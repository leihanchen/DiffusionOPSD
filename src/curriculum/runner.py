"""Single-writer round orchestration with immutable inputs and resumable phases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from config.curriculum_cosmos import get_runner_config, validate_execution_profile
from diffusionopsd.curriculum.diagnosis import validate_audit_approval, validate_diagnosis
from diffusionopsd.curriculum.execution import device_claims, file_hash, file_lock, inventory, invoke
from diffusionopsd.curriculum.feedback import development_contract, summarize_round
from diffusionopsd.curriculum.proposer import atomic_json, content_hash
from diffusionopsd.curriculum.records import manifest_hash, read_manifest, validate_manifest
from diffusionopsd.curriculum.synthesis import next_manifest


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def configured_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class CurriculumRunner:
    def __init__(self, config_path: Path, run_dir: Path):
        self.config_path = Path(config_path).resolve()
        self.root = Path(run_dir).resolve()
        self.config = {**get_runner_config(), **read_json(self.config_path)}
        self.state_path = self.root / 'state.json'
        self.lock_fds = []

    def validate(self):
        cfg = self.config
        for key in ('rounds', 'prompts_per_round', 'development_count', 'development_seeds',
                    'clips_per_prompt', 'phase_timeout_seconds'):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        quotas = cfg['allocation']
        if (not isinstance(quotas, dict) or set(quotas) != {'retained', 'edited', 'rehearsal', 'new'}
                or any(type(value) is not int or value < 0 for value in quotas.values())
                or cfg['clips_per_prompt'] != 4 or sum(quotas.values()) != cfg['prompts_per_round']):
            raise ValueError('Runner requires four clips per prompt and a complete allocation budget')
        if cfg['feedback_mode'] not in {'numeric_only', 'visual_shadow', 'numeric_and_visual'}:
            raise ValueError('Unsupported feedback mode')
        if cfg['curriculum_mode'] not in {'fixed', 'adaptive'}:
            raise ValueError('Unsupported curriculum mode')
        for name in ('trainer_command', 'proposer_command'):
            if not isinstance(cfg[name], list) or not cfg[name]:
                raise ValueError(f'{name} requires an executable argument list')
        if not isinstance(cfg['execution_profile'], dict):
            raise ValueError('A selected execution profile is required')
        revision = cfg['environment'].get('CURRICULUM_VLM_REVISION')
        if not revision or len(revision) != 40:
            raise ValueError('Pinned VLM revision required')
        if cfg['feedback_mode'] != 'numeric_only' and not cfg['diagnosis_command']:
            raise ValueError('Visual feedback requires a diagnosis command')
        if cfg['feedback_mode'] == 'numeric_and_visual':
            if not cfg['diagnosis_audit_path']:
                raise ValueError('Combined mode requires an approved diagnosis audit')
            validate_audit_approval(cfg['diagnosis_audit_path'], cfg['execution_profile']['vlm_model_id'], revision)
        self.observed = inventory(cfg['inventory_command'])
        self.profile = validate_execution_profile(cfg['execution_profile'], self.observed)
        selected_revision = self.profile.get('vlm_revision')
        if selected_revision and selected_revision != revision:
            raise ValueError('Profile and worker revisions differ')
        self.profile['vlm_revision'] = revision
        self.profile['training_seed'] = cfg['seed']
        supported = dict(vlm_dtype='bfloat16', vlm_batch_size=1, frame_limit=8, frame_longest_side=448,
                         input_token_cap=8192, diagnosis_output_cap=1024, proposal_output_cap=2048)
        if (len(self.profile['vlm_device_ids']) != 1
                or any(self.profile.get(key) != value for key, value in supported.items())):
            raise ValueError('Execution profile differs from supported single-device VLM budgets')
        self.worker_environment = dict(
            CURRICULUM_VLM_DEVICE=f"cuda:{self.profile['vlm_device_ids'][0]}",
            CURRICULUM_VLM_MODEL_ID=self.profile['vlm_model_id'])
        for key, value in self.worker_environment.items():
            if key in cfg['environment'] and cfg['environment'][key] != value:
                raise ValueError(f'Worker environment conflicts with execution profile: {key}')
        options = {'--device': self.worker_environment['CURRICULUM_VLM_DEVICE'],
                   '--model-id': self.profile['vlm_model_id'], '--revision': revision,
                   '--model-path': cfg['environment'].get('CURRICULUM_VLM_MODEL_PATH')}
        for command in (cfg['proposer_command'], cfg['diagnosis_command'] or []):
            for index, argument in enumerate(command):
                key, separator, inline = argument.partition('=')
                if key in options:
                    value = inline if separator else command[index + 1] if index + 1 < len(command) else None
                    if value != options[key]:
                        raise ValueError(f'Worker command conflicts with execution profile: {key}')
        self.gpu_type = 'A100' if 'A100' in self.observed['gpu_model'] else 'H200'
        self.gpu_ids = [device['uuid'] for device in self.observed['devices']]
        self.input_hashes = {}
        for name in ('initial_manifest', 'development_manifest', 'test_manifest', 'training_config'):
            if not cfg[name] or not Path(cfg[name]).is_file():
                raise ValueError(f'Missing required input {name}')
            self.input_hashes[name] = file_hash(Path(cfg[name]))
        if cfg['diagnosis_audit_path']:
            self.input_hashes['diagnosis_audit'] = file_hash(Path(cfg['diagnosis_audit_path']))
        for name in ('initial', 'revise', 'diagnose'):
            template = Path(os.environ.get('CURRICULUM_TEMPLATE_DIR', 'config/curriculum_prompts')) / f'{name}.md'
            self.input_hashes[f'template_{name}'] = file_hash(template)
        for path in Path(__file__).parent.glob('*.py'):
            self.input_hashes[f'controller_{path.name}'] = file_hash(path)
        for command in (cfg['trainer_command'], cfg['proposer_command'], cfg['diagnosis_command'] or []):
            for argument in command:
                path = Path(argument)
                if path.suffix in {'.py', '.sh'} and path.is_file():
                    self.input_hashes[f'executable_{path.resolve()}'] = file_hash(path)
        initial = read_manifest(cfg['initial_manifest'])
        development = read_manifest(cfg['development_manifest'])
        test = read_manifest(cfg['test_manifest'])
        validate_manifest([*initial, *development, *test], [])
        if (len(initial) != cfg['prompts_per_round']
                or any(record.split != 'train' or record.round_index != 0 for record in initial)
                or any(record.split != 'test' for record in test)):
            raise ValueError('Initial training or test manifest has invalid partitions/rounds')
        development_contract(development, [*initial, *test], seed=cfg['seed'],
                             seeds_per_prompt=cfg['development_seeds'], expected_count=cfg['development_count'])
        self.initial, self.development, self.test = initial, development, test
        budget = cfg['budget']
        if (budget['max_wall_seconds'] <= 0 or budget['max_gpu_hours'].get(self.gpu_type, 0) <= 0
                or any(budget['phase_estimate_seconds'].get(name, 0) <= 0
                       for name in ('train_one_pass', 'diagnose_clips', 'prepare_next_manifest'))):
            raise ValueError('Positive wall/GPU caps and phase estimates are required')

    def initialize(self):
        identity = content_hash(dict(config=self.config, inputs=self.input_hashes,
                                     profile=self.profile, inventory=self.observed))
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state['identity'] != identity:
                raise ValueError('Configuration, profile, inventory or input hash changed on resume')
            for receipt in self.state['phases'].values():
                if receipt['status'] == 'complete':
                    self.check_outputs(receipt)
            for path, digest in self.state['snapshots'].items():
                if file_hash(Path(path)) != digest:
                    raise ValueError(f'Immutable input snapshot changed: {path}')
            return
        self.state = dict(format_version=1, identity=identity, status='running',
                          completed_rounds=0, phases={}, costs=dict(gpu_hours={}, elapsed_seconds=0),
                          external_initial_acquisition='Prepared externally; acquisition costs unmeasured')
        atomic_json(self.root / 'resolved_config.json', self.config)
        atomic_json(self.root / 'inventory.json', self.observed)
        atomic_json(self.root / 'profile.json', self.profile)
        atomic_json(self.root / 'input_hashes.json', self.input_hashes)
        for name, rows in (('initial', self.initial), ('development', self.development), ('test', self.test)):
            atomic_json(self.root / f'{name}.json', [asdict(record) for record in rows])
        training_config = self.root / ('training_config' + Path(self.config['training_config']).suffix)
        training_config.write_bytes(Path(self.config['training_config']).read_bytes())
        self.state['snapshots'] = {
            str(self.root / f'{name}.json'): file_hash(self.root / f'{name}.json')
            for name in ('initial', 'development', 'test', 'resolved_config', 'inventory', 'profile', 'input_hashes')
        }
        self.state['snapshots'][str(training_config)] = file_hash(training_config)
        self.save()

    def save(self):
        atomic_json(self.state_path, self.state)

    def check_outputs(self, receipt):
        for path, digest in receipt.get('outputs', {}).items():
            if not Path(path).is_file() or file_hash(Path(path)) != digest:
                raise ValueError(f'Committed phase output hash changed: {path}')

    def finish_phase(self, name, key, paths, started):
        elapsed = time.time() - started
        receipt = self.state['phases'][key]
        receipt.update(status='complete', elapsed_seconds=elapsed,
                       outputs={str(path): file_hash(path) for path in paths})
        self.charge(elapsed)
        self.save()

    def charge(self, elapsed):
        costs = self.state['costs']
        costs['elapsed_seconds'] += elapsed
        hours = costs['gpu_hours']
        hours[self.gpu_type] = hours.get(self.gpu_type, 0) + elapsed * len(self.gpu_ids) / 3600

    def phase(self, round_index, name, action):
        key = f'{round_index:03}/{name}'
        receipt = self.state['phases'].get(key)
        if receipt and receipt['status'] == 'complete':
            self.check_outputs(receipt)
            return
        if receipt and receipt['status'] == 'in_flight':
            self.charge(max(0, time.time() - receipt['started_unix']))
            receipt['status'] = 'interrupted'
            self.save()
        budget = self.config['budget']
        estimate = budget['phase_estimate_seconds'].get(name, 0)
        costs = self.state['costs']
        remaining = min(budget['max_wall_seconds'] - costs['elapsed_seconds'],
                        (budget['max_gpu_hours'][self.gpu_type] - costs['gpu_hours'].get(self.gpu_type, 0))
                        * 3600 / len(self.gpu_ids))
        if remaining <= estimate:
            raise RuntimeError(f'Budget exhausted before {name}; inspect saved costs')
        self.timeout = min(self.config['phase_timeout_seconds'], remaining)
        started = time.time()
        attempts = receipt.get('attempts', 0) if receipt else 0
        if attempts >= 3:
            raise RuntimeError(f'Phase {key} failed three attempts; inspect receipts before another run')
        attempt_history = receipt.get('attempt_history', []) if receipt else []
        if receipt:
            attempt_history = [*attempt_history, {field: value for field, value in receipt.items()
                                                  if field != 'attempt_history'}]
        self.state['phases'][key] = dict(status='in_flight', started_unix=started, attempts=attempts + 1,
                                        attempt_history=attempt_history)
        self.save()
        try:
            paths = action()
        except BaseException as exc:
            elapsed = time.time() - started
            self.charge(elapsed)
            self.state['phases'][key].update(status='failed', error=str(exc), elapsed_seconds=elapsed)
            self.save()
            raise
        self.finish_phase(name, key, paths, started)

    def train(self, directory, round_index):
        cfg = self.config
        training = directory / 'training'
        checkpoint = training / 'training_state.pt'
        training_config = self.root / ('training_config' + Path(cfg['training_config']).suffix)
        command = [*cfg['trainer_command'], '--config', str(training_config), '--run-dir', str(training),
                   '--curriculum-manifest', str(directory / 'manifest.json'), '--curriculum-round', str(round_index),
                   '--curriculum-history', str(directory / 'history.json'),
                   '--curriculum-execution-profile', str(self.root / 'profile.json'),
                   '--development-manifest', str(self.root / 'development.json'),
                   '--development-count', str(cfg['development_count']),
                   '--development-seeds', str(cfg['development_seeds'])]
        if checkpoint.exists():
            command.append('--resume')
        elif round_index:
            command.extend(['--previous-round-checkpoint',
                            str(self.root / f'round_{round_index - 1:03}/training/training_state.pt')])
        invoke(command, directory=training, environment=cfg['environment'], timeout=self.timeout,
               lock_fds=self.lock_fds)
        receipt = read_json(training / 'round_state.json')
        records = read_manifest(directory / 'manifest.json')
        if (receipt['checkpoint_sha256'] != file_hash(checkpoint)
                or receipt['manifest_hash'] != manifest_hash(records) or receipt['round_index'] != round_index
                or receipt['next_index'] != len(records) or not receipt['development_done']
                or receipt['round_stats']['prompt_visits'] != len(records)
                or receipt['round_stats']['attempted_clips'] != len(records) * cfg['clips_per_prompt']):
            raise ValueError('Trainer did not commit the complete manifest and development pass')
        if receipt['round_stats']['successful_updates'] < 1:
            raise RuntimeError('Fully rejected round has no successful update; smoke failed')
        return [checkpoint, training / 'round_state.json', training / 'training_feedback.json',
                training / 'development_feedback.json', training / 'costs.json']

    def evaluate_development(self, directory):
        path = directory / 'training/development_feedback.json'
        rows = read_json(path)
        expected = {record.prompt_id for record in self.development}
        if (len(rows) != len(self.development) * self.config['development_seeds']
                or {row['prompt_id'] for row in rows} != expected
                or any(row['split'] != 'development' for row in rows)):
            raise ValueError('Incomplete fixed development evaluation')
        return [path]

    def diagnose(self, directory, records):
        path = directory / 'diagnoses.json'
        if self.config['feedback_mode'] == 'numeric_only':
            atomic_json(path, [])
            return [path]
        requests = []
        lookup = {record.prompt_id: record for record in records}
        rows = read_json(directory / 'training/training_feedback.json')
        for row in rows:
            if not row.get('evidence'):
                continue
            request = dict(mode='diagnose', model_id=self.profile['vlm_model_id'],
                           model_revision=self.profile['vlm_revision'], seed=self.config['seed'],
                           payload=dict(prompt=lookup[row['prompt_id']].text, evidence=row['evidence'],
                                        requirements=['camera_motion', 'stationarity', 'identity', 'geometry']))
            requests.append(request)
        if not requests:
            raise ValueError('Visual phase requires saved frame evidence')
        request_path, response_path = directory / 'diagnosis_requests.json', directory / 'diagnosis_responses.json'
        atomic_json(request_path, requests)
        invoke([*self.config['diagnosis_command'], '--request', str(request_path), '--response', str(response_path)],
               directory=directory / 'diagnosis_worker', environment=self.config['environment'],
               timeout=self.timeout, lock_fds=self.lock_fds)
        responses = read_json(response_path)
        if not isinstance(responses, list) or len(responses) != len(requests):
            raise ValueError('Diagnosis worker returned an incomplete batch')
        diagnoses = []
        for request, response in zip(requests, responses, strict=True):
            if (response.get('model_id'), response.get('model_revision')) != (
                    self.profile['vlm_model_id'], self.profile['vlm_revision']):
                raise ValueError('Diagnosis worker model differs from proposer')
            payload = request['payload']
            result = validate_diagnosis(response['result'], [frame['frame_id'] for frame in payload['evidence']['frames']],
                                        payload['requirements'])
            diagnoses.append(dict(request_hash=content_hash(request), result=result, usage=response.get('usage', {})))
        atomic_json(path, diagnoses)
        return [path, request_path, response_path]

    def results(self):
        proposals = [read_json(path) for path in self.root.glob('round_*/proposals/*/receipt.json')]
        diagnoses = [read_json(path) for path in self.root.glob('round_*/diagnosis_responses.json')]
        return dict(status='complete', completed_rounds=self.state['completed_rounds'],
                    feedback_mode=self.config['feedback_mode'], curriculum_mode=self.config['curriculum_mode'],
                    costs=self.state['costs'],
                    acquisition=dict(proposals=proposals, diagnoses=diagnoses,
                                     interrupted_uncommitted_work='Unknown beyond the last worker/checkpoint receipt'),
                    final_test='Frozen only; evaluate after experiment selection is locked',
                    rounds=[read_json(self.root / f'round_{index:03}/training/round_state.json')
                            for index in range(self.config['rounds'])])

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(self.root / 'run.lock') as run_fd, configured_environment(self.config['environment']):
            self.validate()
            lock_directory = Path(self.config.get('device_lock_dir', '/tmp/diffusionopsd-curriculum-gpus'))
            with device_claims(lock_directory, self.gpu_ids) as gpu_fds, configured_environment(self.worker_environment):
                self.lock_fds = [run_fd, *gpu_fds]
                self.initialize()
                if self.state['status'] == 'complete':
                    return
                history = [*self.development, *self.test]
                for round_index in range(self.config['rounds']):
                    directory = self.root / f'round_{round_index:03}'
                    directory.mkdir(exist_ok=True)

                    def prepare():
                        if round_index == 0:
                            atomic_json(directory / 'manifest.json', [asdict(record) for record in self.initial])
                        records = read_manifest(directory / 'manifest.json')
                        validate_manifest(records, history)
                        atomic_json(directory / 'history.json', [asdict(record) for record in history])
                        return [directory / 'manifest.json', directory / 'history.json']

                    self.phase(round_index, 'prepare_manifest', prepare)
                    records = read_manifest(directory / 'manifest.json')
                    self.phase(round_index, 'train_one_pass', lambda: self.train(directory, round_index))
                    self.phase(round_index, 'evaluate_development',
                               lambda: self.evaluate_development(directory))
                    self.phase(round_index, 'diagnose_clips', lambda: self.diagnose(directory, records))

                    def summarize():
                        summary = summarize_round(read_json(directory / 'training/training_feedback.json'),
                                                  read_json(directory / 'training/development_feedback.json'))
                        atomic_json(directory / 'summary.json', summary)
                        return [directory / 'summary.json']

                    self.phase(round_index, 'summarize', summarize)
                    if round_index + 1 < self.config['rounds']:
                        def synthesize():
                            destination = self.root / f'round_{round_index + 1:03}'
                            destination.mkdir(exist_ok=True)
                            diagnoses = (read_json(directory / 'diagnoses.json')
                                         if self.config['feedback_mode'] == 'numeric_and_visual' else None)
                            rows = next_manifest(records, history, read_json(directory / 'summary.json'),
                                                 config=self.config, round_index=round_index + 1,
                                                 directory=directory, diagnoses=diagnoses,
                                                 timeout_seconds=self.timeout, lock_fds=tuple(self.lock_fds))
                            atomic_json(destination / 'manifest.json', [asdict(record) for record in rows])
                            return [destination / 'manifest.json']

                        self.phase(round_index, 'prepare_next_manifest', synthesize)
                    history.extend(records)
                    self.state['completed_rounds'] = round_index + 1
                    self.save()
                atomic_json(self.root / 'results.json', self.results())
                self.state['snapshots'][str(self.root / 'results.json')] = file_hash(self.root / 'results.json')
                self.state['status'] = 'complete'
                self.save()


def run_experiment(config_path: Path, run_dir: Path) -> None:
    CurriculumRunner(config_path, run_dir).run()
