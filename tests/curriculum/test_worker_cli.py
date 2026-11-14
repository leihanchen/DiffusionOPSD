import json

from PIL import Image
import torch


def test_worker_cli_loads_once_for_both_roles_and_replays_cache(tmp_path, monkeypatch):
    import scripts.curriculum_vlm_worker as worker

    loaded = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.calls = 0

        @property
        def device(self):
            return self.weight.device

        def generate(self, input_ids, **kw):
            self.calls += 1
            return torch.cat([input_ids, torch.tensor([[2]])], 1)

    class Processor:
        def apply_chat_template(self, messages, **kw):
            self.diagnosis = 'requirements' in messages[-1]['content'][-1]['text']
            return {'input_ids': torch.tensor([[1]])}

        def decode(self, *a, **kw):
            return 'response'

        def parse_response(self, *a, **kw):
            value = ({'requirements': [{'requirement_id': 'stationary', 'status': 'unknown',
                                       'frame_ids': [], 'reason': 'Single frame'}]} if self.diagnosis else
                     [{'prompt_id': 'a', 'lineage_id': 'a', 'parent_id': None, 'split': 'train',
                       'text': 'A static scene viewed by an orbiting camera', 'subtype': 'orbit',
                       'attributes': {}, 'source': 'initial', 'round_index': 0}])
            return {'content': json.dumps(value)}

    model = Model()

    def load(*args):
        loaded.append(1)
        return model, Processor()

    monkeypatch.setattr(worker, 'load_frozen_model', load)
    frame = tmp_path / 'frame.png'
    Image.new('RGB', (10, 10)).save(frame)
    base = dict(model_id='google/gemma-4-12B-it', model_revision='a' * 40, seed=42)
    proposal = base | {'mode': 'propose', 'payload': {'domain': 'G1', 'round_index': 0, 'split': 'train',
                       'count': 1, 'sources': {'initial': 1}, 'parents': [], 'feedback': {}}}
    diagnosis = base | {'mode': 'diagnose', 'payload': {'prompt': 'A statue', 'requirements': ['stationary'],
                 'evidence': {'frame_count': 1, 'native_fps': 24.,
                              'frames': [{'frame_id': 0, 'timestamp': 0., 'path': str(frame)}]}}}
    request, response = tmp_path / 'request.json', tmp_path / 'response.json'
    request.write_text(json.dumps([proposal, diagnosis]))
    monkeypatch.setattr('sys.argv', ['worker', '--request', str(request), '--response', str(response),
                                    '--model-path', str(tmp_path), '--model-id', base['model_id'],
                                    '--revision', base['model_revision']])
    worker.main()
    rows = json.loads(response.read_text())
    assert len(loaded) == 1 and model.calls == 2
    assert len(rows) == 2
    worker.main()
    assert len(loaded) == 1 and model.calls == 2  # Fully cached batch needs no model load.
    Image.new('RGB', (10, 10), 'red').save(frame)
    worker.main()
    assert len(loaded) == 2 and model.calls == 3  # File content, not only path, keys evidence cache.


def test_wrapper_requires_isolated_interpreter(tmp_path):
    import subprocess
    import os

    environment = os.environ.copy()
    environment.pop('CURRICULUM_VLM_PYTHON', None)
    result = subprocess.run(['bash', 'scripts/run_curriculum_vlm.sh', '--help'], env=environment,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'CURRICULUM_VLM_PYTHON' in result.stderr


def test_worker_failure_persists_elapsed_and_unknown_usage(tmp_path, monkeypatch):
    import scripts.curriculum_vlm_worker as worker
    import pytest

    def failed_load(*args):
        raise RuntimeError('out of memory')

    monkeypatch.setattr(worker, 'load_frozen_model', failed_load)
    request = {'mode': 'propose', 'model_id': 'google/gemma-4-12B-it', 'model_revision': 'a' * 40,
               'payload': {'domain': 'G1', 'round_index': 0, 'split': 'train', 'count': 1,
                           'sources': {'initial': 1}, 'parents': [], 'feedback': {}}}
    source, response = tmp_path / 'request.json', tmp_path / 'response.json'
    source.write_text(json.dumps(request))
    monkeypatch.setattr('sys.argv', ['worker', '--request', str(source), '--response', str(response),
                                    '--model-path', str(tmp_path), '--revision', 'a' * 40])
    with pytest.raises(RuntimeError):
        worker.main()
    receipt = json.loads(next((tmp_path / '.curriculum_vlm_cache').glob('*.json')).read_text())
    assert receipt['status'] == 'failed'
    assert receipt['usage']['elapsed_seconds'] > 0
    assert receipt['usage']['output_tokens'] is None
    assert receipt['stage'] == 'model_load'
    assert receipt['error_message'] == 'out of memory'


def test_invalid_model_output_exits_nonzero_after_saving_response(tmp_path, monkeypatch):
    import pytest
    import scripts.curriculum_vlm_worker as worker
    from diffusionopsd.curriculum.vlm import InvalidModelResponse

    monkeypatch.setattr(worker, 'load_frozen_model', lambda *args: (object(), object()))

    def invalid(*args):
        raise InvalidModelResponse('invalid answer JSON', {'input_tokens': 10, 'output_tokens': 5})

    monkeypatch.setattr(worker, 'run_requests', invalid)
    source, response = tmp_path / 'request.json', tmp_path / 'response.json'
    source.write_text(json.dumps({'mode': 'propose', 'model_id': 'google/gemma-4-12B-it',
        'model_revision': 'a' * 40, 'payload': {'domain': 'G1', 'round_index': 0, 'split': 'train',
        'count': 1, 'sources': {'initial': 1}, 'parents': [], 'feedback': {}}}))
    monkeypatch.setattr('sys.argv', ['worker', '--request', str(source), '--response', str(response),
                                    '--model-path', str(tmp_path), '--revision', 'a' * 40])
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code != 0
    saved = json.loads(response.read_text())
    assert saved['error'] == 'invalid_response'
    assert saved['attempts'][0]['usage']['output_tokens'] == 5
