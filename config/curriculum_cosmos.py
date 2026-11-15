"""Bounded G1 pilot and explicit heterogeneous execution profiles.

Independent runs scale across nodes. These profiles do not enable distributed Cosmos.
Runtime/asset paths and pinned VLM revision are resolved by execution preflight.
"""

from copy import deepcopy


def get_config() -> dict:
    return dict(prompts_per_round=32, clips_per_prompt=4, rounds=3, seed=42,
                development_prompts=8, development_seeds=2, test_prompts=16, test_seeds=2,
                audit_clip_count=12,
                allocation=dict(retained=8, edited=12, rehearsal=4, new=8),
                subtypes=['orbit', 'dolly', 'lateral'], calibration_split='train',
                proposer_command=None, execution_profile=None)


def get_runner_config() -> dict:
    return dict(rounds=3, prompts_per_round=32, clips_per_prompt=4, seed=42,
                development_count=8, development_seeds=2,
                allocation=dict(retained=8, edited=12, rehearsal=4, new=8),
                feedback_mode='numeric_only', curriculum_mode='adaptive',
                initial_manifest=None, development_manifest=None, test_manifest=None,
                training_config=None, trainer_command=None, proposer_command=None,
                diagnosis_command=None, inventory_command=None, execution_profile=None,
                environment={}, diagnosis_audit_path=None, phase_timeout_seconds=14400,
                budget=dict(max_wall_seconds=43200, max_gpu_hours={'A100': 24},
                            phase_estimate_seconds=dict(train_one_pass=3600, diagnose_clips=3600,
                                                        prepare_next_manifest=600)))


def execution_profile(name: str) -> dict:
    if name not in {'narval_gemma12', 'narval_parallel_arms', 'h200_gemma12_control', 'h200_qwen27_comparison'}:
        raise ValueError(f'Unknown execution profile: {name}')
    narval = name.startswith('narval')
    return dict(profile_name=name, cluster='narval' if narval else 'h200',
                available_nodes=None if narval else 8, nodes_per_run=1,
                gpus_per_node=4 if narval else None, gpu_memory_gb=40 if narval else None,
                train_device_map=dict(policy=0, vae=1, reward=1, judge=1), vlm_device_ids=[0],
                vlm_model_id='Qwen/Qwen3.8-27B' if name == 'h200_qwen27_comparison' else 'google/gemma-4-12B-it',
                vlm_revision=None, vlm_dtype='bfloat16', vlm_batch_size=1,
                frame_limit=8, frame_longest_side=448, input_token_cap=8192,
                diagnosis_output_cap=1024, proposal_output_cap=2048,
                runtime_paths={}, asset_paths={}, max_concurrent_runs=2 if name == 'narval_parallel_arms' else 1)


def validate_execution_profile(profile: dict, inventory: dict) -> dict:
    """Resolve observed local inventory; reject unsupported or impossible placement.

    This validates placement only. Weight paths, revision, and measured memory fit
    are separately required by the VLM worker; no published memory figure proves fit.
    """
    result = deepcopy(profile)
    count, memory = inventory.get('gpus_per_node'), inventory.get('gpu_memory_gb')
    if type(count) is not int or count < 1 or not isinstance(memory, (float, int)) or memory <= 0:
        raise ValueError('Selected profile requires measured GPU count and memory')
    expected = 'A100' if profile['cluster'] == 'narval' else 'H200'
    if expected not in inventory.get('gpu_model', ''):
        raise ValueError('GPU model does not match selected environment')
    if profile['cluster'] == 'narval' and memory < 39:
        raise ValueError('Narval profile requires A100 40 GB devices')
    if profile['nodes_per_run'] != 1:
        raise ValueError('Cross-node Cosmos training is not implemented')
    devices = [*profile['train_device_map'].values(), *profile['vlm_device_ids']]
    if not devices or any(type(i) is not int or i < 0 or i >= count for i in devices):
        raise ValueError('Device map exceeds the local allocation')
    result.update(gpus_per_node=count, gpu_memory_gb=memory)
    return result
