import copy

import pytest

from diffusionopsd.video.distributed_epoch import (
    allocation_layout,
    assign_rollouts,
    canonical_samples,
    route_sample_weights,
    validate_worker_devices,
    prepare_resume,
    training_config_hash,
)


@pytest.mark.parametrize("workers", [4, 8, 16])
def test_sample_assignment_preserves_epoch_and_global_batch(workers):
    order = list(reversed(range(200)))
    seen = []
    for update in range(50):
        assignments = [assign_rollouts(order, update, rank, workers) for rank in range(workers)]
        samples = [(prompt, rollout) for prompt, rollouts in assignments for rollout in rollouts]
        expected = [(prompt, rollout) for prompt in order[update * 4 : update * 4 + 4] for rollout in range(4)]
        assert samples == expected
        assert len(set(samples)) == 16
        assert all(len(rollouts) == 16 // workers for _, rollouts in assignments)
        seen.extend(samples)
    assert len(set(seen)) == 800


@pytest.mark.parametrize("nodes,gpus,expected", [(4, 2, (1, 4)), (4, 4, (2, 8)), (8, 4, (2, 16))])
def test_allocation_derives_workers(nodes, gpus, expected):
    assert allocation_layout(nodes, gpus) == expected


@pytest.mark.parametrize("nodes,gpus", [(0, 4), (4, 3), (4, 0), (3, 2), (12, 4)])
def test_invalid_allocation_rejected(nodes, gpus):
    with pytest.raises(ValueError):
        allocation_layout(nodes, gpus)


@pytest.mark.parametrize("workers", [4, 8, 16])
def test_canonical_order_and_weight_routing(workers):
    order = [9, 2, 7, 1]
    gathered = []
    for rank in range(workers):
        prompt, rollouts = assign_rollouts(order, 0, rank, workers)
        gathered.append(
            [
                {
                    "prompt_index": prompt,
                    "rollout_index": rollout,
                    "prompt": str(prompt),
                    "r": float(prompt + rollout),
                    "m": 1.0,
                    "p_q": 0.5,
                }
                for rollout in reversed(rollouts)
            ]
        )
    canonical = canonical_samples(gathered, order, 0, workers)
    assert [(s["prompt_index"], s["rollout_index"]) for s in canonical] == [
        (prompt, rollout) for prompt in order for rollout in range(4)
    ]
    weights = list(range(16))
    routed = route_sample_weights(canonical, weights, gathered)
    for samples, values in zip(gathered, routed):
        assert values == [order.index(s["prompt_index"]) * 4 + s["rollout_index"] for s in samples]
    gathered[0][0] = copy.deepcopy(gathered[0][-1])
    if len(gathered[0]) == 1:
        gathered[0][0]["rollout_index"] = 99
    with pytest.raises(ValueError):
        canonical_samples(gathered, order, 0, workers)


def test_gpu_pairs_must_be_unique_within_host():
    records = [
        {"rank": 0, "local_rank": 0, "host": "n0", "gpu_uuids": ["a", "b"]},
        {"rank": 1, "local_rank": 1, "host": "n0", "gpu_uuids": ["c", "d"]},
        {"rank": 2, "local_rank": 0, "host": "n1", "gpu_uuids": ["e", "f"]},
        {"rank": 3, "local_rank": 1, "host": "n1", "gpu_uuids": ["g", "h"]},
    ]
    validate_worker_devices(records, 4)
    records[1]["gpu_uuids"][0] = "b"
    with pytest.raises(ValueError, match="overlap"):
        validate_worker_devices(records, 4)


def checkpoint(cfg):
    return {
        "version": 2,
        "training_config": cfg,
        "training_config_hash": training_config_hash(cfg),
        "prompt_hash": "prompts",
        "topology": {"world_size": 4, "workers_per_node": 1, "nodes": 4},
        "rng_states": [dict(rank=i) for i in range(4)],
        "next_update": 4,
        "nonce": 123,
        "max_update_seconds": 3000.0,
        "policy_default": {"x": 1},
        "policy_old": {"x": 2},
        "optimizer": {"step": 4},
    }


def test_resume_exact_or_explicit_migration():
    cfg = {"logdir": "/old", "video": {"width": 1280}, "sample": {"num_image_per_prompt": 4}}
    state = checkpoint(cfg)
    topology = state["topology"]
    result, changed = prepare_resume(state, {**cfg, "logdir": "/new"}, "prompts", topology, False)
    assert not changed
    assert result["rng_states"] == state["rng_states"]
    assert result["max_update_seconds"] == 3000.0
    target = {"world_size": 8, "workers_per_node": 2, "nodes": 4}
    with pytest.raises(ValueError, match="topology"):
        prepare_resume(state, cfg, "prompts", target, False)
    migrated, changed = prepare_resume(state, cfg, "prompts", target, True)
    assert changed and migrated["max_update_seconds"] is None
    assert migrated["next_update"] == 4
    assert migrated["optimizer"] == state["optimizer"]
    assert state["max_update_seconds"] == 3000.0
    with pytest.raises(ValueError, match="config"):
        prepare_resume(state, {**cfg, "video": {"width": 832}}, "prompts", target, True)
    with pytest.raises(ValueError, match="prompt"):
        prepare_resume(state, cfg, "other", target, True)


def test_legacy_hash_is_verified_using_original_run_directory():
    import hashlib
    import json

    cfg = {"logdir": "/old", "video": {"width": 1280}}
    legacy = {
        "version": 1,
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "prompt_hash": "prompts",
        "rng_states": [{"rank": i} for i in range(4)],
        "nonce": 12,
        "next_update": 4,
        "max_update_seconds": 3000.0,
    }
    target = {"world_size": 8, "workers_per_node": 2, "nodes": 4}
    result, changed = prepare_resume(legacy, {**cfg, "logdir": "/new"}, "prompts", target, True, "/old")
    assert changed and result["next_update"] == 4
    with pytest.raises(ValueError, match="config"):
        prepare_resume(legacy, {**cfg, "logdir": "/new"}, "prompts", target, True, "/wrong")


def test_eight_worker_checkpoint_round_trip_and_seed_stability(tmp_path):
    from diffusionopsd.video.distributed_epoch import atomic_checkpoint, deterministic_seed, topology_rng_seed

    cfg = {"logdir": "/old", "learning_rate": 1e-4}
    state = checkpoint(cfg)
    state["topology"] = {"world_size": 8, "workers_per_node": 2, "nodes": 4}
    state["rng_states"] = [{"rank": i} for i in range(8)]
    path = tmp_path / "training_state.pt"
    atomic_checkpoint(path, state)
    import torch

    loaded = torch.load(path, weights_only=False)
    restored, changed = prepare_resume(loaded, cfg, "prompts", state["topology"], False)
    assert not changed and restored["rng_states"] == state["rng_states"]
    assert len({topology_rng_seed(123, 4, rank) for rank in range(8)}) == 8
    assert topology_rng_seed(123, 4, 0) != topology_rng_seed(123, 5, 0)
    expected = {(p, r): deterministic_seed(123, p, r) for p in range(4) for r in range(4)}
    for world in (4, 8, 16):
        actual = {}
        for rank in range(world):
            p, rollouts = assign_rollouts(list(range(4)), 0, rank, world)
            actual.update({(p, r): deterministic_seed(123, p, r) for r in rollouts})
        assert actual == expected


def _uneven_gradient_worker(rank, rendezvous, result_dir, world):
    import torch
    import torch.distributed as dist
    from diffusionopsd.video.distributed_epoch import synchronize_gradients

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    # Same retained samples under every layout; four/eight-worker layouts include empty ranks.
    retained = [i for i in range(16) if i >= 4 and i % 3 and i // (16 // world) == rank]
    for sample in retained:
        (parameter * (sample + 1)).backward()
    count = torch.tensor(len(retained))
    dist.all_reduce(count)
    synchronize_gradients([parameter], int(count))
    torch.save(parameter.grad, f"{result_dir}/gradient_{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_uneven_retention_gradients_match_serial_reference(tmp_path, workers):
    import torch
    import torch.multiprocessing as mp

    mp.spawn(_uneven_gradient_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), workers), nprocs=workers)
    expected = torch.tensor([sum(i + 1 for i in range(4, 16) if i % 3) / sum(bool(i % 3) for i in range(4, 16))])
    for rank in range(workers):
        gradient = torch.load(tmp_path / f"gradient_{rank}.pt", weights_only=True)
        assert torch.allclose(gradient, expected)


def test_adapter_fingerprint_handles_bfloat16_and_order():
    import torch
    from diffusionopsd.video.distributed_epoch import adapter_fingerprint

    state = {"b": torch.tensor([2.0], dtype=torch.bfloat16), "a": torch.tensor([1.0])}
    assert adapter_fingerprint(state) == adapter_fingerprint(dict(reversed(list(state.items()))))
    state2 = {**state, "b": torch.tensor([3.0], dtype=torch.bfloat16)}
    assert adapter_fingerprint(state) != adapter_fingerprint(state2)


def test_single_node_smoke_preserves_all_sixteen_samples():
    from diffusionopsd.video.distributed_epoch import worker_sample_assignments

    order = list(range(200))
    gathered = []
    for rank in range(2):
        assignments = worker_sample_assignments(order, 4, rank, 2, single_node_smoke=True)
        assert len(assignments) == 4
        samples = [
            {
                "prompt_index": prompt,
                "rollout_index": rollout,
                "prompt": str(prompt),
                "r": float(rollout),
                "m": 1.0,
                "p_q": 0.5,
            }
            for prompt, rollouts in assignments
            for rollout in rollouts
        ]
        assert len(samples) == 8
        gathered.append(samples)
    canonical = canonical_samples(gathered, order, 4, 2, single_node_smoke=True)
    assert [(s["prompt_index"], s["rollout_index"]) for s in canonical] == [
        (prompt, rollout) for prompt in (16, 17, 18, 19) for rollout in range(4)
    ]
    with pytest.raises(ValueError):
        worker_sample_assignments(order, 4, 0, 2)
    with pytest.raises(ValueError):
        worker_sample_assignments(order, 4, 0, 4, single_node_smoke=True)
