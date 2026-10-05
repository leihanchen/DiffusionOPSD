import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusionopsd.video.distributed_epoch import (
    atomic_checkpoint,
    deterministic_seed,
    load_checkpoint,
    prompt_schedule,
    should_start_update,
    split_update,
    synchronize_gradients,
)


def test_shuffled_one_pass_and_stable_per_sample_seeds():
    prompts = [f"prompt {i}" for i in range(200)]
    order = prompt_schedule(prompts, nonce=17, workers=4)
    assert sorted(order) == list(range(200))
    assert len(order) == 200
    assert len({tuple(split_update(order, update, rank, 4)) for update in range(50) for rank in range(4)}) == 200
    assert split_update(order, 0, 0, 4) == [order[0]]
    assert split_update(order, 0, 3, 4) == [order[3]]
    assert deterministic_seed(17, order[0], 0) == deterministic_seed(17, order[0], 0)
    assert deterministic_seed(17, order[0], 0) != deterministic_seed(17, order[0], 1)
    with pytest.raises(ValueError, match="divisible"):
        prompt_schedule(prompts[:-1], nonce=17, workers=4)


def test_epoch_preset_has_50_updates_and_three_evaluations():
    from config.wan22_ti2v_epoch import get_config

    config = get_config()
    assert (config.video.width, config.video.height, config.video.num_frames) == (1280, 704, 17)
    assert config.sample.num_steps == 30
    assert config.sample.num_batches_per_epoch * config.num_epochs == 200
    assert config.sample.num_image_per_prompt == 4
    assert (config.train.batch_size, config.sample.train_batch_size) == (1, 1)
    assert (config.save_freq, config.eval_freq, config.eval_num_prompts) == (50, 25, 2)


def test_atomic_checkpoint_validates_identity_and_preserves_previous(tmp_path):
    path = tmp_path / "state.pt"
    state = {"next_update": 8, "prompt_hash": "prompt-v1", "config_hash": "config-v1",
             "optimizer": {"moment": torch.tensor([2.0])}}
    atomic_checkpoint(path, state)
    assert load_checkpoint(path, "prompt-v1", "config-v1")["next_update"] == 8
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="prompt"):
        load_checkpoint(path, "prompt-v2", "config-v1")
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(path, "prompt-v1", "config-v2")
    assert torch.equal(load_checkpoint(path, "prompt-v1", "config-v1")["optimizer"]["moment"],
                       torch.tensor([2.0]))


def test_job_budget_stops_before_another_update():
    assert should_start_update(completed=0, total=50, elapsed=0, max_update_seconds=None, budget=72000)
    assert should_start_update(completed=1, total=50, elapsed=1000, max_update_seconds=3600, budget=72000)
    assert not should_start_update(completed=1, total=50, elapsed=67000, max_update_seconds=3600, budget=72000)
    assert not should_start_update(completed=50, total=50, elapsed=1000, max_update_seconds=3600, budget=72000)


def _gradient_worker(rank, rendezvous, result_dir):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    loss = parameter * (rank + 1)
    loss.backward()
    synchronize_gradients([parameter], total_kept=4)
    torch.save(parameter.grad, f"{result_dir}/gradient_{rank}.pt")
    dist.destroy_process_group()


def test_four_worker_gradients_match_global_mean(tmp_path):
    mp.spawn(_gradient_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=4)
    for rank in range(4):
        gradient = torch.load(tmp_path / f"gradient_{rank}.pt", weights_only=True)
        assert torch.equal(gradient, torch.tensor([2.5]))
