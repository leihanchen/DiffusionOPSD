"""Pinned one-node native MiniMax-H3 t2va OPSD pilot."""
import ml_collections

from diffusionopsd.video.geo_reward import reward_metadata
from diffusionopsd.video.training_data import default_dataset_config


def get_config():
    return ml_collections.ConfigDict(dict(
        model='${VIDEO_REWARD_CKPT_PATH}/MiniMax-H3', model_id='MiniMaxAI/MiniMax-H3',
        model_revision='0919fb684102ef115126612f3fbb84d1027e7317', workflow='t2va',
        conditioning_cache='${VIDEO_REWARD_CKPT_PATH}/minimax-h3-conditioning', conditioning_dtype='bfloat16',
        dataset=default_dataset_config(), geometry_reward=reward_metadata(),
        num_nodes=1, policy_gpus=(0, 1), target_gpus=(2,), policy_microbatch_size=1, candidates_per_node=4, clips_per_prompt=4,
        # Bound idle peers during role-local loads, generation, sequential requests, and fitting.
        collective_timeout_s=3600., service_timeout_s=600.,
        num_frames=124, fps=24.0, height=544, width=960, num_steps=50,
        video_shift=12.0, audio_shift=3.0, query_sigma=0.278,
        learning_rate=1e-5, lora_rank=16, lora_alpha=32,
        beta=1.0, rho=0.10, n_ascent=2, eta=1.0, loss_scale=5.0, max_grad_norm=1.0,
        ema_decay=0.99, seed=42, train_pool_size=256, calibration_prompts=4,
        updates=50, eval_count=8, eval_every=10, checkpoint_every=1,
        max_updates_this_job=0, walltime_seconds=0., checkpoint_publication_reserve_seconds=60.,
        reward_frames=9, reward_height=224, reward_width=384,
        tau_id=-1.0, tau_motion=-1.0, tau_quality=0.4, save_videos=True,
    ))
