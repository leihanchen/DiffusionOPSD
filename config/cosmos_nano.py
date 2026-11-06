"""Native-UniPC Cosmos3-Nano video OPSD defaults for one two-GPU worker."""

import ml_collections
from diffusionopsd.video.geo_reward import reward_metadata


def get_config():
    return ml_collections.ConfigDict(dict(
        geometry_reward=reward_metadata(),
        model="${VIDEO_REWARD_CKPT_PATH}/Cosmos3-Nano",
        model_id="nvidia/Cosmos3-Nano",
        model_revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
        policy_device="cuda:0", vae_device="cuda:1", reward_device="cuda:1", judge_device="cuda:1",
        train_prompts="data/video_motion/train.txt", eval_prompts="data/video_motion/test.txt",
        num_frames=17, height=480, width=832, fps=24.0,
        num_steps=35, guidance_scale=6.0, flow_shift=10.0, query_sigma=0.278,
        updates=50, clips_per_prompt=4, learning_rate=1e-5, lora_rank=16,
        lora_targets=["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        beta=1.0, rho=0.10, n_ascent=2, eta=1.0, loss_scale=5.0, max_grad_norm=1.0,
        ema_decay=0.99, seed=42, calibration_prompts=4,
        tau_id=-1.0, tau_motion=-1.0, tau_quality=0.4,
        reward_frames=9, reward_height=224, reward_width=384,
        eval_every=10, eval_count=2, checkpoint_every=1,
        offload_saved_tensors=True, save_videos=True,
    ))
