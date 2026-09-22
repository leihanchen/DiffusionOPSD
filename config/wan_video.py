import ml_collections
from config.base import get_config as base_config


def get_config():
    config = base_config()
    config.pretrained.model = "${VIDEO_REWARD_CKPT_PATH}/Wan2.1-T2V-1.3B-Diffusers"
    config.resolution = 480
    config.video = video = ml_collections.ConfigDict()
    video.width = 832
    video.height = 480
    video.num_frames = 17
    config.sample.num_steps = 30
    config.sample.guidance_scale = 5.0
    config.sample.num_image_per_prompt = 4          # K clips per prompt
    config.sample.train_batch_size = 1
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 8
    config.train.learning_rate = 1e-4
    config.train.adv_clip_max = 5
    config.beta = 1.0                                # branch coefficient
    config.opa = ml_collections.ConfigDict()
    config.opa.rho = 0.10
    config.opa.n_ascent = 2
    config.opa.eta = 1.0
    config.opa.query_sigma = 0.278
    config.opa.mb = 1                                # target-construction microbatch (spec Sec. 5.2)
    config.gates = gates = ml_collections.ConfigDict()
    gates.tau_id = -1.0                              # -1 = calibrate from base-model 10th percentile at start
    gates.tau_motion = -1.0
    gates.tau_q = 0.4
    gates.calib_prompts = 64
    config.judge = ml_collections.ConfigDict()
    config.judge.device = "cuda:1"
    config.judge.num_frames = 8
    config.reward_device = "cuda:1"
    config.prompt_fn = "text_file"
    config.prompt_fn_kwargs = {"path": "data/video_motion/train.txt"}
    config.eval_prompts = "data/video_motion/test.txt"
    return config
