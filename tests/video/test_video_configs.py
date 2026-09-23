from config.wan22_ti2v import get_config as ti2v_config
from config.wan_video import get_config as wan21_config


def test_wan21_stays_full_finetune_on_the_1_3b_checkpoint():
    cfg = wan21_config()
    assert cfg.use_lora is False
    assert cfg.pretrained.model.endswith("Wan2.1-T2V-1.3B-Diffusers")
    assert (cfg.video.num_frames, cfg.video.height, cfg.video.width) == (17, 480, 832)


def test_ti2v_reuses_the_prompt_grid_and_turns_lora_on():
    cfg = ti2v_config()
    assert cfg.use_lora is True
    assert cfg.pretrained.model.endswith("Wan2.2-TI2V-5B-Diffusers")
    assert cfg.prompt_fn_kwargs["path"] == "data/video_motion/train.txt"
    assert cfg.eval_prompts == "data/video_motion/test.txt"
    assert cfg.opa.query_sigma == 0.278
    assert (cfg.video.num_frames, cfg.video.height, cfg.video.width) == (17, 480, 832)
