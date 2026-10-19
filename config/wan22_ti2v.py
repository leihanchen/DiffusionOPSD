from config.wan_video import get_config as wan21_config


def get_config():
    config = wan21_config()
    config.pretrained.model = "${VIDEO_REWARD_CKPT_PATH}/Wan2.2-TI2V-5B-Diffusers"
    config.resolution = 704
    config.video.width = 1280
    config.video.height = 704
    config.use_lora = True
    return config
