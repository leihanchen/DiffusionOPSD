"""LoRA policy for Wan2.2-TI2V-5B. The base transformer stays frozen."""

from __future__ import annotations

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft import PeftModel

LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


def lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=list(LORA_TARGET_MODULES),
    )


def attach_lora(transformer, lora_path):
    cfg = lora_config()
    if lora_path:
        policy = PeftModel.from_pretrained(transformer, lora_path)
    else:
        policy = get_peft_model(transformer, cfg)
    policy.add_adapter("old", cfg)
    policy.set_adapter("default")
    return policy


def ema_adapter_(model, src="default", dst="old", decay=0.99):
    src_sd = get_peft_model_state_dict(model, adapter_name=src)
    dst_sd = get_peft_model_state_dict(model, adapter_name=dst)
    mixed = {}
    for key, src_value in src_sd.items():
        mixed[key] = dst_sd[key] * decay + src_value.detach() * (1.0 - decay)
    set_peft_model_state_dict(model, mixed, adapter_name=dst)
