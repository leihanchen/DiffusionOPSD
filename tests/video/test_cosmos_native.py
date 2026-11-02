"""CPU integration with the real Cosmos model/pipeline, using random tiny weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

cosmos = pytest.importorskip("diffusers.pipelines.cosmos.pipeline_cosmos3_omni")
from diffusers import Cosmos3OmniTransformer, UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from diffusionopsd.video.cosmos_rollout import CosmosRollout  # noqa: E402
from diffusionopsd.video.wan_policy import ema_adapter_  # noqa: E402


class TinyVideoVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False))
        self.config = SimpleNamespace(
            latents_mean=[0.1], latents_std=[0.7], scale_factor_spatial=16, scale_factor_temporal=4
        )

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def encode(self, pixels):
        return SimpleNamespace(latents=F.avg_pool3d(pixels[:, :1, ::4], (1, 16, 16)))

    def decode(self, latents, return_dict=False):
        return (F.interpolate(latents, size=(5, 32, 32), mode="trilinear").expand(-1, 3, -1, -1, -1).sin(),)


def test_real_cosmos_native_query_suffix_and_lora_gradients():
    torch.manual_seed(8)
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "<|vision_start|>": 2}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    model = Cosmos3OmniTransformer(
        hidden_size=16, intermediate_size=32, head_dim=8, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=1, vocab_size=3, latent_channel=1,
        latent_patch_size=2, patch_latent_dim=4, rope_axes_dim=[2, 1, 1],
    ).requires_grad_(False)
    model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["add_q_proj", "add_v_proj"]))
    model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["add_q_proj", "add_v_proj"]), adapter_name="old")
    set_peft_model_state_dict(model, get_peft_model_state_dict(model), adapter_name="old")
    model.set_adapter("default")
    model.enable_gradient_checkpointing()
    pipe = cosmos.Cosmos3OmniPipeline(
        transformer=model, text_tokenizer=tokenizer, vae=TinyVideoVAE(),
        scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True),
        enable_safety_checker=False,
    )
    roll = CosmosRollout(pipe, num_steps=4, guidance_scale=2.0)
    native = pipe(
        "camera pans right", num_frames=5, height=32, width=32, num_inference_steps=4,
        guidance_scale=2.0, generator=torch.Generator().manual_seed(11), output_type="latent", return_dict=False,
    )[0]
    rec = roll.rollout("camera pans right", num_frames=5, height=32, width=32, sigma_star=0.4,
                       generator=torch.Generator().manual_seed(11))
    assert torch.equal(rec.x0, native)
    assert rec.z_q.shape == (1, 1, 2, 2, 2)
    assert torch.allclose(roll.prediction(rec), rec.y0, atol=1e-6)
    with torch.no_grad():
        assert torch.allclose(roll.continue_from(rec, rec.y0), rec.x0, atol=2e-6)
    before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    prediction = roll.prediction(rec)
    loss = (prediction - rec.y0 - 0.05).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in get_peft_model_state_dict(model).items())
    old_before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model, adapter_name="old").items()}
    ema_adapter_(model, decay=0.99)
    for k, v in get_peft_model_state_dict(model, adapter_name="old").items():
        assert torch.allclose(v, old_before[k] * .99 + get_peft_model_state_dict(model)[k] * .01)
