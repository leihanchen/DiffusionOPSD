import torch
from torch import nn
from diffusionopsd.video.wan_policy import LORA_TARGET_MODULES, attach_lora, ema_adapter_


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


def test_targets_are_bare_suffixes():
    assert "to_q" in LORA_TARGET_MODULES
    assert "to_out.0" in LORA_TARGET_MODULES
    assert all(not name.startswith("attn.") for name in LORA_TARGET_MODULES)


def test_ema_adapter_moves_old_halfway_toward_default():
    torch.manual_seed(0)
    policy = attach_lora(Tiny(), None)
    policy.set_adapter("default")
    with torch.no_grad():
        for param in [p for p in policy.parameters() if p.requires_grad]:
            param.fill_(1.0)
    policy.set_adapter("old")
    with torch.no_grad():
        for param in [p for p in policy.parameters() if p.requires_grad]:
            param.fill_(0.0)
    ema_adapter_(policy, decay=0.5)
    policy.set_adapter("old")
    for param in [p for p in policy.parameters() if p.requires_grad]:
        assert torch.allclose(param, torch.full_like(param, 0.5))
