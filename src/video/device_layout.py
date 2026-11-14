"""Two isolated CUDA devices per video training worker.

The policy/optimizer uses local GPU 0. A frozen inference replica and every
auxiliary model use local GPU 1. Cross-device copies retain policy gradients.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch


def validate_two_gpu_layout(policy_device, frozen_devices, *, visible_count):
    if visible_count != 2:
        raise ValueError(f"Video workers require exactly two visible GPUs; found {visible_count}")
    if torch.device(policy_device) != torch.device("cuda:0"):
        raise ValueError("Trainable policy must use explicit cuda:0")
    if not frozen_devices or any(torch.device(d) != torch.device("cuda:1") for d in frozen_devices):
        raise ValueError("All frozen inference models, text encoder and VAE must use explicit cuda:1")


def module_device(module, *, fallback=None):
    device = getattr(module, "device", None)
    if device is not None:
        return torch.device(device)
    for name in ("parameters", "buffers"):
        members = getattr(module, name, None)
        if members is not None:
            first = next(iter(members()), None)
            if first is not None:
                return first.device
    if fallback is not None:
        return torch.device(fallback)
    raise ValueError("Cannot determine the device of a stateless module without an explicit fallback")


def tensor_tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: tensor_tree_to(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree_to(v, device) for v in value)
    return value


@torch.no_grad()
def encode_wan_prompt(pipe, *args, device, **kwargs):
    """Run the frozen encoder on its own GPU and transfer only its outputs."""
    frozen_device = module_device(pipe.text_encoder)
    result = pipe.encode_prompt(*args, device=frozen_device, **kwargs)
    return tensor_tree_to(result, device)


class InferenceReplica:
    """A detached behavior/current-policy replica; never an optimizer target."""

    def __init__(self, policy, *, device="cuda:1", lora=True):
        self.lora = lora
        # A normal deepcopy first clones every tensor on the source GPU. Large
        # policies fit individually but cannot fit a second transient copy there.
        # Seed deepcopy's memo with destination tensors, preserving tied objects.
        memo = {}
        for parameter in policy.parameters():
            memo[id(parameter)] = torch.nn.Parameter(parameter.detach().to(device, copy=True), requires_grad=False)
        for buffer in policy.buffers():
            memo[id(buffer)] = buffer.detach().to(device, copy=True)
        self.model = copy.deepcopy(policy, memo).to(device).eval().requires_grad_(False)
        if hasattr(self.model, "disable_gradient_checkpointing"):
            self.model.disable_gradient_checkpointing()

    @contextmanager
    def active_on(self, device):
        """Activate an idle replica for generation, then release its GPU memory.

        Cosmos stages its large inference model on CPU between generation calls
        so reward forwards and target backwards have GPU1 activation headroom.
        This context must not enclose a differentiable model forward.
        """
        idle_device = module_device(self.model)
        active_device = torch.device(device)
        try:
            self.model.to(active_device)
            yield self.model
        finally:
            self.model.to(idle_device)
            if active_device.type == "cuda" and idle_device != active_device:
                with torch.cuda.device(active_device):
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def sync(self, policy):
        if self.lora:
            from peft import get_peft_model_state_dict, set_peft_model_state_dict

            for name in ("default", "old"):
                set_peft_model_state_dict(self.model, get_peft_model_state_dict(policy, adapter_name=name),
                                         adapter_name=name)
        else:
            self.model.load_state_dict(policy.state_dict())
        self.model.eval().requires_grad_(False)

    def select(self, name):
        self.model.set_adapter(name)
        # PEFT set_adapter re-enables gradients; inference must remain frozen.
        self.model.eval().requires_grad_(False)
