# Geometry-Consistent DiffusionOPSD for Video — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the DiffusionOPSD codebase so a rectified-flow text-to-video model (default Wan 2.1 T2V-1.3B) can be post-trained with a differentiable GeoFlow-style geometry reward as the local and endpoint reward, an identity gate, a motion floor, and a VLM pairwise quality mask, with a fixed-suffix probe that must pass before training.

**Architecture:** All new code lives in a new `src/video/` package (`diffusionopsd.video`) and three new scripts; nothing in the existing SD3.5-M / Z-Image trainers is modified. The video reward is a plain `torch.nn.Module` over decoded frames `[B,T,3,H,W]` in `[0,1]`, built from three frozen estimators behind small adapter protocols so they can be mocked in tests. The trust-region target step from `scripts/train_opsd_ri_sd3.py::_opa_tr_step` is re-implemented once in N-D form and shared by the probe and the trainer.

**Tech Stack:** Python ≥3.10, torch ≥2.6, diffusers ≥0.36 (`WanPipeline`, `AutoencoderKLWan`), transformers 4.51.x (repo pin), Depth Anything 3 Large v1.1, WAFT, DINOv2-base (`torch.hub facebookresearch/dinov2 dinov2_vitb14`), Qwen2.5-VL-7B-Instruct, pytest.

Spec: `docs/superpowers/specs/2026-09-22-geometry-consistent-opsd-video-design.md`.

## Global Constraints

- Do not edit `scripts/train_opsd_ri_sd3.py`, `scripts/train_opsd_zimage.py`, or `src/rewards.py`. New behavior goes in `src/video/` and new scripts.
- Package import path is `diffusionopsd.video.*` (pyproject maps `diffusionopsd` → `src`). Add `"diffusionopsd.video"` to `[tool.setuptools] packages`.
- Frozen estimators and the VLM judge are never trained: call `.requires_grad_(False)` and `.eval()` on load.
- Decoded frames are always `[B,T,3,H,W]` float32 in `[0,1]`. Latents are `[B,C,T',H',W']`.
- Reward composition is fixed: `R_geo = mean_t(0.5 * R_rigid_t + 0.5 * R_dino_t)`.
- Endpoint weight uses the repo's existing convention `r1 = clamp((clamp(adv,-A,A)/A)/2 + 0.5, 0, 1)` with `A = config.train.adv_clip_max`; masks multiply `r1`.
- Training thresholds: `tau_id`, `tau_motion` = 10th percentile of base-model statistics on the training prompts; `tau_q = 0.4`.
- No model weights ship with this repo or with the OpenResearch demo bundle. Every task that needs weights starts with the download step and must be runnable on CPU with mocks in tests.
- `ruff` line length 120. Tests in `tests/video/`. Run tests with `pytest tests/video -q`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/video/__init__.py` | package marker |
| `src/video/rigid_flow.py` | pure-torch geometry: back-projection, rigid flow from depth + pose, reprojection residual |
| `src/video/gates.py` | identity gate, motion floor, quality mask, `effective_weight` |
| `src/video/opa_video.py` | N-D trust-region ascent/descent step (`opa_tr_step_nd`) |
| `src/video/estimators.py` | adapter protocols + loaders for Depth Anything 3, WAFT, DINOv2 |
| `src/video/geo_reward.py` | `GeoReward` module: composes estimators into `R_geo`, `s_id`, `m` |
| `src/video/quality_judge.py` | `PairwiseVLMJudge` (Qwen2.5-VL) returning `p_q` |
| `src/video/wan_clean_output.py` | Wan 2.1 rollout with query capture, clean-output map, decode |
| `src/video/probe_stats.py` | aggregate fixed-suffix probe records into pass/fail |
| `src/video/branch_loss.py` | paper Eq. 11–13 with the masked weight |
| `src/video/eval_table.py` | spec Sec. 7 success table |
| `config/wan_video.py` | ml_collections config for the video trainer |
| `scripts/download_video_reward_weights.sh` | fetch DA3, WAFT, DINOv2, Qwen2.5-VL, Wan 2.1 |
| `scripts/check_video_reward_setup.py` | one-clip smoke: finite `R_geo` and finite latent gradient |
| `scripts/probe_fixed_suffix_video.py` | Sec. 6 go/no-go probe |
| `scripts/train_opsd_video_wan.py` | training loop |
| `scripts/eval_video_consistency.py` | Sec. 7 held-out table |
| `tests/video/*.py` | unit tests (CPU, mocked estimators) |

---

### Task 1: Rigid flow from depth and pose

**Files:**
- Create: `src/video/__init__.py` (empty)
- Create: `src/video/rigid_flow.py`
- Modify: `pyproject.toml` (`packages` list)
- Test: `tests/video/test_rigid_flow.py`

**Interfaces:**
- Produces:
  - `pixel_grid(h: int, w: int, device, dtype) -> Tensor[3,H,W]` homogeneous pixel coords `(u, v, 1)`.
  - `rigid_flow(depth: Tensor[B,H,W], K: Tensor[B,3,3], T_rel: Tensor[B,4,4]) -> Tensor[B,2,H,W]`. `T_rel` maps points in frame-t camera coordinates to frame-(t+1) camera coordinates.
  - `reprojection_residual(flow: Tensor[B,2,H,W], rigid: Tensor[B,2,H,W], conf: Tensor[B,H,W]) -> Tensor[B]` = confidence-weighted mean L1 residual per sample.

- [ ] **Step 1: Write the failing tests**

```python
# tests/video/test_rigid_flow.py
import torch
from diffusionopsd.video.rigid_flow import pixel_grid, rigid_flow, reprojection_residual


def _K(f=100.0, cx=8.0, cy=6.0):
    K = torch.eye(3)
    K[0, 0] = f; K[1, 1] = f; K[0, 2] = cx; K[1, 2] = cy
    return K[None]


def test_pixel_grid_shape_and_values():
    g = pixel_grid(4, 6, "cpu", torch.float32)
    assert g.shape == (3, 4, 6)
    assert g[0, 0, 5] == 5 and g[1, 3, 0] == 3 and torch.all(g[2] == 1)


def test_identity_pose_gives_zero_flow():
    depth = torch.full((1, 12, 16), 2.0)
    flow = rigid_flow(depth, _K(), torch.eye(4)[None])
    assert flow.shape == (1, 2, 12, 16)
    assert torch.allclose(flow, torch.zeros_like(flow), atol=1e-5)


def test_pure_x_translation_on_fronto_parallel_plane():
    depth = torch.full((1, 12, 16), 2.0)
    T = torch.eye(4)[None].clone(); T[0, 0, 3] = 0.1
    flow = rigid_flow(depth, _K(f=100.0), T)
    # u' - u = f * tx / Z = 100 * 0.1 / 2 = 5 px, no vertical flow
    assert torch.allclose(flow[:, 0], torch.full_like(flow[:, 0], 5.0), atol=1e-4)
    assert torch.allclose(flow[:, 1], torch.zeros_like(flow[:, 1]), atol=1e-4)


def test_reprojection_residual_weights_by_confidence():
    flow = torch.zeros(1, 2, 4, 4)
    rigid = torch.zeros(1, 2, 4, 4); rigid[0, 0, 0, 0] = 8.0
    conf = torch.ones(1, 4, 4)
    assert torch.isclose(reprojection_residual(flow, rigid, conf), torch.tensor([8.0 / 16]))
    conf[0, 0, 0] = 0.0
    assert torch.isclose(reprojection_residual(flow, rigid, conf), torch.tensor([0.0]))


def test_rigid_flow_is_differentiable_wrt_depth():
    depth = torch.full((1, 6, 8), 2.0, requires_grad=True)
    T = torch.eye(4)[None].clone(); T[0, 0, 3] = 0.1
    rigid_flow(depth, _K(), T).sum().backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_rigid_flow.py -q`
Expected: `ModuleNotFoundError: No module named 'diffusionopsd.video'`

- [ ] **Step 3: Implement**

`pyproject.toml`: change `packages = ["diffusionopsd", "diffusionopsd.diffusers_patch"]` to `packages = ["diffusionopsd", "diffusionopsd.diffusers_patch", "diffusionopsd.video"]`. Then `pip install -e .` again.

```python
# src/video/rigid_flow.py
"""Rigid-scene geometry in pure torch: depth + camera motion -> induced optical flow."""

from __future__ import annotations

import torch


def pixel_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=0)  # [3,H,W]


def rigid_flow(depth: torch.Tensor, K: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    """Flow induced by a rigid camera motion.

    depth: [B,H,W] metric depth of frame t.  K: [B,3,3].  T_rel: [B,4,4], frame-t camera -> frame-(t+1) camera.
    Returns flow [B,2,H,W] in pixels (u' - u, v' - v).
    """
    B, H, W = depth.shape
    grid = pixel_grid(H, W, depth.device, depth.dtype).reshape(1, 3, -1).expand(B, 3, -1)  # [B,3,N]
    rays = torch.linalg.solve(K, grid)  # K^-1 [u,v,1]
    X = rays * depth.reshape(B, 1, -1)  # [B,3,N]
    R, t = T_rel[:, :3, :3], T_rel[:, :3, 3:4]
    X2 = R @ X + t
    z2 = X2[:, 2:3].clamp(min=1e-6)
    p2 = (K @ (X2 / z2))[:, :2]  # [B,2,N]
    flow = p2 - grid[:, :2]
    return flow.reshape(B, 2, H, W)


def reprojection_residual(flow: torch.Tensor, rigid: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted mean L1 residual between observed and rigid flow, per sample -> [B]."""
    l1 = (flow - rigid).abs().sum(dim=1)  # [B,H,W]
    w = conf.clamp(min=0.0)
    return (w * l1).flatten(1).sum(1) / (w.flatten(1).sum(1) + 1e-8) * (w.flatten(1).sum(1) > 0)
```

Note on the last line: with all-zero confidence the residual is defined as 0, matching the test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/video/test_rigid_flow.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/video/__init__.py src/video/rigid_flow.py tests/video/test_rigid_flow.py
git commit -m "feat(video): rigid flow from depth and pose, reprojection residual"
```

---

### Task 2: Gates and effective weight

**Files:**
- Create: `src/video/gates.py`
- Test: `tests/video/test_gates.py`

**Interfaces:**
- Produces:
  - `identity_keep(s_id: Tensor[B], tau_id: float) -> BoolTensor[B]` — True keeps the query.
  - `motion_mask(m: Tensor[B], tau_motion: float) -> Tensor[B]` float 0/1.
  - `quality_mask(p_q: Tensor[B], tau_q: float) -> Tensor[B]` float 0/1.
  - `endpoint_weight(adv: Tensor[B], adv_clip_max: float) -> Tensor[B]` = repo `r1`.
  - `effective_weight(adv, adv_clip_max, m, tau_motion, p_q, tau_q) -> Tensor[B]` = `r1 * motion_mask * quality_mask`.
  - `percentile_threshold(values: Tensor[N], q: float) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/video/test_gates.py
import torch
from diffusionopsd.video.gates import (
    identity_keep, motion_mask, quality_mask, endpoint_weight, effective_weight, percentile_threshold,
)


def test_identity_keep():
    keep = identity_keep(torch.tensor([0.9, 0.5, 0.7]), tau_id=0.7)
    assert keep.tolist() == [True, False, True]


def test_motion_and_quality_masks_are_float_indicators():
    assert motion_mask(torch.tensor([0.1, 2.0]), 0.5).tolist() == [0.0, 1.0]
    assert quality_mask(torch.tensor([0.39, 0.4]), 0.4).tolist() == [0.0, 1.0]


def test_endpoint_weight_matches_repo_r1():
    adv = torch.tensor([-10.0, -5.0, 0.0, 2.5, 5.0, 10.0])
    r1 = endpoint_weight(adv, adv_clip_max=5.0)
    assert torch.allclose(r1, torch.tensor([0.0, 0.0, 0.5, 0.75, 1.0, 1.0]))


def test_effective_weight_is_hard_masked():
    adv = torch.tensor([5.0, 5.0, 5.0])
    w = effective_weight(adv, 5.0, m=torch.tensor([1.0, 0.0, 1.0]), tau_motion=0.5,
                         p_q=torch.tensor([0.9, 0.9, 0.1]), tau_q=0.4)
    assert w.tolist() == [1.0, 0.0, 0.0]


def test_percentile_threshold():
    vals = torch.arange(1.0, 101.0)
    assert abs(percentile_threshold(vals, 0.10) - 10.9) < 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_gates.py -q`
Expected: `ModuleNotFoundError: No module named 'diffusionopsd.video.gates'`

- [ ] **Step 3: Implement**

```python
# src/video/gates.py
"""Identity gate, motion floor, quality mask, and the masked endpoint weight (spec Sec. 4.2-4.5)."""

from __future__ import annotations

import torch


def identity_keep(s_id: torch.Tensor, tau_id: float) -> torch.Tensor:
    return s_id >= tau_id


def motion_mask(m: torch.Tensor, tau_motion: float) -> torch.Tensor:
    return (m >= tau_motion).to(m.dtype)


def quality_mask(p_q: torch.Tensor, tau_q: float) -> torch.Tensor:
    return (p_q >= tau_q).to(p_q.dtype)


def endpoint_weight(adv: torch.Tensor, adv_clip_max: float) -> torch.Tensor:
    """Repo convention from scripts/train_opsd_ri_sd3.py: r1 in [0,1] from a clipped advantage."""
    adv_clip = torch.clamp(adv, -adv_clip_max, adv_clip_max)
    return torch.clamp((adv_clip / adv_clip_max) / 2.0 + 0.5, 0, 1)


def effective_weight(adv, adv_clip_max, m, tau_motion, p_q, tau_q) -> torch.Tensor:
    return endpoint_weight(adv, adv_clip_max) * motion_mask(m, tau_motion) * quality_mask(p_q, tau_q)


def percentile_threshold(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().flatten(), q).item())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/video/test_gates.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/gates.py tests/video/test_gates.py
git commit -m "feat(video): identity gate, motion floor, quality mask, effective weight"
```

---

### Task 3: N-D trust-region target step

**Files:**
- Create: `src/video/opa_video.py`
- Test: `tests/video/test_opa_video.py`

**Interfaces:**
- Produces: `opa_tr_step_nd(y0: Tensor[B,...], reward_fn: Callable[[Tensor], Tensor[B]], rho: float, n_ascent: int, eta: float, direction: float, first_grad: Tensor | None = None) -> Tensor[B,...]`. Same semantics as `scripts/train_opsd_ri_sd3.py::_opa_tr_step` with `dir_mode="grad"`, but per-sample norms are computed with `flatten(1)` and broadcast to any rank, and the reward is passed as a closure over latents (so the decoder lives inside `reward_fn`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/video/test_opa_video.py
import torch
from diffusionopsd.video.opa_video import opa_tr_step_nd


def _quadratic_reward(center):
    return lambda y: -((y - center) ** 2).flatten(1).sum(1)


def test_ascent_moves_toward_maximizer_and_respects_radius():
    torch.manual_seed(0)
    y0 = torch.randn(2, 4, 3, 5, 5)  # [B,C,T,H,W]
    center = y0 + 10.0
    y_plus = opa_tr_step_nd(y0, _quadratic_reward(center), rho=0.1, n_ascent=2, eta=1.0, direction=+1.0)
    d = (y_plus - y0).flatten(1).norm(dim=1)
    budget = 0.1 * y0.flatten(1).norm(dim=1)
    assert torch.all(d <= budget * (1 + 1e-4))
    assert torch.all(d >= budget * 0.99)  # gradient is constant, so the full budget is used
    assert torch.all(((y_plus - y0) * (center - y0)).flatten(1).sum(1) > 0)


def test_descent_moves_away():
    y0 = torch.randn(1, 4, 2, 4, 4)
    center = y0 + 1.0
    y_minus = opa_tr_step_nd(y0, _quadratic_reward(center), rho=0.05, n_ascent=1, eta=1.0, direction=-1.0)
    assert torch.all(((y_minus - y0) * (center - y0)).flatten(1).sum(1) < 0)


def test_output_is_detached_and_same_shape():
    y0 = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
    out = opa_tr_step_nd(y0, _quadratic_reward(torch.zeros_like(y0)), 0.1, 1, 1.0, 1.0)
    assert out.shape == y0.shape and not out.requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_opa_video.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/opa_video.py
"""N-D port of scripts/train_opsd_ri_sd3.py::_opa_tr_step (dir_mode='grad') for video latents."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _bshape(x: torch.Tensor):
    return (-1,) + (1,) * (x.ndim - 1)


def opa_tr_step_nd(
    y0: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    rho: float,
    n_ascent: int,
    eta: float,
    direction: float,
    first_grad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x0 = y0.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(_bshape(x0))
    step_len = eta * budget / max(n_ascent, 1)
    x = x0.clone()
    for i in range(n_ascent):
        if i == 0 and first_grad is not None:
            g = first_grad.float()
        else:
            x = x.detach().requires_grad_(True)
            r = reward_fn(x)
            (g,) = torch.autograd.grad(r.sum(), x)
        gn = g.flatten(1).norm(dim=1).view(_bshape(x0)) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gn)
        d = x - x0
        dn = d.flatten(1).norm(dim=1).view(_bshape(x0))
        x = (x0 + d * torch.clamp(budget / (dn + 1e-12), max=1.0)).detach()
    return x.detach()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/video/test_opa_video.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/opa_video.py tests/video/test_opa_video.py
git commit -m "feat(video): N-D trust-region target step"
```

---

### Task 4: Estimator adapters and weight download

**Files:**
- Create: `src/video/estimators.py`
- Create: `scripts/download_video_reward_weights.sh`
- Test: `tests/video/test_estimators.py`

**Interfaces:**
- Produces (protocols; every downstream task consumes only these):
  - `class DepthEstimator(Protocol): def __call__(self, frames01: Tensor[B,T,3,H,W]) -> DepthOutput` where `DepthOutput = NamedTuple(depth: Tensor[B,T,H,W], K: Tensor[B,3,3], poses: Tensor[B,T,4,4] (camera-to-world), conf: Tensor[B,T,H,W] in [0,1])`.
  - `class FlowEstimator(Protocol): def __call__(self, a: Tensor[B,3,H,W], b: Tensor[B,3,H,W]) -> Tensor[B,2,H,W]`.
  - `class PatchFeatureExtractor(Protocol): def __call__(self, imgs: Tensor[B,3,H,W]) -> Tensor[B,D,h,w]` (L2-normalized, `patch=14`).
  - `relative_pose(poses: Tensor[B,T,4,4], t: int) -> Tensor[B,4,4]` = `inv(pose[t+1]) @ pose[t]`, mapping frame-t camera coords to frame-(t+1) camera coords.
  - Loaders: `load_depth_anything3(device) -> DepthEstimator`, `load_waft(device) -> FlowEstimator`, `load_dinov2(device) -> PatchFeatureExtractor`. Each reads its checkpoint under `os.environ["VIDEO_REWARD_CKPT_PATH"]`.

- [ ] **Step 1: Write the failing tests** (pure-math parts and the DINOv2 wrapper contract with a stub backbone)

```python
# tests/video/test_estimators.py
import torch
from diffusionopsd.video.estimators import relative_pose, DinoV2Patches


def test_relative_pose_composes_to_identity_for_static_camera():
    poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    assert torch.allclose(relative_pose(poses, 0), torch.eye(4)[None])


def test_relative_pose_translation_sign():
    poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1).clone()
    poses[0, 1, 0, 3] = 0.1  # camera moved +x in world between t=0 and t=1
    T = relative_pose(poses, 0)
    assert torch.isclose(T[0, 0, 3], torch.tensor(-0.1))  # points shift -x in the new camera frame


class _StubBackbone(torch.nn.Module):
    def forward_features(self, x):
        B, _, H, W = x.shape
        n = (H // 14) * (W // 14)
        return {"x_norm_patchtokens": torch.ones(B, n, 8)}


def test_dino_wrapper_returns_normalized_patch_grid():
    fx = DinoV2Patches(_StubBackbone(), patch=14)
    out = fx(torch.rand(2, 3, 28, 42))
    assert out.shape == (2, 8, 2, 3)
    assert torch.allclose(out.norm(dim=1), torch.ones(2, 2, 3), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_estimators.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/estimators.py
"""Frozen vision estimators behind small protocols so GeoReward can be unit-tested with stubs."""

from __future__ import annotations

import os
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DepthOutput(NamedTuple):
    depth: torch.Tensor  # [B,T,H,W]
    K: torch.Tensor      # [B,3,3]
    poses: torch.Tensor  # [B,T,4,4] camera-to-world
    conf: torch.Tensor   # [B,T,H,W] in [0,1]


class DepthEstimator(Protocol):
    def __call__(self, frames01: torch.Tensor) -> DepthOutput: ...


class FlowEstimator(Protocol):
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


class PatchFeatureExtractor(Protocol):
    def __call__(self, imgs: torch.Tensor) -> torch.Tensor: ...


def relative_pose(poses: torch.Tensor, t: int) -> torch.Tensor:
    """camera(t) coords -> camera(t+1) coords given camera-to-world poses."""
    return torch.linalg.inv(poses[:, t + 1]) @ poses[:, t]


def _ckpt_root() -> str:
    root = os.environ.get("VIDEO_REWARD_CKPT_PATH")
    if not root:
        raise EnvironmentError("Set VIDEO_REWARD_CKPT_PATH (see scripts/download_video_reward_weights.sh)")
    return root


class DinoV2Patches(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, patch: int = 14):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.patch = patch

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = imgs.shape
        h, w = H // self.patch, W // self.patch
        x = F.interpolate(imgs, size=(h * self.patch, w * self.patch), mode="bilinear", align_corners=False)
        x = (x - _IMNET_MEAN.to(x)) / _IMNET_STD.to(x)
        tok = self.backbone.forward_features(x)["x_norm_patchtokens"]  # [B,h*w,D]
        feat = tok.transpose(1, 2).reshape(B, -1, h, w)
        return F.normalize(feat, dim=1)


def load_dinov2(device) -> PatchFeatureExtractor:
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    return DinoV2Patches(backbone).to(device)


class WaftFlow(torch.nn.Module):
    """WAFT adapter. Wraps the model object returned by the official WAFT repo (princeton-vl/WAFT)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # WAFT consumes images in [0,255]; its forward returns a list of flow predictions, last is final.
        out = self.model(a * 255.0, b * 255.0)
        flow = out[-1] if isinstance(out, (list, tuple)) else out
        return flow.float()


def load_waft(device) -> FlowEstimator:
    import sys
    root = os.path.join(_ckpt_root(), "WAFT")
    sys.path.insert(0, root)
    from core.waft import WAFT  # official repo module
    import json
    with open(os.path.join(root, "config", "eval", "sintel.json")) as f:
        cfg = json.load(f)
    model = WAFT(cfg)
    sd = torch.load(os.path.join(_ckpt_root(), "waft_tar_c_t.pth"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return WaftFlow(model).to(device)


class DepthAnything3(torch.nn.Module):
    """Depth Anything 3 adapter. Wraps the official inference model (ByteDance-Seed/Depth-Anything-3)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, frames01: torch.Tensor) -> DepthOutput:
        B, T, _, H, W = frames01.shape
        pred = self.model.inference(frames01.reshape(B * T, 3, H, W), batch_size=T)
        depth = pred["depth"].reshape(B, T, H, W)
        conf = pred["conf"].reshape(B, T, H, W).clamp(0, 1)
        K = pred["intrinsics"].reshape(B, T, 3, 3)[:, 0]
        poses = pred["extrinsics_c2w"].reshape(B, T, 4, 4)
        return DepthOutput(depth.float(), K.float(), poses.float(), conf.float())


def load_depth_anything3(device) -> DepthEstimator:
    from depth_anything_3.api import DepthAnything3 as DA3
    model = DA3.from_pretrained(os.path.join(_ckpt_root(), "depth-anything-3-large-v1.1")).to(device)
    return DepthAnything3(model).to(device)
```

The two vendor adapters (`WaftFlow`, `DepthAnything3`) wrap upstream repos whose exact tensor keys are verified in Task 5 Step 6 (the GPU smoke script), not in unit tests. If a key name differs from the vendor release you install, fix it inside the adapter's `forward` only; the `DepthOutput` / flow contracts are fixed.

```bash
# scripts/download_video_reward_weights.sh
#!/usr/bin/env bash
set -euo pipefail
TARGET=${VIDEO_REWARD_CKPT_PATH:-"$(pwd)/video_reward_ckpts"}
mkdir -p "$TARGET"; export VIDEO_REWARD_CKPT_PATH="$TARGET"
command -v hf >/dev/null 2>&1 || { echo "Missing 'hf' CLI" >&2; exit 2; }

echo "Depth Anything 3 Large v1.1"
hf download depth-anything/DA3LARGE-1.1 --local-dir "$TARGET/depth-anything-3-large-v1.1"
pip install -q "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

echo "WAFT"
[ -d "$TARGET/WAFT" ] || git clone -q https://github.com/princeton-vl/WAFT.git "$TARGET/WAFT"
hf download princeton-vl/WAFT waft_tar_c_t.pth --local-dir "$TARGET"

echo "DINOv2 (torch.hub cache warm-up)"
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"

echo "Qwen2.5-VL-7B-Instruct"
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$TARGET/Qwen2.5-VL-7B-Instruct"

echo "Wan 2.1 T2V-1.3B (diffusers layout)"
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$TARGET/Wan2.1-T2V-1.3B-Diffusers"

echo "export VIDEO_REWARD_CKPT_PATH='$TARGET'"
```

Repository ids in this script are the official Hugging Face ids at the time of writing; if `hf download` reports "not found", check the linked GitHub README for the current id and update only this script.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/video/test_estimators.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/download_video_reward_weights.sh
git add src/video/estimators.py scripts/download_video_reward_weights.sh tests/video/test_estimators.py
git commit -m "feat(video): estimator adapters and weight download script"
```

---

### Task 5: GeoReward module

**Files:**
- Create: `src/video/geo_reward.py`
- Create: `scripts/check_video_reward_setup.py`
- Test: `tests/video/test_geo_reward.py`

**Interfaces:**
- Consumes: `rigid_flow`, `reprojection_residual` (Task 1); `DepthEstimator`, `FlowEstimator`, `PatchFeatureExtractor`, `relative_pose` (Task 4).
- Produces:
  - `class GeoRewardOutput(NamedTuple): geo: Tensor[B]; rigid: Tensor[B]; dino: Tensor[B]; s_id: Tensor[B]; motion: Tensor[B]`.
  - `class GeoReward(torch.nn.Module)`: `__init__(depth: DepthEstimator, flow: FlowEstimator, feats: PatchFeatureExtractor, w_rigid=0.5, w_dino=0.5, residual_scale=1.0)`; `forward(frames01: Tensor[B,T,3,H,W]) -> GeoRewardOutput`. `geo` is differentiable w.r.t. `frames01`. `s_id == dino`, `motion` = mean flow magnitude in pixels.
  - `warp_features(feat_next: Tensor[B,D,h,w], flow: Tensor[B,2,H,W]) -> Tensor[B,D,h,w]` — backward-warp patch features along flow (flow downsampled to the patch grid).

- [ ] **Step 1: Write the failing tests** (stub estimators; no weights)

```python
# tests/video/test_geo_reward.py
import torch
from diffusionopsd.video.estimators import DepthOutput
from diffusionopsd.video.geo_reward import GeoReward, warp_features


class StubDepth:
    def __init__(self, tx=0.0):
        self.tx = tx
    def __call__(self, frames01):
        B, T, _, H, W = frames01.shape
        depth = torch.full((B, T, H, W), 2.0) + 0.0 * frames01.mean(dim=2)  # keep graph
        K = torch.eye(3)[None].repeat(B, 1, 1); K[:, 0, 0] = K[:, 1, 1] = 50.0; K[:, 0, 2] = W / 2; K[:, 1, 2] = H / 2
        poses = torch.eye(4)[None, None].repeat(B, T, 1, 1).clone()
        for t in range(T):
            poses[:, t, 0, 3] = -self.tx * t  # camera moving so that points shift +tx*f/Z per step
        return DepthOutput(depth, K, poses, torch.ones(B, T, H, W))


class StubFlow:
    def __init__(self, du):
        self.du = du
    def __call__(self, a, b):
        B, _, H, W = a.shape
        f = torch.zeros(B, 2, H, W) + 0.0 * a.mean(dim=1, keepdim=True)
        f[:, 0] = self.du
        return f


class StubFeats:
    def __call__(self, imgs):
        B, _, H, W = imgs.shape
        f = imgs.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)[:, :, ::14, ::14]
        return torch.nn.functional.normalize(f + 1e-3, dim=1)


def test_consistent_clip_scores_higher_than_inconsistent():
    frames = torch.rand(1, 3, 3, 28, 42)
    # depth 2, f=50, tx=0.2 per step -> rigid flow = 50*0.2/2 = 5 px
    good = GeoReward(StubDepth(tx=0.2), StubFlow(du=5.0), StubFeats())(frames)
    bad = GeoReward(StubDepth(tx=0.2), StubFlow(du=0.0), StubFeats())(frames)
    assert good.rigid > bad.rigid
    assert good.geo > bad.geo
    assert torch.isclose(good.motion, torch.tensor([5.0]))


def test_geo_is_differentiable_wrt_frames():
    frames = torch.rand(1, 2, 3, 28, 28, requires_grad=True)
    out = GeoReward(StubDepth(0.1), StubFlow(2.5), StubFeats())(frames)
    out.geo.sum().backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all()


def test_warp_features_identity_for_zero_flow():
    feat = torch.rand(1, 4, 2, 3)
    assert torch.allclose(warp_features(feat, torch.zeros(1, 2, 28, 42)), feat, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_geo_reward.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/geo_reward.py
"""GeoFlow-style geometry reward on decoded clips (spec Sec. 4.1-4.3)."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from diffusionopsd.video.estimators import DepthEstimator, FlowEstimator, PatchFeatureExtractor, relative_pose
from diffusionopsd.video.rigid_flow import reprojection_residual, rigid_flow


class GeoRewardOutput(NamedTuple):
    geo: torch.Tensor
    rigid: torch.Tensor
    dino: torch.Tensor
    s_id: torch.Tensor
    motion: torch.Tensor


def warp_features(feat_next: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp next-frame patch features to the current frame along pixel flow."""
    B, D, h, w = feat_next.shape
    H, W = flow.shape[-2:]
    flow_p = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
    flow_p = torch.stack([flow_p[:, 0] * (w / W), flow_p[:, 1] * (h / H)], dim=1)  # to patch units
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    grid = torch.stack([xs, ys], dim=0).float()[None] + flow_p  # [B,2,h,w] sample positions in next frame
    gx = 2 * grid[:, 0] / max(w - 1, 1) - 1
    gy = 2 * grid[:, 1] / max(h - 1, 1) - 1
    return F.grid_sample(feat_next, torch.stack([gx, gy], dim=-1), mode="bilinear",
                         padding_mode="border", align_corners=True)


class GeoReward(torch.nn.Module):
    def __init__(self, depth: DepthEstimator, flow: FlowEstimator, feats: PatchFeatureExtractor,
                 w_rigid: float = 0.5, w_dino: float = 0.5, residual_scale: float = 1.0):
        super().__init__()
        self.depth, self.flow, self.feats = depth, flow, feats
        self.w_rigid, self.w_dino, self.residual_scale = w_rigid, w_dino, residual_scale

    def forward(self, frames01: torch.Tensor) -> GeoRewardOutput:
        B, T, _, H, W = frames01.shape
        d = self.depth(frames01)
        rigid_terms, dino_terms, motion_terms = [], [], []
        for t in range(T - 1):
            a, b = frames01[:, t], frames01[:, t + 1]
            flow = self.flow(a, b)
            rig = rigid_flow(d.depth[:, t], d.K, relative_pose(d.poses, t))
            res = reprojection_residual(flow, rig, d.conf[:, t])
            rigid_terms.append(-self.residual_scale * res)
            fa, fb = self.feats(a), self.feats(b)
            sim = (fa * warp_features(fb, flow)).sum(dim=1)  # cosine, [B,h,w]
            dino_terms.append(sim.flatten(1).mean(1))
            motion_terms.append(flow.norm(dim=1).flatten(1).mean(1))
        rigid = torch.stack(rigid_terms, 1).mean(1)
        dino = torch.stack(dino_terms, 1).mean(1)
        motion = torch.stack(motion_terms, 1).mean(1).detach()
        geo = self.w_rigid * rigid + self.w_dino * dino
        return GeoRewardOutput(geo, rigid, dino, dino.detach(), motion)
```

```python
# scripts/check_video_reward_setup.py
#!/usr/bin/env python3
"""Load the real estimators, score one synthetic clip, and verify a finite frame-space gradient."""
from __future__ import annotations
import argparse, json, math, time
import torch
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda"); p.add_argument("--frames", type=int, default=5)
    p.add_argument("--side", type=int, default=224)
    a = p.parse_args()
    t0 = time.perf_counter()
    reward = GeoReward(load_depth_anything3(a.device), load_waft(a.device), load_dinov2(a.device))
    clip = torch.rand(1, a.frames, 3, a.side, a.side, device=a.device, requires_grad=True)
    out = reward(clip)
    (g,) = torch.autograd.grad(out.geo.sum(), clip)
    rec = {"geo": float(out.geo), "rigid": float(out.rigid), "dino": float(out.dino), "motion": float(out.motion),
           "frame_grad_norm": float(g.norm()), "elapsed_s": time.perf_counter() - t0,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else None}
    for k in ("geo", "frame_grad_norm"):
        if not math.isfinite(rec[k]) or (k == "frame_grad_norm" and rec[k] <= 0):
            raise RuntimeError(f"bad {k}: {rec[k]}")
    print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/video/test_geo_reward.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/geo_reward.py scripts/check_video_reward_setup.py tests/video/test_geo_reward.py
git commit -m "feat(video): differentiable GeoReward with identity and motion statistics"
```

- [ ] **Step 6: GPU smoke (requires downloaded weights)**

Run: `bash scripts/download_video_reward_weights.sh && export VIDEO_REWARD_CKPT_PATH=$PWD/video_reward_ckpts && python scripts/check_video_reward_setup.py --device cuda`
Expected: one JSON line with finite `geo` and `frame_grad_norm > 0`. If a vendor tensor key in `estimators.py` does not match the installed release, fix that adapter's `forward` and re-run; then commit with `git commit -am "fix(video): align estimator adapter with vendor release"`.

---

### Task 6: Pairwise VLM quality judge

**Files:**
- Create: `src/video/quality_judge.py`
- Test: `tests/video/test_quality_judge.py`

**Interfaces:**
- Produces: `class PairwiseVLMJudge`: `__init__(model, processor, device, num_frames: int = 8)`; `p_win(clip_a: Tensor[T,3,H,W], clip_b: Tensor[T,3,H,W], prompt: str) -> float` = probability that A is preferred, computed as softmax over the next-token logits for `"A"` vs `"B"` after a fixed instruction, averaged over both orderings to cancel position bias. `load_qwen25_vl(device) -> PairwiseVLMJudge`.
- Constant `JUDGE_PROMPT` (exact text below).

- [ ] **Step 1: Write the failing test** (stub model exposing `generate_logits(inputs) -> Tensor[V]`)

```python
# tests/video/test_quality_judge.py
import torch
from diffusionopsd.video.quality_judge import PairwiseVLMJudge, JUDGE_PROMPT


class _StubProc:
    tokenizer = type("T", (), {"convert_tokens_to_ids": staticmethod(lambda s: {"A": 1, "B": 2}[s])})()
    def build(self, frames_a, frames_b, prompt):
        return {"which_first": "A" if frames_a.mean() > frames_b.mean() else "B"}


class _StubModel:
    def next_token_logits(self, inputs):
        logits = torch.full((10,), -10.0)
        # prefer the brighter clip regardless of ordering
        logits[1 if inputs["which_first"] == "A" else 2] = 5.0
        return logits


def test_p_win_prefers_brighter_clip_and_is_order_symmetric():
    judge = PairwiseVLMJudge(_StubModel(), _StubProc(), device="cpu")
    bright, dark = torch.full((4, 3, 8, 8), 0.9), torch.full((4, 3, 8, 8), 0.1)
    assert judge.p_win(bright, dark, "a cat") > 0.99
    assert judge.p_win(dark, bright, "a cat") < 0.01


def test_prompt_mentions_overall_quality_not_geometry():
    assert "overall quality" in JUDGE_PROMPT and "geometr" not in JUDGE_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/video/test_quality_judge.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/quality_judge.py
"""Pairwise VLM quality judge used only as a hard mask on omega and as a held-out metric (spec Sec. 4.4)."""

from __future__ import annotations

import os
import torch

JUDGE_PROMPT = (
    "You are shown two short video clips, A then B, both generated for the prompt: \"{prompt}\". "
    "Judge overall quality: visual fidelity, temporal smoothness, absence of artifacts, and how well the clip "
    "matches the prompt. Answer with a single letter, A or B, for the better clip."
)


class PairwiseVLMJudge:
    def __init__(self, model, processor, device, num_frames: int = 8):
        self.model, self.processor, self.device, self.num_frames = model, processor, device, num_frames
        tok = processor.tokenizer
        self.id_a, self.id_b = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")

    def _subsample(self, clip: torch.Tensor) -> torch.Tensor:
        idx = torch.linspace(0, clip.shape[0] - 1, self.num_frames).round().long()
        return clip[idx]

    @torch.no_grad()
    def _p_first(self, first, second, prompt) -> float:
        inputs = self.processor.build(self._subsample(first), self._subsample(second), JUDGE_PROMPT.format(prompt=prompt))
        logits = self.model.next_token_logits(inputs)
        two = torch.stack([logits[self.id_a], logits[self.id_b]]).float()
        return float(torch.softmax(two, 0)[0])

    def p_win(self, clip_a, clip_b, prompt) -> float:
        return 0.5 * (self._p_first(clip_a, clip_b, prompt) + (1.0 - self._p_first(clip_b, clip_a, prompt)))


class _QwenProcessorAdapter:
    def __init__(self, processor):
        self.processor, self.tokenizer = processor, processor.tokenizer

    def build(self, frames_a, frames_b, text):
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": [f for f in frames_a]}, {"type": "video", "video": [f for f in frames_b]},
            {"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[chat], videos=[frames_a, frames_b], return_tensors="pt")


class _QwenModelAdapter:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def next_token_logits(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits[0, -1]


def load_qwen25_vl(device) -> PairwiseVLMJudge:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = os.path.join(os.environ["VIDEO_REWARD_CKPT_PATH"], "Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(path)
    return PairwiseVLMJudge(_QwenModelAdapter(model, device), _QwenProcessorAdapter(proc), device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/video/test_quality_judge.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/quality_judge.py tests/video/test_quality_judge.py
git commit -m "feat(video): order-symmetric pairwise VLM quality judge"
```

---

### Task 7: Wan 2.1 clean-output map, rollout with query capture, decode

**Files:**
- Create: `src/video/wan_clean_output.py`
- Create: `config/wan_video.py`
- Test: `tests/video/test_wan_clean_output.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `clean_output(z: Tensor, v: Tensor, sigma: Tensor[B]) -> Tensor` = `z - sigma * v` with `sigma` broadcast over trailing dims.
  - `select_query_index(sigmas: Tensor[S], sigma_star: float) -> int` = argmin `|sigma_j - sigma*|`.
  - `class WanRollout`: `__init__(pipeline: WanPipeline, num_steps: int, guidance_scale: float)`; `rollout(prompt_embeds, negative_embeds, generator, sigma_star: float) -> RolloutRecord` with fields `x0: Tensor[B,C,T',H',W']`, `z_q: Tensor`, `sigma_q: float`, `q_index: int`, `v_old_q: Tensor`, `sigmas: Tensor[S]`; `continue_from(z_q, q_index, forced_y: Tensor, prompt_embeds, negative_embeds) -> Tensor x0` = the fixed-suffix operator: inserts velocity `(z_q - forced_y)/sigma_q` at step `q_index` and finishes with the frozen pipeline transformer; `velocity(z, sigma, prompt_embeds, negative_embeds, transformer=None) -> Tensor` (CFG-combined); `decode01(latents) -> Tensor[B,T,3,H,W]` in `[0,1]` using `pipeline.vae` (differentiable).
  - `get_config()` in `config/wan_video.py` returning the ml_collections config below.

- [ ] **Step 1: Write the failing tests** (pure functions only; the pipeline needs weights)

```python
# tests/video/test_wan_clean_output.py
import torch
from diffusionopsd.video.wan_clean_output import clean_output, select_query_index


def test_clean_output_recovers_data_on_rectified_path():
    torch.manual_seed(0)
    y = torch.randn(2, 4, 3, 5, 5); eps = torch.randn_like(y); sigma = torch.tensor([0.3, 0.7])
    s = sigma.view(-1, 1, 1, 1, 1)
    z = (1 - s) * y + s * eps
    v = eps - y
    assert torch.allclose(clean_output(z, v, sigma), y, atol=1e-5)


def test_select_query_index_nearest():
    sig = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    assert select_query_index(sig, 0.278) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/video/test_wan_clean_output.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/wan_clean_output.py
"""Rectified-flow clean-output map, rollout with low-noise query capture, and fixed-suffix continuation for Wan 2.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def clean_output(z: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s = sigma.to(z).view(-1, *([1] * (z.ndim - 1)))
    return z - s * v


def select_query_index(sigmas: torch.Tensor, sigma_star: float) -> int:
    return int(torch.argmin((sigmas - sigma_star).abs()).item())


@dataclass
class RolloutRecord:
    x0: torch.Tensor
    z_q: torch.Tensor
    sigma_q: float
    q_index: int
    v_old_q: torch.Tensor
    sigmas: torch.Tensor


class WanRollout:
    def __init__(self, pipeline, num_steps: int, guidance_scale: float):
        self.pipe, self.num_steps, self.g = pipeline, num_steps, guidance_scale

    def _sigmas(self, device):
        self.pipe.scheduler.set_timesteps(self.num_steps, device=device)
        return self.pipe.scheduler.sigmas.to(device)  # length num_steps+1, ends at 0

    def velocity(self, z, sigma, prompt_embeds, negative_embeds, transformer=None):
        tr = transformer or self.pipe.transformer
        t = (sigma * 1000.0).expand(z.shape[0]).to(z)
        v_c = tr(hidden_states=z, timestep=t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        if self.g == 1.0:
            return v_c
        v_u = tr(hidden_states=z, timestep=t, encoder_hidden_states=negative_embeds, return_dict=False)[0]
        return v_u + self.g * (v_c - v_u)

    @torch.no_grad()
    def rollout(self, prompt_embeds, negative_embeds, latents, sigma_star: float) -> RolloutRecord:
        sig = self._sigmas(latents.device)
        q = select_query_index(sig[:-1], sigma_star)
        z, z_q, v_old_q = latents, None, None
        for j in range(self.num_steps):
            v = self.velocity(z, sig[j], prompt_embeds, negative_embeds)
            if j == q:
                z_q, v_old_q = z.clone(), v.clone()
            z = z + (sig[j + 1] - sig[j]) * v  # Euler step on the rectified path
        return RolloutRecord(z, z_q, float(sig[q]), q, v_old_q, sig)

    @torch.no_grad()
    def continue_from(self, z_q, q_index, forced_y, prompt_embeds, negative_embeds):
        sig = self._sigmas(z_q.device)
        v = (z_q - forced_y) / sig[q_index]
        z = z_q + (sig[q_index + 1] - sig[q_index]) * v
        for j in range(q_index + 1, self.num_steps):
            z = z + (sig[j + 1] - sig[j]) * self.velocity(z, sig[j], prompt_embeds, negative_embeds)
        return z

    def decode01(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        lat = latents / std + mean
        video = vae.decode(lat.to(vae.dtype), return_dict=False)[0]  # [B,3,T,H,W] in [-1,1]
        return (video / 2 + 0.5).clamp(0, 1).float().permute(0, 2, 1, 3, 4)
```

```python
# config/wan_video.py
import ml_collections
from config.base import get_config as base_config


def get_config():
    config = base_config()
    config.pretrained.model = "${VIDEO_REWARD_CKPT_PATH}/Wan2.1-T2V-1.3B-Diffusers"
    config.resolution = 480
    config.video = video = ml_collections.ConfigDict()
    video.width = 832; video.height = 480; video.num_frames = 17
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
    config.opa.rho = 0.10; config.opa.n_ascent = 2; config.opa.eta = 1.0
    config.opa.query_sigma = 0.278
    config.opa.mb = 1                                # target-construction microbatch (spec Sec. 5.2)
    config.gates = gates = ml_collections.ConfigDict()
    gates.tau_id = -1.0; gates.tau_motion = -1.0     # -1 = calibrate from base-model 10th percentile at start
    gates.tau_q = 0.4
    gates.calib_prompts = 64
    config.judge = ml_collections.ConfigDict(); config.judge.device = "cuda:1"; config.judge.num_frames = 8
    config.reward_device = "cuda:1"
    config.prompt_fn = "text_file"
    config.prompt_fn_kwargs = {"path": "data/video_motion/train.txt"}
    config.eval_prompts = "data/video_motion/test.txt"
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/video/test_wan_clean_output.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/wan_clean_output.py config/wan_video.py tests/video/test_wan_clean_output.py
git commit -m "feat(video): Wan 2.1 clean-output map, query-capturing rollout, fixed-suffix continuation"
```

---

### Task 8: Fixed-suffix probe (go/no-go)

**Files:**
- Create: `src/video/probe_stats.py`
- Create: `scripts/probe_fixed_suffix_video.py`
- Test: `tests/video/test_probe_stats.py`

**Interfaces:**
- Consumes: `opa_tr_step_nd` (T3), `GeoReward` + loaders (T4/T5), `WanRollout`, `clean_output` (T7).
- Produces: `probe_summary(records: list[dict]) -> dict` with keys `n`, `median_G_construct`, `frac_alignment_pos`, `mean_G_realized`, `mean_G_fit`, `pass` (bool: `median_G_construct > 0 and frac_alignment_pos >= 0.6 and mean_G_realized > 0`). Per-record keys: `G_construct`, `alignment`, `G_realized`, `G_fit`.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_probe_stats.py
from diffusionopsd.video.probe_stats import probe_summary


def test_probe_summary_pass_criteria():
    recs = [{"G_construct": 0.02, "alignment": 0.5, "G_realized": 0.001, "G_fit": 0.019}] * 7 + \
           [{"G_construct": -0.01, "alignment": -0.2, "G_realized": -0.002, "G_fit": -0.008}] * 3
    s = probe_summary(recs)
    assert s["n"] == 10 and s["median_G_construct"] > 0 and abs(s["frac_alignment_pos"] - 0.7) < 1e-9
    assert s["pass"] is True
    s2 = probe_summary(recs[7:])
    assert s2["pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/video/test_probe_stats.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/probe_stats.py
"""Aggregate the fixed-suffix probe (spec Sec. 6; paper Eq. 19)."""
from __future__ import annotations
import statistics


def probe_summary(records: list[dict]) -> dict:
    n = len(records)
    med = statistics.median(r["G_construct"] for r in records)
    frac = sum(r["alignment"] > 0 for r in records) / n
    real = sum(r["G_realized"] for r in records) / n
    fit = sum(r["G_fit"] for r in records) / n
    return {"n": n, "median_G_construct": med, "frac_alignment_pos": frac, "mean_G_realized": real,
            "mean_G_fit": fit, "pass": bool(med > 0 and frac >= 0.6 and real > 0)}
```

```python
# scripts/probe_fixed_suffix_video.py
#!/usr/bin/env python3
"""Same-query probe: G_construct, alignment, G_realized, G_fit with the geometry reward on a Wan 2.1 policy."""
from __future__ import annotations
import argparse, copy, json
import torch
from diffusers import WanPipeline
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.opa_video import opa_tr_step_nd
from diffusionopsd.video.probe_stats import probe_summary
from diffusionopsd.video.wan_clean_output import WanRollout, clean_output
from config.wan_video import get_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True); p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="probe_fixed_suffix.jsonl"); p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()
    cfg = get_config(); dev = "cuda:0"
    pipe = WanPipeline.from_pretrained(cfg.pretrained.model, torch_dtype=torch.bfloat16).to(dev)
    roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device))
    R = lambda lat: reward(roll.decode01(lat).to(cfg.reward_device)).geo.to(lat.device)
    base_state = copy.deepcopy(pipe.transformer.state_dict())
    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n]
    recs = []
    for prompt in prompts:
        pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev)[:2]
        shape = (1, pipe.transformer.config.in_channels, (cfg.video.num_frames - 1) // 4 + 1, cfg.video.height // 8, cfg.video.width // 8)
        z_T = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        rec = roll.rollout(pe, ne, z_T, cfg.opa.query_sigma)
        y0 = clean_output(rec.z_q, rec.v_old_q, torch.tensor([rec.sigma_q], device=dev)).float()
        Fq = lambda y: R(roll.continue_from(rec.z_q, rec.q_index, y.to(rec.z_q.dtype), pe, ne))
        # construction
        y_plus = opa_tr_step_nd(y0, R, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, +1.0)
        F0, Fplus = float(Fq(y0)), float(Fq(y_plus))
        yg = y0.clone().requires_grad_(True); (g_local,) = torch.autograd.grad(R(yg).sum(), yg)
        yf = y0.clone().requires_grad_(True); (g_suffix,) = torch.autograd.grad(Fq(yf).sum(), yf)
        align = float(torch.nn.functional.cosine_similarity(g_local.flatten(), g_suffix.flatten(), dim=0))
        # one fresh fitting update on a copy of the policy (positive branch only, paper Sec. 4.5 protocol)
        pipe.transformer.load_state_dict(base_state)
        opt = torch.optim.AdamW(pipe.transformer.parameters(), lr=a.lr)
        v_theta = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device=dev), pe, ne)
        y_theta = clean_output(rec.z_q, v_theta, torch.tensor([rec.sigma_q], device=dev)).float()
        wf = (y_theta - y_plus).abs().mean().clamp(min=1e-5).detach()
        (((y_theta - y_plus) ** 2) / wf).mean().backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            v_after = roll.velocity(rec.z_q, torch.tensor(rec.sigma_q, device=dev), pe, ne)
            y_after = clean_output(rec.z_q, v_after, torch.tensor([rec.sigma_q], device=dev)).float()
            Fafter = float(Fq(y_after))
        pipe.transformer.load_state_dict(base_state)
        r = {"prompt": prompt, "G_construct": Fplus - F0, "alignment": align, "G_realized": Fafter - F0, "G_fit": Fplus - Fafter}
        recs.append(r); print(json.dumps(r))
        with open(a.out, "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps(probe_summary(recs), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test**

Run: `pytest tests/video/test_probe_stats.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/probe_stats.py scripts/probe_fixed_suffix_video.py tests/video/test_probe_stats.py
git commit -m "feat(video): fixed-suffix go/no-go probe for the geometry reward"
```

- [ ] **Step 6: Run the probe (GPU, weights required)**

Run: `python scripts/probe_fixed_suffix_video.py --prompts data/video_motion/test.txt --n 64`
Expected: final JSON with `"pass": true`. If `pass` is false, stop here and revisit spec Sec. 4.1 (e.g. `residual_scale`, `rho`) before Task 9.

---

### Task 9: Training script

**Files:**
- Create: `src/video/branch_loss.py`
- Create: `scripts/train_opsd_video_wan.py`
- Create: `data/video_motion/README.md`, `data/video_motion/train.txt`, `data/video_motion/test.txt`
- Test: `tests/video/test_train_step_math.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `branch_loss(y_theta, y0, y_plus, y_minus, w_eff, beta) -> Tensor` (pure function, tested), CLI `python scripts/train_opsd_video_wan.py --config config/wan_video.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_train_step_math.py
import torch
from diffusionopsd.video.branch_loss import branch_loss


def test_branch_loss_weights_and_zero_mask():
    y0 = torch.zeros(2, 4, 2, 3, 3); y_plus = y0 + 1.0; y_minus = y0 - 1.0
    y_theta = y0 + 0.5
    w = torch.tensor([1.0, 0.0])
    l = branch_loss(y_theta, y0, y_plus, y_minus, w, beta=1.0)
    assert l.shape == (2,)
    # sample 0: only positive branch -> ((0.5-1)^2)/|0.5-1| = 0.5 ; sample 1: only negative branch -> ((0.5+1)^2)/1.5 = 1.5
    assert torch.allclose(l, torch.tensor([0.5, 1.5]), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/video/test_train_step_math.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/branch_loss.py
"""Paper Eq. 11-13 with the masked endpoint weight from gates.effective_weight."""
from __future__ import annotations
import torch


def branch_loss(y_theta, y0, y_plus, y_minus, w_eff, beta: float) -> torch.Tensor:
    rd = tuple(range(1, y0.ndim))
    y_pos = beta * y_theta + (1 - beta) * y0
    y_neg = (1 + beta) * y0 - beta * y_theta
    with torch.no_grad():
        wf_p = (y_pos - y_plus).abs().mean(dim=rd, keepdim=True).clamp(min=1e-5)
        wf_n = (y_neg - y_minus).abs().mean(dim=rd, keepdim=True).clamp(min=1e-5)
    pos = ((y_pos - y_plus) ** 2 / wf_p).mean(dim=rd)
    neg = ((y_neg - y_minus) ** 2 / wf_n).mean(dim=rd)
    return w_eff * pos + (1 - w_eff) * neg
```

```python
# scripts/train_opsd_video_wan.py
#!/usr/bin/env python3
"""DiffusionOPSD for Wan 2.1 with the geometry reward (spec Sec. 5.2). Single-node; policy on cuda:0, rewards on cuda:1."""
from __future__ import annotations
import copy, json, os, random
from absl import app, flags
from ml_collections import config_flags
import torch
from diffusers import WanPipeline
from diffusionopsd.video.branch_loss import branch_loss
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.gates import effective_weight, identity_keep, percentile_threshold
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.opa_video import opa_tr_step_nd
from diffusionopsd.video.quality_judge import load_qwen25_vl
from diffusionopsd.video.wan_clean_output import WanRollout, clean_output
from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.stat_tracking import PerPromptStatTracker

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/wan_video.py")


def _latent_shape(pipe, cfg):
    return (1, pipe.transformer.config.in_channels, (cfg.video.num_frames - 1) // 4 + 1, cfg.video.height // 8, cfg.video.width // 8)


def main(_):
    cfg = FLAGS.config; dev = "cuda:0"
    pipe = WanPipeline.from_pretrained(os.path.expandvars(cfg.pretrained.model), torch_dtype=torch.bfloat16).to(dev)
    pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    policy = pipe.transformer; behavior = copy.deepcopy(policy).requires_grad_(False)
    roll_old = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device))
    judge = load_qwen25_vl(cfg.judge.device)
    R_geo = lambda lat: reward(roll_old.decode01(lat).to(cfg.reward_device)).geo.to(lat.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.adam_weight_decay)
    ema = EMAModuleWrapper(policy.parameters(), decay=0.99, update_step_interval=1, device=dev)
    tracker = PerPromptStatTracker(cfg.sample.global_std)
    prompts = [l.strip() for l in open(cfg.prompt_fn_kwargs["path"]) if l.strip()]
    K = cfg.sample.num_image_per_prompt

    # fixed base-model reference clips + threshold calibration (spec Sec. 4.4, Sec. 7)
    refs, s_ids, motions = {}, [], []
    with torch.no_grad():
        for prompt in random.Random(0).sample(prompts, min(cfg.gates.calib_prompts, len(prompts))):
            pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev)[:2]
            g = torch.Generator(dev).manual_seed(abs(hash(prompt)) % 2**31)
            rec = roll_old.rollout(pe, ne, torch.randn(_latent_shape(pipe, cfg), device=dev, dtype=torch.bfloat16, generator=g), cfg.opa.query_sigma)
            clip = roll_old.decode01(rec.x0); refs[prompt] = clip[0].cpu()
            out = reward(clip.to(cfg.reward_device)); s_ids.append(out.s_id.cpu()); motions.append(out.motion.cpu())
    tau_id = cfg.gates.tau_id if cfg.gates.tau_id >= 0 else percentile_threshold(torch.cat(s_ids), 0.10)
    tau_motion = cfg.gates.tau_motion if cfg.gates.tau_motion >= 0 else percentile_threshold(torch.cat(motions), 0.10)
    print(json.dumps({"tau_id": tau_id, "tau_motion": tau_motion, "tau_q": cfg.gates.tau_q}))

    for epoch in range(cfg.num_epochs):
        batch_prompts = random.sample(prompts, cfg.sample.num_batches_per_epoch)
        tuples = []
        # 1) rollouts with the frozen behavior policy
        pipe.transformer = behavior
        for prompt in batch_prompts:
            pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev)[:2]
            if prompt not in refs:
                with torch.no_grad():
                    g = torch.Generator(dev).manual_seed(abs(hash(prompt)) % 2**31)
                    refs[prompt] = roll_old.decode01(roll_old.rollout(pe, ne, torch.randn(_latent_shape(pipe, cfg), device=dev, dtype=torch.bfloat16, generator=g), cfg.opa.query_sigma).x0)[0].cpu()
            for _ in range(K):
                rec = roll_old.rollout(pe, ne, torch.randn(_latent_shape(pipe, cfg), device=dev, dtype=torch.bfloat16), cfg.opa.query_sigma)
                with torch.no_grad():
                    clip = roll_old.decode01(rec.x0); out = reward(clip.to(cfg.reward_device))
                    p_q = judge.p_win(clip[0].to(cfg.judge.device), refs[prompt].to(cfg.judge.device), prompt)
                tuples.append({"prompt": prompt, "pe": pe, "ne": ne, "rec": rec, "r": float(out.geo), "m": float(out.motion), "p_q": p_q})
        # 2) endpoint weights
        adv = tracker.update([t["prompt"] for t in tuples], torch.tensor([t["r"] for t in tuples]).numpy())
        adv = torch.tensor(adv, dtype=torch.float32)
        for t, a in zip(tuples, adv):
            t["w"] = float(effective_weight(a.view(1), cfg.train.adv_clip_max, torch.tensor([t["m"]]), tau_motion, torch.tensor([t["p_q"]]), cfg.gates.tau_q))
        # 3) identity gate + targets
        kept = []
        for t in tuples:
            rec = t["rec"]
            y0 = clean_output(rec.z_q, rec.v_old_q, torch.tensor([rec.sigma_q], device=dev)).float()
            with torch.no_grad():
                s_id = reward(roll_old.decode01(y0).to(cfg.reward_device)).s_id
            if not bool(identity_keep(s_id, tau_id).all()):
                continue
            yg = y0.clone().requires_grad_(True); (g0,) = torch.autograd.grad(R_geo(yg).sum(), yg)
            t["y0"] = y0
            t["y_plus"] = opa_tr_step_nd(y0, R_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, +1.0, first_grad=g0)
            t["y_minus"] = opa_tr_step_nd(y0, R_geo, cfg.opa.rho, cfg.opa.n_ascent, cfg.opa.eta, -1.0, first_grad=g0)
            kept.append(t)
        # 4) one finite-fitting update (M_fit = 1)
        pipe.transformer = policy; policy.train(); opt.zero_grad()
        for t in kept:
            rec = t["rec"]
            v_theta = roll_old.velocity(rec.z_q, torch.tensor(rec.sigma_q, device=dev), t["pe"], t["ne"], transformer=policy)
            y_theta = clean_output(rec.z_q, v_theta, torch.tensor([rec.sigma_q], device=dev)).float()
            loss = branch_loss(y_theta, t["y0"], t["y_plus"], t["y_minus"], torch.tensor([t["w"]], device=dev), cfg.beta).mean()
            (loss * cfg.train.adv_clip_max / max(len(kept), 1)).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.max_grad_norm)
        opt.step(); ema.step(policy.parameters(), epoch)
        # 5) behavior EMA refresh (paper Eq. 18)
        with torch.no_grad():
            for pb, pp in zip(behavior.parameters(), policy.parameters()):
                pb.mul_(0.99).add_(pp.detach(), alpha=0.01)
        print(json.dumps({"epoch": epoch, "n_rollouts": len(tuples), "n_kept": len(kept),
                          "mean_r": sum(t["r"] for t in tuples) / len(tuples),
                          "mean_w": sum(t["w"] for t in tuples) / len(tuples),
                          "frac_motion_masked": sum(t["m"] < tau_motion for t in tuples) / len(tuples),
                          "frac_quality_masked": sum(t["p_q"] < cfg.gates.tau_q for t in tuples) / len(tuples)}))
        if (epoch + 1) % cfg.save_freq == 0:
            os.makedirs(cfg.logdir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(cfg.logdir, f"policy_{epoch+1}.pt"))


if __name__ == "__main__":
    app.run(main)
```

`data/video_motion/README.md`: one paragraph stating that `train.txt` and `test.txt` are one prompt per line, motion-explicit (camera pan/orbit/dolly or object translation), disjoint, and that `test.txt` is drawn from the camera-control and 3D-consistency subsets of WorldScore and VBench-2.0 prompt suites. Seed both files with at least 200 training and 64 test prompts before running Task 8 Step 6.

- [ ] **Step 4: Run unit test**

Run: `pytest tests/video/test_train_step_math.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/branch_loss.py scripts/train_opsd_video_wan.py data/video_motion/
git add tests/video/test_train_step_math.py
git commit -m "feat(video): DiffusionOPSD trainer for Wan 2.1 with geometry reward and masked weight"
```

- [ ] **Step 6: Short GPU dry run**

Run: `python scripts/train_opsd_video_wan.py --config config/wan_video.py --config.num_epochs 2 --config.sample.num_batches_per_epoch 1`
Expected: a `tau_*` JSON line, then two epoch JSON lines with `n_kept >= 1` and finite `mean_r`.

---

### Task 10: Held-out evaluation

**Files:**
- Create: `src/video/eval_table.py`
- Create: `scripts/eval_video_consistency.py`
- Test: `tests/video/test_eval_table.py`

**Interfaces:**
- Consumes: `GeoReward`, `PairwiseVLMJudge`, `WanRollout`.
- Produces: `success_table(base: dict, tuned: dict) -> dict` where each input has keys `geo`, `s_id`, `motion` (means) and `tuned` also has `p_q` (mean win-prob vs base). Output keys `geometry_up`, `identity_ok`, `quality_ok`, `not_frozen`, `all_pass` with thresholds from spec Sec. 7: `s_id >= base - 0.02`, `p_q >= 0.45`, `motion >= 0.9 * base`.

- [ ] **Step 1: Write the failing test**

```python
# tests/video/test_eval_table.py
from diffusionopsd.video.eval_table import success_table


def test_success_table_thresholds():
    base = {"geo": -0.30, "s_id": 0.80, "motion": 4.0}
    ok = {"geo": -0.25, "s_id": 0.79, "motion": 3.7, "p_q": 0.46}
    t = success_table(base, ok)
    assert t == {"geometry_up": True, "identity_ok": True, "quality_ok": True, "not_frozen": True, "all_pass": True}
    frozen = dict(ok, motion=3.0)
    assert success_table(base, frozen)["not_frozen"] is False and success_table(base, frozen)["all_pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/video/test_eval_table.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/video/eval_table.py
"""Spec Sec. 7 automatic success criteria."""
from __future__ import annotations


def success_table(base: dict, tuned: dict) -> dict:
    t = {
        "geometry_up": tuned["geo"] > base["geo"],
        "identity_ok": tuned["s_id"] >= base["s_id"] - 0.02,
        "quality_ok": tuned["p_q"] >= 0.45,
        "not_frozen": tuned["motion"] >= 0.9 * base["motion"],
    }
    t["all_pass"] = all(t.values())
    return t
```

```python
# scripts/eval_video_consistency.py
#!/usr/bin/env python3
"""Held-out comparison of a tuned policy against the base model on the same prompts and seeds."""
from __future__ import annotations
import argparse, json, os
import torch
from diffusers import WanPipeline
from diffusionopsd.video.estimators import load_depth_anything3, load_dinov2, load_waft
from diffusionopsd.video.eval_table import success_table
from diffusionopsd.video.geo_reward import GeoReward
from diffusionopsd.video.quality_judge import load_qwen25_vl
from diffusionopsd.video.wan_clean_output import WanRollout
from config.wan_video import get_config


def _gen(pipe, roll, cfg, prompt, seed, dev):
    pe, ne = pipe.encode_prompt(prompt, negative_prompt="", do_classifier_free_guidance=True, device=dev)[:2]
    shape = (1, pipe.transformer.config.in_channels, (cfg.video.num_frames - 1) // 4 + 1, cfg.video.height // 8, cfg.video.width // 8)
    z = torch.randn(shape, device=dev, dtype=torch.bfloat16, generator=torch.Generator(dev).manual_seed(seed))
    return roll.decode01(roll.rollout(pe, ne, z, cfg.opa.query_sigma).x0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True); p.add_argument("--prompts", default=None); p.add_argument("--out", default="eval_video.json")
    a = p.parse_args(); cfg = get_config(); dev = "cuda:0"
    pipe = WanPipeline.from_pretrained(os.path.expandvars(cfg.pretrained.model), torch_dtype=torch.bfloat16).to(dev)
    roll = WanRollout(pipe, cfg.sample.num_steps, cfg.sample.guidance_scale)
    reward = GeoReward(load_depth_anything3(cfg.reward_device), load_waft(cfg.reward_device), load_dinov2(cfg.reward_device))
    judge = load_qwen25_vl(cfg.judge.device)
    prompts = [l.strip() for l in open(a.prompts or cfg.eval_prompts) if l.strip()]
    base_sd = {k: v.clone() for k, v in pipe.transformer.state_dict().items()}
    tuned_sd = torch.load(a.policy, map_location=dev)
    acc = {"base": {"geo": [], "s_id": [], "motion": []}, "tuned": {"geo": [], "s_id": [], "motion": [], "p_q": []}}
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            pipe.transformer.load_state_dict(base_sd); cb = _gen(pipe, roll, cfg, prompt, i, dev)
            pipe.transformer.load_state_dict(tuned_sd); ct = _gen(pipe, roll, cfg, prompt, i, dev)
            ob, ot = reward(cb.to(cfg.reward_device)), reward(ct.to(cfg.reward_device))
            for k in ("geo", "s_id", "motion"):
                acc["base"][k].append(float(getattr(ob, k))); acc["tuned"][k].append(float(getattr(ot, k)))
            acc["tuned"]["p_q"].append(judge.p_win(ct[0].to(cfg.judge.device), cb[0].to(cfg.judge.device), prompt))
    means = {s: {k: sum(v) / len(v) for k, v in d.items()} for s, d in acc.items()}
    result = {"n": len(prompts), "means": means, "success": success_table(means["base"], means["tuned"])}
    json.dump(result, open(a.out, "w"), indent=2); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test**

Run: `pytest tests/video/test_eval_table.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/video/eval_table.py scripts/eval_video_consistency.py tests/video/test_eval_table.py
git commit -m "feat(video): held-out consistency evaluation and success table"
```

- [ ] **Step 6: Full suite**

Run: `pytest tests/video -q && ruff check src/video scripts/*video*.py scripts/probe_fixed_suffix_video.py`
Expected: all tests pass, ruff clean.

---

## Spec coverage

| Spec section | Task |
|---|---|
| 4.1 geometry reward, frozen estimators, differentiable through decoder | 1, 4, 5, 7 (`decode01`) |
| 4.2 identity gate on the anchor | 2, 9 (step 3) |
| 4.3 motion floor | 2, 5 (`motion`), 9 |
| 4.4 quality mask, fixed base references | 6, 9 (refs dict) |
| 4.5 effective weight in Eq. 13 | 2, 9 (`branch_loss`) |
| 5.1 assets / downloads, no bundled weights | 4 |
| 5.2 algorithm deltas, microbatch 1, rewards on separate GPU | 7, 9, `config/wan_video.py` |
| 5.3 motion-explicit prompts, WorldScore/VBench-2.0 held-out | 9 (`data/video_motion/`) |
| 6 fixed-suffix probe and pass criteria | 8 |
| 7 success criteria and threshold calibration | 9 (calibration), 10 |

Not covered by code, by design: the external WorldScore / VBench-2.0 dimension scores in spec Sec. 7 are produced by those benchmarks' own toolkits on the clips saved from `eval_video_consistency.py`; add `--save_dir` to that script when wiring them in.
