"""Oracle tests against the independently loaded, pinned GeoFlow release.

Set GEOFLOW_REFERENCE to a checkout at the revision in geo_reward.py.
The AST loader executes only pure helper definitions, avoiding model downloads.
"""
import ast
import os
from pathlib import Path
import subprocess

import pytest
import torch

from diffusionopsd.video.geo_reward import GeoReward
from test_geo_reward import StubDepth, StubFlow, StubFeats


@pytest.fixture
def oracle():
    root = os.environ.get("GEOFLOW_REFERENCE")
    if not root:
        pytest.skip("Set GEOFLOW_REFERENCE to the pinned GeoFlow checkout")
    assert subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() == (
        "fac2426820e303eb1c0ffd1cc43e58585f99f87b"
    )
    tree = ast.parse((Path(root) / "geoflow/reward_server.py").read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.lineno < 460], type_ignores=[])
    import numpy as np
    namespace = {"np": np}
    exec(compile(module, "pinned_geoflow_helpers", "exec"), namespace)
    return namespace


def test_identical_static_frames_have_zero_reference_reward():
    frames = torch.rand(1, 1, 3, 28, 42).repeat(1, 3, 1, 1, 1)
    out = GeoReward(StubDepth(), StubFlow(0), StubFeats())(frames)
    torch.testing.assert_close(out.geo, torch.zeros(1), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out.s_id, torch.ones(1), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("key,value", [("residual_scale", 2), ("w_rigid", 0.7), ("w_dino", 0.2)])
def test_legacy_scale_override_is_rejected(key, value):
    with pytest.raises(ValueError, match="GeoFlow"):
        GeoReward(StubDepth(), StubFlow(0), StubFeats(), **{key: value})


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("confidence", [False, True])
def test_pair_scores_match_pinned_oracle(oracle, empty, confidence):
    from diffusionopsd.video.geoflow_math import pair_rewards

    g = torch.Generator().manual_seed(71)
    flow = torch.rand(2, 2, 16, 24, generator=g) * 8 - 4
    rigid = torch.rand(2, 2, 16, 24, generator=g) * 5
    target = torch.rand(2, 16, 24, generator=g) + 0.5
    warped = target * 1.2
    warped[:, :3] = 0
    if empty:
        warped.zero_()
    src = torch.rand(2, 4, 2, 3, generator=g)
    dst = torch.rand(2, 4, 2, 3, generator=g)
    valid = torch.ones(2, 2, 3)
    valid[:, 0] = 0
    if empty:
        valid.zero_()
    conf = torch.rand(2, 16, 24, generator=g) if confidence else None
    noc = torch.ones_like(target)
    noc[:, :10] = 0
    expected = oracle["compute_pair_rewards"](
        flow, rigid, target, warped, src.permute(0, 2, 3, 1), dst.permute(0, 2, 3, 1), valid,
        conf_pair=conf, noc_mask=noc,
    )
    actual = pair_rewards(flow, rigid, target, warped, src, dst, valid, conf, noc)
    for key in ("r_geo", "r_dino", "r_combined", "dino_counts"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-4)


def test_geometry_matches_pinned_oracle(oracle):
    from diffusionopsd.video.geoflow_math import project_depth, warp_depth, depth_visibility

    depth = torch.ones(2, 16, 24) * 2
    depth[:, :2] = 0
    source = torch.eye(3).repeat(2, 1, 1)
    target = source.clone()
    target[:, 0, 0] = 1.2
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, 0, 3] = 0.4
    pose[1, 2, 3] = -3  # behind the camera
    rigid, projected = project_depth(depth, source, target, pose)
    expected = oracle["depth_pose_to_flow_batch"](depth, source, target, pose[:, :3, :3], pose[:, :3, 3], "cpu")
    torch.testing.assert_close(rigid, expected, atol=1e-5, rtol=1e-4)
    expected_depth = oracle["warp_depth_batch"](depth, source, target, pose[:, :3, :3], pose[:, :3, 3], "cpu")
    torch.testing.assert_close(warp_depth(rigid, projected), expected_depth, atol=1e-5, rtol=1e-4)
    expected_mask = oracle["depth_occlusion_mask"](
        depth, depth, source, pose[:, :3, :3], pose[:, :3, 3], rigid, "cpu"
    )
    torch.testing.assert_close(depth_visibility(projected, depth, rigid), expected_mask)


@pytest.mark.parametrize("length", [2, 9, 17])
def test_complete_server_score_with_identical_estimators(oracle, length):
    """Execute upstream score_video itself, including preprocessing and long pairs."""
    import cv2
    import numpy as np
    import time
    from types import SimpleNamespace
    import torch.nn.functional as F
    from diffusionopsd.video.geo_reward import quantize_frames

    root = Path(os.environ["GEOFLOW_REFERENCE"])
    tree = ast.parse((root / "geoflow/reward_server.py").read_text())
    worker_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastRewardWorker")
    namespace = dict(oracle, cv2=cv2, np=np, time=time)
    exec(compile(ast.Module(body=[worker_node], type_ignores=[]), "pinned_server", "exec"), namespace)
    worker = namespace["FastRewardWorker"].__new__(namespace["FastRewardWorker"])
    worker.torch, worker.TF, worker.device = torch, F, "cpu"
    worker.TVF = SimpleNamespace(resize=lambda x, size, antialias: F.interpolate(
        x[None] if x.ndim == 3 else x, size=size, mode="bilinear", align_corners=False, antialias=antialias
    ).squeeze(0) if x.ndim == 3 else F.interpolate(x, size=size, mode="bilinear", align_corners=False,
                                                 antialias=antialias))
    worker.process_res = 504
    worker.use_confidence_gating = worker.use_occlusion_masking = worker.use_dynamic_masking = False
    worker.use_long_pairs = True
    worker.long_pair_gap, worker.long_pair_stride = 8, 4
    worker.long_pair_max_flow, worker.long_pair_geo_weight = 4.0, 0.25
    worker.min_weight, worker.motion_floor_px = 0.3, 0.0
    frames = torch.rand(1, length, 3, 16, 24, generator=torch.Generator().manual_seed(4))
    depth = StubDepth(tx=0.01)
    prediction = depth(frames)
    worker._da3 = SimpleNamespace(inference=lambda *args, **kwargs: SimpleNamespace(
        depth=prediction.depth[0].numpy(), intrinsics=prediction.K[0].repeat(length, 1, 1).numpy(),
        extrinsics=torch.linalg.inv(prediction.poses[0])[:, :3].numpy(), conf=prediction.conf[0].numpy(),
    ))
    flow = StubFlow(0.6)
    def feats(x):
        return F.avg_pool2d(x, 4)
    worker._waft_flow_batch = flow
    worker._dino_extract_batch = lambda x: (feats(x).permute(0, 2, 3, 1), 4, 6)
    rgb8 = quantize_frames(frames)[0].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    pngs = [cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes() for frame in rgb8]
    with torch.no_grad():
        expected = worker.score_video(pngs, fps=8)
        actual = GeoReward(depth, flow, feats)(frames)
    for value, name in ((actual.geo, "score"), (actual.rigid, "mean_r_geo"), (actual.dino, "mean_r_dino")):
        torch.testing.assert_close(value, torch.tensor([expected[name]]), atol=1e-5, rtol=1e-4)


def test_quantization_preserves_rgb8_values_and_straight_through_gradient():
    from diffusionopsd.video.geo_reward import quantize_frames
    x = torch.tensor([0.101, 0.501, 0.999], requires_grad=True)
    y = quantize_frames(x)
    torch.testing.assert_close(y, x.detach().mul(255).round().div(255), atol=0, rtol=0)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_partial_gradient_matches_finite_difference_away_from_masks():
    from diffusionopsd.video.geoflow_math import pair_rewards, warp_image
    torch.manual_seed(15)
    source = torch.rand(1, 3, 4, 5, dtype=torch.double)
    target = torch.rand_like(source)
    rigid = torch.ones(1, 2, 4, 5, dtype=torch.double) * 0.12
    depth = torch.ones(1, 4, 5, dtype=torch.double)
    flow = torch.full_like(rigid, 0.31, requires_grad=True)

    def score(value):
        return pair_rewards(value, rigid, depth, depth, source, warp_image(target, value),
                            torch.ones_like(depth))["r_combined"]
    assert torch.autograd.gradcheck(score, (flow,), atol=1e-5, rtol=1e-4)


def test_latent_targets_have_gradients_with_detached_depth():
    from diffusionopsd.video.opa_video import opa_tr_step_nd
    import torch.nn.functional as F

    class DetachedDepth(StubDepth):
        def __call__(self, frames):
            assert not torch.is_grad_enabled()
            return super().__call__(frames)

    class Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
        def forward(self, a, b):
            return (a[:, :2] - b[:, :2]) * self.scale + 0.3

    flow = Flow()
    reward = GeoReward(DetachedDepth(), flow, lambda x: F.avg_pool2d(x, 4))
    latent = torch.randn(1, 2, 3, 16, 24, requires_grad=True)
    def score(z):
        return reward(z.sigmoid()).geo
    grad, = torch.autograd.grad(score(latent).sum(), latent)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert flow.scale.grad is None and not flow.scale.requires_grad
    for direction in (1, -1):
        target = opa_tr_step_nd(latent.detach(), score, rho=0.01, n_ascent=1, eta=1, direction=direction)
        assert torch.isfinite(target).all()
        assert not target.requires_grad
        assert ((target - latent.detach()) * grad).sum() * direction > 0


def test_custom_locked_config_receives_reward_provenance():
    import ml_collections
    from diffusionopsd.video.geo_reward import configure_reward, REWARD_VERSION
    config = ml_collections.ConfigDict()
    config.lock()
    configure_reward(config)
    assert config.geometry_reward.version == REWARD_VERSION
    assert config.is_locked
