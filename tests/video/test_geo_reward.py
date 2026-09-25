import torch
from diffusionopsd.video.estimators import DepthOutput
from diffusionopsd.video.geo_reward import GeoReward, warp_features


class StubDepth:
    def __init__(self, tx=0.0):
        self.tx = tx

    def __call__(self, frames01):
        B, T, _, H, W = frames01.shape
        depth = torch.full((B, T, H, W), 2.0) + 0.0 * frames01.mean(dim=2)  # keep graph
        K = torch.eye(3)[None].repeat(B, 1, 1)
        K[:, 0, 0] = K[:, 1, 1] = 50.0
        K[:, 0, 2] = W / 2
        K[:, 1, 2] = H / 2
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
