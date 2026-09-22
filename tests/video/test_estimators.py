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
