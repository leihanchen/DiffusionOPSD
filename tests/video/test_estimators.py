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


class _Prediction:
    def __init__(self):
        self.depth = torch.ones(2, 4, 6).numpy()
        self.conf = torch.full((2, 4, 6), 2.0).numpy()
        self.intrinsics = torch.tensor(
            [[[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]] * 2
        ).numpy()
        w2c = torch.eye(4).repeat(2, 1, 1)
        w2c[1, 0, 3] = -0.2
        self.extrinsics = w2c.numpy()


class _StubDA3:
    def inference(self, images, process_res=None):
        assert len(images) == 2
        assert images[0].shape == (8, 10, 3)
        return _Prediction()


def test_da3_adapter_maps_prediction_to_frame_space():
    from diffusionopsd.video.estimators import DepthAnything3

    out = DepthAnything3(_StubDA3(), process_res=14)(torch.rand(1, 2, 3, 8, 10))
    assert out.depth.shape == (1, 2, 8, 10)
    assert out.conf.shape == (1, 2, 8, 10)
    assert torch.equal(out.conf, torch.ones(1, 2, 8, 10))
    assert torch.isclose(out.K[0, 0, 0], torch.tensor(10.0))
    assert torch.isclose(out.poses[0, 1, 0, 3], torch.tensor(0.2))
