import torch
import pytest
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
    def __init__(self, pose_rows=4):
        self.depth = torch.ones(2, 16, 24).numpy()
        self.conf = torch.full((2, 16, 24), 2.0).numpy()
        self.intrinsics = torch.tensor([[[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]] * 2).numpy()
        w2c = torch.eye(4).repeat(2, 1, 1)
        w2c[1, 0, 3] = -0.2
        self.extrinsics = w2c[:, :pose_rows].numpy()


class _StubDA3(torch.nn.Module):
    def __init__(self, pose_rows=4):
        super().__init__()
        self.pose_rows = pose_rows

    def inference(self, images, **kwargs):
        assert len(images) == 2
        assert images[0].shape == (8, 10, 3)
        return _Prediction(self.pose_rows)


@pytest.mark.parametrize("pose_rows", [3, 4])
def test_da3_adapter_preserves_server_processing_space(pose_rows):
    from diffusionopsd.video.estimators import DepthAnything3

    out = DepthAnything3(_StubDA3(pose_rows), process_res=14)(torch.rand(1, 2, 3, 8, 10))
    assert out.depth.shape == (1, 2, 16, 24)
    assert out.conf.shape == (1, 2, 16, 24)
    assert torch.equal(out.conf, torch.ones(1, 2, 16, 24))
    assert torch.isclose(out.K[0, 0, 0, 0], torch.tensor(6.0))
    assert out.poses.shape == (1, 2, 4, 4)
    assert torch.isclose(out.poses[0, 1, 0, 3], torch.tensor(0.2, dtype=torch.float64))


def test_da3_server_resolution_and_per_frame_intrinsics():
    from diffusionopsd.video.estimators import DepthAnything3

    class ServerDA3(torch.nn.Module):
        def inference(self, images, **kwargs):
            assert kwargs == dict(process_res=504, process_res_method="upper_bound_resize", export_dir=None)
            prediction = _Prediction()
            prediction.depth = torch.ones(2, 18, 26).numpy()
            prediction.conf = torch.ones(2, 18, 26).numpy()
            prediction.intrinsics[1, 0, 0] = 12
            return prediction

    out = DepthAnything3(ServerDA3())(torch.rand(1, 2, 3, 32, 48, requires_grad=True))
    assert out.depth.shape == (1, 2, 16, 24)
    assert out.K.shape == (1, 2, 3, 3)
    torch.testing.assert_close(out.K[0, :, 0, 0], torch.tensor([6, 12]) * (24 / 26))
    assert not out.depth.requires_grad


def test_dino_resize_matches_reference_antialias():
    import torch.nn.functional as F
    from diffusionopsd.video.estimators import _IMNET_MEAN, _IMNET_STD

    class Capture(torch.nn.Module):
        def forward_features(self, x):
            self.seen = x
            return {"x_norm_patchtokens": torch.ones(x.shape[0], 4, 8)}

    model = Capture()
    frames = torch.rand(1, 3, 31, 33, requires_grad=True)
    DinoV2Patches(model)(frames)
    expected = (F.interpolate(frames, (28, 28), mode="bilinear", align_corners=False, antialias=True)
                - _IMNET_MEAN) / _IMNET_STD
    torch.testing.assert_close(model.seen, expected)


def test_da3_loader_accepts_namespace_package(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from diffusionopsd.video import estimators

    package = ModuleType("depth_anything_3")
    package.__file__ = None
    package.__path__ = [str(tmp_path)]
    api = ModuleType("depth_anything_3.api")
    api.__file__ = str(tmp_path / "api.py")
    api.DepthAnything3 = object
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    monkeypatch.delenv("GEOFLOW_DA3_DIR", raising=False)
    monkeypatch.setenv("VIDEO_REWARD_CKPT_PATH", str(tmp_path))
    monkeypatch.setattr(estimators, "_require_revision", lambda *args: None)
    with pytest.raises(FileNotFoundError, match="weights missing"):
        estimators.load_depth_anything3("cpu")
