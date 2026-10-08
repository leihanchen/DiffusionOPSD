import torch
from diffusionopsd.video.rigid_flow import pixel_grid, rigid_flow, reprojection_residual


def _K(f=100.0, cx=8.0, cy=6.0):
    K = torch.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
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
    T = torch.eye(4)[None].clone()
    T[0, 0, 3] = 0.1
    flow = rigid_flow(depth, _K(f=100.0), T)
    # u' - u = f * tx / Z = 100 * 0.1 / 2 = 5 px, no vertical flow
    assert torch.allclose(flow[:, 0], torch.full_like(flow[:, 0], 5.0), atol=1e-4)
    assert torch.allclose(flow[:, 1], torch.zeros_like(flow[:, 1]), atol=1e-4)


def test_reprojection_residual_weights_by_confidence():
    flow = torch.zeros(1, 2, 4, 4)
    rigid = torch.zeros(1, 2, 4, 4)
    rigid[0, 0, 0, 0] = 8.0
    conf = torch.ones(1, 4, 4)
    assert torch.isclose(reprojection_residual(flow, rigid, conf), torch.tensor([8.0 / 16]))
    conf[0, 0, 0] = 0.0
    assert torch.isclose(reprojection_residual(flow, rigid, conf), torch.tensor([0.0]))


def test_rigid_flow_is_differentiable_wrt_depth():
    depth = torch.full((1, 6, 8), 2.0, requires_grad=True)
    T = torch.eye(4)[None].clone()
    T[0, 0, 3] = 0.1
    rigid_flow(depth, _K(), T).sum().backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
