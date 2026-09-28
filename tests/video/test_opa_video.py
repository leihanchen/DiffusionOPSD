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
