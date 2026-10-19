import torch
from diffusionopsd.video.branch_loss import branch_loss


def test_branch_loss_weights_and_zero_mask():
    y0 = torch.zeros(2, 4, 2, 3, 3)
    y_plus = y0 + 1.0
    y_minus = y0 - 1.0
    y_theta = y0 + 0.5
    w = torch.tensor([1.0, 0.0])
    loss = branch_loss(y_theta, y0, y_plus, y_minus, w, beta=1.0)
    assert loss.shape == (2,)
    # beta=1: y_pos=y_theta=0.5, y_neg=2*y0-y_theta=-0.5; wf_p=|0.5-1|=0.5, pos=0.5;
    # wf_n=|-0.5-(-1)|=0.5, neg=0.5; w=1 -> 0.5, w=0 -> 0.5 (Eq. 11 y_neg, not y_theta on neg branch)
    assert torch.allclose(loss, torch.tensor([0.5, 0.5]), atol=1e-6)
