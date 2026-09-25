import torch
from diffusionopsd.video.wan_clean_output import clean_output, select_query_index


def test_clean_output_recovers_data_on_rectified_path():
    torch.manual_seed(0)
    y = torch.randn(2, 4, 3, 5, 5)
    eps = torch.randn_like(y)
    sigma = torch.tensor([0.3, 0.7])
    s = sigma.view(-1, 1, 1, 1, 1)
    z = (1 - s) * y + s * eps
    v = eps - y
    assert torch.allclose(clean_output(z, v, sigma), y, atol=1e-5)


def test_select_query_index_nearest():
    sig = torch.tensor([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    assert select_query_index(sig, 0.278) == 3
