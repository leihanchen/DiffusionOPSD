from diffusionopsd.video.probe_stats import probe_summary


def test_probe_summary_pass_criteria():
    recs = [{"G_construct": 0.02, "alignment": 0.5, "G_realized": 0.001, "G_fit": 0.019}] * 7 + \
           [{"G_construct": -0.01, "alignment": -0.2, "G_realized": -0.002, "G_fit": -0.008}] * 3
    s = probe_summary(recs)
    assert s["n"] == 10 and s["median_G_construct"] > 0 and abs(s["frac_alignment_pos"] - 0.7) < 1e-9
    assert s["pass"] is True
    s2 = probe_summary(recs[7:])
    assert s2["pass"] is False
