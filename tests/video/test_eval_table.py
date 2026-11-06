from diffusionopsd.video.eval_table import success_table


def test_success_table_thresholds():
    base = {"geo": -0.30, "s_id": 0.80, "motion": 4.0}
    ok = {"geo": -0.25, "s_id": 0.79, "motion": 3.7, "p_q": 0.46}
    t = success_table(base, ok)
    assert t == {
        "geometry_up": True,
        "identity_ok": True,
        "quality_ok": True,
        "not_frozen": True,
        "all_pass": True,
    }
    frozen = dict(ok, motion=3.0)
    assert success_table(base, frozen)["not_frozen"] is False
    assert success_table(base, frozen)["all_pass"] is False


def test_reward_versions_prevent_cross_objective_comparisons():
    import pytest
    base = dict(geo=-0.3, s_id=0.8, motion=4)
    tuned = dict(geo=-0.2, s_id=0.8, motion=4, p_q=0.5, reward_version="geoflow-server")
    with pytest.raises(ValueError, match="reward"):
        success_table(base, tuned)
