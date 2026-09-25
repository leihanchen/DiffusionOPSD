"""Spec Sec. 7 automatic success criteria."""
from __future__ import annotations


def success_table(base: dict, tuned: dict) -> dict:
    t = {
        "geometry_up": tuned["geo"] > base["geo"],
        "identity_ok": tuned["s_id"] >= base["s_id"] - 0.02,
        "quality_ok": tuned["p_q"] >= 0.45,
        "not_frozen": tuned["motion"] >= 0.9 * base["motion"],
    }
    t["all_pass"] = all(t.values())
    return t
