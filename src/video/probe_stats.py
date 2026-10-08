"""Aggregate the fixed-suffix probe (spec Sec. 6; paper Eq. 19)."""
from __future__ import annotations

import statistics


def probe_summary(records: list[dict]) -> dict:
    n = len(records)
    med = statistics.median(r["G_construct"] for r in records)
    frac = sum(r["alignment"] > 0 for r in records) / n
    real = sum(r["G_realized"] for r in records) / n
    fit = sum(r["G_fit"] for r in records) / n
    return {"n": n, "median_G_construct": med, "frac_alignment_pos": frac, "mean_G_realized": real,
            "mean_G_fit": fit, "pass": bool(med > 0 and frac >= 0.6 and real > 0)}
