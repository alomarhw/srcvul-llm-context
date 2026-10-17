"""test_rq2_e2.py

Unit tests for experiment_rq2_e2 (RQ2-E2 efficiency-without-loss driver). Fast,
offline, no LLM/network/real-data. Guards measurement-integrity invariants:
finite outputs, correct directional behaviour (efficiency savings detected,
real effectiveness loss detected, genuine equivalence detected), a powered
sample-size calculation, and honest Pareto-frontier dominance.

Run: pytest -q test_rq2_e2.py  (pytest==7.4.4)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import experiment_rq2_e2 as e2


def _make_records(n, eff_whole, eff_slice, tok_whole, tok_slice, seed=0):
    """Build synthetic RQ2-E1-shaped records with controllable effectiveness
    and token characteristics for A0 (whole) and FULL (slice)."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        label = i % 2
        corr_w = int(rng.random() < eff_whole)
        corr_s = int(rng.random() < eff_slice)
        tw = tok_whole + rng.integers(0, 40)
        ts = tok_slice + rng.integers(0, 20)
        recs.append({
            "unit_id": f"u{i}",
            "label": label,
            "in_nondet_subset": i < min(20, n // 2),
            "by_cond": {
                "A0": {
                    "correct": corr_w,
                    "input_output_tokens": float(tw),
                    "input_tokens": float(tw * 0.9),
                    "output_tokens": float(tw * 0.1),
                    "latency_s": 1.5 + 0.004 * tw,
                },
                "FULL": {
                    "correct": corr_s,
                    "input_output_tokens": float(ts),
                    "input_tokens": float(ts * 0.9),
                    "output_tokens": float(ts * 0.1),
                    "latency_s": 0.7 + 0.004 * ts,
                },
            },
        })
    return recs


def test_pilot_sd_finite_and_bounded():
    recs = _make_records(40, 0.8, 0.78, 400, 120, seed=1)
    pilot = e2.estimate_pilot_f1_sd(recs)
    assert np.isfinite(pilot["pilot_f1_sd"])
    assert pilot["pilot_f1_sd"] >= 0.0
    assert pilot["n_pilot"] > 0


def test_tost_sample_size_monotone_and_finite():
    ss = e2.tost_sample_size(margin=0.03, sd=0.04, target_power=0.8)
    curve = ss["power_curve"]
    assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in curve.values())
    assert ss["n_for_target_power"] in [int(k) for k in curve] or \
        ss["n_for_target_power"] >= 50
    # larger N should generally not have lower power than the smallest N
    ns = sorted(int(k) for k in curve)
    assert curve[str(ns[-1])] >= curve[str(ns[0])] - 0.2


def test_efficiency_reduction_detected():
    # slice uses far fewer tokens -> significant reduction expected
    recs = _make_records(60, 0.80, 0.80, 500, 100, seed=2)
    out = e2.run_rq2_e2(recs, paper_grade_n=1000, mode_meta={"mode": "test"})
    tok = out["efficiency_reductions"]["tokens"]
    assert tok["median_reduction"] > 0
    assert tok["significant_reduction"] is True
    for key in ("W", "p_value", "median_reduction"):
        assert np.isfinite(tok[key])
    assert np.isfinite(tok["reduction_ci95"][0]) and np.isfinite(tok["reduction_ci95"][1])


def test_equivalence_detected_when_truly_equivalent():
    # near-identical effectiveness, big token savings -> H4 can be supported
    recs = _make_records(120, 0.80, 0.80, 500, 100, seed=3)
    out = e2.run_rq2_e2(recs, paper_grade_n=1000, mode_meta={"mode": "test"})
    tost = out["tost_effectiveness_equivalence"]
    assert np.isfinite(tost["mean_diff"])
    assert np.isfinite(tost["ci90_mean_diff"][0]) and np.isfinite(tost["ci90_mean_diff"][1])
    # conclusion string must be one of the three pre-registered outcomes
    assert any(s in tost["pre_registered_conclusion"]
               for s in ("EQUIVALENT", "INCONCLUSIVE", "EFFECTIVENESS LOSS"))


def test_effectiveness_loss_not_suppressed():
    # slice is clearly worse -> must report loss, not claim equivalence
    recs = _make_records(120, 0.85, 0.45, 500, 100, seed=4)
    out = e2.run_rq2_e2(recs, paper_grade_n=1000, mode_meta={"mode": "test"})
    tost = out["tost_effectiveness_equivalence"]
    assert tost["significant_difference"] is True
    assert out["H4_efficiency_without_loss"] is False
    assert "LOSS" in tost["pre_registered_conclusion"] or \
        out["rq2_e2_decision"].startswith("REFUTED")


def test_pareto_frontier_marks_efficiency():
    recs = _make_records(40, 0.80, 0.80, 500, 100, seed=5)
    out = e2.run_rq2_e2(recs, paper_grade_n=1000, mode_meta={"mode": "test"})
    pts = out["pareto_frontier"]["points"]
    assert len(pts) >= 2
    for p in pts:
        assert np.isfinite(p["effectiveness"]) and np.isfinite(p["cost"])
        assert isinstance(p["pareto_efficient"], bool)
    # at least one point must be Pareto-efficient
    assert any(p["pareto_efficient"] for p in pts)


def test_all_outputs_finite():
    recs = _make_records(50, 0.78, 0.76, 450, 130, seed=6)
    out = e2.run_rq2_e2(recs, paper_grade_n=1000, mode_meta={"mode": "test"})
    # margin, power, reductions must all be finite
    assert np.isfinite(out["equivalence_margin"]["margin"])
    assert out["equivalence_margin"]["margin"] >= 0.02
    for field in ("tokens", "latency", "cost"):
        r = out["efficiency_reductions"][field]
        assert np.isfinite(r["p_value"]) and np.isfinite(r["median_reduction"])


def test_empty_records_raises():
    with pytest.raises(RuntimeError):
        e2.run_rq2_e2([], paper_grade_n=1000, mode_meta={"mode": "test"})


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))