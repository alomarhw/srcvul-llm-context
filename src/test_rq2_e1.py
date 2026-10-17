"""test_rq2_e1.py

Unit tests for RQ2-E1 statistical and oracle helpers. These run fast (no LLM,
no network, no real data) and guard the measurement-integrity invariants the
acceptance checks require: finite outputs, correct directions, and sane
negative/identity controls.

Run: pytest -q test_rq2_e1.py  (pytest==7.4.4)
"""

from __future__ import annotations

import numpy as np
import pytest

import oracle_incompleteness as oi
import rq2_stats as rs


# --------------------------------------------------------------------------- #
# oracle_incompleteness
# --------------------------------------------------------------------------- #

SRC = (
    "int f(char *buf, int n) {\n"
    "  int i = 0;\n"
    "  char *p = buf;\n"
    "  for (i = 0; i < n; i++) {\n"
    "    p[i] = read_byte(i);\n"
    "  }\n"
    "  return check(p, n);\n"
    "}\n"
)


def test_build_oracle_nonempty():
    g = oi.build_oracle("u1", SRC)
    assert g.total_edges() > 0
    # pointer/alias operations must be detected (p[i], *p, ->)
    assert len(g.ptr_lines) >= 1
    # call edges must be detected (read_byte, check)
    assert len(g.call_edges) >= 2


def test_incompleteness_finite_and_bounded():
    g = oi.build_oracle("u1", SRC)
    partial = {"slice_lines": [2, 3, 4, 5], "cfuncs": [[5, "read_byte"]]}
    inc = oi.measure_incompleteness(g, partial)
    for k, v in inc.items():
        assert np.isfinite(v), f"{k} not finite"
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_full_slice_is_least_incomplete_control():
    """Negative control: a slice covering the whole function and all callees
    must be no more incomplete than a partial slice."""
    g = oi.build_oracle("u1", SRC)
    n_lines = len(SRC.splitlines())
    full = {"slice_lines": list(range(1, n_lines + 1)),
            "cfuncs": [[5, "read_byte"], [7, "check"]]}
    partial = {"slice_lines": [2, 3], "cfuncs": []}
    inc_full = oi.measure_incompleteness(g, full)
    inc_partial = oi.measure_incompleteness(g, partial)
    assert inc_full["incompleteness_overall"] <= inc_partial["incompleteness_overall"]


def test_correlate_incompleteness_negative():
    # construct aligned dicts where more incompleteness => lower performance
    rng = np.random.default_rng(0)
    uids = [f"u{i}" for i in range(30)]
    inc = {u: float(np.clip(i / 30.0 + rng.normal(0, 0.02), 0, 1)) for i, u in enumerate(uids)}
    perf = {u: float(0.9 - inc[u] * 0.5 + rng.normal(0, 0.02)) for u in uids}
    res = oi.correlate_incompleteness_vs_performance(inc, perf)
    assert res["n"] == 30
    assert np.isfinite(res["spearman_rho"]) and np.isfinite(res["p_value"])
    assert res["spearman_rho"] < 0  # real incompleteness tracks performance loss


def test_build_subset_reproducible():
    units = [{"unit_id": f"u{i}", "func_before": SRC} for i in range(20)]
    slices = {f"u{i}": {"slice_lines": [2, 3, 4], "cfuncs": []} for i in range(20)}
    a = oi.build_incompleteness_subset(units, slices, max_units=10, seed=42)
    b = oi.build_incompleteness_subset(units, slices, max_units=10, seed=42)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert len(a) == 10


# --------------------------------------------------------------------------- #
# rq2_stats
# --------------------------------------------------------------------------- #

def test_jonckheere_detects_decreasing_trend():
    rng = np.random.default_rng(1)
    doses = [0, 10, 20, 40, 60, 80]
    groups = [list(rng.normal(0.8 - d / 200.0, 0.03, 25)) for d in doses]
    jt = rs.jonckheere_terpstra_trend(groups)
    assert np.isfinite(jt["p_value"]) and np.isfinite(jt["z"])
    assert jt["p_value"] < 0.05  # strong monotone decline is significant


def test_jonckheere_flat_not_significant():
    rng = np.random.default_rng(2)
    groups = [list(rng.normal(0.8, 0.03, 25)) for _ in range(6)]
    jt = rs.jonckheere_terpstra_trend(groups)
    assert np.isfinite(jt["p_value"])
    # a flat trend should not be strongly significant
    assert jt["p_value"] > 0.01


def test_spearman_dose_response_negative():
    doses = [0, 10, 20, 40, 60, 80]
    perf = [0.80, 0.76, 0.72, 0.65, 0.58, 0.50]
    res = rs.spearman_dose_response(doses, perf)
    assert res["spearman_rho"] < 0
    assert np.isfinite(res["p_value"])


def test_tost_equivalence_for_near_identical():
    rng = np.random.default_rng(3)
    ref = rng.normal(0.75, 0.04, 80)
    cond = ref + rng.normal(0.0, 0.008, 80)
    margin = rs.derive_tost_margin(ref)
    res = rs.tost_equivalence(ref, cond, margin)
    assert np.isfinite(res["tost_p"]) and np.isfinite(res["mean_diff"])
    assert res["equivalent"] is True


def test_tost_not_equivalent_for_real_loss():
    rng = np.random.default_rng(4)
    ref = rng.normal(0.80, 0.04, 80)
    cond = ref - rng.normal(0.10, 0.04, 80)  # clear effectiveness loss
    margin = 0.02
    res = rs.tost_equivalence(ref, cond, margin)
    assert res["equivalent"] is False


def test_tost_power_in_unit_interval():
    margin = 0.03
    pw = rs.tost_power(n=100, margin=margin, true_diff=0.0, sd=0.03, n_sim=500)
    assert 0.0 <= pw["power"] <= 1.0


def test_paired_wilcoxon_holm_finite_and_corrected():
    rng = np.random.default_rng(5)
    a0 = rng.normal(0.80, 0.05, 60)
    comps = {
        "A0_vs_full": (a0, a0 - rng.normal(0.10, 0.04, 60)),
        "A1_vs_full": (a0, a0 - rng.normal(0.05, 0.04, 60)),
        "A2_vs_full": (a0, a0 - rng.normal(0.02, 0.04, 60)),
    }
    res = rs.paired_wilcoxon_holm(comps)
    assert set(res) == set(comps)
    for name, r in res.items():
        for key in ("W", "p_raw", "p_holm", "cohens_d", "ci_low", "ci_high"):
            assert np.isfinite(r[key]), f"{name}.{key} not finite"
        assert r["p_holm"] >= r["p_raw"] - 1e-9  # Holm never decreases p


def test_derive_margin_respects_floor():
    # tiny-SD pilot -> margin floored at 0.02
    pilot = [0.75, 0.751, 0.749, 0.7505]
    m = rs.derive_tost_margin(pilot, floor=0.02)
    assert m >= 0.02


def test_degenerate_inputs_return_finite():
    # zero-variance / tiny inputs must not produce NaN/inf
    assert np.isfinite(rs.cohens_d_paired([1, 1, 1], [1, 1, 1]))
    lo, hi = rs.bootstrap_ci([0.5])
    assert np.isfinite(lo) and np.isfinite(hi)
    jt = rs.jonckheere_terpstra_trend([[0.5], [0.5]])
    assert np.isfinite(jt["p_value"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))