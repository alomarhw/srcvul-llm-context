"""
test_rq3.py -- unit tests for RQ3 repair components. Fast, offline, no LLM calls.

Guards the measurement-integrity invariants the acceptance checks require:
  * independent static oracle rejects a self-copy (negative control) and accepts
    the official fix (identity-of-fix control);
  * srcDiff reintegration splices slice edits into the whole function;
  * census honestly reports executable level-4 N (0 without build/PoC metadata);
  * no-LLM static-template baseline is deterministic;
  * analysis_rq3 produces finite metrics, complete baseline arms, McNemar with a
    finite OR+CI, contamination CIs, and a NULL (never simulated) Cohen's kappa.

Run: pytest -q test_rq3.py  (pytest==7.4.4)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import repair_pipeline as rp
import feasibility_census as fc
import experiment_rq3 as erq3
import analysis_rq3 as arq3


BEFORE = "int f(char *p){\n char buf[8];\n strcpy(buf, p);\n return 0;\n}"
AFTER = "int f(char *p){\n char buf[8];\n strncpy(buf, p, 7);\n buf[7]=0;\n return 0;\n}"
VUL = [3]


# --------------------------------------------------------------------------- #
# independent static oracle
# --------------------------------------------------------------------------- #
def test_oracle_rejects_self_copy_negative_control():
    o = rp.static_elimination_oracle(BEFORE, AFTER, BEFORE, VUL)
    assert o["eliminated"] is False  # grading against itself must NOT pass


def test_oracle_accepts_official_fix():
    o = rp.static_elimination_oracle(BEFORE, AFTER, AFTER, VUL)
    assert o["parses"] is True
    assert o["closer_to_official_fix"] is True
    assert o["eliminated"] is True


def test_oracle_subchecks_present_and_boolean():
    o = rp.static_elimination_oracle(BEFORE, AFTER, AFTER, VUL)
    for k in ("parses", "vul_pattern_changed", "closer_to_official_fix",
              "signature_severed", "eliminated"):
        assert isinstance(o[k], bool)


# --------------------------------------------------------------------------- #
# reintegration
# --------------------------------------------------------------------------- #
def test_reintegrate_full_function_reply():
    cand = rp.reintegrate_patch(BEFORE, AFTER, VUL)
    assert rp._balanced(cand)
    assert "strncpy" in cand


def test_reintegrate_slice_edit_splices():
    patch = " strncpy(buf, p, 7); buf[7]=0;"
    cand = rp.reintegrate_patch(BEFORE, patch, VUL)
    assert rp._balanced(cand)
    assert "strncpy" in cand


def test_reintegrate_empty_patch_returns_original():
    cand = rp.reintegrate_patch(BEFORE, "", VUL)
    assert cand == BEFORE


# --------------------------------------------------------------------------- #
# feasibility census
# --------------------------------------------------------------------------- #
def test_census_level0_without_metadata():
    u = {"unit_id": "u", "extra": {}}
    assert fc.classify_unit(u) == "0"


def test_census_detects_poc_and_build():
    assert fc.classify_unit({"unit_id": "u", "extra": {"poc": "x"}}) == "4a"
    assert fc.classify_unit({"unit_id": "u", "extra": {"build_cmd": "make"}}) == "2"
    assert fc.classify_unit({"unit_id": "u", "commit_id": "abc"}) == "1"


def test_census_executable_N_zero_without_poc():
    units = [{"unit_id": f"u{i}", "commit_id": "c", "extra": {}} for i in range(10)]
    man = fc.run_census(units)
    assert man["executable_level4_N"] == 0
    assert man["static_only_N"] == 10
    assert "DESIGN_NEEDED" in man  # honestly flags missing executable benchmark


# --------------------------------------------------------------------------- #
# no-LLM static baseline
# --------------------------------------------------------------------------- #
def test_no_llm_baseline_deterministic():
    cand1 = erq3._no_llm_static_template(BEFORE, VUL)
    cand2 = erq3._no_llm_static_template(BEFORE, VUL)
    assert cand1 == cand2
    assert "removed vulnerable stmt" in cand1


# --------------------------------------------------------------------------- #
# analysis_rq3 end-to-end on synthetic records
# --------------------------------------------------------------------------- #
def _make_units(n):
    units = []
    for i in range(n):
        units.append({
            "unit_id": f"u{i}",
            "cwe": f"CWE-{100 + (i % 3)}",
            "project": "p",
            "commit_id": "c",
            "split": "post_cutoff" if i % 4 == 0 else "test",
            "func_before": BEFORE,
            "func_after": AFTER,
            "vul_lines": VUL,
            "is_post_cutoff": (i % 4 == 0),
            "extra": {},
        })
    return units


def _make_records(units, slice_elim=0.7, whole_elim=0.4, seed=0):
    """Synthetic RQ3-E1 records: slice (R4) eliminates more than whole (R1)."""
    rng = np.random.default_rng(seed)
    recs = []
    for u in units:
        for model in erq3.MODELS_UNDER_TEST:
            for rep, (name, prob) in enumerate(
                    [("R1", whole_elim), ("R4", slice_elim)]):
                elim = bool(rng.random() < prob)
                recs.append({
                    "unit_id": u["unit_id"], "cwe": u["cwe"], "split": u["split"],
                    "model": model, "representation": name, "repeat": 0,
                    "patch_applied": True, "candidate_len": 50,
                    "static_oracle": {"parses": True, "vul_pattern_changed": True,
                                      "closer_to_official_fix": elim,
                                      "signature_severed": elim, "eliminated": elim},
                    "eliminated_static": elim,
                    "dynamic_attempted": False, "agentic": True,
                    "agentic_improved": bool(name == "R4" and not elim and rng.random() < 0.3),
                })
    return recs


def test_analysis_rq3_finite_and_complete():
    units = _make_units(40)
    records = _make_records(units, slice_elim=0.75, whole_elim=0.40, seed=1)
    no_llm = [{"unit_id": u["unit_id"], "cwe": u["cwe"], "split": u["split"],
               "model": "no_llm_static_template", "representation": "none",
               "repeat": 0, "patch_applied": True,
               "static_oracle": {"parses": True, "eliminated": False},
               "eliminated_static": False, "dynamic_attempted": False}
              for u in units]
    e1 = {"records": records, "no_llm_baseline": no_llm, "llm_stats": {}}
    e2 = erq3.run_rq3_e2(records)
    census = fc.run_census(units)
    executable_ids = set()
    out = arq3.analyze_rq3(e1, e2, census, units, executable_ids,
                           mode_meta={"label": "test"})
    rq3 = out["RQ3"]
    # all metrics finite
    for k in arq3.METRIC_KEYS:
        assert math.isfinite(float(rq3["metrics"][k])), k
    # all baseline arms complete + finite
    for arm, mm in rq3["baseline_comparison"].items():
        for k in arq3.METRIC_KEYS:
            assert math.isfinite(float(mm[k])), f"{arm}.{k}"
    # statistical test finite
    st = rq3["statistical_tests"]
    assert math.isfinite(st["p_value"])
    assert math.isfinite(st["effect_size"])
    assert all(math.isfinite(x) for x in st["confidence_interval"])


def test_analysis_rq3_slice_advantage_positive():
    units = _make_units(60)
    records = _make_records(units, slice_elim=0.80, whole_elim=0.30, seed=2)
    no_llm = [{"unit_id": u["unit_id"], "cwe": u["cwe"], "split": u["split"],
               "model": "no_llm_static_template", "representation": "none",
               "repeat": 0, "patch_applied": True,
               "static_oracle": {"parses": True, "eliminated": False},
               "eliminated_static": False}
              for u in units]
    e1 = {"records": records, "no_llm_baseline": no_llm}
    e2 = erq3.run_rq3_e2(records)
    census = fc.run_census(units)
    out = arq3.analyze_rq3(e1, e2, census, units, set(), mode_meta={"label": "test"})
    rq3 = out["RQ3"]
    # slice (R4) elimination should exceed whole-function (R1); advantage > 0
    assert rq3["metrics"]["slice_vs_whole_function_advantage_delta_rq3"] > 0
    # elimination rate non-degenerate
    assert 0.0 < rq3["metrics"]["independent_vulnerability_elimination_rate_rq3"] <= 1.0


def test_analysis_rq3_kappa_is_null_not_simulated():
    units = _make_units(30)
    records = _make_records(units, seed=3)
    e1 = {"records": records, "no_llm_baseline": []}
    e2 = erq3.run_rq3_e2(records)
    census = fc.run_census(units)
    out = arq3.analyze_rq3(e1, e2, census, units, set(), mode_meta={"label": "test"})
    kappa = out["RQ3"]["_detail"]["cohens_kappa_adjudication"]
    assert kappa["value"] is None
    assert "not_available" in kappa


def test_analysis_rq3_negative_control_guard():
    units = _make_units(40)
    records = _make_records(units, slice_elim=0.70, whole_elim=0.40, seed=4)
    no_llm = [{"unit_id": u["unit_id"], "cwe": u["cwe"], "split": u["split"],
               "model": "no_llm_static_template", "representation": "none",
               "repeat": 0, "patch_applied": True,
               "static_oracle": {"parses": True, "eliminated": False},
               "eliminated_static": False}
              for u in units]
    e1 = {"records": records, "no_llm_baseline": no_llm}
    e2 = erq3.run_rq3_e2(records)
    census = fc.run_census(units)
    out = arq3.analyze_rq3(e1, e2, census, units, set(), mode_meta={"label": "test"})
    guard = out["RQ3"]["_detail"]["negative_control_guard"]
    assert guard["best_slice_beats_no_llm"] is True


def test_mcnemar_finite_on_discordant():
    a = {f"u{i}": int(i % 2 == 0) for i in range(20)}
    b = {f"u{i}": int(i % 3 == 0) for i in range(20)}
    res = arq3._mcnemar_paired(a, b, "a vs b")
    assert math.isfinite(res["p_value"])
    assert math.isfinite(res["odds_ratio"])
    assert all(math.isfinite(x) for x in res["or_ci95"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))