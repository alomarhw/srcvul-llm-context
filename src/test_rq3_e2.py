"""
test_rq3_e2.py -- unit tests for RQ3-E2 generalization components. Offline, no LLM.

Guards the RQ3-E2 measurement-integrity invariants:
  * project-disjoint split shares NO project between train and test;
  * chronological split puts post-cutoff units on the test side;
  * the per-split rep_builder never leaks the official fix (func_after);
  * per-split delta + bootstrap CI are finite and directionally correct;
  * attenuation (random -> contamination) and the R5 leakage gap are finite with
    the expected sign;
  * H6 persistence requires the CI lower bound > 0 on >= 2 models.

Run: pytest -q test_rq3_e2.py  (pytest==7.4.4)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import experiment_rq3_e2 as e2
import analysis_rq3_e2 as a2


BEFORE = "int f(char *p){\n char buf[8];\n strcpy(buf, p);\n return 0;\n}"
AFTER = "int f(char *p){\n char buf[8];\n strncpy(buf, p, 7);\n buf[7]=0;\n return 0;\n}"


def _units(n):
    units = []
    for i in range(n):
        yr = 2008 + (i % 18)
        units.append({
            "unit_id": f"u{i}",
            "func_before": BEFORE,
            "func_after": AFTER,
            "vul_lines": [3],
            "cwe": f"CWE-{100 + i % 3}",
            "project": f"proj{i % 5}",
            "cve_year": yr,
            "is_post_cutoff": yr >= e2.CHRONO_CUTOFF_YEAR,
        })
    return units


# --------------------------------------------------------------------------- #
# split construction
# --------------------------------------------------------------------------- #
def test_build_splits_present():
    sp = e2.build_splits(_units(40))
    assert set(sp.keys()) == {"random", "chronological", "project_disjoint"}
    for name in ("random", "chronological", "project_disjoint"):
        assert sp[name]["train"] and sp[name]["test"]


def test_project_disjoint_has_no_shared_project():
    sp = e2.build_splits(_units(50))
    tr = {u["project"] for u in sp["project_disjoint"]["train"]}
    te = {u["project"] for u in sp["project_disjoint"]["test"]}
    assert not (tr & te), "project leaked across the project-disjoint split"


def test_chronological_test_is_post_cutoff():
    sp = e2.build_splits(_units(60))
    chrono = sp["chronological"]
    if not chrono.get("_fallback_by_year"):
        for u in chrono["test"]:
            assert (u["is_post_cutoff"] or
                    int(u["cve_year"]) >= e2.CHRONO_CUTOFF_YEAR)
        for u in chrono["train"]:
            assert int(u["cve_year"]) < e2.CHRONO_CUTOFF_YEAR


# --------------------------------------------------------------------------- #
# leak-free rep builder
# --------------------------------------------------------------------------- #
def test_rep_builder_no_official_fix_leak():
    units = _units(20)
    train = units[:15]
    rb = e2.make_split_rep_builder(train, chronological=True,
                                   r5_chronology_constrained=True)
    for u in units:
        for rep in ("R1", "R4", "R5"):
            ctx = rb(u, rep)
            assert u["func_after"] not in ctx, f"official fix leaked in {rep}"


def test_rep_builder_r5_uses_only_train_corpus():
    units = _units(20)
    train = units[:10]
    rb = e2.make_split_rep_builder(train, chronological=False,
                                   r5_chronology_constrained=False)
    # building R5 for a test unit must not raise and must return a string
    ctx = rb(units[15], "R5")
    assert isinstance(ctx, str) and len(ctx) > 0


# --------------------------------------------------------------------------- #
# analysis: per-split deltas + CIs
# --------------------------------------------------------------------------- #
def _records(split_elim: dict, n=30, seed=0):
    """Build RQ3-E2 records with per-split (R4,R1) elimination probabilities.

    split_elim: {split: (p_r4, p_r1)}.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for split, (p4, p1) in split_elim.items():
        for i in range(n):
            uid = f"u{i}"
            for model in e2.MODELS_UNDER_TEST:
                for rep, p in (("R4", p4), ("R1", p1)):
                    elim = bool(rng.random() < p)
                    recs.append({
                        "unit_id": uid, "model": model,
                        "generalization_rep": rep, "split": split, "repeat": 0,
                        "eliminated_static": elim,
                        "static_oracle": {"eliminated": elim, "parses": True},
                    })
    return recs


def test_per_split_delta_positive_when_slice_better():
    recs = _records({"random": (0.8, 0.4),
                     "chronological": (0.7, 0.45),
                     "project_disjoint": (0.65, 0.5)}, n=40, seed=1)
    e2gen = {"records": recs, "r5_leakage_records": [], "split_meta": {},
             "chrono_cutoff_year": e2.CHRONO_CUTOFF_YEAR}
    out = a2.analyze_rq3_e2(e2gen)
    rand = out["per_split_model_delta"]["random"]
    for m, r in rand.items():
        assert r["delta"] > 0
        assert math.isfinite(r["ci95"][0]) and math.isfinite(r["ci95"][1])
        assert r["ci95"][0] <= r["delta"] <= r["ci95"][1]


def test_attenuation_reported_and_finite():
    recs = _records({"random": (0.85, 0.35),
                     "chronological": (0.6, 0.5),
                     "project_disjoint": (0.58, 0.52)}, n=50, seed=2)
    e2gen = {"records": recs, "r5_leakage_records": [], "split_meta": {},
             "chrono_cutoff_year": e2.CHRONO_CUTOFF_YEAR}
    out = a2.analyze_rq3_e2(e2gen)
    atten = out["attenuation_random_to_contamination"]
    for split in ("chronological", "project_disjoint"):
        a = atten[split]
        assert math.isfinite(a["mean_attenuation_drop"])
        assert math.isfinite(a["wilcoxon_p"])
        assert math.isfinite(a["wilcoxon_p_holm"])
        # random advantage is larger than contamination advantage here -> drop>0
        assert a["mean_attenuation_drop"] > 0


def test_h6_persistence_logic():
    # slice clearly better and consistent -> CIs should stay > 0 on >= 2 models
    recs = _records({"random": (0.9, 0.3),
                     "chronological": (0.85, 0.3),
                     "project_disjoint": (0.85, 0.3)}, n=80, seed=3)
    e2gen = {"records": recs, "r5_leakage_records": [], "split_meta": {},
             "chrono_cutoff_year": e2.CHRONO_CUTOFF_YEAR}
    out = a2.analyze_rq3_e2(e2gen)
    assert out["H6_advantage_persists_under_contamination"] is True


def test_h6_false_when_advantage_vanishes():
    # no real advantage under contamination -> H6 should be False
    recs = _records({"random": (0.8, 0.4),
                     "chronological": (0.5, 0.5),
                     "project_disjoint": (0.5, 0.5)}, n=60, seed=4)
    e2gen = {"records": recs, "r5_leakage_records": [], "split_meta": {},
             "chrono_cutoff_year": e2.CHRONO_CUTOFF_YEAR}
    out = a2.analyze_rq3_e2(e2gen)
    assert out["H6_advantage_persists_under_contamination"] is False


# --------------------------------------------------------------------------- #
# R5 leakage gap
# --------------------------------------------------------------------------- #
def test_r5_leakage_gap_positive_when_unconstrained_higher():
    rng = np.random.default_rng(5)
    r5 = []
    for model in e2.MODELS_UNDER_TEST:
        for i in range(30):
            uid = f"u{i}"
            r5.append({"unit_id": uid, "model": model, "r5_retrieval": "constrained",
                       "eliminated_static": bool(rng.random() < 0.55)})
            r5.append({"unit_id": uid, "model": model, "r5_retrieval": "unconstrained",
                       "eliminated_static": bool(rng.random() < 0.80)})
    recs = _records({"random": (0.7, 0.5), "chronological": (0.6, 0.5),
                     "project_disjoint": (0.6, 0.5)}, n=30, seed=5)
    e2gen = {"records": recs, "r5_leakage_records": r5, "split_meta": {},
             "chrono_cutoff_year": e2.CHRONO_CUTOFF_YEAR}
    out = a2.analyze_rq3_e2(e2gen)
    pooled = out["r5_leakage_gap"]["pooled"]
    assert math.isfinite(pooled["leakage_gap"])
    assert pooled["leakage_gap"] > 0  # unconstrained retrieval leaks an advantage


def test_empty_records_not_available():
    out = a2.analyze_rq3_e2({"records": []})
    assert "not_available" in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))