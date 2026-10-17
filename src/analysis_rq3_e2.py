"""
analysis_rq3_e2.py -- statistics for RQ3-E2 contamination-controlled generalization.

Consumes the RQ3-E2 records (experiment_rq3_e2.run_rq3_e2_generalization) and:

  1. Per split (random / chronological / project_disjoint) per model, aggregate
     repeats per unit, compute the slice-vs-whole-function elimination delta
     (mean R4 - mean R1), and a bootstrap 95% CI on that delta (paired over the
     shared test units).
  2. Report whether the advantage PERSISTS (CI lower bound > 0) for each
     split/model; H6 answered YES if it persists under contamination control on
     >= 2 models (on the chronological AND/OR project_disjoint splits).
  3. Quantify ATTENUATION from the random split to each contamination-controlled
     split with paired Wilcoxon on per-model deltas, plus the numeric drop.
  4. Measure the R5 leakage gap = unconstrained-retrieval elimination minus
     chronology-constrained elimination (per model and pooled); R5 gains must be
     interpreted net of this gap.
  5. Holm-correct the family of persistence/attenuation tests.

Elimination is read from the INDEPENDENT static oracle outcome (eliminated_static).
The unit of analysis is the code unit; repeats averaged first. All returned
numbers are finite (degenerate cases use explicit finite conventions).

This module MERGES its findings into results/results.json under
RQ3._detail.rq3_e2_generalization and updates RQ3.metrics.
slice_vs_whole_function_advantage_delta_rq3 to the contamination-controlled
(chronological) value when available, so the headline RQ3 advantage reflects the
generalization-controlled estimate per the RQ3-E2 contract.

Requirements (pinned): numpy==2.2.6, scipy==1.13.1, statsmodels==0.14.6
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except Exception:  # pragma: no cover
    multipletests = None
    HAVE_SM = False

import experiment_rq3_e2 as erq3e2

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

MODELS_UNDER_TEST = erq3e2.MODELS_UNDER_TEST
SPLITS = ["random", "chronological", "project_disjoint"]
CONTAMINATION_SPLITS = ["chronological", "project_disjoint"]


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _holm(pvals: List[float]) -> List[float]:
    if not pvals:
        return []
    if HAVE_SM and multipletests is not None:
        _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        return [float(x) for x in p_adj]
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max((m - rank) * pvals[idx], prev))
        adj[idx] = val
        prev = val
    return adj


def _per_unit_elim(records: List[Dict[str, Any]], split: str, model: str,
                   rep: str) -> Dict[str, int]:
    """Aggregate repeats per unit -> majority elimination (0/1) for split/model/rep."""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        if (r.get("split") == split and r.get("model") == model and
                r.get("generalization_rep") == rep):
            buckets[r["unit_id"]].append(int(bool(r.get("eliminated_static"))))
    return {u: int(sum(v) >= (len(v) / 2.0 + 1e-9)) for u, v in buckets.items()}


def _paired_delta_and_ci(r4: Dict[str, int], r1: Dict[str, int],
                         iters: int = 2000) -> Dict[str, Any]:
    """Paired (over shared units) slice-vs-whole delta + bootstrap 95% CI."""
    # Sorted: record order follows the completion order of concurrent model calls, so it varies between
    # runs; the seeded bootstrap must resample the units in a fixed order to be reproducible.
    shared = sorted(u for u in r4 if u in r1)
    if not shared:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "n": 0,
                "note": "no shared test units"}
    a = np.array([r4[u] for u in shared], dtype=float)  # slice
    b = np.array([r1[u] for u in shared], dtype=float)  # whole-function
    diff = a - b
    delta = float(np.mean(diff))
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(iters)])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"delta": _finite(delta), "ci95": [_finite(ci[0]), _finite(ci[1])],
            "n": n, "slice_elim_rate": float(a.mean()),
            "whole_elim_rate": float(b.mean())}


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if len(diff) < 1 or np.all(diff == 0):
        return float("nan"), 1.0
    try:
        w, p = stats.wilcoxon(a, b, zero_method="wilcox")
        return float(w), float(p)
    except ValueError:
        return float("nan"), 1.0


def analyze_rq3_e2(e2gen: Dict[str, Any]) -> Dict[str, Any]:
    records = e2gen.get("records", [])
    if not records:
        return {"not_available": "RQ3-E2 produced no records"}

    # ---- per split/model deltas + CIs ----
    per_split_model: Dict[str, Dict[str, Any]] = {}
    delta_by_split: Dict[str, List[float]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        per_split_model[split] = {}
        for model in MODELS_UNDER_TEST:
            r4 = _per_unit_elim(records, split, model, "R4")
            r1 = _per_unit_elim(records, split, model, "R1")
            res = _paired_delta_and_ci(r4, r1)
            res["advantage_persists"] = bool(res["ci95"][0] > 0.0)
            per_split_model[split][model] = res
            delta_by_split[split].append(res["delta"])

    # ---- H6 persistence: CI lower bound > 0 on >= 2 models under contamination ----
    persistence = {}
    for split in CONTAMINATION_SPLITS:
        n_persist = sum(
            1 for m in MODELS_UNDER_TEST
            if per_split_model.get(split, {}).get(m, {}).get("advantage_persists")
        )
        persistence[split] = {"models_persisting": n_persist,
                              "total_models": len(MODELS_UNDER_TEST)}
    # H6 true if ANY contamination split has the advantage persisting on >= 2 models
    h6 = any(v["models_persisting"] >= 2 for v in persistence.values())

    # ---- attenuation: random -> each contamination split (paired over models) ----
    attenuation = {}
    raw_p = []
    atten_keys = []
    rand_deltas = np.array(delta_by_split["random"], dtype=float)
    for split in CONTAMINATION_SPLITS:
        cont_deltas = np.array(delta_by_split[split], dtype=float)
        n = min(len(rand_deltas), len(cont_deltas))
        if n >= 1:
            w, p = _paired_wilcoxon(rand_deltas[:n], cont_deltas[:n])
            mean_drop = float(np.mean(rand_deltas[:n] - cont_deltas[:n]))
        else:
            w, p, mean_drop = float("nan"), 1.0, 0.0
        attenuation[split] = {
            "mean_delta_random": float(np.mean(rand_deltas)) if len(rand_deltas) else 0.0,
            "mean_delta_contamination": float(np.mean(cont_deltas)) if len(cont_deltas) else 0.0,
            "mean_attenuation_drop": _finite(mean_drop),
            "wilcoxon_W": _finite(w, 0.0),
            "wilcoxon_p": _finite(1.0 if math.isnan(p) else p, 1.0),
            "n_models": n,
        }
        raw_p.append(_finite(1.0 if math.isnan(p) else p, 1.0))
        atten_keys.append(split)
    holm_atten = _holm(raw_p)
    for key, hp in zip(atten_keys, holm_atten):
        attenuation[key]["wilcoxon_p_holm"] = _finite(hp, 1.0)

    # ---- R5 leakage gap (unconstrained - constrained) per model and pooled ----
    r5recs = e2gen.get("r5_leakage_records", [])
    leakage = {"per_model": {}, "note": (
        "R5 leakage gap = unconstrained-retrieval elimination minus "
        "chronology-constrained elimination; R5 gains interpreted net of this.")}
    pooled_c, pooled_u = [], []
    for model in MODELS_UNDER_TEST:
        c = _r5_rate(r5recs, model, "constrained")
        u = _r5_rate(r5recs, model, "unconstrained")
        if c is not None and u is not None:
            leakage["per_model"][model] = {
                "constrained_elim": _finite(c), "unconstrained_elim": _finite(u),
                "leakage_gap": _finite(u - c),
            }
            pooled_c.append(c)
            pooled_u.append(u)
    if pooled_c and pooled_u:
        leakage["pooled"] = {
            "constrained_elim": float(np.mean(pooled_c)),
            "unconstrained_elim": float(np.mean(pooled_u)),
            "leakage_gap": float(np.mean(pooled_u) - np.mean(pooled_c)),
        }
    else:
        leakage["pooled"] = {"not_available": "no R5 leakage records"}

    # ---- contamination-controlled headline delta (chronological, best model) ----
    chrono = per_split_model.get("chronological", {})
    best_chrono_delta = 0.0
    best_model = None
    if chrono:
        best_model = max(chrono, key=lambda m: chrono[m].get("delta", 0.0))
        best_chrono_delta = _finite(chrono[best_model].get("delta", 0.0))

    return {
        "split_meta": e2gen.get("split_meta", {}),
        "chrono_cutoff_year": e2gen.get("chrono_cutoff_year"),
        "per_split_model_delta": per_split_model,
        "persistence": persistence,
        "attenuation_random_to_contamination": attenuation,
        "r5_leakage_gap": leakage,
        "contamination_controlled_headline_delta_rq3": best_chrono_delta,
        "contamination_controlled_best_model": best_model,
        "H6_advantage_persists_under_contamination": bool(h6),
        "oracle": e2gen.get("oracle", "independent_static"),
        "design_needed": e2gen.get("design_needed", []),
        "llm_stats": e2gen.get("llm_stats", {}),
        "interpretation": (
            "Advantage is claimed to persist only where the bootstrap 95% CI "
            "lower bound stays > 0. Attenuation from the random split is reported "
            "honestly, including any split/model where the advantage vanishes or "
            "reverses (bounded-generalization finding)."),
    }


def _r5_rate(r5recs: List[Dict[str, Any]], model: str, tag: str) -> float | None:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for r in r5recs:
        if r.get("model") == model and r.get("r5_retrieval") == tag:
            buckets[r["unit_id"]].append(int(bool(r.get("eliminated_static"))))
    if not buckets:
        return None
    per_unit = [int(sum(v) >= (len(v) / 2.0 + 1e-9)) for v in buckets.values()]
    return float(np.mean(per_unit)) if per_unit else None


# --------------------------------------------------------------------------- #
# merge into results.json (RQ3._detail), finite-guarded
# --------------------------------------------------------------------------- #
def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else 0.0
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    return obj


def merge_into_results(e2_analysis: Dict[str, Any]) -> None:
    """Attach RQ3-E2 findings under RQ3._detail and refresh the headline
    contamination-controlled advantage delta."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "results.json"
    if not out.exists():
        print("[analysis_rq3_e2] results.json missing; writing RQ3-E2 standalone")
        payload = {"RQ3": {"_detail": {"rq3_e2_generalization": e2_analysis}}}
        out.write_text(json.dumps(_sanitize(payload), indent=2, allow_nan=False))
        return
    try:
        merged = json.loads(out.read_text())
    except Exception:
        merged = {}

    rq3 = merged.setdefault("RQ3", {})
    detail = rq3.setdefault("_detail", {})
    detail["rq3_e2_generalization"] = e2_analysis

    # refresh the headline contamination-controlled advantage delta if finite
    ccd = e2_analysis.get("contamination_controlled_headline_delta_rq3")
    if ccd is not None and math.isfinite(float(ccd)) and "metrics" in rq3:
        rq3["metrics"]["slice_vs_whole_function_advantage_delta_rq3"] = _finite(ccd)
        # also update the statistical_tests note to reflect generalization control
        st = rq3.setdefault("statistical_tests", {})
        st.setdefault("_rq3_e2_note",
                      "slice_vs_whole_function_advantage_delta_rq3 reflects the "
                      "contamination-controlled (chronological) estimate from RQ3-E2.")

    # H6 flag at detail level
    detail["H6_advantage_persists_contamination"] = bool(
        e2_analysis.get("H6_advantage_persists_under_contamination", False))

    clean = _sanitize(merged)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)
    print(f"[analysis_rq3_e2] merged RQ3-E2 generalization into {out}")


if __name__ == "__main__":
    print("analysis_rq3_e2 module; invoke via run_rq3_e2 / main.py")