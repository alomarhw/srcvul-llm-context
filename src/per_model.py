"""per_model.py -- RQ1/RQ2 results for EACH model under test, not only the cross-model majority vote.

The pooled analyses (analysis.analyze / analysis_rq2.analyze_rq2) score one majority vote over all
models and repeats per unit, which can hide a model that behaves differently. Here every model is
re-scored on its own by re-running RQ1-E1 / RQ2-E1 with MODELS_UNDER_TEST narrowed to that model.
All responses replay from the frozen rp_llm_cache, so this adds no model calls.

Writes results["RQ1"]["_detail"]["per_model"] and results["RQ2"]["_detail"]["per_model"].
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"

RQ1_KEYS = ("precision", "recall", "macro_f1", "false_positive_rate_fpr", "false_negative_rate_fnr",
            "top_1", "top_5", "line_precision", "line_recall")
RQ2_KEYS = ("precision", "recall", "macro_f1", "false_positive_rate_fpr", "input_output_token_count_rq2")


def _san(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_san(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _pick(d: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def _pair_metrics(records: List[Dict[str, Any]], model: str, rep: str, top_cwe: int = 10) -> Dict[str, Any]:
    """Pair-wise detection (P-C, VP-S, both-vulnerable / both-safe rates), FPR, parse failures and a per-CWE
    breakdown for one model and representation, from RQ1-E1 records (repeat 0)."""
    pred, parsed, cwe, label, loc = {}, {}, {}, {}, {}
    for r in records:
        entry = (r["by_rep"].get(rep) or [{}])[0].get(model)
        if not entry:
            continue
        uid = r["unit_id"]
        pred[uid], parsed[uid], cwe[uid], label[uid] = entry["pred"], entry.get("parsed", True), r.get("cwe") or "?", r["label"]
        if r["label"] == 1 and r.get("vul_lines"):
            gt = set(r["vul_lines"]); cited = [int(x) for x in entry.get("cited_lines") or []]
            loc[uid] = len(gt & set(cited)) / len(gt)
    pairs: Dict[str, Dict[int, int]] = {}
    for uid, p in pred.items():
        pairs.setdefault(uid.rsplit("__", 1)[0], {})[label[uid]] = p
    full = {k: d for k, d in pairs.items() if 0 in d and 1 in d}

    def summary(keys) -> Dict[str, Any]:
        ds = [full[k] for k in keys]
        n = len(ds)
        pc = sum(1 for d in ds if d[1] == 1 and d[0] == 0); rev = sum(1 for d in ds if d[1] == 0 and d[0] == 1)
        return {"n_pairs": n, "pair_correct": pc, "pair_correct_rate": pc / n if n else 0.0, "vp_s": pc - rev,
                "pair_both_vulnerable_rate": sum(1 for d in ds if d[1] == 1 and d[0] == 1) / n if n else 0.0,
                "pair_both_safe_rate": sum(1 for d in ds if d[1] == 0 and d[0] == 0) / n if n else 0.0}

    neg = [u for u in pred if label[u] == 0]
    out = summary(list(full))
    out["fpr"] = sum(pred[u] for u in neg) / len(neg) if neg else 0.0
    out["parse_failure_rate"] = sum(1 for u in pred if not parsed[u]) / len(pred) if pred else 0.0
    pair_cwe = {k: cwe[k + "__vul"] for k in full if k + "__vul" in cwe}
    counts: Dict[str, int] = {}
    for c in pair_cwe.values():
        counts[c] = counts.get(c, 0) + 1
    per_cwe = {}
    for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_cwe]:
        keys = [k for k, v in pair_cwe.items() if v == c]
        lr = [loc[k + "__vul"] for k in keys if k + "__vul" in loc]
        per_cwe[c] = {**summary(keys), "line_recall": float(np.mean(lr)) if lr else None}
    out["per_cwe_top10"] = per_cwe
    return out


def _nondeterminism(records: List[Dict[str, Any]], model: str, rep: str) -> Dict[str, Any]:
    """Flip rate (share of repeated units whose verdict differs across repeats) and Cohen's kappa between
    repeats 0 and 1, on the RQ1 nondeterminism subset."""
    runs = []
    for r in records:
        entries = r["by_rep"].get(rep) or []
        preds = [e[model]["pred"] for e in entries if model in e]
        if len(preds) >= 2:
            runs.append(preds)
    if not runs:
        return {"n_units": 0, "flip_rate": None, "kappa_rep0_rep1": None}
    flip = sum(1 for ps in runs if len(set(ps)) > 1) / len(runs)
    a = np.array([ps[0] for ps in runs]); b = np.array([ps[1] for ps in runs])
    po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n_units": len(runs), "flip_rate": flip, "kappa_rep0_rep1": kappa}


def run(units: List[Any], rq1_e2: Dict[str, Any], rq1_e0: Dict[str, Any], mode: Dict[str, Any],
        models: List[str], dose_levels: List[int]) -> Dict[str, Any]:
    import analysis
    import analysis_rq2
    import experiment_rq1
    import experiment_rq2_e1
    import experiment_rq2_e2

    saved = (list(experiment_rq1.MODELS_UNDER_TEST), list(experiment_rq2_e1.MODELS_UNDER_TEST))
    rq1_pm: Dict[str, Any] = {}
    rq2_pm: Dict[str, Any] = {}
    try:
        for m in models:
            print(f"[per_model] {m}: RQ1 + RQ2 from the frozen cache ...", flush=True)
            experiment_rq1.MODELS_UNDER_TEST[:] = [m]
            e1 = experiment_rq1.run_rq1_e1(units, repeats=mode["repeats"], nondet_subset=mode["nondet"])
            r1 = analysis.analyze(e1, rq1_e2, rq1_e0, mode_meta=mode)["RQ1"]
            d1 = r1.get("_detail", {})
            rq1_pm[m] = {
                "per_representation": {k: _pick(v, RQ1_KEYS)
                                       for k, v in (d1.get("per_representation_metrics") or {}).items()},
                "pairwise": {r: _pair_metrics(e1["records"], m, r) for r in experiment_rq1.REPRESENTATIONS},
                "nondeterminism": {r: _nondeterminism(e1["records"], m, r) for r in experiment_rq1.REPRESENTATIONS},
                "best_non_metadata": d1.get("best_non_metadata"),
                "r4_vs_best_test": r1.get("statistical_tests"),
                "srcvul_vs_best_llm": d1.get("mcnemar_srcvul_vs_best_llm"),
                "contamination_pre_post": d1.get("contamination_pre_post"),
                "rq1_decision": d1.get("rq1_decision"),
            }
            experiment_rq2_e1.MODELS_UNDER_TEST[:] = [m]
            e2 = experiment_rq2_e1.run_rq2_e1(units, repeats=mode["repeats"], dose_levels=dose_levels,
                                              nondet_subset=mode["nondet"], oracle_subset=mode["oracle"])
            r2 = analysis_rq2.analyze_rq2(e2, mode_meta=mode)["RQ2"]
            d2 = r2.get("_detail", {})
            tost = experiment_rq2_e2.run_rq2_e2(e2["records"], paper_grade_n=2000, mode_meta=mode)
            rq2_pm[m] = {
                "per_condition": {k: _pick(v, RQ2_KEYS) for k, v in (d2.get("per_condition_metrics") or {}).items()},
                "ablation_test": r2.get("statistical_tests"),
                "efficiency_without_loss_tost": (tost or {}).get("tost_effectiveness_equivalence"),
                "efficiency_reductions": (tost or {}).get("efficiency_reductions"),
                "rq2_e2_decision": (tost or {}).get("rq2_e2_decision"),
                "rq2_decision": d2.get("rq2_decision"),
            }
    finally:
        experiment_rq1.MODELS_UNDER_TEST[:] = saved[0]
        experiment_rq2_e1.MODELS_UNDER_TEST[:] = saved[1]

    out = RESULTS / "results.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    note = ("Each model scored on its own (the headline RQ metrics are a cross-model majority vote); "
            "replayed from the frozen response cache, no extra model calls.")
    merged.setdefault("RQ1", {}).setdefault("_detail", {})["per_model"] = _san({"note": note, "models": rq1_pm})
    merged.setdefault("RQ2", {}).setdefault("_detail", {})["per_model"] = _san({"note": note, "models": rq2_pm})
    out.write_text(json.dumps(merged, indent=2))
    print(f"[per_model] wrote per-model RQ1/RQ2 results for {len(models)} models", flush=True)
    return {"RQ1": rq1_pm, "RQ2": rq2_pm}
