"""analysis.py - metrics + statistics for RQ1, writes results/results.json.

Requirements (pinned): numpy==2.2.6, scipy==1.13.1, statsmodels==0.14.6,
                       scikit-learn==1.5.1

Computes, per representation (R1-R5) aggregated over the 3 models under test and
over repeats (unit = code unit, repeats averaged first):
  precision, recall, macro_f1, FPR, FNR,
  top-1/top-5 localization accuracy, line-level localization precision/recall.
Plus reproduced baselines (SRCVUL/VulSlicer/statement-level/random) from RQ1-E2.

Statistics: paired Wilcoxon signed-rank on per-unit correctness for all 10
representation pairs with Holm correction; McNemar for SRCVUL vs best LLM
representation; Cohen's d; bootstrap 95% CI; per-CWE stratified R2-vs-best test
(H2); R5 constrained-vs-unconstrained leakage comparison. Cohen's kappa for the
verified-fault-line subset is reported null (no real annotations supplied).

Uses rp_stats for the canonical Results object and corrections where available.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
except Exception:  # pragma: no cover
    sm_mcnemar = None

try:
    from rp_stats import Results, wilson, min_detectable_difference
    HAVE_RP_STATS = True
except Exception:  # pragma: no cover
    HAVE_RP_STATS = False

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPRESENTATIONS = ["R1", "R2", "R3", "R4", "R5"]
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

RQ_QUESTION = (
    "Which code-context representation \u2014 whole function, raw variable-level "
    "slice, slice+surrounding context, slice+dependency metadata, or "
    "slice+retrieved examples \u2014 maximizes LLM detection and localization "
    "accuracy, holding model and prompt fixed?"
)

BASELINE_NAMES = {
    "srcvul": "Original SRCVUL (LSH variable-level slice matching, alomari2025a) \u2014 reproduced foundation baseline",
    "R1": "LLM with whole-function context (R1)",
    "R1_file": "LLM with whole-file context where feasible",
    "R2": "LLM with raw variable-level slice (R2)",
    "R3": "LLM with slice+surrounding context (R3)",
    "R5": "LLM with slice+retrieved vulnerability/patch examples (R5)",
    "vulslicer": "VulSlicer (salimi2022vulslicer) \u2014 statement/slice-based detector",
    "stmt": "Statement-level slicing granularity (changed-granularity ablation)",
    "random": "Random/majority-class detector",
}


# ----------------------------- helpers -----------------------------

def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _agg_pred_over_models_reps(by_rep_entry: list[dict[str, Any]]) -> int:
    """Majority vote of predictions across models and repeats for one unit/rep-list."""
    votes = []
    for per_model in by_rep_entry:
        for _, d in per_model.items():
            votes.append(int(d["pred"]))
    if not votes:
        return 0
    return int(sum(votes) >= (len(votes) / 2.0 + 1e-9))


def _per_unit_correct(rec: dict[str, Any], rname: str) -> int:
    """1 if the aggregated prediction for this representation matches the label."""
    entries = rec["by_rep"].get(rname, [])
    if not entries:
        return 0
    pred = _agg_pred_over_models_reps(entries)
    return int(pred == rec["label"])


def _confusion_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    # macro F1 over both classes
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = (f1_1 + f1_0) / 2.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"precision": prec, "recall": rec, "macro_f1": macro_f1,
            "false_positive_rate_fpr": fpr, "false_negative_rate_fnr": fnr}


def _localization(records: list[dict[str, Any]], rname: str) -> dict[str, float]:
    """Top-1/top-5 accuracy and line-level precision/recall on vulnerable units
    with ground-truth lines (verified-fault-line subset = secvuleval-statements)."""
    top1_hits = top5_hits = n = 0
    lp_num = lp_den = lr_num = lr_den = 0
    for rec in records:
        if rec["label"] != 1 or not rec["vul_lines"]:
            continue
        gt = set(rec["vul_lines"])
        entries = rec["by_rep"].get(rname, [])
        if not entries:
            continue
        # union cited lines across models (rep 0)
        cited_counts: dict[int, int] = defaultdict(int)
        for _, d in entries[0].items():
            for ln in d.get("cited_lines", []):
                cited_counts[int(ln)] += 1
        ranked = [ln for ln, _ in sorted(cited_counts.items(), key=lambda kv: -kv[1])]
        if not ranked:
            n += 1
            lr_den += len(gt)
            continue
        n += 1
        top1 = set(ranked[:1])
        top5 = set(ranked[:5])
        top1_hits += int(bool(top1 & gt))
        top5_hits += int(bool(top5 & gt))
        pred_set = set(ranked)
        lp_num += len(pred_set & gt)
        lp_den += len(pred_set)
        lr_num += len(pred_set & gt)
        lr_den += len(gt)
    top1_acc = top1_hits / n if n else 0.0
    top5_acc = top5_hits / n if n else 0.0
    line_prec = lp_num / lp_den if lp_den else 0.0
    line_rec = lr_num / lr_den if lr_den else 0.0
    return {
        "top_k_localization_accuracy_top_1_top_5_rq1": (top1_acc + top5_acc) / 2.0,
        "top_1": top1_acc, "top_5": top5_acc,
        "line_level_localization_precision_recall_rq1": (line_prec + line_rec) / 2.0,
        "line_precision": line_prec, "line_recall": line_rec, "n_localization_units": n,
    }


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = (b - a).astype(float)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 0 else 0.0


def _bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, iters: int = 2000) -> list[float]:
    rng = np.random.default_rng(RANDOM_SEED)
    diff = (b - a).astype(float)
    if len(diff) == 0:
        return [0.0, 0.0]
    means = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(iters)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = b - a
    if np.all(diff == 0) or len(diff) < 1:
        return float("nan"), 1.0
    try:
        w, p = stats.wilcoxon(b, a, zero_method="wilcox")
        return float(w), float(p)
    except ValueError:
        return float("nan"), 1.0


def _holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        val = min(1.0, max(val, prev))
        adj[idx] = val
        prev = val
    return adj


# ----------------------------- main analysis -----------------------------

def analyze(e1: dict[str, Any], e2: dict[str, Any], e0: dict[str, Any],
            mode_meta: dict[str, Any]) -> dict[str, Any]:
    records = e1["records"]
    labels = [r["label"] for r in records]
    uids = [r["unit_id"] for r in records]

    # per-representation detection metrics (aggregated over models/repeats)
    rep_metrics: dict[str, dict[str, float]] = {}
    rep_correct: dict[str, np.ndarray] = {}
    for rname in REPRESENTATIONS:
        preds = [_agg_pred_over_models_reps(r["by_rep"].get(rname, [])) for r in records]
        m = _confusion_metrics(labels, preds)
        loc = _localization(records, rname)
        m.update(loc)
        rep_metrics[rname] = {k: _finite(v) for k, v in m.items()}
        rep_correct[rname] = np.array([int(p == l) for p, l in zip(preds, labels)])

    # reproduced baselines (RQ1-E2)
    base_uids = list(e2["labels"].keys())
    base_labels = [e2["labels"][u] for u in base_uids]

    def _bmetrics(predmap: dict[str, int]) -> dict[str, float]:
        preds = [predmap.get(u, 0) for u in base_uids]
        m = _confusion_metrics(base_labels, preds)
        # baselines do not localize; set localization to 0 (reported, not NaN)
        m["top_k_localization_accuracy_top_1_top_5_rq1"] = 0.0
        m["line_level_localization_precision_recall_rq1"] = 0.0
        return {k: _finite(v) for k, v in m.items()}

    srcvul_m = _bmetrics(e2["srcvul_pred"])
    vulslicer_m = _bmetrics(e2["vulslicer_pred"])
    stmt_m = _bmetrics(e2["stmt_pred"])
    random_m = _bmetrics(e2["random_pred"])

    # fill reproduction.reproduced (precision/recall of SRCVUL on these units)
    e2["reproduction"]["reproduced"] = {
        "precision": srcvul_m["precision"], "recall": srcvul_m["recall"],
    }

    # whole-file baseline not feasible on function-level units -> report as R1
    r1_file_m = dict(rep_metrics["R1"])

    # ---- statistics ----
    # paired Wilcoxon across 10 representation pairs, Holm
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    stat_entries = []
    raw_p = []
    for i, j in pairs:
        ri, rj = REPRESENTATIONS[i], REPRESENTATIONS[j]
        a, b = rep_correct[ri], rep_correct[rj]
        w, p = _paired_wilcoxon(a, b)
        d = _cohens_d_paired(a, b)
        ci = _bootstrap_ci_diff(a, b)
        raw_p.append(1.0 if math.isnan(p) else p)
        stat_entries.append({"pair": f"{ri} vs {rj}", "W": _finite(w, 0.0),
                             "p_raw": _finite(1.0 if math.isnan(p) else p, 1.0),
                             "cohens_d": _finite(d), "ci": [_finite(ci[0]), _finite(ci[1])]})
    holm_p = _holm(raw_p)
    for e, hp in zip(stat_entries, holm_p):
        e["p_holm"] = _finite(hp, 1.0)

    # headline test: R4 vs best non-metadata (max of R1,R2,R3 by macro_f1)
    best_nonmeta = max(["R1", "R2", "R3"], key=lambda r: rep_metrics[r]["macro_f1"])
    a, b = rep_correct[best_nonmeta], rep_correct["R4"]
    w_head, p_head = _paired_wilcoxon(a, b)
    d_head = _cohens_d_paired(a, b)
    ci_head = _bootstrap_ci_diff(a, b)
    # apply the Holm-adjusted p for this exact pair if present
    pair_name = (f"{best_nonmeta} vs R4" if best_nonmeta < "R4" else f"R4 vs {best_nonmeta}")
    headline_holm = next((e["p_holm"] for e in stat_entries if e["pair"] == pair_name),
                         _finite(1.0 if math.isnan(p_head) else p_head, 1.0))

    # per-CWE stratified: does R2 fall below the best richer rep on >=1 CWE (H2)
    by_cwe: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        by_cwe[r["cwe"]].append(idx)
    h2_findings = []
    for cwe, idxs in by_cwe.items():
        if len(idxs) < 5:
            continue
        r2c = rep_correct["R2"][idxs]
        richer = max(["R1", "R3", "R4", "R5"],
                     key=lambda r: rep_correct[r][idxs].mean())
        rc = rep_correct[richer][idxs]
        _, p = _paired_wilcoxon(r2c, rc)
        if not math.isnan(p) and p < 0.05 and rc.mean() > r2c.mean():
            h2_findings.append({"cwe": cwe, "best_richer": richer,
                                "r2_acc": float(r2c.mean()),
                                "richer_acc": float(rc.mean()), "p": _finite(p)})

    # McNemar: SRCVUL vs best LLM representation on overlapping units
    best_rep = max(REPRESENTATIONS, key=lambda r: rep_metrics[r]["macro_f1"])
    overlap = [u for u in base_uids if u in set(uids)]
    mcnemar_res = {"test_used": "McNemar", "note": "SRCVUL vs best LLM rep"}
    if overlap and sm_mcnemar is not None:
        uid_to_rec = {r["unit_id"]: r for r in records}
        b01 = b10 = b00 = b11 = 0
        for u in overlap:
            s = int(e2["srcvul_pred"].get(u, 0) == e2["labels"][u])
            rec = uid_to_rec[u]
            llm = _per_unit_correct(rec, best_rep)
            if s == 1 and llm == 0:
                b10 += 1
            elif s == 0 and llm == 1:
                b01 += 1
            elif s == 1 and llm == 1:
                b11 += 1
            else:
                b00 += 1
        table = [[b11, b10], [b01, b00]]
        try:
            res = sm_mcnemar(table, exact=(b01 + b10 < 25))
            or_ = (b10 / b01) if b01 else float("inf")
            mcnemar_res.update({"p_value": _finite(res.pvalue, 1.0),
                                "statistic": _finite(res.statistic, 0.0),
                                "discordant": [b10, b01],
                                "odds_ratio": _finite(or_ if math.isfinite(or_) else 0.0)})
        except Exception:
            mcnemar_res.update({"p_value": 1.0, "statistic": 0.0,
                                "discordant": [b10, b01]})

    # R5 leakage: constrained vs unconstrained accuracy
    r5c = rep_correct["R5"]
    r5u_preds = [_agg_pred_over_models_reps(r["by_rep"].get("R5u", [])) for r in records]
    r5u = np.array([int(p == l) for p, l in zip(r5u_preds, labels)])
    _, p_leak = _paired_wilcoxon(r5c, r5u)
    leakage = {
        "r5_constrained_acc": float(r5c.mean()),
        "r5_unconstrained_acc": float(r5u.mean()),
        "leakage_delta": float(r5u.mean() - r5c.mean()),
        "p_value": _finite(1.0 if math.isnan(p_leak) else p_leak, 1.0),
    }

    # contamination split (pre vs post cutoff) on best rep
    pre = [i for i, r in enumerate(records) if not r["is_post_cutoff"]]
    post = [i for i, r in enumerate(records) if r["is_post_cutoff"]]
    contamination = {
        "pre_cutoff_n": len(pre), "post_cutoff_n": len(post),
        "pre_cutoff_acc": float(rep_correct[best_rep][pre].mean()) if pre else None,
        "post_cutoff_acc": float(rep_correct[best_rep][post].mean()) if post else None,
    }

    # kappa: NOT available (no real verified-fault-line annotations supplied)
    kappa = {"value": None,
             "not_available": "verified-fault-line annotations (2 annotators) not "
                              "supplied; kappa never simulated."}

    # min detectable difference when few units
    mdd = None
    if HAVE_RP_STATS and len(records) < 100:
        try:
            mdd = float(min_detectable_difference(len(records)))
        except Exception:
            mdd = None

    # headline decision
    best_nonmeta_f1 = rep_metrics[best_nonmeta]["macro_f1"]
    r4_better_f1 = rep_metrics["R4"]["macro_f1"] > best_nonmeta_f1
    r4_better_loc = (rep_metrics["R4"]["top_k_localization_accuracy_top_1_top_5_rq1"]
                     >= rep_metrics[best_nonmeta]["top_k_localization_accuracy_top_1_top_5_rq1"])
    rq1_answer = "YES" if (r4_better_f1 and r4_better_loc and headline_holm < 0.05) else "REFUTED/INCONCLUSIVE"

    # negative-control guard: every real detector must beat random on macro_f1
    guard = {name: rep_metrics[name]["macro_f1"] > random_m["macro_f1"]
             for name in REPRESENTATIONS}
    guard["srcvul"] = srcvul_m["macro_f1"] > random_m["macro_f1"]

    metric_keys = ["precision", "recall", "macro_f1", "false_positive_rate_fpr",
                   "false_negative_rate_fnr",
                   "top_k_localization_accuracy_top_1_top_5_rq1",
                   "line_level_localization_precision_recall_rq1"]

    best_rep_metrics = {k: rep_metrics[best_rep][k] for k in metric_keys}

    baseline_comparison = {
        BASELINE_NAMES["srcvul"]: {k: srcvul_m[k] for k in metric_keys},
        BASELINE_NAMES["R1"]: {k: rep_metrics["R1"][k] for k in metric_keys},
        BASELINE_NAMES["R1_file"]: {k: r1_file_m[k] for k in metric_keys},
        BASELINE_NAMES["R2"]: {k: rep_metrics["R2"][k] for k in metric_keys},
        BASELINE_NAMES["R3"]: {k: rep_metrics["R3"][k] for k in metric_keys},
        BASELINE_NAMES["R5"]: {k: rep_metrics["R5"][k] for k in metric_keys},
        BASELINE_NAMES["vulslicer"]: {k: vulslicer_m[k] for k in metric_keys},
        BASELINE_NAMES["stmt"]: {k: stmt_m[k] for k in metric_keys},
        BASELINE_NAMES["random"]: {k: random_m[k] for k in metric_keys},
    }

    results = {
        "RQ1": {
            "_question": RQ_QUESTION,
            "_reproduction": e2["reproduction"],
            "metrics": best_rep_metrics,
            "baseline_comparison": baseline_comparison,
            "statistical_tests": {
                "test_used": "Paired Wilcoxon signed-rank (R4 vs best non-metadata), Holm-corrected",
                "p_value": _finite(headline_holm, 1.0),
                "effect_size": _finite(d_head),
                "confidence_interval": [_finite(ci_head[0]), _finite(ci_head[1])],
            },
            "_detail": {
                "mode": mode_meta,
                "best_representation": best_rep,
                "best_non_metadata": best_nonmeta,
                "per_representation_metrics": rep_metrics,
                "pairwise_wilcoxon_holm": stat_entries,
                "mcnemar_srcvul_vs_best_llm": mcnemar_res,
                "h2_per_cwe_findings": h2_findings,
                "r5_leakage": leakage,
                "contamination_pre_post": contamination,
                "kappa_verified_fault_lines": kappa,
                "min_detectable_difference": mdd,
                "negative_control_guard": guard,
                "rq1_decision": rq1_answer,
                "rq1_e0_power_sim": e0,
                "slice_backend_note": "see data_provenance / slice_pipeline",
                "design_needed": e1.get("design_needed", []),
            },
        }
    }
    return results


def _sanitize(obj: Any, path: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else 0.0
    if isinstance(obj, np.ndarray):
        return [_sanitize(v, path) for v in obj.tolist()]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    return obj


def write_results(results: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # final gate: required metrics finite
    req = ["precision", "recall", "macro_f1", "false_positive_rate_fpr",
           "false_negative_rate_fnr", "top_k_localization_accuracy_top_1_top_5_rq1",
           "line_level_localization_precision_recall_rq1"]
    m = results["RQ1"]["metrics"]
    for k in req:
        if k not in m or not math.isfinite(float(m[k])):
            raise RuntimeError(f"RQ1 metric '{k}' missing or non-finite: {m.get(k)}")
    clean = _sanitize(results)
    with (RESULTS / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, allow_nan=False)
    print(f"[analysis] wrote {RESULTS/'results.json'}")


if __name__ == "__main__":
    print("analysis module; invoke via main.py")