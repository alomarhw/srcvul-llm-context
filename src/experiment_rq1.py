"""experiment_rq1.py - RQ1-E1 and RQ1-E2 experiment drivers.

Requirements (pinned): numpy==2.2.6, scikit-learn==1.5.1

RQ1-E1: within-subjects paired comparison of 5 representations (R1-R5) over the
SAME code units across the 3 models under test, temperature=0. For the
nondeterminism subset (<=200 units) we run `repeats` samples to measure flip
rate. We also run an UNCONSTRAINED R5 variant for leakage quantification.

RQ1-E2: reproduce foundation baselines on the SAME units:
  - Original SRCVUL (LSH variable-level slice matching) -> detection by slice
    similarity to the known-vulnerable corpus.
  - VulSlicer (statement/slice-based detector).
  - Statement-level granularity ablation.
  - Random/majority-class negative control.
  DESIGN NEEDED (todo): which 2025-2026 SOTA detector (VulWeaver vs Multi-View-
  Graph) and the fine-tuned-LLM training protocol are NOT specified; we emit a
  design-needed note instead of inventing it.

This module returns a dict of per-unit records; analysis.py computes metrics and
runs the statistics. LLM outputs come ONLY from the frozen cache.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from slice_pipeline import compute_slice, build_representations, statement_level_slice, BACKEND
from retrieval import RetrievalCorpus
from llm_runner import LLMRunner, MODELS_UNDER_TEST
from corruption_injectors import uniform_random_drop

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))
REPRESENTATIONS = ["R1", "R2", "R3", "R4", "R5"]

DESIGN_NEEDED = [
    "DESIGN NEEDED: RQ1-E2 2025-2026 SOTA detector choice (VulWeaver cao2026vulweaver "
    "vs Multi-View-Graph dou2025a) and the fine-tuned-LLM training protocol/"
    "contamination-controlled split are not specified; not inventing a substitute.",
    "DESIGN NEEDED: verified-fault-line annotation (two annotators, ~200 units, "
    "Cohen's kappa) is a researcher task; kappa is reported null until real "
    "annotations are supplied (never simulated).",
]


def _criterion_vars(unit: Any) -> list[str]:
    """Slicing criterion WITHOUT ground truth (blind_criterion_vars): the same rule for every function, so the two
    versions of a pair are sliced alike and the slice cannot encode the label or the vulnerable lines."""
    from slice_pipeline import blind_criterion_vars
    return blind_criterion_vars(unit.code)


# ------------------------- RQ1-E1 (LLM representations) -------------------------

def build_unit_representations(units: list[Any], ctx_lines: int = 2,
                               k_retrieve: int = 2) -> dict[str, Any]:
    """Precompute slice + R1..R5 (constrained) and an R5_unconstrained per unit."""
    corpus = RetrievalCorpus(units)
    per_unit: dict[str, Any] = {}
    for u in units:
        meta = compute_slice(u.code, criterion_vars=_criterion_vars(u))
        ex_constrained = corpus.retrieve(u, k=k_retrieve, chronological=True)
        ex_unconstrained = corpus.retrieve(u, k=k_retrieve, chronological=False)
        reps = build_representations(u.code, meta, retrieved_examples=ex_constrained,
                                     ctx_lines=ctx_lines)
        reps_unc = build_representations(u.code, meta, retrieved_examples=ex_unconstrained,
                                         ctx_lines=ctx_lines)
        per_unit[u.unit_id] = {
            "unit": u, "meta": meta, "reps": reps,
            "R5_unconstrained": reps_unc["R5"],
        }
    return per_unit


def run_rq1_e1(units: list[Any], repeats: int, nondet_subset: int = 200) -> dict[str, Any]:
    """Return per-unit model outputs for every representation/model/repeat."""
    print(f"[RQ1-E1] slicing backend: {BACKEND}; building representations for "
          f"{len(units)} units")
    per_unit = build_unit_representations(units)

    # nondeterminism subset: run `repeats` samples for these; others get 1 sample
    rng = random.Random(RANDOM_SEED)
    unit_ids = [u.unit_id for u in units]
    nondet_ids = set(rng.sample(unit_ids, min(nondet_subset, len(unit_ids))))

    runner = LLMRunner()
    jobs: list[dict[str, Any]] = []
    for uid, info in per_unit.items():
        n_rep = repeats if uid in nondet_ids else 1
        for model in MODELS_UNDER_TEST:
            for rep in range(n_rep):
                for rname in REPRESENTATIONS:
                    jobs.append({
                        "key": f"{uid}|{model}|{rname}|{rep}",
                        "model": model, "context": info["reps"][rname], "sample": rep,
                    })
                # unconstrained R5 for leakage measurement (rep 0 only)
                if rep == 0:
                    jobs.append({
                        "key": f"{uid}|{model}|R5u|{rep}",
                        "model": model, "context": info["R5_unconstrained"], "sample": rep,
                    })
    print(f"[RQ1-E1] submitting {len(jobs)} LLM calls through frozen cache")
    raw = runner.run_batch(jobs)

    # assemble per-unit structured records
    records = []
    for uid, info in per_unit.items():
        u = info["unit"]
        rec: dict[str, Any] = {
            "unit_id": uid, "label": int(u.label), "cwe": u.cwe,
            "is_post_cutoff": bool(u.is_post_cutoff), "vul_lines": list(u.vul_lines),
            "vul_lines_source": u.vul_lines_source, "code_nlines": len(u.code.splitlines()),
            "in_nondet_subset": uid in nondet_ids, "by_rep": {},
        }
        for rname in REPRESENTATIONS + ["R5u"]:
            reps_out = []
            n_rep = repeats if uid in nondet_ids else 1
            for rep in range(n_rep if rname != "R5u" else 1):
                key = f"{uid}|{{m}}|{rname}|{rep}"
                per_model = {}
                for model in MODELS_UNDER_TEST:
                    r = raw.get(key.format(m=model))
                    if r is None:
                        continue
                    per_model[model] = {
                        "pred": int(bool(r["vulnerable"])),
                        "parsed": bool(r.get("parsed", True)),
                        "cited_lines": r["cited_lines"],
                        "input_tokens": r["input_tokens"],
                        "output_tokens": r["output_tokens"],
                        "latency_s": r["latency_s"],
                    }
                reps_out.append(per_model)
            rec["by_rep"][rname] = reps_out
        records.append(rec)

    return {"records": records, "nondet_ids": sorted(nondet_ids),
            "llm_stats": runner.stats(), "design_needed": DESIGN_NEEDED}


# ------------------------- RQ1-E2 (reproduced baselines) -------------------------

def _slice_fingerprint(code: str, crit: list[str]) -> np.ndarray:
    meta = compute_slice(code, criterion_vars=crit)
    lines = code.splitlines()
    text = "\n".join(lines[i - 1] for i in meta.slice_lines if 1 <= i <= len(lines))
    hv = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")
    return hv.transform([text or code]).toarray()[0]


def run_rq1_e2(units: list[Any]) -> dict[str, Any]:
    """Reproduce SRCVUL (LSH variable-level slice matching), VulSlicer
    (statement/slice), statement-level ablation, and random/majority control.

    SRCVUL: build a database of known-vulnerable slice fingerprints; a unit is
    flagged vulnerable if its nearest DB fingerprint (cosine via LSH-style hashed
    vectors) exceeds a threshold. We use a leave-pair-out protocol so a unit is
    never matched against its own sibling.
    """
    import re

    def crit(u):
        return _criterion_vars(u)

    # database = vulnerable units; leave out the queried pair
    fps = {u.unit_id: _slice_fingerprint(u.code, crit(u)) for u in units}
    vul_ids = [u.unit_id for u in units if u.label == 1]

    def base(uid):
        return uid.rsplit("__", 1)[0]

    # threshold chosen on fingerprint similarity distribution (median of positives)
    sims_pos = []
    for u in units:
        if u.label != 1:
            continue
        qb = base(u.unit_id)
        cand = [vid for vid in vul_ids if base(vid) != qb]
        if not cand:
            continue
        best = max(float(np.dot(fps[u.unit_id], fps[vid])) for vid in cand)
        sims_pos.append(best)
    thr = float(np.median(sims_pos)) if sims_pos else 0.5

    srcvul_pred, vulslicer_pred, stmt_pred, random_pred = {}, {}, {}, {}
    rng = random.Random(RANDOM_SEED)
    pos_rate = sum(u.label for u in units) / max(1, len(units))

    for u in units:
        qb = base(u.unit_id)
        cand = [vid for vid in vul_ids if base(vid) != qb]
        best = max((float(np.dot(fps[u.unit_id], fps[vid])) for vid in cand), default=0.0)
        srcvul_pred[u.unit_id] = int(best >= thr)

        # VulSlicer: statement/slice-based - flag if the slice contains known
        # unsafe sink calls (strcpy/memcpy/free/system/sprintf/gets) and matches DB.
        slice_meta = compute_slice(u.code, criterion_vars=crit(u))
        sinks = {"strcpy", "strcat", "sprintf", "gets", "memcpy", "system",
                 "free", "malloc", "alloca"}
        has_sink = bool(set(slice_meta.cfuncs) & sinks)
        vulslicer_pred[u.unit_id] = int(has_sink and best >= thr * 0.8)

        # statement-level granularity ablation: coarser slice -> same sink rule
        stmt_lines = statement_level_slice(u.code)
        stmt_text = "\n".join(u.code.splitlines()[i - 1] for i in stmt_lines
                              if 1 <= i <= len(u.code.splitlines()))
        stmt_pred[u.unit_id] = int(bool(set(re.findall(r"[A-Za-z_]\w*", stmt_text)) & sinks))

        # random / majority-class control
        random_pred[u.unit_id] = int(rng.random() < pos_rate)

    labels = {u.unit_id: int(u.label) for u in units}
    cwes = {u.unit_id: u.cwe for u in units}
    post = {u.unit_id: bool(u.is_post_cutoff) for u in units}

    return {
        "labels": labels, "cwes": cwes, "is_post_cutoff": post,
        "srcvul_pred": srcvul_pred, "vulslicer_pred": vulslicer_pred,
        "stmt_pred": stmt_pred, "random_pred": random_pred,
        "srcvul_threshold": thr,
        "reproduction": {
            # foundation SRCVUL reported 91% precision / 75% recall on its own
            # databases; we compute precision/recall here on PrimeVul units.
            "reported": {"precision": 0.91, "recall": 0.75},
            "reproduced": None,  # filled by analysis.py on these units
            "citation": "alomari2025a (SRCVUL, ICPC 2025)",
            "note": "Reproduced on PrimeVul units, NOT the original VULSLICER/VUDDY "
                    "databases; a different corpus so an exact match is not expected.",
        },
        "design_needed": DESIGN_NEEDED,
    }


if __name__ == "__main__":
    from data_prep import load_units, stratified_sample
    units, _ = load_units(max_rows=10, allow_synthetic=True, strict=False)
    units = stratified_sample(units, n_pairs=5)
    e2 = run_rq1_e2(units)
    print("SRCVUL threshold:", e2["srcvul_threshold"])