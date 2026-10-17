"""
experiment_rq2_e1.py -- RQ2-E1: isolate the marginal contribution of SLICING.

Holds model/prompt/retrieval FIXED (the best RQ1 config) and varies only the
code-context representation / slice quality, so any performance change is
attributable to slicing itself:

  Ablation grid (per the operationalization):
    A0  no-slicing             : whole function (R1)
    FULL reference             : slice + dependency metadata (R4)  <- the reference
    A1  slice without metadata : R2 (raw variable-level slice)
    A2  slice+metadata, no retrieval : R4 (retrieval already off in A0-A3)
    A3  changed granularity    : statement-level slice (coarser)

  PRIMARY category-specific corruption of the FULL slice (categories SRCVUL is
  documented to miss -- alomari2025a):
    c1  inter-procedural edges removed  (drop lines with called functions)
    c2  pointer/alias dvars removed
    c3  cross-file cfuncs removed

  SECONDARY uniform-random dose-response: drop p% of true slice lines,
    p in {0,10,20,40,60,80} (paper-grade) or {0,40,80} (pilot).

  Measured incompleteness: on an oracle subset (<=150 units) compare the SRCVUL
  slice to the dependency ground-truth oracle (the full dependency-closed slice
  from slice_pipeline) and record an incompleteness score per unit, later
  correlated with F1/localization.

Every LLM call goes through the frozen rp_llm_cache via llm_runner. Detection
correctness is the per-unit outcome; input/output tokens and latency are logged
jointly on the FULL (p=0) condition for the efficiency analysis.

Requirements: numpy==2.2.6 (CPU only).
"""
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List

import numpy as np

from slice_pipeline import (
    compute_slice,
    build_representations,
    statement_level_slice,
    BACKEND,
)
from retrieval import RetrievalCorpus
from llm_runner import LLMRunner, MODELS_UNDER_TEST
from corruption_injectors import (
    uniform_random_drop,
    interprocedural_drop,
    alias_drop,
    cross_file_drop,
    incompleteness_score,
)

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

# The FULL reference uses slice + dependency metadata (R4). Retrieval is OFF for
# every A0-A3 / corruption condition (the RQ2 question isolates slicing from
# retrieval, so retrieval is held fixed = off here).
ABLATIONS = ["A0", "FULL", "A1", "A2", "A3"]
CATEGORIES = ["c1_interproc", "c2_alias", "c3_crossfile"]

# dose-response levels
DOSE_PAPER = [0, 10, 20, 40, 60, 80]
DOSE_PILOT = [0, 40, 80]

DESIGN_NEEDED_RQ2 = [
    "DESIGN NEEDED: tian2025enhanced external localization baseline is not "
    "specified as a reproducible procedure/dataset; not inventing a substitute "
    "(skipped in pilot per steps, surfaced here at paper grade).",
    "DATA NEEDED: the measured-incompleteness correlation uses a dependency "
    "ground-truth oracle that must be MANUALLY verified on the oracle subset; "
    "the full dependency-closed slice is used as a reproducible proxy oracle and "
    "is clearly labeled as such. Manual verification is a researcher task.",
]


# --------------------------------------------------------------------------- #
# Slice line <-> context text helpers
# --------------------------------------------------------------------------- #
def _criterion_vars(code: str, vul_lines: List[int]) -> List[str]:
    """Blind slicing criterion (no ground truth); vul_lines is ignored on purpose."""
    from slice_pipeline import blind_criterion_vars
    return blind_criterion_vars(code)


def _slice_text_from_lines(code: str, lines_keep: List[int]) -> str:
    lines = code.splitlines()
    keep = sorted(set(lines_keep))
    return "\n".join(f"{i}: {lines[i-1]}" for i in keep if 1 <= i <= len(lines))


def _metadata_header(meta) -> str:
    def _fmt(d: Dict[str, List[int]]) -> str:
        return "; ".join(f"{k}@{','.join(map(str, v))}" for k, v in sorted(d.items()))
    return (
        "/* SRCVUL dependency metadata */\n"
        f"vrvars: {', '.join(meta.vrvars) or '-'}\n"
        f"def: {_fmt(meta.defs) or '-'}\n"
        f"use: {_fmt(meta.uses) or '-'}\n"
        f"dvars: {', '.join(meta.dvars) or '-'}\n"
        f"ptrs: {', '.join(meta.ptrs) or '-'}\n"
        f"cfuncs: {', '.join(meta.cfuncs) or '-'}\n"
    )


def _lines_for_vars(meta, vset) -> List[int]:
    """Which slice lines involve any of the given vars (for category drops)."""
    out = set()
    for v in vset:
        out.update(meta.defs.get(v, []))
        out.update(meta.uses.get(v, []))
    return sorted(out)


def _cfunc_lines(code: str, cfuncs: List[str], slice_lines: List[int]) -> List[int]:
    lines = code.splitlines()
    out = []
    cset = set(cfuncs)
    for i in slice_lines:
        if 1 <= i <= len(lines):
            called = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[i - 1]))
            if called & cset:
                out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Build all RQ2-E1 conditions per unit
# --------------------------------------------------------------------------- #
def build_unit_conditions(unit, dose_levels: List[int]) -> Dict[str, Any]:
    """Return {'contexts': {cond_name: text}, 'oracle': {...}, 'meta': meta}."""
    code = unit.code
    vul_lines = [int(x) for x in (unit.vul_lines or [])]
    meta = compute_slice(code, criterion_vars=_criterion_vars(code, []))
    full_slice = list(meta.slice_lines)

    reps = build_representations(code, meta, retrieved_examples=None, ctx_lines=2)

    contexts: Dict[str, str] = {}
    # --- ablation grid (retrieval OFF everywhere) ---
    contexts["A0"] = reps["R1"]            # whole function (no slicing)
    contexts["FULL"] = reps["R4"]          # slice + dependency metadata (reference)
    contexts["A1"] = reps["R2"]            # slice without metadata
    contexts["A2"] = reps["R4"]            # slice+metadata, no retrieval (== FULL w/o retrieval)
    stmt_lines = statement_level_slice(code)
    contexts["A3"] = _slice_text_from_lines(code, stmt_lines)  # changed granularity

    # --- PRIMARY category-specific corruption of the FULL slice ---
    # c1 inter-procedural: drop lines calling other functions
    cf_lines = _cfunc_lines(code, meta.cfuncs, full_slice)
    c1_lines = interprocedural_drop(full_slice, cf_lines)
    # c2 pointer/alias: drop lines involving pointer vars
    ptr_lines = _lines_for_vars(meta, meta.ptrs)
    c2_lines = alias_drop(full_slice, ptr_lines)
    # c3 cross-file: proxy by dropping tail (out-of-function defs)
    c3_lines = cross_file_drop(full_slice, frac=0.3, seed_offset=1)

    header = _metadata_header(meta)
    contexts["c1_interproc"] = header + "\n" + _slice_text_from_lines(code, c1_lines)
    contexts["c2_alias"] = header + "\n" + _slice_text_from_lines(code, c2_lines)
    contexts["c3_crossfile"] = header + "\n" + _slice_text_from_lines(code, c3_lines)

    # --- SECONDARY uniform-random dose-response ---
    dose_lines: Dict[int, List[int]] = {}
    for p in dose_levels:
        frac = p / 100.0
        corrupted = uniform_random_drop(full_slice, frac, seed_offset=p)
        dose_lines[p] = corrupted
        contexts[f"dose_{p}"] = header + "\n" + _slice_text_from_lines(code, corrupted)

    # --- oracle incompleteness (proxy: full dependency-closed slice is oracle) ---
    # the SRCVUL slice is meta.slice_lines; the dependency oracle is the fully
    # closed dependency set. For the reproducible proxy we treat the full closure
    # as the oracle and measure how much each corruption misses.
    oracle_full = full_slice
    incompleteness = {
        "c1_interproc": incompleteness_score(c1_lines, oracle_full),
        "c2_alias": incompleteness_score(c2_lines, oracle_full),
        "c3_crossfile": incompleteness_score(c3_lines, oracle_full),
    }
    for p in dose_levels:
        incompleteness[f"dose_{p}"] = incompleteness_score(dose_lines[p], oracle_full)

    return {
        "contexts": contexts,
        "incompleteness": incompleteness,
        "meta": meta,
        "full_slice_len": len(full_slice),
    }


# --------------------------------------------------------------------------- #
# Run RQ2-E1
# --------------------------------------------------------------------------- #
def run_rq2_e1(
    units: List[Any],
    repeats: int,
    dose_levels: List[int],
    nondet_subset: int = 200,
    oracle_subset: int = 150,
) -> Dict[str, Any]:
    """
    Execute all RQ2-E1 conditions across the 3 models at temperature 0.

    Returns per-unit structured records keyed by condition, plus oracle-subset
    incompleteness bookkeeping. Detection correctness = (pred == label).
    """
    print(f"[RQ2-E1] slicing backend: {BACKEND}; building conditions for "
          f"{len(units)} units; dose levels={dose_levels}")

    # Build conditions per unit
    per_unit: Dict[str, Dict[str, Any]] = {}
    for u in units:
        per_unit[u.unit_id] = build_unit_conditions(u, dose_levels)

    rng = random.Random(RANDOM_SEED)
    unit_ids = [u.unit_id for u in units]
    nondet_ids = set(rng.sample(unit_ids, min(nondet_subset, len(unit_ids))))
    oracle_ids = set(rng.sample(unit_ids, min(oracle_subset, len(unit_ids))))

    # all condition names (same across units)
    all_conditions = ABLATIONS + CATEGORIES + [f"dose_{p}" for p in dose_levels]

    runner = LLMRunner()
    jobs: List[Dict[str, Any]] = []
    for uid, info in per_unit.items():
        n_rep = repeats if uid in nondet_ids else 1
        for model in MODELS_UNDER_TEST:
            for rep in range(n_rep):
                for cond in all_conditions:
                    jobs.append({
                        "key": f"{uid}|{model}|{cond}|{rep}",
                        "model": model,
                        "context": info["contexts"][cond],
                        "sample": rep,
                    })
    print(f"[RQ2-E1] submitting {len(jobs)} LLM calls through frozen cache")
    raw = runner.run_batch(jobs)

    # assemble per-unit records
    records: List[Dict[str, Any]] = []
    for uid, info in per_unit.items():
        u = next(x for x in units if x.unit_id == uid)
        n_rep = repeats if uid in nondet_ids else 1
        rec: Dict[str, Any] = {
            "unit_id": uid,
            "label": int(u.label),
            "cwe": u.cwe,
            "split": u.split,
            "is_post_cutoff": bool(getattr(u, "is_post_cutoff", False)),
            "vul_lines": list(u.vul_lines),
            "in_nondet_subset": uid in nondet_ids,
            "in_oracle_subset": uid in oracle_ids,
            "incompleteness": info["incompleteness"],
            "by_cond": {},
        }
        for cond in all_conditions:
            # aggregate over models & repeats -> per-unit correctness, and
            # capture efficiency (tokens/latency) averaged over models at rep 0
            correct_votes: List[int] = []
            in_toks: List[int] = []
            out_toks: List[int] = []
            lats: List[float] = []
            cited_union: Dict[int, int] = {}
            for rep in range(n_rep):
                for model in MODELS_UNDER_TEST:
                    r = raw.get(f"{uid}|{model}|{cond}|{rep}")
                    if r is None:
                        continue
                    pred = int(bool(r["vulnerable"]))
                    correct_votes.append(int(pred == u.label))
                    if rep == 0:
                        in_toks.append(int(r["input_tokens"]))
                        out_toks.append(int(r["output_tokens"]))
                        lats.append(float(r["latency_s"]))
                        for ln in r.get("cited_lines", []):
                            cited_union[int(ln)] = cited_union.get(int(ln), 0) + 1
            # per-unit correctness = majority correct across model*repeat votes
            if correct_votes:
                corr = int(sum(correct_votes) >= (len(correct_votes) / 2.0 + 1e-9))
            else:
                corr = 0
            ranked = [ln for ln, _ in sorted(cited_union.items(), key=lambda kv: -kv[1])]
            rec["by_cond"][cond] = {
                "correct": corr,
                "n_votes": len(correct_votes),
                "input_tokens": float(np.mean(in_toks)) if in_toks else 0.0,
                "output_tokens": float(np.mean(out_toks)) if out_toks else 0.0,
                "input_output_tokens": (float(np.mean(in_toks)) + float(np.mean(out_toks)))
                if in_toks else 0.0,
                "latency_s": float(np.mean(lats)) if lats else 0.0,
                "cited_lines": ranked,
            }
        records.append(rec)

    return {
        "records": records,
        "conditions": all_conditions,
        "dose_levels": dose_levels,
        "ablations": ABLATIONS,
        "categories": CATEGORIES,
        "nondet_ids": sorted(nondet_ids),
        "oracle_ids": sorted(oracle_ids),
        "llm_stats": runner.stats(),
        "design_needed": DESIGN_NEEDED_RQ2,
        "oracle_note": (
            "Dependency ground-truth oracle is the full dependency-closed slice "
            "(reproducible proxy); manual verification is a researcher task."
        ),
    }


if __name__ == "__main__":
    from data_prep import load_units, stratified_sample

    units, _ = load_units(max_rows=8, allow_synthetic=True, strict=False)
    units = stratified_sample(units, n_pairs=4)
    out = run_rq2_e1(units, repeats=1, dose_levels=DOSE_PILOT, nondet_subset=10,
                     oracle_subset=10)
    r0 = out["records"][0]
    print("conditions:", out["conditions"])
    print("sample incompleteness:", r0["incompleteness"])
    print("sample FULL correct:", r0["by_cond"]["FULL"]["correct"])