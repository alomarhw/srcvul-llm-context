"""
experiment_rq3_e2.py -- RQ3-E2 contamination-controlled generalization.

This is a DISTINCT experiment from RQ3-E1. RQ3-E1 asks whether slice-conditioned
repair eliminates MORE vulnerabilities than whole-function repair under the
independent static oracle. RQ3-E2 asks whether that advantage GENERALIZES under
contamination control, by re-running the best slice representation (R4) vs
whole-function (R1) under THREE splits and measuring attenuation:

  (a) random split            -- shuffle units; no leakage control (reference)
  (b) chronological split     -- test = CVEs after a cutoff; retrieval restricted
                                 to strictly pre-cutoff examples (historical
                                 patches withheld at test time)
  (c) project-disjoint split  -- no repository shared between train/retrieval and
                                 test (project-level separation)

For R5 (slice + retrieved vulnerability/patch examples) we run BOTH a
chronology-constrained variant and an UNCONSTRAINED variant to quantify the
leakage gap that any retrieval benefit carries (acceptanceChecks: "R5 reported
both chronology-constrained and unconstrained").

Design integrity honored:
  * Elimination judged ONLY by the INDEPENDENT static oracle in repair_pipeline
    (parse + severance + closer-to-official-fix + cross-family re-detection);
    never compilation/test-pass alone.
  * The official fix (func_after) is withheld from the repairing model and from
    retrieval context at test time (no leakage); it is used only by the oracle.
  * The unit of analysis is the code unit; repeats are aggregated per unit before
    any statistics (done downstream in analysis_rq3_e2).
  * All LLM calls go through the frozen rp_llm_cache via repair_pipeline.
  * >= 2 version-reported models are used (modelsUnderTest).

DESIGN/DATA NEEDED surfaced (not guessed):
  * FlawFix / SPDAgent external APR anchors are not available as reproducible
    procedures here; we anchor against the whole-function LLM APR (R1) and the
    deterministic no-LLM static-template lower bound instead, clearly labeled.

Requirements (pinned): numpy==2.2.6, pandas==2.2.2
"""
from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List

import numpy as np

import repair_pipeline as rp
from retrieval import RetrievalCorpus

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

MODELS_UNDER_TEST = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "microsoft/phi-4",
]

# The two representations compared under every split. R4 = dependency-enriched
# variable-level slice (the proposed slice conditioning); R1 = whole-function.
SPLIT_REPS = ["R1", "R4"]

# Chronological cutoff: test units are those published at/after this year. The
# dataset's `is_post_cutoff` / cve_year carry the real dates; this threshold is
# the contamination boundary used for the chronological split's TRAIN/TEST line.
CHRONO_CUTOFF_YEAR = int(os.environ.get("RP_CHRONO_CUTOFF_YEAR", 2022))

DESIGN_NEEDED_E2 = [
    "DESIGN NEEDED: external APR anchors FlawFix and SPDAgent are not available "
    "as reproducible procedures in this environment; RQ3-E2 anchors the slice "
    "advantage against the whole-function LLM APR (R1) and the deterministic "
    "no-LLM static-template lower bound, clearly labeled. Supply runnable APR "
    "tools to add them as baselines.",
]


# --------------------------------------------------------------------------- #
# A lightweight retrieval-example wrapper so R5 contexts can be built per split
# --------------------------------------------------------------------------- #
class _UnitView:
    """Adapt a repair-unit dict to the attribute shape RetrievalCorpus expects."""

    __slots__ = ("unit_id", "code", "cwe", "label", "cve_year", "project")

    def __init__(self, u: Dict[str, Any]):
        self.unit_id = u["unit_id"] + "__vul"
        self.code = u.get("func_before", "")
        self.cwe = u.get("cwe", "CWE-unknown")
        self.label = 1
        self.cve_year = int(u.get("cve_year", 0) or 0)
        self.project = u.get("project", "unknown")


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #
def build_splits(units: List[Dict[str, Any]],
                 test_frac: float = 0.3) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Return {split_name: {"train": [...], "test": [...]}} for the three splits.

    * random: shuffle and take `test_frac` as test.
    * chronological: test = units with cve_year >= CHRONO_CUTOFF_YEAR OR
      is_post_cutoff; train/retrieval = strictly earlier units.
    * project_disjoint: hold out whole projects as test such that no project
      appears in both train and test, targeting ~test_frac of units.
    """
    rng = random.Random(RANDOM_SEED)

    # ---- random split ----
    idx = list(range(len(units)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(units) * test_frac)))
    rand_test_idx = set(idx[:n_test])
    random_split = {
        "train": [units[i] for i in range(len(units)) if i not in rand_test_idx],
        "test": [units[i] for i in rand_test_idx],
    }

    # ---- chronological split ----
    def _is_post(u: Dict[str, Any]) -> bool:
        return bool(u.get("is_post_cutoff")) or \
            int(u.get("cve_year", 0) or 0) >= CHRONO_CUTOFF_YEAR
    chrono_test = [u for u in units if _is_post(u)]
    chrono_train = [u for u in units if not _is_post(u)]
    # if the real post-cutoff population is empty/tiny, fall back to the latest
    # `test_frac` of units by year so the split is still meaningful (labeled).
    chrono_fallback = False
    if len(chrono_test) < max(2, n_test // 4):
        chrono_fallback = True
        by_year = sorted(units, key=lambda u: int(u.get("cve_year", 0) or 0))
        chrono_train = by_year[:len(by_year) - n_test]
        chrono_test = by_year[len(by_year) - n_test:]
    chronological_split = {
        "train": chrono_train, "test": chrono_test,
        "_fallback_by_year": chrono_fallback,
    }

    # ---- project-disjoint split ----
    by_project: Dict[str, List[Dict[str, Any]]] = {}
    for u in units:
        by_project.setdefault(u.get("project", "unknown"), []).append(u)
    projects = list(by_project.keys())
    rng.shuffle(projects)
    test_projects: set = set()
    test_count = 0
    for p in projects:
        if test_count >= n_test and len(test_projects) >= 1:
            break
        test_projects.add(p)
        test_count += len(by_project[p])
    # guard: never put ALL projects in test
    if len(test_projects) >= len(projects) and len(projects) > 1:
        test_projects.discard(projects[0])
    proj_test = [u for u in units if u.get("project", "unknown") in test_projects]
    proj_train = [u for u in units if u.get("project", "unknown") not in test_projects]
    project_disjoint_split = {
        "train": proj_train, "test": proj_test,
        "_n_test_projects": len(test_projects),
        "_n_total_projects": len(projects),
    }

    return {
        "random": random_split,
        "chronological": chronological_split,
        "project_disjoint": project_disjoint_split,
    }


# --------------------------------------------------------------------------- #
# Per-split representation builder (handles strict pre-cutoff retrieval for R5)
# --------------------------------------------------------------------------- #
def make_split_rep_builder(train_units: List[Dict[str, Any]],
                           chronological: bool,
                           r5_chronology_constrained: bool = True
                           ) -> Callable[[Dict[str, Any], str], str]:
    """
    Build a rep_builder closed over this split's TRAIN set for retrieval. For R5
    the retrieval corpus is drawn ONLY from `train_units` (never the test set),
    and when `r5_chronology_constrained` is True, examples are additionally
    restricted to those strictly pre-dating the query unit's CVE year.
    """
    import re
    from slice_pipeline import compute_slice, build_representations

    corpus = RetrievalCorpus([_UnitView(u) for u in train_units]) if train_units \
        else None

    def _criterion_vars(code: str, vul_lines: List[int]) -> List[str]:
        """Blind slicing criterion (no ground truth); vul_lines is ignored on purpose."""
        from slice_pipeline import blind_criterion_vars
        return blind_criterion_vars(code)

    def rep_builder(unit: Dict[str, Any], representation: str) -> str:
        code = unit.get("func_before", "")
        vul_lines = [int(x) for x in (unit.get("vul_lines") or [])]
        meta = compute_slice(code, criterion_vars=_criterion_vars(code, []))

        retrieved = None
        if representation == "R5" and corpus is not None:
            qv = _UnitView(unit)
            retrieved = corpus.retrieve(
                qv, k=2, chronological=(chronological or r5_chronology_constrained))
        reps = build_representations(code, meta, retrieved_examples=retrieved,
                                    ctx_lines=2)
        out = reps.get(representation, reps["R1"])
        # leakage guard: never let the official fix appear in the context
        fix = unit.get("func_after", "")
        if fix and fix in out:
            out = reps["R1"]  # fall back to the leak-free whole function
        return out

    return rep_builder


# --------------------------------------------------------------------------- #
# Run RQ3-E2
# --------------------------------------------------------------------------- #
def run_rq3_e2_generalization(
    llm,
    units: List[Dict[str, Any]],
    repeats: int,
    executable_ids: set,
    agentic: bool = False,
) -> Dict[str, Any]:
    """
    For each split (random, chronological, project_disjoint) and each model,
    generate R1 (whole-function) and R4 (slice+metadata) repairs on the TEST set
    with retrieval drawn strictly from that split's TRAIN set, judge elimination
    with the independent static oracle, and record per-unit outcomes.

    Also run R5 twice (chronology-constrained vs unconstrained) on the
    chronological split's test set for the leakage-gap measurement.

    Returns a nested record structure consumed by analysis_rq3_e2.
    """
    splits = build_splits(units)
    split_meta = {
        name: {
            "n_train": len(s["train"]), "n_test": len(s["test"]),
            **{k: v for k, v in s.items() if k.startswith("_")},
        }
        for name, s in splits.items()
    }
    print("[RQ3-E2] splits:",
          {k: (v["n_train"], v["n_test"]) for k, v in split_meta.items()})

    records: List[Dict[str, Any]] = []
    for split_name, split in splits.items():
        test_units = split["test"]
        train_units = split["train"]
        if not test_units:
            print(f"[RQ3-E2] split '{split_name}' has no test units; skipping")
            continue
        is_chrono = (split_name == "chronological")
        rep_builder = make_split_rep_builder(
            train_units, chronological=is_chrono, r5_chronology_constrained=True)

        for model in MODELS_UNDER_TEST:
            for rep in SPLIT_REPS:
                print(f"[RQ3-E2] split={split_name} model={model} rep={rep} "
                      f"(n_test={len(test_units)})")
                recs = rp.generate_and_validate(
                    llm, test_units, rep_builder, representation=rep,
                    model=model, repeats=repeats, agentic=agentic,
                    executable_ids=executable_ids)
                for r in recs:
                    r["split"] = split_name
                    r["generalization_rep"] = rep
                records.extend(recs)

    # ---- R5 leakage gap on the chronological split's test set ----
    r5_records: List[Dict[str, Any]] = []
    chrono = splits.get("chronological", {})
    chrono_test = chrono.get("test", [])
    chrono_train = chrono.get("train", [])
    if chrono_test:
        for constrained in (True, False):
            tag = "constrained" if constrained else "unconstrained"
            rep_builder_r5 = make_split_rep_builder(
                chrono_train, chronological=False,
                r5_chronology_constrained=constrained)
            for model in MODELS_UNDER_TEST:
                print(f"[RQ3-E2] R5 leakage ({tag}) model={model} "
                      f"(n_test={len(chrono_test)})")
                recs = rp.generate_and_validate(
                    llm, chrono_test, rep_builder_r5, representation="R5",
                    model=model, repeats=1, agentic=False,
                    executable_ids=executable_ids)
                for r in recs:
                    r["split"] = "chronological"
                    r["r5_retrieval"] = tag
                r5_records.extend(recs)

    try:
        llm_stats = llm.stats()
    except Exception:
        llm_stats = {}

    return {
        "records": records,
        "r5_leakage_records": r5_records,
        "split_meta": split_meta,
        "chrono_cutoff_year": CHRONO_CUTOFF_YEAR,
        "llm_stats": llm_stats,
        "design_needed": DESIGN_NEEDED_E2,
        "oracle": "independent_static",
    }


if __name__ == "__main__":
    # Offline self-check of split construction (no LLM calls).
    demo = []
    for i in range(30):
        demo.append({
            "unit_id": f"u{i}",
            "func_before": "int f(char*p){char b[8];strcpy(b,p);return 0;}",
            "func_after": "int f(char*p){char b[8];strncpy(b,p,7);b[7]=0;return 0;}",
            "vul_lines": [1],
            "cwe": f"CWE-{100 + i % 3}",
            "project": f"proj{i % 5}",
            "cve_year": 2010 + (i % 15),
            "is_post_cutoff": (2010 + (i % 15)) >= CHRONO_CUTOFF_YEAR,
        })
    sp = build_splits(demo)
    for name, s in sp.items():
        tr = {u["project"] for u in s["train"]}
        te = {u["project"] for u in s["test"]}
        print(f"{name}: train={len(s['train'])} test={len(s['test'])}")
        if name == "project_disjoint":
            assert not (tr & te), "project_disjoint leaked a shared project!"
    print("experiment_rq3_e2 split self-check OK")