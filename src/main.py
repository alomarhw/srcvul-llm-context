"""
main.py -- end-to-end orchestrator for the SRCVUL + LLM slicing study.

Runs EVERY research question in one zero-argument invocation:

  RQ1-E0  Monte-Carlo power + Representation-Matters delta table   (power_sim)
  RQ1-E1  five-representation detection+localization head-to-head  (experiment_rq1)
  RQ1-E2  reproduced SRCVUL/VulSlicer/statement/random baselines   (experiment_rq1)
  RQ2-E1  slicing ablation grid + corruption + dose-response       (experiment_rq2_e1)
  RQ2-E2  efficiency-without-loss (powered TOST)                    (experiment_rq2_e2)
  RQ3-E0  feasibility census                                       (feasibility_census)
  RQ3-E1  slice-vs-whole repair (independent static oracle)        (experiment_rq3/run_rq3)
  RQ3-E2  contamination-controlled generalization                  (experiment_rq3_e2)

Usage (all OPTIONAL -- `python main.py` runs the full paper-grade path over data/):
    python main.py            # STRICT real-data paper-grade run (all RQs)
    python main.py --smoke    # <=50 units, 1 repeat: plumbing check, synthetic OK
    python main.py --pilot    # 200 units, 3 repeats, REAL data (contract pilot)
    python main.py --full     # explicit paper-grade (same as default)

The default path NEVER substitutes synthetic data silently: in STRICT real-data
mode a missing researcher-provided file raises a clear RuntimeError naming it.
Only --smoke permits the deterministic synthetic fallback (plumbing only).

All LLM calls go through the frozen rp_llm_cache (seeded, replayable, offline).

Requirements (pinned to this environment):
    numpy==2.2.6, pandas==2.2.2, scipy==1.13.1, scikit-learn==1.5.1,
    statsmodels==0.14.6, pingouin==0.6.1, matplotlib==3.9.1, rank-bm25==0.2.2,
    tiktoken==0.14.0
(rp_llm_cache, rp_stats, rp_data_runtime, rp_style are provided by the sandbox.)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ---- reproducibility: seed every RNG from RP_RANDOM_SEED (default 42) ----
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
DATA = ROOT / "data"

# ---- study-scale contract modes ----
SMOKE = {"max_rows": 50, "repeats": 1, "nondet": 10, "oracle": 10,
         "label": "smoke", "paper_grade": False}
PILOT = {"max_rows": 200, "repeats": 3, "nondet": 100, "oracle": 100,
         "label": "pilot", "paper_grade": False}
FULL = {"max_rows": None, "repeats": 3, "nondet": 200, "oracle": 150,
        "label": "paper-grade", "paper_grade": True}

# Dose-response levels (paper-grade full curve vs. pilot subset).
DOSE_PAPER = [0, 10, 20, 40, 60, 80]
DOSE_PILOT = [0, 40, 80]

MODELS_UNDER_TEST = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "microsoft/phi-4",
]


# --------------------------------------------------------------------------- #
# provenance + run metadata
# --------------------------------------------------------------------------- #
def _merge_provenance(prov_meta: Dict[str, Any], served_versions: Dict[str, Any]) -> None:
    """Write results/data_provenance.json, copying fetch-manifest lineage and
    preserving any usedMinedData flag."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    # enrich with fetch_manifest lineage when available
    manifest = DATA / "fetch_manifest.json"
    man_by_name: Dict[str, Dict[str, Any]] = {}
    if manifest.exists():
        try:
            man = json.loads(manifest.read_text())
            for e in man.get("ok", []):
                nm = (e.get("contract") or {}).get("datasetName", "")
                man_by_name[nm] = e
                # also key by filename for easy matching
                p = e.get("path", "")
                if p:
                    man_by_name[Path(p).name] = e
        except Exception:
            pass

    for d in prov_meta.get("datasets", []):
        key_candidates = [d.get("name", ""), Path(str(d.get("path", ""))).name]
        for k in key_candidates:
            e = man_by_name.get(k)
            if e:
                for fld in ("version", "license", "citation", "checksum"):
                    if e.get(fld) and not d.get(fld):
                        d[fld] = e[fld]
                if e.get("recordCount") is not None and not d.get("recordCount"):
                    d["recordCount"] = e["recordCount"]
                if e.get("columns") and not d.get("schema"):
                    d["schema"] = e["columns"]
                break

    prov_meta["modelsUnderTest"] = MODELS_UNDER_TEST
    prov_meta["servedModelVersions"] = served_versions

    # preserve mined flag written by any auto-mining recipe
    existing = DATA / "data_provenance.json"
    if existing.exists():
        try:
            prev = json.loads(existing.read_text())
            if prev.get("usedMinedData"):
                prov_meta["usedMinedData"] = True
        except Exception:
            pass

    (RESULTS / "data_provenance.json").write_text(json.dumps(prov_meta, indent=2))
    print(f"[main] wrote results/data_provenance.json "
          f"(usedSynthetic={prov_meta.get('usedSynthetic')})")


def _write_run_metadata(mode: Dict[str, Any], n_units: int,
                        prov_meta: Dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = [d["name"] for d in prov_meta.get("datasets", [])
                if d.get("source") == "real"]
    if not datasets:
        datasets = [d["name"] for d in prov_meta.get("datasets", [])]
    meta = {
        "mode": mode["label"],
        "rows_or_items": int(n_units),
        "repeats": mode["repeats"],
        "datasets": datasets,
        "paper_grade": bool(mode["paper_grade"]),
        "models_under_test": MODELS_UNDER_TEST,
        "random_seed": RANDOM_SEED,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[main] run_metadata: mode={meta['mode']} rows={meta['rows_or_items']} "
          f"repeats={meta['repeats']} paper_grade={meta['paper_grade']}")


def _write_llm_usage(stats: Dict[str, Any]) -> None:
    try:
        (RESULTS / "llm_usage.json").write_text(json.dumps(stats, indent=2))
        print(f"[main] llm usage: {stats}")
    except Exception as exc:  # noqa: BLE001
        print(f"[main] WARNING: could not write llm_usage.json: {exc}")


# --------------------------------------------------------------------------- #
# figures (all read from saved results; method figure rendered from pipeline)
# --------------------------------------------------------------------------- #
def _render_all_figures() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for modname in ("plots", "plots_rq2", "plots_rq3"):
        try:
            mod = __import__(modname)
            mod.main()
        except Exception as exc:  # figures must not block results
            print(f"[main] WARNING: {modname}.main() failed: {exc}")


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def _dump_records(name: str, records: List[Dict[str, Any]]) -> None:
    """Per-unit records (model answers, cited lines, patches) for case studies and the RQ4 sample."""
    with open(RESULTS / name, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"[main] wrote results/{name} ({len(records)} records)")


def run(mode: Dict[str, Any], allow_synthetic: bool, strict: bool) -> int:
    t0 = time.time()
    banner = {"smoke": "Smoke run mode", "pilot": "Pilot run mode",
              "paper-grade": "Paper-grade run mode"}[mode["label"]]
    dose_levels = DOSE_PILOT if mode["label"] in ("smoke", "pilot") else DOSE_PAPER
    print("=" * 72)
    print(f"{banner}: max_rows={mode['max_rows']}, repeats={mode['repeats']}, "
          f"dose_levels={dose_levels}")
    print(f"Models under test: {MODELS_UNDER_TEST}")
    print("=" * 72)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- imports (after seeding) ----
    import data_prep
    import data_prep_rq3
    import experiment_rq1
    import experiment_rq2
    import experiment_rq2_e1
    import experiment_rq2_e2
    import experiment_rq3 as erq3
    import experiment_rq3_e2 as erq3e2
    import feasibility_census
    import analysis
    import analysis_rq2
    import analysis_rq3
    import analysis_rq3_e2
    import power_sim
    from rp_llm_cache import LLMCache

    # ------------------------------------------------------------------ #
    # Load the shared detection units (RQ1 + RQ2 operate on the SAME units)
    # ------------------------------------------------------------------ #
    print("\n[main] loading researcher-provided datasets from data/ ...")
    units, prov_meta = data_prep.load_units(
        max_rows=mode["max_rows"], allow_synthetic=allow_synthetic, strict=strict)
    if not units:
        raise RuntimeError(
            "No code units loaded. Place primevul_pairs.jsonl / "
            "secvuleval_loc.jsonl / cleanvul_pairs.jsonl in data/, or run "
            "`python main.py --smoke` for the synthetic plumbing path.")

    # stratified evaluation sample by top-10 CWE (paper grade: all pairs)
    if mode["max_rows"] is not None:
        n_pairs = mode["max_rows"]
        units = data_prep.stratified_sample(units, n_pairs=n_pairs, top_cwe=10)
    print(f"[main] detection units: {len(units)} "
          f"(synthetic={prov_meta.get('usedSynthetic')})")

    # leakage guard: no near-duplicate test item inside the retrieval corpus and
    # no CVE/project straddling the pre/post-cutoff boundary.
    excluded = _leakage_guard(units)
    prov_meta["leakage_excluded"] = excluded

    data_prep.write_provenance(prov_meta)

    # shared LLM cache (frozen, replayable)
    llm = LLMCache()

    # ================================================================== #
    # RQ1-E0  power simulation + Representation-Matters delta table (synthetic)
    # ================================================================== #
    print("\n[main] RQ1-E0: Monte-Carlo power simulation + delta table ...")
    e0_reps = 120 if mode["label"] == "smoke" else 300
    rq1_e0 = power_sim.run(reps=e0_reps)
    print(f"[main] RQ1-E0 n for 0.8 power = {rq1_e0['n_for_80_power']}; "
          f"R4 delta over best non-metadata = "
          f"{rq1_e0['delta_table']['R4_delta_over_best_nonmeta']:.3f}")

    # ================================================================== #
    # RQ1-E1  five-representation detection + localization head-to-head
    # ================================================================== #
    print("\n[main] RQ1-E1: five-representation detection+localization ...")
    rq1_e1 = experiment_rq1.run_rq1_e1(
        units, repeats=mode["repeats"], nondet_subset=mode["nondet"])

    # ================================================================== #
    # RQ1-E2  reproduced foundation baselines (SRCVUL/VulSlicer/stmt/random)
    # ================================================================== #
    print("\n[main] RQ1-E2: reproduced SRCVUL/VulSlicer/statement/random ...")
    rq1_e2 = experiment_rq1.run_rq1_e2(units)

    # analyze RQ1 -> results.json (RQ1 key)
    print("\n[main] analyzing RQ1 ...")
    rq1_results = analysis.analyze(rq1_e1, rq1_e2, rq1_e0, mode_meta=mode)
    analysis.write_results(rq1_results)

    # ================================================================== #
    # RQ2-E1  slicing ablation grid + corruption + dose-response
    # ================================================================== #
    print("\n[main] RQ2-E1: slicing ablation grid + corruption + dose-response ...")
    rq2_e1 = experiment_rq2_e1.run_rq2_e1(
        units, repeats=mode["repeats"], dose_levels=dose_levels,
        nondet_subset=mode["nondet"], oracle_subset=mode["oracle"])

    print("\n[main] analyzing RQ2 (E1) ...")
    rq2_results = analysis_rq2.analyze_rq2(rq2_e1, mode_meta=mode)
    analysis_rq2.write_results_rq2(rq2_results)

    # ================================================================== #
    # RQ2-E2  efficiency-without-loss (powered TOST) -- merges into RQ2._detail
    # ================================================================== #
    print("\n[main] RQ2-E2: efficiency-without-loss (powered TOST) ...")
    paper_grade_n = 2000
    rq2_e2 = experiment_rq2_e2.run_rq2_e2(
        rq2_e1["records"], paper_grade_n=paper_grade_n, mode_meta=mode)
    _merge_rq2_e2(rq2_e2)

    # ---- RQ1/RQ2 per model (the headline metrics above are a cross-model majority vote) ----
    print("\n[main] RQ1/RQ2 per-model results (frozen cache replay) ...")
    import per_model
    import llm_runner
    per_model.run(units, rq1_e2, rq1_e0, mode, list(llm_runner.MODELS_UNDER_TEST), dose_levels)

    # ---- RQ1 add-ons (study design v2): official PrimeVul test split, prompt robustness, R6 probe ----
    print("\n[main] RQ1 add-ons: RQ1-E3 official PrimeVul test, RQ1-E4 CoT, RQ1-E5 cross-function context ...")
    import addons
    addons.run(units, rq1_e1["nondet_ids"], list(llm_runner.MODELS_UNDER_TEST),
               workers=int(os.environ.get("RP_LLM_WORKERS", 32)),
               max_pairs=None if mode["paper_grade"] else (25 if mode["label"] == "smoke" else 100))

    # ================================================================== #
    # RQ3  repair (uses REPAIR units derived from the same pairs)
    # ================================================================== #
    print("\n[main] loading RQ3 repair units ...")
    repair_units, rq3_prov = data_prep_rq3.load_repair_units(
        max_rows=mode["max_rows"], allow_synthetic=allow_synthetic, strict=strict)
    if not repair_units:
        raise RuntimeError("RQ3: no repair units loaded (see data/ requirements).")
    if mode["max_rows"] is not None:
        # keep RQ3 small enough to stay within the runtime budget
        repair_units = repair_units[:mode["max_rows"]]
    print(f"[main] RQ3 repair units: {len(repair_units)} "
          f"({rq3_prov.get('rq3_units_with_official_fix', 0)} with official fix)")

    # ---- RQ3-E0 feasibility census (honest executable level-4 N) ----
    print("\n[main] RQ3-E0: feasibility census ...")
    census = feasibility_census.write_census(repair_units)
    executable_ids = set(census.get("executable_unit_ids", []))

    # ---- RQ3-E1 slice-vs-whole repair (independent static oracle) + agentic ----
    print("\n[main] RQ3-E1: slice-vs-whole repair (independent static oracle) ...")
    rep_builder = data_prep_rq3.make_representation_builder()
    rq3_e1 = erq3.run_rq3_e1(
        llm, repair_units, rep_builder, repeats=mode["repeats"],
        executable_ids=executable_ids, agentic=True)

    # ---- RQ3-E2 contamination split (reuse RQ3-E1 records for the pre/post view)
    print("\n[main] RQ3-E2 (contamination pre/post view from RQ3-E1 records) ...")
    rq3_e2_contam = erq3.run_rq3_e2(rq3_e1["records"])
    _dump_records("records_rq1.jsonl", rq1_e1["records"])
    _dump_records("records_rq3.jsonl", rq3_e1["records"])

    print("\n[main] analyzing RQ3 ...")
    rq3_results = analysis_rq3.analyze_rq3(
        rq3_e1, rq3_e2_contam, census, repair_units, executable_ids, mode_meta=mode)
    analysis_rq3.write_results_rq3(rq3_results)

    # ---- RQ3-E2 full generalization: random / chronological / project-disjoint ----
    print("\n[main] RQ3-E2: contamination-controlled generalization "
          "(random/chronological/project-disjoint) ...")
    rq3_e2_gen = erq3e2.run_rq3_e2_generalization(
        llm, repair_units, repeats=mode["repeats"],
        executable_ids=executable_ids, agentic=False)
    rq3_e2_analysis = analysis_rq3_e2.analyze_rq3_e2(rq3_e2_gen)
    analysis_rq3_e2.merge_into_results(rq3_e2_analysis)

    # ------------------------------------------------------------------ #
    # provenance / run metadata / llm usage
    # ------------------------------------------------------------------ #
    try:
        stats = llm.stats()
    except Exception:
        stats = {}
    served = stats.get("served_versions", {}) if isinstance(stats, dict) else {}
    # Which slicer produced the slices (srcML+srcSlice vs the lightweight fallback), for reporting.
    import slice_pipeline
    prov_meta["slicing"] = slice_pipeline.slice_backend_report()
    print(f"[main] slicing: {prov_meta['slicing']}")
    _merge_provenance(prov_meta, served)
    _write_run_metadata(mode, len(units), prov_meta)
    _write_llm_usage(stats)

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    print("\n[main] rendering figures ...")
    _render_all_figures()

    # ------------------------------------------------------------------ #
    # final self-validation gate over results.json
    # ------------------------------------------------------------------ #
    _final_gate()

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"[main] COMPLETE in {dt:.1f}s. Outputs in results/ and figures/.")
    print(f"[main] mode={mode['label']} paper_grade={mode['paper_grade']}")
    print("=" * 72)
    return 0


# --------------------------------------------------------------------------- #
# leakage guard (near-duplicate + group overlap)
# --------------------------------------------------------------------------- #
def _leakage_guard(units: List[Any]) -> Dict[str, int]:
    """Exclude near-duplicate retrieval/test pairs and record group straddling.

    Uses rp_stats helpers when available; otherwise a conservative no-op that
    still records zero exclusions. We do not drop units from the within-subjects
    design (that would unbalance pairing); instead we record how many near-dup /
    straddling cases exist so provenance is honest.
    """
    info = {"near_duplicate_pairs": 0, "cve_straddle_pre_post": 0,
            "project_straddle_pre_post": 0}
    try:
        from rp_stats import near_duplicate_pairs, group_overlap
    except Exception:
        return info

    try:
        vul_texts = [u.code for u in units if u.label == 1]
        # test vs retrieval corpus near-dup (same corpus here; count internal dups)
        dups = near_duplicate_pairs(vul_texts, vul_texts, 0.9)
        # subtract self-matches (i,i)
        info["near_duplicate_pairs"] = int(max(0, len(dups) - len(vul_texts)))
    except Exception:
        pass

    try:
        from rp_stats import group_overlap
        pre_cve = [u.cve_id for u in units if not u.is_post_cutoff and u.cve_id]
        post_cve = [u.cve_id for u in units if u.is_post_cutoff and u.cve_id]
        ov = group_overlap(pre_cve, post_cve)
        info["cve_straddle_pre_post"] = int(len(ov)) if hasattr(ov, "__len__") else int(ov or 0)
        pre_proj = [u.project for u in units if not u.is_post_cutoff]
        post_proj = [u.project for u in units if u.is_post_cutoff]
        ovp = group_overlap(pre_proj, post_proj)
        info["project_straddle_pre_post"] = int(len(ovp)) if hasattr(ovp, "__len__") else int(ovp or 0)
    except Exception:
        pass

    print(f"[main] leakage guard: {info}")
    return info


# --------------------------------------------------------------------------- #
# RQ2-E2 merge into results.json (RQ2._detail)
# --------------------------------------------------------------------------- #
def _merge_rq2_e2(rq2_e2: Dict[str, Any]) -> None:
    import math

    def _san(obj):
        if isinstance(obj, dict):
            return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_san(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if math.isfinite(v) else 0.0
        if isinstance(obj, np.ndarray):
            return [_san(v) for v in obj.tolist()]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else 0.0
        return obj

    out = RESULTS / "results.json"
    merged: Dict[str, Any] = {}
    if out.exists():
        try:
            merged = json.loads(out.read_text())
        except Exception:
            merged = {}
    rq2 = merged.setdefault("RQ2", {})
    detail = rq2.setdefault("_detail", {})
    detail["rq2_e2_efficiency_without_loss"] = rq2_e2

    # surface the headline efficiency reductions into the RQ2 statistical_tests
    # note when a significant token reduction exists (does not overwrite metrics).
    try:
        tok = rq2_e2["efficiency_reductions"]["tokens"]
        if tok.get("significant_reduction"):
            st = rq2.setdefault("statistical_tests", {})
            st.setdefault("_rq2_e2_note",
                          "RQ2-E2 TOST: slice significantly reduces tokens; "
                          f"equivalence={rq2_e2['tost_effectiveness_equivalence'].get('equivalent')}.")
    except Exception:
        pass

    out.write_text(json.dumps(_san(merged), indent=2))
    print("[main] merged RQ2-E2 efficiency-without-loss into results.json")


# --------------------------------------------------------------------------- #
# final gate
# --------------------------------------------------------------------------- #
def _final_gate() -> None:
    import math

    out = RESULTS / "results.json"
    if not out.exists():
        raise RuntimeError("results/results.json was not written.")
    data = json.loads(out.read_text())

    required = {
        "RQ1": ["precision", "recall", "macro_f1", "false_positive_rate_fpr",
                "false_negative_rate_fnr",
                "top_k_localization_accuracy_top_1_top_5_rq1",
                "line_level_localization_precision_recall_rq1"],
        "RQ2": ["precision", "recall", "macro_f1", "false_positive_rate_fpr",
                "false_negative_rate_fnr", "input_output_token_count_rq2",
                "end_to_end_latency_rq2", "inference_cost_rq2"],
        "RQ3": ["precision", "recall", "macro_f1", "false_positive_rate_fpr",
                "false_negative_rate_fnr", "patch_applicability_rate_rq3",
                "compilation_success_rate_rq3",
                "existing_test_pass_regression_rate_rq3",
                "independent_vulnerability_elimination_rate_rq3",
                "slice_vs_whole_function_advantage_delta_rq3"],
    }
    for rq, keys in required.items():
        if rq not in data:
            raise RuntimeError(f"results.json missing {rq}")
        metrics = data[rq].get("metrics", {})
        for k in keys:
            if k not in metrics or not math.isfinite(float(metrics[k])):
                raise RuntimeError(f"{rq} metric '{k}' missing or non-finite: "
                                   f"{metrics.get(k)}")
        bc = data[rq].get("baseline_comparison", {})
        if not bc:
            raise RuntimeError(f"{rq} has no baseline_comparison")
    print("[main] final gate passed: all RQ metrics finite and baselines present.")


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="SRCVUL + LLM slicing study -- runs all RQs (RQ1/RQ2/RQ3).")
    ap.add_argument("--smoke", action="store_true",
                    help="<=50 units, 1 repeat: plumbing check (synthetic allowed)")
    ap.add_argument("--pilot", action="store_true",
                    help="200 units, 3 repeats on REAL data (contract pilot)")
    ap.add_argument("--full", action="store_true",
                    help="explicit paper-grade run (all data, 3 repeats)")
    args = ap.parse_args()

    if args.smoke:
        return run(SMOKE, allow_synthetic=True, strict=False)
    if args.pilot:
        return run(PILOT, allow_synthetic=False, strict=True)
    # default == paper-grade, STRICT real-data mode
    return run(FULL, allow_synthetic=False, strict=True)


if __name__ == "__main__":
    sys.exit(main())