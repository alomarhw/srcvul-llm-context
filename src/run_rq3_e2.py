"""
run_rq3_e2.py -- end-to-end RQ3-E2 contamination-controlled generalization.

This is the dedicated runner for the RQ3-E2 experiment (distinct from RQ3-E1).
It can run standalone (`python run_rq3_e2.py`) or be invoked by main.py.

Pipeline:
  1. load researcher-provided repair units (data_prep_rq3) -- STRICT real-data by
     default; --smoke uses the deterministic synthetic fallback for plumbing only.
  2. RQ3-E0 feasibility census (so executable level-4 N is honest before claims).
  3. RQ3-E2 generalization: slice (R4) vs whole-function (R1) repair under
     random / chronological / project-disjoint splits across the models under
     test, with retrieval strictly from each split's train set and the official
     fix withheld; plus the R5 chronology leakage gap.
  4. analysis_rq3_e2: per-split/per-model deltas + bootstrap 95% CIs, paired
     Wilcoxon attenuation, Holm correction, R5 leakage gap, H6 persistence.
  5. merge into results/results.json (RQ3._detail.rq3_e2_generalization) and
     refresh the headline contamination-controlled advantage delta.
  6. write run_metadata.json + data_provenance.json; render figures via plots_rq3.

Default (no args) = STRICT paper-grade over data/. --smoke = synthetic plumbing.
--pilot = contract pilot size (200 units, 3 repeats) on real data. --full =
explicit paper-grade.

Requirements (pinned): numpy==2.2.6, pandas==2.2.2
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from rp_llm_cache import LLMCache

import data_prep
import data_prep_rq3
import feasibility_census
import experiment_rq3_e2 as erq3e2
import analysis_rq3_e2 as arq3e2

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

SMOKE = {"max_rows": 25, "repeats": 1, "label": "smoke", "paper_grade": False}
PILOT = {"max_rows": 200, "repeats": 3, "label": "pilot", "paper_grade": False}
FULL = {"max_rows": None, "repeats": 3, "label": "paper-grade", "paper_grade": True}


def _write_run_metadata(mode: Dict[str, Any], units: List[Dict[str, Any]],
                        prov: Dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = [d["name"] for d in prov.get("datasets", [])
                if d.get("source") == "real"]
    if not datasets:
        datasets = [d["name"] for d in prov.get("datasets", [])]
    meta = {
        "experiment": "RQ3-E2",
        "mode": mode["label"],
        "rows_or_items": len(units),
        "repeats": mode["repeats"],
        "datasets": datasets,
        "paper_grade": bool(mode["paper_grade"]),
        "models_under_test": erq3e2.MODELS_UNDER_TEST,
    }
    # preserve any existing run_metadata (e.g. written by run_rq3) by merging
    existing = {}
    p = RESULTS / "run_metadata.json"
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:
            existing = {}
    existing.setdefault("runs", [])
    if isinstance(existing.get("runs"), list):
        existing["runs"].append(meta)
    existing.update({k: meta[k] for k in ("mode", "rows_or_items", "repeats",
                                          "datasets", "paper_grade")})
    existing["models_under_test"] = meta["models_under_test"]
    p.write_text(json.dumps(existing, indent=2))
    print(f"[run_rq3_e2] run_metadata: mode={meta['mode']} "
          f"rows={meta['rows_or_items']} repeats={meta['repeats']} "
          f"paper_grade={meta['paper_grade']}")


def run(mode: Dict[str, Any], allow_synthetic: bool, strict: bool) -> None:
    label = mode["label"]
    banner = "Smoke run mode" if label == "smoke" else \
        ("Pilot run mode" if label == "pilot" else "Paper-grade run mode")
    print(f"{banner}: max_rows={mode['max_rows']}, repeats={mode['repeats']}")

    # 1. load units
    units, prov = data_prep_rq3.load_repair_units(
        max_rows=mode["max_rows"], allow_synthetic=allow_synthetic, strict=strict)
    if not units:
        raise RuntimeError(
            "RQ3-E2: no repair units loaded. Place primevul_pairs.jsonl / "
            "secvuleval_loc.jsonl / cleanvul_pairs.jsonl (and optionally "
            "postcutoff_pairs.jsonl) in data/, or run with --smoke.")
    print(f"[run_rq3_e2] loaded {len(units)} repair units "
          f"({prov.get('rq3_units_with_official_fix', 0)} with official fix)")
    prov["mode"] = label
    data_prep.write_provenance(prov)

    # 2. feasibility census (honest executable level-4 N)
    census = feasibility_census.write_census(units)
    executable_ids = set(census.get("executable_unit_ids", []))

    # 3. generalization experiment (all LLM calls via frozen cache)
    llm = LLMCache()
    e2gen = erq3e2.run_rq3_e2_generalization(
        llm, units, repeats=mode["repeats"], executable_ids=executable_ids,
        agentic=False)
    try:
        (RESULTS / "llm_usage.json").write_text(
            json.dumps(e2gen.get("llm_stats", {}), indent=2))
        print("[run_rq3_e2] llm stats:", e2gen.get("llm_stats", {}))
    except Exception:
        pass

    # 4. analyze
    e2_analysis = arq3e2.analyze_rq3_e2(e2gen)

    # 5. merge into results.json
    arq3e2.merge_into_results(e2_analysis)

    # self-validation gate: the contamination-controlled headline delta finite
    ccd = e2_analysis.get("contamination_controlled_headline_delta_rq3", 0.0)
    if not math.isfinite(float(ccd)):
        raise RuntimeError("RQ3-E2 contamination-controlled delta is non-finite")

    # 6. run metadata + figures
    _write_run_metadata(mode, units, prov)
    try:
        import plots_rq3
        plots_rq3.main()
    except Exception as exc:  # figures must not block results
        print(f"[run_rq3_e2] WARNING: plotting failed: {exc}")

    # print a concise summary
    print("[run_rq3_e2] H6 (advantage persists under contamination):",
          e2_analysis.get("H6_advantage_persists_under_contamination"))
    for split, pm in e2_analysis.get("per_split_model_delta", {}).items():
        for m, r in pm.items():
            print(f"   {split:16s} {m:28s} delta={r.get('delta', 0.0):+.3f} "
                  f"CI={r.get('ci95')} persists={r.get('advantage_persists')}")
    print("[run_rq3_e2] RQ3-E2 pipeline complete.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RQ3-E2 contamination-controlled generalization")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic plumbing check (<=25 units, 1 repeat)")
    ap.add_argument("--pilot", action="store_true",
                    help="contract pilot size on REAL data (200 units, 3 repeats)")
    ap.add_argument("--full", action="store_true",
                    help="explicit paper-grade run (all data, 3 repeats)")
    args = ap.parse_args()

    if args.smoke:
        run(SMOKE, allow_synthetic=True, strict=False)
    elif args.pilot:
        run(PILOT, allow_synthetic=False, strict=True)
    else:
        run(FULL, allow_synthetic=False, strict=True)


if __name__ == "__main__":
    main()