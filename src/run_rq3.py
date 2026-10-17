"""
run_rq3.py -- end-to-end RQ3 pipeline (invoked by main.py or directly).

Pipeline:
  1. load researcher-provided repair units (data_prep_rq3)
  2. RQ3-E0 feasibility census (feasibility_census) -> results/census_manifest.json
  3. RQ3-E1 slice (R4) vs whole-function (R1) repair with the INDEPENDENT static
     oracle + agentic revision loop + deterministic no-LLM static baseline
     (experiment_rq3 / repair_pipeline); all LLM calls via frozen rp_llm_cache
  4. RQ3-E2 contamination split (pre/post cutoff)
  5. analysis_rq3 -> results/results.json (RQ3 key) + run_metadata + provenance
  6. plots_rq3 -> figures/

Default (no args) runs the STRICT paper-grade path over data/. --smoke runs a
small synthetic plumbing check; --pilot runs the contract pilot size on real
data; --full is the explicit paper-grade path.

Requirements (pinned): numpy==2.2.6, pandas==2.2.2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from rp_llm_cache import LLMCache

import data_prep
import data_prep_rq3
import feasibility_census
import experiment_rq3 as erq3
import analysis_rq3

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

# contract scale
SMOKE = {"max_rows": 25, "repeats": 1, "label": "smoke", "paper_grade": False}
PILOT = {"max_rows": 200, "repeats": 3, "label": "pilot", "paper_grade": False}
FULL = {"max_rows": None, "repeats": 3, "label": "paper-grade", "paper_grade": True}


def _write_run_metadata(mode: Dict[str, Any], units: List[Dict[str, Any]],
                        datasets_meta: Dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": mode["label"],
        "rows_or_items": len(units),
        "repeats": mode["repeats"],
        "datasets": [d["name"] for d in datasets_meta.get("datasets", [])
                     if d.get("source") == "real"] or
                    [d["name"] for d in datasets_meta.get("datasets", [])],
        "paper_grade": bool(mode["paper_grade"]),
        "models_under_test": erq3.MODELS_UNDER_TEST,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[run_rq3] run_metadata: mode={meta['mode']} rows={meta['rows_or_items']} "
          f"repeats={meta['repeats']} paper_grade={meta['paper_grade']}")


def run(mode: Dict[str, Any], allow_synthetic: bool, strict: bool) -> None:
    print(f"{'Smoke' if mode['label']=='smoke' else mode['label'].title()} run mode: "
          f"max_rows={mode['max_rows']}, repeats={mode['repeats']}")

    # 1. load repair units
    units, prov = data_prep_rq3.load_repair_units(
        max_rows=mode["max_rows"], allow_synthetic=allow_synthetic, strict=strict)
    if not units:
        raise RuntimeError("RQ3: no repair units loaded. Place "
                           "primevul_pairs.jsonl / secvuleval_loc.jsonl / "
                           "cleanvul_pairs.jsonl in data/ or run with --smoke.")
    print(f"[run_rq3] loaded {len(units)} repair units "
          f"({prov.get('rq3_units_with_official_fix', 0)} with official fix)")

    # write provenance (preserve mined flag)
    data_prep.write_provenance(prov)

    # 2. RQ3-E0 feasibility census
    census = feasibility_census.write_census(units)
    executable_ids = set(census.get("executable_unit_ids", []))

    # 3. RQ3-E1 repair (slice vs whole) + agentic + no-LLM baseline
    llm = LLMCache()
    rep_builder = data_prep_rq3.make_representation_builder()
    e1 = erq3.run_rq3_e1(
        llm, units, rep_builder, repeats=mode["repeats"],
        executable_ids=executable_ids, agentic=True)
    try:
        e1["llm_stats"] = llm.stats()
    except Exception:
        e1["llm_stats"] = {}
    # persist llm usage
    try:
        (RESULTS / "llm_usage.json").write_text(json.dumps(e1["llm_stats"], indent=2))
        print("[run_rq3] llm stats:", e1["llm_stats"])
    except Exception:
        pass

    # 4. RQ3-E2 contamination split (reuse the same records)
    e2 = erq3.run_rq3_e2(e1["records"])

    # 5. analysis -> results.json (RQ3 key)
    results_rq3 = analysis_rq3.analyze_rq3(
        e1, e2, census, units, executable_ids, mode_meta=mode)
    analysis_rq3.write_results_rq3(results_rq3)

    # run metadata
    _write_run_metadata(mode, units, prov)

    # 6. figures
    try:
        import plots_rq3
        plots_rq3.main()
    except Exception as exc:  # figures must not block results
        print(f"[run_rq3] WARNING: plotting failed: {exc}")

    print("[run_rq3] RQ3 pipeline complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ3 slice-local repair pipeline")
    ap.add_argument("--smoke", action="store_true",
                    help="small synthetic plumbing check (<=25 units, 1 repeat)")
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
        # default == paper-grade (STRICT real-data mode)
        run(FULL, allow_synthetic=False, strict=True)


if __name__ == "__main__":
    main()