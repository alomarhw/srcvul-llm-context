"""
feasibility_census.py -- RQ3-E0 executable-feasibility census.

Honest buildability classification of code units BEFORE any repair/elimination
claim is made. We never pretend a unit is dynamically validatable unless the
researcher-provided metadata says so. The census produces:
  - level-0: parse-only (we always have the function source)
  - level-1: has a fix commit / project pointer (static re-detection possible)
  - level-2: has build metadata in `extra` (buildable, static+build)
  - level-3: has a test harness pointer in `extra`
  - level-4a: dynamic PoC available (extra.poc / extra.sanitizer)
  - level-4b: executable test-harness that exercises the vulnerable path

For the uploaded PrimeVul/SecVulEval/CleanVul schema, build/PoC metadata is
generally ABSENT, so the honest achievable executable level is 0/1. We record
that honestly rather than inflating it. RQ3 elimination therefore defaults to
the INDEPENDENT STATIC oracle (see repair_pipeline.py); dynamic validation is
reported separately and only when real build/PoC metadata exists.

Requirements: pandas==2.2.2, numpy==2.2.6 (CPU only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

RESULTS = Path("results")


def _has(extra: Dict[str, Any], *keys: str) -> bool:
    if not isinstance(extra, dict):
        return False
    for k in keys:
        v = extra.get(k)
        if v not in (None, "", [], {}):
            return True
    return False


def classify_unit(unit: Dict[str, Any]) -> str:
    """Return the highest honestly-supported feasibility level for one unit."""
    extra = unit.get("extra") or {}
    if _has(extra, "poc", "sanitizer", "asan_report", "crash_input"):
        return "4a"
    if _has(extra, "test_harness", "exec_test", "poc_test"):
        return "4b"
    if _has(extra, "test_cmd", "tests", "test_file"):
        return "3"
    if _has(extra, "build_cmd", "build_system", "makefile", "cmake"):
        return "2"
    if unit.get("commit_id") or unit.get("project"):
        return "1"
    return "0"


def run_census(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify every unit and assemble executable vs static-only manifests."""
    counts: Dict[str, int] = {lvl: 0 for lvl in ["0", "1", "2", "3", "4a", "4b"]}
    executable: List[str] = []
    static_only: List[str] = []
    for u in units:
        lvl = classify_unit(u)
        counts[lvl] += 1
        if lvl in ("4a", "4b"):
            executable.append(u.get("unit_id", ""))
        else:
            static_only.append(u.get("unit_id", ""))

    manifest = {
        "total_units": len(units),
        "level_counts": counts,
        "executable_level4_N": len(executable),
        "static_only_N": len(static_only),
        "executable_unit_ids": executable[:5000],
        "notes": (
            "Honest achievable executable (level-4a/4b) N is derived ONLY from "
            "real build/PoC metadata in each unit's `extra` field. The uploaded "
            "PrimeVul/SecVulEval/CleanVul schema carries no build/PoC metadata, "
            "so executable_level4_N is expected to be 0. RQ3 elimination is then "
            "computed by the INDEPENDENT STATIC oracle; dynamic validation is "
            "reported separately only when executable_level4_N > 0."
        ),
    }
    if len(executable) == 0:
        manifest["DESIGN_NEEDED"] = (
            "DATA NEEDED: executable CVE benchmark with build/PoC/test-harness "
            "metadata (ARVO/OSS-Fuzz/Magma-derived) to populate level-4a/4b for "
            "dynamic vulnerability-elimination validation. Static oracle used."
        )
    return manifest


def write_census(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = run_census(units)
    out = RESULTS / "census_manifest.json"
    with out.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[feasibility_census] wrote {out}")
    print(f"[feasibility_census] level counts: {manifest['level_counts']}")
    print(f"[feasibility_census] executable level-4 N = {manifest['executable_level4_N']}")
    return manifest


if __name__ == "__main__":
    # Standalone smoke: classify a few synthetic units.
    demo = [
        {"unit_id": "u1", "commit_id": "abc", "project": "p", "extra": {}},
        {"unit_id": "u2", "extra": {"build_cmd": "make"}},
        {"unit_id": "u3", "extra": {"poc": "crash.bin"}},
        {"unit_id": "u4", "extra": {}},
    ]
    write_census(demo)