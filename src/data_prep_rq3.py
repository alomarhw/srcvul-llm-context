"""
data_prep_rq3.py -- shape researcher-provided data into RQ3 repair units.

RQ3 is a DISTINCT experiment from RQ1/RQ2: the unit of analysis is a vulnerable
function that must be REPAIRED (not merely classified). Each JSONL pair gives us
exactly ONE RQ3 unit (the vulnerable func_before + its official fix func_after +
localization vul_lines). func_after is the OFFICIAL FIX and is used only by the
INDEPENDENT static oracle for adjudication; it is withheld from the repairing
model and from any retrieval context at test time (except in a labeled oracle
condition, which this study does not auto-run).

We reuse data_prep.load_units (the shared loader) and convert its Unit objects
into the plain dicts experiment_rq3 / feasibility_census / repair_pipeline read.

Requirements (pinned): pandas==2.2.2, numpy==2.2.6
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from data_prep import load_units, Unit, POST_CUTOFF_YEAR

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))


def _unit_to_repair_dict(vul_unit: Unit, fix_unit: Unit | None) -> Dict[str, Any]:
    """Build one RQ3 repair unit from a vulnerable unit (+ its sibling fix)."""
    func_after = fix_unit.code if fix_unit is not None else ""
    return {
        "unit_id": vul_unit.unit_id.rsplit("__", 1)[0],  # pair-level id
        "cve_id": vul_unit.cve_id,
        "cwe": vul_unit.cwe,
        "project": vul_unit.project,
        "commit_id": vul_unit.commit_id,
        "split": vul_unit.split,
        "func_before": vul_unit.code,        # VULNERABLE (to repair)
        "func_after": func_after,            # OFFICIAL FIX (oracle only; withheld from model)
        "vul_lines": list(vul_unit.vul_lines),
        "vul_lines_source": vul_unit.vul_lines_source,
        "cve_year": int(vul_unit.cve_year or 0),
        "is_post_cutoff": bool(vul_unit.is_post_cutoff),
        # `extra` carries any build/PoC/test-harness metadata; the uploaded
        # schema rarely has it, so the census honestly reports executable N.
        "extra": {},
    }


def load_repair_units(max_rows: int | None = None,
                      allow_synthetic: bool = False,
                      strict: bool = True) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (repair_units, provenance_meta).

    repair_units: list of dicts (one per vulnerable function to repair).
    max_rows caps pairs per dataset (smoke/pilot sizing).
    """
    units, meta = load_units(max_rows=max_rows, allow_synthetic=allow_synthetic,
                             strict=strict)

    # group units by pair base id so the vulnerable unit keeps its official fix
    by_pair: Dict[str, Dict[str, Unit]] = {}
    for u in units:
        base = u.unit_id.rsplit("__", 1)[0]
        slot = "vul" if u.label == 1 else "fix"
        by_pair.setdefault(base, {})[slot] = u

    repair_units: List[Dict[str, Any]] = []
    for base, slots in by_pair.items():
        vul = slots.get("vul")
        if vul is None:
            # no vulnerable function -> nothing to repair; skip (honest)
            continue
        fix = slots.get("fix")
        repair_units.append(_unit_to_repair_dict(vul, fix))

    meta = dict(meta)
    meta["rq3_repair_units"] = len(repair_units)
    meta["rq3_units_with_official_fix"] = sum(
        1 for r in repair_units if r["func_after"]
    )
    return repair_units, meta


def split_contamination(units: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Chronological contamination split: pre- vs post-cutoff."""
    pre = [u for u in units if not u.get("is_post_cutoff")]
    post = [u for u in units if u.get("is_post_cutoff")]
    return {"pre_cutoff": pre, "post_cutoff": post}


def make_representation_builder():
    """
    Return a callable rep_builder(unit_dict, representation) -> str.

    Reuses the shared slice_pipeline to build the two RQ3 representations:
      R1  whole-function context (no slicing)
      R4  dependency-enriched variable-level slice (slice + SRCVUL metadata)

    The OFFICIAL FIX (func_after) is NEVER placed in the representation text, so
    the repairing model never sees the ground-truth patch (no leakage).
    """
    import re
    from slice_pipeline import compute_slice, build_representations

    def _criterion_vars(code: str, vul_lines: List[int]) -> List[str]:
        """Blind slicing criterion (no ground truth); vul_lines is ignored on purpose."""
        from slice_pipeline import blind_criterion_vars
        return blind_criterion_vars(code)

    def rep_builder(unit: Dict[str, Any], representation: str) -> str:
        code = unit.get("func_before", "")
        meta = compute_slice(code, criterion_vars=_criterion_vars(code, []))
        reps = build_representations(code, meta, retrieved_examples=None, ctx_lines=2)
        if representation == "R1":
            return reps["R1"]
        if representation == "R4":
            return reps["R4"]
        # default to whole function
        return reps.get(representation, reps["R1"])

    return rep_builder


if __name__ == "__main__":
    units, meta = load_repair_units(max_rows=5, allow_synthetic=True, strict=False)
    print(f"repair units: {len(units)} (with fix: {meta['rq3_units_with_official_fix']})")
    rb = make_representation_builder()
    if units:
        r1 = rb(units[0], "R1")
        r4 = rb(units[0], "R4")
        assert units[0]["func_after"] not in r1, "leakage: official fix in R1"
        assert units[0]["func_after"] not in r4 or not units[0]["func_after"], \
            "leakage: official fix in R4"
        print("R1 len:", len(r1), "R4 len:", len(r4))
    splits = split_contamination(units)
    print("pre/post:", len(splits["pre_cutoff"]), len(splits["post_cutoff"]))