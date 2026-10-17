"""oracle_incompleteness.py

RQ2-E1 helper: build a *dependency ground-truth oracle* over a subset of real
units and measure how incomplete the SRCVUL-style variable-level slice is
against that oracle. The measured per-unit incompleteness score is then
correlated (Spearman) with per-unit detection/localization performance to show
that *real* slice incompleteness (not only injected corruption) tracks
performance loss.

The oracle here is a lightweight, deterministic static def-use / call-graph
approximation computed from the source itself. It is a MEASUREMENT instrument,
never the proposed method's own output, so correlating slice incompleteness
against it is an independent check (not tautological).

IMPORTANT (design integrity):
  The *manually-verified* dependency oracle requested by the study
  (two-annotator def-use/call graph verification) is human work and is NOT
  simulated here. This module computes a reproducible *static approximation*
  oracle and clearly labels it as such in provenance. Where the study requires
  the manually-verified oracle for a paper-grade claim, that is surfaced as a
  DESIGN NEEDED note by the caller (experiment_rq2_e1.py / analysis_rq2.py).

Requirements (pinned to this environment):
  numpy==2.2.6, scipy==1.13.1
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

# ---- oracle data types -----------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PTR_RE = re.compile(r"(\*|->|\[)")
# tokens that are language keywords, not variables / callees
_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "break",
    "continue", "return", "goto", "sizeof", "typedef", "struct", "union",
    "enum", "const", "static", "void", "int", "char", "short", "long",
    "unsigned", "signed", "float", "double", "bool", "true", "false", "null",
    "NULL", "extern", "inline", "volatile", "register", "auto",
}


@dataclass
class OracleGraph:
    """Independent static dependency ground truth for one unit."""

    unit_id: str
    def_use_edges: set = field(default_factory=set)   # (line_a, line_b) data-dep
    call_edges: set = field(default_factory=set)      # (line, callee) call-dep
    ptr_lines: set = field(default_factory=set)       # lines with pointer/alias ops
    cross_file_calls: set = field(default_factory=set)  # callees not defined locally

    def total_edges(self) -> int:
        return (
            len(self.def_use_edges)
            + len(self.call_edges)
            + len(self.ptr_lines)
            + len(self.cross_file_calls)
        )


def build_oracle(unit_id: str, source: str) -> OracleGraph:
    """Build a deterministic static dependency oracle from C/C++ source.

    This is an APPROXIMATE, reproducible def-use/call-graph instrument used to
    MEASURE slice incompleteness. It does not define correctness of the
    proposed method; it is an external reference.
    """
    lines = source.splitlines()
    locally_defined: set = set()
    # first pass: locate locally-defined function names (name followed by '(' and
    # eventually '{' on the same or a near line) to classify cross-file calls.
    for ln in lines:
        m = re.match(r"\s*[A-Za-z_][\w\s\*\&]*?\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?", ln)
        if m:
            locally_defined.add(m.group(1))

    g = OracleGraph(unit_id=unit_id)
    # last line a given variable was assigned on -> def-use edges
    last_def: Dict[str, int] = {}
    assign_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:[-+*/|&^]?=)\s*[^=]")

    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("*") or line.startswith("/*"):
            continue

        # pointer / alias operations
        if _PTR_RE.search(line):
            g.ptr_lines.add(idx)

        # call edges
        for cm in _CALL_RE.finditer(line):
            callee = cm.group(1)
            if callee in _KEYWORDS:
                continue
            g.call_edges.add((idx, callee))
            if callee not in locally_defined:
                g.cross_file_calls.add((idx, callee))

        # def-use: a USE of a previously DEFINED variable creates a data-dep edge
        idents = [t for t in _IDENT_RE.findall(line) if t not in _KEYWORDS]
        for name in idents:
            if name in last_def and last_def[name] != idx:
                g.def_use_edges.add((last_def[name], idx))

        # record (re)definitions
        am = assign_re.match(raw)
        if am:
            last_def[am.group(1)] = idx

    return g


# ---- incompleteness scoring ------------------------------------------------

def _slice_lineset(slice_record: dict) -> set:
    """Extract the set of source line numbers covered by a slice record.

    Accepts the slice_pipeline record shape; tolerant of several key names.
    """
    for key in ("slice_lines", "lines", "vrslice_lines", "line_numbers"):
        v = slice_record.get(key)
        if isinstance(v, (list, tuple, set)) and v:
            try:
                return {int(x) for x in v}
            except (TypeError, ValueError):
                continue
    # fall back: parse integers out of a textual slice if present
    txt = slice_record.get("slice_text") or slice_record.get("text") or ""
    nums = re.findall(r"^\s*(\d+)\s*[:|]", txt, flags=re.MULTILINE)
    return {int(n) for n in nums} if nums else set()


def _slice_callset(slice_record: dict) -> set:
    cf = slice_record.get("cfuncs") or slice_record.get("cfuncs_called") or []
    out = set()
    for c in cf:
        if isinstance(c, (list, tuple)) and len(c) == 2:
            out.add((int(c[0]), str(c[1])))
        elif isinstance(c, str):
            out.add((0, c))
    return out


def measure_incompleteness(oracle: OracleGraph, slice_record: dict) -> Dict[str, float]:
    """Fraction of oracle dependency elements NOT captured by the SRCVUL slice.

    Returns per-category and overall incompleteness in [0, 1]; higher = more
    missing dependency structure. Guards degenerate (empty-oracle) cases.
    """
    covered_lines = _slice_lineset(slice_record)
    covered_calls = _slice_callset(slice_record)

    def _frac_missing(oracle_lines: Sequence[int]) -> float:
        total = len(set(oracle_lines))
        if total == 0:
            return 0.0
        missing = sum(1 for ln in set(oracle_lines) if ln not in covered_lines)
        return missing / total

    # def-use: an edge is "covered" if BOTH endpoints are in the slice
    du_total = len(oracle.def_use_edges)
    if du_total == 0:
        du_inc = 0.0
    else:
        du_missing = sum(
            1 for (a, b) in oracle.def_use_edges
            if a not in covered_lines or b not in covered_lines
        )
        du_inc = du_missing / du_total

    # pointer/alias lines
    ptr_inc = _frac_missing(list(oracle.ptr_lines))

    # call edges: covered if line is in slice OR the callee name is listed
    call_total = len(oracle.call_edges)
    if call_total == 0:
        call_inc = 0.0
    else:
        callee_names = {c for (_, c) in covered_calls}
        call_missing = sum(
            1 for (ln, callee) in oracle.call_edges
            if ln not in covered_lines and callee not in callee_names
        )
        call_inc = call_missing / call_total

    # cross-file calls (SRCVUL-missed category c3)
    cf_total = len(oracle.cross_file_calls)
    if cf_total == 0:
        cf_inc = 0.0
    else:
        callee_names = {c for (_, c) in covered_calls}
        cf_missing = sum(
            1 for (ln, callee) in oracle.cross_file_calls
            if ln not in covered_lines and callee not in callee_names
        )
        cf_inc = cf_missing / cf_total

    total = oracle.total_edges()
    if total == 0:
        overall = 0.0
    else:
        overall = (
            (du_inc * du_total)
            + (ptr_inc * len(oracle.ptr_lines))
            + (call_inc * call_total)
            + (cf_inc * cf_total)
        ) / total

    out = {
        "incompleteness_overall": float(np.clip(overall, 0.0, 1.0)),
        "incompleteness_def_use": float(np.clip(du_inc, 0.0, 1.0)),
        "incompleteness_pointer_alias": float(np.clip(ptr_inc, 0.0, 1.0)),
        "incompleteness_interprocedural": float(np.clip(call_inc, 0.0, 1.0)),
        "incompleteness_cross_file": float(np.clip(cf_inc, 0.0, 1.0)),
    }
    # all finite by construction, but assert for safety
    for k, v in out.items():
        if not np.isfinite(v):
            raise RuntimeError(f"oracle_incompleteness: non-finite {k} for {oracle.unit_id}")
    return out


def build_incompleteness_subset(
    units: Sequence[dict],
    slices_by_unit: Dict[str, dict],
    max_units: int,
    seed: int = RANDOM_SEED,
) -> List[Tuple[str, Dict[str, float]]]:
    """Measure incompleteness on a reproducible subset of real units.

    units: iterable of unit dicts with at least {'unit_id', 'func_before'/'source'}.
    slices_by_unit: unit_id -> slice record from slice_pipeline.
    Returns a list of (unit_id, incompleteness-dict) for up to max_units.
    """
    rng = np.random.default_rng(seed)
    # only vulnerable units (label 1) with source + a slice are meaningful here
    pool = []
    for u in units:
        uid = u.get("unit_id")
        src = u.get("func_before") or u.get("source") or u.get("code") or ""
        if not uid or not src or uid not in slices_by_unit:
            continue
        pool.append((uid, src))
    if not pool:
        return []
    idx = np.arange(len(pool))
    rng.shuffle(idx)
    idx = idx[: min(max_units, len(pool))]

    out: List[Tuple[str, Dict[str, float]]] = []
    for i in idx:
        uid, src = pool[int(i)]
        oracle = build_oracle(uid, src)
        inc = measure_incompleteness(oracle, slices_by_unit[uid])
        out.append((uid, inc))
    return out


def correlate_incompleteness_vs_performance(
    incompleteness: Dict[str, float],
    performance: Dict[str, float],
) -> Dict[str, float]:
    """Spearman rho between per-unit incompleteness and per-unit performance.

    Expects aligned dicts keyed by unit_id. Returns rho, p and n. For a real
    "incompleteness tracks performance loss" claim we expect NEGATIVE rho
    (more incomplete -> lower F1/localization). Degenerate (n<3 or zero
    variance) cases return a finite, explicit convention.
    """
    common = [u for u in incompleteness if u in performance]
    x = np.array([incompleteness[u] for u in common], dtype=float)
    y = np.array([performance[u] for u in common], dtype=float)
    n = int(len(common))
    if n < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n": n,
                "note": "degenerate (n<3 or zero variance); finite convention used"}
    rho, p = sps.spearmanr(x, y)
    rho = float(rho) if np.isfinite(rho) else 0.0
    p = float(p) if np.isfinite(p) else 1.0
    return {"spearman_rho": rho, "p_value": p, "n": n}


if __name__ == "__main__":
    # tiny self-check on a representative snippet (illustrative, not evidence)
    demo = (
        "int f(char *buf, int n) {\n"
        "  int i = 0;\n"
        "  char *p = buf;\n"
        "  for (i = 0; i < n; i++) {\n"
        "    p[i] = read_byte(i);\n"
        "  }\n"
        "  return check(p, n);\n"
        "}\n"
    )
    g = build_oracle("demo-1", demo)
    print(f"oracle edges: def_use={len(g.def_use_edges)} call={len(g.call_edges)} "
          f"ptr={len(g.ptr_lines)} cross_file={len(g.cross_file_calls)}")
    # a slice that keeps only the loop body (misses the cross-file call on return)
    partial = {"slice_lines": [2, 3, 4, 5], "cfuncs": [[5, "read_byte"]]}
    inc = measure_incompleteness(g, partial)
    for k, v in inc.items():
        print(f"  {k}: {v:.3f}")
    assert all(np.isfinite(v) for v in inc.values())
    # identical-copy negative control: a slice covering everything -> ~0 incompleteness
    full = {"slice_lines": list(range(1, len(demo.splitlines()) + 1)),
            "cfuncs": [[5, "read_byte"], [7, "check"]]}
    inc_full = measure_incompleteness(g, full)
    print(f"  control (full slice) overall incompleteness = {inc_full['incompleteness_overall']:.3f}")
    assert inc_full["incompleteness_overall"] <= inc["incompleteness_overall"], \
        "negative control failed: full slice must be no more incomplete than partial"
    print("oracle_incompleteness self-check OK")