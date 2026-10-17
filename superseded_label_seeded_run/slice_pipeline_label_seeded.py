"""slice_pipeline.py - SRCVUL variable-level slicing + representation builder.

Requirements (pinned): numpy==2.2.6

This extends the researcher's SRCVUL pipeline (srcML + srcSlice, variable-level
decompositional slicing). srcML/srcSlice are native tools and are NOT pip
packages in this environment (preflight: srcml/srcslice 404 on PyPI). If the
`srcml` and `srcslice` binaries are on PATH we invoke them (srcml --position, then
srcslice) and build the SRCVUL variable-level slice from srcSlice's profiles: def/use
lines of the seed variables closed over their dependent variables (dvars) and pointer
aliases. Functions srcML/srcSlice cannot process fall back to a lightweight C-token
slicer with the same fields; every slice records which backend produced it and
SLICE_STATS counts both, so the fallback rate is reported.

Representations (the ONLY thing varied in RQ1):
  R1 whole function
  R2 raw variable-level slice (vrslice)
  R3 slice + surrounding context (+-N lines)
  R4 slice + explicit dependency metadata header (vrvars/def-use/dvars/ptrs/cfuncs)
  R5 slice + retrieved vulnerability/patch examples (retrieval.py supplies examples)
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

SRCML_BIN = shutil.which("srcml")
SRCSLICE_BIN = shutil.which("srcslice")
BACKEND = "srcml+srcslice" if (SRCML_BIN and SRCSLICE_BIN) else "lightweight-static-fallback"

# C identifiers; keep simple, deterministic.
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PTR_OP = re.compile(r"(\*|->|&)")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*=[^=]")
_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "return", "break",
    "continue", "sizeof", "struct", "union", "enum", "typedef", "const",
    "static", "void", "int", "char", "long", "short", "unsigned", "signed",
    "float", "double", "default", "goto", "NULL",
}


@dataclass
class SliceMeta:
    vrvars: list[str] = field(default_factory=list)       # vulnerability-related vars
    defs: dict[str, list[int]] = field(default_factory=dict)   # var -> def lines
    uses: dict[str, list[int]] = field(default_factory=dict)   # var -> use lines
    dvars: list[str] = field(default_factory=list)       # dependent vars
    ptrs: list[str] = field(default_factory=list)        # pointer-involved vars
    cfuncs: list[str] = field(default_factory=list)      # called functions in slice
    slice_lines: list[int] = field(default_factory=list)  # 1-based lines in slice
    backend: str = BACKEND


def _tokens(line: str) -> list[str]:
    return [t for t in _ID.findall(line) if t not in _KEYWORDS]


def _lightweight_slice(code: str, criterion_vars: list[str] | None = None) -> SliceMeta:
    """Fallback C-token slicer: variable-level slice (forward+backward) around vrvars.

    criterion_vars: when known (e.g. from the vulnerable statement) the slice is
    seeded on those variables; otherwise the most-referenced variables seed it.
    """
    lines = code.splitlines()
    n = len(lines)
    defs: dict[str, list[int]] = {}
    uses: dict[str, list[int]] = {}
    ptr_vars: set[str] = set()
    called: set[str] = set()
    for i, raw in enumerate(lines, start=1):
        line = raw
        for c in _CALL.findall(line):
            if c not in _KEYWORDS:
                called.add(c)
        m = _ASSIGN.match(line)
        if m:
            defs.setdefault(m.group(1), []).append(i)
        for tok in _tokens(line):
            uses.setdefault(tok, []).append(i)
        if _PTR_OP.search(line):
            for tok in _tokens(line):
                ptr_vars.add(tok)

    # seed variables
    if criterion_vars:
        seeds = [v for v in criterion_vars if v in uses or v in defs]
    else:
        seeds = []
    if not seeds:
        # pick most-referenced non-call identifiers
        freq = {v: len(uses.get(v, [])) for v in uses if v not in called}
        seeds = [v for v, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:3]]

    # dependency closure: variables appearing on the same line as a seed var
    dependent: set[str] = set(seeds)
    changed = True
    rounds = 0
    while changed and rounds < 5:
        changed = False
        rounds += 1
        for i, raw in enumerate(lines, start=1):
            toks = set(_tokens(raw))
            if toks & dependent:
                for t in toks:
                    if t not in dependent and t not in called:
                        dependent.add(t)
                        changed = True

    slice_line_set: set[int] = set()
    for v in dependent:
        slice_line_set.update(defs.get(v, []))
        slice_line_set.update(uses.get(v, []))
    slice_lines = sorted(slice_line_set) or list(range(1, n + 1))

    vrvars = seeds
    dvars = sorted(dependent - set(seeds))
    ptrs = sorted(ptr_vars & dependent)
    cfuncs = sorted(called)
    meta = SliceMeta(
        vrvars=vrvars,
        defs={v: defs.get(v, []) for v in dependent if v in defs},
        uses={v: uses.get(v, []) for v in dependent if v in uses},
        dvars=dvars, ptrs=ptrs, cfuncs=cfuncs, slice_lines=slice_lines, backend="lightweight-static-fallback",
    )
    return meta


# ----------------------------- srcSlice backend -----------------------------
import hashlib
import os
import tempfile
from functools import lru_cache

SLICE_STATS: dict[str, int] = {"srcslice": 0, "fallback": 0}
_COVERAGE: list[float] = []  # share of function lines kept by each srcSlice slice
_PROFILE = re.compile(r"^(?P<file>[^,]*),(?P<func>[^,]*),(?P<var>[^,]*),def\{(?P<defs>[^}]*)\},use\{(?P<uses>[^}]*)\},"
                      r"dvars\{(?P<dvars>[^}]*)\},pointers\{(?P<ptrs>[^}]*)\},cfuncs\{(?P<cfuncs>.*)\}\s*$")


def _ints(text: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\d+", text) if int(x) > 0})


def _names(text: str) -> list[str]:
    return [x for x in (t.strip() for t in text.split(",")) if x]


@lru_cache(maxsize=20000)
def _srcslice_profiles(code: str) -> tuple:
    """srcSlice profiles for one function: ((var, defs, uses, dvars, ptrs, cfuncs), ...); () on failure."""
    if not (SRCML_BIN and SRCSLICE_BIN):
        return ()
    for ext in (".c", ".cpp"):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "unit" + ext)
            xml = os.path.join(td, "unit.xml")
            with open(src, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(code)
            try:
                subprocess.run([SRCML_BIN, "--position", src, "-o", xml], check=True,
                               capture_output=True, timeout=30)
                out = subprocess.run([SRCSLICE_BIN, xml], capture_output=True, timeout=30, text=True,
                                     errors="replace").stdout
            except (subprocess.SubprocessError, OSError):
                continue
        rows = []
        for line in out.splitlines():
            m = _PROFILE.match(line.strip())
            if not m:
                continue
            cf = re.sub(r"\{[^}]*\}", "", m.group("cfuncs"))
            rows.append((m.group("var"), tuple(_ints(m.group("defs"))), tuple(_ints(m.group("uses"))),
                         tuple(_names(m.group("dvars"))), tuple(_names(m.group("ptrs"))), tuple(_names(cf))))
        if any(r[1] or r[2] for r in rows):
            return tuple(rows)
    return ()


def compute_slice(code: str, criterion_vars: list[str] | None = None) -> SliceMeta:
    """SRCVUL variable-level slice from srcSlice profiles (fallback: _lightweight_slice).

    Seeds are criterion_vars when known (variables on the vulnerable statements), else the three
    most-used variables. The slice is the def/use lines of the seeds closed forward and backward
    over srcSlice's dependent variables (dvars) and over pointer aliases.
    """
    profiles = _srcslice_profiles(code)
    if not profiles:
        SLICE_STATS["fallback"] += 1
        return _lightweight_slice(code, criterion_vars)
    prof = {r[0]: r for r in profiles}
    seeds = [v for v in (criterion_vars or []) if v in prof]
    if not seeds:
        seeds = [v for v, _ in sorted(prof.items(), key=lambda kv: (-len(kv[1][2]), kv[0]))[:3]]
    # Forward (srcSlice dvars: variables v's value flows into), backward (variables whose dvars
    # include v) and pointer aliases, so the slice is SRCVUL's forward+backward variable slice.
    flows_from: dict[str, set[str]] = {}
    for v, r in prof.items():
        for w in r[3]:
            flows_from.setdefault(w, set()).add(v)
    closure, queue = set(seeds), list(seeds)
    while queue:
        v = queue.pop()
        for w in list(prof[v][3]) + list(prof[v][4]) + sorted(flows_from.get(v, ())):
            if w in prof and w not in closure:
                closure.add(w)
                queue.append(w)
    n = len(code.splitlines())
    lines = sorted({ln for v in closure for ln in prof[v][1] + prof[v][2] if 1 <= ln <= n})
    if not lines:
        SLICE_STATS["fallback"] += 1
        return _lightweight_slice(code, criterion_vars)
    SLICE_STATS["srcslice"] += 1
    _COVERAGE.append(len(lines) / n if n else 1.0)
    return SliceMeta(
        vrvars=list(seeds),
        defs={v: list(prof[v][1]) for v in closure if prof[v][1]},
        uses={v: list(prof[v][2]) for v in closure if prof[v][2]},
        dvars=sorted(closure - set(seeds)),
        ptrs=sorted({w for v in closure for w in prof[v][4]}),
        cfuncs=sorted({f for v in closure for f in prof[v][5]}),
        slice_lines=lines, backend="srcml+srcslice",
    )


def slice_backend_report() -> dict[str, Any]:
    total = SLICE_STATS["srcslice"] + SLICE_STATS["fallback"]
    cov = sorted(_COVERAGE)
    q = lambda f: cov[min(len(cov) - 1, int(f * len(cov)))] if cov else None  # noqa: E731
    return {"srcslice": SLICE_STATS["srcslice"], "fallback": SLICE_STATS["fallback"],
            "fallback_rate": (SLICE_STATS["fallback"] / total) if total else 0.0,
            "srcml_available": bool(SRCML_BIN and SRCSLICE_BIN),
            "coverage_median": q(0.5), "coverage_p25": q(0.25), "coverage_p75": q(0.75)}


def _slice_text(code: str, slice_lines: list[int]) -> str:
    lines = code.splitlines()
    keep = sorted(set(slice_lines))
    return "\n".join(f"{i}: {lines[i-1]}" for i in keep if 1 <= i <= len(lines))


def _context_text(code: str, slice_lines: list[int], ctx: int = 2) -> str:
    lines = code.splitlines()
    keep: set[int] = set()
    for i in slice_lines:
        for j in range(i - ctx, i + ctx + 1):
            if 1 <= j <= len(lines):
                keep.add(j)
    return "\n".join(f"{i}: {lines[i-1]}" for i in sorted(keep))


def _metadata_header(meta: SliceMeta) -> str:
    def _fmt(d: dict[str, list[int]]) -> str:
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


def build_representations(code: str, meta: SliceMeta,
                          retrieved_examples: list[str] | None = None,
                          ctx_lines: int = 2) -> dict[str, str]:
    """Return R1..R5 strings. Statement-level granularity handled by analysis
    via a coarser slice (see statement_level_slice)."""
    numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(code.splitlines(), 1))
    r2 = _slice_text(code, meta.slice_lines)
    reps = {
        "R1": numbered,
        "R2": r2,
        "R3": _context_text(code, meta.slice_lines, ctx=ctx_lines),
        "R4": _metadata_header(meta) + "\n" + r2,
    }
    ex = retrieved_examples or []
    ex_block = "\n\n".join(f"/* retrieved example {k+1} */\n{e}" for k, e in enumerate(ex))
    reps["R5"] = (r2 + ("\n\n" + ex_block if ex_block else ""))
    return reps


def statement_level_slice(code: str) -> list[int]:
    """Changed-granularity ablation: whole statements (lines) containing any
    def/use, i.e. coarser than variable-level."""
    lines = code.splitlines()
    keep = [i for i, ln in enumerate(lines, 1) if _ID.search(ln) and ";" in ln]
    return keep or list(range(1, len(lines) + 1))


if __name__ == "__main__":
    print(f"slice backend: {BACKEND}")
    sample = ("void copy(char *dst, const char *src) {\n"
              "    char buf[8];\n    strcpy(buf, src);\n    memcpy(dst, buf, 8);\n}")
    m = compute_slice(sample, criterion_vars=["buf", "src"])
    print("vrvars:", m.vrvars, "ptrs:", m.ptrs, "cfuncs:", m.cfuncs)
    reps = build_representations(sample, m, retrieved_examples=["/* patch */"])
    for k in ("R1", "R2", "R3", "R4", "R5"):
        print(f"--- {k} ---\n{reps[k][:120]}")