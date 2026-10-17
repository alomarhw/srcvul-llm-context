"""data_prep.py - load and shape the RQ1 datasets into labeled code units.

Requirements (pinned): pandas==2.2.2, numpy==2.2.6

Each JSONL row is a vulnerable/fixed PAIR. We derive TWO units per pair:
    - func_before -> label 1 (vulnerable), with vul_lines localization ground truth
    - func_after  -> label 0 (non-vulnerable)

Strict real-data mode: if a REQUIRED file is missing we raise a clear
RuntimeError naming the dataset. Synthetic generation is ONLY produced when the
caller passes allow_synthetic=True (used by the explicit --smoke path and the
RQ1-E0 synthetic power simulation, which is [data: synthetic acceptable]).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rp_data_runtime import resolve_data_path, load_manifest
except Exception:  # pragma: no cover - fallback when helper absent
    resolve_data_path = None
    load_manifest = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

# Model training-cutoff upper bound used for the contamination control split.
POST_CUTOFF_YEAR = 2026

REQUIRED_FILES = {
    "primevul": "primevul_pairs.jsonl",
    "secvuleval": "secvuleval_loc.jsonl",
    "cleanvul": "cleanvul_pairs.jsonl",
}
OPTIONAL_FILES = {"postcutoff": "postcutoff_pairs.jsonl"}


@dataclass
class Unit:
    unit_id: str
    label: int                      # 1 vulnerable, 0 non-vulnerable
    source_dataset: str
    split: str
    cve_id: str
    cwe: str
    project: str
    commit_id: str
    code: str                       # function source
    vul_lines: list[int] = field(default_factory=list)  # 1-based, only for label==1
    vul_lines_source: str = ""
    cve_year: int = 0
    is_post_cutoff: bool = False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _resolve(fname: str, key: str) -> Path | None:
    """Resolve a dataset path via manifest first, then the literal data/ path."""
    if resolve_data_path is not None:
        for probe in (key, fname, fname.replace(".jsonl", "")):
            try:
                p = Path(resolve_data_path(probe, data_root=str(DATA), required=False))
                if p and p.exists():
                    return p
            except Exception:
                continue
    p = DATA / fname
    return p if p.exists() else None


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pair_to_units(row: dict[str, Any], src: str) -> list[Unit]:
    out: list[Unit] = []
    base_id = str(row.get("unit_id") or _sha256(json.dumps(row, sort_keys=True)[:200]))
    split = str(row.get("split", "test"))
    cve = str(row.get("cve_id", "") or "")
    cwe = str(row.get("cwe", "") or "CWE-unknown")
    project = str(row.get("project", "") or "unknown")
    commit = str(row.get("commit_id", "") or "")
    year = int(row.get("cve_year", 0) or 0)
    is_post = split == "post_cutoff" or year >= POST_CUTOFF_YEAR
    vul_lines = row.get("vul_lines") or []
    try:
        vul_lines = [int(x) for x in vul_lines]
    except Exception:
        vul_lines = []
    vls = str(row.get("vul_lines_source", "") or "")

    before = row.get("func_before")
    after = row.get("func_after")
    if before:
        out.append(Unit(
            unit_id=f"{base_id}__vul", label=1, source_dataset=src, split=split,
            cve_id=cve, cwe=cwe, project=project, commit_id=commit, code=str(before),
            vul_lines=vul_lines, vul_lines_source=vls, cve_year=year, is_post_cutoff=is_post,
        ))
    if after:
        out.append(Unit(
            unit_id=f"{base_id}__fix", label=0, source_dataset=src, split=split,
            cve_id=cve, cwe=cwe, project=project, commit_id=commit, code=str(after),
            vul_lines=[], vul_lines_source="", cve_year=year, is_post_cutoff=is_post,
        ))
    return out


# ----------------------------- synthetic fallback -----------------------------

_SYN_TEMPLATES_VUL = [
    ("CWE-119", "void copy(char *dst, const char *src) {{\n    char buf[{n}];\n"
     "    strcpy(buf, src);   // overflow: no bounds check\n    memcpy(dst, buf, {n});\n}}", [3]),
    ("CWE-416", "int use(obj_t *p) {{\n    free(p);\n    return p->val;   // use after free\n}}", [3]),
    ("CWE-476", "int deref(node_t *n) {{\n    node_t *m = n->next;\n"
     "    return m->data;   // null deref possible\n}}", [3]),
    ("CWE-190", "int add(int a, int b) {{\n    int s = a + b;   // integer overflow unchecked\n"
     "    return alloc(s);\n}}", [2]),
]
_SYN_TEMPLATES_FIX = [
    "void copy(char *dst, const char *src) {{\n    char buf[{n}];\n"
    "    strncpy(buf, src, {n}-1);\n    buf[{n}-1]=0;\n    memcpy(dst, buf, {n});\n}}",
    "int use(obj_t *p) {{\n    int v = p->val;\n    free(p);\n    return v;\n}}",
    "int deref(node_t *n) {{\n    if (!n) return -1;\n    node_t *m = n->next;\n"
    "    if (!m) return -1;\n    return m->data;\n}}",
    "int add(int a, int b) {{\n    if (a > INT_MAX - b) return -1;\n    int s = a + b;\n"
    "    return alloc(s);\n}}",
]


def _synthetic_units(n_pairs: int, src: str = "synthetic") -> list[Unit]:
    rng = random.Random(RANDOM_SEED)
    units: list[Unit] = []
    for i in range(n_pairs):
        t = i % len(_SYN_TEMPLATES_VUL)
        cwe, tmpl, vl = _SYN_TEMPLATES_VUL[t]
        nbuf = rng.choice([8, 16, 32, 64])
        before = tmpl.format(n=nbuf)
        after = _SYN_TEMPLATES_FIX[t].format(n=nbuf)
        year = rng.choice([2010, 2015, 2019, 2021, 2026])
        split = "post_cutoff" if year >= POST_CUTOFF_YEAR else "test"
        units.append(Unit(
            unit_id=f"syn{i}__vul", label=1, source_dataset=src, split=split,
            cve_id=f"CVE-{year}-{1000+i}", cwe=cwe, project=f"proj{t}", commit_id=f"c{i}",
            code=before, vul_lines=vl, vul_lines_source="secvuleval-statements",
            cve_year=year, is_post_cutoff=year >= POST_CUTOFF_YEAR))
        units.append(Unit(
            unit_id=f"syn{i}__fix", label=0, source_dataset=src, split=split,
            cve_id=f"CVE-{year}-{1000+i}", cwe=cwe, project=f"proj{t}", commit_id=f"c{i}",
            code=after, vul_lines=[], vul_lines_source="", cve_year=year,
            is_post_cutoff=year >= POST_CUTOFF_YEAR))
    return units


# ----------------------------- public API -----------------------------

def load_units(max_rows: int | None = None,
               allow_synthetic: bool = False,
               strict: bool = True) -> tuple[list[Unit], dict[str, Any]]:
    """Load all pairs -> units. Returns (units, provenance_datasets_list_meta).

    max_rows caps the number of PAIRS per dataset (smoke/pilot sizing).
    """
    prov_datasets: list[dict[str, Any]] = []
    used_synthetic = False
    all_units: list[Unit] = []

    manifest_meta: dict[str, dict[str, Any]] = {}
    if load_manifest is not None:
        try:
            man = load_manifest(data_root=str(DATA))
            for e in man.get("ok", []):
                name = (e.get("contract") or {}).get("datasetName", "")
                manifest_meta[name] = e
        except Exception:
            pass

    missing_required: list[str] = []
    for key, fname in {**REQUIRED_FILES, **OPTIONAL_FILES}.items():
        p = _resolve(fname, key)
        required = key in REQUIRED_FILES
        if p is None:
            if required:
                missing_required.append(fname)
            prov_datasets.append({
                "name": fname, "source": "missing", "path": str(DATA / fname),
                "rows": 0, "recordCount": 0, "schema": [],
            })
            continue
        rows = _read_jsonl(p)
        if max_rows is not None:
            rows = rows[:max_rows]
        units = []
        for r in rows:
            units.extend(_pair_to_units(r, key))
        all_units.extend(units)
        mm = next((v for k, v in manifest_meta.items() if key in k.lower() or fname in str(v.get("path", ""))), {})
        prov_datasets.append({
            "name": fname, "source": "real", "path": str(p),
            "rows": len(units), "recordCount": len(rows),
            "version": mm.get("version", ""), "license": mm.get("license", ""),
            "citation": mm.get("citation", ""), "checksum": mm.get("checksum", ""),
            "schema": sorted({k for r in rows[:5] for k in r.keys()}),
        })

    if missing_required:
        if strict and not allow_synthetic:
            raise RuntimeError(
                "STRICT_REAL_DATA_MODE: required researcher-provided dataset(s) "
                f"missing from data/: {missing_required}. Place the real files "
                "(primevul_pairs.jsonl, secvuleval_loc.jsonl, cleanvul_pairs.jsonl) "
                "in data/ or run with --smoke to use the synthetic smoke path."
            )
        if allow_synthetic:
            n = max_rows or 25
            syn = _synthetic_units(n)
            all_units.extend(syn)
            used_synthetic = True
            prov_datasets.append({
                "name": "synthetic_pairs", "source": "synthetic",
                "path": "(generated)", "rows": len(syn), "recordCount": n, "schema": [],
            })

    meta = {"usedSynthetic": used_synthetic, "datasets": prov_datasets}
    return all_units, meta


def stratified_sample(units: list[Unit], n_pairs: int, top_cwe: int = 10) -> list[Unit]:
    """Stratify by top-k CWE and sample ~n_pairs pairs (each pair = 2 units)."""
    rng = random.Random(RANDOM_SEED)
    # group by base id so paired structure is preserved
    by_pair: dict[str, list[Unit]] = {}
    for u in units:
        base = u.unit_id.rsplit("__", 1)[0]
        by_pair.setdefault(base, []).append(u)
    pairs = list(by_pair.values())
    # rank CWEs
    from collections import Counter
    cwe_counts = Counter(p[0].cwe for p in pairs)
    top = {c for c, _ in cwe_counts.most_common(top_cwe)}
    pairs_top = [p for p in pairs if p[0].cwe in top] or pairs
    rng.shuffle(pairs_top)
    chosen = pairs_top[:n_pairs]
    out: list[Unit] = []
    for p in chosen:
        out.extend(p)
    return out


def write_provenance(meta: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # preserve mined flag if present
    existing = DATA / "data_provenance.json"
    if existing.exists():
        try:
            prev = json.loads(existing.read_text())
            if prev.get("usedMinedData"):
                meta["usedMinedData"] = True
        except Exception:
            pass
    (RESULTS / "data_provenance.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    units, meta = load_units(max_rows=10, allow_synthetic=True, strict=False)
    print(f"loaded {len(units)} units (synthetic={meta['usedSynthetic']})")
    for d in meta["datasets"]:
        print(" ", d["name"], d["source"], d["rows"])