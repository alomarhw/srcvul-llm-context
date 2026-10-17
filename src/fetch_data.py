"""fetch_data.py - best-effort data acquisition for RQ1 experiments.

Requirements (pinned to this environment):
    pandas==2.2.2, numpy==2.2.6, requests (stdlib urllib fallback)

This script NEVER aborts the pipeline. It tries to download the datasets the
engineer contract lists and writes data/fetch_manifest.json describing what was
actually written. main.py / data_prep.py enforce the strict real-data policy.

Datasets (engineer contract):
  - PrimeVul paired          -> data/primevul_pairs.jsonl
  - SecVulEval localization  -> data/secvuleval_loc.jsonl
  - CleanVul label-quality   -> data/cleanvul_pairs.jsonl
  - Post-cutoff CVE pairs    -> data/postcutoff_pairs.jsonl  (mined; see DATA_CARD)
  - VULSLICER database       -> data/vulslicer_database/ (zenodo 6059924)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Expected researcher-provided files (already mounted in data/ in normal runs).
EXPECTED = {
    "PrimeVul paired (data/primevul_pairs.jsonl)": "primevul_pairs.jsonl",
    "SecVulEval statement-level localization (data/secvuleval_loc.jsonl)": "secvuleval_loc.jsonl",
    "CleanVul score-4 label-quality check (data/cleanvul_pairs.jsonl)": "cleanvul_pairs.jsonl",
    "Post-cutoff CVE pairs (data/postcutoff_pairs.jsonl)": "postcutoff_pairs.jsonl",
}

HF_IDS = {
    "primevul_pairs.jsonl": "colin/PrimeVul",
    "secvuleval_loc.jsonl": "arag0rn/SecVulEval",
    "cleanvul_pairs.jsonl": "yikun-li/CleanVul",
}


def _count_jsonl(p: Path) -> int:
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def _try_zenodo_vulslicer() -> None:
    target = DATA / "vulslicer_database"
    if target.exists() and any(target.iterdir()):
        print(f"[fetch] VULSLICER already present at {target}")
        return
    try:
        import urllib.request

        api = "https://zenodo.org/api/records/6059924"
        print(f"[fetch] GET {api}")
        with urllib.request.urlopen(api, timeout=30) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        files = meta.get("files", [])
        if not files:
            print("[fetch] WARNING: zenodo record 6059924 has no files listed")
            return
        target.mkdir(parents=True, exist_ok=True)
        # download the smallest tabular/archive file to keep it light
        f = files[0]
        url = f.get("links", {}).get("self")
        name = f.get("key", "vulslicer_file")
        if url:
            out = target / name
            print(f"[fetch] downloading {name} ...")
            urllib.request.urlretrieve(url, out)
            print(f"[fetch] saved {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] WARNING: VULSLICER fetch failed: {exc}")


def main() -> int:
    ok, failed = [], []

    for human, fname in EXPECTED.items():
        p = DATA / fname
        if p.exists() and p.stat().st_size > 0:
            n = _count_jsonl(p)
            print(f"[fetch] present: {fname} ({n} rows)")
            ok.append({"path": str(p), "contract": {"datasetName": human}, "recordCount": n})
            continue
        # try huggingface if available
        hf_id = HF_IDS.get(fname)
        fetched = False
        if hf_id:
            try:
                from datasets import load_dataset  # type: ignore

                print(f"[fetch] load_dataset('{hf_id}') ...")
                ds = load_dataset(hf_id)
                split = next(iter(ds.keys()))
                rows = ds[split]
                with p.open("w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                n = _count_jsonl(p)
                print(f"[fetch] wrote {fname} ({n} rows) from HF")
                ok.append({"path": str(p), "contract": {"datasetName": human}, "recordCount": n})
                fetched = True
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] WARNING: HF fetch for {fname} failed: {exc}")
        if not fetched:
            print(f"[fetch] MISSING: {fname} (main.py will enforce policy)")
            failed.append({"path": str(p), "contract": {"datasetName": human}})

    _try_zenodo_vulslicer()

    manifest = {"ok": ok, "failed": failed}
    (DATA / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fetch] wrote manifest with {len(ok)} ok, {len(failed)} failed")
    return 0


if __name__ == "__main__":
    # Never abort the pipeline.
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] WARNING: unexpected error (continuing): {exc}")
        sys.exit(0)