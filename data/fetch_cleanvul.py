"""Rebuild data/cleanvul_pairs.jsonl from the public CleanVul release.

CleanVul (Li et al., 2024; huggingface.co/datasets/yikun-li/CleanVul) lists no license, so this package does not
redistribute its source code. data/cleanvul_index.jsonl holds every field of the 300 pairs we used except the two
functions, plus a SHA-256 of each. This script downloads the score-4 file, finds each function by its hash, and
writes the pairs in the original order and format. The result is checked against CHECKSUMS.json.

Usage: python data/fetch_cleanvul.py
"""

import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/yikun-li/CleanVul/resolve/main/vulnerability_score_4.csv"
KEYS = ["unit_id", "dataset", "split", "cve_id", "cwe", "project", "commit_id", "func_before", "func_after",
        "vul_lines", "vul_lines_source", "cve_year", "extra"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> int:
    raw = ROOT / "data" / "raw" / "cleanvul_vulnerability_score_4.csv"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    csv.field_size_limit(sys.maxsize)
    by_hash = {}
    with open(raw, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("func_before", "func_after"):
                if row.get(col):
                    by_hash[sha(row[col])] = row[col]
    out, missing = [], 0
    for line in (ROOT / "data" / "cleanvul_index.jsonl").read_text().splitlines():
        r = json.loads(line)
        before, after = by_hash.get(r.pop("sha256_func_before")), by_hash.get(r.pop("sha256_func_after"))
        if before is None or after is None:
            missing += 1
            continue
        r["func_before"], r["func_after"] = before, after
        out.append({k: r[k] for k in KEYS})
    if missing:
        print(f"ERROR: {missing} of 300 functions not found in the downloaded file")
        return 1
    dst = ROOT / "data" / "cleanvul_pairs.jsonl"
    dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    want = json.loads((ROOT / "CHECKSUMS.json").read_text())["data/cleanvul_pairs.jsonl"]
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"wrote {dst.relative_to(ROOT)} ({len(out)} pairs); checksum {'OK' if got == want else 'MISMATCH'}")
    return 0 if got == want else 1


if __name__ == "__main__":
    sys.exit(main())
