"""Decompress the .xz files (frozen response cache, large data and record files) and check every file listed in
CHECKSUMS.json. Usage: python scripts/unpack.py"""

import hashlib
import json
import lzma
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for xz in sorted(ROOT.rglob("*.xz")):
        dst = xz.with_suffix("")
        if not dst.exists():
            with lzma.open(xz) as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g)
            print(f"unpacked {dst.relative_to(ROOT)}")
    bad = 0
    for rel, want in json.loads((ROOT / "CHECKSUMS.json").read_text()).items():
        p = ROOT / rel
        if not p.exists():
            note = " (run: python data/fetch_cleanvul.py)" if rel.endswith("cleanvul_pairs.jsonl") else ""
            print(f"missing  {rel}{note}")
            continue
        ok = hashlib.sha256(p.read_bytes()).hexdigest() == want
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} {rel}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
