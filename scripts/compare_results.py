"""Compare a replayed results.json with the shipped one. Every numeric value must match to 1e-9, except timing
fields (latency, wall-clock) and file locations, which depend on the machine. Usage: python scripts/compare_results.py NEW.json OLD.json"""

import json
import math
import sys

SKIP = ("latency", "wall", "elapsed", "duration", "seconds", "timestamp", "started", "finished", "time_s")
PATH_KEYS = (".cache", ".path", ".dir", ".file")  # machine-specific locations


def walk(a, b, path, diffs):
    if any(s in path.lower() for s in SKIP) or path.lower().endswith(PATH_KEYS):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: only in {'new' if k in a else 'shipped'}")
            else:
                walk(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) or (math.isnan(a) and math.isnan(b))):
            diffs.append(f"{path}: {a} vs {b}")
    elif a != b:
        diffs.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


def main() -> int:
    new, old = (json.load(open(p)) for p in sys.argv[1:3])
    diffs = []
    walk(new, old, "", diffs)
    for d in diffs[:50]:
        print(d)
    print(f"{len(diffs)} difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
