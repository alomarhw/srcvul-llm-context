"""Add-on datasets for project DF89E (seed 42), built next to the files from build_df89e_data.py:

  primevul_official_test_pairs.jsonl  ALL 435 pairs of PrimeVul's paired test split, unfiltered, so results are
                                      directly comparable with work that reports pair-wise correct predictions
                                      on that split (e.g. VulTrial, ReVD, Aegis). Same schema as primevul_pairs.
  primevul_train_lora.jsonl           PrimeVul paired TRAIN pairs for the fine-tuned (LoRA) baseline, disjoint by
                                      CVE id, commit and normalised function text from EVERY evaluation file
                                      (including the 325 train pairs evaluated in primevul_pairs.jsonl).

Usage: python build_df89e_addons.py RAW_DIR OUT_DIR   (RAW_DIR holds primevul_{test,train}_paired.jsonl)
"""
import difflib, json, re, sys
from collections import Counter
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
EVAL_FILES = ["primevul_pairs.jsonl", "secvuleval_loc.jsonl", "cleanvul_pairs.jsonl", "postcutoff_pairs.jsonl"]


def norm(s):
    return re.sub(r"\s+", "", s or "")


def fix_lines(before, after):
    sm = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines(), autojunk=False)
    return [i for t, i1, i2, _, _ in sm.get_opcodes() if t in ("replace", "delete") for i in range(i1 + 1, i2 + 1)]


def year(cve):
    m = re.match(r"CVE-(\d{4})-", cve or "")
    return int(m.group(1)) if m else None


def pairs(split, prefix, same_commit=True):
    """Pairs from a PrimeVul paired file. The official test split keeps 2 pairs whose patched version comes from a
    different commit than the vulnerable one; pass same_commit=False to keep them (flagged) for comparability."""
    rows = [json.loads(l) for l in open(RAW / f"primevul_{split}_paired.jsonl")]
    out = []
    for v, p in zip(rows[0::2], rows[1::2]):  # vulnerable function, then its patched version
        if not (v["target"] == 1 and p["target"] == 0):
            continue
        if same_commit and v["commit_id"] != p["commit_id"]:
            continue
        cwe = v.get("cwe")
        out.append({"unit_id": f"{prefix}-{v['idx']}", "dataset": f"PrimeVul-paired-{split}", "split": split,
                    "cve_id": v.get("cve") or "", "cwe": (cwe or ["?"])[0] if isinstance(cwe, list) else (cwe or "?"),
                    "project": v["project"], "commit_id": v["commit_id"], "func_before": v["func"],
                    "func_after": p["func"], "vul_lines": fix_lines(v["func"], p["func"]),
                    "vul_lines_source": "fix-diff", "cve_year": year(v.get("cve")),
                    "extra": {"cve_desc": v.get("cve_desc"), "file_name": v.get("file_name"),
                              "commit_mismatch": v["commit_id"] != p["commit_id"]}})
    return out


def write(name, rows):
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows), "pairs | projects", len({r["project"] for r in rows}),
          "| top CWEs", Counter(r["cwe"] for r in rows).most_common(5))


test = pairs("test", "PVT", same_commit=False)
assert len(test) == 435, f"expected the 435 official test pairs, got {len(test)}"
write("primevul_official_test_pairs.jsonl", test)

# Everything any experiment evaluates on: the LoRA training data must share nothing with it.
held = [json.loads(l) for f in EVAL_FILES for l in open(OUT / f)] + test
held_cves = {r["cve_id"] for r in held if r.get("cve_id")}
held_commits = {r["commit_id"] for r in held if r.get("commit_id")}
held_funcs = {norm(r[k]) for r in held for k in ("func_before", "func_after") if r.get(k)}
train = [r for r in pairs("train", "PVTR")
         if not (r["cve_id"] and r["cve_id"] in held_cves) and r["commit_id"] not in held_commits
         and norm(r["func_before"]) not in held_funcs and norm(r["func_after"]) not in held_funcs]
assert not ({r["cve_id"] for r in train if r["cve_id"]} & held_cves)
assert not ({norm(r["func_before"]) for r in train} & held_funcs)
write("primevul_train_lora.jsonl", train)
