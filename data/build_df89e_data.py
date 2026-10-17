"""Datasets for project DF89E (seed 42). Writes one JSONL per source with a common schema:

  unit_id, dataset, split, cve_id, cwe, project, commit_id, func_before (vulnerable), func_after (fixed),
  vul_lines (1-based lines of func_before), vul_lines_source, cve_year, extra

Sources (all public, MIT or open licenses):
  primevul_pairs.jsonl   PrimeVul paired test+valid splits (Ding et al., ICSE 2025; colin/PrimeVul): the
                         main detection/repair benchmark. Chronological split; NVD-verified labels.
                         vul_lines = lines removed/modified by the fix (fix-diff).
  secvuleval_loc.jsonl   SecVulEval (arag0rn/SecVulEval, 2025): vulnerable functions with their fixed version
                         and curated STATEMENT-level labels -> localization ground truth; also the dataset's
                         own context fields (globals, arguments, external functions) in `extra.context`.
  cleanvul_pairs.jsonl   CleanVul score-4 C/C++ pairs (yikun-li/CleanVul): LLM-filtered high-confidence labels,
                         a label-quality robustness check.
Filters everywhere: 5-200 lines, a real statement change, one function per CVE per source.
Usage: python build_df89e_data.py RAW_DIR OUT_DIR
"""
import difflib, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
rng = random.Random(42)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def stmts(code):
    code = re.sub(r"/\*.*?\*/", " ", code or "", flags=re.S)
    return Counter(x for x in (norm(re.sub(r"//.*", "", l)) for l in code.splitlines()) if x and x not in {"{", "}", "};"})


def fix_lines(before, after):
    sm = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines(), autojunk=False)
    return [i for t, i1, i2, _, _ in sm.get_opcodes() if t in ("replace", "delete") for i in range(i1 + 1, i2 + 1)]


def ok(before, after):
    n = before.count("\n") + 1
    return 5 <= n <= 200 and stmts(before) != stmts(after)


def year(cve):
    m = re.match(r"CVE-(\d{4})-", cve or "")
    return int(m.group(1)) if m else None


def write(name, rows):
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows), "pairs |", Counter(r["split"] for r in rows), "| projects", len({r["project"] for r in rows}),
          "| top CWEs", Counter(r["cwe"] for r in rows).most_common(5))


# ---- PrimeVul paired (test + valid) ----
pv = []
for split in ("test", "valid", "train"):
    rows = [json.loads(l) for l in open(RAW / f"primevul_{split}_paired.jsonl")]
    # paired files list the vulnerable function followed by its patched version
    for v, p in zip(rows[0::2], rows[1::2]):
        if not (v["target"] == 1 and p["target"] == 0 and v["commit_id"] == p["commit_id"]):
            continue
        if not ok(v["func"], p["func"]):
            continue
        pv.append({"unit_id": f"PV-{v['idx']}", "dataset": "PrimeVul-paired", "split": split, "cve_id": v.get("cve") or "",
                   "cwe": (v.get("cwe") or ["?"])[0] if isinstance(v.get("cwe"), list) else (v.get("cwe") or "?"),
                   "project": v["project"], "commit_id": v["commit_id"], "func_before": v["func"], "func_after": p["func"],
                   "vul_lines": fix_lines(v["func"], p["func"]), "vul_lines_source": "fix-diff",
                   "cve_year": year(v.get("cve")), "extra": {"cve_desc": v.get("cve_desc"), "nvd_url": v.get("nvd_url"),
                                                            "file_name": v.get("file_name")}})
# All unique-CVE test+valid pairs (the newest, chronologically held-out splits), topped up to 1,000 with a
# seeded sample of older train pairs.
seen, pv_unique, train_pool = set(), [], []
for r in pv:
    key = r["cve_id"] or r["commit_id"]
    if key in seen:
        continue
    seen.add(key)
    (train_pool if r["split"] == "train" else pv_unique).append(r)
rng.shuffle(train_pool)
pv_unique += train_pool[:max(0, 1000 - len(pv_unique))]
write("primevul_pairs.jsonl", sorted(pv_unique, key=lambda x: (x["split"], x["unit_id"])))
# The other sets are kept disjoint from PrimeVul (no shared CVE, no identical vulnerable function).
PV_CVES = {r["cve_id"] for r in pv_unique if r["cve_id"]}
PV_FUNCS = {norm(r["func_before"]) for r in pv_unique}


def disjoint(r):
    return not (r["cve_id"] and r["cve_id"] in PV_CVES) and norm(r["func_before"]) not in PV_FUNCS

# ---- SecVulEval: statement-level localization ----
d = pd.read_parquet(RAW / "secvuleval.parquet")
by_idx = d.set_index("idx")
cands = []
for _, r in d[d.is_vulnerable & d.fixed_func_idx.notna()].iterrows():
    fidx = int(r.fixed_func_idx)
    if fidx not in by_idx.index:
        continue
    before, after = r.func_body, by_idx.loc[fidx].func_body
    if not isinstance(after, str) or not ok(before, after):
        continue
    try:
        labelled = json.loads(r.changed_statements)
    except (ValueError, TypeError):
        continue
    lines = before.splitlines()
    rel = sorted({i + 1 for _, st in labelled for i, l in enumerate(lines) if norm(st) and norm(st) in norm(l)})
    if not rel:
        continue
    cves = list(r.cve_list) if r.cve_list is not None else []
    cwes = list(r.cwe_list) if r.cwe_list is not None else []
    ctx = {k: (list(v) if v is not None else []) for k, v in (r.context or {}).items()}
    cands.append({"unit_id": f"SVE-{r.idx}", "dataset": "SecVulEval", "split": "eval", "cve_id": cves[0] if cves else "",
                  "cwe": cwes[0] if cwes else "?", "project": r.project, "commit_id": r.commit_id,
                  "func_before": before, "func_after": after, "vul_lines": rel, "vul_lines_source": "secvuleval-statements",
                  "cve_year": year(cves[0] if cves else ""), "extra": {"context": ctx, "func_name": r.func_name}})
seen, uniq = set(), []
rng.shuffle(cands)
per_project = Counter()
for c in cands:                       # one per CVE, at most 40 per project (SecVulEval is Linux-heavy)
    if c["cve_id"] in seen or per_project[c["project"]] >= 40 or not disjoint(c):
        continue
    seen.add(c["cve_id"]); per_project[c["project"]] += 1
    uniq.append(c)
write("secvuleval_loc.jsonl", sorted(uniq[:500], key=lambda x: x["unit_id"]))

# ---- CleanVul score 4, C/C++ ----
c = pd.read_csv(RAW / "cleanvul_4.csv")
c = c[c.extension.astype(str).str.lower().isin(["c", "cpp", "cc", "cxx", "h", "hpp"]) & (c.is_test.astype(str).str.lower() != "true")]
cv, seen = [], set()
for _, r in c.sample(frac=1, random_state=42).iterrows():
    cve = "" if pd.isna(r.cve_id) else str(r.cve_id)
    key = cve or str(r.commit_url)
    if key in seen or not isinstance(r.func_before, str) or not isinstance(r.func_after, str) or not ok(r.func_before, r.func_after):
        continue
    if (cve and cve in PV_CVES) or norm(r.func_before) in PV_FUNCS:
        continue
    seen.add(key)
    m = re.search(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-f]+)", str(r.commit_url))
    cv.append({"unit_id": f"CV-{len(cv)}", "dataset": "CleanVul-score4", "split": "eval", "cve_id": cve, "cwe": "" if pd.isna(r.cwe_id) else str(r.cwe_id),
               "project": m.group(1) if m else "?", "commit_id": m.group(2) if m else "", "func_before": r.func_before,
               "func_after": r.func_after, "vul_lines": fix_lines(r.func_before, r.func_after), "vul_lines_source": "fix-diff",
               "cve_year": year(cve), "extra": {"date": str(r.date), "file_name": r.file_name}})
    if len(cv) >= 300:
        break
write("cleanvul_pairs.jsonl", cv)
