"""rq4_sample.py -- stratified RQ4 failure sample for double coding (see Docs/df89e_rq4_codebook.md, section 4).

Reads results/records_rq1.jsonl and results/records_rq3.jsonl (written by main.py) plus the data files, and writes
  <out>/df89e_rq4_sample.csv  coder sheet: no model names, shuffled, empty coding columns
  <out>/df89e_rq4_key.csv     case_id -> model, representation, unit (for analysis only; not for coders)
Strata (20 each, seed 42, balanced across models): detection FP, detection FN, localization miss, repair failure,
slice effect (10 where R4 was right and R1 wrong, 10 the reverse).
Usage: python rq4_sample.py OUT_DIR   (run inside the paper-grade run directory; needs srcML/srcSlice on PATH)
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from experiment_rq1 import _criterion_vars
from data_prep import _pair_to_units
from slice_pipeline import build_representations, compute_slice

ROOT = Path(__file__).resolve().parent
PER_STRATUM, SEED = 20, 42
DATA_FILES = {"primevul": "primevul_pairs.jsonl", "secvuleval": "secvuleval_loc.jsonl",
              "cleanvul": "cleanvul_pairs.jsonl", "postcutoff": "postcutoff_pairs.jsonl"}
CODING_COLS = ["primary_category", "secondary_category", "slice_effect", "confidence", "note"]


def load_units():
    units = {}
    for src, f in DATA_FILES.items():
        p = ROOT / "data" / f
        if p.exists():
            for row in map(json.loads, open(p)):
                for u in _pair_to_units(row, src):
                    units[u.unit_id] = (u, row)
    return units


def balanced(cands, k, rng):
    """Pick k candidates round-robin over models so every model is represented."""
    by_model = defaultdict(list)
    for c in cands:
        by_model[c["model"]].append(c)
    for v in by_model.values():
        rng.shuffle(v)
    out = []
    while len(out) < k and any(by_model.values()):
        for m in sorted(by_model):
            if by_model[m] and len(out) < k:
                out.append(by_model[m].pop())
    return out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    units = load_units()
    rq1 = [json.loads(l) for l in open(ROOT / "results" / "records_rq1.jsonl")]
    rq3 = [json.loads(l) for l in open(ROOT / "results" / "records_rq3.jsonl")]
    strata = defaultdict(list)
    for r in rq1:
        e1 = (r["by_rep"].get("R1") or [{}])[0]; e4 = (r["by_rep"].get("R4") or [{}])[0]
        for m in set(e1) | set(e4):
            for rep, e in (("R1", e1.get(m)), ("R4", e4.get(m))):
                if not e:
                    continue
                c = {"unit_id": r["unit_id"], "model": m, "rep": rep, "pred": e["pred"], "cited": e.get("cited_lines") or [],
                     "label": r["label"], "vul_lines": r.get("vul_lines") or []}
                if r["label"] == 0 and e["pred"] == 1:
                    strata["detection_false_positive"].append(c)
                elif r["label"] == 1 and e["pred"] == 0:
                    strata["detection_false_negative"].append(c)
                elif r["label"] == 1 and c["vul_lines"] and not set(c["vul_lines"]) & set(map(int, c["cited"])):
                    strata["localization_miss"].append(c)
            if e1.get(m) and e4.get(m) and e1[m]["pred"] != e4[m]["pred"]:
                right = "R4" if e4[m]["pred"] == r["label"] else "R1"
                strata["slice_helped" if right == "R4" else "slice_hurt"].append(
                    {"unit_id": r["unit_id"], "model": m, "rep": "R1 vs R4", "label": r["label"],
                     "r1": e1[m], "r4": e4[m], "vul_lines": r.get("vul_lines") or []})
    for r in rq3:
        if r.get("repeat", 0) == 0 and not r.get("eliminated_static") and r.get("candidate"):
            strata["repair_failure"].append({"unit_id": r["unit_id"], "model": r["model"], "rep": r["representation"],
                                             "patch": r["candidate"], "oracle": r.get("static_oracle")})
    picks = []
    for name in ("detection_false_positive", "detection_false_negative", "localization_miss", "repair_failure"):
        picks += [(name, c) for c in balanced(strata[name], PER_STRATUM, rng)]
    picks += [("slice_effect", c) for c in balanced(strata["slice_helped"], PER_STRATUM // 2, rng)]
    picks += [("slice_effect", c) for c in balanced(strata["slice_hurt"], PER_STRATUM // 2, rng)]
    rng.shuffle(picks)

    sheet, key = [], []
    for i, (stratum, c) in enumerate(picks, 1):
        uid = c["unit_id"]
        repair = stratum == "repair_failure"
        u, row = units.get(uid if not repair else f"{uid}__vul", (None, None))
        if u is None:
            continue
        reps = build_representations(u.code, compute_slice(u.code, criterion_vars=_criterion_vars(u)))
        shown = reps["R4"] if c["rep"] in ("R4",) else reps["R1"]
        case = f"C{i:03d}"
        sheet.append({
            "case_id": case, "stage": stratum.replace("_", " "),
            "code_shown_to_model": reps["R1"] if stratum == "slice_effect" else shown,
            "slice_shown (slice-effect cases)": reps["R4"] if stratum == "slice_effect" else "",
            "ground_truth_label": "vulnerable" if u.label == 1 else "safe (fixed version)",
            "ground_truth_lines": json.dumps(u.vul_lines),
            "official_fix (repair cases)": row.get("func_after", "") if repair else "",
            "model_verdict": "" if stratum in ("slice_effect", "repair_failure") else ("vulnerable" if c["pred"] else "safe"),
            "model_cited_lines": json.dumps(c.get("cited", [])),
            "whole_function_answer (slice-effect)": json.dumps(c["r1"]) if stratum == "slice_effect" else "",
            "slice_answer (slice-effect)": json.dumps(c["r4"]) if stratum == "slice_effect" else "",
            "model_patch (repair cases)": c.get("patch", "") if repair else "",
            **{k: "" for k in CODING_COLS}})
        key.append({"case_id": case, "stratum": stratum, "model": c["model"], "representation": c["rep"], "unit_id": uid,
                    "oracle": json.dumps(c.get("oracle")) if repair else ""})
    for name, rows in (("df89e_rq4_sample.csv", sheet), ("df89e_rq4_key.csv", key)):
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    counts = defaultdict(int)
    for k in key:
        counts[k["stratum"]] += 1
    print(f"RQ4 sample: {len(sheet)} cases {dict(counts)} -> {out_dir}")


if __name__ == "__main__":
    main()
