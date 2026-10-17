"""rq4_cot_sample.py -- RQ4 layer 2: failures WITH the model's own reasoning, for coding with the Aegis categories.

Reads the chain-of-thought answers of RQ1-E4 (the nondeterminism subset; COT_TEMPLATE, 1,024 tokens) from the frozen
cache (cache hits only; an uncached request is skipped, never sent), keeps the failures, and draws a stratified
sample (seed 42, balanced across models and representations):
  14 detection false positives, 14 detection false negatives, 12 localization misses.
Writes
  <out>/df89e_rq4_cot_sample.csv  coder sheet: no model names, shuffled, empty coding columns
  <out>/df89e_rq4_cot_key.csv     case_id -> model, representation, unit (for analysis only; not for coders)
Usage: python rq4_cot_sample.py OUT_DIR   (run inside the paper-grade run directory)
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import addons
import data_prep
import llm_runner as L
from rp_llm_cache import LLMCache

ROOT = Path(__file__).resolve().parent
SEED = 42
STRATA = {"detection_false_positive": 14, "detection_false_negative": 14, "localization_miss": 12}
CODING_COLS = ["primary_category", "secondary_category", "confidence", "note"]


def reasoning_and_verdict(text: str):
    """The verdict is the last JSON object; the reasoning is everything else (models put it before or after)."""
    import re
    tail_at = text.rfind("{")
    res = L.parse_response(text[tail_at:] if tail_at >= 0 else text)
    if not res.get("parsed"):
        res = L.parse_response(text)
    end = text.find("}", tail_at) + 1 if tail_at >= 0 else -1
    rest = (text[:tail_at] + text[end:]) if tail_at >= 0 and end > 0 else text
    rest = re.sub(r"```(json)?", "", rest).strip()
    return rest, res


def main() -> None:
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    recs = {json.loads(l)["unit_id"]: json.loads(l) for l in open(ROOT / "results" / "records_rq1.jsonl")}
    sub_ids = {u for u, r in recs.items() if r.get("in_nondet_subset")}
    units, _ = data_prep.load_units(max_rows=None, allow_synthetic=False, strict=True)
    units = [u for u in units if u.unit_id in sub_ids]
    llm, models = LLMCache(), list(L.MODELS_UNDER_TEST)
    cands, skipped = defaultdict(list), 0
    present = defaultdict(lambda: [0, 0])  # model -> [answers with reasoning, answers]
    for u in units:
        reps = addons._reps_basic(u)
        for rep in ("R1", "R4"):
            user = addons.COT_TEMPLATE.format(context=reps[rep])
            for m in models:
                if not addons._cached(llm, m, user, 1024):
                    skipped += 1
                    continue
                text = llm.complete(L.SYSTEM_PROMPT, user, model=m, max_tokens=1024, temperature=0.0, sample=0)
                reasoning, res = reasoning_and_verdict(text)
                present[m][1] += 1; present[m][0] += int(len(reasoning) >= 40)
                if not res.get("parsed") or len(reasoning) < 40:
                    continue
                pred, cited = int(bool(res["vulnerable"])), list(res.get("cited_lines") or [])
                vul = list(recs[u.unit_id].get("vul_lines") or [])
                row = {"unit": u, "model": m, "rep": rep, "code": reps[rep], "reasoning": reasoning,
                       "pred": pred, "cited": cited, "cwe_pred": res.get("cwe", ""), "vul": vul}
                if u.label == 0 and pred == 1:
                    cands["detection_false_positive"].append(row)
                elif u.label == 1 and pred == 0:
                    cands["detection_false_negative"].append(row)
                elif u.label == 1 and pred == 1 and vul and not set(cited) & set(vul):
                    cands["localization_miss"].append(row)
    rng = random.Random(SEED)
    picked = []
    for stratum, k in STRATA.items():
        pool = cands[stratum]
        by = defaultdict(list)
        for r in pool:
            by[(r["model"], r["rep"])].append(r)
        for v in by.values():
            rng.shuffle(v)
        keys = sorted(by); i = 0; got = []
        while len(got) < k and any(by.values()):
            key = keys[i % len(keys)]; i += 1
            if by[key]:
                got.append(by[key].pop())
        picked += [(stratum, r) for r in got]
    rng.shuffle(picked)
    stage = {"detection_false_positive": "detection", "detection_false_negative": "detection",
             "localization_miss": "localization"}
    with open(out_dir / "df89e_rq4_cot_sample.csv", "w", newline="") as fs, \
         open(out_dir / "df89e_rq4_cot_key.csv", "w", newline="") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["case_id", "stage", "code_shown_to_model", "ground_truth_label", "ground_truth_lines",
                     "model_reasoning", "model_verdict", "model_cited_lines", "model_cwe"] + CODING_COLS)
        wk.writerow(["case_id", "stratum", "model", "representation", "unit_id"])
        for n, (stratum, r) in enumerate(picked, 1):
            cid = f"T{n:03d}"
            ws.writerow([cid, stage[stratum], r["code"], "vulnerable" if r["unit"].label else "safe (fixed version)",
                         " ".join(map(str, r["vul"])) if r["unit"].label else "", r["reasoning"],
                         "vulnerable" if r["pred"] else "safe", " ".join(map(str, r["cited"])), r["cwe_pred"]]
                        + [""] * len(CODING_COLS))
            wk.writerow([cid, stratum, r["model"], r["rep"], r["unit"].unit_id])
    print("RQ4 CoT sample:", len(picked), "cases", {s: sum(1 for t, _ in picked if t == s) for s in STRATA},
          "| pools", {s: len(v) for s, v in cands.items()}, "| uncached skipped", skipped, "->", out_dir)


if __name__ == "__main__":
    main()
