"""ft_logprob_rank.py -- label-token ranking of the fine-tuned LoRA baseline vs. its base model (RQ1-E2 follow-up).

The adapter's greedy verdicts are constant ("safe"), so this reads the model's score instead: with the RQ1 system
prompt and user template, the assistant turn is prefilled with '{"vulnerable": ' and the next-token log-probabilities
of "true" and "false" give s = P(true) / (P(true) + P(false)). Per representation (R1, R4) and model it reports the
share of pairs whose vulnerable version scores above its fix (ties 0.5), unit-level ROC-AUC, and pair-correctness at
a threshold chosen by 5-fold cross-validation over pairs. Raw scores are frozen in results/ft_logprob_scores.jsonl,
so the analysis replays without a server (python ft_logprob_rank.py --replay).
Server: vLLM at RP_LOCAL_LLM_BASE_URL serving the base model and the adapter (local:qwen7b-primevul-lora-blind).
"""
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm_runner as L

ROOT = Path(__file__).resolve().parent
OUT_SCORES = ROOT / "results" / "ft_logprob_scores.jsonl"
OUT = ROOT / "results" / "ft_logprob_rank.json"
MODELS = {"lora": os.environ.get("RP_FINETUNED_MODEL", "local:qwen7b-primevul-lora-blind"),
          "base": "Qwen/Qwen2.5-Coder-7B-Instruct"}
PREFIX = '{"vulnerable": '
BASE = os.environ.get("RP_LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")


def score(model: str, user: str) -> dict:
    body = {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": L.SYSTEM_PROMPT}, {"role": "user", "content": user},
                         {"role": "assistant", "content": PREFIX}],
            "add_generation_prompt": False, "continue_final_message": True}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(20):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600))
            break
        except urllib.error.HTTPError as e:
            if e.code == 400:  # context overflow
                return {"s": None, "overflow": True}
            if attempt == 19: raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 19: raise
        import time; time.sleep(10)
    top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {}
    for t in top:
        k = t["token"].strip().lower()
        if k in ("true", "false") and k not in lp:
            lp[k] = t["logprob"]
    floor = min(t["logprob"] for t in top) - 1.0  # token outside the top 20: bound its probability from above
    pt, pf = math.exp(lp.get("true", floor)), math.exp(lp.get("false", floor))
    return {"s": pt / (pt + pf), "lp_true": lp.get("true"), "lp_false": lp.get("false"), "overflow": False}


def collect(units) -> list[dict]:
    from addons import _reps_basic
    jobs = []
    for u in units:
        reps = _reps_basic(u)
        for rep in ("R1", "R4"):
            user = L.USER_TEMPLATE.format(context=reps[rep])
            for tag, m in MODELS.items():
                jobs.append((u.unit_id, u.label, rep, tag, m, user))
    rows = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("RP_LLM_WORKERS", 32))) as ex:
        futs = {ex.submit(score, j[4], j[5]): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            uid, lab, rep, tag, m, _ = futs[f]
            rows.append({"unit": uid, "label": lab, "rep": rep, "model": tag, "model_id": m, **f.result()})
            if i % 1000 == 0 or i == len(futs):
                print(f"[ft_logprob] {i}/{len(futs)}", flush=True)
    return rows


def auc(pos, neg) -> float:
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        for k in range(i, j): ranks[k] = (i + j + 1) / 2
        i = j
    rp = sum(ranks[k] for k, (v, y) in enumerate(allv) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


def analyse(rows) -> dict:
    res = {}
    for tag in MODELS:
        for rep in ("R1", "R4"):
            sub = [r for r in rows if r["model"] == tag and r["rep"] == rep and r["s"] is not None]
            pairs = {}
            for r in sub:
                pairs.setdefault(r["unit"].rsplit("__", 1)[0], {})[r["label"]] = r["s"]
            full = [(d[1], d[0]) for d in pairs.values() if 0 in d and 1 in d]
            wins = sum(1.0 if v > f else 0.5 if v == f else 0.0 for v, f in full)
            # pair-correct at a threshold chosen on the other folds (5-fold CV over pairs)
            idx = list(range(len(full))); random.Random(42).shuffle(idx)
            folds = [idx[k::5] for k in range(5)]; pc = 0
            for k in range(5):
                train = [full[i] for j in range(5) if j != k for i in folds[j]]
                cands = sorted({x for p in train for x in p})
                best = max(cands, key=lambda t: sum(1 for v, f in train if v >= t > f))
                pc += sum(1 for i in folds[k] if full[i][0] >= best > full[i][1])
            ss = [r["s"] for r in sub]
            res[f"{tag}|{rep}"] = {
                "n_pairs": len(full), "overflow_units": sum(1 for r in rows if r["model"] == tag and r["rep"] == rep and r["s"] is None),
                "pair_rank_accuracy": wilson(wins, len(full)) if full else None,
                "ties": sum(1 for v, f in full if v == f),
                "roc_auc": auc([r["s"] for r in sub if r["label"] == 1], [r["s"] for r in sub if r["label"] == 0]),
                "pair_correct_cv_threshold": wilson(pc, len(full)) if full else None,
                "score_mean": sum(ss) / len(ss), "score_min": min(ss), "score_max": max(ss),
                "greedy_vulnerable_share": sum(1 for x in ss if x > 0.5) / len(ss)}
    return res


def main() -> None:
    if "--replay" in sys.argv:
        src = OUT_SCORES if OUT_SCORES.exists() else ROOT / "data" / "ft_logprob_scores.jsonl"
        rows = [json.loads(l) for l in open(src)]
    else:
        import data_prep
        units, _ = data_prep.load_units(max_rows=None, allow_synthetic=False, strict=True)
        if "--smoke" in sys.argv:
            units = units[:8]
        rows = collect(units)
        if "--smoke" not in sys.argv:
            OUT_SCORES.write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = analyse(rows)
    if "--smoke" not in sys.argv:
        OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
