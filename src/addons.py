"""addons.py -- RQ1 add-on experiments (study design v2), run per model under test:

  RQ1-E3  comparability on the unfiltered official PrimeVul 435-pair test split (R1-R5): FPR, pair-wise correct
          (P-C), VP-S = P-C - reversed pairs, macro-F1, plus context overflows and parse failures.
  RQ1-E4  prompt robustness: the RQ1 zero-shot prompt vs. a chain-of-thought variant, R1 and R4, on the RQ1
          nondeterminism subset.
  RQ1-E5  cross-function context probe on SecVulEval: R6 = R4 + the dataset's inter-procedural context NAMES
          (globals, external functions, arguments, type declarations; never its 'Explanation' field, which could
          leak the label), compared with R1 and R4.

All calls go through rp_llm_cache with the RQ1 system prompt; the zero-shot calls reuse the RQ1 user template, so
units shared with RQ1 replay from the cache. A prompt longer than the model's context window is recorded as an
overflow and scored as a wrong answer (reported), never truncated silently. Writes results["RQ1"]["_detail"]["addons"].
"""
from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import llm_runner as L
from data_prep import _pair_to_units
from slice_pipeline import build_representations, compute_slice

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
RANDOM_SEED = 42

COT_TEMPLATE = (
    "Analyze the following code context and decide if it contains a software security vulnerability.\n\n"
    "First reason step by step: identify untrusted inputs, follow their data flow through the code, and check "
    "whether each dangerous operation is protected by a sufficient check. Keep the reasoning under 200 words.\n"
    "Then, on the LAST line, return STRICT JSON exactly in this schema:\n"
    '{{"vulnerable": true or false, "cited_lines": [list of 1-based line numbers], "cwe": "CWE-id or unknown"}}\n\n'
    "Code context (lines are prefixed with their 1-based line number):\n```c\n{context}\n```\n"
)
R6_FIELDS = ("Globals", "External Function", "Function Argument", "Type Execution Declaration")


def _is_overflow(exc: Exception) -> bool:
    text = str(exc).lower()
    return "maximum context length" in text or "context length" in text or "too long" in text


# Self-hosted models (pinned revisions) and their context window. A prompt that does not fit is an overflow;
# it is decided from the model's own tokenizer and chat template (the rule vLLM applies), so it is the same online
# and in an offline replay, where nothing is cached for a rejected request.
MODEL_REVISIONS = {"Qwen/Qwen2.5-Coder-14B-Instruct": "aedcc2d42b622764e023cf882b6652e646b95671",
                   "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "dcaee4d4dfc5ee71ad501f01f530e5652438fde0",
                   "microsoft/phi-4": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"}
MAX_MODEL_LEN = 16384
_TOKENIZERS: dict[str, Any] = {}
_TOK_LOCK = __import__("threading").Lock()


def _prompt_tokens(model: str, system: str, user: str) -> int | None:
    if model not in MODEL_REVISIONS:
        return None
    with _TOK_LOCK:
        if model not in _TOKENIZERS:
            from transformers import AutoTokenizer
            _TOKENIZERS[model] = AutoTokenizer.from_pretrained(model, revision=MODEL_REVISIONS[model])
        tok = _TOKENIZERS[model]
    return len(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       add_generation_prompt=True, tokenize=True))


def _cached(llm, model: str, user: str, max_tokens: int) -> bool:
    k = llm.key(model, L.SYSTEM_PROMPT, user, max_tokens, 0.0, 0)
    with llm._lock:
        return llm.db.execute("SELECT 1 FROM c WHERE k=?", (k,)).fetchone() is not None


def _call(llm, model: str, template: str, context: str, max_tokens: int) -> dict[str, Any]:
    user = template.format(context=context)
    if not _cached(llm, model, user, max_tokens):
        n = _prompt_tokens(model, L.SYSTEM_PROMPT, user)
        if n is not None and n + max_tokens > MAX_MODEL_LEN:
            return {"vulnerable": False, "cited_lines": [], "parsed": False, "overflow": True}
    try:
        text = llm.complete(L.SYSTEM_PROMPT, user, model=model,
                            max_tokens=max_tokens, temperature=0.0, sample=0)
    except Exception as exc:  # noqa: BLE001 -- only context overflow is tolerated; everything else surfaces
        if type(exc).__name__ == "BadRequestError" and _is_overflow(exc):
            return {"vulnerable": False, "cited_lines": [], "parsed": False, "overflow": True}
        raise
    if template is COT_TEMPLATE:  # the verdict is the last JSON object in the answer
        tail = text[text.rfind("{"):] if "{" in text else text
        res = L.parse_response(tail)
        if not res.get("parsed"):
            res = L.parse_response(text)
    else:
        res = L.parse_response(text)
    res["overflow"] = False
    return res


def _run(jobs: list[tuple], llm, workers: int) -> dict[tuple, dict[str, Any]]:
    """jobs: (key, model, template, context, max_tokens) -> {key: parsed result}."""
    out: dict[tuple, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_call, llm, m, t, c, mt): k for k, m, t, c, mt in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if i % 500 == 0 or i == len(futs):
                print(f"[addons] {i}/{len(futs)} calls", flush=True)
    return out


def _detection(preds: dict[str, int], units: list[Any]) -> dict[str, Any]:
    """Pair-aware detection metrics over units named '<pair>__vul' / '<pair>__fix'."""
    y = np.array([u.label for u in units]); p = np.array([preds[u.unit_id] for u in units])
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0; rec0 = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    pairs: dict[str, dict[int, int]] = {}
    for u in units:
        pairs.setdefault(u.unit_id.rsplit("__", 1)[0], {})[u.label] = preds[u.unit_id]
    full = [d for d in pairs.values() if 0 in d and 1 in d]
    pc = sum(1 for d in full if d[1] == 1 and d[0] == 0)
    reversed_ = sum(1 for d in full if d[1] == 0 and d[0] == 1)
    both_vul = sum(1 for d in full if d[1] == 1 and d[0] == 1)
    both_safe = sum(1 for d in full if d[1] == 0 and d[0] == 0)
    return {"n_units": len(units), "n_pairs": len(full), "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "pair_correct": pc, "vp_s": pc - reversed_, "pair_reversed": reversed_,
            "pair_both_vulnerable": both_vul, "pair_both_safe": both_safe,
            "precision": prec, "recall": rec, "macro_f1": (f1 + f10) / 2, "accuracy": (tp + tn) / len(units)}


def _localization(res: dict[str, dict], units: list[Any]) -> dict[str, float]:
    """Line-level recall/precision and top-1 hit on vulnerable units that carry ground-truth lines."""
    rec, prec, top1 = [], [], []
    for u in units:
        if u.label != 1 or not u.vul_lines:
            continue
        cited = [int(x) for x in res[u.unit_id].get("cited_lines") or []]
        gt = set(u.vul_lines)
        hit = len(gt & set(cited))
        rec.append(hit / len(gt)); prec.append(hit / len(set(cited)) if cited else 0.0)
        top1.append(1.0 if cited and cited[0] in gt else 0.0)
    m = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {"n_localization_units": len(rec), "line_recall": m(rec), "line_precision": m(prec), "top_1": m(top1)}


def _counts(res: dict[str, dict], units: list[Any]) -> dict[str, int]:
    return {"overflow": sum(1 for u in units if res[u.unit_id].get("overflow")),
            "parse_failures": sum(1 for u in units if not res[u.unit_id].get("parsed") and not res[u.unit_id].get("overflow"))}


def _reps_basic(u: Any) -> dict[str, str]:
    from experiment_rq1 import _criterion_vars
    return build_representations(u.code, compute_slice(u.code, criterion_vars=_criterion_vars(u)))


def e3_official_primevul(models: list[str], llm, workers: int, max_pairs: int | None = None) -> dict[str, Any]:
    from experiment_rq1 import REPRESENTATIONS, build_unit_representations
    rows = [json.loads(l) for l in open(DATA / "primevul_official_test_pairs.jsonl")][:max_pairs]
    units = [u for r in rows for u in _pair_to_units(r, "primevul_official")]
    print(f"[addons] RQ1-E3: {len(rows)} official PrimeVul test pairs ({len(units)} units)", flush=True)
    per_unit = build_unit_representations(units)
    jobs = [((u.unit_id, m, r), m, L.USER_TEMPLATE, per_unit[u.unit_id]["reps"][r], 256)
            for u in units for m in models for r in REPRESENTATIONS]
    raw = _run(jobs, llm, workers)
    out = {}
    for m in models:
        out[m] = {}
        for r in REPRESENTATIONS:
            res = {u.unit_id: raw[(u.unit_id, m, r)] for u in units}
            out[m][r] = {**_detection({k: int(bool(v["vulnerable"])) for k, v in res.items()}, units),
                         **_localization(res, units), **_counts(res, units)}
    return {"note": "Unfiltered official PrimeVul paired test split; overflows scored as wrong.", "results": out,
            "cited_baselines_pc_on_same_split": {
                "Aegis (DeepSeek-V3.1, arXiv 2603.20637)": 122, "VulTrial (GPT-4o, trained)": 96,
                "ReVD (Qwen2.5-Coder-7B, fine-tuned)": 78, "CoT (GPT-4o, Ding et al.)": 40},
            "cited_note": "Pair-wise correct predictions as reported in Fang et al. (arXiv 2603.20637, Table 1); not reproduced."}


def e4_prompt_robustness(units: list[Any], nondet_ids: list[str], models: list[str], llm, workers: int) -> dict[str, Any]:
    sub = [u for u in units if u.unit_id in set(nondet_ids)]
    print(f"[addons] RQ1-E4: zero-shot vs CoT on {len(sub)} units", flush=True)
    reps = {u.unit_id: _reps_basic(u) for u in sub}
    jobs = [((u.unit_id, m, r, p), m, t, reps[u.unit_id][r], mt)
            for u in sub for m in models for r in ("R1", "R4")
            for p, t, mt in (("zero_shot", L.USER_TEMPLATE, 256), ("cot", COT_TEMPLATE, 1024))]
    raw = _run(jobs, llm, workers)
    out = {}
    for m in models:
        out[m] = {}
        for p in ("zero_shot", "cot"):
            out[m][p] = {}
            for r in ("R1", "R4"):
                res = {u.unit_id: raw[(u.unit_id, m, r, p)] for u in sub}
                out[m][p][r] = {**_detection({k: int(bool(v["vulnerable"])) for k, v in res.items()}, sub),
                                **_localization(res, sub), **_counts(res, sub)}
    return {"n_units": len(sub), "results": out}


def e5_crossfn_context(units: list[Any], models: list[str], llm, workers: int) -> dict[str, Any]:
    rows = {str(json.loads(l)["unit_id"]): json.loads(l) for l in open(DATA / "secvuleval_loc.jsonl")}
    sub = [u for u in units if u.unit_id.rsplit("__", 1)[0] in rows]
    print(f"[addons] RQ1-E5: R6 cross-function context on {len(sub)} SecVulEval units", flush=True)

    def r6_block(u: Any) -> str:
        ctx = (rows[u.unit_id.rsplit("__", 1)[0]].get("extra") or {}).get("context") or {}
        parts = [f"{f}: {', '.join(map(str, ctx.get(f) or []))[:400]}" for f in R6_FIELDS if ctx.get(f)]
        return "/* cross-function context (names from the enclosing program) */\n" + "\n".join(parts) if parts else ""

    reps = {}
    for u in sub:
        b = _reps_basic(u); blk = r6_block(u)
        reps[u.unit_id] = {"R1": b["R1"], "R4": b["R4"], "R6": (blk + "\n" + b["R4"]) if blk else b["R4"]}
    jobs = [((u.unit_id, m, r), m, L.USER_TEMPLATE, reps[u.unit_id][r], 256)
            for u in sub for m in models for r in ("R1", "R4", "R6")]
    raw = _run(jobs, llm, workers)
    out = {}
    for m in models:
        out[m] = {}
        for r in ("R1", "R4", "R6"):
            res = {u.unit_id: raw[(u.unit_id, m, r)] for u in sub}
            out[m][r] = {**_detection({k: int(bool(v["vulnerable"])) for k, v in res.items()}, sub),
                         **_localization(res, sub), **_counts(res, sub)}
    with_ctx = sum(1 for u in sub if reps[u.unit_id]["R6"] != reps[u.unit_id]["R4"])
    return {"n_units": len(sub), "units_with_context": with_ctx, "fields_used": list(R6_FIELDS),
            "note": "Names of cross-function entities only (not their definitions); the dataset's Explanation field is excluded.",
            "results": out}


FT_LABEL = "Fine-tuned LLM (LoRA on Qwen2.5-Coder-7B-Instruct, PrimeVul train split)"


def ed_finetuned(units: list[Any], model: str, llm, workers: int) -> dict[str, Any]:
    """Fine-tuned baseline: the LoRA adapter (served by vLLM as `model`) on the RQ1 units, R1 and R4, rep 0."""
    print(f"[addons] RQ1-E2 fine-tuned baseline {model} on {len(units)} units", flush=True)
    reps = {u.unit_id: _reps_basic(u) for u in units}
    jobs = [((u.unit_id, r), model, L.USER_TEMPLATE, reps[u.unit_id][r], 256) for u in units for r in ("R1", "R4")]
    raw = _run(jobs, llm, workers)
    out = {}
    for r in ("R1", "R4"):
        res = {u.unit_id: raw[(u.unit_id, r)] for u in units}
        out[r] = {**_detection({k: int(bool(v["vulnerable"])) for k, v in res.items()}, units),
                  **_localization(res, units), **_counts(res, units)}
    return {"model": model, "results": out,
            "training": "LoRA r=16, alpha=32, 1 epoch on 10,877 R1+R4 examples from 3,051 PrimeVul train pairs "
                        "disjoint from every evaluation set (see train_config.json)"}


def _san(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def run(units: list[Any], nondet_ids: list[str], models: list[str], workers: int = 32,
        max_pairs: int | None = None) -> dict[str, Any]:
    from rp_llm_cache import LLMCache
    llm = LLMCache()
    out = {"RQ1-E3": e3_official_primevul(models, llm, workers, max_pairs),
           "RQ1-E4": e4_prompt_robustness(units, nondet_ids, models, llm, workers),
           "RQ1-E5": e5_crossfn_context(units, models, llm, workers)}
    import os
    # The fine-tuned adapter's name comes from RP_FINETUNED_MODEL (while it is being served) or from
    # data/finetuned_model.json (written once its answers are in the frozen cache, so replays include it).
    ft_model = os.environ.get("RP_FINETUNED_MODEL")
    marker = DATA / "finetuned_model.json"
    if not ft_model and marker.exists():
        ft_model = json.loads(marker.read_text()).get("model")
    if ft_model:
        out["RQ1-E2-finetuned"] = ed_finetuned(units, ft_model, llm, workers)
        scores = DATA / "ft_logprob_scores.jsonl"  # frozen label-token scores (ft_logprob_rank.py)
        if scores.exists():
            import ft_logprob_rank
            rows = [json.loads(l) for l in open(scores)]
            out["RQ1-E2-finetuned"]["label_token_ranking"] = ft_logprob_rank.analyse(rows)
    path = RESULTS / "results.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.setdefault("RQ1", {}).setdefault("_detail", {})["addons"] = _san(out)
    if ft_model:
        best = max(out["RQ1-E2-finetuned"]["results"].values(), key=lambda v: v["macro_f1"])
        merged["RQ1"].setdefault("baseline_comparison", {})[FT_LABEL] = _san(
            {k: best[k] for k in ("precision", "recall", "macro_f1", "fpr", "pair_correct", "vp_s", "line_recall")})
    path.write_text(json.dumps(merged, indent=2))
    print("[addons] wrote RQ1-E3/E4/E5 into results.json", flush=True)
    return out
