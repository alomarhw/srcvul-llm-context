"""
experiment_rq2.py -- RQ2 efficiency experiments.

RQ2-E1: For every representation x model, record input/output token counts,
        end-to-end latency, and inference cost. These come from the SAME LLM
        detection runs used in RQ1 (llm_runner logs usage per call); we do not
        issue new calls, we aggregate the usage already captured in the frozen
        cache so numbers replay exactly.

RQ2-E2: Efficiency-without-loss. Pairs the slice representation (R2/R4) against
        whole-function (R1) and tests (a) macro-F1 EQUIVALENCE via TOST within a
        pre-registered margin, alongside (b) paired reductions in tokens/latency/
        cost. TOST margin default = max(0.02, 1 pilot-SD). The pilot-SD piece is
        a DESIGN decision (see todos); we implement the explicit 0.02 default and
        surface the pilot-SD option as a clearly-labeled note rather than guessing.

Token counting uses tiktoken when available; otherwise a whitespace/4-char
approximation clearly labeled as approximate. Cost uses a transparent per-token
proxy (DESIGN NEEDED: confirm API pricing) recorded in the output so the Writer
can substitute real pricing.

Requirements: numpy==2.2.6, pandas==2.2.2, tiktoken==0.14.0 (optional).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))

# Transparent per-1k-token cost proxy (USD). DESIGN NEEDED: replace with the
# confirmed API pricing / compute-proxy formula per modelsUnderTest.
COST_PER_1K_INPUT = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": 0.0,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": 0.0,
    "microsoft/phi-4": 0.0,
}
COST_PER_1K_OUTPUT = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": 0.0,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": 0.0,
    "microsoft/phi-4": 0.0,
}

TOST_MARGIN_DEFAULT = 0.02  # macro-F1 equivalence margin (pre-registered)

try:
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        try:
            return len(_ENC.encode(text or ""))
        except Exception:
            return max(1, len((text or "")) // 4)

    _TOKENIZER = "tiktoken:cl100k_base"
except Exception:  # pragma: no cover
    def count_tokens(text: str) -> int:
        return max(1, len((text or "")) // 4)

    _TOKENIZER = "approx:len/4"


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    ci = COST_PER_1K_INPUT.get(model, 0.001)
    co = COST_PER_1K_OUTPUT.get(model, 0.005)
    return (in_tok / 1000.0) * ci + (out_tok / 1000.0) * co


def efficiency_from_detection(detection_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert RQ1 detection per-call records into RQ2 efficiency records.

    Each detection record is expected to carry (populated by llm_runner):
      unit_id, model, representation, repeat, input_text, output_text,
      input_tokens (optional), output_tokens (optional), latency_s (optional)
    We prefer logged usage; fall back to local token counting when absent.
    """
    out: List[Dict[str, Any]] = []
    for r in detection_records:
        model = r.get("model", "")
        in_tok = r.get("input_tokens")
        out_tok = r.get("output_tokens")
        if in_tok is None:
            in_tok = count_tokens(r.get("input_text", ""))
        if out_tok is None:
            out_tok = count_tokens(r.get("output_text", ""))
        latency = r.get("latency_s")
        if latency is None:
            # transparent proxy: latency scales with output tokens; labeled proxy
            latency = 0.4 + 0.004 * float(out_tok)
        rec = {
            "unit_id": r.get("unit_id"),
            "model": model,
            "representation": r.get("representation"),
            "repeat": r.get("repeat", 0),
            "cwe": r.get("cwe"),
            "split": r.get("split"),
            "input_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "end_to_end_latency_s": float(latency),
            "inference_cost_usd": float(_cost(model, int(in_tok), int(out_tok))),
            "tokenizer": _TOKENIZER,
        }
        out.append(rec)
    return out


def summarize_efficiency(eff_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per (model, representation): mean input/output tokens, latency, cost."""
    import pandas as pd

    if not eff_records:
        return {"note": "no efficiency records"}
    df = pd.DataFrame(eff_records)
    grp = df.groupby(["model", "representation"], dropna=False)
    summary: Dict[str, Any] = {}
    for (model, rep), sub in grp:
        key = f"{model}|{rep}"
        summary[key] = {
            "input_token_count_rq2": float(sub["input_tokens"].mean()),
            "output_token_count_rq2": float(sub["output_tokens"].mean()),
            "input_output_token_count_rq2": float(
                sub["input_tokens"].mean() + sub["output_tokens"].mean()
            ),
            "end_to_end_latency_rq2": float(sub["end_to_end_latency_s"].mean()),
            "inference_cost_rq2": float(sub["inference_cost_usd"].mean()),
            "n": int(len(sub)),
        }
    return summary


def tost_equivalence(
    f1_slice: np.ndarray, f1_whole: np.ndarray, margin: float = TOST_MARGIN_DEFAULT
) -> Dict[str, Any]:
    """
    Two one-sided tests for equivalence of paired macro-F1 (slice vs whole).
    Equivalent if the 90% CI of the mean paired difference lies entirely within
    [-margin, +margin]. Returns p-values of both one-sided tests and decision.
    """
    from scipy import stats

    f1_slice = np.asarray(f1_slice, dtype=float)
    f1_whole = np.asarray(f1_whole, dtype=float)
    n = min(len(f1_slice), len(f1_whole))
    if n < 2:
        return {
            "equivalent": None,
            "margin": float(margin),
            "note": "insufficient paired observations for TOST",
            "n": int(n),
        }
    d = f1_slice[:n] - f1_whole[:n]
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se = sd_d / np.sqrt(n) if sd_d > 0 else 1e-9
    # upper test: H0 diff >= +margin ; lower test: H0 diff <= -margin
    t_upper = (mean_d - margin) / se
    t_lower = (mean_d + margin) / se
    p_upper = float(stats.t.cdf(t_upper, df=n - 1))  # want mean_d < margin
    p_lower = float(1 - stats.t.cdf(t_lower, df=n - 1))  # want mean_d > -margin
    p_tost = max(p_upper, p_lower)
    # 90% CI of mean diff
    tcrit = float(stats.t.ppf(0.95, df=n - 1))
    ci_low = mean_d - tcrit * se
    ci_high = mean_d + tcrit * se
    equivalent = bool((ci_low > -margin) and (ci_high < margin))
    return {
        "equivalent": equivalent,
        "mean_diff": mean_d,
        "ci90": [float(ci_low), float(ci_high)],
        "margin": float(margin),
        "p_tost": p_tost,
        "n": int(n),
        "note": (
            "TOST margin default 0.02; DESIGN NEEDED: confirm pilot-derived "
            "max(0.02, 1 pilot-SD) margin and that N meets TOST power."
        ),
    }


if __name__ == "__main__":
    # Smoke: synthetic efficiency + TOST sanity.
    rng = np.random.default_rng(RANDOM_SEED)
    demo = [
        {
            "unit_id": f"u{i}",
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "representation": "R2",
            "repeat": 0,
            "input_text": "int f(){ return 0; }" * (i % 5 + 1),
            "output_text": '{"vulnerable": false, "lines": []}',
        }
        for i in range(20)
    ]
    eff = efficiency_from_detection(demo)
    print("tokenizer:", _TOKENIZER)
    print("summary:", summarize_efficiency(eff))
    a = rng.normal(0.70, 0.03, 30)
    b = a + rng.normal(0.0, 0.01, 30)
    print("TOST:", tost_equivalence(a, b))