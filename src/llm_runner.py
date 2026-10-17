"""llm_runner.py - frozen-cache LLM querying for detection + localization.

Requirements (pinned): numpy==2.2.6, tiktoken==0.14.0 (optional, for token counts)

Every model call goes through rp_llm_cache.LLMCache (thread-safe, frozen,
replayable). Models under test (EXACT ids, this order):
    Qwen/Qwen2.5-Coder-14B-Instruct, Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, microsoft/phi-4
    (open weights, self-hosted with vLLM at RP_LOCAL_LLM_BASE_URL)

temperature handling: Claude Opus/Sonnet reject temperature; the three models
here accept temperature=0.0. We set temperature=0.0 for all three.

Output is parsed from a strict JSON schema the prompt requests:
    {"vulnerable": true|false, "cited_lines": [<int>, ...], "cwe": "<str>"}
If a call fails, the error SURFACES (no heuristic fallback).
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rp_llm_cache import LLMCache

MODELS_UNDER_TEST = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "microsoft/phi-4",
]

MAX_WORKERS = int(os.environ.get("RP_LLM_WORKERS", 8))

SYSTEM_PROMPT = (
    "You are a precise C/C++ security code reviewer. You classify a code context "
    "as vulnerable or not and cite the exact 1-based line numbers (as shown in the "
    "numbered context) most responsible for any vulnerability. Respond with STRICT "
    "JSON only, no prose."
)

USER_TEMPLATE = (
    "Analyze the following code context and decide if it contains a software "
    "security vulnerability.\n\n"
    "Return STRICT JSON exactly in this schema:\n"
    '{{"vulnerable": true or false, "cited_lines": [list of 1-based line numbers], '
    '"cwe": "CWE-id or unknown"}}\n\n'
    "Code context (lines are prefixed with their 1-based line number):\n"
    "```c\n{context}\n```\n"
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENC = _try_tokenizer()


def _count_tokens(text: str) -> int:
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _first_verdict_object(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "vulnerable" in obj:
                return obj
    return None


def parse_response(text: str) -> dict[str, Any]:
    """Parse the model's JSON. Raises nothing; returns a well-formed dict.

    A malformed response is treated as a NON-detection with no cited lines,
    which is a legitimate (if wrong) model output, not a heuristic substitute.
    """
    m = _JSON_RE.search(text or "")
    if not m:
        return {"vulnerable": False, "cited_lines": [], "cwe": "unknown", "parsed": False}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Models sometimes add prose (with braces) after the JSON: take the first decodable JSON object
        # that carries the verdict.
        obj = _first_verdict_object(text)
        if obj is None:
            return {"vulnerable": False, "cited_lines": [], "cwe": "unknown", "parsed": False}
    vul = bool(obj.get("vulnerable", False))
    cited = obj.get("cited_lines", []) or []
    try:
        cited = [int(x) for x in cited]
    except Exception:
        cited = []
    return {"vulnerable": vul, "cited_lines": cited,
            "cwe": str(obj.get("cwe", "unknown")), "parsed": True}


class LLMRunner:
    def __init__(self):
        self.llm = LLMCache()

    def _one_call(self, model: str, context: str, sample: int) -> dict[str, Any]:
        user = USER_TEMPLATE.format(context=context)
        in_tokens = _count_tokens(SYSTEM_PROMPT + user)
        t0 = time.time()
        text = self.llm.complete(
            SYSTEM_PROMPT, user, model=model,
            max_tokens=256, temperature=0.0, sample=sample,
        )
        latency = time.time() - t0
        out_tokens = _count_tokens(text or "")
        res = parse_response(text)
        res.update({
            "model": model, "input_tokens": in_tokens, "output_tokens": out_tokens,
            "latency_s": latency, "raw": text,
        })
        return res

    def run_batch(self, jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """jobs: list of {"key": str, "model": str, "context": str, "sample": int}.
        Returns key -> result dict. Concurrent, resumes free from cache.
        """
        results: dict[str, dict[str, Any]] = {}
        total = len(jobs)
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut2key = {
                ex.submit(self._one_call, j["model"], j["context"], j.get("sample", 0)): j["key"]
                for j in jobs
            }
            for fut in as_completed(fut2key):
                key = fut2key[fut]
                results[key] = fut.result()  # let errors surface
                done += 1
                if done % 200 == 0 or done == total:
                    print(f"[llm] {done}/{total} calls complete")
        return results

    def stats(self) -> dict[str, Any]:
        try:
            return self.llm.stats()
        except Exception:
            return {}


if __name__ == "__main__":
    print("models under test:", MODELS_UNDER_TEST)
    print("parse test:", parse_response('{"vulnerable": true, "cited_lines": [3], "cwe":"CWE-119"}'))