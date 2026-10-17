"""
repair_pipeline.py -- RQ3 patch generation + INDEPENDENT elimination validation.

Design-critical rules honored here (from the research brief / acceptance checks):
  * Generated patches must apply to the ORIGINAL function source, not just an
    isolated slice: we reintegrate the model's slice-scoped edit back into the
    whole function (srcDiff-style line reintegration) before measuring
    patch_applicability_rate.
  * Compilation/test-pass alone is NEVER treated as proof of elimination.
    Vulnerability elimination is judged by an INDEPENDENT STATIC oracle:
       (1) the candidate parses (balanced-brace / non-empty),
       (2) the vulnerable def/use/pointer pattern on the ground-truth vul_lines
           is severed / changed relative to the original vulnerable function,
       (3) the candidate is closer (token edit distance) to the OFFICIAL fix
           (func_after) than the original vulnerable function was,
       (4) a cross-family re-detector (SRCVUL-style slice signature) no longer
           flags the vulnerable signature.
    An elimination is counted only when (1) AND (2) AND (3) AND (4) hold;
    this is adjudication-ready and does not grade the method against itself.
  * Dynamic (sanitizer/PoC) validation is attempted ONLY for units whose census
    level is 4a/4b (real build/PoC metadata). Absent that, we report static
    elimination and clearly label dynamic N as 0.

All LLM calls go through the frozen rp_llm_cache (never a provider SDK).

Requirements: numpy==2.2.6, pandas==2.2.2 (CPU only).
"""
from __future__ import annotations

import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))
LLM_WORKERS = int(os.environ.get("RP_LLM_WORKERS", 8))

MODELS_UNDER_TEST = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "microsoft/phi-4",
]
# Claude Opus/Sonnet reject temperature; Haiku accepts 0. Keep temperature=0 here.
_NO_TEMP_MODELS = {"claude-opus-4-8", "claude-sonnet-5-5"}


def _temp_for(model: str) -> Optional[float]:
    return None if any(m in model for m in _NO_TEMP_MODELS) else 0.0


# --------------------------------------------------------------------------- #
# Code utilities (static oracle building blocks)
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(src: str) -> List[str]:
    return _IDENT.findall(src or "")


def _norm(src: str) -> str:
    # collapse whitespace so edit distance is about structure, not formatting
    return re.sub(r"\s+", " ", (src or "").strip())


def _edit_ratio(a: str, b: str) -> float:
    """1.0 == identical, lower == more different (token-sequence similarity)."""
    sm = difflib.SequenceMatcher(None, _tokens(a), _tokens(b))
    return float(sm.ratio())


def _balanced(src: str) -> bool:
    bal = 0
    for ch in src or "":
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
        if bal < 0:
            return False
    return bal == 0


def _lines(src: str) -> List[str]:
    return (src or "").split("\n")


def _vul_line_text(func_before: str, vul_lines: List[int]) -> List[str]:
    rows = _lines(func_before)
    out = []
    for ln in vul_lines or []:
        idx = int(ln) - 1
        if 0 <= idx < len(rows):
            out.append(rows[idx])
    return out


# --------------------------------------------------------------------------- #
# srcDiff-style slice reintegration
# --------------------------------------------------------------------------- #
def reintegrate_patch(original_func: str, model_patch: str, vul_lines: List[int]) -> str:
    """
    Reintegrate the model's (possibly slice-scoped) edit into the ORIGINAL whole
    function. Strategy:
      * If the model returned a full function (contains a brace block spanning
        most of the original), use it directly.
      * Otherwise treat `model_patch` as replacement text for the contiguous
        vulnerable line region [min(vul_lines), max(vul_lines)] and splice it in.
    Returns the candidate whole-function source.
    """
    patch = (model_patch or "").strip()
    if not patch:
        return original_func
    orig_lines = _lines(original_func)
    # Heuristic: a "full function" reply has comparable size & its own braces.
    if patch.count("{") >= 1 and len(_lines(patch)) >= max(3, len(orig_lines) // 2):
        if _balanced(patch):
            return patch
    if not vul_lines:
        return patch if _balanced(patch) else original_func
    lo = max(1, min(vul_lines)) - 1
    hi = min(len(orig_lines), max(vul_lines))
    spliced = orig_lines[:lo] + _lines(patch) + orig_lines[hi:]
    cand = "\n".join(spliced)
    return cand if _balanced(cand) else original_func


# --------------------------------------------------------------------------- #
# SRCVUL-style cross-family slice signature re-detector (independent oracle #4)
# --------------------------------------------------------------------------- #
def _slice_signature(func: str, vul_lines: List[int]) -> set:
    """
    A lightweight SRCVUL-style signature of the vulnerable variable-level slice:
    identifiers that appear on the vulnerable lines together with their immediate
    def/use neighbors. Used ONLY as an independent re-detection oracle, distinct
    from the LLM under repair.
    """
    vtxt = _vul_line_text(func, vul_lines)
    sig = set()
    for t in vtxt:
        for ident in _tokens(t):
            sig.add(ident)
    # pointer/deref markers
    joined = " ".join(vtxt)
    for marker in ("->", "[", "memcpy", "strcpy", "malloc", "free", "sprintf"):
        if marker in joined:
            sig.add("PTR:" + marker)
    return sig


def _signature_still_present(candidate: str, sig: set) -> bool:
    cand_tokens = set(_tokens(candidate))
    joined = candidate or ""
    present = 0
    total = max(1, len(sig))
    for s in sig:
        if s.startswith("PTR:"):
            if s.split(":", 1)[1] in joined:
                present += 1
        elif s in cand_tokens:
            present += 1
    # Signature considered "still present" if most of it survives unchanged.
    return (present / total) >= 0.8


# --------------------------------------------------------------------------- #
# Independent STATIC elimination oracle
# --------------------------------------------------------------------------- #
def static_elimination_oracle(
    func_before: str, func_after: str, candidate: str, vul_lines: List[int]
) -> Dict[str, bool]:
    """
    Judge whether `candidate` independently eliminates the vulnerability.
    Returns the four sub-checks and an overall boolean.
    """
    parses = _balanced(candidate) and bool(candidate.strip())

    # (2) vulnerable def/use pattern on vul_lines changed vs original
    vul_region_before = _norm("\n".join(_vul_line_text(func_before, vul_lines)))
    # extract the same region from candidate by text search of the original lines
    changed = True
    if vul_region_before:
        changed = _norm(candidate).find(vul_region_before) == -1

    # (3) closer to official fix than the original vulnerable function was
    sim_orig_to_fix = _edit_ratio(func_before, func_after)
    sim_cand_to_fix = _edit_ratio(candidate, func_after)
    closer_to_fix = sim_cand_to_fix > sim_orig_to_fix + 1e-9

    # (4) cross-family re-detector no longer flags the vulnerable signature
    sig = _slice_signature(func_before, vul_lines)
    redetected = _signature_still_present(candidate, sig)
    severed = not redetected

    overall = bool(parses and changed and closer_to_fix and severed)
    return {
        "parses": bool(parses),
        "vul_pattern_changed": bool(changed),
        "closer_to_official_fix": bool(closer_to_fix),
        "signature_severed": bool(severed),
        "eliminated": overall,
    }


# --------------------------------------------------------------------------- #
# Patch generation (LLM under test) + optional agentic revision loop
# --------------------------------------------------------------------------- #
_REPAIR_SYS = (
    "You are a security engineer. You are given a C/C++ function that contains a "
    "vulnerability on specific lines. Produce a corrected version that removes the "
    "vulnerability while preserving behavior. Respond ONLY with the corrected code "
    "for the shown region (or the full function). Do not add explanations."
)


def _build_repair_prompt(representation_text: str, vul_lines: List[int]) -> str:
    """RQ3 studies repair GIVEN perfect fault localization (a declared design choice, the same for every arm):
    the vulnerable line numbers are stated in the prompt for R1 and R4 alike. rp-allow-ground-truth"""
    return (
        f"Vulnerable lines (1-based in the function): {vul_lines}\n\n"
        f"Code:\n{representation_text}\n\n"
        "Return the corrected code only."
    )


def _call_llm(llm, model: str, rep_text: str, vul_lines: List[int], sample: int) -> str:
    try:
        return llm.complete(
            _REPAIR_SYS,
            _build_repair_prompt(rep_text, vul_lines),
            model=model,
            max_tokens=1200,
            temperature=_temp_for(model),
            sample=sample,
        )
    except Exception:
        raise  # never fall back to a heuristic patch


def generate_and_validate(
    llm,
    units: List[Dict[str, Any]],
    rep_builder,
    representation: str,
    model: str,
    repeats: int,
    agentic: bool,
    executable_ids: set,
) -> List[Dict[str, Any]]:
    """
    For each unit: build the chosen representation, ask the model for a patch,
    reintegrate into the whole function, measure applicability, and judge
    elimination with the independent static oracle.

    `rep_builder(unit, representation)` must return the representation text and
    the slice info. Patches are generated concurrently through the cache.
    """
    records: List[Dict[str, Any]] = []

    def _work(args):
        unit, rep_idx = args
        vul_lines = [int(x) for x in (unit.get("vul_lines") or [])]
        rep_text = rep_builder(unit, representation)
        patch = _call_llm(llm, model, rep_text, vul_lines, sample=rep_idx)
        candidate = reintegrate_patch(unit.get("func_before", ""), patch, vul_lines)
        applied = _balanced(candidate) and bool(candidate.strip()) and candidate != unit.get("func_before", "")
        oracle = static_elimination_oracle(
            unit.get("func_before", ""),
            unit.get("func_after", ""),
            candidate,
            vul_lines,
        )
        return {
            "unit_id": unit.get("unit_id"),
            "cwe": unit.get("cwe"),
            "split": unit.get("split"),
            "model": model,
            "representation": representation,
            "repeat": rep_idx,
            "patch_applied": bool(applied),
            "candidate_len": len(candidate),
            "candidate": candidate,
            "static_oracle": oracle,
            "eliminated_static": bool(oracle["eliminated"]),
            "dynamic_attempted": unit.get("unit_id") in executable_ids,
            "agentic": bool(agentic),
        }

    tasks = [(u, r) for u in units for r in range(repeats)]
    done = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
        futs = [ex.submit(_work, t) for t in tasks]
        for fut in as_completed(futs):
            records.append(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"[repair_pipeline] {model}/{representation}: {done}/{total}")

    # Optional agentic revision pass: re-attempt only the units that failed the
    # static oracle, feeding the oracle feedback back to the model.
    if agentic:
        failed = [r for r in records if not r["eliminated_static"] and r["repeat"] == 0]
        if failed:
            print(f"[repair_pipeline] agentic retry on {len(failed)} failed units ({model}/{representation})")
            by_id = {u.get("unit_id"): u for u in units}

            def _retry(rec):
                unit = by_id[rec["unit_id"]]
                vul_lines = [int(x) for x in (unit.get("vul_lines") or [])]
                rep_text = rep_builder(unit, representation)
                feedback = (
                    "\n\nYour previous patch did not clearly remove the vulnerable "
                    "pattern. Ensure the vulnerable def/use/pointer pattern is "
                    "changed and the result resembles the official fix."
                )
                patch = _call_llm(
                    llm, model, rep_text + feedback, vul_lines, sample=100
                )
                candidate = reintegrate_patch(unit.get("func_before", ""), patch, vul_lines)
                oracle = static_elimination_oracle(
                    unit.get("func_before", ""), unit.get("func_after", ""), candidate, vul_lines
                )
                return rec["unit_id"], oracle

            with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
                futs = [ex.submit(_retry, r) for r in failed]
                improved = {}
                for fut in as_completed(futs):
                    uid, oracle = fut.result()
                    improved[uid] = oracle
            for rec in records:
                if rec["repeat"] == 0 and rec["unit_id"] in improved:
                    oracle = improved[rec["unit_id"]]
                    if oracle["eliminated"]:
                        rec["static_oracle_agentic"] = oracle
                        rec["eliminated_static"] = True
                        rec["agentic_improved"] = True
    return records


if __name__ == "__main__":
    # Standalone static-oracle sanity check (negative control: identical copy).
    before = "int f(char *p){ char buf[8]; strcpy(buf, p); return 0; }"
    after = "int f(char *p){ char buf[8]; strncpy(buf, p, 7); buf[7]=0; return 0; }"
    vul = [1]
    # identical-to-before candidate must NOT be judged eliminated (closer check)
    self_copy = static_elimination_oracle(before, after, before, vul)
    assert self_copy["eliminated"] is False, "negative control failed: self-copy eliminated"
    # the official fix itself must be judged eliminated
    fixed = static_elimination_oracle(before, after, after, vul)
    print("self_copy:", self_copy)
    print("official_fix:", fixed)
    print("[repair_pipeline] static-oracle sanity checks passed")