"""build_lora_data.py -- training examples for the fine-tuned baseline (RQ1-E2, E-D).

Source: data/primevul_train_lora.jsonl (PrimeVul paired train pairs, disjoint by CVE, commit and function from
every evaluation file). Each function becomes two chat examples, one with the whole function (R1) and one with
the srcSlice slice + dependency metadata (R4), using EXACTLY the RQ1 system prompt and user template, so the
adapter is evaluated in the same format it was trained on. Target = the JSON verdict: vulnerable functions cite
the fix-diff lines; fixed functions are safe with no lines. Examples longer than MAX_CHARS are skipped (reported).
Writes data/lora_train.jsonl ({"messages": [...], "rep": "R1"|"R4", "label": 0|1}).
"""
import json
import random
from pathlib import Path

import llm_runner as L
from data_prep import _pair_to_units
from experiment_rq1 import _criterion_vars
from slice_pipeline import build_representations, compute_slice, slice_backend_report

ROOT = Path(__file__).resolve().parent
MAX_CHARS = 12000  # about 3-4k tokens: keeps training sequences short on a 48 GB GPU


def main() -> None:
    rows = [json.loads(l) for l in open(ROOT / "data" / "primevul_train_lora.jsonl")]
    out, skipped = [], 0
    for row in rows:
        for u in _pair_to_units(row, "primevul_train_lora"):
            reps = build_representations(u.code, compute_slice(u.code, criterion_vars=_criterion_vars(u)))
            target = json.dumps({"vulnerable": bool(u.label), "cited_lines": list(u.vul_lines) if u.label else [],
                                 "cwe": (u.cwe if u.label else "unknown") or "unknown"})
            for rep in ("R1", "R4"):
                user = L.USER_TEMPLATE.format(context=reps[rep])
                if len(user) > MAX_CHARS:
                    skipped += 1
                    continue
                out.append({"messages": [{"role": "system", "content": L.SYSTEM_PROMPT},
                                         {"role": "user", "content": user},
                                         {"role": "assistant", "content": target}],
                            "rep": rep, "label": u.label, "unit_id": u.unit_id})
    random.Random(42).shuffle(out)
    with open(ROOT / "data" / "lora_train.jsonl", "w") as fh:
        for ex in out:
            fh.write(json.dumps(ex) + "\n")
    print(f"lora_train.jsonl: {len(out)} examples ({sum(e['label'] for e in out)} vulnerable), skipped {skipped} "
          f"over {MAX_CHARS} chars; slicing {slice_backend_report()}")


if __name__ == "__main__":
    main()
