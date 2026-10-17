"""ft_eval.py -- evaluate the fine-tuned LoRA baseline (RQ1-E2, E-D) on all paper-grade RQ1 units, R1 and R4.

The adapter must be served by vLLM under the name given in RP_FINETUNED_MODEL (routed through
RP_LOCAL_LLM_BASE_URL). Answers go into the frozen cache; afterwards the marker data/finetuned_model.json (and the
same file in every DATA_MARKER_DIRS entry) lets the final replay include the baseline without the server.
Usage: RP_FINETUNED_MODEL=local:qwen7b-primevul-lora python ft_eval.py [extra marker dirs...]
"""
import json
import os
import sys
from pathlib import Path

import addons
import data_prep
from rp_llm_cache import LLMCache

ROOT = Path(__file__).resolve().parent


def main() -> None:
    model = os.environ["RP_FINETUNED_MODEL"]
    units, _ = data_prep.load_units(max_rows=None, allow_synthetic=False, strict=True)
    out = addons.ed_finetuned(units, model, LLMCache(), int(os.environ.get("RP_LLM_WORKERS", 32)))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "finetuned_eval.json").write_text(json.dumps(addons._san(out), indent=1))
    for d in [ROOT / "data", *map(Path, sys.argv[1:])]:
        (d / "finetuned_model.json").write_text(json.dumps({"model": model}))
    print(json.dumps({r: {k: v[k] for k in ("macro_f1", "fpr", "pair_correct", "vp_s", "line_recall")}
                      for r, v in out["results"].items()}))


if __name__ == "__main__":
    main()
