"""train_lora.py -- fine-tuned baseline for #DF89E (RQ1-E2, E-D). Runs on the GPU server.

LoRA on Qwen2.5-Coder-7B-Instruct, trained on data/lora_train.jsonl (built by build_lora_data.py from PrimeVul
train pairs disjoint from every evaluation set). Loss on the assistant (JSON verdict) tokens only. All
hyperparameters are fixed here and written to <out>/train_config.json for the paper.

Usage: python train_lora.py lora_train.jsonl OUT_DIR
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

CFG = {
    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lr": 2e-4, "epochs": 1, "micro_batch": 2, "grad_accum": 8, "max_len": 4096, "warmup_ratio": 0.03,
    "weight_decay": 0.0, "seed": 42, "dtype": "bfloat16",
}


def encode(tok, ex):
    prompt = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True)
    full = prompt + ex["messages"][2]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:CFG["max_len"]]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels[:len(ids)]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    tok = AutoTokenizer.from_pretrained(CFG["base_model"])
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                                             lora_dropout=CFG["lora_dropout"], target_modules=CFG["target_modules"],
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    data = [encode(tok, json.loads(l)) for l in open(src)]
    data = [d for d in data if any(x != -100 for x in d[1])]
    steps = math.ceil(len(data) / (CFG["micro_batch"] * CFG["grad_accum"])) * CFG["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    warm = max(1, int(CFG["warmup_ratio"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    model.train(); step = 0; t0 = time.time(); log = []
    for epoch in range(CFG["epochs"]):
        random.shuffle(data)
        for i in range(0, len(data), CFG["micro_batch"]):
            batch = data[i:i + CFG["micro_batch"]]
            L = max(len(x[0]) for x in batch)
            ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch], device="cuda")
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device="cuda")
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch], device="cuda")
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / CFG["grad_accum"]
            loss.backward()
            if (i // CFG["micro_batch"] + 1) % CFG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 20 == 0 or step == steps:
                    rec = {"step": step, "of": steps, "loss": round(loss.item() * CFG["grad_accum"], 4),
                           "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
                    log.append(rec); print(json.dumps(rec), flush=True)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "train_config.json").write_text(json.dumps({**CFG, "n_examples": len(data), "optimizer_steps": step,
                                                       "train_minutes": round((time.time() - t0) / 60, 1),
                                                       "log": log}, indent=1))
    print("saved adapter to", out / "adapter")


if __name__ == "__main__":
    main()
