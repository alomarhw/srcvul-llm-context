# LoRA fine-tuned baseline

This is the fine-tuned baseline of the article (Appendix H): a LoRA adapter for **Qwen2.5-Coder-7B-Instruct** (revision `c03e6d35`). It was trained with label-free seeding on `data/lora_train.jsonl`: 10,869 examples over R1 and R4, built from 3,051 PrimeVul training pairs that are disjoint from every evaluation file. After truncation at 4,096 tokens, 10,686 examples remained.

| Setting | Value |
|---|---|
| LoRA | r = 16, alpha = 32, dropout 0.05; q/k/v/o and gate/up/down projections |
| Training | 1 epoch, learning rate 2e-4, micro-batch 2 x gradient accumulation 8, warmup 3%, bf16, seed 42 |
| Size | 10,686 examples, 667 optimizer steps, 203 minutes on one RTX 6000 Ada |
| Software | PyTorch 2.6.0 (cu124), transformers 5.19.0, peft 0.21.2 |

## Files

- **`lora_adapter/`**: the adapter, with the base model's tokenizer files and chat template. The weights are split in two parts because of GitHub's file-size limit. `make unpack` joins them into `adapter_model.safetensors` and checks its SHA-256.
- **`training/train_config.json`** and **`training/train.log`**: the exact configuration and the per-step training log, written by `src/train_lora.py`.

## Use

- **Loading the adapter:** serve the base model with vLLM using `--enable-lora --lora-modules qwen7b-primevul-lora-blind=model/lora_adapter`, or load it with `peft.PeftModel.from_pretrained`.
- **Scoring:** `src/ft_eval.py` scores greedy outputs, and `src/ft_logprob_rank.py` computes the label-token ranking.
- **Replay without a GPU:** the scores used in the article are frozen in `data/ft_logprob_scores.jsonl`, so `python src/ft_logprob_rank.py --replay` reproduces the analysis.

The base model and its tokenizer are distributed by Qwen under the Apache 2.0 license.
