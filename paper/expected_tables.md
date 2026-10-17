# #DF89E paper tables (generated from results/results.json by paper_tables_df89e.py)

## T1 Detection per model and representation (3,670 units, 1,835 pairs)

| Model | Input | Macro-F1 | Recall | FPR | Pair-correct | VP-S | Both vuln. / both safe | Parse fail |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 0.508 | 0.474 | 0.457 | 127 (6.9%) | 30 | 40% / 47% | 0.005 |
| Qwen2.5-Coder-14B | R2 raw slice | 0.482 | 0.660 | 0.669 | 88 (4.8%) | -16 | 61% / 28% | 0.010 |
| Qwen2.5-Coder-14B | R3 slice + context | 0.511 | 0.539 | 0.517 | 142 (7.7%) | 39 | 46% / 41% | 0.009 |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 0.463 | 0.737 | 0.747 | 85 (4.6%) | -19 | 69% / 21% | 0.008 |
| Qwen2.5-Coder-14B | R5 slice + retrieved examples | 0.494 | 0.560 | 0.568 | 109 (5.9%) | -18 | 50% / 37% | 0.019 |
| Qwen3-Coder-30B | R1 whole function | 0.409 | 0.095 | 0.070 | 65 (3.5%) | 44 | 6% / 89% | 0.000 |
| Qwen3-Coder-30B | R2 raw slice | 0.404 | 0.095 | 0.089 | 51 (2.8%) | 15 | 7% / 88% | 0.000 |
| Qwen3-Coder-30B | R3 slice + context | 0.406 | 0.093 | 0.076 | 64 (3.5%) | 31 | 6% / 89% | 0.000 |
| Qwen3-Coder-30B | R4 SRCVUL slice | 0.408 | 0.097 | 0.081 | 67 (3.7%) | 30 | 6% / 88% | 0.000 |
| Qwen3-Coder-30B | R5 slice + retrieved examples | 0.483 | 0.292 | 0.278 | 140 (7.6%) | 25 | 22% / 65% | 0.035 |
| Phi-4 | R1 whole function | 0.442 | 0.838 | 0.829 | 76 (4.1%) | 15 | 80% / 13% | 0.004 |
| Phi-4 | R2 raw slice | 0.421 | 0.875 | 0.872 | 46 (2.5%) | 5 | 85% / 10% | 0.001 |
| Phi-4 | R3 slice + context | 0.438 | 0.841 | 0.836 | 66 (3.6%) | 9 | 80% / 13% | 0.002 |
| Phi-4 | R4 SRCVUL slice | 0.379 | 0.941 | 0.942 | 23 (1.3%) | -2 | 93% / 5% | 0.002 |
| Phi-4 | R5 slice + retrieved examples | 0.397 | 0.911 | 0.913 | 63 (3.4%) | -2 | 88% / 5% | 0.077 |

## T2 Localization per model and representation (1,433 vulnerable units with labelled lines)

| Model | Input | Top-1 | Top-5 | Line precision | Line recall |
|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 0.133 | 0.174 | 0.201 | 0.060 |
| Qwen2.5-Coder-14B | R2 raw slice | 0.145 | 0.206 | 0.151 | 0.077 |
| Qwen2.5-Coder-14B | R3 slice + context | 0.137 | 0.177 | 0.158 | 0.055 |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 0.153 | 0.223 | 0.150 | 0.083 |
| Qwen2.5-Coder-14B | R5 slice + retrieved examples | 0.121 | 0.191 | 0.158 | 0.091 |
| Qwen3-Coder-30B | R1 whole function | 0.038 | 0.054 | 0.208 | 0.018 |
| Qwen3-Coder-30B | R2 raw slice | 0.031 | 0.049 | 0.197 | 0.018 |
| Qwen3-Coder-30B | R3 slice + context | 0.037 | 0.049 | 0.180 | 0.016 |
| Qwen3-Coder-30B | R4 SRCVUL slice | 0.029 | 0.051 | 0.217 | 0.018 |
| Qwen3-Coder-30B | R5 slice + retrieved examples | 0.061 | 0.098 | 0.157 | 0.040 |
| Phi-4 | R1 whole function | 0.188 | 0.263 | 0.156 | 0.111 |
| Phi-4 | R2 raw slice | 0.207 | 0.297 | 0.167 | 0.115 |
| Phi-4 | R3 slice + context | 0.186 | 0.264 | 0.162 | 0.104 |
| Phi-4 | R4 SRCVUL slice | 0.211 | 0.334 | 0.146 | 0.148 |
| Phi-4 | R5 slice + retrieved examples | 0.176 | 0.242 | 0.143 | 0.108 |

## T3 Representation tests (pooled over models; paired Wilcoxon, Holm)

| Comparison | Cohen's d | 95% CI (diff. in correct) | p | Holm p |
|---|---|---|---|---|
| R1 vs R2 | -0.023 | [-0.031, 0.006] | 0.164 | 1.000 |
| R1 vs R3 | 0.001 | [-0.011, 0.014] | 0.934 | 1.000 |
| R1 vs R4 | -0.024 | [-0.035, 0.005] | 0.151 | 1.000 |
| R1 vs R5 | -0.017 | [-0.031, 0.009] | 0.315 | 1.000 |
| R2 vs R3 | 0.027 | [-0.003, 0.031] | 0.106 | 0.953 |
| R2 vs R4 | -0.004 | [-0.014, 0.010] | 0.792 | 1.000 |
| R2 vs R5 | 0.004 | [-0.018, 0.022] | 0.811 | 1.000 |
| R3 vs R4 | -0.028 | [-0.033, 0.002] | 0.095 | 0.949 |
| R3 vs R5 | -0.018 | [-0.032, 0.009] | 0.286 | 1.000 |
| R4 vs R5 | 0.007 | [-0.016, 0.024] | 0.690 | 1.000 |

| Model | R4 vs best non-metadata: effect | 95% CI | Holm p |
|---|---|---|---|
| Qwen2.5-Coder-14B | -0.029 | [-0.034, 0.002] | 0.733 |
| Qwen3-Coder-30B | -0.013 | [-0.014, 0.006] | 1.000 |
| Phi-4 | -0.012 | [-0.018, 0.008] | 1.000 |

## T4 Baselines (pooled over models; whole-file context is omitted because it equals R1 here)

| Baseline | Precision | Recall | Macro-F1 | FPR |
|---|---|---|---|---|
| Original SRCVUL (LSH variable-level slice matching, alomari2025a) — re | 0.499 | 0.500 | 0.499 | 0.501 |
| LLM with whole-function context (R1) | 0.512 | 0.471 | 0.510 | 0.450 |
| LLM with raw variable-level slice (R2) | 0.498 | 0.659 | 0.484 | 0.664 |
| LLM with slice+surrounding context (R3) | 0.511 | 0.534 | 0.511 | 0.511 |
| LLM with slice+retrieved vulnerability/patch examples (R5) | 0.500 | 0.586 | 0.496 | 0.586 |
| VulSlicer (salimi2022vulslicer) — statement/slice-based detector | 0.499 | 0.137 | 0.424 | 0.138 |
| Statement-level slicing granularity (changed-granularity ablation) | 0.499 | 0.188 | 0.446 | 0.189 |
| Random/majority-class detector | 0.501 | 0.490 | 0.501 | 0.488 |
| Fine-tuned LLM (LoRA on Qwen2.5-Coder-7B-Instruct, PrimeVul train spli | 0.000 | 0.000 | 0.333 | 0.000 |

## T2b Localization tests per model (exact McNemar on paired top-k hits vs R1, Holm over 12 tests)

| Model | Comparison | k | Hit rate R1 -> slice | Slice-only / whole-only hits | p | Holm p |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R2 vs R1 | top-1 | 0.133 -> 0.145 | +108 / -90 | 0.2269 | 0.7701 |
| Qwen2.5-Coder-14B | R2 vs R1 | top-5 | 0.174 -> 0.206 | +147 / -101 | 0.0042 | 0.0418 |
| Qwen2.5-Coder-14B | R4 vs R1 | top-1 | 0.133 -> 0.153 | +125 / -96 | 0.0594 | 0.4157 |
| Qwen2.5-Coder-14B | R4 vs R1 | top-5 | 0.174 -> 0.223 | +179 / -108 | 0.0000 | 0.0004 |
| Qwen3-Coder-30B | R2 vs R1 | top-1 | 0.038 -> 0.031 | +24 / -35 | 0.1925 | 0.7701 |
| Qwen3-Coder-30B | R2 vs R1 | top-5 | 0.054 -> 0.049 | +40 / -48 | 0.4557 | 0.9115 |
| Qwen3-Coder-30B | R4 vs R1 | top-1 | 0.038 -> 0.029 | +24 / -37 | 0.1237 | 0.6187 |
| Qwen3-Coder-30B | R4 vs R1 | top-5 | 0.054 -> 0.051 | +47 / -52 | 0.6879 | 0.9115 |
| Phi-4 | R2 vs R1 | top-1 | 0.188 -> 0.207 | +134 / -107 | 0.0938 | 0.5626 |
| Phi-4 | R2 vs R1 | top-5 | 0.263 -> 0.297 | +191 / -142 | 0.0084 | 0.0759 |
| Phi-4 | R4 vs R1 | top-1 | 0.188 -> 0.211 | +143 / -110 | 0.0440 | 0.3522 |
| Phi-4 | R4 vs R1 | top-5 | 0.263 -> 0.334 | +224 / -122 | 0.0000 | 0.0000 |

## T2c Localization against chance (random picks matched to the number of lines each model cited)

Labelled line visible in the input: R1 100%, R2 76%, R4 76%; mean lines shown: R1 65.1, R4 23.7.

| Model | Comparison | k | Lift over chance, R1 (pts) | Lift, slice (pts) | Difference [95% CI] | Holm p |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R2 - R1 | top-1 | +7.7 | +4.4 | -3.2 [-5.0, -1.4] | 0.004 |
| Qwen2.5-Coder-14B | R2 - R1 | top-5 | +8.0 | +3.4 | -4.6 [-6.6, -2.7] | <0.001 |
| Qwen2.5-Coder-14B | R4 - R1 | top-1 | +7.7 | +4.5 | -3.1 [-5.0, -1.2] | 0.006 |
| Qwen2.5-Coder-14B | R4 - R1 | top-5 | +8.0 | +3.7 | -4.3 [-6.3, -2.3] | <0.001 |
| Qwen3-Coder-30B | R2 - R1 | top-1 | +2.4 | +1.2 | -1.2 [-2.1, -0.3] | 0.038 |
| Qwen3-Coder-30B | R2 - R1 | top-5 | +2.5 | +1.0 | -1.6 [-2.6, -0.6] | 0.014 |
| Qwen3-Coder-30B | R4 - R1 | top-1 | +2.4 | +0.9 | -1.5 [-2.4, -0.5] | 0.014 |
| Qwen3-Coder-30B | R4 - R1 | top-5 | +2.5 | +0.9 | -1.6 [-2.7, -0.6] | 0.014 |
| Phi-4 | R2 - R1 | top-1 | +9.8 | +7.5 | -2.3 [-4.3, -0.4] | 0.038 |
| Phi-4 | R2 - R1 | top-5 | +9.0 | +6.1 | -2.9 [-5.1, -0.7] | 0.038 |
| Phi-4 | R4 - R1 | top-1 | +9.8 | +6.5 | -3.3 [-5.4, -1.3] | 0.005 |
| Phi-4 | R4 - R1 | top-5 | +9.0 | +6.2 | -2.8 [-5.1, -0.5] | 0.038 |

## T5 Official PrimeVul paired test set (435 pairs; RQ1-E3)

| Model | Input | Pair-correct (of 435) | VP-S | FPR | Macro-F1 |
|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 28 | 8 | 0.405 | 0.506 |
| Qwen2.5-Coder-14B | R2 raw slice | 19 | -10 | 0.637 | 0.480 |
| Qwen2.5-Coder-14B | R3 slice + context | 28 | 9 | 0.441 | 0.509 |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 18 | -3 | 0.729 | 0.470 |
| Qwen2.5-Coder-14B | R5 slice + retrieved examples | 33 | 14 | 0.395 | 0.512 |
| Qwen3-Coder-30B | R1 whole function | 10 | 5 | 0.067 | 0.395 |
| Qwen3-Coder-30B | R2 raw slice | 8 | 0 | 0.074 | 0.389 |
| Qwen3-Coder-30B | R3 slice + context | 13 | 5 | 0.067 | 0.395 |
| Qwen3-Coder-30B | R4 SRCVUL slice | 15 | 9 | 0.053 | 0.395 |
| Qwen3-Coder-30B | R5 slice + retrieved examples | 28 | 14 | 0.271 | 0.493 |
| Phi-4 | R1 whole function | 20 | 0 | 0.814 | 0.445 |
| Phi-4 | R2 raw slice | 16 | 8 | 0.853 | 0.435 |
| Phi-4 | R3 slice + context | 17 | 0 | 0.818 | 0.444 |
| Phi-4 | R4 SRCVUL slice | 9 | 3 | 0.894 | 0.410 |
| Phi-4 | R5 slice + retrieved examples | 12 | -1 | 0.816 | 0.444 |

Cited on the same split (other papers, not rerun): Aegis (DeepSeek-V3.1, arXiv 2603.20637): 122; VulTrial (GPT-4o, trained): 96; ReVD (Qwen2.5-Coder-7B, fine-tuned): 78; CoT (GPT-4o, Ding et al.): 40

## T6 Prompt robustness: zero-shot vs chain-of-thought (nondeterminism subset; RQ1-E4)

| Model | Input | F1 zero-shot | F1 CoT | Pair-correct zero-shot | Pair-correct CoT | CoT answers with reasoning |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 0.442 | 0.437 | 1 | 1 | 0% |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 0.489 | 0.507 | 1 | 1 | 0% |
| Qwen3-Coder-30B | R1 whole function | 0.374 | 0.498 | 0 | 1 | 77% |
| Qwen3-Coder-30B | R4 SRCVUL slice | 0.390 | 0.510 | 1 | 3 | 77% |
| Phi-4 | R1 whole function | 0.404 | 0.429 | 1 | 1 | 62% |
| Phi-4 | R4 SRCVUL slice | 0.368 | 0.376 | 0 | 1 | 62% |

The 200-unit subset samples units, not pairs, so few complete pairs are in it; pair-correct counts here are not comparable to T1.

## T7 Cross-function context probe (SecVulEval; RQ1-E5)

| Model | Input | Macro-F1 | Pair-correct | Top-1 | Line recall |
|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 0.502 | 32 | 0.132 | 0.106 |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 0.473 | 27 | 0.132 | 0.136 |
| Qwen2.5-Coder-14B | R6 slice + cross-function names | 0.468 | 27 | 0.158 | 0.166 |
| Qwen3-Coder-30B | R1 whole function | 0.405 | 18 | 0.028 | 0.028 |
| Qwen3-Coder-30B | R4 SRCVUL slice | 0.403 | 16 | 0.022 | 0.033 |
| Qwen3-Coder-30B | R6 slice + cross-function names | 0.403 | 18 | 0.030 | 0.033 |
| Phi-4 | R1 whole function | 0.455 | 31 | 0.194 | 0.163 |
| Phi-4 | R4 SRCVUL slice | 0.403 | 12 | 0.216 | 0.230 |
| Phi-4 | R6 slice + cross-function names | 0.392 | 10 | 0.248 | 0.263 |

Units with dataset context: 980; fields: Globals, External Function, Function Argument, Type Execution Declaration

## T8 Fine-tuned baseline (LoRA, Qwen2.5-Coder-7B, blind seeding; RQ1-E2)

| Input | Macro-F1 (greedy) | FPR | Pair-correct | VP-S |
|---|---|---|---|---|
| R1 | 0.333 | 0.000 | 0 | 0 |
| R4 | 0.333 | 0.000 | 0 | 0 |

| Model / input | Pair rank accuracy [95% CI] | ROC-AUC | Pair-correct at CV threshold | Mean score |
|---|---|---|---|---|
| lora / R1 | 52.7% [50.4, 55.0] | 0.514 | 12.3% [10.9, 13.9] | 0.499 |
| lora / R4 | 52.5% [50.2, 54.8] | 0.509 | 11.4% [10.1, 13.0] | 0.478 |
| base / R1 | 49.0% [46.7, 51.3] | 0.495 | 4.4% [3.6, 5.5] | 0.306 |
| base / R4 | 49.5% [47.3, 51.8] | 0.495 | 7.2% [6.1, 8.5] | 0.513 |

## T9 Nondeterminism (200 units, 3 repeats) and contamination split

| Model | Input | Flip rate | kappa rep0-rep1 |
|---|---|---|---|
| Qwen2.5-Coder-14B | R1 whole function | 0.015 | 0.979 |
| Qwen2.5-Coder-14B | R2 raw slice | 0.015 | 0.979 |
| Qwen2.5-Coder-14B | R3 slice + context | 0.015 | 0.990 |
| Qwen2.5-Coder-14B | R4 SRCVUL slice | 0.010 | 0.987 |
| Qwen2.5-Coder-14B | R5 slice + retrieved examples | 0.010 | 0.979 |
| Qwen3-Coder-30B | R1 whole function | 0.005 | 0.957 |
| Qwen3-Coder-30B | R2 raw slice | 0.020 | 0.862 |
| Qwen3-Coder-30B | R3 slice + context | 0.020 | 0.835 |
| Qwen3-Coder-30B | R4 SRCVUL slice | 0.015 | 0.881 |
| Qwen3-Coder-30B | R5 slice + retrieved examples | 0.050 | 0.906 |
| Phi-4 | R1 whole function | 0.000 | 1.000 |
| Phi-4 | R2 raw slice | 0.010 | 1.000 |
| Phi-4 | R3 slice + context | 0.005 | 0.980 |
| Phi-4 | R4 SRCVUL slice | 0.000 | 1.000 |
| Phi-4 | R5 slice + retrieved examples | 0.025 | 0.895 |

| Model | Pre-cutoff n | Pre acc. | Post-cutoff n | Post acc. |
|---|---|---|---|---|
| Qwen2.5-Coder-14B | 3600 | 0.511 | 70 | 0.486 |
| Qwen3-Coder-30B | 3600 | 0.507 | 70 | 0.486 |
| Phi-4 | 3600 | 0.504 | 70 | 0.514 |

## T10 RQ2 slicing ablation (macro-F1 per condition)

| Model | FULL | A0 | A1 | A2 | A3 | c1_interproc | c2_alias | c3_crossfile |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-14B | 0.463 | 0.508 | 0.482 | 0.463 | 0.486 | 0.474 | 0.475 | 0.478 |
| Qwen3-Coder-30B | 0.408 | 0.409 | 0.404 | 0.408 | 0.383 | 0.362 | 0.388 | 0.367 |
| Phi-4 | 0.379 | 0.442 | 0.421 | 0.379 | 0.408 | 0.392 | 0.380 | 0.395 |

Dose response (pooled F1 by % of slice removed): 0%: 0.464, 10%: 0.459, 20%: 0.458, 40%: 0.469, 60%: 0.490, 80%: 0.494; Jonckheere-Terpstra p = 0.864; Spearman rho = 0.001 (p = 0.844).

Efficiency without loss, pre-registered TOST (margin 0.05, from the pilot's seed-level F1 SD): mean diff -0.0147, 90% CI [-0.0316, 0.0021], p_TOST = 0.00029, equivalent = True.

Sensitivity, stricter margin 0.02: p_TOST = 0.303, equivalent = False.

Tokens: median 534 (slice) vs 746 (whole function), Wilcoxon p = 4.9e-185.

## T11 RQ3 repair per model (1,835 units; independent static oracle)

| Model / input | Vulnerability eliminated | Patch applies | Compiles |
|---|---|---|---|
| Qwen2.5-Coder-14B-Instruct / R1 | 0.0182 | 0.836 | 0.991 |
| Qwen2.5-Coder-14B-Instruct / R4 | 0.0173 | 0.666 | 0.991 |
| Qwen3-Coder-30B-A3B-Instruct-FP8 / R1 | 0.0222 | 0.762 | 0.992 |
| Qwen3-Coder-30B-A3B-Instruct-FP8 / R4 | 0.0169 | 0.815 | 0.992 |
| phi-4 / R1 | 0.0211 | 0.855 | 0.992 |
| phi-4 / R4 | 0.0109 | 0.603 | 0.991 |

| Comparison | Slice only | Whole only | Odds ratio | Holm p |
|---|---|---|---|---|
| Qwen2.5-Coder-14B-Instruct: R4(slice) vs R1(whole-function) | 25 | 24 | 1.041 | 1.000 |
| Qwen3-Coder-30B-A3B-Instruct-FP8: R4(slice) vs R1(whole-function) | 24 | 35 | 0.690 | 0.965 |
| phi-4: R4(slice) vs R1(whole-function) | 13 | 31 | 0.429 | 0.062 |

| Split | Model | Slice - whole (elimination) | 95% CI | n test |
|---|---|---|---|---|
| random | Qwen2.5-Coder-14B | 0.0018 | [-0.0127, 0.0164] | 550 |
| random | Qwen3-Coder-30B | -0.0055 | [-0.0200, 0.0091] | 550 |
| random | Phi-4 | -0.0091 | [-0.0218, 0.0036] | 550 |
| chronological | Qwen2.5-Coder-14B | 0.0122 | [-0.0031, 0.0306] | 327 |
| chronological | Qwen3-Coder-30B | 0.0092 | [-0.0092, 0.0275] | 327 |
| chronological | Phi-4 | 0.0000 | [-0.0153, 0.0153] | 327 |
| project_disjoint | Qwen2.5-Coder-14B | -0.0092 | [-0.0214, 0.0015] | 655 |
| project_disjoint | Qwen3-Coder-30B | -0.0137 | [-0.0275, -0.0015] | 655 |
| project_disjoint | Phi-4 | -0.0122 | [-0.0244, 0.0000] | 655 |
