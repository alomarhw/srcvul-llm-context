# Shorter, Not Smarter: replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23432671.svg)](https://doi.org/10.5281/zenodo.23432671)

Archived version 1.1.0 (with the LoRA adapter): https://doi.org/10.5281/zenodo.23432671 (all versions: https://doi.org/10.5281/zenodo.23430105).

Replication package for the article **"Shorter, Not Smarter: Dependency Slices as Context for LLM Vulnerability Detection, Localization, and Repair"** by Hakam Alomari and Mohammad Abdur Rafi Farhab (Miami University), a journal extension of the ICPC 2025 paper on SRCVUL.

The study tests whether SRCVUL's variable-level, dependency-preserving slices are better context for an LLM than the whole function. It covers vulnerability detection, localization, and repair, on 1,835 vulnerable/fixed C/C++ pairs, with three open-weight models (Qwen2.5-Coder-14B, Qwen3-Coder-30B, Phi-4).

Every model response is stored in a **frozen response cache**. All results, tables, and figures can therefore be regenerated **without a GPU and without any model call**.

## Quick start

Requirements:
- Python 3.12 (the pinned versions in `environment/requirements.txt` matter for exact replay);
- `xz` support, which ships with Python's `lzma`;
- for the full replay only: [srcML 1.1.0](https://www.srcml.org) and [srcSlice](https://github.com/srcML/srcSlice) on `PATH`.

```bash
make setup                   # create .venv with the pinned environment
. .venv/bin/activate
make unpack                  # decompress cache and large files, verify SHA-256 checksums
make tables                  # regenerate every paper table from results/ and diff against the paper (~1 min)
make figures                 # regenerate the result figures into out/figures
make rq4                     # recompute RQ4 inter-coder agreement from the coding sheets
make replay                  # rerun the whole pipeline from the frozen cache (~2 h on a laptop CPU)
```

What each target needs:
- `make tables`, `make figures`, and `make rq4` use only the shipped results.
- `make replay` also:
  - rebuilds the CleanVul pairs from the public release (`make data`; see *Data licensing*);
  - downloads the VulSlicer database from Zenodo;
  - recomputes slices with srcML/srcSlice;
  - serves every model call from the cache;
  - checks the new `results.json` against the shipped one, with every number required to match to 1e-9 (timings excepted).

## Where each result comes from

| Paper element | Produced by | Shipped output |
|---|---|---|
| RQ1–RQ3 metrics, tests, per-model results | `src/main.py` (all stages) | `results/results.json` |
| Chance-adjusted localization lift (Sec. 5.1; App. G) | `src/loc_chance.py`, `src/loc_lift_test.py` | `results/loc_chance.json`, `results/loc_lift_test.json` |
| Raw localization McNemar tests (App. G) | `src/loc_tests.py` | `results/loc_tests.json` |
| Chain-of-thought reasoning compliance | `src/addons.py` | `results/rq1_e4_cot_compliance.json` |
| LoRA label-token ranking (App. H) | `src/ft_logprob_rank.py --replay` | `results/ft_logprob_rank.json`, `data/ft_logprob_scores.jsonl` |
| All tables in the article and appendix | `paper/paper_tables_df89e.py` | `paper/expected_tables.md` |
| Result figures and the power figure | `paper/paper_figures_df89e.py` | `paper/figures/` |
| Figure 1 (study overview) | `paper/fig1_pipeline.tex` (TikZ) | `paper/figures/df89e_fig0_pipeline.pdf` |
| RQ4 agreement, adjudication, codes (Sec. 5.4; App. I) | `rq4/scripts/rq4_agreement.py`, `rq4_tables.py` | `rq4/df89e_rq4_agreement.json`, `rq4/df89e_rq4_results.*` |
| Per-unit model outputs | `src/main.py` | `results/records_rq1.jsonl`, `results/records_rq3.jsonl.xz` |

## Layout

```
src/            experiment pipeline: slicing and label-free seeding (slice_pipeline.py), representations R1-R6,
                corruption (corruption_injectors.py), BM25 retrieval (retrieval.py), cached model calls
                (llm_runner.py, rp_llm_cache.py), repair and static oracle (repair_pipeline.py), analyses,
                LoRA training (train_lora.py) and label-token scoring (ft_logprob_rank.py)
data/           the evaluation pairs (JSON lines, one vulnerable/fixed pair per line; see data/DATA_CARD.md),
                the scripts that built them, and the CleanVul index (see Data licensing)
cache/          the frozen response cache (SQLite, xz-compressed; 73,398 responses keyed by a hash of model,
                revision, prompt, and representation)
results/        the outputs of the reported run
paper/          table and figure generators, the expected tables, and the figures as they appear in the article
rq4/            codebook v1.2, the blinded samples, both coders' sheets, adjudication, final labels, agreement,
                the model-identity keys, the case studies, and the sampling and agreement scripts
model/          the LoRA fine-tuned baseline: adapter weights, training configuration and log (see model/README.md)
superseded_label_seeded_run/   the earlier run whose slices were seeded on the labelled lines (it leaked the
                answer and is not reported); kept so the leak can be checked against the label-free run
environment/    pinned Python requirements (analysis/replay; GPU inference)
```

## RQ4 coding data

- **Coders.** Coder 1 is the first author (`*_hakam.csv`) and coder 2 the second author (`*_student.csv`). They coded independently, blind to the model.
- **Keys.** The `*_key_DO_NOT_SHARE_WITH_CODERS.csv` files map each case to the model and representation. They were withheld during coding and are published here so the analysis can be checked.
- **Codebook revision.** `df89e_rq4_hakam_OTHER_recode.csv` records coder 1's recoding of uncategorized cases under codebook v1.2. The revision was made before coder 2 started.

## Re-running inference (optional, needs a GPU)

The reported run served the models with vLLM 0.31.0 on one NVIDIA RTX 6000 Ada (48 GB), at temperature 0 and seed 0, with `--max-model-len 16384`. The pinned revisions were:

| Model | Revision |
|---|---|
| Qwen2.5-Coder-14B-Instruct | `aedcc2d4` |
| Qwen3-Coder-30B-A3B-Instruct-FP8 | `dcaee4d4` |
| microsoft/phi-4 | `2db69c1c` |

To run it:
1. Start an OpenAI-compatible vLLM server.
2. Set `RP_LOCAL_LLM_BASE_URL` to its URL.
3. Point `RP_LLM_CACHE` at a new, empty cache.
4. Run `python src/main.py --full`.

New responses will differ slightly from the frozen ones because GPU inference is not bit-exact. The 200-unit repeat runs bound that variation (flip rate at most 0.05). The trained LoRA adapter, its configuration, and the training log are in `model/` (Appendix H of the article).

## Data licensing

- **PrimeVul** ([colin/PrimeVul](https://huggingface.co/datasets/colin/PrimeVul)) and **SecVulEval** ([arag0rn/SecVulEval](https://huggingface.co/datasets/arag0rn/SecVulEval)) are MIT-licensed. Our selected subsets are included with attribution.
- **CleanVul** ([yikun-li/CleanVul](https://huggingface.co/datasets/yikun-li/CleanVul)) lists no license, so its code is not redistributed. `data/cleanvul_index.jsonl` keeps every other field plus a SHA-256 of each function. `make data` downloads the public release and rebuilds the identical file, verified by checksum.
- **The post-cutoff pairs** are functions from public fix commits of OpenSSL, PHP, libexpat, and curl, included for research under their projects' open-source licenses.
- **The VulSlicer database** is downloaded from [Zenodo 6059924](https://zenodo.org/records/6059924) at replay time.

## License and citation

- **Code** is under the MIT license (`LICENSE`).
- **Our annotations, coding data, and results** are under CC BY 4.0 (`LICENSE-DATA`).
- **Third-party data** remains under its original licenses.

Please cite the article; see `CITATION.cff`.
