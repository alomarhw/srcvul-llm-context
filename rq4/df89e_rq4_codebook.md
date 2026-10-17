# #DF89E RQ4: failure-analysis codebook and coding instructions

**For:** the two human coders (the author and a student).
**Version:** 1.2, 2026-10-16.

**RQ4:** *Why do LLMs fail at vulnerability detection, localization and repair, and where do program slices help or hurt?*

RQ4 is coded in two layers. Each layer has its own sheet and its own rules.

| Layer | Sheet | Cases | What you code | Time |
|---|---|---|---|---|
| 1. What goes wrong | `Docs/df89e_rq4_sample.csv` | 100 | the observable failure, plus the slice effect | about 5–8 h |
| 2. Why, in the model's own words | `Docs/df89e_rq4_reasoning_sample.csv` | 40 | the reasoning flaw, from the model's written reasoning | about 2–3 h |

In layer 1 the models answered with a verdict, cited lines and a CWE only, so there is **no reasoning text**. Code only what the row shows. In layer 2 every model wrote its reasoning before the verdict, and that reasoning is what you code.

---

## 1. Procedure (both layers)

1. **Independence.** Code on your own copy of each sheet, for example `..._hakam.csv` and `..._student.csv`. Do not discuss cases or look at the other coder's sheet until both of you have finished.
2. **Blinding.** The sheets do not say which model produced an answer. Do not try to identify the model.
3. **Columns to fill in:**
   - `primary_category` (required): one code from §2 (layer 1) or §3 (layer 2).
   - `secondary_category` (optional): one more code, only if it clearly applies too.
   - `slice_effect` (layer 1 only): see §4.
   - `confidence` (required): see §5.
   - `note` (required for `OTHER`, `LB`, `CX` and `UN`, and for `helped`/`hurt`): one sentence.
4. **Precedence when more than one code fits.** Take the first that applies, in this order:
   1. `LB` (label or ground-truth problem): never a model failure, so it always wins.
   2. The layer's main codes: the observable failure in layer 1 (§2), the reasoning flaw in layer 2 (§3).
   3. `CX` (missing cross-function context): primary only if nothing above fits; otherwise use it as the secondary code.
   4. `UN` (undeterminable from the row).
   5. `OTHER`: only if even `UN` does not fit.
5. **New categories.** `OTHER` should now be rare. If you still use it, describe the case in the note; recurring `OTHER` cases become new codes in a later version (§8).
6. **Agreement.** I compute Cohen's κ per layer on `primary_category`, and in layer 1 also on `slice_effect`. The target is κ ≥ 0.6. You then adjudicate disagreements together. We report both the raw κ and the adjudicated distribution.

---

## 2. Layer 1: observable failure codes (main sheet)

Use only what the row shows: the code given to the model, the ground truth, the official fix (repair cases), and the model's verdict, cited lines and patch. **Do not guess the model's reasoning.**

### Detection errors (rows `detection false positive` / `detection false negative`)

| Code | Name | Use when… |
|---|---|---|
| DC | Flags a guarded operation | false positive whose cited lines contain an operation that a check visible in the shown code already protects (e.g., `memcpy` after a bounds check) |
| DU | Unsupported flag | false positive whose cited lines show no dangerous operation, or none related to the CWE the model named |
| DE | Empty flag | "vulnerable" with no cited lines, or with cited lines outside the shown code |
| DM | Missed visible flaw | false negative where the vulnerable lines are in the shown code and visibly unsafe |
| DS | Flaw outside the shown code | false negative where the vulnerable lines, or the operation that makes them unsafe, are **not** in the code shown (e.g., the slice dropped them) |
| DG | Guard dropped by the slice | false positive because the code shown (usually a slice) omits the check, early return, error path or other control flow that protects the cited operation; the false-positive counterpart of DS |

### Localization and repair errors

| Code | Name | Use when… |
|---|---|---|
| LN | Near-miss localization | the cited lines are on the right data flow (same variable or statement chain) but not the labelled lines |
| LW | Wrong-region localization | the cited lines are unrelated to the vulnerability |
| RP | Partial repair | the patch makes part of the official fix (e.g., one of two checks) |
| RW | Wrong repair | the patch edits the right place, but in a way that does not remove the vulnerability (e.g., checks the wrong bound) |
| RX | Broken repair | the patch does not parse, changes unrelated code, or breaks functionality |
| RN | No-op repair | the patch leaves the vulnerable statements unchanged, or reproduces the function without the fix (formatting or unrelated edits only) |

### Always available

| Code | Name | Use when… |
|---|---|---|
| CX | Missing cross-function context | the right answer needs information from another function or file that the model was not shown; say what was missing in the note |
| LB | Label or ground-truth problem | the model's answer looks right and the dataset label or fix looks wrong or incomplete; reported separately, never counted as a model failure |
| UN | Undeterminable | the row does not give enough to judge the failure (e.g., the labelled lines are blank or diagnostic statements and no fix is shown); use LB instead if the label looks *wrong* |
| OTHER | Other | nothing fits, not even UN; describe the case in the note |

**Reasoning-flaw codes in layer 1.** You may add a §3 code as the `secondary_category`, but only when the row shows it without any guessing. Example: the model cites a `strcpy` that is bounded two lines earlier, so `DC` plus secondary `PM`. If you would have to imagine what the model was thinking, leave the secondary empty.

**Slice-effect rows** (`stage = slice effect`). Code the failure of whichever answer was wrong with the codes above, and fill in `slice_effect` (§4).

---

## 3. Layer 2: reasoning-flaw codes (reasoning sheet)

Each row gives the code the model saw, the ground truth, the model's **written reasoning**, and its verdict, cited lines and CWE. Code the flaw in the reasoning that best explains the wrong answer. The categories follow Fang et al., *Aegis* (arXiv 2603.20637, §2.4 and Table 3), and are cited in the paper.

| Code | Name | Use when the reasoning… |
|---|---|---|
| PM | Pattern-Matching | flags (or clears) code because of a surface pattern (e.g., `memcpy`, `strcpy`, pointer arithmetic) without checking whether the data flow makes it exploitable |
| SM | Semantic Misunderstanding | misreads what an API or construct does (e.g., the bounds `strncpy` enforces, integer promotion) |
| AE | Absence-as-Evidence | concludes "vulnerable" because it cannot see a check, when the check is outside the shown code, or its absence does not make the code exploitable |
| OT | Over-Trust | assumes callers, libraries or inputs are safe without evidence in the code |
| SP | Speculation | relies on behaviour of code it was not shown (callers, configuration, runtime state) |
| AN | Anchoring | fixates on the first suspicious line and ignores evidence elsewhere |
| PH | Phantom Mitigation | cites a check or sanitisation that is not on the relevant path, or not in the code at all |
| SC | Scope Creep | argues about issues unrelated to the vulnerability in question |
| CX | Missing cross-function context | the reasoning is sound for the code shown, but the answer needs code it was not shown |
| LB | Label or ground-truth problem | the reasoning and answer look right and the label looks wrong |
| NF | No reasoning flaw | the reasoning is sound for the code shown, and the answer is wrong only because the flaw lies outside that code (often with secondary DS) |
| RV | Reasoning–verdict mismatch | the written reasoning reaches one conclusion and the recorded verdict states the other |
| UN | Undeterminable | the row does not give enough to tell whether the reasoning is flawed (e.g., the ground truth is only a signature or blank lines) |
| OTHER | Other | none fits, not even UN; describe the case in the note |

In layer 2, `secondary_category` can be another §3 code or a §2 code (e.g., `LN`) when the reasoning also shows a localization pattern.

---

## 4. Slice effect (layer 1, slice-effect rows only)

These rows show both answers: whole function (R1) and slice (R4). For every other row, enter `n/a`.

| Code | Meaning |
|---|---|
| `helped` | the slice answer is correct and the whole-function answer is not. Say *why* in the note, e.g., "slice removed distracting code", "metadata exposed the tainted variable". |
| `hurt` | the whole-function answer is correct and the slice answer is not. Say *why*, e.g., "slice dropped the bounds check on line 12", "slice lost the caller's context". |
| `neutral` | both answers are wrong, for different reasons |
| `n/a` | not a slice-effect row |

---

## 5. Confidence (both layers)

| Value | Use when… |
|---|---|
| `high` | the evidence in the row clearly supports your primary code and no other code is a close second |
| `low` | two codes are plausible, or the decision depends on code or context the row does not show |

Use `low` freely; it does not lower the quality of the coding. We report κ for all cases and for the `high`-confidence cases separately.

---

## 6. Sampling (for the paper's method section)

**Layer 1:** 100 cases, seed 42, balanced across the three models and both representations within each stratum. Built by `rq4_sample.py`.

| Stratum | Cases |
|---|---|
| Detection false positives (fixed version flagged) | 20 |
| Detection false negatives (vulnerable version cleared) | 20 |
| Localization misses (vulnerable detected, labelled lines not cited) | 20 |
| Repair failures (patch rejected by the oracle) | 20 |
| Slice-effect cases (R1 and R4 disagree; half helped, half hurt) | 20 |

**Layer 2:** 40 failures from the chain-of-thought runs (RQ1-E4), where the model reasons before its verdict. Seed 42, balanced across models and representations. Built by `rq4_cot_sample.py`.
- **Strata:** 14 detection false positives, 14 detection false negatives, 12 localization misses.
- **Models:** only answers that contain written reasoning are eligible. One model never wrote reasoning under the chain-of-thought prompt (0 of 400 answers), so layer 2 covers the other two models; the paper reports this.

---

## 7. Worked examples (illustrative, not from our data)

```c
1: int copy(char *dst, const char *src, size_t n) {
2:   char buf[16];
3:   if (n > sizeof(buf)) n = sizeof(buf);
4:   memcpy(buf, src, n);
5:   return process(dst, buf, n);
6: }
```
The label is *safe* (the fixed version), and line 3 bounds `n`.

- **Layer 1:** the model says *vulnerable* and cites line 4, with no reasoning text. Code `primary = DC` (it flags an operation guarded by line 3) and `secondary = PM`: the row shows the pattern without guessing, because the guard is two lines above the cited call. `confidence = high`. `note`: "cites memcpy bounded on line 3".
- **Layer 2:** the reasoning says "line 4 copies `n` bytes into a 16-byte buffer with no size check". Code `primary = PM`: the model flags the copy by pattern and misses a check that is present. (`PH` would be the reverse: citing a check that is *not* there.) `confidence = high`. `note`: "ignores the clamp on line 3".
- **Layer 2, another answer:** the reasoning says "`process()` may not null-terminate `dst`". Code `primary = SP` (it relies on code it was not shown) and `secondary = SC` (unrelated to the bounded copy). `confidence = low`.

---

## 8. Change log
- **v1 (2026-10-13):** initial categories; the reasoning-flaw categories follow Aegis (arXiv 2603.20637).
- **v1.1 (2026-10-16):**
  - **Two layers:** layer 1 codes observable failures on the main sheet; layer 2 codes reasoning flaws on a new 40-case reasoning sheet.
  - **New layer-1 detection codes:** DC, DU, DE, DM, DS.
  - **Reasoning-flaw codes in layer 1:** allowed only as an unambiguous secondary code.
  - **Defined:** the confidence scale and the precedence rule.
- **v1.2 (2026-10-16):** five codes from the first coder's `OTHER` cases (19 of 140), added before the second coder started:
  - layer 1: **DG** guard dropped by the slice, **RN** no-op repair;
  - layer 2: **NF** no reasoning flaw, **RV** reasoning–verdict mismatch;
  - both layers: **UN** undeterminable from the row.
