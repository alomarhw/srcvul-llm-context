# #DF89E RQ4: failure-analysis results (final, after adjudication)

**What this is:** the RQ4 results and a draft of the method and results text.
**Data:**
- final labels: `Docs/df89e_rq4_final_labels.csv`;
- numbers: `Docs/df89e_rq4_results.json` (tables) and `Docs/df89e_rq4_agreement.json` (κ);
- scripts in the package: `rq4_sample.py`, `rq4_cot_sample.py`, `rq4_agreement.py`, `rq4_tables.py`.

The samples are stratified by design, with 20 cases per stratum in layer 1, so the counts below are **within-stratum shares, not population rates**.

---

## 1. Reliability

| | Cases | Raw agreement | Cohen's κ [95% CI] |
|---|---|---|---|
| Layer 1, primary code | 100 | 76% | 0.74 [0.64, 0.82] |
| Layer 1, both coders high-confidence | 42 | 93% | 0.92 [0.81, 1.00] |
| Layer 1, slice effect | 20 | 100% | 1.00 (near-mechanical: each row shows which answer was correct) |
| Layer 2, primary code | 40 | 70% | 0.66 [0.49, 0.80] |
| Layer 2, both coders high-confidence | 17 | 88% | 0.84 [0.63, 1.00] |

The 36 disagreements were adjudicated jointly; 13 were resolved to coder 1's code and 23 to coder 2's. The codebook was revised once (v1.2) from coder 1's `OTHER` cases, before coder 2 started. Twelve of the disagreements came from that revision: coder 1 had used DC/DU before the new code DG existed. Ten of those were adjudicated as DG.

## 2. Layer 1: what goes wrong (100 cases, 3 models)

| Stratum (20 each) | Final primary codes |
|---|---|
| Detection false positives | **DG guard dropped by slice 10**, DU unsupported flag 4, DC flags a guarded operation 4, LB 2 |
| Detection false negatives | **CX missing cross-function context 10**, DS flaw outside shown code 4, DM missed visible flaw 4, UN 2 |
| Localization misses | **LN near-miss (right data flow, wrong lines) 12**, UN 4, LW wrong region 3, LB 1 |
| Repair failures | RX broken 8, RW wrong change at right place 6, RN no-op 5, LB 1 |
| Slice-effect cases | DG 5, DM 5, CX 4, DU 3, UN 2, DS 1 |

**Whole function vs. slice (detection strata):**

| | Whole function (R1) | Slice (R4) |
|---|---|---|
| False positives caused by a dropped guard (DG) | 0 of 6 | **10 of 14** |
| False negatives caused by missing cross-function context (CX) | **9 of 13** | 1 of 7 |
| False negatives where the flaw is not in the shown code (DS) | 0 of 13 | 4 of 7 |

**Where slices helped or hurt (20 cases, 10 each):**
- **Helped:** the whole-function answer had missed a flaw visible in the code (DM 5) or lacked context (CX 4). The slice focused attention on the vulnerable flow.
- **Hurt:** the slice dropped a guard (DG 5), produced an unsupported flag (DU 3), or cut the flaw out (DS 1).

**Other observations:**
- Missing cross-function context appears in **32 of 100** cases, as the primary or a secondary code.
- **4 label problems** (C046, C048, C052, C070) go into the threats section as dataset issues.
- By model, the codes spread similarly across all three models; with about 33 cases per model, there are no reliable model differences.

## 3. Layer 2: why, in the models' own reasoning (40 cases, Qwen3-Coder and Phi-4)

| Code | Count |
|---|---|
| SM Semantic Misunderstanding | 10 |
| AE Absence-as-Evidence | 9 |
| OT Over-Trust | 4 |
| PM Pattern-Matching | 4 |
| UN Undeterminable | 4 |
| SP Speculation | 3 |
| NF No reasoning flaw (flaw outside the shown code) | 3 |
| CX, PH, RV | 1 each |

- **By stratum:**
  - false positives: SM 6, AE 4;
  - false negatives: OT 4, NF 3;
  - localization misses: AE 5, SM 4.
- **By input:** absence-as-evidence is more common on slices (6 of 20) than on whole functions (3 of 20), consistent with the dropped-guard pattern in layer 1.
- **Coverage:** Qwen2.5-Coder-14B is absent from layer 2, because it wrote no reasoning in any of its 400 chain-of-thought answers (`results/rq1_e4_cot_compliance.json`).

---

## 4. Draft text for the paper

### RQ4 method (draft)

> To explain the failures, two authors coded a stratified sample in two layers using a codebook seeded with the reasoning-flaw categories of Aegis~\cite{aegis} and extended with stage-specific codes. *Layer 1* covers 100 failures from the paper-grade runs of all three models: 20 each of detection false positives, detection false negatives, localization misses, repair failures, and cases where the whole-function and slice answers disagree. Because the main runs return only a verdict, cited lines and a CWE, layer 1 codes only the observable failure. *Layer 2* covers 40 failures from the chain-of-thought runs, where models write their reasoning first, and codes the reasoning flaw. The coders worked independently and blind to the model; agreement was substantial (layer 1: κ = 0.74, 95% CI [0.64, 0.82]; layer 2: κ = 0.66, [0.49, 0.80]; 0.92 and 0.84 on cases both coders rated high-confidence). Disagreements were adjudicated jointly. The codebook was revised once, before the second coder started, to add five codes derived from the first coder's uncategorized cases.

### RQ4 results (draft)

> Slices trade one blind spot for another. On whole functions, most missed vulnerabilities lack context from other functions (9 of 13 false negatives). On slices, that cause almost disappears (1 of 7), but two new ones take its place: the slice drops the guard that makes the cited operation safe, which causes 10 of 14 false positives on slices and none on whole functions, or the slice omits the flawed statement itself (4 of 7 false negatives). Where slices helped, they did so by focusing the model on a flaw it had overlooked in the full function; where they hurt, they removed the guard or the flaw. Localization failures are mostly near misses on the right data flow (12 of 20), and failed repairs split between broken patches, wrong changes at the right place, and no-op patches. In the models' own reasoning, the most common flaws are semantic misunderstanding of an API or construct (10 of 40) and absence-as-evidence (9 of 40), which is more frequent on slices (6 vs. 3 of 20), consistent with the dropped guards. Missing cross-function context appears in a third of all coded failures, supporting the view that intraprocedural evidence, whole or sliced, is the limiting factor.

**Caveats for the threats section:**
- the sample is stratified, so the counts are within-stratum shares;
- the counts per cell are small; the one formal contrast is false positives from dropped guards, R4 vs. R1, Fisher p = 0.011, but DG can arise only when code is omitted, so it is partly definitional;
- slice-effect agreement is near-mechanical;
- layer 2 covers two of the three models.
