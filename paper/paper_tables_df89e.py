"""paper_tables_df89e.py -- markdown tables for the #DF89E paper, read only from results/results.json (+ the CoT
compliance file). Usage: python paper_tables_df89e.py RESULTS_DIR OUT.md"""
import json
import sys
from pathlib import Path

SHORT = {"Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder-14B", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": "Qwen3-Coder-30B",
         "microsoft/phi-4": "Phi-4"}
REPS = ["R1", "R2", "R3", "R4", "R5"]
REPN = {"R1": "R1 whole function", "R2": "R2 raw slice", "R3": "R3 slice + context", "R4": "R4 SRCVUL slice",
        "R5": "R5 slice + retrieved examples", "R6": "R6 slice + cross-function names"}


def f(x, d=3):
    return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    res_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    r = json.loads((res_dir / "results.json").read_text())
    d1 = r["RQ1"]["_detail"]; pm = d1["per_model"]["models"]; models = list(pm)
    md = ["# #DF89E paper tables (generated from results/results.json by paper_tables_df89e.py)\n"]

    md.append("## T1 Detection per model and representation (3,670 units, 1,835 pairs)\n")
    rows = []
    for m in models:
        for rep in REPS:
            a, p = pm[m]["per_representation"][rep], pm[m]["pairwise"][rep]
            rows.append([SHORT[m], REPN[rep], f(a["macro_f1"]), f(a["recall"]), f(a["false_positive_rate_fpr"]),
                         f"{p['pair_correct']} ({p['pair_correct_rate']*100:.1f}%)", p["vp_s"],
                         f"{p['pair_both_vulnerable_rate']*100:.0f}% / {p['pair_both_safe_rate']*100:.0f}%", f(p["parse_failure_rate"])])
    md.append(table(["Model", "Input", "Macro-F1", "Recall", "FPR", "Pair-correct", "VP-S", "Both vuln. / both safe", "Parse fail"], rows))

    md.append("## T2 Localization per model and representation (1,433 vulnerable units with labelled lines)\n")
    rows = [[SHORT[m], REPN[rep], f(pm[m]["per_representation"][rep]["top_1"]), f(pm[m]["per_representation"][rep]["top_5"]),
             f(pm[m]["per_representation"][rep]["line_precision"]), f(pm[m]["per_representation"][rep]["line_recall"])]
            for m in models for rep in REPS]
    md.append(table(["Model", "Input", "Top-1", "Top-5", "Line precision", "Line recall"], rows))

    md.append("## T3 Representation tests (pooled over models; paired Wilcoxon, Holm)\n")
    rows = [[t["pair"], f(t["cohens_d"]), f"[{t['ci'][0]:.3f}, {t['ci'][1]:.3f}]", f(t["p_raw"], 3), f(t["p_holm"], 3)]
            for t in d1["pairwise_wilcoxon_holm"]]
    md.append(table(["Comparison", "Cohen's d", "95% CI (diff. in correct)", "p", "Holm p"], rows))
    md.append(table(["Model", "R4 vs best non-metadata: effect", "95% CI", "Holm p"],
                    [[SHORT[m], f(pm[m]["r4_vs_best_test"]["effect_size"]),
                      f"[{pm[m]['r4_vs_best_test']['confidence_interval'][0]:.3f}, {pm[m]['r4_vs_best_test']['confidence_interval'][1]:.3f}]",
                      f(pm[m]["r4_vs_best_test"]["p_value"])] for m in models]))

    md.append("## T4 Baselines (pooled over models; whole-file context is omitted because it equals R1 here)\n")
    rows = []
    for k, v in r["RQ1"]["baseline_comparison"].items():
        if "whole-file" in k:
            continue  # identical to R1: whole files are not available for these units
        rows.append([k[:70], f(v.get("precision")), f(v.get("recall")), f(v.get("macro_f1")), f(v.get("false_positive_rate_fpr", v.get("fpr")))])
    md.append(table(["Baseline", "Precision", "Recall", "Macro-F1", "FPR"], rows))

    ad = d1["addons"]
    lt = res_dir / "loc_tests.json"
    if lt.exists():
        md.append("## T2b Localization tests per model (exact McNemar on paired top-k hits vs R1, Holm over 12 tests)\n")
        rows = [[SHORT.get(t["model"], t["model"]), t["comparison"], f"top-{t['k']}", f"{t['rate_R1']:.3f} -> {t['rate_slice']:.3f}",
                 f"+{t['slice_only']} / -{t['whole_only']}", f(t["p"], 4), f(t["p_holm"], 4)] for t in json.loads(lt.read_text())]
        md.append(table(["Model", "Comparison", "k", "Hit rate R1 -> slice", "Slice-only / whole-only hits", "p", "Holm p"], rows))
    lc, ll = res_dir / "loc_chance.json", res_dir / "loc_lift_test.json"
    if lc.exists() and ll.exists():
        c = json.loads(lc.read_text())
        md.append("## T2c Localization against chance (random picks matched to the number of lines each model cited)\n")
        md.append(f"Labelled line visible in the input: R1 {c['labelled_line_visible_share']['R1']*100:.0f}%, R2 {c['labelled_line_visible_share']['R2']*100:.0f}%, R4 {c['labelled_line_visible_share']['R4']*100:.0f}%; mean lines shown: R1 {c['mean_shown_lines']['R1']:.1f}, R4 {c['mean_shown_lines']['R4']:.1f}.\n")
        rows = [[SHORT.get(t["model"], t["model"]), t["comparison"], f"top-{t['k']}", f"{t['lift_R1']*100:+.1f}", f"{t['lift_slice']*100:+.1f}",
                 f"{t['diff']*100:+.1f} [{t['ci95'][0]*100:+.1f}, {t['ci95'][1]*100:+.1f}]", "<0.001" if t["p_holm"] < 0.001 else f(t["p_holm"], 3)]
                for t in json.loads(ll.read_text())]
        md.append(table(["Model", "Comparison", "k", "Lift over chance, R1 (pts)", "Lift, slice (pts)", "Difference [95% CI]", "Holm p"], rows))
    md.append("## T5 Official PrimeVul paired test set (435 pairs; RQ1-E3)\n")
    e3 = ad["RQ1-E3"]["results"]
    rows = [[SHORT[m], REPN[rep], e3[m][rep]["pair_correct"], e3[m][rep]["vp_s"], f(e3[m][rep]["fpr"]), f(e3[m][rep]["macro_f1"])]
            for m in e3 for rep in REPS]
    md.append(table(["Model", "Input", "Pair-correct (of 435)", "VP-S", "FPR", "Macro-F1"], rows))
    md.append("Cited on the same split (other papers, not rerun): " + "; ".join(f"{k}: {v}" for k, v in ad["RQ1-E3"]["cited_baselines_pc_on_same_split"].items()) + "\n")

    md.append("## T6 Prompt robustness: zero-shot vs chain-of-thought (nondeterminism subset; RQ1-E4)\n")
    e4 = ad["RQ1-E4"]["results"]
    comp = {}
    cp = res_dir / "rq1_e4_cot_compliance.json"
    if cp.exists():
        comp = json.loads(cp.read_text())
    rows = []
    for m in e4:
        for rep in ("R1", "R4"):
            z, c = e4[m]["zero_shot"][rep], e4[m]["cot"][rep]
            rows.append([SHORT[m], REPN[rep], f(z["macro_f1"]), f(c["macro_f1"]), z.get("pair_correct"), c.get("pair_correct"),
                         f"{comp[m]['share']*100:.0f}%" if m in comp else "--"])
    md.append(table(["Model", "Input", "F1 zero-shot", "F1 CoT", "Pair-correct zero-shot", "Pair-correct CoT", "CoT answers with reasoning"], rows))
    md.append("The 200-unit subset samples units, not pairs, so few complete pairs are in it; pair-correct counts here are not comparable to T1.\n")

    md.append("## T7 Cross-function context probe (SecVulEval; RQ1-E5)\n")
    e5 = ad["RQ1-E5"]["results"]
    rows = [[SHORT[m], REPN[rep], f(e5[m][rep]["macro_f1"]), e5[m][rep].get("pair_correct"), f(e5[m][rep].get("top_1")), f(e5[m][rep].get("line_recall"))]
            for m in e5 for rep in ("R1", "R4", "R6")]
    md.append(table(["Model", "Input", "Macro-F1", "Pair-correct", "Top-1", "Line recall"], rows))
    md.append(f"Units with dataset context: {ad['RQ1-E5']['units_with_context']}; fields: {', '.join(ad['RQ1-E5']['fields_used'])}\n")

    md.append("## T8 Fine-tuned baseline (LoRA, Qwen2.5-Coder-7B, blind seeding; RQ1-E2)\n")
    ft = ad["RQ1-E2-finetuned"]
    rows = [[rep, f(ft["results"][rep]["macro_f1"]), f(ft["results"][rep]["fpr"]), ft["results"][rep]["pair_correct"], ft["results"][rep]["vp_s"]] for rep in ft["results"]]
    md.append(table(["Input", "Macro-F1 (greedy)", "FPR", "Pair-correct", "VP-S"], rows))
    lr = ft.get("label_token_ranking", {})
    rows = [[k.replace('|', ' / '), f"{v['pair_rank_accuracy'][0]*100:.1f}% [{v['pair_rank_accuracy'][1]*100:.1f}, {v['pair_rank_accuracy'][2]*100:.1f}]", f(v["roc_auc"]),
             f"{v['pair_correct_cv_threshold'][0]*100:.1f}% [{v['pair_correct_cv_threshold'][1]*100:.1f}, {v['pair_correct_cv_threshold'][2]*100:.1f}]", f(v["score_mean"])]
            for k, v in lr.items()]
    md.append(table(["Model / input", "Pair rank accuracy [95% CI]", "ROC-AUC", "Pair-correct at CV threshold", "Mean score"], rows))

    md.append("## T9 Nondeterminism (200 units, 3 repeats) and contamination split\n")
    rows = [[SHORT[m], REPN[rep], f(pm[m]["nondeterminism"][rep]["flip_rate"]), f(pm[m]["nondeterminism"][rep]["kappa_rep0_rep1"])] for m in models for rep in REPS]
    md.append(table(["Model", "Input", "Flip rate", "kappa rep0-rep1"], rows))
    rows = [[SHORT[m], pm[m]["contamination_pre_post"]["pre_cutoff_n"], f(pm[m]["contamination_pre_post"]["pre_cutoff_acc"]),
             pm[m]["contamination_pre_post"]["post_cutoff_n"], f(pm[m]["contamination_pre_post"]["post_cutoff_acc"])] for m in models]
    md.append(table(["Model", "Pre-cutoff n", "Pre acc.", "Post-cutoff n", "Post acc."], rows))

    d2 = r["RQ2"]["_detail"]; p2 = d2["per_model"]["models"]
    md.append("## T10 RQ2 slicing ablation (macro-F1 per condition)\n")
    conds = ["FULL", "A0", "A1", "A2", "A3", "c1_interproc", "c2_alias", "c3_crossfile"]
    rows = [[SHORT[m]] + [f(p2[m]["per_condition"][c]["macro_f1"]) for c in conds] for m in p2]
    md.append(table(["Model"] + conds, rows))
    dr = d2["dose_response"]
    md.append(f"Dose response (pooled F1 by % of slice removed): {', '.join(f'{k}%: {v:.3f}' for k, v in dr['f1_by_level'].items())}; Jonckheere-Terpstra p = {dr['jonckheere_terpstra_p']:.3f}; Spearman rho = {dr['spearman_rho_level_vs_correct']:.3f} (p = {dr['spearman_p']:.3f}).\n")
    pr = d2["rq2_e2_efficiency_without_loss"]["tost_effectiveness_equivalence"]
    md.append(f"Efficiency without loss, pre-registered TOST (margin {pr['margin']}, from the pilot's seed-level F1 SD): mean diff {pr['mean_diff']:.4f}, 90% CI [{pr['ci90_mean_diff'][0]:.4f}, {pr['ci90_mean_diff'][1]:.4f}], p_TOST = {pr['tost_p']:.2g}, equivalent = {pr['equivalent']}.\n")
    t = d2["efficiency_without_loss_tost"]
    md.append(f"Sensitivity, stricter margin {t['margin']}: p_TOST = {t['p_tost']:.3f}, equivalent = {t['equivalent']}.\n")
    e = d2["efficiency_reductions"]["tokens"]
    md.append(f"Tokens: median {e['median_full']:.0f} (slice) vs {e['median_whole_function']:.0f} (whole function), Wilcoxon p = {e['p_value']:.1e}.\n")

    d3 = r["RQ3"]["_detail"]
    md.append("## T11 RQ3 repair per model (1,835 units; independent static oracle)\n")
    rows = [[k.replace("|", " / ").replace("Qwen/", "").replace("microsoft/", ""), f(v.get("independent_vulnerability_elimination_rate_rq3"), 4),
             f(v.get("patch_applicability_rate_rq3")), f(v.get("compilation_success_rate_rq3"))] for k, v in d3["per_model_rep_summary"].items()]
    md.append(table(["Model / input", "Vulnerability eliminated", "Patch applies", "Compiles"], rows))
    rows = [[x["comparison"].replace("Qwen/", "").replace("microsoft/", ""), x["discordant_a_only"], x["discordant_b_only"], f(x["odds_ratio"]), f(x["p_holm"])] for x in d3["mcnemar_slice_vs_whole"]]
    md.append(table(["Comparison", "Slice only", "Whole only", "Odds ratio", "Holm p"], rows))
    g = d3["rq3_e2_generalization"]["per_split_model_delta"]
    rows = [[sp, SHORT.get(m, m), f(v["delta"], 4), f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]", v["n"]] for sp in g for m, v in g[sp].items()]
    md.append(table(["Split", "Model", "Slice - whole (elimination)", "95% CI", "n test"], rows))
    out.write_text("\n".join(md))
    print("wrote", out)


if __name__ == "__main__":
    main()
