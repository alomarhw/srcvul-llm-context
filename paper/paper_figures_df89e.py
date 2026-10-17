"""paper_figures_df89e.py -- paper figures for #DF89E, drawn only from verified results (no model calls).

Reads RESULTS_DIR/{results.json, loc_lift_test.json, records_rq3.jsonl} and DOCS_DIR/df89e_rq4_final_labels.csv.
Writes OUT_DIR/df89e_fig{1..6}_*.{pdf,png}. Usage: python paper_figures_df89e.py RESULTS_DIR DOCS_DIR OUT_DIR
Palette: blue / orange / aqua per model (fixed order), grey for references; every series also has its own marker.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MODELS = [("Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen2.5-Coder-14B", "#2a78d6", "o"),
          ("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "Qwen3-Coder-30B", "#eb6834", "s"),
          ("microsoft/phi-4", "Phi-4", "#1baf7a", "^")]
REPS = ["R1", "R2", "R3", "R4", "R5"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "figure.constrained_layout.use": True, "lines.linewidth": 1.5, "lines.markersize": 4.5})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    res, docs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    r = json.loads((res / "results.json").read_text()); pm = r["RQ1"]["_detail"]["per_model"]["models"]
    style()

    # Fig 1: detection bias -- recall vs FPR per model across representations
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(0.63, 0.43, "chance line\n(recall = FPR)", color=INK2, fontsize=6.3, ha="left")
    for key, name, col, mk in MODELS:
        pts = [(pm[key]["per_representation"][rp]["false_positive_rate_fpr"], pm[key]["per_representation"][rp]["recall"]) for rp in REPS]
        ax.plot([p[0] for p in pts[:4]], [p[1] for p in pts[:4]], color=col, lw=0.8, alpha=0.5)
        for rp, (x, y) in zip(REPS, pts):
            ax.plot(x, y, mk, color=col, ms=5 if rp != "R1" else 6.5, mfc=col if rp == "R1" else "white", mew=1.2)
            if rp in ("R1", "R4"):
                ax.annotate(rp, (x, y), textcoords="offset points", xytext=(-14, 4) if rp == "R1" else (5, -9), fontsize=6.2, color=INK2)
        ax.plot([], [], mk, color=col, label=name)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("Recall (vulnerable flagged)")
    ax.plot([], [], "o", color=INK2, label="filled: R1 whole function"); ax.plot([], [], "o", color=INK2, mfc="white", label="open: R2-R5 slice inputs")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    ax.set_title("Recall vs. false-positive rate")
    save(fig, out, "df89e_fig1_detection_bias")

    # Fig 2: localization raw vs chance-adjusted (top-5)
    lt = json.loads((res / "loc_lift_test.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    xs = {"R1": 0, "R2": 1, "R4": 2}
    for key, name, col, mk in MODELS:
        rows = [t for t in lt if t["model"] == key and t["k"] == 5]
        rr = {t["comparison"].split()[0]: t for t in rows}
        raw = [pm[key]["per_representation"][rp]["top_5"] for rp in ("R1", "R2", "R4")]
        lift = [rows[0]["lift_R1"], rr["R2"]["lift_slice"], rr["R4"]["lift_slice"]]
        axs[0].plot(list(xs.values()), raw, "-" + mk, color=col, label=name)
        axs[1].plot(list(xs.values()), lift, "-" + mk, color=col)
        for i, v in enumerate(raw): axs[0].annotate(f"{v*100:.0f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.2)
        for i, v in enumerate(lift): axs[1].annotate(f"{v*100:+.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.2)
    for ax in axs:
        ax.set_xticks(list(xs.values()), ["R1 whole function", "R2 raw slice", "R4 SRCVUL slice"]); ax.set_xlim(-0.3, 2.3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
    axs[0].set_title("(a) Raw top-5 localization (%)"); axs[0].set_ylim(0, 0.42); axs[0].legend(loc="upper left", frameon=False)
    axs[1].axhline(0, color=INK2, lw=0.8); axs[1].set_title("(b) Top-5 lift over chance (points)"); axs[1].set_ylim(-0.005, 0.11)
    save(fig, out, "df89e_fig2_localization_chance")

    # Fig 3: RQ2 dose-response per model + efficiency
    p2 = r["RQ2"]["_detail"]["per_model"]["models"]; levels = [0, 10, 20, 40, 60, 80]
    pr = r["RQ2"]["_detail"]["rq2_e2_efficiency_without_loss"]["tost_effectiveness_equivalence"]
    tk = r["RQ2"]["_detail"]["efficiency_reductions"]["tokens"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.4, 1]})
    for key, name, col, mk in MODELS:
        axs[0].plot(levels, [p2[key]["per_condition"][f"dose_{l}"]["macro_f1"] for l in levels], "-" + mk, color=col, label=name)
    axs[0].set_xlabel("Share of slice statements removed (%)"); axs[0].set_ylabel("Macro-F1"); axs[0].set_ylim(0.3, 0.6)
    axs[0].axhline(0.5, color=INK2, lw=0.8, ls="--"); axs[0].text(1, 0.505, "random detector", fontsize=6.2, color=INK2)
    axs[0].set_title("(a) Macro-F1 under slice corruption"); axs[0].legend(loc="lower right", frameon=False, ncol=3)
    ax = axs[1]
    ax.axvspan(-0.05, 0.05, color="#2a78d6", alpha=0.08); ax.axvspan(-0.02, 0.02, color="#2a78d6", alpha=0.12)
    lo, hi = pr["ci90_mean_diff"]; ax.errorbar([pr["mean_diff"]], [0], xerr=[[pr["mean_diff"] - lo], [hi - pr["mean_diff"]]], fmt="o", color=INK, capsize=3)
    ax.set_yticks([]); ax.set_xlim(-0.07, 0.07); ax.set_xlabel("Macro-F1 difference, slice - whole function (90% CI)")
    ax.text(-0.048, 0.35, "pre-registered\nmargin ±0.05", fontsize=6.2, color="#2a78d6"); ax.text(-0.019, -0.45, "±0.02", fontsize=6.2, color="#2a78d6")
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("(b) Equivalence, slice vs. whole function")
    save(fig, out, "df89e_fig3_rq2_dose_efficiency")

    # Fig 4: RQ3 repair -- main set and post-cutoff subset
    d3 = r["RQ3"]["_detail"]["per_model_rep_summary"]; mc = {x["comparison"].split(":")[0]: x for x in r["RQ3"]["_detail"]["mcnemar_slice_vs_whole"]}
    post = defaultdict(lambda: [0, 0])
    for line in open(res / "records_rq3.jsonl"):
        if '"post_cutoff"' not in line: continue
        x = json.loads(line); post[(x["model"], x["unit_id"], x["representation"])][0] += int(bool(x["eliminated_static"]))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    w = 0.36
    for i, (key, name, col, mk) in enumerate(MODELS):
        e1, e4 = d3[f"{key}|R1"]["independent_vulnerability_elimination_rate_rq3"], d3[f"{key}|R4"]["independent_vulnerability_elimination_rate_rq3"]
        axs[0].bar(i - w / 2, e1, w, color="#8a8984", edgecolor="white"); axs[0].bar(i + w / 2, e4, w, color="#2a78d6", edgecolor="white")
        for x0, v in ((i - w / 2, e1), (i + w / 2, e4)): axs[0].text(x0, v + 0.0004, f"{v*100:.1f}", ha="center", fontsize=6.2)
        axs[0].text(i, max(e1, e4) + 0.0032, f"Holm p = {mc[key]['p_holm']:.2f}", ha="center", fontsize=6.2, color=INK2)
        units = sorted({u for (m, u, rp) in post if m == key})
        a1 = [post[(key, u, "R1")][0] >= 2 for u in units]; a4 = [post[(key, u, "R4")][0] >= 2 for u in units]  # majority of 3 runs, as in all RQ3 numbers
        b = sum(y and not x for x, y in zip(a1, a4)); c = sum(x and not y for x, y in zip(a1, a4))
        axs[1].bar(i - w / 2, sum(a1), w, color="#8a8984", edgecolor="white"); axs[1].bar(i + w / 2, sum(a4), w, color="#2a78d6", edgecolor="white")
        for x0, v in ((i - w / 2, sum(a1)), (i + w / 2, sum(a4))): axs[1].text(x0, v + 0.1, str(v), ha="center", fontsize=6.2)
        axs[1].text(i, max(sum(a1), sum(a4)) + 0.85, f"exact p = {mcnemar(b, c):.3f}", ha="center", fontsize=6.2, color=INK2)
    for ax in axs: ax.set_xticks(range(3), [m[1] for m in MODELS])
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.1f}")); axs[0].set_ylim(0, 0.032)
    axs[0].set_title("(a) Vulnerability removed, 1,835 units (%)"); axs[1].set_title("(b) Post-cutoff CVEs repaired (of 35; exploratory)")
    axs[1].set_ylim(0, 6.5); axs[0].bar(0, 0, color="#8a8984", label="whole function (R1)"); axs[0].bar(0, 0, color=INK2, alpha=0.0)
    axs[0].legend(handles=[matplotlib.patches.Patch(color="#8a8984", label="whole function (R1)"), matplotlib.patches.Patch(color="#2a78d6", label="SRCVUL slice (R4)")], loc="upper left", frameon=False, fontsize=6.2)
    save(fig, out, "df89e_fig4_rq3_repair")

    # Fig 5: RQ4 failure causes on whole functions vs slices (detection strata)
    lab = [x for x in csv.DictReader(open(docs / "df89e_rq4_final_labels.csv")) if x["layer"] == "layer1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    names = {"CX": "missing cross-function context", "DS": "flaw outside shown code", "DM": "missed visible flaw", "UN": "undeterminable",
             "DG": "guard dropped by slice", "DU": "unsupported flag", "DC": "flags a guarded operation", "LB": "label problem"}
    pals = {"detection_false_negative": ["#2a78d6", "#eb6834", "#1baf7a", "#8a8984"],
            "detection_false_positive": ["#7b4fa8", "#e0a526", "#8fc6ec", "#c9c4bc"]}
    light = {"#e0a526", "#8fc6ec", "#c9c4bc"}
    for ax, stratum, title in ((axs[0], "detection_false_negative", "(a) Missed vulnerabilities (false negatives)"),
                               (axs[1], "detection_false_positive", "(b) False alarms (false positives)")):
        codes = [c for c, _ in Counter(x["primary"] for x in lab if x["stratum"] == stratum).most_common()]
        for j, rp in enumerate(("R1", "R4")):
            rows = [x for x in lab if x["stratum"] == stratum and x["representation"] == rp]
            cnt = Counter(x["primary"] for x in rows); left = 0
            for k, c in enumerate(codes):
                v = cnt.get(c, 0) / len(rows)
                colr = pals[stratum][k % 4]
                ax.barh(j, v, left=left, color=colr, edgecolor="white", label=names.get(c, c) if j == 0 else None)
                if cnt.get(c, 0): ax.text(left + v / 2, j, str(cnt[c]), ha="center", va="center", fontsize=6.5,
                                          color=INK if colr in light else "white")
                left += v
        ax.set_yticks([0, 1], [f"whole function (n={sum(1 for x in lab if x['stratum']==stratum and x['representation']=='R1')})",
                               f"slice (n={sum(1 for x in lab if x['stratum']==stratum and x['representation']=='R4')})"])
        ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_title(title); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=6.2)
    save(fig, out, "df89e_fig5_rq4_causes")

    # Fig 6 (appendix): power simulation
    pw = r["RQ1"]["_detail"]["rq1_e0_power_sim"]["power_curve"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ns = sorted(int(k) for k in pw); ax.plot(ns, [pw[str(n)] for n in ns], "-o", color="#2a78d6")
    ax.axhline(0.8, color=INK2, ls="--", lw=0.8); ax.text(2100, 0.83, "80% power", fontsize=6.2, color=INK2); ax.axvline(3670, color="#eb6834", lw=0.8); ax.text(3620, 0.25, "our N = 3,670", fontsize=6.2, color="#eb6834", ha="right")
    ax.set_xlabel("Units"); ax.set_ylabel("Power (paired Wilcoxon)"); ax.set_xlim(0, 4000); ax.set_ylim(0, 1.05)
    ax.set_title(f"Power for the pre-registered effect (d = {r['RQ1']['_detail']['rq1_e0_power_sim']['effect_size_cohens_d']:.2f})")
    save(fig, out, "df89e_fig6_power")
    print("wrote", sorted(p.name for p in out.glob("df89e_fig*.png")))


if __name__ == "__main__":
    main()
