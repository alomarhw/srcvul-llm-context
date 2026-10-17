"""
plots_rq2.py -- RQ2 figures from results/results.json.

Requirements: matplotlib==3.9.1, numpy==2.2.6

Produces (contract successCriteria.mustProduce for RQ2):
  figures/graph_slice_worked_example.png  (method worked example)
  figures/rq2_e1_results.png              (ablation drops + dose-response curve)
  figures/rq2_e2_results.png              (efficiency-without-loss: tokens/latency/cost + TOST)

The method figure is rendered from the ACTUAL slice pipeline on a representative
example; result figures read saved RQ2 results.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

try:
    import rp_style

    def _save(fig, path):
        rp_style.save(fig, path)
except Exception:
    def _save(fig, path):
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _load():
    p = RESULTS / "results.json"
    if not p.exists():
        raise RuntimeError("results/results.json not found; run main.py first")
    return json.loads(p.read_text())


def fig_method_worked_example():
    """Representative program -> typed dependence graph -> criterion -> backward
    slice (kept vs removed). Rendered from the real slice pipeline."""
    from slice_pipeline import compute_slice

    code = ("void copy(char *dst, const char *src) {\n"
            "    int len = strlen(src);\n"
            "    char buf[8];\n"
            "    strcpy(buf, src);\n"
            "    memcpy(dst, buf, len);\n"
            "    return;\n}")
    meta = compute_slice(code, criterion_vars=["buf", "src"])
    lines = code.splitlines()
    slice_set = set(meta.slice_lines)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("DepSlice (RQ2): backward slice over criterion {buf, src}")
    ax1.axis("off")
    y = 0.95
    for i, ln in enumerate(lines, 1):
        kept = i in slice_set
        color = "#1a7f37" if kept else "#999999"
        mark = "KEEP" if kept else "drop"
        crit = any(v in ln for v in meta.vrvars)
        prefix = ">>" if crit else "  "
        ax1.text(0.02, y, f"{prefix}{i:2d} [{mark}] {ln}", color=color,
                 fontsize=9, family="monospace", transform=ax1.transAxes)
        y -= 0.11

    ax2.set_title("Typed dependence graph (ops + typed edges)")
    ax2.axis("off")
    nodes = list(dict.fromkeys(meta.vrvars + meta.dvars + meta.cfuncs))[:6] or ["buf"]
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a))
           for n, a in zip(nodes, angles)}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            style = "--" if (a in meta.cfuncs or b in meta.cfuncs) else "-"
            col = "#d1242f" if style == "--" else "#0969da"
            ax2.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                     style, color=col, alpha=0.5, lw=1.2)
    for n, (x, yy) in pos.items():
        is_crit = n in meta.vrvars
        fc = "#ffd33d" if is_crit else "#ddf4ff"
        ax2.scatter([x], [yy], s=1400, c=fc, edgecolors="black", zorder=3)
        ax2.text(x, yy, n, ha="center", va="center", fontsize=9, zorder=4)
    ax2.plot([], [], "-", color="#0969da", label="data dependency")
    ax2.plot([], [], "--", color="#d1242f", label="inter-proc/pointer edge")
    ax2.scatter([], [], c="#ffd33d", edgecolors="black", label="slicing criterion (vrvar)")
    ax2.legend(loc="lower center", fontsize=8)
    FIGS.mkdir(parents=True, exist_ok=True)
    _save(fig, FIGS / "graph_slice_worked_example.png")
    print("[plots_rq2] wrote figures/graph_slice_worked_example.png")


def fig_e1(res):
    detail = res["RQ2"]["_detail"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: ablation / category drop vs FULL with CI
    fam = detail.get("ablation_family_wilcoxon_holm", [])
    names = [e["comparison"].split(" vs ")[0] for e in fam]
    drops = [e.get("mean_drop_vs_full", 0.0) for e in fam]
    sig = [e.get("p_holm", 1.0) < 0.05 for e in fam]
    colors = ["#d1242f" if s else "#8c959f" for s in sig]
    ax1.barh(names, drops, color=colors)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("mean per-unit correctness drop vs FULL slice")
    ax1.set_title("RQ2-E1: ablation / category-specific corruption (red = Holm-sig.)")
    ax1.invert_yaxis()

    # right: uniform-random dose-response curve
    dr = detail.get("dose_response", {})
    f1_by = dr.get("f1_by_level", {})
    if f1_by:
        levels = sorted(int(k) for k in f1_by)
        f1 = [f1_by[str(l)] for l in levels]
        ax2.plot(levels, f1, "o-", color="#0969da")
        ax2.set_xlabel("uniform slice corruption (%)")
        ax2.set_ylabel("macro-F1")
        rho = dr.get("spearman_rho_level_vs_correct", 0.0)
        jtp = dr.get("jonckheere_terpstra_p", 1.0)
        ax2.set_title(f"Dose-response  (Spearman rho={rho:.2f}, JT p={jtp:.3f})")
    else:
        ax2.text(0.5, 0.5, "no dose data", ha="center")
    _save(fig, FIGS / "rq2_e1_results.png")
    print("[plots_rq2] wrote figures/rq2_e1_results.png")


def fig_e2(res):
    detail = res["RQ2"]["_detail"]
    eff = detail.get("efficiency_reductions", {})
    tost = detail.get("efficiency_without_loss_tost", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: whole-function (A0) vs FULL slice medians for tokens/latency/cost
    fields = ["tokens", "latency", "cost"]
    a0 = [eff.get(f, {}).get("median_whole_function", 0.0) for f in fields]
    full = [eff.get(f, {}).get("median_full", 0.0) for f in fields]
    # normalize each field to A0=1 for a comparable bar chart
    a0n, fulln = [], []
    for av, fv in zip(a0, full):
        denom = av if av else 1.0
        a0n.append(1.0)
        fulln.append(fv / denom if denom else 0.0)
    x = np.arange(len(fields))
    ax1.bar(x - 0.2, a0n, 0.4, label="whole-function (A0)", color="#8c959f")
    ax1.bar(x + 0.2, fulln, 0.4, label="FULL slice", color="#1a7f37")
    ax1.set_xticks(x, ["tokens", "latency", "cost"])
    ax1.set_ylabel("relative to whole-function (A0=1)")
    ax1.set_title("RQ2-E2: efficiency reductions (slice vs whole-function)")
    ax1.legend()
    for i, f in enumerate(fields):
        if eff.get(f, {}).get("significant_reduction"):
            ax1.text(i + 0.2, fulln[i] + 0.02, "*", ha="center", fontsize=14,
                     color="#d1242f")

    # right: TOST equivalence of effectiveness (FULL vs A0)
    ax2.axis("off")
    if tost.get("ci90"):
        ci = tost["ci90"]
        margin = tost.get("margin", 0.02)
        ax2.set_title("Efficiency-without-loss (TOST on effectiveness)")
        ax2.axhline(0.5, color="grey", lw=0.5)
        ax2.axvspan(-margin, margin, color="#ddf4ff", alpha=0.6)
        ax2.plot(ci, [0.5, 0.5], "o-", color="#0969da", lw=3)
        ax2.plot([tost.get("mean_diff", 0.0)], [0.5], "D", color="#d1242f")
        ax2.axvline(-margin, ls="--", color="#d1242f")
        ax2.axvline(margin, ls="--", color="#d1242f")
        ax2.set_xlim(-max(0.1, margin * 3), max(0.1, margin * 3))
        ax2.set_ylim(0, 1)
        eq = tost.get("equivalent")
        ax2.text(0, 0.75,
                 f"equivalent within ±{margin}: {eq}\n90% CI of mean diff "
                 f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                 ha="center", fontsize=10)
        ax2.axis("on")
        ax2.set_yticks([])
        ax2.set_xlabel("FULL - whole-function effectiveness difference")
    else:
        ax2.text(0.5, 0.5, "insufficient data for TOST", ha="center")
    _save(fig, FIGS / "rq2_e2_results.png")
    print("[plots_rq2] wrote figures/rq2_e2_results.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_method_worked_example()
    res = _load()
    if "RQ2" not in res:
        print("[plots_rq2] RQ2 not in results.json; skipping result figures")
        return
    fig_e1(res)
    fig_e2(res)
    print("[plots_rq2] all RQ2 figures generated")


if __name__ == "__main__":
    main()