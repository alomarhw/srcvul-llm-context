"""plots.py - figures for RQ1 from results/results.json.

Requirements (pinned): matplotlib==3.9.1, numpy==2.2.6

Produces:
  figures/graph_slice_worked_example.png  (FIG_METHOD - method worked example)
  figures/rq1_e0_results.png              (FIG0 - power curve)
  figures/rq1_e1_results.png              (FIG1 - F1 / top-5 heatmap by representation)
  figures/rq1_e2_results.png              (reproduced baselines macro-F1 bar)
All read from saved results; the method figure is rendered from the actual slice
pipeline on a representative example.
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
    """FIG_METHOD: representative program -> typed dependence graph -> criterion ->
    backward slice (kept vs removed)."""
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
    # left: source with kept/removed
    ax1.set_title("DepSlice-Prompt: backward slice (criterion = buf,src)")
    ax1.axis("off")
    y = 0.95
    for i, ln in enumerate(lines, 1):
        kept = i in slice_set
        color = "#1a7f37" if kept else "#999999"
        mark = "KEEP" if kept else "drop"
        crit = i in {j for j in meta.slice_lines if any(
            v in ln for v in meta.vrvars)}
        prefix = ">>" if crit else "  "
        ax1.text(0.02, y, f"{prefix}{i:2d} [{mark}] {ln}", color=color,
                 fontsize=9, family="monospace", transform=ax1.transAxes)
        y -= 0.11

    # right: typed dependence graph of slice variables
    ax2.set_title("Typed dependence graph (operation nodes + typed edges)")
    ax2.axis("off")
    nodes = list(dict.fromkeys(meta.vrvars + meta.dvars + meta.cfuncs))[:6] or ["buf"]
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a))
           for n, a in zip(nodes, angles)}
    # edges: data (solid) between vrvars/dvars; call (dashed) to cfuncs
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
    ax2.plot([], [], "--", color="#d1242f", label="call/pointer edge")
    ax2.scatter([], [], c="#ffd33d", edgecolors="black", label="slicing criterion (vrvar)")
    ax2.legend(loc="lower center", fontsize=8, ncol=1)
    FIGS.mkdir(parents=True, exist_ok=True)
    _save(fig, FIGS / "graph_slice_worked_example.png")
    print("[plots] wrote figures/graph_slice_worked_example.png")


def fig_e0_power(res):
    detail = res["RQ1"]["_detail"].get("rq1_e0_power_sim", {})
    curve = detail.get("power_curve", {})
    fig, ax = plt.subplots(figsize=(7, 5))
    if curve:
        ns = sorted(int(k) for k in curve)
        p = [curve[str(n)] if str(n) in curve else curve[n] for n in ns]
        ax.plot(ns, p, "o-", color="#0969da")
        ax.axhline(0.8, ls="--", color="grey", label="0.8 power")
        ax.set_xlabel("sample size (units)")
        ax.set_ylabel("power (paired Wilcoxon)")
        ax.legend()
    ax.set_title("RQ1-E0 Monte Carlo power simulation")
    _save(fig, FIGS / "rq1_e0_results.png")
    print("[plots] wrote figures/rq1_e0_results.png")


def fig_e1_heatmap(res):
    per = res["RQ1"]["_detail"]["per_representation_metrics"]
    reps = ["R1", "R2", "R3", "R4", "R5"]
    metrics = ["macro_f1", "recall", "precision",
               "top_k_localization_accuracy_top_1_top_5_rq1"]
    labels = ["macro-F1", "recall", "precision", "top-k loc"]
    mat = np.array([[per[r][m] for r in reps] for m in metrics])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(reps)), reps)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("RQ1-E1: detection + localization by representation")
    _save(fig, FIGS / "rq1_e1_results.png")
    print("[plots] wrote figures/rq1_e1_results.png")


def fig_e2_baselines(res):
    bc = res["RQ1"]["baseline_comparison"]
    names, vals = [], []
    for name, m in bc.items():
        short = name.split("(")[0].strip()[:24]
        names.append(short)
        vals.append(m["macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d1242f" if "Random" in n else "#0969da" for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("macro-F1")
    ax.set_title("RQ1-E2: reproduced baselines vs representations (macro-F1)")
    ax.invert_yaxis()
    _save(fig, FIGS / "rq1_e2_results.png")
    print("[plots] wrote figures/rq1_e2_results.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_method_worked_example()
    res = _load()
    fig_e0_power(res)
    fig_e1_heatmap(res)
    fig_e2_baselines(res)
    print("[plots] all RQ1 figures generated")


if __name__ == "__main__":
    main()