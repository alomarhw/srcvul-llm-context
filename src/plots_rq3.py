"""
plots_rq3.py -- RQ3 figures from results/results.json and census_manifest.json.

Requirements (pinned): matplotlib==3.9.1, numpy==2.2.6

Produces (contract successCriteria.mustProduce for RQ3):
  figures/graph_slice_worked_example.png  (method worked example; rendered from
                                           the real slice pipeline)
  figures/rq3_e0_results.png              (census: honest level-4 N)
  figures/rq3_e1_results.png              (repair hierarchy: slice R4 vs whole R1
                                           elimination, per model, with McNemar sig.)
  figures/rq3_e2_results.png              (contamination generalization delta + CIs)
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
    """Representative program -> typed dependence graph -> slicing criterion ->
    backward slice (kept vs removed). Rendered from the actual slice pipeline."""
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
    ax1.set_title("Slice-local repair (RQ3): backward slice over {buf, src}")
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
    ax1.text(0.02, y - 0.02, "repair edits on KEEP lines are reintegrated into",
             fontsize=8, color="#444444", transform=ax1.transAxes)
    ax1.text(0.02, y - 0.10, "the ORIGINAL whole function (srcDiff reintegration)",
             fontsize=8, color="#444444", transform=ax1.transAxes)

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
    print("[plots_rq3] wrote figures/graph_slice_worked_example.png")


def fig_e0_census():
    FIGS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "census_manifest.json"
    fig, ax = plt.subplots(figsize=(8, 5))
    if p.exists():
        man = json.loads(p.read_text())
        counts = man.get("level_counts", {})
        levels = ["0", "1", "2", "3", "4a", "4b"]
        vals = [counts.get(l, 0) for l in levels]
        colors = ["#8c959f", "#8c959f", "#0969da", "#0969da", "#1a7f37", "#1a7f37"]
        ax.bar(levels, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
        ax.set_xlabel("feasibility level (0=parse .. 4a/4b=executable PoC/test)")
        ax.set_ylabel("unit count")
        ax.set_title(f"RQ3-E0 feasibility census "
                     f"(executable level-4 N = {man.get('executable_level4_N', 0)})")
    else:
        ax.text(0.5, 0.5, "census_manifest.json not found", ha="center")
    _save(fig, FIGS / "rq3_e0_results.png")
    print("[plots_rq3] wrote figures/rq3_e0_results.png")


def fig_e1(res):
    detail = res["RQ3"]["_detail"]
    summary = detail.get("per_model_rep_summary", {})
    mcn = {m["comparison"].split(":")[0]: m
           for m in detail.get("mcnemar_slice_vs_whole", [])}
    models = detail.get("_models", None)
    # derive model list from summary keys
    model_set = sorted({k.split("|")[0] for k in summary})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_set))
    r1 = [summary.get(f"{m}|R1", {}).get("independent_vulnerability_elimination_rate_rq3", 0.0)
          for m in model_set]
    r4 = [summary.get(f"{m}|R4", {}).get("independent_vulnerability_elimination_rate_rq3", 0.0)
          for m in model_set]
    ax.bar(x - 0.2, r1, 0.4, label="whole-function (R1)", color="#8c959f")
    ax.bar(x + 0.2, r4, 0.4, label="slice+metadata (R4)", color="#1a7f37")
    ax.set_xticks(x, [m.split("-")[0] for m in model_set], rotation=15)
    ax.set_ylabel("independent vulnerability-elimination rate")
    ax.set_title("RQ3-E1: slice-vs-whole repair elimination (independent static oracle)")
    ax.legend()
    # significance stars
    for i, m in enumerate(model_set):
        res_m = mcn.get(m, {})
        if res_m.get("p_holm", 1.0) < 0.05:
            ax.text(i + 0.2, r4[i] + 0.02, "*", ha="center", fontsize=16, color="#d1242f")
    _save(fig, FIGS / "rq3_e1_results.png")
    print("[plots_rq3] wrote figures/rq3_e1_results.png")


def fig_e2(res):
    detail = res["RQ3"]["_detail"]
    cont = detail.get("contamination_generalization", {})
    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(cont.keys())
    if not models:
        ax.text(0.5, 0.5, "no contamination data", ha="center")
        _save(fig, FIGS / "rq3_e2_results.png")
        print("[plots_rq3] wrote figures/rq3_e2_results.png")
        return
    deltas = [cont[m].get("slice_advantage_delta_post", 0.0) for m in models]
    cis = [cont[m].get("slice_advantage_ci95_post", [0.0, 0.0]) for m in models]
    y = np.arange(len(models))
    lowers = [d - c[0] for d, c in zip(deltas, cis)]
    uppers = [c[1] - d for d, c in zip(deltas, cis)]
    colors = ["#1a7f37" if cont[m].get("advantage_persists") else "#8c959f"
              for m in models]
    ax.errorbar(deltas, y, xerr=[lowers, uppers], fmt="o", color="#0969da",
                ecolor="#444444", capsize=4)
    for i, c in enumerate(colors):
        ax.scatter([deltas[i]], [y[i]], color=c, s=80, zorder=3)
    ax.axvline(0, color="#d1242f", ls="--", lw=1)
    ax.set_yticks(y, [m.split("-")[0] for m in models])
    ax.set_xlabel("post-cutoff slice advantage delta (R4 - R1), 95% CI")
    ax.set_title("RQ3-E2: contamination generalization "
                 "(green = CI lower bound > 0, advantage persists)")
    _save(fig, FIGS / "rq3_e2_results.png")
    print("[plots_rq3] wrote figures/rq3_e2_results.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_method_worked_example()
    fig_e0_census()
    res = _load()
    if "RQ3" not in res:
        print("[plots_rq3] RQ3 not in results.json; skipping result figures")
        return
    fig_e1(res)
    fig_e2(res)
    print("[plots_rq3] all RQ3 figures generated")


if __name__ == "__main__":
    main()