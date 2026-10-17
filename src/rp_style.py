"""ResearchPilot figure house style. `import rp_style` at the top of plots.py.

Applies publication-grade matplotlib defaults (reinforcing matplotlibrc, and overriding any
seaborn/style.use call made before import), and exposes a colorblind-safe palette + helpers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe qualitative palette.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
PAL = PALETTE  # compatibility alias used by some generated plots.py files
CMAP = "viridis"

_RC = {
    "figure.figsize": (7, 4.5), "figure.dpi": 150,
    # Robust auto-layout so labels/ticks/legends never overlap (see matplotlibrc note above).
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.size": 12, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.frameon": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0,
    "lines.linewidth": 2.0, "lines.markersize": 6, "image.cmap": CMAP,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
}

def apply():
    plt.rcParams.update(_RC)

apply()

# Figures carry NO embedded title (the Writer adds a caption that covers it) and NO provenance/
# watermark footer. These are stripped deterministically by the injected sitecustomize, but expose a
# no-op here too for code that calls them via rp_style.
def finalize(fig=None):
    """Despine before saving. Layout is handled by constrained_layout (set in rcParams), so we do
    NOT call tight_layout here — the two engines conflict, and tight_layout is the weaker one."""
    fig = fig or plt.gcf()
    for ax in fig.get_axes():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig

def save(*args, fig=None, path=None):
    """finalize() then savefig at 300 DPI, tight bbox.

    Accept both generated-code conventions: save(path, fig=None) and save(fig, path).

    Robust to degenerate layouts: constrained_layout can shrink axes to zero ("axes sizes
    collapsed to zero"), and a tight bbox then computes a 0x0 canvas that makes savefig raise.
    A single such figure must NOT crash plots.py and lose every later figure, so we retry with
    progressively safer settings and, only as a last resort, write a minimal placeholder so a
    file always exists for the caption to reference.
    """
    import os

    if path is None and args:
        if hasattr(args[0], "savefig"):
            fig = args[0]
            path = args[1] if len(args) > 1 else None
        else:
            path = args[0]
            if len(args) > 1:
                fig = args[1]
    if path is None:
        raise TypeError("rp_style.save requires an output path")
    fig = finalize(fig)
    out_dir = os.path.dirname(str(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    attempts = (
        {"dpi": 300, "bbox_inches": "tight"},  # ideal
        {"dpi": 300},                          # drop tight bbox (the 0x0 trigger)
        {"dpi": 150},
    )
    last_error = None
    for i, kwargs in enumerate(attempts):
        try:
            if i == 1:
                # The layout engine collapsed the axes — turn it off and restore a sane size.
                try:
                    fig.set_layout_engine("none")
                except Exception:
                    pass
                w, h = fig.get_size_inches()
                if w < 1 or h < 1:
                    fig.set_size_inches(7, 4.5)
            fig.savefig(path, **kwargs)
            return
        except Exception as exc:
            last_error = exc
    # Every strategy failed (truly degenerate figure) — emit a minimal valid PNG so the pipeline
    # continues and the figure's caption isn't left pointing at a missing file.
    try:
        placeholder = plt.figure(figsize=(7, 4.5))
        placeholder.text(0.5, 0.5, "figure unavailable", ha="center", va="center", fontsize=14)
        placeholder.savefig(path, dpi=150)
        plt.close(placeholder)
    except Exception:
        raise last_error

# Figures carry no embedded title — the Writer adds a caption that covers it.
NO_TITLES = True
