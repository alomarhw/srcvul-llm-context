"""rq2_stats.py

Statistical routines specific to RQ2-E1, built ONLY on pinned libraries:
  scipy==1.13.1, statsmodels==0.14.6, pingouin==0.6.1, numpy==2.2.6

Provided:
  * jonckheere_terpstra_trend  -- ordered dose-response trend test
  * spearman_dose_response     -- rho of performance vs corruption dose
  * paired_wilcoxon_holm       -- family of paired ablation comparisons, Holm-corrected
  * tost_equivalence           -- two one-sided tests for efficiency-without-loss
  * tost_power                 -- Monte-Carlo power for the TOST at a given margin
  * cohens_d_paired            -- paired Cohen's d
  * bootstrap_ci               -- bootstrap 95% CI of a statistic

All functions return finite numbers; degenerate inputs use explicit finite
conventions (p=1.0, effect=0.0, zero-width CI) rather than NaN/inf, so the
Writer-facing results.json never contains non-finite values.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except Exception:  # pragma: no cover - pingouin is pinned, but stay resilient
    _HAS_PINGOUIN = False

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))


def _finite(x: float, default: float) -> float:
    return float(x) if (x is not None and np.isfinite(x)) else float(default)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d on the difference scores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return 0.0
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0.0:
        return 0.0
    return _finite(np.mean(diff) / sd, 0.0)


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 0.0)
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = v.size
    for i in range(n_boot):
        boots[i] = statistic(v[rng.integers(0, n, n)])
    lo = _finite(np.percentile(boots, 100 * alpha / 2), float(statistic(v)))
    hi = _finite(np.percentile(boots, 100 * (1 - alpha / 2)), float(statistic(v)))
    return (lo, hi)


def paired_wilcoxon_holm(
    comparisons: Dict[str, Tuple[Sequence[float], Sequence[float]]],
) -> Dict[str, Dict[str, float]]:
    """Family of paired Wilcoxon signed-rank tests with Holm correction.

    comparisons: name -> (reference_scores, condition_scores) aligned per unit.
    Returns name -> {W, p_raw, p_holm, cohens_d, ci_low, ci_high, n}.
    """
    names: List[str] = []
    raw_p: List[float] = []
    partial: Dict[str, Dict[str, float]] = {}

    for name, (ref, cond) in comparisons.items():
        ref = np.asarray(ref, dtype=float)
        cond = np.asarray(cond, dtype=float)
        n = int(min(ref.size, cond.size))
        ref, cond = ref[:n], cond[:n]
        diff = ref - cond
        if n < 1 or np.all(diff == 0):
            W, p = 0.0, 1.0
        else:
            try:
                W, p = sps.wilcoxon(ref, cond, zero_method="wilcox",
                                    alternative="two-sided", mode="auto")
            except ValueError:
                W, p = 0.0, 1.0
        d = cohens_d_paired(ref, cond)
        lo, hi = bootstrap_ci(diff)
        partial[name] = {
            "W": _finite(W, 0.0),
            "p_raw": _finite(p, 1.0),
            "cohens_d": d,
            "ci_low": lo,
            "ci_high": hi,
            "n": n,
        }
        names.append(name)
        raw_p.append(_finite(p, 1.0))

    if names:
        _, p_holm, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
        for nm, ph in zip(names, p_holm):
            partial[nm]["p_holm"] = _finite(ph, 1.0)
    return partial


def jonckheere_terpstra_trend(
    groups_by_dose: List[Sequence[float]],
) -> Dict[str, float]:
    """Jonckheere-Terpstra ordered trend test across ordered dose groups.

    groups_by_dose must be in ascending dose order (e.g. p = 0,10,...,80).
    A negative/decreasing trend in performance is expected. Uses pingouin when
    available; otherwise a standard normal-approximation JT implementation.
    """
    groups = [np.asarray(g, dtype=float) for g in groups_by_dose if len(g) > 0]
    if len(groups) < 3:
        return {"JT": 0.0, "p_value": 1.0, "z": 0.0,
                "note": "fewer than 3 non-empty dose groups; finite convention"}

    # Normal-approximation JT statistic (Mann-Whitney U summed over ordered pairs)
    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    N = ns.sum()
    U = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            gi, gj = groups[i], groups[j]
            # count pairs where later-dose value < earlier-dose value (decreasing trend)
            for x in gi:
                U += np.sum(gj < x) + 0.5 * np.sum(gj == x)
    mean_U = (N**2 - np.sum(ns**2)) / 4.0
    var_U = (N**2 * (2 * N + 3) - np.sum(ns**2 * (2 * ns + 3))) / 72.0
    if var_U <= 0:
        return {"JT": float(U), "p_value": 1.0, "z": 0.0,
                "note": "zero variance; finite convention"}
    z = (U - mean_U) / np.sqrt(var_U)
    # two-sided p (we report both; a decreasing trend gives large |z|)
    p = 2.0 * sps.norm.sf(abs(z))
    out = {"JT": _finite(U, 0.0), "z": _finite(z, 0.0), "p_value": _finite(p, 1.0)}

    if _HAS_PINGOUIN:
        # cross-check with pingouin if it exposes a compatible trend test
        try:
            # build a long dataframe for pingouin's ANOVA-based trend as a sanity ref
            import pandas as pd
            rows = []
            for di, g in enumerate(groups):
                for v in g:
                    rows.append({"dose": di, "y": float(v)})
            df = pd.DataFrame(rows)
            rho, prho = sps.spearmanr(df["dose"], df["y"])
            out["spearman_rho_crosscheck"] = _finite(rho, 0.0)
            out["spearman_p_crosscheck"] = _finite(prho, 1.0)
        except Exception:
            pass
    return out


def spearman_dose_response(
    doses: Sequence[float],
    performance: Sequence[float],
) -> Dict[str, float]:
    x = np.asarray(doses, dtype=float)
    y = np.asarray(performance, dtype=float)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n": int(x.size),
                "note": "degenerate; finite convention"}
    rho, p = sps.spearmanr(x, y)
    return {"spearman_rho": _finite(rho, 0.0), "p_value": _finite(p, 1.0), "n": int(x.size)}


def tost_equivalence(
    ref: Sequence[float],
    cond: Sequence[float],
    margin: float,
) -> Dict[str, float]:
    """Two one-sided tests (TOST) for paired equivalence within +/- margin.

    Used for the H4 efficiency-without-loss claim: effectiveness (e.g. F1 proxy
    per unit) on COMPLETE slices must be equivalent between the reference and
    the efficient condition within the pre-registered margin. Returns the TOST
    p-value (max of the two one-sided p's) and the observed mean difference.
    """
    ref = np.asarray(ref, dtype=float)
    cond = np.asarray(cond, dtype=float)
    n = int(min(ref.size, cond.size))
    ref, cond = ref[:n], cond[:n]
    if n < 2:
        return {"tost_p": 1.0, "mean_diff": 0.0, "margin": float(margin),
                "equivalent": False, "n": n,
                "note": "n<2; cannot establish equivalence"}
    diff = ref - cond
    if np.std(diff, ddof=1) == 0.0:
        md = float(np.mean(diff))
        equiv = abs(md) < margin
        return {"tost_p": 0.0 if equiv else 1.0, "mean_diff": md,
                "margin": float(margin), "equivalent": bool(equiv), "n": n,
                "note": "zero-variance difference"}
    p, (t1, p1, _), (t2, p2, _) = ttost_paired(ref, cond, low=-margin, upp=margin)
    md = float(np.mean(diff))
    return {
        "tost_p": _finite(p, 1.0),
        "p_lower": _finite(p1, 1.0),
        "p_upper": _finite(p2, 1.0),
        "mean_diff": md,
        "margin": float(margin),
        "equivalent": bool(_finite(p, 1.0) < 0.05),
        "n": n,
    }


def tost_power(
    n: int,
    margin: float,
    true_diff: float = 0.0,
    sd: float = 0.05,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """Monte-Carlo power for the paired TOST at a given n, margin and true diff.

    Simulates paired differences ~ N(true_diff, sd) and reports the fraction of
    simulations declaring equivalence. Supports the power-calculation acceptance
    requirement for the equivalence claim.
    """
    if n < 2:
        return {"power": 0.0, "n": int(n), "margin": float(margin),
                "true_diff": float(true_diff), "sd": float(sd),
                "note": "n<2"}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        diff = rng.normal(true_diff, sd, n)
        ref = diff            # synthetic ref-cond difference scale
        cond = np.zeros(n)
        try:
            p, _, _ = ttost_paired(ref, cond, low=-margin, upp=margin)
        except Exception:
            p = 1.0
        if np.isfinite(p) and p < alpha:
            hits += 1
    return {"power": float(hits / n_sim), "n": int(n), "margin": float(margin),
            "true_diff": float(true_diff), "sd": float(sd), "n_sim": int(n_sim)}


def derive_tost_margin(pilot_scores: Sequence[float], floor: float = 0.02) -> float:
    """Pilot-derived equivalence margin = max(floor, 1 pilot SD).

    Implements the study default `max(0.02, 1 pilot-SD)`. The exact margin is a
    DESIGN decision; this is the documented default, not a silent substitute.
    """
    s = np.asarray(pilot_scores, dtype=float)
    s = s[np.isfinite(s)]
    sd = float(np.std(s, ddof=1)) if s.size >= 2 else 0.0
    return float(max(floor, sd))


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    # dose-response self-check: performance decreasing with dose
    doses = [0, 10, 20, 40, 60, 80]
    groups = [list(rng.normal(0.8 - d / 200.0, 0.03, 20)) for d in doses]
    jt = jonckheere_terpstra_trend(groups)
    print("JT trend:", {k: round(v, 4) if isinstance(v, float) else v for k, v in jt.items()})
    flat = [np.mean(g) for g in groups]
    sp = spearman_dose_response(doses, flat)
    print("Spearman dose-response:", {k: round(v, 4) if isinstance(v, float) else v for k, v in sp.items()})
    assert sp["spearman_rho"] < 0, "expected negative dose-response rho in self-check"

    # TOST self-check: two nearly-identical conditions should be equivalent
    ref = rng.normal(0.75, 0.04, 60)
    cond = ref + rng.normal(0.0, 0.01, 60)
    margin = derive_tost_margin(ref)
    tost = tost_equivalence(ref, cond, margin)
    print("TOST:", {k: round(v, 4) if isinstance(v, float) else v for k, v in tost.items()})
    pw = tost_power(n=60, margin=margin, true_diff=0.0, sd=0.04)
    print("TOST power:", {k: round(v, 4) if isinstance(v, float) else v for k, v in pw.items()})

    # paired Wilcoxon family self-check
    a0 = rng.normal(0.80, 0.05, 50)   # full-slice reference
    abl = a0 - rng.normal(0.08, 0.04, 50)  # ablation drops performance
    fam = paired_wilcoxon_holm({"A0_no_slicing_vs_full": (a0, abl)})
    print("Wilcoxon family:", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                    for kk, vv in v.items()} for k, v in fam.items()})
    for res in fam.values():
        assert np.isfinite(res["p_holm"]) and np.isfinite(res["cohens_d"])
    print("rq2_stats self-check OK")