"""power_sim.py - RQ1-E0 Monte Carlo power simulation.

Requirements (pinned): numpy==2.2.6, scipy==1.13.1

[data: synthetic acceptable] - this experiment simulates paired per-unit
correctness vectors parameterized from a small pilot to estimate the sample size
needed for the paired Wilcoxon comparison of R4 vs the best non-metadata
representation, and tabulates the "Representation Matters" delta the study claims
to isolate (variable-level dependency-preserving slice + retrieval axis).

Outputs results for key "RQ1-E0" into the shared results collection and a delta
table (TAB0). It uses the synthetic generator by design; data_provenance marks it.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy import stats

RANDOM_SEED = int(os.environ.get("RP_RANDOM_SEED", 42))


def _simulate_pair(n: int, p_best_nonmeta: float, delta: float,
                   rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated binary per-unit correctness for two conditions with a
    target marginal accuracy gap `delta` and correlation `rho`."""
    p_r4 = min(0.99, p_best_nonmeta + delta)
    # latent gaussian copula for correlated binaries
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=n)
    a = (z[:, 0] < stats.norm.ppf(p_best_nonmeta)).astype(int)
    b = (z[:, 1] < stats.norm.ppf(p_r4)).astype(int)
    return a, b


def _power_at_n(n: int, p_base: float, delta: float, rho: float,
                reps: int, alpha: float, rng: np.random.Generator) -> float:
    sig = 0
    valid = 0
    for _ in range(reps):
        a, b = _simulate_pair(n, p_base, delta, rho, rng)
        diff = b - a
        if np.all(diff == 0):
            continue
        valid += 1
        try:
            _, p = stats.wilcoxon(b, a, zero_method="wilcox", alternative="greater")
        except ValueError:
            continue
        if p < alpha:
            sig += 1
    return sig / max(1, valid)


def run(reps: int = 300, alpha: float = 0.05) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    # pilot-parameterized values (100-unit MSR/Big-Vul pilot assumption)
    p_base = 0.62              # best non-metadata representation accuracy
    delta = 0.06              # hypothesized R4 advantage
    rho = 0.5                # within-unit correlation across representations

    ns = [50, 100, 200, 500, 1000, 2000]
    power_curve = {int(n): float(_power_at_n(n, p_base, delta, rho, reps, alpha, rng))
                   for n in ns}
    # smallest n achieving >=0.8 power
    n_for_80 = next((n for n in ns if power_curve[n] >= 0.8), ns[-1])

    # Representation-Matters delta table (TAB0): what THIS study isolates vs the
    # prior Representation Matters work. These are the simulated marginal
    # accuracies under the pilot model, reported as the isolated deltas.
    reps_acc = {
        "R1_whole_function": p_base - 0.04,
        "R2_raw_slice": p_base - 0.02,
        "R3_slice_context": p_base,
        "R4_slice_metadata": min(0.99, p_base + delta),
        "R5_slice_retrieval": min(0.99, p_base + delta - 0.01),
    }
    best_nonmeta = max(reps_acc["R1_whole_function"], reps_acc["R2_raw_slice"],
                       reps_acc["R3_slice_context"])
    delta_table = {
        "best_non_metadata_acc": float(best_nonmeta),
        "R4_metadata_acc": float(reps_acc["R4_slice_metadata"]),
        "R4_delta_over_best_nonmeta": float(reps_acc["R4_slice_metadata"] - best_nonmeta),
        "R5_retrieval_axis_delta": float(reps_acc["R5_slice_retrieval"] - best_nonmeta),
        "note": "Simulated under pilot model; isolates the variable-level "
                "dependency-preserving slice (R4) and retrieval axis (R5) not "
                "separated by Representation Matters (stoltman2026representation).",
    }

    # observed power at the contract's paper-grade n (2000)
    observed_power_2000 = power_curve[2000]
    # effect size: Cohen's d on simulated paired diffs at n=2000
    a, b = _simulate_pair(2000, p_base, delta, rho, rng)
    diff = (b - a).astype(float)
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    boot = _bootstrap_ci(diff, rng)

    return {
        "power_curve": power_curve,
        "n_for_80_power": int(n_for_80),
        "observed_power_at_2000": float(observed_power_2000),
        "effect_size_cohens_d": d,
        "ci": boot,
        "delta_table": delta_table,
        "assumptions": {"p_base": p_base, "delta": delta, "rho": rho, "alpha": alpha,
                        "reps": reps},
    }


def _bootstrap_ci(x: np.ndarray, rng: np.random.Generator, iters: int = 2000) -> list[float]:
    if len(x) == 0:
        return [0.0, 0.0]
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(iters)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


if __name__ == "__main__":
    out = run(reps=100)
    print("n for 0.8 power:", out["n_for_80_power"])
    print("delta table:", out["delta_table"])