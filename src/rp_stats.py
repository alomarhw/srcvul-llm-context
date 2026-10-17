"""Shared statistics for experiments: the tests, corrections and controls a reviewer expects.

Self-contained on purpose (numpy + scipy only): the Engineer ships it into the experiment sandbox as
``rp_stats.py`` so generated code never hand-rolls statistics. Typical use::

    from rp_stats import Results
    res = Results()
    res.metric("RQ1", "vpr", 0.051, arm="full_agent")
    res.metric("RQ1", "vpr", 0.047, arm="ungrounded", baseline=True)
    res.paired_counts("RQ1", "full vs ungrounded (touch)", full_counts, base_counts, family="RQ1-touch",
                      metric="touch", baseline="ungrounded")          # Wilcoxon on per-unit counts
    res.control("RQ1", "oracle accepts official fixes", kind="positive",
                expected="150/150", observed=f"{k}/150", passed=(k == 150))
    res.write("results/results.json")                                  # Holm per family, then write

Every test lands in ``results[rq].statistical_tests.tests`` with a ``name``, ``metric``, ``baseline``
and (after :meth:`Results.finalize`) a Holm-adjusted ``p_holm`` within its ``family``; the paper's claim
checker matches each claim to its own comparison through these labels. Controls land in
``results[rq].controls`` and gate the run: a control that did not behave fails it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import stats as _st

# --------------------------------------------------------------------------------------------
# Proportions and effect sizes
# --------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a proportion k/n."""
    if n <= 0:
        return {"p": None, "lo": None, "hi": None, "k": k, "n": n}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": int(k), "n": int(n)}


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def bootstrap_ci(values: Sequence[float], stat: Callable = np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI of ``stat`` over ``values``."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"estimate": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    return {"estimate": float(stat(x)), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2))}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap CI of mean(a - b) over paired units (same items, two conditions)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, np.mean, n_boot, alpha, seed)


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test on paired binary outcomes (same units under two conditions)."""
    a = [bool(x) for x in a]
    b = [bool(x) for x in b]
    if len(a) != len(b):
        raise ValueError("mcnemar_exact needs paired outcomes of equal length")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    p = float(_st.binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"test": "exact McNemar", "a_only": a_only, "b_only": b_only, "p_value": p, "n": len(a)}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test on paired per-unit scores (e.g. successes out of k repeats per CVE),
    with the matched-pairs rank-biserial correlation r as effect size (+1: a always better)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("wilcoxon_paired needs paired scores of equal length")
    d = a - b
    nz = d[d != 0]
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    if nz.size == 0:
        return {"test": "Wilcoxon signed-rank", "p_value": 1.0, "effect_size": 0.0, "effect": "rank-biserial r",
                "better": 0, "worse": 0, "n": int(d.size)}
    ranks = _st.rankdata(np.abs(nz))
    r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    # Zero differences are dropped BEFORE testing, so small samples get the exact null distribution
    # (a normal approximation on a handful of non-zero pairs, which SciPy uses when zeros are passed
    # in, can understate p badly: 4-vs-0 gives 0.046 approximate vs 0.125 exact).
    method = "exact" if nz.size <= 50 else "approx"
    p = float(_st.wilcoxon(nz, method=method).pvalue)
    return {"test": "Wilcoxon signed-rank", "method": method, "p_value": p, "effect_size": r,
            "effect": "rank-biserial r", "better": better, "worse": worse, "n": int(d.size)}


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two independent samples (e.g. per-unit scores of two different CVE sets). Repeats of one unit
    must be averaged first so they are never counted as independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = _st.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * float(res.statistic) / (a.size * b.size) if a.size and b.size else 0.0
    return {"test": "Mann-Whitney U", "p_value": float(res.pvalue), "effect_size": -r,
            "effect": "rank-biserial r", "n1": int(a.size), "n2": int(b.size)}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family {name: p}."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None), key=lambda t: t[0])
    m = len(items)
    out: dict = {}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def min_detectable_difference(n: int, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference in proportions a paired/independent comparison of n units can detect at the
    given power (normal approximation) — report it next to a null result on a small sample."""
    if n <= 0:
        return float("nan")
    z = _st.norm.ppf(1 - alpha / 2) + _st.norm.ppf(power)
    return float(z * math.sqrt(2 * p0 * (1 - p0) / n))


# --------------------------------------------------------------------------------------------
# Leakage checks
# --------------------------------------------------------------------------------------------

_TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|\S")


def _shingles(text: str, k: int = 5) -> set:
    toks = _TOK.findall(text or "")
    return {tuple(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def jaccard(a: str, b: str, k: int = 5) -> float:
    sa, sb = _shingles(a, k), _shingles(b, k)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def near_duplicate_pairs(queries: Sequence[str], corpus: Sequence[str], threshold: float = 0.9,
                         k: int = 5) -> list[tuple[int, int, float]]:
    """(query index, corpus index, jaccard) for every pair at or above ``threshold``. Use it to keep a
    test item's near-copies out of training data or a retrieval corpus."""
    corpus_sh = [_shingles(c, k) for c in corpus]
    out = []
    for i, q in enumerate(queries):
        sq = _shingles(q, k)
        for j, sc in enumerate(corpus_sh):
            if sq and sc:
                small, large = (sq, sc) if len(sq) <= len(sc) else (sc, sq)
                if len(small) / len(large) < threshold:  # Jaccard ≤ |small|/|large|: cannot reach it
                    continue
                s = len(sq & sc) / len(sq | sc)
                if s >= threshold:
                    out.append((i, j, s))
    return out


def group_overlap(train_groups: Iterable, test_groups: Iterable) -> list:
    """Groups (e.g. CVE ids, projects) present on both sides of a split — must be empty."""
    return sorted(set(train_groups) & set(test_groups), key=str)


# --------------------------------------------------------------------------------------------
# Results builder
# --------------------------------------------------------------------------------------------


class Results:
    """Builds ``results.json`` in the shape the pipeline reads: per RQ ``metrics``,
    ``baseline_comparison``, ``statistical_tests`` (named tests, Holm per family), ``controls`` and
    raw per-unit ``outcomes`` for re-analysis."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.data: dict = {}

    def _rq(self, rq: str) -> dict:
        return self.data.setdefault(rq, {"metrics": {}, "baseline_comparison": {},
                                         "statistical_tests": {"tests": []}, "controls": [], "outcomes": {}})

    def metric(self, rq: str, name: str, value: float, arm: Optional[str] = None, baseline: bool = False,
               ci: Optional[dict] = None) -> None:
        """Record a metric for the proposed method (default) or a named baseline arm."""
        r = self._rq(rq)
        target = r["baseline_comparison"].setdefault(arm or "baseline", {}) if baseline else r["metrics"]
        target[name] = float(value)
        if ci:
            r.setdefault("confidence_intervals", {})[f"{arm or 'method'}:{name}"] = ci

    def outcomes(self, rq: str, arm: str, values: Sequence) -> None:
        """Raw per-unit outcomes (one entry per unit, e.g. successes out of k repeats per CVE)."""
        self._rq(rq)["outcomes"][arm] = [float(v) for v in values]

    def add_test(self, rq: str, name: str, result: dict, *, family: Optional[str] = None,
                 metric: str = "", baseline: str = "", subset: str = "") -> dict:
        row = {"name": name, "metric": metric, "baseline": baseline, "subset": subset,
               "family": family or rq, **result}
        self._rq(rq)["statistical_tests"]["tests"].append(row)
        return row

    def paired_binary(self, rq, name, a, b, **labels) -> dict:
        return self.add_test(rq, name, mcnemar_exact(a, b), **labels)

    def paired_counts(self, rq, name, a, b, **labels) -> dict:
        return self.add_test(rq, name, wilcoxon_paired(a, b), **labels)

    def independent(self, rq, name, a, b, **labels) -> dict:
        return self.add_test(rq, name, mann_whitney(a, b), **labels)

    def control(self, rq: str, name: str, *, kind: str, expected, observed, passed: bool) -> None:
        """A control and whether it behaved. kind: positive (the experiment can detect an effect),
        negative (a known-null case reads as null), calibration (a metric/oracle on known cases), or
        contamination (a post-cutoff split, variants, or a membership probe)."""
        self._rq(rq)["controls"].append({"name": name, "kind": kind, "expected": str(expected),
                                         "observed": str(observed), "passed": bool(passed)})

    def finalize(self) -> dict:
        """Holm-adjust p-values within each family (across RQs) and set RQ-level summary fields."""
        families: dict = {}
        for rq, r in self.data.items():
            for i, t in enumerate(r["statistical_tests"]["tests"]):
                if isinstance(t.get("p_value"), (int, float)):
                    families.setdefault(t["family"], {})[(rq, i)] = t["p_value"]
        for fam in families.values():
            for (rq, i), p_adj in holm(fam).items():
                self.data[rq]["statistical_tests"]["tests"][i]["p_holm"] = p_adj
        for r in self.data.values():
            tests = r["statistical_tests"]["tests"]
            if tests:
                best = min(tests, key=lambda t: t.get("p_holm", t.get("p_value", 1.0)))
                r["statistical_tests"].update({"p_value": best.get("p_value"), "p_holm": best.get("p_holm"),
                                               "effect_size": best.get("effect_size")})
        return self.data

    def write(self, path: str = "results/results.json", extra: Optional[dict] = None) -> dict:
        data = self.finalize()
        out = {**data, **(extra or {})}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=2, default=float))
        return out


SANDBOX_MODULE = "rp_stats.py"


def stats_artifact() -> dict:
    """This module as a code artifact for the experiment sandbox (importable as ``rp_stats``)."""
    return {"filename": SANDBOX_MODULE, "content": Path(__file__).read_text(encoding="utf-8"),
            "language": "python", "purpose": "Shared statistics: tests, Holm, CIs, controls, leakage checks."}
