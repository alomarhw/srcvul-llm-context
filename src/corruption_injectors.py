"""corruption_injectors.py - slice corruption for robustness/ablation analysis.

Requirements (pinned): numpy==2.2.6

Given a full slice (line set) from slice_pipeline, produce corrupted slices that
drop lines in three category-specific modes plus uniform-random, and score the
incompleteness relative to the dependency oracle (the full slice).
"""
from __future__ import annotations

import random
from typing import Any

RANDOM_SEED = 42


def _rng(seed_offset: int = 0) -> random.Random:
    import os
    base = int(os.environ.get("RP_RANDOM_SEED", RANDOM_SEED))
    return random.Random(base + seed_offset)


def uniform_random_drop(slice_lines: list[int], frac: float, seed_offset: int = 0) -> list[int]:
    """Drop `frac` of the slice lines uniformly at random."""
    if not slice_lines or frac <= 0:
        return list(slice_lines)
    rng = _rng(seed_offset)
    keep_n = max(1, int(round(len(slice_lines) * (1.0 - frac))))
    kept = rng.sample(slice_lines, keep_n)
    return sorted(kept)


def interprocedural_drop(slice_lines: list[int], cfunc_lines: list[int]) -> list[int]:
    """Category-specific: drop lines involving called functions (simulates missing
    inter-procedural dependency resolution)."""
    cf = set(cfunc_lines)
    return sorted([l for l in slice_lines if l not in cf]) or list(slice_lines)


def alias_drop(slice_lines: list[int], ptr_lines: list[int]) -> list[int]:
    """Category-specific: drop pointer/alias lines (simulates imprecise alias
    analysis)."""
    pl = set(ptr_lines)
    return sorted([l for l in slice_lines if l not in pl]) or list(slice_lines)


def cross_file_drop(slice_lines: list[int], frac: float = 0.3, seed_offset: int = 1) -> list[int]:
    """Category-specific: simulate missing cross-file definitions by dropping the
    last `frac` of lines (proxy for out-of-function defs)."""
    if not slice_lines:
        return []
    cut = max(1, int(round(len(slice_lines) * (1.0 - frac))))
    return slice_lines[:cut]


def incompleteness_score(corrupted: list[int], oracle: list[int]) -> float:
    """Fraction of oracle slice lines missing from the corrupted slice."""
    if not oracle:
        return 0.0
    missing = len(set(oracle) - set(corrupted))
    return missing / len(oracle)


if __name__ == "__main__":
    full = [1, 2, 3, 4, 5, 6, 7, 8]
    c = uniform_random_drop(full, 0.5)
    print("corrupted:", c, "incompleteness:", incompleteness_score(c, full))