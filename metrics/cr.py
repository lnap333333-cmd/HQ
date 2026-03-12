"""Coverage rate (CR) metric (简化版)."""

from __future__ import annotations

from typing import Iterable

from ..types import Objectives


def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def coverage_rate(a: Iterable[Objectives], b: Iterable[Objectives]) -> float:
    A = list(a)
    B = list(b)
    if not A or not B:
        return 0.0
    count = 0
    for b_obj in B:
        if any(dominates(a_obj, b_obj) for a_obj in A):
            count += 1
    return count / len(B)

