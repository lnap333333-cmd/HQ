"""Marginal contribution MC and overlap/corr helpers."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from ..types import Objectives


def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated(points: Iterable[Objectives]) -> List[Objectives]:
    pts = list(points)
    nd: List[Objectives] = []
    for p in pts:
        if any(dominates(q, p) for q in pts if q is not p):
            continue
        nd.append(p)
    return nd


def overlap_ratio(a: Iterable[Objectives], b: Iterable[Objectives], all_pts: Iterable[Objectives]) -> float:
    A = set(a)
    B = set(b)
    ALL = set(all_pts)
    if not ALL:
        return 0.0
    return len(A & B) / len(ALL)


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    n = min(len(x), len(y))
    x = x[-n:]
    y = y[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)) * sum((y[i] - my) ** 2 for i in range(n)))
    return 0.0 if den == 0 else num / den
