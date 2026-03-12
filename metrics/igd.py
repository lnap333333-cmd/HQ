"""IGD metric (简化版)."""

from __future__ import annotations

import math
from typing import Iterable, List

from ..types import Objectives


def igd(reference: Iterable[Objectives], approx: Iterable[Objectives]) -> float:
    ref = list(reference)
    app = list(approx)
    if not ref or not app:
        return 0.0
    total = 0.0
    for r in ref:
        best = float("inf")
        for a in app:
            dist = math.sqrt(sum((r[i] - a[i]) ** 2 for i in range(3)))
            if dist < best:
                best = dist
        total += best
    return total / len(ref)

