"""Spacing metric."""

from __future__ import annotations

import math
from typing import Iterable, List

from ..types import Objectives


def spacing(points: Iterable[Objectives]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    dists: List[float] = []
    for i, a in enumerate(pts):
        best = float("inf")
        for j, b in enumerate(pts):
            if i == j:
                continue
            dist = math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))
            if dist < best:
                best = dist
        dists.append(best)
    mean = sum(dists) / len(dists)
    return math.sqrt(sum((d - mean) ** 2 for d in dists) / len(dists))
