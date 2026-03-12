"""Hypervolume (HV) metric (近似版)."""

from __future__ import annotations

from typing import Iterable, Tuple

from ..types import Objectives


def hv_approx(points: Iterable[Objectives], ref: Objectives) -> float:
    """
    Simple HV approximation for 3 objectives by summing cuboids to reference.
    Not exact; intended as lightweight placeholder.
    """
    total = 0.0
    for p in points:
        dx = max(0.0, ref[0] - p[0])
        dy = max(0.0, ref[1] - p[1])
        dz = max(0.0, ref[2] - p[2])
        total += dx * dy * dz
    return total

