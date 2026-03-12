"""C2: Rhythm-embedded cooperation (双向精英互换)."""

from __future__ import annotations

from typing import List, Tuple

from ..encoding import Encoding
from ..instance import ProblemInstance


def apply(
    instance: ProblemInstance,
    pop_a: List[Encoding],
    pop_b: List[Encoding],
    rate: float = 0.1,
) -> Tuple[List[Encoding], List[Encoding]]:
    if not pop_a or not pop_b:
        return pop_a, pop_b
    k = max(1, int(min(len(pop_a), len(pop_b)) * rate))
    a_best = sorted(pop_a, key=lambda e: sum(instance.evaluate(e)))[:k]
    b_best = sorted(pop_b, key=lambda e: sum(instance.evaluate(e)))[:k]
    a_rest = sorted(pop_a, key=lambda e: sum(instance.evaluate(e)))[k:]
    b_rest = sorted(pop_b, key=lambda e: sum(instance.evaluate(e)))[k:]
    return a_best + b_rest, b_best + a_rest

