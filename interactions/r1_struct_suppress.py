"""R1: Structural suppression (抑制弱者结构)."""

from __future__ import annotations

from typing import List, Tuple

from ..encoding import Encoding
from ..instance import ProblemInstance


def apply(
    instance: ProblemInstance,
    winner: List[Encoding],
    loser: List[Encoding],
    rate: float = 0.2,
) -> Tuple[List[Encoding], List[Encoding]]:
    if not winner or not loser:
        return winner, loser
    k = max(1, int(len(loser) * rate))
    loser_sorted = sorted(loser, key=lambda e: sum(instance.evaluate(e)), reverse=True)
    winner_sorted = sorted(winner, key=lambda e: sum(instance.evaluate(e)))
    replaced = winner_sorted[:k]
    new_loser = replaced + loser_sorted[k:]
    return winner, new_loser

