"""R2: Territorial invasion (领地入侵)."""

from __future__ import annotations

from typing import List, Tuple

from ..encoding import Encoding
from ..instance import ProblemInstance


def apply(
    instance: ProblemInstance,
    winner: List[Encoding],
    loser: List[Encoding],
    rate: float = 0.1,
) -> Tuple[List[Encoding], List[Encoding]]:
    if not winner or not loser:
        return winner, loser
    k = max(1, int(len(loser) * rate))
    winner_sorted = sorted(winner, key=lambda e: sum(instance.evaluate(e)))
    loser_sorted = sorted(loser, key=lambda e: sum(instance.evaluate(e)), reverse=True)
    invaded = winner_sorted[:k]
    new_loser = invaded + loser_sorted[k:]
    return winner, new_loser

