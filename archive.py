"""Elite archive EA / non-dominated maintenance (精英档案)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .encoding import Encoding
from .types import Objectives


def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


@dataclass
class ArchiveEntry:
    encoding: Encoding
    objectives: Objectives
    source: str | None = None


class EliteArchive:
    """
    全局精英档案 EA:
    - 保持非支配解
    - 目前不做 crowding 截断（后续再加）
    """

    def __init__(self) -> None:
        self.entries: List[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def as_objective_matrix(self) -> List[Objectives]:
        return [e.objectives for e in self.entries]

    def update(self, candidates: Iterable[ArchiveEntry]) -> Tuple[int, int]:
        added = 0
        removed = 0
        for c in candidates:
            if any(dominates(e.objectives, c.objectives) for e in self.entries):
                continue

            before = len(self.entries)
            self.entries = [e for e in self.entries if not dominates(c.objectives, e.objectives)]
            removed += before - len(self.entries)

            if not any(e.objectives == c.objectives and e.encoding == c.encoding for e in self.entries):
                self.entries.append(c)
                added += 1

        return added, removed

