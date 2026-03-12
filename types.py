"""Common types and dataclasses (基础数据结构)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

JobId = int
FactoryId = int
StageId = int
MachineId = int

Objectives = Tuple[float, float, float]


@dataclass(frozen=True)
class ScheduleEntry:
    job: JobId
    factory: FactoryId
    stage: StageId
    machine: MachineId
    start: float
    end: float

