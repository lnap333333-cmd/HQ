"""Objective functions f1/f2/f3 (目标函数，均为最小化)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .decoder import DecodedSolution
from .instance import ProblemInstance
from .types import Objectives


@dataclass(frozen=True)
class ObjectiveBreakdown:
    f1_makespan: float
    f2_weighted_tardiness: float
    f3_factory_load_imbalance: float
    factory_loads: List[float]


def compute_objectives(
    instance: ProblemInstance, decoded: DecodedSolution
) -> Tuple[Objectives, ObjectiveBreakdown]:
    """
    - f1: makespan = max completion time
    - f2: weighted total tardiness = sum w_j * max(0, C_j - d_j)
    - f3: factory load imbalance = std-dev of factory loads (from decoded schedule)
    """
    instance.validate()
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    C = decoded.completion_time
    if len(C) != J:
        raise ValueError("completion_time length mismatch")

    f1 = float(max(C)) if C else 0.0
    f2 = 0.0
    for j in range(J):
        tard = max(0.0, float(C[j]) - float(instance.due_date[j]))
        f2 += float(instance.weight[j]) * tard

    loads = [0.0 for _ in range(F)]
    for e in decoded.schedule:
        loads[e.factory] += float(e.end - e.start)

    mean = sum(loads) / F if F > 0 else 0.0
    var = sum((x - mean) ** 2 for x in loads) / F if F > 0 else 0.0
    f3 = float(math.sqrt(var))

    objs: Objectives = (f1, f2, f3)
    return objs, ObjectiveBreakdown(
        f1_makespan=f1,
        f2_weighted_tardiness=f2,
        f3_factory_load_imbalance=f3,
        factory_loads=loads,
    )

