"""Deterministic decoding and schedule table (解码与调度表)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .encoding import Encoding
from .instance import ProblemInstance
from .types import ScheduleEntry


@dataclass(frozen=True)
class DecodedSolution:
    encoding: Encoding
    schedule: List[ScheduleEntry]
    completion_time: List[float]


def decode(instance: ProblemInstance, enc: Encoding) -> DecodedSolution:
    """
    确定性解码（最小口径，统一用于评价）:
    - 各工厂按 job_sequence 过滤得到作业顺序
    - 每作业依次流经 S 工序
    - 每工序在该工厂的并行机上选定 machine_assignment
    """
    instance.validate()
    F, J, S = instance.num_factories, instance.num_jobs, instance.num_stages
    if len(enc.factory_assignment) != J:
        raise ValueError("factory_assignment length mismatch")
    if sorted(enc.job_sequence) != list(range(J)):
        raise ValueError("job_sequence must be a permutation of jobs")
    if len(enc.machine_assignment) != J or any(len(row) != S for row in enc.machine_assignment):
        raise ValueError("machine_assignment must be shape [J][S]")

    availability: List[List[List[float]]] = []
    for f in range(F):
        availability.append([[0.0 for _ in range(instance.machines[f][s])] for s in range(S)])

    job_stage_completion = [[0.0 for _ in range(S)] for _ in range(J)]
    schedule: List[ScheduleEntry] = []

    for f in range(F):
        jobs_f = [j for j in enc.job_sequence if enc.factory_assignment[j] == f]
        for j in jobs_f:
            for s in range(S):
                m_count = instance.machines[f][s]
                m = enc.machine_assignment[j][s] % m_count
                prev = 0.0 if s == 0 else job_stage_completion[j][s - 1]
                start = max(prev, availability[f][s][m])
                end = start + float(instance.processing_time[j][s])
                availability[f][s][m] = end
                job_stage_completion[j][s] = end
                schedule.append(
                    ScheduleEntry(
                        job=j,
                        factory=f,
                        stage=s,
                        machine=m,
                        start=start,
                        end=end,
                    )
                )

    completion = [job_stage_completion[j][S - 1] for j in range(J)]
    return DecodedSolution(encoding=enc, schedule=schedule, completion_time=completion)

