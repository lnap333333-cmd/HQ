"""Three-part solution encoding (三段式编码)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from .instance import ProblemInstance


@dataclass(frozen=True)
class Encoding:
    """
    三段式编码:
    - factory_assignment[j] = 工厂编号
    - job_sequence: 作业全排列
    - machine_assignment[j][s] = 工序s的机器编号
    """

    factory_assignment: List[int]
    job_sequence: List[int]
    machine_assignment: List[List[int]]


def encoding_from_dict(d: Dict) -> Encoding:
    return Encoding(
        factory_assignment=list(map(int, d["factory_assignment"])),
        job_sequence=list(map(int, d["job_sequence"])),
        machine_assignment=[list(map(int, row)) for row in d["machine_assignment"]],
    )


def encoding_to_dict(enc: Encoding) -> Dict:
    return {
        "factory_assignment": enc.factory_assignment,
        "job_sequence": enc.job_sequence,
        "machine_assignment": enc.machine_assignment,
    }


def random_encoding(instance: ProblemInstance, rng: random.Random) -> Encoding:
    instance.validate()
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    fa = [rng.randrange(F) for _ in range(J)]
    seq = list(range(J))
    rng.shuffle(seq)
    ma = [[rng.randrange(instance.machines[fa[j]][s]) for s in range(S)] for j in range(J)]
    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)

