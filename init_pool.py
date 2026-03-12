"""Unified LHS candidate pool (Ω0)."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .encoding import Encoding
from .instance import ProblemInstance

_POOL_CACHE: Dict[Tuple[int, int], Tuple[List[Encoding], List[Tuple[float, float, float]]]] = {}


def ensure_pool(instance: ProblemInstance, size: int, seed: int = 1) -> None:
    key = (id(instance), size)
    if key in _POOL_CACHE:
        return
    rng = random.Random(seed)
    encs = _lhs_encodings(instance, size, rng)
    objs = [instance.evaluate(e) for e in encs]
    _POOL_CACHE[key] = (encs, objs)


def get_pool(
    instance: ProblemInstance, size: int, seed: int = 1
) -> Tuple[List[Encoding], List[Tuple[float, float, float]]]:
    key = (id(instance), size)
    if key not in _POOL_CACHE:
        ensure_pool(instance, size, seed=seed)
    return _POOL_CACHE[key]


def _lhs_samples(n: int, d: int, rng: random.Random) -> List[List[float]]:
    # Latin Hypercube Sampling in [0,1]
    samples = [[0.0] * d for _ in range(n)]
    for j in range(d):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(n):
            u = (perm[i] + rng.random()) / n
            samples[i][j] = u
    return samples


def _lhs_encodings(instance: ProblemInstance, n: int, rng: random.Random) -> List[Encoding]:
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    dim = J + J + J * S
    samples = _lhs_samples(n, dim, rng)
    encs: List[Encoding] = []
    for row in samples:
        idx = 0
        fa = [min(F - 1, int(row[idx + j] * F)) for j in range(J)]
        idx += J
        keys = row[idx : idx + J]
        idx += J
        seq = sorted(range(J), key=lambda j: keys[j])
        ma: List[List[int]] = []
        for j in range(J):
            row_ma = []
            for s in range(S):
                m = instance.machines[fa[j]][s]
                row_ma.append(min(m - 1, int(row[idx] * m)))
                idx += 1
            ma.append(row_ma)
        encs.append(Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma))
    return encs
