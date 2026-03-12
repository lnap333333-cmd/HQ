"""IG (Iterative Greedy) 最小可运行实现."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from .base import Algorithm
from ..encoding import Encoding
from ..init_pool import get_pool
from ..instance import ProblemInstance


@dataclass(frozen=True)
class IGConfig:
    destruction_rate: float = 0.15
    repair_rule: str = "earliest_completion"
    local_search_steps: int = 5
    # SA-style acceptance (classic IG)
    initial_temp: float = 0.8
    cooling_rate: float = 0.995
    min_temp: float = 1e-3


class IG(Algorithm):
    name = "IG"

    def __init__(self, config: IGConfig) -> None:
        self.config = config
        self._rng = random.Random()
        self._temp = config.initial_temp

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def initialize(self, instance: ProblemInstance, population_size: int) -> List[Encoding]:
        """
        初始化解:
        - 随机生成可行解
        """
        instance.validate()
        pool_size = max(population_size * 4, population_size)
        pool, objs = get_pool(instance, pool_size)
        # elite perturbation from nondominated pool
        elite = select_elite(pool, objs, population_size)
        return elite

    def step(self, instance: ProblemInstance, population: List[Encoding]) -> List[Encoding]:
        """
        迭代贪婪:
        - 破坏
        - 修复
        - 接受准则
        """
        instance.validate()
        if not population:
            return []

        new_pop: List[Encoding] = []
        for ind in population:
            candidate = destroy(self._rng, ind, self.config.destruction_rate)
            candidate = repair(self._rng, instance, candidate, self.config.repair_rule)
            candidate = local_search(self._rng, instance, candidate, self.config.local_search_steps)
            new_pop.append(accept_sa(self._rng, instance, ind, candidate, self._temp))
        self._cool_down()
        return new_pop

    def _cool_down(self) -> None:
        if self._temp > self.config.min_temp:
            self._temp = max(self.config.min_temp, self._temp * self.config.cooling_rate)


def random_population(
    rng: random.Random, instance: ProblemInstance, size: int
) -> List[Encoding]:
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    pop: List[Encoding] = []
    for _ in range(size):
        fa = [rng.randrange(F) for _ in range(J)]
        seq = list(range(J))
        rng.shuffle(seq)
        ma = [
            [rng.randrange(instance.machines[f][s]) for s in range(S)]
            for f in fa
        ]
        pop.append(Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma))
    return pop


def select_elite(
    pool: List[Encoding], objs: List[tuple], k: int
) -> List[Encoding]:
    if k <= 0:
        return []
    nd_idx = []
    for i, a in enumerate(objs):
        if any(dominates(b, a) for j, b in enumerate(objs) if j != i):
            continue
        nd_idx.append(i)
    # if not enough, fill by best sum
    if len(nd_idx) < k:
        rest = sorted(range(len(pool)), key=lambda i: sum(objs[i]))
        nd_idx = (nd_idx + [i for i in rest if i not in nd_idx])[:k]
    selected = [pool[i] for i in nd_idx[:k]]
    return selected


def destroy(rng: random.Random, enc: Encoding, rate: float) -> Encoding:
    J = len(enc.job_sequence)
    k = max(1, int(J * rate))
    seq = enc.job_sequence[:]
    removed = rng.sample(seq, k)
    seq = [j for j in seq if j not in removed]
    return Encoding(
        factory_assignment=enc.factory_assignment[:],
        job_sequence=seq,
        machine_assignment=[row[:] for row in enc.machine_assignment],
    )


def repair(
    rng: random.Random, instance: ProblemInstance, enc: Encoding, rule: str
) -> Encoding:
    # 修复: 将缺失作业插入序列；默认使用最小化目标的贪婪插入
    J = len(enc.factory_assignment)
    present = set(enc.job_sequence)
    missing = [j for j in range(J) if j not in present]
    seq = enc.job_sequence[:]
    for j in missing:
        if rule == "earliest_completion":
            remaining = [x for x in missing if x != j and x not in seq]
            seq = greedy_insert(instance, enc, seq, j, remaining)
        else:
            pos = rng.randrange(len(seq) + 1)
            seq.insert(pos, j)
    return Encoding(
        factory_assignment=enc.factory_assignment[:],
        job_sequence=seq,
        machine_assignment=[row[:] for row in enc.machine_assignment],
    )


def accept_sa(
    rng: random.Random,
    instance: ProblemInstance,
    current: Encoding,
    candidate: Encoding,
    temp: float,
) -> Encoding:
    c_obj = instance.evaluate(current)
    n_obj = instance.evaluate(candidate)
    if dominates(n_obj, c_obj):
        return candidate
    if dominates(c_obj, n_obj):
        return current
    # If non-dominated, use crowding distance to prefer diversity
    crowd = crowding_distance([c_obj, n_obj])
    if crowd[1] > crowd[0]:
        return candidate
    if crowd[0] > crowd[1]:
        return current
    # SA-style acceptance as tie-breaker
    delta = sum(n_obj) - sum(c_obj)
    if delta <= 0:
        return candidate
    if temp <= 0:
        return current
    prob = math.exp(-delta / temp)
    return candidate if rng.random() < prob else current


def dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def greedy_insert(
    instance: ProblemInstance, enc: Encoding, seq: List[int], job: int, remaining: List[int]
) -> List[int]:
    # 选择非支配且拥挤度最大的插入位置（用完整序列评估）
    candidates = []
    cand_seqs = []
    for pos in range(len(seq) + 1):
        cand_seq = seq[:]
        cand_seq.insert(pos, job)
        full_seq = cand_seq + remaining
        cand = Encoding(
            factory_assignment=enc.factory_assignment[:],
            job_sequence=full_seq,
            machine_assignment=[row[:] for row in enc.machine_assignment],
        )
        candidates.append(instance.evaluate(cand))
        cand_seqs.append(cand_seq)

    # select non-dominated; tie-break by crowding
    nd_idx = []
    for i, a in enumerate(candidates):
        if any(dominates(b, a) for j, b in enumerate(candidates) if j != i):
            continue
        nd_idx.append(i)
    if not nd_idx:
        return cand_seqs[0]
    if len(nd_idx) == 1:
        return cand_seqs[nd_idx[0]]
    crowd = crowding_distance([candidates[i] for i in nd_idx])
    best_local = max(range(len(nd_idx)), key=lambda k: crowd[k])
    return cand_seqs[nd_idx[best_local]]


def local_search(
    rng: random.Random, instance: ProblemInstance, enc: Encoding, steps: int
) -> Encoding:
    best = enc
    best_score = sum(instance.evaluate(enc))
    seq = enc.job_sequence[:]
    for _ in range(steps):
        if len(seq) < 2:
            break
        i, k = rng.sample(range(len(seq)), 2)
        cand_seq = seq[:]
        cand_seq[i], cand_seq[k] = cand_seq[k], cand_seq[i]
        cand = Encoding(
            factory_assignment=enc.factory_assignment[:],
            job_sequence=cand_seq,
            machine_assignment=[row[:] for row in enc.machine_assignment],
        )
        score_vec = instance.evaluate(cand)
        if dominates(score_vec, instance.evaluate(best)):
            best = cand
            seq = cand_seq
        # Try local machine reassignment on a random job/stage
        j = rng.randrange(len(enc.factory_assignment))
        s = rng.randrange(instance.num_stages)
        f = enc.factory_assignment[j]
        best_m = enc.machine_assignment[j][s]
        for m in range(instance.machines[f][s]):
            if m == best_m:
                continue
            cand_ma = [row[:] for row in enc.machine_assignment]
            cand_ma[j][s] = m
            cand2 = Encoding(
                factory_assignment=enc.factory_assignment[:],
                job_sequence=seq[:],
                machine_assignment=cand_ma,
            )
            score2 = instance.evaluate(cand2)
            if dominates(score2, instance.evaluate(best)):
                best = cand2
                best_m = m

        # Try factory reassignment for load balance (f3)
        j2 = rng.randrange(len(enc.factory_assignment))
        current_f = enc.factory_assignment[j2]
        for f2 in range(instance.num_factories):
            if f2 == current_f:
                continue
            cand_fa = enc.factory_assignment[:]
            cand_fa[j2] = f2
            cand_ma = [row[:] for row in enc.machine_assignment]
            for s2 in range(instance.num_stages):
                cand_ma[j2][s2] = rng.randrange(instance.machines[f2][s2])
            cand3 = Encoding(
                factory_assignment=cand_fa,
                job_sequence=seq[:],
                machine_assignment=cand_ma,
            )
            cur_f3 = instance.evaluate(best)[2]
            new_f3 = instance.evaluate(cand3)[2]
            if new_f3 < cur_f3:
                best = cand3
    return best


def crowding_distance(objs: List[tuple]) -> List[float]:
    # return crowding for given list (same order)
    n = len(objs)
    if n == 0:
        return []
    dist = [0.0] * n
    for m in range(3):
        idx = sorted(range(n), key=lambda i: objs[i][m])
        dist[idx[0]] = float("inf")
        dist[idx[-1]] = float("inf")
        min_v = objs[idx[0]][m]
        max_v = objs[idx[-1]][m]
        if max_v == min_v:
            continue
        for k in range(1, n - 1):
            prev_v = objs[idx[k - 1]][m]
            next_v = objs[idx[k + 1]][m]
            dist[idx[k]] += (next_v - prev_v) / (max_v - min_v)
    return dist

