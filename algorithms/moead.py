"""MOEA/D (最小可运行实现)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .base import Algorithm
from ..encoding import Encoding
from ..init_pool import get_pool
from ..instance import ProblemInstance
from ..types import Objectives


@dataclass(frozen=True)
class MOEADConfig:
    num_subproblems: int = 25
    neighborhood_size: int = 10
    decomposition: str = "tchebycheff"
    weight_init: str = "uniform"
    crossover_prob: float = 0.9
    mutation_prob: float = 0.0  # 若为0则使用 1/J
    delta: float = 0.9  # 使用邻域的概率
    nr: int = 2  # 最大更新邻居数量
    neighborhood_min_ratio: float = 0.4
    weight_reset_period: int = 25
    weight_reset_ratio: float = 0.1
    global_replace_trials: int = 6
    tardiness_refine_prob: float = 0.55
    tardiness_refine_trials: int = 5
    load_refine_prob: float = 0.35
    load_refine_trials: int = 4
    makespan_refine_prob: float = 0.25
    makespan_refine_trials: int = 3
    # 论文: ATM-MOEA/D 停滞检测触发权重自适应 (arxiv 2502.16481)
    stagnation_window: int = 15  # 检测停滞的迭代窗口
    stagnation_reset_ratio: float = 0.2  # 停滞时额外重置比例


class MOEAD(Algorithm):
    name = "MOEA/D"

    def __init__(self, config: MOEADConfig) -> None:
        self.config = config
        self._rng = random.Random()
        self._weights: List[Tuple[float, float, float]] = []
        self._neighbors: List[List[int]] = []
        self._iter = 0
        self._ideal_history: List[Objectives] = []

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def initialize(self, instance: ProblemInstance, population_size: int) -> List[Encoding]:
        """
        初始化:
        - 生成权重向量
        - 初始化子问题解
        """
        instance.validate()
        num = self.config.num_subproblems
        self._weights = uniform_weights(num, self._rng)
        self._neighbors = build_neighbors(self._weights, self.config.neighborhood_size)
        pool_size = max(num * 4, num)
        pool, objs = get_pool(instance, pool_size)
        pop: List[Encoding] = []
        for w in self._weights:
            best_idx = min(range(len(pool)), key=lambda i: tchebycheff(objs[i], ideal_point(objs), w))
            pop.append(pool[best_idx])
        return pop[:num]

    def step(self, instance: ProblemInstance, population: List[Encoding]) -> List[Encoding]:
        """
        单代进化:
        - 邻域选择
        - 变异/交叉产生子代
        - Tchebycheff 更新
        """
        instance.validate()
        if not population:
            return []
        num = min(self.config.num_subproblems, len(population))
        if num == 0:
            return []
        if len(population) != num:
            population = population[:num]
        if len(self._weights) != num or len(self._neighbors) != num:
            self._weights = uniform_weights(num, self._rng)
        k_cur = self._current_neighborhood_size(num)
        self._neighbors = build_neighbors(self._weights, k_cur)

        objs = [instance.evaluate(ind) for ind in population]
        ideal = [min(v[i] for v in objs) for i in range(3)]
        nadir = [max(v[i] for v in objs) for i in range(3)]

        new_pop = population[:]
        for i in range(num):
            neigh = self._neighbors[i]
            use_neigh = self._rng.random() < self.config.delta
            pool = neigh if use_neigh else list(range(num))
            if not pool:
                pool = list(range(num))
            p1 = new_pop[self._rng.choice(pool)]
            p2 = new_pop[self._rng.choice(pool)]
            child = crossover(self._rng, instance, p1, p2, self.config.crossover_prob)
            child = mutate(
                self._rng,
                instance,
                child,
                self._mutation_prob(instance),
            )
            child = refine_makespan(
                self._rng,
                instance,
                child,
                self.config.makespan_refine_prob,
                self.config.makespan_refine_trials,
            )
            child = refine_tardiness(
                self._rng,
                instance,
                child,
                self.config.tardiness_refine_prob,
                self.config.tardiness_refine_trials,
            )
            child = refine_load_balance(
                self._rng,
                instance,
                child,
                self.config.load_refine_prob,
                self.config.load_refine_trials,
            )
            child_obj = instance.evaluate(child)
            for k in range(3):
                if child_obj[k] < ideal[k]:
                    ideal[k] = child_obj[k]
                if child_obj[k] > nadir[k]:
                    nadir[k] = child_obj[k]

            replaced = 0
            for j in neigh:
                if scalar_value(child_obj, tuple(ideal), tuple(nadir), self._weights[j]) <= scalar_value(
                    objs[j], tuple(ideal), tuple(nadir), self._weights[j]
                ):
                    new_pop[j] = child
                    objs[j] = child_obj
                    replaced += 1
                    if replaced >= self.config.nr:
                        break
            if replaced == 0 and self.config.global_replace_trials > 0:
                # fallback: try a few global directions to avoid stagnation
                for _ in range(self.config.global_replace_trials):
                    j = self._rng.randrange(num)
                    if scalar_value(child_obj, tuple(ideal), tuple(nadir), self._weights[j]) <= scalar_value(
                        objs[j], tuple(ideal), tuple(nadir), self._weights[j]
                    ):
                        new_pop[j] = child
                        objs[j] = child_obj
                        break

        self._iter += 1
        self._ideal_history.append(tuple(ideal))
        if len(self._ideal_history) > self.config.stagnation_window:
            self._ideal_history.pop(0)
        # 论文: 停滞检测触发更强权重重置 (ATM-MOEA/D)
        stagnation = self._detect_stagnation()
        reset_ratio = self.config.weight_reset_ratio
        if stagnation:
            reset_ratio = min(0.5, reset_ratio + self.config.stagnation_reset_ratio)
        if self.config.weight_reset_period > 0 and self._iter % self.config.weight_reset_period == 0:
            self._periodic_reset(instance, new_pop, objs, tuple(ideal), reset_ratio)
        return new_pop

    def _mutation_prob(self, instance: ProblemInstance) -> float:
        if self.config.mutation_prob > 0:
            return self.config.mutation_prob
        return 1.0 / max(1, instance.num_jobs)

    def _current_neighborhood_size(self, num: int) -> int:
        k_max = min(self.config.neighborhood_size, num)
        k_min = max(2, int(round(k_max * self.config.neighborhood_min_ratio)))
        if k_min >= k_max:
            return k_max
        # Stage schedule: wide -> medium -> narrow
        ratio = min(1.0, self._iter / 200.0)
        if ratio < 1 / 3:
            return k_max
        if ratio < 2 / 3:
            return max(k_min, int(round((k_max + k_min) / 2)))
        return k_min

    def _detect_stagnation(self) -> bool:
        if len(self._ideal_history) < self.config.stagnation_window:
            return False
        first = self._ideal_history[0]
        last = self._ideal_history[-1]
        return all(last[i] >= first[i] - 1e-9 for i in range(3))

    def _periodic_reset(
        self,
        instance: ProblemInstance,
        population: List[Encoding],
        objs: List[Objectives],
        ideal: Objectives,
        reset_ratio: float | None = None,
    ) -> None:
        num = len(population)
        if num == 0:
            return
        ratio = reset_ratio if reset_ratio is not None else self.config.weight_reset_ratio
        n_reset = max(1, int(round(num * ratio)))
        # worst subproblems by decomposition value
        nadir = (
            max(o[0] for o in objs),
            max(o[1] for o in objs),
            max(o[2] for o in objs),
        )
        scores = [
            (scalar_value(objs[i], ideal, nadir, self._weights[i]), i)
            for i in range(num)
        ]
        scores.sort(reverse=True)
        target_idx = [i for _, i in scores[:n_reset]]

        pool, pool_objs = get_pool(instance, max(4 * n_reset, n_reset))
        for i in target_idx:
            best = min(
                range(len(pool)),
                key=lambda p: scalar_value(pool_objs[p], ideal, nadir, self._weights[i]),
            )
            population[i] = pool[best]
            objs[i] = pool_objs[best]


def uniform_weights(n: int, rng: random.Random) -> List[Tuple[float, float, float]]:
    if n <= 0:
        return []
    # 极端权重：确保覆盖 f1/f2/f3 三个方向，与 IG 形成互补，提升整体 Pareto 质量
    eps = 1e-3
    extreme = [
        (1 - 2 * eps, eps, eps),  # f1 主导
        (eps, 1 - 2 * eps, eps),  # f2 主导
        (eps, eps, 1 - 2 * eps),  # f3 主导
    ]
    if n <= 3:
        return extreme[:n]
    H = 1
    while (H + 2) * (H + 1) // 2 < n - 3:
        H += 1
    weights: List[Tuple[float, float, float]] = list(extreme)
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            w = [i / H, j / H, k / H]
            if sum(w) > 0:
                w = [max(eps, x) for x in w]
                s = sum(w)
                weights.append((w[0] / s, w[1] / s, w[2] / s))
    if len(weights) > n:
        rng.shuffle(weights[3:])  # 只打乱非极端部分
        keep = min(len(weights[3:]), max(0, n - 3))
        weights = weights[:3] + rng.sample(weights[3:], keep)
    return weights[:n]


def build_neighbors(
    weights: List[Tuple[float, float, float]], k: int
) -> List[List[int]]:
    def dist(w1: Tuple[float, float, float], w2: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((w1[i] - w2[i]) ** 2 for i in range(3)))

    neighbors: List[List[int]] = []
    for i, w in enumerate(weights):
        dists = sorted(((dist(w, w2), j) for j, w2 in enumerate(weights)), key=lambda x: x[0])
        neighbors.append([j for _, j in dists[:k]])
    return neighbors


def tchebycheff(obj: Objectives, ideal: Objectives, w: Tuple[float, float, float]) -> float:
    return max(w[i] * abs(obj[i] - ideal[i]) for i in range(3))


def scalar_value(
    obj: Objectives,
    ideal: Objectives,
    nadir: Objectives,
    w: Tuple[float, float, float],
) -> float:
    # normalized Tchebycheff to handle scale imbalance among f1/f2/f3.
    vals = []
    for i in range(3):
        den = max(1e-9, nadir[i] - ideal[i])
        vals.append(w[i] * abs(obj[i] - ideal[i]) / den)
    return max(vals)


def ideal_point(objs: List[Objectives]) -> Objectives:
    return (
        min(o[0] for o in objs),
        min(o[1] for o in objs),
        min(o[2] for o in objs),
    )


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


def crossover(
    rng: random.Random, instance: ProblemInstance, p1: Encoding, p2: Encoding, prob: float
) -> Encoding:
    if rng.random() > prob:
        return p1
    J = len(p1.job_sequence)
    cut1 = rng.randrange(J)
    cut2 = rng.randrange(cut1, J)
    seq = order_crossover(p1.job_sequence, p2.job_sequence, cut1, cut2)
    fa, ma = [], []
    for j in range(J):
        if rng.random() < 0.5:
            fa.append(p1.factory_assignment[j])
        else:
            fa.append(p2.factory_assignment[j])
        row = []
        for s in range(instance.num_stages):
            m_max = instance.machines[fa[j]][s] - 1
            if rng.random() < 0.5:
                row.append(min(p1.machine_assignment[j][s], m_max))
            else:
                row.append(min(p2.machine_assignment[j][s], m_max))
        ma.append(row)
    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


def order_crossover(p1: List[int], p2: List[int], cut1: int, cut2: int) -> List[int]:
    child = [None] * len(p1)
    child[cut1:cut2] = p1[cut1:cut2]
    fill = [x for x in p2 if x not in child[cut1:cut2]]
    idx = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child  # type: ignore[return-value]


def mutate(rng: random.Random, instance: ProblemInstance, enc: Encoding, prob: float) -> Encoding:
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    fa = enc.factory_assignment[:]
    seq = enc.job_sequence[:]
    ma = [row[:] for row in enc.machine_assignment]

    for j in range(J):
        if rng.random() < prob:
            fa[j] = rng.randrange(F)
            for s in range(S):
                ma[j][s] = rng.randrange(instance.machines[fa[j]][s])

    if rng.random() < prob and J >= 2:
        i, k = rng.sample(range(J), 2)
        seq[i], seq[k] = seq[k], seq[i]

    for j in range(J):
        for s in range(S):
            if rng.random() < prob:
                ma[j][s] = rng.randrange(instance.machines[fa[j]][s])

    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


def refine_makespan(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    trials: int = 1,
) -> Encoding:
    if rng.random() > prob:
        return enc
    J = len(enc.job_sequence)
    if J < 2:
        return enc
    best = enc
    best_f1 = instance.evaluate(enc)[0]
    for _ in range(max(1, trials)):
        i, k = rng.sample(range(J), 2)
        seq = best.job_sequence[:]
        seq[i], seq[k] = seq[k], seq[i]
        cand = Encoding(
            factory_assignment=best.factory_assignment[:],
            job_sequence=seq,
            machine_assignment=[row[:] for row in best.machine_assignment],
        )
        f1_new = instance.evaluate(cand)[0]
        if f1_new < best_f1:
            best = cand
            best_f1 = f1_new
    return best


def refine_tardiness(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    trials: int = 1,
) -> Encoding:
    if rng.random() > prob:
        return enc
    J = len(enc.job_sequence)
    if J < 2:
        return enc
    best = enc
    best_f2 = instance.evaluate(enc)[1]
    for _ in range(max(1, trials)):
        i, k = rng.sample(range(J), 2)
        seq = best.job_sequence[:]
        seq[i], seq[k] = seq[k], seq[i]
        cand = Encoding(
            factory_assignment=best.factory_assignment[:],
            job_sequence=seq,
            machine_assignment=[row[:] for row in best.machine_assignment],
        )
        f2_new = instance.evaluate(cand)[1]
        if f2_new < best_f2:
            best = cand
            best_f2 = f2_new
    return best


def refine_load_balance(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    trials: int = 1,
) -> Encoding:
    if rng.random() > prob or instance.num_factories <= 1:
        return enc
    J = len(enc.factory_assignment)
    if J == 0:
        return enc
    best = enc
    best_f3 = instance.evaluate(enc)[2]
    for _ in range(max(1, trials)):
        j = rng.randrange(J)
        cur_f = best.factory_assignment[j]
        cand_f = rng.randrange(instance.num_factories)
        if cand_f == cur_f:
            continue
        fa = best.factory_assignment[:]
        fa[j] = cand_f
        ma = [row[:] for row in best.machine_assignment]
        for s in range(instance.num_stages):
            ma[j][s] = rng.randrange(instance.machines[cand_f][s])
        cand = Encoding(
            factory_assignment=fa,
            job_sequence=best.job_sequence[:],
            machine_assignment=ma,
        )
        f3_new = instance.evaluate(cand)[2]
        if f3_new < best_f3:
            best = cand
            best_f3 = f3_new
    return best

