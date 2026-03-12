"""NSGA-II (骨架实现，流程与参数结构)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import Algorithm
from ..encoding import Encoding
from ..init_pool import get_pool
from ..instance import ProblemInstance
from ..types import Objectives


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 25
    crossover_prob: float = 0.9
    eta_c: float = 20.0
    mutation_prob: float = 0.0  # 若为0则使用 1/J
    eta_m: float = 20.0
    tardiness_refine_prob: float = 0.50
    load_refine_prob: float = 0.25
    makespan_refine_prob: float = 0.25  # 降低IG占比：强化f1竞争
    refine_trials: int = 10
    seed_ratio: float = 0.3
    anchor_keep: bool = True
    # 论文: 对更多个体应用邻域结构可提高多样性 (hrpub/7080, flowshop)
    refine_fronts: int = 2
    refine_front3_ratio: float = 0.55  # 提高以增强对IG的竞争力


class NSGA2(Algorithm):
    name = "NSGA-II"

    def __init__(self, config: NSGA2Config) -> None:
        self.config = config
        self._rng = random.Random()

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def initialize(self, instance: ProblemInstance, population_size: int) -> List[Encoding]:
        """
        初始化种群（占位）:
        - 根据统一编码生成初始解
        """
        instance.validate()
        pop: List[Encoding] = []
        seed_count = max(0, int(population_size * self.config.seed_ratio))
        if seed_count > 0:
            pop.extend(seed_population(self._rng, instance, seed_count))
        # LHS pool selection for diversity
        pool_size = max(population_size * 4, population_size)
        pool, objs = get_pool(instance, pool_size)
        pop.extend(select_diverse(pool, objs, population_size - len(pop)))
        return pop[:population_size]

    def step(self, instance: ProblemInstance, population: List[Encoding]) -> List[Encoding]:
        """
        单代进化（占位）:
        - Fast non-dominated sorting
        - Crowding distance
        - Selection / Crossover / Mutation
        """
        instance.validate()
        if not population:
            return []

        pop_size = len(population)
        objectives = [instance.evaluate(ind) for ind in population]
        fronts = fast_nondominated_sort(objectives)
        crowd = crowding_distance(fronts, objectives)

        # Binary tournament selection
        mating_pool = [
            tournament_select(self._rng, population, objectives, fronts, crowd) for _ in range(pop_size)
        ]

        # Variation
        offspring: List[Encoding] = []
        for i in range(0, pop_size, 2):
            p1 = mating_pool[i]
            p2 = mating_pool[(i + 1) % pop_size]
            c1, c2 = crossover(self._rng, instance, p1, p2, self.config.crossover_prob, self.config.eta_c)
            c1 = mutate(
                self._rng,
                instance,
                c1,
                self._mutation_prob(instance),
                self.config.eta_m,
            )
            c2 = mutate(
                self._rng,
                instance,
                c2,
                self._mutation_prob(instance),
                self.config.eta_m,
            )
            c1 = refine_makespan(
                self._rng,
                instance,
                c1,
                self.config.makespan_refine_prob,
                self.config.refine_trials,
            )
            c2 = refine_makespan(
                self._rng,
                instance,
                c2,
                self.config.makespan_refine_prob,
                self.config.refine_trials,
            )
            c1 = refine_tardiness(
                self._rng,
                instance,
                c1,
                self.config.tardiness_refine_prob,
                self.config.refine_trials,
            )
            c2 = refine_tardiness(
                self._rng,
                instance,
                c2,
                self.config.tardiness_refine_prob,
                self.config.refine_trials,
            )
            c1 = refine_load_balance(
                self._rng,
                instance,
                c1,
                self.config.load_refine_prob,
                self.config.refine_trials,
            )
            c2 = refine_load_balance(
                self._rng,
                instance,
                c2,
                self.config.load_refine_prob,
                self.config.refine_trials,
            )
            offspring.extend([c1, c2])

        combined = population + offspring[:pop_size]
        combined_objs = [instance.evaluate(ind) for ind in combined]
        new_fronts = fast_nondominated_sort(combined_objs)
        new_pop: List[Encoding] = []
        for front in new_fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(combined[i] for i in front)
            else:
                dist = crowding_distance([front], combined_objs)[0]
                sorted_front = sorted(front, key=lambda idx: dist[idx], reverse=True)
                new_pop.extend(combined[i] for i in sorted_front[: pop_size - len(new_pop)])
                break
        # 论文改进: 对前N个前沿的更多个体应用邻域局部搜索 (hrpub, flowshop)
        objs_new = [instance.evaluate(x) for x in new_pop]
        fronts_new = fast_nondominated_sort(objs_new)
        refine_idx: List[int] = []
        for fi, front in enumerate(fronts_new):
            if fi < self.config.refine_fronts:
                refine_idx.extend(front)
            elif fi == self.config.refine_fronts:
                n_take = max(0, int(len(front) * self.config.refine_front3_ratio))
                if n_take > 0:
                    dist = crowding_distance([front], objs_new)[0]
                    sorted_f = sorted(front, key=lambda i: dist.get(i, 0.0), reverse=True)
                    refine_idx.extend(sorted_f[:n_take])
                break
        for i in refine_idx:
            if i >= len(new_pop):
                continue
            x = refine_makespan(
                self._rng,
                instance,
                new_pop[i],
                prob=1.0,
                trials=self.config.refine_trials,
            )
            x = refine_tardiness(
                self._rng,
                instance,
                x,
                prob=1.0,
                trials=self.config.refine_trials,
            )
            x = refine_load_balance(
                self._rng,
                instance,
                x,
                prob=1.0,
                trials=self.config.refine_trials,
            )
            new_pop[i] = x

        if self.config.anchor_keep and new_pop:
            # 保持前沿极值 + 膝点，提升整体 Pareto 质量
            mins = [min(o[m] for o in combined_objs) for m in range(3)]
            maxs = [max(o[m] for o in combined_objs) for m in range(3)]
            scales = [max(1e-9, maxs[m] - mins[m]) for m in range(3)]

            def norm_sum(i: int) -> float:
                return sum((combined_objs[i][m] - mins[m]) / scales[m] for m in range(3))

            knee_idx = min(range(len(combined_objs)), key=norm_sum)
            anchors = [
                combined[min(range(len(combined_objs)), key=lambda i: combined_objs[i][0])],
                combined[min(range(len(combined_objs)), key=lambda i: combined_objs[i][1])],
                combined[min(range(len(combined_objs)), key=lambda i: combined_objs[i][2])],
                combined[knee_idx],
            ]
            for a in anchors:
                if a in new_pop:
                    continue
                worst_idx = max(
                    range(len(new_pop)),
                    key=lambda i: sum(instance.evaluate(new_pop[i])),
                )
                new_pop[worst_idx] = a
        return new_pop

    def _mutation_prob(self, instance: ProblemInstance) -> float:
        if self.config.mutation_prob > 0:
            return self.config.mutation_prob
        return 1.0 / max(1, instance.num_jobs)


def dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_nondominated_sort(objs: List[Objectives]) -> List[List[int]]:
    S: List[List[int]] = [[] for _ in range(len(objs))]
    n = [0 for _ in range(len(objs))]
    fronts: List[List[int]] = [[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(fronts: List[List[int]], objs: List[Objectives]) -> List[Dict[int, float]]:
    distances: List[Dict[int, float]] = []
    for front in fronts:
        dist = {i: 0.0 for i in front}
        if not front:
            distances.append(dist)
            continue
        if len(front) <= 2:
            for i in front:
                dist[i] = float("inf")
            distances.append(dist)
            continue
        for m in range(3):
            front_sorted = sorted(front, key=lambda i: objs[i][m])
            dist[front_sorted[0]] = float("inf")
            dist[front_sorted[-1]] = float("inf")
            min_v = objs[front_sorted[0]][m]
            max_v = objs[front_sorted[-1]][m]
            if max_v == min_v:
                continue
            for k in range(1, len(front_sorted) - 1):
                prev_v = objs[front_sorted[k - 1]][m]
                next_v = objs[front_sorted[k + 1]][m]
                dist[front_sorted[k]] += (next_v - prev_v) / (max_v - min_v)

        # Truthful-like component: nearest-neighbor distance in objective space.
        # This avoids overly optimistic crowding scores in 3-objective fronts.
        mins = [min(objs[i][m] for i in front) for m in range(3)]
        maxs = [max(objs[i][m] for i in front) for m in range(3)]
        for i in front:
            if dist[i] == float("inf"):
                continue
            nn = float("inf")
            for j in front:
                if i == j:
                    continue
                d2 = 0.0
                for m in range(3):
                    scale = (maxs[m] - mins[m]) + 1e-12
                    d = (objs[i][m] - objs[j][m]) / scale
                    d2 += d * d
                nn = min(nn, d2 ** 0.5)
            dist[i] += nn
        distances.append(dist)
    return distances


def tournament_select(
    rng: random.Random,
    population: List[Encoding],
    objs: List[Objectives],
    fronts: List[List[int]],
    crowd: List[Dict[int, float]],
) -> Encoding:
    a = rng.randrange(len(population))
    b = rng.randrange(len(population))
    rank = build_rank_map(fronts)
    if rank[a] < rank[b]:
        return population[a]
    if rank[b] < rank[a]:
        return population[b]
    ca = crowd[rank[a]].get(a, 0.0)
    cb = crowd[rank[b]].get(b, 0.0)
    if ca > cb:
        return population[a]
    if cb > ca:
        return population[b]
    # deterministic tie-break: prioritize lower tardiness then lower imbalance
    oa = objs[a]
    ob = objs[b]
    if oa[1] < ob[1]:
        return population[a]
    if ob[1] < oa[1]:
        return population[b]
    if oa[2] < ob[2]:
        return population[a]
    if ob[2] < oa[2]:
        return population[b]
    return population[a] if a <= b else population[b]


def build_rank_map(fronts: List[List[int]]) -> Dict[int, int]:
    rank: Dict[int, int] = {}
    for i, front in enumerate(fronts):
        for idx in front:
            rank[idx] = i
    return rank


def crossover(
    rng: random.Random,
    instance: ProblemInstance,
    p1: Encoding,
    p2: Encoding,
    prob: float,
    eta_c: float,
) -> Tuple[Encoding, Encoding]:
    if rng.random() > prob:
        return p1, p2

    J = len(p1.job_sequence)
    cut1 = rng.randrange(J)
    cut2 = rng.randrange(cut1, J)
    child_seq1 = order_crossover(p1.job_sequence, p2.job_sequence, cut1, cut2)
    child_seq2 = order_crossover(p2.job_sequence, p1.job_sequence, cut1, cut2)

    # SBX-like crossover for discrete factory/machine assignment
    fa1, fa2 = [], []
    ma1, ma2 = [], []
    for j in range(J):
        f1, f2 = p1.factory_assignment[j], p2.factory_assignment[j]
        c_f1, c_f2 = sbx_int(rng, f1, f2, 0, instance.num_factories - 1, eta_c)
        fa1.append(c_f1)
        fa2.append(c_f2)

        row1, row2 = [], []
        for s in range(instance.num_stages):
            m_max1 = instance.machines[c_f1][s] - 1
            m_max2 = instance.machines[c_f2][s] - 1
            m1, m2 = p1.machine_assignment[j][s], p2.machine_assignment[j][s]
            c_m1, _ = sbx_int(rng, m1, m2, 0, m_max1, eta_c)
            _, c_m2 = sbx_int(rng, m1, m2, 0, m_max2, eta_c)
            row1.append(c_m1)
            row2.append(c_m2)
        ma1.append(row1)
        ma2.append(row2)

    return (
        Encoding(factory_assignment=fa1, job_sequence=child_seq1, machine_assignment=ma1),
        Encoding(factory_assignment=fa2, job_sequence=child_seq2, machine_assignment=ma2),
    )


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


def mutate(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    eta_m: float,
) -> Encoding:
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    fa = enc.factory_assignment[:]
    seq = enc.job_sequence[:]
    ma = [row[:] for row in enc.machine_assignment]

    # factory assignment mutation
    for j in range(J):
        if rng.random() < prob:
            fa[j] = poly_mutation_int(rng, fa[j], 0, F - 1, eta_m)
            for s in range(S):
                m_max = instance.machines[fa[j]][s] - 1
                ma[j][s] = poly_mutation_int(rng, ma[j][s], 0, m_max, eta_m)

    # swap mutation on sequence
    if rng.random() < prob and J >= 2:
        i, k = rng.sample(range(J), 2)
        seq[i], seq[k] = seq[k], seq[i]

    # machine assignment mutation
    for j in range(J):
        for s in range(S):
            if rng.random() < prob:
                m_max = instance.machines[fa[j]][s] - 1
                ma[j][s] = poly_mutation_int(rng, ma[j][s], 0, m_max, eta_m)

    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


def refine_makespan(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    trials: int = 1,
) -> Encoding:
    """局部搜索降低 f1 (makespan)，与 IG 竞争"""
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


def seed_population(rng: random.Random, instance: ProblemInstance, size: int) -> List[Encoding]:
    # EDD-based seeding with greedy factory load assignment
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    jobs = list(range(J))
    jobs.sort(key=lambda j: instance.due_date[j])
    pop: List[Encoding] = []
    for _ in range(size):
        loads = [0.0 for _ in range(F)]
        factory_assignment = [0 for _ in range(J)]
        for j in jobs:
            f = min(range(F), key=lambda x: loads[x])
            factory_assignment[j] = f
            loads[f] += sum(instance.processing_time[j][s] for s in range(S))
        job_sequence = jobs[:]
        machine_assignment = [
            [rng.randrange(instance.machines[factory_assignment[j]][s]) for s in range(S)]
            for j in range(J)
        ]
        pop.append(
            Encoding(
                factory_assignment=factory_assignment,
                job_sequence=job_sequence,
                machine_assignment=machine_assignment,
            )
        )
    return pop


def select_diverse(
    pool: List[Encoding], objs: List[Objectives], k: int
) -> List[Encoding]:
    if k <= 0:
        return []
    selected: List[int] = []
    # start from best by sum
    start = min(range(len(pool)), key=lambda i: sum(objs[i]))
    selected.append(start)
    while len(selected) < k and len(selected) < len(pool):
        def min_dist(i: int) -> float:
            return min(
                ((objs[i][0] - objs[j][0]) ** 2 + (objs[i][1] - objs[j][1]) ** 2 + (objs[i][2] - objs[j][2]) ** 2)
                for j in selected
            )
        next_idx = max(
            (i for i in range(len(pool)) if i not in selected),
            key=min_dist,
        )
        selected.append(next_idx)
    return [pool[i] for i in selected]


def sbx_int(
    rng: random.Random, x1: int, x2: int, lo: int, hi: int, eta: float
) -> Tuple[int, int]:
    if x1 == x2:
        return x1, x2
    if x1 > x2:
        x1, x2 = x2, x1
    u = rng.random()
    beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    c1 = int(round(min(max(c1, lo), hi)))
    c2 = int(round(min(max(c2, lo), hi)))
    return c1, c2


def poly_mutation_int(
    rng: random.Random, x: int, lo: int, hi: int, eta: float
) -> int:
    if lo == hi:
        return lo
    u = rng.random()
    delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
    y = x + delta * (hi - lo)
    y = int(round(min(max(y, lo), hi)))
    return y

