"""HHO (最小可运行实现)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .base import Algorithm
from ..encoding import Encoding
from ..init_pool import get_pool
from ..instance import ProblemInstance


@dataclass(frozen=True)
class HHOConfig:
    e0: float = 2.0
    energy_decay: str = "nonlinear"
    besiege_mode: str = "soft_hard"
    max_iters: int = 200
    levy_beta: float = 1.5
    leader_pool_size: int = 10
    tardiness_leader_prob: float = 0.35  # 多目标引导：f1/f2/f3 轮换，保持整体优化
    makespan_leader_prob: float = 0.35
    load_leader_prob: float = 0.30
    refine_prob: float = 0.4
    load_refine_prob: float = 0.3
    makespan_refine_prob: float = 0.2
    assignment_refine_prob: float = 0.35
    refine_trials: int = 5
    elite_keep_ratio: float = 0.15
    gaussian_prob: float = 0.05
    opposition_ratio: float = 0.0
    seed_ratio: float = 0.2
    # EHHO论文(Soft Computing 2023): DE式变异 + 混沌策略
    de_crossover_prob: float = 0.15  # 以DE式生成新个体的概率
    chaos_mutation_prob: float = 0.1  # 混沌映射替代均匀随机的概率
    chaos_init_ratio: float = 0.4  # 混沌初始化占比，提升初始多样性


def _chaos_logistic(x: float, mu: float = 4.0) -> float:
    """Logistic 映射 x_{n+1}=mu*x*(1-x)，mu=4 时完全混沌"""
    return mu * x * (1.0 - x)


def _chaos_sinusoidal(ch: float, a: float = 2.3) -> float:
    """EHHO论文: 正弦混沌映射 ch_{i+1}=a*ch^2*sin(pi*ch), a=2.3"""
    return a * ch * ch * math.sin(math.pi * ch)


def _discrete_de(
    rng: random.Random,
    instance: ProblemInstance,
    r1: Encoding,
    r2: Encoding,
    r3: Encoding,
    f: float,
) -> Encoding:
    """离散DE: V = r1 + F*(r2-r3), 适配三段式编码"""
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    fa, seq, ma = list(r1.factory_assignment), list(r1.job_sequence), [row[:] for row in r1.machine_assignment]
    for j in range(J):
        if rng.random() < f:
            if r2.factory_assignment[j] != r3.factory_assignment[j]:
                fa[j] = r2.factory_assignment[j]
            else:
                fa[j] = r3.factory_assignment[j]
            for s in range(S):
                m_max = instance.machines[fa[j]][s] - 1
                if r2.machine_assignment[j][s] != r3.machine_assignment[j][s]:
                    ma[j][s] = min(r2.machine_assignment[j][s], m_max)
                else:
                    ma[j][s] = min(r3.machine_assignment[j][s], m_max)
    if rng.random() < f and J >= 2:
        i, k = rng.sample(range(J), 2)
        seq[i], seq[k] = seq[k], seq[i]
    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


class HHO(Algorithm):
    name = "HHO"

    def __init__(self, config: HHOConfig) -> None:
        self.config = config
        self._rng = random.Random()
        self._iter = 0
        self._chaos_state: float = 0.7

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def initialize(self, instance: ProblemInstance, population_size: int) -> List[Encoding]:
        """
        初始化鹰群:
        - 混沌初始化 + 启发式种子 + 池中多样性选择
        """
        instance.validate()
        pop: List[Encoding] = []
        chaos_count = max(0, int(population_size * self.config.chaos_init_ratio))
        seed_count = max(0, int(population_size * self.config.seed_ratio))
        if chaos_count > 0:
            pop.extend(chaos_init_population(self._rng, instance, chaos_count))
        if seed_count > 0:
            pop.extend(seed_population(self._rng, instance, seed_count))
        remain = population_size - len(pop)
        if remain > 0:
            pool_size = max(population_size * 4, population_size)
            pool, objs = get_pool(instance, pool_size)
            pop.extend(select_diverse(pool, objs, remain))
        return pop[:population_size]

    def step(self, instance: ProblemInstance, population: List[Encoding]) -> List[Encoding]:
        """
        单代更新:
        - Exploration / Soft besiege / Hard besiege / Levy
        """
        instance.validate()
        if not population:
            return []

        objs = [instance.evaluate(ind) for ind in population]
        leaders = select_leader_pool(objs, self.config.leader_pool_size)
        best_f1_idx = min(range(len(objs)), key=lambda i: objs[i][0])
        best_f2_idx = min(range(len(objs)), key=lambda i: objs[i][1])
        best_f3_idx = min(range(len(objs)), key=lambda i: objs[i][2])

        e = self._energy()
        new_pop: List[Encoding] = []
        for ind in population:
            if abs(e) < 0.8 and leaders and self._rng.random() < 0.35:
                # Late-stage: sample leaders from ND crowding pool for diversity.
                leader = population[leaders[self._rng.randrange(len(leaders))]]
            else:
                r_lead = self._rng.random()
                if r_lead < self.config.makespan_leader_prob:
                    leader = population[best_f1_idx]
                elif r_lead < self.config.makespan_leader_prob + self.config.tardiness_leader_prob:
                    leader = population[best_f2_idx]
                else:
                    leader = population[best_f3_idx]
            # EHHO: 混沌映射生成F ∈ [0.5,1]
            self._chaos_state = _chaos_sinusoidal(self._chaos_state) % 1.0
            if self._chaos_state <= 0:
                self._chaos_state = 0.7
            f_de = 0.5 + 0.5 * self._chaos_state
            q = self._rng.random()
            r = self._rng.random()
            if abs(e) >= 1.0:
                # Exploration
                if q >= 0.5:
                    cand = perturb(self._rng, instance, ind, rate=0.25)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, r3, f_de)
                    new_pop.append(cand)
                else:
                    rand = population[self._rng.randrange(len(population))]
                    cand = perturb(self._rng, instance, rand, rate=0.25)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, rand, f_de)
                    new_pop.append(cand)
            else:
                if r >= 0.5 and abs(e) >= 0.5:
                    # Soft besiege
                    cand = attract(self._rng, instance, ind, leader, rate=0.6)
                    cand = perturb(self._rng, instance, cand, rate=0.1)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, r3, f_de)
                    new_pop.append(cand)
                elif r >= 0.5 and abs(e) < 0.5:
                    # Hard besiege
                    cand = attract(self._rng, instance, ind, leader, rate=0.9)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, r3, f_de)
                    new_pop.append(cand)
                elif r < 0.5 and abs(e) >= 0.5:
                    # Soft besiege + rapid dives (Levy)
                    cand = attract(self._rng, instance, ind, leader, rate=0.6)
                    cand = levy_dive(self._rng, instance, cand, beta=self.config.levy_beta)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, r3, f_de)
                    new_pop.append(cand)
                else:
                    # Hard besiege + rapid dives
                    cand = attract(self._rng, instance, ind, leader, rate=0.9)
                    cand = levy_dive(self._rng, instance, cand, beta=self.config.levy_beta)
                    cand = gaussian_research(self._rng, instance, cand, self.config.gaussian_prob)
                    cand = refine_makespan(
                        self._rng, instance, cand,
                        self.config.makespan_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_tardiness(
                        self._rng, instance, cand,
                        self.config.refine_prob, self.config.refine_trials,
                    )
                    cand = refine_load_balance(
                        self._rng, instance, cand,
                        self.config.load_refine_prob, self.config.refine_trials,
                    )
                    cand = refine_assignment(
                        self._rng, instance, cand,
                        self.config.assignment_refine_prob, self.config.refine_trials,
                    )
                    if self.config.de_crossover_prob > 0 and self._rng.random() < self.config.de_crossover_prob:
                        r3 = population[self._rng.randrange(len(population))]
                        cand = _discrete_de(self._rng, instance, cand, leader, r3, f_de)
                    new_pop.append(cand)

        # Environmental selection (non-dominated + crowding)
        combined = population + new_pop
        combined_objs = [instance.evaluate(ind) for ind in combined]
        fronts = fast_nondominated_sort(combined_objs)
        new_sel: List[Encoding] = []
        elite_keep = max(1, int(len(population) * self.config.elite_keep_ratio))
        elite_pool = select_leader_pool(objs, elite_keep)
        for idx in elite_pool:
            if len(new_sel) < len(population):
                new_sel.append(population[idx])
        for front in fronts:
            if len(new_sel) + len(front) <= len(population):
                for i in front:
                    cand = combined[i]
                    if cand not in new_sel:
                        new_sel.append(cand)
            else:
                dist = crowding_distance(front, combined_objs)
                sorted_front = sorted(front, key=lambda idx: dist.get(idx, 0.0), reverse=True)
                for i in sorted_front:
                    if len(new_sel) >= len(population):
                        break
                    cand = combined[i]
                    if cand not in new_sel:
                        new_sel.append(cand)
                break
        if self.config.opposition_ratio > 0:
            new_sel = apply_opposition(
                self._rng,
                instance,
                new_sel,
                self.config.opposition_ratio,
            )
        self._iter += 1
        return new_sel

    def _energy(self) -> float:
        if self.config.energy_decay == "linear":
            t = min(self._iter, self.config.max_iters)
            return 2 * self.config.e0 * (1 - t / self.config.max_iters) - self.config.e0
        if self.config.energy_decay == "nonlinear":
            t = min(self._iter, self.config.max_iters)
            r = t / max(1, self.config.max_iters)
            # Nonlinear decay: slower early decrease, stronger late exploitation.
            return 2 * self.config.e0 * (1 - r * r) - self.config.e0
        return self.config.e0


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


def perturb(rng: random.Random, instance: ProblemInstance, enc: Encoding, rate: float) -> Encoding:
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    fa = enc.factory_assignment[:]
    seq = enc.job_sequence[:]
    ma = [row[:] for row in enc.machine_assignment]

    for j in range(J):
        if rng.random() < rate:
            fa[j] = rng.randrange(F)
            for s in range(S):
                ma[j][s] = rng.randrange(instance.machines[fa[j]][s])

    if rng.random() < rate and J >= 2:
        i, k = rng.sample(range(J), 2)
        seq[i], seq[k] = seq[k], seq[i]

    for j in range(J):
        for s in range(S):
            if rng.random() < rate:
                ma[j][s] = rng.randrange(instance.machines[fa[j]][s])

    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


def attract(
    rng: random.Random, instance: ProblemInstance, enc: Encoding, best: Encoding, rate: float
) -> Encoding:
    J, S = instance.num_jobs, instance.num_stages
    fa = enc.factory_assignment[:]
    seq = enc.job_sequence[:]
    ma = [row[:] for row in enc.machine_assignment]

    for j in range(J):
        if rng.random() < rate:
            fa[j] = best.factory_assignment[j]
        for s in range(S):
            if rng.random() < rate:
                ma[j][s] = best.machine_assignment[j][s]

    if rng.random() < rate:
        seq = best.job_sequence[:]
        if rng.random() < 0.3 and J >= 2:
            i, k = rng.sample(range(J), 2)
            seq[i], seq[k] = seq[k], seq[i]

    return Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)


def levy_dive(
    rng: random.Random, instance: ProblemInstance, enc: Encoding, beta: float
) -> Encoding:
    # Discrete Levy: perform a number of swaps proportional to Levy step
    import math

    sigma = (
        (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.random() * sigma
    v = max(1e-9, rng.random())
    step = abs(u / (v ** (1 / beta)))
    swaps = max(1, int(step) % max(1, instance.num_jobs))

    seq = enc.job_sequence[:]
    for _ in range(swaps):
        if len(seq) < 2:
            break
        i, k = rng.sample(range(len(seq)), 2)
        seq[i], seq[k] = seq[k], seq[i]

    return Encoding(
        factory_assignment=enc.factory_assignment[:],
        job_sequence=seq,
        machine_assignment=[row[:] for row in enc.machine_assignment],
    )


def dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def select_leader_pool(objs: List[Tuple[float, float, float]], k: int) -> List[int]:
    # Select top-k leaders from non-dominated set using crowding distance
    if not objs:
        return [0]
    nd_idx = []
    for i, a in enumerate(objs):
        if any(dominates(b, a) for j, b in enumerate(objs) if j != i):
            continue
        nd_idx.append(i)
    if not nd_idx:
        return [0]
    if len(nd_idx) <= k:
        return nd_idx
    crowd = crowding_distance(nd_idx, objs)
    sorted_nd = sorted(nd_idx, key=lambda i: crowd.get(i, 0.0), reverse=True)
    return sorted_nd[:k]


def crowding_distance(indices: List[int], objs: List[Tuple[float, float, float]]) -> dict:
    dist = {i: 0.0 for i in indices}
    if not indices:
        return dist
    for m in range(3):
        sorted_idx = sorted(indices, key=lambda i: objs[i][m])
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")
        min_v = objs[sorted_idx[0]][m]
        max_v = objs[sorted_idx[-1]][m]
        if max_v == min_v:
            continue
        for k in range(1, len(sorted_idx) - 1):
            prev_v = objs[sorted_idx[k - 1]][m]
            next_v = objs[sorted_idx[k + 1]][m]
            dist[sorted_idx[k]] += (next_v - prev_v) / (max_v - min_v)
    return dist


def fast_nondominated_sort(objs: List[Tuple[float, float, float]]) -> List[List[int]]:
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


def seed_population(rng: random.Random, instance: ProblemInstance, size: int) -> List[Encoding]:
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


def chaos_init_population(
    rng: random.Random, instance: ProblemInstance, size: int
) -> List[Encoding]:
    """混沌初始化：用 Logistic 映射生成具有遍历性的初始解，提升多样性"""
    J, S, F = instance.num_jobs, instance.num_stages, instance.num_factories
    pop: List[Encoding] = []
    x = rng.random() * 0.9 + 0.05  # 避免 0/1，保证混沌
    for _ in range(size):
        fa, seq, ma = [], list(range(J)), []
        for j in range(J):
            x = _chaos_logistic(x)
            f = min(int(x * F) % F, F - 1)
            fa.append(f)
        for i in range(J - 1, 0, -1):
            x = _chaos_logistic(x)
            k = int(x * (i + 1)) % (i + 1)
            seq[i], seq[k] = seq[k], seq[i]
        for j in range(J):
            row = []
            for s in range(S):
                x = _chaos_logistic(x)
                m_max = instance.machines[fa[j]][s]
                m = int(x * m_max) % m_max if m_max > 0 else 0
                row.append(m)
            ma.append(row)
        pop.append(
            Encoding(
                factory_assignment=fa,
                job_sequence=seq,
                machine_assignment=ma,
            )
        )
    return pop


def select_diverse(
    pool: List[Encoding], objs: List[Tuple[float, float, float]], k: int
) -> List[Encoding]:
    if k <= 0:
        return []
    selected: List[int] = []
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


def refine_assignment(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
    trials: int = 1,
) -> Encoding:
    """Jointly refine factory/machine assignment for f2+f3."""
    if rng.random() > prob or instance.num_factories <= 1:
        return enc
    J, S = instance.num_jobs, instance.num_stages
    if J == 0:
        return enc
    best = enc
    f1, f2, f3 = instance.evaluate(enc)
    best_score = f2 + 0.25 * f3
    for _ in range(max(1, trials)):
        j = rng.randrange(J)
        cur_f = best.factory_assignment[j]
        local_best = best
        local_score = best_score
        local_f1 = f1
        for cand_f in range(instance.num_factories):
            if cand_f == cur_f:
                continue
            fa = best.factory_assignment[:]
            fa[j] = cand_f
            ma = [row[:] for row in best.machine_assignment]
            for s in range(S):
                ma[j][s] = rng.randrange(instance.machines[cand_f][s])
            cand = Encoding(
                factory_assignment=fa,
                job_sequence=best.job_sequence[:],
                machine_assignment=ma,
            )
            c1, c2, c3 = instance.evaluate(cand)
            c_score = c2 + 0.25 * c3
            if c_score < local_score or (c_score == local_score and c1 < local_f1):
                local_best = cand
                local_f1 = c1
                local_score = c_score
        if local_best is not best:
            best = local_best
            f1, f2, f3 = instance.evaluate(best)
            best_score = local_score
    return best


def gaussian_research(
    rng: random.Random,
    instance: ProblemInstance,
    enc: Encoding,
    prob: float,
) -> Encoding:
    if rng.random() > prob:
        return enc
    J = len(enc.job_sequence)
    if J < 2:
        return enc
    seq = enc.job_sequence[:]
    swaps = max(1, int(abs(rng.gauss(0.0, 1.0)) * 2))
    for _ in range(swaps):
        i, k = rng.sample(range(J), 2)
        seq[i], seq[k] = seq[k], seq[i]
    return Encoding(
        factory_assignment=enc.factory_assignment[:],
        job_sequence=seq,
        machine_assignment=[row[:] for row in enc.machine_assignment],
    )


def apply_opposition(
    rng: random.Random,
    instance: ProblemInstance,
    population: List[Encoding],
    ratio: float,
) -> List[Encoding]:
    if not population:
        return population
    n = max(1, int(len(population) * ratio))
    scored = [(sum(instance.evaluate(ind)), i) for i, ind in enumerate(population)]
    scored.sort(reverse=True)  # worst first (minimization)
    new_pop = population[:]
    for _, idx in scored[:n]:
        ind = new_pop[idx]
        fa = [instance.num_factories - 1 - f for f in ind.factory_assignment]
        ma: List[List[int]] = []
        for j, row in enumerate(ind.machine_assignment):
            opp_row: List[int] = []
            for s, m in enumerate(row):
                m_max = instance.machines[fa[j]][s] - 1
                opp_row.append(max(0, m_max - min(m, m_max)))
            ma.append(opp_row)
        seq = list(reversed(ind.job_sequence))
        cand = Encoding(factory_assignment=fa, job_sequence=seq, machine_assignment=ma)
        if sum(instance.evaluate(cand)) < sum(instance.evaluate(ind)):
            new_pop[idx] = cand
    return new_pop

