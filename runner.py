"""General execution procedure (总体执行流程)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .algorithms.hho import HHO
from .algorithms.ig import IG
from .algorithms.moead import MOEAD
from .algorithms.nsga2 import NSGA2
from .archive import ArchiveEntry, EliteArchive
from .config import GlobalConfig, HighLevelConfig, LowLevelConfig
from .encoding import Encoding
from .init_pool import ensure_pool
from .interactions.c1_elite_migration import apply as apply_c1
from .interactions.c2_rhythm_coop import apply as apply_c2
from .interactions.r1_struct_suppress import apply as apply_r1
from .interactions.r2_territorial_invade import apply as apply_r2
from .instance import ProblemInstance
from .logging_utils import JsonlLogger
from .metrics.cv import cv
from .metrics.contribution import nondominated, overlap_ratio, pearson
from .metrics.hv import hv_approx
from .rl.high_level_q import HighLevelQLearning, RelationMode
from .rl.low_level_q import InteractionOp, LowLevelQLearning


@dataclass(frozen=True)
class RunnerConfig:
    output_dir: str = "outputs/run"
    log_every: int = 10
    reference_point: Tuple[float, float, float] = (1e4, 1e4, 1e4)
    verbose: bool = True


def run(instance: ProblemInstance, cfg: GlobalConfig, runner_cfg: RunnerConfig) -> EliteArchive:
    instance.validate()
    out_dir = Path(runner_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLogger(out_dir / "progress.jsonl")

    # 统一候选池 (LHS)
    max_n = max(cfg.nsga2.population_size, cfg.moead.num_subproblems, cfg.population_size // 4)
    ensure_pool(instance, size=max_n * 4, seed=1)

    # 初始化算法
    nsga = NSGA2(cfg.nsga2)
    moead = MOEAD(cfg.moead)
    hho = HHO(cfg.hho)
    ig = IG(cfg.ig)

    # 初始种群
    nsga_size = cfg.nsga2.population_size
    moead_size = cfg.moead.num_subproblems
    ig_size = max(1, int(round(cfg.population_size * 0.2)))
    hho_size = cfg.population_size - nsga_size - moead_size - ig_size
    if hho_size < 1:
        # Fallback to equal split when sizes are not feasible
        hho_size = max(1, cfg.population_size // 4)
        ig_size = max(1, cfg.population_size // 4)
    pop_nsga = nsga.initialize(instance, nsga_size)
    pop_moead = moead.initialize(instance, moead_size)
    pop_hho = hho.initialize(instance, hho_size)
    pop_ig = ig.initialize(instance, ig_size)

    archive = EliteArchive()
    step = 0
    fe_count = 0

    qh = HighLevelQLearning(
        alpha=cfg.high_level.alpha,
        gamma=cfg.high_level.gamma,
        epsilon=cfg.high_level.epsilon,
        switch_cost=cfg.high_level.switch_cost,
    )
    ql = LowLevelQLearning(
        alpha=cfg.low_level.alpha, gamma=cfg.low_level.gamma, epsilon=cfg.low_level.epsilon
    )
    stagnation = 0
    last_hv = 0.0
    last_cv = 0.0
    mc_history: Dict[str, List[float]] = {"nsga": [], "moead": [], "hho": [], "ig": []}

    hv_history: List[float] = []
    cv_history: List[float] = []
    mc_smoothed: Dict[str, float] = {"nsga": 0.0, "moead": 0.0, "hho": 0.0, "ig": 0.0}
    assist_credit: Dict[str, float] = {"nsga": 0.0, "moead": 0.0, "hho": 0.0, "ig": 0.0}
    assist_mc_raw: Dict[str, float] = {"nsga": 0.0, "moead": 0.0, "hho": 0.0, "ig": 0.0}
    pending_interactions: List[Dict[str, Any]] = []

    mode_counts: Dict[str, int] = {"independent": 0, "cooperation": 0, "competition": 0}
    op_counts: Dict[str, int] = {"c1_elite_migration": 0, "c2_rhythm_coop": 0, "r1_struct_suppress": 0, "r2_territorial_invade": 0}
    op_success: Dict[str, int] = {"c1_elite_migration": 0, "c2_rhythm_coop": 0, "r1_struct_suppress": 0, "r2_territorial_invade": 0}

    while step < cfg.max_iters:
        for k in assist_credit.keys():
            assist_credit[k] *= cfg.low_level.assist_decay

        if runner_cfg.verbose:
            print(f"[迭代 {step}] 步骤1：原生演化")
            print(f"[迭代 {step}]  ├─ NSGA-II：进化一代")
        pop_nsga = nsga.step(instance, pop_nsga)
        if runner_cfg.verbose:
            print(f"[迭代 {step}]  ├─ MOEA/D：进化一代")
        pop_moead = moead.step(instance, pop_moead)
        if runner_cfg.verbose:
            print(f"[迭代 {step}]  ├─ HHO：更新一代")
        pop_hho = hho.step(instance, pop_hho)
        if runner_cfg.verbose:
            print(f"[迭代 {step}]  └─ IG：迭代贪婪")
        pop_ig = ig.step(instance, pop_ig)

        # Step2: 更新档案
        if runner_cfg.verbose:
            print(f"[迭代 {step}] 步骤2：更新档案")
        entries: List[ArchiveEntry] = []
        for name, pop in (("nsga", pop_nsga), ("moead", pop_moead), ("hho", pop_hho), ("ig", pop_ig)):
            for ind in pop:
                entries.append(ArchiveEntry(ind, instance.evaluate(ind), source=name))
                fe_count += 1
        archive.update(entries)

        # Step3: 高层决策
        if runner_cfg.verbose:
            print(f"[迭代 {step}] 步骤3：高层决策")
        hv_val = hv_approx(archive.as_objective_matrix(), runner_cfg.reference_point)
        cv_val = cv(archive.as_objective_matrix())
        settled_next: List[Dict[str, Any]] = []
        for rec in pending_interactions:
            if rec["settle_step"] > step:
                settled_next.append(rec)
                continue
            gain = _delayed_credit_gain(
                hv_base=float(rec["hv_base"]),
                cv_base=float(rec["cv_base"]),
                hv_now=hv_val,
                cv_now=cv_val,
                mc_weight=cfg.low_level.mc_weight,
                reward_scale=cfg.low_level.assist_reward_scale,
            )
            if gain <= 0:
                continue
            shares: Dict[str, float] = rec.get("shares", {})
            for algo, share in shares.items():
                credit = gain * max(0.0, float(share))
                if credit <= 0:
                    continue
                assist_mc_raw[algo] = assist_mc_raw.get(algo, 0.0) + credit
                assist_credit[algo] = min(1.0, assist_credit.get(algo, 0.0) + credit)
        pending_interactions = settled_next
        hv_history.append(hv_val)
        cv_history.append(cv_val)
        T = cfg.high_level.relation_step
        if len(hv_history) > T:
            delta_hv = hv_val - hv_history[-T - 1]
            delta_cv = cv_val - cv_history[-T - 1]
        else:
            delta_hv = hv_val - last_hv
            delta_cv = cv_val - last_cv
        if delta_hv <= 1e-9:
            stagnation += 1
        else:
            stagnation = 0
        state_h = qh.observe(delta_hv, delta_cv, stagnation)
        source_share = _source_share(archive.entries, ["nsga", "moead", "hho", "ig"])
        mode_bias = _mode_bias(
            step=step,
            max_iters=cfg.max_iters,
            stagnation=stagnation,
            source_share=source_share,
            high_cfg=cfg.high_level,
            low_cfg=cfg.low_level,
        )
        decision = qh.select_action_with_bias(state_h, mode_bias)
        decision = qh.select_action_with_bias(
            state_h,
            {
                _scheduled_mode(decision.mode, step, cfg.max_iters, stagnation, cfg.high_level): 0.15
            },
        )
        qh.record_action(decision.mode)
        mode_counts[decision.mode.value] = mode_counts.get(decision.mode.value, 0) + 1

        # Step4: 低层交互
        if runner_cfg.verbose:
            print(f"[迭代 {step}] 步骤4：低层交互（{decision.mode.value}）")
        pops: Dict[str, List[Encoding]] = {
            "nsga": pop_nsga,
            "moead": pop_moead,
            "hho": pop_hho,
            "ig": pop_ig,
        }
        # MC based on archive sources (EA_i)
        ea_all = [e.objectives for e in archive.entries]
        hv_ea = hv_approx(ea_all, runner_cfg.reference_point)
        cv_ea = cv(ea_all)
        mc: Dict[str, float] = {}
        for k in pops.keys():
            ea_i = [e.objectives for e in archive.entries if e.source == k]
            ea_minus = [e.objectives for e in archive.entries if e.source != k]
            hv_minus = hv_approx(ea_minus, runner_cfg.reference_point) if ea_minus else 0.0
            cv_minus = cv(ea_minus) if ea_minus else 0.0
            q = (hv_ea - hv_minus) / (hv_ea + 1e-9)
            s = max(0.0, cv_minus - cv_ea)
            raw = cfg.low_level.mc_weight * q + (1 - cfg.low_level.mc_weight) * s
            # smoothing
            mc_smoothed[k] = cfg.low_level.mc_smooth * mc_smoothed[k] + (1 - cfg.low_level.mc_smooth) * raw
            mc[k] = mc_smoothed[k]
            mc_history[k].append(mc[k])

        if decision.mode != RelationMode.INDEPENDENT:
            keys = list(pops.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]
                    ea_i = [e.objectives for e in archive.entries if e.source == ki]
                    ea_j = [e.objectives for e in archive.entries if e.source == kj]
                    overlap = overlap_ratio(ea_i, ea_j, ea_all)
                    corr = pearson(mc_history[ki], mc_history[kj])
                    state_l = ql.observe(mc[ki], mc[kj], overlap, corr)
                    decision_l = ql.select_action_with_bias(
                        state_l,
                        _op_bias(
                            mode=decision.mode,
                            ki=ki,
                            kj=kj,
                            stagnation=stagnation,
                            source_share=source_share,
                            low_cfg=cfg.low_level,
                        ),
                    )

                    # choose winner/loser by fairness-adjusted MC
                    score = _adjusted_mc_scores(
                        mc=mc,
                        archive_entries=archive.entries,
                        progress=(step + 1) / max(1, cfg.max_iters),
                        assist_credit=assist_credit,
                        low_cfg=cfg.low_level,
                    )
                    if score[ki] >= score[kj]:
                        winner, loser = ki, kj
                    else:
                        winner, loser = kj, ki

                    if decision.mode == RelationMode.COOPERATION:
                        executed_op = "c2_rhythm_coop"
                        if decision_l.op.value == "c1_elite_migration":
                            if runner_cfg.verbose:
                                print(f"[迭代 {step}]  ├─ C1 协同：{winner} -> {loser}")
                            pops[winner], pops[loser] = apply_c1(
                                instance,
                                pops[winner],
                                pops[loser],
                                rate=_interaction_rate(
                                    mode=decision.mode,
                                    op="c1_elite_migration",
                                    winner=winner,
                                    loser=loser,
                                    stagnation=stagnation,
                                    source_share=source_share,
                                    low_cfg=cfg.low_level,
                                ),
                            )
                            op_counts["c1_elite_migration"] += 1
                            executed_op = "c1_elite_migration"
                        else:
                            if runner_cfg.verbose:
                                print(f"[迭代 {step}]  ├─ C2 协同：{ki} <-> {kj}")
                            pops[ki], pops[kj] = apply_c2(
                                instance,
                                pops[ki],
                                pops[kj],
                                rate=_interaction_rate(
                                    mode=decision.mode,
                                    op="c2_rhythm_coop",
                                    winner=winner,
                                    loser=loser,
                                    stagnation=stagnation,
                                    source_share=source_share,
                                    low_cfg=cfg.low_level,
                                ),
                            )
                            op_counts["c2_rhythm_coop"] += 1
                    elif decision.mode == RelationMode.COMPETITION:
                        executed_op = "r2_territorial_invade"
                        if decision_l.op.value == "r1_struct_suppress":
                            if runner_cfg.verbose:
                                print(f"[迭代 {step}]  ├─ R1 竞争：{winner} 抑制 {loser}")
                            pops[winner], pops[loser] = apply_r1(
                                instance,
                                pops[winner],
                                pops[loser],
                                rate=_interaction_rate(
                                    mode=decision.mode,
                                    op="r1_struct_suppress",
                                    winner=winner,
                                    loser=loser,
                                    stagnation=stagnation,
                                    source_share=source_share,
                                    low_cfg=cfg.low_level,
                                ),
                            )
                            op_counts["r1_struct_suppress"] += 1
                            executed_op = "r1_struct_suppress"
                        else:
                            if runner_cfg.verbose:
                                print(f"[迭代 {step}]  ├─ R2 竞争：{winner} 入侵 {loser}")
                            pops[winner], pops[loser] = apply_r2(
                                instance,
                                pops[winner],
                                pops[loser],
                                rate=_interaction_rate(
                                    mode=decision.mode,
                                    op="r2_territorial_invade",
                                    winner=winner,
                                    loser=loser,
                                    stagnation=stagnation,
                                    source_share=source_share,
                                    low_cfg=cfg.low_level,
                                ),
                            )
                            op_counts["r2_territorial_invade"] += 1

                    next_state = ql.observe(mc[ki], mc[kj], overlap, corr)
                    prev_sum = mc_history[ki][-2] + mc_history[kj][-2] if len(mc_history[ki]) > 1 else 0.0
                    reward_l = (mc[ki] + mc[kj]) - prev_sum
                    ql.update(state_l, decision_l, reward_l, next_state)
                    if reward_l > 0:
                        op_success[executed_op] += 1
                        _update_assist_credit(
                            assist_credit=assist_credit,
                            mode=decision.mode,
                            op=executed_op,
                            winner=winner,
                            loser=loser,
                        )
                    pending_interactions.append(
                        {
                            "settle_step": step + max(1, cfg.low_level.assist_delay_steps),
                            "hv_base": hv_val,
                            "cv_base": cv_val,
                            "shares": _interaction_credit_shares(
                                executed_op=executed_op,
                                ki=ki,
                                kj=kj,
                                winner=winner,
                                loser=loser,
                                low_cfg=cfg.low_level,
                            ),
                        }
                    )

            pop_nsga, pop_moead, pop_hho, pop_ig = (
                pops["nsga"],
                pops["moead"],
                pops["hho"],
                pops["ig"],
            )

        # High-level update
        reward_h = qh.compute_reward(delta_hv, delta_cv, stagnation, decision.mode)
        next_state_h = qh.observe(delta_hv, delta_cv, stagnation)
        qh.update(state_h, decision, reward_h, next_state_h)

        # Step5: 记录
        if runner_cfg.verbose:
            print(f"[迭代 {step}] 步骤5：记录与更新")
        if step % runner_cfg.log_every == 0:
            logger.log(
                {
                    "step": step,
                    "fe": fe_count,
                    "archive_size": len(archive),
                    "hv": hv_val,
                    "cv": cv_val,
                    "mode": decision.mode.value,
                    "mc": mc,
                }
            )

        last_hv = hv_val
        last_cv = cv_val
        step += 1

    _save_archive(out_dir / "archive.json", archive)
    _save_summary(
        out_dir / "summary.json",
        archive,
        mode_counts,
        op_counts,
        op_success,
        reference_point=runner_cfg.reference_point,
        mc_weight=cfg.low_level.mc_weight,
        assist_mc_raw=assist_mc_raw,
        assist_blend_lambda=cfg.low_level.assist_blend_lambda,
    )
    return archive


def run_single(
    instance: ProblemInstance,
    cfg: GlobalConfig,
    runner_cfg: RunnerConfig,
    algo: str,
) -> EliteArchive:
    """
    Run a single algorithm independently for comparison.
    algo: "nsga", "moead", "hho", "ig"
    """
    instance.validate()
    out_dir = Path(runner_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "progress.jsonl")

    max_n = max(cfg.nsga2.population_size, cfg.moead.num_subproblems, cfg.population_size // 4)
    ensure_pool(instance, size=max_n * 4, seed=1)

    if algo == "nsga":
        alg = NSGA2(cfg.nsga2)
        pop = alg.initialize(instance, cfg.nsga2.population_size)
    elif algo == "moead":
        alg = MOEAD(cfg.moead)
        pop = alg.initialize(instance, cfg.moead.num_subproblems)
    elif algo == "hho":
        alg = HHO(cfg.hho)
        pop = alg.initialize(instance, cfg.population_size // 4)
    elif algo == "ig":
        alg = IG(cfg.ig)
        pop = alg.initialize(instance, cfg.population_size // 4)
    else:
        raise ValueError(f"unknown algo: {algo}")

    archive = EliteArchive()
    step = 0
    fe_count = 0
    last_hv = 0.0
    last_cv = 0.0

    while step < cfg.max_iters:
        if runner_cfg.verbose:
            print(f"[迭代 {step}] 单算法({algo})：进化一代")
        pop = alg.step(instance, pop)

        entries: List[ArchiveEntry] = []
        for ind in pop:
            entries.append(ArchiveEntry(ind, instance.evaluate(ind), source=algo))
            fe_count += 1
        archive.update(entries)

        hv_val = hv_approx(archive.as_objective_matrix(), runner_cfg.reference_point)
        cv_val = cv(archive.as_objective_matrix())

        if step % runner_cfg.log_every == 0:
            logger.log(
                {
                    "step": step,
                    "fe": fe_count,
                    "archive_size": len(archive),
                    "hv": hv_val,
                    "cv": cv_val,
                    "mode": "single",
                }
            )

        last_hv = hv_val
        last_cv = cv_val
        step += 1

    _save_archive(out_dir / "archive.json", archive)
    _save_summary(out_dir / "summary.json", archive, None, None, None)
    return archive


def _save_archive(path: Path, archive: EliteArchive) -> None:
    import json

    payload = {
        "size": len(archive),
        "objectives": [list(e.objectives) for e in archive.entries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_summary(
    path: Path,
    archive: EliteArchive,
    mode_counts: Dict[str, int] | None,
    op_counts: Dict[str, int] | None,
    op_success: Dict[str, int] | None,
    reference_point: Tuple[float, float, float] = (1e4, 1e4, 1e4),
    mc_weight: float = 0.3,
    assist_mc_raw: Dict[str, float] | None = None,
    assist_blend_lambda: float = 0.35,
) -> None:
    import json

    total = len(archive.entries)
    contrib: Dict[str, int] = {}
    for e in archive.entries:
        if e.source is None:
            continue
        contrib[e.source] = contrib.get(e.source, 0) + 1
    cr = {k: (v / total if total > 0 else 0.0) for k, v in contrib.items()}
    marginal_raw, marginal_rate = _compute_marginal_contribution(
        archive.entries, reference_point, mc_weight
    )
    algos = ("nsga", "moead", "hho", "ig")
    assist_raw = {k: float((assist_mc_raw or {}).get(k, 0.0)) for k in algos}
    assist_rate = _normalize_rates(assist_raw)
    total_raw = {k: float(marginal_raw.get(k, 0.0)) + assist_raw[k] for k in algos}
    total_rate = _normalize_rates(total_raw)
    blend_lambda = min(1.0, max(0.0, assist_blend_lambda))
    blended_rate = {
        k: (1.0 - blend_lambda) * float(marginal_rate.get(k, 0.0)) + blend_lambda * assist_rate[k]
        for k in algos
    }
    blended_rate = _normalize_rates(blended_rate)
    blended_raw = {
        k: (1.0 - blend_lambda) * float(marginal_raw.get(k, 0.0)) + blend_lambda * assist_raw[k]
        for k in algos
    }
    payload = {
        "archive_size": total,
        "contribution_rate": cr,
        "marginal_contribution": marginal_raw,
        "marginal_contribution_rate": marginal_rate,
        "assist_marginal_contribution": assist_raw,
        "assist_marginal_contribution_rate": assist_rate,
        "total_marginal_contribution": total_raw,
        "total_marginal_contribution_rate": total_rate,
        "blended_marginal_contribution": blended_raw,
        "blended_marginal_contribution_rate": blended_rate,
        "mode_counts": mode_counts,
        "operator_counts": op_counts,
        "operator_success": op_success,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_marginal_contribution(
    entries: List[ArchiveEntry],
    reference_point: Tuple[float, float, float],
    mc_weight: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    algos = ["nsga", "moead", "hho", "ig"]
    if not entries:
        zero = {k: 0.0 for k in algos}
        return zero, zero

    ea_all = [e.objectives for e in entries]
    hv_all = hv_approx(ea_all, reference_point)
    cv_all = cv(ea_all)

    raw: Dict[str, float] = {}
    for k in algos:
        ea_minus = [e.objectives for e in entries if e.source != k]
        if not ea_minus:
            raw[k] = 0.0
            continue
        hv_minus = hv_approx(ea_minus, reference_point)
        cv_minus = cv(ea_minus)
        q = (hv_all - hv_minus) / (hv_all + 1e-9)
        s = max(0.0, cv_minus - cv_all)
        raw[k] = mc_weight * q + (1 - mc_weight) * s

    positive = {k: max(0.0, v) for k, v in raw.items()}
    total_pos = sum(positive.values())
    if total_pos <= 1e-12:
        rate = {k: 0.0 for k in algos}
    else:
        rate = {k: positive[k] / total_pos for k in algos}
    return raw, rate


def _normalize_rates(raw: Dict[str, float]) -> Dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in raw.items()}
    total_pos = sum(positive.values())
    if total_pos <= 1e-12:
        return {k: 0.0 for k in raw.keys()}
    return {k: positive[k] / total_pos for k in raw.keys()}


def _delayed_credit_gain(
    hv_base: float,
    cv_base: float,
    hv_now: float,
    cv_now: float,
    mc_weight: float,
    reward_scale: float,
) -> float:
    hv_gain = max(0.0, (hv_now - hv_base) / (abs(hv_base) + 1e-9))
    cv_gain = max(0.0, cv_base - cv_now)
    return max(0.0, reward_scale * (mc_weight * hv_gain + (1 - mc_weight) * cv_gain))


def _interaction_credit_shares(
    executed_op: str,
    ki: str,
    kj: str,
    winner: str,
    loser: str,
    low_cfg: LowLevelConfig,
) -> Dict[str, float]:
    if executed_op == "c2_rhythm_coop":
        return {ki: 0.5, kj: 0.5}
    return {
        winner: max(0.0, low_cfg.assist_winner_share),
        loser: max(0.0, low_cfg.assist_loser_share),
    }


def _scheduled_mode(
    base_mode: RelationMode,
    step: int,
    max_iters: int,
    stagnation: int,
    high_cfg: HighLevelConfig,
) -> RelationMode:
    progress = (step + 1) / max(1, max_iters)
    if progress < high_cfg.explore_phase_ratio:
        if stagnation >= high_cfg.stagnation_force_competition:
            return RelationMode.COMPETITION
        # Early phase: softly encourage cooperation only for independent actions.
        if base_mode == RelationMode.INDEPENDENT:
            return RelationMode.COOPERATION
        return base_mode

    if progress < high_cfg.exploit_phase_ratio:
        if stagnation >= high_cfg.stagnation_force_competition:
            return RelationMode.COMPETITION
        if base_mode == RelationMode.INDEPENDENT and stagnation >= high_cfg.stagnation_force_coop:
            return RelationMode.COOPERATION
        return base_mode

    # Late phase: prefer stability and targeted cooperation
    if stagnation >= high_cfg.stagnation_force_coop:
        return RelationMode.COOPERATION
    if base_mode == RelationMode.COMPETITION and stagnation == 0:
        return RelationMode.INDEPENDENT
    return base_mode


def _mode_bias(
    step: int,
    max_iters: int,
    stagnation: int,
    source_share: Dict[str, float],
    high_cfg: HighLevelConfig,
    low_cfg: LowLevelConfig,
) -> Dict[RelationMode, float]:
    progress = (step + 1) / max(1, max_iters)
    bias = {
        RelationMode.INDEPENDENT: 0.0,
        RelationMode.COOPERATION: 0.0,
        RelationMode.COMPETITION: 0.0,
    }
    if progress < high_cfg.explore_phase_ratio:
        bias[RelationMode.COOPERATION] += 0.08
        bias[RelationMode.COMPETITION] += 0.06
        bias[RelationMode.INDEPENDENT] -= 0.02
    elif progress >= high_cfg.exploit_phase_ratio:
        bias[RelationMode.INDEPENDENT] += 0.08
        bias[RelationMode.COMPETITION] += 0.04
        bias[RelationMode.COOPERATION] -= 0.03

    if stagnation >= high_cfg.stagnation_force_coop:
        bias[RelationMode.COOPERATION] += 0.10
    if stagnation >= high_cfg.stagnation_force_competition:
        bias[RelationMode.COMPETITION] += 0.10

    # Weak HHO acts as a soft signal to increase cooperation, not a hard override.
    hho_share = source_share.get("hho", 0.0)
    weak_thr = max(low_cfg.weak_share_threshold, low_cfg.target_share * 0.5)
    if hho_share < weak_thr:
        bias[RelationMode.COOPERATION] += 0.03
        bias[RelationMode.COMPETITION] -= 0.01
    return bias


def _op_bias(
    mode: RelationMode,
    ki: str,
    kj: str,
    stagnation: int,
    source_share: Dict[str, float],
    low_cfg: LowLevelConfig,
) -> Dict[InteractionOp, float]:
    bias: Dict[InteractionOp, float] = {}
    if "hho" not in (ki, kj):
        return bias
    hho_share = source_share.get("hho", 0.0)
    weak_thr = max(low_cfg.weak_share_threshold, low_cfg.target_share * 0.5)
    if hho_share >= weak_thr:
        return bias
    if mode == RelationMode.COOPERATION:
        bias[InteractionOp.C2] = low_cfg.hho_pair_c2_bias
        bias[InteractionOp.C1] = -0.5 * low_cfg.hho_pair_c2_bias
        if stagnation > 0:
            bias[InteractionOp.C2] += low_cfg.hho_stagnation_c2_bonus
    elif mode == RelationMode.COMPETITION:
        bias[InteractionOp.R2] = low_cfg.hho_pair_r2_bias
        bias[InteractionOp.R1] = -0.4 * low_cfg.hho_pair_r2_bias
        if stagnation > 0:
            bias[InteractionOp.R2] += low_cfg.hho_stagnation_r2_bonus
    return bias


def _interaction_rate(
    mode: RelationMode,
    op: str,
    winner: str,
    loser: str,
    stagnation: int,
    source_share: Dict[str, float],
    low_cfg: LowLevelConfig,
) -> float:
    base = 0.1
    if op == "r1_struct_suppress":
        base = 0.2
    hho_share = source_share.get("hho", 0.0)
    weak_thr = max(low_cfg.weak_share_threshold, low_cfg.target_share * 0.5)
    if hho_share >= weak_thr:
        return base
    if mode == RelationMode.COOPERATION and op == "c1_elite_migration" and loser == "hho":
        return max(0.02, base * low_cfg.hho_coop_absorb_rate_scale)
    if mode == RelationMode.COMPETITION and loser == "hho":
        return max(0.02, base * low_cfg.hho_compete_loss_rate_scale)
    if mode == RelationMode.COMPETITION and winner == "hho":
        boost = low_cfg.hho_compete_win_rate_scale * (1.2 if stagnation > 0 else 1.0)
        return min(0.25, base * boost)
    return base


def _adjusted_mc_scores(
    mc: Dict[str, float],
    archive_entries: List[ArchiveEntry],
    progress: float,
    assist_credit: Dict[str, float],
    low_cfg: LowLevelConfig,
) -> Dict[str, float]:
    total = len(archive_entries)
    share = {k: 0.0 for k in mc.keys()}
    if total > 0:
        counts = {k: 0 for k in mc.keys()}
        for e in archive_entries:
            if e.source in counts:
                counts[e.source] += 1
        share = {k: counts[k] / total for k in counts.keys()}

    score: Dict[str, float] = {}
    for k, v in mc.items():
        deficit = max(0.0, low_cfg.target_share - share.get(k, 0.0))
        s = v + low_cfg.fairness_boost * deficit
        s += low_cfg.assist_boost * assist_credit.get(k, 0.0)
        if k == "hho":
            s += low_cfg.hho_priority_boost * deficit
        if progress < low_cfg.early_phase_ratio and k == "ig":
            # Suppress early IG dominance to keep multi-algorithm exploration active.
            surplus = max(0.0, share.get(k, 0.0) - low_cfg.target_share)
            s -= low_cfg.early_ig_penalty * surplus
        score[k] = s
    return score


def _source_share(entries: List[ArchiveEntry], keys: List[str]) -> Dict[str, float]:
    total = len(entries)
    share = {k: 0.0 for k in keys}
    if total <= 0:
        return share
    counts = {k: 0 for k in keys}
    for e in entries:
        if e.source in counts:
            counts[e.source] += 1
    return {k: counts[k] / total for k in keys}


def _update_assist_credit(
    assist_credit: Dict[str, float],
    mode: RelationMode,
    op: str,
    winner: str,
    loser: str,
) -> None:
    if mode == RelationMode.COOPERATION:
        if op == "c1_elite_migration":
            assist_credit[winner] = min(1.0, assist_credit.get(winner, 0.0) + 0.10)
            assist_credit[loser] = min(1.0, assist_credit.get(loser, 0.0) + 0.08)
        else:  # c2_rhythm_coop
            assist_credit[winner] = min(1.0, assist_credit.get(winner, 0.0) + 0.08)
            assist_credit[loser] = min(1.0, assist_credit.get(loser, 0.0) + 0.08)
    elif mode == RelationMode.COMPETITION:
        if op == "r1_struct_suppress":
            assist_credit[winner] = min(1.0, assist_credit.get(winner, 0.0) + 0.10)
            assist_credit[loser] = min(1.0, assist_credit.get(loser, 0.0) + 0.04)
        else:  # r2_territorial_invade
            assist_credit[winner] = min(1.0, assist_credit.get(winner, 0.0) + 0.09)
            assist_credit[loser] = min(1.0, assist_credit.get(loser, 0.0) + 0.03)

