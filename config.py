"""Unified configuration (参数配置)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .algorithms.hho import HHOConfig
from .algorithms.ig import IGConfig
from .algorithms.moead import MOEADConfig
from .algorithms.nsga2 import NSGA2Config


@dataclass(frozen=True)
class HighLevelConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    switch_cost: float = 1.0
    relation_step: int = 5
    explore_phase_ratio: float = 0.35
    exploit_phase_ratio: float = 0.8
    stagnation_force_coop: int = 6
    stagnation_force_competition: int = 10


@dataclass(frozen=True)
class LowLevelConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    mc_weight: float = 0.3
    mc_smooth: float = 0.3
    target_share: float = 0.25
    fairness_boost: float = 0.4
    early_ig_penalty: float = 0.3
    early_phase_ratio: float = 0.4
    assist_boost: float = 0.25
    assist_decay: float = 0.97
    assist_delay_steps: int = 3
    assist_reward_scale: float = 0.15
    assist_winner_share: float = 0.65
    assist_loser_share: float = 0.35
    assist_blend_lambda: float = 0.45
    weak_share_threshold: float = 0.12
    hho_priority_boost: float = 0.6
    hho_competition_protect: bool = True
    hho_coop_absorb_rate_scale: float = 0.25
    hho_compete_loss_rate_scale: float = 0.25
    hho_pair_c2_bias: float = 0.18
    hho_pair_r2_bias: float = 0.14
    hho_stagnation_c2_bonus: float = 0.1
    hho_stagnation_r2_bonus: float = 0.1
    hho_compete_win_rate_scale: float = 1.6


@dataclass(frozen=True)
class GlobalConfig:
    population_size: int = 100
    max_iters: int = 200
    nsga2: NSGA2Config = NSGA2Config()
    moead: MOEADConfig = MOEADConfig()
    hho: HHOConfig = HHOConfig()
    ig: IGConfig = IGConfig()
    high_level: HighLevelConfig = HighLevelConfig()
    low_level: LowLevelConfig = LowLevelConfig()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GlobalConfig":
        return GlobalConfig(
            population_size=int(d.get("population_size", 100)),
            max_iters=int(d.get("max_iters", 200)),
            nsga2=NSGA2Config(**d.get("nsga2", {})),
            moead=MOEADConfig(**d.get("moead", {})),
            hho=HHOConfig(**d.get("hho", {})),
            ig=IGConfig(**d.get("ig", {})),
            high_level=HighLevelConfig(**d.get("high_level", {})),
            low_level=LowLevelConfig(**d.get("low_level", {})),
        )
