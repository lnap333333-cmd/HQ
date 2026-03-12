"""High-level Q-learning (mode scheduler) (高层Q-learning)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class RelationMode(str, Enum):
    INDEPENDENT = "independent"
    COOPERATION = "cooperation"
    COMPETITION = "competition"


@dataclass(frozen=True)
class HighLevelState:
    delta_hv: float
    delta_cv: float
    stagnation: int


@dataclass(frozen=True)
class HighLevelDecision:
    mode: RelationMode


class HighLevelQLearning:
    """
    高层Q-learning框架:
    - observe(): 观测状态
    - select_action(): 选择模式
    - update(): Q更新
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        switch_cost: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.switch_cost = switch_cost
        self._rng = random.Random()
        self._q: Dict[Tuple[int, int, int], Dict[RelationMode, float]] = {}
        self._last_action: RelationMode | None = None

    def observe(self, delta_hv: float, delta_cv: float, stagnation: int) -> HighLevelState:
        return HighLevelState(delta_hv=delta_hv, delta_cv=delta_cv, stagnation=stagnation)

    def select_action(self, state: HighLevelState) -> HighLevelDecision:
        return self.select_action_with_bias(state, {})

    def select_action_with_bias(
        self, state: HighLevelState, mode_bias: Dict[RelationMode, float]
    ) -> HighLevelDecision:
        s = self._discretize(state)
        if self._rng.random() < self.epsilon:
            mode = self._rng.choice(list(RelationMode))
        else:
            q = self._q.get(s, {})
            mode = max(RelationMode, key=lambda a: q.get(a, 0.0) + mode_bias.get(a, 0.0))
        return HighLevelDecision(mode=mode)

    def update(
        self,
        prev_state: HighLevelState,
        action: HighLevelDecision,
        reward: float,
        next_state: HighLevelState,
    ) -> None:
        s = self._discretize(prev_state)
        ns = self._discretize(next_state)
        q = self._q.setdefault(s, {})
        cur = q.get(action.mode, 0.0)
        next_q = self._q.get(ns, {})
        max_next = max(next_q.values(), default=0.0)
        q[action.mode] = cur + self.alpha * (reward + self.gamma * max_next - cur)

    def compute_reward(
        self, delta_hv: float, delta_cv: float, stagnation: int, mode: RelationMode
    ) -> float:
        switch_cost = self.switch_cost if self._last_action and self._last_action != mode else 0.0
        return 1.0 * delta_hv - 1.0 * delta_cv - 0.1 * stagnation - switch_cost

    def record_action(self, mode: RelationMode) -> None:
        self._last_action = mode

    def _discretize(self, s: HighLevelState) -> Tuple[int, int, int]:
        dh = 1 if s.delta_hv > 0 else 0
        dc = 1 if s.delta_cv > 0 else 0
        st = 0 if s.stagnation < 5 else 1 if s.stagnation < 15 else 2
        return (dh, dc, st)

