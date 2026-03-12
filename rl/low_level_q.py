"""Low-level Q-learning (interaction selector) (低层Q-learning)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class InteractionOp(str, Enum):
    C1 = "c1_elite_migration"
    C2 = "c2_rhythm_coop"
    R1 = "r1_struct_suppress"
    R2 = "r2_territorial_invade"


@dataclass(frozen=True)
class LowLevelState:
    mc_i: float
    mc_j: float
    overlap: float
    corr: float


@dataclass(frozen=True)
class LowLevelDecision:
    op: InteractionOp


class LowLevelQLearning:
    """
    低层Q-learning框架:
    - observe(): 观测算法对状态
    - select_action(): 选择交互算子
    - update(): Q更新
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random()
        self._q: Dict[Tuple[int, int, int, int], Dict[InteractionOp, float]] = {}

    def observe(self, mc_i: float, mc_j: float, overlap: float, corr: float) -> LowLevelState:
        return LowLevelState(mc_i=mc_i, mc_j=mc_j, overlap=overlap, corr=corr)

    def select_action(self, state: LowLevelState) -> LowLevelDecision:
        return self.select_action_with_bias(state, {})

    def select_action_with_bias(
        self, state: LowLevelState, op_bias: Dict[InteractionOp, float]
    ) -> LowLevelDecision:
        s = self._discretize(state)
        if self._rng.random() < self.epsilon:
            op = self._rng.choice(list(InteractionOp))
        else:
            q = self._q.get(s, {})
            op = max(InteractionOp, key=lambda a: q.get(a, 0.0) + op_bias.get(a, 0.0))
        return LowLevelDecision(op=op)

    def update(
        self,
        prev_state: LowLevelState,
        action: LowLevelDecision,
        reward: float,
        next_state: LowLevelState,
    ) -> None:
        s = self._discretize(prev_state)
        ns = self._discretize(next_state)
        q = self._q.setdefault(s, {})
        cur = q.get(action.op, 0.0)
        next_q = self._q.get(ns, {})
        max_next = max(next_q.values(), default=0.0)
        q[action.op] = cur + self.alpha * (reward + self.gamma * max_next - cur)

    def _discretize(self, s: LowLevelState) -> Tuple[int, int, int, int]:
        def bucket(x: float) -> int:
            if x < -0.1:
                return -1
            if x < 0.1:
                return 0
            return 1

        mc_i = bucket(s.mc_i)
        mc_j = bucket(s.mc_j)
        ov = 0 if s.overlap < 0.2 else 1 if s.overlap < 0.5 else 2
        corr = -1 if s.corr < -0.2 else 0 if s.corr < 0.2 else 1
        return (mc_i, mc_j, ov, corr)

