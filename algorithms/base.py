"""Algorithm interface base class (算法接口基类)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from ..encoding import Encoding
from ..instance import ProblemInstance
from ..types import Objectives


@dataclass(frozen=True)
class Candidate:
    encoding: Encoding
    objectives: Objectives


class Algorithm(Protocol):
    """
    最小算法接口（供 NSGA-II / MOEA-D / HHO / IG 统一调用）:
    - initialize(): 生成初始种群
    - step(): 执行一代演化（或一次迭代）
    统一评价接口由 ProblemInstance.evaluate() 负责（目标均为最小化）
    """

    name: str

    def initialize(self, instance: ProblemInstance, population_size: int) -> List[Encoding]:
        ...

    def step(self, instance: ProblemInstance, population: List[Encoding]) -> List[Encoding]:
        ...

