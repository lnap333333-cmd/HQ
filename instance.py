"""Problem instance definition and JSON IO (问题实例与IO)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ProblemInstance:
    """
    DHFSP 实例 (最小可用):
    - F: 工厂数量
    - J: 作业数量
    - S: 工序数量
    - machines[f][s]: 工厂f在工序s的并行机数量
    - processing_time[j][s]: 作业j在工序s的加工时间
    - due_date[j], weight[j]: 拖期与权重
    """

    num_factories: int
    num_jobs: int
    num_stages: int
    machines: List[List[int]]
    processing_time: List[List[float]]
    due_date: List[float]
    weight: List[float]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProblemInstance":
        return ProblemInstance(
            num_factories=int(d["num_factories"]),
            num_jobs=int(d["num_jobs"]),
            num_stages=int(d["num_stages"]),
            machines=[list(map(int, row)) for row in d["machines"]],
            processing_time=[list(map(float, row)) for row in d["processing_time"]],
            due_date=list(map(float, d["due_date"])),
            weight=list(map(float, d["weight"])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_factories": self.num_factories,
            "num_jobs": self.num_jobs,
            "num_stages": self.num_stages,
            "machines": self.machines,
            "processing_time": self.processing_time,
            "due_date": self.due_date,
            "weight": self.weight,
        }

    @staticmethod
    def load_json(path: str | Path) -> "ProblemInstance":
        p = Path(path)
        return ProblemInstance.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def validate(self) -> None:
        F, J, S = self.num_factories, self.num_jobs, self.num_stages
        if len(self.machines) != F or any(len(row) != S for row in self.machines):
            raise ValueError("machines must be shape [F][S]")
        if len(self.processing_time) != J or any(len(row) != S for row in self.processing_time):
            raise ValueError("processing_time must be shape [J][S]")
        if len(self.due_date) != J or len(self.weight) != J:
            raise ValueError("due_date/weight must have J values")
        if any(m <= 0 for row in self.machines for m in row):
            raise ValueError("machines must be positive")
        if any(p <= 0 for row in self.processing_time for p in row):
            raise ValueError("processing_time must be positive")

    def evaluate(self, encoding: "Encoding", return_decoded: bool = False):
        """
        统一评估接口（最小化）:
        - 返回 (f1, f2, f3)
        - 可选返回解码后的调度表/完工时间
        """
        from .decoder import decode
        from .encoding import Encoding
        from .objectives import compute_objectives

        if not isinstance(encoding, Encoding):
            raise TypeError("encoding must be Encoding")

        decoded = decode(self, encoding)
        objectives, breakdown = compute_objectives(self, decoded)
        if return_decoded:
            return objectives, decoded, breakdown
        return objectives

