"""Pareto scatter plot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt

from ..types import Objectives

# CJK font fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_pareto(points: Iterable[Objectives], path: str | Path) -> None:
    pts = list(points)
    if not pts:
        return
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    z = [p[2] for p in pts]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=10, c="tab:blue", alpha=0.7)
    ax.set_xlabel("f1 (Makespan)")
    ax.set_ylabel("f2 (Weighted Tardiness)")
    ax.set_zlabel("f3 (Load Imbalance)")
    ax.set_title("Pareto Front (Approx)")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
