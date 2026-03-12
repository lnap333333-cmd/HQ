"""Gantt chart visualization."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from ..types import ScheduleEntry

# CJK font fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_gantt(schedule: List[ScheduleEntry], path: str | Path) -> None:
    if not schedule:
        return

    # Group by (factory, stage, machine)
    lanes = {}
    for e in schedule:
        key = (e.factory, e.stage, e.machine)
        lanes.setdefault(key, []).append(e)

    fig, ax = plt.subplots(figsize=(8, 4))
    yticks = []
    ylabels = []
    y = 0
    for key, entries in sorted(lanes.items()):
        entries = sorted(entries, key=lambda x: x.start)
        for e in entries:
            ax.barh(y, e.end - e.start, left=e.start, height=0.6, color="tab:blue", alpha=0.7)
        yticks.append(y)
        ylabels.append(f"F{key[0]}-S{key[1]}-M{key[2]}")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title("Schedule Gantt")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
