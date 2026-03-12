"""Pareto comparison plot (f1,f2,f3) across algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

# Try common CJK fonts to avoid missing glyph warnings
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ..types import Objectives


def _load_objectives(path: Path) -> List[Objectives]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(obj) for obj in data.get("objectives", [])]


def plot_pareto_compare(
    run_dirs: Dict[str, Path],
    title: str,
    out_path: str | Path,
) -> None:
    styles = {
        "HQ": ("o", "tab:red"),
        "NSGA-II": ("s", "tab:blue"),
        "MOEA/D": ("^", "tab:green"),
        "HHO": ("v", "tab:orange"),
        "IG": ("<", "tab:purple"),
    }
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    for label, d in run_dirs.items():
        objs = _load_objectives(d / "archive.json")
        if not objs:
            continue
        marker, color = styles.get(label, ("o", None))
        x = [o[0] for o in objs]
        y = [o[1] for o in objs]
        z = [o[2] for o in objs]
        ax.scatter(x, y, z, s=30, marker=marker, label=label, alpha=0.8, c=color)

    ax.set_title(title)
    ax.set_xlabel("完工时间 (Makespan)")
    ax.set_ylabel("总拖期 (Total Tardiness)")
    ax.set_zlabel("负载不均衡 (Load Imbalance)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
