"""Comparison plots across algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# CJK font fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_hv_series(path: str | Path) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    hv: List[float] = []
    path = Path(path)
    if not path.exists():
        return steps, hv
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        steps.append(int(rec.get("step", 0)))
        hv.append(float(rec.get("hv", 0.0)))
    return steps, hv


def plot_hv_compare(run_dirs: Dict[str, Path], out_path: str | Path) -> None:
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    max_hv = 0.0
    for name, d in run_dirs.items():
        steps, hv = load_hv_series(d / "progress.jsonl")
        series[name] = (steps, hv)
        if hv:
            max_hv = max(max_hv, max(hv))
    if max_hv <= 0:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (steps, hv) in series.items():
        if not steps:
            continue
        hv_norm = [v / max_hv for v in hv]
        ax.plot(steps, hv_norm, marker="o", markersize=3, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized HV")
    ax.set_title("HV Comparison (Normalized)")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
