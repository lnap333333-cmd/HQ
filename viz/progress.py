"""Progress visualization (normalized HV)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

# CJK font fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_progress(path: str | Path) -> Tuple[List[int], List[float], List[float]]:
    steps: List[int] = []
    hv: List[float] = []
    cv: List[float] = []
    path = Path(path)
    if not path.exists():
        return steps, hv, cv
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        steps.append(int(rec.get("step", 0)))
        hv.append(float(rec.get("hv", 0.0)))
        cv.append(float(rec.get("cv", 0.0)))
    return steps, hv, cv


def plot_progress(progress_path: str | Path, out_dir: str | Path) -> None:
    steps, hv, cv = load_progress(progress_path)
    if not steps:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_hv = max(hv) if max(hv) > 0 else 1.0
    hv_norm = [v / max_hv for v in hv]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, hv_norm, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized HV")
    ax.set_title("HV (Normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / "hv_norm.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, cv, marker="o", markersize=3, color="tab:orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CV")
    ax.set_title("CV")
    fig.tight_layout()
    fig.savefig(out_dir / "cv.png", dpi=150)
    plt.close(fig)
