"""Minimal demo entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script: python path/to/demo.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.algorithms.nsga2 import NSGA2Config
from hq_dhfsp.algorithms.moead import MOEADConfig
from hq_dhfsp.instance import ProblemInstance
from hq_dhfsp.runner import RunnerConfig, run
from hq_dhfsp.decoder import decode
from hq_dhfsp.viz.gantt import plot_gantt
from hq_dhfsp.viz.pareto import plot_pareto


def main() -> None:
    # Small demo instance
    inst = ProblemInstance(
        num_factories=2,
        num_jobs=12,
        num_stages=3,
        machines=[[2, 2, 1], [1, 2, 2]],
        processing_time=[[3, 4, 5] for _ in range(12)],
        due_date=[20 for _ in range(12)],
        weight=[1 for _ in range(12)],
    )
    cfg = GlobalConfig(
        population_size=40,
        max_iters=20,
        nsga2=NSGA2Config(population_size=20),
        moead=MOEADConfig(num_subproblems=20),
    )
    runner_cfg = RunnerConfig(output_dir=str(Path("outputs") / "demo_run"))
    archive = run(inst, cfg, runner_cfg)
    # Pareto plot
    plot_pareto([e.objectives for e in archive.entries], Path(runner_cfg.output_dir) / "pareto.png")
    # Gantt plot for best (by sum of objectives)
    if archive.entries:
        best = min(archive.entries, key=lambda e: sum(e.objectives))
        decoded = decode(inst, best.encoding)
        plot_gantt(decoded.schedule, Path(runner_cfg.output_dir) / "gantt.png")


if __name__ == "__main__":
    main()

