from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hq_dhfsp.algorithms.moead import MOEADConfig
from hq_dhfsp.algorithms.nsga2 import NSGA2Config
from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.decoder import decode
from hq_dhfsp.instance import ProblemInstance
from hq_dhfsp.runner import RunnerConfig, run
from hq_dhfsp.viz.gantt import plot_gantt
from hq_dhfsp.viz.pareto import plot_pareto
from hq_dhfsp.viz.progress import plot_progress


def build_instance(seed: int = 2) -> ProblemInstance:
    random.seed(seed)
    J, F, S = 70, 4, 4
    machines = [[3, 3, 2, 2], [2, 3, 3, 2], [3, 2, 3, 3], [2, 2, 3, 3]]
    processing_time = [[random.randint(1, 20) for _ in range(S)] for _ in range(J)]
    due_date = [random.randint(30, 200) for _ in range(J)]
    weight = [random.randint(1, 5) for _ in range(J)]
    return ProblemInstance(
        num_factories=F,
        num_jobs=J,
        num_stages=S,
        machines=machines,
        processing_time=processing_time,
        due_date=due_date,
        weight=weight,
    )


def main() -> None:
    inst = build_instance()
    cfg = GlobalConfig(
        population_size=100,
        max_iters=100,
        nsga2=NSGA2Config(population_size=100),
        moead=MOEADConfig(num_subproblems=100),
    )

    out_dir = Path("outputs") / "hq_full_run"
    archive = run(inst, cfg, RunnerConfig(output_dir=str(out_dir), log_every=20, verbose=True))

    plot_progress(out_dir / "progress.jsonl", out_dir)
    plot_pareto([e.objectives for e in archive.entries], out_dir / "pareto.png")
    if archive.entries:
        best = min(archive.entries, key=lambda e: sum(e.objectives))
        plot_gantt(decode(inst, best.encoding).schedule, out_dir / "gantt.png")

    print("\n结果输出路径：")
    print(f"- 目录：{out_dir}")
    print(f"- 进度：{out_dir / 'progress.jsonl'}")
    print(f"- 帕累托：{out_dir / 'pareto.png'}")
    print(f"- 甘特图：{out_dir / 'gantt.png'}")
    print(f"- HV/CV：{out_dir / 'hv_norm.png'} / {out_dir / 'cv.png'}")


if __name__ == "__main__":
    main()
