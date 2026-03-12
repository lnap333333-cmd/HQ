from __future__ import annotations

import random
import sys
from pathlib import Path

# allow running as a script without installing package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hq_dhfsp.algorithms.moead import MOEADConfig
from hq_dhfsp.algorithms.nsga2 import NSGA2Config
from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.instance import ProblemInstance
from hq_dhfsp.runner import RunnerConfig, run
from hq_dhfsp.viz.progress import plot_progress


def main() -> None:
    random.seed(2)

    # Medium-scale instance
    J, F, S = 40, 3, 4
    machines = [[3, 3, 2, 2], [2, 3, 3, 2], [3, 2, 3, 3]]
    processing_time = [[random.randint(1, 20) for _ in range(S)] for _ in range(J)]
    due_date = [random.randint(30, 200) for _ in range(J)]
    weight = [random.randint(1, 5) for _ in range(J)]

    inst = ProblemInstance(
        num_factories=F,
        num_jobs=J,
        num_stages=S,
        machines=machines,
        processing_time=processing_time,
        due_date=due_date,
        weight=weight,
    )

    cfg = GlobalConfig(
        population_size=60,
        max_iters=60,
        nsga2=NSGA2Config(population_size=30),
        moead=MOEADConfig(num_subproblems=30),
    )
    out_dir = Path("outputs") / "medium_run"
    run(inst, cfg, RunnerConfig(output_dir=str(out_dir), log_every=20, verbose=True))

    # Normalized HV/CV plots
    plot_progress(out_dir / "progress.jsonl", out_dir)

    print("\n结果输出路径：")
    print(f"- 目录：{out_dir}")
    print(f"- 进度：{out_dir / 'progress.jsonl'}")
    print(f"- HV/CV：{out_dir / 'hv_norm.png'} / {out_dir / 'cv.png'}")


if __name__ == "__main__":
    main()
