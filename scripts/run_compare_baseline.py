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
from hq_dhfsp.runner import RunnerConfig, run_single
from hq_dhfsp.viz.compare import plot_hv_compare
from hq_dhfsp.viz.gantt import plot_gantt
from hq_dhfsp.viz.pareto import plot_pareto
from hq_dhfsp.viz.pareto_compare import plot_pareto_compare
from hq_dhfsp.viz.progress import plot_progress


def build_instance(seed: int = 2) -> ProblemInstance:
    random.seed(seed)
    J, F, S = 40, 3, 4
    machines = [[3, 3, 2, 2], [2, 3, 3, 2], [3, 2, 3, 3]]
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
        population_size=60,
        max_iters=60,
        nsga2=NSGA2Config(population_size=30),
        moead=MOEADConfig(num_subproblems=30),
    )

    base_out = Path("outputs") / "compare_baseline"
    base_out.mkdir(parents=True, exist_ok=True)

    # Single algorithms only
    for algo in ("nsga", "moead", "hho", "ig"):
        out_dir = base_out / algo
        arch = run_single(inst, cfg, RunnerConfig(output_dir=str(out_dir), log_every=20, verbose=True), algo)
        plot_progress(out_dir / "progress.jsonl", out_dir)
        plot_pareto([e.objectives for e in arch.entries], out_dir / "pareto.png")
        if arch.entries:
            best = min(arch.entries, key=lambda e: sum(e.objectives))
            plot_gantt(decode(inst, best.encoding).schedule, out_dir / "gantt.png")

    # HV comparison across baselines
    plot_hv_compare(
        {
            "nsga": base_out / "nsga",
            "moead": base_out / "moead",
            "hho": base_out / "hho",
            "ig": base_out / "ig",
        },
        base_out / "hv_compare.png",
    )

    # Pareto comparison (f1 vs f2)
    title = f"{inst.num_jobs}J{inst.num_stages}S{inst.num_factories}F - 帕累托前沿对比"
    plot_pareto_compare(
        {
            "NSGA-II": base_out / "nsga",
            "MOEA/D": base_out / "moead",
            "HHO": base_out / "hho",
            "IG": base_out / "ig",
        },
        title,
        base_out / "pareto_compare.png",
    )

    print("\n结果输出路径：")
    print(f"- 目录：{base_out}")
    print(f"- 子算法目录：{base_out / 'nsga'}, {base_out / 'moead'}, {base_out / 'hho'}, {base_out / 'ig'}")
    print(f"- HV对比：{base_out / 'hv_compare.png'}")
    print(f"- 帕累托对比：{base_out / 'pareto_compare.png'}")


if __name__ == "__main__":
    main()
