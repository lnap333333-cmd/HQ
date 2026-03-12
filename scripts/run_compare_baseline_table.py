from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hq_dhfsp.algorithms.moead import MOEADConfig
from hq_dhfsp.algorithms.nsga2 import NSGA2Config
from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.instance import ProblemInstance
from hq_dhfsp.runner import RunnerConfig, run_single
from hq_dhfsp.viz.pareto_compare import plot_pareto_compare


@dataclass
class AlgoStats:
    makespan_best: float
    makespan_mean: float
    tardiness_best: float
    tardiness_mean: float
    imbalance_best: float
    imbalance_mean: float
    runtime_mean: float
    pareto_points: List[Tuple[float, float]]


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


def summarize_runs(
    algo: str,
    inst: ProblemInstance,
    cfg: GlobalConfig,
    runs: int,
    out_dir: Path,
) -> AlgoStats:
    makespan_vals: List[float] = []
    tardiness_vals: List[float] = []
    imbalance_vals: List[float] = []
    runtimes: List[float] = []
    pareto_points: List[Tuple[float, float]] = []

    for i in range(runs):
        t0 = time.perf_counter()
        arch = run_single(
            inst,
            cfg,
            RunnerConfig(output_dir=str(out_dir / f"run_{i+1}"), log_every=20, verbose=True),
            algo,
        )
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)

        if arch.entries:
            f1 = [e.objectives[0] for e in arch.entries]
            f2 = [e.objectives[1] for e in arch.entries]
            f3 = [e.objectives[2] for e in arch.entries]
            makespan_vals.append(min(f1))
            tardiness_vals.append(min(f2))
            imbalance_vals.append(min(f3))
            if i == 0:
                pareto_points = list(zip(f1, f2))
        else:
            makespan_vals.append(float("inf"))
            tardiness_vals.append(float("inf"))
            imbalance_vals.append(float("inf"))

    def mean(vals: List[float]) -> float:
        valid = [v for v in vals if v != float("inf")]
        return sum(valid) / len(valid) if valid else float("inf")

    return AlgoStats(
        makespan_best=min(makespan_vals),
        makespan_mean=mean(makespan_vals),
        tardiness_best=min(tardiness_vals),
        tardiness_mean=mean(tardiness_vals),
        imbalance_best=min(imbalance_vals),
        imbalance_mean=mean(imbalance_vals),
        runtime_mean=sum(runtimes) / len(runtimes) if runtimes else 0.0,
        pareto_points=pareto_points,
    )


def print_table(stats: Dict[str, AlgoStats]) -> None:
    header = [
        "算法",
        "完工时间-最优",
        "完工时间-均值",
        "总拖期-最优",
        "总拖期-均值",
        "负载不均衡-最优",
        "负载不均衡-均值",
        "运行时间(s)",
    ]
    row_fmt = "{:<10} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}"
    print("\n算法对比（最优/均值/运行时间）")
    print(row_fmt.format(*header))
    for name, st in stats.items():
        print(
            row_fmt.format(
                name,
                f"{st.makespan_best:.2f}",
                f"{st.makespan_mean:.2f}",
                f"{st.tardiness_best:.2f}",
                f"{st.tardiness_mean:.2f}",
                f"{st.imbalance_best:.2f}",
                f"{st.imbalance_mean:.2f}",
                f"{st.runtime_mean:.2f}",
            )
        )


def main() -> None:
    inst = build_instance()
    cfg = GlobalConfig(
        population_size=100,
        max_iters=100,
        nsga2=NSGA2Config(population_size=100),
        moead=MOEADConfig(num_subproblems=100),
    )

    out_dir = Path("outputs") / "compare_baseline_table"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline algorithms in our framework
    algos = [("NSGA-II", "nsga"), ("MOEA/D", "moead"), ("HHO", "hho"), ("IG", "ig")]
    stats: Dict[str, AlgoStats] = {}

    for label, algo in algos:
        stats[label] = summarize_runs(algo, inst, cfg, runs=3, out_dir=out_dir / algo)

    print_table(stats)

    # Pareto comparison plot (f1 vs f2) using first-run points
    title = f"{inst.num_jobs}J{inst.num_stages}S{inst.num_factories}F - 帕累托前沿对比"
    plot_pareto_compare(
        {
            "NSGA-II": out_dir / "nsga" / "run_1",
            "MOEA/D": out_dir / "moead" / "run_1",
            "HHO": out_dir / "hho" / "run_1",
            "IG": out_dir / "ig" / "run_1",
        },
        title,
        out_dir / "pareto_compare.png",
    )

    # Save table to file
    report = out_dir / "baseline_table.txt"
    with report.open("w", encoding="utf-8") as f:
        header = [
            "算法",
            "完工时间-最优",
            "完工时间-均值",
            "总拖期-最优",
            "总拖期-均值",
            "负载不均衡-最优",
            "负载不均衡-均值",
            "运行时间(s)",
        ]
        f.write("算法对比（最优/均值/运行时间）\n")
        f.write("{:<10} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}\n".format(*header))
        for name, st in stats.items():
            f.write(
                "{:<10} {:>12} {:>12} {:>12} {:>12} {:>14} {:>14} {:>12}\n".format(
                    name,
                    f"{st.makespan_best:.2f}",
                    f"{st.makespan_mean:.2f}",
                    f"{st.tardiness_best:.2f}",
                    f"{st.tardiness_mean:.2f}",
                    f"{st.imbalance_best:.2f}",
                    f"{st.imbalance_mean:.2f}",
                    f"{st.runtime_mean:.2f}",
                )
            )

    # Print output paths
    print("\n结果输出路径：")
    print(f"- 表格：{report}")
    print(f"- 帕累托对比图：{out_dir / 'pareto_compare.png'}")
    print(f"- 各算法目录：{out_dir / 'nsga'}, {out_dir / 'moead'}, {out_dir / 'hho'}, {out_dir / 'ig'}")


if __name__ == "__main__":
    main()
