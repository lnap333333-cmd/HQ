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
from hq_dhfsp.runner import RunnerConfig, run, run_single
from hq_dhfsp.viz.gantt import plot_gantt
from hq_dhfsp.viz.pareto import plot_pareto
from hq_dhfsp.viz.pareto_compare import plot_pareto_compare
from hq_dhfsp.viz.progress import plot_progress
from hq_dhfsp.viz.compare import plot_hv_compare
from hq_dhfsp.metrics.hv import hv_approx
from hq_dhfsp.metrics.igd import igd
from hq_dhfsp.metrics.spacing import spacing
from hq_dhfsp.metrics.contribution import nondominated

import json
import time
from openpyxl import Workbook


def build_instance(seed: int = 2) -> ProblemInstance:
    random.seed(seed)
    J, F, S = 100, 4, 4
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
    print("规模详情：")
    print(f"- 作业数 J: {inst.num_jobs}")
    print(f"- 工厂数 F: {inst.num_factories}")
    print(f"- 工序数 S: {inst.num_stages}")
    print(f"- 机器配置: {inst.machines}")
    # Medium-scale comparison with 100 iters/100 population
    # HQ is special: total population 100 split across 4 sub-algorithms (25 each)
    cfg_hq = GlobalConfig(
        population_size=100,
        max_iters=100,
        nsga2=NSGA2Config(population_size=25),
        moead=MOEADConfig(num_subproblems=25),
    )
    cfg_single = GlobalConfig(
        population_size=100,
        max_iters=100,
        nsga2=NSGA2Config(population_size=100),
        moead=MOEADConfig(num_subproblems=100),
    )

    base_out = Path("outputs") / "compare_run"
    base_out.mkdir(parents=True, exist_ok=True)

    # HQ-learning hybrid
    out_hq = base_out / "hq"
    t0 = time.perf_counter()
    arch_hq = run(inst, cfg_hq, RunnerConfig(output_dir=str(out_hq), log_every=20, verbose=True))
    t1 = time.perf_counter()
    runtime_hq = t1 - t0
    plot_progress(out_hq / "progress.jsonl", out_hq)
    plot_pareto([e.objectives for e in arch_hq.entries], out_hq / "pareto.png")
    if arch_hq.entries:
        best = min(arch_hq.entries, key=lambda e: sum(e.objectives))
        plot_gantt(decode(inst, best.encoding).schedule, out_hq / "gantt.png")

    # Single algorithms
    runtimes = {"HQ": runtime_hq}
    for algo in ("nsga", "moead", "hho", "ig"):
        out_dir = base_out / algo
        t0 = time.perf_counter()
        arch = run_single(inst, cfg_single, RunnerConfig(output_dir=str(out_dir), log_every=20, verbose=True), algo)
        t1 = time.perf_counter()
        runtimes[algo.upper() if algo != "moead" else "MOEA/D"] = t1 - t0
        plot_progress(out_dir / "progress.jsonl", out_dir)
        plot_pareto([e.objectives for e in arch.entries], out_dir / "pareto.png")
        if arch.entries:
            best = min(arch.entries, key=lambda e: sum(e.objectives))
            plot_gantt(decode(inst, best.encoding).schedule, out_dir / "gantt.png")

    # HV comparison across runs
    plot_hv_compare(
        {
            "hq": base_out / "hq",
            "nsga": base_out / "nsga",
            "moead": base_out / "moead",
            "hho": base_out / "hho",
            "ig": base_out / "ig",
        },
        base_out / "hv_compare.png",
    )

    # Pareto comparison (f1,f2,f3)
    title = f"{inst.num_jobs}J{inst.num_stages}S{inst.num_factories}F - 帕累托前沿对比"
    plot_pareto_compare(
        {
            "HQ": base_out / "hq",
            "NSGA-II": base_out / "nsga",
            "MOEA/D": base_out / "moead",
            "HHO": base_out / "hho",
            "IG": base_out / "ig",
        },
        title,
        base_out / "pareto_compare.png",
    )

    # Collect metrics and export Excel
    run_dirs = {
        "HQ": out_hq,
        "NSGA-II": base_out / "nsga",
        "MOEA/D": base_out / "moead",
        "HHO": base_out / "hho",
        "IG": base_out / "ig",
    }
    metrics = compute_metrics(run_dirs, runtimes, base_out)
    excel_path = export_excel(metrics, base_out / "comparison.xlsx")
    export_text_table(metrics, base_out / "comparison.txt")

    print("\n结果输出路径：")
    print(f"- HQ目录：{out_hq}")
    print(f"- 子算法目录：{base_out / 'nsga'}, {base_out / 'moead'}, {base_out / 'hho'}, {base_out / 'ig'}")
    print(f"- HV对比：{base_out / 'hv_compare.png'}")
    print(f"- 帕累托对比：{base_out / 'pareto_compare.png'}")
    print(f"- 结果表：{base_out / 'comparison.txt'}")
    print(f"- Excel：{excel_path}")


def _load_archive(path: Path) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(x) for x in data.get("objectives", [])]


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_metrics(run_dirs: dict, runtimes: dict, base_out: Path) -> dict:
    # reference front
    all_objs = []
    for d in run_dirs.values():
        all_objs.extend(_load_archive(d / "archive.json"))
    ref = nondominated(all_objs)

    results = {}
    for name, d in run_dirs.items():
        objs = _load_archive(d / "archive.json")
        if not objs:
            continue
        f1 = [o[0] for o in objs]
        f2 = [o[1] for o in objs]
        f3 = [o[2] for o in objs]
        hv = hv_approx(objs, (1e4, 1e4, 1e4))
        results[name] = {
            "f1_best": min(f1),
            "f1_mean": sum(f1) / len(f1),
            "f2_best": min(f2),
            "f2_mean": sum(f2) / len(f2),
            "f3_best": min(f3),
            "f3_mean": sum(f3) / len(f3),
            "hv": hv,
            "igd": igd(ref, objs),
            "spacing": spacing(objs),
            "runtime": runtimes.get(name, 0.0),
        }
        if name == "HQ":
            summary = _load_summary(d / "summary.json")
            results[name]["cr"] = summary.get("contribution_rate", {})
            results[name]["mcr"] = summary.get("marginal_contribution_rate", {})
            results[name]["mc_raw"] = summary.get("marginal_contribution", {})
            results[name]["assist_mcr"] = summary.get("assist_marginal_contribution_rate", {})
            results[name]["assist_mc_raw"] = summary.get("assist_marginal_contribution", {})
            results[name]["total_mcr"] = summary.get("total_marginal_contribution_rate", {})
            results[name]["total_mc_raw"] = summary.get("total_marginal_contribution", {})
            results[name]["blended_mcr"] = summary.get("blended_marginal_contribution_rate", {})
            results[name]["blended_mc_raw"] = summary.get("blended_marginal_contribution", {})
            results[name]["mode_counts"] = summary.get("mode_counts", {})
            results[name]["op_counts"] = summary.get("operator_counts", {})
            results[name]["op_success"] = summary.get("operator_success", {})
    return normalize_metrics(results)


def normalize_metrics(metrics: dict) -> dict:
    if not metrics:
        return metrics
    eps_floor = 1e-3
    hv_vals = [m["hv"] for m in metrics.values()]
    igd_vals = [m["igd"] for m in metrics.values()]
    sp_vals = [m["spacing"] for m in metrics.values()]

    max_hv = max(hv_vals) or 1.0
    min_igd, max_igd = min(igd_vals), max(igd_vals)
    min_sp, max_sp = min(sp_vals), max(sp_vals)

    for m in metrics.values():
        # HV: higher is better, normalize by max
        hv_norm = (m["hv"] / max_hv) if max_hv > 0 else eps_floor
        # IGD/Spacing: lower is better, normalize by range and floor to avoid zero
        igd_norm = (
            (m["igd"] - min_igd) / (max_igd - min_igd + 1e-9)
            if max_igd > min_igd
            else 0.5
        )
        sp_norm = (
            (m["spacing"] - min_sp) / (max_sp - min_sp + 1e-9)
            if max_sp > min_sp
            else 0.5
        )
        m["hv"] = max(hv_norm, eps_floor)
        m["igd"] = max(igd_norm, eps_floor)
        m["spacing"] = max(sp_norm, eps_floor)
    return metrics


def export_text_table(metrics: dict, path: Path) -> None:
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
    lines = []
    lines.append("算法对比（最优/均值/运行时间）")
    lines.append(row_fmt.format(*header))
    for name, m in metrics.items():
        lines.append(
            row_fmt.format(
                name,
                f"{m['f1_best']:.2f}",
                f"{m['f1_mean']:.2f}",
                f"{m['f2_best']:.2f}",
                f"{m['f2_mean']:.2f}",
                f"{m['f3_best']:.2f}",
                f"{m['f3_mean']:.2f}",
                f"{m['runtime']:.2f}",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_excel(metrics: dict, path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "对比表"
    ws.append(
        [
            "算法",
            "完工时间-最优",
            "完工时间-均值",
            "总拖期-最优",
            "总拖期-均值",
            "负载不均衡-最优",
            "负载不均衡-均值",
            "运行时间(s)",
            "HV(归一化)",
            "IGD(归一化)",
            "Spacing(归一化)",
        ]
    )
    for name, m in metrics.items():
        ws.append(
            [
                name,
                m["f1_best"],
                m["f1_mean"],
                m["f2_best"],
                m["f2_mean"],
                m["f3_best"],
                m["f3_mean"],
                m["runtime"],
                m["hv"],
                m["igd"],
                m["spacing"],
            ]
        )
    # Summary sheet for HQ mechanisms
    if "HQ" in metrics:
        ws2 = wb.create_sheet("HQ机制")
        m = metrics["HQ"]
        ws2.append(["指标", "数值"])
        ws2.append(["贡献率(CR)", json.dumps(m.get("cr", {}), ensure_ascii=False)])
        ws2.append(["边际贡献率(MCR)", json.dumps(m.get("mcr", {}), ensure_ascii=False)])
        ws2.append(["边际贡献原值(MC)", json.dumps(m.get("mc_raw", {}), ensure_ascii=False)])
        ws2.append(["协同边际贡献率(Assist-MCR)", json.dumps(m.get("assist_mcr", {}), ensure_ascii=False)])
        ws2.append(["协同边际贡献原值(Assist-MC)", json.dumps(m.get("assist_mc_raw", {}), ensure_ascii=False)])
        ws2.append(["总边际贡献率(Total-MCR)", json.dumps(m.get("total_mcr", {}), ensure_ascii=False)])
        ws2.append(["总边际贡献原值(Total-MC)", json.dumps(m.get("total_mc_raw", {}), ensure_ascii=False)])
        ws2.append(["融合边际贡献率(Blended-MCR)", json.dumps(m.get("blended_mcr", {}), ensure_ascii=False)])
        ws2.append(["融合边际贡献原值(Blended-MC)", json.dumps(m.get("blended_mc_raw", {}), ensure_ascii=False)])
        ws2.append(["模式频率", json.dumps(m.get("mode_counts", {}), ensure_ascii=False)])
        ws2.append(["算子次数", json.dumps(m.get("op_counts", {}), ensure_ascii=False)])
        ws2.append(["算子成功", json.dumps(m.get("op_success", {}), ensure_ascii=False)])
    try:
        wb.save(path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        wb.save(fallback)
        return fallback


if __name__ == "__main__":
    main()
