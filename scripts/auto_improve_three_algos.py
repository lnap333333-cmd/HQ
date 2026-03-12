from __future__ import annotations

import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hq_dhfsp.algorithms.hho import HHOConfig
from hq_dhfsp.algorithms.moead import MOEADConfig
from hq_dhfsp.algorithms.nsga2 import NSGA2Config
from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.instance import ProblemInstance
from hq_dhfsp.runner import RunnerConfig, run, run_single


def build_instance(spec: Dict[str, int], seed: int) -> ProblemInstance:
    rng = random.Random(seed)
    j, f, s = spec["J"], spec["F"], spec["S"]
    machines = [[rng.randint(2, 4) for _ in range(s)] for _ in range(f)]
    processing_time = [[rng.randint(1, 20) for _ in range(s)] for _ in range(j)]
    due_date = [rng.randint(max(10, int(7.5 * s)), int(50 * s)) for _ in range(j)]
    weight = [rng.randint(1, 5) for _ in range(j)]
    return ProblemInstance(
        num_factories=f,
        num_jobs=j,
        num_stages=s,
        machines=machines,
        processing_time=processing_time,
        due_date=due_date,
        weight=weight,
    )


def _load_archive(path: Path) -> List[Tuple[float, float, float]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(x) for x in data.get("objectives", [])]


def _best(objs: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not objs:
        return (1e18, 1e18, 1e18)
    return (
        min(o[0] for o in objs),
        min(o[1] for o in objs),
        min(o[2] for o in objs),
    )


def _hq_ig_cr(path: Path) -> float:
    if not path.exists():
        return 1.0
    data = json.loads(path.read_text(encoding="utf-8"))
    return float((data.get("contribution_rate", {}) or {}).get("ig", 1.0))


def evaluate_candidate(
    base_out: Path,
    name: str,
    cfg_hq: GlobalConfig,
    cfg_single: GlobalConfig,
    specs: List[Dict[str, int]],
) -> Dict[str, float]:
    out = base_out / name
    out.mkdir(parents=True, exist_ok=True)

    hq_wins = 0
    cases = 0
    cr_ig_vals: List[float] = []
    margin_f2: List[float] = []
    margin_f3: List[float] = []

    for i, spec in enumerate(specs, start=1):
        case_dir = out / f"case_{i:02d}_{spec['J']}J{spec['S']}S{spec['F']}F"
        case_dir.mkdir(parents=True, exist_ok=True)
        inst = build_instance(spec, seed=500 + i)

        hq_dir = case_dir / "hq"
        ig_dir = case_dir / "ig"
        run(inst, cfg_hq, RunnerConfig(output_dir=str(hq_dir), log_every=20, verbose=False))
        run_single(inst, cfg_single, RunnerConfig(output_dir=str(ig_dir), log_every=20, verbose=False), "ig")

        hq_objs = _load_archive(hq_dir / "archive.json")
        ig_objs = _load_archive(ig_dir / "archive.json")
        hq_best = _best(hq_objs)
        ig_best = _best(ig_objs)

        # lower is better on all three
        hq_better_count = int(hq_best[0] <= ig_best[0]) + int(hq_best[1] <= ig_best[1]) + int(
            hq_best[2] <= ig_best[2]
        )
        if hq_better_count >= 2:
            hq_wins += 1
        cases += 1

        margin_f2.append(ig_best[1] - hq_best[1])  # >0 means HQ better
        margin_f3.append(ig_best[2] - hq_best[2])  # >0 means HQ better
        cr_ig_vals.append(_hq_ig_cr(hq_dir / "summary.json"))

    # score: prefer HQ beating IG in f2/f3 and lower IG CR domination
    score = (
        2.0 * (hq_wins / max(1, cases))
        + 0.002 * statistics.mean(margin_f2)
        + 0.2 * statistics.mean(margin_f3)
        - 1.2 * statistics.mean(cr_ig_vals)
    )
    return {
        "score": score,
        "hq_win_rate_vs_ig": hq_wins / max(1, cases),
        "mean_f2_margin_ig_minus_hq": statistics.mean(margin_f2),
        "mean_f3_margin_ig_minus_hq": statistics.mean(margin_f3),
        "mean_cr_ig": statistics.mean(cr_ig_vals),
    }


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = Path("outputs") / f"auto_improve_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    # representative scales (small / medium / large-ish)
    specs = [
        {"J": 30, "F": 2, "S": 3},
        {"J": 80, "F": 3, "S": 4},
        {"J": 120, "F": 4, "S": 5},
    ]

    # keep IG unchanged; only improve three algorithms.
    base_hq = GlobalConfig(population_size=100, max_iters=50, nsga2=NSGA2Config(population_size=25), moead=MOEADConfig(num_subproblems=25))
    base_single = GlobalConfig(population_size=100, max_iters=50, nsga2=NSGA2Config(population_size=100), moead=MOEADConfig(num_subproblems=100))

    candidates = [
        {
            "name": "cand_a",
            "nsga2": NSGA2Config(population_size=25, tardiness_refine_prob=0.25, load_refine_prob=0.20, refine_trials=4, seed_ratio=0.25),
            "moead": MOEADConfig(num_subproblems=25, delta=0.85, nr=3, neighborhood_min_ratio=0.45, weight_reset_period=20, weight_reset_ratio=0.12, tardiness_refine_prob=0.25),
            "hho": HHOConfig(refine_prob=0.25, elite_keep_ratio=0.12, gaussian_prob=0.22, opposition_ratio=0.12, tardiness_leader_prob=0.25),
        },
        {
            "name": "cand_b",
            "nsga2": NSGA2Config(population_size=25, tardiness_refine_prob=0.20, load_refine_prob=0.25, refine_trials=5, seed_ratio=0.2),
            "moead": MOEADConfig(num_subproblems=25, delta=0.80, nr=4, neighborhood_min_ratio=0.40, weight_reset_period=18, weight_reset_ratio=0.15, tardiness_refine_prob=0.20),
            "hho": HHOConfig(refine_prob=0.30, elite_keep_ratio=0.15, gaussian_prob=0.20, opposition_ratio=0.15, tardiness_leader_prob=0.20),
        },
        {
            "name": "cand_c",
            "nsga2": NSGA2Config(population_size=25, tardiness_refine_prob=0.30, load_refine_prob=0.18, refine_trials=3, seed_ratio=0.3),
            "moead": MOEADConfig(num_subproblems=25, delta=0.90, nr=3, neighborhood_min_ratio=0.50, weight_reset_period=25, weight_reset_ratio=0.10, tardiness_refine_prob=0.30),
            "hho": HHOConfig(refine_prob=0.22, elite_keep_ratio=0.10, gaussian_prob=0.18, opposition_ratio=0.10, tardiness_leader_prob=0.30),
        },
    ]

    results = []
    best = None
    best_cfg = None
    for c in candidates:
        cfg_hq = GlobalConfig(
            population_size=base_hq.population_size,
            max_iters=base_hq.max_iters,
            nsga2=c["nsga2"],
            moead=c["moead"],
            hho=c["hho"],
            ig=base_hq.ig,
            high_level=base_hq.high_level,
            low_level=base_hq.low_level,
        )
        cfg_single = GlobalConfig(
            population_size=base_single.population_size,
            max_iters=base_single.max_iters,
            nsga2=NSGA2Config(population_size=100, tardiness_refine_prob=c["nsga2"].tardiness_refine_prob, load_refine_prob=c["nsga2"].load_refine_prob, refine_trials=c["nsga2"].refine_trials, seed_ratio=c["nsga2"].seed_ratio),
            moead=MOEADConfig(num_subproblems=100, delta=c["moead"].delta, nr=c["moead"].nr, neighborhood_min_ratio=c["moead"].neighborhood_min_ratio, weight_reset_period=c["moead"].weight_reset_period, weight_reset_ratio=c["moead"].weight_reset_ratio, tardiness_refine_prob=c["moead"].tardiness_refine_prob),
            hho=c["hho"],
            ig=base_single.ig,
            high_level=base_single.high_level,
            low_level=base_single.low_level,
        )
        metric = evaluate_candidate(root, c["name"], cfg_hq, cfg_single, specs)
        item = {"name": c["name"], **metric}
        results.append(item)
        if best is None or item["score"] > best["score"]:
            best = item
            best_cfg = c
        print(f"[{c['name']}] score={item['score']:.4f}, HQ_win_rate={item['hq_win_rate_vs_ig']:.3f}, CR_IG={item['mean_cr_ig']:.3f}")

    (root / "tuning_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if best is not None and best_cfg is not None:
        (root / "best_candidate.json").write_text(
            json.dumps(
                {
                    "best": best,
                    "config": {
                        "nsga2": asdict(best_cfg["nsga2"]),
                        "moead": asdict(best_cfg["moead"]),
                        "hho": asdict(best_cfg["hho"]),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\nBest candidate:", best["name"])
        print("Output:", root)


if __name__ == "__main__":
    main()

