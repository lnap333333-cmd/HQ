"""快速测试：单实例单次运行，验证改进后算法是否正常执行。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hq_dhfsp.config import GlobalConfig
from hq_dhfsp.runner import RunnerConfig, run_single
from hq_dhfsp.instance import ProblemInstance

def main():
    print("=" * 50)
    print("快速测试：改进后算法验证")
    print("=" * 50)
    cfg = GlobalConfig(max_iters=5)
    inst = ProblemInstance.from_dict({
        "num_jobs": 20, "num_stages": 2, "num_factories": 2,
        "machines": [[2, 2], [2, 2]],
        "processing_time": [[10, 15], [12, 10], [8, 12], [11, 14], [9, 11]] * 4,
        "due_date": [100, 120, 90, 150, 110, 130, 95, 140, 105, 125] * 2,
    })
    runner_cfg = RunnerConfig(output_dir=str(Path("outputs") / "quick_test"))
    print("\n运行单算法各 5 步...")
    for algo in ["nsga", "moead", "hho", "ig"]:
        archive = run_single(inst, cfg, runner_cfg, algo)
        n = len(archive)
        print(f"  {algo.upper()}: 归档 {n} 个解")
    print("\n✓ 测试通过，改进后算法正常执行。")
    print("  完整对比请运行: python scripts/run_compare_batch.py")

if __name__ == "__main__":
    main()
