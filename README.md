## HQ-learning Cooperative–Competitive Multi-Algorithm DHFSP (Skeleton)

本仓库提供“HQ-learning 驱动的多算法协同–竞争进化框架”在 **DHFSP 三目标**场景下的**工程骨架**，用于快速落地实现与论文复现实验。

### 目录结构

- `src/hq_dhfsp/`: 主包
  - `instance.py`: 实例数据结构与 JSON IO
  - `solution.py`: 三段式编码与解码（生成调度表）
  - `objectives.py`: \(f1,f2,f3\) 计算
  - `archive.py`: 全局精英档案 EA（非支配筛选等）
  - `algorithms/`: NSGA-II / MOEA-D / HHO / IG 算法接口与骨架
  - `rl/`: 高层/低层 Q-learning 骨架
  - `interactions/`: C1/C2/R1/R2 关系动作骨架
  - `metrics/`: HV/CV/IGD/Spread/CR 等指标骨架
  - `runner.py`: 总体执行流程（主循环）
  - `logging_utils.py`: JSONL 日志
  - `viz/`: 甘特图/帕累托图（占位）
- `examples/`: 示例实例与输出（占位）

### 环境

- Python 3.10+

安装依赖（骨架阶段尽量轻量）：

```bash
python -m pip install -r requirements.txt
```

### 快速运行（骨架自检）

```bash
python -m hq_dhfsp.demo
```

该命令会：
- 构造一个小型随机实例
- 跑一小段 `runner` 主循环（使用算法/交互的占位实现）
- 输出 `outputs/demo_run/` 下的日志与若干结果文件

### 说明

当前是**文件骨架**：核心接口、数据结构和主流程都已搭好，但具体算子（NSGA-II、MOEA/D、HHO、IG、C1/C2/R1/R2）的实现细节仍需逐步填充。

