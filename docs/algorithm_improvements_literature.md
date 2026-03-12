# HQ-Learning 最新算法说明文档

本文档描述当前代码中已经实现并正在使用的 HQ 算法版本，目标是与 `src/hq_dhfsp` 下的最新实现完全一致，用于说明算法结构、决策逻辑、子算法职责以及贡献指标体系。

---

## 1. 总体定位

当前 HQ 算法是一个三层协同框架，用于求解三目标 DHFSP：

- `f1`: makespan
- `f2`: total tardiness
- `f3`: load imbalance

整体思想不是简单地把多个元启发式并列运行，而是建立一个分层控制体系：

1. 高层智能体负责**关系模式调度**
2. 低层智能体负责**交互算子选择**
3. 底层子算法负责**独立进化、协同进化、竞争进化**

对应主题可概括为：

- 高层 agent：基于演化状态进行模式调度
- 低层 agent：针对算法对选择交互算子
- 底层子算法：在独立、合作、竞争三种关系下共同演化

---

## 2. 框架组成

当前 HQ 由 4 个底层子算法组成：

- `NSGA-II`
- `MOEA/D`
- `HHO`
- `IG`

其中 HQ 在每一代的执行顺序为：

1. 四个子算法各自演化一代
2. 将四个群体统一并入精英档案
3. 高层 Q-learning 选择本代关系模式
4. 低层 Q-learning 为每一对算法选择交互算子
5. 记录模式、算子、贡献、档案与学习信号

---

## 3. 高层：关系模式调度

高层采用 `HighLevelQLearning`，动作空间为三种关系模式：

- `independent`
- `cooperation`
- `competition`

### 3.1 高层状态

高层状态由演化改进情况构成，核心量包括：

- `delta_hv`: 最近窗口或相邻代的超体积增量
- `delta_cv`: 最近窗口或相邻代的覆盖/分散变化
- `stagnation`: 停滞代数

### 3.2 高层调度逻辑

高层不是硬编码切换模式，而是“Q 值 + 软偏置”联合决策：

- 早期阶段更偏向 `cooperation`
- 中后期逐步提升 `independent` 与定向 `competition`
- 当停滞明显时，提高 `cooperation` 或 `competition` 的偏置
- 当 `HHO` 份额过弱时，额外给 `cooperation` 轻微正偏置

此外还存在一个轻量的阶段调度器 `_scheduled_mode()`，用于在不同演化阶段对高层动作做柔性修正，而不是直接覆盖学习过程。

### 3.3 高层当前参数

- `relation_step = 5`
- `explore_phase_ratio = 0.35`
- `exploit_phase_ratio = 0.8`
- `stagnation_force_coop = 6`
- `stagnation_force_competition = 10`

---

## 4. 低层：交互算子选择

低层采用 `LowLevelQLearning`，针对每一对算法单独决策交互算子。

### 4.1 低层状态

低层状态由算法对之间的关系特征构成：

- `MC_i`, `MC_j`: 两个算法当前边际贡献强度
- `overlap`: 两者在档案中的覆盖重叠度
- `correlation`: 两者贡献变化序列的相关性

### 4.2 动作空间

低层动作空间为四个交互算子：

- `C1`: elite migration
- `C2`: rhythm cooperation
- `R1`: structural suppression
- `R2`: territorial invasion

其中：

- 在 `cooperation` 模式下，仅使用 `C1 / C2`
- 在 `competition` 模式下，仅使用 `R1 / R2`
- 在 `independent` 模式下，不触发交互

### 4.3 低层当前偏置逻辑

低层不是单纯依靠原始 `MC` 排序，而是通过 `_adjusted_mc_scores()` 做公平化修正，综合考虑：

- 当前档案占比与目标份额偏差
- `assist_credit` 协同信用
- 早期对 `IG` 的抑制
- 对弱势 `HHO` 的优先扶持

当配对中包含 `HHO` 且其份额偏弱时：

- 在合作模式下，偏向 `C2`
- 在竞争模式下，偏向 `R2`
- 若 `HHO` 是竞争中的 loser，则降低交互强度
- 若 `HHO` 偏弱，则适当降低其被吸收/被打压的速率

---

## 5. 底层交互算子含义

### 5.1 Cooperation

- `C1 (elite migration)`
  - 由赢家向输家注入精英结构
  - 强调定向知识转移

- `C2 (rhythm cooperation)`
  - 双向节律协同
  - 更强调互补结构交换与协同探索

### 5.2 Competition

- `R1 (structural suppression)`
  - 赢家压制输家结构空间
  - 偏向限制性竞争

- `R2 (territorial invasion)`
  - 赢家用自身结构入侵输家群体
  - 偏向更强的替换/占领

---

## 6. 贡献度评价体系

当前 HQ 不是只看最终档案占位，而是同时维护 5 类贡献指标。

### 6.1 CR

`CR (Contribution Rate)` 表示最终档案中各算法来源占比。

特点：

- 直观
- 适合看“最后是谁留下来了”
- 容易低估“铺路型算法”

### 6.2 MCR

`MCR (Marginal Contribution Rate)` 基于移除某算法后造成的整体质量下降计算。

当前原值由两部分构成：

- `q`: 移除后 `HV` 的损失比例
- `s`: 移除后 `CV` 的恶化量

综合公式为：

- `raw = mc_weight * q + (1 - mc_weight) * s`

特点：

- 比单纯 CR 更公平
- 能反映直接边际贡献
- 仍然偏向“最终显式留在档案中的算法”

### 6.3 Assist-MCR

为解决“协作铺路但最终被别人收割”的问题，当前版本新增了**延迟协同信用**。

机制如下：

1. 每次交互后不立即记功
2. 将交互事件放入 `pending_interactions`
3. 在 `assist_delay_steps = 3` 代后结算
4. 若之后 `HV` 提升、`CV` 改善，则把收益分给触发交互的算法对

当前参数：

- `assist_reward_scale = 0.15`
- `assist_winner_share = 0.65`
- `assist_loser_share = 0.35`

若是 `C2`，则双方均分信用；若是 `C1/R1/R2`，则按 winner/loser 比例分配。

### 6.4 Total-MCR

`Total-MCR = 直接边际贡献 + 协同边际贡献`

它反映“某算法既直接留下了什么，又通过交互间接促成了什么”。

### 6.5 Blended-MCR

当前主分析指标为：

- `Blended-MCR = (1 - λ) * MCR + λ * Assist-MCR`

当前默认：

- `assist_blend_lambda = 0.45`

采用这个指标的原因是：

- 比 `MCR` 更能体现 `HHO` 这类协作型算法的价值
- 比 `Total-MCR` 更克制，不会把所有算法贡献拉得过平
- 更贴合 HQ 框架“合作-竞争-独立共演化”的主题

---

## 7. 四个子算法的当前职责

### 7.1 NSGA-II

当前 `NSGA-II` 在 HQ 中主要负责：

- 稳定前沿收敛
- 边界解与膝点保持
- 提供较强的 `f1/f2` 竞争能力

当前实现特征：

- `seed_population + pool 多样性选择` 初始化
- 二元锦标赛选择
- 交叉与变异
- `makespan / tardiness / load` 三类精炼
- 对前若干前沿进行强化精炼
- `anchor_keep` 保留三个目标极值解与膝点解

关键参数：

- `population_size = 25`
- `tardiness_refine_prob = 0.50`
- `load_refine_prob = 0.25`
- `makespan_refine_prob = 0.25`
- `refine_trials = 10`
- `refine_fronts = 2`

### 7.2 MOEA/D

当前 `MOEA/D` 在 HQ 中主要负责：

- 从分解视角提供结构化多目标推进
- 补充前沿不同方向上的收敛压力
- 通过权重与邻域机制维持方向性搜索

当前实现特征：

- 三目标 `Tchebycheff` 分解
- 极端权重 + 均匀权重联合构造
- 动态邻域大小
- `makespan / tardiness / load` 分解方向精炼
- 停滞窗口检测
- 周期性权重重置
- 全局替换 fallback，避免邻域停滞
- 对小规模子问题数做了安全权重采样修复

关键参数：

- `num_subproblems = 25`
- `neighborhood_size = 10`
- `delta = 0.9`
- `nr = 2`
- `weight_reset_period = 25`
- `stagnation_window = 15`

### 7.3 HHO

当前 `HHO` 是 HQ 中重点突出创新性的子算法，其职责不是简单复制 IG，而是强调：

- 多目标 leader 驱动
- 强探索与结构跳跃
- 在合作/竞争关系中提供差异化搜索结构

当前实现特征：

- `Logistic` 混沌初始化
- 正弦混沌驱动 DE 变异因子
- 多目标 leader 选择：`f1 / f2 / f3` 轮换
- `exploration / soft besiege / hard besiege / levy dive`
- 离散 DE 交叉
- 高斯微扰
- `makespan / tardiness / load / assignment` 联合精炼
- 后期从非支配拥挤 leader 池中取 leader
- 环境选择时保留 elite pool

关键参数：

- `chaos_init_ratio = 0.4`
- `de_crossover_prob = 0.15`
- `leader_pool_size = 10`
- `refine_prob = 0.4`
- `assignment_refine_prob = 0.35`
- `elite_keep_ratio = 0.15`

### 7.4 IG

当前 `IG` 在 HQ 中主要负责：

- 强后期收敛
- 快速利用与局部重构
- 为 HQ 提供高效的修复与精修能力

当前实现特征：

- 迭代贪婪破坏-修复
- `earliest_completion` 贪婪修复
- 局部搜索
- 非支配优先 + crowding tie-break
- SA 风格接受准则

关键参数：

- `destruction_rate = 0.15`
- `local_search_steps = 5`
- `initial_temp = 0.8`
- `cooling_rate = 0.995`

---

## 8. HQ 当前一代完整流程

### 8.1 原生演化

四个子算法先各自独立运行一代：

- `NSGA-II.step()`
- `MOEA/D.step()`
- `HHO.step()`
- `IG.step()`

### 8.2 更新档案

将四个群体全部转为 `ArchiveEntry`，并统一更新精英档案。

### 8.3 结算延迟协同信用

对于已到结算代数的交互记录：

- 若后续 `HV` 变好、`CV` 变好，则产生协同收益
- 收益写入：
  - `assist_credit`
  - `assist_mc_raw`

### 8.4 高层选择模式

高层根据：

- 演化阶段
- 停滞程度
- 当前来源结构

选择本代关系模式。

### 8.5 低层选择算子

对每个算法对：

- 计算 `MC / overlap / correlation`
- 选择 `C1/C2/R1/R2`
- 根据公平修正后的分数确定 winner/loser
- 根据算法状态动态调节交互强度

### 8.6 更新信用与日志

每次有效交互会：

- 更新算子成功次数
- 生成一条延迟协同记录
- 更新 `assist_credit`
- 周期性写入 `progress.jsonl`

### 8.7 输出汇总

最终会输出：

- `CR`
- `MCR`
- `Assist-MCR`
- `Total-MCR`
- `Blended-MCR`
- 模式统计
- 算子统计

---

## 9. 当前版本的设计重点

这版 HQ 的核心不是“强行提高 HHO 占位”，而是同时满足以下目标：

1. 保持 HQ 的整体求解质量
2. 不破坏三层结构主题
3. 让 `HHO` 的直接贡献和协同贡献都能被观测到
4. 让消融实验能同时解释“谁直接留下了解”和“谁促成了改进”

因此当前最推荐的贡献解释口径是：

- `CR`：最终档案占位
- `MCR`：直接边际贡献
- `Blended-MCR`：当前主分析指标

---

## 10. 当前默认参数结论

根据最近多轮实验，当前默认推荐值为：

- `assist_reward_scale = 0.15`
- `assist_delay_steps = 3`
- `assist_blend_lambda = 0.45`

该组参数的作用是：

- 避免协同信用过强，导致所有算法贡献被“拉平”
- 保留 `HHO` 的协同价值
- 让 `Blended-MCR` 能更自然地体现 HQ 的分层协作机制

---

## 11. 文档适用范围

本说明文档对应当前代码主干版本，主要参考以下实现：

- `src/hq_dhfsp/config.py`
- `src/hq_dhfsp/runner.py`
- `src/hq_dhfsp/algorithms/nsga2.py`
- `src/hq_dhfsp/algorithms/moead.py`
- `src/hq_dhfsp/algorithms/hho.py`
- `src/hq_dhfsp/algorithms/ig.py`
- `scripts/run_compare_batch.py`

若后续修改高层状态、低层奖励、交互算子或贡献定义，应同步更新本文档。
