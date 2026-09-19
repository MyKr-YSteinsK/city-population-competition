# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- Main 正式判定 **S1｜数据建模与审计 COMPLETE**。
- Main 已完成对 Lab final report 与 GitHub `main` 实际 S2 证据的复审，正式判定 **S2｜Validation Architecture COMPLETE**。
- 当前进入 **S3｜Baseline Ladder**。
- 当前工程事实源：GitHub 私有仓库 `MyKr-YSteinsK/city-population-competition` 的当前 `main`；S2 Main review 基线为 `2045280bdc7d1ee2d97d2f128c1dff235152f9ba`。
- 历史聊天数字、旧 handoff 或旧假设不得覆盖当前仓库和后续新证据。

## Confirmed Mastery / Stable Understanding

### S1

- 能区分 raw / canonical / modeling dataset，理解 canonical 层不是训练样本。
- 能解释 grain、候选 key、key uniqueness、join cardinality，以及重复 key 为什么不能靠静默去重解决。
- 能区分内部缺失、时间覆盖边界、实体覆盖缺失，以及 0 可能是合法值或 sentinel。
- 能说明 quarantine、provenance、derived value 与 raw observation 的边界。
- 能用跨表一致性和局部证据判断修复规则，理解局部规律不能直接提升为全局规则。

### S2

- 理解随机 IID split 不适合本题的核心原因：真实任务是时间外推，验证必须复刻 forecast-time information boundary。
- 理解真实 2023 target history 截止 2021，因此历史 pseudo-target `T` 的 target-history cutoff 为 `T-2`，不能用 `T-1` 人口把任务偷偷变成一步预测。
- 理解 current file max year 是当前 snapshot fact，不等于历史 publication schedule；source availability 必须作为 replay assumption 显式建模。
- 理解 multi-origin expanding backtest 与 latest out-of-time holdout 回答不同问题：前者用于稳定性与模型选择主证据，后者用于最终任务接近性 / stress-window 诊断。
- 理解 pooled city-year MSE 与 macro-origin MSE 是不同 aggregation contract；当前数值相等来自每个 origin 都有相同 scoreable city count，不是数学恒等式。
- 理解 feature engineering、preprocessing、imputation、selection 与 model fit 都必须在每个 pseudo-origin information boundary 内独立完成，不能先看完整时间轴再切分。
- 理解 resident target 仍是主代理而非官方映射已被证明，registered 必须作为 sensitivity target 保留。

## Meaningful Evidence

### S1 frozen foundation

- 合法 raw source、canonical tables、quarantine、provenance 和 S1 rebuild contract 已冻结并可重建。
- `test.xls` 保持 quarantine，不进入 canonical / validation / modeling truth。

### S2 final evidence

- Lab final synthesis：`S2_VALIDATION_ARCHITECTURE_FINAL_REPORT`。
- GitHub `reports/S2_HANDOFFS/S2-Plan01..05/` 保存逐步实验与 handoff 证据。
- Plan05 生成 3,520 prediction/residual rows 和 3,520 leakage audit rows；leakage audit 全部 PASS。
- S2 使用 deterministic target-history-only probes 验证 architecture，没有越界进入正式 ML modeling / tuning / submission。
- common strict origin domain 冻结为 `2011–2021`；resident-only 2010 仅可作为辅助诊断，不混入 resident/registered 主比较。
- target-history horizon = 2 years：pseudo-target `T` 只允许使用 population target history through `T-2`。
- primary target proxy：`常住人口（万人）`；mandatory sensitivity：`户籍人口（万人）`。
- primary metric：MSE；auxiliary metric：RMSE；competition-score transform `1/(1+MSE)` 只有在 MSE scope 明确时才可解释。
- validation architecture：multi-origin expanding backtest 为主模型选择 / 稳定性证据，2021 latest holdout 为 final-task proximity diagnostic。
- primary aggregation：pooled city-year MSE；mandatory sensitivity：macro-origin MSE。若未来 coverage 不等导致 ranking disagreement，必须触发 contract review。
- R1 `REFERENCE_SNAPSHOT_LAG` 与 R2 `CONSERVATIVE_PUBLICATION_STRESS` 保留为 source-specific information-set replay assumptions，不宣称是真实 publication facts。
- Plan05 中 2021 是所有固定 naive probe / target 组合的高误差甚至最差 origin，支持“不能只靠 latest holdout 做模型选择”。
- per-origin stability 与 dominant-city SSE concentration 被纳入后续 S3–S5 必须报告的诊断边界。

### S3 Plan01 baseline evidence

- S3 Plan01 在不修改冻结 S2 contract 的前提下，正式重跑 B0 persistence、B1 recent two-point trend，并新增 B2 expanding per-city linear trend。
- common origins 仍为 `2011–2021`，resident primary 与 registered sensitivity 均完整覆盖；prediction/residual rows 为 5,280，leakage audit 为 5,280，全部 PASS。
- B1 在 resident 与 registered 的 pooled/macro MSE 上均为当前最强 aggregate baseline；B2 相比 B0 有改善，但相比 B1 在两个 target 上都变差。
- B2 的 resident 结果显示 recent-vs-long-history 的明确 trade-off：2021 latest MSE 略优于 B1，且跨 origin MSE dispersion 更低；registered 在 2021 和 dispersion 上均不如 B1。B2 改善 B0 的贡献在 resident 受 city3 集中影响，registered 则更 broad-based。
- R1/R2 对三个 target-history-only baseline 数值完全一致；B0/B1 与 S2 Plan05 的既有 probe 逐行兼容。S3 Plan01 未使用 covariates、复杂 ML、tuning、ensemble、2023 forecast 或 submission。

## Frozen S2 Validation Contract

后续 S3–S5 的 material model comparison 默认必须：

1. 使用 common origins `2011–2021`；
2. target history obey `T-2`；
3. feature/source information set obey per-origin replay boundary；
4. resident 作为 primary target proxy；
5. registered 作为 mandatory sensitivity；
6. pooled city-year MSE 为 primary aggregate；
7. macro-origin MSE 必须同时报告；
8. latest 2021 holdout 单独报告，不替代 multi-origin evidence；
9. 检查 per-origin instability；
10. 检查 dominant-city SSE concentration；
11. 保留 leakage audit；
12. 使用 covariates 时执行 R1 / R2 sensitivity，除非有明确证据允许删除其中一个 scenario。

任何后续 Stage 不得为了更好分数静默修改以上 contract。若新证据要求修改，必须先回 Main 做 validation-contract review。

## Remaining Uncertainty / Boundaries

这些是真实未决边界，不再视为 S2 未完成：

- “总人口”到 resident / registered 的官方精确映射仍没有直接官方证明；采用 resident primary + registered sensitivity。
- 真实 source publication timing 未知；采用 R1/R2 replay assumptions。
- PDF formal section 的 MSE 与 FAQ RMSE 冲突没有新的官方澄清；本地 contract 冻结为 MSE primary / RMSE auxiliary。
- Plan05 target-only probes 无法产生 R1/R2 数值差异；真正的 covariate sensitivity 留待 S3/S4 自然触发。
- 2021 属于明显困难的 late regime；后续模型必须同时看 historical stability 与 latest-origin realism。
- 一些指标可能被少数城市的 SSE 主导；后续 improvement 必须判断是 broad gain 还是单城修复。

## Active Anchor / Tracks

- Anchor：城市人口分析与预测原始赛题 + GitHub `city-population-competition`。
- Main：**S3｜Baseline Ladder**。
- S2 Lab：阶段任务结束，不再继续 Plan06。
- S3 Plan01 baseline evidence 已完成 Codex 交付，等待 Lab/Main 复审；后续仍需新 Plan，不由本轮自动推进。

## Stage Experiment Workflow

从 S2 起，阶段实验采用“独立实验对话 → 连续 Plan → Codex 执行 → 证据回传 → 实验对话复审 → 下一 Plan”的工作流。

- 每个 Stage 在需要正式实验时新建独立实验对话。
- 实验对话负责读取最新 repo 与证据、提出当前最高信息价值的问题、制定下一份 Plan、审查 Codex handoff，并迭代到 Stage evidence sufficient。
- Plan 是目标、边界、验收与必要上下文，不是逐行实现教程；给 Codex 保留局部设计、诊断、验证和合理决策空间。
- Codex 必须回传 actual changes、actual runs、evidence、failure/deviation、remaining uncertainty；不能用“完成了”代替证据。
- 每个 Plan 的审查材料进入 Stage 专属独立 handoff 文件夹。
- Stage 结束时实验对话生成完整 Markdown final report，重点记录真实参赛者视角的 `Question → Hypothesis → Experiment → Evidence → Interpretation → Decision` 链。
- 只有 Main 有权正式判定 Stage complete 并推进下一 Stage。

## Next Entry Point

进入 S3｜Baseline Ladder。

S3 的中心问题：

> 在不依赖复杂 ML 的情况下，哪些简单、可解释、严格遵守 S2 validation contract 的预测规则已经构成有竞争力的 baseline，以及后续 ML 至少必须超过什么标准才有价值？

Main/Lab 先复审 `reports/S3_HANDOFFS/S3-Plan01/`，决定是否保留 B1 为当前 strongest baseline，并为 recency–stability trade-off 设计下一份有限、预注册的 S3 问题；本 Plan 不宣布 S3 COMPLETE，也不进入 S4。

## Do Not Repeat

- 不重新展开 S1 数据考古，除非 GitHub `main` 出现会改变 source-level contract 的新证据。
- 不重新设计 S2 validation architecture，除非后续实验发现 contract-level contradiction 或取得新的官方 target/publication evidence。
- 不要求用户对已经明显掌握的简单概念做低价值解释性问答。
- 不回到逐步手把手 Guided Lab 作为默认实验方式。
- 不允许单个 Codex Plan 越过当前 Stage 或替代 Main 的 Stage completion 判定。
