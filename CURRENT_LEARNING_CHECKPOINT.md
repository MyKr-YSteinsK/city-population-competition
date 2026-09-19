# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- Main 正式判定 **S1｜数据建模与审计 COMPLETE**。
- Main 正式判定 **S2｜Validation Architecture COMPLETE**。
- Main 已复审 S3 final report 与 GitHub `main` 的 S3-Plan01..03 实际证据，正式判定 **S3｜Baseline Ladder COMPLETE**。
- 当前进入 **S4｜Feature & ML Modeling**。
- 当前工程事实源：GitHub 私有仓库 `MyKr-YSteinsK/city-population-competition` 当前 `main`；S3 Main review 基线为 `b3d0030113d3aa7a97acd0e99e2e639a2e3232c5`。
- 历史聊天数字、旧 handoff 或旧假设不得覆盖当前仓库和后续新证据。

## Confirmed Mastery / Stable Understanding

### S1

- 能区分 raw / canonical / modeling dataset，理解 canonical 层不是训练样本。
- 能解释 grain、候选 key、key uniqueness、join cardinality，以及重复 key 为什么不能靠静默去重解决。
- 能区分内部缺失、时间覆盖边界、实体覆盖缺失，以及 0 可能是合法值或 sentinel。
- 能说明 quarantine、provenance、derived value 与 raw observation 的边界。
- 能用跨表一致性和局部证据判断修复规则，理解局部规律不能直接提升为全局规则。

### S2

- 理解真实任务是时间外推，validation 必须复刻 forecast-time information boundary。
- 理解 target-history cutoff 为 `T-2`，不能使用 `T-1` 人口把任务偷换成一步预测。
- 理解 current snapshot max year 不等于历史 publication schedule；source availability 必须作为 replay assumption 显式建模。
- 理解 multi-origin expanding backtest 与 latest out-of-time holdout 回答不同问题。
- 理解 pooled city-year MSE 与 macro-origin MSE 是不同 aggregation contract。
- 理解 preprocessing、feature engineering、imputation、selection 和 model fit 都必须在每个 pseudo-origin boundary 内独立完成。
- 理解 resident 是 primary proxy 而非官方 target mapping 已被证明，registered 必须保留 sensitivity。

### S3

- 理解 baseline 的作用不是“先跑弱模型”，而是建立后续复杂模型必须真正超过的可解释 benchmark。
- 理解本题人口自身历史中 trend signal 很强，persistence 明显不足。
- 理解 recent-history length 存在 recency–stability trade-off：2 点趋势偏高方差，过长历史又过度平滑；当前 recent-3 local OLS 是最强固定基线。
- 理解 city-specific slope 是真实信号；强制 pooled shared absolute slope 会产生明显 systematic error。
- 理解 city heterogeneity 不等于可以事后给每个 city 选择不同模型；oracle headroom 与 deployable selector 必须严格区分。
- 理解 aggregate leader 与 latest-origin leader 可以冲突，resident 2021 是明确案例。
- 理解总体 MSE 可能被少数城市支配，因此后续 improvement 必须检查 broad gain、concentration 和 leave-one-city/origin robustness。

## Meaningful Evidence

### Frozen S2 Validation Contract

后续 S4–S5 material model comparison 默认必须：

1. common origins = `2011–2021`；
2. target history obey `T-2`；
3. feature/source information set obey per-origin replay boundary；
4. resident = primary target proxy；
5. registered = mandatory sensitivity；
6. pooled city-year MSE = primary aggregate；
7. macro-origin MSE 必须同时报告；
8. latest 2021 holdout 单独报告，不替代 multi-origin evidence；
9. 检查 per-origin instability；
10. 检查 dominant/top-3 city SSE concentration；
11. 保留 leakage audit；
12. 使用 covariates 时真正执行 R1 / R2 sensitivity，除非出现足以修改 contract 的新证据。

不得为了更好分数静默修改以上 contract；若新证据要求修改，必须先回 Main 做 contract review。

### S3 final baseline evidence

- S3-Plan01..03 完整覆盖 B0–B7，未越界运行正式复杂 ML、tuning、ensemble、2023 final forecast 或 submission。
- 所有 material experiment 的 leakage、R1/R2 target-only equivalence、S2 compatibility 与前序 Plan compatibility checks 均通过。
- 当前 strongest deployable fixed baseline：
  - `B3_RECENT_3POINT_OLS_TREND`
  - 每个 city / pseudo-origin `T` 只使用 `T-2` 以前最近 3 个 non-missing target observations；
  - per-city OLS：`population = intercept + slope × year`；
  - 预测 target year `T`。
- B3 resident primary pooled/macro MSE = `872.452651515143`。
- B3 registered sensitivity pooled/macro MSE = `225.421212121213`。
- Resident 2021 B3 MSE = `3705.42777777773`，不是 latest leader；B1/B2 在该 latest window 更好。后续模型不能用 aggregate improvement 掩盖 latest-regime degradation。
- Registered 2021 B3 MSE = `1485.60486111112`，aggregate/latest 当前一致。
- leave-one-origin-out：resident B3 保持 leader `10/11`，registered `11/11`。
- pooled shared-slope baselines B5/B6 明显失败，确认 local city-specific dynamics 有预测价值。
- log-growth B7 接近 B3，但 full-domain 仍未超过 B3；multiplicative growth 可作为后续 feature hypothesis，不是 baseline leader。
- leakage-safe city selector 没有形成统一收益；oracle headroom 明确只作为 non-deployable upper bound。
- B3 residual SSE 仍有明显 city concentration，后续 ML 改善必须检查是否只是修复少数 dominant cities。

## Frozen S3 Baseline Hurdle

S4/S5 的 material ML 至少需要满足：

### Primary numerical hurdle

在 resident primary、冻结 S2 contract 下：

`pooled MSE < 872.452651515143`

只超过 B0/B1 或只提高 score 不构成 ML 价值。

### Required guardrails

任何声称“优于 baseline”的 material model 必须同时报告并解释：

- macro-origin MSE；
- per-origin win/loss 与 instability；
- latest 2021；
- registered sensitivity；
- dominant-city / top-3 SSE concentration；
- leave-one-origin robustness；
- leave-one-city sensitivity；
- improved-city / city-origin coverage；
- per-origin leakage audit；
- 使用 covariates 时的 R1 / R2 sensitivity。

如果只在 aggregate 改善而 latest 明显恶化，只能称 historical aggregate improvement，不能无条件称为更强最终方案。

## Remaining Uncertainty / Boundaries

这些不构成 S3 未完成：

- 官方 target mapping 仍未直接证明；resident primary + registered sensitivity 继续保留。
- 真实 publication timing 仍未知；R1/R2 继续作为 replay assumptions。
- PDF MSE / FAQ RMSE 冲突仍无官方澄清；MSE primary / RMSE auxiliary contract 不变。
- resident latest regime 与 full-domain aggregate 存在明确 tension；这是后续模型必须解决/解释的问题。
- city-level oracle headroom 较大但 routing evidence 不稳定；不能事后手工给城市分模型。

## Active Anchor / Tracks

- Anchor：城市人口分析与预测原始赛题 + GitHub `city-population-competition`。
- Main：**S4｜Feature & ML Modeling**。
- S3 Lab：阶段结束，不再继续 S3 Plan04。
- S4 正式实验尚未激活；需要实验时新建 `Lab｜S4 实验`，继续采用阶段连续 Plan / Codex / handoff / review 工作流。

## S4 Entry Goal

S4 的中心问题：

> 怎样把 S1 的多源历史信息，在严格遵守 S2 information boundary 的前提下，转化为可学习的特征，并建立第一套真正的 ML pipeline，最终判断多源特征 + ML 是否能稳定超过 B3？

优先学习/实验问题：

- modeling dataset 应如何按 pseudo-origin 动态生成，而不是提前在完整时间轴一次性造特征；
- target-history feature：latest / lag / difference / growth / recent trend / slope 如何编码；
- 多源 covariates 在 R1/R2 下如何形成合法 as-of features；
- missingness / provenance / invalidated / derived rows 如何进入 feature policy；
- global model 与 local signal 如何结合；
- 第一批候选模型以机制覆盖为目的，不做 model sweep；
- 何时需要 Ridge / ElasticNet、tree / GBDT / LightGBM 等 JIT 理论；
- 如何用 ablation 证明提升来自哪些 feature group，而不是黑箱复杂度。

## Stage Experiment Workflow

继续采用：

`Main → 独立 Stage Lab → Plan → Codex → handoff → Lab review → 下一 Plan → Stage final report → Main`

- Plan 是研究合同和边界，不是逐行实现教程。
- Codex 保留局部数据设计、实现、诊断、验证和合理实验决策空间。
- 每次实验对话生成 Plan 时，必须**直接生成可下载文件**交给用户，不在聊天正文中预览或重复展示 Plan 全文。
- 每次生成 Plan 前，聊天中先给用户一个**简短说明**，概括本轮预计做什么、为什么做、预期获得什么证据/成果；保持简洁，不展开成 Plan 正文。
- 每份 Plan 必须使用清晰的**中文标题式文件名**，格式优先为：`S<Stage>-Plan<NN>｜<中文实验标题>.md`。例如：`S4-Plan01｜近期人口动态特征与 Ridge 趋势修正实验.md`。标题应反映本轮核心实验问题，不使用泛化名称如“实验1”“优化计划”。
- 每个 Plan 的证据进入 `reports/S4_HANDOFFS/S4-PlanXX/`。
- S4 Lab 不能自行进入 S5。
- Stage 结束时生成 `S4_FEATURE_ML_MODELING_FINAL_REPORT.md`，重点保存真实参赛者视角的 Hypothesis → Experiment → Evidence → Interpretation → Decision 链。

## Next Entry Point

Main 进入 S4，先建立 feature/modeling 的最小机制框架和第一轮实验问题；随后激活 `Lab｜S4 实验`。

## Do Not Repeat

- 不重新展开 S1 数据考古，除非最新 repo 出现会改变 source-level contract 的新证据。
- 不重新设计 S2 validation architecture，除非出现 contract-level contradiction 或新的官方证据。
- 不继续扩展类似 naive baseline；B3 已作为 S4 的 frozen hurdle。
- 不要求用户对已经明显掌握的简单概念做低价值问答。
- 不回到逐步手把手 Guided Lab 作为默认实验方式。
- 不允许单个 Codex Plan 无边界完成整个 S4/S5 或替代 Main 的 Stage completion 判定。
