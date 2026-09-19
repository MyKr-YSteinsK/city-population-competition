# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- **S1｜数据建模与审计 COMPLETE**
- **S2｜Validation Architecture COMPLETE**
- **S3｜Baseline Ladder COMPLETE**
- Main 已复审 S4 final report、Wave A/B summaries 与 machine summary，正式判定 **S4｜Feature & ML Modeling COMPLETE**。
- 当前进入 **S5｜Evidence-driven Iteration**。
- 当前工程事实源：GitHub 私有仓库 `MyKr-YSteinsK/city-population-competition` 当前 `main`。
- S4 Main review 基线：`0bfc07be5c229d7da41bd0edbd1e64998f2b2b07`。

## Confirmed Mastery

### S1–S3

此前 checkpoint 中已确认的数据审计、validation architecture 与 baseline reasoning 继续有效，不重复展开。

### S4

- 能解释为什么 B3 在本题很难被当前 residual-correction ML 超越：recent-3 local trend 已吸收主要人口动态信号，新增 target-history / covariate features 多数只是在重解释弱残差信号。
- 能区分 **latest 2021 improvement** 与 **multi-origin aggregate improvement**；latest 局部提升不能覆盖 historical aggregate 恶化。
- 能解释 R1-only 或 R1 更强的 covariate gain 为什么需要警惕：真实 publication timing 未知，freshness-dependent gain 在 R2 下退化说明结论依赖 information-boundary assumption。
- 能解释 RidgeCV 多数 outer fit 选择最大正则 `alpha=100`、同时 coefficient direction 跨 origin/scenario 翻号，属于增量 signal 弱且不稳定的证据，而不是“应该继续调更多参数”的信号。
- 能根据多组独立 falsification evidence 判断停止 feature/model fishing，而不是因为“还有模型没试过”继续机械 sweep。
- 已建立对 leakage-safe pseudo-origin feature generation、source-specific as-of covariates、provenance/quarantine/regime-aware handling、Ridge/HGB/RidgeCV pipeline、ablation 与 R1/R2 sensitivity 的工作理解。

## Meaningful S4 Evidence

- Frozen strongest model 仍为 `B3_RECENT_3POINT_OLS_TREND`。
- B3 resident pooled MSE = `872.452651515143`；registered pooled MSE = `225.421212121213`。
- S4 已实际覆盖 target-history Ridge / shape、urbanization、low-capacity HGB、density、wage、employment、city-level error memory、cross-city relative features、compact multi-source RidgeCV。
- 没有 material S4 candidate 在 resident primary multi-origin aggregate 上稳定超过 B3。
- 多个 candidate 只改善 latest 2021，同时恶化 historical aggregate，形成明确 latest-regime tension。
- density / employment / multisource 等 R1 较新信息的局部 gain 在 R2 下退化或 reversal，说明 freshness dependence 真实存在。
- multi-source RidgeCV 多数 outer fit 选择最大正则 `alpha=100`，多个 coefficient 跨 origin/scenario 翻号。
- leakage、T-2、source as-of、preprocessing、R1/R2 与 compatibility checks 均通过；负结果没有明显工程边界错误可解释。
- S4 已形成 reproducible formal ML pipeline 与多机制 falsification evidence，即使没有产生分数提升，也满足 Stage completion signal。
- 未生成 2023 final prediction / submission，未提前进入 S5。

## Partial / Fragile

- stronger boosting、tuning、ensemble/blending、partial pooling 尚未形成正向实验证据；它们仍是 Conditional，不因进入 S5 自动激活。
- 对少数 dominant cities / late-regime residual 的可预测结构仍未充分解释。
- 对“如何把 local trend 的高适应性与更稳定的 shrinkage / pooling 结合”仍缺实证。
- 学习者已能做当前阶段的机制/决策解释；更强的独立 Defense / changed-case Transfer 留到 S5/S7 继续取证。

## Remaining Uncertainty

- 官方 target mapping 仍未直接证明；resident primary + registered sensitivity 保留。
- 真实 source publication timing 仍未知；R1/R2 replay assumptions 保留。
- PDF MSE / FAQ RMSE 冲突无新的官方澄清；MSE primary / RMSE auxiliary 不变。
- B3 是否已经接近本题在当前数据条件下的可实现上限未知。
- residual concentration / late-regime difficulty 是否能被低容量、可解释方法稳定利用未知。

## Frozen Contracts Entering S5

### Validation

继续冻结 S2 contract：

- common origins `2011–2021`
- target history `T-2`
- resident primary / registered sensitivity
- pooled city-year MSE primary
- macro-origin MSE mandatory sensitivity
- latest 2021 separate diagnostic
- per-origin instability + city concentration
- leakage audit
- covariate experiments执行 R1/R2 sensitivity

### Baseline

B3 继续作为 strongest deployable fixed reference。任何“改进”必须在相同 scope 下比较，不能只凭单一 origin / target / scenario / score。

## S5 Entry Strategy

S5 中心问题：

> 从 S3/S4 已有误差和负结果证据出发，哪些剩余 hypothesis 真正值得测试；如何快速淘汰无信号方向，只对有证据的 winner 深挖，而不是继续 feature/model fishing？

初始优先级：

1. 先做 residual / regime decomposition，确认 dominant-city、late-regime、recency-vs-stability tension 是否提供可操作结构。
2. 优先考虑低容量、机制明确的迭代，例如 local-trend shrinkage / partial pooling、预注册 blend 或 robust local trend；这些方向直接回应 S3/S4 已发现的问题。
3. remaining source groups 只在存在明确 residual hypothesis 时做独立 screening，不做 feature dump。
4. stronger boosting 暂不自动激活：low-capacity HGB 与多源 RidgeCV 已给出负证据；只有 residual structure 明确支持 nonlinear interaction 时再开。
5. tuning 暂不激活：没有值得 tuning 的正向候选。
6. ensemble/blending / partial pooling 为 Conditional；若低容量 screening 有稳定信号再 deep dive。

## Parallel-first Experiment Topology

从 S5 起正式采用：

`统一 frozen base → 独立 hypotheses 并行 screening → 统一 evidence review/integration → 只对 winner deep dive → 下一 wave`

适合并行的条件：

- hypothesis / source / mechanism / robustness test 相互独立；
- 共享 frozen validation/baseline；
- 一个结果不是另一个实验设计的必要输入。

必须串行的情况：

- 后续设计依赖前一实验真实结果；
- 会修改 shared data / validation / model contract；
- 下一实验需要上一实验选出 feature/model；
- 多 worker 会同时修改同一 shared core implementation。

并行 branch 默认：

- 每 worker 独立 branch/worktree；
- 同一 frozen base；
- 不直接改 `main`；
- 不修改 checkpoint；
- 尽量不修改 shared context files；
- screening branch 只产 evidence；
- wave 结束后统一 integration；
- source branches 暂留作审计引用；
- integration 后下一 wave 使用新的统一基点；
- 长期悬空 branch 不成为事实源。

## Evidence Budget

默认采用：

`快速 screening → 有信号才 deep dive`

- stable infrastructure 不重复做低价值全量审计，只验证当前 Plan 新增风险。
- 明显失败 candidate 允许早停。
- 失败 screening 不为了 artifact 完整性强行追加 leave-one-city / leave-one-origin / concentration / 大量 ablation / model sweep。
- promising candidate 才进入完整 robustness、ablation、stability 与机制解释。

## Stage Experiment Workflow

- Main 拥有 Stage authority。
- S5 Lab 负责 wave design、Plan、Codex handoff review 与局部决策，不得自行宣布 S5 COMPLETE / 进入 S6。
- Plan 只规定核心问题、frozen boundary、forbidden actions、success/failure rule、required evidence；Codex 保留 code organization、debugging、lightweight sanity checks 和合理实现决策空间。
- 每次生成 Plan 前，聊天中先简要说明“预计做什么 / 为什么做 / 预计得到什么证据”。
- 每次 Plan **直接生成可下载 Markdown 文件，不在聊天中预览正文**。
- Plan 文件名使用：`S5-PlanNN｜中文实验标题.md`。
- 每个 Plan / parallel worker 的 evidence 进入 `reports/S5_HANDOFFS/...` 的独立目录。
- 每个 wave 结束先做统一 review/integration，再决定下一 wave。
- S5 最终生成阶段报告，保存真实参赛者视角的 `Hypothesis → Experiment → Evidence → Interpretation → Decision` 链，再交 Main 判定。

## Next Entry Point

创建 `Lab｜S5 实验`。

第一个高价值问题不是“再试哪个模型”，而是：

> B3 的失败主要来自哪些可重复 residual regimes；这些模式是否支持一组相互独立、低容量、可解释的 shrinkage / robust-trend / blend hypotheses，并值得进入 parallel screening？

## Do Not Repeat

- 不重新展开 S1 数据考古。
- 不重新设计 S2 contract，除非出现 contract-level contradiction / 新官方证据。
- 不继续扩展 S3 naive baseline ladder。
- 不重跑 S4 已明确 falsified 的同类 feature/model correction。
- 不因为阶段编号进入 S5 就自动开启 stronger boosting / tuning / ensemble。
- 不做无假设的大规模 feature/model sweep。
