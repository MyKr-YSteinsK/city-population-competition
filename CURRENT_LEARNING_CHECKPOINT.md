# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- **S1｜数据建模与审计 COMPLETE**
- **S2｜Validation Architecture COMPLETE**
- **S3｜Baseline Ladder COMPLETE**
- **S4｜Feature & ML Modeling COMPLETE**
- **S5｜Evidence-driven Iteration COMPLETE**
- Main 已复审 S6 Plan02–04 的 clean rebuild、final fit/prediction、final freeze 与 provenance 证据，正式判定 **S6｜Final Solution & Reproduction COMPLETE**。
- 当前正式进入 **S7｜Transfer & Defense**。
- S6 closeout engineering base：GitHub `main` @ `bfe265d3d57f43938e967890567a4162d7b3c896`。
- 当前阶段不再继续优化 final model；S7 只做能力整合、Defense、Transfer 与最终 Success Definition 审查。

## S6 Confirmed

- 已将 S5 的 resident-primary candidate 固化为可执行 final solution，而不是继续 model search。
- 已完成 disposable-worktree clean raw rebuild：
  - official PDF + 7 legal XLSX；
  - S1 raw→canonical equivalence PASS；
  - S5 Plan04 1760-row / 144-feature matrix rebuild PASS；
  - Plan04B 2320-row replay rebuild PASS；
  - source-cap / quarantine / benchmark / population-history / B3-history boundary violations = 0。
- 已建立 2023 current feature foundation：
  - 160 rows；
  - resident / registered × R1 / R2 各 40 rows；
  - 2240 boundary audit rows；
  - no target label after 2021。
- 已基于 rebuilt foundation 完成 final-fit candidate generation：
  - resident F4 R1 selected config = `FRM_C3_FLEXIBLE_REGULARIZED`；
  - training rows = 573；
  - max label year = 2021；
  - seed = 20260919；
  - prediction year = 2023。
- 最终方案已冻结为：
  - target = resident；
  - framing = `F4_LOG_GROWTH_FROM_LATEST`；
  - scenario = `R1_REFERENCE_SNAPSHOT_LAG`；
  - model = CatBoost C3 / `FRM_C3_FLEXIBLE_REGULARIZED`。
- 最终 submission 已生成并冻结：
  - path = `reports/S6_HANDOFFS/S6-Plan04/submission_final.csv`；
  - 40 rows；
  - `city1..city40`；
  - year = 2023；
  - columns = `city_id,year,pred`；
  - 2-decimal fixed-point rendering；
  - no clipping / routing / ensemble / manual city correction。
- final freeze 保留高精度 provenance，并记录 exact immutable source commit 与 SHA256。
- `test.xls`、B-data、external 2022/2023 truth 未进入 final solution。
- registered alternatives 已在 Plan03 单独生成并审查；B3/RLT3 对 city40 出现合法历史外推导致的负值，因此未通过 positive-prediction schema，也未通过 clipping 或人工修正强行提交。

## Reproduction Assessment

S6 的核心 reproduction gate 判定为 **通过，但带明确债务**。

已实际发生的链路是：

`raw official sources → canonical rebuild → 144-feature/replay/current foundation rebuild → final fit → 2023 candidate predictions → frozen final submission`

因此满足 Brief/Plan 对“原始数据可重新构建 final solution、关键中间产物可解释、不能依赖手工 Excel 改表”的核心要求。

未完成的更强复验包括：

- full historical F4 OOF replay；
- exact all-origin F4 selection reproduction；
- full RLT3 historical regeneration；
- 更深的一次性 fresh end-to-end release proof。

这些被保留为 **reproduction debt / supplemental verification**，而不是当前 S6 blocker。S7 最终项目审查时必须如实说明，不能把它们描述成已经完成。

## Final Solution Status

### Frozen resident-primary final solution

`resident / F4_LOG_GROWTH_FROM_LATEST / R1_REFERENCE_SNAPSHOT_LAG / CatBoost C3`

状态：

**FINAL_SOLUTION_FROZEN_FOR_PROJECT_REVIEW**

关键边界：

- local historical evidence 支持 resident F4；
- R1 historically stronger than R2；
- leaderboard score 尚不可用；
- 官方“总人口”到 resident / registered 的语义映射仍未外部确认；
- 因此它是当前项目 contract 下的 final solution，不是已被 leaderboard 或官方 target mapping 证明的全局最优解。

### Reference

`B3_RECENT_3POINT_OLS_TREND`

继续作为 simple benchmark / historical reference。

### Registered alternative

registered 路径已保留实验与 candidate evidence，但未形成可直接提交的 final alternative；city40 的合法历史异常导致 B3/RLT3 负外推。不得事后 clipping 或人工修值伪造一个“可用” registered final solution。

## S6 Partial / Fragile

- leaderboard score unavailable；当前 final solution 尚未获得平台外部反馈。
- official “总人口” target mapping 仍 unresolved。
- full historical F4 replay 被延期。
- current R1 final choice依赖历史 replay assumption；真实 publication timing 不可直接恢复。
- final submission 的数值已冻结，但对 leaderboard metric / score-scale 的 sanity question 仍保留；首次真实提交后再决定是否专项复盘。
- registered sensitivity 仍是语义风险，不应因为 resident final freeze 而被遗忘。

## Remaining Uncertainty

- 官方 target mapping：resident vs registered。
- PDF MSE / FAQ RMSE 与 leaderboard score-scale 之间的解释差异。
- leaderboard score 尚未取得。
- full historical F4 reproduction debt。
- 一个旧 S5 半成品模型仍在独立运行；它不是 S7 blocker。未来结果只作为 late supplemental evidence 合并，除非出现足以实质改变 current final-solution conclusion 的证据。

## S7 Entry Point

S7 中心问题：

> S1–S6 的能力是否已经真正属于学习者，而不是只存在于 repo、报告和 Codex 输出里？

S7 不重新教学 S1–S6，也不重新跑模型搜索。

评估分三条主线：

1. **Project Decision Chain Integration**
   - 从任务取证、数据审计、validation、baseline、feature/model、S5 iteration 到 S6 final freeze，能否解释每个关键决策为什么发生、替代方案为什么被拒绝。

2. **Defense**
   - problem framing；
   - target / horizon / information boundary；
   - validation architecture；
   - leakage；
   - B3 为什么强；
   - F4 为什么成为 final framing；
   - multi-source / stronger model 为什么多数失败；
   - R1/R2、registered sensitivity、tail risk、city concentration；
   - final solution 的 reproduction boundary 与 remaining uncertainty。

3. **Transfer**
   - 改 forecast horizon；
   - 改 feature availability / publication lag；
   - 换城市范围或加入 unseen cities；
   - 改 metric；
   - 改 target mapping；
   - 截断历史或删最新 source；
   - 要求判断哪些 contract / validation / feature / model 必须重设计，哪些可以保留。

## S7 Assessment Strategy

- 新建独立 `Review｜迁移与答辩` 对话，减少旧教学上下文对答题的提示。
- 以 learner-owned explanation、diagnosis、changed-case redesign 为主要证据。
- 不考 API 背诵，不机械复述报告数字。
- 可使用 repo / final report 做事实核对，但核心判断必须由学习者给出。
- Review 只回传证据与脆弱点；最终项目 COMPLETE 仍由 Main 对照 Brief 的 Reproduction + Defense + Transfer Success Definition 决定。

## Do Not Repeat

- 不重新开启 S5/S6 model sweep、tuning、Optuna、ensemble、routing、city gating。
- 不为了补 artifact 仪式感重跑所有 historical worker。
- 不把 deferred full historical F4 replay 说成已完成。
- 不把 leaderboard unavailable 说成模型已在线验证。
- 不把 resident final solution 说成官方 target mapping 已确认。
- 不重新教授 S1–S6 已经有稳定证据的基础内容。
- S7 不负责改变 mainline；若 Review 暴露重大 blocking gap，回 Main 决定是否做最小 remediation。
