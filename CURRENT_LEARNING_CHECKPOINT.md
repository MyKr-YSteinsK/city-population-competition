# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- **S1｜数据建模与审计 COMPLETE**
- **S2｜Validation Architecture COMPLETE**
- **S3｜Baseline Ladder COMPLETE**
- **S4｜Feature & ML Modeling COMPLETE**
- **S5｜Evidence-driven Iteration COMPLETE**
- Main 已复审 S6 Plan02–04 的 clean rebuild、final fit/prediction、final freeze 与 provenance 证据，正式判定 **S6｜Final Solution & Reproduction COMPLETE**。
- **S7｜Transfer & Defense 已启动，但当前暂停最终 closeout**：Review 已提供了有价值的迁移/答辩证据，并暴露出“真实 2023 target 与最终测试 regime 可能与历史 backtest 明显不同”的新问题。
- 用户已明确批准新增 **S8｜真实目标重建与答案驱动的可推广建模（Post-hoc Target Reconstruction & Reverse Modeling）**。S8 继续负责 candidate truth / identity / year-alignment / regime forensic，不宣告完成。
- 用户进一步明确批准新增 **S9｜文献驱动的人口预测方法与盲测挑战（Literature-Guided Forecast Benchmark & Procedurally Blinded 2023 Holdout）**，当前 Mainline 的模型研究入口转入 S9；S8 作为独立 truth/evaluation support track 保留。
- S6 closeout engineering base：GitHub `main` @ `bfe265d3d57f43938e967890567a4162d7b3c896`。
- S6 frozen final solution 继续保持不变；S8/S9 都是后续 research extensions，不回写或伪装成原 competition-legal final solution。
- S9 完成后与 S8 证据一并返回 S7，整合后再做最终 Reproduction / Defense / Transfer Success Definition 审查。

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

## Supplemental Jump Investigation

- S7 Supplemental Jump Event Study 已由 Main 复核，worker commit：`7b35fec79e3b4b2e1145127fa415b9aa8deb83ca`，结论正式接受为 **`H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE`**。
- 该 investigation 没有训练 classifier / hurdle / mixture / magnitude model，也没有生成新 2023 prediction 或 submission；frozen S6 final solution 不变。
- q90 jump label 使用 outer-training universe 内 `pseudo_target_year <= T-2` 定义，没有使用当前 outer target 调 threshold；primary sample 有 68 个 unique jump positives，只有 6/11 origins 至少有 2 个 positives，且事件明显集中在少数年份。
- F4 remaining-tail relevance 成立：jump rows 的 F4 MSE 约为 normal rows 的 `23.29×`（R1）/`25.66×`（R2），说明 unresolved error 确实集中在 structural-surprise regime。
- 但 precursor gate 失败：预注册 population / employment / sector / density / urbanization / wage / living-standard 等 pre-origin features 中，没有 feature 在跨-origin稳定性与 R1/R2 robustness 上通过 Gate B/C。部分 pooled AUC / standardized difference 看似有方向，但 origin-level consistency 不足，不能升级为 deployable jump detector。
- 旧 Plan06B CatBoost F2 的 conditional re-analysis 也未通过 Gate D：pooled jump rows 相对 B3 仍恶化约 `5.29%`（R1）/`6.81%`（R2），normal rows 恶化更明显；虽然 jump stratum 在 6/11 origins 局部获胜，但不足以支持“先识别 jump 再用 F2 修正”的可部署结构。
- Main 决定：**关闭 supplemental jump-model track**。不再授权 jump classifier、hurdle/mixture、direction/magnitude model、city routing 或基于已知 worst-city 的条件修正。该结果作为一个重要 limitation 保留到 S7 Defense/Transfer：本题可识别“哪里风险高/哪里 tail 集中”，但现有合法信息不足以稳定提前识别具体 jump event。
- 只有未来出现新的、合法且 materially stronger pre-origin evidence，才允许 Main 重新评估；现有证据本身不构成 reopening 条件。

## S8｜真实目标重建与答案驱动的可推广建模

### Stage Purpose

S8 允许在 **post-hoc research sandbox** 中使用外部真实城市统计资料与高置信度重建的 2023 candidate truth，回答两个新问题：

1. DataFountain competition year=2023 的隐藏 target 到底最可能对应什么现实统计口径、统计年份和城市实体？
2. 已经看到候选答案以后，能否从误差结构中反推出一个 **不是 city-by-city 手调、而是能在历史 analogue / changed-case 中重新成立的预测机制**？

本阶段不是重新解释 S6 final，也不以“把 candidate truth 拟合到最低误差”为目标。

### Current Forensic Evidence

- 已构造一份 40-city candidate mapping 与 candidate 2023 truth，用于 post-hoc validation；它目前是 **高置信度重建假设，不是 DataFountain 官方 hidden-label 文件**。
- 当前工作假设是 competition year 与现实统计年存在约 +1 对齐关系，因此 competition 2023 可能对应 real 2022 resident population；该假设必须在 S8 中用多表、多年份、官方统计来源继续取证。
- 对当前 candidate truth 的离线比较显示：
  - frozen resident F4-R1 约 MSE 1912；
  - resident B3 约 MSE 4225；
  - simple persistence（沿用 competition-2021 resident）约 MSE 122。
- 这个排序与历史 multi-origin backtest 明显冲突，提示 final test 可能存在 post-census / post-revision regime shift；在完成 truth reconstruction audit 前，不把该解释写成事实。

### Hard Separation: Competition-Legal vs Post-hoc

S0–S6 原竞赛方案继续遵守“禁止外部数据”。

S8 新增一个明确隔离的外部取证层：

- **允许**：真实城市反匿名、多年度常住/户籍人口、统计公报、人口普查、人口密度/城镇化/就业/工资/生活水平等外部资料，用于 identity verification、year alignment、candidate truth reconstruction、error forensics。
- **禁止**：把 real 2021/2022/2023 等在原预测时不可见的外部数据作为 feature，再声称模型是原比赛可部署方案。
- 外部未来数据只能作为 post-hoc truth / diagnostic evidence。
- 所有 S8 产物必须显式标注 `POST_HOC_EXTERNAL_RESEARCH`，不得覆盖 `submission_final.csv`。

### Answer-aware Modeling Contract

S8 允许“从答案出发”发现 hypothesis，但不允许直接把答案拟合成规则。

允许：

`candidate truth → error pattern → mechanism hypothesis → historical analogue test → deployability judgment`

禁止：

`candidate truth → per-city correction / tuned coefficients → declare generalizable`

任何新模型若要获得“可推广 / deployable”标签，至少必须：

- 只使用原 competition information boundary 内可获得的变量；
- 不使用 2023 candidate truth 做 row-specific routing；
- 不使用真实城市名称作为隐藏 shortcut，除非研究目标明确变成“已知真实城市身份后的外部数据预测”，且与原赛题模型严格分开；
- 在多个历史 pseudo-origin 上重新验证；
- 对 census/revision hypothesis，优先使用 **2010 census analogue**：把 competition 2011 附近视作类似基准重估点，检查从该时点向后两年的模型是否也应采用 level anchoring / growth shrinkage，而不是只在 2023 上成立；
- 2023 candidate truth 只能作为 post-hoc target，不能成为唯一 acceptance evidence。

### S8 Workstreams

1. **S8-A｜Target Forensics**
   - 40 city de-anonymization；
   - competition year ↔ real statistical year 对齐；
   - resident vs registered 目标语义；
   - 多年度 population + registered + density + urbanization + employment + wage + age structure 等交叉验证；
   - official source 优先；
   - 每个 city 输出 mapping confidence 与 conflicting evidence。

2. **S8-B｜Candidate Truth Reconstruction**
   - 收集 real 2021 / 2022 resident population；
   - 明确年末人口、七普口径、统计公报修订、整数万人 round/截断等；
   - 形成 versioned candidate truth；
   - 对不确定 city 保留区间或多个候选，不强行单值。

3. **S8-C｜Frozen Solution Error Autopsy**
   - 统一比较 persistence / B3 / F4-R1 / F4-R2 / registered controls；
   - decomposition by city / province / scale / census jump / pre-2020 trend；
   - 判断系统 bias、tail、regime shift 与 statistical re-basing 的关系。

4. **S8-D｜Reverse Hypothesis Mining**
   - 从答案发现可解释机制，如：
     - census-aware level anchoring；
     - growth shrinkage toward persistence；
     - breakpoint-aware trend；
     - pre-census trend 与 census jump 分离；
     - low-capacity partial pooling。
   - 每次只允许一个机制假设，不进行大模型 sweep。

5. **S8-E｜Generalization Gate**
   - 把 candidate mechanism 放回历史数据；
   - 使用 multi-origin + census analogue + changed-case 验证；
   - 检查其提升是否仅来自 2023 hindsight；
   - 若历史 evidence 不支持，则明确标记为 `ANSWER_FIT_NOT_DEPLOYABLE`。

6. **S8-F｜Post-hoc Research Solution**
   - 只有通过 Generalization Gate 的机制，才形成 post-hoc research model；
   - 与 frozen S6 solution 分开报告；
   - 说明它若在原比赛前部署，需要哪些可见信息与先验假设。

### S8 Stop / Success Conditions

S8 不以 candidate-truth MSE 越低越好作为唯一目标。

阶段完成需要：

- 40-city mapping / year alignment / target semantics 达到足够高的取证质量，剩余不确定项显式隔离；
- candidate truth 可重复构造并有 provenance；
- 冻结模型真实误差得到可信复盘；
- 至少一个 answer-derived mechanism 被送回历史 analogue 验证；
- 最终明确区分：
  - `POST_HOC_ONLY`
  - `HISTORICALLY_SUPPORTED_BUT_UNPROVEN`
  - `DEPLOYABLE_UNDER_ORIGINAL_INFORMATION_BOUNDARY`
- 不论是否找到更强模型，都形成“为什么 2023 与历史 validation 排序不同”的可辩护结论。

### Return Point

S8 完成后返回 **S7｜Transfer & Defense**。

S7 最终答辩需要新增一项：

> 已知 final test 发生 distribution / statistical-regime shift 时，怎样区分“validation 错了”“数据口径变了”“真实机制变了”“post-hoc answer fitting”，以及怎样把答案启发转化为新的可推广 hypothesis？

## S9｜文献驱动的人口预测方法与盲测挑战

### Stage Purpose

重新模拟一次真实比赛：把 S8 的 2023 candidate truth 当作不可见测试集，只用原 competition-legal 历史数据、S2 时间验证与人口预测文献启发的方法完成训练、调整、筛选与最终 2023 预测；**模型/预测冻结后才一次性 reveal 2023**。

由于学习者与 Main 已经见过当前 candidate truth，S9 只能称为 **procedurally blinded holdout**，不能声称是认知上完全未知的 blind test。实验完整性依靠数据隔离、worker boundary、commit/hash freeze 与 one-time reveal。

### Blind Boundary

- 训练/调参 worker 不得读取 S8 candidate truth、real-2022 population、逐城市 2023 error 或答案驱动的 city-specific correction。
- 所有 candidate promotion 只允许使用 historical OOF evidence。
- 2023 predictions 必须在 reveal 前生成并 commit/hash。
- reveal 后计算 2023 MSE 只用于最终外部评估；如果 reveal 后继续调模型，必须另开 `POST_REVEAL_RESEARCH` round，不得把修改后的结果称为 blind result。
- S8 truth 版本在 reveal 前应锁定 provenance/version；若后续 truth reconstruction 发生 materially changed revision，需要把旧/新 truth 作为不同 evaluator version 报告。

### Frozen Internal Validation

- resident primary；
- common expanding pseudo-origins 2011–2021；
- target history cutoff `<= T-2`；
- pooled city-year MSE primary；
- macro-origin / per-origin / RMSE / MAE / bias / coverage / leakage 作为稳定性证据；
- 预注册 `T=2013` 为 **census analogue stress slice**：其 cutoff `<=2011` 与 final `T=2023` cutoff `<=2021` 在 census-anchor geometry 上对应，用于检验 census jump 是否应继续外推。

### S9 Research Lines

1. **Local statistical forecasting**
   - Persistence / B3 / RLT3 references；
   - SES；
   - Holt；
   - damped Holt / ETS；
   - Theta；
   - bounded ARIMA benchmark。

2. **F4 framing × learner**
   - Ridge；
   - LightGBM；
   - XGBoost；
   - CatBoost existing reference。
   - 目的：区分 target framing improvement 与 learner effect，而不是继续模型扫货。

3. **Forecast combination**
   - simple mean；
   - median / trimmed robust combination；
   - historical-error-based static weighting。
   - 权重只来自历史 inner/OOF evidence。

4. **Dynamic expert gating / FFORMA-like MoE**
   - expert pool 控制在 3–5 个机制互补模型；
   - low-capacity soft gate；
   - gate features 只允许 pre-origin state information，例如 recent growth / slope / acceleration / volatility / scale / expert disagreement / census-distance proxy；
   - 必须 nested / OOF，禁止 current outer target leakage。

5. **Census-aware mechanism**
   - pre-census slope；
   - census jump；
   - post-census growth；
   - level reset vs persistent growth；
   - trend/growth damping。
   - 先检验 2013 analogue，再看完整 multi-origin；不得根据 2023 truth 调 shrinkage。

### Internal Freeze

S9 champion 必须在 reveal 前根据内部 evidence 冻结。

每个 material candidate 至少报告：

- pooled historical MSE；
- RMSE / MAE / bias；
- macro-origin MSE；
- origin wins/losses；
- latest-origin MSE；
- 2013 census analogue MSE；
- coverage / leakage audit；
- model complexity / deployability。

可以同时冻结少量 diagnostic finalists，但 champion 身份不得在 reveal 后改写。

### One-time 2023 Reveal

当 champion + finalists 的 2023 predictions 已 commit/hash 后：

1. 解锁 S8 candidate truth；
2. 统一计算 40-city 2023 MSE / RMSE / MAE / bias；
3. 同时展示 `internal historical MSE` 与 `2023 holdout MSE`；
4. 检查模型排序是否一致；
5. 分析 regime shift / census rebasing / expert selection 是否解释 ranking reversal。

### Current Candidate Priority

第一条正式新实验线：

`Persistence → SES → Holt → Damped Holt/ETS → Theta → bounded ARIMA`

原因不是补模型名单，而是它们直接检验“人口趋势应继续、阻尼还是重置”的机制。

之后再进入：

`F4 × learner → forecast combination → dynamic gate → census-aware mechanism`

### Relationship to Prior Routing Decision

S7 supplemental jump investigation 关闭的是当时证据下的 jump classifier / routing。S9 是用户基于新的文献与 post-hoc evidence 明确授权的新 research stage，因此允许重新研究 **general expert combination / dynamic gating**，但必须遵守新的 blind/nested validation contract；这不等于回写 S6 或否定旧结论。

### Return Point

S9 完成后与 S8 truth/evidence 一并返回 S7。

S7 最终 Transfer/Defense 需要回答：

> 如果内部 backtest 与 blind final holdout 给出不同模型排序，应如何解释、怎样判断 validation misspecification、regime shift、statistical rebasing 与 post-hoc overfitting？

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
