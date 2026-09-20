# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- **S1｜数据建模与审计 COMPLETE**
- **S2｜Validation Architecture COMPLETE**
- **S3｜Baseline Ladder COMPLETE**
- **S4｜Feature & ML Modeling COMPLETE**
- Main 已复审 S5 final report、Main handoff、candidate comparison、experiment registry 与异常/recovery 边界，正式判定 **S5｜Evidence-driven Iteration COMPLETE**。
- 当前正式进入 **S6｜Final Solution & Reproduction**。
- S5 Main review 使用的 canonical engineering base：GitHub `main` @ `2fbba5f2362e4dc1b1d8f5679d0fd13453fa840e`。
- 后续工程事实继续以 GitHub 私有仓库 `MyKr-YSteinsK/city-population-competition` 当前 `main` 为准。

## S5 Confirmed

- 能从 residual / stability / negative evidence 出发提出高信息价值实验，而不是按“还没试过什么模型”继续搜索。
- 能区分 primary resident improvement 与 registered sensitivity：resident 上的真实提升不能包装成 universal target improvement。
- 能把 R1/R2 差异作为 information-boundary sensitivity 解释，不把 R1 更好自动当成模型更强。
- 能区分完整 performance evidence、recovery-only evidence 与 coverage-incomplete evidence；不可恢复或不完整结果不得进入 ranking。
- 能执行 screening-first / winner-only deep dive，并在强模型、local/hierarchical、target-framing 等多类 hypothesis 后主动收缩候选集。
- 能解释为什么 tail metric、origin wins、latest、scale bin、city concentration 与 primary pooled MSE 需要共同阅读，而不是由单一数字决定。
- 能在 S5 结束时停止 model fishing；没有为了阶段“必须更复杂”继续新增模型。

## Meaningful Evidence

- S5 Experiment Registry 已形成完整的 `Hypothesis → Experiment → Evidence → Interpretation → Decision` 链。
- Frozen validation contract 未被 S5 为了分数静默修改；common origins、T-2、resident primary / registered sensitivity、R1/R2、pooled/macro 与 latest diagnostics 继续有效。
- B3 resident pooled MSE = `872.4527`，仍是冻结 benchmark。
- `F4_LOG_GROWTH_FROM_LATEST` 在 resident primary 上形成当前最强有效 signal：
  - R1 pooled MSE = `739.6007`，相对 B3 改善约 `15.227%`；
  - R2 pooled MSE = `802.4457`，改善约 `8.024%`；
  - origin wins = `5/11`、`7/11`；
  - latest-2021 MSE = `2956.2890`、`3595.0013`；
  - RMSE / MAE / median AE / bias / city SSE concentration 总体优于 B3；
  - resident low / medium / high scale bins 在 R1/R2 均改善。
- F4 的 P95 tail error 并未全面优于 B3，因此它不是全维度统治型 improvement。
- `F3_DELTA_FROM_LATEST` 是次级有效 framing signal，但 resident pooled improvement 小于 F4。
- Plan06A–D 的强模型 residual controls 没有在 resident primary 上替代 B3；CatBoost F2 只是其中最强有效 control。
- registered sensitivity：
  - B3 = `225.4212`；
  - RLT3 = `223.6909`，约 `+0.768%`，`8/11` origin wins；
  - F3 = `228.3363 / 227.5234`；
  - F4 = `243.2445 / 246.8899`。
  因此 F4 不能解释成 resident + registered 的 universal improvement。
- Plan06F = recovery-only：执行/selection facts 可恢复，但核心 OOF/performance evidence 不可恢复，不参与排名。
- Plan06G = coverage incomplete：不完整 OOF 与 outer failures 使其 performance 不可用于 candidate promotion。
- Plan06H = city-local unsupported：不支持 routing/gating。
- S5 Final Candidate Panel 已主动收缩为 B3、RLT3、CatBoost F2 control、F3、F4，而不是保留所有尝试模型。
- Pre-S6 cleanup 已将实验 branch/worktree 收束回 canonical `main`；历史 unique worker commits 通过 immutable archive tags 保留审计引用。

## Final Candidate Status

### Primary final-solution candidate for S6

`F4_LOG_GROWTH_FROM_LATEST`

状态：

**promoted_to_S6_resident_primary_candidate**

理由：

- 在冻结 resident primary contract 下，R1 / R2 pooled MSE 都实质低于 B3；
- 两个 scenario 下 latest-2021 均优于 B3；
- typical error、bias、scale bins 和 city concentration 提供与 primary MSE 同方向的辅助证据；
- improvement 不是只依赖单一 scenario 或单一 scale bin。

### Reference / fallback

`B3_RECENT_3POINT_OLS_TREND`

状态：

**frozen_reference_and_fallback**

B3 仍是最重要的简单可复现 benchmark；S6 reproduction 必须能够同时重现它，以校验 final pipeline 没有破坏历史合同。

### Sensitivity comparator

`RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND`

状态：

**registered_sensitivity_comparator_only**

它在 registered 上有小幅稳定信号，但 resident 未达到 S5 strong gate，不作为 primary final model。

### Important boundary

不存在“resident / registered 两个 target 上都统一胜出”的 universal winner。S6 冻结的是当前项目本地 contract 下的 **resident-primary final candidate**，不是已经取得官方 target mapping 证明的全局竞赛真值最优模型。

## Partial / Fragile

- F4 的 resident origin wins 不是压倒性一致，R1 为 `5/11`、R2 为 `7/11`；其优势主要由 aggregate + 多项辅助指标共同支持。
- F4 的 tail P95 不是全面优于 B3，最终方案解释必须保留 tail-risk caveat。
- F4 在 registered 上明显劣于 B3/RLT3；如果未来官方 target mapping 证明目标是 registered，当前 final candidate 必须重新选择。
- F4 的 framing signal 已有强 evidence，但学习者对完整 final-training / current-2023 information-set reconstruction / submission generation 的独立复现能力尚未验证；这正是 S6 的任务。
- S5 的 partial-pooling / EBM 等方向没有形成可采信正向结论；不应在 S6 再开启模型研究。

## Official Classic-Mode Clarification

- 当前用户参加的是 DataFountain **经典赛（classic competition）**，不是 2023 年已结束的正式初赛/决赛。
- DataFountain 经典赛规则明确：当前提交会进行自动评测，并显示在 **经典赛排行榜**；评测采用“原赛题初赛 A 榜评测规则”。因此当前提交不写入历史“初赛 A 榜”或“初赛 B 榜”，也不属于原决赛排名。
- 官方 dataset 页面仍保留历史资产：原“数据文件夹”、单独的“B榜数据”、以及“提交样例”。用户上传的 `B-data.zip` MD5 为 `6DC5FD08E6C8B13271521BB23122EB2C`，与官方 B 榜数据 MD5 完全一致；`submission_sample.csv` MD5 为 `882A25A3037ABC8A5531CD0991EDBFD9`，与官方提交样例完全一致。
- 对上传资产的直接检查确认：提交样例为 40 行 `city1..city40`、`year=2023`、列为 `city_id, year, pred`。
- B 榜数据不是当前 classic submission 的默认建模输入：它是原正式比赛 B 榜阶段的独立历史资产。其 40 个匿名 city 标签虽然仍为 `city1..city40`，但人口历史与原数据文件夹大量不同，且人口规模可覆盖到 2022，说明不能把两套同名 city ID 当作同一实体直接拼接。
- 因此当前 S6 应继续以原 A-stage “数据文件夹”对应的项目 canonical data 为建模事实源，并使用官方 submission sample 冻结输出 schema；B 榜数据只作为历史赛制研究/迁移材料，除非 Main 另行开启明确的 B 榜复现实验。

## Remaining Uncertainty

- 官方“总人口”究竟对应 resident 还是 registered 仍没有直接证明。
- 真实 source publication timing 仍未知；历史 R1/R2 只是 replay assumptions。
- PDF formal MSE / FAQ RMSE 冲突仍未获得官方澄清；本地 contract 继续 MSE primary / RMSE auxiliary。
- F4 在真实 2023 当前 snapshot 上的最终 prediction 尚未生成，因此不存在 final submission evidence。
- 鉴于官方“总人口”到常住人口/户籍人口的精确映射仍未直接证明，S6 final pipeline 必须同时保留两套目标路径：resident primary 与 registered alternative。两者应分别训练、分别生成 2023 predictions、分别生成符合官方 schema 的 submission 候选；不得在没有新证据时平均或混合。首次真实平台提交后，可根据实际 score 再做 target-mapping 复盘。
- final training 时怎样把历史 replay geometry 映射到真实 2023 information set，必须在 S6 明确冻结，不能凭历史 backtest runner 隐式推断。
- submission schema / exact output coverage 仍需 S6 做最终事实检查。
- 经典赛 leaderboard score 与本地 MSE→score 数量级存在值得警惕的 sanity question，但当前不作为 S6 blocker。先按既定 S6 路线完成首次真实 submission；拿到平台实际评分后，再决定是否开启 metric / target-scale / leaderboard 专项复盘。

## S6 Entry Point

S6 中心问题：

> 怎样把已经选出的 resident-primary F4 candidate，从“validated experiment”冻结为一个 raw → canonical → feature → train → 2023 prediction → submission 的可重复 final solution，同时确保最终训练没有悄悄改变 S2–S5 的信息边界和模型定义？

S6 第一入口不是继续试模型，而是建立 **Final Solution Contract + Reproduction Contract**。

第一任务应完成：

1. 从当前 `main` 精确提取 F4 的真实实现、feature/replay/model dependencies 与必要 artifacts；
2. 冻结 final target/model/feature/current-2023 information-set/fallback/seed/environment/output-schema；
3. 明确 historical validation geometry 与真实 2023 final-fit geometry 的对应关系；
4. 明确 B3 reproduction control；
5. 定义 fresh rebuild：从 raw source 清理可重建派生产物后，如何重新得到 canonical、final feature/model 和最终 prediction；
6. 在 contract review 通过前，不生成正式 submission，也不再做模型选择实验。

## S6 Execution Strategy

S6 以 **serial reproducibility workflow** 为主，而不是 S5 的 parallel model screening：

`contract freeze → clean rebuild → historical reproduction controls → final fit/prediction → submission validation → fresh end-to-end rerun`

只有彼此完全独立的环境/schema checks 才适合并行；任何会改变 final contract 的步骤必须串行。

每个 S6 Plan 仍遵守项目 Plan 交付规则：

- 聊天先简短说明预计做什么 / 为什么 / 预计成果；
- Plan 直接生成可下载 Markdown 文件，不在聊天中预览全文；
- 文件名使用 `S6-PlanNN｜中文标题.md`；
- Plan 给 Codex 保留合理实现与验证空间，但不得重新开启 model selection/tuning。

## Next Entry Point

新建 `Lab｜S6 Final Solution & Reproduction`。

第一份 Plan 应围绕：

**Final Solution Contract Freeze**

目标不是预测，而是把 F4 从实验候选固化成唯一可复现的 final pipeline specification，并定义随后 clean-room reproduction 的验收标准。

## Do Not Repeat

- 不重新开启 S5 model sweep。
- 不重跑已关闭的 LightGBM / XGBoost / ExtraTrees / EBM / hierarchical / city-local family 只为了寻找更低分。
- 不调参、Optuna、stacking、routing 或 post-hoc city gating，除非 Main 因 contract-level contradiction 正式退回 S5。
- 不把 Plan06F recovery-only 或 Plan06G incomplete evidence 重新解释成 performance evidence。
- 不把 F4 的 resident improvement 说成 registered / universal improvement。
- 不在 Final Solution Contract 冻结前生成正式 2023 submission。
- 不依赖已存在的派生 CSV / cache 作为唯一复现路径；S6 必须证明 raw source 可以重新构建 final solution。
