# Current Learning Checkpoint｜城市人口预测竞赛实战

## Current Position

- Main 正式判定 **S1｜数据建模与审计 COMPLETE**。
- 当前进入 **S2｜Validation Architecture**。
- S1 工程 freeze 基线：GitHub `MyKr-YSteinsK/city-population-competition` 的 `main`，S1 freeze commit `2de1a80aba16875eaeaf07b0cbdca0689f9b2105`。
- 后续代码、数据、canonicalization 与 S1 工程事实以远端仓库当前 `main` 为准；历史聊天数字不覆盖仓库。

## Confirmed Mastery

- 能区分 raw / canonical / modeling dataset，理解 canonical 层不是训练样本。
- 能解释 grain、候选 key、key uniqueness、join cardinality，以及重复 key 为什么不能靠静默去重解决。
- 能区分内部缺失、时间覆盖边界、实体覆盖缺失，以及 0 可能是合法值或 sentinel。
- 能说明 quarantine、provenance、derived value 与 raw observation 的边界。
- 能用跨表一致性和局部证据判断修复规则，理解局部规律不能直接提升为全局规则。
- 理解 feature availability / information set 的基本含义：不同字段可用到的最新年份可以不同。

## Meaningful Evidence

- 仓库包含 7 个合法 XLSX + 参考 PDF 的只读 raw source；来源未解析的 `test.xls` 已进入 quarantine，不参与 canonical。
- `scripts/run_s1.py` 统一重建 7 个领域审计脚本，并断言 raw hash 与 canonical/quarantine/provenance/regime 输出重建前后保持一致。
- 当前 S1 contract 有 14 张逻辑 canonical 表，所有 canonical `(city, year)` key 无重复。
- quality registry 显式保留 unresolved truth、coverage gap、derived provenance、density regime 与 income shift 等边界，不通过静默填补制造确定性。
- 员工派生行保留独立 provenance；隔离值保持缺失，不回灌为伪装的原始事实。
- Main 已对照 Brief / Plan 与仓库 handoff 完成最终验收，S1 不再继续扩展。

## Partial / Fragile

- 尚未学习并独立设计可信的 out-of-time validation / backtesting。
- 各源的真实发布时间与 forecast-origin availability 仍未知；仓库明确标记为 unknown，应在 S2 作为 information-set 约束处理。
- 赛题 target 精确口径与 PDF 中 MSE / RMSE 表述冲突仍是赛题源材料层面的待决问题；正式 S2 实验前必须明确采用的本地 validation contract。

## Open Questions

- 真实 2023 预测时，每一类 source-level feature 到底允许使用到哪个年份？
- 历史 pseudo-test 应如何复刻真实 horizon 与各字段 freshness，而不是统一按 `T-1` 截断？
- 应使用单一 out-of-time holdout，还是多个 forecast origins 的 expanding / rolling backtest？
- 本地 evaluator 如何在 MSE / RMSE 源材料冲突下定义主指标与辅助指标？

## Active Anchor / Tracks

- Anchor：城市人口分析与预测原始赛题 + GitHub `city-population-competition`。
- Main：S2｜Validation Architecture。
- Lab：当前未激活 S2 实验。

## Stage Experiment Workflow

从 S2 起，阶段实验采用“独立实验对话 → 连续 Plan → Codex 执行 → 证据回传 → 实验对话复审 → 下一 Plan”的工作流。

- 每个 Stage 在需要正式实验时新建独立实验对话，例如 `Lab｜S2 实验`。
- 实验对话不直接替 Codex 完成实现；它负责理解阶段目标、审查当前仓库与上一轮证据、制定下一份 Plan，并在 Codex 回传后重新判断。
- Plan 是目标、边界、验收和必要上下文，不是逐行实现说明。必须给 Codex 保留局部方案设计、实现、诊断、验证和合理决策空间。
- Codex 的重要实验仍须围绕 `Hypothesis / Question → Experiment → Evidence → Interpretation / Decision`，并回传真实运行证据、偏差、失败和剩余不确定性。
- 每个 Codex Plan 的交付材料放入该 Stage 专属的独立 handoff 文件夹；用户将整个 handoff 文件夹内容提供给对应实验对话复审。不得仅凭 Codex 的文字“完成”判断实验结论。
- 实验对话按序生成下一份 Plan，直到该 Stage 的实验问题形成足够证据；实验对话本身不得自行推进 Mainline Stage。
- Stage 实验结束时，实验对话必须生成一份 Markdown 阶段报告。报告重点不是流水账，而是从真实参赛者视角记录：问题与约束、当时证据、关键假设、为什么选择某实验、实际验证结果、失败/反证、如何据此做决策、最终方案/结果、剩余边界，以及交给 Main 的阶段结论。
- Main 阅读阶段报告和必要原始证据后，负责正式 Stage completion 判定、能力判断、下一 Stage 教学和 checkpoint 合并。
- 已确认简单且低认知价值的重复性工程可交由 Codex 自主完成；关键 validation、leakage、feature/model 选择、实验解释仍要求人类拥有判断。

## Next Entry Point

S2 Main 先完成必要的 Validation Architecture 教学与实验问题定义；需要落地验证时再创建 `Lab｜S2 实验`。实验对话使用上述连续 Plan/Codex/handoff/review 工作流，而不是 Guided Implementation，也不让一次 Plan 无边界扫完整个后续 Stage。

## Do Not Repeat

- 不重新展开 S1 数据考古，除非 GitHub `main` 出现会改变数据契约的新证据。
- 不把历史聊天中的旧计数覆盖当前仓库事实。
- 不再要求用户对显然已掌握的问题做低价值解释性问答。
- 不再采用 S1 早期的逐步手把手 Guided Lab 作为默认实验方式。
- 不让单个 Plan 越过当前 Stage 或替代 Main 的阶段判定。
