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
- Lab：当前未激活 S2 实验。只有 Main 明确实验目标与边界后，才新建 `Lab｜S2 实验`。

## Next Entry Point

先建立 S2 的 Validation Contract：明确 `forecast origin`、`forecast horizon`、`information set`、`pseudo-test target year` 四者的关系；再比较随机切分、单一时间外 holdout 与多时点 backtest 各自在回答什么问题。

## Do Not Repeat

- 不重新展开 S1 数据考古，除非 GitHub `main` 出现会改变数据契约的新证据。
- 不把历史聊天中的旧计数覆盖当前仓库事实。
- 不在 Main 尚未定义实验问题前让 Codex 自动完成完整 validation / modeling pipeline。
