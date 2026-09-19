# S4-Plan04B Handoff｜工资水平发展信号与 B3 残差修正快速筛选

## 状态与结论

- Plan 状态：`evidence complete`；本 worker 只完成 screening，不自动执行 deep dive。
- screening decision：`unsupported`；`deep_dive_recommended=false`。
- `s4_complete=false`、`s5_entered=false`、`final_2023_forecast_run=false`、`submission_run=false`。
- B3 仍是本轮 frozen comparison baseline；R1/R2 wage cap 完全等价，因此 M4B 只在 R1 实际 fit 一次，R2 复用等价性结论而不复制模型 workload。

## Actual changes

1. 新增 `scripts/build_s4_plan04b_wage_screen.py`。
2. 新增本目录下的 `HANDOFF.md`、`plan04b_summary.json`、`screening_metrics.csv`、`wage_boundary_audit.csv`。
3. 未修改 `main`、S2 validation contract、S3/S4 前序 handoff、canonical/raw data、三个并行 Wave A worker 文件或项目状态文档。

## Execution contract

- frozen base / execution starting SHA：`b8c302abe7ebf1710869aaa55a00d4b926d2bcac`。
- current evidence-generation HEAD：`b8c302abe7ebf1710869aaa55a00d4b926d2bcac`。
- branch：`s4/wave-a-wage`；本 Plan 在独立 worktree 执行；未 merge `main`。
- common origins：`2011–2021`；resident 为 primary，registered 为 sensitivity；population target history 为 `T-2`。

## Actual model and source geometry

- candidate：`M4B_B3_PLUS_RIDGE_WAGE`；唯一 feature group：`WAGE_RECENT_V1`。
- features：`log1p(latest positive averageWage)`、最近两个 positive observation 的 actual-year `log(latest/previous) / gap`、最近三个 positive observation 的 actual-year OLS `log(wage)` slope。
- preprocessing：每个 target/origin 独立使用 `SimpleImputer(median, add_indicator=True) + StandardScaler + Ridge(alpha=1.0)`；只在 outer-origin training rows fit。
- wage source：`data/normalized/工资水平_职工平均工资.csv`；source status=`source_plus_invalidation`；observed latest=`2021`；observed lag=`2`；invalidated key=`city31/2011`。
- S2 required lag：R1=`2`、R2=`2`；common-origin cap mismatch=`0`；`scenario_equivalent=true`。
- 未使用 density、urbanization、employment、city ID；没有 tuning、GBDT、lag/rolling/polynomial 扩展、2023 forecast 或 submission。

## Core screening metrics

数值为 common strict `40 cities × 11 origins` 的 pooled MSE / macro-origin MSE / latest 2021 MSE；relative improvement 为 `(B3 - M4B) / B3`，正值表示 candidate 更好。

| target | B3 pooled / macro / latest | M4B pooled / macro / latest | M4B origin wins/losses/ties |
|---|---:|---:|---:|
| resident | 872.452652 / 872.452652 / 3705.427778 | 943.919472 / 943.919472 / 3627.050361 | 5/6/0 |
| registered | 225.421212 / 225.421212 / 1485.604861 | 225.850578 / 225.850578 / 1499.100610 | 3/8/0 |

## Handoff questions

1. **wage R1/R2 是否确实等价？** 是。S2 required lag 为 R1=`2`、R2=`2`，2011–2021 每个 origin 的 source cap 也完全一致；因此没有重复 fit R2。
2. **M4B resident 是否超过 B3？** 否。 B3 pooled MSE=`872.452652`，M4B=`943.919472`，relative improvement=`-8.191484%`。
3. **registered 是否支持？** 不支持。 B3 pooled MSE=`225.421212`，M4B=`225.850578`，relative improvement=`-0.190473%`。
4. **per-origin wins/losses？** resident=`5/6/0`，registered=`3/8/0`，顺序为 wins/losses/ties。
5. **latest 2021 是否支持或冲突？** resident M4B latest=`3627.050361` vs B3=`3705.427778`，relative improvement=`2.115206%`；registered M4B latest=`1499.100610` vs B3=`1485.604861`，relative improvement=`-0.908435%`。latest 只作 proximity diagnostic，不能覆盖 pooled 结论。
6. **invalidated value 是否零回流？** 是。canonical invalidated key 的 feature-used value reintroduction count=`0`，used-key count=`0`；candidate-hit 只记录 allowed-history 交集，不把 raw 异常值带回模型。guard=`PASS`。
7. **是否值得 deep dive？** `deep_dive_recommended=false`。判定依据：resident pooled hurdle、origin majority 与 latest proximity；本 worker 不自动生成 leave-one-origin/city 或 concentration artifacts。

## Verification and evidence index

- B3 row compatibility：`PASS`；B3 pooled/macro/latest metric compatibility：`PASS`。
- wage boundary / invalidation / T-2 / preprocessing guards：`PASS`。
- `screening_metrics.csv`：aggregate、per-origin、pairwise、R1/R2 equivalence 与 focused sanity checks。
- `wage_boundary_audit.csv`：required cap、actual latest year、staleness、feature source years、invalidated candidate-hit 与 zero-reintroduction boundary。
- `plan04b_summary.json`：机器可读 screening decision、metrics、scope flags 与 evidence counts。

## Failure / deviation / remaining uncertainty

- Plan04B 的 R1/R2 wage cap 等价性使单次 M4B fit 足够；这是计划明确授权的 workload reduction，不是遗漏 R2。
- source publication timing 未被证明；R1/R2 仍是 S2 observed-cap replay assumptions，不应解释为真实发布时间因果。
- wage history 的 missingness / invalidation 保持显式；缺失只由 outer-training fold 的 median + missing indicator 处理，未插值、future fill 或回填 raw quarantine value。
- deep dive 未执行；不据此宣布 S4 complete 或进入 S5。
