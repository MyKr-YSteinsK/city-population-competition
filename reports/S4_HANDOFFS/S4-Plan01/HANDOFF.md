# S4-Plan01 Handoff — 近期人口动态特征与 Ridge 趋势修正实验

## 结论先行

S4-Plan01 的 evidence 已完成，但本实验没有产生可替代冻结 B3 的 material ML improvement：

- resident primary pooled city-year MSE：B3 `872.452651515143`，M1A `910.349894805637`，M1B `938.694952017902`。
- registered sensitivity pooled city-year MSE：B3 `225.421212121213`，M1A `231.360015887421`，M1B `238.433125004615`。
- M1A 在 resident 的 2021 latest origin 上优于 B3（`3413.978047342008` vs `3705.42777777773`），但历史 aggregate 变差；registered sensitivity 则 aggregate 和 latest 都变差。
- M1B 相对 M1A 的 shape ablation 在 resident、registered 都为负增量，且 11 个 origin 中均为 `4` 胜、`7` 负；不能声称 shape extension 有独立且稳定的价值。
- 因此 `s4_complete=false`，且未进入 S5；本交接只完成 Plan01，不授权 2023 forecast、submission 或下一 Plan 的自动实施。

## 1. Actual changes

本次在执行时已处于最新 `origin/main`：Plan01 开始前记录的 execution baseline 为
`50f22159e1f38332967dd66bd9ddfb9a369fc21c`。在该 baseline 上实际变更为：

1. 新增 [build_s4_plan01_ridge_delta_dynamics.py](/D:/CS/MLDM/city-population-competition/scripts/build_s4_plan01_ridge_delta_dynamics.py)：实现 target-history-only 的 pseudo-origin feature builder、outer-origin Ridge pipeline、B3 兼容复现、指标/稳定性/集中度/泄漏证据生成。
2. `requirements.txt` 增加 `scikit-learn`，用于冻结的 `SimpleImputer(median, add_indicator=True)`、`StandardScaler`、`Ridge(alpha=1.0)`。
3. 新增本目录下的机器可读 artifacts 与本 HANDOFF；其中 `feature_missingness_imputation.csv` 按 feature/target/scenario/origin/model 显式记录 missing 与 imputation counts。
4. 更新 [PROJECT_CONTEXT.md](/D:/CS/MLDM/city-population-competition/docs/PROJECT_CONTEXT.md)、[PROJECT_FILE_MAP.md](/D:/CS/MLDM/city-population-competition/docs/PROJECT_FILE_MAP.md)、[CODEX_INSTRUCTIONS.md](/D:/CS/MLDM/city-population-competition/docs/CODEX_INSTRUCTIONS.md)，登记 S4 Plan01 的执行边界和证据位置。

`CURRENT_LEARNING_CHECKPOINT.md` 按 Plan01 要求保持只读；没有把 S4 Plan01 误写成整个 S4 完成，也没有进入 S5。

## 2. Actual runs and modeling geometry

运行命令为：

```text
python -m py_compile scripts/build_s4_plan01_ridge_delta_dynamics.py
python scripts/build_s4_plan01_ridge_delta_dynamics.py
```

冻结的比较范围仍为 `2011–2021` 共 11 个 origins、40 个城市。每个 target/scenario/model 的 validation 都覆盖 `40 × 11` 个 scoreable city-year rows；总预测行数为 `5280`，其中正式 ML 行为 `3520`。R1 与 R2 均保留，即使本 Plan 没有 covariates。

第一套 feature builder 的边界是：

- outer validation origin `T`：可用 target history 的最大年份固定为 `T-2`；
- training pseudo-row `(city, y)`：label 年份满足 `y <= T-2`，该行所有 target-history feature 只从 `year <= y-2` 构造；
- feature row 以自己的 pseudo-target year 为边界，不使用预先在完整时间轴上计算的未来值；
- 最近三点 OLS 使用实际年份差，不假设年份连续；最近历史不足三点的 training candidate 被记录为 `insufficient_recent3_history` 并排除，未静默填 0；
- 预处理仅在每个 outer-origin 的训练行上 fit，缺失值使用训练 fold median 加 missing indicator，不把缺失映射为零。

两个正式候选为：

| model | feature group | features | target |
| --- | --- | --- | --- |
| `M1A_RIDGE_DELTA_TH_CORE` | `TH_CORE` | `latest_value`, `recent3_ols_projected_change`, `latest_gap_years`, `recent3_span_years` | `future_delta = target_at_y - latest_allowed_value_before_y` |
| `M1B_RIDGE_DELTA_TH_SHAPE` | `TH_SHAPE` | TH_CORE 四项，加 `slope_disagreement_2v3`, `recent_relative_slope` | 同上 |

模型按 target、scenario、outer origin 分别拟合；不使用 city ID、city one-hot、static routing、covariates、tree/GBDT、tuning、ensemble 或 arbitrary feature dump。

## 3. B3 compatibility and primary metrics

B3 兼容性通过：

- `b3_compatibility_check.csv`：逐 city-origin prediction/residual/squared error 与 S3 Plan03 frozen artifact 在容差内一致；
- `b3_metric_compatibility_check.csv`：pooled、macro、latest 指标与冻结参考一致；
- resident B3 primary pooled MSE 为 `872.4526515151429`，registered sensitivity 为 `225.4212121212132`。

主结果如下（R1；R2 数值完全相同）：

| target | predictor | pooled city-year MSE | macro-origin MSE | latest 2021 MSE | vs B3 pooled |
| --- | --- | ---: | ---: | ---: | ---: |
| resident | B3 | 872.452652 | 872.452652 | 3705.427778 | reference |
| resident | M1A | 910.349895 | 910.349895 | 3413.978047 | -4.343759% |
| resident | M1B | 938.694952 | 938.694952 | 3429.895874 | -7.592653% |
| registered | B3 | 225.421212 | 225.421212 | 1485.604861 | reference |
| registered | M1A | 231.360016 | 231.360016 | 1507.968092 | -2.634536% |
| registered | M1B | 238.433125 | 238.433125 | 1518.535458 | -5.772266% |

这里的 `improvement_vs_B3_pct` 按 Plan 定义为 `(B3_MSE - candidate_MSE) / B3_MSE`；负值代表候选更差。由于每个 origin 都是 40 个城市，当前 pooled 与 macro 数值相同，但两类聚合仍分别落盘并单独审查。

## 4. Hypothesis support / falsification

### 问题 1：第一套 origin-specific feature builder 怎样保证每个 row 的 `T-2` / `y-2`？

由 `build_feature_row(series, row_year)` 先过滤 `year <= row_year - 2`，再计算 latest、最近三点 OLS、最近两点 slope、gap、span 和 shape features。外层训练只接收 `y <= T-2` 的 label，validation row 传入 `T`。`training_geometry.csv` 和 `leakage_audit.csv` 对每个 target/scenario/origin/model 逐行记录最大 history year，并全部 PASS。

### 问题 2：B3 是否在当前 main 被精确复现？

是。当前 main 的 B3 frozen row-level 与 metric-level 兼容性均 PASS，覆盖同一 `40 × 11` scope；详见 `b3_compatibility_check.csv` 与 `b3_metric_compatibility_check.csv`。

### 问题 3：M1A 是否超过 B3？

否。resident pooled MSE 为 `910.349895`，高于 B3 的 `872.452652`，相对 B3 为 `-4.343759%`。M1A 只在 resident latest 2021 上改善，不足以称为 aggregate 或 material ML improvement。registered 也未超过 B3。

### 问题 4：M1B 是否超过 B3？

否。resident pooled MSE 为 `938.694952`，相对 B3 为 `-7.592653%`；registered 为 `238.433125`，相对 B3 为 `-5.772266%`。latest 2021 虽略优于 B3（resident），但 aggregate 仍明显更差。

### 问题 5：shape extension 相对 core 的增量来自哪些 origin / cities，是否 broad-based？

不是 broad-based。M1B 相对 M1A 的 pooled origin 对比在 resident 和 registered 都是 4 胜、7 负：

- resident：M1B 胜于 M1A 的 origins 为 `2014, 2016, 2017, 2018`；M1A 胜于 M1B 的 origins 为 `2011, 2012, 2013, 2015, 2019, 2020, 2021`；
- registered：M1B 胜于 M1A 的 origins 为 `2014, 2015, 2016, 2017`；M1A 胜于 M1B 的 origins 为 `2011, 2012, 2013, 2018, 2019, 2020, 2021`。

city-origin 残差中，M1B 相对 M1A 的较大正向差异集中在 resident 的 `city10/city9`、registered 的 `city9/city40/city35`，而不是均匀覆盖城市。`ablation_comparison.csv`、`per_origin_metrics.csv`、`per_origin_predictions_or_residuals.csv` 保留完整可复核明细；最终 ablation 结论是 shape extension unsupported。

### 问题 6：resident aggregate 与 2021 latest 是否仍冲突？

是。M1A 的 resident latest 2021 MSE `3413.978047` 低于 B3 的 `3705.427778`，但 full-domain pooled MSE `910.349895` 高于 B3 的 `872.452652`；M1B 也同方向但 aggregate 更差。这是 latest-regime improvement 与 historical aggregate degradation 的明确冲突，不能只用 2021 latest 选择模型。

### 问题 7：registered sensitivity 是否支持同一机制，还是发生 reversal？

发生 reversal。registered 下 M1A/M1B 不仅 pooled aggregate 都差于 B3，2021 latest 也分别为 `1507.968092`、`1518.535458`，高于 B3 的 `1485.604861`。因此 resident latest 的局部改善不能解释为稳定的 target-history dynamics 机制。

### 问题 8：improvement 是否依赖少数 dominant cities？

候选并非完全由单个城市决定，但改善不够广泛，并且误差仍有集中：

- resident M1A/M1B 的 dominant city SSE share 为 `27.17%` / `27.90%`，top-3 share 为 `47.93%` / `48.41%`；相对 B3 改善的城市数仅 `11/40`、`13/40`，city-origin improved share 为 `37.05%`、`38.86%`；
- registered M1A/M1B 的 dominant city SSE share 为 `41.91%` / `40.17%`，top-3 share 为 `81.62%` / `79.63%`；改善城市数为 `17/40`、`16/40`，city-origin improved share 为 `41.82%`、`39.77%`。

所以不能写成 broad/stable gain；完整判断见 `city_improvement_concentration.csv` 与 `leave_one_city_sensitivity.csv`。

### 问题 9：leave-one-origin / leave-one-city 后结论是否稳定？

在当前 B3/M1A/M1B 比较集下，resident 的 full-domain leader B3 在 leave-one-origin 中保留 `9/11`；去掉 2013 时 M1B 成为 leader，去掉 2014 时 M1A 成为 leader，其余 origins B3 保持 leader。registered 的 full-domain B3 leader 在 `11/11` 个 leave-one-origin 中保持。leave-one-city 结果完整落盘，且没有通过城市重加权掩盖敏感性；候选相对 B3 的多数负差异并非只由一个城市删除后消失。

### 问题 10：R1/R2 是否按预期完全等价？

是。`r1_r2_equivalence_check.csv` 共 `66` 行（3 models × 2 targets × 11 origins），全部 PASS；`R2_CONSERVATIVE_PUBLICATION_STRESS` 与 R1 的 prediction/residual/score 数值一致。这是 target-history-only pipeline 的预期，不是额外实证发现。

### 问题 11：是否存在 leakage / coverage / missingness / preprocessing boundary deviation？

没有发现 deviation：

- `leakage_audit.csv` 共 `26460` 行，全部 PASS；
- `feature_missingness_imputation.csv` 按 feature/target/scenario/origin/model 逐项记录，所有 training/validation missing 与 imputation counts 均为 `0`；
- validation 每个模型/target/scenario/origin 都是 40 个城市，训练和 validation 最大 history year 均满足外层 cutoff；
- 正式 ML 训练覆盖 88 个 outer target×scenario×origin scopes；training feature missing cells 和 validation feature missing cells 均为 0；
- `SimpleImputer` 仅在 training fold fit，模型 fit row count 与 preprocessing fit row count 相等；
- 训练 candidate 中因不足最近三点历史排除的行被明确计数，未隐式填补；
- 未使用 covariates、city identity、future target、T-1/T、跨 target 信息。

### 问题 12：下一轮最高信息价值问题是什么？

Recommendation：不要继续堆叠相似的 target-history lag/shape features，也不要直接进入 S5。若 Lab 授权下一轮，应只预注册一个清晰的 source-specific covariate group，在同一 R1/R2、同一 `T-2` boundary、同一 B3 comparator 下，检验“外生结构信息是否能解释 M1A 不能稳定解决的城市异质性”。验收仍应以 raw pooled MSE、latest 2021、per-origin、leave-one-origin/city、concentration 和 registered sensitivity 的联合证据为准；本建议不自动实施下一 Plan。

## 5. Failures, local decisions, deviations, uncertainty

### Actual failures and repairs

- 首次完整运行的模型和指标已生成，但 summary 序列化处因字典 comprehension 变量作用域错误触发 `NameError`；修正 summary 后重新运行，最终 runner PASS。
- 实现复核时发现 outer-origin fit 曾嵌套在 city loop 中的重复风险，已在成功运行前收敛为“每个 target×scenario×origin×model 只 fit 一次、随后预测 40 城市”的结构；最终行数与覆盖审计通过。

这些是实现级修复，不改变冻结数据、目标、origin、指标或候选模型定义。

### Local technical decisions

- `recent_relative_slope` 的分母非法或缺失时显式记录 missing，由训练 fold imputer 处理；不以 0 代替。
- 最近三点不足的 training row 不强行扩展窗口，因为 Plan01 已预注册 recent-3 semantics；排除原因写入 diagnostics。
- `scikit-learn` 作为正式 runtime dependency 写入 `requirements.txt`；未使用 `--ignore-scripts` 等规避方式。

### Remaining uncertainty

该 Plan 只识别了 target-history-only Ridge 在当前 historical pseudo-origin geometry 下的能力边界，不能证明所有 covariates 无价值，也不能生成 2023 最终预测。registered 与 resident reversal、2021 latest 与 full aggregate 冲突、以及 city/error concentration 都应保留到后续 Lab 决策中。

## 6. Artifacts and delivery

本目录的 required artifacts 已全部生成：

`HANDOFF.md`、`feature_manifest.csv`、`feature_missingness_imputation.csv`、`modeling_dataset_diagnostics.csv`、`training_geometry.csv`、`aggregate_metrics.csv`、`per_origin_metrics.csv`、`per_origin_predictions_or_residuals.csv`、`pairwise_b3_comparison.csv`、`ablation_comparison.csv`、`latest_origin_comparison.csv`、`city_improvement_concentration.csv`、`leave_one_origin_out.csv`、`leave_one_city_sensitivity.csv`、`leakage_audit.csv`、`r1_r2_equivalence_check.csv`、`plan01_summary.json`。

兼容性辅助 artifacts 为 `b3_compatibility_check.csv` 和 `b3_metric_compatibility_check.csv`。

交付状态：

- `formal_ml_model_used=true`
- `covariates_used=false`
- `tuning_run=false`
- `ensemble_run=false`
- `final_2023_forecast_run=false`
- `submission_run=false`
- `s4_complete=false`
- `s5_entered=false`

主体 focused commit：`dd2734ee0519cc197f1bed89f0964492b0640930`，已 push 到 `origin/main`。本次回填只更新本 HANDOFF 的交付元数据，随后以 metadata-only commit 收口，并再次执行 JSON、artifact、upstream equality 和 clean worktree 验收。
