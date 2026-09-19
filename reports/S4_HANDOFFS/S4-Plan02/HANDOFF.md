# S4-Plan02 Handoff｜城镇化率外生信号与 B3 残差修正

## 状态与结论

- Plan 状态：`evidence complete`；本 Plan 不支持把城镇化率升级为可部署的增量信号。
- 当前 strongest fixed baseline 仍为 `B3_RECENT_3POINT_OLS_TREND`。
- 最终结论：`urbanization unsupported`。`D2` 全局残差偏差校正也没有超过 B3，因此不能把结果解释为“只是 global residual bias”；M2A level 与 M2B dynamics 均没有形成 aggregate、稳定、跨 scenario 的增益。
- `s4_complete=false`、`s5_entered=false`、`final_2023_forecast_run=false`、`submission_run=false`。

## Actual changes

1. 新增 `scripts/build_s4_plan02_urbanization_residual.py`，复用 S2 的 `required_lag_years` 与 S3 的 B3 `fit_prediction`，不重新硬编码 source cutoff 或重写 B3。
2. 新增本目录下的完整 Plan02 evidence：feature manifest、R1/R2 freshness、training geometry、missingness/imputation、prediction/residual、aggregate/per-origin/latest、ablation、concentration、leave-one-origin/city、coefficient/correlation、leakage 与 compatibility artifacts，以及 `plan02_summary.json`。
3. 更新 `docs/PROJECT_CONTEXT.md`、`docs/PROJECT_FILE_MAP.md`、`docs/CODEX_INSTRUCTIONS.md`，把当前上下文从 S4 Plan01 扩展为已完成 Plan01/02 evidence，但保留未进入 S5 和未执行最终预测的边界。
4. 未修改 raw XLSX、canonical data、S2 contract 或 `CURRENT_LEARNING_CHECKPOINT.md`。

## Execution contract

- 实际执行起始 SHA：`71f76664426a3b720f9b03f77792c8f0905e9f92`。
- 主体 evidence delivery commit：`1aed8c111fe356881cca64426070d27fd964daec`，已推送到 `origin/main`。
- 起始时 `main` 与 `origin/main` 对齐；工作树无用户本地修改。
- 已执行：

  ```text
  git fetch origin main
  python -m py_compile scripts/build_s4_plan02_urbanization_residual.py
  python scripts/build_s4_plan02_urbanization_residual.py
  ```

- runner 输出的关键审计均为 PASS：`b3_compatibility_pass`、`b3_metric_compatibility_pass`、`urbanization_year_audit_pass`、`population_t_minus_2_audit_pass`、`leakage_pass`、`full_coverage`。

## Important local technical decisions

- 城镇化率 canonical source 为 `data/normalized/城镇化率.csv`，source status 为 `source_preserved`；S2 reference geometry 的 observed latest year 为 2022、observed lag 为 1 年，2022 有 29 城市记录。publication timing 未被证明，R1/R2 只作为 frozen observed-cap replay assumption。
- R1 使用 S2 共享规则得到的 `required_lag_years=1`，row cap 为 `T-1`；R2 对 annual source 使用共享规则得到 `required_lag_years=2`，row cap 为 `T-2`。R1/R2 是独立 feature table 与独立 model run，不能只复制 scenario label。
- 人口侧仍遵守 B3 的 target-history boundary：训练 pseudo-target year `y` 只用 population history `<= y-2`，validation origin `T` 只用 `<= T-2`。
- M2A 只把 `urbanization_latest` 放入模型；`urbanization_latest_year` 与 staleness 只做 audit。M2B 在 M2A 基础上增加按实际年份差计算的 `urbanization_recent_change` 和 actual-year OLS `urbanization_recent3_slope`。
- 每个 target/scenario/origin/model 独立 fit `SimpleImputer(strategy="median", add_indicator=True)`、`StandardScaler` 与 `Ridge(alpha=1.0)`；median 只来自 outer-origin training rows。训练早期信息不足的缺失不填 0、不插值；validation 侧 40 城市均无 feature missing。
- D2 只做训练折内全局 B3 residual mean correction，是 bias-control diagnostic，不使用 urbanization feature，也不作为已授权的部署模型。
- 没有 city identity、其他 covariates、tuning、ensemble、future 2023 forecast 或 submission。

## Core aggregate metrics

这里的 pooled MSE 是 40 城市 × 11 origins 的 primary aggregate；latest 是 2021 origin。R1/R2 对 B3 与 D2 相同，因为它们不使用城镇化率，M2A/M2B 则分别独立重跑。

| target | scenario | B3 pooled / latest | D2 pooled / latest | M2A pooled / latest | M2B pooled / latest |
|---|---|---:|---:|---:|---:|
| resident | R1 | 872.452652 / 3705.427778 | 883.812539 / 3678.580511 | 903.774480 / 3569.820046 | 899.082499 / 3510.675476 |
| resident | R2 | 872.452652 / 3705.427778 | 883.812539 / 3678.580511 | 911.627743 / 3604.557881 | 925.764100 / 3623.037697 |
| registered | R1 | 225.421212 / 1485.604861 | 225.592430 / 1492.275461 | 228.373477 / 1505.252375 | 228.937284 / 1506.585970 |
| registered | R2 | 225.421212 / 1485.604861 | 225.592430 / 1492.275461 | 228.716508 / 1506.450615 | 230.017342 / 1506.707978 |

相对 B3 的 pooled MSE 变化：D2 为 resident `+1.3021%`、registered `+0.0760%`；M2A 为 resident R1/R2 `+3.5901%`/`+4.4902%`、registered `+1.3097%`/`+1.4618%`；M2B 为 resident `+3.0523%`/`+6.1105%`、registered `+1.5598%`/`+2.0389%`。因此局部 latest 改善不能抵消全域 historical aggregate 变差。

## Handoff questions

1. **当前 `main` 上 B3 是否再次精确复现？** 是。B3 row-level compatibility 与 S3 Plan03 frozen rows 通过，metric-level compatibility 也通过；resident pooled/macro `872.452651515143`、registered pooled/macro `225.421212121213`，R1/R2 均一致。

2. **R1 与 R2 实际分别给 urbanization 多新的信息？** R1 可用到 `T-1`，R2 收紧到 `T-2`；每个 origin 都有 40/40 latest availability，且每个 origin 都确认 R1/R2 feature 不同。以 2021 为例，35 个城市的 latest value、27 个城市的 recent change、26 个城市的 recent-3 slope 不同。该差异是 replay cap 的结果，不是已证明的 publication-date fact。

3. **40 city × 11 origins 的 urbanization validation coverage 是否完整？** 是。R1/R2、resident/registered、M2A/M2B 的每个 validation scope 均为 `40 × 11 = 440`，`full_coverage=true`。训练早期行存在可审计的 source absence/insufficient-history；这些行没有被删除后伪装成完整数据，而是按 outer-training median + missing indicator 处理；validation missing count 为 0。

4. **D2 仅做 global residual bias correction 后效果如何？** resident pooled MSE `872.452652 → 883.812539`，registered `225.421212 → 225.592430`，均变差；但 resident latest 从 `3705.427778` 降到 `3678.580511`，registered latest 反而升到 `1492.275461`。所以 D2 不是可部署增益，也不能支持“只缺一个 global bias correction”。

5. **M2A urbanization level 是否真实优于 B3 与 D2？** 否。M2A 在两个 target、两个 scenario 的 pooled MSE 都高于 B3，也高于 D2；R1 resident latest 虽降至 `3569.820046`，但这只是 latest 局部现象。

6. **M2B dynamics 是否真实优于 M2A？** 否。R1 resident M2B 比 M2A 小幅改善 `0.5192%`，但 R2 resident 变差 `1.5507%`；registered R1/R2 均变差 `0.2469%`/`0.5688%`，origin wins 也不占多数。

7. **resident primary 的 aggregate improvement 是否存在？** 不存在。所有 correction model 的 pooled/macro aggregate 都高于 B3；D2 是 correction 中最接近的 resident 模型，但仍差 `11.359888` MSE。

8. **registered sensitivity 是否支持相同结论？** 支持“不支持增量”的结论，而且更严格：D2 与 M2A/M2B 的 pooled MSE 都高于 B3，latest 也全部恶化。

9. **latest 2021 是否支持还是反对 historical aggregate 结论？** resident latest 支持局部修正：D2/M2A/M2B 都低于 B3；但 historical aggregate 反对，因此不能把 latest 局部改善当成稳定信号。registered latest 与 historical aggregate 一致，全部反对 correction。

10. **per-origin improvement 是否 broad-based？** 否。相对 B3 的 per-origin wins 虽然存在，但不是稳定多数：resident R1 为 D2/M2A/M2B `4/7/8` 个 origin，R2 为 `4/7/5`；registered R1 为 `5/4/3`，R2 为 `5/3/2`。pooled aggregate 没有因此改善。

11. **improvement 是否被少数 city SSE 主导？** 是。registered candidate 的 top-1 positive-delta city 贡献约 `81%–85%`，top-3 约 `93%–97%`；resident candidate top-1 约 `21%–28%`，top-3 约 `63%–70%`。candidate 的 top-three total SSE share 虽有时下降，但这没有转化成总 MSE 改善；移除最大 positive-improvement city 后，M2A/M2B 仍没有形成正的 aggregate improvement。

12. **leave-one-origin / leave-one-city 后结论是否稳定？** 基本稳定。B3 full-domain leader 在 resident 的 leave-one-origin 中存活 `10/11`，registered 存活 `9/11`；resident 仅在省略 2014 时出现候选短暂夺冠，registered 省略 2014 或 2021 时 D2 可局部夺冠。leave-one-city 中 M2A/M2B 在所有 removed-city 场景仍为负 improvement；registered D2 只有一个 removed-city 情况出现约 `+0.59` 的轻微局部 reversal，不足以改变结论。

13. **R1→R2 后收益是否显著衰减或 reversal？** 是，且只有城镇化率模型受影响。resident M2A pooled `+7.853262`、latest `+34.737836`，M2B pooled `+26.681601`、latest `+112.362221`；registered M2A pooled `+0.343031`、latest `+1.198239`，M2B pooled `+1.080058`、latest `+0.122008`。B3/D2 的 R1→R2 数值变化为 0。

14. **Ridge coefficient / residual correlation 是否有方向稳定的 mechanism evidence？** 没有联合稳定证据。resident 的 level coefficient 在 22 个 origin×scenario fit 中为正，但 registered level coefficient 只有 4 正、18 负；M2B dynamics 的 slope/change 在 target/scenario 间出现 mixed signs。训练 residual correlation 也随 origin 变化，validation post-hoc 方向不稳定。因此不能把正的 resident level coefficient 当成已验证机制。

15. **是否有 source missingness、future availability、quarantine、pipeline leakage deviation？** 未发现未记录的 deviation。source missingness 与 early-history insufficiency 已在 `feature_missingness_imputation.csv` 记录并仅在 training fold impute；validation 40 城市无缺失。future availability 的 publication timing 未证明，已明确标注为 observed-cap replay assumption；没有把 quarantine/source-derived 行并入模型。population `T-2`、urbanization cap、feature/target boundary 与全量 leakage audit 均 PASS。

16. **本 Plan 最终支持哪一种结论？** 支持 `urbanization unsupported`。不支持 `level-only signal`、`dynamics incremental signal`、`R1-only information-set-sensitive signal`，也不支持 `global residual bias rather than covariate signal`；后者被 D2 未超过 B3 的结果否定。

17. **下一轮最高信息价值问题是什么？** 先由 Main/Lab 评审是否值得测试另一个具有独立 source-quality provenance、较完整 40-city 历史覆盖且可复用 R1/R2 boundary 的预注册信号组，并优先验证其是否能在 B3 之上形成跨 origin/target/scenario 的 aggregate gain。不要因为本 Plan 的 latest 局部改善自动进入 S5，也不要自动执行下一 Plan。

## Evidence index

- `feature_manifest.csv`：`URB_LEVEL_V1`、`URB_DYNAMICS_V1` 与模型使用边界。
- `urbanization_information_boundary.csv`、`r1_r2_feature_freshness_comparison.csv`：R1/R2 source cap、lag、40-city availability 与实际 feature difference。
- `modeling_dataset_diagnostics.csv`、`training_geometry.csv`、`feature_missingness_imputation.csv`：outer-origin train/validation geometry 与 fold-local imputation。
- `aggregate_metrics.csv`、`per_origin_metrics.csv`、`per_origin_predictions_or_residuals.csv`：primary metrics 与可复核预测/残差。
- `pairwise_b3_comparison.csv`、`bias_control_comparison.csv`、`level_vs_dynamics_ablation.csv`、`latest_origin_comparison.csv`：B3/D2/M2A/M2B 比较。
- `city_improvement_concentration.csv`、`leave_one_origin_out.csv`、`leave_one_city_sensitivity.csv`：稳定性和集中度。
- `ridge_coefficient_diagnostics.csv`、`residual_feature_relationship.csv`：机制诊断，不作为因果证明。
- `leakage_audit.csv`、`b3_compatibility_check.csv`、`b3_metric_compatibility_check.csv`：边界与 baseline compatibility。
- `plan02_summary.json`：机器可读结论与 acceptance flags。

## Failure / deviation / risk notes

- 本轮没有 runtime failure；py_compile 与 focused runner 均通过。
- 主要风险不是代码错误，而是 source publication timing 未被历史材料证明；因此 R1/R2 不能被叙述为真实发布时间可用性，只能作为 frozen replay stress boundary。
- training missingness 是数据边界的真实表现，不是补造观测；任何后续模型若改变 imputation、lag 或 source contract，必须重新生成全部 evidence，不能复用本轮结论。
