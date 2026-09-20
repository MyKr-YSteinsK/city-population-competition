# S5-Plan06G Handoff — Hierarchical / Partial Pooling 城市异质性 screening

## Result

- Plan ID：S5-Plan06G
- Stage：S5_EVIDENCE_DRIVEN_ITERATION
- Worker branch：s5/plan06g-hierarchical-partial-pooling
- Exact base SHA：1f97f67488cd7aec120ef8b9fa04407dc13d54ba
- Worker status：complete_with_incomplete_coverage
- Runner commit used for execution：56b5a6d45ed3d8f22e23d7270205000c01b8cca7
- Official OOF coverage：combined 2760 / 3520 rows — FAIL；per framing {"F1_DIRECT": {"candidate_oof_rows": 1760, "required_oof_rows": 1760, "status": "PASS"}, "F2_B3_RESIDUAL": {"candidate_oof_rows": 1000, "required_oof_rows": 1760, "status": "FAIL"}}
- Classification：coverage_incomplete

## Dependency

- `statsmodels==0.15.0` install/import/pip-check：PASS / PASS / PASS
- Python：3.14.6；actual statsmodels：0.15.0
- `requirements.txt` 未修改；未替换为 PyMC 或其他 Bayesian stack。

## Method and boundary

- F1_DIRECT 使用 `actual_target_value`；F2_B3_RESIDUAL 使用 `b3_residual_actual_minus_prediction`，最终预测为 B3 + residual prediction。
- 每个 inner/outer training fold 独立执行 median imputation + missing indicators + standardization + PCA；PCA 不进入 `city_id`，随机效应分组为 raw `city_id`。
- C1/C2 是 city random intercept；C3 是 city random intercept + standardized `pop_recent3_ols_slope` random slope；REML=false。
- 未见过的验证城市使用随机效应 0；未使用 held-out target 估计验证城市随机效应。

## Coverage and convergence

- Outer fit groups：88 selected framing groups；model failures：19。
- Fit counters：{"fit_failed_units": 361, "fit_success_units": 1239, "model_fit_units": 1600, "outer_fit_failure_count": 19, "primary_attempts": 1600, "primary_exception_units": 20, "primary_nonconverged_units": 341}
- Selected config counts：{"HIER_C1_RI_PCA8": 72, "HIER_C2_RI_PCA16": 3, "HIER_C3_RI_RS_PCA12": 13}
- Unseen-city counts：{"by_target_scenario": {"registered|R1_REFERENCE_SNAPSHOT_LAG": {"max_unseen_city_count": 13, "mean_unseen_city_count": 1.1818181818181819, "outer_fit_count": 22, "total_unseen_city_counts": 26}, "registered|R2_CONSERVATIVE_PUBLICATION_STRESS": {"max_unseen_city_count": 13, "mean_unseen_city_count": 1.1818181818181819, "outer_fit_count": 22, "total_unseen_city_counts": 26}, "resident|R1_REFERENCE_SNAPSHOT_LAG": {"max_unseen_city_count": 2, "mean_unseen_city_count": 0.18181818181818182, "outer_fit_count": 22, "total_unseen_city_counts": 4}, "resident|R2_CONSERVATIVE_PUBLICATION_STRESS": {"max_unseen_city_count": 2, "mean_unseen_city_count": 0.18181818181818182, "outer_fit_count": 22, "total_unseen_city_counts": 4}}, "total_unseen_city_counts": 60}

## Pooled metrics

| framing | target | scenario | pooled MSE | B3 MSE | improvement % | origin wins | latest-2021 MSE | top-1 city SSE share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| F1_DIRECT | resident | R1 | 3112.837948 | 872.452652 | -256.792 | 0 | 5301.697333 | 0.213 |
| F1_DIRECT | resident | R2 | 3501.301020 | 872.452652 | -301.317 | 0 | 5884.348888 | 0.178 |
| F1_DIRECT | registered | R1 | 1937.609461 | 225.421212 | -759.551 | 0 | 1805.029491 | 0.323 |
| F1_DIRECT | registered | R2 | 1877.520659 | 225.421212 | -732.894 | 0 | 1895.932385 | 0.330 |
| F2_B3_RESIDUAL | resident | R1 | 1450.465322 | 872.452652 | -66.251 | 3 | 3557.378852 | 0.296 |
| F2_B3_RESIDUAL | resident | R2 | 1562.211338 | 872.452652 | -79.060 | 3 | 3687.748051 | 0.271 |
| F2_B3_RESIDUAL | registered | R1 | 25.132387 | 225.421212 | 88.851 | 0 | NA | 0.123 |
| F2_B3_RESIDUAL | registered | R2 | 375.547045 | 225.421212 | -66.598 | 2 | 1636.369491 | 0.402 |

Resident/registered R1/R2 的 macro-origin MSE、RMSE、MAE、bias、per-origin、city-level SSE/MSE 与 top-3 集中度均在 CSV artifacts 中提供。

## F1/F2 comparison

- resident/R1：F2_B3_RESIDUAL pooled MSE 更低 (F1=3112.837948, F2=1450.465322)。
- resident/R2：F2_B3_RESIDUAL pooled MSE 更低 (F1=3501.301020, F2=1562.211338)。
- registered/R1：F2_B3_RESIDUAL pooled MSE 更低 (F1=1937.609461, F2=25.132387)。
- registered/R2：F2_B3_RESIDUAL pooled MSE 更低 (F1=1877.520659, F2=375.547045)。

## Variance and pooling interpretation

- Variance summary：{"random_intercept_variance_mean": 17283.02309690249, "random_intercept_variance_median": 23485.97890009763, "random_slope_variance_mean": 34.75503617118498, "random_slope_variance_meaningfully_nonzero_fraction": 1.0, "random_slope_variance_median": 40.27706168410581, "random_slope_variance_nonnull_count": 13, "residual_variance_mean": 269.41957698613686, "successful_outer_fit_count": 69}
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: fixed-effect residual city-mean variance averaged 31433.2 versus raw between-city variance 44421.6.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (132.703 vs 159.251); this is a diagnostic, not routing evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.091; stability must be judged with the variance and convergence evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 8.40371 versus raw between-city variance 8.53312.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (5.38035e-05 vs 0.00029914); this is a diagnostic, not routing evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: fixed-effect residual city-mean variance averaged 30803.3 versus raw between-city variance 44421.6.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (132.006 vs 156.16); this is a diagnostic, not routing evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.727; stability must be judged with the variance and convergence evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 12.9387 versus raw between-city variance 5.69807.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (1.12082 vs 2.19524); this is a diagnostic, not routing evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.571; stability must be judged with the variance and convergence evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: fixed-effect residual city-mean variance averaged 24632.5 versus raw between-city variance 63876.8.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (119.115 vs 148.864); this is a diagnostic, not routing evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 64.5709 versus raw between-city variance 85.3513.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (0.677346 vs 1.29815); this is a diagnostic, not routing evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: fixed-effect residual city-mean variance averaged 24631 versus raw between-city variance 63876.8.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (118.799 vs 150.418); this is a diagnostic, not routing evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 70.3961 versus raw between-city variance 87.789.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (0.740011 vs 1.63738); this is a diagnostic, not routing evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.

## Boundary and deviations

- 未修改 Plan04 的 144 个特征、replay rows、validation geometry、Plan04/04B/05 artifacts 或 `CURRENT_LEARNING_CHECKPOINT.md`。
- 未执行 city routing、post-hoc selector、ensemble/stacking/gating、2023 prediction/submission；未声明 S5 complete，未进入 S6。
- pooling strength 仅为诊断，不转化为城市路由或后验配置选择。

## Artifacts

- `HANDOFF.md`
- `worker_environment.json`
- `oof_predictions.csv`
- `aggregate_metrics.csv`
- `per_origin_metrics.csv`
- `configuration_selection.csv`
- `inner_selection_evidence.csv`
- `city_model_error.csv`
- `missingness_diagnostics.csv`
- `random_effects_diagnostics.csv`
- `variance_components.csv`
- `pooling_strength_summary.csv`
- `model_summary.json`
