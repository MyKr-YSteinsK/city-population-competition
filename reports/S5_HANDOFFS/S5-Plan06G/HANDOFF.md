# S5-Plan06G Handoff — Hierarchical / Partial Pooling 城市异质性 screening

## Result

- Plan ID：S5-Plan06G
- Stage：S5_EVIDENCE_DRIVEN_ITERATION
- Worker branch：s5/plan06g-hierarchical-partial-pooling
- Exact base SHA：1f97f67488cd7aec120ef8b9fa04407dc13d54ba
- Runner commit used for execution：62385e984b65cfc8ed5026ead57074d41e61d656
- Official OOF coverage：2920 / 3520 rows — FAIL
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

- Outer fit groups：88 selected framing groups；model failures：15。
- Fit counters：{"fallback_attempts": 663, "fallback_exception_units": 342, "fallback_nonconverged_units": 106, "fallback_success_units": 215, "fit_failed_units": 448, "fit_success_units": 1152, "model_fit_units": 1600, "outer_fit_failure_count": 15, "primary_attempts": 1600, "primary_nonconverged_units": 663}
- Selected config counts：{"HIER_C1_RI_PCA8": 58, "HIER_C2_RI_PCA16": 2, "HIER_C3_RI_RS_PCA12": 28}
- Unseen-city counts：{"by_target_scenario": {"registered|R1_REFERENCE_SNAPSHOT_LAG": {"max_unseen_city_count": 13, "mean_unseen_city_count": 1.1818181818181819, "outer_fit_count": 22, "total_unseen_city_counts": 26}, "registered|R2_CONSERVATIVE_PUBLICATION_STRESS": {"max_unseen_city_count": 13, "mean_unseen_city_count": 1.1818181818181819, "outer_fit_count": 22, "total_unseen_city_counts": 26}, "resident|R1_REFERENCE_SNAPSHOT_LAG": {"max_unseen_city_count": 2, "mean_unseen_city_count": 0.18181818181818182, "outer_fit_count": 22, "total_unseen_city_counts": 4}, "resident|R2_CONSERVATIVE_PUBLICATION_STRESS": {"max_unseen_city_count": 2, "mean_unseen_city_count": 0.18181818181818182, "outer_fit_count": 22, "total_unseen_city_counts": 4}}, "total_unseen_city_counts": 60}

## Pooled metrics

| framing | target | scenario | pooled MSE | B3 MSE | improvement % | origin wins | latest-2021 MSE | top-1 city SSE share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| F1_DIRECT | resident | R1 | 3112.837948 | 872.452652 | -256.792 | 0 | 5301.697333 | 0.213 |
| F1_DIRECT | resident | R2 | 3501.301020 | 872.452652 | -301.317 | 0 | 5884.348888 | 0.178 |
| F1_DIRECT | registered | R1 | 1938.059300 | 225.421212 | -759.750 | 0 | 1805.029491 | 0.328 |
| F1_DIRECT | registered | R2 | 1978.288064 | 225.421212 | -777.596 | 0 | 1895.932385 | 0.317 |
| F2_B3_RESIDUAL | resident | R1 | 431466.454036 | 872.452652 | -49354.426 | 0 | 567213.067960 | 0.131 |
| F2_B3_RESIDUAL | resident | R2 | 436919.461241 | 872.452652 | -49979.447 | 0 | 564691.258178 | 0.131 |
| F2_B3_RESIDUAL | registered | R1 | 314475.828801 | 225.421212 | -139405.873 | 0 | 335386.008733 | 0.083 |
| F2_B3_RESIDUAL | registered | R2 | 313177.895596 | 225.421212 | -138830.091 | 0 | 333204.188311 | 0.083 |

Resident/registered R1/R2 的 macro-origin MSE、RMSE、MAE、bias、per-origin、city-level SSE/MSE 与 top-3 集中度均在 CSV artifacts 中提供。

## F1/F2 comparison

- resident/R1：F1_DIRECT pooled MSE 更低 (F1=3112.837948, F2=431466.454036)。
- resident/R2：F1_DIRECT pooled MSE 更低 (F1=3501.301020, F2=436919.461241)。
- registered/R1：F1_DIRECT pooled MSE 更低 (F1=1938.059300, F2=314475.828801)。
- registered/R2：F1_DIRECT pooled MSE 更低 (F1=1978.288064, F2=313177.895596)。

## Variance and pooling interpretation

- Variance summary：{"random_intercept_variance_mean": 16251.960611293714, "random_intercept_variance_median": 23167.91663794281, "random_slope_variance_mean": 46.70921735118103, "random_slope_variance_meaningfully_nonzero_fraction": 1.0, "random_slope_variance_median": 48.108464865988935, "random_slope_variance_nonnull_count": 28, "residual_variance_mean": 226.12668436776818, "successful_outer_fit_count": 73}
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: fixed-effect residual city-mean variance averaged 30079 versus raw between-city variance 44421.6.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (129.846 vs 156.724); this is a diagnostic, not routing evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.636; stability must be judged with the variance and convergence evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 16.9096 versus raw between-city variance 3.88968.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (1.89871 vs 4.16181); this is a diagnostic, not routing evidence.
- registered/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.889; stability must be judged with the variance and convergence evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: fixed-effect residual city-mean variance averaged 31213.3 versus raw between-city variance 44421.6.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (133.581 vs 156.273); this is a diagnostic, not routing evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.364; stability must be judged with the variance and convergence evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 14.5373 versus raw between-city variance 4.43119.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (1.98091 vs 4.12332); this is a diagnostic, not routing evidence.
- registered/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 1.000; stability must be judged with the variance and convergence evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: fixed-effect residual city-mean variance averaged 24632.5 versus raw between-city variance 63876.8.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (119.115 vs 148.864); this is a diagnostic, not routing evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 63.8473 versus raw between-city variance 94.9924.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (0.948714 vs 1.64892); this is a diagnostic, not routing evidence.
- resident/R1_REFERENCE_SNAPSHOT_LAG/F2_B3_RESIDUAL: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: fixed-effect residual city-mean variance averaged 24631 versus raw between-city variance 63876.8.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: low-history mean absolute random intercept was smaller than high-history (118.799 vs 150.418); this is a diagnostic, not routing evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F1_DIRECT: selected-fit C3/non-C3 slope-nonzero fraction was 0.000; stability must be judged with the variance and convergence evidence.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: fixed-effect residual city-mean variance averaged 74.2541 versus raw between-city variance 103.635.
- resident/R2_CONSERVATIVE_PUBLICATION_STRESS/F2_B3_RESIDUAL: low-history mean absolute random intercept was smaller than high-history (0.64807 vs 1.04752); this is a diagnostic, not routing evidence.
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
