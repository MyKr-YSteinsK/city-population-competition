# S5-Plan06A Handoff - LightGBM full multi-source screening

## Result

- Status: Complete
- Branch: s5/plan06a-lightgbm-screening
- Exact execution base: 1f97f67488cd7aec120ef8b9fa04407dc13d54ba
- Runner source commit for OOF: 1e61fe9af746fc17fdf4acde5072c58eb27255e6
- Family classification: family_competitive_but_unstable
- Best resident framing by pooled MSE: F2_B3_RESIDUAL

## Frozen data and nested selection

- Formal matrix: 1760 rows; replay matrix: 2320 rows; numeric features: 144.
- Outer origins: [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]; cities: 40.
- Eligible inner folds: 264 / 484; fallback outer fits: 8 per framing.
- Selection used only eligible inner replay folds; outer-origin labels were not used for configuration selection.
- Numeric NaN was kept raw for LightGBM native missing handling; city identity used fixed-vocabulary one-hot encoding.

## Pooled metrics versus B3

framing | target | scenario | pooled_mse | b3_mse | improvement_pct | origin_wins | latest_2021_mse
F1_DIRECT | resident | R1_REFERENCE_SNAPSHOT_LAG | 3675.171201 | 872.452652 | -321.246 | 1 | 2452.492794
F1_DIRECT | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 3612.083229 | 872.452652 | -314.015 | 1 | 2829.592947
F1_DIRECT | registered | R1_REFERENCE_SNAPSHOT_LAG | 739.368774 | 225.421212 | -227.994 | 0 | 1695.634682
F1_DIRECT | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 728.461345 | 225.421212 | -223.156 | 0 | 1645.342353
F2_B3_RESIDUAL | resident | R1_REFERENCE_SNAPSHOT_LAG | 1007.799663 | 872.452652 | -15.513 | 4 | 3139.932562
F2_B3_RESIDUAL | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 1022.184514 | 872.452652 | -17.162 | 4 | 3685.919565
F2_B3_RESIDUAL | registered | R1_REFERENCE_SNAPSHOT_LAG | 228.445205 | 225.421212 | -1.341 | 4 | 1528.280892
F2_B3_RESIDUAL | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 230.075037 | 225.421212 | -2.065 | 4 | 1511.330763

## Configuration and diagnostics

- Selected configurations: {"F1_DIRECT|LGB_C1_CONSERVATIVE":8,"F1_DIRECT|LGB_C3_FLEXIBLE_REGULARIZED":36,"F2_B3_RESIDUAL|LGB_C1_CONSERVATIVE":42,"F2_B3_RESIDUAL|LGB_C2_MEDIUM":2}.
- Fallback fits: {"F1_DIRECT":8,"F2_B3_RESIDUAL":8}.
- Top feature groups by normalized LightGBM gain: [{"feature_count":17,"feature_group":"FG-P","normalized_gain_mean":0.6747531641236009},{"feature_count":10,"feature_group":"FG-EMP","normalized_gain_mean":0.09681497239760196},{"feature_count":16,"feature_group":"FG-AGE","normalized_gain_mean":0.05328557396367615},{"feature_count":14,"feature_group":"FG-SECTOR","normalized_gain_mean":0.047092230696605196},{"feature_count":9,"feature_group":"FG-D","normalized_gain_mean":0.03731493766317695}].
- City error concentration is recorded in city_model_error.csv; origin-band feature importance is recorded in feature_importance.csv.

## Boundary and governance

- No tuning beyond the three pre-registered configurations.
- No Optuna, Bayesian optimization, random search, large grid search, routing, gating, ensemble or blend.
- No feature changes; no Plan04, Plan04B or Plan05 artifact changes; no requirements change.
- No 2023 prediction, no submission, S5 not declared complete, S6 not entered.

## Artifacts

- oof_predictions.csv
- aggregate_metrics.csv
- per_origin_metrics.csv
- configuration_selection.csv
- inner_selection_evidence.csv
- city_model_error.csv
- missingness_diagnostics.csv
- feature_importance.csv
- model_summary.json

## Next boundary

- Main/Lab should compare this family evidence with the other Plan06 workers under the frozen Wave contract. This worker does not select a global winner or start tuning.
