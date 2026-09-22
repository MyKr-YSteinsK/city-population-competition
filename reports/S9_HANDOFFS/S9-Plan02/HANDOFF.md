# TASK_RESULT — S9-Plan02

## Result
- Status: `PLAN02_EVIDENCE_COMPLETE_READY_FOR_LAB_REVIEW`
- Branch: `codex/s9-plan02-f4-learner-cross`
- Model-track parent SHA: `098a4e7e169d528c378803078e33ec5f1450fd03`
- Runner execution commit: `2281166d474fdf4903a05c796092738489d43126`
- Push state: `pushed`
- Runtime seconds: `3.298`

## Goal
Compare pre-registered learner families under the frozen F4 log-growth target, separating framing, learner, and native city-identity effects.

## Current Evidence
- B3 remains the frozen simple reference.
- RLT3 is the stable local control, not a mechanism-diverse ensemble member by default.
- ARIMA_011 is retained as the Plan01 diversity diagnostic only.

## Hypothesis / Question
Whether historical F4 gains persist across learner families and whether native city identity accounts for material CatBoost performance.

## MUST / Non-goals
- F4 only; 144 frozen numeric features; frozen Plan04B nested geometry; resident primary and registered sensitivity.
- No 2023 prediction, truth access, Plan03 combination, gate, routing, feature change, clipping, merge to main, or champion freeze.

## Acceptance
- Contract/geometry checks: `True`; historical CatBoost reproduction: `True`.
- Coverage: `True`; leakage: `True`; preprocessing leakage: `True`; metric recomputation: `True`.
- Blind boundary: `True`; no truth: `True`; no 2023 prediction: `True`.

## Required Evidence
- Actual fits: inner `3960`, outer `220`, total `4180`.
- Nested selections: `220` learner/target/scenario/origin selections; fallback selections `40`.

## Results — Resident Primary
| Pipeline | R1 pooled MSE | R2 pooled MSE | origin wins vs B3 R1/R2 | 2021 MSE R1/R2 | 2013 MSE R1/R2 | corr vs RLT3 R1/R2 | Plan02 gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `RIDGE_F4_NUMERIC_ONLY` | 1773.438847 | 2006.678820 | 2/2 | 2600.877499/3696.771070 | 1074.544192/1827.757701 | 0.6310/0.7135 | `NOT_PROMOTED` |
| `LIGHTGBM_F4_NUMERIC_ONLY` | 813.586429 | 897.800130 | 7/6 | 2795.582423/3777.441467 | 1206.579603/1110.204181 | 0.9307/0.9401 | `NOT_PROMOTED` |
| `XGBOOST_F4_NUMERIC_ONLY` | 804.063117 | 833.385219 | 7/8 | 2999.259895/3648.685700 | 1183.630467/914.191151 | 0.9341/0.9451 | `PROMOTE_COMPETITIVE` |
| `CATBOOST_F4_NUMERIC_ONLY` | 732.859080 | 811.757167 | 6/7 | 2918.270040/3598.825029 | 962.321814/926.496018 | 0.9490/0.9672 | `PROMOTE_COMPETITIVE` |
| `CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE` | 739.600665 | 802.445747 | 5/7 | 2956.289001/3595.001285 | 932.932348/960.784902 | 0.9516/0.9574 | `PROMOTE_COMPETITIVE` |
| `B3_RECENT_3POINT_OLS_TREND` | 872.452652 | 872.452652 | 0/0 | 3705.427778/3705.427778 | 922.348611/922.348611 | 0.9976/0.9976 | `REFERENCE_ONLY` |
| `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND` | 861.490909 | 861.490909 | 6/6 | 3666.800000/3666.800000 | 1149.750000/1149.750000 | 1.0000/1.0000 | `REFERENCE_ONLY` |
| `ARIMA_011_FIXED_THETA05` | 850.057600 | 850.057600 | 3/3 | 3362.767055/3362.767055 | 1154.360516/1154.360516 | 0.8811/0.8811 | `REFERENCE_ONLY` |

## Results — Registered Sensitivity

| Pipeline | R1 pooled MSE | R2 pooled MSE | origin wins vs B3 R1/R2 |
|---|---:|---:|---:|
| `RIDGE_F4_NUMERIC_ONLY` | 367.450297 | 352.128474 | 0/0 |
| `LIGHTGBM_F4_NUMERIC_ONLY` | 265.194626 | 294.629582 | 2/0 |
| `XGBOOST_F4_NUMERIC_ONLY` | 258.752589 | 270.287849 | 3/4 |
| `CATBOOST_F4_NUMERIC_ONLY` | 241.639492 | 247.093630 | 4/4 |
| `CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE` | 243.244531 | 246.889893 | 4/4 |
| `B3_RECENT_3POINT_OLS_TREND` | 225.421212 | 225.421212 | 0/0 |
| `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND` | 223.690909 | 223.690909 | 8/8 |
| `ARIMA_011_FIXED_THETA05` | 241.035627 | 241.035627 | 5/5 |

## Historical CatBoost Compatibility
- All archived F4 native-city prediction, aggregate, per-origin, and selected-configuration checks: `True`.
- Archived Plan06E identity: artifact tip `343970c...`; source commit `469787d...`; CatBoost native-city reproduction is kept separate from Layer A.

## Learner-vs-Framing Interpretation
1. F4 signal across learners: learner-agnostic support is present across multiple families
2. Native `city_id` contribution: native city identity effect is mixed or not uniformly beneficial.
3. Individually competitive candidates: `XGBOOST_F4_NUMERIC_ONLY, CATBOOST_F4_NUMERIC_ONLY`.
4. Diversity candidates: `none`.
5. Plan03 complementary evidence: `False`; Plan03 was not started.

## Failures / Deviations
- None. Fixed configurations, gates, and input boundaries were not changed after results were observed.

## Decision
`PLAN02_EVIDENCE_COMPLETE_READY_FOR_LAB_REVIEW`

## Remaining Uncertainty
- Worker branch: `codex/s9-plan02-f4-learner-cross`; base SHA: `098a4e7e169d528c378803078e33ec5f1450fd03`; runner execution commit: `2281166d474fdf4903a05c796092738489d43126`.
- Fixed parameter families were not expanded; no Optuna, early stopping, feature selection, new feature, routing, combination, or census correction was evaluated.
- No S8 truth access, no 2023 prediction, and no blind champion freeze.

## Artifacts
- `learner_definitions.csv`, `preprocessing_contract.csv`, `source_lineage.json`, `historical_catboost_compatibility.csv`
- `configuration_selection.csv`, `inner_selection_evidence.csv`, `oof_predictions.csv`, `aggregate_metrics.csv`, `per_origin_metrics.csv`, `city_error_metrics.csv`
- `error_correlation_matrix.csv`, `r1_r2_sensitivity.csv`, `2013_stress_slice.csv`, `latest_2021_metrics.csv`, `coverage_audit.csv`
- `leakage_audit.csv`, `preprocessing_leakage_audit.csv`, `finite_prediction_audit.csv`, `promotion_gate.csv`, `plan02_summary.json`, `HANDOFF.md`

## Suggested Next Boundary
Lab review only. Plan03 requires a separately generated plan and must not start automatically from this task.
