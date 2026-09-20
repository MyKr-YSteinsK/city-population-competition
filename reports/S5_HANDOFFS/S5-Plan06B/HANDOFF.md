# S5-Plan06B Handoff — CatBoost 城市感知全量多表强模型 screening

## Result

- Status: `COMPLETE` (family screening only)
- Classification: `family_competitive_but_unstable`
- Worker branch: `s5/plan06b-catboost-screening`
- Wave base SHA: `1f97f67488cd7aec120ef8b9fa04407dc13d54ba`
- Artifact-generation source commit: `1f97f67488cd7aec120ef8b9fa04407dc13d54ba`
- Final worker tip is returned in the TASK_RESULT and was not used as an experiment baseline.
- S5 is not declared complete; S6 is not entered.

## Frozen contract evidence

- Formal matrix: 1760 rows; replay matrix: 2320 rows.
- Model features: 144 numeric features plus raw `city_id` categorical identity.
- Outer origins: `[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]`; legal training cutoff is `outer_origin_year - 2`.
- Inner folds: 264 eligible / 484 total; selection never uses outer labels.
- Both framings share the same replay rows, folds, and outer validation geometry.

## Method and compatibility

- `CatBoostRegressor` with the three pre-registered configurations only; fixed seed `20260919`, CPU `thread_count=1`, no early stopping.
- Numeric missing values remain raw `NaN`; no imputer or scaler. `city_id` remains a raw string native categorical feature.
- CatBoost CPU accepted all listed parameters, including `rsm`; no parameter substitution was made.
- No city routing/gating, ensemble/blend, target encoding, tuning beyond the frozen three-config inner selection, 2023 prediction, or submission was run.

## Aggregate metrics

| target | scenario | framing | model MSE | B3 MSE | improvement | origin wins | latest 2021 MSE | top1 city SSE | top3 city SSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| registered | R1_REFERENCE_SNAPSHOT_LAG | F1_DIRECT | 1219.774371 | 225.421212 | -441.109% | 0 | 2088.890277 | 35.22% | 55.14% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F2_B3_RESIDUAL | 227.346773 | 225.421212 | -0.854% | 7 | 1535.784266 | 41.67% | 77.67% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F1_DIRECT | 1266.359344 | 225.421212 | -461.775% | 0 | 2020.922922 | 39.54% | 58.48% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F2_B3_RESIDUAL | 227.125265 | 225.421212 | -0.756% | 7 | 1508.447867 | 41.70% | 76.96% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F1_DIRECT | 4165.627483 | 872.452652 | -377.462% | 0 | 4568.949442 | 55.41% | 75.91% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F2_B3_RESIDUAL | 973.309135 | 872.452652 | -11.560% | 4 | 3401.460584 | 27.65% | 50.86% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F1_DIRECT | 4254.604034 | 872.452652 | -387.660% | 0 | 4540.145142 | 55.21% | 76.09% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F2_B3_RESIDUAL | 992.191731 | 872.452652 | -13.724% | 3 | 3378.640515 | 28.37% | 50.71% |

## Configuration and capacity diagnostics

- Selected configuration counts:
  - `F1_DIRECT` / `CAT_C1_CONSERVATIVE`: 18 outer fits
  - `F1_DIRECT` / `CAT_C2_MEDIUM`: 26 outer fits
  - `F2_B3_RESIDUAL` / `CAT_C1_CONSERVATIVE`: 23 outer fits
  - `F2_B3_RESIDUAL` / `CAT_C2_MEDIUM`: 5 outer fits
  - `F2_B3_RESIDUAL` / `CAT_C3_FLEXIBLE_REGULARIZED`: 16 outer fits
- Fallback fits: 16 across both framings (8 per framing).
- Later-origin capacity distribution is retained in `configuration_selection.csv`; no post-score configuration was added.

## City identity and feature-group diagnostics

- Overall native `city_id` importance: mean 0.165214 (0.165% of mean CatBoost importance across selected outer fits).
- Top feature groups by mean native importance:
  - `FG-P`: 3.642047 (50.69%)
  - `FG-EMP`: 1.063073 (14.80%)
  - `FG-D`: 0.430865 (6.00%)
  - `FG-AGE`: 0.429762 (5.98%)
  - `FG-SECTOR`: 0.413767 (5.76%)
- City breadth and concentration evidence is in `city_model_error.csv`; feature importance alone is not treated as causal evidence.

## Classification evidence

- `family_competitive_but_unstable` under the frozen Plan05 family-signal rule.
- Operational spread check: at least two origin wins and at least two improved cities in the relevant scenario(s).
- This is a family-level screening result only; no global winner is declared.

## Artifacts

- `oof_predictions.csv` — 1,760 rows per framing with the frozen OOF columns.
- `aggregate_metrics.csv`, `per_origin_metrics.csv`, `city_model_error.csv` — required pooled, origin, and city evidence.
- `configuration_selection.csv`, `inner_selection_evidence.csv` — nested selection and fold-level evidence.
- `missingness_diagnostics.csv`, `feature_importance.csv`, `model_summary.json` — missingness, native importance, and runtime summary.

## Deviations / failures

- None observed.

## Explicit boundary statement

No routing, gating, ensemble, blend, city-specific parameter search, outer-label rescue, 2023 forecast, submission, S5 completion declaration, or S6 entry was performed.
