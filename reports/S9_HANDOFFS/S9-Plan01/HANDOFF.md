# TASK_RESULT — S9-Plan01

## Result
- Status: Complete
- Branch: `codex/s9-plan01-local-statistical-forecasting`
- Recovery base: `0a74372ea4e3555b3d659e6a2c77647b92eabea4`
- Runner commit: `e815f5ff501059a4373f45b7e0ffe321f397ba43`
- Runtime: `2.458` seconds
- State: historical evidence complete; no S9 blind champion or final prediction was frozen

## Goal
Evaluate pre-registered local statistical forecasting mechanisms under the competition-legal target-history boundary.

## Current Evidence
- Frozen S2 geometry: resident primary, registered sensitivity, R1/R2, origins 2011–2021, target history through T-2.
- B3 was taken from the existing S3 implementation and frozen S3-Plan03 historical rows.
- RLT3 provenance was audited against the frozen S5-Plan02 source definition and `fit_median_slope` implementation.

## Hypothesis / Question
Whether level persistence, recent trend, smoothing, damping, Theta, or bounded low-order differenced dynamics add stable historical evidence beyond B3.

## MUST / Non-goals
- MUST: use only legal target history <= T-2; retain resident-primary interpretation; expose fallbacks and leakage evidence.
- Non-goals: no 2023 forecast, no truth reveal, no S8 access, no blend, routing, gating, champion freeze, or automatic Plan02 start.

## Acceptance / Required Evidence
- OOF coverage: `40` target/scenario/candidate groups; each expected 440 rows.
- B3 resident R1 pooled MSE: `872.452652`; all aggregate and origin evidence is in machine-readable CSVs.
- RLT3 provenance pass: `True`; compatibility checks: `[('B0_PERSISTENCE_LAST_OBSERVED_frozen_s3_compatibility', 'yes'), ('B1_TWO_POINT_LINEAR_TREND_frozen_s3_compatibility', 'yes'), ('B3_RECENT_3POINT_OLS_TREND_frozen_s3_compatibility', 'yes')]`.

## Actual Changes
- Added one pre-registered runner and this Plan01 contract.
- Added only `reports/S9_HANDOFFS/S9-Plan01/` evidence artifacts.

## Actual Runs
- Runner syntax compilation.
- Historical OOF generation for 10 fixed candidates over the full 1,760-row target/scenario/origin/city universe.
- Independent coverage, finite-prediction, leakage, compatibility, R1/R2, and metric consistency checks.

## Results — resident primary
| Candidate | R1/R2 pooled MSE | R1/R2 origin wins vs B3 | Plan01 gate |
|---|---:|---:|---|
| `B0_PERSISTENCE_LAST_OBSERVED` | 1926.168182 / 1926.168182 | 1 / 1 | `not_promoted` |
| `B1_TWO_POINT_LINEAR_TREND` | 968.286364 / 968.286364 | 5 / 5 | `distinct_error_profile` |
| `B3_RECENT_3POINT_OLS_TREND` | 872.452652 / 872.452652 | 0 / 0 | `reference_control` |
| `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND` | 861.490909 / 861.490909 | 6 / 6 | `competitive_stable` |
| `SES_ALPHA_0_5` | 3404.712307 / 3404.712307 | 0 / 0 | `not_promoted` |
| `HOLT_LINEAR_A05_B03` | 1041.771921 / 1041.771921 | 3 / 3 | `distinct_error_profile` |
| `HOLT_DAMPED_A05_B03_PHI080` | 1160.689070 / 1160.689070 | 3 / 3 | `not_promoted` |
| `THETA_THETA2_ALPHA05` | 1940.343193 / 1940.343193 | 0 / 0 | `not_promoted` |
| `ARIMA_110_BOUNDED` | 992.809639 / 992.809639 | 4 / 4 | `distinct_error_profile` |
| `ARIMA_011_FIXED_THETA05` | 850.057600 / 850.057600 | 3 / 3 | `distinct_error_profile` |

## Results — registered sensitivity
| Candidate | R1/R2 pooled MSE | R1/R2 origin wins vs B3 |
|---|---:|---:|
| `B0_PERSISTENCE_LAST_OBSERVED` | 420.963636 / 420.963636 | 0 / 0 |
| `B1_TWO_POINT_LINEAR_TREND` | 239.209091 / 239.209091 | 3 / 3 |
| `B3_RECENT_3POINT_OLS_TREND` | 225.421212 / 225.421212 | 0 / 0 |
| `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND` | 223.690909 / 223.690909 | 8 / 8 |
| `SES_ALPHA_0_5` | 635.553294 / 635.553294 | 0 / 0 |
| `HOLT_LINEAR_A05_B03` | 272.818626 / 272.818626 | 4 / 4 |
| `HOLT_DAMPED_A05_B03_PHI080` | 302.949699 / 302.949699 | 3 / 3 |
| `THETA_THETA2_ALPHA05` | 411.976394 / 411.976394 | 0 / 0 |
| `ARIMA_110_BOUNDED` | 226.901527 / 226.901527 | 5 / 5 |
| `ARIMA_011_FIXED_THETA05` | 241.035627 / 241.035627 | 5 / 5 |

## Failures / Deviations
- None in the final run. No candidate was silently removed for a poor score; declared deterministic fallbacks are recorded per row.

## Interpretation / Decision
- This Plan does not declare a champion. Candidates are handed to Lab/Main for the Plan01 promotion gate review.
- The 2013 analogue is retained as a diagnostic slice only; it does not select a candidate.
- R1/R2 equivalence is expected for these target-history-only methods and is audited rather than used to claim scenario robustness of covariates.

## Remaining Uncertainty
- Fixed smoothing/ARIMA parameters are intentionally low-capacity and were not searched; a result is evidence for the tested structures only.
- RLT3 is included only as a provenance-audited historical expert; its Plan01 rows do not use post-reveal information.
- No later combination, gating, census mechanism, or final 2023 fit has been evaluated.

## Material Project-State Changes
- PROJECT_BRIEF: none.
- DECISIONS: none; Plan01 does not choose a champion.
- CURRENT_STATE: S9-Plan01 historical local-statistical evidence exists on this branch; blind freeze is not reached.

## Artifacts
- `S9-Plan01-Local-Statistical-Forecasting.md`
- `oof_predictions.csv`
- `aggregate_metrics.csv`
- `per_origin_metrics.csv`
- `city_error_metrics.csv`
- `coverage.csv`
- `leakage_audit.csv`
- `stability_summary.csv`
- `2013_stress_slice.csv`
- `r1_r2_equivalence.csv`
- `compatibility_checks.csv`
- `selection_history.csv`
- `blind_boundary_audit.csv`
- `model_configuration.json`
- `plan01_summary.json`
- `HANDOFF.md`

## Suggested Next Boundary
Lab/Main review of this evidence. Plan02 must not start automatically from this task.
