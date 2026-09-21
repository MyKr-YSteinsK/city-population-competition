# S9-Plan01｜Local Statistical Forecasting

## Status

This is the freshly generated execution contract from remote `main` at
`0a74372ea4e3555b3d659e6a2c77647b92eabea4`. It is registered before Plan01
OOF results are scored. The plan stops at historical evidence and does not
generate a 2023 prediction or reveal any evaluator truth.

Branch: `codex/s9-plan01-local-statistical-forecasting`

## Goal

Test whether competition-legal local target-history forecasting benefits from
persistence, recent trend, smoothing, damping, Theta decomposition, or a
bounded low-order ARIMA process, under the frozen S2 historical validation
contract.

## Allowed input allowlist

The runner may read only these project artifacts:

- `data/normalized/人口规模.csv`
- `docs/s1_data_contract.csv`
- `docs/s1_quality_issue_registry.csv`
- `reports/S2_HANDOFFS/S2-Plan05/plan05_summary.json`
- `reports/S2_HANDOFFS/S2-Plan05/origin_information_boundary.csv`
- `reports/S2_HANDOFFS/S2-Plan02/replay_scenario_definitions.csv`
- `reports/S3_HANDOFFS/S3-Plan02/baseline_predictor_definitions.csv`
- `reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv`
- `scripts/build_s3_plan02_recent_window_and_selector.py`
- `reports/S5_HANDOFFS/S5-Plan02/candidate_definitions.csv`
- `scripts/build_s5_plan02_robust_local_trend_screen.py`

The runner must not read S8 branches, S8 truth artifacts, reconstructed
future labels, `test.xls`, external data, leaderboard data, or any 2022/2023
target value. S6 final-solution and S5 model-evaluation summaries are not
inputs to candidate choice in this Plan.

## Frozen validation contract

- Primary target: `resident`
- Sensitivity target: `registered`
- Scenarios: `R1_REFERENCE_SNAPSHOT_LAG`, `R2_CONSERVATIVE_PUBLICATION_STRESS`
- Outer origins: `2011..2021`
- Target history cutoff: `max used target year <= origin_year - 2`
- Primary metric: pooled city-year MSE
- Required diagnostics: RMSE, MAE, signed bias, macro-origin MSE, per-origin
  results, latest-origin/2021 result, 2013 analogue, coverage, leakage audit,
  finite-prediction audit, and R1/R2 equivalence.

R1 and R2 are retained in the output even though target-history-only methods
are expected to be numerically identical across them.

## Pre-registered candidates

All candidates use only the latest non-missing target history through `T-2`.
No candidate uses covariates, current outer labels, or parameter selection on
the current outer target.

| Candidate | Fixed mechanism / parameters | Fallback |
|---|---|---|
| `B0_PERSISTENCE_LAST_OBSERVED` | Frozen S3 implementation | No history is unscoreable |
| `B1_TWO_POINT_LINEAR_TREND` | Frozen S3 two-point slope | No history is unscoreable |
| `B3_RECENT_3POINT_OLS_TREND` | Frozen S3 B3 implementation and reference rows | Frozen B1 fallback inside source implementation |
| `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND` | Frozen S5 source implementation: latest 3 points, pairwise slope median and median intercept | Frozen B1 fallback, then B0 only if necessary |
| `SES_ALPHA_0_5` | Simple exponential smoothing, fixed `alpha=0.5` | B0 if history is empty |
| `HOLT_LINEAR_A05_B03` | Calendar-year Holt level/trend, fixed `alpha=0.5`, `beta=0.3` | SES if fewer than 2 points |
| `HOLT_DAMPED_A05_B03_PHI080` | Holt level/trend with fixed damping `phi=0.8`, `alpha=0.5`, `beta=0.3` | SES if fewer than 2 points |
| `THETA_THETA2_ALPHA05` | Theta decomposition with fixed `theta=2.0`, OLS trend, SES `alpha=0.5` | B0 if fewer than 2 points |
| `ARIMA_110_BOUNDED` | Differenced AR(1) with OLS intercept/phi and fixed coefficient clip `[-0.95,0.95]` | B3 if fewer than 3 points or invalid fit |
| `ARIMA_011_FIXED_THETA05` | Differenced MA(1)-style forecast with fixed `theta=0.5` and historical mean drift | B3 if fewer than 3 points or invalid fit |

The two ARIMA candidates are fixed bounded structures, not an automatic
`(p,d,q)` search. There is no Optuna, unrestricted grid, model leaderboard,
window search, or parameter-budget expansion.

## Promotion / stop gate

Plan01 does not select a final S9 champion. A new local-statistical method
adds useful evidence for Lab review only if it satisfies either pre-registered
condition:

1. `resident` pooled MSE is no worse than 2% above B3, with at least 5/11
   origin wins and no non-2013 origin loss greater than four times the B3 MSE
   for that origin; or
2. its resident pooled MSE is no worse than 20% above B3, it wins at least
   2/11 origins, and its pooled signed-error correlation with B3 is below
   `0.90`, demonstrating a materially different error profile.

The gate is evidence classification only. It cannot promote a candidate to a
blind champion, start Plan02 automatically, or use registered results to
override resident-primary evidence. The 2013 slice is diagnostic only and
cannot select a candidate alone.

## Acceptance

- 10 candidates × 2 targets × 2 scenarios × 11 origins × 40 cities = 17,600
  scoreable historical prediction rows, or an explicitly documented structural
  coverage failure.
- B0, B1, and B3 predictions agree with the frozen S3 source within the
  serialized numeric tolerance.
- RLT3 implementation identity and formula provenance are audited before its
  rows are scored.
- Every fit/history audit passes the `T-2` boundary; no outer target label is
  used for fitting.
- All predictions are finite; declared fallbacks are visible rather than
  hidden.
- Machine-readable prediction, metric, origin, coverage, leakage,
  configuration, selection-history, compatibility, R1/R2, and 2013 artifacts
  are emitted under `reports/S9_HANDOFFS/S9-Plan01/`.
- No final 2023 prediction, submission, truth reveal, routing, blending,
  gating, or Plan02 execution is performed.
