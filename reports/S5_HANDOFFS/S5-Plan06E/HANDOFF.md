# S5-Plan06E Handoff — Delta/Growth target framing screening

## Result

- Status: `COMPLETE` (framing screening only)
- F3 classification: `framing_signal_found`
- F4 classification: `framing_signal_found`
- Overall classification: `framing_signal_found`
- Worker branch: `s5/plan06e-delta-growth-framing`
- Wave base SHA: `1f97f67488cd7aec120ef8b9fa04407dc13d54ba`
- Artifact-generation source commit: `469787d00b1dc220a017df02c6fbb6fcb76d2605`
- S5 is not declared complete; S6 is not entered.

## Frozen contract and method

- Formal validation: 1,760 rows; replay: 2,320 rows; 144 numeric Plan04 features plus raw string `city_id` categorical metadata.
- CatBoostRegressor uses only the three pre-registered configurations, seed `20260919`, raw numeric NaN, CPU `thread_count=1`, and no outer-label early stopping.
- F3 label is `actual_target_value - pop_latest_level`; F4 label is `log(actual_target_value / pop_latest_level)`; reconstruction is un-clipped.
- Configuration selection uses only eligible inner folds inside each outer training pool; no outer labels enter selection.

## Aggregate metrics

| target | scenario | framing | pooled MSE | macro-origin MSE | origin wins vs B3 | latest-2021 MSE | RMSE | MAE | bias | top1 city SSE | top3 city SSE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| registered | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | 228.336277 | 228.336277 | 5 | 1542.851264 | 15.110800 | 6.163030 | 0.166993 | 42.29% | 80.62% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | 243.244531 | 243.244531 | 4 | 1563.975996 | 15.596299 | 6.555444 | 0.347977 | 40.04% | 76.31% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | 227.523387 | 227.523387 | 5 | 1516.440780 | 15.083878 | 6.213451 | 0.191784 | 42.52% | 80.29% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | 246.889893 | 246.889893 | 4 | 1558.691800 | 15.712730 | 6.463560 | 0.230483 | 39.36% | 77.21% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | 788.841312 | 788.841312 | 4 | 2988.542197 | 28.086319 | 14.242944 | -1.192178 | 25.51% | 47.96% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | 739.600665 | 739.600665 | 5 | 2956.289001 | 27.195600 | 13.677303 | -1.207226 | 24.69% | 48.55% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | 818.447737 | 818.447737 | 5 | 3469.764492 | 28.608526 | 14.227051 | -1.990988 | 26.70% | 49.52% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | 802.445747 | 802.445747 | 7 | 3595.001285 | 28.327473 | 14.026808 | -1.182331 | 26.85% | 50.09% |

## Configuration selections

- `F3_DELTA_FROM_LATEST` / `FRM_C1_CONSERVATIVE`: 13 outer fits
- `F3_DELTA_FROM_LATEST` / `FRM_C2_MEDIUM`: 18 outer fits
- `F3_DELTA_FROM_LATEST` / `FRM_C3_FLEXIBLE_REGULARIZED`: 13 outer fits
- `F4_LOG_GROWTH_FROM_LATEST` / `FRM_C1_CONSERVATIVE`: 31 outer fits
- `F4_LOG_GROWTH_FROM_LATEST` / `FRM_C2_MEDIUM`: 3 outer fits
- `F4_LOG_GROWTH_FROM_LATEST` / `FRM_C3_FLEXIBLE_REGULARIZED`: 10 outer fits
- Fallback conservative selections: 16 total, 8 per framing.

## Scale-bin behavior

Bins are low/medium/high tertiles of `pop_latest_level` computed inside each relevant outer training universe.
| target | scenario | framing | bin | rows | model MSE | B3 MSE | improvement |
|---|---|---|---|---:|---:|---:|---:|
| registered | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | high | 169 | 274.595817 | 288.818705 | 4.92% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | low | 111 | 429.473649 | 404.699449 | -6.12% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | medium | 160 | 39.935587 | 34.083333 | -17.17% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | high | 169 | 311.676417 | 288.818705 | -7.91% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | low | 111 | 422.775280 | 404.699449 | -4.47% |
| registered | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | medium | 160 | 46.413895 | 34.083333 | -36.18% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | high | 169 | 270.827266 | 288.818705 | 6.23% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | low | 111 | 431.463853 | 404.699449 | -6.61% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | medium | 160 | 40.299965 | 34.083333 | -18.24% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | high | 169 | 320.315896 | 288.818705 | -10.91% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | low | 111 | 424.275000 | 404.699449 | -4.84% |
| registered | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | medium | 160 | 46.272758 | 34.083333 | -35.76% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | high | 166 | 1832.030955 | 2023.407129 | 9.46% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | low | 121 | 43.067711 | 53.676538 | 19.76% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F3_DELTA_FROM_LATEST | medium | 153 | 246.809449 | 271.233479 | 9.00% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | high | 166 | 1716.593486 | 2023.407129 | 15.16% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | low | 121 | 39.003398 | 53.676538 | 27.34% |
| resident | R1_REFERENCE_SNAPSHOT_LAG | F4_LOG_GROWTH_FROM_LATEST | medium | 153 | 233.662502 | 271.233479 | 13.85% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | high | 166 | 1903.261328 | 2023.407129 | 5.94% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | low | 121 | 47.950551 | 53.676538 | 10.67% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F3_DELTA_FROM_LATEST | medium | 153 | 250.807890 | 271.233479 | 7.53% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | high | 166 | 1869.214329 | 2023.407129 | 7.62% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | low | 121 | 42.089833 | 53.676538 | 21.59% |
| resident | R2_CONSERVATIVE_PUBLICATION_STRESS | F4_LOG_GROWTH_FROM_LATEST | medium | 153 | 246.363923 | 271.233479 | 9.17% |

## Reconstruction and boundary statement

- Pathology check: none observed; all formal/replay `pop_latest_level` values were positive and no clipping was applied.
- No Plan04 feature change, replay-row change, post-hoc city routing, gating, ensemble, stacking, 2023 prediction, or submission was performed.
- This worker does not declare S5 complete and does not enter S6.

## Artifacts

- `oof_predictions.csv`, `aggregate_metrics.csv`, `per_origin_metrics.csv`, `configuration_selection.csv`, `inner_selection_evidence.csv`, `city_model_error.csv`.
- `scale_error_diagnostics.csv`, `scale_bin_metrics.csv`, and `framing_summary.json`.
