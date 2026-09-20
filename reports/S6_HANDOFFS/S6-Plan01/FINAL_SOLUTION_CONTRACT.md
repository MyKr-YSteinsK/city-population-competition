# S6-Plan01 Final Solution Contract

## Freeze result

- Contract ID: `S6_FINAL_F4_RESIDENT_PRIMARY_2023_V1`
- Status: `PARTIALLY_FROZEN_SUBMISSION_BLOCKED`
- Ready for Main stage review: `yes`
- Ready for the later clean raw rebuild: `yes`
- Formal 2023 prediction/submission generated: `no`
- Source main SHA audited: `f9b97f32adc0e60e6d270db16392e4d46e7ddbe4`

## Target and framing

- Primary target: `resident`, source column `常住人口（万人）`, local role `primary_proxy`.
- Official wording: `总人口`; official mapping to resident versus registered remains `UNKNOWN`.
- The later final pipeline must retain two separate target paths: resident `primary` and registered `alternative`; this Plan freezes the paths but does not fit, predict, average, or mix them.
- F4: `F4_LOG_GROWTH_FROM_LATEST`.
- Training label: `log(actual_target_value / pop_latest_level)`.
- Reconstruction: `pop_latest_level * exp(raw_prediction)`.
- Population history and final labels are capped at `T-2`; for 2023 this is `<= 2021`.
- No clipping, no 2022/2023 target truth, and positive/finite checks are required.

## Feature contract

- Feature names are read from Plan04 `feature_definitions.csv` and checked against the Plan04B machine contract: **144 numeric model features**.
- `city_id` is the only categorical metadata and is handled natively by CatBoost.
- Numeric missingness remains raw NaN. No imputation, scaling, one-hot encoding, target encoding, selector, or post-hoc feature change is allowed.
- Source quarantine, provenance, density-regime, and age benchmark guards remain active.

## Current 2023 information set

- Existing Plan02 mappings are reused without new source policy: R1_REFERENCE_SNAPSHOT_LAG=EXACT_R1; R2_CONSERVATIVE_PUBLICATION_STRESS=EXACT_R2.
- The audit covers all 14 logical sources backed by the 7 legal XLSX workbooks.
- Publication timing remains unproven; R1/R2 remain explicit sensitivity assumptions, not historical facts.
- Detailed evidence: `current_2023_information_set_audit.csv`.

## Final training geometry

- Conceptual outer prediction year: `2023`.
- Allowed historical target labels: `pseudo_target_year <= 2021`.
- Resident primary geometry: R1_REFERENCE_SNAPSHOT_LAG=573 rows/40 cities; R2_CONSERVATIVE_PUBLICATION_STRESS=573 rows/40 cities.
- The replay source has 2,320 rows, unique `(target, scenario, pseudo_target_year, city_id)` keys, and zero source-cap/quarantine/benchmark boundary violations in the audited rows.
- The builder excluded `1,040` pre-matrix rows for documented reasons (`b3_prediction_unavailable:480; no_target_label:560`); no fallback rows are invented.
- Detailed evidence: `final_training_geometry.csv`.

## Model and selection

- CatBoostRegressor `1.2.10`, RMSE, seed `20260919`, `thread_count=1`, `random_strength=1.0`, no outer-label early stopping.
- Only the three historical configurations C1/C2/C3 are allowed.
- Inner selection uses only eligible folds inside each outer training pool; C1 is selected when eligible fold count is below 3. This is distinct from the B3 solution fallback.
- No 2023 label, score, per-city gate, routing, ensemble, or stacking is allowed.

## B3 control and fallback

- Authoritative implementation: `scripts/build_s3_plan02_recent_window_and_selector.py`.
- B3 is the frozen reference and whole-solution fallback, not a per-city mixture with F4.
- Frozen resident/registered pooled MSE: `872.452651515143` / `225.421212121213`.

## Submission boundary

- Official PDF confirms CSV `submission.csv` with `city_id`, `year`, and `pred`; Main evidence confirms 40 rows `city1..city40` and `year=2023`.
- Official target wording is `总人口`, but official mapping and unit are not safe to freeze from available evidence. The formal submission remains blocked and is not generated.
- When later authorized, resident and registered must produce separate candidates; no averaging or mixing is permitted without new evidence.
- Detailed evidence: `submission_schema_evidence.md`.

## Rebuild boundary

- Preserve: official PDF/7 XLSX, Git history/tags, and this final contract/evidence.
- Rebuildable: normalized/quarantine/provenance/regime CSVs, Plan04/04B matrices, model/prediction artifacts, caches/temp.
- Forbidden: `data/quarantine/source/test.xls` content, external/B-data as default input, manually edited truth, 2022/2023 target truth, and worker OOF as final-fit shortcut.
- Detailed manifest: `rebuild_manifest.csv`.
