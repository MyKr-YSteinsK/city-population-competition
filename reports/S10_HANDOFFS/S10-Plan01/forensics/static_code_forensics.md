# S10-Plan01 Static Code Forensics — `forecast_population.py`

## Scope and immutable identity

This document records static conclusions from the user-supplied script before any execution. The immutable intake copy is `reports/S10_HANDOFFS/S10-Plan01/intake/original/forecast_population.py`; its expected and observed identity is 20,738 bytes, 441 LF-split lines including the terminal empty line, and SHA256 `f719fa577d2c5b8fd51a024af0dd084390f849e7c3564a34ac816a4b35fb8342`. The external source file is not edited.

The script defines `BASE = Path(__file__).resolve().parent` and `RAW = BASE / "raw"`. `_pick(pattern)` sorts `RAW.glob(pattern)` and silently selects the first match. Therefore runtime exact paths must be recorded; a glob pattern alone is not sufficient evidence.

## Static dataflow

`raw/*.xlsx` → `load_data()` → population reshape plus auxiliary city-year reshapes → per-frame numeric `groupby([city, year]).mean()` and outer merges → `repair_obvious_target_anomalies()` → target/exogenous maps with as-of lookup → `build_features()` → expanding target-history validation and panel model fitting → local/model predictions → method metrics and per-city MAE selector → nonnegative, two-decimal `submission_resident.csv` and `submission_registered.csv`.

The script explicitly searches these seven patterns: `*人口规模*.xlsx`, `*人口密度*.xlsx`, `*城镇化率*.xlsx`, `*就业信息*.xlsx`, `*工资水平*.xlsx`, `*年龄结构*.xlsx`, and `*生活水平*.xlsx`. It has no static reference to `test.xls`, B-data, S8 truth, or external future population files. The disposable runtime must contain only the seven legal XLSX files so that this static fact is also true at runtime.

## Targets and hard-coded mutation

- Targets are handled separately as `resident` and `registered` (lines 424–426).
- `repair_obvious_target_anomalies()` checks only `(city == "city40", year == 2021)` in the registered column (lines 150–160).
- The trigger is at least two non-null prior values from 2017–2020 and `median(prior) / max(current, 1.0) > 2.0` (lines 156–159).
- The replacement is the median of the last two prior values. This is a manual target mutation in the script contract, not a general data-cleaning rule.
- The function writes only `registered`; resident is not modified by this function. Because the repaired population is passed into target maps, features, validation actuals, and final registered prediction, the registered path is affected wherever that row enters history or scoring. The exact source value, trigger result, replacement, and downstream row counts are runtime evidence to be recorded separately.

## Local candidate pool

`local_candidates()` statically defines `local_last`, `local_slope3`, `local_slope5`, `local_mean_slope5`, `local_linear3`, and `local_linear5`. It adds `density_hold` and `density_plus_slope5` only when target is resident, the area ratio is available, and density is observed exactly at `cutoff + 1`; it does not carry an older density into this candidate. The slope candidates use the last `k + 1` history points, so the three-point and five-point labels refer to slope windows of three and five annual differences respectively. The method list also names five pooled model candidates.

## Pooled model pool and exact hyperparameters

`_model()` defines:

- `Ridge(alpha=20.0)`;
- `ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=-1)`;
- `RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=-1)`;
- `HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=5.0, random_state=42)`;
- `GradientBoostingRegressor(n_estimators=220, learning_rate=0.03, max_depth=2, loss="huber", random_state=42)`.

No CatBoost, LightGBM, XGBoost, sweep, ensemble, or hyperparameter search is present. ExtraTrees and RandomForest request `n_jobs=-1`.

## Feature timing and feature construction

For a feature row for target year `year`:

- target-history objects used for slopes and area ratio are capped at `year - 2`;
- target lags are `_asof(..., year - lag)` for lag 2, 3, 4, and 5;
- target slopes and the linear slope are derived from history through `year - 2`;
- every exogenous column is `_asof(..., year - 1)`;
- `resident_density_proxy` uses a resident/density area ratio through `year - 2` and density as-of `year - 1`;
- `registered_resident_ratio` uses the target lag-2 values;
- `year_since_2000` is deterministic.

Thus the target-history cutoff is T-2 for a validation target T, while exogenous values are requested at T-1. `_asof()` carries the latest available observation at or before the requested year; this can be stale carry-forward when a source stops earlier. The local density candidates use `_exact(cutoff + 1)`, a different and stricter rule. For the 2023 final target, the feature row requests exogenous values as-of 2022 and target history through 2021; final local candidates also use cutoff 2021 and horizon 2.

## Preprocessing scope

`_matrices()` concatenates train and test feature rows before `get_dummies(..., columns=["city"])`, so the dummy-column schema is the train/test union. The `SimpleImputer(strategy="median", add_indicator=True)` is fit only on the train matrix and then applied to test. Ridge fits `StandardScaler` only on the imputed train matrix and transforms test; tree models use the imputed matrices without scaling. There is no full-data numerical imputer or scaler fit. The union operation can reveal category-column structure, but it does not compute numerical statistics from test rows.

## Validation geometry and metrics

`evaluate_and_forecast()` uses validation years 2017, 2018, 2019, 2020, and 2021. Each validation year T uses cutoff T-2; training rows have feature year `<= cutoff` and non-null current target; the validation label at T is merged only for scoring. The final fit uses cutoff 2021 and target year 2023. The script therefore has five historical origins, horizon 2, and final target-history cutoff 2021.

The method-level table aggregates valid predictions by method and prints `n`, MAE, MedAE, and RMSE. `RMSE` is `sqrt(mean(squared error))`; the imported `mean_absolute_error` and `mean_squared_error` functions are not used, and the script does not print or save MSE. The method table is sorted by MAE then RMSE.

The final selector is a separate object: for each city, it aggregates the same 2017–2021 validation records by city and method, requires `n >= 2`, sorts by that city's historical MAE then method name, and uses the selected method's 2023 prediction. If the selected prediction is missing it falls back through globally sorted methods. Final predictions are lower-bounded at zero and rounded to two decimals only in the CSV. The same validation records are therefore used for historical method evaluation and city-specific final selection; this is a non-nested selector diagnostic, not an unbiased model-comparison estimator.

## Static evidence boundaries

The script reads the current raw population table as its target source and has a code-level registered-target repair. It does not contain the project S1 canonical/provenance/quarantine logic. The following are intentionally not judged from static code alone: exact glob hits and raw hashes, actual duplicate counts, actual city40 values and repair trigger, dependency versions, runtime failures, output hashes, and the provenance of any external claim about `675`. Those require the bounded runtime and independent harness artifacts.
