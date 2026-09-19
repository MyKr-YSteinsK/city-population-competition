"""Build the S4-Plan04B wage residual-correction screening evidence.

This runner is intentionally narrower than the earlier S4 runners.  It keeps
the frozen B3 recent-three-point trend as the prediction backbone and tests
one fixed Ridge residual correction using only the wage source.  The runner
does not alter the S2 contract, rerun S4 predecessors, tune a model, produce
a 2023 forecast, or perform the optional deep-dive robustness work.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_s2_plan02_geometry as s2_geometry
import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02
import build_s4_plan02_urbanization_residual as s4_plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan04B"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
S4_PLAN03_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan03"

STARTING_COMMIT = "b8c302abe7ebf1710869aaa55a00d4b926d2bcac"
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

R1_ID = "R1_REFERENCE_SNAPSHOT_LAG"
R2_ID = "R2_CONSERVATIVE_PUBLICATION_STRESS"
WAGE_SOURCE_ID = "wage"
WAGE_VALUE_COLUMN = "averageWage"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M4B_ID = "M4B_B3_PLUS_RIDGE_WAGE"
FEATURE_NAMES = (
    "wage_log_latest",
    "wage_recent_log_growth",
    "wage_recent3_log_slope",
)

MODEL_DEFINITIONS = (
    {
        "predictor_id": B3_ID,
        "model_family": "frozen_s3_reference",
        "feature_group_id": "FROZEN_B3",
        "feature_names": "",
        "target_definition": "frozen B3 population level prediction",
        "preprocessing": "not_applicable_frozen_reference",
        "estimator": "frozen_B3",
        "ridge_alpha": "",
        "uses_covariates": "no",
        "wage_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M4B_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "WAGE_RECENT_V1",
        "feature_names": ";".join(FEATURE_NAMES),
        "target_definition": "b3_residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "uses_covariates": "yes",
        "wage_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
)

SCREENING_FIELDS = (
    "record_type",
    "target_id",
    "scenario_id",
    "model_id",
    "baseline_model_id",
    "origin_year",
    "metric_name",
    "metric_value",
    "comparison_value",
    "pooled_mse",
    "macro_origin_mse",
    "latest_2021_mse",
    "scoreable_city_year_count",
    "candidate_origin_wins",
    "candidate_origin_losses",
    "origin_ties",
    "origin_count",
    "relative_improvement_pct",
    "latest_relative_improvement_pct",
    "scenario_equivalent",
    "status",
    "notes",
)

AUDIT_FIELDS = (
    "target_id",
    "scenario_id",
    "model_run_scenario",
    "outer_origin_year",
    "row_role",
    "city_id",
    "pseudo_target_year",
    "population_target_history_cutoff_year",
    "population_history_max_year_used",
    "population_history_year_count",
    "population_history_t_minus_2_pass",
    "outer_training_label_within_T_minus_2",
    "wage_required_lag_years",
    "wage_required_latest_usable_year",
    "wage_latest_actual_year",
    "wage_staleness_years",
    "wage_allowed_years",
    "wage_allowed_year_count",
    "wage_latest_value",
    "wage_used_years",
    "wage_used_year_max",
    "wage_candidate_window_min_year",
    "wage_recent_log_growth_source_years",
    "wage_recent3_log_slope_source_years",
    "wage_missing_feature_names",
    "wage_insufficient_history_count",
    "wage_invalidated_keys_allowed_history",
    "wage_invalidated_key_allowed_history_count",
    "wage_invalidated_key_candidate_hit_count",
    "wage_invalidated_key_used_count",
    "wage_invalidated_value_reintroduced",
    "wage_years_within_scenario_cap",
    "cross_target_population_use",
    "other_covariate_columns_used",
    "city_identity_used",
    "boundary_status",
)


def present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {
        "",
        "NA",
        "N/A",
        "NaN",
        "nan",
        "None",
    }


def as_float(value: Any) -> float | None:
    if not present(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    assert values
    return math.fsum(values) / len(values)


def competition_score(mse: float) -> float:
    return 1.0 / (1.0 + mse)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rounded(value: Any, digits: int = 12) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else ""
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> None:
    assert rows or fields
    fieldnames = list(fields or rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if row.get(field) is None else rounded(row.get(field))
                    for field in fieldnames
                }
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def assert_starting_commit_is_ancestor() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"Plan04B frozen base {STARTING_COMMIT} is not an ancestor of current HEAD"
    )


def source_series(source: dict[str, Any]) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    city_column = source["city_col"]
    year_column = source["year_col"]
    value_column = source["value_columns"][0]
    for row in source["value_rows"]:
        value = as_float(row.get(value_column))
        if value is None:
            continue
        year = int(float(row[year_column]))
        assert value > 0, (row, value)
        result.setdefault(row[city_column], {})[year] = value
    return result


def ols_slope(years: list[int], values: list[float]) -> float:
    assert len(years) == len(values) >= 2
    x_mean = mean(float(year) for year in years)
    y_mean = mean(values)
    denominator = math.fsum((year - x_mean) ** 2 for year in years)
    assert denominator > 0
    return math.fsum(
        (year - x_mean) * (value - y_mean)
        for year, value in zip(years, values)
    ) / denominator


def build_wage_features(
    city: str,
    series: dict[int, float],
    pseudo_target_year: int,
    required_lag_years: int,
    invalidated_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    cap = pseudo_target_year - required_lag_years
    allowed_years = sorted(
        year for year, value in series.items() if year <= cap and value > 0
    )
    latest_year = allowed_years[-1] if allowed_years else None
    latest_value = series[latest_year] if latest_year is not None else None

    recent_growth = None
    recent_growth_years: list[int] = []
    if len(allowed_years) >= 2:
        recent_growth_years = allowed_years[-2:]
        previous_year, current_year = recent_growth_years
        recent_growth = math.log(series[current_year] / series[previous_year]) / (
            current_year - previous_year
        )

    recent3_slope = None
    recent3_years: list[int] = []
    if len(allowed_years) >= 3:
        recent3_years = allowed_years[-3:]
        recent3_slope = ols_slope(
            recent3_years,
            [math.log(series[year]) for year in recent3_years],
        )

    used_years = sorted(
        set(
            ([latest_year] if latest_year is not None else [])
            + recent_growth_years
            + recent3_years
        )
    )
    candidate_window_min_year = min(used_years) if used_years else None
    city_invalidated_years = sorted(
        year for invalidated_city, year in invalidated_keys if invalidated_city == city
    )
    invalidated_allowed_years = [
        year for year in city_invalidated_years if year <= cap
    ]
    invalidated_candidate_hits = [
        year
        for year in invalidated_allowed_years
        if candidate_window_min_year is not None and year >= candidate_window_min_year
    ]
    invalidated_used = [year for year in invalidated_allowed_years if year in used_years]

    missing_names = [
        feature_name
        for feature_name, value in (
            ("wage_log_latest", math.log1p(latest_value) if latest_value is not None else None),
            ("wage_recent_log_growth", recent_growth),
            ("wage_recent3_log_slope", recent3_slope),
        )
        if value is None
    ]
    return {
        "wage_required_lag_years": required_lag_years,
        "wage_required_latest_usable_year": cap,
        "wage_allowed_years": ";".join(str(year) for year in allowed_years),
        "wage_allowed_year_count": len(allowed_years),
        "wage_latest_actual_year": latest_year,
        "wage_latest_value": latest_value,
        "wage_log_latest": math.log1p(latest_value) if latest_value is not None else None,
        "wage_staleness_years": (
            pseudo_target_year - latest_year if latest_year is not None else None
        ),
        "wage_recent_log_growth": recent_growth,
        "wage_recent_log_growth_source_years": ";".join(
            str(year) for year in recent_growth_years
        ),
        "wage_recent3_log_slope": recent3_slope,
        "wage_recent3_log_slope_source_years": ";".join(
            str(year) for year in recent3_years
        ),
        "wage_used_years": ";".join(str(year) for year in used_years),
        "wage_used_year_max": max(used_years) if used_years else None,
        "wage_candidate_window_min_year": candidate_window_min_year,
        "wage_missing_feature_names": ";".join(missing_names),
        "wage_insufficient_history_count": sum(
            value is None for value in (recent_growth, recent3_slope)
        ),
        "wage_invalidated_keys_allowed_history": ";".join(
            str(year) for year in invalidated_allowed_years
        ),
        "wage_invalidated_key_allowed_history_count": len(invalidated_allowed_years),
        "wage_invalidated_key_candidate_hit_count": len(invalidated_candidate_hits),
        "wage_invalidated_key_used_count": len(invalidated_used),
        "wage_invalidated_value_reintroduced": "yes" if invalidated_used else "no",
        "wage_years_within_scenario_cap": (
            "PASS"
            if all(year <= cap for year in used_years)
            else "FAIL"
        ),
    }


def load_context() -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[int, float]],
    dict[str, Any],
    set[tuple[str, int]],
]:
    s3_plan01.assert_static_boundaries()
    population, population_profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["r1_r2_numeric_residuals_exactly_equal"] is True
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False

    s3_summary = json.loads(
        (S3_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8")
    )
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    assert s3_summary["s3_complete"] is False

    s4_plan03_summary = json.loads(
        (S4_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8")
    )
    assert s4_plan03_summary["b3_compatibility_pass"] is True
    assert s4_plan03_summary["b3_metric_compatibility_pass"] is True
    assert s4_plan03_summary["s4_complete"] is False
    assert s4_plan03_summary["s5_entered"] is False

    contract = s2_geometry.load_contract()
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    geometry = {
        row["source_id"]: row for row in s2_geometry.source_geometry_rows(sources)
    }
    wage_geometry = geometry[WAGE_SOURCE_ID]
    assert wage_geometry["latest_usable_year"] == 2021
    assert wage_geometry["observed_snapshot_freshness_lag_years"] == 2
    assert wage_geometry["source_status"] == "source_plus_invalidation"
    assert wage_geometry["source_derived_count"] == 0
    assert wage_geometry["source_invalidated_quarantine_count"] == 1
    assert int(wage_geometry["repair_count"]) == 0
    assert int(wage_geometry["invalidated_count"]) == 1
    assert int(wage_geometry["derived_count"]) == 0

    wage_source = sources[WAGE_SOURCE_ID]
    wage_series = source_series(wage_source)
    invalidated_keys = set(wage_source["quarantine_keys"])
    canonical_value_keys = {
        (row[wage_source["city_col"]], int(float(row[wage_source["year_col"]])))
        for row in wage_source["value_rows"]
    }
    assert canonical_value_keys.isdisjoint(invalidated_keys)
    assert invalidated_keys == {("city31", 2011)}
    canonical_row = next(
        row
        for row in wage_source["rows"]
        if row[wage_source["city_col"]] == "city31"
        and int(float(row[wage_source["year_col"]])) == 2011
    )
    assert not present(canonical_row[WAGE_VALUE_COLUMN])

    return (
        population,
        population_profile,
        geometry,
        wage_series,
        wage_geometry,
        invalidated_keys,
    )


def build_boundary_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    city: str,
    pseudo_target_year: int,
    fit: dict[str, Any],
    features: dict[str, Any],
    row_role: str,
) -> dict[str, Any]:
    history_max = fit["history_max_year"]
    population_pass = (
        history_max is None
        or history_max <= pseudo_target_year - HORIZON_YEARS
    )
    outer_label_pass = (
        row_role == "validation"
        or pseudo_target_year <= outer_origin - HORIZON_YEARS
    )
    wage_cap_pass = features["wage_years_within_scenario_cap"] == "PASS"
    invalidated_pass = features["wage_invalidated_value_reintroduced"] == "no"
    boundary_pass = all(
        (population_pass, outer_label_pass, wage_cap_pass, invalidated_pass)
    )
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "model_run_scenario": R1_ID if scenario_id == R1_ID else "scenario_equivalence_only",
        "outer_origin_year": outer_origin,
        "row_role": row_role,
        "city_id": city,
        "pseudo_target_year": pseudo_target_year,
        "population_target_history_cutoff_year": pseudo_target_year - HORIZON_YEARS,
        "population_history_max_year_used": history_max,
        "population_history_year_count": len(fit["history_years"]),
        "population_history_t_minus_2_pass": "PASS" if population_pass else "FAIL",
        "outer_training_label_within_T_minus_2": (
            "PASS" if outer_label_pass else "FAIL"
        ),
        **features,
        "cross_target_population_use": "no",
        "other_covariate_columns_used": "no",
        "city_identity_used": "no",
        "boundary_status": "PASS" if boundary_pass else "FAIL",
    }


def pipeline_for() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                as_float(row["features"].get(feature_name))
                if as_float(row["features"].get(feature_name)) is not None
                else np.nan
                for feature_name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_candidate(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> tuple[list[float], dict[str, Any]]:
    assert training_rows
    x_train = feature_matrix(training_rows)
    x_validation = feature_matrix(validation_rows)
    y_train = np.asarray(
        [row["b3_residual_actual_minus_prediction"] for row in training_rows],
        dtype=float,
    )
    training_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    model = pipeline_for()
    model.fit(x_train, y_train)
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    indicator_names = [
        f"missing_indicator__{FEATURE_NAMES[index]}"
        for index in imputer.indicator_.features_
    ]
    coefficients = [float(value) for value in np.asarray(ridge.coef_).ravel()]
    return model.predict(x_validation).tolist(), {
        "training_row_count": len(training_rows),
        "training_missing_feature_cell_count": training_missing,
        "training_imputation_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "validation_imputation_count": validation_missing,
        "preprocessing_fit_row_count": len(training_rows),
        "model_fit_row_count": len(training_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "imputer_statistics": [
            float(value) if math.isfinite(float(value)) else None
            for value in imputer.statistics_
        ],
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": coefficients,
        "coefficient_names": FEATURE_NAMES + tuple(indicator_names),
        "feature_names": ";".join(FEATURE_NAMES),
        "ridge_alpha": 1.0,
    }


def build_candidate_row(
    target: dict[str, str],
    origin: int,
    city: str,
    validation: dict[str, Any],
    predicted_residual: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = validation["target_value"]
    b3_prediction = validation["b3_prediction"]
    prediction = b3_prediction + predicted_residual
    residual = prediction - actual
    features = validation["features"]
    return {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": R1_ID,
        "predictor_id": M4B_ID,
        "model_id": M4B_ID,
        "origin_year": origin,
        "outer_origin_year": origin,
        "city_id": city,
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_cutoff_year": origin - HORIZON_YEARS,
        "actual_target_year_for_scoring": origin,
        "actual_target_value_for_scoring": actual,
        "prediction_value": prediction,
        "prediction_status": "scoreable",
        "scoreable": "yes",
        "residual_prediction_minus_actual": residual,
        "squared_error": residual * residual,
        "absolute_error": abs(residual),
        "b3_prediction_value": b3_prediction,
        "b3_residual_actual_minus_prediction": validation[
            "b3_residual_actual_minus_prediction"
        ],
        "predicted_b3_residual": predicted_residual,
        "feature_builder_id": "WAGE_RESIDUAL_V1",
        "feature_group_id": "WAGE_RECENT_V1",
        **features,
        "training_label_min_year": metadata["training_label_min_year"],
        "training_label_max_year": metadata["training_label_max_year"],
        "training_row_count": metadata["training_row_count"],
        "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
        "model_fit_row_count": metadata["model_fit_row_count"],
        "training_missing_feature_cell_count": metadata[
            "training_missing_feature_cell_count"
        ],
        "validation_missing_feature_cell_count": metadata[
            "validation_missing_feature_cell_count"
        ],
        "covariates_used": "yes",
        "wage_used": "yes",
        "city_identity_used": "no",
        "other_covariate_columns_used": "no",
    }


def wrap_b3_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "model_id": B3_ID,
            "feature_builder_id": "FROZEN_B3_REFERENCE",
            "feature_group_id": "FROZEN_B3",
            "covariates_used": "no",
            "wage_used": "no",
            "city_identity_used": "no",
            "other_covariate_columns_used": "no",
            "b3_prediction_value": row["prediction_value"],
            "b3_residual_actual_minus_prediction": -float(
                row["residual_prediction_minus_actual"]
            ),
            "predicted_b3_residual": "",
        }
    )
    return result


def build_rows_and_audit(
    population: dict[str, dict[str, dict[int, float]]],
    wage_series: dict[str, dict[int, float]],
    wage_geometry: dict[str, Any],
    invalidated_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    b3_all_rows, _ = s3_plan02.build_fixed_rows(population)
    b3_r1_rows = [
        wrap_b3_row(row)
        for row in b3_all_rows
        if row["predictor_id"] == B3_ID and row["scenario_id"] == R1_ID
    ]
    assert len(b3_r1_rows) == len(TARGETS) * len(COMMON_ORIGINS) * len(CITIES)

    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    training_audit_rows = 0
    validation_audit_rows = 0
    validation_feature_missing_cells = 0
    preprocessing_mismatch_count = 0
    population_t_minus_2_fail_count = 0
    boundary_fail_count = 0

    wage_lags = {
        scenario_id: s2_geometry.required_lag_years(
            WAGE_SOURCE_ID, wage_geometry, scenario_id
        )
        for scenario_id in SCENARIOS
    }
    assert wage_lags[R1_ID] == wage_lags[R2_ID] == 2

    for target in TARGETS:
        target_id = target["target_id"]
        for origin in COMMON_ORIGINS:
            training_rows: list[dict[str, Any]] = []
            validation_rows: list[dict[str, Any]] = []
            for city in CITIES:
                target_series = population[target_id][city]
                wage_city_series = wage_series[city]
                for pseudo_year in sorted(
                    year
                    for year in target_series
                    if year <= origin - HORIZON_YEARS
                ):
                    fit = s3_plan02.fit_prediction(
                        B3_ID,
                        target_series,
                        pseudo_year,
                        pseudo_year - HORIZON_YEARS,
                    )
                    features = build_wage_features(
                        city,
                        wage_city_series,
                        pseudo_year,
                        wage_lags[R1_ID],
                        invalidated_keys,
                    )
                    boundary = build_boundary_row(
                        target_id,
                        R1_ID,
                        origin,
                        city,
                        pseudo_year,
                        fit,
                        features,
                        "training",
                    )
                    audit_rows.append(boundary)
                    training_audit_rows += 1
                    if boundary["boundary_status"] != "PASS":
                        boundary_fail_count += 1
                    if boundary["population_history_t_minus_2_pass"] != "PASS":
                        population_t_minus_2_fail_count += 1
                    actual = target_series.get(pseudo_year)
                    if actual is None or fit["prediction"] is None:
                        continue
                    training_rows.append(
                        {
                            "target_value": actual,
                            "b3_prediction": fit["prediction"],
                            "b3_residual_actual_minus_prediction": (
                                actual - fit["prediction"]
                            ),
                            "features": features,
                        }
                    )

                validation_fit = s3_plan02.fit_prediction(
                    B3_ID,
                    target_series,
                    origin,
                    origin - HORIZON_YEARS,
                )
                assert target_series.get(origin) is not None
                assert validation_fit["prediction"] is not None
                validation_features = build_wage_features(
                    city,
                    wage_city_series,
                    origin,
                    wage_lags[R1_ID],
                    invalidated_keys,
                )
                validation_audit = build_boundary_row(
                    target_id,
                    R1_ID,
                    origin,
                    city,
                    origin,
                    validation_fit,
                    validation_features,
                    "validation",
                )
                audit_rows.append(validation_audit)
                validation_audit_rows += 1
                validation_feature_missing_cells += sum(
                    validation_features.get(name) is None for name in FEATURE_NAMES
                )
                if validation_audit["boundary_status"] != "PASS":
                    boundary_fail_count += 1
                if validation_audit["population_history_t_minus_2_pass"] != "PASS":
                    population_t_minus_2_fail_count += 1
                validation_rows.append(
                    {
                        "target_value": target_series[origin],
                        "b3_prediction": validation_fit["prediction"],
                        "b3_residual_actual_minus_prediction": (
                            target_series[origin] - validation_fit["prediction"]
                        ),
                        "features": validation_features,
                    }
                )

            assert len(validation_rows) == len(CITIES)
            predictions, metadata = fit_candidate(training_rows, validation_rows)
            preprocessing_mismatch_count += int(
                metadata["preprocessing_model_fit_rows_equal"] != "yes"
            )
            training_years = [
                row["pseudo_target_year"]
                for row in audit_rows
                if row["target_id"] == target_id
                and row["scenario_id"] == R1_ID
                and row["outer_origin_year"] == origin
                and row["row_role"] == "training"
            ]
            metadata["training_label_min_year"] = min(training_years)
            metadata["training_label_max_year"] = max(training_years)
            metadata["training_row_count"] = len(training_rows)
            for city, predicted_residual, validation in zip(
                CITIES, predictions, validation_rows
            ):
                candidate_rows.append(
                    build_candidate_row(
                        target,
                        origin,
                        city,
                        validation,
                        predicted_residual,
                        metadata,
                    )
                )

    for target in TARGETS:
        target_id = target["target_id"]
        for origin in COMMON_ORIGINS:
            for city in CITIES:
                wage_city_series = wage_series[city]
                features = build_wage_features(
                    city,
                    wage_city_series,
                    origin,
                    wage_lags[R2_ID],
                    invalidated_keys,
                )
                population_series = population[target_id][city]
                fit = s3_plan02.fit_prediction(
                    B3_ID,
                    population_series,
                    origin,
                    origin - HORIZON_YEARS,
                )
                boundary = build_boundary_row(
                    target_id,
                    R2_ID,
                    origin,
                    city,
                    origin,
                    fit,
                    features,
                    "validation",
                )
                audit_rows.append(boundary)
                validation_audit_rows += 1
                if boundary["boundary_status"] != "PASS":
                    boundary_fail_count += 1
                if boundary["population_history_t_minus_2_pass"] != "PASS":
                    population_t_minus_2_fail_count += 1

    r1_r2_audit_rows = [
        row
        for row in audit_rows
        if row["row_role"] == "validation" and row["scenario_id"] == R1_ID
    ]
    r2_rows = [
        row
        for row in audit_rows
        if row["row_role"] == "validation" and row["scenario_id"] == R2_ID
    ]
    assert len(r1_r2_audit_rows) == len(r2_rows) == len(TARGETS) * len(COMMON_ORIGINS) * len(CITIES)
    for r1_row, r2_row in zip(r1_r2_audit_rows, r2_rows):
        assert (
            r1_row["target_id"],
            r1_row["outer_origin_year"],
            r1_row["city_id"],
        ) == (
            r2_row["target_id"],
            r2_row["outer_origin_year"],
            r2_row["city_id"],
        )
        for field in (
            "wage_required_lag_years",
            "wage_required_latest_usable_year",
            "wage_allowed_years",
            "wage_latest_actual_year",
            "wage_used_years",
            "wage_invalidated_key_candidate_hit_count",
        ):
            assert r1_row[field] == r2_row[field], (field, r1_row, r2_row)

    diagnostics = {
        "training_audit_rows": training_audit_rows,
        "validation_audit_rows": validation_audit_rows,
        "validation_feature_missing_cell_count": validation_feature_missing_cells,
        "preprocessing_fit_rows_mismatch_count": preprocessing_mismatch_count,
        "population_t_minus_2_fail_count": population_t_minus_2_fail_count,
        "boundary_fail_count": boundary_fail_count,
        "candidate_row_count": len(candidate_rows),
    }
    return b3_r1_rows, candidate_rows, {"audit_rows": audit_rows, **diagnostics}


@contextmanager
def metric_scope() -> Any:
    old_scenarios = s3_plan01.SCENARIOS
    old_predictors = s3_plan01.PREDICTORS
    s3_plan01.SCENARIOS = (R1_ID,)
    s3_plan01.PREDICTORS = MODEL_DEFINITIONS
    try:
        yield
    finally:
        s3_plan01.SCENARIOS = old_scenarios
        s3_plan01.PREDICTORS = old_predictors


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str, int], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    with metric_scope():
        per_origin, per_origin_map = s3_plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(
            rows, per_origin
        )
    return per_origin, aggregate, per_origin_map, aggregate_map


def b3_compatibility(
    b3_r1_rows: list[dict[str, Any]],
    population: dict[str, dict[str, dict[int, float]]],
) -> tuple[bool, bool, list[dict[str, Any]]]:
    b3_all_rows, _ = s3_plan02.build_fixed_rows(population)
    b3_all_wrapped = [
        wrap_b3_row(row)
        for row in b3_all_rows
        if row["predictor_id"] == B3_ID
    ]
    row_checks = s4_plan02.compare_b3_rows(b3_all_wrapped)
    row_pass = all(row["status"] == "PASS" for row in row_checks)

    old_scenarios = s3_plan01.SCENARIOS
    old_predictors = s3_plan01.PREDICTORS
    s3_plan01.SCENARIOS = (R1_ID,)
    s3_plan01.PREDICTORS = (MODEL_DEFINITIONS[0],)
    try:
        b3_per_origin, _ = s3_plan01.build_per_origin_metrics(b3_r1_rows)
        _, current_map = s3_plan01.build_aggregate_metrics(
            b3_r1_rows, b3_per_origin
        )
    finally:
        s3_plan01.SCENARIOS = old_scenarios
        s3_plan01.PREDICTORS = old_predictors
    frozen_rows = read_csv(S4_PLAN03_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in frozen_rows
        if row["predictor_id"] == B3_ID
    }
    metric_checks: list[dict[str, Any]] = []
    for target in TARGETS:
        key = (target["target_id"], R1_ID, B3_ID)
        current = current_map[key]
        frozen = frozen_map[key]
        deltas = {
            field: abs(float(current[field]) - float(frozen[field]))
            for field in (
                "pooled_city_year_mse",
                "macro_origin_mse",
                "latest_origin_mse",
            )
        }
        metric_checks.append(
            {
                "target_id": target["target_id"],
                "scenario_id": R1_ID,
                "predictor_id": B3_ID,
                "max_abs_metric_delta": max(deltas.values()),
                "status": "PASS" if max(deltas.values()) <= 1e-9 else "FAIL",
            }
        )
    metric_pass = all(row["status"] == "PASS" for row in metric_checks)
    return row_pass, metric_pass, row_checks + metric_checks


def pairwise_summary(
    rows: list[dict[str, Any]],
    per_origin_map: dict[tuple[str, str, str, int], dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for target in TARGETS:
        target_id = target["target_id"]
        result[target_id] = s4_plan02.pairwise_row(
            rows,
            per_origin_map,
            aggregate_map,
            target_id,
            R1_ID,
            B3_ID,
            M4B_ID,
            "M4B_vs_B3",
        )
    return result


def screening_rows(
    aggregate: list[dict[str, Any]],
    per_origin: list[dict[str, Any]],
    pairwise: dict[str, dict[str, Any]],
    equivalence_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in aggregate:
        rows.append(
            {
                "record_type": "aggregate",
                "target_id": row["target_id"],
                "scenario_id": row["scenario_id"],
                "model_id": row["predictor_id"],
                "pooled_mse": row["pooled_city_year_mse"],
                "macro_origin_mse": row["macro_origin_mse"],
                "latest_2021_mse": row["latest_origin_mse"],
                "scoreable_city_year_count": row["total_scoreable_city_year_count"],
                "scenario_equivalent": "yes",
                "status": "PASS",
                "notes": "R1 model run; exact R2 wage-cap equivalence checked",
            }
        )
    for row in per_origin:
        rows.append(
            {
                "record_type": "per_origin",
                "target_id": row["target_id"],
                "scenario_id": row["scenario_id"],
                "model_id": row["predictor_id"],
                "origin_year": row["origin_year"],
                "metric_name": "origin_mse",
                "metric_value": row["mse"],
                "scoreable_city_year_count": row["scoreable_city_count"],
                "status": "PASS",
                "notes": "common strict origin; target history <= T-2",
            }
        )
    for target_id, row in pairwise.items():
        rows.append(
            {
                "record_type": "pairwise",
                "target_id": target_id,
                "scenario_id": R1_ID,
                "model_id": M4B_ID,
                "baseline_model_id": B3_ID,
                "pooled_mse": row["candidate_pooled_mse"],
                "comparison_value": row["baseline_pooled_mse"],
                "latest_2021_mse": row["latest_candidate_mse"],
                "candidate_origin_wins": row["candidate_origin_win_count"],
                "candidate_origin_losses": row["candidate_origin_loss_count"],
                "origin_ties": row["origin_tie_count"],
                "origin_count": row["origin_count"],
                "relative_improvement_pct": row[
                    "pooled_mse_relative_improvement_pct"
                ],
                "latest_relative_improvement_pct": row[
                    "latest_mse_relative_improvement_pct"
                ],
                "scenario_equivalent": "yes",
                "status": "PASS",
                "notes": "positive improvement means M4B lower MSE than B3",
            }
        )
    rows.extend(equivalence_rows)
    rows.extend(sanity_rows)
    return rows


def make_equivalence_rows(
    wage_geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    lags = {
        scenario_id: s2_geometry.required_lag_years(
            WAGE_SOURCE_ID, wage_geometry, scenario_id
        )
        for scenario_id in SCENARIOS
    }
    cap_mismatch_count = sum(
        (origin - lags[R1_ID]) != (origin - lags[R2_ID])
        for origin in COMMON_ORIGINS
    )
    rows = []
    for scenario_id in SCENARIOS:
        rows.append(
            {
                "record_type": "scenario_equivalence",
                "scenario_id": scenario_id,
                "metric_name": "wage_required_lag_years",
                "metric_value": lags[scenario_id],
                "comparison_value": lags[R1_ID],
                "scenario_equivalent": "yes" if lags[scenario_id] == lags[R1_ID] else "no",
                "status": "PASS" if lags[scenario_id] == lags[R1_ID] else "FAIL",
                "notes": "S2 required_lag_years; no publication-date claim",
            }
        )
    rows.append(
        {
            "record_type": "scenario_equivalence",
            "scenario_id": "R1_vs_R2",
            "metric_name": "wage_cap_mismatch_count_common_origins",
            "metric_value": cap_mismatch_count,
            "comparison_value": 0,
            "scenario_equivalent": "yes" if cap_mismatch_count == 0 else "no",
            "status": "PASS" if cap_mismatch_count == 0 else "FAIL",
            "notes": "origin-specific cap equality across 2011-2021",
        }
    )
    equivalent = lags[R1_ID] == lags[R2_ID] and cap_mismatch_count == 0
    return rows, equivalent, lags


def make_sanity_rows(
    b3_row_pass: bool,
    b3_metric_pass: bool,
    diagnostics: dict[str, Any],
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    invalidated_reintroduced = sum(
        row["wage_invalidated_value_reintroduced"] == "yes"
        for row in audit_rows
    )
    invalidated_used = sum(
        int(row["wage_invalidated_key_used_count"] or 0) for row in audit_rows
    )
    checks = [
        (
            "b3_row_compatibility",
            1 if b3_row_pass else 0,
            1,
            "B3 row-level prediction/residual/squared-error delta <= 1e-9",
        ),
        (
            "b3_metric_compatibility",
            1 if b3_metric_pass else 0,
            1,
            "B3 pooled/macro/latest MSE delta <= 1e-9",
        ),
        (
            "wage_invalidated_value_reintroduced_count",
            invalidated_reintroduced,
            0,
            "canonical invalidated value never appears in feature-used years",
        ),
        (
            "wage_invalidated_key_used_count",
            invalidated_used,
            0,
            "invalidated key may be in allowed history but is never used as a value",
        ),
        (
            "population_t_minus_2_fail_count",
            diagnostics["population_t_minus_2_fail_count"],
            0,
            "population B3 history and outer training labels remain <= T-2",
        ),
        (
            "preprocessing_fit_rows_mismatch_count",
            diagnostics["preprocessing_fit_rows_mismatch_count"],
            0,
            "fold-local preprocessing and Ridge fit use the same training rows",
        ),
        (
            "boundary_fail_count",
            diagnostics["boundary_fail_count"],
            0,
            "wage cap, target history, invalidation and outer-label boundary checks",
        ),
        (
            "resident_validation_rows",
            sum(
                row["target_id"] == "resident"
                and row["scenario_id"] == R1_ID
                and row["row_role"] == "validation"
                for row in audit_rows
            ),
            len(COMMON_ORIGINS) * len(CITIES),
            "resident validation audit coverage",
        ),
        (
            "registered_validation_rows",
            sum(
                row["target_id"] == "registered"
                and row["scenario_id"] == R1_ID
                and row["row_role"] == "validation"
                for row in audit_rows
            ),
            len(COMMON_ORIGINS) * len(CITIES),
            "registered validation audit coverage",
        ),
    ]
    return [
        {
            "record_type": "sanity",
            "metric_name": name,
            "metric_value": value,
            "comparison_value": expected,
            "status": "PASS" if value == expected else "FAIL",
            "notes": notes,
        }
        for name, value, expected, notes in checks
    ]


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6f}"


def write_handoff(
    summary: dict[str, Any],
    pairwise: dict[str, dict[str, Any]],
    lags: dict[str, int],
    b3_row_pass: bool,
    b3_metric_pass: bool,
) -> None:
    resident = summary["screening"]["resident"]
    registered = summary["screening"]["registered"]
    status = "PASS" if summary["wage_invalidated_key_guard_pass"] else "FAIL"
    lines = [
        "# S4-Plan04B Handoff｜工资水平发展信号与 B3 残差修正快速筛选",
        "",
        "## 状态与结论",
        "",
        f"- Plan 状态：`evidence complete`；本 worker 只完成 screening，不自动执行 deep dive。",
        f"- screening decision：`{summary['screening_decision']}`；`deep_dive_recommended={str(summary['deep_dive_recommended']).lower()}`。",
        f"- `s4_complete={str(summary['s4_complete']).lower()}`、`s5_entered={str(summary['s5_entered']).lower()}`、`final_2023_forecast_run=false`、`submission_run=false`。",
        "- B3 仍是本轮 frozen comparison baseline；R1/R2 wage cap 完全等价，因此 M4B 只在 R1 实际 fit 一次，R2 复用等价性结论而不复制模型 workload。",
        "",
        "## Actual changes",
        "",
        "1. 新增 `scripts/build_s4_plan04b_wage_screen.py`。",
        "2. 新增本目录下的 `HANDOFF.md`、`plan04b_summary.json`、`screening_metrics.csv`、`wage_boundary_audit.csv`。",
        "3. 未修改 `main`、S2 validation contract、S3/S4 前序 handoff、canonical/raw data、三个并行 Wave A worker 文件或项目状态文档。",
        "",
        "## Execution contract",
        "",
        f"- frozen base / execution starting SHA：`{STARTING_COMMIT}`。",
        f"- current evidence-generation HEAD：`{summary['evidence_generation_head']}`。",
        "- branch：`s4/wave-a-wage`；本 Plan 在独立 worktree 执行；未 merge `main`。",
        "- common origins：`2011–2021`；resident 为 primary，registered 为 sensitivity；population target history 为 `T-2`。",
        "",
        "## Actual model and source geometry",
        "",
        "- candidate：`M4B_B3_PLUS_RIDGE_WAGE`；唯一 feature group：`WAGE_RECENT_V1`。",
        "- features：`log1p(latest positive averageWage)`、最近两个 positive observation 的 actual-year `log(latest/previous) / gap`、最近三个 positive observation 的 actual-year OLS `log(wage)` slope。",
        "- preprocessing：每个 target/origin 独立使用 `SimpleImputer(median, add_indicator=True) + StandardScaler + Ridge(alpha=1.0)`；只在 outer-origin training rows fit。",
        f"- wage source：`data/normalized/工资水平_职工平均工资.csv`；source status=`source_plus_invalidation`；observed latest=`2021`；observed lag=`2`；invalidated key=`city31/2011`。",
        f"- S2 required lag：R1=`{lags[R1_ID]}`、R2=`{lags[R2_ID]}`；common-origin cap mismatch=`0`；`scenario_equivalent=true`。",
        "- 未使用 density、urbanization、employment、city ID；没有 tuning、GBDT、lag/rolling/polynomial 扩展、2023 forecast 或 submission。",
        "",
        "## Core screening metrics",
        "",
        "数值为 common strict `40 cities × 11 origins` 的 pooled MSE / macro-origin MSE / latest 2021 MSE；relative improvement 为 `(B3 - M4B) / B3`，正值表示 candidate 更好。",
        "",
        "| target | B3 pooled / macro / latest | M4B pooled / macro / latest | M4B origin wins/losses/ties |",
        "|---|---:|---:|---:|",
        f"| resident | {fmt(resident['b3_pooled_mse'])} / {fmt(resident['b3_macro_origin_mse'])} / {fmt(resident['b3_latest_2021_mse'])} | {fmt(resident['candidate_pooled_mse'])} / {fmt(resident['candidate_macro_origin_mse'])} / {fmt(resident['candidate_latest_2021_mse'])} | {resident['origin_wins']}/{resident['origin_losses']}/{resident['origin_ties']} |",
        f"| registered | {fmt(registered['b3_pooled_mse'])} / {fmt(registered['b3_macro_origin_mse'])} / {fmt(registered['b3_latest_2021_mse'])} | {fmt(registered['candidate_pooled_mse'])} / {fmt(registered['candidate_macro_origin_mse'])} / {fmt(registered['candidate_latest_2021_mse'])} | {registered['origin_wins']}/{registered['origin_losses']}/{registered['origin_ties']} |",
        "",
        "## Handoff questions",
        "",
        f"1. **wage R1/R2 是否确实等价？** 是。S2 required lag 为 R1=`{lags[R1_ID]}`、R2=`{lags[R2_ID]}`，2011–2021 每个 origin 的 source cap 也完全一致；因此没有重复 fit R2。",
        f"2. **M4B resident 是否超过 B3？** {'是，进入 deep-dive 候选。' if resident['candidate_beats_b3'] else '否。'} B3 pooled MSE=`{fmt(resident['b3_pooled_mse'])}`，M4B=`{fmt(resident['candidate_pooled_mse'])}`，relative improvement=`{fmt(resident['relative_improvement_pct'])}%`。",
        f"3. **registered 是否支持？** {'支持。' if registered['candidate_beats_b3'] else '不支持。'} B3 pooled MSE=`{fmt(registered['b3_pooled_mse'])}`，M4B=`{fmt(registered['candidate_pooled_mse'])}`，relative improvement=`{fmt(registered['relative_improvement_pct'])}%`。",
        f"4. **per-origin wins/losses？** resident=`{resident['origin_wins']}/{resident['origin_losses']}/{resident['origin_ties']}`，registered=`{registered['origin_wins']}/{registered['origin_losses']}/{registered['origin_ties']}`，顺序为 wins/losses/ties。",
        f"5. **latest 2021 是否支持或冲突？** resident M4B latest=`{fmt(resident['candidate_latest_2021_mse'])}` vs B3=`{fmt(resident['b3_latest_2021_mse'])}`，relative improvement=`{fmt(resident['latest_relative_improvement_pct'])}%`；registered M4B latest=`{fmt(registered['candidate_latest_2021_mse'])}` vs B3=`{fmt(registered['b3_latest_2021_mse'])}`，relative improvement=`{fmt(registered['latest_relative_improvement_pct'])}%`。latest 只作 proximity diagnostic，不能覆盖 pooled 结论。",
        f"6. **invalidated value 是否零回流？** 是。canonical invalidated key 的 feature-used value reintroduction count=`{summary['invalidated_value_reintroduced_count']}`，used-key count=`{summary['invalidated_key_used_count']}`；candidate-hit 只记录 allowed-history 交集，不把 raw 异常值带回模型。guard=`{status}`。",
        f"7. **是否值得 deep dive？** `deep_dive_recommended={str(summary['deep_dive_recommended']).lower()}`。判定依据：resident pooled hurdle、origin majority 与 latest proximity；本 worker 不自动生成 leave-one-origin/city 或 concentration artifacts。",
        "",
        "## Verification and evidence index",
        "",
        f"- B3 row compatibility：`{'PASS' if b3_row_pass else 'FAIL'}`；B3 pooled/macro/latest metric compatibility：`{'PASS' if b3_metric_pass else 'FAIL'}`。",
        f"- wage boundary / invalidation / T-2 / preprocessing guards：`{'PASS' if summary['all_boundary_guards_pass'] else 'FAIL'}`。",
        "- `screening_metrics.csv`：aggregate、per-origin、pairwise、R1/R2 equivalence 与 focused sanity checks。",
        "- `wage_boundary_audit.csv`：required cap、actual latest year、staleness、feature source years、invalidated candidate-hit 与 zero-reintroduction boundary。",
        "- `plan04b_summary.json`：机器可读 screening decision、metrics、scope flags 与 evidence counts。",
        "",
        "## Failure / deviation / remaining uncertainty",
        "",
        "- Plan04B 的 R1/R2 wage cap 等价性使单次 M4B fit 足够；这是计划明确授权的 workload reduction，不是遗漏 R2。",
        "- source publication timing 未被证明；R1/R2 仍是 S2 observed-cap replay assumptions，不应解释为真实发布时间因果。",
        "- wage history 的 missingness / invalidation 保持显式；缺失只由 outer-training fold 的 median + missing indicator 处理，未插值、future fill 或回填 raw quarantine value。",
        "- deep dive 未执行；不据此宣布 S4 complete 或进入 S5。",
    ]
    (OUTPUT_DIR / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_starting_commit_is_ancestor()
    execution_head = git_head()
    population, population_profile, geometry, wage_series, wage_geometry, invalidated_keys = load_context()
    equivalence_rows, scenario_equivalent, lags = make_equivalence_rows(wage_geometry)
    assert scenario_equivalent

    b3_rows, candidate_rows, diagnostic_bundle = build_rows_and_audit(
        population,
        wage_series,
        wage_geometry,
        invalidated_keys,
    )
    audit_rows = diagnostic_bundle.pop("audit_rows")
    all_model_rows = b3_rows + candidate_rows
    per_origin, aggregate, per_origin_map, aggregate_map = calculate_metrics(all_model_rows)
    pairwise = pairwise_summary(all_model_rows, per_origin_map, aggregate_map)

    b3_row_pass, b3_metric_pass, b3_checks = b3_compatibility(b3_rows, population)
    sanity_rows = make_sanity_rows(
        b3_row_pass,
        b3_metric_pass,
        diagnostic_bundle,
        audit_rows,
    )
    screening = {}
    for target in TARGETS:
        target_id = target["target_id"]
        base = aggregate_map[(target_id, R1_ID, B3_ID)]
        candidate = aggregate_map[(target_id, R1_ID, M4B_ID)]
        pair = pairwise[target_id]
        screening[target_id] = {
            "b3_pooled_mse": float(base["pooled_city_year_mse"]),
            "b3_macro_origin_mse": float(base["macro_origin_mse"]),
            "b3_latest_2021_mse": float(base["latest_origin_mse"]),
            "candidate_pooled_mse": float(candidate["pooled_city_year_mse"]),
            "candidate_macro_origin_mse": float(candidate["macro_origin_mse"]),
            "candidate_latest_2021_mse": float(candidate["latest_origin_mse"]),
            "relative_improvement_pct": float(
                pair["pooled_mse_relative_improvement_pct"]
            ),
            "latest_relative_improvement_pct": float(
                pair["latest_mse_relative_improvement_pct"]
            ),
            "origin_wins": int(pair["candidate_origin_win_count"]),
            "origin_losses": int(pair["candidate_origin_loss_count"]),
            "origin_ties": int(pair["origin_tie_count"]),
            "candidate_beats_b3": bool(
                candidate["pooled_city_year_mse"] < base["pooled_city_year_mse"]
            ),
            "scoreable_city_year_count": int(
                candidate["total_scoreable_city_year_count"]
            ),
        }

    resident_screen = screening["resident"]
    resident_within_one_pct = (
        resident_screen["relative_improvement_pct"] >= -1.0
    )
    resident_origin_majority = resident_screen["origin_wins"] > resident_screen["origin_losses"]
    resident_latest_not_severe_reversal = (
        resident_screen["latest_relative_improvement_pct"] >= -1.0
    )
    deep_dive_recommended = bool(
        resident_screen["candidate_beats_b3"]
        or (
            resident_within_one_pct
            and resident_origin_majority
            and resident_latest_not_severe_reversal
        )
    )
    screening_decision = (
        "promising_for_lab_deep_dive" if deep_dive_recommended else "unsupported"
    )

    invalidated_value_reintroduced_count = sum(
        row["wage_invalidated_value_reintroduced"] == "yes"
        for row in audit_rows
    )
    invalidated_key_used_count = sum(
        int(row["wage_invalidated_key_used_count"] or 0) for row in audit_rows
    )
    all_sanity_pass = all(row["status"] == "PASS" for row in sanity_rows)
    summary = {
        "plan_id": "S4-Plan04B",
        "stage": "S4 Feature & ML Modeling",
        "branch": "s4/wave-a-wage",
        "starting_commit": STARTING_COMMIT,
        "evidence_generation_head": execution_head,
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "city_count": len(CITIES),
        "primary_target": "resident",
        "sensitivity_target": "registered",
        "target_history_cutoff_rule": "population target history <= T-2",
        "strongest_fixed_baseline": B3_ID,
        "candidate_id": M4B_ID,
        "feature_group_id": "WAGE_RECENT_V1",
        "feature_names": list(FEATURE_NAMES),
        "model_definitions": list(MODEL_DEFINITIONS),
        "wage_source_id": WAGE_SOURCE_ID,
        "wage_geometry": wage_geometry,
        "wage_required_lag_years": lags,
        "scenario_equivalent": scenario_equivalent,
        "candidate_model_run_scenario": R1_ID,
        "r2_candidate_model_rerun": False,
        "r1_r2_cap_mismatch_count": 0,
        "screening": screening,
        "screening_decision": screening_decision,
        "deep_dive_recommended": deep_dive_recommended,
        "invalidated_key_count": len(invalidated_keys),
        "invalidated_value_reintroduced_count": invalidated_value_reintroduced_count,
        "invalidated_key_used_count": invalidated_key_used_count,
        "wage_invalidated_key_guard_pass": (
            invalidated_value_reintroduced_count == 0
            and invalidated_key_used_count == 0
        ),
        "population_t_minus_2_audit_pass": diagnostic_bundle[
            "population_t_minus_2_fail_count"
        ]
        == 0,
        "preprocessing_fit_boundary_pass": diagnostic_bundle[
            "preprocessing_fit_rows_mismatch_count"
        ]
        == 0,
        "wage_boundary_audit_pass": diagnostic_bundle["boundary_fail_count"] == 0,
        "all_boundary_guards_pass": all_sanity_pass,
        "candidate_prediction_rows": len(candidate_rows),
        "training_audit_rows": diagnostic_bundle["training_audit_rows"],
        "validation_audit_rows": diagnostic_bundle["validation_audit_rows"],
        "validation_feature_missing_cell_count": diagnostic_bundle[
            "validation_feature_missing_cell_count"
        ],
        "b3_row_compatibility_pass": b3_row_pass,
        "b3_metric_compatibility_pass": b3_metric_pass,
        "s4_complete": False,
        "s5_entered": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "tuning_run": False,
        "gbdt_run": False,
        "other_covariates_used": False,
        "city_identity_used": False,
        "population_profile": population_profile,
        "evidence_files": [
            "HANDOFF.md",
            "plan04b_summary.json",
            "screening_metrics.csv",
            "wage_boundary_audit.csv",
            "scripts/build_s4_plan04b_wage_screen.py",
        ],
        "b3_checks": b3_checks,
        "sanity_checks_pass": all_sanity_pass,
    }

    metrics_rows = screening_rows(
        aggregate,
        per_origin,
        pairwise,
        equivalence_rows,
        sanity_rows,
    )
    write_csv(OUTPUT_DIR / "screening_metrics.csv", metrics_rows, SCREENING_FIELDS)
    write_csv(OUTPUT_DIR / "wage_boundary_audit.csv", audit_rows, AUDIT_FIELDS)
    write_json(OUTPUT_DIR / "plan04b_summary.json", summary)
    write_handoff(summary, pairwise, lags, b3_row_pass, b3_metric_pass)

    assert all(row["status"] == "PASS" for row in equivalence_rows)
    assert summary["wage_invalidated_key_guard_pass"]
    assert summary["population_t_minus_2_audit_pass"]
    assert summary["preprocessing_fit_boundary_pass"]
    assert summary["wage_boundary_audit_pass"]
    assert summary["b3_row_compatibility_pass"]
    assert summary["b3_metric_compatibility_pass"]
    assert summary["s4_complete"] is False
    assert summary["s5_entered"] is False


if __name__ == "__main__":
    main()
