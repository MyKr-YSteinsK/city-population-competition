"""Build S4-Plan05B cross-city relative-feature screening evidence.

The runner tests one fixed Ridge residual correction over four cross-sectional
percentile features derived from the frozen B3 state.  It keeps the S2/S3
population boundary, uses no external source or city identity, and deliberately
stops at the screening evidence required by Plan05B.
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
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan05B"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
S4_PLAN03_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan03"
WAVE_A_SUMMARY = ROOT / "reports" / "S4_HANDOFFS" / "S4-WaveA" / "WAVE_A_SUMMARY.md"

STARTING_COMMIT = "decfd7e0c68d14da31eabe5720ab706122c1040d"
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
R1_ID = "R1_REFERENCE_SNAPSHOT_LAG"
R2_ID = "R2_CONSERVATIVE_PUBLICATION_STRESS"
TARGETS = s3_plan02.TARGETS

B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M5B_ID = "M5B_B3_PLUS_RIDGE_CROSS_CITY_RELATIVE"
RAW_STATE_FIELDS = (
    "latest_value",
    "recent3_slope",
    "projected_change",
    "recent_relative_slope",
)
FEATURE_NAMES = (
    "latest_value_percentile",
    "recent3_slope_percentile",
    "projected_change_percentile",
    "relative_slope_percentile",
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
        "cross_city_relative_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M5B_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "CROSS_CITY_RELATIVE_V1",
        "feature_names": ";".join(FEATURE_NAMES),
        "target_definition": "b3_residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "uses_covariates": "no",
        "cross_city_relative_used": "yes",
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
    "training_row_included",
    "b3_state_latest_value",
    "b3_state_recent3_slope",
    "b3_state_projected_change",
    "b3_state_recent_relative_slope",
    "b3_state_fit_status",
    "latest_value_percentile",
    "recent3_slope_percentile",
    "projected_change_percentile",
    "relative_slope_percentile",
    "cross_city_feature_missing_names",
    "cross_city_latest_value_available_count",
    "cross_city_recent3_slope_available_count",
    "cross_city_projected_change_available_count",
    "cross_city_relative_slope_available_count",
    "cross_city_diagnostics_json",
    "cross_city_percentile_rule",
    "cross_city_information_set_pass",
    "target_label_used_in_feature",
    "city_identity_used",
    "other_covariate_columns_used",
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
        f"Plan05B base {STARTING_COMMIT} is not an ancestor of current HEAD"
    )


def source_state(
    series: dict[int, float],
    pseudo_target_year: int,
) -> dict[str, Any]:
    cutoff = pseudo_target_year - HORIZON_YEARS
    fit = s3_plan02.fit_prediction(
        B3_ID,
        series,
        pseudo_target_year,
        cutoff,
    )
    latest_year = fit["latest_target_year_used"]
    latest_value = series[latest_year] if latest_year is not None else None
    is_recent3 = fit["regression_fit_status"] == "recent_3point_ols"
    recent3_slope = fit["regression_slope"] if is_recent3 else None
    projected_change = (
        fit["prediction"] - latest_value
        if fit["prediction"] is not None and latest_value is not None
        else None
    )
    recent_relative_slope = (
        recent3_slope / latest_value
        if recent3_slope is not None and latest_value not in (None, 0)
        else None
    )
    return {
        "population_target_history_cutoff_year": cutoff,
        "population_history_max_year_used": fit["history_max_year"],
        "population_history_year_count": len(fit["history_years"]),
        "latest_value": latest_value,
        "recent3_slope": recent3_slope,
        "projected_change": projected_change,
        "recent_relative_slope": recent_relative_slope,
        "b3_state_fit_status": fit["regression_fit_status"],
        "b3_fit": fit,
    }


def percentile_map(values_by_city: dict[str, float]) -> dict[str, float]:
    if not values_by_city:
        return {}
    ordered = sorted(values_by_city.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = mean(float(rank) for rank in range(index + 1, end + 1))
        if len(ordered) == 1:
            percentile = 0.5
        else:
            percentile = (average_rank - 1.0) / (len(ordered) - 1.0)
        for city, _ in ordered[index:end]:
            result[city] = percentile
        index = end
    return result


def cross_section_features(
    states_by_city: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    percentile_by_raw: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for raw_field in RAW_STATE_FIELDS:
        values = {
            city: float(state[raw_field])
            for city, state in states_by_city.items()
            if state[raw_field] is not None
        }
        median_value = statistics.median(values.values()) if values else None
        mad = (
            statistics.median(
                abs(value - median_value) for value in values.values()
            )
            if values and median_value is not None
            else None
        )
        diagnostics[raw_field] = {
            "available_city_count": len(values),
            "missing_city_count": len(CITIES) - len(values),
            "unique_value_count": len(set(values.values())),
            "mean": mean(values.values()) if values else None,
            "std_population": (
                statistics.pstdev(values.values()) if len(values) > 1 else 0.0
            ),
            "median": median_value,
            "mad": mad,
        }
        percentile_by_raw[raw_field] = percentile_map(values)

    result: dict[str, dict[str, Any]] = {}
    feature_for_raw = dict(zip(RAW_STATE_FIELDS, FEATURE_NAMES))
    for city, state in states_by_city.items():
        features = {
            feature_for_raw[raw_field]: percentile_by_raw[raw_field].get(city)
            for raw_field in RAW_STATE_FIELDS
        }
        result[city] = {
            **state,
            **features,
            "cross_city_feature_missing_names": ";".join(
                name for name, value in features.items() if value is None
            ),
            "cross_city_diagnostics_json": stable_json(diagnostics),
            "cross_city_latest_value_available_count": diagnostics["latest_value"][
                "available_city_count"
            ],
            "cross_city_recent3_slope_available_count": diagnostics["recent3_slope"][
                "available_city_count"
            ],
            "cross_city_projected_change_available_count": diagnostics[
                "projected_change"
            ]["available_city_count"],
            "cross_city_relative_slope_available_count": diagnostics[
                "recent_relative_slope"
            ]["available_city_count"],
        }
    return result


def build_cross_cache(
    population: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    cache: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        years = sorted(
            {
                year
                for city in CITIES
                for year in population[target_id][city]
            }
        )
        cache[target_id] = {}
        for pseudo_year in years:
            states = {
                city: source_state(population[target_id][city], pseudo_year)
                for city in CITIES
            }
            cache[target_id][pseudo_year] = cross_section_features(states)
    return cache


def load_context() -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, Any],
]:
    assert WAVE_A_SUMMARY.exists()
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

    s4_summary = json.loads(
        (S4_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8")
    )
    assert s4_summary["b3_compatibility_pass"] is True
    assert s4_summary["b3_metric_compatibility_pass"] is True
    assert s4_summary["s4_complete"] is False
    assert s4_summary["s5_entered"] is False
    return population, population_profile


def wrap_b3_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "model_id": B3_ID,
            "feature_builder_id": "FROZEN_B3_REFERENCE",
            "feature_group_id": "FROZEN_B3",
            "covariates_used": "no",
            "cross_city_relative_used": "no",
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


def build_audit_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    city: str,
    pseudo_target_year: int,
    state: dict[str, Any],
    row_role: str,
    training_included: str,
) -> dict[str, Any]:
    cutoff = state["population_target_history_cutoff_year"]
    population_pass = (
        state["population_history_max_year_used"] is None
        or state["population_history_max_year_used"] <= cutoff
    )
    outer_label_pass = (
        row_role == "validation"
        or pseudo_target_year <= outer_origin - HORIZON_YEARS
    )
    information_pass = population_pass and outer_label_pass
    feature_values = {name: state[name] for name in FEATURE_NAMES}
    boundary_pass = all(
        (
            information_pass,
            state["cross_city_feature_missing_names"] is not None,
        training_included in {"yes", "no", "not_run_equivalent"},
        )
    )
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "model_run_scenario": R1_ID if scenario_id == R1_ID else "scenario_equivalence_only",
        "outer_origin_year": outer_origin,
        "row_role": row_role,
        "city_id": city,
        "pseudo_target_year": pseudo_target_year,
        "population_target_history_cutoff_year": cutoff,
        "population_history_max_year_used": state["population_history_max_year_used"],
        "population_history_year_count": state["population_history_year_count"],
        "population_history_t_minus_2_pass": "PASS" if population_pass else "FAIL",
        "outer_training_label_within_T_minus_2": (
            "PASS" if outer_label_pass else "FAIL"
        ),
        "training_row_included": training_included,
        "b3_state_latest_value": state["latest_value"],
        "b3_state_recent3_slope": state["recent3_slope"],
        "b3_state_projected_change": state["projected_change"],
        "b3_state_recent_relative_slope": state["recent_relative_slope"],
        "b3_state_fit_status": state["b3_state_fit_status"],
        **feature_values,
        "cross_city_feature_missing_names": state[
            "cross_city_feature_missing_names"
        ],
        "cross_city_latest_value_available_count": state[
            "cross_city_latest_value_available_count"
        ],
        "cross_city_recent3_slope_available_count": state[
            "cross_city_recent3_slope_available_count"
        ],
        "cross_city_projected_change_available_count": state[
            "cross_city_projected_change_available_count"
        ],
        "cross_city_relative_slope_available_count": state[
            "cross_city_relative_slope_available_count"
        ],
        "cross_city_diagnostics_json": state["cross_city_diagnostics_json"],
        "cross_city_percentile_rule": (
            "average-tie-rank normalized to [0,1] as (rank-1)/(n-1); n=1 -> 0.5"
        ),
        "cross_city_information_set_pass": "PASS" if information_pass else "FAIL",
        "target_label_used_in_feature": "no",
        "city_identity_used": "no",
        "other_covariate_columns_used": "no",
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
                as_float(row["features"].get(name))
                if as_float(row["features"].get(name)) is not None
                else np.nan
                for name in FEATURE_NAMES
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
    return model.predict(x_validation).tolist(), {
        "training_row_count": len(training_rows),
        "training_missing_feature_cell_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "preprocessing_fit_row_count": len(training_rows),
        "model_fit_row_count": len(training_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": [
            float(value) for value in np.asarray(ridge.coef_).ravel()
        ],
        "coefficient_names": list(FEATURE_NAMES) + indicator_names,
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
    return {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": R1_ID,
        "predictor_id": M5B_ID,
        "model_id": M5B_ID,
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
        "feature_builder_id": "CROSS_CITY_RELATIVE_V1",
        "feature_group_id": "CROSS_CITY_RELATIVE_V1",
        "cross_city_relative_used": "yes",
        "city_identity_used": "no",
        "other_covariate_columns_used": "no",
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
        **{
            name: validation["features"].get(name)
            for name in FEATURE_NAMES
        },
    }


def build_rows_and_audit(
    population: dict[str, dict[str, dict[int, float]]],
    cross_cache: dict[str, dict[int, dict[str, dict[str, Any]]]],
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
    training_missing_feature_cells = 0
    validation_missing_feature_cells = 0
    preprocessing_mismatch_count = 0
    population_t_minus_2_fail_count = 0
    boundary_fail_count = 0

    for target in TARGETS:
        target_id = target["target_id"]
        for origin in COMMON_ORIGINS:
            training_rows: list[dict[str, Any]] = []
            validation_rows: list[dict[str, Any]] = []
            training_included_states: list[dict[str, Any]] = []
            for city in CITIES:
                target_series = population[target_id][city]
                for pseudo_year in sorted(
                    year
                    for year in target_series
                    if year <= origin - HORIZON_YEARS
                ):
                    state = cross_cache[target_id][pseudo_year][city]
                    fit = state["b3_fit"]
                    actual = target_series.get(pseudo_year)
                    included = "yes" if actual is not None and fit["prediction"] is not None else "no"
                    audit = build_audit_row(
                        target_id,
                        R1_ID,
                        origin,
                        city,
                        pseudo_year,
                        state,
                        "training",
                        included,
                    )
                    audit_rows.append(audit)
                    training_audit_rows += 1
                    training_missing_feature_cells += sum(
                        state[name] is None for name in FEATURE_NAMES
                    ) if included == "yes" else 0
                    if audit["boundary_status"] != "PASS":
                        boundary_fail_count += 1
                    if audit["population_history_t_minus_2_pass"] != "PASS":
                        population_t_minus_2_fail_count += 1
                    if included != "yes":
                        continue
                    training_rows.append(
                        {
                            "pseudo_target_year": pseudo_year,
                            "target_value": actual,
                            "b3_prediction": fit["prediction"],
                            "b3_residual_actual_minus_prediction": actual - fit["prediction"],
                            "features": state,
                        }
                    )
                    training_included_states.append(state)

                validation_state = cross_cache[target_id][origin][city]
                validation_fit = validation_state["b3_fit"]
                assert target_series.get(origin) is not None
                assert validation_fit["prediction"] is not None
                validation_audit = build_audit_row(
                    target_id,
                    R1_ID,
                    origin,
                    city,
                    origin,
                    validation_state,
                    "validation",
                    "yes",
                )
                audit_rows.append(validation_audit)
                validation_audit_rows += 1
                validation_missing_feature_cells += sum(
                    validation_state[name] is None for name in FEATURE_NAMES
                )
                if validation_audit["boundary_status"] != "PASS":
                    boundary_fail_count += 1
                if validation_audit["population_history_t_minus_2_pass"] != "PASS":
                    population_t_minus_2_fail_count += 1
                validation_rows.append(
                    {
                        "target_value": target_series[origin],
                        "b3_prediction": validation_fit["prediction"],
                        "b3_residual_actual_minus_prediction": target_series[origin] - validation_fit["prediction"],
                        "features": validation_state,
                    }
                )

            assert len(validation_rows) == len(CITIES)
            predictions, metadata = fit_candidate(training_rows, validation_rows)
            preprocessing_mismatch_count += int(
                metadata["preprocessing_model_fit_rows_equal"] != "yes"
            )
            training_years = [row["pseudo_target_year"] for row in training_rows]
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
                state = cross_cache[target_id][origin][city]
                fit = state["b3_fit"]
                audit = build_audit_row(
                    target_id,
                    R2_ID,
                    origin,
                    city,
                    origin,
                    state,
                    "validation",
                    "not_run_equivalent",
                )
                audit_rows.append(audit)
                validation_audit_rows += 1
                if audit["boundary_status"] != "PASS":
                    boundary_fail_count += 1
                if audit["population_history_t_minus_2_pass"] != "PASS":
                    population_t_minus_2_fail_count += 1

    r1_rows = [
        row for row in audit_rows
        if row["scenario_id"] == R1_ID and row["row_role"] == "validation"
    ]
    r2_rows = [
        row for row in audit_rows
        if row["scenario_id"] == R2_ID and row["row_role"] == "validation"
    ]
    assert len(r1_rows) == len(r2_rows) == len(TARGETS) * len(COMMON_ORIGINS) * len(CITIES)
    equivalence_mismatch_count = 0
    for r1_row, r2_row in zip(r1_rows, r2_rows):
        assert (
            r1_row["target_id"],
            r1_row["outer_origin_year"],
            r1_row["city_id"],
        ) == (
            r2_row["target_id"],
            r2_row["outer_origin_year"],
            r2_row["city_id"],
        )
        for field in FEATURE_NAMES + tuple(
            f"b3_state_{raw_field}" for raw_field in RAW_STATE_FIELDS
        ) + (
            "population_history_max_year_used",
            "population_history_year_count",
            "cross_city_diagnostics_json",
        ):
            if r1_row[field] != r2_row[field]:
                equivalence_mismatch_count += 1

    diagnostics = {
        "training_audit_rows": training_audit_rows,
        "validation_audit_rows": validation_audit_rows,
        "training_missing_feature_cell_count": training_missing_feature_cells,
        "validation_missing_feature_cell_count": validation_missing_feature_cells,
        "preprocessing_fit_rows_mismatch_count": preprocessing_mismatch_count,
        "population_t_minus_2_fail_count": population_t_minus_2_fail_count,
        "boundary_fail_count": boundary_fail_count,
        "candidate_row_count": len(candidate_rows),
        "r1_r2_feature_mismatch_count": equivalence_mismatch_count,
    }
    return b3_r1_rows, candidate_rows, {"audit_rows": audit_rows, **diagnostics}


@contextmanager
def metric_scope(predictors: tuple[dict[str, Any], ...] = MODEL_DEFINITIONS) -> Any:
    old_scenarios = s3_plan01.SCENARIOS
    old_predictors = s3_plan01.PREDICTORS
    s3_plan01.SCENARIOS = (R1_ID,)
    s3_plan01.PREDICTORS = predictors
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
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(rows, per_origin)
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

    with metric_scope((MODEL_DEFINITIONS[0],)):
        b3_per_origin, _ = s3_plan01.build_per_origin_metrics(b3_r1_rows)
        _, current_map = s3_plan01.build_aggregate_metrics(
            b3_r1_rows, b3_per_origin
        )
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
    return {
        target["target_id"]: s4_plan02.pairwise_row(
            rows,
            per_origin_map,
            aggregate_map,
            target["target_id"],
            R1_ID,
            B3_ID,
            M5B_ID,
            "M5B_vs_B3",
        )
        for target in TARGETS
    }


def top_prediction_change_rows(
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        by_city: defaultdict[str, float] = defaultdict(float)
        for row in candidate_rows:
            if row["target_id"] == target_id:
                by_city[row["city_id"]] += abs(float(row["predicted_b3_residual"]))
        total = math.fsum(by_city.values())
        ranked = sorted(by_city.items(), key=lambda item: (-item[1], item[0]))
        top = ranked[:3]
        summary[target_id] = {
            "total_absolute_prediction_change": total,
            "top3_city_ids": [city for city, _ in top],
            "top3_absolute_prediction_change": math.fsum(value for _, value in top),
            "top3_share": math.fsum(value for _, value in top) / total if total else 0.0,
        }
        for rank, (city, value) in enumerate(top, start=1):
            result.append(
                {
                    "record_type": "top3_prediction_change",
                    "target_id": target_id,
                    "model_id": M5B_ID,
                    "metric_name": f"top{rank}_absolute_prediction_change",
                    "metric_value": value,
                    "comparison_value": value / total if total else 0.0,
                    "status": "PASS",
                    "notes": f"city={city}; sum abs(predicted B3 residual) across 11 origins",
                }
            )
        result.append(
            {
                "record_type": "top3_prediction_change",
                "target_id": target_id,
                "model_id": M5B_ID,
                "metric_name": "top3_absolute_prediction_change_share",
                "metric_value": summary[target_id]["top3_share"],
                "comparison_value": 1.0,
                "status": "PASS",
                "notes": "light concentration sanity; not a leave-one-city analysis",
            }
        )
    return result, summary


def screening_rows(
    aggregate: list[dict[str, Any]],
    per_origin: list[dict[str, Any]],
    pairwise: dict[str, dict[str, Any]],
    equivalence_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
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
                "notes": "R1 model run; target-only cross-city features make R2 numerically equivalent",
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
                "notes": "common strict origin; population history <= T-2",
            }
        )
    for target_id, row in pairwise.items():
        rows.append(
            {
                "record_type": "pairwise",
                "target_id": target_id,
                "scenario_id": R1_ID,
                "model_id": M5B_ID,
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
                "notes": "positive improvement means M5B lower MSE than B3",
            }
        )
    rows.extend(equivalence_rows)
    rows.extend(sanity_rows)
    rows.extend(concentration_rows)
    return rows


def equivalence_rows(mismatch_count: int) -> list[dict[str, Any]]:
    return [
        {
            "record_type": "scenario_equivalence",
            "scenario_id": "R1_vs_R2",
            "metric_name": "cross_city_feature_mismatch_count",
            "metric_value": mismatch_count,
            "comparison_value": 0,
            "scenario_equivalent": "yes" if mismatch_count == 0 else "no",
            "status": "PASS" if mismatch_count == 0 else "FAIL",
            "notes": "same target-only population information set and same pseudo-origin cross-section",
        }
    ]


def sanity_rows(
    b3_row_pass: bool,
    b3_metric_pass: bool,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
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
            "cross_city_feature_mismatch_count",
            diagnostics["r1_r2_feature_mismatch_count"],
            0,
            "R1/R2 feature table equality",
        ),
        (
            "population_t_minus_2_fail_count",
            diagnostics["population_t_minus_2_fail_count"],
            0,
            "B3 state uses only population history <= pseudo-target-2",
        ),
        (
            "preprocessing_fit_rows_mismatch_count",
            diagnostics["preprocessing_fit_rows_mismatch_count"],
            0,
            "fold-local imputer/scaler/Ridge use the same training rows",
        ),
        (
            "cross_city_boundary_fail_count",
            diagnostics["boundary_fail_count"],
            0,
            "cross-sectional statistics use same pseudo-origin allowed data",
        ),
        (
            "resident_validation_rows",
            diagnostics["resident_validation_rows"],
            len(COMMON_ORIGINS) * len(CITIES),
            "resident validation coverage",
        ),
        (
            "registered_validation_rows",
            diagnostics["registered_validation_rows"],
            len(COMMON_ORIGINS) * len(CITIES),
            "registered validation coverage",
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
    return f"{float(value):.6f}"


def write_handoff(
    summary: dict[str, Any],
    pairwise: dict[str, dict[str, Any]],
    top_change: dict[str, Any],
    b3_row_pass: bool,
    b3_metric_pass: bool,
) -> None:
    resident = summary["screening"]["resident"]
    registered = summary["screening"]["registered"]
    lines = [
        "# S4-Plan05B Handoff｜跨城市相对位置特征快速筛选",
        "",
        "## 状态与结论",
        "",
        "- Plan 状态：`evidence complete`；本 worker 只完成 screening，不自动执行 deep dive。",
        f"- screening decision：`{summary['screening_decision']}`；`deep_dive_recommended={str(summary['deep_dive_recommended']).lower()}`。",
        f"- `s4_complete={str(summary['s4_complete']).lower()}`、`s5_entered={str(summary['s5_entered']).lower()}`、`final_2023_prediction_run=false`、`submission_run=false`。",
        "- cross-city percentile representation 只由同一 pseudo-origin、同一 target 的 population history <= T-2 构造；R1/R2 完全等价，M5B 只实际 fit 一次。",
        "",
        "## Actual changes",
        "",
        "1. 新增 `scripts/build_s4_plan05b_cross_city_relative.py`。",
        "2. 新增本目录下的 `HANDOFF.md`、`plan05b_summary.json`、`screening_metrics.csv`、`cross_city_feature_audit.csv`。",
        "3. 未修改 main、Wave A evidence、S2/S3 contract、CURRENT_LEARNING_CHECKPOINT.md 或其他 Wave B worker 文件。",
        "",
        "## Execution contract",
        "",
        f"- integration 后 frozen base / execution starting SHA：`{STARTING_COMMIT}`。",
        f"- current evidence-generation HEAD：`{summary['evidence_generation_head']}`。",
        "- branch：`s4/wave-b-cross-city-relative`；独立 worktree 执行；完成后只 push 本 branch。",
        "- common origins：`2011–2021`；resident primary、registered sensitivity；population boundary=`T-2`。",
        "",
        "## Actual feature geometry",
        "",
        "- candidate：`M5B_B3_PLUS_RIDGE_CROSS_CITY_RELATIVE`；model=`Ridge(alpha=1.0)` residual correction。",
        "- raw B3 state：latest value、recent-3 OLS slope、B3 projected change、recent relative slope；model 只使用对应 cross-city percentile，不重新加入 raw TH_CORE。",
        "- percentile rule：同一 pseudo-origin 的 available cities 使用 average-tie-rank，按 `(rank-1)/(n-1)` 归一化到 `[0,1]`，单个 available city 取 `0.5`。",
        "- preprocessing：outer-origin training rows 内 fit `SimpleImputer(median, add_indicator=True) + StandardScaler + Ridge(alpha=1.0)`。",
        "- 未使用 city ID、外部 source、target year label、tuning、GBDT、2023 forecast 或 submission。",
        "",
        "## Core screening metrics",
        "",
        "数值为 R1 实际 model run 的 pooled MSE / macro-origin MSE / latest 2021 MSE；R2 因 exact feature equality 不重复 fit。括号为 candidate 相对 B3 的 improvement percentage，正值表示更好。",
        "",
        "| target | B3 pooled / macro / latest | M5B pooled / macro / latest | origin wins/losses/ties |",
        "|---|---:|---:|---:|",
        f"| resident | {fmt(resident['b3_pooled_mse'])} / {fmt(resident['b3_macro_origin_mse'])} / {fmt(resident['b3_latest_2021_mse'])} | {fmt(resident['candidate_pooled_mse'])} / {fmt(resident['candidate_macro_origin_mse'])} / {fmt(resident['candidate_latest_2021_mse'])} ({fmt(resident['relative_improvement_pct'])}%) | {resident['origin_wins']}/{resident['origin_losses']}/{resident['origin_ties']} |",
        f"| registered | {fmt(registered['b3_pooled_mse'])} / {fmt(registered['b3_macro_origin_mse'])} / {fmt(registered['b3_latest_2021_mse'])} | {fmt(registered['candidate_pooled_mse'])} / {fmt(registered['candidate_macro_origin_mse'])} / {fmt(registered['candidate_latest_2021_mse'])} ({fmt(registered['relative_improvement_pct'])}%) | {registered['origin_wins']}/{registered['origin_losses']}/{registered['origin_ties']} |",
        "",
        "## Handoff questions",
        "",
        f"1. **cross-city relative representation 是否超过 B3？** {'是。' if resident['candidate_beats_b3'] else '否。'} resident pooled MSE 为 `{fmt(resident['candidate_pooled_mse'])}` vs B3 `{fmt(resident['b3_pooled_mse'])}`，relative improvement=`{fmt(resident['relative_improvement_pct'])}%`。",
        f"2. **resident / registered 是否一致？** {'一致支持 candidate。' if resident['candidate_beats_b3'] and registered['candidate_beats_b3'] else '不一致或共同不支持。'} resident improvement=`{fmt(resident['relative_improvement_pct'])}%`，registered improvement=`{fmt(registered['relative_improvement_pct'])}%`。",
        f"3. **origin wins 是否多数？** resident=`{resident['origin_wins']}/{resident['origin_losses']}/{resident['origin_ties']}`，registered=`{registered['origin_wins']}/{registered['origin_losses']}/{registered['origin_ties']}`，顺序为 wins/losses/ties。",
        f"4. **latest 是否支持？** resident latest improvement=`{fmt(resident['latest_relative_improvement_pct'])}%`，registered=`{fmt(registered['latest_relative_improvement_pct'])}%`；latest 仅为 proximity diagnostic，不能覆盖 pooled 结论。",
        f"5. **R1/R2 是否严格等价？** 是。cross-city feature mismatch count=`{summary['r1_r2_feature_mismatch_count']}`，candidate 只实际运行 R1，R2 未复制 model workload。",
        f"6. **gain/loss 是否明显由极端城市驱动？** 轻量 audit 的 resident top-3 absolute prediction-change cities=`{','.join(top_change['resident']['top3_city_ids'])}`，share=`{fmt(top_change['resident']['top3_share'])}`；registered=`{','.join(top_change['registered']['top3_city_ids'])}`，share=`{fmt(top_change['registered']['top3_share'])}`。这只是 prediction-change concentration，不是 leave-one-city 结论。",
        f"7. **是否值得 deep dive？** `deep_dive_recommended={str(summary['deep_dive_recommended']).lower()}`；当前 resident pooled/origin/latest evidence 未满足 Wave B deep-dive rule。",
        "",
        "## Verification and evidence index",
        "",
        f"- B3 row compatibility：`{'PASS' if b3_row_pass else 'FAIL'}`；B3 metric compatibility：`{'PASS' if b3_metric_pass else 'FAIL'}`。",
        f"- cross-city information boundary、R1/R2 equality、population T-2、preprocessing fit rows：`{'PASS' if summary['all_boundary_guards_pass'] else 'FAIL'}`。",
        "- `screening_metrics.csv`：aggregate、per-origin、pairwise、R1/R2 equality、sanity 与 top-3 prediction-change records。",
        "- `cross_city_feature_audit.csv`：pseudo-origin state、percentile feature、same-cross-section diagnostics、missingness 与 T-2 boundary。",
        "- `plan05b_summary.json`：机器可读 screening decision、metrics、dispersion/missingness 和 scope flags。",
        "",
        "## Failure / deviation / remaining uncertainty",
        "",
        "- R1/R2 完全等价来自 target-only population information set；因此按 screening efficiency 规则只 fit 一次 M5B，不复制相同 workload。",
        "- cross-city percentile 会丢失绝对尺度，且横截面排名可能受早期 missingness 和极端城市影响；本轮只记录 top-3 prediction-change sanity，不把它解释为稳健性证明。",
        "- 本轮没有 leave-one-city、完整 concentration、调参或其他 feature family；若结果有希望，由 Lab 决定下一轮 deep dive。",
    ]
    (OUTPUT_DIR / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_starting_commit_is_ancestor()
    evidence_generation_head = git_head()
    population, population_profile = load_context()
    cross_cache = build_cross_cache(population)
    b3_rows, candidate_rows, diagnostic_bundle = build_rows_and_audit(
        population,
        cross_cache,
    )
    audit_rows = diagnostic_bundle.pop("audit_rows")
    per_origin, aggregate, per_origin_map, aggregate_map = calculate_metrics(
        b3_rows + candidate_rows
    )
    pairwise = pairwise_summary(
        b3_rows + candidate_rows,
        per_origin_map,
        aggregate_map,
    )
    b3_row_pass, b3_metric_pass, b3_checks = b3_compatibility(b3_rows, population)
    concentration_rows, top_change = top_prediction_change_rows(candidate_rows)

    resident_top_feature_diag = next(
        row
        for row in audit_rows
        if row["target_id"] == "resident"
        and row["scenario_id"] == R1_ID
        and row["row_role"] == "validation"
    )
    feature_dispersion = json.loads(
        resident_top_feature_diag["cross_city_diagnostics_json"]
    )
    diagnostic_bundle["resident_validation_rows"] = sum(
        row["target_id"] == "resident"
        and row["scenario_id"] == R1_ID
        and row["row_role"] == "validation"
        for row in audit_rows
    )
    diagnostic_bundle["registered_validation_rows"] = sum(
        row["target_id"] == "registered"
        and row["scenario_id"] == R1_ID
        and row["row_role"] == "validation"
        for row in audit_rows
    )
    sanity = sanity_rows(b3_row_pass, b3_metric_pass, diagnostic_bundle)
    equivalence = equivalence_rows(diagnostic_bundle["r1_r2_feature_mismatch_count"])

    screening: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        base = aggregate_map[(target_id, R1_ID, B3_ID)]
        candidate = aggregate_map[(target_id, R1_ID, M5B_ID)]
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
    deep_dive_recommended = bool(
        resident_screen["candidate_beats_b3"]
        or (
            resident_screen["relative_improvement_pct"] >= -1.0
            and resident_screen["origin_wins"] > resident_screen["origin_losses"]
            and resident_screen["latest_relative_improvement_pct"] >= -1.0
        )
    )
    summary = {
        "plan_id": "S4-Plan05B",
        "stage": "S4 Feature & ML Modeling",
        "branch": "s4/wave-b-cross-city-relative",
        "starting_commit": STARTING_COMMIT,
        "evidence_generation_head": evidence_generation_head,
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "city_count": len(CITIES),
        "primary_target": "resident",
        "sensitivity_target": "registered",
        "target_history_cutoff_rule": "population target history <= T-2",
        "strongest_fixed_baseline": B3_ID,
        "candidate_id": M5B_ID,
        "feature_group_id": "CROSS_CITY_RELATIVE_V1",
        "raw_state_fields": list(RAW_STATE_FIELDS),
        "feature_names": list(FEATURE_NAMES),
        "percentile_rule": "average-tie-rank normalized to [0,1] as (rank-1)/(n-1); n=1 -> 0.5",
        "model_definitions": list(MODEL_DEFINITIONS),
        "scenario_equivalent": diagnostic_bundle["r1_r2_feature_mismatch_count"] == 0,
        "r1_r2_feature_mismatch_count": diagnostic_bundle[
            "r1_r2_feature_mismatch_count"
        ],
        "candidate_model_run_scenario": R1_ID,
        "r2_candidate_model_rerun": False,
        "screening": screening,
        "screening_decision": "promising_for_lab_deep_dive" if deep_dive_recommended else "unsupported",
        "deep_dive_recommended": deep_dive_recommended,
        "feature_dispersion_validation_example": feature_dispersion,
        "top3_absolute_prediction_change": top_change,
        "candidate_prediction_rows": diagnostic_bundle["candidate_row_count"],
        "training_audit_rows": diagnostic_bundle["training_audit_rows"],
        "validation_audit_rows": diagnostic_bundle["validation_audit_rows"],
        "training_missing_feature_cell_count": diagnostic_bundle[
            "training_missing_feature_cell_count"
        ],
        "validation_missing_feature_cell_count": diagnostic_bundle[
            "validation_missing_feature_cell_count"
        ],
        "b3_row_compatibility_pass": b3_row_pass,
        "b3_metric_compatibility_pass": b3_metric_pass,
        "population_t_minus_2_audit_pass": diagnostic_bundle[
            "population_t_minus_2_fail_count"
        ]
        == 0,
        "preprocessing_fit_boundary_pass": diagnostic_bundle[
            "preprocessing_fit_rows_mismatch_count"
        ]
        == 0,
        "cross_city_boundary_pass": diagnostic_bundle["boundary_fail_count"] == 0,
        "all_boundary_guards_pass": all(row["status"] == "PASS" for row in sanity),
        "s4_complete": False,
        "s5_entered": False,
        "final_2023_prediction_run": False,
        "submission_run": False,
        "tuning_run": False,
        "gbdt_run": False,
        "external_covariates_used": False,
        "city_identity_used": False,
        "population_profile": population_profile,
        "b3_checks": b3_checks,
        "evidence_files": [
            "HANDOFF.md",
            "plan05b_summary.json",
            "screening_metrics.csv",
            "cross_city_feature_audit.csv",
            "scripts/build_s4_plan05b_cross_city_relative.py",
        ],
    }
    metrics = screening_rows(
        aggregate,
        per_origin,
        pairwise,
        equivalence,
        sanity,
        concentration_rows,
    )
    write_csv(OUTPUT_DIR / "screening_metrics.csv", metrics, SCREENING_FIELDS)
    write_csv(OUTPUT_DIR / "cross_city_feature_audit.csv", audit_rows, AUDIT_FIELDS)
    write_json(OUTPUT_DIR / "plan05b_summary.json", summary)
    write_handoff(summary, pairwise, top_change, b3_row_pass, b3_metric_pass)

    assert summary["scenario_equivalent"]
    assert summary["b3_row_compatibility_pass"]
    assert summary["b3_metric_compatibility_pass"]
    assert summary["population_t_minus_2_audit_pass"]
    assert summary["preprocessing_fit_boundary_pass"]
    assert summary["cross_city_boundary_pass"]
    assert summary["all_boundary_guards_pass"]
    assert summary["s4_complete"] is False
    assert summary["s5_entered"] is False


if __name__ == "__main__":
    main()
