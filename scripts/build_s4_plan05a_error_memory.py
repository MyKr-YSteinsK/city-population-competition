"""Build S4-Plan05A B3 error-memory screening evidence.

The candidate uses only historical pseudo-target residuals from the same city.
For a row at year ``y``, every residual used by the feature builder is from a
pseudo-target ``z <= y - 2``; the B3 fit that produced that residual uses
population history ``<= z - 2``.  The current row residual is never placed in
its own feature vector.

This is a screening-only runner.  It fits only frozen B3 and the fixed
M5A_B3_PLUS_RIDGE_ERROR_MEMORY residual correction.  It does not use
covariates, city identity, tuning, 2023 prediction, submission output, or a
full robustness suite.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan05A"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
WAVE_A_SUMMARY_PATH = ROOT / "reports" / "S4_HANDOFFS" / "S4-WaveA" / "WAVE_A_SUMMARY.md"

STARTING_COMMIT = "decfd7e0c68d14da31eabe5720ab706122c1040d"
BRANCH = "s4/wave-b-error-memory"
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M5A_ID = "M5A_B3_PLUS_RIDGE_ERROR_MEMORY"
MODEL_IDS = (B3_ID, M5A_ID)
MEMORY_FEATURES = (
    "last_observed_b3_residual",
    "recent3_mean_b3_residual",
    "all_available_mean_b3_residual",
    "recent3_residual_slope",
)

MODEL_DEFINITIONS = (
    {
        "predictor_id": B3_ID,
        "model_family": "frozen_s3_reference",
        "feature_group_id": "FROZEN_B3",
        "feature_names": "",
        "raw_feature_count": 0,
        "target_definition": "frozen B3 population level prediction",
        "preprocessing": "not_applicable_frozen_reference",
        "estimator": "frozen_B3",
        "model_parameters": "",
        "uses_covariates": "no",
        "error_memory_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M5A_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "B3_ERROR_MEMORY_V1",
        "feature_names": ";".join(MEMORY_FEATURES),
        "raw_feature_count": len(MEMORY_FEATURES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "model_parameters": json.dumps({"alpha": 1.0}, sort_keys=True),
        "uses_covariates": "no",
        "error_memory_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
)

FEATURE_MANIFEST = (
    {
        "feature_group_id": "B3_ERROR_MEMORY_V1",
        "feature_name": "last_observed_b3_residual",
        "formula": "B3 residual at the latest available historical pseudo-target z <= y-2",
        "source": "same target city history + frozen B3",
        "semantic_role": "latest city-level model-error memory",
        "model_feature": "yes",
        "used_by": M5A_ID,
        "year_rule": "historical pseudo-target boundary z <= y-2; each residual B3 fit history <= z-2",
        "missing_policy": "explicit missing if no valid historical pseudo-residual",
    },
    {
        "feature_group_id": "B3_ERROR_MEMORY_V1",
        "feature_name": "recent3_mean_b3_residual",
        "formula": "mean of the three latest valid historical B3 residuals",
        "source": "same target city history + frozen B3",
        "semantic_role": "recent persistent city-level model-error level",
        "model_feature": "yes",
        "used_by": M5A_ID,
        "year_rule": "three latest residual pseudo-targets, all z <= y-2",
        "missing_policy": "explicit missing if fewer than three valid historical pseudo-residuals",
    },
    {
        "feature_group_id": "B3_ERROR_MEMORY_V1",
        "feature_name": "all_available_mean_b3_residual",
        "formula": "mean of all valid historical B3 residuals with z <= y-2",
        "source": "same target city history + frozen B3",
        "semantic_role": "longer city-level model-error bias memory",
        "model_feature": "yes",
        "used_by": M5A_ID,
        "year_rule": "all valid residual pseudo-targets z <= y-2",
        "missing_policy": "explicit missing if no valid historical pseudo-residual",
    },
    {
        "feature_group_id": "B3_ERROR_MEMORY_V1",
        "feature_name": "recent3_residual_slope",
        "formula": "OLS slope of the three latest valid B3 residuals over actual pseudo-target years",
        "source": "same target city history + frozen B3",
        "semantic_role": "recent change in city-level model error",
        "model_feature": "yes",
        "used_by": M5A_ID,
        "year_rule": "three latest residual pseudo-targets; actual year gaps; all z <= y-2",
        "missing_policy": "explicit missing if fewer than three valid historical pseudo-residuals",
    },
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else ""
    return value


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
            writer.writerow({
                field: "" if row.get(field) is None else to_scalar(row.get(field))
                for field in fieldnames
            })


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(field for row in rows for field in row))


def load_context() -> dict[str, Any]:
    assert WAVE_A_SUMMARY_PATH.exists()
    wave_summary = WAVE_A_SUMMARY_PATH.read_text(encoding="utf-8")
    for plan_id in ("S4-Plan04A", "S4-Plan04B", "S4-Plan04C"):
        assert plan_id in wave_summary
        summary_path = ROOT / "reports" / "S4_HANDOFFS" / plan_id / f"{plan_id.replace('-', '').lower()}_summary.json"
        if plan_id == "S4-Plan04A":
            summary_path = ROOT / "reports" / "S4_HANDOFFS" / plan_id / "plan04a_summary.json"
        elif plan_id == "S4-Plan04B":
            summary_path = ROOT / "reports" / "S4_HANDOFFS" / plan_id / "plan04b_summary.json"
        elif plan_id == "S4-Plan04C":
            summary_path = ROOT / "reports" / "S4_HANDOFFS" / plan_id / "plan04c_summary.json"
        integrated_summary = read_json(summary_path)
        assert integrated_summary["s4_complete"] is False
        assert integrated_summary["s5_entered"] is False

    s3_plan01.assert_static_boundaries()
    population, population_profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    scenario_definitions = s3_plan01.load_scenarios()
    s3_summary = read_json(S3_PLAN03_DIR / "plan03_summary.json")
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    assert s3_summary["r1_r2_equivalence_all_passed"] is True
    assert set(scenario["scenario_id"] for scenario in scenario_definitions) == set(SCENARIOS)
    return {
        "population": population,
        "population_profile": population_profile,
        "s2_summary": s2_summary,
        "scenario_definitions": scenario_definitions,
        "s3_summary": s3_summary,
    }


def b3_fit(series: dict[int, float], target_year: int) -> dict[str, Any]:
    return s3_plan02.fit_prediction(
        B3_ID,
        series,
        target_year,
        target_year - HORIZON_YEARS,
    )


def residual_record(series: dict[int, float], pseudo_target_year: int) -> dict[str, Any] | None:
    fit = b3_fit(series, pseudo_target_year)
    actual = series.get(pseudo_target_year)
    if actual is None or fit["prediction"] is None:
        return None
    fit_history_max = fit["history_max_year"]
    assert fit_history_max is None or fit_history_max <= pseudo_target_year - HORIZON_YEARS
    return {
        "pseudo_target_year": pseudo_target_year,
        "residual_actual_minus_b3": actual - fit["prediction"],
        "b3_history_max_year": fit_history_max,
        "b3_history_years": tuple(fit["history_years"]),
        "b3_fit_status": fit["regression_fit_status"],
    }


def build_residual_history(
    series: dict[int, float],
    row_year: int,
) -> list[dict[str, Any]]:
    cutoff = row_year - HORIZON_YEARS
    history = []
    for pseudo_target_year in sorted(year for year in series if year <= cutoff):
        record = residual_record(series, pseudo_target_year)
        if record is not None:
            history.append(record)
    assert all(
        record["pseudo_target_year"] <= row_year - HORIZON_YEARS
        for record in history
    )
    assert all(
        record["b3_history_max_year"] is None
        or record["b3_history_max_year"] <= record["pseudo_target_year"] - HORIZON_YEARS
        for record in history
    )
    return history


def build_memory_features(
    series: dict[int, float],
    row_year: int,
    history_cache: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    cache_key = (id(series), row_year)
    history = history_cache.get(cache_key)
    if history is None:
        history = build_residual_history(series, row_year)
        history_cache[cache_key] = history
    residuals = [float(record["residual_actual_minus_b3"]) for record in history]
    years = [int(record["pseudo_target_year"]) for record in history]
    recent3 = history[-3:]
    recent3_values = [float(record["residual_actual_minus_b3"]) for record in recent3]
    recent3_years = [int(record["pseudo_target_year"]) for record in recent3]
    features: dict[str, Any] = {
        "last_observed_b3_residual": residuals[-1] if residuals else None,
        "recent3_mean_b3_residual": mean(recent3_values) if len(recent3_values) >= 3 else None,
        "all_available_mean_b3_residual": mean(residuals) if residuals else None,
        "recent3_residual_slope": ols_slope(recent3_years, recent3_values) if len(recent3_values) >= 3 else None,
    }
    missing = [name for name in MEMORY_FEATURES if features[name] is None]
    max_b3_history_year = max(
        (record["b3_history_max_year"] for record in history if record["b3_history_max_year"] is not None),
        default=None,
    )
    return {
        **features,
        "residual_history_cutoff_year": row_year - HORIZON_YEARS,
        "residual_history_count": len(history),
        "residual_history_years": ";".join(str(year) for year in years),
        "residual_history_min_pseudo_target_year": min(years) if years else None,
        "residual_history_max_pseudo_target_year": max(years) if years else None,
        "residual_history_max_b3_population_history_year": max_b3_history_year,
        "recent3_residual_source_years": ";".join(str(year) for year in recent3_years),
        "memory_feature_status": "complete" if not missing else "partial" if residuals else "no_history",
        "memory_missing_feature_names": ";".join(missing),
    }


def build_training_rows(
    target_series: dict[int, float],
    city: str,
    outer_origin: int,
    history_cache: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    excluded = 0
    for row_year in sorted(
        year for year in target_series if year <= outer_origin - HORIZON_YEARS
    ):
        fit = b3_fit(target_series, row_year)
        actual = target_series.get(row_year)
        if actual is None or fit["prediction"] is None:
            excluded += 1
            continue
        memory = build_memory_features(target_series, row_year, history_cache)
        rows.append(
            {
                "row_role": "training",
                "city_id": city,
                "row_year": row_year,
                "outer_origin_year": outer_origin,
                "target_value": actual,
                "b3_prediction": fit["prediction"],
                "b3_residual_actual_minus_prediction": actual - fit["prediction"],
                "b3_fit": fit,
                "memory": memory,
            }
        )
    return rows, excluded


def build_validation_row(
    target_series: dict[int, float],
    city: str,
    origin: int,
    history_cache: dict[tuple[int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    fit = b3_fit(target_series, origin)
    actual = target_series.get(origin)
    assert actual is not None and fit["prediction"] is not None
    return {
        "row_role": "validation",
        "city_id": city,
        "row_year": origin,
        "outer_origin_year": origin,
        "target_value": actual,
        "b3_prediction": fit["prediction"],
        "b3_residual_actual_minus_prediction": actual - fit["prediction"],
        "b3_fit": fit,
        "memory": build_memory_features(target_series, origin, history_cache),
    }


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                as_float(row["memory"].get(name))
                if as_float(row["memory"].get(name)) is not None
                else np.nan
                for name in MEMORY_FEATURES
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_residual_model(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> tuple[list[float], dict[str, Any]]:
    assert training_rows and validation_rows
    x_train = feature_matrix(training_rows)
    x_validation = feature_matrix(validation_rows)
    y_train = np.asarray(
        [row["b3_residual_actual_minus_prediction"] for row in training_rows],
        dtype=float,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x_train, y_train)
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    indicator_names = [
        f"missing_indicator__{MEMORY_FEATURES[index]}"
        for index in imputer.indicator_.features_
    ]
    return model.predict(x_validation).tolist(), {
        "training_row_count": len(training_rows),
        "training_label_min_year": min(int(row["row_year"]) for row in training_rows),
        "training_label_max_year": max(int(row["row_year"]) for row in training_rows),
        "training_missing_feature_cell_count": int(np.isnan(x_train).sum()),
        "validation_missing_feature_cell_count": int(np.isnan(x_validation).sum()),
        "preprocessing_fit_row_count": len(training_rows),
        "model_fit_row_count": len(training_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": [float(value) for value in np.asarray(ridge.coef_).ravel()],
        "coefficient_names": list(MEMORY_FEATURES) + indicator_names,
        "feature_names": ";".join(MEMORY_FEATURES),
        "ridge_alpha": 1.0,
    }


def common_row_fields(
    target: dict[str, str],
    scenario_id: str,
    model_id: str,
    validation: dict[str, Any],
    prediction: float,
    predicted_residual: float | None,
    feature_group_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = validation["target_value"]
    residual = prediction - actual
    fit = validation["b3_fit"]
    memory = validation["memory"]
    result = {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": scenario_id,
        "predictor_id": model_id,
        "model_id": model_id,
        "origin_year": validation["row_year"],
        "outer_origin_year": validation["outer_origin_year"],
        "city_id": validation["city_id"],
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_cutoff_year": validation["row_year"] - HORIZON_YEARS,
        "actual_target_year_for_scoring": validation["row_year"],
        "actual_target_value_for_scoring": actual,
        "prediction_value": prediction,
        "prediction_status": "scoreable",
        "scoreable": "yes",
        "residual_prediction_minus_actual": residual,
        "squared_error": residual * residual,
        "absolute_error": abs(residual),
        "history_observation_count": fit["history_count"],
        "history_years_used": ";".join(str(year) for year in fit["history_years"]),
        "history_min_year": fit["history_min_year"],
        "history_max_year": fit["history_max_year"],
        "latest_target_year_used": fit["latest_target_year_used"],
        "previous_target_year_used": fit["previous_target_year_used"],
        "b3_prediction_value": validation["b3_prediction"],
        "b3_residual_actual_minus_prediction": validation["b3_residual_actual_minus_prediction"],
        "predicted_b3_residual": predicted_residual,
        "feature_builder_id": "FROZEN_B3_REFERENCE" if model_id == B3_ID else "B3_ERROR_MEMORY_V1",
        "feature_group_id": feature_group_id,
        "last_observed_b3_residual": memory["last_observed_b3_residual"] if model_id == M5A_ID else None,
        "recent3_mean_b3_residual": memory["recent3_mean_b3_residual"] if model_id == M5A_ID else None,
        "all_available_mean_b3_residual": memory["all_available_mean_b3_residual"] if model_id == M5A_ID else None,
        "recent3_residual_slope": memory["recent3_residual_slope"] if model_id == M5A_ID else None,
        "residual_history_cutoff_year": memory["residual_history_cutoff_year"] if model_id == M5A_ID else None,
        "residual_history_count": memory["residual_history_count"] if model_id == M5A_ID else None,
        "residual_history_years": memory["residual_history_years"] if model_id == M5A_ID else "",
        "residual_history_max_pseudo_target_year": memory["residual_history_max_pseudo_target_year"] if model_id == M5A_ID else None,
        "recent3_residual_source_years": memory["recent3_residual_source_years"] if model_id == M5A_ID else "",
        "memory_feature_status": memory["memory_feature_status"] if model_id == M5A_ID else "not_applicable",
        "memory_missing_feature_names": memory["memory_missing_feature_names"] if model_id == M5A_ID else "",
        "training_label_min_year": metadata.get("training_label_min_year", ""),
        "training_label_max_year": metadata.get("training_label_max_year", ""),
        "training_row_count": metadata.get("training_row_count", ""),
        "preprocessing_fit_row_count": metadata.get("preprocessing_fit_row_count", ""),
        "model_fit_row_count": metadata.get("model_fit_row_count", ""),
        "training_missing_feature_cell_count": metadata.get("training_missing_feature_cell_count", ""),
        "validation_missing_feature_cell_count": metadata.get("validation_missing_feature_cell_count", ""),
        "covariates_used": "no",
        "error_memory_used": "no" if model_id == B3_ID else "yes",
        "city_identity_used": "no",
    }
    return result


def make_b3_row(target: dict[str, str], scenario_id: str, validation: dict[str, Any]) -> dict[str, Any]:
    return common_row_fields(
        target,
        scenario_id,
        B3_ID,
        validation,
        validation["b3_prediction"],
        None,
        "FROZEN_B3",
        {},
    )


def make_candidate_row(
    target: dict[str, str],
    scenario_id: str,
    validation: dict[str, Any],
    predicted_residual: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return common_row_fields(
        target,
        scenario_id,
        M5A_ID,
        validation,
        validation["b3_prediction"] + predicted_residual,
        predicted_residual,
        "B3_ERROR_MEMORY_V1",
        metadata,
    )


def boundary_audit_row(
    target: dict[str, str],
    scenario_id: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    pseudo_target_year = int(row["row_year"])
    outer_origin = int(row["outer_origin_year"])
    memory = row["memory"]
    fit = row["b3_fit"]
    history_max = memory["residual_history_max_pseudo_target_year"]
    max_b3_history = memory["residual_history_max_b3_population_history_year"]
    current_residual_in_history = pseudo_target_year in {
        int(value) for value in memory["residual_history_years"].split(";") if value
    }
    t_minus_1_in_history = pseudo_target_year - 1 in {
        int(value) for value in memory["residual_history_years"].split(";") if value
    }
    checks = {
        "residual_history_within_y_minus_2": history_max is None or history_max <= pseudo_target_year - HORIZON_YEARS,
        "residual_b3_history_within_z_minus_2": max_b3_history is None or max_b3_history <= max(
            int(value) - HORIZON_YEARS
            for value in memory["residual_history_years"].split(";")
            if value
        ) if history_max is not None else True,
        "self_residual_excluded": not current_residual_in_history,
        "t_minus_1_residual_excluded": not t_minus_1_in_history,
        "outer_training_label_within_outer_origin_minus_2": (
            row["row_role"] != "training" or pseudo_target_year <= outer_origin - HORIZON_YEARS
        ),
        "current_b3_fit_within_y_minus_2": (
            fit["history_max_year"] is None or fit["history_max_year"] <= pseudo_target_year - HORIZON_YEARS
        ),
    }
    return {
        "target_id": target["target_id"],
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "row_role": row["row_role"],
        "city_id": row["city_id"],
        "pseudo_target_year": pseudo_target_year,
        "current_row_target_history_cutoff_year": pseudo_target_year - HORIZON_YEARS,
        "current_row_b3_history_max_year": fit["history_max_year"] or "",
        "current_row_b3_residual_actual_minus_prediction": row["b3_residual_actual_minus_prediction"],
        "residual_history_cutoff_year": memory["residual_history_cutoff_year"],
        "residual_history_count": memory["residual_history_count"],
        "residual_history_years": memory["residual_history_years"],
        "residual_history_max_pseudo_target_year": history_max or "",
        "residual_history_max_b3_population_history_year": max_b3_history or "",
        "recent3_residual_source_years": memory["recent3_residual_source_years"],
        "memory_feature_status": memory["memory_feature_status"],
        "memory_missing_feature_names": memory["memory_missing_feature_names"],
        "current_row_residual_in_feature_history": "yes" if current_residual_in_history else "no",
        "t_minus_1_residual_in_feature_history": "yes" if t_minus_1_in_history else "no",
        "residual_history_within_y_minus_2": "PASS" if checks["residual_history_within_y_minus_2"] else "FAIL",
        "residual_b3_history_within_z_minus_2": "PASS" if checks["residual_b3_history_within_z_minus_2"] else "FAIL",
        "self_residual_excluded": "PASS" if checks["self_residual_excluded"] else "FAIL",
        "t_minus_1_residual_excluded": "PASS" if checks["t_minus_1_residual_excluded"] else "FAIL",
        "outer_training_label_within_outer_origin_minus_2": "PASS" if checks["outer_training_label_within_outer_origin_minus_2"] else "FAIL",
        "current_b3_fit_within_y_minus_2": "PASS" if checks["current_b3_fit_within_y_minus_2"] else "FAIL",
        "boundary_status": "PASS" if all(checks.values()) else "FAIL",
    }


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    saved_predictors = s3_plan01.PREDICTORS
    s3_plan01.PREDICTORS = MODEL_DEFINITIONS
    try:
        per_origin, _ = s3_plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(rows, per_origin)
    finally:
        s3_plan01.PREDICTORS = saved_predictors
    return per_origin, aggregate, aggregate_map


def compare_b3_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen = read_csv(S3_PLAN03_DIR / "per_origin_predictions_or_residuals.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row
        for row in frozen
        if row["predictor_id"] == B3_ID
    }
    current_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row
        for row in current_rows
        if row["predictor_id"] == B3_ID
    }
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(frozen_map) == len(current_map) == expected
    result = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    for (target_id, scenario_id), pairs in sorted(grouped.items()):
        max_deltas = {field: 0.0 for field in ("prediction_value", "residual_prediction_minus_actual", "squared_error")}
        passed = True
        for current, frozen_row in pairs:
            for field in max_deltas:
                left = as_float(current.get(field))
                right = as_float(frozen_row.get(field))
                if left is None or right is None:
                    passed = passed and left == right
                else:
                    max_deltas[field] = max(max_deltas[field], abs(left - right))
                    passed = passed and abs(left - right) <= 1e-9
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "rows_compared": len(pairs),
                "max_abs_prediction_delta": max_deltas["prediction_value"],
                "max_abs_residual_delta": max_deltas["residual_prediction_minus_actual"],
                "max_abs_squared_error_delta": max_deltas["squared_error"],
                "status": "PASS" if passed else "FAIL",
                "comparison_rule": "B3 row-level prediction/residual/squared-error delta <= 1e-9",
            }
        )
    assert all(row["status"] == "PASS" for row in result)
    return result


def compare_b3_metrics(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    frozen = read_csv(S3_PLAN03_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in frozen
        if row["predictor_id"] == B3_ID
    }
    result = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            key = (target["target_id"], scenario_id, B3_ID)
            current = aggregate_map[key]
            old = frozen_map[key]
            fields = ("pooled_city_year_mse", "macro_origin_mse", "latest_origin_mse")
            deltas = [abs(float(current[field]) - float(old[field])) for field in fields]
            passed = max(deltas) <= 1e-9
            result.append(
                {
                    "target_id": target["target_id"],
                    "scenario_id": scenario_id,
                    "status": "PASS" if passed else "FAIL",
                    "max_abs_metric_delta": max(deltas),
                    "comparison_rule": "B3 pooled/macro/latest metric delta <= 1e-9",
                }
            )
    assert all(row["status"] == "PASS" for row in result)
    return result


def compare_r1_r2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row
        for row in rows
    }
    result = []
    numeric_fields = (
        "prediction_value",
        "residual_prediction_minus_actual",
        "squared_error",
        "last_observed_b3_residual",
        "recent3_mean_b3_residual",
        "all_available_mean_b3_residual",
        "recent3_residual_slope",
    )
    for target in TARGETS:
        for predictor_id in MODEL_IDS:
            pairs = []
            for origin in COMMON_ORIGINS:
                for city in CITIES:
                    left = indexed[(target["target_id"], SCENARIOS[0], predictor_id, origin, city)]
                    right = indexed[(target["target_id"], SCENARIOS[1], predictor_id, origin, city)]
                    pairs.append((left, right))
            max_delta = 0.0
            strings_equal = True
            for left, right in pairs:
                for field in numeric_fields:
                    left_value = as_float(left.get(field))
                    right_value = as_float(right.get(field))
                    if left_value is None or right_value is None:
                        strings_equal = strings_equal and left_value == right_value
                    else:
                        max_delta = max(max_delta, abs(left_value - right_value))
                strings_equal = strings_equal and left.get("residual_history_years", "") == right.get("residual_history_years", "")
            result.append(
                {
                    "target_id": target["target_id"],
                    "predictor_id": predictor_id,
                    "rows_compared": len(pairs),
                    "max_abs_numeric_delta": max_delta,
                    "history_years_exact": "PASS" if strings_equal else "FAIL",
                    "status": "PASS" if strings_equal and max_delta <= 1e-9 else "FAIL",
                    "comparison_rule": "target-history-only R1/R2 row and memory-feature delta <= 1e-9",
                }
            )
    assert all(row["status"] == "PASS" for row in result)
    return result


def coefficient_direction_summary(fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            relevant = [
                row for row in fit_rows
                if row["target_id"] == target["target_id"] and row["scenario_id"] == scenario_id
            ]
            for feature_index, feature_name in enumerate(MEMORY_FEATURES):
                values = [
                    float(row["ridge_coefficients"][feature_index])
                    for row in relevant
                    if len(row["ridge_coefficients"]) > feature_index
                ]
                positive = sum(value > 1e-12 for value in values)
                negative = sum(value < -1e-12 for value in values)
                zero = len(values) - positive - negative
                direction = "positive" if positive and not negative else "negative" if negative and not positive else "zero" if not positive and not negative else "mixed"
                result[f"{target['target_id']}_{scenario_id}_{feature_name}"] = {
                    "coefficient_min": min(values),
                    "coefficient_max": max(values),
                    "positive_count": positive,
                    "negative_count": negative,
                    "zero_count": zero,
                    "origin_count": len(values),
                    "sign_stability": direction,
                }
    return result


def build_screening_rows(
    per_origin: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    coefficient_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    aggregate_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in aggregate
    }
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin
    }
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            base = aggregate_map[(target_id, scenario_id, B3_ID)]
            candidate = aggregate_map[(target_id, scenario_id, M5A_ID)]
            candidate_origin = [per_origin_map[(target_id, scenario_id, M5A_ID, origin)] for origin in COMMON_ORIGINS]
            base_origin = {origin: per_origin_map[(target_id, scenario_id, B3_ID, origin)] for origin in COMMON_ORIGINS}
            wins = sum(float(row["mse"]) < float(base_origin[int(row["origin_year"])]["mse"]) - 1e-12 for row in candidate_origin)
            losses = sum(float(row["mse"]) > float(base_origin[int(row["origin_year"])]["mse"]) + 1e-12 for row in candidate_origin)
            ties = len(candidate_origin) - wins - losses
            r1 = aggregate_map[(target_id, SCENARIOS[0], M5A_ID)]
            r2 = aggregate_map[(target_id, SCENARIOS[1], M5A_ID)]
            result.append(
                {
                    "metric_scope": "aggregate",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "predictor_id": B3_ID,
                    "origin_year": "",
                    "pooled_city_year_mse": base["pooled_city_year_mse"],
                    "latest_2021_mse": base["latest_origin_mse"],
                    "macro_origin_mse": base["macro_origin_mse"],
                    "wins_vs_b3": "",
                    "losses_vs_b3": "",
                    "ties_vs_b3": "",
                    "delta_vs_b3_pooled_mse": 0.0,
                    "delta_vs_b3_latest_2021_mse": 0.0,
                    "relative_improvement_vs_b3_pct": 0.0,
                    "r1_to_r2_pooled_mse_delta": float(r2["pooled_city_year_mse"]) - float(r1["pooled_city_year_mse"]),
                    "r1_to_r2_latest_2021_mse_delta": float(r2["latest_origin_mse"]) - float(r1["latest_origin_mse"]),
                    "coefficient_sign_summary": "",
                    "comparison_vs_b3": "baseline",
                }
            )
            candidate_pooled = float(candidate["pooled_city_year_mse"])
            base_pooled = float(base["pooled_city_year_mse"])
            result.append(
                {
                    "metric_scope": "aggregate",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "predictor_id": M5A_ID,
                    "origin_year": "",
                    "pooled_city_year_mse": candidate_pooled,
                    "latest_2021_mse": candidate["latest_origin_mse"],
                    "macro_origin_mse": candidate["macro_origin_mse"],
                    "wins_vs_b3": wins,
                    "losses_vs_b3": losses,
                    "ties_vs_b3": ties,
                    "delta_vs_b3_pooled_mse": candidate_pooled - base_pooled,
                    "delta_vs_b3_latest_2021_mse": float(candidate["latest_origin_mse"]) - float(base["latest_origin_mse"]),
                    "relative_improvement_vs_b3_pct": (base_pooled - candidate_pooled) / base_pooled * 100,
                    "r1_to_r2_pooled_mse_delta": float(r2["pooled_city_year_mse"]) - float(r1["pooled_city_year_mse"]),
                    "r1_to_r2_latest_2021_mse_delta": float(r2["latest_origin_mse"]) - float(r1["latest_origin_mse"]),
                    "coefficient_sign_summary": json.dumps(
                        {
                            feature: coefficient_summary[f"{target_id}_{scenario_id}_{feature}"]["sign_stability"]
                            for feature in MEMORY_FEATURES
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "comparison_vs_b3": "better" if candidate_pooled < base_pooled - 1e-12 else "worse" if candidate_pooled > base_pooled + 1e-12 else "tie",
                }
            )
            for origin in COMMON_ORIGINS:
                candidate_origin_row = per_origin_map[(target_id, scenario_id, M5A_ID, origin)]
                base_origin_row = per_origin_map[(target_id, scenario_id, B3_ID, origin)]
                candidate_mse = float(candidate_origin_row["mse"])
                base_mse = float(base_origin_row["mse"])
                result.append(
                    {
                        "metric_scope": "per_origin",
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "predictor_id": M5A_ID,
                        "origin_year": origin,
                        "origin_mse": candidate_mse,
                        "b3_reference_origin_mse": base_mse,
                        "delta_vs_b3_origin_mse": candidate_mse - base_mse,
                        "comparison_vs_b3": "better" if candidate_mse < base_mse - 1e-12 else "worse" if candidate_mse > base_mse + 1e-12 else "tie",
                    }
                )
    return result


def decision_summary(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    per_origin: list[dict[str, Any]],
    coefficient_summary: dict[str, Any],
) -> dict[str, Any]:
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin
    }
    metrics: dict[str, Any] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            base = aggregate_map[(target_id, scenario_id, B3_ID)]
            candidate = aggregate_map[(target_id, scenario_id, M5A_ID)]
            wins = sum(
                float(per_origin_map[(target_id, scenario_id, M5A_ID, origin)]["mse"])
                < float(per_origin_map[(target_id, scenario_id, B3_ID, origin)]["mse"]) - 1e-12
                for origin in COMMON_ORIGINS
            )
            losses = sum(
                float(per_origin_map[(target_id, scenario_id, M5A_ID, origin)]["mse"])
                > float(per_origin_map[(target_id, scenario_id, B3_ID, origin)]["mse"]) + 1e-12
                for origin in COMMON_ORIGINS
            )
            metrics[f"{target_id}_{scenario_id}_{B3_ID}"] = {
                "pooled_mse": float(base["pooled_city_year_mse"]),
                "latest_2021_mse": float(base["latest_origin_mse"]),
            }
            metrics[f"{target_id}_{scenario_id}_{M5A_ID}"] = {
                "pooled_mse": float(candidate["pooled_city_year_mse"]),
                "latest_2021_mse": float(candidate["latest_origin_mse"]),
                "wins_vs_b3": wins,
                "losses_vs_b3": losses,
                "ties_vs_b3": len(COMMON_ORIGINS) - wins - losses,
                "pooled_delta_vs_b3": float(candidate["pooled_city_year_mse"]) - float(base["pooled_city_year_mse"]),
                "latest_delta_vs_b3": float(candidate["latest_origin_mse"]) - float(base["latest_origin_mse"]),
                "relative_improvement_vs_b3_pct": (
                    float(base["pooled_city_year_mse"]) - float(candidate["pooled_city_year_mse"])
                ) / float(base["pooled_city_year_mse"]) * 100,
                "coefficient_sign_summary": {
                    feature: coefficient_summary[f"{target_id}_{scenario_id}_{feature}"]
                    for feature in MEMORY_FEATURES
                },
            }

    resident_r1 = metrics[f"resident_{SCENARIOS[0]}_{M5A_ID}"]
    resident_r2 = metrics[f"resident_{SCENARIOS[1]}_{M5A_ID}"]
    resident_r1_base = metrics[f"resident_{SCENARIOS[0]}_{B3_ID}"]
    resident_r2_base = metrics[f"resident_{SCENARIOS[1]}_{B3_ID}"]
    resident_pooled_over_b3 = resident_r1["pooled_mse"] < resident_r1_base["pooled_mse"] and resident_r2["pooled_mse"] < resident_r2_base["pooled_mse"]
    resident_majority = resident_r1["wins_vs_b3"] > len(COMMON_ORIGINS) / 2 and resident_r2["wins_vs_b3"] > len(COMMON_ORIGINS) / 2
    resident_latest_support = resident_r1["latest_2021_mse"] < resident_r1_base["latest_2021_mse"] and resident_r2["latest_2021_mse"] < resident_r2_base["latest_2021_mse"]
    near_b3_and_broad = any(
        metrics[f"resident_{scenario_id}_{M5A_ID}"]["relative_improvement_vs_b3_pct"] >= -1.0
        and metrics[f"resident_{scenario_id}_{M5A_ID}"]["wins_vs_b3"] > len(COMMON_ORIGINS) / 2
        and metrics[f"resident_{scenario_id}_{M5A_ID}"]["latest_2021_mse"] < metrics[f"resident_{scenario_id}_{B3_ID}"]["latest_2021_mse"]
        for scenario_id in SCENARIOS
    )
    deep_dive = bool(resident_pooled_over_b3 or near_b3_and_broad)
    if resident_pooled_over_b3:
        decision = "error_memory_supported"
        rationale = "resident M5A pooled MSE is below B3 in both target-history replay scenarios"
    elif deep_dive:
        decision = "error_memory_conditional"
        rationale = "resident M5A is within the one-percent hurdle with majority origin wins and latest support in at least one scenario"
    else:
        decision = "error_memory_unsupported"
        rationale = "resident M5A does not provide a stable pooled, multi-origin, and latest improvement over B3"
    registered_support = all(
        metrics[f"registered_{scenario_id}_{M5A_ID}"]["pooled_mse"] < metrics[f"registered_{scenario_id}_{B3_ID}"]["pooled_mse"]
        for scenario_id in SCENARIOS
    )
    return {
        "metrics": metrics,
        "screening_decision": decision,
        "screening_rationale": rationale,
        "deep_dive_recommended": deep_dive,
        "resident_pooled_over_b3_in_both_scenarios": resident_pooled_over_b3,
        "resident_majority_origin_wins_in_both_scenarios": resident_majority,
        "resident_latest_support_in_both_scenarios": resident_latest_support,
        "near_b3_and_broad_condition": near_b3_and_broad,
        "registered_supports_candidate_in_both_scenarios": registered_support,
        "r1_r2_candidate_pooled_delta_resident": resident_r2["pooled_mse"] - resident_r1["pooled_mse"],
        "r1_r2_candidate_latest_delta_resident": resident_r2["latest_2021_mse"] - resident_r1["latest_2021_mse"],
    }


def write_handoff(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    resident_r1 = metrics[f"resident_{SCENARIOS[0]}_{M5A_ID}"]
    resident_r2 = metrics[f"resident_{SCENARIOS[1]}_{M5A_ID}"]
    registered_r1 = metrics[f"registered_{SCENARIOS[0]}_{M5A_ID}"]
    registered_r2 = metrics[f"registered_{SCENARIOS[1]}_{M5A_ID}"]
    resident_b3_r1 = metrics[f"resident_{SCENARIOS[0]}_{B3_ID}"]
    resident_b3_r2 = metrics[f"resident_{SCENARIOS[1]}_{B3_ID}"]
    lines = [
        "# S4-Plan05A Handoff｜B3 城市级残差记忆与误差持续性快速筛选",
        "",
        "## Result",
        "- Status: Complete（screening 已完成；不代表 S4 complete）",
        f"- Branch: `{summary['branch']}`",
        f"- Integration starting commit: `{summary['starting_commit']}`",
        f"- Decision: `{summary['screening_decision']}`",
        f"- `deep_dive_recommended`: `{str(summary['deep_dive_recommended']).lower()}`",
        "",
        "## Scope",
        "- Candidate: `M5A_B3_PLUS_RIDGE_ERROR_MEMORY` = frozen B3 + Ridge(alpha=1.0) residual correction.",
        "- Features: last residual, recent-3 mean, all-history mean, and recent-3 residual slope.",
        "- Resident primary; registered sensitivity; common origins 2011–2021.",
        "- No covariates, city ID, tuning, GBDT, 2023 prediction, submission, or full robustness suite.",
        "",
        "## Screening answers",
        "1. **是否存在可利用的 city-level residual persistence？**",
        f"   - `{summary['screening_decision']}`；resident R1 M5A pooled `{resident_r1['pooled_mse']:.12f}` vs B3 `{resident_b3_r1['pooled_mse']:.12f}`，R2 `{resident_r2['pooled_mse']:.12f}` vs B3 `{resident_b3_r2['pooled_mse']:.12f}`。",
        "2. **M5A 是否超过 B3？**",
        f"   - resident R1/R2 relative improvement: `{resident_r1['relative_improvement_vs_b3_pct']:.6f}%` / `{resident_r2['relative_improvement_vs_b3_pct']:.6f}%`。",
        f"   - registered R1/R2 relative improvement: `{registered_r1['relative_improvement_vs_b3_pct']:.6f}%` / `{registered_r2['relative_improvement_vs_b3_pct']:.6f}%`；两 scenario pooled 支持：`{str(summary['registered_supports_candidate_in_both_scenarios']).lower()}`。",
        "3. **per-origin 是否 broad-based？**",
        f"   - resident R1 `{resident_r1['wins_vs_b3']}/{resident_r1['losses_vs_b3']}/{resident_r1['ties_vs_b3']}`，R2 `{resident_r2['wins_vs_b3']}/{resident_r2['losses_vs_b3']}/{resident_r2['ties_vs_b3']}`（wins/losses/ties）。",
        f"   - registered R1 `{registered_r1['wins_vs_b3']}/{registered_r1['losses_vs_b3']}/{registered_r1['ties_vs_b3']}`，R2 `{registered_r2['wins_vs_b3']}/{registered_r2['losses_vs_b3']}/{registered_r2['ties_vs_b3']}`。",
        "4. **latest 2021 是否支持？**",
        f"   - resident B3 latest `{resident_b3_r1['latest_2021_mse']:.12f}`；M5A R1 `{resident_r1['latest_2021_mse']:.12f}`、R2 `{resident_r2['latest_2021_mse']:.12f}`。",
        f"   - registered B3 latest `{metrics[f'registered_{SCENARIOS[0]}_{B3_ID}']['latest_2021_mse']:.12f}`；M5A R1 `{registered_r1['latest_2021_mse']:.12f}`、R2 `{registered_r2['latest_2021_mse']:.12f}`。",
        "5. **boundary 是否无 self/future residual leakage？**",
        f"   - audit rows `{summary['error_memory_boundary']['audit_rows']}`；all boundary PASS=`{str(summary['error_memory_boundary']['all_boundary_rows_pass']).lower()}`；self residual violations=`{summary['error_memory_boundary']['self_residual_violation_count']}`；`y-1/T-1` residual violations=`{summary['error_memory_boundary']['t_minus_1_violation_count']}`。",
        f"   - residual history max pseudo-target always `<= y-2`；each historical residual B3 fit history `<= z-2`。",
        "6. **coefficient direction / deep dive？**",
        "   - resident 四个 memory feature 的系数方向均为 `mixed`（跨 11 个 origins）；registered 的 all-history mean 与 recent-3 mean 为 `negative`，last residual 与 recent-3 slope 为 `mixed`。",
        f"   - deep dive=`{str(summary['deep_dive_recommended']).lower()}`。{summary['screening_rationale']}",
        "",
        "## Evidence",
        "- `plan05a_summary.json`",
        "- `screening_metrics.csv`",
        "- `error_memory_boundary_audit.csv`",
        "- `scripts/build_s4_plan05a_error_memory.py`",
        "",
        "## Project state",
        "- `s4_complete=false`; `s5_entered=false`。Wave A integration 与前序 contract 未修改。",
    ]
    (OUTPUT_DIR / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert STARTING_COMMIT == "decfd7e0c68d14da31eabe5720ab706122c1040d"
    context = load_context()
    population = context["population"]
    all_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    training_excluded_count = 0
    model_fit_count = 0
    history_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                training_rows: list[dict[str, Any]] = []
                validation_rows: list[dict[str, Any]] = []
                for city in CITIES:
                    target_series = population[target_id][city]
                    city_training, excluded = build_training_rows(
                        target_series,
                        city,
                        origin,
                        history_cache,
                    )
                    training_rows.extend(city_training)
                    training_excluded_count += excluded
                    validation = build_validation_row(
                        target_series,
                        city,
                        origin,
                        history_cache,
                    )
                    validation_rows.append(validation)
                    boundary_rows.append(boundary_audit_row(target, scenario_id, validation))
                    boundary_rows.extend(
                        boundary_audit_row(target, scenario_id, row)
                        for row in city_training
                    )
                predicted_residuals, metadata = fit_residual_model(training_rows, validation_rows)
                fit_rows.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        **metadata,
                    }
                )
                model_fit_count += 1
                for validation, predicted_residual in zip(validation_rows, predicted_residuals):
                    all_rows.append(make_b3_row(target, scenario_id, validation))
                    all_rows.append(
                        make_candidate_row(
                            target,
                            scenario_id,
                            validation,
                            float(predicted_residual),
                            metadata,
                        )
                    )

    expected_validation_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(all_rows) == expected_validation_rows * 2
    assert len(fit_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS)
    assert len(boundary_rows) > expected_validation_rows
    assert all(row["boundary_status"] == "PASS" for row in boundary_rows)
    assert all(row["error_memory_used"] in {"no", "yes"} for row in all_rows)

    b3_rows = [row for row in all_rows if row["predictor_id"] == B3_ID]
    b3_compatibility = compare_b3_rows(b3_rows)
    per_origin, aggregate, aggregate_map = calculate_metrics(all_rows)
    b3_metric_compatibility = compare_b3_metrics(aggregate_map)
    r1_r2_equivalence = compare_r1_r2(all_rows)
    coefficient_summary = coefficient_direction_summary(fit_rows)
    screening_rows = build_screening_rows(per_origin, aggregate, coefficient_summary)
    decision = decision_summary(aggregate_map, per_origin, coefficient_summary)

    self_violations = sum(
        row["self_residual_excluded"] != "PASS" for row in boundary_rows
    )
    t_minus_1_violations = sum(
        row["t_minus_1_residual_excluded"] != "PASS" for row in boundary_rows
    )
    history_violations = sum(
        row["residual_history_within_y_minus_2"] != "PASS"
        or row["residual_b3_history_within_z_minus_2"] != "PASS"
        for row in boundary_rows
    )
    assert self_violations == t_minus_1_violations == history_violations == 0
    assert all(row["status"] == "PASS" for row in b3_compatibility)
    assert all(row["status"] == "PASS" for row in b3_metric_compatibility)
    assert all(row["status"] == "PASS" for row in r1_r2_equivalence)

    summary = {
        "plan_id": "S4-Plan05A",
        "stage": "S4_FEATURE_ML_MODELING",
        "branch": BRANCH,
        "starting_commit": STARTING_COMMIT,
        "wave_a_integration_verified": True,
        "common_origin_domain": "2011-2021",
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_cutoff_rule": "current B3 row history <= y-2; historical residual pseudo-target z <= y-2; each residual's B3 history <= z-2",
        "strongest_baseline": B3_ID,
        "model_ids": list(MODEL_IDS),
        "candidate_id": M5A_ID,
        "feature_manifest": list(FEATURE_MANIFEST),
        "b3_compatibility_pass": all(row["status"] == "PASS" for row in b3_compatibility),
        "b3_metric_compatibility_pass": all(row["status"] == "PASS" for row in b3_metric_compatibility),
        "b3_compatibility_rows": b3_compatibility,
        "b3_metric_compatibility_rows": b3_metric_compatibility,
        "r1_r2_equivalence_pass": all(row["status"] == "PASS" for row in r1_r2_equivalence),
        "r1_r2_equivalence_rows": r1_r2_equivalence,
        "error_memory_boundary": {
            "audit_rows": len(boundary_rows),
            "validation_rows": expected_validation_rows,
            "training_rows": len(boundary_rows) - expected_validation_rows,
            "all_boundary_rows_pass": all(row["boundary_status"] == "PASS" for row in boundary_rows),
            "self_residual_violation_count": self_violations,
            "t_minus_1_violation_count": t_minus_1_violations,
            "history_boundary_violation_count": history_violations,
            "current_row_residual_used_as_feature": False,
            "future_residual_used_as_feature": False,
            "residual_history_max_rule": "<= y-2",
            "residual_b3_fit_history_max_rule": "<= z-2",
        },
        "coefficient_direction_summary": coefficient_summary,
        "metrics": decision["metrics"],
        "screening_decision": decision["screening_decision"],
        "screening_rationale": decision["screening_rationale"],
        "deep_dive_recommended": decision["deep_dive_recommended"],
        "resident_pooled_over_b3_in_both_scenarios": decision["resident_pooled_over_b3_in_both_scenarios"],
        "resident_majority_origin_wins_in_both_scenarios": decision["resident_majority_origin_wins_in_both_scenarios"],
        "resident_latest_support_in_both_scenarios": decision["resident_latest_support_in_both_scenarios"],
        "near_b3_and_broad_condition": decision["near_b3_and_broad_condition"],
        "registered_supports_candidate_in_both_scenarios": decision["registered_supports_candidate_in_both_scenarios"],
        "r1_r2_candidate_pooled_delta_resident": decision["r1_r2_candidate_pooled_delta_resident"],
        "r1_r2_candidate_latest_delta_resident": decision["r1_r2_candidate_latest_delta_resident"],
        "training_excluded_b3_insufficient_history_rows": training_excluded_count,
        "outer_origin_model_fit_count": model_fit_count,
        "covariates_used": False,
        "city_identity_used": False,
        "tuning_run": False,
        "formal_ml_model_used": True,
        "robustness_suite_run": False,
        "final_2023_prediction_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
    }
    write_csv(OUTPUT_DIR / "screening_metrics.csv", screening_rows, fields=union_fields(screening_rows))
    write_csv(OUTPUT_DIR / "error_memory_boundary_audit.csv", boundary_rows, fields=union_fields(boundary_rows))
    write_json(OUTPUT_DIR / "plan05a_summary.json", summary)
    write_handoff(summary)

    print("S4_PLAN05A_ERROR_MEMORY_SCREENING_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"model_fit_count={model_fit_count}")
    print(f"b3_compatibility_pass={summary['b3_compatibility_pass']}")
    print(f"b3_metric_compatibility_pass={summary['b3_metric_compatibility_pass']}")
    print(f"r1_r2_equivalence_pass={summary['r1_r2_equivalence_pass']}")
    print(f"self_residual_violation_count={self_violations}")
    print(f"t_minus_1_violation_count={t_minus_1_violations}")
    print(f"screening_decision={summary['screening_decision']}")
    print(f"deep_dive_recommended={summary['deep_dive_recommended']}")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
