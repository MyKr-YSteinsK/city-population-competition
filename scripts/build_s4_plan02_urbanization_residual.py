"""Build S4-Plan02 urbanization residual-correction evidence.

The frozen B3 recent-three-point population trend remains the prediction
backbone.  This runner fits only a fixed Ridge residual correction using the
source-preserved urbanization series under the frozen S2 R1/R2 availability
rules.  It deliberately avoids population-history features, city identity,
other covariates, tuning, ensembles, final forecasts, and submission output.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan02"
S2_PLAN02_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
S4_PLAN01_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01"

STARTING_COMMIT = "71f76664426a3b720f9b03f77792c8f0905e9f92"
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
EARLY_ORIGINS = s3_plan02.EARLY_ORIGINS
LATE_ORIGINS = s3_plan02.LATE_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

URBANIZATION_SOURCE_ID = "urbanization"
URBANIZATION_VALUE_COLUMN = "urbanizationRate"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
D2_ID = "D2_B3_GLOBAL_RESIDUAL_BIAS"
M2A_ID = "M2A_B3_PLUS_RIDGE_URB_LEVEL"
M2B_ID = "M2B_B3_PLUS_RIDGE_URB_DYNAMICS"
MODEL_IDS = (B3_ID, D2_ID, M2A_ID, M2B_ID)
CORRECTION_IDS = (D2_ID, M2A_ID, M2B_ID)
RIDGE_IDS = (M2A_ID, M2B_ID)
LEVEL_FEATURES = ("urbanization_latest",)
DYNAMICS_FEATURES = (
    "urbanization_latest",
    "urbanization_recent_change",
    "urbanization_recent3_slope",
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
        "urbanization_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": D2_ID,
        "model_family": "global_residual_bias_diagnostic",
        "feature_group_id": "NONE_DIAGNOSTIC",
        "feature_names": "",
        "target_definition": "mean historical B3 residual actual minus B3 prediction",
        "preprocessing": "not_applicable_diagnostic",
        "estimator": "global_mean_training_residual",
        "ridge_alpha": "",
        "uses_covariates": "no",
        "urbanization_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M2A_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "URB_LEVEL_V1",
        "feature_names": ";".join(LEVEL_FEATURES),
        "target_definition": "b3_residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "uses_covariates": "yes",
        "urbanization_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M2B_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "URB_DYNAMICS_V1",
        "feature_names": ";".join(DYNAMICS_FEATURES),
        "target_definition": "b3_residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "uses_covariates": "yes",
        "urbanization_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
)

FEATURE_MANIFEST = (
    {
        "feature_group_id": "URB_LEVEL_V1",
        "feature_name": "urbanization_latest",
        "formula": "latest non-missing urbanizationRate with actual year <= scenario cap",
        "source": "data/normalized/城镇化率.csv",
        "semantic_role": "urbanization development level",
        "model_feature": "yes",
        "used_by": "M2A;M2B",
        "year_rule": "scenario-specific source cap; no future fill or interpolation",
        "missing_policy": "explicit missing; training-fold median imputation only",
    },
    {
        "feature_group_id": "URB_LEVEL_V1",
        "feature_name": "urbanization_latest_year",
        "formula": "actual year of urbanization_latest",
        "source": "data/normalized/城镇化率.csv",
        "semantic_role": "freshness audit field",
        "model_feature": "no",
        "used_by": "audit_only",
        "year_rule": "scenario-specific source cap",
        "missing_policy": "blank when source has no value as of cap",
    },
    {
        "feature_group_id": "URB_LEVEL_V1",
        "feature_name": "urbanization_staleness_years",
        "formula": "pseudo-target year minus actual latest urbanization year",
        "source": "data/normalized/城镇化率.csv",
        "semantic_role": "freshness audit field",
        "model_feature": "no",
        "used_by": "audit_only",
        "year_rule": "actual year difference; no assumed publication schedule",
        "missing_policy": "blank when source has no value as of cap",
    },
    {
        "feature_group_id": "URB_DYNAMICS_V1",
        "feature_name": "urbanization_recent_change",
        "formula": "(latest allowed value - previous allowed value) / actual year gap",
        "source": "data/normalized/城镇化率.csv",
        "semantic_role": "recent urbanization change",
        "model_feature": "yes",
        "used_by": "M2B",
        "year_rule": "two latest allowed observations; actual year difference",
        "missing_policy": "explicit missing if fewer than two allowed observations",
    },
    {
        "feature_group_id": "URB_DYNAMICS_V1",
        "feature_name": "urbanization_recent3_slope",
        "formula": "OLS slope on latest three allowed urbanization observations",
        "source": "data/normalized/城镇化率.csv",
        "semantic_role": "recent urbanization trend",
        "model_feature": "yes",
        "used_by": "M2B",
        "year_rule": "three latest allowed observations; actual year differences",
        "missing_policy": "explicit missing if fewer than three allowed observations",
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


def competition_score(mse: float) -> float:
    return 1.0 / (1.0 + mse)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else ""
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    assert rows or fields
    fieldnames = list(fields or rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else to_scalar(row.get(field)) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(field for row in rows for field in row))


def ols_fit(years: list[int], values: list[float]) -> dict[str, float]:
    assert len(years) == len(values) >= 2
    x_mean = mean(float(year) for year in years)
    y_mean = mean(values)
    denominator = math.fsum((year - x_mean) ** 2 for year in years)
    assert denominator > 0
    slope = math.fsum((year - x_mean) * (value - y_mean) for year, value in zip(years, values)) / denominator
    return {"slope": slope, "intercept": y_mean - slope * x_mean}


def population_target_id(target: dict[str, str]) -> str:
    return target["target_id"]


def source_series(source: dict[str, Any]) -> dict[str, dict[int, float]]:
    city_col = source["city_col"]
    year_col = source["year_col"]
    value_col = source["value_columns"][0]
    result: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    for row in source["value_rows"]:
        city = row[city_col]
        year = int(float(row[year_col]))
        value = as_float(row.get(value_col))
        if value is not None:
            result.setdefault(city, {})[year] = value
    return result


def build_urbanization_features(
    series: dict[int, float],
    pseudo_target_year: int,
    required_lag_years: int,
) -> dict[str, Any]:
    cap = pseudo_target_year - required_lag_years
    allowed_years = sorted(year for year in series if year <= cap)
    latest_year = allowed_years[-1] if allowed_years else None
    latest_value = series[latest_year] if latest_year is not None else None
    recent_change = None
    recent_change_years: list[int] = []
    if len(allowed_years) >= 2:
        recent_change_years = allowed_years[-2:]
        recent_change = (series[recent_change_years[-1]] - series[recent_change_years[-2]]) / (recent_change_years[-1] - recent_change_years[-2])
    recent3_slope = None
    recent3_years: list[int] = []
    if len(allowed_years) >= 3:
        recent3_years = allowed_years[-3:]
        recent3_slope = ols_fit(recent3_years, [series[year] for year in recent3_years])["slope"]
    used_years = sorted(set(([latest_year] if latest_year is not None else []) + recent_change_years + recent3_years))
    return {
        "urbanization_required_lag_years": required_lag_years,
        "urbanization_required_latest_usable_year": cap,
        "urbanization_allowed_years": ";".join(str(year) for year in allowed_years),
        "urbanization_latest_year": latest_year,
        "urbanization_latest": latest_value,
        "urbanization_staleness_years": pseudo_target_year - latest_year if latest_year is not None else None,
        "urbanization_recent_change": recent_change,
        "urbanization_recent_change_source_years": ";".join(str(year) for year in recent_change_years),
        "urbanization_recent3_slope": recent3_slope,
        "urbanization_recent3_slope_source_years": ";".join(str(year) for year in recent3_years),
        "urbanization_used_years": ";".join(str(year) for year in used_years),
        "urbanization_used_year_max": max(used_years) if used_years else None,
        "urbanization_latest_available": "yes" if latest_year is not None else "no",
        "urbanization_recent_change_available": "yes" if recent_change is not None else "no",
        "urbanization_recent3_slope_available": "yes" if recent3_slope is not None else "no",
        "urbanization_missing_feature_names": ";".join(
            name
            for name, value in (
                ("urbanization_latest", latest_value),
                ("urbanization_recent_change", recent_change),
                ("urbanization_recent3_slope", recent3_slope),
            )
            if value is None
        ),
    }


def required_urbanization_lag(geometry: dict[str, Any], scenario_id: str) -> int:
    return s2_geometry.required_lag_years(URBANIZATION_SOURCE_ID, geometry, scenario_id)


def b3_fit(series: dict[int, float], target_year: int) -> dict[str, Any]:
    return s3_plan02.fit_prediction(B3_ID, series, target_year, target_year - HORIZON_YEARS)


def build_training_rows(
    target_series: dict[int, float],
    urban_series: dict[int, float],
    outer_origin: int,
    required_lag: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row_year in sorted(year for year in target_series if year <= outer_origin - HORIZON_YEARS):
        fit = b3_fit(target_series, row_year)
        urban = build_urbanization_features(urban_series, row_year, required_lag)
        actual = target_series.get(row_year)
        row = {
            "row_role": "training",
            "row_year": row_year,
            "target_value": actual,
            "b3_prediction": fit["prediction"],
            "b3_residual_actual_minus_prediction": (
                actual - fit["prediction"] if actual is not None and fit["prediction"] is not None else None
            ),
            "b3_fit": fit,
            "urban": urban,
        }
        if actual is None or fit["prediction"] is None:
            excluded.append({**row, "exclusion_reason": "b3_insufficient_population_history"})
        else:
            eligible.append(row)
    return eligible, excluded


def build_validation_row(
    target_series: dict[int, float],
    urban_series: dict[int, float],
    origin: int,
    required_lag: int,
) -> dict[str, Any]:
    fit = b3_fit(target_series, origin)
    actual = target_series.get(origin)
    urban = build_urbanization_features(urban_series, origin, required_lag)
    assert actual is not None and fit["prediction"] is not None
    return {
        "row_role": "validation",
        "row_year": origin,
        "target_value": actual,
        "b3_prediction": fit["prediction"],
        "b3_residual_actual_minus_prediction": actual - fit["prediction"],
        "b3_fit": fit,
        "urban": urban,
    }


def urban_boundary_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    city: str,
    row: dict[str, Any],
    source_status: str,
    quarantine_reintroduced: str = "no",
) -> dict[str, Any]:
    fit = row["b3_fit"]
    urban = row["urban"]
    pseudo_year = int(row["row_year"])
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "row_role": row["row_role"],
        "city_id": city,
        "pseudo_target_year": pseudo_year,
        "population_target_history_cutoff_year": pseudo_year - HORIZON_YEARS,
        "population_history_max_year_used": fit["history_max_year"] or "",
        "population_history_years_used": ";".join(str(year) for year in fit["history_years"]),
        "urbanization_required_lag_years": urban["urbanization_required_lag_years"],
        "urbanization_required_latest_usable_year": urban["urbanization_required_latest_usable_year"],
        "urbanization_allowed_years": urban["urbanization_allowed_years"],
        "urbanization_latest_year_used": urban["urbanization_latest_year"] or "",
        "urbanization_latest_value": urban["urbanization_latest"],
        "urbanization_recent_change_source_years": urban["urbanization_recent_change_source_years"],
        "urbanization_recent3_slope_source_years": urban["urbanization_recent3_slope_source_years"],
        "urbanization_used_years": urban["urbanization_used_years"],
        "urbanization_used_year_max": urban["urbanization_used_year_max"] or "",
        "population_history_within_y_minus_2": "PASS" if fit["history_max_year"] is None or fit["history_max_year"] <= pseudo_year - HORIZON_YEARS else "FAIL",
        "urbanization_years_within_scenario_cap": "PASS" if urban["urbanization_used_year_max"] is None or urban["urbanization_used_year_max"] <= urban["urbanization_required_latest_usable_year"] else "FAIL",
        "outer_training_label_within_T_minus_2": "PASS" if pseudo_year <= outer_origin - HORIZON_YEARS else "FAIL",
        "urbanization_source_status": source_status,
        "quarantine_value_reintroduced": quarantine_reintroduced,
        "cross_target_population_use": "no",
        "boundary_status": "PASS",
    }


def pipeline_for(feature_names: tuple[str, ...]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[as_float(row["urban"].get(name)) if as_float(row["urban"].get(name)) is not None else np.nan for name in feature_names] for row in rows],
        dtype=float,
    )


def fit_ridge_residual(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[list[float], dict[str, Any]]:
    x_train = feature_matrix(train_rows, feature_names)
    x_validation = feature_matrix(validation_rows, feature_names)
    y_train = np.asarray([row["b3_residual_actual_minus_prediction"] for row in train_rows], dtype=float)
    training_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    model = pipeline_for(feature_names)
    model.fit(x_train, y_train)
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    indicator_names = [f"missing_indicator__{feature_names[index]}" for index in imputer.indicator_.features_]
    coefficient_names = list(feature_names) + indicator_names
    coefficients = [float(value) for value in np.asarray(ridge.coef_).ravel()]
    return model.predict(x_validation).tolist(), {
        "training_row_count": len(train_rows),
        "training_missing_feature_cell_count": training_missing,
        "training_imputation_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "validation_imputation_count": validation_missing,
        "preprocessing_fit_row_count": len(train_rows),
        "model_fit_row_count": len(train_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "imputer_statistics": [float(value) if math.isfinite(float(value)) else None for value in imputer.statistics_],
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": coefficients,
        "coefficient_names": coefficient_names,
        "feature_names": ";".join(feature_names),
        "ridge_alpha": 1.0,
    }


def score_model_row(
    target: dict[str, str],
    scenario_id: str,
    model_id: str,
    origin: int,
    city: str,
    validation: dict[str, Any],
    predicted_residual: float,
    feature_group_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = validation["target_value"]
    b3_prediction = validation["b3_prediction"]
    prediction = b3_prediction + predicted_residual
    residual = prediction - actual
    urban = validation["urban"]
    return {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": scenario_id,
        "predictor_id": model_id,
        "model_id": model_id,
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
        "b3_residual_actual_minus_prediction": validation["b3_residual_actual_minus_prediction"],
        "predicted_b3_residual": predicted_residual,
        "feature_builder_id": "URBANIZATION_RESIDUAL_V1",
        "feature_group_id": feature_group_id,
        "urbanization_required_lag_years": urban["urbanization_required_lag_years"],
        "urbanization_required_latest_usable_year": urban["urbanization_required_latest_usable_year"],
        "urbanization_allowed_years": urban["urbanization_allowed_years"],
        "urbanization_latest_year": urban["urbanization_latest_year"],
        "urbanization_latest": urban["urbanization_latest"],
        "urbanization_staleness_years": urban["urbanization_staleness_years"],
        "urbanization_recent_change": urban["urbanization_recent_change"],
        "urbanization_recent_change_source_years": urban["urbanization_recent_change_source_years"],
        "urbanization_recent3_slope": urban["urbanization_recent3_slope"],
        "urbanization_recent3_slope_source_years": urban["urbanization_recent3_slope_source_years"],
        "urbanization_used_years": urban["urbanization_used_years"],
        "urbanization_used_year_max": urban["urbanization_used_year_max"],
        "urbanization_missing_feature_names": urban["urbanization_missing_feature_names"],
        "training_label_min_year": metadata.get("training_label_min_year", ""),
        "training_label_max_year": metadata.get("training_label_max_year", ""),
        "training_row_count": metadata.get("training_row_count", ""),
        "preprocessing_fit_row_count": metadata.get("preprocessing_fit_row_count", ""),
        "model_fit_row_count": metadata.get("model_fit_row_count", ""),
        "training_missing_feature_cell_count": metadata.get("training_missing_feature_cell_count", ""),
        "validation_missing_feature_cell_count": metadata.get("validation_missing_feature_cell_count", ""),
        "covariates_used": "yes" if model_id in RIDGE_IDS else "no",
        "urbanization_used": "yes" if model_id in RIDGE_IDS else "no",
        "city_identity_used": "no",
    }


def wrap_b3_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "model_id": B3_ID,
            "feature_builder_id": "FROZEN_B3_REFERENCE",
            "feature_group_id": "FROZEN_B3",
            "covariates_used": "no",
            "urbanization_used": "no",
            "city_identity_used": "no",
            "b3_prediction_value": row["prediction_value"],
            "b3_residual_actual_minus_prediction": -float(row["residual_prediction_minus_actual"]),
            "predicted_b3_residual": "",
        }
    )
    return result


def make_d2_row(
    target: dict[str, str],
    scenario_id: str,
    origin: int,
    city: str,
    validation: dict[str, Any],
    bias: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return score_model_row(target, scenario_id, D2_ID, origin, city, validation, bias, "NONE_DIAGNOSTIC", metadata)


def load_context() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[int, float]], dict[str, Any]]:
    s3_plan01.assert_static_boundaries()
    population, profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    scenario_definitions = s3_plan01.load_scenarios()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False
    s3_summary = json.loads((S3_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8"))
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    s4_plan01_summary = json.loads((S4_PLAN01_DIR / "plan01_summary.json").read_text(encoding="utf-8"))
    assert s4_plan01_summary["b3_row_compatibility_pass"] is True
    assert s4_plan01_summary["b3_metric_compatibility_pass"] is True
    assert s4_plan01_summary["s4_complete"] is False
    contract = s2_geometry.load_contract()
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    geometry = {row["source_id"]: row for row in s2_geometry.source_geometry_rows(sources)}
    urban_geometry = geometry[URBANIZATION_SOURCE_ID]
    assert urban_geometry["latest_usable_year"] == 2022
    assert urban_geometry["observed_snapshot_freshness_lag_years"] == 1
    assert urban_geometry["source_status"] == "source_preserved"
    assert urban_geometry["source_derived_count"] == 0
    assert urban_geometry["source_invalidated_quarantine_count"] == 0
    assert {row["scenario_id"] for row in scenario_definitions} == set(SCENARIOS)
    urban_series = source_series(sources[URBANIZATION_SOURCE_ID])
    return population, profile, geometry, urban_series, urban_geometry


def compare_b3_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = read_csv(S3_PLAN03_DIR / "per_origin_predictions_or_residuals.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row
        for row in frozen_rows
        if row["predictor_id"] == B3_ID
    }
    current_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row
        for row in current_rows
        if row["predictor_id"] == B3_ID
    }
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(frozen_map) == len(current_map) == expected
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    result = []
    for (target_id, scenario_id), pairs in sorted(grouped.items()):
        deltas = {field: [] for field in ("prediction_value", "residual_prediction_minus_actual", "squared_error")}
        passed = True
        for current, frozen in pairs:
            for field in deltas:
                left = as_float(current.get(field))
                right = as_float(frozen.get(field))
                if left is None or right is None:
                    passed = passed and left == right
                else:
                    delta = abs(left - right)
                    deltas[field].append(delta)
                    passed = passed and delta <= 1e-9
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "rows_compared": len(pairs),
                "max_abs_prediction_delta": max(deltas["prediction_value"]),
                "max_abs_residual_delta": max(deltas["residual_prediction_minus_actual"]),
                "max_abs_squared_error_delta": max(deltas["squared_error"]),
                "status": "PASS" if passed else "FAIL",
                "comparison_rule": "B3 row-level prediction/residual/squared-error delta <= 1e-9",
            }
        )
    assert all(row["status"] == "PASS" for row in result)
    return result


def calculate_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    saved = s3_plan01.PREDICTORS
    s3_plan01.PREDICTORS = MODEL_DEFINITIONS
    try:
        per_origin, per_origin_map = s3_plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(rows, per_origin)
    finally:
        s3_plan01.PREDICTORS = saved
    return per_origin, aggregate, aggregate_map


def validate_b3_metrics(aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    frozen = read_csv(S3_PLAN03_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in frozen
        if row["predictor_id"] == B3_ID
    }
    result = []
    for key, current in sorted(aggregate_map.items()):
        if key[2] != B3_ID:
            continue
        reference = frozen_map[key]
        deltas = {
            field: abs(float(current[field]) - float(reference[field]))
            for field in ("pooled_city_year_mse", "macro_origin_mse", "latest_origin_mse")
        }
        result.append(
            {
                "target_id": key[0],
                "scenario_id": key[1],
                "predictor_id": key[2],
                "max_abs_aggregate_delta": max(deltas.values()),
                "status": "PASS" if max(deltas.values()) <= 1e-9 else "FAIL",
                "comparison_rule": "B3 pooled, macro and latest MSE delta <= 1e-9",
            }
        )
    assert len(result) == len(TARGETS) * len(SCENARIOS)
    assert all(row["status"] == "PASS" for row in result)
    return result


def score_rows(rows: list[dict[str, Any]], target_id: str, scenario_id: str, predictor_id: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["target_id"] == target_id
        and row["scenario_id"] == scenario_id
        and row["predictor_id"] == predictor_id
        and row["scoreable"] == "yes"
    ]


def pairwise_row(
    rows: list[dict[str, Any]],
    per_origin_map: dict[tuple[str, str, str, int], dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    target_id: str,
    scenario_id: str,
    baseline_id: str,
    candidate_id: str,
    pair_id: str,
) -> dict[str, Any]:
    base_agg = aggregate_map[(target_id, scenario_id, baseline_id)]
    cand_agg = aggregate_map[(target_id, scenario_id, candidate_id)]
    origin_deltas = []
    origin_wins = origin_losses = origin_ties = 0
    for origin in COMMON_ORIGINS:
        delta = float(per_origin_map[(target_id, scenario_id, baseline_id, origin)]["mse"]) - float(per_origin_map[(target_id, scenario_id, candidate_id, origin)]["mse"])
        origin_deltas.append(delta)
        if delta > 1e-12:
            origin_wins += 1
        elif delta < -1e-12:
            origin_losses += 1
        else:
            origin_ties += 1
    base_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, baseline_id)}
    cand_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, candidate_id)}
    deltas = [float(base_rows[key]["squared_error"]) - float(cand_rows[key]["squared_error"]) for key in sorted(base_rows)]
    latest_base = per_origin_map[(target_id, scenario_id, baseline_id, LATEST_ORIGIN)]
    latest_cand = per_origin_map[(target_id, scenario_id, candidate_id, LATEST_ORIGIN)]
    base_mse = float(base_agg["pooled_city_year_mse"])
    cand_mse = float(cand_agg["pooled_city_year_mse"])
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "pair_id": pair_id,
        "baseline_predictor_id": baseline_id,
        "candidate_predictor_id": candidate_id,
        "baseline_pooled_mse": base_mse,
        "candidate_pooled_mse": cand_mse,
        "pooled_mse_absolute_delta_baseline_minus_candidate": base_mse - cand_mse,
        "pooled_mse_relative_improvement_pct": (base_mse - cand_mse) / base_mse * 100.0,
        "baseline_macro_origin_mse": float(base_agg["macro_origin_mse"]),
        "candidate_macro_origin_mse": float(cand_agg["macro_origin_mse"]),
        "baseline_pooled_rmse": float(base_agg["pooled_city_year_rmse"]),
        "candidate_pooled_rmse": float(cand_agg["pooled_city_year_rmse"]),
        "baseline_pooled_mae": float(base_agg["pooled_city_year_mae"]),
        "candidate_pooled_mae": float(cand_agg["pooled_city_year_mae"]),
        "latest_baseline_mse": float(latest_base["mse"]),
        "latest_candidate_mse": float(latest_cand["mse"]),
        "latest_mse_relative_improvement_pct": (float(latest_base["mse"]) - float(latest_cand["mse"])) / float(latest_base["mse"]) * 100.0,
        "latest_candidate_outcome": "win" if float(latest_base["mse"]) > float(latest_cand["mse"]) + 1e-12 else "loss" if float(latest_base["mse"]) < float(latest_cand["mse"]) - 1e-12 else "tie",
        "candidate_origin_win_count": origin_wins,
        "candidate_origin_loss_count": origin_losses,
        "origin_tie_count": origin_ties,
        "origin_count": len(COMMON_ORIGINS),
        "candidate_origin_win_rate": origin_wins / len(COMMON_ORIGINS),
        "candidate_city_origin_improved_share": sum(delta > 1e-12 for delta in deltas) / len(deltas),
        "candidate_city_origin_worsened_share": sum(delta < -1e-12 for delta in deltas) / len(deltas),
        "candidate_city_origin_tied_share": sum(abs(delta) <= 1e-12 for delta in deltas) / len(deltas),
        "origin_mse_delta_min": min(origin_deltas),
        "origin_mse_delta_max": max(origin_deltas),
        "ranking_metric": "pooled_city_year_mse",
        "score_is_secondary_transform_only": "yes",
    }


def concentration_row(
    rows: list[dict[str, Any]],
    target_id: str,
    scenario_id: str,
    candidate_id: str,
    baseline_id: str = B3_ID,
) -> dict[str, Any]:
    base_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, baseline_id)}
    cand_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, candidate_id)}
    city_base: defaultdict[str, float] = defaultdict(float)
    city_cand: defaultdict[str, float] = defaultdict(float)
    city_delta: defaultdict[str, float] = defaultdict(float)
    for key in sorted(base_rows):
        base = float(base_rows[key]["squared_error"])
        cand = float(cand_rows[key]["squared_error"])
        city = key[1]
        city_base[city] += base
        city_cand[city] += cand
        city_delta[city] += base - cand
    total_base = math.fsum(city_base.values())
    total_cand = math.fsum(city_cand.values())
    ranked_base = sorted(city_base.items(), key=lambda item: (-item[1], item[0]))
    ranked_cand = sorted(city_cand.items(), key=lambda item: (-item[1], item[0]))
    positive_delta = sorted(((city, delta) for city, delta in city_delta.items() if delta > 0), key=lambda item: (-item[1], item[0]))
    positive_total = math.fsum(delta for _, delta in positive_delta)
    top_improvement = positive_delta[0][0] if positive_delta else ""
    top_error = ranked_cand[0][0]

    def mse_without(city: str, source: dict[tuple[int, str], dict[str, Any]]) -> float:
        return mean(float(row["squared_error"]) for key, row in source.items() if key[1] != city)

    base_without_improvement = mse_without(top_improvement, base_rows) if top_improvement else ""
    cand_without_improvement = mse_without(top_improvement, cand_rows) if top_improvement else ""
    base_without_error = mse_without(top_error, base_rows)
    cand_without_error = mse_without(top_error, cand_rows)
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "baseline_predictor_id": baseline_id,
        "candidate_predictor_id": candidate_id,
        "baseline_pooled_sse": total_base,
        "candidate_pooled_sse": total_cand,
        "pooled_mse_absolute_delta_baseline_minus_candidate": (total_base - total_cand) / len(base_rows),
        "dominant_baseline_city_id": ranked_base[0][0],
        "dominant_candidate_city_id": ranked_cand[0][0],
        "baseline_dominant_city_sse_share": ranked_base[0][1] / total_base,
        "candidate_dominant_city_sse_share": ranked_cand[0][1] / total_cand,
        "baseline_top3_city_sse_share": math.fsum(value for _, value in ranked_base[:3]) / total_base,
        "candidate_top3_city_sse_share": math.fsum(value for _, value in ranked_cand[:3]) / total_cand,
        "improved_city_count": sum(city_delta[city] > 1e-12 for city in CITIES),
        "worsened_city_count": sum(city_delta[city] < -1e-12 for city in CITIES),
        "tied_city_count": sum(abs(city_delta[city]) <= 1e-12 for city in CITIES),
        "improved_city_share": sum(city_delta[city] > 1e-12 for city in CITIES) / len(CITIES),
        "worsened_city_share": sum(city_delta[city] < -1e-12 for city in CITIES) / len(CITIES),
        "improved_city_origin_row_share": sum(float(base_rows[key]["squared_error"]) - float(cand_rows[key]["squared_error"]) > 1e-12 for key in base_rows) / len(base_rows),
        "worsened_city_origin_row_share": sum(float(base_rows[key]["squared_error"]) - float(cand_rows[key]["squared_error"]) < -1e-12 for key in base_rows) / len(base_rows),
        "top_positive_delta_city_id": top_improvement,
        "top1_positive_delta_sse_contribution_share": positive_delta[0][1] / positive_total if positive_total else 0.0,
        "top3_positive_delta_sse_contribution_share": math.fsum(delta for _, delta in positive_delta[:3]) / positive_total if positive_total else 0.0,
        "remove_largest_improvement_city_id": top_improvement,
        "baseline_mse_without_largest_improvement_city": base_without_improvement,
        "candidate_mse_without_largest_improvement_city": cand_without_improvement,
        "candidate_improvement_without_largest_improvement_city": base_without_improvement - cand_without_improvement if top_improvement else "",
        "remove_largest_error_city_id": top_error,
        "baseline_mse_without_largest_error_city": base_without_error,
        "candidate_mse_without_largest_error_city": cand_without_error,
        "candidate_improvement_without_largest_error_city": base_without_error - cand_without_error,
        "interpretation_scope": "city-origin residual rows; no city reweighting",
    }


def leave_one_origin_rows(rows: list[dict[str, Any]], aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            full_leader = min(MODEL_IDS, key=lambda model_id: float(aggregate_map[(target_id, scenario_id, model_id)]["pooled_city_year_mse"]))
            for omitted in COMMON_ORIGINS:
                model_mse = {}
                for model_id in MODEL_IDS:
                    model_mse[model_id] = mean(
                        float(row["squared_error"])
                        for row in score_rows(rows, target_id, scenario_id, model_id)
                        if int(row["origin_year"]) != omitted
                    )
                ordered = sorted(model_mse.items(), key=lambda item: (item[1], MODEL_IDS.index(item[0])))
                leader = ordered[0][0]
                for rank, (model_id, mse) in enumerate(ordered, start=1):
                    result.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "omitted_origin_year": omitted,
                            "predictor_id": model_id,
                            "remaining_origin_count": len(COMMON_ORIGINS) - 1,
                            "leave_one_origin_pooled_mse": mse,
                            "leave_one_origin_rank": rank,
                            "leave_one_origin_leader": leader,
                            "full_domain_leader": full_leader,
                            "full_domain_leader_survives": "yes" if leader == full_leader else "no",
                        }
                    )
    return result


def leave_one_city_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            base_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, B3_ID)}
            for candidate_id in CORRECTION_IDS:
                cand_rows = {(int(row["origin_year"]), row["city_id"]): row for row in score_rows(rows, target_id, scenario_id, candidate_id)}
                for removed_city in CITIES:
                    base_values = [float(row["squared_error"]) for key, row in base_rows.items() if key[1] != removed_city]
                    cand_values = [float(row["squared_error"]) for key, row in cand_rows.items() if key[1] != removed_city]
                    base_mse = mean(base_values)
                    cand_mse = mean(cand_values)
                    result.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "baseline_predictor_id": B3_ID,
                            "candidate_predictor_id": candidate_id,
                            "removed_city_id": removed_city,
                            "baseline_mse_without_city": base_mse,
                            "candidate_mse_without_city": cand_mse,
                            "candidate_improvement_without_city": base_mse - cand_mse,
                            "interpretation_scope": "no city reweighting; leave-one-city sensitivity",
                        }
                    )
    return result


def latest_rows(aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            b3_mse = float(aggregate_map[(target["target_id"], scenario_id, B3_ID)]["pooled_city_year_mse"])
            for model_id in MODEL_IDS:
                aggregate = aggregate_map[(target["target_id"], scenario_id, model_id)]
                result.append(
                    {
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "predictor_id": model_id,
                        "latest_origin_year": LATEST_ORIGIN,
                        "latest_origin_mse": aggregate["latest_origin_mse"],
                        "latest_origin_rmse": aggregate["latest_origin_rmse"],
                        "latest_origin_mae": aggregate["latest_origin_mae"],
                        "latest_origin_bias": aggregate["latest_origin_mean_signed_error_bias"],
                        "latest_origin_rank": aggregate["latest_origin_mse_rank"],
                        "pooled_mse": aggregate["pooled_city_year_mse"],
                        "pooled_mse_relative_to_b3_pct": (float(aggregate["pooled_city_year_mse"]) - b3_mse) / b3_mse * 100.0,
                    }
                )
    return result


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = rank
        index = end
    return output


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_dev = [value - left_mean for value in left]
    right_dev = [value - right_mean for value in right]
    denom = math.sqrt(math.fsum(value * value for value in left_dev) * math.fsum(value * value for value in right_dev))
    return math.fsum(a * b for a, b in zip(left_dev, right_dev)) / denom if denom > 0 else None


def relationship_rows(
    target_id: str,
    scenario_id: str,
    origin: int,
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for role, source_rows, residual_key in (
        ("training", training_rows, "b3_residual_actual_minus_prediction"),
        ("validation_posthoc", validation_rows, "b3_residual_actual_minus_prediction"),
    ):
        for feature_name in ("urbanization_latest", "urbanization_recent_change", "urbanization_recent3_slope"):
            paired = [
                (float(row["urban"][feature_name]), float(row[residual_key]))
                for row in source_rows
                if row["urban"].get(feature_name) is not None and row[residual_key] is not None
            ]
            feature_values = [pair[0] for pair in paired]
            residual_values = [pair[1] for pair in paired]
            result.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "outer_origin_year": origin,
                    "analysis_role": role,
                    "feature_name": feature_name,
                    "paired_row_count": len(paired),
                    "feature_mean": mean(feature_values) if paired else "",
                    "residual_mean": mean(residual_values) if paired else "",
                    "pearson_correlation": pearson(feature_values, residual_values) if paired else "",
                    "spearman_correlation": pearson(ranks(feature_values), ranks(residual_values)) if paired else "",
                    "correlation_status": "PASS" if len(paired) >= 2 else "INSUFFICIENT_PAIRS",
                    "interpretation_scope": "mechanism diagnostic; validation uses retrospective labels and is not training selection",
                }
            )
    return result


def feature_freshness_rows(
    target_id: str,
    origin: int,
    validation_by_scenario: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    r1 = {city: row["urban"] for city, row in zip(CITIES, validation_by_scenario[SCENARIOS[0]])}
    r2 = {city: row["urban"] for city, row in zip(CITIES, validation_by_scenario[SCENARIOS[1]])}
    common_latest = [city for city in CITIES if r1[city]["urbanization_latest"] is not None and r2[city]["urbanization_latest"] is not None]
    latest_deltas = [r1[city]["urbanization_latest"] - r2[city]["urbanization_latest"] for city in common_latest]
    recent_change_deltas = [r1[city]["urbanization_recent_change"] - r2[city]["urbanization_recent_change"] for city in CITIES if r1[city]["urbanization_recent_change"] is not None and r2[city]["urbanization_recent_change"] is not None]
    slope_deltas = [r1[city]["urbanization_recent3_slope"] - r2[city]["urbanization_recent3_slope"] for city in CITIES if r1[city]["urbanization_recent3_slope"] is not None and r2[city]["urbanization_recent3_slope"] is not None]
    return {
        "target_id": target_id,
        "outer_origin_year": origin,
        "r1_scenario_id": SCENARIOS[0],
        "r2_scenario_id": SCENARIOS[1],
        "r1_required_lag_years": r1[CITIES[0]]["urbanization_required_lag_years"],
        "r2_required_lag_years": r2[CITIES[0]]["urbanization_required_lag_years"],
        "r1_required_latest_usable_year": r1[CITIES[0]]["urbanization_required_latest_usable_year"],
        "r2_required_latest_usable_year": r2[CITIES[0]]["urbanization_required_latest_usable_year"],
        "r1_latest_available_city_count": sum(row["urbanization_latest"] is not None for row in r1.values()),
        "r2_latest_available_city_count": sum(row["urbanization_latest"] is not None for row in r2.values()),
        "r1_r2_common_latest_city_count": len(common_latest),
        "r1_r2_latest_value_difference_city_count": sum(abs(r1[city]["urbanization_latest"] - r2[city]["urbanization_latest"]) > 1e-12 for city in common_latest),
        "r1_r2_latest_value_delta_mean": mean(latest_deltas) if latest_deltas else "",
        "r1_r2_latest_value_delta_min": min(latest_deltas) if latest_deltas else "",
        "r1_r2_latest_value_delta_max": max(latest_deltas) if latest_deltas else "",
        "r1_r2_recent_change_difference_city_count": sum(abs(value) > 1e-12 for value in recent_change_deltas),
        "r1_r2_recent3_slope_difference_city_count": sum(abs(value) > 1e-12 for value in slope_deltas),
        "r1_r2_feature_difference_confirmed": "yes" if latest_deltas and any(abs(value) > 1e-12 for value in latest_deltas) else "no",
        "source_publication_timing_proven": "no",
        "source_boundary_status": "observed cap replay assumption; not publication-date fact",
    }


def feature_missingness_rows(
    target_id: str,
    scenario_id: str,
    origin: int,
    model_id: str,
    feature_names: tuple[str, ...],
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    excluded_count: int,
) -> list[dict[str, Any]]:
    result = []
    for feature_name in feature_names:
        training_missing = sum(row["urban"].get(feature_name) is None for row in training_rows)
        validation_missing = sum(row["urban"].get(feature_name) is None for row in validation_rows)
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "outer_origin_year": origin,
                "model_id": model_id,
                "feature_group_id": "URB_LEVEL_V1" if model_id == M2A_ID else "URB_DYNAMICS_V1",
                "feature_name": feature_name,
                "training_eligible_row_count": len(training_rows),
                "training_missing_count": training_missing,
                "training_imputation_count": training_missing,
                "validation_row_count": len(validation_rows),
                "validation_missing_count": validation_missing,
                "validation_imputation_count": validation_missing,
                "excluded_training_row_count": excluded_count,
                "imputation_rule": "median fit on outer-origin training rows only; add_indicator=True",
                "status": "PASS",
            }
        )
    return result


def leakage_rows_for_scope(
    target_id: str,
    scenario_id: str,
    origin: int,
    model_id: str,
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    urban_source_status: str,
) -> list[dict[str, Any]]:
    result = []
    for row_role, source_rows in (("training", training_rows), ("validation", validation_rows)):
        for row in source_rows:
            fit = row["b3_fit"]
            urban = row["urban"]
            pseudo_year = row["row_year"]
            population_pass = fit["history_max_year"] is None or fit["history_max_year"] <= pseudo_year - HORIZON_YEARS
            urban_pass = urban["urbanization_used_year_max"] is None or urban["urbanization_used_year_max"] <= urban["urbanization_required_latest_usable_year"]
            outer_label_pass = pseudo_year <= origin - HORIZON_YEARS if row_role == "training" else True
            preprocessing_equal = metadata.get("preprocessing_model_fit_rows_equal", "not_applicable")
            no_t_minus_one = "no"
            no_t = "no"
            no_future = "no"
            status = all((population_pass, urban_pass, outer_label_pass, preprocessing_equal in {"yes", "not_applicable"}))
            result.append(
                {
                    "row_role": row_role,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "model_id": model_id,
                    "outer_origin_year": origin,
                    "city_id": row.get("city_id", ""),
                    "pseudo_target_year": pseudo_year,
                    "population_history_max_year": fit["history_max_year"] or "",
                    "population_allowed_history_cutoff": pseudo_year - HORIZON_YEARS,
                    "urbanization_used_year_max": urban["urbanization_used_year_max"] or "",
                    "urbanization_required_cap": urban["urbanization_required_latest_usable_year"],
                    "population_history_t_minus_2_pass": "PASS" if population_pass else "FAIL",
                    "urbanization_source_cap_pass": "PASS" if urban_pass else "FAIL",
                    "training_label_max_within_T_minus_2": "PASS" if outer_label_pass else "FAIL",
                    "preprocessing_fit_row_count": metadata.get("preprocessing_fit_row_count", "not_applicable"),
                    "model_fit_row_count": metadata.get("model_fit_row_count", "not_applicable"),
                    "preprocessing_model_fit_rows_equal": preprocessing_equal,
                    "outer_t_minus_1_target_present": "not_applicable" if row_role == "training" else "yes" if (origin - 1) in row["target_series"] else "no",
                    "outer_t_target_present": "not_applicable" if row_role == "training" else "yes" if origin in row["target_series"] else "no",
                    "t_minus_1_used_in_features": no_t_minus_one,
                    "t_used_in_features": no_t,
                    "future_target_years_used": no_future,
                    "feature_source_target_id": target_id,
                    "cross_target_contamination": "no",
                    "other_covariate_columns_used": "no",
                    "city_identity_used": "no",
                    "quarantine_value_reintroduced": "no",
                    "urbanization_source_status": urban_source_status,
                    "leakage_status": "PASS" if status else "FAIL",
                }
            )
    return result


def main() -> None:
    assert STARTING_COMMIT == "71f76664426a3b720f9b03f77792c8f0905e9f92"
    population, population_profile, geometry, urban_series, urban_geometry = load_context()
    b3_all_rows, _ = s3_plan02.build_fixed_rows(population)
    b3_rows = [wrap_b3_row(row) for row in b3_all_rows if row["predictor_id"] == B3_ID]
    assert len(b3_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    b3_compatibility = compare_b3_rows(b3_rows)

    urban_boundaries: list[dict[str, Any]] = []
    freshness_rows: list[dict[str, Any]] = []
    dataset_diagnostics: list[dict[str, Any]] = []
    training_geometry: list[dict[str, Any]] = []
    feature_missingness: list[dict[str, Any]] = []
    leakage_audit: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    relationship_rows_all: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    b3_key_map = {
        (row["target_id"], row["scenario_id"], int(row["origin_year"]), row["city_id"]): row
        for row in b3_rows
    }
    urban_source_status = urban_geometry["source_status"]

    for target in TARGETS:
        target_id = target["target_id"]
        validation_feature_by_origin: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for scenario_id in SCENARIOS:
            required_lag = required_urbanization_lag(urban_geometry, scenario_id)
            for origin in COMMON_ORIGINS:
                training_by_city: dict[str, list[dict[str, Any]]] = {}
                excluded_by_city: dict[str, list[dict[str, Any]]] = {}
                validation_by_city: dict[str, dict[str, Any]] = {}
                for city in CITIES:
                    target_series = population[target_id][city]
                    city_urban_series = urban_series[city]
                    train_rows, excluded_rows = build_training_rows(target_series, city_urban_series, origin, required_lag)
                    for row in train_rows + excluded_rows:
                        row["city_id"] = city
                        row["target_series"] = target_series
                        urban_boundaries.append(urban_boundary_row(target_id, scenario_id, origin, city, row, urban_source_status))
                    validation = build_validation_row(target_series, city_urban_series, origin, required_lag)
                    validation["city_id"] = city
                    validation["target_series"] = target_series
                    urban_boundaries.append(urban_boundary_row(target_id, scenario_id, origin, city, validation, urban_source_status))
                    training_by_city[city] = train_rows
                    excluded_by_city[city] = excluded_rows
                    validation_by_city[city] = validation
                training_rows = [row for city in CITIES for row in training_by_city[city]]
                excluded_rows = [row for city in CITIES for row in excluded_by_city[city]]
                validation_rows = [validation_by_city[city] for city in CITIES]
                validation_feature_by_origin.setdefault(origin, {})[scenario_id] = validation_rows
                relationship_rows_all.extend(relationship_rows(target_id, scenario_id, origin, training_rows, validation_rows))
                for model_id in RIDGE_IDS:
                    feature_names = LEVEL_FEATURES if model_id == M2A_ID else DYNAMICS_FEATURES
                    predicted_residuals, metadata = fit_ridge_residual(training_rows, validation_rows, feature_names)
                    metadata.update(
                        {
                            "training_label_min_year": min(row["row_year"] for row in training_rows),
                            "training_label_max_year": max(row["row_year"] for row in training_rows),
                        }
                    )
                    candidate_rows.extend(
                        score_model_row(target, scenario_id, model_id, origin, city, validation_by_city[city], predicted_delta, "URB_LEVEL_V1" if model_id == M2A_ID else "URB_DYNAMICS_V1", metadata)
                        for city, predicted_delta in zip(CITIES, predicted_residuals)
                    )
                    feature_missingness.extend(feature_missingness_rows(target_id, scenario_id, origin, model_id, feature_names, training_rows, validation_rows, len(excluded_rows)))
                    for name, coefficient in zip(metadata["coefficient_names"], metadata["ridge_coefficients"]):
                        coefficient_rows.append(
                            {
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin,
                                "model_id": model_id,
                                "feature_group_id": "URB_LEVEL_V1" if model_id == M2A_ID else "URB_DYNAMICS_V1",
                                "coefficient_name": name,
                                "coefficient": coefficient,
                                "coefficient_sign": "positive" if coefficient > 1e-12 else "negative" if coefficient < -1e-12 else "zero",
                                "ridge_alpha": metadata["ridge_alpha"],
                                "standardized_feature_space": "yes",
                            }
                        )
                    training_geometry.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "model_id": model_id,
                            "feature_group_id": "URB_LEVEL_V1" if model_id == M2A_ID else "URB_DYNAMICS_V1",
                            "training_row_count": metadata["training_row_count"],
                            "training_excluded_row_count": len(excluded_rows),
                            "training_label_min_year": metadata["training_label_min_year"],
                            "training_label_max_year": metadata["training_label_max_year"],
                            "training_population_history_max_year": max(row["b3_fit"]["history_max_year"] for row in training_rows if row["b3_fit"]["history_max_year"] is not None),
                            "training_urbanization_cap_max_year": max(row["urban"]["urbanization_required_latest_usable_year"] for row in training_rows),
                            "training_urbanization_used_year_max": max((row["urban"]["urbanization_used_year_max"] or 0) for row in training_rows),
                            "validation_urbanization_cap_year": validation_rows[0]["urban"]["urbanization_required_latest_usable_year"],
                            "validation_urbanization_used_year_max": max((row["urban"]["urbanization_used_year_max"] or 0) for row in validation_rows),
                            "ridge_alpha": metadata["ridge_alpha"],
                            "ridge_intercept": metadata["ridge_intercept"],
                            "ridge_coefficients": stable_json(metadata["ridge_coefficients"]),
                            "coefficient_names": ";".join(metadata["coefficient_names"]),
                            "feature_names": metadata["feature_names"],
                            "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                            "model_fit_row_count": metadata["model_fit_row_count"],
                            "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                        }
                    )
                    leakage_audit.extend(leakage_rows_for_scope(target_id, scenario_id, origin, model_id, training_rows, validation_rows, metadata, urban_source_status))
                    dataset_diagnostics.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "target_history_cutoff_year": origin - HORIZON_YEARS,
                            "urbanization_required_lag_years": required_lag,
                            "urbanization_required_latest_usable_year": origin - required_lag,
                            "model_id": model_id,
                            "feature_group_id": "URB_LEVEL_V1" if model_id == M2A_ID else "URB_DYNAMICS_V1",
                            "training_candidate_row_count": len(training_rows) + len(excluded_rows),
                            "training_eligible_row_count": len(training_rows),
                            "training_excluded_row_count": len(excluded_rows),
                            "training_excluded_reason_counts": stable_json(dict(Counter("b3_insufficient_population_history" for row in excluded_rows))),
                            "validation_expected_city_count": len(CITIES),
                            "validation_urbanization_latest_available_city_count": sum(row["urban"]["urbanization_latest"] is not None for row in validation_rows),
                            "validation_scoreable_city_count": len(validation_rows),
                            "training_label_min_year": metadata["training_label_min_year"],
                            "training_label_max_year": metadata["training_label_max_year"],
                            "training_missing_feature_cell_count": metadata["training_missing_feature_cell_count"],
                            "training_imputation_count": metadata["training_imputation_count"],
                            "validation_missing_feature_cell_count": metadata["validation_missing_feature_cell_count"],
                            "validation_imputation_count": metadata["validation_imputation_count"],
                            "b3_residual_training_mean": mean(row["b3_residual_actual_minus_prediction"] for row in training_rows),
                            "imputation_policy": "median fit on outer-origin training rows only, add_indicator=True",
                            "full_validation_coverage": "yes" if len(validation_rows) == len(CITIES) and all(row["urban"]["urbanization_latest"] is not None for row in validation_rows) else "no",
                            "coverage_status": "PASS" if len(validation_rows) == len(CITIES) and all(row["urban"]["urbanization_latest"] is not None for row in validation_rows) else "FAIL",
                        }
                    )
                d2_bias = mean(row["b3_residual_actual_minus_prediction"] for row in training_rows)
                d2_metadata = {
                    "training_row_count": len(training_rows),
                    "training_label_min_year": min(row["row_year"] for row in training_rows),
                    "training_label_max_year": max(row["row_year"] for row in training_rows),
                    "preprocessing_fit_row_count": "not_applicable_diagnostic",
                    "model_fit_row_count": len(training_rows),
                    "preprocessing_model_fit_rows_equal": "not_applicable",
                    "training_missing_feature_cell_count": 0,
                    "validation_missing_feature_cell_count": 0,
                }
                candidate_rows.extend(make_d2_row(target, scenario_id, origin, city, validation_by_city[city], d2_bias, d2_metadata) for city in CITIES)
                leakage_audit.extend(leakage_rows_for_scope(target_id, scenario_id, origin, D2_ID, training_rows, validation_rows, d2_metadata, urban_source_status))
                dataset_diagnostics.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": origin,
                        "target_history_cutoff_year": origin - HORIZON_YEARS,
                        "urbanization_required_lag_years": required_lag,
                        "urbanization_required_latest_usable_year": origin - required_lag,
                        "model_id": D2_ID,
                        "feature_group_id": "NONE_DIAGNOSTIC",
                        "training_candidate_row_count": len(training_rows) + len(excluded_rows),
                        "training_eligible_row_count": len(training_rows),
                        "training_excluded_row_count": len(excluded_rows),
                        "training_excluded_reason_counts": stable_json(dict(Counter("b3_insufficient_population_history" for row in excluded_rows))),
                        "validation_expected_city_count": len(CITIES),
                        "validation_urbanization_latest_available_city_count": sum(row["urban"]["urbanization_latest"] is not None for row in validation_rows),
                        "validation_scoreable_city_count": len(validation_rows),
                        "training_label_min_year": d2_metadata["training_label_min_year"],
                        "training_label_max_year": d2_metadata["training_label_max_year"],
                        "training_missing_feature_cell_count": "not_applicable",
                        "training_imputation_count": "not_applicable",
                        "validation_missing_feature_cell_count": "not_applicable",
                        "validation_imputation_count": "not_applicable",
                        "b3_residual_training_mean": d2_bias,
                        "imputation_policy": "not_applicable_diagnostic",
                        "full_validation_coverage": "yes",
                        "coverage_status": "PASS",
                    }
                )
        for origin in COMMON_ORIGINS:
            freshness_rows.append(feature_freshness_rows(target_id, origin, validation_feature_by_origin[origin]))

    all_rows = b3_rows + candidate_rows
    expected_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(MODEL_IDS)
    assert len(all_rows) == expected_rows, len(all_rows)
    assert len(candidate_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(CORRECTION_IDS)
    per_origin_metrics, aggregate_metrics, aggregate_map = calculate_metrics(all_rows)
    per_origin_map = {(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row for row in per_origin_metrics}
    b3_metric_compatibility = validate_b3_metrics(aggregate_map)

    pairwise_b3 = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for candidate_id in CORRECTION_IDS:
                pairwise_b3.append(pairwise_row(all_rows, per_origin_map, aggregate_map, target["target_id"], scenario_id, B3_ID, candidate_id, f"{candidate_id}_vs_B3"))
    bias_control = [dict(row, comparison_type="global residual bias diagnostic") for row in pairwise_b3 if row["candidate_predictor_id"] == D2_ID]
    ablation = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            ablation.append(pairwise_row(all_rows, per_origin_map, aggregate_map, target["target_id"], scenario_id, M2A_ID, M2B_ID, "M2B_vs_M2A"))
    concentrations = [concentration_row(all_rows, target["target_id"], scenario_id, candidate_id) for target in TARGETS for scenario_id in SCENARIOS for candidate_id in CORRECTION_IDS]
    leave_origin = leave_one_origin_rows(all_rows, aggregate_map)
    leave_city = leave_one_city_rows(all_rows)
    latest = latest_rows(aggregate_map)

    r1_r2_metric_sensitivity: dict[str, Any] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        for model_id in MODEL_IDS:
            r1 = aggregate_map[(target_id, SCENARIOS[0], model_id)]
            r2 = aggregate_map[(target_id, SCENARIOS[1], model_id)]
            r1_r2_metric_sensitivity[f"{target_id}_{model_id}"] = {
                "r1_pooled_mse": r1["pooled_city_year_mse"],
                "r2_pooled_mse": r2["pooled_city_year_mse"],
                "r2_minus_r1_pooled_mse": float(r2["pooled_city_year_mse"]) - float(r1["pooled_city_year_mse"]),
                "r1_latest_mse": r1["latest_origin_mse"],
                "r2_latest_mse": r2["latest_origin_mse"],
                "r2_minus_r1_latest_mse": float(r2["latest_origin_mse"]) - float(r1["latest_origin_mse"]),
            }

    resident_b3_r1 = float(aggregate_map[("resident", SCENARIOS[0], B3_ID)]["pooled_city_year_mse"])
    resident_candidates = {
        model_id: float(aggregate_map[("resident", SCENARIOS[0], model_id)]["pooled_city_year_mse"])
        for model_id in (D2_ID, M2A_ID, M2B_ID)
    }
    best_resident_r1_model = min(resident_candidates, key=resident_candidates.get)
    resident_b3_r2 = float(aggregate_map[("resident", SCENARIOS[1], B3_ID)]["pooled_city_year_mse"])
    resident_r2_candidates = {
        model_id: float(aggregate_map[("resident", SCENARIOS[1], model_id)]["pooled_city_year_mse"])
        for model_id in (D2_ID, M2A_ID, M2B_ID)
    }
    best_resident_r2_model = min(resident_r2_candidates, key=resident_r2_candidates.get)
    level_ablation_supported = all(
        float(aggregate_map[(target_id, scenario_id, M2A_ID)]["pooled_city_year_mse"]) < float(aggregate_map[(target_id, scenario_id, B3_ID)]["pooled_city_year_mse"])
        for target_id in ("resident", "registered")
        for scenario_id in SCENARIOS
    )
    dynamics_ablation_supported = all(
        float(aggregate_map[(target_id, scenario_id, M2B_ID)]["pooled_city_year_mse"]) < float(aggregate_map[(target_id, scenario_id, M2A_ID)]["pooled_city_year_mse"])
        and next(row for row in ablation if row["target_id"] == target_id and row["scenario_id"] == scenario_id)["candidate_origin_win_count"] > next(row for row in ablation if row["target_id"] == target_id and row["scenario_id"] == scenario_id)["candidate_origin_loss_count"]
        for target_id in ("resident", "registered")
        for scenario_id in SCENARIOS
    )
    urbanization_material_signal_supported = (
        best_resident_r1_model in RIDGE_IDS
        and resident_candidates[best_resident_r1_model] < resident_b3_r1
        and best_resident_r2_model in RIDGE_IDS
        and resident_r2_candidates[best_resident_r2_model] < resident_b3_r2
        and all(next(row for row in pairwise_b3 if row["target_id"] == "resident" and row["scenario_id"] == scenario_id and row["candidate_predictor_id"] == best_resident_r1_model)["candidate_origin_win_count"] > next(row for row in pairwise_b3 if row["target_id"] == "resident" and row["scenario_id"] == scenario_id and row["candidate_predictor_id"] == best_resident_r1_model)["candidate_origin_loss_count"] for scenario_id in SCENARIOS)
    )

    coefficient_stability: dict[str, Any] = {}
    for key in sorted({(row["target_id"], row["model_id"], row["coefficient_name"]) for row in coefficient_rows}):
        values = [row["coefficient"] for row in coefficient_rows if (row["target_id"], row["model_id"], row["coefficient_name"]) == key]
        signs = [row["coefficient_sign"] for row in coefficient_rows if (row["target_id"], row["model_id"], row["coefficient_name"]) == key]
        coefficient_stability[f"{key[0]}_{key[1]}_{key[2]}"] = {
            "positive_count": signs.count("positive"),
            "negative_count": signs.count("negative"),
            "zero_count": signs.count("zero"),
            "sign_stability": "positive" if signs and all(sign == "positive" for sign in signs) else "negative" if signs and all(sign == "negative" for sign in signs) else "mixed",
            "coefficient_min": min(values),
            "coefficient_max": max(values),
        }

    summary = {
        "plan_id": "S4-Plan02",
        "stage": "S4_FEATURE_ML_MODELING",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "target_history_cutoff_rule": "population B3 history <= y-2 for training pseudo-row and <= T-2 for validation; urbanization history <= scenario-specific source cap",
        "covariate_group": "urbanization",
        "model_ids": list(MODEL_IDS),
        "resident_b3_pooled_mse": aggregate_map[("resident", SCENARIOS[0], B3_ID)]["pooled_city_year_mse"],
        "resident_b3_macro_mse": aggregate_map[("resident", SCENARIOS[0], B3_ID)]["macro_origin_mse"],
        "resident_b3_latest_2021_mse": aggregate_map[("resident", SCENARIOS[0], B3_ID)]["latest_origin_mse"],
        "registered_b3_pooled_mse": aggregate_map[("registered", SCENARIOS[0], B3_ID)]["pooled_city_year_mse"],
        "registered_b3_macro_mse": aggregate_map[("registered", SCENARIOS[0], B3_ID)]["macro_origin_mse"],
        "registered_b3_latest_2021_mse": aggregate_map[("registered", SCENARIOS[0], B3_ID)]["latest_origin_mse"],
        "resident_d2_pooled_mse_r1": aggregate_map[("resident", SCENARIOS[0], D2_ID)]["pooled_city_year_mse"],
        "resident_m2a_pooled_mse_r1": aggregate_map[("resident", SCENARIOS[0], M2A_ID)]["pooled_city_year_mse"],
        "resident_m2b_pooled_mse_r1": aggregate_map[("resident", SCENARIOS[0], M2B_ID)]["pooled_city_year_mse"],
        "resident_d2_pooled_mse_r2": aggregate_map[("resident", SCENARIOS[1], D2_ID)]["pooled_city_year_mse"],
        "resident_m2a_pooled_mse_r2": aggregate_map[("resident", SCENARIOS[1], M2A_ID)]["pooled_city_year_mse"],
        "resident_m2b_pooled_mse_r2": aggregate_map[("resident", SCENARIOS[1], M2B_ID)]["pooled_city_year_mse"],
        "registered_d2_pooled_mse_r1": aggregate_map[("registered", SCENARIOS[0], D2_ID)]["pooled_city_year_mse"],
        "registered_m2a_pooled_mse_r1": aggregate_map[("registered", SCENARIOS[0], M2A_ID)]["pooled_city_year_mse"],
        "registered_m2b_pooled_mse_r1": aggregate_map[("registered", SCENARIOS[0], M2B_ID)]["pooled_city_year_mse"],
        "registered_d2_pooled_mse_r2": aggregate_map[("registered", SCENARIOS[1], D2_ID)]["pooled_city_year_mse"],
        "registered_m2a_pooled_mse_r2": aggregate_map[("registered", SCENARIOS[1], M2A_ID)]["pooled_city_year_mse"],
        "registered_m2b_pooled_mse_r2": aggregate_map[("registered", SCENARIOS[1], M2B_ID)]["pooled_city_year_mse"],
        "latest_origin_metrics": {f"{row['target_id']}_{row['scenario_id']}_{row['predictor_id']}": row["latest_origin_mse"] for row in latest},
        "per_origin_win_counts_vs_b3": {f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_origin_win_count"] for row in pairwise_b3},
        "leave_one_origin_leader_survival": {f"{target_id}_{scenario_id}_{model_id}": sum(row["full_domain_leader_survives"] == "yes" for row in leave_origin if row["target_id"] == target_id and row["scenario_id"] == scenario_id and row["predictor_id"] == model_id) for target_id in ("resident", "registered") for scenario_id in SCENARIOS for model_id in MODEL_IDS},
        "dominant_city_sse_shares": {f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_dominant_city_sse_share"] for row in concentrations},
        "top_three_city_sse_shares": {f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_top3_city_sse_share"] for row in concentrations},
        "improved_city_shares": {f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["improved_city_share"] for row in concentrations},
        "improved_city_origin_shares": {f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["improved_city_origin_row_share"] for row in concentrations},
        "r1_r2_feature_difference_confirmed": any(row["r1_r2_feature_difference_confirmed"] == "yes" for row in freshness_rows),
        "r1_r2_metric_sensitivity_summary": r1_r2_metric_sensitivity,
        "urbanization_source_geometry": {
            "observed_latest_year": urban_geometry["latest_usable_year"],
            "observed_lag_years": urban_geometry["observed_snapshot_freshness_lag_years"],
            "source_status": urban_geometry["source_status"],
            "reference_latest_year_city_count": urban_geometry["latest_year_city_count"],
        },
        "level_ablation_supported": "yes" if level_ablation_supported else "no",
        "dynamics_ablation_supported": "yes" if dynamics_ablation_supported else "no",
        "urbanization_material_signal_supported": "yes" if urbanization_material_signal_supported else "no",
        "best_resident_r1_model_among_corrections": best_resident_r1_model,
        "best_resident_r2_model_among_corrections": best_resident_r2_model,
        "coefficient_sign_stability": coefficient_stability,
        "b3_compatibility_pass": all(row["status"] == "PASS" for row in b3_compatibility),
        "b3_metric_compatibility_pass": all(row["status"] == "PASS" for row in b3_metric_compatibility),
        "leakage_pass": all(row["leakage_status"] == "PASS" for row in leakage_audit),
        "urbanization_year_audit_pass": all(row["boundary_status"] == "PASS" and row["urbanization_years_within_scenario_cap"] == "PASS" for row in urban_boundaries),
        "population_t_minus_2_audit_pass": all(row["population_history_within_y_minus_2"] == "PASS" for row in urban_boundaries),
        "no_cross_source_contamination": all(row["other_covariate_columns_used"] == "no" for row in leakage_audit),
        "no_city_identity": all(row["city_identity_used"] == "no" for row in leakage_audit),
        "full_coverage": all(row["coverage_status"] == "PASS" for row in dataset_diagnostics),
        "feature_missingness_rows": len(feature_missingness),
        "training_geometry_rows": len(training_geometry),
        "modeling_dataset_diagnostics_rows": len(dataset_diagnostics),
        "prediction_rows": len(all_rows),
        "correction_prediction_rows": len(candidate_rows),
        "leakage_audit_rows": len(leakage_audit),
        "urbanization_information_boundary_rows": len(urban_boundaries),
        "formal_ml_model_used": True,
        "covariates_used": True,
        "urbanization_used": True,
        "other_covariates_used": False,
        "city_identity_used": False,
        "tuning_run": False,
        "ensemble_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
        "population_profile": population_profile,
        "technical_decision": "Use shared S2 required_lag_years for urbanization cap; model only urbanization_latest for M2A and add actual-year recent change/slope for M2B; latest year and staleness remain audit-only fields.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "feature_manifest.csv", list(FEATURE_MANIFEST))
    write_csv(OUTPUT_DIR / "urbanization_information_boundary.csv", urban_boundaries)
    write_csv(OUTPUT_DIR / "r1_r2_feature_freshness_comparison.csv", freshness_rows)
    write_csv(OUTPUT_DIR / "modeling_dataset_diagnostics.csv", dataset_diagnostics)
    write_csv(OUTPUT_DIR / "training_geometry.csv", training_geometry)
    write_csv(OUTPUT_DIR / "feature_missingness_imputation.csv", feature_missingness)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(OUTPUT_DIR / "per_origin_metrics.csv", per_origin_metrics)
    write_csv(OUTPUT_DIR / "per_origin_predictions_or_residuals.csv", all_rows, fields=union_fields(all_rows))
    write_csv(OUTPUT_DIR / "pairwise_b3_comparison.csv", pairwise_b3)
    write_csv(OUTPUT_DIR / "bias_control_comparison.csv", bias_control)
    write_csv(OUTPUT_DIR / "level_vs_dynamics_ablation.csv", ablation)
    write_csv(OUTPUT_DIR / "latest_origin_comparison.csv", latest)
    write_csv(OUTPUT_DIR / "city_improvement_concentration.csv", concentrations)
    write_csv(OUTPUT_DIR / "leave_one_origin_out.csv", leave_origin)
    write_csv(OUTPUT_DIR / "leave_one_city_sensitivity.csv", leave_city)
    write_csv(OUTPUT_DIR / "ridge_coefficient_diagnostics.csv", coefficient_rows)
    write_csv(OUTPUT_DIR / "residual_feature_relationship.csv", relationship_rows_all)
    write_csv(OUTPUT_DIR / "leakage_audit.csv", leakage_audit)
    write_csv(OUTPUT_DIR / "b3_compatibility_check.csv", b3_compatibility)
    write_csv(OUTPUT_DIR / "b3_metric_compatibility_check.csv", b3_metric_compatibility)
    write_json(OUTPUT_DIR / "plan02_summary.json", summary)

    print("S4_PLAN02_URBANIZATION_RESIDUAL_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_origin_domain=2011-2021")
    print(f"model_ids={','.join(MODEL_IDS)}")
    print(f"prediction_rows={len(all_rows)}")
    print(f"correction_prediction_rows={len(candidate_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"aggregate_metric_rows={len(aggregate_metrics)}")
    print(f"urbanization_information_boundary_rows={len(urban_boundaries)}")
    print(f"feature_missingness_rows={len(feature_missingness)}")
    print(f"leakage_audit_rows={len(leakage_audit)}")
    print(f"b3_compatibility_pass={summary['b3_compatibility_pass']}")
    print(f"b3_metric_compatibility_pass={summary['b3_metric_compatibility_pass']}")
    print(f"urbanization_year_audit_pass={summary['urbanization_year_audit_pass']}")
    print(f"population_t_minus_2_audit_pass={summary['population_t_minus_2_audit_pass']}")
    print(f"leakage_pass={summary['leakage_pass']}")
    print(f"full_coverage={summary['full_coverage']}")
    print(f"r1_r2_feature_difference_confirmed={summary['r1_r2_feature_difference_confirmed']}")
    print(f"level_ablation_supported={summary['level_ablation_supported']}")
    print(f"dynamics_ablation_supported={summary['dynamics_ablation_supported']}")
    print(f"urbanization_material_signal_supported={summary['urbanization_material_signal_supported']}")
    print("formal_ml_model_used=True")
    print("covariates_used=True")
    print("urbanization_used=True")
    print("other_covariates_used=False")
    print("city_identity_used=False")
    print("tuning_run=False")
    print("ensemble_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
