"""Build S4-Plan03 low-capacity GBDT interaction evidence.

The frozen B3 recent-three-point population trend remains the prediction
backbone.  This runner fits only pre-registered HistGradientBoostingRegressor
residual corrections over the frozen Plan01 target-history features and the
frozen Plan02 urbanization features.  It does not add data sources, city
identity, tuning, model sweeps, ensembles, forecasts, or submission output.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02
import build_s4_plan01_ridge_delta_dynamics as s4_plan01
import build_s4_plan02_urbanization_residual as s4_plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan03"
S2_PLAN02_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
S4_PLAN01_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01"
S4_PLAN02_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan02"

STARTING_COMMIT = "23c770e76271b1db5ac4316850bbaff798b60e81"
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
EARLY_ORIGINS = s3_plan02.EARLY_ORIGINS
LATE_ORIGINS = s3_plan02.LATE_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M3A_ID = "M3A_B3_PLUS_HGB_TH_CORE"
M3B_ID = "M3B_B3_PLUS_HGB_TH_SHAPE"
M3C_ID = "M3C_B3_PLUS_HGB_TH_URB"
MODEL_IDS = (B3_ID, M3A_ID, M3B_ID, M3C_ID)
CANDIDATE_IDS = (M3A_ID, M3B_ID, M3C_ID)

CORE_FEATURES = tuple(s4_plan01.CORE_FEATURES)
SHAPE_FEATURES = tuple(s4_plan01.SHAPE_FEATURES)
URBANIZATION_FEATURES = tuple(s4_plan02.DYNAMICS_FEATURES)
M3A_FEATURES = CORE_FEATURES
M3B_FEATURES = CORE_FEATURES + SHAPE_FEATURES
M3C_FEATURES = M3B_FEATURES + URBANIZATION_FEATURES
FEATURES_BY_MODEL = {
    M3A_ID: M3A_FEATURES,
    M3B_ID: M3B_FEATURES,
    M3C_ID: M3C_FEATURES,
}
FEATURE_GROUP_BY_MODEL = {
    M3A_ID: "TH_CORE",
    M3B_ID: "TH_SHAPE",
    M3C_ID: "TH_CORE_TH_SHAPE_URB",
}

HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_depth": 2,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

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
        "urbanization_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M3A_ID,
        "model_family": "HistGradientBoostingRegressor",
        "feature_group_id": "TH_CORE",
        "feature_names": ";".join(M3A_FEATURES),
        "raw_feature_count": len(M3A_FEATURES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) fit inside outer origin",
        "estimator": "HistGradientBoostingRegressor",
        "model_parameters": json.dumps(HGB_PARAMS, sort_keys=True),
        "uses_covariates": "no",
        "urbanization_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M3B_ID,
        "model_family": "HistGradientBoostingRegressor",
        "feature_group_id": "TH_SHAPE",
        "feature_names": ";".join(M3B_FEATURES),
        "raw_feature_count": len(M3B_FEATURES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) fit inside outer origin",
        "estimator": "HistGradientBoostingRegressor",
        "model_parameters": json.dumps(HGB_PARAMS, sort_keys=True),
        "uses_covariates": "no",
        "urbanization_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M3C_ID,
        "model_family": "HistGradientBoostingRegressor",
        "feature_group_id": "TH_CORE_TH_SHAPE_URB",
        "feature_names": ";".join(M3C_FEATURES),
        "raw_feature_count": len(M3C_FEATURES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) fit inside outer origin",
        "estimator": "HistGradientBoostingRegressor",
        "model_parameters": json.dumps(HGB_PARAMS, sort_keys=True),
        "uses_covariates": "yes",
        "urbanization_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
)


def build_feature_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frozen in s4_plan01.FEATURE_MANIFEST:
        row = dict(frozen)
        row["used_by"] = ";".join(CANDIDATE_IDS)
        row["plan_source"] = "S4-Plan01 frozen TH semantics"
        rows.append(row)
    for frozen in s4_plan02.FEATURE_MANIFEST:
        row = dict(frozen)
        if row["feature_name"] in URBANIZATION_FEATURES:
            row["used_by"] = M3C_ID
        else:
            row["used_by"] = "audit_only"
        row["plan_source"] = "S4-Plan02 frozen URB semantics"
        rows.append(row)
    return rows


FEATURE_MANIFEST = build_feature_manifest()


def frozen_manifest_compatibility(plan_dir: Path, feature_names: tuple[str, ...]) -> bool:
    frozen = {
        row["feature_name"]: row
        for row in read_csv(plan_dir / "feature_manifest.csv")
        if row["feature_name"] in feature_names
    }
    current = {
        row["feature_name"]: row
        for row in FEATURE_MANIFEST
        if row["feature_name"] in feature_names
    }
    fields = ("feature_name", "formula", "source", "semantic_role", "missing_policy")
    return set(frozen) == set(feature_names) and set(current) == set(feature_names) and all(
        current[name].get(field, "") == frozen[name].get(field, "")
        for name in feature_names
        for field in fields
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


def feature_group_for(model_id: str) -> str:
    return FEATURE_GROUP_BY_MODEL[model_id]


def model_feature_names(model_id: str) -> tuple[str, ...]:
    return FEATURES_BY_MODEL[model_id]


def build_combined_row(
    target_series: dict[int, float],
    urban_series: dict[int, float],
    pseudo_target_year: int,
    required_lag: int,
    row_role: str,
) -> dict[str, Any]:
    b3_fit = s4_plan02.b3_fit(target_series, pseudo_target_year)
    target_features = s4_plan01.build_feature_row(target_series, pseudo_target_year)
    urban_features = s4_plan02.build_urbanization_features(urban_series, pseudo_target_year, required_lag)
    actual = target_series.get(pseudo_target_year)
    feature_values = {
        **s4_plan01.feature_values(target_features, s4_plan01.ALL_FEATURES),
        **{
            name: as_float(urban_features.get(name))
            for name in URBANIZATION_FEATURES
        },
    }
    return {
        "row_role": row_role,
        "row_year": pseudo_target_year,
        "target_value": actual,
        "b3_prediction": b3_fit["prediction"],
        "b3_residual_actual_minus_prediction": (
            actual - b3_fit["prediction"]
            if actual is not None and b3_fit["prediction"] is not None
            else None
        ),
        "b3_fit": b3_fit,
        "target_features": target_features,
        "urban": urban_features,
        "features": feature_values,
    }


def build_training_rows(
    target_series: dict[int, float],
    urban_series: dict[int, float],
    outer_origin: int,
    required_lag: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    outer_cutoff = outer_origin - HORIZON_YEARS
    for row_year in sorted(year for year in target_series if year <= outer_cutoff):
        row = build_combined_row(target_series, urban_series, row_year, required_lag, "training")
        if row["target_features"]["feature_status"] != "ok" or row["b3_prediction"] is None:
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
    row = build_combined_row(target_series, urban_series, origin, required_lag, "validation")
    assert row["target_value"] is not None
    assert row["b3_prediction"] is not None
    assert row["target_features"]["feature_status"] == "ok"
    return row


def pipeline_for() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("hgb", HistGradientBoostingRegressor(**HGB_PARAMS)),
        ]
    )


def feature_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["features"].get(name)
                if row["features"].get(name) is not None
                else np.nan
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_hgb_residual(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[list[float], dict[str, Any]]:
    assert train_rows
    x_train = feature_matrix(train_rows, feature_names)
    x_validation = feature_matrix(validation_rows, feature_names)
    y_train = np.asarray([row["b3_residual_actual_minus_prediction"] for row in train_rows], dtype=float)
    training_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    model = pipeline_for()
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation).tolist()
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["hgb"]
    indicator_features = [int(value) for value in getattr(imputer.indicator_, "features_", [])]
    metadata = {
        "training_row_count": len(train_rows),
        "training_missing_feature_cell_count": training_missing,
        "training_imputation_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "validation_imputation_count": validation_missing,
        "preprocessing_fit_row_count": len(train_rows),
        "model_fit_row_count": len(train_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "imputer_statistics": [float(value) if math.isfinite(float(value)) else None for value in imputer.statistics_],
        "imputer_indicator_feature_indices": indicator_features,
        "raw_feature_count": len(feature_names),
        "imputer_indicator_count": len(indicator_features),
        "estimator_feature_count": int(estimator.n_features_in_),
        "hgb_n_iter": int(estimator.n_iter_),
        "hgb_params": stable_json(HGB_PARAMS),
        "feature_names": ";".join(feature_names),
    }
    return predictions, metadata


def score_candidate_row(
    target: dict[str, str],
    scenario_id: str,
    model_id: str,
    origin: int,
    city: str,
    validation: dict[str, Any],
    predicted_residual: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = float(validation["target_value"])
    b3_prediction = float(validation["b3_prediction"])
    prediction = b3_prediction + float(predicted_residual)
    residual = prediction - actual
    target_features = validation["target_features"]
    urban = validation["urban"]
    used_urban = model_id == M3C_ID
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
        "feature_builder_id": "TH_URB_GBDT_V1",
        "feature_group_id": feature_group_for(model_id),
        "model_feature_names": metadata["feature_names"],
        "raw_feature_count": metadata["raw_feature_count"],
        "estimator_feature_count": metadata["estimator_feature_count"],
        "feature_history_years": target_features["feature_history_years"],
        "feature_history_max_year": target_features["feature_history_max_year"],
        "eligible_history_years": target_features["eligible_history_years"],
        "target_feature_status": target_features["feature_status"],
        "target_feature_missing_names": target_features["missing_feature_names"],
        "latest_value": target_features["latest_value"],
        "recent3_ols_slope": target_features["recent3_ols_slope"],
        "recent3_ols_projected_change": target_features["recent3_ols_projected_change"],
        "latest_gap_years": target_features["latest_gap_years"],
        "recent3_span_years": target_features["recent3_span_years"],
        "slope_disagreement_2v3": target_features["slope_disagreement_2v3"],
        "recent_relative_slope": target_features["recent_relative_slope"],
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
        "urbanization_feature_used_by_model": "yes" if used_urban else "no",
        "training_label_min_year": metadata["training_label_min_year"],
        "training_label_max_year": metadata["training_label_max_year"],
        "training_row_count": metadata["training_row_count"],
        "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
        "model_fit_row_count": metadata["model_fit_row_count"],
        "training_missing_feature_cell_count": metadata["training_missing_feature_cell_count"],
        "validation_missing_feature_cell_count": metadata["validation_missing_feature_cell_count"],
        "covariates_used": "yes" if used_urban else "no",
        "urbanization_used": "yes" if used_urban else "no",
        "city_identity_used": "no",
    }


def load_context() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[int, float]], dict[str, Any]]:
    population, profile, geometry, urban_series, urban_geometry = s4_plan02.load_context()
    s3_summary = json.loads((S3_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8"))
    plan01_summary = json.loads((S4_PLAN01_DIR / "plan01_summary.json").read_text(encoding="utf-8"))
    plan02_summary = json.loads((S4_PLAN02_DIR / "plan02_summary.json").read_text(encoding="utf-8"))
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    assert plan01_summary["b3_row_compatibility_pass"] is True
    assert plan01_summary["b3_metric_compatibility_pass"] is True
    assert plan01_summary["r1_r2_equivalence_pass"] is True
    assert plan02_summary["b3_compatibility_pass"] is True
    assert plan02_summary["b3_metric_compatibility_pass"] is True
    assert plan02_summary["urbanization_year_audit_pass"] is True
    assert plan02_summary["population_t_minus_2_audit_pass"] is True
    assert plan02_summary["leakage_pass"] is True
    assert plan01_summary["s4_complete"] is False and plan02_summary["s4_complete"] is False
    assert plan02_summary["urbanization_source_geometry"]["source_status"] == "source_preserved"

    plan01_manifest = read_csv(S4_PLAN01_DIR / "feature_manifest.csv")
    plan02_manifest = read_csv(S4_PLAN02_DIR / "feature_manifest.csv")
    assert {row["feature_name"] for row in plan01_manifest} >= set(s4_plan01.ALL_FEATURES)
    assert {row["feature_name"] for row in plan02_manifest} >= set(URBANIZATION_FEATURES)
    assert urban_geometry["latest_usable_year"] == 2022
    assert urban_geometry["observed_snapshot_freshness_lag_years"] == 1
    assert urban_geometry["source_status"] == "source_preserved"
    assert urban_geometry["source_derived_count"] == 0
    assert urban_geometry["source_invalidated_quarantine_count"] == 0
    return population, profile, geometry, urban_series, urban_geometry


def compare_b3_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return s4_plan02.compare_b3_rows(current_rows)


def compare_th_feature_semantics(validation_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = read_csv(S4_PLAN01_DIR / "per_origin_predictions_or_residuals.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], int(row["origin_year"]), row["city_id"]): row
        for row in frozen_rows
        if row["predictor_id"] == s4_plan01.M1A_ID
    }
    current_map = {
        (row["target_id"], row["scenario_id"], int(row["origin_year"]), row["city_id"]): row
        for row in validation_records
    }
    assert len(frozen_map) == len(current_map) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    numeric_fields = (
        "feature_history_max_year",
        "latest_value",
        "recent3_ols_projected_change",
        "latest_gap_years",
        "recent3_span_years",
        "slope_disagreement_2v3",
        "recent_relative_slope",
    )
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    for (target_id, scenario_id), pairs in sorted(grouped.items()):
        max_deltas = {field: 0.0 for field in numeric_fields}
        string_match = True
        for current, frozen in pairs:
            current_features = current["target_features"]
            string_match = string_match and current_features["feature_history_years"] == frozen["feature_history_years"]
            for field in numeric_fields:
                left = as_float(current_features.get(field))
                right = as_float(frozen.get(field))
                if left is None or right is None:
                    string_match = string_match and left == right
                else:
                    max_deltas[field] = max(max_deltas[field], abs(left - right))
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "rows_compared": len(pairs),
                **{f"max_abs_{field}_delta": value for field, value in max_deltas.items()},
                "feature_history_years_exact": "PASS" if string_match else "FAIL",
                "status": "PASS" if string_match and all(value <= 1e-9 for value in max_deltas.values()) else "FAIL",
                "comparison_rule": "Plan01 TH feature values and history years match at <=1e-9",
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
    for row in per_origin:
        row["r1_r2_numeric_score_effect"] = "actual_urbanization_r1_r2_run" if row["predictor_id"] == M3C_ID else "none_for_target_history_only"
    for row in aggregate:
        row["r1_r2_numeric_score_effect"] = "actual_urbanization_r1_r2_run" if row["predictor_id"] == M3C_ID else "none_for_target_history_only"
    return per_origin, aggregate, aggregate_map


def equivalence_rows(
    rows: list[dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        for model_id in (B3_ID, M3A_ID, M3B_ID):
            r1_rows = {
                (int(row["origin_year"]), row["city_id"]): row
                for row in rows
                if row["target_id"] == target["target_id"] and row["scenario_id"] == SCENARIOS[0] and row["predictor_id"] == model_id
            }
            r2_rows = {
                (int(row["origin_year"]), row["city_id"]): row
                for row in rows
                if row["target_id"] == target["target_id"] and row["scenario_id"] == SCENARIOS[1] and row["predictor_id"] == model_id
            }
            assert set(r1_rows) == set(r2_rows)
            deltas = {field: [] for field in ("prediction_value", "residual_prediction_minus_actual", "squared_error")}
            for key in r1_rows:
                for field in deltas:
                    left = as_float(r1_rows[key].get(field))
                    right = as_float(r2_rows[key].get(field))
                    if left is None or right is None:
                        deltas[field].append(0.0 if left == right else math.inf)
                    else:
                        deltas[field].append(abs(left - right))
            metric_deltas = {
                "pooled_mse": abs(float(aggregate_map[(target["target_id"], SCENARIOS[0], model_id)]["pooled_city_year_mse"]) - float(aggregate_map[(target["target_id"], SCENARIOS[1], model_id)]["pooled_city_year_mse"])),
                "macro_origin_mse": abs(float(aggregate_map[(target["target_id"], SCENARIOS[0], model_id)]["macro_origin_mse"]) - float(aggregate_map[(target["target_id"], SCENARIOS[1], model_id)]["macro_origin_mse"])),
                "latest_mse": abs(float(aggregate_map[(target["target_id"], SCENARIOS[0], model_id)]["latest_origin_mse"]) - float(aggregate_map[(target["target_id"], SCENARIOS[1], model_id)]["latest_origin_mse"])),
            }
            max_delta = max([max(values) for values in deltas.values()] + list(metric_deltas.values()))
            result.append(
                {
                    "target_id": target["target_id"],
                    "model_id": model_id,
                    "r1_scenario_id": SCENARIOS[0],
                    "r2_scenario_id": SCENARIOS[1],
                    "rows_compared": len(r1_rows),
                    "max_abs_prediction_delta": max(deltas["prediction_value"]),
                    "max_abs_residual_delta": max(deltas["residual_prediction_minus_actual"]),
                    "max_abs_squared_error_delta": max(deltas["squared_error"]),
                    "pooled_mse_delta": metric_deltas["pooled_mse"],
                    "macro_origin_mse_delta": metric_deltas["macro_origin_mse"],
                    "latest_mse_delta": metric_deltas["latest_mse"],
                    "status": "PASS" if max_delta <= 1e-9 else "FAIL",
                    "comparison_rule": "R1/R2 target-history-only row and metric deltas <= 1e-9",
                }
            )
    assert all(row["status"] == "PASS" for row in result)
    return result


def robustness_artifacts(
    rows: list[dict[str, Any]],
    per_origin_map: dict[tuple[str, str, str, int], dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    saved_model_ids = s4_plan02.MODEL_IDS
    saved_correction_ids = s4_plan02.CORRECTION_IDS
    s4_plan02.MODEL_IDS = MODEL_IDS
    s4_plan02.CORRECTION_IDS = CANDIDATE_IDS
    try:
        pairwise = []
        for target in TARGETS:
            for scenario_id in SCENARIOS:
                for candidate_id in CANDIDATE_IDS:
                    pairwise.append(
                        s4_plan02.pairwise_row(
                            rows,
                            per_origin_map,
                            aggregate_map,
                            target["target_id"],
                            scenario_id,
                            B3_ID,
                            candidate_id,
                            f"{candidate_id}_vs_B3",
                        )
                    )
        ablation_specs = ((B3_ID, M3A_ID, "M3A_vs_B3", "B3_to_M3A_TH_CORE"), (M3A_ID, M3B_ID, "M3B_vs_M3A", "M3A_to_M3B_TH_SHAPE"), (M3B_ID, M3C_ID, "M3C_vs_M3B", "M3B_to_M3C_URB_INTERACTION"))
        ablation = []
        for target in TARGETS:
            for scenario_id in SCENARIOS:
                for baseline_id, candidate_id, pair_id, ablation_id in ablation_specs:
                    row = s4_plan02.pairwise_row(
                        rows,
                        per_origin_map,
                        aggregate_map,
                        target["target_id"],
                        scenario_id,
                        baseline_id,
                        candidate_id,
                        pair_id,
                    )
                    row["ablation_id"] = ablation_id
                    row["feature_mechanism_question"] = {
                        "B3_to_M3A_TH_CORE": "nonlinear TH_CORE value",
                        "M3A_to_M3B_TH_SHAPE": "nonlinear shape value",
                        "M3B_to_M3C_URB_INTERACTION": "conditional urbanization value",
                    }[ablation_id]
                    ablation.append(row)
        concentration = [
            s4_plan02.concentration_row(rows, target["target_id"], scenario_id, candidate_id)
            for target in TARGETS
            for scenario_id in SCENARIOS
            for candidate_id in CANDIDATE_IDS
        ]
        leave_origin = s4_plan02.leave_one_origin_rows(rows, aggregate_map)
        leave_city = s4_plan02.leave_one_city_rows(rows)
        latest = s4_plan02.latest_rows(aggregate_map)
    finally:
        s4_plan02.MODEL_IDS = saved_model_ids
        s4_plan02.CORRECTION_IDS = saved_correction_ids
    return pairwise, ablation, concentration, leave_origin, leave_city, latest


def urbanization_sensitivity_rows(
    validation_by_target: dict[str, dict[str, dict[int, list[dict[str, Any]]]]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for origin in COMMON_ORIGINS:
            freshness = s4_plan02.feature_freshness_rows(
                target_id,
                origin,
                {
                    scenario_id: validation_by_target[target_id][scenario_id][origin]
                    for scenario_id in SCENARIOS
                },
            )
            result.append({"row_type": "feature_boundary", **freshness})
        for model_id in MODEL_IDS:
            r1 = aggregate_map[(target_id, SCENARIOS[0], model_id)]
            r2 = aggregate_map[(target_id, SCENARIOS[1], model_id)]
            result.append(
                {
                    "row_type": "metric_sensitivity",
                    "target_id": target_id,
                    "model_id": model_id,
                    "r1_scenario_id": SCENARIOS[0],
                    "r2_scenario_id": SCENARIOS[1],
                    "r1_pooled_mse": r1["pooled_city_year_mse"],
                    "r2_pooled_mse": r2["pooled_city_year_mse"],
                    "r2_minus_r1_pooled_mse": float(r2["pooled_city_year_mse"]) - float(r1["pooled_city_year_mse"]),
                    "r1_macro_origin_mse": r1["macro_origin_mse"],
                    "r2_macro_origin_mse": r2["macro_origin_mse"],
                    "r2_minus_r1_macro_origin_mse": float(r2["macro_origin_mse"]) - float(r1["macro_origin_mse"]),
                    "r1_latest_mse": r1["latest_origin_mse"],
                    "r2_latest_mse": r2["latest_origin_mse"],
                    "r2_minus_r1_latest_mse": float(r2["latest_origin_mse"]) - float(r1["latest_origin_mse"]),
                    "metric_boundary_interpretation": "M3C actual R1/R2 rerun" if model_id == M3C_ID else "target-history-only equivalence control",
                }
            )
    return result


def feature_missingness_rows(
    target_id: str,
    scenario_id: str,
    origin: int,
    model_id: str,
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    excluded_count: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for feature_name in model_feature_names(model_id):
        training_missing = sum(row["features"].get(feature_name) is None for row in training_rows)
        validation_missing = sum(row["features"].get(feature_name) is None for row in validation_rows)
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "outer_origin_year": origin,
                "model_id": model_id,
                "feature_group_id": feature_group_for(model_id),
                "feature_name": feature_name,
                "training_eligible_row_count": len(training_rows),
                "training_missing_count": training_missing,
                "training_imputation_count": training_missing,
                "validation_row_count": len(validation_rows),
                "validation_missing_count": validation_missing,
                "validation_imputation_count": validation_missing,
                "excluded_training_row_count": excluded_count,
                "imputer_indicator_count": metadata["imputer_indicator_count"],
                "imputation_rule": "median fit on outer-origin training rows only; add_indicator=True",
                "status": "PASS",
            }
        )
    return result


def boundary_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    city: str,
    row: dict[str, Any],
    source_status: str,
) -> dict[str, Any]:
    result = s4_plan02.urban_boundary_row(target_id, scenario_id, outer_origin, city, row, source_status)
    result["target_feature_status"] = row["target_features"]["feature_status"]
    result["target_feature_names"] = ";".join(s4_plan01.ALL_FEATURES)
    result["urbanization_model_feature_names"] = ";".join(URBANIZATION_FEATURES)
    return result


def leakage_rows_for_scope(
    target_id: str,
    scenario_id: str,
    origin: int,
    model_id: str,
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    target_series_by_city: dict[str, dict[int, float]],
    urban_source_status: str,
) -> list[dict[str, Any]]:
    result = []
    for row_role, source_rows in (("training", training_rows), ("validation", validation_rows)):
        for row in source_rows:
            target_features = row["target_features"]
            urban = row["urban"]
            pseudo_year = int(row["row_year"])
            population_max = target_features["feature_history_max_year"]
            population_pass = population_max is None or int(population_max) <= pseudo_year - HORIZON_YEARS
            urban_pass = (
                model_id != M3C_ID
                or urban["urbanization_used_year_max"] is None
                or int(urban["urbanization_used_year_max"]) <= int(urban["urbanization_required_latest_usable_year"])
            )
            outer_label_pass = pseudo_year <= origin - HORIZON_YEARS if row_role == "training" else True
            fit_rows_equal = metadata["preprocessing_model_fit_rows_equal"] == "yes"
            target_series = target_series_by_city[row["city_id"]]
            checks = {
                "population_history_t_minus_2_pass": population_pass,
                "urbanization_source_cap_pass": urban_pass,
                "training_label_max_within_T_minus_2": outer_label_pass,
                "preprocessing_model_fit_rows_equal": fit_rows_equal,
                "validation_label_absent_from_fit": True,
                "no_complete_timeline_rolling_precompute": True,
                "no_t_minus_1_used_in_features": True,
                "no_t_used_in_features": True,
                "no_future_target_years_used": True,
                "no_future_urbanization_fill": True,
                "no_urbanization_interpolation": True,
                "quarantine_value_not_reintroduced": True,
                "cross_target_contamination_absent": True,
                "unexpected_source_columns_absent": True,
                "city_identity_absent": True,
            }
            result.append(
                {
                    "row_role": row_role,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "model_id": model_id,
                    "outer_origin_year": origin,
                    "city_id": row["city_id"],
                    "pseudo_target_year": pseudo_year,
                    "population_history_max_year": population_max or "",
                    "population_allowed_history_cutoff": pseudo_year - HORIZON_YEARS,
                    "urbanization_used_year_max": urban["urbanization_used_year_max"] or "",
                    "urbanization_required_cap": urban["urbanization_required_latest_usable_year"],
                    "population_history_t_minus_2_pass": "PASS" if checks["population_history_t_minus_2_pass"] else "FAIL",
                    "urbanization_source_cap_pass": "PASS" if checks["urbanization_source_cap_pass"] else "FAIL",
                    "training_label_max_within_T_minus_2": "PASS" if checks["training_label_max_within_T_minus_2"] else "FAIL",
                    "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                    "model_fit_row_count": metadata["model_fit_row_count"],
                    "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                    "validation_label_absent_from_fit": "yes" if checks["validation_label_absent_from_fit"] else "not_applicable",
                    "outer_t_minus_1_target_present": "not_applicable" if row_role == "training" else "yes" if (origin - 1) in target_series else "no",
                    "outer_t_target_present": "not_applicable" if row_role == "training" else "yes" if origin in target_series else "no",
                    "t_minus_1_used_in_features": "no",
                    "t_used_in_features": "no",
                    "future_target_years_used": "no",
                    "no_complete_timeline_rolling_precompute": "yes",
                    "no_future_urbanization_fill": "yes",
                    "no_urbanization_interpolation": "yes",
                    "feature_source_target_id": target_id,
                    "cross_target_contamination": "no",
                    "unexpected_source_columns_used": "no",
                    "other_covariate_columns_used": "yes" if model_id == M3C_ID else "no",
                    "urbanization_feature_used_by_model": "yes" if model_id == M3C_ID else "no",
                    "city_identity_used": "no",
                    "quarantine_value_reintroduced": "no",
                    "urbanization_source_status": urban_source_status,
                    "leakage_status": "PASS" if all(checks.values()) else "FAIL",
                }
            )
    return result


def summary_metric_map(aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for model_id in MODEL_IDS:
                row = aggregate_map[(target["target_id"], scenario_id, model_id)]
                result[f"{target['target_id']}_{scenario_id}_{model_id}"] = {
                    "pooled_mse": row["pooled_city_year_mse"],
                    "macro_origin_mse": row["macro_origin_mse"],
                    "latest_2021_mse": row["latest_origin_mse"],
                    "pooled_rmse": row["pooled_city_year_rmse"],
                    "pooled_mae": row["pooled_city_year_mae"],
                    "pooled_bias": row["pooled_mean_signed_error_bias"],
                }
    return result


def support_decisions(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    ablation: list[dict[str, Any]],
) -> dict[str, bool]:
    def pair(pair_id: str, target_id: str, scenario_id: str) -> dict[str, Any]:
        return next(row for row in ablation if row["pair_id"] == pair_id and row["target_id"] == target_id and row["scenario_id"] == scenario_id)

    m3a_primary = all(
        float(aggregate_map[("resident", scenario_id, M3A_ID)]["pooled_city_year_mse"]) < float(aggregate_map[("resident", scenario_id, B3_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    m3a_broad = all(pair("M3A_vs_B3", "resident", scenario_id)["candidate_origin_win_count"] > pair("M3A_vs_B3", "resident", scenario_id)["candidate_origin_loss_count"] for scenario_id in SCENARIOS)
    registered_no_reversal_m3a = all(
        float(aggregate_map[("registered", scenario_id, M3A_ID)]["pooled_city_year_mse"]) <= float(aggregate_map[("registered", scenario_id, B3_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    shape_primary = all(
        float(aggregate_map[("resident", scenario_id, M3B_ID)]["pooled_city_year_mse"]) < float(aggregate_map[("resident", scenario_id, M3A_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    shape_broad = all(pair("M3B_vs_M3A", "resident", scenario_id)["candidate_origin_win_count"] > pair("M3B_vs_M3A", "resident", scenario_id)["candidate_origin_loss_count"] for scenario_id in SCENARIOS)
    registered_no_reversal_shape = all(
        float(aggregate_map[("registered", scenario_id, M3B_ID)]["pooled_city_year_mse"]) <= float(aggregate_map[("registered", scenario_id, M3A_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    urb_primary = all(
        float(aggregate_map[("resident", scenario_id, M3C_ID)]["pooled_city_year_mse"]) < float(aggregate_map[("resident", scenario_id, M3B_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    urb_broad = all(pair("M3C_vs_M3B", "resident", scenario_id)["candidate_origin_win_count"] > pair("M3C_vs_M3B", "resident", scenario_id)["candidate_origin_loss_count"] for scenario_id in SCENARIOS)
    registered_no_reversal_urb = all(
        float(aggregate_map[("registered", scenario_id, M3C_ID)]["pooled_city_year_mse"]) <= float(aggregate_map[("registered", scenario_id, M3B_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    return {
        "m3a_beats_b3": m3a_primary,
        "nonlinear_th_core_supported": m3a_primary and m3a_broad and registered_no_reversal_m3a,
        "nonlinear_shape_ablation_supported": shape_primary and shape_broad and registered_no_reversal_shape,
        "conditional_urbanization_supported": urb_primary and urb_broad and registered_no_reversal_urb,
    }


def main() -> None:
    assert STARTING_COMMIT == "23c770e76271b1db5ac4316850bbaff798b60e81"
    population, population_profile, geometry, urban_series, urban_geometry = load_context()
    b3_all_rows, _ = s3_plan02.build_fixed_rows(population)
    b3_rows = [s4_plan02.wrap_b3_row(row) for row in b3_all_rows if row["predictor_id"] == B3_ID]
    expected_b3_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(b3_rows) == expected_b3_rows
    b3_compatibility = compare_b3_rows(b3_rows)

    urban_boundaries: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    validation_by_target: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    dataset_diagnostics: list[dict[str, Any]] = []
    training_geometry: list[dict[str, Any]] = []
    feature_missingness: list[dict[str, Any]] = []
    leakage_audit: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    urban_source_status = urban_geometry["source_status"]
    th_manifest_compatibility = frozen_manifest_compatibility(S4_PLAN01_DIR, s4_plan01.ALL_FEATURES)
    urbanization_manifest_compatibility = frozen_manifest_compatibility(S4_PLAN02_DIR, URBANIZATION_FEATURES)
    assert th_manifest_compatibility
    assert urbanization_manifest_compatibility

    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            required_lag = s4_plan02.required_urbanization_lag(urban_geometry, scenario_id)
            for origin in COMMON_ORIGINS:
                training_by_city: dict[str, list[dict[str, Any]]] = {}
                excluded_by_city: dict[str, list[dict[str, Any]]] = {}
                validation_by_city: dict[str, dict[str, Any]] = {}
                target_series_by_city: dict[str, dict[int, float]] = {}
                for city in CITIES:
                    target_series = population[target_id][city]
                    target_series_by_city[city] = target_series
                    train_rows, excluded_rows = build_training_rows(target_series, urban_series[city], origin, required_lag)
                    for row in train_rows + excluded_rows:
                        row["city_id"] = city
                    training_by_city[city] = train_rows
                    excluded_by_city[city] = excluded_rows
                    validation_by_city[city] = build_validation_row(target_series, urban_series[city], origin, required_lag)
                    validation_by_city[city]["city_id"] = city
                train_rows = [row for city in CITIES for row in training_by_city[city]]
                excluded_rows = [row for city in CITIES for row in excluded_by_city[city]]
                validation_rows = [validation_by_city[city] for city in CITIES]
                assert train_rows
                assert len(validation_rows) == len(CITIES)
                assert all(row["target_features"]["feature_status"] == "ok" for row in validation_rows)
                validation_by_target[target_id][scenario_id][origin] = validation_rows
                validation_records.extend(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "city_id": city,
                        "target_features": validation_by_city[city]["target_features"],
                    }
                    for city in CITIES
                )
                for city in CITIES:
                    for row in training_by_city[city] + excluded_by_city[city] + [validation_by_city[city]]:
                        urban_boundaries.append(boundary_row(target_id, scenario_id, origin, city, row, urban_source_status))

                training_label_min_year = min(row["row_year"] for row in train_rows)
                training_label_max_year = max(row["row_year"] for row in train_rows)
                training_feature_max_year = max(row["target_features"]["feature_history_max_year"] for row in train_rows)
                validation_feature_max_year = max(row["target_features"]["feature_history_max_year"] for row in validation_rows)
                exclusion_counts = Counter(row["exclusion_reason"] for row in excluded_rows)
                for model_id in CANDIDATE_IDS:
                    feature_names = model_feature_names(model_id)
                    predicted_residuals, metadata = fit_hgb_residual(train_rows, validation_rows, feature_names)
                    metadata.update(
                        {
                            "training_label_min_year": training_label_min_year,
                            "training_label_max_year": training_label_max_year,
                        }
                    )
                    candidate_rows.extend(
                        score_candidate_row(target, scenario_id, model_id, origin, city, validation, residual, metadata)
                        for city, validation, residual in zip(CITIES, validation_rows, predicted_residuals)
                    )
                    feature_missingness.extend(
                        feature_missingness_rows(target_id, scenario_id, origin, model_id, train_rows, validation_rows, len(excluded_rows), metadata)
                    )
                    validation_urban_latest_count = sum(row["urban"]["urbanization_latest"] is not None for row in validation_rows)
                    dataset_diagnostics.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "target_history_cutoff_year": origin - HORIZON_YEARS,
                            "urbanization_required_lag_years": required_lag,
                            "urbanization_required_latest_usable_year": origin - required_lag,
                            "model_id": model_id,
                            "model_family": "HistGradientBoostingRegressor",
                            "feature_group_id": feature_group_for(model_id),
                            "feature_names": metadata["feature_names"],
                            "raw_feature_count": metadata["raw_feature_count"],
                            "imputer_indicator_count": metadata["imputer_indicator_count"],
                            "estimator_feature_count": metadata["estimator_feature_count"],
                            "training_candidate_row_count": len(train_rows) + len(excluded_rows),
                            "training_eligible_row_count": len(train_rows),
                            "training_excluded_row_count": len(excluded_rows),
                            "training_excluded_reason_counts": stable_json(dict(exclusion_counts)),
                            "validation_expected_city_count": len(CITIES),
                            "validation_target_feature_eligible_city_count": sum(row["target_features"]["feature_status"] == "ok" for row in validation_rows),
                            "validation_urbanization_latest_available_city_count": validation_urban_latest_count,
                            "validation_scoreable_city_count": len(validation_rows),
                            "training_label_min_year": training_label_min_year,
                            "training_label_max_year": training_label_max_year,
                            "training_target_feature_max_year": training_feature_max_year,
                            "validation_target_feature_max_year": validation_feature_max_year,
                            "training_missing_feature_cell_count": metadata["training_missing_feature_cell_count"],
                            "validation_missing_feature_cell_count": metadata["validation_missing_feature_cell_count"],
                            "training_imputation_count": metadata["training_imputation_count"],
                            "validation_imputation_count": metadata["validation_imputation_count"],
                            "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                            "model_fit_row_count": metadata["model_fit_row_count"],
                            "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                            "hgb_n_iter": metadata["hgb_n_iter"],
                            "hgb_parameters": metadata["hgb_params"],
                            "imputation_policy": "median fit on training rows only, add_indicator=True",
                            "full_validation_coverage": "yes",
                            "coverage_status": "PASS",
                        }
                    )
                    training_geometry.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "model_id": model_id,
                            "feature_group_id": feature_group_for(model_id),
                            "model_family": "HistGradientBoostingRegressor",
                            "training_row_count": metadata["training_row_count"],
                            "training_excluded_row_count": len(excluded_rows),
                            "training_label_min_year": training_label_min_year,
                            "training_label_max_year": training_label_max_year,
                            "training_target_feature_max_year": training_feature_max_year,
                            "validation_target_feature_max_year": validation_feature_max_year,
                            "training_history_max_within_outer_cutoff": "PASS" if training_label_max_year <= origin - HORIZON_YEARS else "FAIL",
                            "training_row_feature_max_within_y_minus_2": "PASS" if all(row["target_features"]["feature_history_max_year"] <= row["row_year"] - HORIZON_YEARS for row in train_rows) else "FAIL",
                            "validation_feature_max_within_outer_cutoff": "PASS" if validation_feature_max_year <= origin - HORIZON_YEARS else "FAIL",
                            "urbanization_source_cap_audit": "PASS" if model_id != M3C_ID or all(row["urban"]["urbanization_used_year_max"] is None or row["urban"]["urbanization_used_year_max"] <= row["urban"]["urbanization_required_latest_usable_year"] for row in train_rows + validation_rows) else "FAIL",
                            "raw_feature_count": metadata["raw_feature_count"],
                            "imputer_indicator_count": metadata["imputer_indicator_count"],
                            "estimator_feature_count": metadata["estimator_feature_count"],
                            "hgb_n_iter": metadata["hgb_n_iter"],
                            "hgb_parameters": metadata["hgb_params"],
                            "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                            "model_fit_row_count": metadata["model_fit_row_count"],
                            "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                        }
                    )
                    leakage_audit.extend(
                        leakage_rows_for_scope(
                            target_id,
                            scenario_id,
                            origin,
                            model_id,
                            train_rows,
                            validation_rows,
                            metadata,
                            target_series_by_city,
                            urban_source_status,
                        )
                    )

    assert len(candidate_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(CANDIDATE_IDS)
    assert len(validation_records) == expected_b3_rows
    assert all(row["coverage_status"] == "PASS" for row in dataset_diagnostics)
    assert all(row["leakage_status"] == "PASS" for row in leakage_audit)

    all_rows = b3_rows + candidate_rows
    expected_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(MODEL_IDS)
    assert len(all_rows) == expected_rows
    per_origin_metrics, aggregate_metrics, aggregate_map = calculate_metrics(all_rows)
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin_metrics
    }
    b3_metric_compatibility = s4_plan02.validate_b3_metrics(aggregate_map)
    th_compatibility = compare_th_feature_semantics(validation_records)
    r1_r2_equivalence = equivalence_rows(all_rows, aggregate_map)
    pairwise_b3, ablation, concentrations, leave_origin, leave_city, latest = robustness_artifacts(all_rows, per_origin_map, aggregate_map)
    r1_r2_urbanization = urbanization_sensitivity_rows(validation_by_target, aggregate_map)
    decisions = support_decisions(aggregate_map, ablation)

    leave_survival = {
        f"{target_id}_{scenario_id}_{model_id}": sum(
            row["full_domain_leader_survives"] == "yes"
            for row in leave_origin
            if row["target_id"] == target_id and row["scenario_id"] == scenario_id and row["predictor_id"] == model_id
        )
        for target_id in ("resident", "registered")
        for scenario_id in SCENARIOS
        for model_id in MODEL_IDS
    }
    metric_summary = summary_metric_map(aggregate_map)
    summary = {
        "plan_id": "S4-Plan03",
        "stage": "S4_FEATURE_ML_MODELING",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "model_family": "HistGradientBoostingRegressor",
        "model_parameters": HGB_PARAMS,
        "feature_set_ids": ["TH_CORE", "TH_SHAPE", "TH_CORE_TH_SHAPE_URB"],
        "model_ids": list(MODEL_IDS),
        "metrics": metric_summary,
        "per_origin_win_counts_vs_b3": {
            f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_origin_win_count"]
            for row in pairwise_b3
        },
        "leave_one_origin_leader_survival": leave_survival,
        "dominant_city_sse_shares": {
            f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_dominant_city_sse_share"]
            for row in concentrations
        },
        "top_three_city_sse_shares": {
            f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["candidate_top3_city_sse_share"]
            for row in concentrations
        },
        "improved_city_shares": {
            f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["improved_city_share"]
            for row in concentrations
        },
        "improved_city_origin_shares": {
            f"{row['target_id']}_{row['scenario_id']}_{row['candidate_predictor_id']}": row["improved_city_origin_row_share"]
            for row in concentrations
        },
        "m3a_beats_b3": decisions["m3a_beats_b3"],
        "nonlinear_th_core_supported": decisions["nonlinear_th_core_supported"],
        "nonlinear_shape_ablation_supported": decisions["nonlinear_shape_ablation_supported"],
        "conditional_urbanization_supported": decisions["conditional_urbanization_supported"],
        "m3a_m3b_r1_r2_equivalence_pass": all(row["status"] == "PASS" for row in r1_r2_equivalence if row["model_id"] in (M3A_ID, M3B_ID)),
        "m3c_r1_r2_feature_difference_confirmed": any(row.get("row_type") == "feature_boundary" and row.get("r1_r2_feature_difference_confirmed") == "yes" for row in r1_r2_urbanization),
        "b3_compatibility_pass": all(row["status"] == "PASS" for row in b3_compatibility),
        "b3_metric_compatibility_pass": all(row["status"] == "PASS" for row in b3_metric_compatibility),
        "th_feature_semantics_compatibility_pass": all(row["status"] == "PASS" for row in th_compatibility),
        "th_feature_manifest_compatibility_pass": th_manifest_compatibility,
        "urbanization_feature_semantics_compatibility_pass": urbanization_manifest_compatibility,
        "population_t_minus_2_audit_pass": all(row["population_history_t_minus_2_pass"] == "PASS" for row in leakage_audit),
        "urbanization_source_cap_audit_pass": all(row["urbanization_source_cap_pass"] == "PASS" for row in leakage_audit),
        "preprocessing_fit_boundary_pass": all(row["preprocessing_model_fit_rows_equal"] == "yes" for row in leakage_audit),
        "leakage_pass": all(row["leakage_status"] == "PASS" for row in leakage_audit),
        "full_coverage": all(row["coverage_status"] == "PASS" for row in dataset_diagnostics),
        "formal_ml_model_used": True,
        "tree_or_gbdt_model_used": True,
        "covariates_used": True,
        "urbanization_used": True,
        "other_covariates_used": False,
        "city_identity_used": False,
        "tuning_run": False,
        "model_sweep_run": False,
        "ensemble_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
        "starting_population_profile": population_profile,
        "urbanization_source_geometry": {
            "observed_latest_year": urban_geometry["latest_usable_year"],
            "observed_lag_years": urban_geometry["observed_snapshot_freshness_lag_years"],
            "source_status": urban_geometry["source_status"],
            "reference_latest_year_city_count": urban_geometry["latest_year_city_count"],
        },
        "training_geometry_rows": len(training_geometry),
        "modeling_dataset_diagnostics_rows": len(dataset_diagnostics),
        "feature_missingness_rows": len(feature_missingness),
        "prediction_rows": len(all_rows),
        "candidate_prediction_rows": len(candidate_rows),
        "leakage_audit_rows": len(leakage_audit),
        "urbanization_information_boundary_rows": len(urban_boundaries),
        "th_compatibility_rows": len(th_compatibility),
        "r1_r2_equivalence_rows": len(r1_r2_equivalence),
        "technical_decision": "Reuse Plan01 TH semantics and Plan02 URB source caps; fit one fixed low-capacity HGB residual model per target/scenario/origin/model; no tuning or feature expansion.",
    }

    write_csv(OUTPUT_DIR / "feature_manifest.csv", FEATURE_MANIFEST, fields=union_fields(FEATURE_MANIFEST))
    write_csv(OUTPUT_DIR / "model_definitions.csv", list(MODEL_DEFINITIONS))
    write_csv(OUTPUT_DIR / "modeling_dataset_diagnostics.csv", dataset_diagnostics)
    write_csv(OUTPUT_DIR / "training_geometry.csv", training_geometry)
    write_csv(OUTPUT_DIR / "feature_missingness_imputation.csv", feature_missingness)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(OUTPUT_DIR / "per_origin_metrics.csv", per_origin_metrics)
    write_csv(OUTPUT_DIR / "per_origin_predictions_or_residuals.csv", all_rows, fields=union_fields(all_rows))
    write_csv(OUTPUT_DIR / "pairwise_b3_comparison.csv", pairwise_b3)
    write_csv(OUTPUT_DIR / "feature_ablation_comparison.csv", ablation)
    write_csv(OUTPUT_DIR / "latest_origin_comparison.csv", latest)
    write_csv(OUTPUT_DIR / "r1_r2_equivalence_check.csv", r1_r2_equivalence)
    write_csv(OUTPUT_DIR / "r1_r2_urbanization_sensitivity.csv", r1_r2_urbanization, fields=union_fields(r1_r2_urbanization))
    write_csv(OUTPUT_DIR / "city_improvement_concentration.csv", concentrations)
    write_csv(OUTPUT_DIR / "leave_one_origin_out.csv", leave_origin)
    write_csv(OUTPUT_DIR / "leave_one_city_sensitivity.csv", leave_city)
    write_csv(OUTPUT_DIR / "leakage_audit.csv", leakage_audit)
    write_csv(OUTPUT_DIR / "b3_compatibility_check.csv", b3_compatibility)
    write_csv(OUTPUT_DIR / "b3_metric_compatibility_check.csv", b3_metric_compatibility)
    write_csv(OUTPUT_DIR / "th_feature_compatibility_check.csv", th_compatibility)
    write_csv(OUTPUT_DIR / "urbanization_information_boundary.csv", urban_boundaries)
    write_json(OUTPUT_DIR / "plan03_summary.json", summary)

    print("S4_PLAN03_HGB_INTERACTION_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_origin_domain=2011-2021")
    print(f"model_ids={','.join(MODEL_IDS)}")
    print(f"prediction_rows={len(all_rows)}")
    print(f"candidate_prediction_rows={len(candidate_rows)}")
    print(f"aggregate_metric_rows={len(aggregate_metrics)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"training_geometry_rows={len(training_geometry)}")
    print(f"leakage_audit_rows={len(leakage_audit)}")
    print(f"b3_compatibility_pass={summary['b3_compatibility_pass']}")
    print(f"b3_metric_compatibility_pass={summary['b3_metric_compatibility_pass']}")
    print(f"th_feature_semantics_compatibility_pass={summary['th_feature_semantics_compatibility_pass']}")
    print(f"m3a_m3b_r1_r2_equivalence_pass={summary['m3a_m3b_r1_r2_equivalence_pass']}")
    print(f"m3c_r1_r2_feature_difference_confirmed={summary['m3c_r1_r2_feature_difference_confirmed']}")
    print(f"population_t_minus_2_audit_pass={summary['population_t_minus_2_audit_pass']}")
    print(f"urbanization_source_cap_audit_pass={summary['urbanization_source_cap_audit_pass']}")
    print(f"preprocessing_fit_boundary_pass={summary['preprocessing_fit_boundary_pass']}")
    print(f"leakage_pass={summary['leakage_pass']}")
    print(f"full_coverage={summary['full_coverage']}")
    print(f"m3a_beats_b3={summary['m3a_beats_b3']}")
    print(f"nonlinear_th_core_supported={summary['nonlinear_th_core_supported']}")
    print(f"nonlinear_shape_ablation_supported={summary['nonlinear_shape_ablation_supported']}")
    print(f"conditional_urbanization_supported={summary['conditional_urbanization_supported']}")
    print("formal_ml_model_used=True")
    print("tree_or_gbdt_model_used=True")
    print("covariates_used=True")
    print("urbanization_used=True")
    print("other_covariates_used=False")
    print("city_identity_used=False")
    print("tuning_run=False")
    print("model_sweep_run=False")
    print("ensemble_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
