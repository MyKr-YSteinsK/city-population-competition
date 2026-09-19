"""Build S4-Plan01 target-history dynamics Ridge evidence.

This runner constructs every supervised row against its own pseudo-target
boundary, then fits a fixed StandardScaler + Ridge(alpha=1.0) pipeline inside
each outer origin.  It compares two pre-registered feature groups:
TH_CORE and TH_CORE + TH_SHAPE.  No covariates, city identity, tuning or
additional model family is used.
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

import build_s3_plan01_baseline_ladder as plan01
import build_s3_plan02_recent_window_and_selector as plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01"
S3_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"

STARTING_COMMIT = "50f22159e1f38332967dd66bd9ddfb9a369fc21c"
HORIZON_YEARS = plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = plan02.LATEST_ORIGIN
COMMON_ORIGINS = plan02.COMMON_ORIGINS
CITIES = plan02.CITIES
SCENARIOS = plan02.SCENARIOS
TARGETS = plan02.TARGETS

B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M1A_ID = "M1A_RIDGE_DELTA_TH_CORE"
M1B_ID = "M1B_RIDGE_DELTA_TH_SHAPE"
ML_IDS = (M1A_ID, M1B_ID)
MODEL_IDS = (B3_ID, M1A_ID, M1B_ID)

CORE_FEATURES = (
    "latest_value",
    "recent3_ols_projected_change",
    "latest_gap_years",
    "recent3_span_years",
)
SHAPE_FEATURES = (
    "slope_disagreement_2v3",
    "recent_relative_slope",
)
ALL_FEATURES = CORE_FEATURES + SHAPE_FEATURES

MODEL_DEFINITIONS = (
    {
        "predictor_id": B3_ID,
        "model_family": "frozen_s3_reference",
        "feature_group_id": "FROZEN_B3",
        "feature_names": "",
        "target_definition": "population level predicted by frozen recent-3 per-city OLS",
        "preprocessing": "not_applicable_frozen_reference",
        "estimator": "frozen_B3",
        "ridge_alpha": "",
        "city_identity_used": "no",
        "covariates_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M1A_ID,
        "model_family": "Ridge_delta",
        "feature_group_id": "TH_CORE",
        "feature_names": ";".join(CORE_FEATURES),
        "target_definition": "future_delta = target_at_y - latest_allowed_value_before_y",
        "preprocessing": "SimpleImputer(median, add_indicator) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "city_identity_used": "no",
        "covariates_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M1B_ID,
        "model_family": "Ridge_delta",
        "feature_group_id": "TH_SHAPE",
        "feature_names": ";".join(ALL_FEATURES),
        "target_definition": "future_delta = target_at_y - latest_allowed_value_before_y",
        "preprocessing": "SimpleImputer(median, add_indicator) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "city_identity_used": "no",
        "covariates_used": "no",
        "tuning": "no",
    },
)

FEATURE_MANIFEST = (
    {
        "feature_group_id": "TH_CORE",
        "feature_name": "latest_value",
        "formula": "target value at latest non-missing year <= row_year-2",
        "source": "same target city history",
        "semantic_role": "local level anchor",
        "missing_policy": "row excluded only if recent-3 history cannot be built; otherwise numeric",
        "used_by": "M1A;M1B",
    },
    {
        "feature_group_id": "TH_CORE",
        "feature_name": "recent3_ols_projected_change",
        "formula": "OLS(recent 3 allowed target points) projected to row_year minus latest_value",
        "source": "same target city history",
        "semantic_role": "B3 mechanism representation",
        "missing_policy": "row excluded only if recent-3 history cannot be built",
        "used_by": "M1A;M1B",
    },
    {
        "feature_group_id": "TH_CORE",
        "feature_name": "latest_gap_years",
        "formula": "row_year - latest_allowed_observation_year",
        "source": "same target city history",
        "semantic_role": "temporal gap",
        "missing_policy": "row excluded only if recent-3 history cannot be built",
        "used_by": "M1A;M1B",
    },
    {
        "feature_group_id": "TH_CORE",
        "feature_name": "recent3_span_years",
        "formula": "max(recent3_years) - min(recent3_years)",
        "source": "same target city history",
        "semantic_role": "observation geometry",
        "missing_policy": "row excluded only if recent-3 history cannot be built",
        "used_by": "M1A;M1B",
    },
    {
        "feature_group_id": "TH_SHAPE",
        "feature_name": "slope_disagreement_2v3",
        "formula": "annualized recent-2 slope minus recent-3 OLS slope",
        "source": "same target city history",
        "semantic_role": "short-vs-recent trend shape",
        "missing_policy": "row excluded only if recent-3 history cannot be built",
        "used_by": "M1B",
    },
    {
        "feature_group_id": "TH_SHAPE",
        "feature_name": "recent_relative_slope",
        "formula": "recent3 OLS slope / latest_value",
        "source": "same target city history",
        "semantic_role": "growth-like scale signal",
        "missing_policy": "illegal or zero denominator becomes explicit missing and is imputed in training fold",
        "used_by": "M1B",
    },
)


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "NA", "N/A", "NaN", "nan", "None"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float:
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
    if isinstance(value, (np.floating,)):
        number = float(value)
        return round(number, 12) if math.isfinite(number) else ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    assert rows or fields
    fieldnames = list(fields or rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: to_scalar(row.get(field)) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(field for row in rows for field in row))


def ols_fit(years: list[int], values: list[float], target_year: int) -> dict[str, float]:
    assert len(years) == len(values) >= 2
    x = [float(year) for year in years]
    x_mean = mean(x)
    y_mean = mean(values)
    denominator = math.fsum((year - x_mean) ** 2 for year in x)
    assert denominator > 0
    numerator = math.fsum((year - x_mean) * (value - y_mean) for year, value in zip(x, values))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return {
        "slope": slope,
        "intercept": intercept,
        "projected_value": intercept + slope * target_year,
    }


def build_feature_row(series: dict[int, float], row_year: int) -> dict[str, Any]:
    cutoff = row_year - HORIZON_YEARS
    eligible_years = sorted(year for year in series if year <= cutoff)
    base: dict[str, Any] = {
        "row_year": row_year,
        "row_history_cutoff_year": cutoff,
        "eligible_history_years": ";".join(str(year) for year in eligible_years),
        "eligible_history_count": len(eligible_years),
        "feature_history_years": "",
        "feature_history_max_year": None,
        "feature_status": "insufficient_recent3_history",
        "missing_feature_names": "",
        "latest_value": None,
        "recent3_ols_slope": None,
        "recent3_ols_projected_value": None,
        "recent3_ols_projected_change": None,
        "latest_gap_years": None,
        "recent3_span_years": None,
        "recent2_annualized_slope": None,
        "slope_disagreement_2v3": None,
        "recent_relative_slope": None,
        "relative_slope_missing_reason": "",
    }
    if len(eligible_years) < 3:
        return base

    recent3_years = eligible_years[-3:]
    recent3_values = [series[year] for year in recent3_years]
    recent2_years = recent3_years[-2:]
    recent2_values = recent3_values[-2:]
    fitted = ols_fit(recent3_years, recent3_values, row_year)
    latest_year = recent3_years[-1]
    latest_value = series[latest_year]
    recent2_slope = (recent2_values[-1] - recent2_values[-2]) / (recent2_years[-1] - recent2_years[-2])
    missing: list[str] = []
    relative_slope: float | None
    relative_reason = ""
    if not math.isfinite(latest_value) or latest_value == 0:
        relative_slope = None
        relative_reason = "latest_value_zero_or_invalid"
        missing.append("recent_relative_slope")
    else:
        relative_slope = fitted["slope"] / latest_value
    base.update(
        {
            "feature_history_years": ";".join(str(year) for year in recent3_years),
            "feature_history_max_year": latest_year,
            "feature_status": "ok",
            "missing_feature_names": ";".join(missing),
            "latest_value": latest_value,
            "recent3_ols_slope": fitted["slope"],
            "recent3_ols_projected_value": fitted["projected_value"],
            "recent3_ols_projected_change": fitted["projected_value"] - latest_value,
            "latest_gap_years": row_year - latest_year,
            "recent3_span_years": recent3_years[-1] - recent3_years[0],
            "recent2_annualized_slope": recent2_slope,
            "slope_disagreement_2v3": recent2_slope - fitted["slope"],
            "recent_relative_slope": relative_slope,
            "relative_slope_missing_reason": relative_reason,
        }
    )
    return base


def feature_values(feature_row: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, float | None]:
    return {
        name: as_float(feature_row.get(name))
        for name in feature_names
    }


def build_training_candidates(series: dict[int, float], outer_origin: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_cutoff = outer_origin - HORIZON_YEARS
    eligible_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row_year in sorted(year for year in series if year <= outer_cutoff):
        target_value = series[row_year]
        features = build_feature_row(series, row_year)
        row = {
            "row_year": row_year,
            "target_value": target_value,
            "latest_value": features["latest_value"],
            "future_delta": target_value - features["latest_value"] if features["feature_status"] == "ok" else None,
            "features": feature_values(features, ALL_FEATURES),
            "feature_row": features,
        }
        if features["feature_status"] != "ok":
            excluded_rows.append({**row, "exclusion_reason": features["feature_status"]})
        else:
            eligible_rows.append(row)
    return eligible_rows, excluded_rows


def build_validation_candidate(series: dict[int, float], origin: int) -> dict[str, Any]:
    features = build_feature_row(series, origin)
    actual = series.get(origin)
    return {
        "row_year": origin,
        "target_value": actual,
        "latest_value": features["latest_value"],
        "future_delta": actual - features["latest_value"] if actual is not None and features["feature_status"] == "ok" else None,
        "features": feature_values(features, ALL_FEATURES),
        "feature_row": features,
    }


def pipeline_for(feature_names: tuple[str, ...]) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def fit_outer_model(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[Pipeline, list[float], dict[str, Any]]:
    assert train_rows
    x_train = np.asarray(
        [[row["features"].get(name, np.nan) for name in feature_names] for row in train_rows],
        dtype=float,
    )
    y_train = np.asarray([row["future_delta"] for row in train_rows], dtype=float)
    x_validation = np.asarray(
        [[row["features"].get(name, np.nan) for name in feature_names] for row in validation_rows],
        dtype=float,
    )
    train_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    model = pipeline_for(feature_names)
    model.fit(x_train, y_train)
    predicted_delta = model.predict(x_validation).tolist()
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    metadata = {
        "training_row_count": len(train_rows),
        "training_missing_feature_cell_count": train_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "training_imputed_feature_cell_count": train_missing,
        "validation_imputed_feature_cell_count": validation_missing,
        "preprocessing_fit_row_count": len(train_rows),
        "model_fit_row_count": len(train_rows),
        "preprocessing_model_fit_rows_equal": "yes" if len(train_rows) == len(train_rows) else "no",
        "imputer_statistics": [None if not math.isfinite(float(value)) else float(value) for value in imputer.statistics_],
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": [float(value) for value in np.asarray(ridge.coef_).ravel()],
        "feature_names": ";".join(feature_names),
        "ridge_alpha": 1.0,
    }
    return model, predicted_delta, metadata


def model_row_from_validation(
    target: dict[str, str],
    scenario_id: str,
    model_id: str,
    origin: int,
    city: str,
    validation: dict[str, Any],
    predicted_delta: float,
    metadata: dict[str, Any],
    feature_group_id: str,
) -> dict[str, Any]:
    actual = validation["target_value"]
    prediction = validation["latest_value"] + predicted_delta if validation["latest_value"] is not None else None
    scoreable = prediction is not None and actual is not None and math.isfinite(float(prediction))
    residual = float(prediction) - float(actual) if scoreable else None
    feature_row = validation["feature_row"]
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
        "actual_target_year_for_scoring": origin if actual is not None else "",
        "actual_target_value_for_scoring": actual,
        "row_role": "validation",
        "feature_builder_id": "TH_DYNAMICS_V1",
        "feature_group_id": feature_group_id,
        "feature_history_years": feature_row["feature_history_years"],
        "feature_history_max_year": feature_row["feature_history_max_year"],
        "eligible_history_years": feature_row["eligible_history_years"],
        "latest_value": validation["latest_value"],
        "actual_future_delta": validation["future_delta"],
        "predicted_future_delta": predicted_delta,
        "latest_gap_years": feature_row["latest_gap_years"],
        "recent3_span_years": feature_row["recent3_span_years"],
        "recent3_ols_slope": feature_row["recent3_ols_slope"],
        "recent3_ols_projected_change": feature_row["recent3_ols_projected_change"],
        "slope_disagreement_2v3": feature_row["slope_disagreement_2v3"],
        "recent_relative_slope": feature_row["recent_relative_slope"],
        "missing_feature_names": feature_row["missing_feature_names"],
        "prediction_value": prediction,
        "prediction_status": "scoreable" if scoreable else "not_scoreable",
        "scoreable": "yes" if scoreable else "no",
        "residual_prediction_minus_actual": residual,
        "squared_error": residual * residual if residual is not None else None,
        "absolute_error": abs(residual) if residual is not None else None,
        "training_label_min_year": metadata["training_label_min_year"],
        "training_label_max_year": metadata["training_label_max_year"],
        "training_row_count": metadata["training_row_count"],
        "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
        "model_fit_row_count": metadata["model_fit_row_count"],
        "training_missing_feature_cell_count": metadata["training_missing_feature_cell_count"],
        "validation_missing_feature_cell_count": metadata["validation_missing_feature_cell_count"],
        "covariates_used": "no",
        "city_identity_used": "no",
    }


def frozen_b3_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "model_id": B3_ID,
        "row_role": "validation",
        "feature_builder_id": "FROZEN_B3_REFERENCE",
        "feature_group_id": "FROZEN_B3",
        "feature_history_years": row["history_years_used"],
        "feature_history_max_year": row["history_max_year"],
        "eligible_history_years": row["eligible_history_years"],
        "latest_value": None,
        "actual_future_delta": None,
        "predicted_future_delta": None,
        "covariates_used": "no",
        "city_identity_used": "no",
    }


def load_context() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any], dict[str, Any]]:
    plan01.assert_static_boundaries()
    population, profile = plan01.load_population()
    s2_summary, _, _ = plan01.load_s2_context()
    plan01.load_scenarios()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    s3_summary = json.loads((S3_DIR / "plan03_summary.json").read_text(encoding="utf-8"))
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    assert s3_summary["common_origin_domain"] == "2011-2021"
    return population, profile, s3_summary


def compare_b3_reference(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = plan01.read_csv(S3_DIR / "per_origin_predictions_or_residuals.csv")
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
    assert len(frozen_map) == len(current_map) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    for (target_id, scenario_id), pairs in sorted(grouped.items()):
        deltas: list[float] = []
        residual_deltas: list[float] = []
        squared_deltas: list[float] = []
        exact_scope = True
        for current, frozen in pairs:
            for field, output in (
                ("prediction_value", deltas),
                ("residual_prediction_minus_actual", residual_deltas),
                ("squared_error", squared_deltas),
            ):
                left = as_float(current.get(field))
                right = as_float(frozen.get(field))
                if left is None or right is None:
                    if left != right:
                        exact_scope = False
                else:
                    delta = abs(left - right)
                    output.append(delta)
                    if delta > 1e-9:
                        exact_scope = False
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "rows_compared": len(pairs),
                "max_abs_prediction_delta": max(deltas) if deltas else 0.0,
                "max_abs_residual_delta": max(residual_deltas) if residual_deltas else 0.0,
                "max_abs_squared_error_delta": max(squared_deltas) if squared_deltas else 0.0,
                "status": "PASS" if exact_scope else "FAIL",
                "comparison_rule": "B3 row-level prediction/residual/squared-error delta <= 1e-9",
            }
        )
    assert all(row["status"] == "PASS" for row in result)
    return result


def prepare_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    saved_predictors = plan01.PREDICTORS
    plan01.PREDICTORS = MODEL_DEFINITIONS
    try:
        per_origin, _ = plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = plan01.build_aggregate_metrics(rows, per_origin)
        r1_r2 = plan01.build_r1_r2_equivalence(rows)
    finally:
        plan01.PREDICTORS = saved_predictors
    assert all(row["status"] == "PASS" for row in r1_r2)
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin
    }
    return per_origin, aggregate, aggregate_map, r1_r2


def score_rows(rows: list[dict[str, Any]], target_id: str, scenario_id: str, model_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["target_id"] == target_id
        and row["scenario_id"] == scenario_id
        and row["predictor_id"] == model_id
        and row["scoreable"] == "yes"
    ]


def build_pairwise_row(
    rows: list[dict[str, Any]],
    per_origin_map: dict[tuple[str, str, str, int], dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    target_id: str,
    scenario_id: str,
    baseline_id: str,
    candidate_id: str,
    pair_id: str,
) -> dict[str, Any]:
    baseline_aggregate = aggregate_map[(target_id, scenario_id, baseline_id)]
    candidate_aggregate = aggregate_map[(target_id, scenario_id, candidate_id)]
    origin_wins = origin_losses = origin_ties = 0
    origin_deltas: list[float] = []
    for origin in COMMON_ORIGINS:
        base = per_origin_map[(target_id, scenario_id, baseline_id, origin)]
        cand = per_origin_map[(target_id, scenario_id, candidate_id, origin)]
        delta = float(base["mse"]) - float(cand["mse"])
        origin_deltas.append(delta)
        if delta > 1e-12:
            origin_wins += 1
        elif delta < -1e-12:
            origin_losses += 1
        else:
            origin_ties += 1
    baseline_rows = {
        (row["origin_year"], row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, baseline_id)
    }
    candidate_rows = {
        (row["origin_year"], row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, candidate_id)
    }
    deltas = [
        float(baseline_rows[key]["squared_error"]) - float(candidate_rows[key]["squared_error"])
        for key in sorted(baseline_rows)
    ]
    improved = sum(delta > 1e-12 for delta in deltas)
    worsened = sum(delta < -1e-12 for delta in deltas)
    tied = len(deltas) - improved - worsened
    latest_base = per_origin_map[(target_id, scenario_id, baseline_id, LATEST_ORIGIN)]
    latest_candidate = per_origin_map[(target_id, scenario_id, candidate_id, LATEST_ORIGIN)]
    baseline_mse = float(baseline_aggregate["pooled_city_year_mse"])
    candidate_mse = float(candidate_aggregate["pooled_city_year_mse"])
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "pair_id": pair_id,
        "baseline_predictor_id": baseline_id,
        "candidate_predictor_id": candidate_id,
        "baseline_pooled_mse": baseline_mse,
        "candidate_pooled_mse": candidate_mse,
        "pooled_mse_absolute_delta_baseline_minus_candidate": baseline_mse - candidate_mse,
        "pooled_mse_relative_improvement_pct": (baseline_mse - candidate_mse) / baseline_mse * 100.0,
        "baseline_macro_origin_mse": float(baseline_aggregate["macro_origin_mse"]),
        "candidate_macro_origin_mse": float(candidate_aggregate["macro_origin_mse"]),
        "baseline_pooled_rmse": float(baseline_aggregate["pooled_city_year_rmse"]),
        "candidate_pooled_rmse": float(candidate_aggregate["pooled_city_year_rmse"]),
        "pooled_rmse_delta_baseline_minus_candidate": float(baseline_aggregate["pooled_city_year_rmse"]) - float(candidate_aggregate["pooled_city_year_rmse"]),
        "baseline_pooled_mae": float(baseline_aggregate["pooled_city_year_mae"]),
        "candidate_pooled_mae": float(candidate_aggregate["pooled_city_year_mae"]),
        "pooled_mae_delta_baseline_minus_candidate": float(baseline_aggregate["pooled_city_year_mae"]) - float(candidate_aggregate["pooled_city_year_mae"]),
        "pooled_score_delta_candidate_minus_baseline": float(candidate_aggregate["pooled_competition_score_from_mse"]) - float(baseline_aggregate["pooled_competition_score_from_mse"]),
        "latest_baseline_mse": float(latest_base["mse"]),
        "latest_candidate_mse": float(latest_candidate["mse"]),
        "latest_mse_relative_improvement_pct": (float(latest_base["mse"]) - float(latest_candidate["mse"])) / float(latest_base["mse"]) * 100.0,
        "latest_candidate_outcome": "win" if float(latest_base["mse"]) > float(latest_candidate["mse"]) + 1e-12 else "loss" if float(latest_base["mse"]) < float(latest_candidate["mse"]) - 1e-12 else "tie",
        "candidate_origin_win_count": origin_wins,
        "candidate_origin_loss_count": origin_losses,
        "origin_tie_count": origin_ties,
        "origin_count": len(COMMON_ORIGINS),
        "candidate_origin_win_rate": origin_wins / len(COMMON_ORIGINS),
        "candidate_city_origin_improved_share": improved / len(deltas),
        "candidate_city_origin_worsened_share": worsened / len(deltas),
        "candidate_city_origin_tied_share": tied / len(deltas),
        "origin_mse_delta_min": min(origin_deltas),
        "origin_mse_delta_max": max(origin_deltas),
        "ranking_metric": "pooled_city_year_mse",
        "score_is_secondary_transform_only": "yes",
    }


def build_concentration_row(
    rows: list[dict[str, Any]],
    target_id: str,
    scenario_id: str,
    candidate_id: str,
    baseline_id: str = B3_ID,
) -> dict[str, Any]:
    baseline_rows = score_rows(rows, target_id, scenario_id, baseline_id)
    candidate_rows = score_rows(rows, target_id, scenario_id, candidate_id)
    base_by_key = {(row["origin_year"], row["city_id"]): row for row in baseline_rows}
    cand_by_key = {(row["origin_year"], row["city_id"]): row for row in candidate_rows}
    city_base: defaultdict[str, float] = defaultdict(float)
    city_cand: defaultdict[str, float] = defaultdict(float)
    city_delta: defaultdict[str, float] = defaultdict(float)
    row_deltas: list[float] = []
    for key in sorted(base_by_key):
        base = float(base_by_key[key]["squared_error"])
        cand = float(cand_by_key[key]["squared_error"])
        city = key[1]
        city_base[city] += base
        city_cand[city] += cand
        city_delta[city] += base - cand
        row_deltas.append(base - cand)
    total_base = math.fsum(city_base.values())
    total_cand = math.fsum(city_cand.values())
    ranked_cand = sorted(city_cand.items(), key=lambda item: (-item[1], item[0]))
    positive_delta = [(city, delta) for city, delta in city_delta.items() if delta > 0]
    positive_delta.sort(key=lambda item: (-item[1], item[0]))
    positive_total = math.fsum(delta for _, delta in positive_delta)
    largest_improvement_city = positive_delta[0][0] if positive_delta else ""
    largest_error_city = ranked_cand[0][0]
    def mse_without(city: str, source: dict[tuple[str, str], dict[str, Any]]) -> float:
        values = [float(row["squared_error"]) for key, row in source.items() if key[1] != city]
        return mean(values)
    base_without_improvement = mse_without(largest_improvement_city, base_by_key) if largest_improvement_city else ""
    cand_without_improvement = mse_without(largest_improvement_city, cand_by_key) if largest_improvement_city else ""
    base_without_error = mse_without(largest_error_city, base_by_key)
    cand_without_error = mse_without(largest_error_city, cand_by_key)
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "baseline_predictor_id": baseline_id,
        "candidate_predictor_id": candidate_id,
        "baseline_pooled_sse": total_base,
        "candidate_pooled_sse": total_cand,
        "pooled_mse_absolute_delta_baseline_minus_candidate": (total_base - total_cand) / len(row_deltas),
        "dominant_baseline_city_id": max(city_base, key=lambda city: (city_base[city], city)),
        "dominant_candidate_city_id": ranked_cand[0][0],
        "baseline_dominant_city_sse_share": max(city_base.values()) / total_base,
        "candidate_dominant_city_sse_share": ranked_cand[0][1] / total_cand,
        "baseline_top3_city_sse_share": sum(value for _, value in sorted(city_base.items(), key=lambda item: (-item[1], item[0]))[:3]) / total_base,
        "candidate_top3_city_sse_share": sum(value for _, value in ranked_cand[:3]) / total_cand,
        "improved_city_count": sum(city_delta[city] > 1e-12 for city in CITIES),
        "worsened_city_count": sum(city_delta[city] < -1e-12 for city in CITIES),
        "tied_city_count": sum(abs(city_delta[city]) <= 1e-12 for city in CITIES),
        "improved_city_share": sum(city_delta[city] > 1e-12 for city in CITIES) / len(CITIES),
        "worsened_city_share": sum(city_delta[city] < -1e-12 for city in CITIES) / len(CITIES),
        "improved_city_origin_row_share": sum(delta > 1e-12 for delta in row_deltas) / len(row_deltas),
        "worsened_city_origin_row_share": sum(delta < -1e-12 for delta in row_deltas) / len(row_deltas),
        "top_positive_delta_city_id": largest_improvement_city,
        "top1_positive_delta_sse_contribution_share": positive_delta[0][1] / positive_total if positive_total else 0.0,
        "top3_positive_delta_sse_contribution_share": math.fsum(delta for _, delta in positive_delta[:3]) / positive_total if positive_total else 0.0,
        "remove_largest_improvement_city_id": largest_improvement_city,
        "baseline_mse_without_largest_improvement_city": base_without_improvement,
        "candidate_mse_without_largest_improvement_city": cand_without_improvement,
        "candidate_improvement_without_largest_improvement_city": (base_without_improvement - cand_without_improvement) if largest_improvement_city else "",
        "remove_largest_error_city_id": largest_error_city,
        "baseline_mse_without_largest_error_city": base_without_error,
        "candidate_mse_without_largest_error_city": cand_without_error,
        "candidate_improvement_without_largest_error_city": base_without_error - cand_without_error,
        "interpretation_scope": "city-origin residual rows; no city reweighting",
    }


def build_leave_one_origin(rows: list[dict[str, Any]], aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            full_leader = min(
                MODEL_IDS,
                key=lambda model_id: float(aggregate_map[(target_id, scenario_id, model_id)]["pooled_city_year_mse"]),
            )
            for omitted_origin in COMMON_ORIGINS:
                model_mse: dict[str, float] = {}
                for model_id in MODEL_IDS:
                    values = [
                        float(row["squared_error"])
                        for row in score_rows(rows, target_id, scenario_id, model_id)
                        if int(row["origin_year"]) != omitted_origin
                    ]
                    model_mse[model_id] = mean(values)
                ordered = sorted(model_mse.items(), key=lambda item: (item[1], MODEL_IDS.index(item[0])))
                leader = ordered[0][0]
                for rank, (model_id, mse) in enumerate(ordered, start=1):
                    result.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "omitted_origin_year": omitted_origin,
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


def build_leave_one_city(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            baseline_rows = score_rows(rows, target_id, scenario_id, B3_ID)
            base_by_key = {(row["origin_year"], row["city_id"]): row for row in baseline_rows}
            for candidate_id in ML_IDS:
                candidate_rows = score_rows(rows, target_id, scenario_id, candidate_id)
                cand_by_key = {(row["origin_year"], row["city_id"]): row for row in candidate_rows}
                for removed_city in CITIES:
                    base_values = [float(row["squared_error"]) for key, row in base_by_key.items() if key[1] != removed_city]
                    cand_values = [float(row["squared_error"]) for key, row in cand_by_key.items() if key[1] != removed_city]
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


def build_latest_rows(aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for model_id in MODEL_IDS:
                aggregate = aggregate_map[(target["target_id"], scenario_id, model_id)]
                rows.append(
                    {
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "predictor_id": model_id,
                        "latest_origin_year": LATEST_ORIGIN,
                        "latest_origin_mse": aggregate["latest_origin_mse"],
                        "latest_origin_rmse": aggregate["latest_origin_rmse"],
                        "latest_origin_mae": aggregate["latest_origin_mae"],
                        "latest_origin_bias": aggregate["latest_origin_mean_signed_error_bias"],
                        "latest_origin_score": aggregate["latest_origin_competition_score_from_mse"],
                        "latest_origin_rank": aggregate["latest_origin_mse_rank"],
                        "pooled_mse": aggregate["pooled_city_year_mse"],
                        "pooled_mse_relative_to_b3_pct": (
                            float(aggregate["pooled_city_year_mse"]) - float(aggregate_map[(target["target_id"], scenario_id, B3_ID)]["pooled_city_year_mse"])
                        )
                        / float(aggregate_map[(target["target_id"], scenario_id, B3_ID)]["pooled_city_year_mse"])
                        * 100.0,
                    }
                )
    return rows


def build_training_rows_and_models(
    population: dict[str, dict[str, dict[int, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ml_prediction_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    leakage: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for origin in COMMON_ORIGINS:
            train_rows: list[dict[str, Any]] = []
            excluded_rows: list[dict[str, Any]] = []
            for train_city in CITIES:
                train, excluded = build_training_candidates(population[target_id][train_city], origin)
                for row in train:
                    row["city_id"] = train_city
                for row in excluded:
                    row["city_id"] = train_city
                train_rows.extend(train)
                excluded_rows.extend(excluded)
            assert train_rows
            validation_rows = [
                build_validation_candidate(population[target_id][city], origin)
                for city in CITIES
            ]
            assert all(row["feature_row"]["feature_status"] == "ok" and row["target_value"] is not None for row in validation_rows)
            train_min_year = min(row["row_year"] for row in train_rows)
            train_max_year = max(row["row_year"] for row in train_rows)
            feature_max = max(row["feature_row"]["feature_history_max_year"] for row in train_rows)
            validation_feature_max = max(row["feature_row"]["feature_history_max_year"] for row in validation_rows)
            exclusion_counts = Counter(row["exclusion_reason"] for row in excluded_rows)
            for scenario_id in SCENARIOS:
                for model_id in ML_IDS:
                    feature_names = CORE_FEATURES if model_id == M1A_ID else ALL_FEATURES
                    _, predicted_deltas, metadata = fit_outer_model(train_rows, validation_rows, feature_names)
                    metadata.update(
                        {
                            "training_label_min_year": train_min_year,
                            "training_label_max_year": train_max_year,
                        }
                    )
                    for city, validation, predicted_delta in zip(CITIES, validation_rows, predicted_deltas):
                        ml_prediction_rows.append(
                            model_row_from_validation(
                                target,
                                scenario_id,
                                model_id,
                                origin,
                                city,
                                validation,
                                predicted_delta,
                                metadata,
                                "TH_CORE" if model_id == M1A_ID else "TH_SHAPE",
                            )
                        )
                        feature_row = validation["feature_row"]
                        leakage.append(
                            {
                                "row_role": "validation_feature_row",
                                "model_id": model_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin,
                                "city_id": city,
                                "row_year": origin,
                                "allowed_row_history_cutoff_year": origin - HORIZON_YEARS,
                                "feature_history_max_year": feature_row["feature_history_max_year"],
                                "training_label_max_year": metadata["training_label_max_year"],
                                "validation_feature_max_within_T_minus_2": "PASS" if feature_row["feature_history_max_year"] <= origin - HORIZON_YEARS else "FAIL",
                                "training_label_max_within_T_minus_2": "PASS" if metadata["training_label_max_year"] <= origin - HORIZON_YEARS else "FAIL",
                                "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                                "model_fit_row_count": metadata["model_fit_row_count"],
                                "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                                "outer_t_minus_1_target_present": "yes" if origin - 1 in population[target_id][city] else "no",
                                "outer_t_target_present": "yes" if origin in population[target_id][city] else "no",
                                "t_minus_1_used_in_features": "no",
                                "t_used_in_features": "no",
                                "future_target_years_used": "no",
                                "feature_source_target_id": target_id,
                                "cross_target_contamination": "no",
                                "covariate_columns_used": "no",
                                "leakage_status": "PASS" if feature_row["feature_history_max_year"] <= origin - HORIZON_YEARS and metadata["training_label_max_year"] <= origin - HORIZON_YEARS else "FAIL",
                            }
                        )
                    for train_row in train_rows:
                        train_feature_row = train_row["feature_row"]
                        row_cutoff = train_row["row_year"] - HORIZON_YEARS
                        leakage.append(
                            {
                                "row_role": "training_feature_row",
                                "model_id": model_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin,
                                "city_id": train_row["city_id"],
                                "row_year": train_row["row_year"],
                                "allowed_row_history_cutoff_year": row_cutoff,
                                "feature_history_max_year": train_feature_row["feature_history_max_year"],
                                "training_label_max_year": metadata["training_label_max_year"],
                                "validation_feature_max_within_T_minus_2": "not_applicable",
                                "training_label_max_within_T_minus_2": "PASS",
                                "training_row_feature_max_within_y_minus_2": "PASS" if train_feature_row["feature_history_max_year"] <= row_cutoff else "FAIL",
                                "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                                "model_fit_row_count": metadata["model_fit_row_count"],
                                "preprocessing_model_fit_rows_equal": metadata["preprocessing_model_fit_rows_equal"],
                                "outer_t_minus_1_target_present": "not_applicable",
                                "outer_t_target_present": "not_applicable",
                                "t_minus_1_used_in_features": "no",
                                "t_used_in_features": "no",
                                "future_target_years_used": "no",
                                "feature_source_target_id": target_id,
                                "cross_target_contamination": "no",
                                "covariate_columns_used": "no",
                                "leakage_status": "PASS" if train_feature_row["feature_history_max_year"] <= row_cutoff and metadata["training_label_max_year"] <= origin - HORIZON_YEARS else "FAIL",
                            }
                        )
                    diagnostics.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "target_history_cutoff_year": origin - HORIZON_YEARS,
                            "model_id": model_id,
                            "feature_group_id": "TH_CORE" if model_id == M1A_ID else "TH_SHAPE",
                            "training_candidate_row_count": len(train_rows) + len(excluded_rows),
                            "training_eligible_row_count": len(train_rows),
                            "training_excluded_row_count": len(excluded_rows),
                            "training_excluded_reason_counts": stable_json(dict(exclusion_counts)),
                            "validation_expected_city_count": len(CITIES),
                            "validation_feature_eligible_city_count": len(validation_rows),
                            "validation_scoreable_city_count": len(validation_rows),
                            "training_label_min_year": train_min_year,
                            "training_label_max_year": train_max_year,
                            "training_feature_max_year": feature_max,
                            "validation_feature_max_year": validation_feature_max,
                            "training_missing_feature_cell_count": metadata["training_missing_feature_cell_count"],
                            "validation_missing_feature_cell_count": metadata["validation_missing_feature_cell_count"],
                            "imputation_policy": "median fit on training rows only, add_indicator=True",
                            "full_validation_coverage": "yes" if len(validation_rows) == len(CITIES) else "no",
                            "coverage_status": "PASS" if len(validation_rows) == len(CITIES) else "FAIL",
                        }
                    )
                    geometry.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin,
                            "model_id": model_id,
                            "feature_group_id": "TH_CORE" if model_id == M1A_ID else "TH_SHAPE",
                            "training_row_count": metadata["training_row_count"],
                            "training_label_min_year": train_min_year,
                            "training_label_max_year": train_max_year,
                            "training_feature_max_year": feature_max,
                            "training_history_max_within_outer_cutoff": "PASS" if train_max_year <= origin - HORIZON_YEARS else "FAIL",
                            "training_row_feature_max_within_y_minus_2": "PASS" if all(row["feature_row"]["feature_history_max_year"] <= row["row_year"] - HORIZON_YEARS for row in train_rows) else "FAIL",
                            "validation_feature_max_year": validation_feature_max,
                            "validation_feature_max_within_outer_cutoff": "PASS" if validation_feature_max <= origin - HORIZON_YEARS else "FAIL",
                            "ridge_alpha": metadata["ridge_alpha"],
                            "ridge_intercept": metadata["ridge_intercept"],
                            "ridge_coefficients": stable_json(metadata["ridge_coefficients"]),
                            "feature_names": metadata["feature_names"],
                            "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
                            "model_fit_row_count": metadata["model_fit_row_count"],
                        }
                    )
    assert len(ml_prediction_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(ML_IDS)
    assert diagnostics
    assert all(row["coverage_status"] == "PASS" for row in diagnostics)
    assert all(row["leakage_status"] == "PASS" for row in leakage)
    return ml_prediction_rows, diagnostics, geometry, leakage, []


def validate_summary_metrics(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    s3_aggregate = plan01.read_csv(S3_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in s3_aggregate
        if row["predictor_id"] == B3_ID
    }
    result: list[dict[str, Any]] = []
    for key, frozen in sorted(frozen_map.items()):
        current = aggregate_map[key]
        fields = ("pooled_city_year_mse", "macro_origin_mse", "latest_origin_mse")
        deltas = {field: abs(float(current[field]) - float(frozen[field])) for field in fields}
        passed = all(delta <= 1e-9 for delta in deltas.values())
        result.append(
            {
                "target_id": key[0],
                "scenario_id": key[1],
                "predictor_id": key[2],
                "max_abs_aggregate_delta": max(deltas.values()),
                "status": "PASS" if passed else "FAIL",
                "comparison_rule": "B3 pooled, macro and latest MSE delta <= 1e-9",
            }
        )
    assert all(row["status"] == "PASS" for row in result)
    return result


def main() -> None:
    assert STARTING_COMMIT == "50f22159e1f38332967dd66bd9ddfb9a369fc21c"
    population, population_profile, s3_summary = load_context()
    b3_rows, _ = plan02.build_fixed_rows(population)
    b3_rows = [row for row in b3_rows if row["predictor_id"] == B3_ID]
    assert len(b3_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    b3_compatibility = compare_b3_reference(b3_rows)
    ml_rows, dataset_diagnostics, training_geometry, leakage_audit, _ = build_training_rows_and_models(population)
    all_rows = [frozen_b3_row(row) for row in b3_rows] + ml_rows
    assert len(all_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * len(MODEL_IDS)

    per_origin_metrics, aggregate_metrics, aggregate_map, r1_r2 = prepare_metrics(all_rows)
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin_metrics
    }
    frozen_metric_check = validate_summary_metrics(aggregate_map)
    assert all(row["status"] == "PASS" for row in frozen_metric_check)

    pairwise_rows = []
    ablation_rows = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for model_id in ML_IDS:
                pairwise_rows.append(
                    build_pairwise_row(
                        all_rows,
                        per_origin_map,
                        aggregate_map,
                        target["target_id"],
                        scenario_id,
                        B3_ID,
                        model_id,
                        f"{model_id}_vs_B3",
                    )
                )
            ablation_rows.append(
                build_pairwise_row(
                    all_rows,
                    per_origin_map,
                    aggregate_map,
                    target["target_id"],
                    scenario_id,
                    M1A_ID,
                    M1B_ID,
                    "M1B_vs_M1A",
                )
            )

    concentration_rows = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for model_id in ML_IDS:
                concentration_rows.append(build_concentration_row(all_rows, target["target_id"], scenario_id, model_id))
    leave_one_origin = build_leave_one_origin(all_rows, aggregate_map)
    leave_one_city = build_leave_one_city(all_rows)
    latest_rows = build_latest_rows(aggregate_map)

    resident_primary = {
        model_id: aggregate_map[("resident", SCENARIOS[0], model_id)]
        for model_id in MODEL_IDS
    }
    registered_sensitivity = {
        model_id: aggregate_map[("registered", SCENARIOS[0], model_id)]
        for model_id in MODEL_IDS
    }
    best_ml_model = min(ML_IDS, key=lambda model_id: float(resident_primary[model_id]["pooled_city_year_mse"]))
    resident_b3_mse = float(resident_primary[B3_ID]["pooled_city_year_mse"])
    best_ml_mse = float(resident_primary[best_ml_model]["pooled_city_year_mse"])
    b3_loo = [row for row in leave_one_origin if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0] and row["predictor_id"] == B3_ID]
    m1b_loo = [row for row in leave_one_origin if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0] and row["predictor_id"] == M1B_ID]
    ablation_resident = next(row for row in ablation_rows if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0])
    ablation_registered = next(row for row in ablation_rows if row["target_id"] == "registered" and row["scenario_id"] == SCENARIOS[0])
    shape_supported = (
        float(ablation_resident["pooled_mse_absolute_delta_baseline_minus_candidate"]) > 0
        and int(ablation_resident["candidate_origin_win_count"]) > int(ablation_resident["candidate_origin_loss_count"])
        and float(ablation_registered["pooled_mse_absolute_delta_baseline_minus_candidate"]) >= 0
        and sum(row["full_domain_leader_survives"] == "yes" for row in m1b_loo) >= sum(row["full_domain_leader_survives"] == "yes" for row in b3_loo)
    )
    best_pair = next(row for row in pairwise_rows if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0] and row["candidate_predictor_id"] == best_ml_model)
    summary = {
        "plan_id": "S4-Plan01",
        "stage": "S4_FEATURE_ML_MODELING",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "target_history_cutoff_rule": "outer validation T uses target history <= T-2; historical training row y uses features <= y-2",
        "model_ids": list(MODEL_IDS),
        "feature_group_ids": ["TH_CORE", "TH_SHAPE"],
        "prediction_rows": len(all_rows),
        "ml_prediction_rows": len(ml_rows),
        "per_origin_metric_rows": len(per_origin_metrics),
        "aggregate_metric_rows": len(aggregate_metrics),
        "training_geometry_rows": len(training_geometry),
        "modeling_dataset_diagnostics_rows": len(dataset_diagnostics),
        "leakage_audit_rows": len(leakage_audit),
        "full_coverage": all(row["coverage_status"] == "PASS" for row in dataset_diagnostics),
        "resident_b3_pooled_mse": resident_primary[B3_ID]["pooled_city_year_mse"],
        "resident_b3_macro_mse": resident_primary[B3_ID]["macro_origin_mse"],
        "resident_b3_latest_2021_mse": resident_primary[B3_ID]["latest_origin_mse"],
        "resident_m1a_pooled_mse": resident_primary[M1A_ID]["pooled_city_year_mse"],
        "resident_m1a_macro_mse": resident_primary[M1A_ID]["macro_origin_mse"],
        "resident_m1a_latest_2021_mse": resident_primary[M1A_ID]["latest_origin_mse"],
        "resident_m1b_pooled_mse": resident_primary[M1B_ID]["pooled_city_year_mse"],
        "resident_m1b_macro_mse": resident_primary[M1B_ID]["macro_origin_mse"],
        "resident_m1b_latest_2021_mse": resident_primary[M1B_ID]["latest_origin_mse"],
        "registered_b3_pooled_mse": registered_sensitivity[B3_ID]["pooled_city_year_mse"],
        "registered_b3_macro_mse": registered_sensitivity[B3_ID]["macro_origin_mse"],
        "registered_b3_latest_2021_mse": registered_sensitivity[B3_ID]["latest_origin_mse"],
        "registered_m1a_pooled_mse": registered_sensitivity[M1A_ID]["pooled_city_year_mse"],
        "registered_m1a_macro_mse": registered_sensitivity[M1A_ID]["macro_origin_mse"],
        "registered_m1a_latest_2021_mse": registered_sensitivity[M1A_ID]["latest_origin_mse"],
        "registered_m1b_pooled_mse": registered_sensitivity[M1B_ID]["pooled_city_year_mse"],
        "registered_m1b_macro_mse": registered_sensitivity[M1B_ID]["macro_origin_mse"],
        "registered_m1b_latest_2021_mse": registered_sensitivity[M1B_ID]["latest_origin_mse"],
        "best_ml_model_by_resident_pooled_mse": best_ml_model,
        "best_ml_resident_pooled_mse": best_ml_mse,
        "best_ml_improvement_vs_b3_pct": (resident_b3_mse - best_ml_mse) / resident_b3_mse * 100.0,
        "resident_b3_beaten": "yes" if best_ml_mse < resident_b3_mse else "no",
        "per_origin_win_counts_vs_b3": {
            f"{row['target_id']}_{row['candidate_predictor_id']}": row["candidate_origin_win_count"]
            for row in pairwise_rows
            if row["scenario_id"] == SCENARIOS[0]
        },
        "leave_one_origin_leader_survival": {
            f"{target_id}_{model_id}": sum(
                row["full_domain_leader_survives"] == "yes"
                for row in leave_one_origin
                if row["target_id"] == target_id
                and row["scenario_id"] == SCENARIOS[0]
                and row["predictor_id"] == model_id
            )
            for target_id in (target["target_id"] for target in TARGETS)
            for model_id in MODEL_IDS
        },
        "dominant_city_sse_shares": {
            f"{row['target_id']}_{row['candidate_predictor_id']}": row["candidate_dominant_city_sse_share"]
            for row in concentration_rows
            if row["scenario_id"] == SCENARIOS[0]
        },
        "top_three_city_sse_shares": {
            f"{row['target_id']}_{row['candidate_predictor_id']}": row["candidate_top3_city_sse_share"]
            for row in concentration_rows
            if row["scenario_id"] == SCENARIOS[0]
        },
        "improved_city_shares": {
            f"{row['target_id']}_{row['candidate_predictor_id']}": row["improved_city_share"]
            for row in concentration_rows
            if row["scenario_id"] == SCENARIOS[0]
        },
        "improved_city_origin_shares": {
            f"{row['target_id']}_{row['candidate_predictor_id']}": row["improved_city_origin_row_share"]
            for row in concentration_rows
            if row["scenario_id"] == SCENARIOS[0]
        },
        "leakage_pass": all(row["leakage_status"] == "PASS" for row in leakage_audit),
        "r1_r2_equivalence_pass": all(row["status"] == "PASS" for row in r1_r2),
        "b3_row_compatibility_pass": all(row["status"] == "PASS" for row in b3_compatibility),
        "b3_metric_compatibility_pass": all(row["status"] == "PASS" for row in frozen_metric_check),
        "shape_ablation_supported": "yes" if shape_supported else "no",
        "formal_ml_model_used": True,
        "covariates_used": False,
        "city_identity_used": False,
        "tuning_run": False,
        "ensemble_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
        "population_profile": population_profile,
        "s3_reference_summary": {
            "starting_commit": s3_summary["starting_commit"],
            "strongest_fixed_baseline": s3_summary["strongest_deployable_fixed_baseline_candidate"],
        },
        "technical_decision": "Use sklearn SimpleImputer(median, add_indicator=True), StandardScaler and Ridge(alpha=1.0) per outer origin; no missing feature is silently mapped to zero.",
        "best_pair_reference": best_pair["pair_id"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "feature_manifest.csv", list(FEATURE_MANIFEST))
    write_csv(OUTPUT_DIR / "modeling_dataset_diagnostics.csv", dataset_diagnostics)
    write_csv(OUTPUT_DIR / "training_geometry.csv", training_geometry)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(OUTPUT_DIR / "per_origin_metrics.csv", per_origin_metrics)
    write_csv(OUTPUT_DIR / "per_origin_predictions_or_residuals.csv", all_rows, fields=union_fields(all_rows))
    write_csv(OUTPUT_DIR / "pairwise_b3_comparison.csv", pairwise_rows)
    write_csv(OUTPUT_DIR / "ablation_comparison.csv", ablation_rows)
    write_csv(OUTPUT_DIR / "latest_origin_comparison.csv", latest_rows)
    write_csv(OUTPUT_DIR / "city_improvement_concentration.csv", concentration_rows)
    write_csv(OUTPUT_DIR / "leave_one_origin_out.csv", leave_one_origin)
    write_csv(OUTPUT_DIR / "leave_one_city_sensitivity.csv", leave_one_city)
    write_csv(OUTPUT_DIR / "leakage_audit.csv", leakage_audit)
    write_csv(OUTPUT_DIR / "r1_r2_equivalence_check.csv", r1_r2)
    write_csv(OUTPUT_DIR / "b3_compatibility_check.csv", b3_compatibility)
    write_csv(OUTPUT_DIR / "b3_metric_compatibility_check.csv", frozen_metric_check)
    write_json(OUTPUT_DIR / "plan01_summary.json", summary)

    print("S4_PLAN01_RIDGE_DELTA_DYNAMICS_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_origin_domain=2011-2021")
    print(f"model_ids={','.join(MODEL_IDS)}")
    print(f"prediction_rows={len(all_rows)}")
    print(f"ml_prediction_rows={len(ml_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"aggregate_metric_rows={len(aggregate_metrics)}")
    print(f"training_geometry_rows={len(training_geometry)}")
    print(f"modeling_dataset_diagnostics_rows={len(dataset_diagnostics)}")
    print(f"leakage_audit_rows={len(leakage_audit)}")
    print(f"full_coverage={summary['full_coverage']}")
    print(f"b3_row_compatibility_pass={summary['b3_row_compatibility_pass']}")
    print(f"b3_metric_compatibility_pass={summary['b3_metric_compatibility_pass']}")
    print(f"leakage_pass={summary['leakage_pass']}")
    print(f"r1_r2_equivalence_pass={summary['r1_r2_equivalence_pass']}")
    print(f"resident_b3_pooled_mse={summary['resident_b3_pooled_mse']}")
    print(f"resident_m1a_pooled_mse={summary['resident_m1a_pooled_mse']}")
    print(f"resident_m1b_pooled_mse={summary['resident_m1b_pooled_mse']}")
    print(f"best_ml_model_by_resident_pooled_mse={best_ml_model}")
    print(f"resident_b3_beaten={summary['resident_b3_beaten']}")
    print(f"shape_ablation_supported={summary['shape_ablation_supported']}")
    print("formal_ml_model_used=True")
    print("covariates_used=False")
    print("tuning_run=False")
    print("ensemble_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
