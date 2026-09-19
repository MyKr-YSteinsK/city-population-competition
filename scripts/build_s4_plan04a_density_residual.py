"""Build the S4-Plan04A density residual-correction screening evidence.

This runner keeps the frozen B3 recent-three-point population trend as the
prediction backbone and fits one fixed Ridge(alpha=1.0) correction using only
the canonical population-density source.  R1 and R2 are constructed
independently from the S2 source-specific availability boundary.  Density
dynamics that cross a documented city regime shift are explicit missing
values; the persistent post-shift level is retained as a level feature.

The runner is screening-only.  It does not tune, sweep model families, add
other sources, use city identity, run robustness suites, create a 2023
prediction, or create a submission.
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
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan04A"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
REGIME_REPORT_PATH = ROOT / "data" / "normalized" / "人口密度_regime_report.csv"

STARTING_COMMIT = "b8c302abe7ebf1710869aaa55a00d4b926d2bcac"
BRANCH = "s4/wave-a-density"
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

DENSITY_SOURCE_ID = "density"
DENSITY_VALUE_COLUMN = "人口密度（人/平方公里）"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M4A_ID = "M4A_B3_PLUS_RIDGE_DENSITY"
MODEL_IDS = (B3_ID, M4A_ID)
DENSITY_FEATURES = (
    "density_latest",
    "density_recent_relative_change",
    "density_recent3_relative_slope",
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
        "density_used": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M4A_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_names": ";".join(DENSITY_FEATURES),
        "raw_feature_count": len(DENSITY_FEATURES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "model_parameters": json.dumps({"alpha": 1.0}, sort_keys=True),
        "uses_covariates": "yes",
        "density_used": "yes",
        "city_identity_used": "no",
        "tuning": "no",
    },
)

FEATURE_MANIFEST = (
    {
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_name": "density_latest",
        "formula": "latest non-missing density with actual year <= pseudo-target year minus scenario lag",
        "source": "data/normalized/人口密度.csv",
        "semantic_role": "recent density level",
        "model_feature": "yes",
        "used_by": M4A_ID,
        "year_rule": "S2 source-specific R1/R2 cap; no future fill or interpolation",
        "missing_policy": "explicit missing; training-fold median imputation only",
    },
    {
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_name": "density_recent_relative_change",
        "formula": "((latest density - previous density) / previous density) / actual year gap",
        "source": "data/normalized/人口密度.csv",
        "semantic_role": "annualized recent relative density change",
        "model_feature": "yes",
        "used_by": M4A_ID,
        "year_rule": "two latest allowed observations; actual year difference",
        "missing_policy": "explicit missing if fewer than two observations, zero denominator, or regime crossing",
    },
    {
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_name": "density_recent3_relative_slope",
        "formula": "OLS slope on latest three allowed densities divided by latest density",
        "source": "data/normalized/人口密度.csv",
        "semantic_role": "annualized recent relative density trend",
        "model_feature": "yes",
        "used_by": M4A_ID,
        "year_rule": "three latest allowed observations; actual year differences",
        "missing_policy": "explicit missing if fewer than three observations, zero denominator, or regime crossing",
    },
    {
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_name": "density_latest_year",
        "formula": "actual year of density_latest",
        "source": "data/normalized/人口密度.csv",
        "semantic_role": "freshness audit field",
        "model_feature": "no",
        "used_by": "audit_only",
        "year_rule": "scenario-specific source cap",
        "missing_policy": "blank when source has no value as of cap",
    },
    {
        "feature_group_id": "DENSITY_RECENT_V1",
        "feature_name": "density_regime_audit",
        "formula": "documented city regime shift and dynamic-window crossing diagnostics",
        "source": "data/normalized/人口密度_regime_report.csv",
        "semantic_role": "regime guard audit field",
        "model_feature": "no",
        "used_by": "audit_only",
        "year_rule": "crossing means min(window year) < shift year <= max(window year)",
        "missing_policy": "dynamics explicit missing when crossing is detected",
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


def as_int(value: Any) -> int | None:
    return int(float(value)) if present(value) else None


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
    s3_plan01.assert_static_boundaries()
    population, population_profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    scenario_definitions = s3_plan01.load_scenarios()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False

    contract = s2_geometry.load_contract()
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    density_source = sources[DENSITY_SOURCE_ID]
    geometry = {
        row["source_id"]: row
        for row in s2_geometry.source_geometry_rows(sources)
    }
    density_geometry = geometry[DENSITY_SOURCE_ID]
    assert density_geometry["latest_usable_year"] == 2022
    assert density_geometry["observed_snapshot_freshness_lag_years"] == 1
    assert density_geometry["source_status"] == "source_plus_repair_and_invalidation"
    assert density_geometry["source_invalidated_quarantine_count"] == 13
    assert density_geometry["source_derived_count"] == 0

    density_series: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    density_key_rows = {}
    value_column = density_source["value_columns"][0]
    for row in density_source["rows"]:
        year = as_int(row.get(density_source["year_col"]))
        assert year is not None
        density_key_rows[(row[density_source["city_col"]], year)] = row
    for row in density_source["value_rows"]:
        city = row[density_source["city_col"]]
        year = as_int(row.get(density_source["year_col"]))
        value = as_float(row.get(value_column))
        assert year is not None and value is not None
        density_series.setdefault(city, {})[year] = value

    quarantine_keys = set(density_source["quarantine_keys"])
    quarantine_nonempty_keys = {
        key
        for key in quarantine_keys
        if any(present(density_key_rows[key].get(column)) for column in density_source["value_columns"])
    }
    assert not quarantine_nonempty_keys
    assert not (
        quarantine_keys
        & {
            (city, year)
            for city, series in density_series.items()
            for year in series
        }
    )

    regime_rows = read_csv(REGIME_REPORT_PATH)
    assert len(regime_rows) == 13
    regime_shift_year = {
        row["城市名称"]: int(row["regime_shift_year"])
        for row in regime_rows
        if row["status"] == "retained"
    }
    assert set(regime_shift_year) == {
        "city2", "city5", "city8", "city12", "city13", "city15", "city18",
        "city22", "city26", "city30", "city34", "city37", "city40",
    }
    scenario_lags = {
        scenario_id: s2_geometry.required_lag_years(
            DENSITY_SOURCE_ID, density_geometry, scenario_id
        )
        for scenario_id in SCENARIOS
    }
    assert scenario_lags == {
        "R1_REFERENCE_SNAPSHOT_LAG": 1,
        "R2_CONSERVATIVE_PUBLICATION_STRESS": 2,
    }
    return {
        "population": population,
        "population_profile": population_profile,
        "s2_summary": s2_summary,
        "scenario_definitions": scenario_definitions,
        "density_source": density_source,
        "density_geometry": density_geometry,
        "density_series": density_series,
        "quarantine_keys": quarantine_keys,
        "regime_shift_year": regime_shift_year,
        "scenario_lags": scenario_lags,
    }


def regime_phase(years: list[int], shift_year: int | None) -> str:
    if not years or shift_year is None:
        return "not_applicable" if shift_year is None else "no_window"
    if max(years) < shift_year:
        return "pre"
    if min(years) >= shift_year:
        return "post"
    return "crossing"


def crosses_regime(years: list[int], shift_year: int | None) -> bool:
    return bool(
        years
        and shift_year is not None
        and min(years) < shift_year <= max(years)
    )


def build_density_features(
    series: dict[int, float],
    city: str,
    pseudo_target_year: int,
    required_lag: int,
    quarantine_keys: set[tuple[str, int]],
    regime_shift_year: dict[str, int],
) -> dict[str, Any]:
    cap = pseudo_target_year - required_lag
    allowed_years = sorted(year for year in series if year <= cap)
    latest_year = allowed_years[-1] if allowed_years else None
    latest_value = series[latest_year] if latest_year is not None else None
    recent_change_years = allowed_years[-2:] if len(allowed_years) >= 2 else []
    recent3_years = allowed_years[-3:] if len(allowed_years) >= 3 else []
    shift_year = regime_shift_year.get(city)
    recent_change_crossing = crosses_regime(recent_change_years, shift_year)
    recent3_crossing = crosses_regime(recent3_years, shift_year)

    recent_change = None
    recent_change_missing_reason = ""
    if len(recent_change_years) < 2:
        recent_change_missing_reason = "insufficient_two_observations"
    elif recent_change_crossing:
        recent_change_missing_reason = "documented_regime_crossing"
    else:
        previous_year, latest_change_year = recent_change_years
        previous_value = series[previous_year]
        gap = latest_change_year - previous_year
        if previous_value == 0:
            recent_change_missing_reason = "zero_previous_density"
        else:
            recent_change = (series[latest_change_year] - previous_value) / previous_value / gap

    recent3_relative_slope = None
    recent3_missing_reason = ""
    if len(recent3_years) < 3:
        recent3_missing_reason = "insufficient_three_observations"
    elif recent3_crossing:
        recent3_missing_reason = "documented_regime_crossing"
    elif latest_value in (None, 0):
        recent3_missing_reason = "zero_latest_density"
    else:
        recent3_relative_slope = ols_slope(
            recent3_years,
            [series[year] for year in recent3_years],
        ) / latest_value

    used_years = sorted(
        set(
            ([latest_year] if latest_year is not None else [])
            + recent_change_years
            + recent3_years
        )
    )
    quarantine_hits = sorted(
        (city, year) for year in used_years if (city, year) in quarantine_keys
    )
    suppressed = int(recent_change_crossing) + int(recent3_crossing)
    missing_names = [
        name
        for name, value in (
            ("density_latest", latest_value),
            ("density_recent_relative_change", recent_change),
            ("density_recent3_relative_slope", recent3_relative_slope),
        )
        if value is None
    ]
    pre_regime_years = [year for year in used_years if shift_year is not None and year < shift_year]
    post_regime_years = [year for year in used_years if shift_year is not None and year >= shift_year]
    return {
        "density_required_lag_years": required_lag,
        "density_required_latest_usable_year": cap,
        "density_allowed_years": ";".join(str(year) for year in allowed_years),
        "density_latest_year": latest_year,
        "density_latest": latest_value,
        "density_staleness_years": pseudo_target_year - latest_year if latest_year is not None else None,
        "density_recent_relative_change": recent_change,
        "density_recent_change_source_years": ";".join(str(year) for year in recent_change_years),
        "density_recent_change_regime_phase": regime_phase(recent_change_years, shift_year),
        "density_recent_change_missing_reason": recent_change_missing_reason,
        "density_recent3_relative_slope": recent3_relative_slope,
        "density_recent3_slope_source_years": ";".join(str(year) for year in recent3_years),
        "density_recent3_slope_regime_phase": regime_phase(recent3_years, shift_year),
        "density_recent3_slope_missing_reason": recent3_missing_reason,
        "density_used_years": ";".join(str(year) for year in used_years),
        "density_used_year_max": max(used_years) if used_years else None,
        "density_regime_shift_year": shift_year,
        "density_regime_city_flag": "yes" if shift_year is not None else "no",
        "density_regime_window_phase": regime_phase(used_years, shift_year),
        "density_pre_regime_years": ";".join(str(year) for year in pre_regime_years),
        "density_post_regime_years": ";".join(str(year) for year in post_regime_years),
        "density_suppressed_dynamics_count": suppressed,
        "density_quarantine_candidate_hit_count": len(quarantine_hits),
        "density_quarantine_candidate_hit_years": ";".join(str(year) for _, year in quarantine_hits),
        "density_missing_feature_names": ";".join(missing_names),
    }


def b3_fit(series: dict[int, float], target_year: int) -> dict[str, Any]:
    return s3_plan02.fit_prediction(
        B3_ID,
        series,
        target_year,
        target_year - HORIZON_YEARS,
    )


def build_training_rows(
    target_series: dict[int, float],
    density_series: dict[int, float],
    city: str,
    outer_origin: int,
    required_lag: int,
    quarantine_keys: set[tuple[str, int]],
    regime_shift_year: dict[str, int],
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
        density = build_density_features(
            density_series,
            city,
            row_year,
            required_lag,
            quarantine_keys,
            regime_shift_year,
        )
        rows.append(
            {
                "row_role": "training",
                "city_id": city,
                "row_year": row_year,
                "target_value": actual,
                "b3_prediction": fit["prediction"],
                "b3_residual_actual_minus_prediction": actual - fit["prediction"],
                "b3_fit": fit,
                "density": density,
                "outer_origin_year": outer_origin,
            }
        )
    return rows, excluded


def build_validation_row(
    target_series: dict[int, float],
    density_series: dict[int, float],
    city: str,
    origin: int,
    required_lag: int,
    quarantine_keys: set[tuple[str, int]],
    regime_shift_year: dict[str, int],
) -> dict[str, Any]:
    fit = b3_fit(target_series, origin)
    actual = target_series.get(origin)
    assert actual is not None and fit["prediction"] is not None
    return {
        "row_role": "validation",
        "city_id": city,
        "row_year": origin,
        "target_value": actual,
        "b3_prediction": fit["prediction"],
        "b3_residual_actual_minus_prediction": actual - fit["prediction"],
        "b3_fit": fit,
        "density": build_density_features(
            density_series,
            city,
            origin,
            required_lag,
            quarantine_keys,
            regime_shift_year,
        ),
        "outer_origin_year": origin,
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
                as_float(row["density"].get(name))
                if as_float(row["density"].get(name)) is not None
                else np.nan
                for name in DENSITY_FEATURES
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
    model = pipeline_for()
    model.fit(x_train, y_train)
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    indicator_names = [
        f"missing_indicator__{DENSITY_FEATURES[index]}"
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
        "imputer_statistics": [
            float(value) if math.isfinite(float(value)) else None
            for value in imputer.statistics_
        ],
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": [float(value) for value in np.asarray(ridge.coef_).ravel()],
        "coefficient_names": list(DENSITY_FEATURES) + indicator_names,
        "feature_names": ";".join(DENSITY_FEATURES),
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
    b3_prediction = validation["b3_prediction"]
    residual = prediction - actual
    density = validation["density"]
    return {
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
        "history_observation_count": validation["b3_fit"]["history_count"],
        "history_years_used": ";".join(str(year) for year in validation["b3_fit"]["history_years"]),
        "history_min_year": validation["b3_fit"]["history_min_year"],
        "history_max_year": validation["b3_fit"]["history_max_year"],
        "latest_target_year_used": validation["b3_fit"]["latest_target_year_used"],
        "previous_target_year_used": validation["b3_fit"]["previous_target_year_used"],
        "b3_prediction_value": b3_prediction,
        "b3_residual_actual_minus_prediction": validation["b3_residual_actual_minus_prediction"],
        "predicted_b3_residual": predicted_residual,
        "feature_builder_id": "FROZEN_B3_REFERENCE" if model_id == B3_ID else "DENSITY_RECENT_V1",
        "feature_group_id": feature_group_id,
        "density_required_lag_years": density["density_required_lag_years"],
        "density_required_latest_usable_year": density["density_required_latest_usable_year"],
        "density_allowed_years": density["density_allowed_years"],
        "density_latest_year": density["density_latest_year"],
        "density_latest": density["density_latest"],
        "density_staleness_years": density["density_staleness_years"],
        "density_recent_relative_change": density["density_recent_relative_change"],
        "density_recent_change_source_years": density["density_recent_change_source_years"],
        "density_recent_change_regime_phase": density["density_recent_change_regime_phase"],
        "density_recent3_relative_slope": density["density_recent3_relative_slope"],
        "density_recent3_slope_source_years": density["density_recent3_slope_source_years"],
        "density_recent3_slope_regime_phase": density["density_recent3_slope_regime_phase"],
        "density_used_years": density["density_used_years"],
        "density_used_year_max": density["density_used_year_max"],
        "density_regime_shift_year": density["density_regime_shift_year"],
        "density_regime_city_flag": density["density_regime_city_flag"],
        "density_suppressed_dynamics_count": density["density_suppressed_dynamics_count"],
        "density_quarantine_candidate_hit_count": density["density_quarantine_candidate_hit_count"],
        "density_missing_feature_names": density["density_missing_feature_names"],
        "training_label_min_year": metadata.get("training_label_min_year", ""),
        "training_label_max_year": metadata.get("training_label_max_year", ""),
        "training_row_count": metadata.get("training_row_count", ""),
        "preprocessing_fit_row_count": metadata.get("preprocessing_fit_row_count", ""),
        "model_fit_row_count": metadata.get("model_fit_row_count", ""),
        "training_missing_feature_cell_count": metadata.get("training_missing_feature_cell_count", ""),
        "validation_missing_feature_cell_count": metadata.get("validation_missing_feature_cell_count", ""),
        "covariates_used": "no" if model_id == B3_ID else "yes",
        "density_used": "no" if model_id == B3_ID else "yes",
        "city_identity_used": "no",
    }


def make_b3_row(
    target: dict[str, str],
    scenario_id: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
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
        M4A_ID,
        validation,
        validation["b3_prediction"] + predicted_residual,
        predicted_residual,
        "DENSITY_RECENT_V1",
        metadata,
    )


def density_audit_row(
    target: dict[str, str],
    scenario_id: str,
    row: dict[str, Any],
    density_source: dict[str, Any],
) -> dict[str, Any]:
    fit = row["b3_fit"]
    density = row["density"]
    pseudo_year = int(row["row_year"])
    outer_origin = int(row["outer_origin_year"])
    history_max = fit["history_max_year"]
    density_max = density["density_used_year_max"]
    dynamics_guard_pass = (
        density["density_suppressed_dynamics_count"] == 0
        or (
            density["density_recent_change_regime_phase"] == "crossing"
            and density["density_recent_relative_change"] is None
        )
        or (
            density["density_recent3_slope_regime_phase"] == "crossing"
            and density["density_recent3_relative_slope"] is None
        )
    )
    checks = {
        "population_history_within_y_minus_2": history_max is None or history_max <= pseudo_year - HORIZON_YEARS,
        "density_years_within_scenario_cap": density_max is None or density_max <= density["density_required_latest_usable_year"],
        "outer_training_label_within_outer_origin_minus_2": (
            row["row_role"] != "training" or pseudo_year <= outer_origin - HORIZON_YEARS
        ),
        "quarantine_reintroduced": density["density_quarantine_candidate_hit_count"] == 0,
        "regime_crossing_guard": dynamics_guard_pass,
    }
    return {
        "target_id": target["target_id"],
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "row_role": row["row_role"],
        "city_id": row["city_id"],
        "pseudo_target_year": pseudo_year,
        "population_target_history_cutoff_year": pseudo_year - HORIZON_YEARS,
        "population_history_max_year_used": history_max or "",
        "population_history_years_used": ";".join(str(year) for year in fit["history_years"]),
        "density_required_lag_years": density["density_required_lag_years"],
        "density_required_latest_usable_year": density["density_required_latest_usable_year"],
        "density_allowed_years": density["density_allowed_years"],
        "density_latest_year_used": density["density_latest_year"] or "",
        "density_latest_value": density["density_latest"],
        "density_staleness_years": density["density_staleness_years"],
        "density_recent_change_source_years": density["density_recent_change_source_years"],
        "density_recent_change_regime_phase": density["density_recent_change_regime_phase"],
        "density_recent3_slope_source_years": density["density_recent3_slope_source_years"],
        "density_recent3_slope_regime_phase": density["density_recent3_slope_regime_phase"],
        "density_used_years": density["density_used_years"],
        "density_used_year_max": density_max or "",
        "density_regime_shift_year": density["density_regime_shift_year"] or "",
        "density_regime_city_flag": density["density_regime_city_flag"],
        "density_pre_regime_years": density["density_pre_regime_years"],
        "density_post_regime_years": density["density_post_regime_years"],
        "density_suppressed_dynamics_count": density["density_suppressed_dynamics_count"],
        "density_recent_change_missing_reason": density["density_recent_change_missing_reason"],
        "density_recent3_slope_missing_reason": density["density_recent3_slope_missing_reason"],
        "density_quarantine_candidate_hit_count": density["density_quarantine_candidate_hit_count"],
        "density_quarantine_candidate_hit_years": density["density_quarantine_candidate_hit_years"],
        "density_source_status": density_source["spec"]["source_status"],
        "quarantine_reintroduced": "no" if checks["quarantine_reintroduced"] else "yes",
        "quarantine_reintroduced_count": density["density_quarantine_candidate_hit_count"],
        "population_history_within_y_minus_2": "PASS" if checks["population_history_within_y_minus_2"] else "FAIL",
        "density_years_within_scenario_cap": "PASS" if checks["density_years_within_scenario_cap"] else "FAIL",
        "outer_training_label_within_outer_origin_minus_2": "PASS" if checks["outer_training_label_within_outer_origin_minus_2"] else "FAIL",
        "regime_crossing_guard": "PASS" if checks["regime_crossing_guard"] else "FAIL",
        "boundary_status": "PASS" if all(checks.values()) else "FAIL",
    }


def build_screening_rows(
    all_rows: list[dict[str, Any]],
    per_origin: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in aggregate
    }
    per_origin_map = {
        (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
            int(row["origin_year"]),
        ): row
        for row in per_origin
    }
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            base = aggregate_map[(target_id, scenario_id, B3_ID)]
            candidate = aggregate_map[(target_id, scenario_id, M4A_ID)]
            candidate_origin = [
                per_origin_map[(target_id, scenario_id, M4A_ID, origin)]
                for origin in COMMON_ORIGINS
            ]
            base_origin = {
                origin: per_origin_map[(target_id, scenario_id, B3_ID, origin)]
                for origin in COMMON_ORIGINS
            }
            wins = sum(
                float(row["mse"]) < float(base_origin[int(row["origin_year"])]["mse"]) - 1e-12
                for row in candidate_origin
            )
            losses = sum(
                float(row["mse"]) > float(base_origin[int(row["origin_year"])]["mse"]) + 1e-12
                for row in candidate_origin
            )
            ties = len(candidate_origin) - wins - losses
            r1 = aggregate_map[(target_id, "R1_REFERENCE_SNAPSHOT_LAG", M4A_ID)]
            r2 = aggregate_map[(target_id, "R2_CONSERVATIVE_PUBLICATION_STRESS", M4A_ID)]
            r1_base = aggregate_map[(target_id, "R1_REFERENCE_SNAPSHOT_LAG", B3_ID)]
            r2_base = aggregate_map[(target_id, "R2_CONSERVATIVE_PUBLICATION_STRESS", B3_ID)]
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
                    "pooled_city_year_mae": base["pooled_city_year_mae"],
                    "pooled_mean_signed_error_bias": base["pooled_mean_signed_error_bias"],
                    "wins_vs_b3": "",
                    "losses_vs_b3": "",
                    "ties_vs_b3": "",
                    "delta_vs_b3_pooled_mse": 0.0,
                    "delta_vs_b3_latest_2021_mse": 0.0,
                    "relative_improvement_vs_b3_pct": 0.0,
                    "r1_to_r2_pooled_mse_delta": float(r2_base["pooled_city_year_mse"]) - float(r1_base["pooled_city_year_mse"]),
                    "r1_to_r2_latest_2021_mse_delta": float(r2_base["latest_origin_mse"]) - float(r1_base["latest_origin_mse"]),
                    "comparison_vs_b3": "baseline",
                }
            )
            candidate_pooled = float(candidate["pooled_city_year_mse"])
            base_pooled = float(base["pooled_city_year_mse"])
            candidate_latest = float(candidate["latest_origin_mse"])
            base_latest = float(base["latest_origin_mse"])
            result.append(
                {
                    "metric_scope": "aggregate",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "predictor_id": M4A_ID,
                    "origin_year": "",
                    "pooled_city_year_mse": candidate_pooled,
                    "latest_2021_mse": candidate_latest,
                    "macro_origin_mse": candidate["macro_origin_mse"],
                    "pooled_city_year_mae": candidate["pooled_city_year_mae"],
                    "pooled_mean_signed_error_bias": candidate["pooled_mean_signed_error_bias"],
                    "wins_vs_b3": wins,
                    "losses_vs_b3": losses,
                    "ties_vs_b3": ties,
                    "delta_vs_b3_pooled_mse": candidate_pooled - base_pooled,
                    "delta_vs_b3_latest_2021_mse": candidate_latest - base_latest,
                    "relative_improvement_vs_b3_pct": (base_pooled - candidate_pooled) / base_pooled * 100,
                    "r1_to_r2_pooled_mse_delta": float(r2["pooled_city_year_mse"]) - float(r1["pooled_city_year_mse"]),
                    "r1_to_r2_latest_2021_mse_delta": float(r2["latest_origin_mse"]) - float(r1["latest_origin_mse"]),
                    "comparison_vs_b3": "better" if candidate_pooled < base_pooled - 1e-12 else "worse" if candidate_pooled > base_pooled + 1e-12 else "tie",
                }
            )
            for origin in COMMON_ORIGINS:
                candidate_origin_row = per_origin_map[(target_id, scenario_id, M4A_ID, origin)]
                base_origin_row = per_origin_map[(target_id, scenario_id, B3_ID, origin)]
                candidate_mse = float(candidate_origin_row["mse"])
                base_mse = float(base_origin_row["mse"])
                result.append(
                    {
                        "metric_scope": "per_origin",
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "predictor_id": M4A_ID,
                        "origin_year": origin,
                        "origin_mse": candidate_mse,
                        "b3_reference_origin_mse": base_mse,
                        "delta_vs_b3_origin_mse": candidate_mse - base_mse,
                        "comparison_vs_b3": "better" if candidate_mse < base_mse - 1e-12 else "worse" if candidate_mse > base_mse + 1e-12 else "tie",
                    }
                )
    assert len(result) == len(TARGETS) * len(SCENARIOS) * 2 + len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS)
    return result


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
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    result = []
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
            deltas = {
                field: abs(float(current[field]) - float(old[field]))
                for field in fields
            }
            passed = all(delta <= 1e-9 for delta in deltas.values())
            result.append(
                {
                    "target_id": target["target_id"],
                    "scenario_id": scenario_id,
                    "status": "PASS" if passed else "FAIL",
                    "max_abs_metric_delta": max(deltas.values()),
                    "comparison_rule": "B3 pooled/macro/latest metric delta <= 1e-9",
                }
            )
    assert all(row["status"] == "PASS" for row in result)
    return result


def decision_summary(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    per_origin: list[dict[str, Any]],
) -> dict[str, Any]:
    per_origin_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"])): row
        for row in per_origin
    }
    metric_summary: dict[str, Any] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            base = aggregate_map[(target_id, scenario_id, B3_ID)]
            candidate = aggregate_map[(target_id, scenario_id, M4A_ID)]
            wins = sum(
                float(per_origin_map[(target_id, scenario_id, M4A_ID, origin)]["mse"])
                < float(per_origin_map[(target_id, scenario_id, B3_ID, origin)]["mse"]) - 1e-12
                for origin in COMMON_ORIGINS
            )
            losses = sum(
                float(per_origin_map[(target_id, scenario_id, M4A_ID, origin)]["mse"])
                > float(per_origin_map[(target_id, scenario_id, B3_ID, origin)]["mse"]) + 1e-12
                for origin in COMMON_ORIGINS
            )
            metric_summary[f"{target_id}_{scenario_id}_{B3_ID}"] = {
                "pooled_mse": float(base["pooled_city_year_mse"]),
                "latest_2021_mse": float(base["latest_origin_mse"]),
            }
            metric_summary[f"{target_id}_{scenario_id}_{M4A_ID}"] = {
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
            }

    resident_r1_base = aggregate_map[("resident", SCENARIOS[0], B3_ID)]
    resident_r2_base = aggregate_map[("resident", SCENARIOS[1], B3_ID)]
    resident_r1_candidate = aggregate_map[("resident", SCENARIOS[0], M4A_ID)]
    resident_r2_candidate = aggregate_map[("resident", SCENARIOS[1], M4A_ID)]
    resident_r1_improvement = (
        float(resident_r1_base["pooled_city_year_mse"]) - float(resident_r1_candidate["pooled_city_year_mse"])
    ) / float(resident_r1_base["pooled_city_year_mse"]) * 100
    resident_r2_improvement = (
        float(resident_r2_base["pooled_city_year_mse"]) - float(resident_r2_candidate["pooled_city_year_mse"])
    ) / float(resident_r2_base["pooled_city_year_mse"]) * 100
    resident_r1_row = metric_summary[f"resident_{SCENARIOS[0]}_{M4A_ID}"]
    resident_r2_row = metric_summary[f"resident_{SCENARIOS[1]}_{M4A_ID}"]
    r1_beats = resident_r1_improvement > 0
    r2_beats = resident_r2_improvement > 0
    majority_both = resident_r1_row["wins_vs_b3"] > len(COMMON_ORIGINS) / 2 and resident_r2_row["wins_vs_b3"] > len(COMMON_ORIGINS) / 2
    within_one_pct_and_broad = (
        max(resident_r1_improvement, resident_r2_improvement) >= -1.0
        and min(resident_r1_improvement, resident_r2_improvement) > 0
        and majority_both
        and float(resident_r1_candidate["latest_origin_mse"]) < float(resident_r1_base["latest_origin_mse"])
        and float(resident_r2_candidate["latest_origin_mse"]) < float(resident_r2_base["latest_origin_mse"])
    )
    r1_clear_gain = resident_r1_improvement >= 1.0 and float(resident_r1_candidate["latest_origin_mse"]) < float(resident_r1_base["latest_origin_mse"])
    deep_dive = bool((r1_beats or r2_beats) or within_one_pct_and_broad or r1_clear_gain)
    if r1_beats and r2_beats and majority_both:
        decision = "supported"
        rationale = "resident pooled MSE beats B3 in both R1 and R2 with majority per-origin wins"
    elif deep_dive:
        decision = "conditional_freshness_or_regime_sensitive"
        rationale = "screening shows a candidate gain or a bounded gain pattern that warrants a separate freshness/regime investigation"
    else:
        decision = "unsupported"
        rationale = "resident pooled MSE does not show a sufficiently stable gain over B3 under the frozen R1/R2 screen"

    registered_support = all(
        float(aggregate_map[("registered", scenario_id, M4A_ID)]["pooled_city_year_mse"])
        < float(aggregate_map[("registered", scenario_id, B3_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    return {
        "screening_decision": decision,
        "screening_rationale": rationale,
        "deep_dive_recommended": deep_dive,
        "deep_dive_rule": {
            "resident_pooled_beats_b3": "any R1/R2 strict improvement",
            "near_b3_condition": "within 1% with majority per-origin wins and latest-origin improvement in both scenarios",
            "r1_clear_gain_condition": "R1 pooled improvement >= 1% and latest-origin improvement",
        },
        "registered_supports_candidate_in_both_scenarios": registered_support,
        "resident_r1_improvement_pct": resident_r1_improvement,
        "resident_r2_improvement_pct": resident_r2_improvement,
        "resident_r1_to_r2_pooled_delta": float(resident_r2_candidate["pooled_city_year_mse"]) - float(resident_r1_candidate["pooled_city_year_mse"]),
        "resident_r1_to_r2_latest_delta": float(resident_r2_candidate["latest_origin_mse"]) - float(resident_r1_candidate["latest_origin_mse"]),
        "metrics": metric_summary,
    }


def write_handoff(summary: dict[str, Any]) -> None:
    decision = summary["screening_decision"]
    density = summary["density_boundary"]
    metrics = summary["metrics"]
    resident_r1 = metrics[f"resident_{SCENARIOS[0]}_{M4A_ID}"]
    resident_r2 = metrics[f"resident_{SCENARIOS[1]}_{M4A_ID}"]
    registered_r1 = metrics[f"registered_{SCENARIOS[0]}_{M4A_ID}"]
    registered_r2 = metrics[f"registered_{SCENARIOS[1]}_{M4A_ID}"]
    lines = [
        "# S4-Plan04A Handoff — 人口密度近期信号与 B3 残差修正快速筛选",
        "",
        "## Result",
        f"- Status: Complete（screening 已完成；不代表 S4 complete）",
        f"- Branch: `{summary['branch']}`",
        f"- Frozen base: `{summary['starting_commit']}`",
        f"- Decision: `{decision}`",
        f"- `deep_dive_recommended`: `{str(summary['deep_dive_recommended']).lower()}`",
        "",
        "## Frozen scope",
        "- Candidate: `M4A_B3_PLUS_RIDGE_DENSITY` = frozen B3 + Ridge(alpha=1.0) residual correction.",
        "- Features: `density_latest`, annualized `density_recent_relative_change`, and normalized `density_recent3_relative_slope`.",
        "- Resident / registered are evaluated separately; only common origins 2011–2021 are scored.",
        "- No tuning, GBDT, city ID, other source, 2023 prediction, submission, or full robustness suite was run.",
        "",
        "## Screening answers",
        "1. **M4A vs B3**",
        f"   - resident R1 pooled MSE: `{resident_r1['pooled_mse']:.12f}` vs B3 `{metrics[f'resident_{SCENARIOS[0]}_{B3_ID}']['pooled_mse']:.12f}`; latest 2021 `{resident_r1['latest_2021_mse']:.12f}`.",
        f"   - resident R2 pooled MSE: `{resident_r2['pooled_mse']:.12f}` vs B3 `{metrics[f'resident_{SCENARIOS[1]}_{B3_ID}']['pooled_mse']:.12f}`; latest 2021 `{resident_r2['latest_2021_mse']:.12f}`.",
        "2. **Registered sensitivity**",
        f"   - R1 candidate pooled MSE `{registered_r1['pooled_mse']:.12f}` vs B3 `{metrics[f'registered_{SCENARIOS[0]}_{B3_ID}']['pooled_mse']:.12f}`.",
        f"   - R2 candidate pooled MSE `{registered_r2['pooled_mse']:.12f}` vs B3 `{metrics[f'registered_{SCENARIOS[1]}_{B3_ID}']['pooled_mse']:.12f}`.",
        f"   - Supports candidate in both scenarios: `{str(summary['registered_supports_candidate_in_both_scenarios']).lower()}`.",
        "3. **Per-origin wins / losses vs B3**",
        f"   - resident R1: `{resident_r1['wins_vs_b3']}` wins / `{resident_r1['losses_vs_b3']}` losses / `{resident_r1['ties_vs_b3']}` ties.",
        f"   - resident R2: `{resident_r2['wins_vs_b3']}` wins / `{resident_r2['losses_vs_b3']}` losses / `{resident_r2['ties_vs_b3']}` ties.",
        f"   - registered R1: `{registered_r1['wins_vs_b3']}` wins / `{registered_r1['losses_vs_b3']}` losses / `{registered_r1['ties_vs_b3']}` ties.",
        f"   - registered R2: `{registered_r2['wins_vs_b3']}` wins / `{registered_r2['losses_vs_b3']}` losses / `{registered_r2['ties_vs_b3']}` ties.",
        "4. **R1 → R2**",
        f"   - resident candidate pooled delta R2−R1: `{summary['resident_r1_to_r2_pooled_delta']:.12f}`; latest delta: `{summary['resident_r1_to_r2_latest_delta']:.12f}`.",
        f"   - resident relative improvement vs B3: R1 `{summary['resident_r1_improvement_pct']:.6f}%`, R2 `{summary['resident_r2_improvement_pct']:.6f}%`.",
        "5. **Regime suppression**",
        f"   - audit feature rows: `{density['audit_rows']}`; rows with at least one suppressed dynamic: `{density['rows_with_suppressed_dynamics']}`; suppressed dynamic feature cells: `{density['suppressed_dynamic_feature_cells']}`.",
        "6. **Quarantine guard**",
        f"   - candidate-hit count: `{density['quarantine_candidate_hit_count']}`; reintroduction count: `{density['quarantine_reintroduction_count']}`; guard: `PASS`.",
        "7. **Deep dive**",
        f"   - `{str(summary['deep_dive_recommended']).lower()}` — {summary['screening_rationale']}.",
        "",
        "## Evidence",
        "- `plan04a_summary.json`",
        "- `screening_metrics.csv`",
        "- `density_boundary_audit.csv`",
        "- `scripts/build_s4_plan04a_density_residual.py`",
        "",
        "## Project state",
        "- `s4_complete=false`; `s5_entered=false`.",
        "- Shared checkpoint, project context, file map, S2 contract, and prior S3/S4 handoffs were not modified.",
    ]
    (OUTPUT_DIR / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert STARTING_COMMIT == "b8c302abe7ebf1710869aaa55a00d4b926d2bcac"
    context = load_context()
    population = context["population"]
    density_series = context["density_series"]
    density_source = context["density_source"]
    quarantine_keys = context["quarantine_keys"]
    regime_shift_year = context["regime_shift_year"]
    scenario_lags = context["scenario_lags"]

    all_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    training_excluded_count = 0
    model_fit_count = 0
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            required_lag = scenario_lags[scenario_id]
            for origin in COMMON_ORIGINS:
                training_rows: list[dict[str, Any]] = []
                validation_rows: list[dict[str, Any]] = []
                for city in CITIES:
                    target_series = population[target_id][city]
                    city_training, excluded = build_training_rows(
                        target_series,
                        density_series[city],
                        city,
                        origin,
                        required_lag,
                        quarantine_keys,
                        regime_shift_year,
                    )
                    training_rows.extend(city_training)
                    training_excluded_count += excluded
                    validation = build_validation_row(
                        target_series,
                        density_series[city],
                        city,
                        origin,
                        required_lag,
                        quarantine_keys,
                        regime_shift_year,
                    )
                    validation_rows.append(validation)
                    audit_rows.append(density_audit_row(target, scenario_id, validation, density_source))
                    for row in city_training:
                        audit_rows.append(density_audit_row(target, scenario_id, row, density_source))
                predicted_residuals, metadata = fit_residual_model(training_rows, validation_rows)
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

    assert len(all_rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * 2
    assert len(audit_rows) > len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert model_fit_count == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS)
    assert all(row["density_used"] in {"no", "yes"} for row in all_rows)
    assert all(row["density_quarantine_candidate_hit_count"] == 0 for row in all_rows)
    assert all(row["boundary_status"] == "PASS" for row in audit_rows)

    b3_rows = [row for row in all_rows if row["predictor_id"] == B3_ID]
    b3_compatibility = compare_b3_rows(b3_rows)
    per_origin, aggregate, aggregate_map = calculate_metrics(all_rows)
    b3_metric_compatibility = compare_b3_metrics(aggregate_map)
    screening_rows = build_screening_rows(all_rows, per_origin, aggregate)
    decision = decision_summary(aggregate_map, per_origin)

    suppressed_cells = sum(int(row["density_suppressed_dynamics_count"]) for row in audit_rows)
    rows_with_suppression = sum(
        int(row["density_suppressed_dynamics_count"]) > 0 for row in audit_rows
    )
    suppression_by_role = Counter(
        row["row_role"]
        for row in audit_rows
        if int(row["density_suppressed_dynamics_count"]) > 0
    )
    r1_feature_signatures = {
        (
            row["target_id"], row["origin_year"], row["city_id"],
            row["density_allowed_years"], row["density_latest_year"],
            row["density_recent_relative_change"], row["density_recent3_relative_slope"],
        )
        for row in all_rows
        if row["predictor_id"] == M4A_ID and row["scenario_id"] == SCENARIOS[0]
    }
    r2_feature_signatures = {
        (
            row["target_id"], row["origin_year"], row["city_id"],
            row["density_allowed_years"], row["density_latest_year"],
            row["density_recent_relative_change"], row["density_recent3_relative_slope"],
        )
        for row in all_rows
        if row["predictor_id"] == M4A_ID and row["scenario_id"] == SCENARIOS[1]
    }
    assert r1_feature_signatures != r2_feature_signatures
    quarantine_reintroduction_count = sum(
        int(row["density_quarantine_candidate_hit_count"]) for row in audit_rows
    )
    assert quarantine_reintroduction_count == 0
    assert all(row["status"] == "PASS" for row in b3_compatibility)
    assert all(row["status"] == "PASS" for row in b3_metric_compatibility)

    summary = {
        "plan_id": "S4-Plan04A",
        "stage": "S4_FEATURE_ML_MODELING",
        "branch": BRANCH,
        "starting_commit": STARTING_COMMIT,
        "common_origin_domain": "2011-2021",
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_cutoff_rule": "population B3 history <= pseudo-target year minus 2; validation target history <= origin minus 2",
        "strongest_baseline": B3_ID,
        "model_ids": list(MODEL_IDS),
        "candidate_id": M4A_ID,
        "density_feature_manifest": list(FEATURE_MANIFEST),
        "density_source": {
            "canonical": "data/normalized/人口密度.csv",
            "source_status": density_source["spec"]["source_status"],
            "canonical_value_year_min": context["density_geometry"]["canonical_value_year_min"],
            "canonical_value_year_max": context["density_geometry"]["canonical_value_year_max"],
            "latest_usable_year": context["density_geometry"]["latest_usable_year"],
            "observed_snapshot_freshness_lag_years": context["density_geometry"]["observed_snapshot_freshness_lag_years"],
            "quarantine_count": len(quarantine_keys),
            "regime_city_count": len(regime_shift_year),
        },
        "scenario_lags": scenario_lags,
        "r1_r2_feature_difference_confirmed": True,
        "b3_compatibility_pass": all(row["status"] == "PASS" for row in b3_compatibility),
        "b3_metric_compatibility_pass": all(row["status"] == "PASS" for row in b3_metric_compatibility),
        "b3_compatibility_rows": b3_compatibility,
        "b3_metric_compatibility_rows": b3_metric_compatibility,
        "density_boundary": {
            "audit_rows": len(audit_rows),
            "rows_with_suppressed_dynamics": rows_with_suppression,
            "suppressed_dynamic_feature_cells": suppressed_cells,
            "suppressed_rows_by_role": dict(sorted(suppression_by_role.items())),
            "quarantine_candidate_hit_count": quarantine_reintroduction_count,
            "quarantine_reintroduction_count": quarantine_reintroduction_count,
            "quarantine_guard_pass": quarantine_reintroduction_count == 0,
            "regime_crossing_guard_pass": all(row["regime_crossing_guard"] == "PASS" for row in audit_rows),
            "all_boundary_rows_pass": all(row["boundary_status"] == "PASS" for row in audit_rows),
        },
        "training_excluded_b3_insufficient_history_rows": training_excluded_count,
        "outer_origin_model_fit_count": model_fit_count,
        "metrics": decision["metrics"],
        "screening_decision": decision["screening_decision"],
        "screening_rationale": decision["screening_rationale"],
        "deep_dive_recommended": decision["deep_dive_recommended"],
        "deep_dive_rule": decision["deep_dive_rule"],
        "registered_supports_candidate_in_both_scenarios": decision["registered_supports_candidate_in_both_scenarios"],
        "resident_r1_improvement_pct": decision["resident_r1_improvement_pct"],
        "resident_r2_improvement_pct": decision["resident_r2_improvement_pct"],
        "resident_r1_to_r2_pooled_delta": decision["resident_r1_to_r2_pooled_delta"],
        "resident_r1_to_r2_latest_delta": decision["resident_r1_to_r2_latest_delta"],
        "no_other_sources_used": True,
        "city_identity_used": False,
        "tuning_run": False,
        "robustness_suite_run": False,
        "final_2023_prediction_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
    }

    write_csv(OUTPUT_DIR / "screening_metrics.csv", screening_rows, fields=union_fields(screening_rows))
    write_csv(OUTPUT_DIR / "density_boundary_audit.csv", audit_rows, fields=union_fields(audit_rows))
    write_json(OUTPUT_DIR / "plan04a_summary.json", summary)
    write_handoff(summary)

    print("S4_PLAN04A_SCREENING_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"model_fit_count={model_fit_count}")
    print(f"b3_compatibility_pass={summary['b3_compatibility_pass']}")
    print(f"b3_metric_compatibility_pass={summary['b3_metric_compatibility_pass']}")
    print(f"r1_r2_feature_difference_confirmed={summary['r1_r2_feature_difference_confirmed']}")
    print(f"quarantine_reintroduction_count={quarantine_reintroduction_count}")
    print(f"regime_suppressed_dynamic_feature_cells={suppressed_cells}")
    print(f"screening_decision={summary['screening_decision']}")
    print(f"deep_dive_recommended={summary['deep_dive_recommended']}")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
