"""Build S4-Plan04C employment-scale screening evidence.

This runner tests one pre-registered employment-scale residual correction over
the frozen B3 population trend.  Employment features are reconstructed
independently for each pseudo-target under the S2 R1/R2 source boundary.  The
runner keeps source-derived provenance visible, never reads quarantined values
into the feature map, and intentionally emits screening evidence only.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan04C"
S2_PLAN01_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan01"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
EMPLOYMENT_PATH = ROOT / "data" / "normalized" / "就业信息_从业人员数.csv"
EMPLOYMENT_PROVENANCE_PATH = ROOT / "data" / "normalized" / "就业信息_从业人员数_provenance.csv"
EMPLOYMENT_QUARANTINE_PATH = ROOT / "data" / "quarantine" / "就业信息_从业人员数_quarantine.csv"
S2_POLICY_PATH = S2_PLAN01_DIR / "pseudo_target_source_policy_matrix.csv"

PLAN_ID = "S4-Plan04C"
STARTING_COMMIT = "b8c302abe7ebf1710869aaa55a00d4b926d2bcac"
BRANCH = "s4/wave-a-employment"
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS

B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M4C_ID = "M4C_B3_PLUS_RIDGE_EMPLOYMENT"
MODEL_IDS = (B3_ID, M4C_ID)
FEATURE_NAMES = (
    "employees_log_latest",
    "employees_recent_log_growth",
    "employees_recent3_log_slope",
)
SCENARIO_LAG_YEARS = {
    "R1_REFERENCE_SNAPSHOT_LAG": 1,
    "R2_CONSERVATIVE_PUBLICATION_STRESS": 2,
}
POLICY_BY_SCENARIO = {
    "R1_REFERENCE_SNAPSHOT_LAG": "P1_ASOF_T_MINUS_1",
    "R2_CONSERVATIVE_PUBLICATION_STRESS": "P2_ASOF_T_MINUS_2",
}

MODEL_DEFINITIONS = (
    {
        "predictor_id": B3_ID,
        "model_family": "frozen_s3_reference",
        "feature_group_id": "FROZEN_B3",
        "feature_names": "",
        "target_definition": "frozen B3 recent-three-point population trend",
        "preprocessing": "not_applicable_frozen_reference",
        "estimator": "frozen_B3",
        "ridge_alpha": "",
        "uses_covariates": "no",
        "city_identity_used": "no",
        "tuning": "no",
    },
    {
        "predictor_id": M4C_ID,
        "model_family": "Ridge_residual_correction",
        "feature_group_id": "EMPLOYMENT_RECENT_V1",
        "feature_names": ";".join(FEATURE_NAMES),
        "target_definition": "B3 residual = actual target minus frozen B3 prediction",
        "preprocessing": "SimpleImputer(median, add_indicator=True) + StandardScaler fit inside outer origin",
        "estimator": "Ridge(alpha=1.0)",
        "ridge_alpha": 1.0,
        "uses_covariates": "yes",
        "city_identity_used": "no",
        "tuning": "no",
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


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
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
            writer.writerow({field: to_scalar(row.get(field)) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(field for row in rows for field in row))


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def assert_repo_identity() -> None:
    assert git_text("branch", "--show-current") == BRANCH
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "frozen base must be an ancestor of the worker branch"


def load_context() -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, Any],
    dict[str, Any],
]:
    s3_plan01.assert_static_boundaries()
    population, population_profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    s3_plan01.load_scenarios()
    s3_summary = json.loads(
        (S3_PLAN03_DIR / "plan03_summary.json").read_text(encoding="utf-8")
    )
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == B3_ID
    assert s3_summary["s2_contract_modified"] is False
    assert s3_summary["common_origin_domain"] == "2011-2021"
    return population, population_profile, s3_summary


def load_employment_context() -> tuple[
    dict[str, dict[int, float]],
    dict[str, Any],
    set[tuple[str, int]],
    set[tuple[str, int, float]],
    dict[tuple[int, str], dict[str, str]],
]:
    canonical_rows = read_csv(EMPLOYMENT_PATH)
    provenance_rows = read_csv(EMPLOYMENT_PROVENANCE_PATH)
    quarantine_rows = read_csv(EMPLOYMENT_QUARANTINE_PATH)
    assert canonical_rows and provenance_rows and quarantine_rows

    series: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    canonical_keys: set[tuple[str, int]] = set()
    canonical_values: dict[tuple[str, int], float] = {}
    for row in canonical_rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        value = as_float(row["employeesNumber"])
        assert city in series and year is not None and value is not None
        assert value > 0
        key = (city, year)
        assert key not in canonical_keys
        canonical_keys.add(key)
        canonical_values[key] = value
        series[city][year] = value

    derived_keys: set[tuple[str, int]] = set()
    for row in provenance_rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        value = as_float(row["employeesNumber"])
        assert city in series and year is not None and value is not None
        key = (city, year)
        assert key in canonical_keys
        assert abs(canonical_values[key] - value) <= 1e-12
        assert row["source_type"] == "sector_sum_reconstruction"
        assert key not in derived_keys
        derived_keys.add(key)

    quarantine_keys: set[tuple[str, int]] = set()
    quarantine_triples: set[tuple[str, int, float]] = set()
    for row in quarantine_rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        value = as_float(row["employeesNumber"])
        assert city in series and year is not None and value is not None
        quarantine_keys.add((city, year))
        quarantine_triples.add((city, year, value))

    assert len(canonical_rows) == 773
    assert len(canonical_keys) == 773
    assert len(series) == 40
    assert len(provenance_rows) == 91
    assert len(derived_keys) == 91
    assert len(quarantine_rows) == 80
    assert len(quarantine_triples) == 80
    assert min(year for _, year in canonical_keys) == 2000
    assert max(year for _, year in canonical_keys) == 2022
    assert not any(
        (city, year, value) in quarantine_triples
        for (city, year), value in canonical_values.items()
    )

    policy_rows = read_csv(S2_POLICY_PATH)
    policy_map: dict[tuple[int, str], dict[str, str]] = {}
    for origin in range(2000, max(COMMON_ORIGINS) + 1):
        for scenario_id in SCENARIOS:
            policy_id = POLICY_BY_SCENARIO[scenario_id]
            matches = [
                row
                for row in policy_rows
                if row["source_id"] == "employees"
                and as_int(row["pseudo_target_year"]) == origin
                and row["policy_id"] == policy_id
            ]
            expected_cap = origin - SCENARIO_LAG_YEARS[scenario_id]
            if not matches:
                assert origin == 2000
                policy = {
                    "policy_id": policy_id,
                    "effective_source_cap_year": str(expected_cap),
                    "policy_evidence_scope": "formula_only_pre_contract_training_year",
                }
            else:
                assert len(matches) == 1
                policy = dict(matches[0])
                observed_cap = as_int(policy["effective_source_cap_year"])
                if observed_cap is None:
                    assert expected_cap < min(
                        year for city in CITIES for year in series[city]
                    )
                    policy["policy_evidence_scope"] = (
                        "formula_only_unavailable_source_cutoff"
                    )
                    policy["effective_source_cap_year"] = str(expected_cap)
                else:
                    assert observed_cap == expected_cap
                    policy["policy_evidence_scope"] = "s2_policy_row"
                assert as_int(
                    policy["available_canonical_rows_at_or_before_cutoff"]
                ) == sum(
                    year <= expected_cap
                    for city in CITIES
                    for year in series[city]
                )
            policy_map[(origin, scenario_id)] = policy

    metadata = {
        "canonical_row_count": len(canonical_rows),
        "canonical_city_count": len(series),
        "canonical_year_min": min(year for _, year in canonical_keys),
        "canonical_year_max": max(year for _, year in canonical_keys),
        "source_non_derived_row_count": len(canonical_keys - derived_keys),
        "derived_row_count": len(derived_keys),
        "quarantine_row_count": len(quarantine_rows),
        "quarantine_unique_key_count": len(quarantine_keys),
        "quarantine_exact_value_overlap_count": 0,
        "source_status": "source_plus_remap_quarantine_and_derived",
        "observed_latest_year": max(year for _, year in canonical_keys),
        "observed_snapshot_lag_years": REFERENCE_TARGET_YEAR - max(year for _, year in canonical_keys),
        "provenance_file": str(EMPLOYMENT_PROVENANCE_PATH.relative_to(ROOT)),
        "quarantine_file": str(EMPLOYMENT_QUARANTINE_PATH.relative_to(ROOT)),
    }
    assert metadata["source_non_derived_row_count"] == 682
    assert metadata["observed_snapshot_lag_years"] == 1
    return series, metadata, derived_keys, quarantine_triples, policy_map


def ols_slope(years: list[int], values: list[float]) -> float:
    assert len(years) == len(values) >= 2
    x_mean = mean(float(year) for year in years)
    y_mean = mean(values)
    denominator = math.fsum((float(year) - x_mean) ** 2 for year in years)
    assert denominator > 0
    numerator = math.fsum(
        (float(year) - x_mean) * (value - y_mean)
        for year, value in zip(years, values)
    )
    return numerator / denominator


def build_employment_features(
    city: str,
    series: dict[int, float],
    pseudo_target_year: int,
    scenario_id: str,
    derived_keys: set[tuple[str, int]],
    quarantine_keys: set[tuple[str, int, float]],
    quarantine_key_set: set[tuple[str, int]],
) -> dict[str, Any]:
    lag_years = SCENARIO_LAG_YEARS[scenario_id]
    cap_year = pseudo_target_year - lag_years
    allowed_years = sorted(
        year
        for year, value in series.items()
        if year <= cap_year and value > 0
    )
    latest_year = allowed_years[-1] if allowed_years else None
    latest_value = series[latest_year] if latest_year is not None else None
    recent2_years = allowed_years[-2:] if len(allowed_years) >= 2 else []
    recent3_years = allowed_years[-3:] if len(allowed_years) >= 3 else []
    used_years = sorted(set(([latest_year] if latest_year is not None else []) + recent2_years + recent3_years))

    latest_log = math.log1p(latest_value) if latest_value is not None else None
    recent_log_growth = None
    if len(recent2_years) == 2:
        left, right = recent2_years
        recent_log_growth = (math.log(series[right]) - math.log(series[left])) / (right - left)
    recent3_log_slope = None
    if len(recent3_years) == 3:
        recent3_log_slope = ols_slope(
            recent3_years,
            [math.log(series[year]) for year in recent3_years],
        )

    values = {
        "employees_log_latest": latest_log,
        "employees_recent_log_growth": recent_log_growth,
        "employees_recent3_log_slope": recent3_log_slope,
    }
    missing = [name for name in FEATURE_NAMES if values[name] is None]
    used_derived_count = sum((city, year) in derived_keys for year in used_years)
    recent2_derived_count = sum((city, year) in derived_keys for year in recent2_years)
    recent3_derived_count = sum((city, year) in derived_keys for year in recent3_years)
    quarantine_candidate_hit_count = sum(
        (city, year, series[year]) in quarantine_keys for year in used_years
    )
    quarantine_key_overlap_count = sum(
        (city, year) in quarantine_key_set for year in used_years
    )
    return {
        **values,
        "feature_status": "ok" if not missing else "partial_missing",
        "missing_feature_names": ";".join(missing),
        "scenario_required_lag_years": lag_years,
        "source_cap_year": cap_year,
        "allowed_years": ";".join(str(year) for year in allowed_years),
        "allowed_value_count": len(allowed_years),
        "feature_used_years": ";".join(str(year) for year in used_years),
        "feature_history_max_year": latest_year,
        "latest_feature_year": latest_year,
        "latest_feature_value": latest_value,
        "latest_feature_is_derived": (
            "yes" if latest_year is not None and (city, latest_year) in derived_keys else "no"
        ),
        "recent2_feature_years": ";".join(str(year) for year in recent2_years),
        "recent3_feature_years": ";".join(str(year) for year in recent3_years),
        "recent2_derived_count": recent2_derived_count,
        "recent3_derived_count": recent3_derived_count,
        "feature_used_derived_count": used_derived_count,
        "feature_used_non_derived_count": len(used_years) - used_derived_count,
        "allowed_derived_count": sum((city, year) in derived_keys for year in allowed_years),
        "allowed_non_derived_count": sum((city, year) not in derived_keys for year in allowed_years),
        "quarantine_candidate_hit_count": quarantine_candidate_hit_count,
        "quarantine_key_overlap_count": quarantine_key_overlap_count,
    }


def build_combined_row(
    target_series: dict[int, float],
    employment_series: dict[int, float],
    city: str,
    pseudo_target_year: int,
    scenario_id: str,
    derived_keys: set[tuple[str, int]],
    quarantine_keys: set[tuple[str, int, float]],
    quarantine_key_set: set[tuple[str, int]],
    row_role: str,
) -> dict[str, Any]:
    b3_fit = s3_plan02.fit_prediction(
        B3_ID,
        target_series,
        pseudo_target_year,
        pseudo_target_year - HORIZON_YEARS,
    )
    actual = target_series.get(pseudo_target_year)
    employment_features = build_employment_features(
        city,
        employment_series,
        pseudo_target_year,
        scenario_id,
        derived_keys,
        quarantine_keys,
        quarantine_key_set,
    )
    b3_prediction = b3_fit["prediction"]
    return {
        "row_role": row_role,
        "row_year": pseudo_target_year,
        "city_id": city,
        "target_value": actual,
        "b3_prediction": b3_prediction,
        "b3_residual_actual_minus_prediction": (
            actual - b3_prediction
            if actual is not None and b3_prediction is not None
            else None
        ),
        "b3_fit": b3_fit,
        "employment_features": employment_features,
        "features": {
            name: as_float(employment_features.get(name))
            for name in FEATURE_NAMES
        },
    }


def build_training_rows(
    target_series: dict[int, float],
    employment_series: dict[int, float],
    city: str,
    outer_origin: int,
    scenario_id: str,
    derived_keys: set[tuple[str, int]],
    quarantine_keys: set[tuple[str, int, float]],
    quarantine_key_set: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    outer_cutoff = outer_origin - HORIZON_YEARS
    for row_year in sorted(year for year in target_series if year <= outer_cutoff):
        row = build_combined_row(
            target_series,
            employment_series,
            city,
            row_year,
            scenario_id,
            derived_keys,
            quarantine_keys,
            quarantine_key_set,
            "training",
        )
        if row["target_value"] is None or row["b3_prediction"] is None:
            excluded.append({**row, "exclusion_reason": "b3_insufficient_population_history"})
        else:
            eligible.append(row)
    return eligible, excluded


def build_validation_row(
    target_series: dict[int, float],
    employment_series: dict[int, float],
    city: str,
    origin: int,
    scenario_id: str,
    derived_keys: set[tuple[str, int]],
    quarantine_keys: set[tuple[str, int, float]],
    quarantine_key_set: set[tuple[str, int]],
) -> dict[str, Any]:
    row = build_combined_row(
        target_series,
        employment_series,
        city,
        origin,
        scenario_id,
        derived_keys,
        quarantine_keys,
        quarantine_key_set,
        "validation",
    )
    assert row["target_value"] is not None
    assert row["b3_prediction"] is not None
    return row


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
                row["features"].get(name)
                if row["features"].get(name) is not None
                else np.nan
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_ridge_residual(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> tuple[list[float], dict[str, Any]]:
    assert train_rows
    x_train = feature_matrix(train_rows)
    x_validation = feature_matrix(validation_rows)
    y_train = np.asarray(
        [row["b3_residual_actual_minus_prediction"] for row in train_rows],
        dtype=float,
    )
    assert np.isfinite(y_train).all()
    training_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    model = pipeline_for()
    model.fit(x_train, y_train)
    predicted_residuals = model.predict(x_validation).tolist()
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    indicator = getattr(imputer.indicator_, "features_", [])
    return predicted_residuals, {
        "training_row_count": len(train_rows),
        "training_missing_feature_cell_count": training_missing,
        "training_imputation_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "validation_imputation_count": validation_missing,
        "preprocessing_fit_row_count": len(train_rows),
        "model_fit_row_count": len(train_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "imputer_indicator_count": len(indicator),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": [float(value) for value in np.asarray(ridge.coef_).ravel()],
        "feature_names": ";".join(FEATURE_NAMES),
        "ridge_alpha": 1.0,
    }


def score_candidate_row(
    target: dict[str, str],
    scenario_id: str,
    origin: int,
    validation: dict[str, Any],
    predicted_residual: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = float(validation["target_value"])
    b3_prediction = float(validation["b3_prediction"])
    prediction = b3_prediction + float(predicted_residual)
    residual = prediction - actual
    feature = validation["employment_features"]
    return {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": scenario_id,
        "predictor_id": M4C_ID,
        "model_id": M4C_ID,
        "origin_year": origin,
        "outer_origin_year": origin,
        "city_id": validation["city_id"],
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
        "feature_builder_id": "EMPLOYMENT_RECENT_V1",
        "feature_group_id": "EMPLOYMENT_RECENT_V1",
        "model_feature_names": ";".join(FEATURE_NAMES),
        "feature_status": feature["feature_status"],
        "feature_missing_names": feature["missing_feature_names"],
        "feature_history_years": feature["feature_used_years"],
        "feature_history_max_year": feature["feature_history_max_year"],
        "employees_log_latest": feature["employees_log_latest"],
        "employees_recent_log_growth": feature["employees_recent_log_growth"],
        "employees_recent3_log_slope": feature["employees_recent3_log_slope"],
        "employment_source_cap_year": feature["source_cap_year"],
        "employment_latest_feature_year": feature["latest_feature_year"],
        "employment_latest_feature_is_derived": feature["latest_feature_is_derived"],
        "employment_recent2_feature_years": feature["recent2_feature_years"],
        "employment_recent3_feature_years": feature["recent3_feature_years"],
        "employment_recent2_derived_count": feature["recent2_derived_count"],
        "employment_recent3_derived_count": feature["recent3_derived_count"],
        "employment_feature_used_derived_count": feature["feature_used_derived_count"],
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
        "city_identity_used": "no",
    }


def wrap_b3_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "model_id": B3_ID,
            "feature_builder_id": "FROZEN_B3_REFERENCE",
            "feature_group_id": "FROZEN_B3",
            "b3_prediction_value": row["prediction_value"],
            "b3_residual_actual_minus_prediction": -float(
                row["residual_prediction_minus_actual"]
            ),
            "predicted_b3_residual": "",
            "covariates_used": "no",
            "city_identity_used": "no",
        }
    )
    return result


def make_boundary_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    row: dict[str, Any],
    metadata: dict[str, Any],
    policy_map: dict[tuple[int, str], dict[str, str]],
) -> dict[str, Any]:
    feature = row["employment_features"]
    b3_fit = row["b3_fit"]
    pseudo_year = int(row["row_year"])
    policy = policy_map[(pseudo_year, scenario_id)]
    cap = int(feature["source_cap_year"])
    used_years = [
        int(value)
        for value in feature["feature_used_years"].split(";")
        if value
    ]
    source_cap_pass = all(year <= cap for year in used_years)
    b3_history_max = b3_fit["history_max_year"]
    population_boundary_pass = (
        b3_history_max is None
        or int(b3_history_max) <= pseudo_year - HORIZON_YEARS
    )
    training_label_pass = (
        row["row_role"] != "training"
        or pseudo_year <= outer_origin - HORIZON_YEARS
    )
    quarantine_pass = feature["quarantine_candidate_hit_count"] == 0
    # The feature builder reads only the canonical series.  The explicit count
    # identity plus the exact quarantine triple check are the machine audit
    # boundary; the S2 policy row supplies the independent cap contract.
    provenance_pass = (
        len(used_years)
        == feature["feature_used_derived_count"]
        + feature["feature_used_non_derived_count"]
        and len(used_years) == len(set(used_years))
        and int(policy["effective_source_cap_year"]) == cap
        and feature["quarantine_candidate_hit_count"] == 0
    )
    checks = [
        source_cap_pass,
        population_boundary_pass,
        training_label_pass,
        quarantine_pass,
        provenance_pass,
        len(used_years) == len(set(used_years)),
    ]
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "row_role": row["row_role"],
        "city_id": row["city_id"],
        "pseudo_target_year": pseudo_year,
        "outer_training_cutoff_year": outer_origin - HORIZON_YEARS,
        "employment_policy_id": policy["policy_id"],
        "employment_required_lag_years": feature["scenario_required_lag_years"],
        "employment_source_cap_year": cap,
        "employment_allowed_years": feature["allowed_years"],
        "employment_feature_used_years": feature["feature_used_years"],
        "employment_feature_history_max_year": feature["feature_history_max_year"] or "",
        "employment_latest_feature_year": feature["latest_feature_year"] or "",
        "employment_latest_feature_is_derived": feature["latest_feature_is_derived"],
        "employment_recent2_feature_years": feature["recent2_feature_years"],
        "employment_recent3_feature_years": feature["recent3_feature_years"],
        "employment_recent2_derived_count": feature["recent2_derived_count"],
        "employment_recent3_derived_count": feature["recent3_derived_count"],
        "employment_feature_used_derived_count": feature["feature_used_derived_count"],
        "employment_feature_used_non_derived_count": feature[
            "feature_used_non_derived_count"
        ],
        "employment_allowed_derived_count": feature["allowed_derived_count"],
        "employment_allowed_non_derived_count": feature["allowed_non_derived_count"],
        "employment_feature_status": feature["feature_status"],
        "employment_missing_feature_names": feature["missing_feature_names"],
        "population_history_max_year_used": b3_history_max or "",
        "population_history_within_y_minus_2": "PASS"
        if population_boundary_pass
        else "FAIL",
        "employment_source_cap_within_policy": "PASS"
        if source_cap_pass
        else "FAIL",
        "s2_policy_cap_match": "PASS"
        if int(policy["effective_source_cap_year"]) == cap
        else "FAIL",
        "outer_training_label_within_T_minus_2": (
            "PASS"
            if row["row_role"] == "validation" or training_label_pass
            else "FAIL"
        ),
        "validation_label_absent_from_fit": (
            "yes" if row["row_role"] == "validation" else "not_applicable"
        ),
        "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
        "model_fit_row_count": metadata["model_fit_row_count"],
        "preprocessing_model_fit_rows_equal": metadata[
            "preprocessing_model_fit_rows_equal"
        ],
        "training_label_min_year": metadata["training_label_min_year"],
        "training_label_max_year": metadata["training_label_max_year"],
        "quarantine_key_overlap_count": feature["quarantine_key_overlap_count"],
        "quarantine_candidate_hit_count": feature[
            "quarantine_candidate_hit_count"
        ],
        "quarantine_value_reintroduced": "no"
        if quarantine_pass
        else "yes",
        "provenance_tracking_status": "PASS" if provenance_pass else "FAIL",
        "feature_source_target_id": target_id,
        "unexpected_feature_columns_used": "no",
        "sector_share_features_used": "no",
        "unemployment_features_used": "no",
        "wage_features_used": "no",
        "density_features_used": "no",
        "city_identity_used": "no",
        "leakage_status": "PASS" if all(checks) else "FAIL",
        "boundary_status": "PASS" if all(checks) else "FAIL",
    }


def build_candidate_rows(
    population: dict[str, dict[str, dict[int, float]]],
    employment_series: dict[str, dict[int, float]],
    derived_keys: set[tuple[str, int]],
    quarantine_keys: set[tuple[str, int, float]],
    quarantine_key_set: set[tuple[str, int]],
    policy_map: dict[tuple[int, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    training_geometry: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                training_by_city: dict[str, list[dict[str, Any]]] = {}
                excluded_by_city: dict[str, list[dict[str, Any]]] = {}
                validation_by_city: dict[str, dict[str, Any]] = {}
                for city in CITIES:
                    target_series = population[target_id][city]
                    train_rows, excluded_rows = build_training_rows(
                        target_series,
                        employment_series[city],
                        city,
                        origin,
                        scenario_id,
                        derived_keys,
                        quarantine_keys,
                        quarantine_key_set,
                    )
                    training_by_city[city] = train_rows
                    excluded_by_city[city] = excluded_rows
                    validation_by_city[city] = build_validation_row(
                        target_series,
                        employment_series[city],
                        city,
                        origin,
                        scenario_id,
                        derived_keys,
                        quarantine_keys,
                        quarantine_key_set,
                    )

                train_rows = [
                    row
                    for city in CITIES
                    for row in training_by_city[city]
                ]
                excluded_rows = [
                    row
                    for city in CITIES
                    for row in excluded_by_city[city]
                ]
                validation_rows = [validation_by_city[city] for city in CITIES]
                assert train_rows and len(validation_rows) == len(CITIES)

                predicted_residuals, metadata = fit_ridge_residual(
                    train_rows,
                    validation_rows,
                )
                metadata["training_label_min_year"] = min(
                    int(row["row_year"]) for row in train_rows
                )
                metadata["training_label_max_year"] = max(
                    int(row["row_year"]) for row in train_rows
                )
                for city, validation, predicted_residual in zip(
                    CITIES,
                    validation_rows,
                    predicted_residuals,
                ):
                    candidate_rows.append(
                        score_candidate_row(
                            target,
                            scenario_id,
                            origin,
                            validation,
                            predicted_residual,
                            metadata,
                        )
                    )
                for row in train_rows + excluded_rows + validation_rows:
                    boundary_rows.append(
                        make_boundary_row(
                            target_id,
                            scenario_id,
                            origin,
                            row,
                            metadata,
                            policy_map,
                        )
                    )

                training_geometry.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": origin,
                        "training_row_count": len(train_rows),
                        "training_excluded_row_count": len(excluded_rows),
                        "training_label_min_year": metadata["training_label_min_year"],
                        "training_label_max_year": metadata["training_label_max_year"],
                        "training_missing_feature_cell_count": metadata[
                            "training_missing_feature_cell_count"
                        ],
                        "validation_missing_feature_cell_count": metadata[
                            "validation_missing_feature_cell_count"
                        ],
                        "preprocessing_fit_row_count": metadata[
                            "preprocessing_fit_row_count"
                        ],
                        "model_fit_row_count": metadata["model_fit_row_count"],
                        "preprocessing_model_fit_rows_equal": metadata[
                            "preprocessing_model_fit_rows_equal"
                        ],
                        "ridge_alpha": metadata["ridge_alpha"],
                        "feature_names": metadata["feature_names"],
                        "employment_source_cap_for_validation": (
                            origin - SCENARIO_LAG_YEARS[scenario_id]
                        ),
                    }
                )
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(candidate_rows) == expected
    assert boundary_rows
    return candidate_rows, boundary_rows, training_geometry


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    saved_predictors = s3_plan01.PREDICTORS
    s3_plan01.PREDICTORS = MODEL_DEFINITIONS
    try:
        per_origin, _ = s3_plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(
            rows,
            per_origin,
        )
    finally:
        s3_plan01.PREDICTORS = saved_predictors
    return per_origin, aggregate, aggregate_map


def compare_b3_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = read_csv(
        S3_PLAN03_DIR / "per_origin_predictions_or_residuals.csv"
    )
    frozen_map = {
        (
            row["target_id"],
            row["scenario_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in frozen_rows
        if row["predictor_id"] == B3_ID
    }
    current_map = {
        (
            row["target_id"],
            row["scenario_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in current_rows
        if row["predictor_id"] == B3_ID
    }
    assert set(current_map) == set(frozen_map)
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            pairs = [
                (current_map[key], frozen_map[key])
                for key in sorted(current_map)
                if key[0] == target["target_id"] and key[1] == scenario_id
            ]
            max_deltas = {
                field: max(
                    abs(float(left[field]) - float(right[field]))
                    for left, right in pairs
                )
                for field in (
                    "prediction_value",
                    "residual_prediction_minus_actual",
                    "squared_error",
                )
            }
            result.append(
                {
                    "metric_scope": "b3_sanity_row",
                    "target_id": target["target_id"],
                    "scenario_id": scenario_id,
                    "predictor_id": B3_ID,
                    "rows_compared": len(pairs),
                    "max_abs_prediction_delta": max_deltas["prediction_value"],
                    "max_abs_residual_delta": max_deltas[
                        "residual_prediction_minus_actual"
                    ],
                    "max_abs_squared_error_delta": max_deltas["squared_error"],
                    "status": "PASS"
                    if max(max_deltas.values()) <= 1e-9
                    else "FAIL",
                    "comparison_rule": "B3 row-level prediction/residual/squared-error delta <= 1e-9",
                }
            )
    assert all(row["status"] == "PASS" for row in result)
    return result


def compare_b3_aggregate(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    frozen_rows = read_csv(S3_PLAN03_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in frozen_rows
        if row["predictor_id"] == B3_ID
    }
    result: list[dict[str, Any]] = []
    for key, frozen in sorted(frozen_map.items()):
        current = aggregate_map[key]
        deltas = {
            field: abs(float(current[current_field]) - float(frozen[field]))
            for field, current_field in (
                ("pooled_city_year_mse", "pooled_city_year_mse"),
                ("macro_origin_mse", "macro_origin_mse"),
                ("latest_origin_mse", "latest_origin_mse"),
            )
        }
        result.append(
            {
                "metric_scope": "b3_sanity_aggregate",
                "target_id": key[0],
                "scenario_id": key[1],
                "predictor_id": B3_ID,
                "max_abs_pooled_mse_delta": deltas["pooled_city_year_mse"],
                "max_abs_macro_mse_delta": deltas["macro_origin_mse"],
                "max_abs_latest_mse_delta": deltas["latest_origin_mse"],
                "status": "PASS" if max(deltas.values()) <= 1e-9 else "FAIL",
                "comparison_rule": "B3 pooled, macro and latest MSE delta <= 1e-9",
            }
        )
    assert len(result) == len(TARGETS) * len(SCENARIOS)
    assert all(row["status"] == "PASS" for row in result)
    return result


def score_rows(
    rows: list[dict[str, Any]],
    target_id: str,
    scenario_id: str,
    predictor_id: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["target_id"] == target_id
        and row["scenario_id"] == scenario_id
        and row["predictor_id"] == predictor_id
        and row["scoreable"] == "yes"
    ]


def build_pairwise_row(
    rows: list[dict[str, Any]],
    per_origin_map: dict[tuple[str, str, str, int], dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    target_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    baseline = aggregate_map[(target_id, scenario_id, B3_ID)]
    candidate = aggregate_map[(target_id, scenario_id, M4C_ID)]
    origin_deltas: list[float] = []
    win_years: list[int] = []
    loss_years: list[int] = []
    tie_years: list[int] = []
    for origin in COMMON_ORIGINS:
        base = per_origin_map[(target_id, scenario_id, B3_ID, origin)]
        cand = per_origin_map[(target_id, scenario_id, M4C_ID, origin)]
        delta = float(base["mse"]) - float(cand["mse"])
        origin_deltas.append(delta)
        if delta > 1e-12:
            win_years.append(origin)
        elif delta < -1e-12:
            loss_years.append(origin)
        else:
            tie_years.append(origin)
    base_rows = {
        (int(row["origin_year"]), row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, B3_ID)
    }
    candidate_rows = {
        (int(row["origin_year"]), row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, M4C_ID)
    }
    row_deltas = [
        float(base_rows[key]["squared_error"])
        - float(candidate_rows[key]["squared_error"])
        for key in sorted(base_rows)
    ]
    latest_base = per_origin_map[(target_id, scenario_id, B3_ID, LATEST_ORIGIN)]
    latest_candidate = per_origin_map[
        (target_id, scenario_id, M4C_ID, LATEST_ORIGIN)
    ]
    baseline_mse = float(baseline["pooled_city_year_mse"])
    candidate_mse = float(candidate["pooled_city_year_mse"])
    return {
        "metric_scope": "pairwise_b3",
        "target_id": target_id,
        "scenario_id": scenario_id,
        "baseline_predictor_id": B3_ID,
        "candidate_predictor_id": M4C_ID,
        "baseline_pooled_mse": baseline_mse,
        "candidate_pooled_mse": candidate_mse,
        "pooled_mse_absolute_delta_baseline_minus_candidate": baseline_mse
        - candidate_mse,
        "pooled_mse_relative_improvement_pct": (
            (baseline_mse - candidate_mse) / baseline_mse * 100.0
        ),
        "baseline_macro_origin_mse": float(baseline["macro_origin_mse"]),
        "candidate_macro_origin_mse": float(candidate["macro_origin_mse"]),
        "baseline_latest_2021_mse": float(baseline["latest_origin_mse"]),
        "candidate_latest_2021_mse": float(candidate["latest_origin_mse"]),
        "latest_mse_relative_improvement_pct": (
            (float(baseline["latest_origin_mse"])
             - float(candidate["latest_origin_mse"]))
            / float(baseline["latest_origin_mse"])
            * 100.0
        ),
        "latest_candidate_outcome": (
            "win"
            if float(latest_base["mse"]) > float(latest_candidate["mse"]) + 1e-12
            else "loss"
            if float(latest_base["mse"]) < float(latest_candidate["mse"]) - 1e-12
            else "tie"
        ),
        "candidate_origin_win_count": len(win_years),
        "candidate_origin_loss_count": len(loss_years),
        "origin_tie_count": len(tie_years),
        "origin_win_years": ";".join(map(str, win_years)),
        "origin_loss_years": ";".join(map(str, loss_years)),
        "origin_tie_years": ";".join(map(str, tie_years)),
        "origin_count": len(COMMON_ORIGINS),
        "candidate_origin_win_rate": len(win_years) / len(COMMON_ORIGINS),
        "candidate_city_origin_improved_share": sum(
            delta > 1e-12 for delta in row_deltas
        )
        / len(row_deltas),
        "candidate_city_origin_worsened_share": sum(
            delta < -1e-12 for delta in row_deltas
        )
        / len(row_deltas),
        "ranking_metric": "pooled_city_year_mse",
        "score_is_secondary_transform_only": "yes",
    }


def validation_audit_map(
    boundary_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result = {}
    for row in boundary_rows:
        if row["row_role"] != "validation":
            continue
        key = (
            row["target_id"],
            row["scenario_id"],
            int(row["pseudo_target_year"]),
            row["city_id"],
        )
        assert key not in result
        result[key] = row
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(result) == expected
    return result


def feature_values_differ(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for field in (
        "employment_feature_used_years",
        "employment_latest_feature_is_derived",
        "employment_recent2_feature_years",
        "employment_recent3_feature_years",
    ):
        if left[field] != right[field]:
            return True
    for field in FEATURE_NAMES:
        left_value = as_float(left.get(field))
        right_value = as_float(right.get(field))
        if left_value is None or right_value is None:
            if left_value != right_value:
                return True
        elif abs(left_value - right_value) > 1e-12:
            return True
    return False


def build_r1_r2_sensitivity_rows(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit = validation_audit_map(boundary_rows)
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for model_id in MODEL_IDS:
            r1 = aggregate_map[(target_id, SCENARIOS[0], model_id)]
            r2 = aggregate_map[(target_id, SCENARIOS[1], model_id)]
            keys = [
                (target_id, SCENARIOS[0], origin, city)
                for origin in COMMON_ORIGINS
                for city in CITIES
            ]
            feature_difference_count = 0
            for _, _, origin, city in keys:
                left = audit[(target_id, SCENARIOS[0], origin, city)]
                right = audit[(target_id, SCENARIOS[1], origin, city)]
                if feature_values_differ(left, right):
                    feature_difference_count += 1
            model_feature_difference_count = (
                feature_difference_count if model_id == M4C_ID else 0
            )
            result.append(
                {
                    "metric_scope": "r1_r2_sensitivity",
                    "target_id": target_id,
                    "predictor_id": model_id,
                    "r1_scenario_id": SCENARIOS[0],
                    "r2_scenario_id": SCENARIOS[1],
                    "r1_pooled_mse": r1["pooled_city_year_mse"],
                    "r2_pooled_mse": r2["pooled_city_year_mse"],
                    "r2_minus_r1_pooled_mse": float(
                        r2["pooled_city_year_mse"]
                    )
                    - float(r1["pooled_city_year_mse"]),
                    "r1_latest_2021_mse": r1["latest_origin_mse"],
                    "r2_latest_2021_mse": r2["latest_origin_mse"],
                    "r2_minus_r1_latest_2021_mse": float(
                        r2["latest_origin_mse"]
                    )
                    - float(r1["latest_origin_mse"]),
                    "employment_feature_difference_count": feature_difference_count,
                    "r1_r2_feature_difference_count": model_feature_difference_count,
                    "r1_r2_feature_difference_confirmed": (
                        "yes"
                        if model_id == M4C_ID and model_feature_difference_count > 0
                        else "no"
                    ),
                    "interpretation": (
                        "employment source-boundary rerun"
                        if model_id == M4C_ID
                        else "target-history-only equivalence control"
                    ),
                }
            )
    return result


def build_provenance_segment_rows(
    rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit = validation_audit_map(boundary_rows)
    baseline_map = {
        (row["target_id"], row["scenario_id"], int(row["origin_year"]), row["city_id"]): row
        for row in rows
        if row["predictor_id"] == B3_ID
    }
    candidate_map = {
        (row["target_id"], row["scenario_id"], int(row["origin_year"]), row["city_id"]): row
        for row in rows
        if row["predictor_id"] == M4C_ID
    }
    selectors: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
        ("all_validation_rows", lambda row: True),
        (
            "any_derived_in_recent3",
            lambda row: int(row["employment_recent3_derived_count"]) > 0,
        ),
        (
            "no_derived_in_recent3",
            lambda row: int(row["employment_recent3_derived_count"]) == 0,
        ),
        (
            "derived_heavy_recent3_ge2",
            lambda row: int(row["employment_recent3_derived_count"]) >= 2,
        ),
        (
            "latest_feature_derived",
            lambda row: row["employment_latest_feature_is_derived"] == "yes",
        ),
        (
            "latest_feature_non_derived",
            lambda row: row["employment_latest_feature_is_derived"] == "no",
        ),
    )
    result: list[dict[str, Any]] = []
    association: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            keys = [
                (target["target_id"], scenario_id, origin, city)
                for origin in COMMON_ORIGINS
                for city in CITIES
            ]
            enriched = []
            for key in keys:
                base = baseline_map[key]
                candidate = candidate_map[key]
                boundary = audit[key]
                delta = float(base["squared_error"]) - float(
                    candidate["squared_error"]
                )
                enriched.append((boundary, delta, base, candidate))
            positive_total = sum(max(delta, 0.0) for _, delta, _, _ in enriched)
            derived_window_positive = sum(
                max(delta, 0.0)
                for boundary, delta, _, _ in enriched
                if int(boundary["employment_recent3_derived_count"]) > 0
            )
            derived_heavy_positive = sum(
                max(delta, 0.0)
                for boundary, delta, _, _ in enriched
                if int(boundary["employment_recent3_derived_count"]) >= 2
            )
            derived_window_wins = sum(
                delta > 1e-12
                for boundary, delta, _, _ in enriched
                if int(boundary["employment_recent3_derived_count"]) > 0
            )
            total_wins = sum(delta > 1e-12 for _, delta, _, _ in enriched)
            association.append(
                {
                    "metric_scope": "provenance_association",
                    "target_id": target["target_id"],
                    "scenario_id": scenario_id,
                    "candidate_predictor_id": M4C_ID,
                    "validation_row_count": len(enriched),
                    "positive_sse_gain_total": positive_total,
                    "positive_sse_gain_any_derived_recent3": derived_window_positive,
                    "positive_sse_gain_derived_heavy_recent3": derived_heavy_positive,
                    "derived_window_positive_gain_share": (
                        derived_window_positive / positive_total
                        if positive_total
                        else 0.0
                    ),
                    "derived_heavy_positive_gain_share": (
                        derived_heavy_positive / positive_total
                        if positive_total
                        else 0.0
                    ),
                    "candidate_win_count": total_wins,
                    "candidate_wins_any_derived_recent3": derived_window_wins,
                    "candidate_win_share_any_derived_recent3": (
                        derived_window_wins / total_wins if total_wins else 0.0
                    ),
                    "provenance_sensitive": (
                        "yes"
                        if positive_total
                        and (
                            derived_window_positive / positive_total >= 0.5
                            or derived_heavy_positive / positive_total >= 0.5
                        )
                        else "no"
                    ),
                }
            )
            for segment_name, selector in selectors:
                segment = [
                    item
                    for item in enriched
                    if selector(item[0])
                ]
                deltas = [item[1] for item in segment]
                base_sse = sum(
                    float(item[2]["squared_error"]) for item in segment
                )
                candidate_sse = sum(
                    float(item[3]["squared_error"]) for item in segment
                )
                result.append(
                    {
                        "metric_scope": "provenance_segment",
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "segment_name": segment_name,
                        "candidate_predictor_id": M4C_ID,
                        "segment_row_count": len(segment),
                        "baseline_segment_mse": (
                            base_sse / len(segment) if segment else ""
                        ),
                        "candidate_segment_mse": (
                            candidate_sse / len(segment) if segment else ""
                        ),
                        "segment_sse_delta_baseline_minus_candidate": (
                            base_sse - candidate_sse
                            if segment
                            else ""
                        ),
                        "candidate_win_count": sum(
                            delta > 1e-12 for delta in deltas
                        ),
                        "candidate_loss_count": sum(
                            delta < -1e-12 for delta in deltas
                        ),
                        "candidate_tie_count": sum(
                            abs(delta) <= 1e-12 for delta in deltas
                        ),
                        "positive_sse_gain": sum(
                            max(delta, 0.0) for delta in deltas
                        ),
                    }
                )
    return result, association


def make_decision(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pairwise_map = {
        (row["target_id"], row["scenario_id"]): row for row in pairwise_rows
    }
    resident_beats_b3_all = all(
        float(aggregate_map[("resident", scenario_id, M4C_ID)]["pooled_city_year_mse"])
        < float(aggregate_map[("resident", scenario_id, B3_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    registered_supports = all(
        float(aggregate_map[("registered", scenario_id, M4C_ID)]["pooled_city_year_mse"])
        <= float(aggregate_map[("registered", scenario_id, B3_ID)]["pooled_city_year_mse"])
        for scenario_id in SCENARIOS
    )
    close_with_origin_latest_support = False
    for scenario_id in SCENARIOS:
        pair = pairwise_map[("resident", scenario_id)]
        relative_gap = abs(float(pair["pooled_mse_relative_improvement_pct"]))
        origin_support = int(pair["candidate_origin_win_count"]) > int(
            pair["candidate_origin_loss_count"]
        )
        latest_support = pair["latest_candidate_outcome"] == "win"
        if relative_gap <= 1.0 and origin_support and latest_support:
            close_with_origin_latest_support = True
    association_map = {
        (row["target_id"], row["scenario_id"]): row
        for row in association_rows
    }
    resident_gain = any(
        float(pairwise_map[("resident", scenario_id)][
            "pooled_mse_absolute_delta_baseline_minus_candidate"
        ]) > 0
        for scenario_id in SCENARIOS
    )
    provenance_sensitive_resident = any(
        association_map[("resident", scenario_id)]["provenance_sensitive"] == "yes"
        for scenario_id in SCENARIOS
    )
    deep_dive = (
        resident_beats_b3_all
        or close_with_origin_latest_support
        or (resident_gain and provenance_sensitive_resident)
    )
    if provenance_sensitive_resident:
        screening_decision = "provenance-sensitive employment signal"
    elif resident_beats_b3_all:
        screening_decision = "employment scale signal supported"
    else:
        screening_decision = "unsupported"
    return {
        "screening_decision": screening_decision,
        "resident_beats_b3_in_both_r1_r2": resident_beats_b3_all,
        "registered_supports_candidate": registered_supports,
        "close_with_origin_latest_support": close_with_origin_latest_support,
        "resident_gain_present": resident_gain,
        "provenance_sensitive_resident": provenance_sensitive_resident,
        "deep_dive_recommended": deep_dive,
        "deep_dive_rule": (
            "resident beats B3 in both scenarios; or resident is within 1% "
            "with origin/latest support; or clear resident gain is provenance-sensitive"
        ),
    }


def main() -> None:
    assert_repo_identity()
    population, population_profile, s3_summary = load_context()
    (
        employment_series,
        employment_metadata,
        derived_keys,
        quarantine_triples,
        policy_map,
    ) = load_employment_context()
    quarantine_key_set = {
        (city, year)
        for city, year, _ in quarantine_triples
    }

    b3_all_rows, _ = s3_plan02.build_fixed_rows(population)
    b3_rows = [
        wrap_b3_row(row)
        for row in b3_all_rows
        if row["predictor_id"] == B3_ID
    ]
    expected_scope_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(b3_rows) == expected_scope_rows
    b3_row_sanity = compare_b3_rows(b3_rows)

    candidate_rows, boundary_rows, training_geometry = build_candidate_rows(
        population,
        employment_series,
        derived_keys,
        quarantine_triples,
        quarantine_key_set,
        policy_map,
    )
    assert len(candidate_rows) == expected_scope_rows
    all_rows = b3_rows + candidate_rows
    assert len(all_rows) == expected_scope_rows * len(MODEL_IDS)

    per_origin_metrics, aggregate_metrics, aggregate_map = calculate_metrics(all_rows)
    per_origin_map = {
        (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
            int(row["origin_year"]),
        ): row
        for row in per_origin_metrics
    }
    b3_aggregate_sanity = compare_b3_aggregate(aggregate_map)
    pairwise_rows = [
        build_pairwise_row(
            all_rows,
            per_origin_map,
            aggregate_map,
            target["target_id"],
            scenario_id,
        )
        for target in TARGETS
        for scenario_id in SCENARIOS
    ]
    r1_r2_rows = build_r1_r2_sensitivity_rows(aggregate_map, boundary_rows)
    provenance_segment_rows, association_rows = build_provenance_segment_rows(
        all_rows,
        boundary_rows,
    )
    decision = make_decision(aggregate_map, pairwise_rows, association_rows)

    assert all(row["status"] == "PASS" for row in b3_row_sanity)
    assert all(row["status"] == "PASS" for row in b3_aggregate_sanity)
    assert all(row["boundary_status"] == "PASS" for row in boundary_rows)
    assert all(row["leakage_status"] == "PASS" for row in boundary_rows)
    assert all(
        row["quarantine_value_reintroduced"] == "no"
        for row in boundary_rows
    )
    assert all(
        row["preprocessing_model_fit_rows_equal"] == "yes"
        for row in training_geometry
    )
    assert all(
        row["employment_required_lag_years"] == SCENARIO_LAG_YEARS[row["scenario_id"]]
        for row in boundary_rows
    )
    assert all(
        row["r1_r2_feature_difference_confirmed"] == "yes"
        for row in r1_r2_rows
        if row["predictor_id"] == M4C_ID
    )

    screening_rows: list[dict[str, Any]] = []
    screening_rows.extend(dict(row, metric_scope="per_origin") for row in per_origin_metrics)
    screening_rows.extend(dict(row, metric_scope="aggregate") for row in aggregate_metrics)
    screening_rows.extend(pairwise_rows)
    screening_rows.extend(r1_r2_rows)
    screening_rows.extend(provenance_segment_rows)
    screening_rows.extend(association_rows)
    screening_rows.extend(b3_row_sanity)
    screening_rows.extend(b3_aggregate_sanity)

    aggregate_summary = {}
    for row in aggregate_metrics:
        aggregate_summary[
            f"{row['target_id']}_{row['scenario_id']}_{row['predictor_id']}"
        ] = {
            "pooled_mse": row["pooled_city_year_mse"],
            "macro_origin_mse": row["macro_origin_mse"],
            "latest_2021_mse": row["latest_origin_mse"],
            "pooled_rmse": row["pooled_city_year_rmse"],
            "pooled_mae": row["pooled_city_year_mae"],
        }
    pairwise_summary = {
        f"{row['target_id']}_{row['scenario_id']}": {
            key: row[key]
            for key in (
                "candidate_pooled_mse",
                "pooled_mse_absolute_delta_baseline_minus_candidate",
                "pooled_mse_relative_improvement_pct",
                "baseline_latest_2021_mse",
                "candidate_latest_2021_mse",
                "latest_mse_relative_improvement_pct",
                "candidate_origin_win_count",
                "candidate_origin_loss_count",
                "origin_win_years",
                "origin_loss_years",
            )
        }
        for row in pairwise_rows
    }
    r1_r2_summary = {
        f"{row['target_id']}_{row['predictor_id']}": {
            key: row[key]
            for key in (
                "r1_pooled_mse",
                "r2_pooled_mse",
                "r2_minus_r1_pooled_mse",
                "r1_latest_2021_mse",
                "r2_latest_2021_mse",
                "r2_minus_r1_latest_2021_mse",
                "r1_r2_feature_difference_count",
                "r1_r2_feature_difference_confirmed",
            )
        }
        for row in r1_r2_rows
    }
    association_summary = {
        f"{row['target_id']}_{row['scenario_id']}": row
        for row in association_rows
    }

    summary = {
        "plan_id": PLAN_ID,
        "stage": "S4_FEATURE_ML_MODELING",
        "starting_commit": STARTING_COMMIT,
        "branch": BRANCH,
        "frozen_base_commit": STARTING_COMMIT,
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "target_history_cutoff_rule": "population B3 history <= pseudo-target-2; outer training labels <= outer-origin-2",
        "employment_boundary_rule": "R1 source cap = pseudo-target-1; R2 source cap = pseudo-target-2",
        "model_ids": list(MODEL_IDS),
        "model_family": "Ridge residual correction",
        "ridge_alpha": 1.0,
        "feature_group_id": "EMPLOYMENT_RECENT_V1",
        "feature_names": list(FEATURE_NAMES),
        "feature_formulas": {
            "employees_log_latest": "log1p(latest allowed positive employeesNumber)",
            "employees_recent_log_growth": "annualized [log(recent latest)-log(recent previous)] using latest two allowed positive observations",
            "employees_recent3_log_slope": "OLS slope of log(employeesNumber) against real year over latest three allowed positive observations",
        },
        "employment_source": employment_metadata,
        "population_profile": population_profile,
        "s3_reference_summary": {
            "starting_commit": s3_summary["starting_commit"],
            "strongest_fixed_baseline": s3_summary[
                "strongest_deployable_fixed_baseline_candidate"
            ],
        },
        "metrics": aggregate_summary,
        "pairwise_vs_b3": pairwise_summary,
        "r1_r2_sensitivity": r1_r2_summary,
        "provenance_association": association_summary,
        "b3_row_sanity_pass": True,
        "b3_aggregate_sanity_pass": True,
        "employment_r1_r2_boundary_pass": all(
            row["s2_policy_cap_match"] == "PASS" for row in boundary_rows
        ),
        "provenance_tracking_complete": all(
            row["provenance_tracking_status"] == "PASS" for row in boundary_rows
        ),
        "quarantine_zero_reintroduction": all(
            row["quarantine_value_reintroduced"] == "no"
            for row in boundary_rows
        ),
        "resident_registered_screening_complete": True,
        "resident_b3_pooled_mse": aggregate_map[
            ("resident", SCENARIOS[0], B3_ID)
        ]["pooled_city_year_mse"],
        "registered_b3_pooled_mse": aggregate_map[
            ("registered", SCENARIOS[0], B3_ID)
        ]["pooled_city_year_mse"],
        "screening_decision": decision["screening_decision"],
        "deep_dive_recommended": decision["deep_dive_recommended"],
        "decision_detail": decision,
        "training_geometry_rows": len(training_geometry),
        "boundary_audit_rows": len(boundary_rows),
        "candidate_prediction_rows": len(candidate_rows),
        "screening_metric_rows": len(screening_rows),
        "uses_sector_shares": False,
        "uses_unemployment": False,
        "uses_wage": False,
        "uses_density": False,
        "uses_city_id": False,
        "tuning_run": False,
        "gbdt_run": False,
        "ensemble_run": False,
        "final_2023_prediction_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
    }

    write_csv(OUTPUT_DIR / "screening_metrics.csv", screening_rows)
    write_csv(
        OUTPUT_DIR / "employment_provenance_boundary_audit.csv",
        boundary_rows,
    )
    write_json(OUTPUT_DIR / "plan04c_summary.json", summary)

    print("S4_PLAN04C_EMPLOYMENT_SIGNAL_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"branch={BRANCH}")
    print("common_origin_domain=2011-2021")
    print(f"model_ids={','.join(MODEL_IDS)}")
    print(f"candidate_prediction_rows={len(candidate_rows)}")
    print(f"boundary_audit_rows={len(boundary_rows)}")
    print(f"screening_metric_rows={len(screening_rows)}")
    print(f"b3_row_sanity_pass={summary['b3_row_sanity_pass']}")
    print(f"b3_aggregate_sanity_pass={summary['b3_aggregate_sanity_pass']}")
    print(
        "employment_r1_r2_boundary_pass="
        f"{summary['employment_r1_r2_boundary_pass']}"
    )
    print(
        "provenance_tracking_complete="
        f"{summary['provenance_tracking_complete']}"
    )
    print(
        "quarantine_zero_reintroduction="
        f"{summary['quarantine_zero_reintroduction']}"
    )
    print(f"screening_decision={summary['screening_decision']}")
    print(f"deep_dive_recommended={summary['deep_dive_recommended']}")
    print("uses_sector_shares=False")
    print("uses_unemployment=False")
    print("uses_wage=False")
    print("uses_density=False")
    print("uses_city_id=False")
    print("tuning_run=False")
    print("gbdt_run=False")
    print("ensemble_run=False")
    print("final_2023_prediction_run=False")
    print("submission_run=False")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
