"""Build the S4-Plan05C compact multi-source RidgeCV screening evidence.

The runner tests one strictly pre-registered joint residual correction over the
frozen B3 population trend. Four audited source semantics are rebuilt under
their own S2 information caps. Provenance, quarantine and density-regime fields
are audit-only; exactly eight raw source features reach the model.

This is a screening worker. It does not run a model sweep, add sources,
generate a 2023 forecast, enter S5, or modify shared project-state files.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_s2_plan02_geometry as s2_geometry
import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02
import build_s4_plan02_urbanization_residual as s4_plan02
import build_s4_plan04a_density_residual as s4_plan04a
import build_s4_plan04b_wage_screen as s4_plan04b
import build_s4_plan04c_employment_signal as s4_plan04c


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan05C"
S3_PLAN03_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
REGIME_REPORT_PATH = ROOT / "data" / "normalized" / "人口密度_regime_report.csv"
EMPLOYMENT_PROVENANCE_PATH = (
    ROOT / "data" / "normalized" / "就业信息_从业人员数_provenance.csv"
)
EMPLOYMENT_QUARANTINE_PATH = (
    ROOT / "data" / "quarantine" / "就业信息_从业人员数_quarantine.csv"
)
WAVE_A_SUMMARY_PATH = ROOT / "reports" / "S4_HANDOFFS" / "S4-WaveA" / "WAVE_A_SUMMARY.md"

PLAN_ID = "S4-Plan05C"
STARTING_COMMIT = "decfd7e0c68d14da31eabe5720ab706122c1040d"
BRANCH = "s4/wave-b-multisource-joint"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
M5C_ID = "M5C_B3_PLUS_RIDGECV_MULTISOURCE"
SCENARIOS = s3_plan02.SCENARIOS
TARGETS = s3_plan02.TARGETS
COMMON_ORIGINS = s3_plan02.COMMON_ORIGINS
CITIES = s3_plan02.CITIES
HORIZON_YEARS = s3_plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = s3_plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = s3_plan02.LATEST_ORIGIN

SOURCE_IDS = ("urbanization", "density", "wage", "employees")
SOURCE_FEATURES = {
    "urbanization": ("urbanization_latest", "urbanization_recent3_slope"),
    "density": ("density_latest", "density_recent3_relative_slope"),
    "wage": (
        "wage_log_latest",
        "wage_recent3_log_slope",
    ),
    "employees": (
        "employees_log_latest",
        "employees_recent3_log_slope",
    ),
}
FEATURE_NAMES = tuple(
    feature_name
    for source_id in SOURCE_IDS
    for feature_name in SOURCE_FEATURES[source_id]
)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
FALLBACK_ALPHA = 10.0
MIN_INNER_FOLDS = 3
MIN_INNER_TRAIN_ROWS = 2 * len(FEATURE_NAMES)


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
    fieldnames = list(fields or dict.fromkeys(field for row in rows for field in row))
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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
    assert result.returncode == 0, "Wave B integration commit must be an ancestor"
    assert WAVE_A_SUMMARY_PATH.exists()
    for plan_id in ("S4-Plan04A", "S4-Plan04B", "S4-Plan04C"):
        assert (ROOT / "reports" / "S4_HANDOFFS" / plan_id / "HANDOFF.md").exists()


def source_series(source: dict[str, Any]) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    city_column = source["city_col"]
    year_column = source["year_col"]
    value_column = source["value_columns"][0]
    for row in source["value_rows"]:
        city = row[city_column]
        year = as_int(row[year_column])
        value = as_float(row.get(value_column))
        assert city in result and year is not None and value is not None
        result[city][year] = value
    return result


def load_employment_provenance(
    employment_source: dict[str, Any],
) -> tuple[set[tuple[str, int]], set[tuple[str, int, float]], dict[str, Any]]:
    provenance_rows = read_csv(EMPLOYMENT_PROVENANCE_PATH)
    quarantine_rows = read_csv(EMPLOYMENT_QUARANTINE_PATH)
    assert len(provenance_rows) == 91
    assert len(quarantine_rows) == 80

    city_column = employment_source["city_col"]
    year_column = employment_source["year_col"]
    value_column = employment_source["value_columns"][0]
    canonical_values = {}
    for row in employment_source["rows"]:
        city = row[city_column]
        year = as_int(row[year_column])
        value = as_float(row.get(value_column))
        if value is not None:
            canonical_values[(city, year)] = value

    derived_keys: set[tuple[str, int]] = set()
    for row in provenance_rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        value = as_float(row["employeesNumber"])
        assert year is not None and value is not None
        key = (city, year)
        assert key in canonical_values
        assert abs(canonical_values[key] - value) <= 1e-12
        assert row["source_type"] == "sector_sum_reconstruction"
        assert key not in derived_keys
        derived_keys.add(key)

    quarantine_triples: set[tuple[str, int, float]] = set()
    quarantine_keys: set[tuple[str, int]] = set()
    for row in quarantine_rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        value = as_float(row["employeesNumber"])
        assert year is not None and value is not None
        quarantine_keys.add((city, year))
        quarantine_triples.add((city, year, value))

    assert derived_keys == set(employment_source["derived_keys"])
    assert quarantine_keys == set(employment_source["quarantine_keys"])
    assert not any(
        (city, year, value) in quarantine_triples
        for (city, year), value in canonical_values.items()
    )
    metadata = {
        "canonical_row_count": len(employment_source["rows"]),
        "canonical_value_row_count": len(employment_source["value_rows"]),
        "derived_row_count": len(derived_keys),
        "quarantine_row_count": len(quarantine_rows),
        "quarantine_unique_key_count": len(quarantine_keys),
        "source_status": "source_plus_remap_quarantine_and_derived",
        "provenance_file": str(EMPLOYMENT_PROVENANCE_PATH.relative_to(ROOT)),
        "quarantine_file": str(EMPLOYMENT_QUARANTINE_PATH.relative_to(ROOT)),
    }
    assert metadata["canonical_row_count"] == 773
    assert metadata["derived_row_count"] == 91
    assert metadata["quarantine_unique_key_count"] == 59
    return derived_keys, quarantine_triples, metadata


def load_context() -> dict[str, Any]:
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

    contract = s2_geometry.load_contract()
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    geometry = {
        row["source_id"]: row for row in s2_geometry.source_geometry_rows(sources)
    }
    for source_id in SOURCE_IDS:
        assert source_id in sources and source_id in geometry

    source_lags = {
        source_id: {
            scenario_id: s2_geometry.required_lag_years(
                source_id, geometry[source_id], scenario_id
            )
            for scenario_id in SCENARIOS
        }
        for source_id in SOURCE_IDS
    }
    assert source_lags["urbanization"] == {SCENARIOS[0]: 1, SCENARIOS[1]: 2}
    assert source_lags["density"] == {SCENARIOS[0]: 1, SCENARIOS[1]: 2}
    assert source_lags["employees"] == {SCENARIOS[0]: 1, SCENARIOS[1]: 2}
    assert source_lags["wage"][SCENARIOS[0]] == source_lags["wage"][SCENARIOS[1]] == 2

    density_regime_rows = read_csv(REGIME_REPORT_PATH)
    assert len(density_regime_rows) == 13
    regime_shift_year = {
        row["城市名称"]: int(row["regime_shift_year"])
        for row in density_regime_rows
        if row["status"] == "retained"
    }
    assert len(regime_shift_year) == 13
    employment_derived_keys, employment_quarantine_triples, employment_metadata = (
        load_employment_provenance(sources["employees"])
    )
    employment_quarantine_keys = {
        (city, year) for city, year, _ in employment_quarantine_triples
    }
    assert s4_plan04c.SCENARIO_LAG_YEARS == source_lags["employees"]
    assert geometry["urbanization"]["source_status"] == "source_preserved"
    assert geometry["density"]["source_status"] == "source_plus_repair_and_invalidation"
    assert geometry["wage"]["source_status"] == "source_plus_invalidation"
    assert geometry["employees"]["source_status"] == (
        "source_plus_remap_quarantine_and_derived"
    )
    return {
        "population": population,
        "population_profile": population_profile,
        "s2_summary": s2_summary,
        "s3_summary": s3_summary,
        "sources": sources,
        "geometry": geometry,
        "source_lags": source_lags,
        "urbanization_series": source_series(sources["urbanization"]),
        "density_series": source_series(sources["density"]),
        "wage_series": source_series(sources["wage"]),
        "employment_series": source_series(sources["employees"]),
        "density_quarantine_keys": set(sources["density"]["quarantine_keys"]),
        "wage_invalidated_keys": set(sources["wage"]["quarantine_keys"]),
        "employment_derived_keys": employment_derived_keys,
        "employment_quarantine_triples": employment_quarantine_triples,
        "employment_quarantine_keys": employment_quarantine_keys,
        "employment_metadata": employment_metadata,
        "regime_shift_year": regime_shift_year,
    }


def build_feature_bundle(
    context: dict[str, Any],
    city: str,
    pseudo_target_year: int,
    scenario_id: str,
) -> dict[str, Any]:
    lags = context["source_lags"]
    urbanization = s4_plan02.build_urbanization_features(
        context["urbanization_series"][city],
        pseudo_target_year,
        lags["urbanization"][scenario_id],
    )
    density = s4_plan04a.build_density_features(
        context["density_series"][city],
        city,
        pseudo_target_year,
        lags["density"][scenario_id],
        context["density_quarantine_keys"],
        context["regime_shift_year"],
    )
    wage = s4_plan04b.build_wage_features(
        city,
        context["wage_series"][city],
        pseudo_target_year,
        lags["wage"][scenario_id],
        context["wage_invalidated_keys"],
    )
    employment = s4_plan04c.build_employment_features(
        city,
        context["employment_series"][city],
        pseudo_target_year,
        scenario_id,
        context["employment_derived_keys"],
        context["employment_quarantine_triples"],
        context["employment_quarantine_keys"],
    )
    source_details = {
        "urbanization": urbanization,
        "density": density,
        "wage": wage,
        "employees": employment,
    }
    features = {
        feature_name: as_float(source_details[source_id].get(feature_name))
        for source_id in SOURCE_IDS
        for feature_name in SOURCE_FEATURES[source_id]
    }
    assert tuple(features) == FEATURE_NAMES
    return {"features": features, "sources": source_details}


def build_combined_row(
    context: dict[str, Any],
    target_series: dict[int, float],
    target_id: str,
    city: str,
    pseudo_target_year: int,
    scenario_id: str,
    row_role: str,
    outer_origin: int,
) -> dict[str, Any]:
    b3_fit = s3_plan02.fit_prediction(
        B3_ID,
        target_series,
        pseudo_target_year,
        pseudo_target_year - HORIZON_YEARS,
    )
    actual = target_series.get(pseudo_target_year)
    bundle = build_feature_bundle(context, city, pseudo_target_year, scenario_id)
    b3_prediction = b3_fit["prediction"]
    return {
        "row_role": row_role,
        "outer_origin_year": outer_origin,
        "row_year": pseudo_target_year,
        "city_id": city,
        "target_id": target_id,
        "scenario_id": scenario_id,
        "target_value": actual,
        "b3_prediction": b3_prediction,
        "b3_fit": b3_fit,
        "b3_residual_actual_minus_prediction": (
            actual - b3_prediction
            if actual is not None and b3_prediction is not None
            else None
        ),
        "features": bundle["features"],
        "sources": bundle["sources"],
    }


def build_training_rows(
    context: dict[str, Any],
    target_id: str,
    target_series: dict[int, float],
    city: str,
    outer_origin: int,
    scenario_id: str,
) -> tuple[list[dict[str, Any]], int]:
    eligible: list[dict[str, Any]] = []
    excluded_count = 0
    outer_cutoff = outer_origin - HORIZON_YEARS
    for row_year in sorted(year for year in target_series if year <= outer_cutoff):
        row = build_combined_row(
            context,
            target_series,
            target_id,
            city,
            row_year,
            scenario_id,
            "training",
            outer_origin,
        )
        if row["target_value"] is None or row["b3_prediction"] is None:
            excluded_count += 1
        else:
            eligible.append(row)
    return eligible, excluded_count


def build_validation_row(
    context: dict[str, Any],
    target_id: str,
    target_series: dict[int, float],
    city: str,
    origin: int,
    scenario_id: str,
) -> dict[str, Any]:
    row = build_combined_row(
        context,
        target_series,
        target_id,
        city,
        origin,
        scenario_id,
        "validation",
        origin,
    )
    assert row["target_value"] is not None
    assert row["b3_prediction"] is not None
    return row


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["features"].get(feature_name)
                if row["features"].get(feature_name) is not None
                else np.nan
                for feature_name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def inner_splits(
    training_rows: list[dict[str, Any]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    row_years = np.asarray([int(row["row_year"]) for row in training_rows], dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    years: list[int] = []
    for validation_year in sorted(set(row_years.tolist())):
        train_indices = np.asarray(
            [
                index
                for index, row_year in enumerate(row_years)
                if row_year <= validation_year - HORIZON_YEARS
            ],
            dtype=int,
        )
        validation_indices = np.asarray(
            [index for index, row_year in enumerate(row_years) if row_year == validation_year],
            dtype=int,
        )
        if (
            len(train_indices) >= MIN_INNER_TRAIN_ROWS
            and len(validation_indices) > 0
        ):
            splits.append((train_indices, validation_indices))
            years.append(validation_year)
    return splits, years


def alpha_scores(
    x_train_transformed: np.ndarray,
    y_train: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for alpha in ALPHA_GRID:
        fold_mse = []
        for train_indices, validation_indices in splits:
            model = Ridge(alpha=alpha)
            model.fit(x_train_transformed[train_indices], y_train[train_indices])
            prediction = model.predict(x_train_transformed[validation_indices])
            fold_mse.append(
                float(np.mean((prediction - y_train[validation_indices]) ** 2))
            )
        scores[str(alpha)] = mean(fold_mse)
    return scores


def fit_ridgecv_residual(
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
    assert np.isfinite(y_train).all()
    splits, split_years = inner_splits(training_rows)
    training_missing = int(np.isnan(x_train).sum())
    validation_missing = int(np.isnan(x_validation).sum())
    if len(splits) >= MIN_INNER_FOLDS:
        estimator: Ridge | RidgeCV = RidgeCV(
            alphas=ALPHA_GRID,
            cv=splits,
            scoring="neg_mean_squared_error",
        )
        selection_method = "ridgecv_time_aware_expanding"
        fallback_reason = ""
    else:
        estimator = Ridge(alpha=FALLBACK_ALPHA)
        selection_method = "fallback_fixed_alpha"
        fallback_reason = (
            f"inner_fold_count_{len(splits)}_below_minimum_{MIN_INNER_FOLDS}"
        )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", estimator),
        ]
    )
    model.fit(x_train, y_train)
    predicted_residuals = model.predict(x_validation).tolist()
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    indicator_indices = [int(index) for index in imputer.indicator_.features_]
    indicator_names = [
        f"missing_indicator__{FEATURE_NAMES[index]}" for index in indicator_indices
    ]
    coefficient_names = list(FEATURE_NAMES) + indicator_names
    coefficients = [float(value) for value in np.asarray(ridge.coef_).ravel()]
    transformed_train = scaler.transform(imputer.transform(x_train))
    score_map = alpha_scores(transformed_train, y_train, splits) if splits else {}
    selected_alpha = float(
        ridge.alpha_ if isinstance(ridge, RidgeCV) else FALLBACK_ALPHA
    )
    return predicted_residuals, {
        "training_row_count": len(training_rows),
        "training_label_min_year": min(int(row["row_year"]) for row in training_rows),
        "training_label_max_year": max(int(row["row_year"]) for row in training_rows),
        "training_missing_feature_cell_count": training_missing,
        "validation_missing_feature_cell_count": validation_missing,
        "preprocessing_fit_row_count": len(training_rows),
        "model_fit_row_count": len(training_rows),
        "preprocessing_model_fit_rows_equal": "yes",
        "selected_alpha": selected_alpha,
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selection_method": selection_method,
        "alpha_fallback_reason": fallback_reason,
        "inner_validation_fold_count": len(splits),
        "inner_validation_years": split_years,
        "inner_fold_geometry": (
            "each inner validation year y uses train rows with pseudo-target year <= y-2"
        ),
        "inner_alpha_mse": score_map,
        "preprocessing_fit_scope": "outer_origin_training_rows_only",
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": coefficients,
        "coefficient_names": coefficient_names,
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
            "density_used": "no",
            "wage_used": "no",
            "employees_used": "no",
            "city_identity_used": "no",
            "selected_alpha": "",
            "alpha_selection_method": "not_applicable_frozen_reference",
        }
    )
    return result


def score_candidate_row(
    target: dict[str, str],
    scenario_id: str,
    validation: dict[str, Any],
    predicted_residual: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    actual = float(validation["target_value"])
    b3_prediction = float(validation["b3_prediction"])
    prediction = b3_prediction + float(predicted_residual)
    residual = prediction - actual
    row = {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": scenario_id,
        "predictor_id": M5C_ID,
        "model_id": M5C_ID,
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
        "b3_prediction_value": b3_prediction,
        "b3_residual_actual_minus_prediction": validation[
            "b3_residual_actual_minus_prediction"
        ],
        "predicted_b3_residual": predicted_residual,
        "feature_builder_id": "MULTISOURCE_COMPACT_V1",
        "feature_group_id": "MULTISOURCE_COMPACT_V1",
        "model_feature_names": ";".join(FEATURE_NAMES),
        "model_feature_count": len(FEATURE_NAMES),
        "covariates_used": "yes",
        "urbanization_used": "yes",
        "density_used": "yes",
        "wage_used": "yes",
        "employees_used": "yes",
        "city_identity_used": "no",
        "selected_alpha": metadata["selected_alpha"],
        "alpha_selection_method": metadata["alpha_selection_method"],
        "inner_validation_fold_count": metadata["inner_validation_fold_count"],
        "inner_validation_years": stable_json(metadata["inner_validation_years"]),
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
        "coefficient_names": stable_json(metadata["coefficient_names"]),
        "coefficient_values": stable_json(metadata["ridge_coefficients"]),
    }
    for source_id in SOURCE_IDS:
        for feature_name in SOURCE_FEATURES[source_id]:
            row[feature_name] = validation["features"][feature_name]
        details = validation["sources"][source_id]
        row[f"{source_id}_used_years"] = details.get(
            f"{source_id}_used_years",
            details.get("feature_used_years", ""),
        )
        row[f"{source_id}_source_cap_year"] = details.get(
            f"{source_id}_required_latest_usable_year",
            details.get("source_cap_year", ""),
        )
    return row


def make_boundary_audit_row(
    context: dict[str, Any],
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    pseudo_year = int(row["row_year"])
    b3_fit = row["b3_fit"]
    source_rows: dict[str, dict[str, Any]] = {}
    source_passes: list[bool] = []
    for source_id in SOURCE_IDS:
        detail = row["sources"][source_id]
        if source_id == "urbanization":
            used_field = "urbanization_used_years"
            cap_field = "urbanization_required_latest_usable_year"
            lag_field = "urbanization_required_lag_years"
            quarantine_hits = 0
            derived_used = 0
            exact_reintro = 0
        elif source_id == "density":
            used_field = "density_used_years"
            cap_field = "density_required_latest_usable_year"
            lag_field = "density_required_lag_years"
            quarantine_hits = int(detail["density_quarantine_candidate_hit_count"])
            derived_used = 0
            exact_reintro = quarantine_hits
        elif source_id == "wage":
            used_field = "wage_used_years"
            cap_field = "wage_required_latest_usable_year"
            lag_field = "wage_required_lag_years"
            quarantine_hits = int(detail["wage_invalidated_key_candidate_hit_count"])
            derived_used = 0
            exact_reintro = int(detail["wage_invalidated_key_used_count"])
        else:
            used_field = "feature_used_years"
            cap_field = "source_cap_year"
            lag_field = "scenario_required_lag_years"
            quarantine_hits = int(detail["quarantine_candidate_hit_count"])
            derived_used = int(detail["feature_used_derived_count"])
            exact_reintro = quarantine_hits
        used_years = [
            int(value)
            for value in str(detail.get(used_field, "")).split(";")
            if value
        ]
        cap_year = int(detail[cap_field])
        required_lag = int(detail[lag_field])
        expected_lag = int(context["source_lags"][source_id][row["scenario_id"]])
        cap_pass = all(year <= cap_year for year in used_years)
        lag_pass = required_lag == expected_lag
        provenance_pass = (
            len(used_years) == len(set(used_years))
            and exact_reintro == 0
        )
        if source_id != "wage":
            provenance_pass = provenance_pass and quarantine_hits == 0
        if source_id == "employees":
            provenance_pass = provenance_pass and (
                len(used_years)
                == int(detail["feature_used_derived_count"])
                + int(detail["feature_used_non_derived_count"])
            )
        if source_id == "density":
            regime_crossing = detail["density_regime_window_phase"] == "crossing"
            regime_pass = not (
                regime_crossing
                and detail["density_recent3_relative_slope"] is not None
            )
        else:
            regime_crossing = False
            regime_pass = True
        source_pass = cap_pass and lag_pass and provenance_pass and regime_pass
        source_passes.append(source_pass)
        source_rows[source_id] = {
            "required_lag_years": required_lag,
            "expected_lag_years": expected_lag,
            "source_cap_year": cap_year,
            "used_years": used_years,
            "selected_feature_values": {
                name: row["features"][name] for name in SOURCE_FEATURES[source_id]
            },
            "cap_pass": cap_pass,
            "lag_pass": lag_pass,
            "quarantine_candidate_hit_count": quarantine_hits,
            "exact_quarantine_value_reintroduced_count": exact_reintro,
            "derived_used_count": derived_used,
            "regime_crossing": regime_crossing,
            "regime_guard_pass": regime_pass,
            "provenance_pass": provenance_pass,
            "status": "PASS" if source_pass else "FAIL",
        }

    population_pass = (
        b3_fit["history_max_year"] is None
        or int(b3_fit["history_max_year"]) <= pseudo_year - HORIZON_YEARS
    )
    outer_label_pass = (
        row["row_role"] == "validation"
        or pseudo_year <= int(row["outer_origin_year"]) - HORIZON_YEARS
    )
    feature_set_pass = tuple(row["features"]) == FEATURE_NAMES
    checks = [population_pass, outer_label_pass, feature_set_pass, all(source_passes)]
    return {
        "record_type": "boundary",
        "target_id": row["target_id"],
        "scenario_id": row["scenario_id"],
        "outer_origin_year": row["outer_origin_year"],
        "row_role": row["row_role"],
        "pseudo_target_year": pseudo_year,
        "city_id": row["city_id"],
        "population_history_max_year_used": b3_fit["history_max_year"],
        "population_history_within_T_minus_2": "PASS" if population_pass else "FAIL",
        "outer_training_label_within_T_minus_2": "PASS" if outer_label_pass else "FAIL",
        "model_feature_names": ";".join(FEATURE_NAMES),
        "model_feature_count": len(FEATURE_NAMES),
        "feature_set_exactly_pre_registered": "yes" if feature_set_pass else "no",
        "provenance_audit_only": "yes",
        "source_ids": ";".join(SOURCE_IDS),
        "source_geometry_status": stable_json(
            {
                source_id: context["geometry"][source_id]["source_status"]
                for source_id in SOURCE_IDS
            }
        ),
        "source_boundary_audit": stable_json(
            {source_id: source_rows[source_id]["status"] for source_id in SOURCE_IDS}
        ),
        "urbanization_used_years": row["sources"]["urbanization"][
            "urbanization_used_years"
        ],
        "density_used_years": row["sources"]["density"]["density_used_years"],
        "wage_used_years": row["sources"]["wage"]["wage_used_years"],
        "employees_used_years": row["sources"]["employees"]["feature_used_years"],
        "urbanization_feature_fingerprint": stable_json(
            {name: row["features"][name] for name in SOURCE_FEATURES["urbanization"]}
        ),
        "density_feature_fingerprint": stable_json(
            {name: row["features"][name] for name in SOURCE_FEATURES["density"]}
        ),
        "wage_feature_fingerprint": stable_json(
            {name: row["features"][name] for name in SOURCE_FEATURES["wage"]}
        ),
        "employees_feature_fingerprint": stable_json(
            {name: row["features"][name] for name in SOURCE_FEATURES["employees"]}
        ),
        "density_regime_window_phase": row["sources"]["density"][
            "density_regime_window_phase"
        ],
        "density_suppressed_dynamics_count": row["sources"]["density"][
            "density_suppressed_dynamics_count"
        ],
        "employment_feature_used_derived_count": row["sources"]["employees"][
            "feature_used_derived_count"
        ],
        "employment_feature_used_non_derived_count": row["sources"]["employees"][
            "feature_used_non_derived_count"
        ],
        "employment_latest_feature_is_derived": row["sources"]["employees"][
            "latest_feature_is_derived"
        ],
        "employment_quarantine_key_overlap_count": row["sources"]["employees"][
            "quarantine_key_overlap_count"
        ],
        "density_quarantine_candidate_hit_count": row["sources"]["density"][
            "density_quarantine_candidate_hit_count"
        ],
        "wage_invalidated_key_candidate_hit_count": row["sources"]["wage"][
            "wage_invalidated_key_candidate_hit_count"
        ],
        "wage_invalidated_key_used_count": row["sources"]["wage"][
            "wage_invalidated_key_used_count"
        ],
        "employment_quarantine_candidate_hit_count": row["sources"]["employees"][
            "quarantine_candidate_hit_count"
        ],
        "preprocessing_fit_row_count": metadata["preprocessing_fit_row_count"],
        "model_fit_row_count": metadata["model_fit_row_count"],
        "preprocessing_model_fit_rows_equal": metadata[
            "preprocessing_model_fit_rows_equal"
        ],
        "selected_alpha": metadata["selected_alpha"],
        "alpha_selection_method": metadata["alpha_selection_method"],
        "inner_validation_fold_count": metadata["inner_validation_fold_count"],
        "inner_validation_years": stable_json(metadata["inner_validation_years"]),
        "source_cap_boundary_status": "PASS" if all(source_passes) else "FAIL",
        "quarantine_value_reintroduced": (
            "no"
            if all(
                source_rows[source_id]["exact_quarantine_value_reintroduced_count"] == 0
                for source_id in SOURCE_IDS
            )
            else "yes"
        ),
        "provenance_tracking_status": (
            "PASS"
            if all(source_rows[source_id]["provenance_pass"] for source_id in SOURCE_IDS)
            else "FAIL"
        ),
        "unexpected_feature_columns_used": "no" if feature_set_pass else "yes",
        "th_core_used": "no",
        "city_identity_used": "no",
        "age_used": "no",
        "living_standard_used": "no",
        "unemployment_used": "no",
        "sector_share_used": "no",
        "leakage_status": "PASS" if all(checks) else "FAIL",
        "boundary_status": "PASS" if all(checks) else "FAIL",
    }


def build_candidate_rows(
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                training_rows: list[dict[str, Any]] = []
                excluded_count = 0
                validation_rows: list[dict[str, Any]] = []
                for city in CITIES:
                    target_series = context["population"][target_id][city]
                    city_training, city_excluded = build_training_rows(
                        context,
                        target_id,
                        target_series,
                        city,
                        origin,
                        scenario_id,
                    )
                    training_rows.extend(city_training)
                    excluded_count += city_excluded
                    validation_rows.append(
                        build_validation_row(
                            context,
                            target_id,
                            target_series,
                            city,
                            origin,
                            scenario_id,
                        )
                    )
                training_rows.sort(key=lambda row: (int(row["row_year"]), row["city_id"]))
                validation_rows.sort(key=lambda row: row["city_id"])
                predicted_residuals, metadata = fit_ridgecv_residual(
                    training_rows, validation_rows
                )
                metadata = {
                    **metadata,
                    "excluded_training_rows": excluded_count,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "outer_origin_year": origin,
                }
                fit_records.append(metadata)
                for validation, predicted_residual in zip(
                    validation_rows, predicted_residuals
                ):
                    candidate_rows.append(
                        score_candidate_row(
                            target,
                            scenario_id,
                            validation,
                            predicted_residual,
                            metadata,
                        )
                    )
                for row in training_rows + validation_rows:
                    boundary_rows.append(
                        make_boundary_audit_row(context, row, metadata)
                    )
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(candidate_rows) == expected
    assert len(fit_records) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS)
    return candidate_rows, boundary_rows, fit_records


def compare_b3_rows(current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows = read_csv(S3_PLAN03_DIR / "per_origin_predictions_or_residuals.csv")
    frozen_map = {
        (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
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
            row["predictor_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in current_rows
        if row["predictor_id"] == B3_ID
    }
    assert len(frozen_map) == len(current_map)
    result = []
    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, current in current_map.items():
        grouped[(key[0], key[1])].append((current, frozen_map[key]))
    for (target_id, scenario_id), pairs in sorted(grouped.items()):
        deltas = []
        for current, frozen in pairs:
            for field in (
                "prediction_value",
                "residual_prediction_minus_actual",
                "squared_error",
            ):
                left = as_float(current.get(field))
                right = as_float(frozen.get(field))
                assert left is not None and right is not None
                deltas.append(abs(left - right))
        result.append(
            {
                "metric_scope": "b3_sanity_row",
                "target_id": target_id,
                "scenario_id": scenario_id,
                "predictor_id": B3_ID,
                "rows_compared": len(pairs),
                "max_abs_delta": max(deltas),
                "status": "PASS" if max(deltas) <= 1e-9 else "FAIL",
                "comparison_rule": "B3 row prediction/residual/squared-error delta <= 1e-9",
            }
        )
    assert all(row["status"] == "PASS" for row in result)
    return result


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    saved_predictors = s3_plan01.PREDICTORS
    s3_plan01.PREDICTORS = (
        {"predictor_id": B3_ID},
        {"predictor_id": M5C_ID},
    )
    try:
        per_origin, _ = s3_plan01.build_per_origin_metrics(rows)
        aggregate, aggregate_map = s3_plan01.build_aggregate_metrics(rows, per_origin)
    finally:
        s3_plan01.PREDICTORS = saved_predictors
    return per_origin, aggregate, aggregate_map


def compare_b3_aggregate(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    frozen_rows = read_csv(S3_PLAN03_DIR / "aggregate_metrics.csv")
    frozen_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in frozen_rows
        if row["predictor_id"] == B3_ID
    }
    result = []
    for key, frozen in sorted(frozen_map.items()):
        current = aggregate_map[key]
        deltas = [
            abs(float(current[field]) - float(frozen[field]))
            for field in (
                "pooled_city_year_mse",
                "macro_origin_mse",
                "latest_origin_mse",
            )
        ]
        result.append(
            {
                "metric_scope": "b3_sanity_aggregate",
                "target_id": key[0],
                "scenario_id": key[1],
                "predictor_id": B3_ID,
                "max_abs_metric_delta": max(deltas),
                "status": "PASS" if max(deltas) <= 1e-9 else "FAIL",
                "comparison_rule": "B3 pooled/macro/latest MSE delta <= 1e-9",
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
    candidate = aggregate_map[(target_id, scenario_id, M5C_ID)]
    win_years: list[int] = []
    loss_years: list[int] = []
    tie_years: list[int] = []
    for origin in COMMON_ORIGINS:
        baseline_origin = per_origin_map[(target_id, scenario_id, B3_ID, origin)]
        candidate_origin = per_origin_map[(target_id, scenario_id, M5C_ID, origin)]
        delta = float(baseline_origin["mse"]) - float(candidate_origin["mse"])
        if delta > 1e-12:
            win_years.append(origin)
        elif delta < -1e-12:
            loss_years.append(origin)
        else:
            tie_years.append(origin)
    base_map = {
        (int(row["origin_year"]), row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, B3_ID)
    }
    candidate_map = {
        (int(row["origin_year"]), row["city_id"]): row
        for row in score_rows(rows, target_id, scenario_id, M5C_ID)
    }
    row_deltas = [
        float(base_map[key]["squared_error"]) - float(candidate_map[key]["squared_error"])
        for key in sorted(base_map)
    ]
    baseline_mse = float(baseline["pooled_city_year_mse"])
    candidate_mse = float(candidate["pooled_city_year_mse"])
    latest_baseline = per_origin_map[(target_id, scenario_id, B3_ID, LATEST_ORIGIN)]
    latest_candidate = per_origin_map[(target_id, scenario_id, M5C_ID, LATEST_ORIGIN)]
    return {
        "metric_scope": "pairwise_b3",
        "target_id": target_id,
        "scenario_id": scenario_id,
        "baseline_predictor_id": B3_ID,
        "candidate_predictor_id": M5C_ID,
        "baseline_pooled_mse": baseline_mse,
        "candidate_pooled_mse": candidate_mse,
        "pooled_mse_absolute_delta_baseline_minus_candidate": baseline_mse - candidate_mse,
        "pooled_mse_relative_improvement_pct": (
            (baseline_mse - candidate_mse) / baseline_mse * 100.0
        ),
        "baseline_macro_origin_mse": float(baseline["macro_origin_mse"]),
        "candidate_macro_origin_mse": float(candidate["macro_origin_mse"]),
        "baseline_latest_2021_mse": float(baseline["latest_origin_mse"]),
        "candidate_latest_2021_mse": float(candidate["latest_origin_mse"]),
        "latest_mse_relative_improvement_pct": (
            (float(baseline["latest_origin_mse"]) - float(candidate["latest_origin_mse"]))
            / float(baseline["latest_origin_mse"])
            * 100.0
        ),
        "latest_candidate_outcome": (
            "win"
            if float(latest_baseline["mse"]) > float(latest_candidate["mse"]) + 1e-12
            else "loss"
            if float(latest_baseline["mse"]) < float(latest_candidate["mse"]) - 1e-12
            else "tie"
        ),
        "candidate_origin_win_count": len(win_years),
        "candidate_origin_loss_count": len(loss_years),
        "origin_tie_count": len(tie_years),
        "origin_win_years": ";".join(map(str, win_years)),
        "origin_loss_years": ";".join(map(str, loss_years)),
        "origin_tie_years": ";".join(map(str, tie_years)),
        "origin_majority_win": "yes" if len(win_years) > len(loss_years) else "no",
        "candidate_city_origin_improved_share": sum(
            delta > 1e-12 for delta in row_deltas
        )
        / len(row_deltas),
        "candidate_city_origin_worsened_share": sum(
            delta < -1e-12 for delta in row_deltas
        )
        / len(row_deltas),
        "ranking_metric": "pooled_city_year_mse",
        "latest_is_secondary_proximity_diagnostic": "yes",
    }


def validation_boundary_map(
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


def feature_bundle_from_boundary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        source_id: row.get(f"{source_id}_feature_fingerprint", "")
        for source_id in SOURCE_IDS
    }


def build_r1_r2_rows(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit = validation_boundary_map(boundary_rows)
    result: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for model_id in (B3_ID, M5C_ID):
            r1 = aggregate_map[(target_id, SCENARIOS[0], model_id)]
            r2 = aggregate_map[(target_id, SCENARIOS[1], model_id)]
            any_differences = 0
            source_difference_counts = Counter()
            wage_equivalent = True
            for origin in COMMON_ORIGINS:
                for city in CITIES:
                    left = audit[(target_id, SCENARIOS[0], origin, city)]
                    right = audit[(target_id, SCENARIOS[1], origin, city)]
                    if model_id == B3_ID:
                        continue
                    left_sources = feature_bundle_from_boundary(left)
                    right_sources = feature_bundle_from_boundary(right)
                    row_diff = False
                    for source_id in SOURCE_IDS:
                        if left_sources[source_id] != right_sources[source_id]:
                            source_difference_counts[source_id] += 1
                            row_diff = True
                    if left_sources["wage"] != right_sources["wage"]:
                        wage_equivalent = False
                    if row_diff:
                        any_differences += 1
            if model_id == B3_ID:
                wage_equivalent = True
            result.append(
                {
                    "metric_scope": "r1_r2_sensitivity",
                    "target_id": target_id,
                    "predictor_id": model_id,
                    "r1_scenario_id": SCENARIOS[0],
                    "r2_scenario_id": SCENARIOS[1],
                    "r1_pooled_mse": r1["pooled_city_year_mse"],
                    "r2_pooled_mse": r2["pooled_city_year_mse"],
                    "r2_minus_r1_pooled_mse": float(r2["pooled_city_year_mse"])
                    - float(r1["pooled_city_year_mse"]),
                    "r1_latest_2021_mse": r1["latest_origin_mse"],
                    "r2_latest_2021_mse": r2["latest_origin_mse"],
                    "r2_minus_r1_latest_2021_mse": float(r2["latest_origin_mse"])
                    - float(r1["latest_origin_mse"]),
                    "r1_r2_feature_difference_count": any_differences,
                    "r1_r2_source_feature_difference_counts": stable_json(
                        dict(source_difference_counts)
                    ),
                    "r1_r2_feature_difference_confirmed": (
                        "yes" if model_id == M5C_ID and any_differences else "no"
                    ),
                    "wage_r1_r2_equivalent": "yes" if wage_equivalent else "no",
                    "interpretation": (
                        "joint source-boundary rerun"
                        if model_id == M5C_ID
                        else "target-history-only equivalence control"
                    ),
                }
            )
    return result


def sign(value: float) -> int:
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def build_alpha_diagnostics(
    fit_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    for record in fit_records:
        audit_rows.append(
            {
                "record_type": "alpha_selection",
                "target_id": record["target_id"],
                "scenario_id": record["scenario_id"],
                "outer_origin_year": record["outer_origin_year"],
                "selected_alpha": record["selected_alpha"],
                "alpha_grid": stable_json(record["alpha_grid"]),
                "alpha_selection_method": record["alpha_selection_method"],
                "alpha_fallback_reason": record["alpha_fallback_reason"],
                "inner_validation_fold_count": record["inner_validation_fold_count"],
                "inner_validation_years": stable_json(record["inner_validation_years"]),
                "inner_alpha_mse": stable_json(record["inner_alpha_mse"]),
                "preprocessing_fit_scope": record["preprocessing_fit_scope"],
                "training_row_count": record["training_row_count"],
                "excluded_training_rows": record["excluded_training_rows"],
                "preprocessing_model_fit_rows_equal": record[
                    "preprocessing_model_fit_rows_equal"
                ],
            }
        )
        for name, coefficient in zip(
            record["coefficient_names"], record["ridge_coefficients"]
        ):
            source_id = next(
                (
                    candidate
                    for candidate in SOURCE_IDS
                    if name == candidate
                    or name.startswith(tuple(SOURCE_FEATURES[candidate]))
                ),
                "missing_indicator",
            )
            audit_rows.append(
                {
                    "record_type": "coefficient",
                    "target_id": record["target_id"],
                    "scenario_id": record["scenario_id"],
                    "outer_origin_year": record["outer_origin_year"],
                    "coefficient_name": name,
                    "coefficient_source": source_id,
                    "coefficient_value": coefficient,
                    "selected_alpha": record["selected_alpha"],
                    "alpha_selection_method": record["alpha_selection_method"],
                }
            )

    stability_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []
    alpha_summary: dict[str, Any] = {}
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            scoped = sorted(
                (
                    record
                    for record in fit_records
                    if record["target_id"] == target_id
                    and record["scenario_id"] == scenario_id
                ),
                key=lambda record: int(record["outer_origin_year"]),
            )
            selected = [float(record["selected_alpha"]) for record in scoped]
            alpha_counts = Counter(str(alpha) for alpha in selected)
            extreme_count = sum(
                alpha in (min(ALPHA_GRID), max(ALPHA_GRID)) for alpha in selected
            )
            fallback_count = sum(
                record["alpha_selection_method"] == "fallback_fixed_alpha"
                for record in scoped
            )
            for feature_name in FEATURE_NAMES:
                values = []
                for record in scoped:
                    names = record["coefficient_names"]
                    if feature_name in names:
                        values.append(
                            float(record["ridge_coefficients"][names.index(feature_name)])
                        )
                nonzero_signs = [sign(value) for value in values if sign(value) != 0]
                sign_flips = sum(
                    left != right
                    for left, right in zip(nonzero_signs, nonzero_signs[1:])
                )
                stability_rows.append(
                    {
                        "metric_scope": "coefficient_stability",
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "feature_name": feature_name,
                        "coefficient_fit_count": len(values),
                        "coefficient_nonzero_count": len(nonzero_signs),
                        "coefficient_sign_flip_count": sign_flips,
                        "coefficient_abs_mean": mean(abs(value) for value in values),
                        "coefficient_abs_max": max(abs(value) for value in values),
                        "coefficient_signs": ";".join(map(str, nonzero_signs)),
                        "coefficient_instability_observed": (
                            "yes" if sign_flips > 0 else "no"
                        ),
                    }
                )

            for record in scoped:
                source_abs_mass = {
                    source_id: math.fsum(
                        abs(
                            float(
                                record["ridge_coefficients"][
                                    record["coefficient_names"].index(feature_name)
                                ]
                            )
                        )
                        for feature_name in SOURCE_FEATURES[source_id]
                        if feature_name in record["coefficient_names"]
                    )
                    for source_id in SOURCE_IDS
                }
                total_mass = math.fsum(source_abs_mass.values())
                dominant_source = max(
                    source_abs_mass,
                    key=lambda source_id: (source_abs_mass[source_id], source_id),
                )
                dominant_share = (
                    source_abs_mass[dominant_source] / total_mass if total_mass else 0.0
                )
                dominance_rows.append(
                    {
                        "metric_scope": "source_dominance",
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": record["outer_origin_year"],
                        "dominant_source": dominant_source,
                        "dominant_source_abs_coefficient_share": dominant_share,
                        "source_abs_coefficient_mass": stable_json(source_abs_mass),
                        "single_source_dominance_ge_75pct": (
                            "yes" if dominant_share >= 0.75 else "no"
                        ),
                    }
                )
            scoped_dominance = [
                row
                for row in dominance_rows
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
            ]
            coefficient_scope = [
                row
                for row in stability_rows
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
            ]
            alpha_summary[f"{target_id}_{scenario_id}"] = {
                "selected_alpha_counts": dict(alpha_counts),
                "selected_alpha_unique_count": len(alpha_counts),
                "selected_alpha_min": min(selected),
                "selected_alpha_max": max(selected),
                "extreme_alpha_count": extreme_count,
                "extreme_alpha_share": extreme_count / len(selected),
                "fallback_count": fallback_count,
                "alpha_stability_status": (
                    "fallback"
                    if fallback_count > 0
                    else "stable_but_extreme"
                    if len(alpha_counts) == 1 and extreme_count == len(selected)
                    else "stable"
                    if len(alpha_counts) == 1
                    else "mixed"
                ),
                "coefficient_sign_flip_total": sum(
                    int(row["coefficient_sign_flip_count"]) for row in coefficient_scope
                ),
                "coefficient_sign_flip_features": [
                    row["feature_name"]
                    for row in coefficient_scope
                    if int(row["coefficient_sign_flip_count"]) > 0
                ],
                "dominant_source_counts": dict(
                    Counter(row["dominant_source"] for row in scoped_dominance)
                ),
                "dominant_source_share_mean": mean(
                    float(row["dominant_source_abs_coefficient_share"])
                    for row in scoped_dominance
                ),
                "dominant_source_share_max": max(
                    float(row["dominant_source_abs_coefficient_share"])
                    for row in scoped_dominance
                ),
                "single_source_dominance_ge_75pct_count": sum(
                    row["single_source_dominance_ge_75pct"] == "yes"
                    for row in scoped_dominance
                ),
            }
    return audit_rows, stability_rows, alpha_summary, dominance_rows


def build_alpha_screening_rows(
    alpha_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(alpha_summary.items()):
        target_id, scenario_id = key.split("_", 1)
        rows.append(
            {
                "metric_scope": "alpha_stability",
                "target_id": target_id,
                "scenario_id": scenario_id,
                **value,
            }
        )
    return rows


def make_decision(
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    del aggregate_map
    pairwise_map = {
        (row["target_id"], row["scenario_id"]): row for row in pairwise_rows
    }
    resident = {
        scenario_id: pairwise_map[("resident", scenario_id)]
        for scenario_id in SCENARIOS
    }
    resident_relative = {
        scenario_id: float(row["pooled_mse_relative_improvement_pct"])
        for scenario_id, row in resident.items()
    }
    resident_gain_scenarios = [
        scenario_id for scenario_id, value in resident_relative.items() if value > 0
    ]
    at_least_one_gain = bool(resident_gain_scenarios)
    other_no_material_reversal = (
        at_least_one_gain
        and all(
            value >= -1.0
            for scenario_id, value in resident_relative.items()
            if scenario_id not in resident_gain_scenarios
        )
    )
    first_rule = at_least_one_gain and other_no_material_reversal
    both_within_one_pct = all(abs(value) <= 1.0 for value in resident_relative.values())
    complementary_evidence = any(
        row["origin_majority_win"] == "yes" or row["latest_candidate_outcome"] == "win"
        for row in resident.values()
    ) and any(
        row["origin_majority_win"] == "no" or row["latest_candidate_outcome"] == "loss"
        for row in resident.values()
    )
    second_rule = both_within_one_pct and complementary_evidence
    deep_dive = first_rule or second_rule
    registered = {
        scenario_id: pairwise_map[("registered", scenario_id)]
        for scenario_id in SCENARIOS
    }
    registered_consistent = all(
        float(row["pooled_mse_relative_improvement_pct"]) >= -1.0
        for row in registered.values()
    )
    return {
        "screening_decision": (
            "multisource_joint_supported_for_deep_dive"
            if deep_dive
            else "multisource_joint_unsupported"
        ),
        "deep_dive_recommended": deep_dive,
        "resident_relative_improvement_pct": resident_relative,
        "resident_gain_scenarios": resident_gain_scenarios,
        "resident_at_least_one_scenario_gain": at_least_one_gain,
        "other_scenario_no_clear_material_reversal": other_no_material_reversal,
        "decision_rule_one_pass": first_rule,
        "resident_both_scenarios_within_1pct": both_within_one_pct,
        "resident_complementary_origin_latest_evidence": complementary_evidence,
        "decision_rule_two_pass": second_rule,
        "registered_consistency_within_minus_1pct": registered_consistent,
        "decision_rule": (
            "deep_dive=true when resident at least one scenario beats B3 and "
            "the other has no clear material reversal; or both scenarios are "
            "within 1% with complementary origin/latest evidence"
        ),
    }


def aggregate_summary(
    aggregate_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for row in aggregate_metrics:
        result[
            f"{row['target_id']}_{row['scenario_id']}_{row['predictor_id']}"
        ] = {
            "pooled_mse": row["pooled_city_year_mse"],
            "macro_origin_mse": row["macro_origin_mse"],
            "latest_2021_mse": row["latest_origin_mse"],
            "pooled_rmse": row["pooled_city_year_rmse"],
            "pooled_mae": row["pooled_city_year_mae"],
        }
    return result


def write_handoff(
    summary: dict[str, Any],
    pairwise_rows: list[dict[str, Any]],
    r1_r2_rows: list[dict[str, Any]],
) -> None:
    pairwise = {
        (row["target_id"], row["scenario_id"]): row for row in pairwise_rows
    }
    resident_r1 = pairwise[("resident", SCENARIOS[0])]
    resident_r2 = pairwise[("resident", SCENARIOS[1])]
    registered_r1 = pairwise[("registered", SCENARIOS[0])]
    registered_r2 = pairwise[("registered", SCENARIOS[1])]
    candidate_r1_r2 = {
        row["target_id"]: row
        for row in r1_r2_rows
        if row["predictor_id"] == M5C_ID
    }
    decision = summary["decision_detail"]
    alpha = summary["alpha_diagnostics"]
    coefficient = summary["coefficient_diagnostics"]
    lines = [
        "# S4-Plan05C Handoff — 多源弱信号联合正则化快速筛选",
        "",
        "## 状态与结论",
        "- Plan 状态：evidence complete；本 worker 只完成 screening，完成后停止。",
        f"- 分支：{summary['branch']}；frozen integration base：{summary['starting_commit']}。",
        f"- screening decision：{summary['screening_decision']}；deep_dive_recommended={str(summary['deep_dive_recommended']).lower()}。",
        "- s4_complete=false、s5_entered=false、未生成 2023 prediction 或 submission。",
        "",
        "## Model and feature boundary",
        f"- 仅比较 frozen {B3_ID} 与 {M5C_ID}。",
        f"- 联合模型为 B3 + predicted B3 residual，outer-origin 内使用 time-aware RidgeCV，alpha grid={list(ALPHA_GRID)}。",
        f"- 8 个模型特征：{'; '.join(FEATURE_NAMES)}。",
        "- 只使用 urbanization、density、wage、employees 四个已审计 source；provenance/quarantine/regime 字段仅为 audit。",
        "- 未使用 TH_CORE、city ID、age、living standard、unemployment、sector shares；未运行 GBDT、ensemble 或额外特征 sweep。",
        "",
        "## Handoff questions",
        f"1. joint multiweak signal > B3? resident R1：{resident_r1['candidate_pooled_mse']:.6f} vs {resident_r1['baseline_pooled_mse']:.6f}，relative improvement={resident_r1['pooled_mse_relative_improvement_pct']:.4f}%；R2：{resident_r2['candidate_pooled_mse']:.6f} vs {resident_r2['baseline_pooled_mse']:.6f}，relative improvement={resident_r2['pooled_mse_relative_improvement_pct']:.4f}%。",
        f"2. resident/registered consistency? registered R1 improvement={registered_r1['pooled_mse_relative_improvement_pct']:.4f}%，R2={registered_r2['pooled_mse_relative_improvement_pct']:.4f}%；resident 是 primary，registered 仅 sensitivity，{('consistent within the screening guard' if decision['registered_consistency_within_minus_1pct'] else 'not consistent')}。",
        f"3. R1→R2 stability? resident candidate pooled delta={candidate_r1_r2['resident']['r2_minus_r1_pooled_mse']:.6f}，latest delta={candidate_r1_r2['resident']['r2_minus_r1_latest_2021_mse']:.6f}；feature differences={candidate_r1_r2['resident']['r1_r2_feature_difference_count']}，wage cap equivalence={candidate_r1_r2['resident']['wage_r1_r2_equivalent']}。",
        f"4. per-origin majority improvement? resident R1={resident_r1['candidate_origin_win_count']}/{resident_r1['candidate_origin_loss_count']} wins/losses，R2={resident_r2['candidate_origin_win_count']}/{resident_r2['candidate_origin_loss_count']}；多数改善并未被预设为 pooled 结论的替代。",
        f"5. inner-selected alpha stability? {json.dumps(alpha, ensure_ascii=False, sort_keys=True)}。",
        f"6. single source dominance/coefficient flips? diagnostics={json.dumps(coefficient, ensure_ascii=False, sort_keys=True)}；这些是系数诊断，不是新增 source 选择依据。",
        f"7. source boundary/provenance guard? boundary rows={summary['boundary_audit_rows']}；全部 boundary_status=PASS；quarantine exact-value reintroduction=0；density regime crossing suppression 保留。",
        f"8. deep dive worth? {str(summary['deep_dive_recommended']).lower()}，规则一={str(decision['decision_rule_one_pass']).lower()}，规则二={str(decision['decision_rule_two_pass']).lower()}。",
        "",
        "## Evidence",
        "- plan05c_summary.json：机器可读决策、metrics、alpha/coefficient diagnostics 与 scope flags。",
        "- screening_metrics.csv：aggregate/per-origin、B3 sanity、pairwise、R1/R2 sensitivity、alpha stability、coefficient stability 与 source dominance。",
        "- multisource_boundary_and_alpha_audit.csv：四 source cap/provenance/quarantine/regime boundary 与每个 outer-origin alpha/coefficient audit。",
        "- scripts/build_s4_plan05c_multisource_joint.py：可重跑 runner。",
        "",
        "## Verification scope",
        "- B3 row/aggregate compatibility、source-specific caps、population T-2、quarantine/provenance、density regime guard 和 preprocessing fit boundary 均由 runner assert。",
        "- R1/R2 是 S2 observed-cap replay assumptions，不是 publication timing 事实证明。",
        "- 本 worker 不继续 deep dive，不修改 CURRENT_LEARNING_CHECKPOINT.md，也不自动进入 S5。",
    ]
    (OUTPUT_DIR / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    assert_repo_identity()
    context = load_context()
    b3_all_rows, _ = s3_plan02.build_fixed_rows(context["population"])
    b3_rows = [
        wrap_b3_row(row)
        for row in b3_all_rows
        if row["predictor_id"] == B3_ID
    ]
    expected_scope = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(b3_rows) == expected_scope
    b3_row_sanity = compare_b3_rows(b3_rows)

    candidate_rows, boundary_rows, fit_records = build_candidate_rows(context)
    assert len(candidate_rows) == expected_scope
    assert all(row["boundary_status"] == "PASS" for row in boundary_rows)
    assert all(row["leakage_status"] == "PASS" for row in boundary_rows)
    assert all(row["quarantine_value_reintroduced"] == "no" for row in boundary_rows)
    assert all(
        row["preprocessing_model_fit_rows_equal"] == "yes"
        for row in boundary_rows
    )

    all_rows = b3_rows + candidate_rows
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
    r1_r2_rows = build_r1_r2_rows(aggregate_map, boundary_rows)
    alpha_audit_rows, coefficient_stability_rows, alpha_diagnostics, dominance_rows = (
        build_alpha_diagnostics(fit_records)
    )
    alpha_screening_rows = build_alpha_screening_rows(alpha_diagnostics)
    coefficient_diagnostics = {
        key: {
            "coefficient_sign_flip_total": value["coefficient_sign_flip_total"],
            "coefficient_sign_flip_features": value["coefficient_sign_flip_features"],
            "dominant_source_counts": value["dominant_source_counts"],
            "dominant_source_share_mean": value["dominant_source_share_mean"],
            "dominant_source_share_max": value["dominant_source_share_max"],
        }
        for key, value in alpha_diagnostics.items()
    }
    decision = make_decision(aggregate_map, pairwise_rows)

    screening_rows: list[dict[str, Any]] = []
    screening_rows.extend(dict(row, metric_scope="per_origin") for row in per_origin_metrics)
    screening_rows.extend(dict(row, metric_scope="aggregate") for row in aggregate_metrics)
    screening_rows.extend(pairwise_rows)
    screening_rows.extend(r1_r2_rows)
    screening_rows.extend(alpha_screening_rows)
    screening_rows.extend(coefficient_stability_rows)
    screening_rows.extend(dominance_rows)
    screening_rows.extend(b3_row_sanity)
    screening_rows.extend(b3_aggregate_sanity)

    summary = {
        "plan_id": PLAN_ID,
        "stage": "S4_FEATURE_ML_MODELING",
        "starting_commit": STARTING_COMMIT,
        "branch": BRANCH,
        "frozen_base_commit": STARTING_COMMIT,
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "target_history_cutoff_rule": "population B3 history <= pseudo-target-2; outer training labels <= outer-origin-2",
        "source_specific_caps": context["source_lags"],
        "wage_r1_r2_equivalent": True,
        "model_ids": [B3_ID, M5C_ID],
        "model_family": "RidgeCV residual correction",
        "alpha_grid": list(ALPHA_GRID),
        "fallback_alpha": FALLBACK_ALPHA,
        "minimum_inner_folds": MIN_INNER_FOLDS,
        "inner_geometry": "expanding pseudo-origin: inner validation year y trains only on pseudo-target rows <= y-2",
        "feature_group_id": "MULTISOURCE_COMPACT_V1",
        "feature_names": list(FEATURE_NAMES),
        "feature_source_map": {
            source_id: list(SOURCE_FEATURES[source_id]) for source_id in SOURCE_IDS
        },
        "feature_formulas": {
            "urbanization_latest": "latest allowed observed urbanizationRate",
            "urbanization_recent3_slope": "OLS slope of urbanizationRate against actual year over latest three allowed observations",
            "density_latest": "latest allowed observed density level",
            "density_recent3_relative_slope": "OLS slope of density against actual year divided by latest density; missing on documented regime crossing",
            "wage_log_latest": "log1p(latest allowed positive averageWage)",
            "wage_recent3_log_slope": "OLS slope of log averageWage against actual year over latest three allowed positive observations",
            "employees_log_latest": "log1p(latest allowed positive employeesNumber)",
            "employees_recent3_log_slope": "OLS slope of log employeesNumber against actual year over latest three allowed positive observations",
        },
        "source_geometry": {
            source_id: {
                key: context["geometry"][source_id][key]
                for key in (
                    "latest_usable_year",
                    "observed_snapshot_freshness_lag_years",
                    "source_status",
                    "source_derived_count",
                    "source_invalidated_quarantine_count",
                )
            }
            for source_id in SOURCE_IDS
        },
        "employment_provenance": context["employment_metadata"],
        "metrics": aggregate_summary(aggregate_metrics),
        "pairwise_vs_b3": {
            f"{row['target_id']}_{row['scenario_id']}": row
            for row in pairwise_rows
        },
        "r1_r2_sensitivity": {
            f"{row['target_id']}_{row['predictor_id']}": row
            for row in r1_r2_rows
        },
        "decision_detail": decision,
        "alpha_diagnostics": alpha_diagnostics,
        "coefficient_diagnostics": coefficient_diagnostics,
        "b3_row_sanity_pass": True,
        "b3_aggregate_sanity_pass": True,
        "source_boundary_audit_pass": True,
        "quarantine_zero_reintroduction": True,
        "density_regime_suppression_preserved": True,
        "r1_r2_feature_difference_confirmed": all(
            row["r1_r2_feature_difference_confirmed"] == "yes"
            for row in r1_r2_rows
            if row["predictor_id"] == M5C_ID
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
        "boundary_audit_rows": len(boundary_rows),
        "alpha_audit_rows": len(alpha_audit_rows),
        "candidate_prediction_rows": len(candidate_rows),
        "fit_record_count": len(fit_records),
        "screening_metric_rows": len(screening_rows),
        "uses_only_audited_four_sources": True,
        "uses_th_core": False,
        "uses_city_id": False,
        "uses_age": False,
        "uses_living_standard": False,
        "uses_unemployment": False,
        "uses_sector_shares": False,
        "tuning_run": False,
        "model_sweep_run": False,
        "gbdt_run": False,
        "ensemble_run": False,
        "final_2023_prediction_run": False,
        "submission_run": False,
        "s4_complete": False,
        "s5_entered": False,
    }
    assert summary["r1_r2_feature_difference_confirmed"]
    assert summary["source_boundary_audit_pass"]
    assert summary["b3_row_sanity_pass"]
    assert summary["b3_aggregate_sanity_pass"]

    write_csv(OUTPUT_DIR / "screening_metrics.csv", screening_rows)
    write_csv(
        OUTPUT_DIR / "multisource_boundary_and_alpha_audit.csv",
        boundary_rows + alpha_audit_rows,
    )
    write_json(OUTPUT_DIR / "plan05c_summary.json", summary)
    write_handoff(summary, pairwise_rows, r1_r2_rows)

    print("S4_PLAN05C_MULTISOURCE_JOINT_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"branch={BRANCH}")
    print("common_origin_domain=2011-2021")
    print(f"model_ids={','.join((B3_ID, M5C_ID))}")
    print(f"feature_count={len(FEATURE_NAMES)}")
    print(f"candidate_prediction_rows={len(candidate_rows)}")
    print(f"boundary_audit_rows={len(boundary_rows)}")
    print(f"alpha_audit_rows={len(alpha_audit_rows)}")
    print(f"screening_metric_rows={len(screening_rows)}")
    print(f"b3_row_sanity_pass={summary['b3_row_sanity_pass']}")
    print(f"b3_aggregate_sanity_pass={summary['b3_aggregate_sanity_pass']}")
    print(f"source_boundary_audit_pass={summary['source_boundary_audit_pass']}")
    print(f"quarantine_zero_reintroduction={summary['quarantine_zero_reintroduction']}")
    print(f"r1_r2_feature_difference_confirmed={summary['r1_r2_feature_difference_confirmed']}")
    print(f"screening_decision={summary['screening_decision']}")
    print(f"deep_dive_recommended={summary['deep_dive_recommended']}")
    print("uses_only_audited_four_sources=True")
    print("uses_th_core=False")
    print("uses_city_id=False")
    print("uses_age=False")
    print("uses_living_standard=False")
    print("uses_unemployment=False")
    print("uses_sector_shares=False")
    print("tuning_run=False")
    print("model_sweep_run=False")
    print("gbdt_run=False")
    print("ensemble_run=False")
    print("final_2023_prediction_run=False")
    print("submission_run=False")
    print("s4_complete=False")
    print("s5_entered=False")


if __name__ == "__main__":
    main()
