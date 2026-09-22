#!/usr/bin/env python3
"""S10-Plan02: same-contract replay and temporal selector audit.

This runner is deliberately bounded to the frozen 2011-2021 historical
contract.  It never reads the S8 truth track or any 2022/2023 population
label.  The external script is used only as an immutable semantic reference;
all new replay code lives here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_s2_plan02_geometry as s2_geometry  # noqa: E402
import build_s3_plan01_baseline_ladder as s3_plan01  # noqa: E402


PLAN_ID = "S10-Plan02"
BASE_SHA = "8cb00034791482d34cc3e5d3bc8df7fdcdf9d174"
PLAN01_BRANCH = "codex/s10-plan01-external-script-forensics"
F4_SOURCE_TAG = "archive/pre-s6/s5-plan06e-delta-growth-framing"
F4_SOURCE_PATH = "reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv"
ORIGINAL_SCRIPT = ROOT / "reports" / "S10_HANDOFFS" / "S10-Plan01" / "intake" / "original" / "forecast_population.py"
OUTPUT_DIR = ROOT / "reports" / "S10_HANDOFFS" / PLAN_ID
REFERENCE_B3_PATH = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan02" / "per_origin_predictions_or_residuals.csv"
REFERENCE_COMPARISON_PATH = ROOT / "reports" / "S5_FINAL_CANDIDATE_COMPARISON.csv"

ORIGINS = tuple(range(2011, 2022))
NESTED_ORIGINS = tuple(range(2013, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))
TARGETS = ("resident", "registered")
SCENARIOS = ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS")
MODEL_NAMES = ("ridge", "extra_trees", "random_forest", "hist_gradient", "gradient_huber")
LOCAL_METHODS = (
    "local_last",
    "local_slope3",
    "local_slope5",
    "local_mean_slope5",
    "local_linear3",
    "local_linear5",
)
DENSITY_METHODS = ("density_hold", "density_plus_slope5")
MODEL_METHODS = tuple(f"model_{name}" for name in MODEL_NAMES)
ALL_METHODS = LOCAL_METHODS + DENSITY_METHODS + MODEL_METHODS
SOURCE_SEMANTICS = ("CANONICAL_PROVENANCE_SAFE", "ORIGINAL_SOURCE_SEMANTICS")
PRIMARY_REPAIR = "NO_MANUAL_TARGET_REPAIR"
REPAIR_DIAGNOSTIC = "ORIGINAL_REGISTERED_REPAIR"

SOURCE_FEATURE_COLUMNS = {
    "density": ("density",),
    "urbanization": ("urbanization_rate",),
    "unemployment": ("unemployment_rate",),
    "employees": ("employees_number",),
    "sector": ("primary_employment", "secondary_employment", "tertiary_employment"),
    "wage": ("average_wage",),
    "age": ("age_0_14", "age_15_64", "age_65_plus"),
}
CANONICAL_COLUMN_MAP = {
    "density": {"人口密度（人/平方公里）": "density"},
    "urbanization": {"urbanizationRate": "urbanization_rate"},
    "unemployment": {"unemploymentRate": "unemployment_rate"},
    "employees": {"employeesNumber": "employees_number"},
    "sector": {
        "pi_Employment": "primary_employment",
        "si_Employment": "secondary_employment",
        "ti_Employment": "tertiary_employment",
    },
    "wage": {"averageWage": "average_wage"},
    "age": {"0-14": "age_0_14", "15-64": "age_15_64", "65+": "age_65_plus"},
    "overall_income": {"disposableIncome": "income_disposable"},
    "overall_consumption": {"consumptionExpenditures": "income_consumption"},
    "urban_consumption": {"urbanConsumptionExpenditures": "income_urban_consumption"},
    "rural_consumption": {"ruralConsumptionExpenditures": "income_rural_consumption"},
    "urban_income": {"urbanDisposableIncome": "income_urban_disposable"},
    "rural_income": {"ruralDisposableIncome": "income_rural_disposable"},
}
INCOME_SOURCE_FEATURE_COLUMNS = {
    "overall_income": ("income_disposable",),
    "overall_consumption": ("income_consumption",),
    "urban_consumption": ("income_urban_consumption",),
    "rural_consumption": ("income_rural_consumption",),
    "urban_income": ("income_urban_disposable",),
    "rural_income": ("income_rural_disposable",),
}
ORIGINAL_COLUMN_TO_SOURCE = {
    "density": ("density", "density"),
    "urbanization": ("urbanization_rate", "urbanization_rate"),
    "unemployment": ("unemployment_rate", "unemployment_rate"),
    "employees": ("employees_number", "employees_number"),
    "sector": ("primary_employment", "primary_employment", "secondary_employment", "tertiary_employment"),
    "wage": ("average_wage", "average_wage"),
    "age": ("age_0_14", "age_0_14", "age_15_64", "age_65_plus"),
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    if not fieldnames:
        raise ValueError(f"no fields for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float) and (not math.isfinite(value)):
                    value = None
                clean[field] = value
            writer.writerow(clean)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text in {"", "NA", "N/A", "NaN", "nan", "None"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric(value: Any) -> float | None:
    return finite_float(value)


def nested_copy_population(population: dict[str, dict[str, dict[int, float]]]) -> dict[str, dict[str, dict[int, float]]]:
    return {
        target: {city: dict(values) for city, values in by_city.items()}
        for target, by_city in population.items()
    }


def apply_registered_repair(population: dict[str, dict[str, dict[int, float]]]) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    out = nested_copy_population(population)
    values = out["registered"]["city40"]
    before = values.get(2021)
    prior = [values[year] for year in range(2017, 2021) if year in values]
    triggered = bool(before is not None and len(prior) >= 2 and np.median(prior) / max(before, 1.0) > 2.0)
    after = float(np.median(prior[-2:])) if triggered else before
    if triggered and after is not None:
        values[2021] = after
    return out, {
        "repair_variant": REPAIR_DIAGNOSTIC,
        "city": "city40",
        "year": 2021,
        "registered_before": before,
        "registered_after": after,
        "prior_2017_2020": prior,
        "triggered": triggered,
        "resident_changed": out["resident"]["city40"].get(2021) != population["resident"]["city40"].get(2021),
    }


def source_cap(geometry: dict[str, dict[str, Any]], source_id: str, origin_year: int, scenario_id: str) -> tuple[int, int]:
    lag = s2_geometry.required_lag_years(source_id, geometry[source_id], scenario_id)
    return lag, origin_year - lag


def build_canonical_context() -> dict[str, Any]:
    population, profile = s3_plan01.load_population()
    contract = s2_geometry.load_contract()
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    geometry = {row["source_id"]: row for row in s2_geometry.source_geometry_rows(sources)}
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for source_id, source in sources.items():
        values: dict[str, dict[int, dict[str, float]]] = {city: {} for city in CITIES}
        for row in source["rows"]:
            city = row[source["city_col"]]
            year = int(float(row[source["year_col"]]))
            if (city, year) in source["quarantine_keys"]:
                continue
            cell_values: dict[str, float] = {}
            for column in source["value_columns"]:
                number = finite_float(row.get(column))
                if number is not None:
                    cell_values[CANONICAL_COLUMN_MAP.get(source_id, {}).get(column, column)] = number
            if cell_values:
                values.setdefault(city, {})[year] = cell_values
        source_values[source_id] = values
    return {
        "population": population,
        "population_profile": profile,
        "contract": contract,
        "sources": sources,
        "geometry": geometry,
        "source_values": source_values,
    }


def load_original_source_values() -> tuple[dict[str, dict[str, dict[int, dict[str, float]]]], dict[str, Any]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("s10_plan01_original_immutable", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RAW = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
    _, exog = module.load_data()
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for source_id, mapping in ORIGINAL_COLUMN_TO_SOURCE.items():
        values: dict[str, dict[int, dict[str, float]]] = {city: {} for city in CITIES}
        columns = mapping[1:]
        for row in exog[["city", "year", *dict.fromkeys(columns)]].itertuples(index=False):
            city = str(row[0])
            year = int(row[1])
            cell: dict[str, float] = {}
            for column in columns:
                number = finite_float(getattr(row, column))
                if number is not None:
                    cell[column] = number
            if cell:
                values.setdefault(city, {})[year] = cell
        source_values[source_id] = values
    return source_values, {
        "employment_duplicate_semantics": "original source frames grouped by city/year mean before outer merge",
        "income_parser_semantics": "original _long_income_tables output is empty on current workbook header geometry",
        "income_features_included": False,
        "source_semantics": "ORIGINAL_SOURCE_SEMANTICS",
    }


def canonical_source_column(source_id: str, canonical_name: str) -> str:
    return canonical_name


def get_source_value(
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    source_id: str,
    city: str,
    year: int,
    column: str,
) -> float | None:
    return source_values.get(source_id, {}).get(city, {}).get(year, {}).get(column)


def latest_source_value(
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    source_id: str,
    city: str,
    cap: int,
    column: str,
) -> tuple[int | None, float | None]:
    candidates = [
        (year, cells[column])
        for year, cells in source_values.get(source_id, {}).get(city, {}).items()
        if year <= cap and column in cells
    ]
    if not candidates:
        return None, None
    year, value = max(candidates, key=lambda item: item[0])
    return year, value


def source_years_asof(
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    source_id: str,
    city: str,
    cap: int,
) -> list[int]:
    return sorted(
        year
        for year, cells in source_values.get(source_id, {}).get(city, {}).items()
        if year <= cap and cells
    )


def history(target_values: dict[str, dict[int, float]], city: str, cutoff: int) -> list[tuple[int, float]]:
    return sorted((year, value) for year, value in target_values.get(city, {}).items() if year <= cutoff and finite_float(value) is not None)


def asof_target(target_values: dict[str, dict[int, float]], city: str, year: int) -> tuple[int | None, float | None]:
    points = [(y, v) for y, v in target_values.get(city, {}).items() if y <= year]
    if not points:
        return None, None
    return max(points, key=lambda item: item[0])


def slope(points: list[tuple[int, float]], k: int, mean_slope: bool = False) -> float:
    values = points[-(k + 1):]
    if len(values) < 2:
        return 0.0
    slopes = np.diff([float(value) for _, value in values]) / np.diff([float(year) for year, _ in values])
    return float(np.mean(slopes) if mean_slope else np.median(slopes))


def linear_forecast(points: list[tuple[int, float]], cutoff: int, horizon: int, k: int) -> float | None:
    values = points[-k:]
    if not values:
        return None
    if len(values) < 2:
        return float(values[-1][1])
    return float(np.polyval(np.polyfit([year for year, _ in values], [value for _, value in values], 1), cutoff + horizon))


def area_ratio(
    population: dict[str, dict[str, dict[int, float]]],
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    city: str,
    cutoff: int,
) -> float | None:
    resident = dict(history(population["resident"], city, cutoff))
    density = {
        year: cells["density"]
        for year, cells in source_values.get("density", {}).get(city, {}).items()
        if year <= cutoff and "density" in cells
    }
    paired = [(year, resident[year] / density[year]) for year in resident.keys() & density.keys() if resident[year] > 0 and density[year] > 0]
    return float(np.median([value for _, value in sorted(paired)[-1:]])) if paired else None


def local_predictions(
    population: dict[str, dict[str, dict[int, float]]],
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    geometry: dict[str, dict[str, Any]],
    target: str,
    city: str,
    origin_year: int,
    scenario_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    cutoff = origin_year - 2
    points = history(population[target], city, cutoff)
    if not points:
        return {}, {method: "insufficient_target_history" for method in LOCAL_METHODS}
    last = float(points[-1][1])
    result: dict[str, float] = {
        "local_last": last,
        "local_slope3": last + 2.0 * slope(points, 3),
        "local_slope5": last + 2.0 * slope(points, 5),
        "local_mean_slope5": last + 2.0 * slope(points, 5, mean_slope=True),
    }
    for method, k in (("local_linear3", 3), ("local_linear5", 5)):
        value = linear_forecast(points, cutoff, 2, k)
        if value is not None:
            result[method] = value
    reasons = {method: "" for method in LOCAL_METHODS}
    density_cap = source_cap(geometry, "density", origin_year, scenario_id)[1]
    required_density_year = cutoff + 1
    density_next = get_source_value(source_values, "density", city, required_density_year, "density") if required_density_year <= density_cap else None
    area = area_ratio(population, source_values, city, cutoff)
    if target == "resident" and density_next is not None and area is not None:
        intermediate = area * density_next
        result["density_hold"] = intermediate
        result["density_plus_slope5"] = intermediate + slope(points, 5)
    else:
        reason = "source_cap_before_T_minus_1" if required_density_year > density_cap else "density_exact_T_minus_1_or_area_missing"
        reasons.update({method: reason for method in DENSITY_METHODS})
    return result, reasons


def build_feature_row(
    population: dict[str, dict[str, dict[int, float]]],
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    geometry: dict[str, dict[str, Any]],
    city: str,
    year: int,
    scenario_id: str,
    *,
    include_income: bool,
    source_semantics: str,
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {"city": city, "year": year}
    cutoff = year - 2
    for target in TARGETS:
        points = history(population[target], city, cutoff)
        for lag in (2, 3, 4, 5):
            _, value = asof_target(population[target], city, year - lag)
            row[f"{target}_lag{lag}"] = value
        row[f"{target}_slope3"] = slope(points, 3)
        row[f"{target}_slope5"] = slope(points, 5)
        row[f"{target}_mean_slope5"] = slope(points, 5, mean_slope=True)
        row[f"{target}_linear_slope5"] = (
            float(np.polyfit([y for y, _ in points[-5:]], [v for _, v in points[-5:]], 1)[0])
            if len(points) >= 2 else 0.0
        )

    active_sources = dict(SOURCE_FEATURE_COLUMNS)
    if include_income:
        active_sources.update(INCOME_SOURCE_FEATURE_COLUMNS)
    used_source_years: dict[str, list[int]] = {}
    source_cap_text: list[str] = []
    for source_id, columns in active_sources.items():
        _, cap = source_cap(geometry, source_id, year, scenario_id)
        source_cap_text.append(f"{source_id}<={cap}")
        used_years = source_years_asof(source_values, source_id, city, cap)
        used_source_years[source_id] = used_years
        for column in columns:
            latest_year, value = latest_source_value(source_values, source_id, city, cap, column)
            row[f"exog_{column}"] = value
        audit_rows.append({
            "source_semantics": source_semantics,
            "scenario_id": scenario_id,
            "origin_year": year,
            "city_id": city,
            "source_id": source_id,
            "required_lag_years": source_cap(geometry, source_id, year, scenario_id)[0],
            "required_cap_year": cap,
            "used_years": ";".join(str(value) for value in used_years),
            "latest_used_year": max(used_years) if used_years else None,
            "source_available": "yes" if used_years else "no",
            "source_cap_status": "PASS" if all(value <= cap for value in used_years) else "FAIL",
            "quarantine_status": "NOT_APPLICABLE_OR_EXCLUDED" if source_semantics == "ORIGINAL_SOURCE_SEMANTICS" else "EXCLUDED_BEFORE_LOOKUP",
            "provenance_status": "DIAGNOSTIC_RAW" if source_semantics == "ORIGINAL_SOURCE_SEMANTICS" else "CANONICAL_TRACKED",
        })
    _, density_cap = source_cap(geometry, "density", year, scenario_id)
    _, density_now = latest_source_value(source_values, "density", city, density_cap, "density")
    area = area_ratio(population, source_values, city, cutoff)
    row["resident_density_proxy"] = area * density_now if area is not None and density_now is not None else None
    resident_lag = row["resident_lag2"]
    registered_lag = row["registered_lag2"]
    row["registered_resident_ratio"] = registered_lag / max(float(resident_lag), 1.0) if resident_lag is not None and registered_lag is not None else None
    row["year_since_2000"] = year - 2000
    row["target_history_cutoff_year"] = cutoff
    row["source_cap_summary"] = ";".join(source_cap_text)
    return row


def build_features(
    population: dict[str, dict[str, dict[int, float]]],
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    geometry: dict[str, dict[str, Any]],
    scenario_id: str,
    *,
    include_income: bool,
    source_semantics: str,
    audit_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    # Only years in the observed canonical population are material; no 2022/2023
    # target row is constructed or read.
    years = range(2001, 2022)
    for year in years:
        for city in CITIES:
            rows.append(build_feature_row(
                population, source_values, geometry, city, year, scenario_id,
                include_income=include_income,
                source_semantics=source_semantics,
                audit_rows=audit_rows,
            ))
    frame = pd.DataFrame(rows)
    for target in TARGETS:
        frame[target] = [population[target].get(row.city, {}).get(int(row.year), np.nan) for row in frame.itertuples()]
    return frame


def model(name: str):
    if name == "ridge":
        return Ridge(alpha=20.0)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=1)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=1)
    if name == "hist_gradient":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=5.0, random_state=42)
    if name == "gradient_huber":
        return GradientBoostingRegressor(n_estimators=220, learning_rate=0.03, max_depth=2, loss="huber", random_state=42)
    raise KeyError(name)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, model_name: str) -> np.ndarray:
    feature_cols = [column for column in train.columns if column not in {"resident", "registered", "year", "target_history_cutoff_year", "source_cap_summary"}]
    combined = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    combined = pd.get_dummies(combined, columns=["city"], dtype=float)
    x_train = combined.iloc[: len(train)]
    x_test = combined.iloc[len(train):]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    if model_name == "ridge":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    estimator = model(model_name)
    estimator.fit(x_train, train[target].astype(float).to_numpy())
    return estimator.predict(x_test)


def row_base(
    target: str,
    scenario_id: str,
    origin_year: int,
    city: str,
    actual: float | None,
    method: str,
    prediction: float | None,
    *,
    source_semantics: str,
    repair_variant: str,
    family: str,
    training_max_target_year: int | None,
    unavailable_reason: str = "",
    feature_evidence: str = "",
) -> dict[str, Any]:
    error = prediction - actual if prediction is not None and actual is not None else None
    return {
        "target": target,
        "scenario_id": scenario_id,
        "origin_year": origin_year,
        "city_id": city,
        "method": method,
        "family": family,
        "source_semantics": source_semantics,
        "repair_variant": repair_variant,
        "target_history_cutoff_year": origin_year - 2,
        "training_max_target_year": training_max_target_year,
        "prediction": prediction,
        "actual": actual,
        "squared_error": error * error if error is not None else None,
        "absolute_error": abs(error) if error is not None else None,
        "bias_error_prediction_minus_actual": error,
        "prediction_available": "yes" if prediction is not None else "no",
        "unavailability_reason": unavailable_reason,
        "feature_source_cap_evidence": feature_evidence,
    }


def run_fixed_replay(
    population: dict[str, dict[str, dict[int, float]]],
    source_values: dict[str, dict[str, dict[int, dict[str, float]]]],
    geometry: dict[str, dict[str, Any]],
    *,
    source_semantics: str,
    repair_variant: str,
    include_income: bool,
    run_models: bool,
    feature_cache: dict[tuple[str, str, str, bool], pd.DataFrame],
    source_audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in SCENARIOS:
        feature_key = (source_semantics, repair_variant, scenario_id, include_income)
        if run_models and feature_key not in feature_cache:
            feature_cache[feature_key] = build_features(
                population, source_values, geometry, scenario_id,
                include_income=include_income,
                source_semantics=source_semantics,
                audit_rows=source_audit_rows,
            )
        features = feature_cache.get(feature_key)
        for target in TARGETS:
            for origin_year in ORIGINS:
                actuals = population[target]
                for city in CITIES:
                    actual = actuals.get(city, {}).get(origin_year)
                    local, reasons = local_predictions(population, source_values, geometry, target, city, origin_year, scenario_id)
                    for method in LOCAL_METHODS + DENSITY_METHODS:
                        prediction = local.get(method)
                        reason = reasons.get(method, "") if prediction is None else ""
                        rows.append(row_base(target, scenario_id, origin_year, city, actual, method, prediction,
                                             source_semantics=source_semantics, repair_variant=repair_variant,
                                             family="A_TARGET_ONLY" if method in LOCAL_METHODS else "B_DENSITY",
                                             training_max_target_year=origin_year - 2, unavailable_reason=reason,
                                             feature_evidence="target_history<=T-2;density_cap_checked"))
                if not run_models or features is None:
                    continue
                train = features[features["year"].le(origin_year - 2)].copy()
                test = features[features["year"].eq(origin_year)].copy()
                for model_name in MODEL_NAMES:
                    model_method = f"model_{model_name}"
                    train_target = train[train[target].notna()].copy()
                    if train_target.empty or test.empty:
                        for city in CITIES:
                            rows.append(row_base(target, scenario_id, origin_year, city, actuals.get(city, {}).get(origin_year), model_method, None,
                                                 source_semantics=source_semantics, repair_variant=repair_variant, family="C_POOLED_SKLEARN",
                                                 training_max_target_year=None, unavailable_reason="empty_training_or_test"))
                        continue
                    prediction = fit_predict(train_target, test, target, model_name)
                    training_max = int(train_target["year"].max())
                    assert training_max <= origin_year - 2
                    for city, value in zip(test["city"], prediction):
                        value_float = finite_float(value)
                        rows.append(row_base(target, scenario_id, origin_year, str(city), actuals.get(str(city), {}).get(origin_year), model_method, value_float,
                                             source_semantics=source_semantics, repair_variant=repair_variant, family="C_POOLED_SKLEARN",
                                             training_max_target_year=training_max, feature_evidence=str(test.loc[test["city"].eq(city), "source_cap_summary"].iloc[0])))
    return rows


def method_pool_rows() -> list[dict[str, Any]]:
    pools = {
        "A_TARGET_ONLY": LOCAL_METHODS,
        "B_PLUS_DENSITY": LOCAL_METHODS + DENSITY_METHODS,
        "C_PLUS_POOLED_ML": LOCAL_METHODS + MODEL_METHODS,
        "D_FULL_LEGAL_POOL": LOCAL_METHODS + DENSITY_METHODS + MODEL_METHODS,
    }
    rows = []
    for pool_id, methods in pools.items():
        for order, method in enumerate(methods, 1):
            family = "A_TARGET_ONLY" if method in LOCAL_METHODS else "B_DENSITY" if method in DENSITY_METHODS else "C_POOLED_SKLEARN"
            rows.append({"pool_id": pool_id, "pool_order": order, "method": method, "family": family, "frozen_before_scoring": "yes"})
    return rows


def metrics_for_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    scenario_id: str,
    method: str,
    scoring_origins: tuple[int, ...],
    selector_type: str,
    pool_id: str,
    comparison_label: str,
    source_semantics: str,
    repair_variant: str,
    control_b3: pd.DataFrame | None = None,
    control_f4: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = frame[(frame["target"] == target) & (frame["scenario_id"] == scenario_id) & (frame["method"] == method) & frame["origin_year"].isin(scoring_origins)].copy()
    valid = candidate[candidate["prediction"].notna() & candidate["actual"].notna()].copy()
    origin_rows: list[dict[str, Any]] = []
    for origin, group in valid.groupby("origin_year"):
        origin_rows.append({
            "target": target, "scenario_id": scenario_id, "method": method, "selector_type": selector_type,
            "pool_id": pool_id, "comparison_label": comparison_label, "source_semantics": source_semantics,
            "repair_variant": repair_variant, "origin_year": int(origin), "scored_rows": int(len(group)),
            "origin_mse": float(group["squared_error"].mean()), "origin_mae": float(group["absolute_error"].mean()),
            "origin_bias": float(group["bias_error_prediction_minus_actual"].mean()),
        })
    origin_df = pd.DataFrame(origin_rows, columns=[
        "target", "scenario_id", "method", "selector_type", "pool_id", "comparison_label",
        "source_semantics", "repair_variant", "origin_year", "scored_rows", "origin_mse",
        "origin_mae", "origin_bias",
    ])
    pooled_mse = float(valid["squared_error"].mean()) if not valid.empty else None
    pooled_mae = float(valid["absolute_error"].mean()) if not valid.empty else None
    pooled_bias = float(valid["bias_error_prediction_minus_actual"].mean()) if not valid.empty else None
    latest = valid[valid["origin_year"] == max(scoring_origins)]
    wins_b3 = wins_f4 = losses_b3 = losses_f4 = 0
    if control_b3 is not None:
        joined = origin_df.merge(control_b3, on="origin_year", suffixes=("", "_control"))
        wins_b3 = int((joined["origin_mse"] < joined["origin_mse_control"] - 1e-12).sum())
        losses_b3 = int((joined["origin_mse"] > joined["origin_mse_control"] + 1e-12).sum())
    if control_f4 is not None:
        joined = origin_df.merge(control_f4, on="origin_year", suffixes=("", "_control"))
        wins_f4 = int((joined["origin_mse"] < joined["origin_mse_control"] - 1e-12).sum())
        losses_f4 = int((joined["origin_mse"] > joined["origin_mse_control"] + 1e-12).sum())
    expected_rows = len(scoring_origins) * len(CITIES)
    summary = {
        "method": method, "target": target, "scenario_id": scenario_id, "source_semantics": source_semantics, "repair_variant": repair_variant,
        "selector_type": selector_type, "pool_id": pool_id, "comparison_label": comparison_label,
        "scoring_origins": ";".join(str(origin) for origin in scoring_origins), "origin_count": len(scoring_origins),
        "scored_rows": int(len(valid)), "expected_rows": expected_rows, "coverage": float(len(valid) / expected_rows) if expected_rows else 0.0,
        "pooled_mse": pooled_mse, "rmse": float(math.sqrt(pooled_mse)) if pooled_mse is not None else None,
        "mae": pooled_mae, "bias": pooled_bias,
        "macro_origin_mse": float(origin_df["origin_mse"].mean()) if not origin_df.empty else None,
        "latest_origin_mse": float(latest["squared_error"].mean()) if not latest.empty else None,
        "wins_vs_B3": wins_b3, "losses_vs_B3": losses_b3, "wins_vs_F4": wins_f4, "losses_vs_F4": losses_f4,
        "direct_comparison_valid": "yes" if len(valid) == expected_rows else "matched_subset_only",
        "deployability_status": "deployable_fixed_candidate" if selector_type == "FIXED_METHOD" and len(valid) == expected_rows else "diagnostic_or_partial",
    }
    return summary, origin_rows


def reference_frame_from_s4() -> pd.DataFrame:
    frame = read_csv(REFERENCE_B3_PATH)
    frame = frame[(frame["predictor_id"] == "B3_RECENT_3POINT_OLS_TREND") & frame["target_id"].isin(TARGETS) & frame["origin_year"].between(2011, 2021)].copy()
    return pd.DataFrame({
        "target": frame["target_id"], "scenario_id": frame["scenario_id"], "origin_year": frame["origin_year"].astype(int),
        "city_id": frame["city_id"].astype(str), "method": "B3", "prediction": pd.to_numeric(frame["prediction_value"], errors="coerce"),
        "actual": pd.to_numeric(frame["actual_target_value_for_scoring"], errors="coerce"),
    })


def reference_frame_from_f4() -> tuple[pd.DataFrame, str]:
    raw = subprocess.check_output(["git", "show", f"{F4_SOURCE_TAG}:{F4_SOURCE_PATH}"], cwd=ROOT)
    source_hash = sha256_bytes(raw)
    frame = pd.read_csv(io.BytesIO(raw))
    frame = frame[(frame["framing_id"] == "F4_LOG_GROWTH_FROM_LATEST") & frame["target_id"].isin(TARGETS) & frame["origin_year"].between(2011, 2021)].copy()
    assert len(frame) == 2 * 2 * 11 * 40, len(frame)
    return pd.DataFrame({
        "target": frame["target_id"], "scenario_id": frame["scenario_id"], "origin_year": frame["origin_year"].astype(int),
        "city_id": frame["city_id"].astype(str), "method": "F4", "prediction": pd.to_numeric(frame["prediction_value"], errors="coerce"),
        "actual": pd.to_numeric(frame["actual_target_value"], errors="coerce"),
    }), source_hash


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["squared_error"] = out["error"] ** 2
    out["absolute_error"] = out["error"].abs()
    return out


def control_origin_metrics(frame: pd.DataFrame, target: str, scenario_id: str, origins: tuple[int, ...]) -> pd.DataFrame:
    valid = add_errors(frame[(frame["target"] == target) & (frame["scenario_id"] == scenario_id) & frame["origin_year"].isin(origins)].copy())
    return valid.groupby("origin_year", as_index=False).agg(origin_mse=("squared_error", "mean"), origin_mae=("absolute_error", "mean"), origin_bias=("error", "mean"))


def apparent_selector(fixed: pd.DataFrame, scoring_origins: tuple[int, ...], method_pool: tuple[str, ...], *, target: str, scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = fixed[(fixed["target"] == target) & (fixed["scenario_id"] == scenario_id) & fixed["method"].isin(method_pool)].copy()
    history = subset[subset["prediction"].notna()].groupby(["city_id", "method"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"))
    choices = []
    for city in CITIES:
        candidates = history[(history["city_id"] == city) & (history["method"].isin(method_pool)) & (history["n"] >= 2)].sort_values(["mae", "method"])
        method = str(candidates.iloc[0]["method"]) if not candidates.empty else method_pool[0]
        choices.append({"city_id": city, "method_selected": method, "selector_type": "INVALID_FOR_UNBIASED_HISTORICAL_COMPARISON"})
    assignments = pd.DataFrame(choices)
    selected = subset.merge(assignments, on="city_id", how="inner")
    selected = selected[selected["method"] == selected["method_selected"]].copy()
    selected = selected[selected["origin_year"].isin(scoring_origins)].copy()
    selected["selector_type"] = "NON_NESTED_APPARENT_SELECTOR"
    return selected, assignments


def nested_selector(
    fixed: pd.DataFrame,
    *,
    target: str,
    scenario_id: str,
    pool_id: str,
    method_pool: tuple[str, ...],
    scoring_origins: tuple[int, ...] = NESTED_ORIGINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = fixed[(fixed["target"] == target) & (fixed["scenario_id"] == scenario_id) & fixed["method"].isin(method_pool)].copy()
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    previous_by_city: dict[str, str | None] = {city: None for city in CITIES}
    for origin in scoring_origins:
        prior = subset[(subset["origin_year"] < origin) & subset["prediction"].notna()].copy()
        prior_city = prior.groupby(["city_id", "method"], as_index=False).agg(prior_n=("absolute_error", "size"), prior_mae=("absolute_error", "mean"))
        prior_global = prior.groupby("method", as_index=False).agg(global_n=("absolute_error", "size"), global_mae=("absolute_error", "mean"))
        global_order = prior_global.sort_values(["global_mae", "method"])["method"].tolist()
        current = subset[(subset["origin_year"] == origin) & subset["prediction"].notna()].copy()
        for city in CITIES:
            choices = prior_city[(prior_city["city_id"] == city) & (prior_city["method"].isin(method_pool)) & (prior_city["prior_n"] >= 2)].sort_values(["prior_mae", "method"])
            if not choices.empty:
                proposed = str(choices.iloc[0]["method"])
                selection_reason = "city_prior_mae"
            else:
                proposed = str(global_order[0]) if global_order else method_pool[0]
                selection_reason = "global_prior_mae" if global_order else "pool_order_no_prior"
            available = set(current[current["city_id"] == city]["method"].tolist())
            chosen = proposed
            fallback_reason = ""
            if chosen not in available:
                chosen = next((method for method in global_order if method in available), None)
                if chosen is None:
                    chosen = next((method for method in method_pool if method in available), None)
                fallback_reason = "selected_method_unavailable_current_origin"
            if chosen is None:
                fallback_reason = "no_current_legal_prediction"
            city_prior = {str(row["method"]): (int(row["prior_n"]), float(row["prior_mae"])) for _, row in choices.iterrows()}
            for method in method_pool:
                n, mae = city_prior.get(method, (0, None))
                audit_rows.append({
                    "target": target, "scenario_id": scenario_id, "pool_id": pool_id, "origin_year": origin, "city_id": city,
                    "candidate_method": method, "prior_origins_strictly_less_than_current": ";".join(str(value) for value in sorted(prior["origin_year"].unique())),
                    "prior_n": n, "prior_mae": mae, "proposed_method": proposed, "selected_method": chosen,
                    "selection_reason": selection_reason, "fallback_reason": fallback_reason,
                    "previous_selected_method": previous_by_city[city],
                    "temporal_leakage_status": "PASS",
                })
            previous_by_city[city] = chosen
            if chosen is None:
                continue
            hit = current[(current["city_id"] == city) & (current["method"] == chosen)]
            if hit.empty:
                continue
            selected_rows.append({**hit.iloc[0].to_dict(), "pool_id": pool_id, "selector_type": "VALID_TEMPORAL_SELECTOR_EVALUATION", "selected_method": chosen, "fallback_reason": fallback_reason})
    return pd.DataFrame(selected_rows), pd.DataFrame(audit_rows)


def oracle_selector(fixed: pd.DataFrame, target: str, scenario_id: str, pool_id: str, method_pool: tuple[str, ...], scoring_origins: tuple[int, ...]) -> pd.DataFrame:
    subset = fixed[(fixed["target"] == target) & (fixed["scenario_id"] == scenario_id) & fixed["origin_year"].isin(scoring_origins) & fixed["method"].isin(method_pool) & fixed["prediction"].notna()].copy()
    subset = subset.sort_values(["origin_year", "city_id", "squared_error", "method"])
    selected = subset.groupby(["origin_year", "city_id"], as_index=False).first()
    selected["pool_id"] = pool_id
    selected["selector_type"] = "UNDEPLOYABLE_ORACLE_UPPER_BOUND"
    selected["selected_method"] = selected["method"]
    selected["fallback_reason"] = ""
    return selected


def save_reference_metrics(b3: pd.DataFrame, f4: pd.DataFrame, f4_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, frame, source in (("B3", b3, "S4_HANDOFFS/S4-Plan02/per_origin_predictions_or_residuals.csv"), ("F4", f4, f"git:{F4_SOURCE_TAG}:{F4_SOURCE_PATH}")):
        for target in TARGETS:
            for scenario in SCENARIOS:
                valid = add_errors(frame[(frame["target"] == target) & (frame["scenario_id"] == scenario)].copy())
                rows.append({
                    "method": label, "target": target, "scenario_id": scenario, "scoring_origins": "2011;2012;2013;2014;2015;2016;2017;2018;2019;2020;2021",
                    "origin_count": valid["origin_year"].nunique(), "scored_rows": len(valid), "pooled_mse": valid["squared_error"].mean(),
                    "rmse": math.sqrt(valid["squared_error"].mean()), "mae": valid["absolute_error"].mean(), "bias": valid["error"].mean(),
                    "macro_origin_mse": valid.groupby("origin_year")["squared_error"].mean().mean(), "latest_origin_mse": valid.loc[valid["origin_year"] == 2021, "squared_error"].mean(),
                    "reference_source": source, "f4_oof_sha256": f4_hash if label == "F4" else "",
                })
    return rows


def density_failure_markdown(fixed: pd.DataFrame) -> str:
    density = fixed[(fixed["source_semantics"] == "CANONICAL_PROVENANCE_SAFE") & (fixed["repair_variant"] == PRIMARY_REPAIR) & fixed["method"].isin(DENSITY_METHODS) & (fixed["target"] == "resident")].copy()
    valid = density[density["prediction"].notna()].copy()
    lines = [
        "# Density candidate failure audit",
        "",
        "This is a diagnostic of the frozen density formulas; no density formula was repaired.",
        "",
        "## Per-origin evidence",
        "",
        "| method | origin | rows | MSE | MAE | max absolute error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (method, origin), group in valid.groupby(["method", "origin_year"]):
        lines.append(f"| {method} | {origin} | {len(group)} | {group['squared_error'].mean():.6f} | {group['absolute_error'].mean():.6f} | {group['absolute_error'].max():.6f} |")
    lines += ["", "## Largest observed rows", "", "| method | origin | city | prediction | actual | absolute error |", "|---|---:|---|---:|---:|---:|"]
    for row in valid.sort_values("absolute_error", ascending=False).head(12).itertuples():
        lines.append(f"| {row.method} | {row.origin_year} | {row.city_id} | {row.prediction:.6f} | {row.actual:.6f} | {row.absolute_error:.6f} |")
    lines += [
        "", "## Mechanism interpretation", "",
        "The candidate uses the latest legal resident/density area ratio through T-2 and an exact density observation at T-1 when the source cap permits it. The audit records the exact source cap and lookup availability per row. The 2017/2018 blow-up is therefore retained as observed candidate behavior; this Plan does not infer a unit conversion or regime correction without a separate source-level proof.",
        "", "Selector interaction is recorded in `selector_choice_history.csv`; a density method being selected after a large error is evidence about prior-only learning, not permission to repair the formula.",
    ]
    return "\n".join(lines) + "\n"


def build_master_rows(
    metric_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
    apparent_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = list(reference_rows) + list(metric_rows) + list(nested_rows) + list(apparent_rows) + list(oracle_rows)
    for row in out:
        row.setdefault("wins_vs_B3", None)
        row.setdefault("losses_vs_B3", None)
        row.setdefault("wins_vs_F4", None)
        row.setdefault("losses_vs_F4", None)
        row.setdefault("direct_comparison_valid", "reference_or_diagnostic")
        row.setdefault("deployability_status", "reference_or_diagnostic")
    return out


def main() -> None:
    global ROOT, OUTPUT_DIR, REFERENCE_B3_PATH, REFERENCE_COMPARISON_PATH, ORIGINAL_SCRIPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-parser-repair", action="store_true")
    parser.add_argument("--reuse-fixed-oof", action="store_true")
    parser.add_argument("--skip-repair-sensitivity", action="store_true")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    OUTPUT_DIR = ROOT / "reports" / "S10_HANDOFFS" / PLAN_ID
    REFERENCE_B3_PATH = ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan02" / "per_origin_predictions_or_residuals.csv"
    REFERENCE_COMPARISON_PATH = ROOT / "reports" / "S5_FINAL_CANDIDATE_COMPARISON.csv"
    ORIGINAL_SCRIPT = ROOT / "reports" / "S10_HANDOFFS" / "S10-Plan01" / "intake" / "original" / "forecast_population.py"
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT)
    assert ancestry.returncode == 0, f"Plan02 base {BASE_SHA} is not an ancestor of HEAD"
    population_context = build_canonical_context()
    canonical_population = population_context["population"]
    repaired_population, repair_evidence = apply_registered_repair(canonical_population)
    canonical_sources = population_context["source_values"]
    geometry = population_context["geometry"]
    original_sources, original_source_evidence = load_original_source_values()
    assert max(year for city in CITIES for year in canonical_population["resident"][city]) <= 2021
    assert max(year for city in CITIES for year in canonical_population["registered"][city]) <= 2021

    (OUTPUT_DIR / "contract").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablation").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "contract" / "plan02_contract.json", {
        "plan_id": PLAN_ID, "base_sha": BASE_SHA, "plan01_branch": PLAN01_BRANCH,
        "targets": {"primary": "resident", "sensitivity": "registered"}, "origins": list(ORIGINS),
        "nested_scoring_origins": list(NESTED_ORIGINS), "horizon_years": 2, "target_history_cutoff": "T-2",
        "primary_metric": "pooled city-year MSE", "macro_metric": "macro-origin MSE", "city_count": 40,
        "source_semantics": list(SOURCE_SEMANTICS), "primary_repair_variant": PRIMARY_REPAIR,
        "forbidden_inputs": ["S8 truth", "candidate_truth_2023", "real-2022/2023 population labels", "2023 row-level errors", "S9 merge", "S6 mutation"],
        "fixed_candidate_pool_frozen": True, "candidate_pool": [row for row in method_pool_rows()],
        "model_parameters_fixed_no_sweep": True, "f4_reference_tag": F4_SOURCE_TAG, "f4_reference_path": F4_SOURCE_PATH,
        "population_profile": population_context["population_profile"],
    })
    write_csv(OUTPUT_DIR / "contract" / "candidate_pool.csv", method_pool_rows())
    write_json(OUTPUT_DIR / "contract" / "repair_evidence.json", repair_evidence)

    b3 = reference_frame_from_s4()
    f4, f4_hash = reference_frame_from_f4()
    reference_metric_rows = save_reference_metrics(b3, f4, f4_hash)
    write_csv(OUTPUT_DIR / "contract" / "project_reference_metrics.csv", reference_metric_rows)
    matched_reference_rows: list[dict[str, Any]] = []
    for label, reference in (("B3", b3), ("F4", f4)):
        for target in TARGETS:
            for scenario in SCENARIOS:
                valid = add_errors(reference[(reference["target"] == target) & (reference["scenario_id"] == scenario) & reference["origin_year"].isin(NESTED_ORIGINS)].copy())
                matched_reference_rows.append({
                    "method": label, "target": target, "scenario_id": scenario, "comparison_label": "MATCHED_2013_2021",
                    "scoring_origins": ";".join(str(origin) for origin in NESTED_ORIGINS), "origin_count": valid["origin_year"].nunique(),
                    "scored_rows": len(valid), "expected_rows": len(NESTED_ORIGINS) * len(CITIES),
                    "coverage": len(valid) / (len(NESTED_ORIGINS) * len(CITIES)), "pooled_mse": valid["squared_error"].mean(),
                    "rmse": math.sqrt(valid["squared_error"].mean()), "mae": valid["absolute_error"].mean(), "bias": valid["error"].mean(),
                    "macro_origin_mse": valid.groupby("origin_year")["squared_error"].mean().mean(),
                    "latest_origin_mse": valid.loc[valid["origin_year"] == max(NESTED_ORIGINS), "squared_error"].mean(),
                    "reference_source": "S4_HANDOFFS/S4-Plan02/per_origin_predictions_or_residuals.csv" if label == "B3" else f"git:{F4_SOURCE_TAG}:{F4_SOURCE_PATH}",
                    "f4_oof_sha256": f4_hash if label == "F4" else "",
                })
    write_csv(OUTPUT_DIR / "metrics" / "matched_control_metrics.csv", matched_reference_rows)

    feature_cache: dict[tuple[str, str, str, bool], pd.DataFrame] = {}
    source_audit_rows: list[dict[str, Any]] = []
    fixed_path = OUTPUT_DIR / "oof" / "fixed_method_oof.csv"
    if args.reuse_fixed_oof and fixed_path.exists():
        fixed = read_csv(fixed_path)
        source_audit_path = OUTPUT_DIR / "audit" / "source_cap_legality.csv"
        if source_audit_path.exists():
            source_audit_rows = read_csv(source_audit_path).to_dict("records")
    else:
        fixed_rows: list[dict[str, Any]] = []
        for semantics, sources in (("CANONICAL_PROVENANCE_SAFE", canonical_sources), ("ORIGINAL_SOURCE_SEMANTICS", original_sources)):
            fixed_rows.extend(run_fixed_replay(
                canonical_population, sources, geometry,
                source_semantics=semantics, repair_variant=PRIMARY_REPAIR, include_income=False, run_models=True,
                feature_cache=feature_cache, source_audit_rows=source_audit_rows,
            ))
        fixed = pd.DataFrame(fixed_rows)
        write_csv(fixed_path, fixed.to_dict("records"))
        write_csv(OUTPUT_DIR / "audit" / "source_cap_legality.csv", source_audit_rows)

    parser_rows: list[dict[str, Any]] = []
    if not args.skip_parser_repair:
        parser_rows = run_fixed_replay(
            canonical_population, canonical_sources, geometry,
            source_semantics="PARSER_REPAIRED_ONLY", repair_variant=PRIMARY_REPAIR, include_income=True, run_models=True,
            feature_cache=feature_cache, source_audit_rows=source_audit_rows,
        )
    parser_status = "NOT_RUN_LOW_PRIORITY" if args.skip_parser_repair else "RUN"
    write_csv(OUTPUT_DIR / "ablation" / "parser_repair_ablation.csv", [
        {"status": parser_status, "source_semantics": "PARSER_REPAIRED_ONLY", "income_parser_original_frame_count": 0,
         "known_quarantine_excluded": "yes", "note": "Bounded diagnostic only; not part of primary selector pool."}
    ])

    fixed_primary = fixed[(fixed["source_semantics"] == "CANONICAL_PROVENANCE_SAFE") & (fixed["repair_variant"] == PRIMARY_REPAIR)].copy()
    metric_rows: list[dict[str, Any]] = []
    per_origin_rows: list[dict[str, Any]] = []
    for semantics in SOURCE_SEMANTICS:
        scope = fixed[fixed["source_semantics"] == semantics]
        for target in TARGETS:
            for scenario in SCENARIOS:
                b3_origin = control_origin_metrics(b3, target, scenario, ORIGINS)
                f4_origin = control_origin_metrics(f4, target, scenario, ORIGINS)
                for method in ALL_METHODS:
                    row, origins = metrics_for_frame(scope, target=target, scenario_id=scenario, method=method, scoring_origins=ORIGINS,
                                                      selector_type="FIXED_METHOD", pool_id="", comparison_label="11_ORIGIN",
                                                      source_semantics=semantics, repair_variant=PRIMARY_REPAIR,
                                                      control_b3=b3_origin, control_f4=f4_origin)
                    metric_rows.append(row)
                    per_origin_rows.extend(origins)
                    row9, origins9 = metrics_for_frame(scope, target=target, scenario_id=scenario, method=method, scoring_origins=NESTED_ORIGINS,
                                                        selector_type="FIXED_METHOD", pool_id="", comparison_label="MATCHED_2013_2021",
                                                        source_semantics=semantics, repair_variant=PRIMARY_REPAIR,
                                                        control_b3=control_origin_metrics(b3, target, scenario, NESTED_ORIGINS),
                                                        control_f4=control_origin_metrics(f4, target, scenario, NESTED_ORIGINS))
                    metric_rows.append(row9)
                    per_origin_rows.extend(origins9)
    if parser_rows:
        parser_frame = pd.DataFrame(parser_rows)
        for target in TARGETS:
            for scenario in SCENARIOS:
                for method in MODEL_METHODS:
                    row, origins = metrics_for_frame(parser_frame, target=target, scenario_id=scenario, method=method, scoring_origins=ORIGINS,
                                                      selector_type="FIXED_METHOD_PARSER_REPAIR_DIAGNOSTIC", pool_id="", comparison_label="11_ORIGIN",
                                                      source_semantics="PARSER_REPAIRED_ONLY", repair_variant=PRIMARY_REPAIR)
                    metric_rows.append(row); per_origin_rows.extend(origins)
    write_csv(OUTPUT_DIR / "metrics" / "fixed_method_metrics.csv", metric_rows)
    write_csv(OUTPUT_DIR / "metrics" / "per_origin_metrics.csv", per_origin_rows)
    write_csv(OUTPUT_DIR / "audit" / "coverage_audit.csv", [
        {
            "target": row.get("target"), "scenario_id": row.get("scenario_id"), "method": row.get("method"),
            "selector_type": row.get("selector_type"), "pool_id": row.get("pool_id"),
            "comparison_label": row.get("comparison_label"), "scoring_origins": row.get("scoring_origins"),
            "scored_rows": row.get("scored_rows"), "expected_rows": row.get("expected_rows"),
            "coverage": row.get("coverage"), "coverage_status": "PASS" if float(row.get("coverage") or 0) == 1.0 else "PARTIAL_OR_UNAVAILABLE",
        }
        for row in metric_rows
    ])

    choice_rows: list[dict[str, Any]] = []
    nested_oof_rows: list[dict[str, Any]] = []
    nested_metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    pools = {
        "A_TARGET_ONLY": LOCAL_METHODS,
        "B_PLUS_DENSITY": LOCAL_METHODS + DENSITY_METHODS,
        "C_PLUS_POOLED_ML": LOCAL_METHODS + MODEL_METHODS,
        "D_FULL_LEGAL_POOL": LOCAL_METHODS + DENSITY_METHODS + MODEL_METHODS,
    }
    for target in TARGETS:
        for scenario in SCENARIOS:
            for pool_id, pool in pools.items():
                selected, audit = nested_selector(fixed_primary, target=target, scenario_id=scenario, pool_id=pool_id, method_pool=pool)
                choice_rows.extend(audit.to_dict("records"))
                method_view = selected.copy()
                if not method_view.empty:
                    method_view["method"] = pool_id
                    method_view["prediction"] = pd.to_numeric(method_view["prediction"], errors="coerce")
                    method_view["actual"] = pd.to_numeric(method_view["actual"], errors="coerce")
                    method_view["squared_error"] = (method_view["prediction"] - method_view["actual"]) ** 2
                    method_view["absolute_error"] = (method_view["prediction"] - method_view["actual"]).abs()
                    method_view["bias_error_prediction_minus_actual"] = method_view["prediction"] - method_view["actual"]
                    nested_metric, nested_origins = metrics_for_frame(method_view.assign(method=pool_id), target=target, scenario_id=scenario, method=pool_id,
                                                                       scoring_origins=NESTED_ORIGINS, selector_type="VALID_TEMPORAL_SELECTOR_EVALUATION", pool_id=pool_id,
                                                                       comparison_label="MATCHED_2013_2021", source_semantics="CANONICAL_PROVENANCE_SAFE", repair_variant=PRIMARY_REPAIR,
                                                                       control_b3=control_origin_metrics(b3, target, scenario, NESTED_ORIGINS),
                                                                       control_f4=control_origin_metrics(f4, target, scenario, NESTED_ORIGINS))
                    nested_metric_rows.append(nested_metric)
                    ablation_rows.append(nested_metric)
                    selected["method_selected"] = selected["selected_method"]
                    nested_oof_rows.extend(selected.to_dict("records"))
    write_csv(OUTPUT_DIR / "oof" / "selector_choice_history.csv", choice_rows)
    write_csv(OUTPUT_DIR / "oof" / "nested_selector_oof.csv", nested_oof_rows)
    write_csv(OUTPUT_DIR / "ablation" / "selector_pool_ablation.csv", ablation_rows)

    apparent_summary_rows: list[dict[str, Any]] = []
    oracle_summary_rows: list[dict[str, Any]] = []
    apparent_oof_rows: list[dict[str, Any]] = []
    oracle_oof_rows: list[dict[str, Any]] = []
    full_pool = pools["D_FULL_LEGAL_POOL"]
    for target in TARGETS:
        for scenario in SCENARIOS:
            apparent, assignments = apparent_selector(fixed_primary, ORIGINS, full_pool, target=target, scenario_id=scenario)
            if not apparent.empty:
                apparent["method"] = "NON_NESTED_APPARENT_SELECTOR"
                apparent["squared_error"] = (apparent["prediction"] - apparent["actual"]) ** 2
                apparent["absolute_error"] = (apparent["prediction"] - apparent["actual"]).abs()
                apparent["bias_error_prediction_minus_actual"] = apparent["prediction"] - apparent["actual"]
                apparent_oof_rows.extend(apparent.to_dict("records"))
                summary, origins = metrics_for_frame(apparent, target=target, scenario_id=scenario, method="NON_NESTED_APPARENT_SELECTOR", scoring_origins=ORIGINS,
                                                     selector_type="NON_NESTED_APPARENT_SELECTOR", pool_id="D_FULL_LEGAL_POOL", comparison_label="11_ORIGIN",
                                                     source_semantics="CANONICAL_PROVENANCE_SAFE", repair_variant=PRIMARY_REPAIR,
                                                     control_b3=control_origin_metrics(b3, target, scenario, ORIGINS),
                                                     control_f4=control_origin_metrics(f4, target, scenario, ORIGINS))
                apparent_summary_rows.append(summary)
                summary9, origins9 = metrics_for_frame(apparent, target=target, scenario_id=scenario, method="NON_NESTED_APPARENT_SELECTOR", scoring_origins=NESTED_ORIGINS,
                                                       selector_type="NON_NESTED_APPARENT_SELECTOR", pool_id="D_FULL_LEGAL_POOL", comparison_label="MATCHED_2013_2021",
                                                       source_semantics="CANONICAL_PROVENANCE_SAFE", repair_variant=PRIMARY_REPAIR,
                                                       control_b3=control_origin_metrics(b3, target, scenario, NESTED_ORIGINS),
                                                       control_f4=control_origin_metrics(f4, target, scenario, NESTED_ORIGINS))
                apparent_summary_rows.append(summary9)
            oracle = oracle_selector(fixed_primary, target, scenario, "D_FULL_LEGAL_POOL", full_pool, NESTED_ORIGINS)
            if not oracle.empty:
                oracle["method"] = "UNDEPLOYABLE_ORACLE_UPPER_BOUND"
                oracle["squared_error"] = (oracle["prediction"] - oracle["actual"]) ** 2
                oracle["absolute_error"] = (oracle["prediction"] - oracle["actual"]).abs()
                oracle["bias_error_prediction_minus_actual"] = oracle["prediction"] - oracle["actual"]
                oracle_oof_rows.extend(oracle.to_dict("records"))
                summary, origins = metrics_for_frame(oracle, target=target, scenario_id=scenario, method="UNDEPLOYABLE_ORACLE_UPPER_BOUND", scoring_origins=NESTED_ORIGINS,
                                                     selector_type="UNDEPLOYABLE_ORACLE_UPPER_BOUND", pool_id="D_FULL_LEGAL_POOL", comparison_label="MATCHED_2013_2021",
                                                     source_semantics="CANONICAL_PROVENANCE_SAFE", repair_variant=PRIMARY_REPAIR,
                                                     control_b3=control_origin_metrics(b3, target, scenario, NESTED_ORIGINS),
                                                     control_f4=control_origin_metrics(f4, target, scenario, NESTED_ORIGINS))
                oracle_summary_rows.append(summary)
    write_csv(OUTPUT_DIR / "oof" / "apparent_selector_oof.csv", apparent_oof_rows)
    write_csv(OUTPUT_DIR / "oof" / "oracle_selector_oof.csv", oracle_oof_rows)
    write_csv(OUTPUT_DIR / "metrics" / "selector_comparison.csv", nested_metric_rows + apparent_summary_rows + oracle_summary_rows)

    # Repair sensitivity is intentionally diagnostic: only the resident pooled
    # models are refit so the cross-target registered-history effect is visible.
    repair_rows: list[dict[str, Any]] = []
    if not args.skip_repair_sensitivity:
        repair_feature_cache: dict[tuple[str, str, str, bool], pd.DataFrame] = {}
        repair_audit: list[dict[str, Any]] = []
        for semantics, sources in (("CANONICAL_PROVENANCE_SAFE", canonical_sources),):
            repaired_model_rows = run_fixed_replay(repaired_population, sources, geometry,
                                                   source_semantics=semantics, repair_variant=REPAIR_DIAGNOSTIC, include_income=False, run_models=True,
                                                   feature_cache=repair_feature_cache, source_audit_rows=repair_audit)
            repaired = pd.DataFrame(repaired_model_rows)
            no_repair = fixed_primary[fixed_primary["target"] == "resident"]
            repaired = repaired[repaired["target"] == "resident"]
            keys = ["scenario_id", "origin_year", "city_id", "method"]
            joined = no_repair[no_repair["method"].isin(MODEL_METHODS)][keys + ["prediction"]].merge(repaired[repaired["method"].isin(MODEL_METHODS)][keys + ["prediction"]], on=keys, suffixes=("_no_repair", "_repair"))
            joined["prediction_delta_repair_minus_no_repair"] = joined["prediction_repair"] - joined["prediction_no_repair"]
            joined["changed"] = (joined["prediction_delta_repair_minus_no_repair"].abs() > 1e-12).astype(int)
            joined["source_semantics"] = semantics
            repair_rows.extend(joined.to_dict("records"))
    write_csv(OUTPUT_DIR / "audit" / "target_repair_sensitivity.csv", repair_rows, fields=(
        "scenario_id", "origin_year", "city_id", "method", "prediction_no_repair", "prediction_repair",
        "prediction_delta_repair_minus_no_repair", "changed", "source_semantics",
    ))

    # Method switching and stable identity are selector diagnostics, not model
    # tuning.  Keep them in the machine-readable selector comparison table.
    switch_rows = []
    choices_df = pd.DataFrame(choice_rows)
    if not choices_df.empty:
        for (target, scenario, pool_id, city), group in choices_df.groupby(["target", "scenario_id", "pool_id", "city_id"]):
            chosen = group[["origin_year", "selected_method"]].drop_duplicates().sort_values("origin_year")
            methods = chosen["selected_method"].dropna().tolist()
            switch_rows.append({"target": target, "scenario_id": scenario, "pool_id": pool_id, "city_id": city,
                                "scoring_origins": ";".join(str(value) for value in chosen["origin_year"]),
                                "selected_methods": ";".join(methods), "method_switch_count": sum(left != right for left, right in zip(methods, methods[1:])),
                                "stable_expert_identity": "yes" if len(set(methods)) <= 1 else "no"})
    write_csv(OUTPUT_DIR / "metrics" / "selector_method_switching.csv", switch_rows)

    master = build_master_rows(metric_rows, save_reference_metrics(b3, f4, f4_hash), nested_metric_rows, apparent_summary_rows, oracle_summary_rows)
    write_csv(OUTPUT_DIR / "metrics" / "master_comparison.csv", master)
    (OUTPUT_DIR / "audit" / "density_failure_audit.md").write_text(density_failure_markdown(fixed_primary), encoding="utf-8")
    write_json(OUTPUT_DIR / "audit" / "source_semantics_evidence.json", original_source_evidence)
    write_json(OUTPUT_DIR / "audit" / "selector_temporal_leakage_audit.json", {
        "non_nested_selector": "INVALID_FOR_UNBIASED_HISTORICAL_COMPARISON",
        "nested_selector": "PASS: prior origins strictly less than current; warm-up 2011-2012 excluded from scoring",
        "oracle": "UNDEPLOYABLE_ORACLE_UPPER_BOUND",
        "fallback": "prior-only global ranking then current availability; no current target used",
    })
    write_csv(OUTPUT_DIR / "audit" / "selector_temporal_leakage_audit.csv", [
        {"selector_type": "NON_NESTED_APPARENT_SELECTOR", "status": "INVALID_FOR_UNBIASED_HISTORICAL_COMPARISON", "evidence": "selection uses all 2011-2021 target errors before scoring the same rows"},
        {"selector_type": "VALID_TEMPORAL_SELECTOR_EVALUATION", "status": "PASS", "evidence": "each origin T ranks using only origins < T; 2011-2012 are warm-up only"},
        {"selector_type": "VALID_TEMPORAL_SELECTOR_EVALUATION_FALLBACK", "status": "PASS", "evidence": "fallback uses prior-only global ranking and current availability, never current target"},
        {"selector_type": "UNDEPLOYABLE_ORACLE_UPPER_BOUND", "status": "DIAGNOSTIC_ONLY", "evidence": "uses current true error and is not deployable"},
    ])
    write_json(OUTPUT_DIR / "audit" / "runtime_manifest.json", {
        "base_sha": BASE_SHA, "head_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__,
        "f4_source_tag": F4_SOURCE_TAG, "f4_source_path": F4_SOURCE_PATH, "f4_source_sha256": f4_hash,
        "population_max_year_read": 2021, "truth_access_boundary": "no S8 truth and no 2022/2023 target labels read",
    })
    print(json.dumps({
        "status": "RUN_COMPLETE", "fixed_rows": len(fixed), "nested_rows": len(nested_metric_rows), "choice_rows": len(choice_rows),
        "source_audit_rows": len(source_audit_rows), "repair_sensitivity_rows": len(repair_rows), "output_dir": str(OUTPUT_DIR),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
