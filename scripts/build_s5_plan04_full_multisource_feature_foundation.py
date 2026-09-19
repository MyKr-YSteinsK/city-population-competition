"""Build the S5-Plan04 full multi-source feature foundation.

This runner creates one leakage-safe, model-agnostic matrix for the frozen
S2 historical replay geometry.  It deliberately stops at feature construction:
no estimator is fitted, no feature is selected using outcomes, no 2023
population prediction is produced, and no submission is written.

The implementation reuses the S2 source contract/lag helper and the frozen
S3 B1/B2/B3 predictor implementation.  Source-specific feature builders keep
the S1 quarantine, invalidation, provenance, density-regime, and benchmark
boundaries explicit in both the matrix audit columns and the boundary audit.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import build_s2_plan02_geometry as s2_geometry
import build_s3_plan01_baseline_ladder as s3_plan01
import build_s3_plan02_recent_window_and_selector as s3_plan02


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan04"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
BASE_SHA = "d75ec4b9b1b392003c18201fa4b27899720f3c81"
BRANCH = "s5/plan04-full-multisource-feature-foundation"
OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
REGIME_REPORT_PATH = ROOT / "data" / "normalized" / "人口密度_regime_report.csv"

REFERENCE_TARGET_YEAR = 2023
HORIZON_YEARS = 2
COMMON_ORIGINS = tuple(range(2011, 2022))
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
CITIES = tuple(f"city{i}" for i in range(1, 41))
TARGET_COLUMNS = {
    "resident": "常住人口（万人）",
    "registered": "户籍人口（万人）",
}

SOURCE_ORDER = (
    "population",
    "density",
    "urbanization",
    "unemployment",
    "employees",
    "sector",
    "wage",
    "age",
    "overall_income",
    "overall_consumption",
    "urban_consumption",
    "rural_consumption",
    "urban_income",
    "rural_income",
)

POP_FEATURES = (
    "pop_latest_level",
    "pop_log1p_latest",
    "pop_latest_observation_year",
    "pop_history_observation_count",
    "pop_source_age_years",
    "pop_latest_first_difference",
    "pop_latest_relative_growth",
    "pop_recent3_ols_slope",
    "pop_expanding_ols_slope",
    "pop_recent_vs_expanding_slope_disagreement_abs",
    "pop_history_difference_volatility",
    "pop_relative_history_difference_volatility",
    "pop_recent_acceleration_abs",
    "pop_b1_prediction",
    "pop_b2_prediction",
    "pop_b3_prediction",
    "pop_baseline_prediction_spread",
)

DENSITY_FEATURES = (
    "density_latest_level",
    "density_log1p_latest",
    "density_latest_observation_year",
    "density_source_age_years",
    "density_recent_absolute_change",
    "density_recent_relative_change",
    "density_recent3_relative_slope",
    "density_usable_observation_count",
    "density_available_flag",
)

URBANIZATION_FEATURES = (
    "urbanization_latest_rate",
    "urbanization_latest_observation_year",
    "urbanization_source_age_years",
    "urbanization_recent_absolute_change",
    "urbanization_recent3_slope",
    "urbanization_usable_observation_count",
    "urbanization_available_flag",
)

UNEMPLOYMENT_FEATURES = (
    "unemployment_latest_rate",
    "unemployment_latest_observation_year",
    "unemployment_source_age_years",
    "unemployment_recent_absolute_change",
    "unemployment_recent3_slope",
    "unemployment_usable_observation_count",
    "unemployment_available_flag",
)

EMPLOYEES_FEATURES = (
    "employees_latest_level",
    "employees_log1p_latest",
    "employees_latest_observation_year",
    "employees_source_age_years",
    "employees_recent_absolute_change",
    "employees_recent_relative_change",
    "employees_recent3_log_slope",
    "employees_usable_observation_count",
    "employees_latest_is_derived_flag",
    "employees_available_flag",
)

SECTOR_FEATURES = (
    "sector_latest_primary_level",
    "sector_latest_secondary_level",
    "sector_latest_tertiary_level",
    "sector_latest_total_level",
    "sector_latest_primary_share",
    "sector_latest_secondary_share",
    "sector_latest_tertiary_share",
    "sector_latest_observation_year",
    "sector_source_age_years",
    "sector_recent_primary_share_change",
    "sector_recent_secondary_share_change",
    "sector_recent_tertiary_share_change",
    "sector_usable_common_year_observation_count",
    "sector_available_flag",
)

WAGE_FEATURES = (
    "wage_latest_level",
    "wage_log1p_latest",
    "wage_latest_observation_year",
    "wage_source_age_years",
    "wage_recent_absolute_change",
    "wage_recent_relative_change",
    "wage_recent3_log_slope",
    "wage_usable_observation_count",
    "wage_available_flag",
)

AGE_FEATURES = (
    "age_latest_benchmark_year",
    "age_source_age_years",
    "age_latest_0_14",
    "age_latest_15_64",
    "age_latest_65_plus",
    "age_latest_valid_total",
    "age_youth_share",
    "age_working_age_share",
    "age_elderly_share",
    "age_dependency_like_ratio",
    "age_old_age_dependency_like_ratio",
    "age_youth_share_change",
    "age_working_age_share_change",
    "age_elderly_share_change",
    "age_benchmark_elapsed_years",
    "age_benchmark_available_flag",
)

LIFE_SOURCE_PREFIXES = (
    ("overall_income", "overall_income"),
    ("overall_consumption", "overall_consumption"),
    ("urban_income", "urban_income"),
    ("rural_income", "rural_income"),
    ("urban_consumption", "urban_consumption"),
    ("rural_consumption", "rural_consumption"),
)
LIFE_SERIES_SUFFIXES = (
    "latest_level",
    "log1p_latest",
    "latest_observation_year",
    "source_age_years",
    "recent_relative_change",
    "recent3_log_slope",
    "usable_observation_count",
    "available_flag",
)
LIFE_SERIES_FEATURES = tuple(
    f"{prefix}_{suffix}"
    for _, prefix in LIFE_SOURCE_PREFIXES
    for suffix in LIFE_SERIES_SUFFIXES
)
LIFE_RATIO_FEATURES = (
    "life_overall_consumption_to_overall_income",
    "life_urban_consumption_to_urban_income",
    "life_rural_consumption_to_rural_income",
    "life_urban_income_to_rural_income",
    "life_urban_consumption_to_rural_consumption",
    "life_log_urban_income_gap_rural_income",
    "life_log_urban_consumption_gap_rural_consumption",
)
LIFE_FEATURES = LIFE_SERIES_FEATURES + LIFE_RATIO_FEATURES

GROUPS = (
    {
        "group_id": "FG-P",
        "sources": ("population",),
        "feature_names": POP_FEATURES,
        "mechanism": "target-history level, dynamics, frozen baseline geometry, and pre-origin risk diagnostics",
        "quality_boundary": "population target history <= T-2; canonical missingness preserved; no outcome-derived feature",
        "expected_model_use": "all model families; B3 reference-compatible target-history block",
    },
    {
        "group_id": "FG-D",
        "sources": ("density",),
        "feature_names": DENSITY_FEATURES,
        "mechanism": "density level and recent change under retained density-regime guard",
        "quality_boundary": "S1 density quarantine and documented regime crossings suppress dynamics only; levels are retained",
        "expected_model_use": "all model families; missingness-aware covariate block",
    },
    {
        "group_id": "FG-U",
        "sources": ("urbanization",),
        "feature_names": URBANIZATION_FEATURES,
        "mechanism": "urbanization level and recent development trend",
        "quality_boundary": "canonical coverage and city3 plateau retained; no clipping or interpolation",
        "expected_model_use": "all model families; categorical city handling remains downstream",
    },
    {
        "group_id": "FG-UNEMP",
        "sources": ("unemployment",),
        "feature_names": UNEMPLOYMENT_FEATURES,
        "mechanism": "urban unemployment level and recent trend",
        "quality_boundary": "raw canonical semantics retained; unresolved low-cardinality blocks are not relabeled",
        "expected_model_use": "all model families with explicit missing handling",
    },
    {
        "group_id": "FG-EMP",
        "sources": ("employees",),
        "feature_names": EMPLOYEES_FEATURES,
        "mechanism": "total employees level, growth, trend, and provenance state",
        "quality_boundary": "91 sector-sum reconstructions remain flagged; 80 quarantined source rows never re-enter",
        "expected_model_use": "all model families; provenance flag available for later ablation",
    },
    {
        "group_id": "FG-SECTOR",
        "sources": ("sector",),
        "feature_names": SECTOR_FEATURES,
        "mechanism": "primary/secondary/tertiary employment composition and recent share changes",
        "quality_boundary": "same-year component validity required; 18 conflicting rows quarantined; repair identity is not independent evidence",
        "expected_model_use": "all model families; shares are source-local composition features",
    },
    {
        "group_id": "FG-WAGE",
        "sources": ("wage",),
        "feature_names": WAGE_FEATURES,
        "mechanism": "average-wage level and development dynamics",
        "quality_boundary": "S1 invalidated wage cell remains missing; no model-based filling",
        "expected_model_use": "all model families with explicit missing handling",
    },
    {
        "group_id": "FG-AGE",
        "sources": ("age",),
        "feature_names": AGE_FEATURES,
        "mechanism": "benchmark-only age composition and benchmark-to-benchmark change",
        "quality_boundary": "only {2000,2010,2020}; no annual interpolation; invalidated benchmark rows remain missing",
        "expected_model_use": "benchmark-aware covariate block; downstream models must retain age availability",
    },
    {
        "group_id": "FG-LIFE",
        "sources": tuple(source_id for source_id, _ in LIFE_SOURCE_PREFIXES),
        "feature_names": LIFE_FEATURES,
        "mechanism": "six living-standard levels, recent log dynamics, aligned-year ratios, and log gaps",
        "quality_boundary": "all six canonical tables kept separate; ratios require same allowed year; no accounting identity imputation",
        "expected_model_use": "all model families; ratio features are optional downstream ablations, not preselected winners",
    },
)

MODEL_FEATURES = tuple(
    feature_name
    for group in GROUPS
    for feature_name in group["feature_names"]
)
GROUP_BY_FEATURE = {
    feature_name: group["group_id"]
    for group in GROUPS
    for feature_name in group["feature_names"]
}
SOURCE_BY_FEATURE = {
    feature_name: ";".join(group["sources"])
    for group in GROUPS
    for feature_name in group["feature_names"]
}

IDENTITY_COLUMNS = (
    "target_id",
    "scenario_id",
    "origin_year",
    "city_id",
    "target_history_cutoff_year",
)
LABEL_COLUMNS = ("actual_target_value",)
REFERENCE_COLUMNS = (
    "b3_prediction",
    "b3_residual_actual_minus_prediction",
    "b3_squared_error",
)

SOURCE_AUDIT_SUFFIXES = (
    "required_lag_years",
    "required_cap_year",
    "latest_used_year",
    "source_age_years",
    "usable_observation_count",
    "used_year_min",
    "used_year_max",
    "used_years",
    "availability_state",
    "missing_reason",
    "provenance_state",
    "derived_used_count",
    "quarantine_keys_as_of_cap",
    "quarantine_used_count",
    "domain_type",
    "used_years_within_cap",
    "quarantine_reintroduction_status",
)
SOURCE_AUDIT_EXTRA_SUFFIXES = {
    "population": ("history_cutoff_pass",),
    "density": (
        "regime_shift_year",
        "regime_window_phase",
        "suppressed_dynamics_count",
        "dynamic_guard_status",
    ),
    "employees": ("latest_is_derived", "used_derived_count"),
    "sector": ("latest_components_valid", "valid_component_year_count"),
    "age": ("benchmark_years_used", "domain_violation_count"),
}
AUDIT_COLUMNS = ("feature_contract_version",) + tuple(
    f"{source_id}_{suffix}"
    for source_id in SOURCE_ORDER
    for suffix in SOURCE_AUDIT_SUFFIXES + SOURCE_AUDIT_EXTRA_SUFFIXES.get(source_id, ())
) + tuple(
    f"life_ratio_{name}_latest_year" for name in LIFE_RATIO_FEATURES
) + tuple(
    f"life_ratio_{name}_missing_reason" for name in LIFE_RATIO_FEATURES
)

MATRIX_FIELDS = IDENTITY_COLUMNS + LABEL_COLUMNS + REFERENCE_COLUMNS + MODEL_FEATURES + AUDIT_COLUMNS

OOF_FIELDS = (
    "model_id",
    "framing_id",
    "target_id",
    "scenario_id",
    "origin_year",
    "city_id",
    "actual_target_value",
    "prediction_value",
    "residual_prediction_minus_actual",
    "squared_error",
    "training_max_label_year",
    "feature_contract_version",
    "worker_commit",
)

RATIO_SPECS = (
    ("life_overall_consumption_to_overall_income", "overall_consumption", "overall_income", "consumption / disposable income"),
    ("life_urban_consumption_to_urban_income", "urban_consumption", "urban_income", "urban consumption / urban income"),
    ("life_rural_consumption_to_rural_income", "rural_consumption", "rural_income", "rural consumption / rural income"),
    ("life_urban_income_to_rural_income", "urban_income", "rural_income", "urban income / rural income"),
    ("life_urban_consumption_to_rural_consumption", "urban_consumption", "rural_consumption", "urban consumption / rural consumption"),
    ("life_log_urban_income_gap_rural_income", "urban_income", "rural_income", "log(urban income) - log(rural income)"),
    ("life_log_urban_consumption_gap_rural_consumption", "urban_consumption", "rural_consumption", "log(urban consumption) - log(rural consumption)"),
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
    if not present(value):
        return None
    return int(float(value))


def as_float(value: Any) -> float | None:
    if not present(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def ols_slope(years: list[int], values: list[float]) -> float | None:
    if len(years) != len(values) or len(years) < 2:
        return None
    x_mean = math.fsum(float(year) for year in years) / len(years)
    y_mean = math.fsum(values) / len(values)
    denominator = math.fsum((year - x_mean) ** 2 for year in years)
    if denominator == 0:
        return None
    return math.fsum(
        (year - x_mean) * (value - y_mean)
        for year, value in zip(years, values)
    ) / denominator


def ols_intercept(years: list[int], values: list[float], slope: float | None) -> float | None:
    if slope is None or not values:
        return None
    return (math.fsum(values) / len(values)) - slope * (math.fsum(years) / len(years))


def population_std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def safe_relative_change(previous: float | None, current: float | None, gap: int | None) -> float | None:
    if previous is None or current is None or gap in (None, 0) or previous == 0:
        return None
    return (current - previous) / previous / gap


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]] | None = None,
    fields: Iterable[str] | None = None,
    model_features: set[str] | None = None,
) -> None:
    rows = rows or []
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    assert fieldnames, path
    model_features = model_features or set()
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
            output: dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if value is None:
                    output[field] = "NaN" if field in model_features else ""
                elif isinstance(value, float):
                    output[field] = "NaN" if not math.isfinite(value) else round(value, 12)
                else:
                    output[field] = value
            writer.writerow(output)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def assert_repo_identity() -> None:
    assert git_text("branch", "--show-current") == BRANCH
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Plan04 base must be an ancestor of the worker tip"


def source_row_maps(sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[int, dict[str, str]]]]:
    result: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for source_id, source in sources.items():
        city_col = source["city_col"]
        year_col = source["year_col"]
        by_city: dict[str, dict[int, dict[str, str]]] = {city: {} for city in CITIES}
        for row in source["rows"]:
            city = row[city_col]
            year = as_int(row.get(year_col))
            assert city in by_city and year is not None
            assert year not in by_city[city], (source_id, city, year)
            by_city[city][year] = row
        result[source_id] = by_city
    return result


def scalar_series(
    source_id: str,
    source: dict[str, Any],
    row_maps: dict[str, dict[str, dict[int, dict[str, str]]]],
) -> dict[str, dict[int, float]]:
    value_column = source["value_columns"][0]
    result: dict[str, dict[int, float]] = {city: {} for city in CITIES}
    for city in CITIES:
        for year, row in row_maps[source_id][city].items():
            if (city, year) in source["quarantine_keys"]:
                continue
            value = as_float(row.get(value_column))
            if value is not None:
                result[city][year] = value
    return result


def load_context() -> dict[str, Any]:
    assert_repo_identity()
    s3_plan01.assert_static_boundaries()
    population, population_profile = s3_plan01.load_population()
    s2_summary, _, _ = s3_plan01.load_s2_context()
    s3_summary = json.loads(
        (ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03" / "plan03_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    assert s2_summary["s2_complete"] is False
    assert s2_summary["s3_entered"] is False
    assert s3_summary["strongest_deployable_fixed_baseline_candidate"] == "B3_RECENT_3POINT_OLS_TREND"
    assert s3_summary["s2_contract_modified"] is False
    assert s3_summary["s3_complete"] is False

    contract = s2_geometry.load_contract()
    assert {row["dataset_id"] for row in contract} == set(SOURCE_ORDER)
    sources = s2_geometry.load_source_data(contract)
    s2_geometry.assert_s1_boundaries(contract, sources)
    geometry = {
        row["source_id"]: row for row in s2_geometry.source_geometry_rows(sources)
    }
    assert set(geometry) == set(SOURCE_ORDER)

    regime_rows = read_csv(REGIME_REPORT_PATH)
    assert len(regime_rows) == 13
    regime_shift_year = {
        row["城市名称"]: int(row["regime_shift_year"])
        for row in regime_rows
        if row["status"] == "retained"
    }
    assert len(regime_shift_year) == 13

    row_maps = source_row_maps(sources)
    scalar_series_by_source = {
        source_id: scalar_series(source_id, sources[source_id], row_maps)
        for source_id in SOURCE_ORDER
        if len(sources[source_id]["value_columns"]) == 1
    }
    assert set(scalar_series_by_source) == {
        "density",
        "urbanization",
        "unemployment",
        "employees",
        "wage",
        "overall_income",
        "overall_consumption",
        "urban_consumption",
        "rural_consumption",
        "urban_income",
        "rural_income",
    }
    assert len(sources["employees"]["derived_keys"]) == 91
    assert len(sources["employees"]["quarantine_keys"]) == 59
    return {
        "population": population,
        "population_profile": population_profile,
        "s2_summary": s2_summary,
        "s3_summary": s3_summary,
        "contract": contract,
        "sources": sources,
        "geometry": geometry,
        "row_maps": row_maps,
        "scalar_series": scalar_series_by_source,
        "regime_shift_year": regime_shift_year,
    }


def source_cap(context: dict[str, Any], source_id: str, origin_year: int, scenario_id: str) -> tuple[int, int]:
    lag = s2_geometry.required_lag_years(
        source_id,
        context["geometry"][source_id],
        scenario_id,
    )
    return lag, origin_year - lag


def make_source_audit(
    context: dict[str, Any],
    source_id: str,
    city: str,
    origin_year: int,
    scenario_id: str,
    allowed_years: list[int],
    used_years: list[int],
    latest_year: int | None,
    usable_count: int,
    missing_names: list[str],
    availability_state: str,
    provenance_state: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lag, cap = source_cap(context, source_id, origin_year, scenario_id)
    source = context["sources"][source_id]
    geometry = context["geometry"][source_id]
    qkeys = source["quarantine_keys"]
    derived_keys = source["derived_keys"]
    used_years = sorted(set(year for year in used_years if year is not None))
    q_as_of_cap = sum(
        1
        for key_city, year in qkeys
        if key_city == city and year <= cap
    )
    q_used = sum((city, year) in qkeys for year in used_years)
    derived_used = sum((city, year) in derived_keys for year in used_years)
    within_cap = all(year <= cap for year in used_years)
    audit = {
        f"{source_id}_required_lag_years": lag,
        f"{source_id}_required_cap_year": cap,
        f"{source_id}_latest_used_year": latest_year,
        f"{source_id}_source_age_years": origin_year - latest_year if latest_year is not None else None,
        f"{source_id}_usable_observation_count": usable_count,
        f"{source_id}_used_year_min": min(used_years) if used_years else None,
        f"{source_id}_used_year_max": max(used_years) if used_years else None,
        f"{source_id}_used_years": ";".join(str(year) for year in used_years),
        f"{source_id}_availability_state": availability_state,
        f"{source_id}_missing_reason": ";".join(missing_names),
        f"{source_id}_provenance_state": provenance_state,
        f"{source_id}_derived_used_count": derived_used,
        f"{source_id}_quarantine_keys_as_of_cap": q_as_of_cap,
        f"{source_id}_quarantine_used_count": q_used,
        f"{source_id}_domain_type": geometry["domain_type"],
        f"{source_id}_used_years_within_cap": "PASS" if within_cap else "FAIL",
        f"{source_id}_quarantine_reintroduction_status": "PASS" if q_used == 0 else "FAIL",
    }
    for suffix in SOURCE_AUDIT_EXTRA_SUFFIXES.get(source_id, ()):
        audit.setdefault(f"{source_id}_{suffix}", None)
    if extra:
        audit.update({f"{source_id}_{key}": value for key, value in extra.items()})
    return audit


def scalar_result(
    context: dict[str, Any],
    source_id: str,
    city: str,
    origin_year: int,
    scenario_id: str,
    prefix: str,
    series: dict[int, float],
    feature_names: tuple[str, ...],
    *,
    include_log_latest: bool = False,
    include_recent_absolute: bool = False,
    include_recent_relative: bool = False,
    recent3_mode: str | None = None,
    positive_only: bool = False,
) -> dict[str, Any]:
    _, cap = source_cap(context, source_id, origin_year, scenario_id)
    allowed_points = sorted(
        (year, value)
        for year, value in series.items()
        if year <= cap and (not positive_only or value > 0)
    )
    allowed_years = [year for year, _ in allowed_points]
    latest_year = allowed_years[-1] if allowed_years else None
    latest_value = allowed_points[-1][1] if allowed_points else None
    previous = allowed_points[-2] if len(allowed_points) >= 2 else None
    recent3 = allowed_points[-3:] if len(allowed_points) >= 3 else []

    values: dict[str, Any] = {name: None for name in feature_names}
    if f"{prefix}_latest_level" in feature_names:
        values[f"{prefix}_latest_level"] = latest_value
    if f"{prefix}_latest_rate" in feature_names:
        values[f"{prefix}_latest_rate"] = latest_value
    if include_log_latest:
        values[f"{prefix}_log1p_latest"] = (
            math.log1p(latest_value) if latest_value is not None and latest_value >= 0 else None
        )
    if f"{prefix}_latest_observation_year" in feature_names:
        values[f"{prefix}_latest_observation_year"] = latest_year
    if f"{prefix}_source_age_years" in feature_names:
        values[f"{prefix}_source_age_years"] = (
            origin_year - latest_year if latest_year is not None else None
        )
    if f"{prefix}_usable_observation_count" in feature_names:
        values[f"{prefix}_usable_observation_count"] = len(allowed_points)
    if f"{prefix}_available_flag" in feature_names:
        values[f"{prefix}_available_flag"] = 1 if latest_year is not None else 0

    if include_recent_absolute and previous is not None and latest_year is not None:
        values[f"{prefix}_recent_absolute_change"] = latest_value - previous[1]
    if include_recent_relative and previous is not None and latest_year is not None:
        values[f"{prefix}_recent_relative_change"] = safe_relative_change(
            previous[1], latest_value, latest_year - previous[0]
        )
    if recent3_mode and len(recent3) >= 3:
        years = [year for year, _ in recent3]
        recent_values = [value for _, value in recent3]
        slope = ols_slope(years, recent_values)
        if recent3_mode == "log":
            if all(value > 0 for value in recent_values):
                slope = ols_slope(years, [math.log(value) for value in recent_values])
            else:
                slope = None
        elif recent3_mode == "relative":
            slope = slope / latest_value if slope is not None and latest_value not in (None, 0) else None
        if recent3_mode == "log":
            slope_name = f"{prefix}_recent3_log_slope"
        elif recent3_mode == "relative":
            slope_name = f"{prefix}_recent3_relative_slope"
        else:
            slope_name = f"{prefix}_recent3_slope"
        if slope_name in feature_names:
            values[slope_name] = slope

    missing_names = [name for name in feature_names if values.get(name) is None]
    reasons: list[str] = []
    if not allowed_points:
        reasons.append("no_historical_row_as_of_cap")
    if include_recent_absolute and len(allowed_points) < 2:
        reasons.append("insufficient_two_observations")
    if include_recent_relative and len(allowed_points) >= 2 and allowed_points[-2][1] == 0:
        reasons.append("zero_previous_denominator")
    if recent3_mode and len(recent3) < 3:
        reasons.append("insufficient_three_observations")
    if recent3_mode == "log" and len(recent3) >= 3 and not all(value > 0 for _, value in recent3):
        reasons.append("non_positive_log_window")
    if include_log_latest and latest_value is not None and latest_value < 0:
        reasons.append("non_positive_log_latest")
    if missing_names and not reasons:
        reasons.append("feature_specific_missingness")

    used_years = sorted(
        set(
            ([latest_year] if latest_year is not None else [])
            + ([year for year, _ in allowed_points[-2:]] if include_recent_absolute or include_recent_relative else [])
            + ([year for year, _ in recent3] if recent3_mode else [])
        )
    )
    provenance_state = "source_observed_or_missing"
    if source_id == "employees" and any(
        (city, year) in context["sources"][source_id]["derived_keys"] for year in used_years
    ):
        provenance_state = "derived_rows_included_and_flagged"
    audit = make_source_audit(
        context,
        source_id,
        city,
        origin_year,
        scenario_id,
        allowed_years,
        used_years,
        latest_year,
        len(allowed_points),
        missing_names,
        "available" if latest_year is not None else "no_historical_row_as_of_cap",
        provenance_state,
    )
    return {
        "features": values,
        "audit": audit,
        "allowed_points": allowed_points,
        "used_years": used_years,
        "latest_year": latest_year,
        "latest_value": latest_value,
    }


def build_population_result(
    context: dict[str, Any],
    target_id: str,
    city: str,
    origin_year: int,
    scenario_id: str,
) -> dict[str, Any]:
    series = context["population"][target_id][city]
    cutoff = origin_year - HORIZON_YEARS
    points = sorted((year, value) for year, value in series.items() if year <= cutoff)
    years = [year for year, _ in points]
    values = [value for _, value in points]
    latest_year = years[-1] if years else None
    latest_value = values[-1] if values else None
    previous = points[-2] if len(points) >= 2 else None
    differences = [values[i] - values[i - 1] for i in range(1, len(values))]
    relative_differences = [
        (values[i] - values[i - 1]) / max(abs(values[i - 1]), 1e-12) / (years[i] - years[i - 1])
        for i in range(1, len(values))
    ]
    b1 = s3_plan02.fit_prediction("B1_TWO_POINT_LINEAR_TREND", series, origin_year, cutoff)
    b2 = s3_plan02.fit_prediction("B2_EXPANDING_PER_CITY_LINEAR_TREND", series, origin_year, cutoff)
    b3 = s3_plan02.fit_prediction("B3_RECENT_3POINT_OLS_TREND", series, origin_year, cutoff)
    recent3_slope = b3["regression_slope"]
    expanding_slope = b2["regression_slope"]
    values_map: dict[str, Any] = {
        "pop_latest_level": latest_value,
        "pop_log1p_latest": math.log1p(latest_value) if latest_value is not None and latest_value >= 0 else None,
        "pop_latest_observation_year": latest_year,
        "pop_history_observation_count": len(points),
        "pop_source_age_years": origin_year - latest_year if latest_year is not None else None,
        "pop_latest_first_difference": latest_value - previous[1] if previous and latest_value is not None else None,
        "pop_latest_relative_growth": safe_relative_change(
            previous[1], latest_value, latest_year - previous[0]
        ) if previous and latest_year is not None else None,
        "pop_recent3_ols_slope": recent3_slope,
        "pop_expanding_ols_slope": expanding_slope,
        "pop_recent_vs_expanding_slope_disagreement_abs": abs(recent3_slope - expanding_slope)
        if recent3_slope is not None and expanding_slope is not None else None,
        "pop_history_difference_volatility": population_std(differences),
        "pop_relative_history_difference_volatility": population_std(relative_differences),
        "pop_recent_acceleration_abs": abs(differences[-1] - differences[-2]) if len(differences) >= 2 else None,
        "pop_b1_prediction": b1["prediction"],
        "pop_b2_prediction": b2["prediction"],
        "pop_b3_prediction": b3["prediction"],
        "pop_baseline_prediction_spread": max(
            prediction for prediction in (b1["prediction"], b2["prediction"], b3["prediction"])
            if prediction is not None
        ) - min(
            prediction for prediction in (b1["prediction"], b2["prediction"], b3["prediction"])
            if prediction is not None
        ) if all(prediction is not None for prediction in (b1["prediction"], b2["prediction"], b3["prediction"])) else None,
    }
    missing_names = [name for name in POP_FEATURES if values_map.get(name) is None]
    reasons = []
    if not points:
        reasons.append("no_target_history_as_of_T_minus_2")
    if len(points) < 2:
        reasons.append("insufficient_two_target_observations")
    if len(points) < 3:
        reasons.append("insufficient_three_target_observations")
    if missing_names and not reasons:
        reasons.append("feature_specific_missingness")
    audit = make_source_audit(
        context,
        "population",
        city,
        origin_year,
        scenario_id,
        years,
        years,
        latest_year,
        len(points),
        missing_names,
        "available" if latest_year is not None else "no_historical_row_as_of_cap",
        "target_history_source_observed_or_missing",
        {"history_cutoff_pass": "PASS" if all(year <= cutoff for year in years) else "FAIL"},
    )
    return {
        "features": values_map,
        "audit": audit,
        "allowed_points": points,
        "used_years": years,
        "latest_year": latest_year,
        "latest_value": latest_value,
        "b1_prediction": b1["prediction"],
        "b2_prediction": b2["prediction"],
        "b3_prediction": b3["prediction"],
    }


def build_density_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    result = scalar_result(
        context,
        "density",
        city,
        origin_year,
        scenario_id,
        "density",
        context["scalar_series"]["density"][city],
        DENSITY_FEATURES,
        include_log_latest=True,
        include_recent_absolute=True,
        include_recent_relative=True,
        recent3_mode="relative",
    )
    points = result["allowed_points"]
    shift_year = context["regime_shift_year"].get(city)
    recent2_years = [year for year, _ in points[-2:]] if len(points) >= 2 else []
    recent3_years = [year for year, _ in points[-3:]] if len(points) >= 3 else []
    crosses_recent2 = bool(recent2_years and shift_year is not None and min(recent2_years) < shift_year <= max(recent2_years))
    crosses_recent3 = bool(recent3_years and shift_year is not None and min(recent3_years) < shift_year <= max(recent3_years))
    if crosses_recent2:
        result["features"]["density_recent_absolute_change"] = None
        result["features"]["density_recent_relative_change"] = None
    if crosses_recent3:
        result["features"]["density_recent3_relative_slope"] = None
    missing_names = [name for name in DENSITY_FEATURES if result["features"].get(name) is None]
    reasons = [
        reason
        for reason in (result["audit"]["density_missing_reason"].split(";") if result["audit"]["density_missing_reason"] else [])
        if reason
    ]
    if crosses_recent2 or crosses_recent3:
        reasons.append("documented_regime_crossing")
    result["audit"]["density_missing_reason"] = ";".join(dict.fromkeys(reasons))
    result["audit"].update(
        {
            "density_regime_shift_year": shift_year,
            "density_regime_window_phase": (
                "not_applicable"
                if shift_year is None
                else "crossing"
                if any(
                    years and min(years) < shift_year <= max(years)
                    for years in (recent2_years, recent3_years)
                )
                else "post"
                if result["used_years"] and min(result["used_years"]) >= shift_year
                else "pre"
            ),
            "density_suppressed_dynamics_count": int(crosses_recent2) + int(crosses_recent3),
            "density_dynamic_guard_status": "PASS",
        }
    )
    return result


def build_urbanization_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    return scalar_result(
        context,
        "urbanization",
        city,
        origin_year,
        scenario_id,
        "urbanization",
        context["scalar_series"]["urbanization"][city],
        URBANIZATION_FEATURES,
        include_recent_absolute=True,
        recent3_mode="level",
    )


def build_unemployment_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    return scalar_result(
        context,
        "unemployment",
        city,
        origin_year,
        scenario_id,
        "unemployment",
        context["scalar_series"]["unemployment"][city],
        UNEMPLOYMENT_FEATURES,
        include_recent_absolute=True,
        recent3_mode="level",
    )


def build_employees_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    result = scalar_result(
        context,
        "employees",
        city,
        origin_year,
        scenario_id,
        "employees",
        context["scalar_series"]["employees"][city],
        EMPLOYEES_FEATURES,
        include_log_latest=True,
        include_recent_absolute=True,
        include_recent_relative=True,
        recent3_mode="log",
        positive_only=True,
    )
    latest_year = result["latest_year"]
    derived_keys = context["sources"]["employees"]["derived_keys"]
    latest_is_derived = int(latest_year is not None and (city, latest_year) in derived_keys)
    result["features"]["employees_latest_is_derived_flag"] = latest_is_derived
    result["audit"]["employees_latest_is_derived"] = "yes" if latest_is_derived else "no"
    result["audit"]["employees_used_derived_count"] = sum(
        (city, year) in derived_keys for year in result["used_years"]
    )
    result["audit"]["employees_provenance_state"] = (
        "derived_rows_included_and_flagged"
        if result["audit"]["employees_used_derived_count"]
        else "source_observed_or_missing"
    )
    return result


def build_wage_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    return scalar_result(
        context,
        "wage",
        city,
        origin_year,
        scenario_id,
        "wage",
        context["scalar_series"]["wage"][city],
        WAGE_FEATURES,
        include_log_latest=True,
        include_recent_absolute=True,
        include_recent_relative=True,
        recent3_mode="log",
        positive_only=True,
    )


def build_sector_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    source_id = "sector"
    source = context["sources"][source_id]
    _, cap = source_cap(context, source_id, origin_year, scenario_id)
    rows = context["row_maps"][source_id][city]
    points: list[tuple[int, float, float, float, float, float, float]] = []
    for year, row in sorted(rows.items()):
        if year > cap or (city, year) in source["quarantine_keys"]:
            continue
        values = [as_float(row.get(column)) for column in source["value_columns"]]
        if any(value is None for value in values):
            continue
        primary, secondary, tertiary = values
        total = primary + secondary + tertiary
        if total <= 0:
            continue
        points.append(
            (
                year,
                primary,
                secondary,
                tertiary,
                total,
                primary / total,
                secondary / total,
            )
        )
    latest = points[-1] if points else None
    previous = points[-2] if len(points) >= 2 else None
    features = {name: None for name in SECTOR_FEATURES}
    if latest:
        year, primary, secondary, tertiary, total, primary_share, secondary_share = latest
        tertiary_share = tertiary / total
        features.update(
            {
                "sector_latest_primary_level": primary,
                "sector_latest_secondary_level": secondary,
                "sector_latest_tertiary_level": tertiary,
                "sector_latest_total_level": total,
                "sector_latest_primary_share": primary_share,
                "sector_latest_secondary_share": secondary_share,
                "sector_latest_tertiary_share": tertiary_share,
                "sector_latest_observation_year": year,
                "sector_source_age_years": origin_year - year,
                "sector_usable_common_year_observation_count": len(points),
                "sector_available_flag": 1,
            }
        )
        if previous:
            previous_year, _, _, _, _, prev_primary_share, prev_secondary_share = previous
            prev_tertiary_share = previous[3] / previous[4]
            features.update(
                {
                    "sector_recent_primary_share_change": primary_share - prev_primary_share,
                    "sector_recent_secondary_share_change": secondary_share - prev_secondary_share,
                    "sector_recent_tertiary_share_change": tertiary_share - prev_tertiary_share,
                }
            )
    else:
        features["sector_usable_common_year_observation_count"] = 0
        features["sector_available_flag"] = 0
    used_years = [point[0] for point in points[-2:]] if previous else ([latest[0]] if latest else [])
    missing_names = [name for name in SECTOR_FEATURES if features.get(name) is None]
    reasons = []
    if not points:
        reasons.append("no_valid_same_year_components_as_of_cap")
    if len(points) < 2:
        reasons.append("insufficient_two_common_year_observations")
    if missing_names and not reasons:
        reasons.append("feature_specific_missingness")
    audit = make_source_audit(
        context,
        source_id,
        city,
        origin_year,
        scenario_id,
        [point[0] for point in points],
        used_years,
        latest[0] if latest else None,
        len(points),
        missing_names + reasons,
        "available" if latest else "no_valid_same_year_components_as_of_cap",
        "component_repair_and_quarantine_boundary_preserved",
        {
            "latest_components_valid": "yes" if latest else "no",
            "valid_component_year_count": len(points),
        },
    )
    return {"features": features, "audit": audit, "used_years": used_years, "latest_year": latest[0] if latest else None}


def build_age_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    source_id = "age"
    source = context["sources"][source_id]
    _, cap = source_cap(context, source_id, origin_year, scenario_id)
    benchmark_years = {
        int(part) for part in context["geometry"][source_id]["domain_years"].split(";") if part.strip()
    }
    rows = context["row_maps"][source_id][city]
    points: list[tuple[int, float, float, float, float]] = []
    for year, row in sorted(rows.items()):
        if year > cap or year not in benchmark_years or (city, year) in source["quarantine_keys"]:
            continue
        values = [as_float(row.get(column)) for column in source["value_columns"]]
        if any(value is None for value in values):
            continue
        youth, working, elderly = values
        total = youth + working + elderly
        if total <= 0:
            continue
        points.append((year, youth, working, elderly, total))
    latest = points[-1] if points else None
    previous = points[-2] if len(points) >= 2 else None
    features = {name: None for name in AGE_FEATURES}
    if latest:
        year, youth, working, elderly, total = latest
        youth_share = youth / total
        working_share = working / total
        elderly_share = elderly / total
        features.update(
            {
                "age_latest_benchmark_year": year,
                "age_source_age_years": origin_year - year,
                "age_latest_0_14": youth,
                "age_latest_15_64": working,
                "age_latest_65_plus": elderly,
                "age_latest_valid_total": total,
                "age_youth_share": youth_share,
                "age_working_age_share": working_share,
                "age_elderly_share": elderly_share,
                "age_dependency_like_ratio": (youth + elderly) / working if working != 0 else None,
                "age_old_age_dependency_like_ratio": elderly / working if working != 0 else None,
                "age_benchmark_available_flag": 1,
            }
        )
        if previous:
            previous_year, previous_youth, previous_working, previous_elderly, previous_total = previous
            features.update(
                {
                    "age_youth_share_change": youth_share - previous_youth / previous_total,
                    "age_working_age_share_change": working_share - previous_working / previous_total,
                    "age_elderly_share_change": elderly_share - previous_elderly / previous_total,
                    "age_benchmark_elapsed_years": year - previous_year,
                }
            )
    else:
        features["age_benchmark_available_flag"] = 0
    used_years = [point[0] for point in points[-2:]] if previous else ([latest[0]] if latest else [])
    missing_names = [name for name in AGE_FEATURES if features.get(name) is None]
    reasons = []
    if not points:
        reasons.append("benchmark_not_available_as_of_cap")
    if len(points) < 2:
        reasons.append("insufficient_two_valid_benchmarks_for_change")
    if missing_names and not reasons:
        reasons.append("feature_specific_missingness")
    domain_violations = sum(year not in benchmark_years for year in used_years)
    assert domain_violations == 0
    audit = make_source_audit(
        context,
        source_id,
        city,
        origin_year,
        scenario_id,
        [point[0] for point in points],
        used_years,
        latest[0] if latest else None,
        len(points),
        missing_names + reasons,
        "available" if latest else "benchmark_not_available_as_of_cap",
        "benchmark_only_invalidation_boundary_preserved",
        {
            "benchmark_years_used": ";".join(str(year) for year in used_years),
            "domain_violation_count": domain_violations,
        },
    )
    return {"features": features, "audit": audit, "used_years": used_years, "latest_year": latest[0] if latest else None}


def build_life_result(
    context: dict[str, Any], city: str, origin_year: int, scenario_id: str
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    used_years: list[int] = []
    for source_id, prefix in LIFE_SOURCE_PREFIXES:
        result = scalar_result(
            context,
            source_id,
            city,
            origin_year,
            scenario_id,
            prefix,
            context["scalar_series"][source_id][city],
            tuple(f"{prefix}_{suffix}" for suffix in LIFE_SERIES_SUFFIXES),
            include_log_latest=True,
            include_recent_relative=True,
            recent3_mode="log",
            positive_only=False,
        )
        features.update(result["features"])
        audit.update(result["audit"])
        used_years.extend(result["used_years"])

    for name, numerator_source, denominator_source, _ in RATIO_SPECS:
        _, numerator_cap = source_cap(context, numerator_source, origin_year, scenario_id)
        _, denominator_cap = source_cap(context, denominator_source, origin_year, scenario_id)
        cap = min(numerator_cap, denominator_cap)
        numerator = context["scalar_series"][numerator_source][city]
        denominator = context["scalar_series"][denominator_source][city]
        aligned = sorted(
            year
            for year in set(numerator).intersection(denominator)
            if year <= cap
        )
        ratio_value = None
        latest_ratio_year = None
        missing_reason = ""
        for year in reversed(aligned):
            numerator_value = numerator[year]
            denominator_value = denominator[year]
            if name.startswith("life_log_"):
                if numerator_value > 0 and denominator_value > 0:
                    ratio_value = math.log(numerator_value) - math.log(denominator_value)
                    latest_ratio_year = year
                    break
            elif denominator_value != 0:
                ratio_value = numerator_value / denominator_value
                latest_ratio_year = year
                break
        if latest_ratio_year is None:
            missing_reason = (
                "no_aligned_year_as_of_common_cap"
                if not aligned
                else "ratio_denominator_invalid_or_non_positive_log_inputs"
            )
        features[name] = ratio_value
        audit[f"life_ratio_{name}_latest_year"] = latest_ratio_year
        audit[f"life_ratio_{name}_missing_reason"] = missing_reason
    return {
        "features": features,
        "audit": audit,
        "used_years": sorted(set(used_years)),
    }


def build_row(
    context: dict[str, Any],
    target_id: str,
    scenario_id: str,
    origin_year: int,
    city: str,
) -> dict[str, Any]:
    population_result = build_population_result(context, target_id, city, origin_year, scenario_id)
    results = {
        "FG-P": population_result,
        "FG-D": build_density_result(context, city, origin_year, scenario_id),
        "FG-U": build_urbanization_result(context, city, origin_year, scenario_id),
        "FG-UNEMP": build_unemployment_result(context, city, origin_year, scenario_id),
        "FG-EMP": build_employees_result(context, city, origin_year, scenario_id),
        "FG-SECTOR": build_sector_result(context, city, origin_year, scenario_id),
        "FG-WAGE": build_wage_result(context, city, origin_year, scenario_id),
        "FG-AGE": build_age_result(context, city, origin_year, scenario_id),
        "FG-LIFE": build_life_result(context, city, origin_year, scenario_id),
    }
    features: dict[str, Any] = {}
    audit: dict[str, Any] = {"feature_contract_version": "S5-Plan04-FULL-MULTISOURCE-V1"}
    for group in GROUPS:
        result = results[group["group_id"]]
        features.update(result["features"])
        audit.update(result["audit"])
    assert set(features) == set(MODEL_FEATURES)
    target_series = context["population"][target_id][city]
    actual = target_series.get(origin_year)
    b3_prediction = population_result["b3_prediction"]
    residual = actual - b3_prediction if actual is not None and b3_prediction is not None else None
    row: dict[str, Any] = {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "origin_year": origin_year,
        "city_id": city,
        "target_history_cutoff_year": origin_year - HORIZON_YEARS,
        "actual_target_value": actual,
        "b3_prediction": b3_prediction,
        "b3_residual_actual_minus_prediction": residual,
        "b3_squared_error": residual * residual if residual is not None else None,
        **features,
        **audit,
    }
    assert set(row) == set(MATRIX_FIELDS)
    return row


def build_matrix(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for origin_year in COMMON_ORIGINS:
                for city in CITIES:
                    rows.append(build_row(context, target_id, scenario_id, origin_year, city))
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    assert len(rows) == expected
    keys = [tuple(row[field] for field in ("target_id", "scenario_id", "origin_year", "city_id")) for row in rows]
    assert len(keys) == len(set(keys))
    assert {row["origin_year"] for row in rows} == set(COMMON_ORIGINS)
    assert {row["city_id"] for row in rows} == set(CITIES)
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            subset = [row for row in rows if row["target_id"] == target_id and row["scenario_id"] == scenario_id]
            assert len(subset) == len(COMMON_ORIGINS) * len(CITIES)
    return rows


def b3_compatibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {"resident": 872.452651515143, "registered": 225.421212121213}
    observed: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            subset = [row for row in rows if row["target_id"] == target_id and row["scenario_id"] == scenario_id]
            scoreable = [row for row in subset if row["actual_target_value"] is not None and row["b3_prediction"] is not None]
            assert len(scoreable) == 440, (target_id, scenario_id, len(scoreable))
            pooled = math.fsum(float(row["b3_squared_error"]) for row in scoreable) / len(scoreable)
            origin_mse = {
                origin: math.fsum(
                    float(row["b3_squared_error"])
                    for row in scoreable
                    if row["origin_year"] == origin
                ) / 40
                for origin in COMMON_ORIGINS
            }
            latest = origin_mse[2021]
            observed[f"{target_id}_{scenario_id}"] = {
                "pooled_mse": pooled,
                "macro_origin_mse": math.fsum(origin_mse.values()) / len(origin_mse),
                "latest_2021_mse": latest,
                "scoreable_rows": len(scoreable),
            }
            checks.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "pooled_delta_vs_frozen": pooled - expected[target_id],
                    "macro_delta_vs_frozen": observed[f"{target_id}_{scenario_id}"]["macro_origin_mse"] - expected[target_id],
                    "status": "PASS"
                    if abs(pooled - expected[target_id]) <= 1e-9
                    and abs(observed[f"{target_id}_{scenario_id}"]["macro_origin_mse"] - expected[target_id]) <= 1e-9
                    else "FAIL",
                }
            )
    assert all(check["status"] == "PASS" for check in checks)
    return {"expected_pooled_mse": expected, "observed": observed, "checks": checks, "pass": True}


def r1_r2_feature_differences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {
        (row["target_id"], row["scenario_id"], row["origin_year"], row["city_id"]): row
        for row in rows
    }
    pair_count = 0
    differing_pairs = 0
    group_counts = {group["group_id"]: 0 for group in GROUPS}
    source_counts = {source_id: 0 for source_id in SOURCE_ORDER}
    reference_equivalent = True
    for target_id in TARGETS:
        for origin_year in COMMON_ORIGINS:
            for city in CITIES:
                r1 = index[(target_id, SCENARIOS[0], origin_year, city)]
                r2 = index[(target_id, SCENARIOS[1], origin_year, city)]
                pair_count += 1
                if any(
                    (r1[field] is None) != (r2[field] is None)
                    or (
                        r1[field] is not None
                        and r2[field] is not None
                        and abs(float(r1[field]) - float(r2[field])) > 1e-12
                    )
                    for field in REFERENCE_COLUMNS + LABEL_COLUMNS
                ):
                    reference_equivalent = False
                pair_diff = False
                for group in GROUPS:
                    differs = any(
                        (r1[field] is None) != (r2[field] is None)
                        or (
                            r1[field] is not None
                            and r2[field] is not None
                            and abs(float(r1[field]) - float(r2[field])) > 1e-12
                        )
                        for field in group["feature_names"]
                    )
                    if differs:
                        group_counts[group["group_id"]] += 1
                        pair_diff = True
                        for source_id in group["sources"]:
                            source_counts[source_id] += 1
                differing_pairs += int(pair_diff)
    return {
        "row_pairs_compared": pair_count,
        "row_pairs_with_any_model_feature_difference": differing_pairs,
        "feature_difference_pair_count_by_group": group_counts,
        "feature_difference_pair_count_by_source": source_counts,
        "label_and_reference_columns_r1_r2_equivalent": reference_equivalent,
        "difference_reason": (
            "R2 adds one year of lag only for annual sources whose observed 2023 snapshot lag is one; "
            "population target history remains T-2 and benchmark-only age retains its own boundary."
        ),
    }


def source_boundary_audit(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for source_id in SOURCE_ORDER:
            prefix = f"{source_id}_"
            used_text = row[f"{prefix}used_years"]
            used_years = [int(value) for value in used_text.split(";") if value]
            domain_type = row[f"{prefix}domain_type"]
            domain_years = {
                int(part)
                for part in context["geometry"][source_id]["domain_years"].split(";")
                if part.strip()
            }
            domain_violation = sum(year not in domain_years for year in used_years) if domain_type == "benchmark_only" else 0
            cap = int(row[f"{prefix}required_cap_year"])
            cap_status = all(year <= cap for year in used_years)
            q_status = row[f"{prefix}quarantine_reintroduction_status"]
            status = cap_status and domain_violation == 0 and q_status == "PASS"
            result.append(
                {
                    "target_id": row["target_id"],
                    "scenario_id": row["scenario_id"],
                    "origin_year": row["origin_year"],
                    "city_id": row["city_id"],
                    "source_id": source_id,
                    "required_lag_years": row[f"{prefix}required_lag_years"],
                    "required_cap_year": cap,
                    "domain_type": domain_type,
                    "allowed_domain_years": ";".join(str(year) for year in sorted(domain_years)),
                    "latest_used_year": row[f"{prefix}latest_used_year"],
                    "source_age_years": row[f"{prefix}source_age_years"],
                    "usable_observation_count": row[f"{prefix}usable_observation_count"],
                    "used_years": used_text,
                    "used_year_max": row[f"{prefix}used_year_max"],
                    "cap_violation_count": sum(year > cap for year in used_years),
                    "benchmark_domain_violation_count": domain_violation,
                    "quarantine_keys_as_of_cap": row[f"{prefix}quarantine_keys_as_of_cap"],
                    "quarantine_used_count": row[f"{prefix}quarantine_used_count"],
                    "derived_used_count": row[f"{prefix}derived_used_count"],
                    "provenance_state": row[f"{prefix}provenance_state"],
                    "availability_state": row[f"{prefix}availability_state"],
                    "missing_reason": row[f"{prefix}missing_reason"],
                    "used_years_within_cap": "PASS" if cap_status else "FAIL",
                    "quarantine_reintroduction_status": q_status,
                    "benchmark_domain_status": "PASS" if domain_violation == 0 else "FAIL",
                    "audit_status": "PASS" if status else "FAIL",
                }
            )
    assert len(result) == len(rows) * len(SOURCE_ORDER)
    assert all(row["audit_status"] == "PASS" for row in result)
    assert all(int(row["quarantine_used_count"]) == 0 for row in result)
    return result


def feature_availability_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for origin_year in COMMON_ORIGINS:
                base_rows = [
                    row
                    for row in rows
                    if row["target_id"] == target_id
                    and row["scenario_id"] == scenario_id
                    and row["origin_year"] == origin_year
                ]
                assert len(base_rows) == len(CITIES)
                for group in GROUPS:
                    values = [row[name] for row in base_rows for name in group["feature_names"]]
                    nonmissing = sum(value is not None for value in values)
                    source_ages = [
                        row[f"{source_id}_source_age_years"]
                        for row in base_rows
                        for source_id in group["sources"]
                        if row[f"{source_id}_source_age_years"] is not None
                    ]
                    cities_with_info = sum(
                        any(row[name] is not None for name in group["feature_names"])
                        for row in base_rows
                    )
                    result.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "origin_year": origin_year,
                            "feature_group": group["group_id"],
                            "row_count": len(base_rows),
                            "feature_count": len(group["feature_names"]),
                            "non_missing_feature_cell_rate": nonmissing / (len(base_rows) * len(group["feature_names"])),
                            "cities_with_any_usable_information": cities_with_info,
                            "median_source_age_years": statistics.median(source_ages) if source_ages else None,
                            "max_source_age_years": max(source_ages) if source_ages else None,
                        }
                    )
    assert len(result) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(GROUPS)
    return result


def formula_for_feature(name: str, group_id: str) -> tuple[str, str, str]:
    if group_id == "FG-P":
        formulas = {
            "pop_latest_level": "latest non-missing target value <= origin_year - 2",
            "pop_log1p_latest": "log1p(pop_latest_level)",
            "pop_latest_observation_year": "year of pop_latest_level",
            "pop_history_observation_count": "count of non-missing target observations <= origin_year - 2",
            "pop_source_age_years": "origin_year - pop_latest_observation_year",
            "pop_latest_first_difference": "latest target level minus previous target level",
            "pop_latest_relative_growth": "(latest - previous) / previous / actual year gap",
            "pop_recent3_ols_slope": "OLS slope on the latest three target observations <= origin_year - 2",
            "pop_expanding_ols_slope": "OLS slope on all target observations <= origin_year - 2",
            "pop_recent_vs_expanding_slope_disagreement_abs": "abs(pop_recent3_ols_slope - pop_expanding_ols_slope)",
            "pop_history_difference_volatility": "population standard deviation of all historical first differences <= origin_year - 2",
            "pop_relative_history_difference_volatility": "population standard deviation of annualized relative historical first differences",
            "pop_recent_acceleration_abs": "absolute difference between the latest two historical first differences",
            "pop_b1_prediction": "frozen B1 two-point trend prediction at origin_year",
            "pop_b2_prediction": "frozen B2 expanding per-city OLS prediction at origin_year",
            "pop_b3_prediction": "frozen B3 recent-3 OLS prediction at origin_year",
            "pop_baseline_prediction_spread": "max(B1,B2,B3) - min(B1,B2,B3)",
        }
        return formulas[name], "none", "target history <= T-2"
    if name.endswith("_log1p_latest"):
        return "log1p(latest valid positive source level)", "log1p", "latest source observations <= source-specific cap"
    if name.endswith("_latest_observation_year") or name.endswith("_latest_benchmark_year"):
        return "year of latest valid source observation under the source cap", "calendar_year", "source-specific cap"
    if name.endswith("_source_age_years"):
        return "origin_year - latest valid source observation year", "calendar_year_delta", "source-specific cap"
    if name.endswith("_usable_observation_count") or name.endswith("_usable_common_year_observation_count"):
        return "count of valid observations under the source cap", "count", "source-specific cap"
    if name.endswith("_available_flag") or name.endswith("_flag"):
        return "1 when the corresponding source feature is available, otherwise 0", "binary", "source-specific cap"
    if "recent_absolute_change" in name:
        return "latest valid level minus previous valid level", "difference", "latest two valid observations under source cap"
    if "recent_relative_change" in name:
        return "(latest valid level - previous valid level) / previous level / actual year gap", "annualized_relative_change", "latest two valid observations under source cap"
    if "recent3_log_slope" in name:
        return "OLS slope on log positive values from the latest three valid observations", "log_slope", "latest three valid observations under source cap"
    if "recent3_slope" in name:
        return "OLS slope on the latest three valid source observations", "slope", "latest three valid observations under source cap"
    if name.endswith("_latest_level") or name.endswith("_latest_rate"):
        return "latest valid source level under the source cap", "level", "source-specific cap"
    if name.startswith("sector_latest_"):
        return "latest same-year valid sector component or derived total/share", "level_or_share", "same-year sector components <= source cap"
    if name.startswith("sector_recent_"):
        return "latest valid sector share minus previous valid same-year share", "share_difference", "same-year sector components <= source cap"
    if name.startswith("age_"):
        return "latest valid benchmark age composition or change using actual benchmark years", "benchmark_level_or_share", "benchmark-only cap; no interpolation"
    if name.startswith("life_"):
        return "latest aligned-year living-standard ratio or log gap under both source caps", "ratio_or_log_gap", "same allowed year under both source-specific caps"
    return f"model-agnostic {group_id} feature {name}", "none", "source-specific cap"


def feature_definition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        source_ids = ";".join(group["sources"])
        for name in group["feature_names"]:
            formula, transform, asof_rule = formula_for_feature(name, group["group_id"])
            rows.append(
                {
                    "feature_name": name,
                    "feature_group": group["group_id"],
                    "source_id": source_ids,
                    "role": "model_feature",
                    "formula": formula,
                    "transform": transform,
                    "required_history": "as stated by formula; unavailable history remains missing",
                    "missing_rule": "store NaN and expose source/group audit reason; no Plan04 imputation",
                    "asof_rule": asof_rule,
                    "categorical": "no",
                    "model_feature": "yes",
                    "feature_contract_version": "S5-Plan04-FULL-MULTISOURCE-V1",
                }
            )
    column_rows = (
        [(name, "IDENTITY", "identity", "no") for name in IDENTITY_COLUMNS]
        + [(name, "LABEL", "label_or_reference", "no") for name in LABEL_COLUMNS]
        + [(name, "REFERENCE", "label_or_reference", "no") for name in REFERENCE_COLUMNS]
    )
    for name, group_id, role, categorical in column_rows:
        rows.append(
            {
                "feature_name": name,
                "feature_group": group_id,
                "source_id": "",
                "role": role,
                "formula": "row identity or label/reference-only column; never part of Plan04 model feature set",
                "transform": "none",
                "required_history": "not applicable",
                "missing_rule": "blank when unavailable or not scoreable",
                "asof_rule": "identity/label/reference contract",
                "categorical": "yes" if name == "city_id" else "no",
                "model_feature": "no",
                "feature_contract_version": "S5-Plan04-FULL-MULTISOURCE-V1",
            }
        )
    assert len({row["feature_name"] for row in rows}) == len(rows)
    return rows


def feature_group_manifest_rows() -> list[dict[str, Any]]:
    return [
        {
            "feature_group": group["group_id"],
            "sources": ";".join(group["sources"]),
            "feature_count": len(group["feature_names"]),
            "mechanism": group["mechanism"],
            "quality_boundary": group["quality_boundary"],
            "expected_model_use": group["expected_model_use"],
        }
        for group in GROUPS
    ]


def schema_json() -> dict[str, Any]:
    return {
        "feature_contract_version": "S5-Plan04-FULL-MULTISOURCE-V1",
        "plan_id": PLAN_ID,
        "row_key": ["target_id", "scenario_id", "origin_year", "city_id"],
        "identity_columns": list(IDENTITY_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "reference_columns": list(REFERENCE_COLUMNS),
        "model_features": list(MODEL_FEATURES),
        "categorical_features": ["city_id"],
        "audit_columns": list(AUDIT_COLUMNS),
        "feature_groups": {
            group["group_id"]: {
                "sources": list(group["sources"]),
                "features": list(group["feature_names"]),
            }
            for group in GROUPS
        },
        "missing_value_contract": {
            "numeric_model_features": "NaN in CSV; no global fill, forward fill, zero fill, or model imputation",
            "labels_and_references": "blank when unavailable; actual_target_value is scoring label only",
            "audit_metadata": "blank or explicit state/reason; source-specific missingness is preserved",
            "downstream_linear_policy": "fold-local SimpleImputer plus indicator fitted only inside training fold",
            "downstream_tree_policy": "document model-specific missing handling and fit preprocessing only inside training fold",
        },
        "city_id_policy": "raw categorical identifier retained; no one-hot encoding in Plan04",
        "target_history_boundary": "all population target-history-derived features use observations <= origin_year - 2",
        "source_geometry": "R1_REFERENCE_SNAPSHOT_LAG and R2_CONSERVATIVE_PUBLICATION_STRESS from S2 replay artifacts",
    }


def foundation_summary(
    context: dict[str, Any],
    rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    compatibility: dict[str, Any],
    r1_r2: dict[str, Any],
) -> dict[str, Any]:
    source_feature_nonmissing = {
        group["group_id"]: {
            "feature_count": len(group["feature_names"]),
            "nonmissing_cells": sum(
                row[name] is not None for row in rows for name in group["feature_names"]
            ),
            "total_cells": len(rows) * len(group["feature_names"]),
        }
        for group in GROUPS
    }
    return {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "base_sha": BASE_SHA,
        "worker_branch": BRANCH,
        "build_commit": git_text("rev-parse", "HEAD"),
        "feature_contract_version": "S5-Plan04-FULL-MULTISOURCE-V1",
        "logical_source_groups": list(SOURCE_ORDER),
        "logical_source_group_count": len(SOURCE_ORDER),
        "target_ids": list(TARGETS),
        "scenario_ids": list(SCENARIOS),
        "common_origins": list(COMMON_ORIGINS),
        "city_count": len(CITIES),
        "expected_matrix_rows": 1760,
        "matrix_rows": len(rows),
        "unique_key_count": len({tuple(row[field] for field in ("target_id", "scenario_id", "origin_year", "city_id")) for row in rows}),
        "feature_group_counts": {group["group_id"]: len(group["feature_names"]) for group in GROUPS},
        "model_feature_count": len(MODEL_FEATURES),
        "identity_column_count": len(IDENTITY_COLUMNS),
        "label_column_count": len(LABEL_COLUMNS),
        "reference_column_count": len(REFERENCE_COLUMNS),
        "audit_column_count": len(AUDIT_COLUMNS),
        "feature_nonmissing_summary": source_feature_nonmissing,
        "feature_availability_rows": len(availability_rows),
        "source_boundary_audit_rows": len(boundary_rows),
        "source_status": {
            source_id: {
                "source_status": context["geometry"][source_id]["source_status"],
                "domain_type": context["geometry"][source_id]["domain_type"],
                "observed_latest_year": context["geometry"][source_id]["latest_usable_year"],
                "observed_snapshot_lag_years": context["geometry"][source_id]["observed_snapshot_freshness_lag_years"],
            }
            for source_id in SOURCE_ORDER
        },
        "b3_compatibility": compatibility,
        "r1_r2_feature_difference": r1_r2,
        "boundary_checks": {
            "target_history_max_year_never_exceeds_T_minus_2": all(
                row["population_history_cutoff_pass"] == "PASS" for row in rows
            ),
            "source_used_years_never_exceed_cap": all(
                row["used_years_within_cap"] == "PASS" for row in boundary_rows
            ),
            "quarantine_reintroduction_count": sum(
                int(row["quarantine_used_count"]) for row in boundary_rows
            ),
            "benchmark_domain_violation_count": sum(
                int(row["benchmark_domain_violation_count"]) for row in boundary_rows
            ),
            "density_dynamic_guard_all_pass": all(
                row["density_dynamic_guard_status"] == "PASS" for row in rows
            ),
            "employment_provenance_rows": len(context["sources"]["employees"]["derived_keys"]),
            "new_external_source_used": False,
            "test_xls_used": False,
            "annual_interpolation_used": False,
            "global_fill_used": False,
            "imputer_fitted": False,
            "feature_selection_by_target_performance": False,
            "model_fit_run": False,
            "routing_or_gate_model_run": False,
            "final_2023_forecast_run": False,
            "submission_run": False,
        },
        "oof_prediction_contract_defined": True,
        "oof_predictions_fabricated": False,
        "optional_2023_feature_readiness_audit_executed": False,
        "recommendation": "parallel_plan05_wave_safe_after_main_review_and_integration",
        "result": "full_multisource_foundation_ready",
        "s5_complete": False,
        "s6_entered": False,
    }


def handoff_text(summary: dict[str, Any]) -> str:
    compatibility = summary["b3_compatibility"]["observed"]
    r1_r2 = summary["r1_r2_feature_difference"]
    lines = [
        f"# {PLAN_ID} Handoff — 全量多表特征基座与城市异质性接口",
        "",
        "## Result",
        "",
        f"- Plan ID：`{PLAN_ID}`",
        f"- Stage：`{STAGE}`",
        f"- Frozen base：`{BASE_SHA}`",
        f"- Worker branch：`{BRANCH}`",
        f"- Exact worker commit at handoff generation：`{summary['build_commit']}`",
        "- Foundation result：`full_multisource_foundation_ready`",
        "- S5 未声明完成，S6 未进入；Stage authority 仍为 Main。",
        "",
        "## Scope and actual files",
        "",
        "本 worker 只建立 model-agnostic full multi-source matrix 与接口，不训练模型、不调参、不做 feature selection、不生成 2023 人口预测或 submission。",
        "",
        "实际生成文件：",
        "",
        "- `scripts/build_s5_plan04_full_multisource_feature_foundation.py`",
        "- `reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan04/feature_group_manifest.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan04/feature_availability_summary.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan04/source_boundary_audit.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan04/feature_matrix_schema.json`",
        "- `reports/S5_HANDOFFS/S5-Plan04/foundation_summary.json`",
        "- `reports/S5_HANDOFFS/S5-Plan04/oof_prediction_contract.csv`",
        "",
        "## Commands executed",
        "",
        "- `python -m py_compile scripts/build_s5_plan04_full_multisource_feature_foundation.py`",
        "- `python scripts/build_s5_plan04_full_multisource_feature_foundation.py`",
        "- `git diff --check`",
        "",
        "## Frozen matrix checks",
        "",
        f"- Rows：`{summary['matrix_rows']}` / expected `1760`; unique keys：`{summary['unique_key_count']}`.",
        "- Keys：`(target_id, scenario_id, origin_year, city_id)`; targets resident/registered; scenarios R1/R2; origins 2011–2021; 40 cities.",
        f"- Model features：`{summary['model_feature_count']}` across `{summary['logical_source_group_count']}` logical source groups and `{len(GROUPS)}` semantic feature groups.",
        "- `city_id` remains raw categorical metadata; no one-hot encoding is performed.",
        "- Numeric model missingness is written as `NaN`; no global fill or Plan04 imputer is fitted.",
        "",
        "## Source and leakage evidence",
        "",
        "- S2 `replay_scenario_definitions.csv` and `reference_forecast_geometry.csv` are reused; no publication date is invented.",
        "- Population target-history features and B1/B2/B3 references use `<= T-2` only; origin labels are label/reference-only.",
        "- Annual source caps are checked per source and scenario; age remains benchmark-only `{2000, 2010, 2020}` with no interpolation.",
        "- Density regime crossings suppress only affected dynamics; levels and the retained regime audit remain visible.",
        "- Employment derived rows remain provenance-flagged; quarantined values are not reintroduced.",
        f"- Source boundary audit rows：`{summary['source_boundary_audit_rows']}`; cap violations：`{sum(int(row['cap_violation_count']) for row in read_csv(OUTPUT_DIR / 'source_boundary_audit.csv'))}`; quarantine reintroductions：`{summary['boundary_checks']['quarantine_reintroduction_count']}`.",
        "",
        "## B3 compatibility",
        "",
        f"- resident R1 pooled MSE：`{compatibility['resident_R1_REFERENCE_SNAPSHOT_LAG']['pooled_mse']:.15f}`; R2：`{compatibility['resident_R2_CONSERVATIVE_PUBLICATION_STRESS']['pooled_mse']:.15f}`.",
        f"- registered R1 pooled MSE：`{compatibility['registered_R1_REFERENCE_SNAPSHOT_LAG']['pooled_mse']:.15f}`; R2：`{compatibility['registered_R2_CONSERVATIVE_PUBLICATION_STRESS']['pooled_mse']:.15f}`.",
        "- Frozen expected values passed within floating tolerance: resident `872.452651515143`, registered `225.421212121213`.",
        "",
        "## R1/R2 feature differences",
        "",
        f"- Compared `{r1_r2['row_pairs_compared']}` target×origin×city pairs; pairs with any model-feature difference：`{r1_r2['row_pairs_with_any_model_feature_difference']}`.",
        f"- Label/reference columns numerically equivalent across R1/R2：`{r1_r2['label_and_reference_columns_r1_r2_equivalent']}`.",
        "- Differences are expected from source-specific R2 stress caps; target history and benchmark-only age retain separate geometry.",
        "",
        "## 2023 readiness and forbidden actions",
        "",
        "- Optional 2023 feature-readiness audit：not executed; no 2023 population prediction was created.",
        "- No LightGBM/CatBoost/XGBoost/ExtraTrees/Ridge/HGB/neural training, tuning, target-based feature ranking, routing, blend, gating, or submission was run.",
        "",
        "## Recommendation to Main",
        "",
        "- The common feature contract is safe for review and, after Main integrates this worker tip, can be consumed by parallel Plan05A/B/C/D workers.",
        "- Future workers must consume this exact contract and return `feature_contract_change_required` rather than privately extending the matrix.",
        "- Main must independently review/integrate Plan04; this handoff does not declare S5 complete or authorize S6.",
        "",
        "## Unresolved risks",
        "",
        "- Target proxy and source publication timing remain conditional S2 assumptions.",
        "- Downstream model-specific missing-value handling and city-aware variants remain untested by design.",
        "- Feature foundation completeness does not establish that any future model will beat B3.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    context = load_context()
    rows = build_matrix(context)
    compatibility = b3_compatibility(rows)
    r1_r2 = r1_r2_feature_differences(rows)
    assert r1_r2["label_and_reference_columns_r1_r2_equivalent"]
    boundary_rows = source_boundary_audit(rows, context)
    availability_rows = feature_availability_summary(rows)
    summary = foundation_summary(context, rows, boundary_rows, availability_rows, compatibility, r1_r2)
    assert summary["boundary_checks"]["target_history_max_year_never_exceeds_T_minus_2"]
    assert summary["boundary_checks"]["source_used_years_never_exceed_cap"]
    assert summary["boundary_checks"]["quarantine_reintroduction_count"] == 0
    assert summary["boundary_checks"]["benchmark_domain_violation_count"] == 0
    assert summary["boundary_checks"]["density_dynamic_guard_all_pass"]
    assert summary["logical_source_group_count"] == 14
    assert summary["matrix_rows"] == 1760

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "full_multisource_feature_matrix.csv",
        rows,
        fields=MATRIX_FIELDS,
        model_features=set(MODEL_FEATURES),
    )
    write_csv(OUTPUT_DIR / "feature_definitions.csv", feature_definition_rows())
    write_csv(OUTPUT_DIR / "feature_group_manifest.csv", feature_group_manifest_rows())
    write_csv(OUTPUT_DIR / "feature_availability_summary.csv", availability_rows)
    write_csv(OUTPUT_DIR / "source_boundary_audit.csv", boundary_rows)
    write_json(OUTPUT_DIR / "feature_matrix_schema.json", schema_json())
    write_json(OUTPUT_DIR / "foundation_summary.json", summary)
    write_csv(OUTPUT_DIR / "oof_prediction_contract.csv", fields=OOF_FIELDS)
    (OUTPUT_DIR / "HANDOFF.md").write_text(handoff_text(summary), encoding="utf-8")

    print("S5_PLAN04_FULL_MULTISOURCE_FOUNDATION_READY")
    print(f"base_sha={BASE_SHA}")
    print(f"worker_branch={BRANCH}")
    print(f"matrix_rows={len(rows)}")
    print(f"unique_keys={summary['unique_key_count']}")
    print(f"logical_source_groups={len(SOURCE_ORDER)}")
    print(f"model_feature_count={len(MODEL_FEATURES)}")
    print(f"b3_compatibility_pass={compatibility['pass']}")
    print(f"source_boundary_audit_rows={len(boundary_rows)}")
    print(f"r1_r2_feature_difference_pairs={r1_r2['row_pairs_with_any_model_feature_difference']}")
    print("model_fit_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s5_complete=False")
    print("s6_entered=False")
    print("result=full_multisource_foundation_ready")


if __name__ == "__main__":
    main()
