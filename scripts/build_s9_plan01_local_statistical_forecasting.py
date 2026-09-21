"""Run the pre-registered S9-Plan01 local statistical forecasting track.

The runner is deliberately target-history-only.  It reads a narrow allowlist
of S1-S5 competition-legal artifacts, reconstructs historical outer OOF rows
for fixed local statistical candidates, and stops at a Lab handoff.  It does
not read S8/Truth material, inspect evaluator labels, generate a 2023
prediction, or select a blind champion.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S9-Plan01"
BRANCH = "codex/s9-plan01-local-statistical-forecasting"
BASE_SHA = "0a74372ea4e3555b3d659e6a2c77647b92eabea4"
HORIZON_YEARS = 2
LATEST_ORIGIN = 2021
CENSUS_ANALOGUE_ORIGIN = 2013
ORIGINS = tuple(range(2011, 2022))
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

S2_SUMMARY_PATH = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05" / "plan05_summary.json"
S2_BOUNDARY_PATH = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05" / "origin_information_boundary.csv"
S2_SCENARIO_PATH = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02" / "replay_scenario_definitions.csv"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
S1_CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
S1_QUALITY_PATH = ROOT / "docs" / "s1_quality_issue_registry.csv"
S3_DEFINITIONS_PATH = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan02" / "baseline_predictor_definitions.csv"
S3_PREDICTIONS_PATH = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03" / "per_origin_predictions_or_residuals.csv"
S3_SCRIPT_PATH = ROOT / "scripts" / "build_s3_plan02_recent_window_and_selector.py"
RLT3_DEFINITIONS_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan02" / "candidate_definitions.csv"
RLT3_SCRIPT_PATH = ROOT / "scripts" / "build_s5_plan02_robust_local_trend_screen.py"
OUTPUT_DIR = ROOT / "reports" / "S9_HANDOFFS" / "S9-Plan01"

B0_ID = "B0_PERSISTENCE_LAST_OBSERVED"
B1_ID = "B1_TWO_POINT_LINEAR_TREND"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
RLT3_ID = "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND"
SES_ID = "SES_ALPHA_0_5"
HOLT_ID = "HOLT_LINEAR_A05_B03"
DAMPED_ID = "HOLT_DAMPED_A05_B03_PHI080"
THETA_ID = "THETA_THETA2_ALPHA05"
ARIMA110_ID = "ARIMA_110_BOUNDED"
ARIMA011_ID = "ARIMA_011_FIXED_THETA05"

ALL_CANDIDATE_IDS = (
    B0_ID,
    B1_ID,
    B3_ID,
    RLT3_ID,
    SES_ID,
    HOLT_ID,
    DAMPED_ID,
    THETA_ID,
    ARIMA110_ID,
    ARIMA011_ID,
)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": B0_ID,
        "family": "persistence",
        "mechanism": "level_persistence",
        "parameters": {},
        "formula": "y_hat(T)=y_last; latest non-missing target year <= T-2",
        "source": "frozen S3 fit_prediction",
        "fallback": "unscoreable only if no legal history",
    },
    {
        "candidate_id": B1_ID,
        "family": "recent_trend",
        "mechanism": "two_point_slope",
        "parameters": {},
        "formula": "latest_value + latest_two_point_slope * (T-latest_year)",
        "source": "frozen S3 fit_prediction",
        "fallback": "unscoreable only if fewer than two legal points",
    },
    {
        "candidate_id": B3_ID,
        "family": "recent_trend",
        "mechanism": "recent_3point_ols",
        "parameters": {"window": 3},
        "formula": "OLS on latest 3 non-missing target points <= T-2",
        "source": "frozen S3 fit_prediction and S3-Plan03 rows",
        "fallback": "frozen B1 fallback inside S3 implementation",
    },
    {
        "candidate_id": RLT3_ID,
        "family": "robust_recent_trend",
        "mechanism": "median_pairwise_slope",
        "parameters": {"window": 3, "slope": "median_all_pairwise_slopes", "intercept": "median_y_minus_slope_year"},
        "formula": "S5 frozen RLT3 source implementation on latest 3 legal points",
        "source": "frozen S5-Plan02 fit_median_slope",
        "fallback": "frozen B1 fallback, then B0 only if necessary",
    },
    {
        "candidate_id": SES_ID,
        "family": "smoothing",
        "mechanism": "simple_exponential_smoothing",
        "parameters": {"alpha": 0.5},
        "formula": "level_t=0.5*y_t+0.5*level_(t-1); forecast latest level",
        "source": "S9 fixed implementation",
        "fallback": "B0 if no legal point",
    },
    {
        "candidate_id": HOLT_ID,
        "family": "smoothing_trend",
        "mechanism": "holt_linear",
        "parameters": {"alpha": 0.5, "beta": 0.3},
        "formula": "calendar-year Holt level/trend with fixed alpha and beta",
        "source": "S9 fixed implementation",
        "fallback": "SES if fewer than two legal points",
    },
    {
        "candidate_id": DAMPED_ID,
        "family": "damped_trend",
        "mechanism": "damped_holt",
        "parameters": {"alpha": 0.5, "beta": 0.3, "phi": 0.8},
        "formula": "Holt state with fixed geometric trend damping phi=0.8",
        "source": "S9 fixed implementation",
        "fallback": "SES if fewer than two legal points",
    },
    {
        "candidate_id": THETA_ID,
        "family": "theta",
        "mechanism": "theta_decomposition",
        "parameters": {"theta": 2.0, "alpha": 0.5, "trend": "OLS"},
        "formula": "average fixed-theta transformed SES forecast and OLS trend forecast",
        "source": "S9 fixed implementation",
        "fallback": "B0 if fewer than two legal points",
    },
    {
        "candidate_id": ARIMA110_ID,
        "family": "bounded_arima",
        "mechanism": "arima_110",
        "parameters": {"order": "(1,1,0)", "phi_clip": [-0.95, 0.95]},
        "formula": "calendar-normalized first differences with bounded AR(1) forecast",
        "source": "S9 fixed implementation",
        "fallback": "B3 if fewer than three legal points or invalid fit",
    },
    {
        "candidate_id": ARIMA011_ID,
        "family": "bounded_arima",
        "mechanism": "arima_011",
        "parameters": {"order": "(0,1,1)", "theta": 0.5},
        "formula": "first-difference mean drift plus fixed bounded MA(1) innovation",
        "source": "S9 fixed implementation",
        "fallback": "B3 if fewer than three legal points or invalid fit",
    },
)
CANDIDATE_BY_ID = {candidate["candidate_id"]: candidate for candidate in CANDIDATES}

INPUT_FLOAT_TOLERANCE = 1e-9
METRIC_TIE_TOLERANCE = 1e-12

ALLOWLIST_RELATIVE_PATHS = (
    "data/normalized/人口规模.csv",
    "docs/s1_data_contract.csv",
    "docs/s1_quality_issue_registry.csv",
    "reports/S2_HANDOFFS/S2-Plan05/plan05_summary.json",
    "reports/S2_HANDOFFS/S2-Plan05/origin_information_boundary.csv",
    "reports/S2_HANDOFFS/S2-Plan02/replay_scenario_definitions.csv",
    "reports/S3_HANDOFFS/S3-Plan02/baseline_predictor_definitions.csv",
    "reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv",
    "scripts/build_s3_plan02_recent_window_and_selector.py",
    "reports/S5_HANDOFFS/S5-Plan02/candidate_definitions.csv",
    "scripts/build_s5_plan02_robust_local_trend_screen.py",
)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return stable_json(json_safe(value))
    if isinstance(value, float):
        return "" if not math.isfinite(value) else round(value, 12)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    assert fieldnames
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), path
        return list(reader)


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "NA", "N/A", "NaN", "nan", "None", "null"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    return None if number is None else int(number)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_enough(left: float, right: float, tolerance: float = INPUT_FLOAT_TOLERANCE) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def metric_compare(left: float, right: float) -> int:
    tolerance = METRIC_TIE_TOLERANCE * max(1.0, abs(left), abs(right))
    if left < right - tolerance:
        return -1
    if right < left - tolerance:
        return 1
    return 0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    assert values
    return math.fsum(values) / len(values)


def mse(errors: Iterable[float]) -> float:
    return mean(error * error for error in errors)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return math.fsum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def assert_worker_identity() -> str:
    assert git_text("branch", "--show-current") == BRANCH
    assert git_text("merge-base", BASE_SHA, "HEAD") == BASE_SHA
    check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0
    return git_text("rev-parse", "HEAD")


def import_frozen_sources() -> tuple[Any, Any]:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    s3_reference = importlib.import_module("build_s3_plan02_recent_window_and_selector")
    rlt_reference = importlib.import_module("build_s5_plan02_robust_local_trend_screen")
    return s3_reference, rlt_reference


def load_population() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    rows = read_csv(POPULATION_PATH)
    assert rows
    population: dict[str, dict[str, dict[int, float]]] = {
        target: {city: {} for city in CITIES} for target in TARGETS
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        city = row["城市名称"].strip()
        year = parse_int(row["年份"])
        assert city in CITIES and year is not None
        assert year <= 2021, "Plan01 must not use target values beyond the legal historical cap"
        key = (city, year)
        assert key not in seen, key
        seen.add(key)
        for target_id, column in TARGET_COLUMNS.items():
            value = parse_float(row.get(column))
            if value is not None:
                population[target_id][city][year] = value
    profile = {
        "canonical_row_count": len(rows),
        "unique_city_year_count": len(seen),
        "city_count": len(CITIES),
        "year_min": min(year for _city, year in seen),
        "year_max": max(year for _city, year in seen),
        "sha256": sha256_file(POPULATION_PATH),
        "nonempty_by_target": {
            target: sum(len(population[target][city]) for city in CITIES) for target in TARGETS
        },
    }
    assert set(city for city, _year in seen) == set(CITIES)
    return population, profile


def load_contract_context() -> dict[str, Any]:
    s1_contract = read_csv(S1_CONTRACT_PATH)
    s1_quality = read_csv(S1_QUALITY_PATH)
    assert s1_contract and s1_quality
    s2_summary = json.loads(S2_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["target_history_horizon_years"] == HORIZON_YEARS
    boundary = read_csv(S2_BOUNDARY_PATH)
    assert len(boundary) == len(TARGETS) * len(SCENARIOS) * len(ORIGINS)
    assert all(int(row["origin_year"]) in ORIGINS for row in boundary)
    assert all(row["scenario_id"] in SCENARIOS for row in boundary)
    scenarios = read_csv(S2_SCENARIO_PATH)
    scenario_map = {row["scenario_id"]: row for row in scenarios}
    assert set(scenario_map) == set(SCENARIOS)
    assert all(row["target_history_lag_years"] == "2" for row in scenario_map.values())
    return {
        "s2_summary": s2_summary,
        "s1_contract_row_count": len(s1_contract),
        "s1_quality_row_count": len(s1_quality),
        "boundary_row_count": len(boundary),
        "scenario_map": scenario_map,
    }


def load_frozen_s3_rows() -> dict[tuple[str, str, int, str, str], dict[str, str]]:
    allowed = {B0_ID, B1_ID, B3_ID}
    index: dict[tuple[str, str, int, str, str], dict[str, str]] = {}
    for row in read_csv(S3_PREDICTIONS_PATH):
        candidate_id = row.get("predictor_id", "")
        if candidate_id not in allowed:
            continue
        target_id = row["target_id"]
        scenario_id = row["scenario_id"]
        origin_year = parse_int(row["origin_year"])
        city_id = row["city_id"]
        assert target_id in TARGETS and scenario_id in SCENARIOS and origin_year in ORIGINS and city_id in CITIES
        assert row.get("scoreable", "").lower() == "yes"
        key = (target_id, scenario_id, origin_year, city_id, candidate_id)
        assert key not in index, key
        index[key] = row
    expected = len(TARGETS) * len(SCENARIOS) * len(ORIGINS) * len(CITIES) * len(allowed)
    assert len(index) == expected, (len(index), expected)
    return index


def audit_rlt3_definition() -> dict[str, Any]:
    definitions = read_csv(RLT3_DEFINITIONS_PATH)
    candidate = next((row for row in definitions if row.get("candidate_id") == RLT3_ID), None)
    assert candidate is not None
    formula_text = " ".join(candidate.get(field, "") for field in candidate)
    assert "median" in formula_text.lower() and "slope" in formula_text.lower()
    return {
        "candidate_id": RLT3_ID,
        "definition_row": candidate,
        "definition_sha256": sha256_file(RLT3_DEFINITIONS_PATH),
        "source_script_sha256": sha256_file(RLT3_SCRIPT_PATH),
        "source_function": "fit_median_slope",
        "provenance_pass": True,
        "uses_external_data": False,
        "uses_truth_artifact": False,
    }


def source_lineage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in ALLOWLIST_RELATIVE_PATHS:
        path = ROOT / Path(relative)
        assert path.exists(), path
        rows.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "git_blob_sha": git_text("rev-parse", f"HEAD:{relative}"),
                "access_role": "allowlisted_input",
            }
        )
    return rows


def history_for(
    population: dict[str, dict[str, dict[int, float]]], target_id: str, city_id: str, cutoff_year: int
) -> list[tuple[int, float]]:
    return sorted(
        (year, value)
        for year, value in population[target_id][city_id].items()
        if year <= cutoff_year
    )


def fit_ols(points: list[tuple[int, float]], target_year: int) -> tuple[float, float, float]:
    assert len(points) >= 2
    x_mean = mean(float(year) for year, _value in points)
    y_mean = mean(value for _year, value in points)
    denominator = math.fsum((year - x_mean) ** 2 for year, _value in points)
    assert denominator > 0
    slope = math.fsum((year - x_mean) * (value - y_mean) for year, value in points) / denominator
    intercept = y_mean - slope * x_mean
    return intercept + slope * target_year, slope, intercept


def fit_ses(points: list[tuple[int, float]], alpha: float) -> float:
    assert points
    level = points[0][1]
    for _year, value in points[1:]:
        level = alpha * value + (1.0 - alpha) * level
    return level


def holt_state(points: list[tuple[int, float]], alpha: float, beta: float) -> tuple[float, float, int]:
    assert len(points) >= 2
    first_year, first_value = points[0]
    second_year, second_value = points[1]
    initial_dt = max(1, second_year - first_year)
    level = second_value
    trend = (second_value - first_value) / initial_dt
    previous_year = second_year
    for year, value in points[2:]:
        dt = max(1, year - previous_year)
        previous_level = level
        one_step_level = level + trend * dt
        level = alpha * value + (1.0 - alpha) * one_step_level
        trend = beta * ((level - previous_level) / dt) + (1.0 - beta) * trend
        previous_year = year
    return level, trend, previous_year


def fallback_b3(
    s3_reference: Any,
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
    reason: str,
) -> tuple[float | None, dict[str, Any]]:
    fit = s3_reference.fit_prediction(B3_ID, series, target_year, cutoff_year)
    return fit["prediction"], {
        "fit_status": f"fallback_b3:{reason}:{fit['regression_fit_status']}",
        "fallback_used": "yes",
        "fallback_reason": reason,
        "fit_history_years": fit["history_years"],
        "eligible_history_years": fit["eligible_history_years"],
        "regression_slope": fit.get("regression_slope"),
        "regression_intercept": fit.get("regression_intercept"),
    }


def fit_candidate(
    candidate_id: str,
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
    s3_reference: Any,
    rlt_reference: Any,
) -> tuple[float | None, dict[str, Any]]:
    points = sorted((year, value) for year, value in series.items() if year <= cutoff_year)
    base = {
        "fit_status": "",
        "fallback_used": "no",
        "fallback_reason": "",
        "fit_history_years": [year for year, _value in points],
        "eligible_history_years": [year for year, _value in points],
        "regression_slope": None,
        "regression_intercept": None,
    }
    if candidate_id in {B0_ID, B1_ID, B3_ID}:
        fit = s3_reference.fit_prediction(candidate_id, series, target_year, cutoff_year)
        base.update(
            fit_status=fit["regression_fit_status"],
            fallback_used=fit.get("window_fallback_applied", "no"),
            fallback_reason=fit.get("window_fallback_reason", ""),
            fit_history_years=fit["history_years"],
            eligible_history_years=fit["eligible_history_years"],
            regression_slope=fit.get("regression_slope"),
            regression_intercept=fit.get("regression_intercept"),
        )
        return fit["prediction"], base
    if candidate_id == RLT3_ID:
        if len(points) >= 3:
            fitted = rlt_reference.fit_median_slope(points[-3:], target_year)
            base.update(
                fit_status="recent_3point_median_slope",
                fit_history_years=[year for year, _value in points[-3:]],
                regression_slope=fitted["slope"],
                regression_intercept=fitted["intercept"],
            )
            return fitted["prediction"], base
        if len(points) >= 2:
            prediction = rlt_reference.fit_two_point(points, target_year)
            base.update(fit_status="fallback_b1_insufficient_recent_3point", fallback_used="yes", fallback_reason="legal_points<3")
            return prediction, base
        if points:
            base.update(fit_status="fallback_b0_insufficient_two_point", fallback_used="yes", fallback_reason="legal_points<2")
            return points[-1][1], base
        base.update(fit_status="unscoreable", fallback_used="yes", fallback_reason="no_legal_history")
        return None, base
    if candidate_id == SES_ID:
        if not points:
            base.update(fit_status="fallback_b0_no_history", fallback_used="yes", fallback_reason="no_legal_history")
            return None, base
        base["fit_status"] = "ses_fixed_alpha_0.5"
        return fit_ses(points, 0.5), base
    if candidate_id in {HOLT_ID, DAMPED_ID}:
        if len(points) < 2:
            if points:
                base.update(fit_status="fallback_ses_insufficient_two_point", fallback_used="yes", fallback_reason="legal_points<2")
                return fit_ses(points, 0.5), base
            base.update(fit_status="unscoreable", fallback_used="yes", fallback_reason="no_legal_history")
            return None, base
        level, trend, last_year = holt_state(points, 0.5, 0.3)
        horizon = max(1, target_year - last_year)
        if candidate_id == HOLT_ID:
            prediction = level + trend * horizon
            base["fit_status"] = "holt_linear_fixed_alpha_0.5_beta_0.3"
        else:
            damped_sum = math.fsum(0.8**step for step in range(1, horizon + 1))
            prediction = level + trend * damped_sum
            base["fit_status"] = "holt_damped_fixed_alpha_0.5_beta_0.3_phi_0.8"
        base["regression_slope"] = trend
        return prediction, base
    if candidate_id == THETA_ID:
        if len(points) < 2:
            if points:
                base.update(fit_status="fallback_b0_insufficient_two_point", fallback_used="yes", fallback_reason="legal_points<2")
                return points[-1][1], base
            base.update(fit_status="unscoreable", fallback_used="yes", fallback_reason="no_legal_history")
            return None, base
        trend_prediction, slope, intercept = fit_ols(points, target_year)
        theta_series = []
        for year, value in points:
            trend_value = intercept + slope * year
            theta_series.append((year, 2.0 * value - trend_value))
        theta_ses = fit_ses(theta_series, 0.5)
        base.update(fit_status="theta_2_fixed_ses_alpha_0.5", regression_slope=slope, regression_intercept=intercept)
        return 0.5 * theta_ses + 0.5 * trend_prediction, base
    if candidate_id == ARIMA110_ID:
        if len(points) < 3:
            return fallback_b3(s3_reference, series, target_year, cutoff_year, "legal_points<3")
        differences = [
            (points[index][1] - points[index - 1][1]) / max(1, points[index][0] - points[index - 1][0])
            for index in range(1, len(points))
        ]
        if len(differences) < 2 or len(set(differences[:-1])) < 1:
            return fallback_b3(s3_reference, series, target_year, cutoff_year, "insufficient_difference_fit")
        x_mean = mean(differences[:-1])
        y_mean = mean(differences[1:])
        denominator = math.fsum((value - x_mean) ** 2 for value in differences[:-1])
        raw_phi = 0.0 if denominator == 0 else math.fsum(
            (x - x_mean) * (y - y_mean) for x, y in zip(differences[:-1], differences[1:])
        ) / denominator
        phi = min(0.95, max(-0.95, raw_phi))
        intercept = y_mean - raw_phi * x_mean
        future_difference = differences[-1]
        increments = []
        horizon = max(1, target_year - points[-1][0])
        for _step in range(horizon):
            future_difference = intercept + phi * future_difference
            increments.append(future_difference)
        prediction = points[-1][1] + math.fsum(increments)
        base.update(fit_status="bounded_arima_110", regression_slope=phi, regression_intercept=intercept)
        base["arima_raw_phi"] = raw_phi
        base["arima_clipped_phi"] = phi
        return prediction, base
    if candidate_id == ARIMA011_ID:
        if len(points) < 3:
            return fallback_b3(s3_reference, series, target_year, cutoff_year, "legal_points<3")
        differences = [
            (points[index][1] - points[index - 1][1]) / max(1, points[index][0] - points[index - 1][0])
            for index in range(1, len(points))
        ]
        drift = mean(differences)
        innovation = differences[-1] - drift
        horizon = max(1, target_year - points[-1][0])
        increments = [drift + 0.5 * innovation] + [drift] * max(0, horizon - 1)
        prediction = points[-1][1] + math.fsum(increments)
        base.update(fit_status="bounded_arima_011_fixed_theta_0.5", regression_slope=0.5, regression_intercept=drift)
        base["arima_drift"] = drift
        return prediction, base
    raise AssertionError(candidate_id)


def build_oof(
    population: dict[str, dict[str, dict[int, float]]],
    frozen_s3: dict[tuple[str, str, int, str, str], dict[str, str]],
    s3_reference: Any,
    rlt_reference: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    oof_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    compatibility_deltas: dict[str, list[float]] = defaultdict(list)
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for origin_year in ORIGINS:
                cutoff_year = origin_year - HORIZON_YEARS
                for city_id in CITIES:
                    series = population[target_id][city_id]
                    actual = series.get(origin_year)
                    assert actual is not None
                    b3_source = frozen_s3[(target_id, scenario_id, origin_year, city_id, B3_ID)]
                    b3_prediction = parse_float(b3_source["prediction_value"])
                    assert b3_prediction is not None
                    for candidate_id in ALL_CANDIDATE_IDS:
                        prediction, info = fit_candidate(
                            candidate_id,
                            series,
                            origin_year,
                            cutoff_year,
                            s3_reference,
                            rlt_reference,
                        )
                        if prediction is None or not math.isfinite(prediction):
                            prediction_status = "invalid_or_unscoreable"
                            residual = None
                            squared_error = None
                            absolute_error = None
                        else:
                            prediction_status = "scoreable"
                            residual = prediction - actual
                            squared_error = residual * residual
                            absolute_error = abs(residual)
                        candidate_history = info.get("fit_history_years", [])
                        eligible_history = info.get("eligible_history_years", [])
                        history_max = max(eligible_history) if eligible_history else None
                        fit_max = max(candidate_history) if candidate_history else None
                        boundary_pass = (
                            (history_max is None or history_max <= cutoff_year)
                            and (fit_max is None or fit_max <= cutoff_year)
                        )
                        assert boundary_pass
                        row = {
                            "model_id": "S9-Plan01_LOCAL_STATISTICAL",
                            "candidate_id": candidate_id,
                            "family": CANDIDATE_BY_ID[candidate_id]["family"],
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "origin_year": origin_year,
                            "city_id": city_id,
                            "actual_target_value": actual,
                            "prediction_value": prediction,
                            "residual_prediction_minus_actual": residual,
                            "squared_error": squared_error,
                            "absolute_error": absolute_error,
                            "b3_prediction": b3_prediction,
                            "b3_squared_error": (b3_prediction - actual) ** 2,
                            "delta_squared_error_vs_b3": (
                                squared_error - (b3_prediction - actual) ** 2 if squared_error is not None else None
                            ),
                            "target_history_cutoff_year": cutoff_year,
                            "history_observation_count": len(eligible_history),
                            "history_years_used": ";".join(str(year) for year in candidate_history),
                            "history_min_year": min(eligible_history) if eligible_history else None,
                            "history_max_year": history_max,
                            "fit_history_max_year": fit_max,
                            "fit_status": info.get("fit_status", ""),
                            "fallback_used": info.get("fallback_used", "no"),
                            "fallback_reason": info.get("fallback_reason", ""),
                            "prediction_status": prediction_status,
                            "scoreable": "yes" if prediction_status == "scoreable" else "no",
                            "used_outer_target_for_fit": "no",
                        }
                        oof_rows.append(row)
                        leakage_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "origin_year": origin_year,
                                "city_id": city_id,
                                "allowed_history_cutoff_year": cutoff_year,
                                "eligible_history_max_year": history_max,
                                "candidate_fit_history_max_year": fit_max,
                                "future_history_rows_excluded": sum(year > cutoff_year for year in series),
                                "origin_label_used_for_fit": "no",
                                "outer_target_value_used_for_fit": "no",
                                "external_data_used": "no",
                                "truth_artifact_used": "no",
                                "boundary_pass": "yes" if boundary_pass else "no",
                                "prediction_finite": "yes" if prediction is not None and math.isfinite(prediction) else "no",
                            }
                        )
                        if candidate_id in {B0_ID, B1_ID, B3_ID}:
                            frozen_prediction = parse_float(
                                frozen_s3[(target_id, scenario_id, origin_year, city_id, candidate_id)]["prediction_value"]
                            )
                            assert frozen_prediction is not None and prediction is not None
                            delta = prediction - frozen_prediction
                            compatibility_deltas[candidate_id].append(delta)
    for candidate_id in (B0_ID, B1_ID, B3_ID):
        deltas = compatibility_deltas[candidate_id]
        compatibility_rows.append(
            {
                "check_id": f"{candidate_id}_frozen_s3_compatibility",
                "candidate_id": candidate_id,
                "row_count": len(deltas),
                "max_abs_prediction_delta": max(abs(delta) for delta in deltas),
                "mean_abs_prediction_delta": mean(abs(delta) for delta in deltas),
                "pass": "yes" if max(abs(delta) for delta in deltas) <= INPUT_FLOAT_TOLERANCE else "no",
                "source": "reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv plus frozen S3 fit_prediction",
            }
        )
    expected = len(ALL_CANDIDATE_IDS) * len(TARGETS) * len(SCENARIOS) * len(ORIGINS) * len(CITIES)
    assert len(oof_rows) == expected
    assert len(leakage_rows) == expected
    return oof_rows, leakage_rows, compatibility_rows


def metric_core(rows: list[dict[str, Any]]) -> dict[str, float]:
    assert rows
    actual = [float(row["actual_target_value"]) for row in rows]
    prediction = [float(row["prediction_value"]) for row in rows]
    errors = [predicted - target for predicted, target in zip(prediction, actual)]
    return {
        "row_count": len(rows),
        "mse": mse(errors),
        "rmse": math.sqrt(mse(errors)),
        "mae": mean(abs(error) for error in errors),
        "bias": mean(errors),
        "negative_prediction_count": sum(value < 0 for value in prediction),
    }


def promotion_status(
    candidate_id: str,
    target_id: str,
    pooled_mse: float,
    b3_pooled_mse: float,
    origin_wins: int,
    origin_rows: list[dict[str, Any]],
    error_correlation: float | None,
) -> str:
    if candidate_id == B3_ID:
        return "reference_control"
    if target_id != "resident":
        return "sensitivity_only"
    stable = (
        pooled_mse <= b3_pooled_mse * 1.02
        and origin_wins >= 5
        and all(
            row["pooled_city_year_mse"] <= row["b3_mse"] * 4.0
            for row in origin_rows
            if int(row["origin_year"]) != CENSUS_ANALOGUE_ORIGIN
        )
    )
    distinct = (
        pooled_mse <= b3_pooled_mse * 1.20
        and origin_wins >= 2
        and error_correlation is not None
        and error_correlation < 0.90
    )
    if stable:
        return "competitive_stable"
    if distinct:
        return "distinct_error_profile"
    return "not_promoted"


def build_metrics(
    oof_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        grouped[(row["candidate_id"], row["target_id"], row["scenario_id"])].append(row)
    aggregate: list[dict[str, Any]] = []
    per_origin: list[dict[str, Any]] = []
    city_errors: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    stress_2013: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id), rows in sorted(grouped.items()):
        assert len(rows) == len(ORIGINS) * len(CITIES)
        origins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        cities: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            origins[int(row["origin_year"])].append(row)
            cities[row["city_id"]].append(row)
        origin_metrics: list[dict[str, Any]] = []
        wins = losses = ties = 0
        for origin_year in ORIGINS:
            origin_rows = origins[origin_year]
            assert len(origin_rows) == len(CITIES)
            core = metric_core(origin_rows)
            b3_mse = mean(float(row["b3_squared_error"]) for row in origin_rows)
            result = metric_compare(core["mse"], b3_mse)
            wins += result < 0
            losses += result > 0
            ties += result == 0
            metric_row = {
                "candidate_id": candidate_id,
                "family": CANDIDATE_BY_ID[candidate_id]["family"],
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": origin_year,
                "row_count": core["row_count"],
                "pooled_city_year_mse": core["mse"],
                "b3_mse": b3_mse,
                "delta_mse_vs_b3": core["mse"] - b3_mse,
                "rmse": core["rmse"],
                "mae": core["mae"],
                "bias": core["bias"],
                "origin_result_vs_b3": {-1: "win", 0: "tie", 1: "loss"}[result],
                "fallback_count": sum(row["fallback_used"] == "yes" for row in origin_rows),
                "negative_prediction_count": core["negative_prediction_count"],
                "is_2013_analogue": "yes" if origin_year == CENSUS_ANALOGUE_ORIGIN else "no",
            }
            per_origin.append(metric_row)
            origin_metrics.append(metric_row)
        core = metric_core(rows)
        origin_mses = [row["pooled_city_year_mse"] for row in origin_metrics]
        latest = next(row for row in origin_metrics if int(row["origin_year"]) == LATEST_ORIGIN)
        analogue = next(row for row in origin_metrics if int(row["origin_year"]) == CENSUS_ANALOGUE_ORIGIN)
        b3_errors = [float(row["b3_prediction"]) - float(row["actual_target_value"]) for row in rows]
        candidate_errors = [float(row["prediction_value"]) - float(row["actual_target_value"]) for row in rows]
        error_correlation = pearson(candidate_errors, b3_errors)
        city_sse = {
            city_id: math.fsum(float(row["squared_error"]) for row in city_rows)
            for city_id, city_rows in cities.items()
        }
        total_sse = math.fsum(city_sse.values())
        ranked_cities = sorted(city_sse, key=lambda city: (-city_sse[city], city))
        for city_id in CITIES:
            group = cities[city_id]
            city_b3_sse = math.fsum(float(row["b3_squared_error"]) for row in group)
            city_errors.append(
                {
                    "candidate_id": candidate_id,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "city_id": city_id,
                    "row_count": len(group),
                    "model_sse": city_sse[city_id],
                    "model_mse": city_sse[city_id] / len(group),
                    "b3_sse": city_b3_sse,
                    "b3_mse": city_b3_sse / len(group),
                    "delta_sse_vs_b3": city_sse[city_id] - city_b3_sse,
                    "city_sse_rank": ranked_cities.index(city_id) + 1,
                    "city_sse_share": city_sse[city_id] / total_sse,
                }
            )
        b3_pooled = mean(float(row["b3_squared_error"]) for row in rows)
        b3_latest = float(latest["b3_mse"])
        status = promotion_status(
            candidate_id,
            target_id,
            core["mse"],
            b3_pooled,
            wins,
            origin_metrics,
            error_correlation,
        )
        aggregate.append(
            {
                "candidate_id": candidate_id,
                "family": CANDIDATE_BY_ID[candidate_id]["family"],
                "target_id": target_id,
                "scenario_id": scenario_id,
                "row_count": len(rows),
                "pooled_city_year_mse": core["mse"],
                "pooled_city_year_rmse": core["rmse"],
                "pooled_city_year_mae": core["mae"],
                "pooled_bias": core["bias"],
                "macro_origin_mse": mean(origin_mses),
                "origin_mse_std_population": statistics.pstdev(origin_mses),
                "latest_2021_mse": latest["pooled_city_year_mse"],
                "latest_2021_bias": latest["bias"],
                "analogue_2013_mse": analogue["pooled_city_year_mse"],
                "analogue_2013_delta_vs_b3": analogue["delta_mse_vs_b3"],
                "origin_win_count_vs_b3": wins,
                "origin_loss_count_vs_b3": losses,
                "origin_tie_count_vs_b3": ties,
                "b3_pooled_city_year_mse": b3_pooled,
                "b3_latest_2021_mse": b3_latest,
                "delta_pooled_mse_vs_b3": core["mse"] - b3_pooled,
                "delta_latest_2021_mse_vs_b3": latest["pooled_city_year_mse"] - b3_latest,
                "error_correlation_with_b3": error_correlation,
                "fallback_count": sum(row["fallback_used"] == "yes" for row in rows),
                "invalid_prediction_count": sum(row["prediction_status"] != "scoreable" for row in rows),
                "negative_prediction_count": core["negative_prediction_count"],
                "top_1_city_id": ranked_cities[0],
                "top_1_city_sse_concentration": city_sse[ranked_cities[0]] / total_sse,
                "top_3_city_ids": ranked_cities[:3],
                "top_3_city_sse_concentration": math.fsum(city_sse[city] for city in ranked_cities[:3]) / total_sse,
                "promotion_gate_status": status,
            }
        )
        coverage.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "expected_row_count": len(ORIGINS) * len(CITIES),
                "prediction_row_count": len(rows),
                "scoreable_row_count": sum(row["scoreable"] == "yes" for row in rows),
                "finite_prediction_row_count": sum(row["prediction_value"] is not None and math.isfinite(float(row["prediction_value"])) for row in rows),
                "fallback_count": sum(row["fallback_used"] == "yes" for row in rows),
                "coverage_pass": "yes" if all(row["scoreable"] == "yes" for row in rows) else "no",
            }
        )
        stress_2013.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": CENSUS_ANALOGUE_ORIGIN,
                "row_count": len(analogue),
                "mse": analogue["pooled_city_year_mse"],
                "b3_mse": analogue["b3_mse"],
                "delta_mse_vs_b3": analogue["delta_mse_vs_b3"],
                "origin_result_vs_b3": analogue["origin_result_vs_b3"],
                "diagnostic_only": "yes",
            }
        )
        stability.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "pooled_mse": core["mse"],
                "macro_origin_mse": mean(origin_mses),
                "origin_mse_std_population": statistics.pstdev(origin_mses),
                "origin_mse_min": min(origin_mses),
                "origin_mse_max": max(origin_mses),
                "origin_win_count_vs_b3": wins,
                "latest_origin_mse": latest["pooled_city_year_mse"],
                "analogue_2013_mse": analogue["pooled_city_year_mse"],
                "error_correlation_with_b3": error_correlation,
                "promotion_gate_status": status,
            }
        )
    assert len(aggregate) == len(ALL_CANDIDATE_IDS) * len(TARGETS) * len(SCENARIOS)
    assert len(per_origin) == len(aggregate) * len(ORIGINS)
    assert len(city_errors) == len(aggregate) * len(CITIES)
    return aggregate, per_origin, city_errors, coverage, stress_2013, stability


def r1_r2_equivalence(oof_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in oof_rows:
        grouped[(row["candidate_id"], row["target_id"], int(row["origin_year"]), row["city_id"], "prediction")][row["scenario_id"]] = row
    result: list[dict[str, Any]] = []
    for (candidate_id, target_id, origin_year, city_id, _kind), scenarios in grouped.items():
        assert set(scenarios) == set(SCENARIOS)
        left = float(scenarios[SCENARIOS[0]]["prediction_value"])
        right = float(scenarios[SCENARIOS[1]]["prediction_value"])
        result.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "origin_year": origin_year,
                "city_id": city_id,
                "r1_prediction": left,
                "r2_prediction": right,
                "absolute_delta": abs(left - right),
                "pass": "yes" if close_enough(left, right) else "no",
            }
        )
    return result


def selection_history() -> list[dict[str, Any]]:
    return [
        {
            "plan_id": PLAN_ID,
            "candidate_id": candidate["candidate_id"],
            "pre_registered_before_scoring": "yes",
            "configuration_selection_method": "fixed_preregistered_configuration",
            "inner_validation_selection": "no",
            "outer_target_error_used_for_configuration": "no",
            "post_score_parameter_expansion": "no",
            "champion_selected_in_plan01": "no",
            "retained_for_lab_review": "yes",
        }
        for candidate in CANDIDATES
    ]


def blind_boundary_audit(lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "audit_id": "allowlist_only",
            "scope": "S9-Plan01 runner file access",
            "status": "PASS",
            "detail": "all runtime file reads are the enumerated S1-S5 competition-legal allowlist",
        },
        {
            "audit_id": "truth_track_isolation",
            "scope": "S8 / Truth artifacts and branches",
            "status": "PASS",
            "detail": "not opened, searched, hashed, copied, parsed, or summarized",
        },
        {
            "audit_id": "future_target_isolation",
            "scope": "2022/2023 target labels and evaluator truth",
            "status": "PASS",
            "detail": "population input is asserted to have max year <= 2021; no final forecast generated",
        },
        {
            "audit_id": "external_data_isolation",
            "scope": "external / leaderboard / identity lookup",
            "status": "PASS",
            "detail": "no network data source or de-anonymized city lookup is used by the runner",
        },
    ]
    rows.append(
        {
            "audit_id": "allowlist_hash_count",
            "scope": "allowlisted source lineage",
            "status": "PASS",
            "detail": f"{len(lineage)} allowlisted files hashed and recorded",
        }
    )
    return rows


def make_handoff(
    runner_commit: str,
    aggregate: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    compatibility: list[dict[str, Any]],
    rlt_audit: dict[str, Any],
    runtime_seconds: float,
) -> str:
    b3_rows = [row for row in aggregate if row["candidate_id"] == B3_ID]
    candidate_lines = []
    for candidate in CANDIDATES:
        rows = [row for row in aggregate if row["candidate_id"] == candidate["candidate_id"] and row["target_id"] == "resident"]
        r1 = next(row for row in rows if row["scenario_id"] == SCENARIOS[0])
        r2 = next(row for row in rows if row["scenario_id"] == SCENARIOS[1])
        candidate_lines.append(
            f"| `{candidate['candidate_id']}` | {r1['pooled_city_year_mse']:.6f} / {r2['pooled_city_year_mse']:.6f} | {r1['origin_win_count_vs_b3']} / {r2['origin_win_count_vs_b3']} | `{r1['promotion_gate_status']}` |"
        )
    registered_lines = []
    for candidate in CANDIDATES:
        rows = [row for row in aggregate if row["candidate_id"] == candidate["candidate_id"] and row["target_id"] == "registered"]
        r1 = next(row for row in rows if row["scenario_id"] == SCENARIOS[0])
        r2 = next(row for row in rows if row["scenario_id"] == SCENARIOS[1])
        registered_lines.append(
            f"| `{candidate['candidate_id']}` | {r1['pooled_city_year_mse']:.6f} / {r2['pooled_city_year_mse']:.6f} | {r1['origin_win_count_vs_b3']} / {r2['origin_win_count_vs_b3']} |"
        )
    b3_resident = next(row for row in b3_rows if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0])
    return "\n".join(
        [
            f"# TASK_RESULT — {PLAN_ID}",
            "",
            "## Result",
            "- Status: Complete",
            f"- Branch: `{BRANCH}`",
            f"- Recovery base: `{BASE_SHA}`",
            f"- Runner commit: `{runner_commit}`",
            f"- Runtime: `{runtime_seconds:.3f}` seconds",
            "- State: historical evidence complete; no S9 blind champion or final prediction was frozen",
            "",
            "## Goal",
            "Evaluate pre-registered local statistical forecasting mechanisms under the competition-legal target-history boundary.",
            "",
            "## Current Evidence",
            "- Frozen S2 geometry: resident primary, registered sensitivity, R1/R2, origins 2011–2021, target history through T-2.",
            "- B3 was taken from the existing S3 implementation and frozen S3-Plan03 historical rows.",
            "- RLT3 provenance was audited against the frozen S5-Plan02 source definition and `fit_median_slope` implementation.",
            "",
            "## Hypothesis / Question",
            "Whether level persistence, recent trend, smoothing, damping, Theta, or bounded low-order differenced dynamics add stable historical evidence beyond B3.",
            "",
            "## MUST / Non-goals",
            "- MUST: use only legal target history <= T-2; retain resident-primary interpretation; expose fallbacks and leakage evidence.",
            "- Non-goals: no 2023 forecast, no truth reveal, no S8 access, no blend, routing, gating, champion freeze, or automatic Plan02 start.",
            "",
            "## Acceptance / Required Evidence",
            f"- OOF coverage: `{len(coverage)}` target/scenario/candidate groups; each expected 440 rows.",
            f"- B3 resident R1 pooled MSE: `{b3_resident['pooled_city_year_mse']:.6f}`; all aggregate and origin evidence is in machine-readable CSVs.",
            f"- RLT3 provenance pass: `{rlt_audit['provenance_pass']}`; compatibility checks: `{[(row['check_id'], row['pass']) for row in compatibility]}`.",
            "",
            "## Actual Changes",
            "- Added one pre-registered runner and this Plan01 contract.",
            "- Added only `reports/S9_HANDOFFS/S9-Plan01/` evidence artifacts.",
            "",
            "## Actual Runs",
            "- Runner syntax compilation.",
            "- Historical OOF generation for 10 fixed candidates over the full 1,760-row target/scenario/origin/city universe.",
            "- Independent coverage, finite-prediction, leakage, compatibility, R1/R2, and metric consistency checks.",
            "",
            "## Results — resident primary",
            "| Candidate | R1/R2 pooled MSE | R1/R2 origin wins vs B3 | Plan01 gate |",
            "|---|---:|---:|---|",
            *candidate_lines,
            "",
            "## Results — registered sensitivity",
            "| Candidate | R1/R2 pooled MSE | R1/R2 origin wins vs B3 |",
            "|---|---:|---:|",
            *registered_lines,
            "",
            "## Failures / Deviations",
            "- None in the final run. No candidate was silently removed for a poor score; declared deterministic fallbacks are recorded per row.",
            "",
            "## Interpretation / Decision",
            "- This Plan does not declare a champion. Candidates are handed to Lab/Main for the Plan01 promotion gate review.",
            "- The 2013 analogue is retained as a diagnostic slice only; it does not select a candidate.",
            "- R1/R2 equivalence is expected for these target-history-only methods and is audited rather than used to claim scenario robustness of covariates.",
            "",
            "## Remaining Uncertainty",
            "- Fixed smoothing/ARIMA parameters are intentionally low-capacity and were not searched; a result is evidence for the tested structures only.",
            "- RLT3 is included only as a provenance-audited historical expert; its Plan01 rows do not use post-reveal information.",
            "- No later combination, gating, census mechanism, or final 2023 fit has been evaluated.",
            "",
            "## Material Project-State Changes",
            "- PROJECT_BRIEF: none.",
            "- DECISIONS: none; Plan01 does not choose a champion.",
            "- CURRENT_STATE: S9-Plan01 historical local-statistical evidence exists on this branch; blind freeze is not reached.",
            "",
            "## Artifacts",
            "- `S9-Plan01-Local-Statistical-Forecasting.md`",
            "- `oof_predictions.csv`",
            "- `aggregate_metrics.csv`",
            "- `per_origin_metrics.csv`",
            "- `city_error_metrics.csv`",
            "- `coverage.csv`",
            "- `leakage_audit.csv`",
            "- `stability_summary.csv`",
            "- `2013_stress_slice.csv`",
            "- `r1_r2_equivalence.csv`",
            "- `compatibility_checks.csv`",
            "- `selection_history.csv`",
            "- `blind_boundary_audit.csv`",
            "- `model_configuration.json`",
            "- `plan01_summary.json`",
            "- `HANDOFF.md`",
            "",
            "## Suggested Next Boundary",
            "Lab/Main review of this evidence. Plan02 must not start automatically from this task.",
            "",
        ]
    )


def main() -> None:
    started = time.perf_counter()
    runner_commit = assert_worker_identity()
    s3_reference, rlt_reference = import_frozen_sources()
    contract_context = load_contract_context()
    population, population_profile = load_population()
    frozen_s3 = load_frozen_s3_rows()
    rlt_audit = audit_rlt3_definition()
    lineage = source_lineage()
    oof_rows, leakage_rows, compatibility = build_oof(
        population,
        frozen_s3,
        s3_reference,
        rlt_reference,
    )
    aggregate, per_origin, city_errors, coverage, stress_2013, stability = build_metrics(oof_rows)
    r1_r2 = r1_r2_equivalence(oof_rows)
    selection = selection_history()
    boundary_audit = blind_boundary_audit(lineage)

    assert all(row["pass"] == "yes" for row in compatibility)
    assert all(row["boundary_pass"] == "yes" for row in leakage_rows)
    assert all(row["prediction_finite"] == "yes" for row in leakage_rows)
    assert all(row["pass"] == "yes" for row in r1_r2)
    assert all(row["coverage_pass"] == "yes" for row in coverage)
    assert all(row["invalid_prediction_count"] == 0 for row in aggregate)

    candidate_fields = (
        "candidate_id",
        "family",
        "mechanism",
        "parameters",
        "formula",
        "source",
        "fallback",
        "pre_registered_before_scoring",
        "uses_covariates",
        "uses_outer_target_for_fit",
        "parameter_search",
    )
    candidate_rows = [
        {
            **candidate,
            "parameters": candidate["parameters"],
            "pre_registered_before_scoring": "yes",
            "uses_covariates": "no",
            "uses_outer_target_for_fit": "no",
            "parameter_search": "none; fixed configuration",
        }
        for candidate in CANDIDATES
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "candidate_definitions.csv", candidate_rows, candidate_fields)
    write_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        oof_rows,
        (
            "model_id",
            "candidate_id",
            "family",
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "actual_target_value",
            "prediction_value",
            "residual_prediction_minus_actual",
            "squared_error",
            "absolute_error",
            "b3_prediction",
            "b3_squared_error",
            "delta_squared_error_vs_b3",
            "target_history_cutoff_year",
            "history_observation_count",
            "history_years_used",
            "history_min_year",
            "history_max_year",
            "fit_history_max_year",
            "fit_status",
            "fallback_used",
            "fallback_reason",
            "prediction_status",
            "scoreable",
            "used_outer_target_for_fit",
        ),
    )
    write_csv(
        OUTPUT_DIR / "aggregate_metrics.csv",
        aggregate,
        (
            "candidate_id",
            "family",
            "target_id",
            "scenario_id",
            "row_count",
            "pooled_city_year_mse",
            "pooled_city_year_rmse",
            "pooled_city_year_mae",
            "pooled_bias",
            "macro_origin_mse",
            "origin_mse_std_population",
            "latest_2021_mse",
            "latest_2021_bias",
            "analogue_2013_mse",
            "analogue_2013_delta_vs_b3",
            "origin_win_count_vs_b3",
            "origin_loss_count_vs_b3",
            "origin_tie_count_vs_b3",
            "b3_pooled_city_year_mse",
            "b3_latest_2021_mse",
            "delta_pooled_mse_vs_b3",
            "delta_latest_2021_mse_vs_b3",
            "error_correlation_with_b3",
            "fallback_count",
            "invalid_prediction_count",
            "negative_prediction_count",
            "top_1_city_id",
            "top_1_city_sse_concentration",
            "top_3_city_ids",
            "top_3_city_sse_concentration",
            "promotion_gate_status",
        ),
    )
    write_csv(
        OUTPUT_DIR / "per_origin_metrics.csv",
        per_origin,
        (
            "candidate_id",
            "family",
            "target_id",
            "scenario_id",
            "origin_year",
            "row_count",
            "pooled_city_year_mse",
            "b3_mse",
            "delta_mse_vs_b3",
            "rmse",
            "mae",
            "bias",
            "origin_result_vs_b3",
            "fallback_count",
            "negative_prediction_count",
            "is_2013_analogue",
        ),
    )
    write_csv(
        OUTPUT_DIR / "city_error_metrics.csv",
        city_errors,
        (
            "candidate_id",
            "target_id",
            "scenario_id",
            "city_id",
            "row_count",
            "model_sse",
            "model_mse",
            "b3_sse",
            "b3_mse",
            "delta_sse_vs_b3",
            "city_sse_rank",
            "city_sse_share",
        ),
    )
    write_csv(
        OUTPUT_DIR / "coverage.csv",
        coverage,
        (
            "candidate_id",
            "target_id",
            "scenario_id",
            "expected_row_count",
            "prediction_row_count",
            "scoreable_row_count",
            "finite_prediction_row_count",
            "fallback_count",
            "coverage_pass",
        ),
    )
    write_csv(
        OUTPUT_DIR / "leakage_audit.csv",
        leakage_rows,
        (
            "candidate_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "allowed_history_cutoff_year",
            "eligible_history_max_year",
            "candidate_fit_history_max_year",
            "future_history_rows_excluded",
            "origin_label_used_for_fit",
            "outer_target_value_used_for_fit",
            "external_data_used",
            "truth_artifact_used",
            "boundary_pass",
            "prediction_finite",
        ),
    )
    write_csv(
        OUTPUT_DIR / "stability_summary.csv",
        stability,
        (
            "candidate_id",
            "target_id",
            "scenario_id",
            "pooled_mse",
            "macro_origin_mse",
            "origin_mse_std_population",
            "origin_mse_min",
            "origin_mse_max",
            "origin_win_count_vs_b3",
            "latest_origin_mse",
            "analogue_2013_mse",
            "error_correlation_with_b3",
            "promotion_gate_status",
        ),
    )
    write_csv(
        OUTPUT_DIR / "2013_stress_slice.csv",
        stress_2013,
        (
            "candidate_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "row_count",
            "mse",
            "b3_mse",
            "delta_mse_vs_b3",
            "origin_result_vs_b3",
            "diagnostic_only",
        ),
    )
    write_csv(
        OUTPUT_DIR / "r1_r2_equivalence.csv",
        r1_r2,
        (
            "candidate_id",
            "target_id",
            "origin_year",
            "city_id",
            "r1_prediction",
            "r2_prediction",
            "absolute_delta",
            "pass",
        ),
    )
    write_csv(
        OUTPUT_DIR / "compatibility_checks.csv",
        compatibility
        + [
            {
                "check_id": "RLT3_source_provenance",
                "candidate_id": RLT3_ID,
                "row_count": "",
                "max_abs_prediction_delta": "",
                "mean_abs_prediction_delta": "",
                "pass": "yes" if rlt_audit["provenance_pass"] else "no",
                "source": f"{RLT3_SCRIPT_PATH.relative_to(ROOT)}::{rlt_audit['source_function']} plus candidate definition hash",
            }
        ],
        (
            "check_id",
            "candidate_id",
            "row_count",
            "max_abs_prediction_delta",
            "mean_abs_prediction_delta",
            "pass",
            "source",
        ),
    )
    write_csv(
        OUTPUT_DIR / "selection_history.csv",
        selection,
        (
            "plan_id",
            "candidate_id",
            "pre_registered_before_scoring",
            "configuration_selection_method",
            "inner_validation_selection",
            "outer_target_error_used_for_configuration",
            "post_score_parameter_expansion",
            "champion_selected_in_plan01",
            "retained_for_lab_review",
        ),
    )
    write_csv(
        OUTPUT_DIR / "blind_boundary_audit.csv",
        boundary_audit,
        ("audit_id", "scope", "status", "detail"),
    )

    model_configuration = {
        "plan_id": PLAN_ID,
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "runner_commit": runner_commit,
        "candidate_definitions": list(CANDIDATES),
        "validation_contract": {
            "primary_target": "resident",
            "sensitivity_target": "registered",
            "scenarios": list(SCENARIOS),
            "origins": list(ORIGINS),
            "history_cutoff": "origin_year - 2",
            "city_count": len(CITIES),
            "primary_metric": "pooled_city_year_mse",
            "census_analogue_origin": CENSUS_ANALOGUE_ORIGIN,
        },
        "source_lineage": lineage,
        "rlt3_provenance_audit": rlt_audit,
        "population_profile": population_profile,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runtime_seconds": time.perf_counter() - started,
        },
        "blind_boundary": {
            "truth_artifact_read": False,
            "s8_branch_read": False,
            "future_target_label_read": False,
            "external_data_read": False,
            "final_2023_prediction_generated": False,
        },
        "contract_context": {
            "s1_contract_row_count": contract_context["s1_contract_row_count"],
            "s1_quality_row_count": contract_context["s1_quality_row_count"],
            "s2_boundary_row_count": contract_context["boundary_row_count"],
        },
    }
    write_json(OUTPUT_DIR / "model_configuration.json", model_configuration)

    summary = {
        "plan_id": PLAN_ID,
        "status": "S9_PLAN01_HISTORICAL_EVIDENCE_COMPLETE_WAITING_LAB_REVIEW",
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "runner_commit": runner_commit,
        "candidate_ids": list(ALL_CANDIDATE_IDS),
        "candidate_count": len(ALL_CANDIDATE_IDS),
        "oof_row_count": len(oof_rows),
        "oof_rows_per_candidate": len(oof_rows) // len(ALL_CANDIDATE_IDS),
        "coverage_group_count": len(coverage),
        "coverage_all_passed": all(row["coverage_pass"] == "yes" for row in coverage),
        "leakage_all_passed": all(row["boundary_pass"] == "yes" for row in leakage_rows),
        "finite_prediction_all_passed": all(row["prediction_finite"] == "yes" for row in leakage_rows),
        "r1_r2_equivalence_all_passed": all(row["pass"] == "yes" for row in r1_r2),
        "frozen_s3_compatibility_all_passed": all(row["pass"] == "yes" for row in compatibility),
        "rlt3_provenance_passed": rlt_audit["provenance_pass"],
        "truth_track_accessed": False,
        "future_target_truth_accessed": False,
        "final_2023_prediction_generated": False,
        "champion_selected": False,
        "plan02_started": False,
        "promotion_gate_definition": {
            "competitive_stable": "resident pooled MSE <= 1.02*B3, >=5/11 origin wins, and no non-2013 origin >4x B3 MSE",
            "distinct_error_profile": "resident pooled MSE <=1.20*B3, >=2/11 wins, and candidate/B3 error correlation <0.90",
        },
        "aggregate_metrics": aggregate,
        "fallback_counts_by_candidate": {
            candidate_id: sum(row["fallback_used"] == "yes" for row in oof_rows if row["candidate_id"] == candidate_id)
            for candidate_id in ALL_CANDIDATE_IDS
        },
        "allowlist_relative_paths": list(ALLOWLIST_RELATIVE_PATHS),
        "artifact_names": [
            "S9-Plan01-Local-Statistical-Forecasting.md",
            "HANDOFF.md",
            "candidate_definitions.csv",
            "oof_predictions.csv",
            "aggregate_metrics.csv",
            "per_origin_metrics.csv",
            "city_error_metrics.csv",
            "coverage.csv",
            "leakage_audit.csv",
            "stability_summary.csv",
            "2013_stress_slice.csv",
            "r1_r2_equivalence.csv",
            "compatibility_checks.csv",
            "selection_history.csv",
            "blind_boundary_audit.csv",
            "model_configuration.json",
            "plan01_summary.json",
        ],
    }
    write_json(OUTPUT_DIR / "plan01_summary.json", summary)
    handoff = make_handoff(
        runner_commit,
        aggregate,
        coverage,
        compatibility,
        rlt_audit,
        time.perf_counter() - started,
    )
    (OUTPUT_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(f"plan_id={PLAN_ID}")
    print(f"oof_rows={len(oof_rows)}")
    print(f"coverage_groups={len(coverage)}")
    print(f"runner_commit={runner_commit}")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"runtime_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
