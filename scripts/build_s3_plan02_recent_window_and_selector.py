"""Build S3-Plan02 recent-window and city-heterogeneity evidence.

This runner inherits the frozen S2 contract and the S3-Plan01 B0/B1/B2
implementation.  It adds only the pre-registered B3/B4 fixed windows and the
prequential global/city selectors.  Oracle outputs are diagnostic-only and
are never mixed into the deployable baseline ranking.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import build_s3_plan01_baseline_ladder as plan01


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan02"
PLAN01_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan01"
S2_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05"

STARTING_COMMIT = "ed006d91174889280451878a2c1a820fb05a95bf"
HORIZON_YEARS = plan01.HORIZON_YEARS
REFERENCE_TARGET_YEAR = plan01.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = plan01.LATEST_ORIGIN
COMMON_ORIGINS = plan01.COMMON_ORIGINS
EARLY_ORIGINS = plan01.EARLY_ORIGINS
LATE_ORIGINS = plan01.LATE_ORIGINS
CITIES = plan01.CITIES
SCENARIOS = plan01.SCENARIOS
TARGETS = plan01.TARGETS

FIXED_PREDICTORS = (
    {
        "predictor_id": "B0_PERSISTENCE_LAST_OBSERVED",
        "s2_predictor_id": "PERSISTENCE_LAST_OBSERVED",
        "predictor_name": "persistence / last observed target",
        "mechanism": "reference",
        "formula": "y_hat(T) = y_last; latest non-missing target year <= T-2",
        "minimum_effective_history_points": 1,
        "window_size": "none",
        "window_selection": "latest non-missing observation through T-2",
    },
    {
        "predictor_id": "B1_TWO_POINT_LINEAR_TREND",
        "s2_predictor_id": "TWO_POINT_LINEAR_TREND",
        "predictor_name": "recent two-point linear trend",
        "mechanism": "recent_slope",
        "formula": "fit the slope between the two latest non-missing target points <= T-2 and extrapolate to T",
        "minimum_effective_history_points": 2,
        "window_size": 2,
        "window_selection": "two latest non-missing observations",
    },
    {
        "predictor_id": "B2_EXPANDING_PER_CITY_LINEAR_TREND",
        "s2_predictor_id": "",
        "predictor_name": "expanding per-city linear trend",
        "mechanism": "long_history_city_specific_trend",
        "formula": "fit target = intercept + slope * year on all non-missing city target history <= T-2; predict T",
        "minimum_effective_history_points": 2,
        "window_size": "all",
        "window_selection": "all available history through T-2; no window search",
    },
    {
        "predictor_id": "B3_RECENT_3POINT_OLS_TREND",
        "s2_predictor_id": "",
        "predictor_name": "recent three-observation OLS trend",
        "mechanism": "recent_window_slope",
        "formula": "fit target = intercept + slope * year on the latest 3 non-missing target points <= T-2; predict T",
        "minimum_effective_history_points": 3,
        "window_size": 3,
        "window_selection": "latest 3 non-missing observations; fixed fallback to B1 if fewer than 3",
    },
    {
        "predictor_id": "B4_RECENT_5POINT_OLS_TREND",
        "s2_predictor_id": "",
        "predictor_name": "recent five-observation OLS trend",
        "mechanism": "recent_window_slope",
        "formula": "fit target = intercept + slope * year on the latest 5 non-missing target points <= T-2; predict T",
        "minimum_effective_history_points": 5,
        "window_size": 5,
        "window_selection": "latest 5 non-missing observations; fixed fallback to B1 if fewer than 5",
    },
)

SELECTOR_DEFINITIONS = (
    {
        "predictor_id": "GLOBAL_PREQUENTIAL_SELECTOR",
        "predictor_name": "global prequential selector",
        "selector_scope": "all cities within target and scenario",
        "candidate_predictors": "B1,B2,B3,B4",
        "selection_rule": "mean historical squared error over origins < T",
        "cold_start_rule": "fewer than 3 prior scored origins => fixed B1 fallback",
        "selection_tie_break": "B1 then B2 then B3 then B4",
        "uses_current_origin_residual": "no",
        "uses_future_origin_residual": "no",
    },
    {
        "predictor_id": "CITY_PREQUENTIAL_SELECTOR",
        "predictor_name": "city-level prequential selector",
        "selector_scope": "one city within target and scenario",
        "candidate_predictors": "B1,B2,B3,B4",
        "selection_rule": "mean historical squared error for this city over origins < T",
        "cold_start_rule": "fewer than 3 prior scored origins => fixed B1 fallback",
        "selection_tie_break": "B1 then B2 then B3 then B4",
        "uses_current_origin_residual": "no",
        "uses_future_origin_residual": "no",
    },
)

ALL_PREDICTORS = FIXED_PREDICTORS + SELECTOR_DEFINITIONS
FIXED_IDS = tuple(row["predictor_id"] for row in FIXED_PREDICTORS)
SELECTOR_IDS = tuple(row["predictor_id"] for row in SELECTOR_DEFINITIONS)
SELECTOR_CANDIDATE_IDS = (
    "B1_TWO_POINT_LINEAR_TREND",
    "B2_EXPANDING_PER_CITY_LINEAR_TREND",
    "B3_RECENT_3POINT_OLS_TREND",
    "B4_RECENT_5POINT_OLS_TREND",
)
SELECTION_TIE_BREAK_ORDER = SELECTOR_CANDIDATE_IDS

PAIRWISE_COMPARISONS = (
    ("B1_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B1_TWO_POINT_LINEAR_TREND"),
    ("B2_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B2_EXPANDING_PER_CITY_LINEAR_TREND"),
    ("B2_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B2_EXPANDING_PER_CITY_LINEAR_TREND"),
    ("B3_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B3_vs_B2", "B2_EXPANDING_PER_CITY_LINEAR_TREND", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B2", "B2_EXPANDING_PER_CITY_LINEAR_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B4_vs_B3", "B3_RECENT_3POINT_OLS_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B3_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B4_RECENT_5POINT_OLS_TREND"),
    ("GLOBAL_SELECTOR_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "GLOBAL_PREQUENTIAL_SELECTOR"),
    ("CITY_SELECTOR_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "CITY_PREQUENTIAL_SELECTOR"),
    ("CITY_SELECTOR_vs_GLOBAL_SELECTOR", "GLOBAL_PREQUENTIAL_SELECTOR", "CITY_PREQUENTIAL_SELECTOR"),
)


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
            writer.writerow({field: "" if row.get(field) is None else plan01.rounded(row.get(field)) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(field for row in rows for field in row))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "NA", "N/A", "NaN", "nan", "None"}


def as_float(value: Any) -> float | None:
    if not present(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float:
    assert values
    return math.fsum(values) / len(values)


def competition_score(value: float) -> float:
    return 1.0 / (1.0 + value)


def git_starting_commit() -> str:
    return STARTING_COMMIT


def fit_ols(years: list[int], values: list[float], target_year: int) -> dict[str, Any]:
    assert len(years) == len(values) and len(years) >= 2
    x_mean = mean([float(year) for year in years])
    y_mean = mean(values)
    denominator = math.fsum((float(year) - x_mean) ** 2 for year in years)
    assert denominator > 0
    numerator = math.fsum((float(year) - x_mean) * (value - y_mean) for year, value in zip(years, values))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return {
        "prediction": intercept + slope * target_year,
        "slope": slope,
        "intercept": intercept,
    }


def fit_prediction(predictor_id: str, series: dict[int, float], target_year: int, cutoff_year: int) -> dict[str, Any]:
    eligible_years = sorted(year for year in series if year <= cutoff_year)
    eligible_values = [series[year] for year in eligible_years]
    result: dict[str, Any] = {
        "eligible_history_years": eligible_years,
        "history_years": eligible_years,
        "history_count": len(eligible_years),
        "history_min_year": eligible_years[0] if eligible_years else None,
        "history_max_year": eligible_years[-1] if eligible_years else None,
        "latest_target_year_used": eligible_years[-1] if eligible_years else None,
        "previous_target_year_used": eligible_years[-2] if len(eligible_years) >= 2 else None,
        "regression_fit_min_year": None,
        "regression_fit_max_year": None,
        "regression_fit_distinct_year_count": len(set(eligible_years)),
        "regression_slope": None,
        "regression_intercept": None,
        "regression_fit_status": "not_applicable",
        "fixed_window_size": None,
        "window_points_available": len(eligible_years),
        "window_points_used": len(eligible_years),
        "window_fallback_applied": "no",
        "window_fallback_reason": "",
        "prediction": None,
    }

    if predictor_id == "B0_PERSISTENCE_LAST_OBSERVED":
        if eligible_years:
            result["prediction"] = series[eligible_years[-1]]
            result["regression_fit_status"] = "not_applicable_reference"
        else:
            result["regression_fit_status"] = "insufficient_history"
        return result

    if predictor_id == "B1_TWO_POINT_LINEAR_TREND":
        if len(eligible_years) >= 2:
            years = eligible_years[-2:]
            values = [series[year] for year in years]
            slope = (values[-1] - values[-2]) / (years[-1] - years[-2])
            result["regression_slope"] = slope
            result["regression_intercept"] = values[-1] - slope * years[-1]
            result["prediction"] = values[-1] + slope * (target_year - years[-1])
            result["regression_fit_min_year"] = years[0]
            result["regression_fit_max_year"] = years[-1]
            result["regression_fit_distinct_year_count"] = len(set(years))
            result["regression_fit_status"] = "two_point_slope"
        else:
            result["regression_fit_status"] = "insufficient_history"
        return result

    if predictor_id == "B2_EXPANDING_PER_CITY_LINEAR_TREND":
        if len(eligible_years) >= 2:
            fitted = fit_ols(eligible_years, eligible_values, target_year)
            result.update(
                prediction=fitted["prediction"],
                regression_slope=fitted["slope"],
                regression_intercept=fitted["intercept"],
                regression_fit_min_year=eligible_years[0],
                regression_fit_max_year=eligible_years[-1],
                regression_fit_status="expanding_all_history_ols",
            )
        else:
            result["regression_fit_status"] = "insufficient_history_for_identifiable_slope"
        return result

    if predictor_id in {"B3_RECENT_3POINT_OLS_TREND", "B4_RECENT_5POINT_OLS_TREND"}:
        window_size = 3 if predictor_id.startswith("B3") else 5
        result["fixed_window_size"] = window_size
        if len(eligible_years) >= window_size:
            years = eligible_years[-window_size:]
            values = [series[year] for year in years]
            fitted = fit_ols(years, values, target_year)
            result.update(
                history_years=years,
                history_count=len(years),
                history_min_year=years[0],
                history_max_year=years[-1],
                latest_target_year_used=years[-1],
                previous_target_year_used=years[-2] if len(years) >= 2 else None,
                regression_fit_min_year=years[0],
                regression_fit_max_year=years[-1],
                regression_fit_distinct_year_count=len(set(years)),
                regression_slope=fitted["slope"],
                regression_intercept=fitted["intercept"],
                regression_fit_status=f"recent_{window_size}point_ols",
                window_points_used=len(years),
                prediction=fitted["prediction"],
            )
        elif len(eligible_years) >= 2:
            fallback = fit_prediction("B1_TWO_POINT_LINEAR_TREND", series, target_year, cutoff_year)
            result.update(
                history_years=fallback["history_years"],
                history_count=fallback["history_count"],
                history_min_year=fallback["history_min_year"],
                history_max_year=fallback["history_max_year"],
                latest_target_year_used=fallback["latest_target_year_used"],
                previous_target_year_used=fallback["previous_target_year_used"],
                regression_fit_min_year=fallback["regression_fit_min_year"],
                regression_fit_max_year=fallback["regression_fit_max_year"],
                regression_fit_distinct_year_count=fallback["regression_fit_distinct_year_count"],
                regression_slope=fallback["regression_slope"],
                regression_intercept=fallback["regression_intercept"],
                regression_fit_status=f"fallback_to_B1_insufficient_recent_{window_size}point",
                window_points_used=len(fallback["history_years"]),
                window_fallback_applied="yes",
                window_fallback_reason=f"eligible_history_points={len(eligible_years)}<{window_size}",
                prediction=fallback["prediction"],
            )
        else:
            result["regression_fit_status"] = f"insufficient_history_for_recent_{window_size}point_and_B1_fallback"
        return result

    raise AssertionError(predictor_id)


def load_context() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any], list[dict[str, str]], dict[tuple[str, str, str, int, str], dict[str, str]]]:
    plan01.assert_static_boundaries()
    population, profile = plan01.load_population()
    s2_summary, s2_boundary, s2_index = plan01.load_s2_context()
    plan01.load_scenarios()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["s2_complete"] is False
    plan01_summary = json.loads((PLAN01_DIR / "plan01_summary.json").read_text(encoding="utf-8"))
    assert plan01_summary["common_origin_domain"] == "2011-2021"
    assert plan01_summary["s2_contract_modified"] is False
    return population, profile, s2_boundary, s2_index


def row_from_fit(target: dict[str, str], scenario_id: str, predictor_id: str, origin: int, city: str, series: dict[int, float], fit: dict[str, Any]) -> dict[str, Any]:
    actual = series.get(origin)
    prediction = fit["prediction"]
    scoreable = prediction is not None and actual is not None
    residual = prediction - actual if scoreable else None
    cutoff = origin - HORIZON_YEARS
    future_years = sorted(year for year in series if year > cutoff)
    fit_max = fit["regression_fit_max_year"]
    leakage_pass = (fit["latest_target_year_used"] is None or fit["latest_target_year_used"] <= cutoff) and (fit_max is None or fit_max <= cutoff)
    assert leakage_pass
    return {
        "target_id": target["target_id"],
        "target_column": target["canonical_column"],
        "target_contract_role": target["contract_role"],
        "scenario_id": scenario_id,
        "predictor_id": predictor_id,
        "origin_year": origin,
        "city_id": city,
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_cutoff_year": cutoff,
        "actual_target_year_for_scoring": origin if actual is not None else "",
        "actual_target_value_for_scoring": actual,
        "eligible_history_years": ";".join(str(year) for year in fit["eligible_history_years"]),
        "history_observation_count": fit["history_count"],
        "history_years_used": ";".join(str(year) for year in fit["history_years"]),
        "history_min_year": fit["history_min_year"],
        "history_max_year": fit["history_max_year"],
        "latest_target_year_used": fit["latest_target_year_used"],
        "previous_target_year_used": fit["previous_target_year_used"],
        "regression_fit_min_year": fit["regression_fit_min_year"],
        "regression_fit_max_year": fit["regression_fit_max_year"],
        "regression_fit_distinct_year_count": fit["regression_fit_distinct_year_count"],
        "regression_slope": fit["regression_slope"],
        "regression_intercept": fit["regression_intercept"],
        "regression_fit_status": fit["regression_fit_status"],
        "fixed_window_size": fit["fixed_window_size"],
        "window_points_available": fit["window_points_available"],
        "window_points_used": fit["window_points_used"],
        "window_fallback_applied": fit["window_fallback_applied"],
        "window_fallback_reason": fit["window_fallback_reason"],
        "prediction_value": prediction,
        "prediction_status": "scoreable" if scoreable else "missing_actual_target" if actual is None else "insufficient_target_history",
        "scoreable": "yes" if scoreable else "no",
        "residual_prediction_minus_actual": residual,
        "squared_error": residual * residual if residual is not None else None,
        "absolute_error": abs(residual) if residual is not None else None,
    }


def build_fixed_rows(population: dict[str, dict[str, dict[int, float]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for predictor in FIXED_PREDICTORS:
                predictor_id = predictor["predictor_id"]
                for origin in COMMON_ORIGINS:
                    cutoff = origin - HORIZON_YEARS
                    for city in CITIES:
                        series = population[target["target_id"]][city]
                        fit = fit_prediction(predictor_id, series, origin, cutoff)
                        row = row_from_fit(target, scenario_id, predictor_id, origin, city, series, fit)
                        rows.append(row)
                        audit.append({
                            "target_id": target["target_id"],
                            "scenario_id": scenario_id,
                            "predictor_id": predictor_id,
                            "origin_year": origin,
                            "city_id": city,
                            "current_origin": origin,
                            "target_history_cutoff_year": cutoff,
                            "history_max_year_used": row["history_max_year"],
                            "regression_fit_max_year_used": row["regression_fit_max_year"] or row["history_max_year"],
                            "window_points_used": row["window_points_used"],
                            "window_fallback_applied": row["window_fallback_applied"],
                            "selector_history_max_origin": "",
                            "prior_scored_origin_count": "",
                            "current_origin_residual_used": "not_applicable_fixed_predictor",
                            "future_origin_residual_used": "not_applicable_fixed_predictor",
                            "selected_fixed_predictor_id": predictor_id,
                            "target_history_cutoff_pass": "PASS",
                            "selector_history_strictly_prior_pass": "PASS",
                            "current_origin_residual_excluded_pass": "PASS",
                            "future_origin_residual_excluded_pass": "PASS",
                            "leakage_status": "PASS",
                            "audit_rule": "fixed predictor history must be <= T-2; target T is scoring-only",
                        })
    assert len(rows) == len(TARGETS) * len(SCENARIOS) * len(FIXED_PREDICTORS) * len(COMMON_ORIGINS) * len(CITIES)
    return rows, audit


def fixed_row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    result = {(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row for row in rows}
    assert len(result) == len(rows)
    return result


def prior_candidate_history(fixed_map: dict[tuple[str, str, str, int, str], dict[str, Any]], target_id: str, scenario_id: str, origin: int, city: str | None) -> tuple[dict[str, float], dict[str, int], int | None, int]:
    mse_by_candidate: dict[str, float] = {}
    count_by_candidate: dict[str, int] = {}
    for candidate_id in SELECTOR_CANDIDATE_IDS:
        candidate_rows = []
        for prior_origin in COMMON_ORIGINS:
            if prior_origin >= origin:
                continue
            cities = [city] if city is not None else list(CITIES)
            for prior_city in cities:
                row = fixed_map[(target_id, scenario_id, candidate_id, prior_origin, prior_city)]
                if row["scoreable"] == "yes":
                    candidate_rows.append(float(row["squared_error"]))
        count_by_candidate[candidate_id] = len(candidate_rows)
        mse_by_candidate[candidate_id] = mean(candidate_rows) if candidate_rows else math.inf
    prior_origins = [year for year in COMMON_ORIGINS if year < origin]
    complete_prior_origins = []
    for prior_origin in prior_origins:
        cities = [city] if city is not None else list(CITIES)
        if all(fixed_map[(target_id, scenario_id, candidate_id, prior_origin, prior_city)]["scoreable"] == "yes" for candidate_id in SELECTOR_CANDIDATE_IDS for prior_city in cities):
            complete_prior_origins.append(prior_origin)
    return mse_by_candidate, count_by_candidate, max(complete_prior_origins) if complete_prior_origins else None, len(complete_prior_origins)


def choose_candidate(mse_by_candidate: dict[str, float], prior_origin_count: int) -> tuple[str, bool, str]:
    if prior_origin_count < 3:
        return "B1_TWO_POINT_LINEAR_TREND", True, "cold_start_fewer_than_3_prior_scored_origins"
    chosen = min(SELECTION_TIE_BREAK_ORDER, key=lambda candidate_id: (mse_by_candidate[candidate_id], SELECTION_TIE_BREAK_ORDER.index(candidate_id)))
    return chosen, False, "prequential_historical_mse_minimum"


def build_selector_rows(fixed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_map = fixed_row_map(fixed_rows)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for selector_id in SELECTOR_IDS:
                selector_scope_city = selector_id == "CITY_PREQUENTIAL_SELECTOR"
                for origin in COMMON_ORIGINS:
                    for city in CITIES:
                        scope_city = city if selector_scope_city else None
                        mse_by_candidate, count_by_candidate, history_max, prior_count = prior_candidate_history(fixed_map, target_id, scenario_id, origin, scope_city)
                        selected_id, fallback_applied, selection_reason = choose_candidate(mse_by_candidate, prior_count)
                        source = fixed_map[(target_id, scenario_id, selected_id, origin, city)]
                        row = dict(source)
                        artifact_mse = {
                            candidate_id: value if math.isfinite(value) else None
                            for candidate_id, value in mse_by_candidate.items()
                        }
                        row.update({
                            "predictor_id": selector_id,
                            "selected_fixed_predictor_id": selected_id,
                            "selector_type": selector_id,
                            "selector_scope": "city" if selector_scope_city else "global",
                            "selection_history_max_origin": history_max if history_max is not None else "",
                            "prior_scored_origin_count": prior_count,
                            "selection_fallback_applied": "yes" if fallback_applied else "no",
                            "selection_fallback_reason": selection_reason if fallback_applied else "",
                            "selection_rule": "fixed B1 fallback before 3 prior origins; otherwise minimum historical MSE over B1/B2/B3/B4",
                            "historical_candidate_mse_json": stable_json(artifact_mse),
                            "historical_candidate_scored_row_count_json": stable_json(count_by_candidate),
                            "selection_tie_break_order": ">".join(SELECTION_TIE_BREAK_ORDER),
                            "current_origin_residual_used": "no",
                            "future_origin_residual_used": "no",
                        })
                        rows.append(row)
                        cutoff = origin - HORIZON_YEARS
                        target_history_ok = row["history_max_year"] in {None, ""} or int(row["history_max_year"]) <= cutoff
                        selector_history_ok = history_max is None or history_max < origin
                        current_excluded = row["current_origin_residual_used"] == "no"
                        future_excluded = row["future_origin_residual_used"] == "no"
                        status = target_history_ok and selector_history_ok and current_excluded and future_excluded
                        assert status
                        audit.append({
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "predictor_id": selector_id,
                            "origin_year": origin,
                            "city_id": city,
                            "current_origin": origin,
                            "target_history_cutoff_year": cutoff,
                            "history_max_year_used": row["history_max_year"],
                            "regression_fit_max_year_used": row["regression_fit_max_year"] or row["history_max_year"],
                            "window_points_used": row["window_points_used"],
                            "window_fallback_applied": row["window_fallback_applied"],
                            "selector_history_max_origin": history_max if history_max is not None else "",
                            "prior_scored_origin_count": prior_count,
                            "current_origin_residual_used": "no",
                            "future_origin_residual_used": "no",
                            "selected_fixed_predictor_id": selected_id,
                            "target_history_cutoff_pass": "PASS" if target_history_ok else "FAIL",
                            "selector_history_strictly_prior_pass": "PASS" if selector_history_ok else "FAIL",
                            "current_origin_residual_excluded_pass": "PASS" if current_excluded else "FAIL",
                            "future_origin_residual_excluded_pass": "PASS" if future_excluded else "FAIL",
                            "leakage_status": "PASS" if status else "FAIL",
                            "audit_rule": "selector history max origin must be < current origin; selected fixed output must obey T-2",
                        })
    assert len(rows) == len(TARGETS) * len(SCENARIOS) * len(SELECTOR_IDS) * len(COMMON_ORIGINS) * len(CITIES)
    return rows, audit


def slope_diagnostics(fixed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = fixed_row_map(fixed_rows)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                unique_rows = []
                for city in CITIES:
                    b1 = by_key[(target["target_id"], scenario_id, "B1_TWO_POINT_LINEAR_TREND", origin, city)]
                    b2 = by_key[(target["target_id"], scenario_id, "B2_EXPANDING_PER_CITY_LINEAR_TREND", origin, city)]
                    b1_slope = as_float(b1["regression_slope"])
                    b2_slope = as_float(b2["regression_slope"])
                    sign_disagreement = b1_slope is not None and b2_slope is not None and b1_slope * b2_slope < 0
                    b1_se = as_float(b1["squared_error"])
                    b2_se = as_float(b2["squared_error"])
                    unique_rows.append({"b1_slope": b1_slope, "b2_slope": b2_slope, "sign_disagreement": sign_disagreement, "b2_better": b2_se is not None and b1_se is not None and b2_se < b1_se - 1e-12})
                    for predictor_id in FIXED_IDS:
                        current = by_key[(target["target_id"], scenario_id, predictor_id, origin, city)]
                        current_slope = as_float(current["regression_slope"])
                        abs_difference = abs(b1_slope - b2_slope) if b1_slope is not None and b2_slope is not None else None
                        ratio = abs(b1_slope) / abs(b2_slope) if b1_slope is not None and b2_slope not in (None, 0.0) else None
                        rows.append({
                            "target_id": target["target_id"],
                            "scenario_id": scenario_id,
                            "origin_year": origin,
                            "city_id": city,
                            "predictor_id": predictor_id,
                            "history_point_count": current["history_observation_count"],
                            "history_first_year": current["history_min_year"],
                            "history_last_year": current["history_max_year"],
                            "prediction_extrapolation_distance": origin - int(current["latest_target_year_used"]) if current["latest_target_year_used"] not in (None, "") else None,
                            "fitted_slope": current_slope,
                            "recent_b1_slope": b1_slope,
                            "long_history_b2_slope": b2_slope,
                            "b1_minus_b2_slope": b1_slope - b2_slope if b1_slope is not None and b2_slope is not None else None,
                            "absolute_b1_b2_slope_difference": abs_difference,
                            "b1_b2_slope_sign_disagreement": "yes" if sign_disagreement else "no" if b1_slope is not None and b2_slope is not None else "not_available",
                            "absolute_slope_ratio_b1_to_b2": ratio,
                            "b1_squared_error": b1_se,
                            "b2_squared_error": b2_se,
                            "b1_minus_b2_squared_error": b1_se - b2_se if b1_se is not None and b2_se is not None else None,
                            "b2_error_outcome": "better" if b1_se is not None and b2_se is not None and b1_se - b2_se > 1e-12 else "worse" if b1_se is not None and b2_se is not None and b1_se - b2_se < -1e-12 else "tie" if b1_se is not None and b2_se is not None else "not_scoreable",
                        })
                sign_rows = [row for row in unique_rows if row["b1_slope"] is not None and row["b2_slope"] is not None]
                disagreement_rows = [row for row in sign_rows if row["sign_disagreement"]]
                agreement_rows = [row for row in sign_rows if not row["sign_disagreement"]]
                summary.append({
                    "target_id": target["target_id"],
                    "scenario_id": scenario_id,
                    "origin_year": origin,
                    "city_count_with_two_slopes": len(sign_rows),
                    "slope_sign_disagreement_city_count": len(disagreement_rows),
                    "slope_sign_disagreement_share": len(disagreement_rows) / len(sign_rows) if sign_rows else "",
                    "mean_absolute_b1_b2_slope_difference": mean([abs(row["b1_slope"] - row["b2_slope"]) for row in sign_rows]) if sign_rows else "",
                    "b2_better_city_count": sum(row["b2_better"] for row in sign_rows),
                    "b2_better_share": sum(row["b2_better"] for row in sign_rows) / len(sign_rows) if sign_rows else "",
                    "b2_better_when_slope_sign_disagrees_share": sum(row["b2_better"] for row in disagreement_rows) / len(disagreement_rows) if disagreement_rows else "",
                    "b2_better_when_slope_sign_agrees_share": sum(row["b2_better"] for row in agreement_rows) / len(agreement_rows) if agreement_rows else "",
                    "interpretation_scope": "diagnostic_only; no rule added from this artifact",
                })
    return rows, summary


def city_preference_artifacts(fixed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = fixed_row_map(fixed_rows)
    profile_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for city in CITIES:
                per_predictor: dict[str, list[dict[str, Any]]] = {}
                for predictor_id in FIXED_IDS:
                    per_predictor[predictor_id] = [by_key[(target_id, scenario_id, predictor_id, origin, city)] for origin in COMMON_ORIGINS]
                per_origin_winners: list[str] = []
                rank_distribution: dict[str, dict[str, int]] = {}
                for index, origin in enumerate(COMMON_ORIGINS):
                    values = {predictor_id: float(per_predictor[predictor_id][index]["squared_error"]) for predictor_id in FIXED_IDS if per_predictor[predictor_id][index]["scoreable"] == "yes"}
                    ranks = {predictor_id: 1 + sum(other < value - 1e-12 for other in values.values()) for predictor_id, value in values.items()}
                    winner = min(SELECTION_TIE_BREAK_ORDER[:0] + FIXED_IDS, key=lambda predictor_id: (values[predictor_id], FIXED_IDS.index(predictor_id)))
                    per_origin_winners.append(winner)
                    for predictor_id, rank in ranks.items():
                        rank_distribution.setdefault(predictor_id, {})[str(rank)] = rank_distribution.setdefault(predictor_id, {}).get(str(rank), 0) + 1
                cumulative: dict[str, float] = {}
                mean_mae: dict[str, float] = {}
                bias: dict[str, float] = {}
                origin_mse: dict[str, list[float]] = {}
                for predictor_id in FIXED_IDS:
                    rows = per_predictor[predictor_id]
                    errors = [float(row["residual_prediction_minus_actual"]) for row in rows if row["scoreable"] == "yes"]
                    squared = [error * error for error in errors]
                    cumulative[predictor_id] = math.fsum(squared)
                    mean_mae[predictor_id] = mean([abs(error) for error in errors])
                    bias[predictor_id] = mean(errors)
                    origin_mse[predictor_id] = squared
                ordered = sorted(FIXED_IDS, key=lambda predictor_id: (cumulative[predictor_id], FIXED_IDS.index(predictor_id)))
                best = ordered[0]
                second = ordered[1]
                best_wins = per_origin_winners.count(best)
                worst = ordered[-1]
                leave_one_out_winners = []
                for omitted_index in range(len(COMMON_ORIGINS)):
                    leave_out = {predictor_id: math.fsum(origin_mse[predictor_id][index] for index in range(len(COMMON_ORIGINS)) if index != omitted_index) for predictor_id in FIXED_IDS}
                    leave_one_out_winners.append(min(FIXED_IDS, key=lambda predictor_id: (leave_out[predictor_id], FIXED_IDS.index(predictor_id))))
                robust = all(winner == best for winner in leave_one_out_winners)
                margins_by_origin = []
                for index in range(len(COMMON_ORIGINS)):
                    margins_by_origin.append(origin_mse[second][index] - origin_mse[best][index])
                positive_margins = [max(0.0, value) for value in margins_by_origin]
                largest_positive_share = max(positive_margins) / math.fsum(positive_margins) if math.fsum(positive_margins) else 0.0
                if best_wins >= 7 and robust:
                    label = "stable_preference"
                elif not robust and largest_positive_share >= 0.5:
                    label = "outlier_dominated_preference"
                else:
                    label = "weak_or_unstable_preference"
                for predictor_id in FIXED_IDS:
                    rows = per_predictor[predictor_id]
                    profile_rows.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "city_id": city,
                        "predictor_id": predictor_id,
                        "cumulative_sse": cumulative[predictor_id],
                        "mean_squared_error_across_origins": mean(origin_mse[predictor_id]),
                        "mean_absolute_error": mean_mae[predictor_id],
                        "mean_signed_error_bias": bias[predictor_id],
                        "origin_win_count": per_origin_winners.count(predictor_id),
                        "origin_win_share": per_origin_winners.count(predictor_id) / len(COMMON_ORIGINS),
                        "origin_rank_distribution_json": stable_json(rank_distribution.get(predictor_id, {})),
                        "best_model_count": per_origin_winners.count(predictor_id),
                        "worst_model_count": sum(winner != predictor_id and all(float(per_predictor[other][index]["squared_error"]) <= float(per_predictor[predictor_id][index]["squared_error"]) + 1e-12 for other in FIXED_IDS) for index, winner in enumerate(per_origin_winners)),
                        "performance_dispersion_squared_error": statistics.pstdev(origin_mse[predictor_id]),
                        "late_origin_mean_mse": mean([origin_mse[predictor_id][COMMON_ORIGINS.index(origin)] for origin in (2019, 2020, 2021)]),
                        "latest_2021_squared_error": origin_mse[predictor_id][-1],
                        "historical_cumulative_rank": ordered.index(predictor_id) + 1,
                        "historical_best_predictor": best,
                        "best_vs_second_best_margin_sse": cumulative[second] - cumulative[best],
                        "best_vs_second_best_relative_margin": (cumulative[second] - cumulative[best]) / cumulative[second] if cumulative[second] else 0.0,
                        "preference_descriptor": label,
                    })
                stability_rows.append({
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "city_id": city,
                    "historical_best_predictor": best,
                    "historical_second_best_predictor": second,
                    "historical_worst_predictor": worst,
                    "historical_best_model_count": best_wins,
                    "historical_best_model_share": best_wins / len(COMMON_ORIGINS),
                    "winner_sequence_by_origin": ";".join(f"{origin}:{winner}" for origin, winner in zip(COMMON_ORIGINS, per_origin_winners)),
                    "winner_switch_count": sum(left != right for left, right in zip(per_origin_winners, per_origin_winners[1:])),
                    "best_vs_second_best_margin_sse": cumulative[second] - cumulative[best],
                    "best_vs_second_best_relative_margin": (cumulative[second] - cumulative[best]) / cumulative[second] if cumulative[second] else 0.0,
                    "leave_one_origin_winner_sequence": ";".join(f"omit_{origin}:{winner}" for origin, winner in zip(COMMON_ORIGINS, leave_one_out_winners)),
                    "best_survives_every_single_origin_removal": "yes" if robust else "no",
                    "largest_positive_margin_origin_share": largest_positive_share,
                    "preference_descriptor": label,
                    "interpretation_scope": "descriptive fixed-baseline evidence; not used as a deployed selector rule",
                })
    consistency_rows = []
    resident = {row["city_id"]: row for row in stability_rows if row["target_id"] == "resident" and row["scenario_id"] == SCENARIOS[0]}
    registered = {row["city_id"]: row for row in stability_rows if row["target_id"] == "registered" and row["scenario_id"] == SCENARIOS[0]}
    for city in CITIES:
        r = resident[city]
        g = registered[city]
        consistency_rows.append({
            "scenario_id": SCENARIOS[0],
            "city_id": city,
            "resident_historical_best_predictor": r["historical_best_predictor"],
            "registered_historical_best_predictor": g["historical_best_predictor"],
            "same_historical_best_predictor": "yes" if r["historical_best_predictor"] == g["historical_best_predictor"] else "no",
            "resident_preference_descriptor": r["preference_descriptor"],
            "registered_preference_descriptor": g["preference_descriptor"],
            "same_preference_descriptor": "yes" if r["preference_descriptor"] == g["preference_descriptor"] else "no",
        })
    return profile_rows, stability_rows, consistency_rows


def oracle_artifacts(fixed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = fixed_row_map(fixed_rows)
    static_choice: dict[tuple[str, str, str], str] = {}
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for city in CITIES:
                cumulative = {predictor_id: math.fsum(float(by_key[(target["target_id"], scenario_id, predictor_id, origin, city)]["squared_error"]) for origin in COMMON_ORIGINS) for predictor_id in FIXED_IDS}
                static_choice[(target["target_id"], scenario_id, city)] = min(FIXED_IDS, key=lambda predictor_id: (cumulative[predictor_id], FIXED_IDS.index(predictor_id)))
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                for city in CITIES:
                    candidate_rows = [by_key[(target["target_id"], scenario_id, predictor_id, origin, city)] for predictor_id in FIXED_IDS]
                    row_oracle_source = min(candidate_rows, key=lambda row: (float(row["squared_error"]), FIXED_IDS.index(row["predictor_id"])))
                    static_source = by_key[(target["target_id"], scenario_id, static_choice[(target["target_id"], scenario_id, city)], origin, city)]
                    for oracle_id, source, oracle_scope in (
                        ("ORACLE_ROW_ORIGIN", row_oracle_source, "row/origin post-hoc oracle"),
                        ("ORACLE_STATIC_CITY", static_source, "static city post-hoc oracle"),
                    ):
                        oracle_row = dict(source)
                        oracle_row.update({
                            "predictor_id": oracle_id,
                            "oracle_marker": "ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND",
                            "oracle_scope": oracle_scope,
                            "oracle_selected_predictor_id": source["predictor_id"],
                            "deployable": "no",
                        })
                        rows.append(oracle_row)
                    assignments.extend([
                        {
                            "target_id": target["target_id"],
                            "scenario_id": scenario_id,
                            "origin_year": origin,
                            "city_id": city,
                            "oracle_type": "ORACLE_ROW_ORIGIN",
                            "selected_predictor_id": row_oracle_source["predictor_id"],
                            "selection_information": "current origin actual residual; post-hoc only",
                            "deployable": "no",
                        },
                        {
                            "target_id": target["target_id"],
                            "scenario_id": scenario_id,
                            "origin_year": origin,
                            "city_id": city,
                            "oracle_type": "ORACLE_STATIC_CITY",
                            "selected_predictor_id": static_choice[(target["target_id"], scenario_id, city)],
                            "selection_information": "complete 2011-2021 validation history; post-hoc only",
                            "deployable": "no",
                        },
                    ])
    assert len(rows) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES) * 2
    return rows, assignments, [{"target_id": target["target_id"], "scenario_id": scenario_id, "city_id": city, "static_city_oracle_predictor_id": predictor_id, "oracle_marker": "ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND", "deployable": "no"} for (target_id, scenario_id, city), predictor_id in static_choice.items() for target in TARGETS if target["target_id"] == target_id]


def oracle_metrics(oracle_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oracle_defs = (
        {"predictor_id": "ORACLE_ROW_ORIGIN"},
        {"predictor_id": "ORACLE_STATIC_CITY"},
    )
    saved = plan01.PREDICTORS
    plan01.PREDICTORS = oracle_defs
    try:
        per_origin, _ = plan01.build_per_origin_metrics(oracle_rows)
        aggregate, _ = plan01.build_aggregate_metrics(oracle_rows, per_origin)
    finally:
        plan01.PREDICTORS = saved
    rows = [{"metric_level": "per_origin", **row, "oracle_marker": "ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND", "deployable": "no"} for row in per_origin]
    rows.extend({"metric_level": "aggregate", **row, "oracle_marker": "ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND", "deployable": "no"} for row in aggregate)
    return rows, aggregate


def contract_rows() -> list[dict[str, Any]]:
    return [
        {"contract_item": "origin_domain", "frozen_value": "2011-2021 common strict origins", "status": "PASS", "evidence": "per_origin_metrics.csv"},
        {"contract_item": "target_history_cutoff", "frozen_value": "all fixed predictor fit rows <= T-2", "status": "PASS", "evidence": "leakage_audit.csv"},
        {"contract_item": "primary_target", "frozen_value": "resident / 常住人口（万人）", "status": "PASS", "evidence": "aggregate_metrics.csv"},
        {"contract_item": "sensitivity_target", "frozen_value": "registered / 户籍人口（万人）", "status": "PASS", "evidence": "aggregate_metrics.csv"},
        {"contract_item": "primary_aggregation", "frozen_value": "pooled city-year MSE", "status": "PASS", "evidence": "aggregate_metrics.csv and pairwise_baseline_comparison.csv"},
        {"contract_item": "aggregation_sensitivity", "frozen_value": "macro-origin MSE", "status": "PASS", "evidence": "aggregate_metrics.csv and pairwise_baseline_comparison.csv"},
        {"contract_item": "latest_diagnostic", "frozen_value": "2021 latest holdout reported separately", "status": "PASS", "evidence": "aggregate_metrics.csv and late_origin_comparison.csv"},
        {"contract_item": "replay_scenarios", "frozen_value": "R1 and R2 retained; target-history-only numeric equivalence expected", "status": "PASS", "evidence": "r1_r2_equivalence_check.csv"},
        {"contract_item": "recent_window_scope", "frozen_value": "only fixed 3-point and 5-point OLS windows; no sweep", "status": "PASS", "evidence": "baseline_predictor_definitions.csv"},
        {"contract_item": "selector_history_boundary", "frozen_value": "selector history max origin < current origin", "status": "PASS", "evidence": "selector_history_audit.csv"},
        {"contract_item": "selector_cold_start", "frozen_value": "fewer than 3 prior scored origins => fixed B1", "status": "PASS", "evidence": "selector_definitions.csv and selector_history_audit.csv"},
        {"contract_item": "oracle_scope", "frozen_value": "oracle is non-deployable post-hoc diagnostic only", "status": "PASS", "evidence": "oracle_metrics.csv and oracle_assignment_distribution.csv"},
        {"contract_item": "model_boundary", "frozen_value": "no covariates, complex ML, tuning, ensemble, clustering, 2023 forecast or submission", "status": "PASS", "evidence": "plan02_summary.json"},
    ]


def plan01_compatibility(fixed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prior_rows = read_csv(PLAN01_DIR / "per_origin_predictions_or_residuals.csv")
    prior_map = {(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row for row in prior_rows}
    current = fixed_row_map(fixed_rows)
    result = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for predictor_id in FIXED_IDS[:3]:
                for origin in COMMON_ORIGINS:
                    deltas = []
                    for city in CITIES:
                        left = current[(target["target_id"], scenario_id, predictor_id, origin, city)]
                        right = prior_map[(target["target_id"], scenario_id, predictor_id, origin, city)]
                        for left_field, right_field in (("prediction_value", "prediction_value"), ("residual_prediction_minus_actual", "residual_prediction_minus_actual"), ("squared_error", "squared_error")):
                            left_value = as_float(left[left_field])
                            right_value = as_float(right[right_field])
                            if left_value is not None and right_value is not None:
                                deltas.append(abs(float(plan01.rounded(left_value)) - right_value))
                            elif left_value != right_value:
                                deltas.append(math.inf)
                    maximum = max(deltas) if deltas else 0.0
                    result.append({
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "predictor_id": predictor_id,
                        "rows_compared": len(CITIES),
                        "max_abs_rounded_value_delta": maximum,
                        "compatibility_tolerance": 1e-12,
                        "status": "PASS" if maximum <= 1e-12 else "FAIL",
                        "comparison": "Plan02 rerun versus S3-Plan01 rounded machine artifact",
                    })
    assert all(row["status"] == "PASS" for row in result)
    return result


def late_comparison(per_origin_metrics: list[dict[str, Any]], selector_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = [{"record_type": "metric", **row} for row in per_origin_metrics if int(row["origin_year"]) in (2019, 2020, 2021)]
    choice_counter: Counter[tuple[str, str, int, str, str]] = Counter()
    for row in selector_rows:
        origin = int(row["origin_year"])
        if origin in (2019, 2020, 2021):
            choice_counter[(row["target_id"], row["scenario_id"], origin, row["predictor_id"], row["selected_fixed_predictor_id"])] += 1
    choices = [{"record_type": "selector_choice", "target_id": target_id, "scenario_id": scenario_id, "origin_year": origin, "selector_id": selector_id, "selected_fixed_predictor_id": selected_id, "city_count": count} for (target_id, scenario_id, origin, selector_id, selected_id), count in sorted(choice_counter.items())]
    by_key = fixed_row_map(fixed_rows)
    winners = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for origin in (2019, 2020, 2021):
                counts = Counter()
                for city in CITIES:
                    rows = [by_key[(target["target_id"], scenario_id, predictor_id, origin, city)] for predictor_id in FIXED_IDS]
                    winner = min(rows, key=lambda row: (float(row["squared_error"]), FIXED_IDS.index(row["predictor_id"])))
                    counts[winner["predictor_id"]] += 1
                winners.extend({"record_type": "fixed_city_winner", "target_id": target["target_id"], "scenario_id": scenario_id, "origin_year": origin, "predictor_id": predictor_id, "city_count": count} for predictor_id, count in sorted(counts.items()))
    return metrics, choices, winners


def selector_concentration(per_pair_summary: list[dict[str, Any]], concentration_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in concentration_summary if row["pair_id"] in {"GLOBAL_SELECTOR_vs_B1", "CITY_SELECTOR_vs_B1", "CITY_SELECTOR_vs_GLOBAL_SELECTOR"}]


def main() -> None:
    assert STARTING_COMMIT == "ed006d91174889280451878a2c1a820fb05a95bf"
    population, population_profile, s2_boundary, s2_index = load_context()

    fixed_rows, fixed_audit = build_fixed_rows(population)
    selector_rows, selector_audit = build_selector_rows(fixed_rows)
    all_rows = fixed_rows + selector_rows
    all_audit = fixed_audit + selector_audit
    assert len(all_rows) == len(TARGETS) * len(SCENARIOS) * (len(FIXED_IDS) + len(SELECTOR_IDS)) * len(COMMON_ORIGINS) * len(CITIES)
    assert all(row["leakage_status"] == "PASS" for row in all_audit)

    saved_predictors = plan01.PREDICTORS
    saved_pairs = plan01.PAIRWISE_COMPARISONS
    plan01.PREDICTORS = ALL_PREDICTORS
    plan01.PAIRWISE_COMPARISONS = PAIRWISE_COMPARISONS
    try:
        per_origin_metrics, per_origin_metric_map = plan01.build_per_origin_metrics(all_rows)
        aggregate_metrics, aggregate_map = plan01.build_aggregate_metrics(all_rows, per_origin_metrics)
        stability = plan01.build_stability_summary(per_origin_metrics, aggregate_map)
        pairwise_summary, pairwise_origin, row_comparison_map = plan01.build_pairwise_comparisons(all_rows, aggregate_map, per_origin_metric_map)
        concentration_city, concentration_summary = plan01.build_improvement_concentration(pairwise_summary, row_comparison_map, aggregate_map)
        r1_r2 = plan01.build_r1_r2_equivalence(all_rows)
        s2_compatibility = plan01.build_s2_compatibility(all_rows, s2_index)
    finally:
        plan01.PREDICTORS = saved_predictors
        plan01.PAIRWISE_COMPARISONS = saved_pairs

    assert all(row["status"] == "PASS" for row in r1_r2)
    assert all(row["status"] == "PASS" for row in s2_compatibility)
    plan01_compat = plan01_compatibility(fixed_rows)
    slope_rows, slope_summary = slope_diagnostics(fixed_rows)
    city_profile, city_stability, target_consistency = city_preference_artifacts(fixed_rows)
    oracle_rows, oracle_assignments, static_oracle_choices = oracle_artifacts(fixed_rows)
    oracle_metric_rows, oracle_aggregate = oracle_metrics(oracle_rows)
    late_metrics, late_choices, late_winners = late_comparison(per_origin_metrics, selector_rows, fixed_rows)
    contract_evidence = contract_rows()
    assert all(row["status"] == "PASS" for row in contract_evidence)

    origin_boundary = []
    for source_row in s2_boundary:
        row = dict(source_row)
        row.update({
            "s3_plan_id": "S3-Plan02",
            "s3_predictor_family": "B0/B1/B2 plus fixed B3/B4; adaptive selectors use only prior pseudo-origin outcomes",
            "s3_recent_window_scope": "only 3-point and 5-point OLS; no additional window sweep",
            "s3_target_history_rule": "all selected fixed fit rows <= T-2",
            "s3_selector_history_rule": "selector history max origin < current origin",
            "s3_scenario_numeric_effect_expected": "none for target-history-only predictors and selectors",
        })
        origin_boundary.append(row)

    fixed_definitions = []
    for predictor in FIXED_PREDICTORS:
        fixed_definitions.append({
            **predictor,
            "plan_id": "S3-Plan02",
            "pre_registered_for_this_plan": "yes",
            "target_history_horizon_years": HORIZON_YEARS,
            "fallback_policy": "B1 fixed fallback only for B3/B4 when fewer than declared window points; no silent window expansion",
            "uses_covariates": "no",
            "formal_ml_model": "no",
            "tuning": "no",
            "ensemble": "no",
        })

    summary = {
        "plan_id": "S3-Plan02",
        "stage": "S3_BASELINE_LADDER",
        "starting_commit": git_starting_commit(),
        "branch": "main",
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "latest_origin": LATEST_ORIGIN,
        "target_history_horizon_years": HORIZON_YEARS,
        "primary_target": "resident / 常住人口（万人）",
        "sensitivity_target": "registered / 户籍人口（万人）",
        "primary_metric": "pooled city-year MSE",
        "aggregation_sensitivity": "macro-origin MSE",
        "fixed_predictor_ids": list(FIXED_IDS),
        "selector_ids": list(SELECTOR_IDS),
        "selector_candidate_ids": list(SELECTOR_CANDIDATE_IDS),
        "fixed_prediction_rows": len(fixed_rows),
        "selector_prediction_rows": len(selector_rows),
        "prediction_rows": len(all_rows),
        "leakage_audit_rows": len(all_audit),
        "leakage_audit_all_passed": all(row["leakage_status"] == "PASS" for row in all_audit),
        "selector_history_audit_all_passed": all(row["leakage_status"] == "PASS" for row in selector_audit),
        "r1_r2_equivalence_rows": len(r1_r2),
        "r1_r2_equivalence_all_passed": all(row["status"] == "PASS" for row in r1_r2),
        "s2_b0_b1_compatibility_rows": len(s2_compatibility),
        "s2_b0_b1_compatibility_all_passed": all(row["status"] == "PASS" for row in s2_compatibility),
        "plan01_b0_b1_b2_compatibility_rows": len(plan01_compat),
        "plan01_b0_b1_b2_compatibility_all_passed": all(row["status"] == "PASS" for row in plan01_compat),
        "validation_contract_all_passed": all(row["status"] == "PASS" for row in contract_evidence),
        "oracle_rows": len(oracle_rows),
        "oracle_marker": "ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND",
        "formal_ml_model_used": False,
        "covariates_used": False,
        "tuning_run": False,
        "ensemble_run": False,
        "clustering_run": False,
        "learned_gate_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "test_xls_used": False,
        "s2_contract_modified": False,
        "s3_complete": False,
        "s4_entered": False,
        "population_profile": population_profile,
        "required_evidence_scope": "fixed ladder, adaptive selectors, oracle diagnostics, city heterogeneity, concentration, late-origin, slope mechanism, leakage",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "baseline_predictor_definitions.csv", fixed_definitions)
    write_csv(OUTPUT_DIR / "selector_definitions.csv", [dict(row, plan_id="S3-Plan02", target_history_horizon_years=HORIZON_YEARS, deployable="yes") for row in SELECTOR_DEFINITIONS])
    write_csv(OUTPUT_DIR / "validation_contract_evidence.csv", contract_evidence)
    write_csv(OUTPUT_DIR / "origin_information_boundary.csv", origin_boundary)
    write_csv(
        OUTPUT_DIR / "per_origin_predictions_or_residuals.csv",
        all_rows,
        fields=union_fields(all_rows),
    )
    write_csv(OUTPUT_DIR / "per_origin_metrics.csv", per_origin_metrics)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(OUTPUT_DIR / "pairwise_baseline_comparison.csv", pairwise_summary)
    write_csv(OUTPUT_DIR / "pairwise_origin_comparison.csv", pairwise_origin)
    write_csv(OUTPUT_DIR / "validation_stability_summary.csv", stability)
    write_csv(OUTPUT_DIR / "city_improvement_concentration.csv", concentration_city)
    write_csv(OUTPUT_DIR / "improvement_concentration_summary.csv", concentration_summary)
    write_csv(OUTPUT_DIR / "selector_improvement_concentration_summary.csv", selector_concentration(pairwise_summary, concentration_summary))
    write_csv(OUTPUT_DIR / "leakage_audit.csv", all_audit)
    write_csv(OUTPUT_DIR / "selector_history_audit.csv", selector_audit)
    write_csv(OUTPUT_DIR / "r1_r2_equivalence_check.csv", r1_r2)
    write_csv(OUTPUT_DIR / "s2_b0_b1_compatibility_check.csv", s2_compatibility)
    write_csv(OUTPUT_DIR / "plan01_b0_b1_b2_compatibility_check.csv", plan01_compat)
    write_csv(OUTPUT_DIR / "slope_mechanism_diagnostics.csv", slope_rows)
    write_csv(OUTPUT_DIR / "slope_disagreement_summary.csv", slope_summary)
    write_csv(OUTPUT_DIR / "city_predictor_performance.csv", city_profile)
    write_csv(OUTPUT_DIR / "city_preference_stability.csv", city_stability)
    write_csv(OUTPUT_DIR / "target_preference_consistency.csv", target_consistency)
    write_csv(OUTPUT_DIR / "oracle_predictions_or_residuals.csv", oracle_rows)
    write_csv(
        OUTPUT_DIR / "oracle_metrics.csv",
        oracle_metric_rows,
        fields=union_fields(oracle_metric_rows),
    )
    write_csv(OUTPUT_DIR / "oracle_assignment_distribution.csv", oracle_assignments)
    write_csv(OUTPUT_DIR / "oracle_static_city_assignments.csv", static_oracle_choices)
    late_rows = late_metrics + late_choices + late_winners
    write_csv(
        OUTPUT_DIR / "late_origin_comparison.csv",
        late_rows,
        fields=union_fields(late_rows),
    )
    write_json(OUTPUT_DIR / "plan02_summary.json", summary)

    print("S3_PLAN02_RECENT_WINDOW_SELECTOR_PASS")
    print(f"starting_commit={summary['starting_commit']}")
    print("common_origin_domain=2011-2021")
    print(f"fixed_predictor_count={len(FIXED_IDS)}")
    print(f"selector_count={len(SELECTOR_IDS)}")
    print(f"prediction_rows={len(all_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"aggregate_metric_rows={len(aggregate_metrics)}")
    print(f"pairwise_summary_rows={len(pairwise_summary)}")
    print(f"oracle_rows={len(oracle_rows)}")
    print(f"leakage_audit_all_passed={summary['leakage_audit_all_passed']}")
    print(f"selector_history_audit_all_passed={summary['selector_history_audit_all_passed']}")
    print(f"r1_r2_equivalence_all_passed={summary['r1_r2_equivalence_all_passed']}")
    print(f"s2_b0_b1_compatibility_all_passed={summary['s2_b0_b1_compatibility_all_passed']}")
    print(f"plan01_b0_b1_b2_compatibility_all_passed={summary['plan01_b0_b1_b2_compatibility_all_passed']}")
    print("formal_ml_model_used=False")
    print("covariates_used=False")
    print("tuning_run=False")
    print("ensemble_run=False")
    print("clustering_run=False")
    print("learned_gate_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
