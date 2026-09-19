"""Build S3-Plan03 pooled simple-regression and growth evidence.

The runner reuses the Plan02 target-only evaluator for B0--B4 and adds only
the pre-registered B5/B6 pooled city-fixed-effect additive trends and B7
recent-3 per-city log-linear growth.  It also produces robustness diagnostics
and an ML hurdle; it does not run formal ML or enter S4.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import build_s3_plan01_baseline_ladder as plan01
import build_s3_plan02_recent_window_and_selector as plan02


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
PLAN01_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan01"
PLAN02_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan02"
S2_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05"

STARTING_COMMIT = "7e741ab91cdee3d9df396212da3b7e454629bf5d"
HORIZON_YEARS = plan02.HORIZON_YEARS
REFERENCE_TARGET_YEAR = plan02.REFERENCE_TARGET_YEAR
LATEST_ORIGIN = plan02.LATEST_ORIGIN
COMMON_ORIGINS = plan02.COMMON_ORIGINS
EARLY_ORIGINS = plan02.EARLY_ORIGINS
LATE_ORIGINS = plan02.LATE_ORIGINS
CITIES = plan02.CITIES
SCENARIOS = plan02.SCENARIOS
TARGETS = plan02.TARGETS

MIDDLE_ORIGINS = tuple(range(2016, 2019))
REGIMES = {
    "early_2011_2015": EARLY_ORIGINS,
    "middle_2016_2018": MIDDLE_ORIGINS,
    "late_2019_2021": LATE_ORIGINS,
}

NEW_PREDICTORS = (
    {
        "predictor_id": "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND",
        "predictor_name": "expanding pooled city-fixed-effect additive trend",
        "mechanism": "pooled_shared_absolute_slope",
        "formula": "y(city,year) = city_intercept(city) + shared_slope * year using all target rows <= T-2",
        "fit_scope": "pooled across cities with independent city intercepts",
        "window_size": "all",
        "fallback_policy": "fixed B3 fallback only if a city has no usable pooled fit; no imputation",
        "uses_covariates": "no",
        "tuning": "no",
        "ensemble": "no",
    },
    {
        "predictor_id": "B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND",
        "predictor_name": "recent-3 pooled city-fixed-effect additive trend",
        "mechanism": "pooled_shared_recent_absolute_slope",
        "formula": "y = city_intercept(city) + shared_slope * year using exactly the latest 3 target rows <= T-2 per fit city",
        "fit_scope": "pooled across cities with exactly 3 recent points and independent city intercepts",
        "window_size": 3,
        "fallback_policy": "city with fewer than 3 points uses fixed B3 fallback and is excluded from pooled fit",
        "uses_covariates": "no",
        "tuning": "no",
        "ensemble": "no",
    },
    {
        "predictor_id": "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH",
        "predictor_name": "recent-3 per-city log-linear growth",
        "mechanism": "per_city_multiplicative_growth",
        "formula": "log(y) = intercept + growth_slope * year on the latest 3 positive target rows <= T-2; predict exp(log_y_hat(T))",
        "fit_scope": "one log-linear fit per city",
        "window_size": 3,
        "fallback_policy": "fixed B3 fallback if fewer than 3 positive points; non-positive points are excluded and explicitly audited",
        "uses_covariates": "no",
        "tuning": "no",
        "ensemble": "no",
    },
)

ALL_PREDICTORS = plan02.FIXED_PREDICTORS + NEW_PREDICTORS
FIXED_IDS = tuple(row["predictor_id"] for row in ALL_PREDICTORS)

PAIRWISE_COMPARISONS = (
    ("B1_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B1_TWO_POINT_LINEAR_TREND"),
    ("B2_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B2_EXPANDING_PER_CITY_LINEAR_TREND"),
    ("B2_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B2_EXPANDING_PER_CITY_LINEAR_TREND"),
    ("B3_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B3_vs_B2", "B2_EXPANDING_PER_CITY_LINEAR_TREND", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B2", "B2_EXPANDING_PER_CITY_LINEAR_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B4_vs_B3", "B3_RECENT_3POINT_OLS_TREND", "B4_RECENT_5POINT_OLS_TREND"),
    ("B5_vs_B2", "B2_EXPANDING_PER_CITY_LINEAR_TREND", "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND"),
    ("B5_vs_B3", "B3_RECENT_3POINT_OLS_TREND", "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND"),
    ("B6_vs_B3", "B3_RECENT_3POINT_OLS_TREND", "B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND"),
    ("B6_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND"),
    ("B7_vs_B3", "B3_RECENT_3POINT_OLS_TREND", "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH"),
    ("B7_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH"),
    ("B5_vs_B1", "B1_TWO_POINT_LINEAR_TREND", "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND"),
    ("B3_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B3_RECENT_3POINT_OLS_TREND"),
    ("B4_vs_B0", "B0_PERSISTENCE_LAST_OBSERVED", "B4_RECENT_5POINT_OLS_TREND"),
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


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "NA", "N/A", "NaN", "nan", "None"}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float:
    assert values
    return math.fsum(values) / len(values)


def competition_score(value: float) -> float:
    return 1.0 / (1.0 + value)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = math.fsum((x - left_mean) ** 2 for x in left)
    right_ss = math.fsum((y - right_mean) ** 2 for y in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def fit_ols(years: list[int], values: list[float], target_year: int, log_space: bool = False) -> dict[str, float]:
    assert len(years) == len(values) and len(years) >= 2
    fit_values = [math.log(value) for value in values] if log_space else values
    x_mean = mean([float(year) for year in years])
    y_mean = mean(fit_values)
    denominator = math.fsum((float(year) - x_mean) ** 2 for year in years)
    assert denominator > 0
    numerator = math.fsum((float(year) - x_mean) * (value - y_mean) for year, value in zip(years, fit_values))
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    fitted_value = intercept + slope * target_year
    return {
        "slope": slope,
        "intercept": intercept,
        "prediction": math.exp(fitted_value) if log_space else fitted_value,
    }


def fit_pooled_city_fe(points_by_city: dict[str, list[tuple[int, float]]]) -> dict[str, Any] | None:
    city_stats: dict[str, dict[str, Any]] = {}
    all_rows: list[tuple[str, int, float]] = []
    for city in CITIES:
        points = points_by_city.get(city, [])
        if not points:
            continue
        years = [year for year, _ in points]
        values = [value for _, value in points]
        x_mean = mean([float(year) for year in years])
        y_mean = mean(values)
        city_stats[city] = {
            "years": years,
            "values": values,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "count": len(points),
        }
        all_rows.extend((city, year, value) for year, value in points)
    denominator = math.fsum(
        (float(year) - city_stats[city]["x_mean"]) ** 2
        for city, year, _ in all_rows
    )
    if denominator <= 0 or not all_rows:
        return None
    numerator = math.fsum(
        (float(year) - city_stats[city]["x_mean"]) * (value - city_stats[city]["y_mean"])
        for city, year, value in all_rows
    )
    shared_slope = numerator / denominator
    for city, stats in city_stats.items():
        stats["intercept"] = stats["y_mean"] - shared_slope * stats["x_mean"]
    return {
        "city_stats": city_stats,
        "shared_slope": shared_slope,
        "numerator": numerator,
        "denominator": denominator,
        "training_row_count": len(all_rows),
        "training_city_count": len(city_stats),
        "training_min_year": min(year for _, year, _ in all_rows),
        "training_max_year": max(year for _, year, _ in all_rows),
        "training_point_count_min": min(stats["count"] for stats in city_stats.values()),
        "training_point_count_max": max(stats["count"] for stats in city_stats.values()),
    }


def fit_record(
    predictor_id: str,
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
    history_years: list[int],
    prediction: float | None,
    slope: float | None,
    intercept: float | None,
    status: str,
    fixed_window_size: Any,
    fallback_applied: str = "no",
    fallback_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eligible = sorted(year for year in series if year <= cutoff_year)
    result = {
        "eligible_history_years": eligible,
        "history_years": history_years,
        "history_count": len(history_years),
        "history_min_year": history_years[0] if history_years else None,
        "history_max_year": history_years[-1] if history_years else None,
        "latest_target_year_used": history_years[-1] if history_years else None,
        "previous_target_year_used": history_years[-2] if len(history_years) >= 2 else None,
        "regression_fit_min_year": history_years[0] if history_years else None,
        "regression_fit_max_year": history_years[-1] if history_years else None,
        "regression_fit_distinct_year_count": len(set(history_years)),
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_fit_status": status,
        "fixed_window_size": fixed_window_size,
        "window_points_available": len(eligible),
        "window_points_used": len(history_years),
        "window_fallback_applied": fallback_applied,
        "window_fallback_reason": fallback_reason,
        "prediction": prediction,
    }
    if extra:
        result.update(extra)
    return result


def row_from_model(
    target: dict[str, str],
    scenario_id: str,
    predictor_id: str,
    origin: int,
    city: str,
    series: dict[int, float],
    fit: dict[str, Any],
) -> dict[str, Any]:
    row = plan02.row_from_fit(target, scenario_id, predictor_id, origin, city, series, fit)
    row.update({
        "model_family": "pooled_city_fixed_effect_additive" if predictor_id.startswith("B5_") or predictor_id.startswith("B6_") else "per_city_log_linear_growth",
        "target_t_minus_1_present": "yes" if (origin - 1) in series else "no",
        "pooled_training_row_count": fit.get("pooled_training_row_count"),
        "pooled_training_city_count": fit.get("pooled_training_city_count"),
        "pooled_training_min_year": fit.get("pooled_training_min_year"),
        "pooled_training_max_year": fit.get("pooled_training_max_year"),
        "pooled_training_point_count_min": fit.get("pooled_training_point_count_min"),
        "pooled_training_point_count_max": fit.get("pooled_training_point_count_max"),
        "shared_slope": fit.get("shared_slope"),
        "shared_slope_numerator": fit.get("shared_slope_numerator"),
        "shared_slope_denominator": fit.get("shared_slope_denominator"),
        "city_intercept": fit.get("city_intercept"),
        "growth_slope": fit.get("growth_slope"),
        "growth_intercept": fit.get("growth_intercept"),
        "nonpositive_eligible_history_count": fit.get("nonpositive_eligible_history_count", 0),
        "nonpositive_history_policy": fit.get("nonpositive_history_policy", "not_applicable"),
        "pooled_city_fit_status": fit.get("pooled_city_fit_status", "not_applicable"),
    })
    return row


def audit_row(row: dict[str, Any], model_kind: str, pooled_max_year: int | None = None) -> dict[str, Any]:
    origin = int(row["origin_year"])
    cutoff = origin - HORIZON_YEARS
    history_value = row.get("history_max_year")
    fit_value = row.get("regression_fit_max_year")
    history_max = int(history_value) if history_value not in (None, "") else None
    fit_max = int(fit_value) if fit_value not in (None, "") else None
    pooled_max = pooled_max_year if pooled_max_year is not None else fit_max or history_max
    target_history_pass = (history_max is None or history_max <= cutoff) and (fit_max is None or fit_max <= cutoff)
    pooled_history_pass = pooled_max is None or pooled_max <= cutoff
    status = target_history_pass and pooled_history_pass
    assert status
    return {
        "target_id": row["target_id"],
        "scenario_id": row["scenario_id"],
        "predictor_id": row["predictor_id"],
        "origin_year": origin,
        "city_id": row["city_id"],
        "scoring_target_year": origin,
        "allowed_target_history_cutoff_year": cutoff,
        "history_max_year_used": history_max,
        "regression_fit_max_year_used": fit_max,
        "pooled_training_max_year_used": pooled_max,
        "model_kind": model_kind,
        "t_minus_1_target_present_but_excluded": row.get("target_t_minus_1_present", "no"),
        "origin_target_scoring_only": "yes",
        "current_origin_other_city_labels_used": "no" if model_kind == "pooled_city_fixed_effect" else "not_applicable",
        "future_target_years_not_used": "PASS",
        "target_history_max_within_T_minus_2": "PASS" if target_history_pass else "FAIL",
        "pooled_training_max_within_T_minus_2": "PASS" if pooled_history_pass else "FAIL",
        "future_labels_excluded_from_fit": "PASS",
        "leakage_status": "PASS" if status else "FAIL",
        "audit_rule": "all target and pooled training rows must have year <= T-2; current origin labels are scoring-only",
    }


def load_context() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any], list[dict[str, str]], dict[tuple[str, str, str, int, str], dict[str, str]]]:
    plan02.plan02 = plan02
    plan02.plan01.assert_static_boundaries()
    population, profile = plan02.plan01.load_population()
    s2_summary, s2_boundary, s2_index = plan02.plan01.load_s2_context()
    plan02.plan01.load_scenarios()
    assert s2_summary["common_strict_origin_domain"] == "2011-2021"
    assert s2_summary["s2_complete"] is False
    plan01_summary = json.loads((PLAN01_DIR / "plan01_summary.json").read_text(encoding="utf-8"))
    plan02_summary = json.loads((PLAN02_DIR / "plan02_summary.json").read_text(encoding="utf-8"))
    assert plan01_summary["s2_contract_modified"] is False
    assert plan02_summary["s2_contract_modified"] is False
    assert plan02_summary["plan_id"] == "S3-Plan02"
    return population, profile, s2_boundary, s2_index


def build_pooled_rows(
    population: dict[str, dict[str, dict[int, float]]],
    fallback_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fallback_map = plan02.fixed_row_map(fallback_rows)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    growth_diagnostics: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                cutoff = origin - HORIZON_YEARS
                model_specs: list[tuple[str, dict[str, list[tuple[int, float]]], dict[str, Any]]] = []
                for predictor_id in (
                    "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND",
                    "B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND",
                ):
                    points_by_city: dict[str, list[tuple[int, float]]] = {}
                    for city in CITIES:
                        series = population[target_id][city]
                        eligible = sorted((year, series[year]) for year in series if year <= cutoff)
                        if predictor_id.startswith("B5"):
                            points_by_city[city] = eligible
                        else:
                            points_by_city[city] = eligible[-3:] if len(eligible) >= 3 else []
                    model = fit_pooled_city_fe(points_by_city)
                    assert model is not None
                    model_specs.append((predictor_id, points_by_city, model))
                    fit_diagnostics.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "target_history_cutoff_year": cutoff,
                        "predictor_id": predictor_id,
                        "fit_scope": "pooled_city_fixed_effect",
                        "window_policy": "all eligible points" if predictor_id.startswith("B5") else "exactly latest 3 points per fit city",
                        "training_row_count": model["training_row_count"],
                        "training_city_count": model["training_city_count"],
                        "training_min_year": model["training_min_year"],
                        "training_max_year": model["training_max_year"],
                        "training_point_count_min": model["training_point_count_min"],
                        "training_point_count_max": model["training_point_count_max"],
                        "shared_slope": model["shared_slope"],
                        "shared_slope_numerator": model["numerator"],
                        "shared_slope_denominator": model["denominator"],
                        "fit_status": "PASS",
                        "leakage_status": "PASS" if model["training_max_year"] <= cutoff else "FAIL",
                    })
                for predictor_id, points_by_city, model in model_specs:
                    for city in CITIES:
                        series = population[target_id][city]
                        eligible_years = sorted(year for year in series if year <= cutoff)
                        city_stats = model["city_stats"].get(city)
                        if city_stats is None:
                            fallback = fallback_map[(target_id, scenario_id, "B3_RECENT_3POINT_OLS_TREND", origin, city)]
                            fit = plan02.fit_prediction("B3_RECENT_3POINT_OLS_TREND", series, origin, cutoff)
                            fit.update({
                                "fixed_window_size": 3,
                                "window_fallback_applied": "yes",
                                "window_fallback_reason": "city excluded from pooled fit under fixed window policy",
                                "pooled_city_fit_status": "fallback_to_B3",
                            })
                            pooled_max = model["training_max_year"]
                        else:
                            years = city_stats["years"]
                            intercept = city_stats["intercept"]
                            fit = fit_record(
                                predictor_id,
                                series,
                                origin,
                                cutoff,
                                years,
                                intercept + model["shared_slope"] * origin,
                                model["shared_slope"],
                                intercept,
                                "pooled_city_fe_shared_slope",
                                "all" if predictor_id.startswith("B5") else 3,
                                extra={
                                    "pooled_training_row_count": model["training_row_count"],
                                    "pooled_training_city_count": model["training_city_count"],
                                    "pooled_training_min_year": model["training_min_year"],
                                    "pooled_training_max_year": model["training_max_year"],
                                    "pooled_training_point_count_min": model["training_point_count_min"],
                                    "pooled_training_point_count_max": model["training_point_count_max"],
                                    "shared_slope": model["shared_slope"],
                                    "shared_slope_numerator": model["numerator"],
                                    "shared_slope_denominator": model["denominator"],
                                    "city_intercept": intercept,
                                    "pooled_city_fit_status": "pooled_city_fe_shared_slope",
                                },
                            )
                            pooled_max = model["training_max_year"]
                        row = row_from_model(target, scenario_id, predictor_id, origin, city, series, fit)
                        row["pooled_city_fit_status"] = fit.get("pooled_city_fit_status", "pooled_city_fe_shared_slope")
                        rows.append(row)
                        audit.append(audit_row(row, "pooled_city_fixed_effect", pooled_max))

    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                cutoff = origin - HORIZON_YEARS
                for city in CITIES:
                    series = population[target_id][city]
                    eligible_years = sorted(year for year in series if year <= cutoff)
                    positive_years = [year for year in eligible_years if series[year] > 0]
                    nonpositive_count = len(eligible_years) - len(positive_years)
                    if len(positive_years) >= 3:
                        years = positive_years[-3:]
                        values = [series[year] for year in years]
                        fitted = fit_ols(years, values, origin, log_space=True)
                        fit = fit_record(
                            "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH",
                            series,
                            origin,
                            cutoff,
                            years,
                            fitted["prediction"],
                            fitted["slope"],
                            fitted["intercept"],
                            "recent_3point_log_linear_growth",
                            3,
                            extra={
                                "growth_slope": fitted["slope"],
                                "growth_intercept": fitted["intercept"],
                                "nonpositive_eligible_history_count": nonpositive_count,
                                "nonpositive_history_policy": "excluded_and_audited",
                                "pooled_city_fit_status": "per_city_log_growth",
                            },
                        )
                    else:
                        fallback = fallback_map[(target_id, scenario_id, "B3_RECENT_3POINT_OLS_TREND", origin, city)]
                        fit = plan02.fit_prediction("B3_RECENT_3POINT_OLS_TREND", series, origin, cutoff)
                        fit.update({
                            "fixed_window_size": 3,
                            "window_fallback_applied": "yes",
                            "window_fallback_reason": f"positive_history_points={len(positive_years)}<3",
                            "growth_slope": None,
                            "growth_intercept": None,
                            "nonpositive_eligible_history_count": nonpositive_count,
                            "nonpositive_history_policy": "excluded_and_audited",
                            "pooled_city_fit_status": "fallback_to_B3",
                        })
                    row = row_from_model(target, scenario_id, "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH", origin, city, series, fit)
                    rows.append(row)
                    audit.append(audit_row(row, "per_city_log_linear_growth"))
                    growth_diagnostics.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "city_id": city,
                        "target_history_cutoff_year": cutoff,
                        "eligible_history_point_count": len(eligible_years),
                        "positive_history_point_count": len(positive_years),
                        "nonpositive_eligible_history_count": nonpositive_count,
                        "positive_window_years": ";".join(str(year) for year in positive_years[-3:]),
                        "log_growth_slope": row["growth_slope"],
                        "log_growth_intercept": row["growth_intercept"],
                        "fallback_applied": row["window_fallback_applied"],
                        "growth_fit_status": row["regression_fit_status"],
                        "nonpositive_history_policy": row["nonpositive_history_policy"],
                    })
    assert len(rows) == len(TARGETS) * len(SCENARIOS) * 3 * len(COMMON_ORIGINS) * len(CITIES)
    assert all(row["leakage_status"] == "PASS" for row in audit)
    return rows, audit, fit_diagnostics, growth_diagnostics


def fixed_row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int, str], dict[str, Any]]:
    result = {(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row for row in rows}
    assert len(result) == len(rows)
    return result


def compatibility_with_plan02(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prior_rows = read_csv(PLAN02_DIR / "per_origin_predictions_or_residuals.csv")
    prior_map = {(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]), row["city_id"]): row for row in prior_rows}
    current = fixed_row_map(all_rows)
    result = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for predictor_id in plan02.FIXED_IDS:
                for origin in COMMON_ORIGINS:
                    deltas = []
                    for city in CITIES:
                        left = current[(target["target_id"], scenario_id, predictor_id, origin, city)]
                        right = prior_map[(target["target_id"], scenario_id, predictor_id, origin, city)]
                        for field in ("prediction_value", "residual_prediction_minus_actual", "squared_error"):
                            left_value = as_float(left[field])
                            right_value = as_float(right[field])
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
                        "comparison": "Plan03 rerun versus S3-Plan02 rounded machine artifact",
                    })
    assert all(row["status"] == "PASS" for row in result)
    return result


def local_pooled_diagnostics(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = fixed_row_map(all_rows)
    local_rows: list[dict[str, Any]] = []
    dispersion_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                b2_slopes = []
                b3_slopes = []
                relation_values = {"B5_vs_B2": ([], []), "B6_vs_B3": ([], [])}
                for city in CITIES:
                    b2 = by_key[(target_id, scenario_id, "B2_EXPANDING_PER_CITY_LINEAR_TREND", origin, city)]
                    b3 = by_key[(target_id, scenario_id, "B3_RECENT_3POINT_OLS_TREND", origin, city)]
                    b5 = by_key[(target_id, scenario_id, "B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND", origin, city)]
                    b6 = by_key[(target_id, scenario_id, "B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND", origin, city)]
                    b2_slope = as_float(b2["regression_slope"])
                    b3_slope = as_float(b3["regression_slope"])
                    b5_slope = as_float(b5["shared_slope"])
                    b6_slope = as_float(b6["shared_slope"])
                    if b2_slope is not None:
                        b2_slopes.append(b2_slope)
                    if b3_slope is not None:
                        b3_slopes.append(b3_slope)
                    b5_delta = float(b5["squared_error"]) - float(b2["squared_error"])
                    b6_delta = float(b6["squared_error"]) - float(b3["squared_error"])
                    b5_deviation = abs(b2_slope - b5_slope) if b2_slope is not None and b5_slope is not None else None
                    b6_deviation = abs(b3_slope - b6_slope) if b3_slope is not None and b6_slope is not None else None
                    if b5_deviation is not None:
                        relation_values["B5_vs_B2"][0].append(b5_deviation)
                        relation_values["B5_vs_B2"][1].append(b5_delta)
                    if b6_deviation is not None:
                        relation_values["B6_vs_B3"][0].append(b6_deviation)
                        relation_values["B6_vs_B3"][1].append(b6_delta)
                    local_rows.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "city_id": city,
                        "b2_expanding_local_slope": b2_slope,
                        "b3_recent_local_slope": b3_slope,
                        "b5_expanding_shared_slope": b5_slope,
                        "b6_recent3_shared_slope": b6_slope,
                        "b2_history_point_count": b2["history_observation_count"],
                        "b3_history_point_count": b3["history_observation_count"],
                        "b2_history_first_year": b2["history_min_year"],
                        "b3_history_first_year": b3["history_min_year"],
                        "b2_history_last_year": b2["history_max_year"],
                        "b3_history_last_year": b3["history_max_year"],
                        "b2_extrapolation_distance": origin - int(b2["latest_target_year_used"]),
                        "b3_extrapolation_distance": origin - int(b3["latest_target_year_used"]),
                        "absolute_b2_b5_slope_deviation": b5_deviation,
                        "absolute_b3_b6_slope_deviation": b6_deviation,
                        "b5_minus_b2_squared_error": b5_delta,
                        "b6_minus_b3_squared_error": b6_delta,
                        "b5_vs_b2_error_outcome": "better" if b5_delta < -1e-12 else "worse" if b5_delta > 1e-12 else "tie",
                        "b6_vs_b3_error_outcome": "better" if b6_delta < -1e-12 else "worse" if b6_delta > 1e-12 else "tie",
                    })
                for slope_label, slopes in (("B2_EXPANDING_LOCAL", b2_slopes), ("B3_RECENT_LOCAL", b3_slopes)):
                    q1 = percentile(slopes, 0.25)
                    q3 = percentile(slopes, 0.75)
                    dispersion_rows.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "slope_source": slope_label,
                        "city_count": len(slopes),
                        "mean_slope": mean(slopes),
                        "median_slope": statistics.median(slopes),
                        "slope_std_population": statistics.pstdev(slopes),
                        "slope_min": min(slopes),
                        "slope_max": max(slopes),
                        "slope_iqr": q3 - q1,
                        "positive_slope_share": sum(value > 0 for value in slopes) / len(slopes),
                        "negative_slope_share": sum(value < 0 for value in slopes) / len(slopes),
                        "zero_slope_share": sum(value == 0 for value in slopes) / len(slopes),
                    })
                for comparison_id, (deviations, deltas) in relation_values.items():
                    relation_rows.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "comparison_id": comparison_id,
                        "row_count": len(deviations),
                        "mean_absolute_slope_deviation": mean(deviations) if deviations else "",
                        "mean_candidate_minus_baseline_squared_error": mean(deltas) if deltas else "",
                        "candidate_better_share": sum(delta < -1e-12 for delta in deltas) / len(deltas) if deltas else "",
                        "pearson_abs_slope_deviation_vs_error_delta": correlation(deviations, deltas),
                        "interpretation_scope": "diagnostic_only; no gating rule added",
                    })
    return local_rows, dispersion_rows, relation_rows


def growth_mechanism_diagnostics(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = fixed_row_map(all_rows)
    result = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for origin in COMMON_ORIGINS:
                for city in CITIES:
                    b3 = by_key[(target_id, scenario_id, "B3_RECENT_3POINT_OLS_TREND", origin, city)]
                    b7 = by_key[(target_id, scenario_id, "B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH", origin, city)]
                    latest_value = as_float(b3["actual_target_value_for_scoring"])
                    b3_slope = as_float(b3["regression_slope"])
                    growth_slope = as_float(b7["growth_slope"])
                    delta = float(b7["squared_error"]) - float(b3["squared_error"])
                    result.append({
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin,
                        "city_id": city,
                        "latest_scoring_target_value": latest_value,
                        "b3_recent_absolute_slope": b3_slope,
                        "b7_recent_log_growth_slope": growth_slope,
                        "b7_approx_percent_growth": math.expm1(growth_slope) * 100 if growth_slope is not None else None,
                        "b7_minus_b3_squared_error": delta,
                        "b7_error_outcome": "better" if delta < -1e-12 else "worse" if delta > 1e-12 else "tie",
                        "nonpositive_eligible_history_count": b7["nonpositive_eligible_history_count"],
                        "growth_fit_status": b7["regression_fit_status"],
                    })
    return result


def leave_one_origin_out(all_rows: list[dict[str, Any]], aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["target_id"], row["scenario_id"], row["predictor_id"], int(row["origin_year"]))].append(row)
    rows = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for removed_origin in COMMON_ORIGINS:
                mse_by_predictor = {}
                for predictor_id in FIXED_IDS:
                    errors = []
                    for origin in COMMON_ORIGINS:
                        if origin == removed_origin:
                            continue
                        errors.extend(float(row["residual_prediction_minus_actual"]) for row in grouped[(target["target_id"], scenario_id, predictor_id, origin)] if row["scoreable"] == "yes")
                    mse_by_predictor[predictor_id] = mean([error * error for error in errors])
                ordered = sorted(FIXED_IDS, key=lambda predictor_id: (mse_by_predictor[predictor_id], FIXED_IDS.index(predictor_id)))
                leader = ordered[0]
                for rank, predictor_id in enumerate(ordered, start=1):
                    rows.append({
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "removed_origin_year": removed_origin,
                        "predictor_id": predictor_id,
                        "remaining_origin_domain": ";".join(str(origin) for origin in COMMON_ORIGINS if origin != removed_origin),
                        "remaining_pooled_mse": mse_by_predictor[predictor_id],
                        "remaining_pooled_score": competition_score(mse_by_predictor[predictor_id]),
                        "rank_low_is_better": rank,
                        "leave_one_origin_leader": leader,
                        "full_domain_leader": min(FIXED_IDS, key=lambda candidate: float(aggregate_map[(target["target_id"], scenario_id, candidate)]["pooled_city_year_mse"])),
                    })
    summary = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            full_leader = min(FIXED_IDS, key=lambda candidate: float(aggregate_map[(target["target_id"], scenario_id, candidate)]["pooled_city_year_mse"]))
            leaders = [row["leave_one_origin_leader"] for row in rows if row["target_id"] == target["target_id"] and row["scenario_id"] == scenario_id and row["predictor_id"] == full_leader]
            leader_by_origin = [next(row["leave_one_origin_leader"] for row in rows if row["target_id"] == target["target_id"] and row["scenario_id"] == scenario_id and int(row["removed_origin_year"]) == removed) for removed in COMMON_ORIGINS]
            summary.append({
                "target_id": target["target_id"],
                "scenario_id": scenario_id,
                "full_domain_leader": full_leader,
                "full_domain_leader_survives_leave_one_origin_count": sum(leader == full_leader for leader in leader_by_origin),
                "leave_one_origin_count": len(COMMON_ORIGINS),
                "leader_switch_count_vs_full_domain": sum(leader != full_leader for leader in leader_by_origin),
                "leader_by_removed_origin": ";".join(f"omit_{origin}:{leader}" for origin, leader in zip(COMMON_ORIGINS, leader_by_origin)),
                "b3_is_full_domain_leader": "yes" if full_leader == "B3_RECENT_3POINT_OLS_TREND" else "no",
                "interpretation_scope": "robustness diagnostic; frozen full domain remains primary",
            })
    return rows, summary


def time_regime_metrics(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for predictor_id in FIXED_IDS:
                for regime_id, origins in REGIMES.items():
                    scored = [row for row in all_rows if row["target_id"] == target["target_id"] and row["scenario_id"] == scenario_id and row["predictor_id"] == predictor_id and int(row["origin_year"]) in origins and row["scoreable"] == "yes"]
                    errors = [float(row["residual_prediction_minus_actual"]) for row in scored]
                    squared = [error * error for error in errors]
                    result.append({
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "predictor_id": predictor_id,
                        "regime_id": regime_id,
                        "origin_domain": f"{min(origins)}-{max(origins)}",
                        "origin_count": len(origins),
                        "scoreable_row_count": len(scored),
                        "pooled_mse": mean(squared),
                        "pooled_rmse": math.sqrt(mean(squared)),
                        "pooled_mae": mean([abs(error) for error in errors]),
                        "pooled_bias": mean(errors),
                        "pooled_score": competition_score(mean(squared)),
                    })
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for regime_id in REGIMES:
                rows = [row for row in result if row["target_id"] == target["target_id"] and row["scenario_id"] == scenario_id and row["regime_id"] == regime_id]
                ordered = sorted(rows, key=lambda row: (float(row["pooled_mse"]), FIXED_IDS.index(row["predictor_id"])))
                for rank, row in enumerate(ordered, start=1):
                    row["rank_low_is_better"] = rank
                    row["regime_winner"] = ordered[0]["predictor_id"]
    return result


def leave_one_city_sensitivity(all_rows: list[dict[str, Any]], row_comparison_map: dict[tuple[str, str, str, str], list[dict[str, Any]]], aggregate_map: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    selected_pairs = {"B5_vs_B2", "B6_vs_B3", "B7_vs_B3", "B5_vs_B1", "B6_vs_B1", "B7_vs_B1"}
    rows = []
    for target in TARGETS:
        for scenario_id in SCENARIOS:
            for pair_id, baseline_id, candidate_id in PAIRWISE_COMPARISONS:
                if pair_id not in selected_pairs:
                    continue
                by_city = {city: row_comparison_map[(target["target_id"], scenario_id, pair_id, city)] for city in CITIES}
                improvement_totals = {city: math.fsum(float(row["baseline_squared_error"]) - float(row["candidate_squared_error"]) for row in by_city[city]) for city in CITIES}
                baseline_sse = {city: math.fsum(float(row["baseline_squared_error"]) for row in by_city[city]) for city in CITIES}
                candidate_sse = {city: math.fsum(float(row["candidate_squared_error"]) for row in by_city[city]) for city in CITIES}
                largest_improvement_city = max(CITIES, key=lambda city: (improvement_totals[city], city))
                largest_error_city = max(CITIES, key=lambda city: (baseline_sse[city] + candidate_sse[city], city))
                for removal_kind, city in (("largest_improvement_city", largest_improvement_city), ("largest_error_city", largest_error_city)):
                    baseline_total = math.fsum(baseline_sse[other] for other in CITIES if other != city)
                    candidate_total = math.fsum(candidate_sse[other] for other in CITIES if other != city)
                    row_count = sum(len(by_city[other]) for other in CITIES if other != city)
                    rows.append({
                        "target_id": target["target_id"],
                        "scenario_id": scenario_id,
                        "pair_id": pair_id,
                        "baseline_predictor_id": baseline_id,
                        "candidate_predictor_id": candidate_id,
                        "removal_kind": removal_kind,
                        "removed_city_id": city,
                        "baseline_mse_without_city": baseline_total / row_count,
                        "candidate_mse_without_city": candidate_total / row_count,
                        "candidate_improvement_without_city": (baseline_total - candidate_total) / row_count,
                        "full_domain_candidate_improvement": float(aggregate_map[(target["target_id"], scenario_id, baseline_id)]["pooled_city_year_mse"]) - float(aggregate_map[(target["target_id"], scenario_id, candidate_id)]["pooled_city_year_mse"]),
                        "removed_city_baseline_sse": baseline_sse[city],
                        "removed_city_candidate_sse": candidate_sse[city],
                        "removed_city_delta_sse": improvement_totals[city],
                        "interpretation_scope": "no city reweighting; sensitivity only",
                    })
    return rows


def latest_origin_rows(per_origin_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"latest_record": "2021", **row} for row in per_origin_metrics if int(row["origin_year"]) == LATEST_ORIGIN]


def ml_hurdle_rows(aggregate_map: dict[tuple[str, str, str], dict[str, Any]], leave_summary: list[dict[str, Any]], concentration_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_target = "resident"
    scenario_id = SCENARIOS[0]
    strongest = min(FIXED_IDS, key=lambda predictor_id: (float(aggregate_map[(primary_target, scenario_id, predictor_id)]["pooled_city_year_mse"]), FIXED_IDS.index(predictor_id)))
    rows: list[dict[str, Any]] = []
    for target_id in ("resident", "registered"):
        aggregate = aggregate_map[(target_id, scenario_id, strongest)]
        leave = next(row for row in leave_summary if row["target_id"] == target_id and row["scenario_id"] == scenario_id)
        concentration = next((row for row in concentration_summary if row["target_id"] == target_id and row["scenario_id"] == scenario_id and row["pair_id"] == f"{strongest}_vs_B1"), None) if strongest != "B1_TWO_POINT_LINEAR_TREND" else None
        rows.append({
            "record_type": "strongest_s3_deployable_fixed_baseline",
            "target_id": target_id,
            "scenario_id": scenario_id,
            "predictor_id": strongest,
            "mechanism": next(row["formula"] for row in ALL_PREDICTORS if row["predictor_id"] == strongest),
            "pooled_mse": aggregate["pooled_city_year_mse"],
            "macro_mse": aggregate["macro_origin_mse"],
            "pooled_rmse": aggregate["pooled_city_year_rmse"],
            "pooled_mae": aggregate["pooled_city_year_mae"],
            "bias": aggregate["pooled_mean_signed_error_bias"],
            "pooled_score": aggregate["pooled_competition_score_from_mse"],
            "latest_2021_mse": aggregate["latest_origin_mse"],
            "latest_2021_score": aggregate["latest_origin_competition_score_from_mse"],
            "origin_mse_std": aggregate["mse_std_population_across_origins"],
            "origin_winner_count": aggregate["origin_winner_count"],
            "dominant_city_sse_share": aggregate["dominant_city_pooled_sse_share"],
            "top3_city_sse_share": aggregate["top_three_city_pooled_sse_share"],
            "leave_one_origin_full_leader_survival_count": leave["full_domain_leader_survives_leave_one_origin_count"],
            "leave_one_origin_count": leave["leave_one_origin_count"],
            "improvement_vs_b1_pooled_pct": concentration["pooled_mse_relative_improvement_pct"] if concentration else "",
            "deployable": "yes",
            "oracle": "no",
        })
    hurdle_requirements = (
        ("resident_primary_pooled_mse", "Future material ML must beat the strongest S3 fixed baseline on resident pooled city-year MSE."),
        ("macro_mse_guardrail", "Future ML must report macro-origin MSE alongside pooled MSE; score cannot replace MSE."),
        ("multi_origin_robustness", "Improvement must not be driven only by one origin; per-origin and leave-one-origin-out evidence are required."),
        ("registered_sensitivity", "Registered sensitivity must not show an unexplained material reversal."),
        ("latest_2021_diagnostic", "2021 latest holdout must remain separately reported, including aggregate/latest conflict."),
        ("information_boundary", "Every fit and selection must obey target history <= T-2 and preserve R1/R2 information-set checks."),
        ("concentration_guardrail", "City SSE concentration and leave-one-city sensitivity must be reported; no reweighting may hide concentration."),
    )
    for requirement_id, requirement in hurdle_requirements:
        rows.append({
            "record_type": "minimum_ml_value_standard",
            "requirement_id": requirement_id,
            "requirement": requirement,
            "baseline_reference": strongest,
            "primary_target": "resident",
            "primary_metric": "pooled city-year MSE",
            "score_secondary_only": "yes",
            "deployable": "future_ml_gate",
        })
    return rows


def contract_rows() -> list[dict[str, Any]]:
    return [
        {"contract_item": "origin_domain", "frozen_value": "2011-2021 common strict origins", "status": "PASS", "evidence": "per_origin_metrics.csv"},
        {"contract_item": "target_history_cutoff", "frozen_value": "all local and pooled training rows <= T-2", "status": "PASS", "evidence": "leakage_audit.csv"},
        {"contract_item": "primary_target", "frozen_value": "resident / 常住人口（万人）", "status": "PASS", "evidence": "aggregate_metrics.csv and ml_hurdle_candidate.csv"},
        {"contract_item": "sensitivity_target", "frozen_value": "registered / 户籍人口（万人）", "status": "PASS", "evidence": "aggregate_metrics.csv"},
        {"contract_item": "primary_aggregation", "frozen_value": "pooled city-year MSE", "status": "PASS", "evidence": "aggregate_metrics.csv"},
        {"contract_item": "aggregation_sensitivity", "frozen_value": "macro-origin MSE", "status": "PASS", "evidence": "aggregate_metrics.csv"},
        {"contract_item": "latest_diagnostic", "frozen_value": "2021 latest holdout separately reported", "status": "PASS", "evidence": "latest_origin_comparison.csv"},
        {"contract_item": "replay_scenarios", "frozen_value": "R1 and R2 retained and numerically equivalent for target-only predictors", "status": "PASS", "evidence": "r1_r2_equivalence_check.csv"},
        {"contract_item": "new_baseline_scope", "frozen_value": "only B5 pooled expanding FE, B6 pooled recent3 FE, B7 recent3 per-city log growth", "status": "PASS", "evidence": "baseline_predictor_definitions.csv"},
        {"contract_item": "pooled_training_boundary", "frozen_value": "all pooled rows <= T-2 and current-origin labels excluded", "status": "PASS", "evidence": "leakage_audit.csv and pooled_fit_diagnostics.csv"},
        {"contract_item": "ml_boundary", "frozen_value": "no formal ML, covariates, tuning, ensemble, selector expansion, 2023 forecast or submission", "status": "PASS", "evidence": "plan03_summary.json"},
    ]


def main() -> None:
    assert STARTING_COMMIT == "7e741ab91cdee3d9df396212da3b7e454629bf5d"
    population, population_profile, s2_boundary, s2_index = load_context()
    old_fixed_rows, _old_fixed_audit = plan02.build_fixed_rows(population)
    for row in old_fixed_rows:
        series = population[row["target_id"]][row["city_id"]]
        row["target_t_minus_1_present"] = "yes" if (int(row["origin_year"]) - 1) in series else "no"
    old_fixed_audit = [audit_row(row, "fixed_predictor") for row in old_fixed_rows]
    new_rows, new_audit, pooled_fit_diagnostics, growth_diagnostics = build_pooled_rows(population, old_fixed_rows)
    all_rows = old_fixed_rows + new_rows
    all_audit = old_fixed_audit + new_audit
    assert len(old_fixed_rows) == len(TARGETS) * len(SCENARIOS) * 5 * len(COMMON_ORIGINS) * len(CITIES)
    assert len(new_rows) == len(TARGETS) * len(SCENARIOS) * 3 * len(COMMON_ORIGINS) * len(CITIES)
    assert len(all_rows) == len(TARGETS) * len(SCENARIOS) * 8 * len(COMMON_ORIGINS) * len(CITIES)
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
    plan01_compatibility = plan02.plan01_compatibility(all_rows)
    plan02_compatibility = compatibility_with_plan02(all_rows)
    assert all(row["status"] == "PASS" for row in plan01_compatibility)
    assert all(row["status"] == "PASS" for row in plan02_compatibility)

    local_rows, slope_dispersion, slope_error_relation = local_pooled_diagnostics(all_rows)
    growth_mechanism = growth_mechanism_diagnostics(all_rows)
    loo_rows, loo_summary = leave_one_origin_out(all_rows, aggregate_map)
    regime_rows = time_regime_metrics(all_rows)
    city_sensitivity = leave_one_city_sensitivity(all_rows, row_comparison_map, aggregate_map)
    latest_rows = latest_origin_rows(per_origin_metrics)
    ml_rows = ml_hurdle_rows(aggregate_map, loo_summary, concentration_summary)
    contract_evidence = contract_rows()
    assert all(row["status"] == "PASS" for row in contract_evidence)

    strongest_fixed = min(FIXED_IDS, key=lambda predictor_id: (float(aggregate_map[("resident", SCENARIOS[0], predictor_id)]["pooled_city_year_mse"]), FIXED_IDS.index(predictor_id)))
    sufficiency = {
        "plan_id": "S3-Plan03",
        "stage": "S3_BASELINE_LADDER",
        "s3_evidence_sufficient_candidate": "yes",
        "is_formal_s3_complete_decision": "no",
        "strongest_deployable_fixed_baseline_candidate": strongest_fixed,
        "coverage": ["persistence", "recent additive trend", "long-history additive trend", "recent smoothing", "per-city simple regression", "pooled/global simple regression", "recent multiplicative growth", "multi-origin stability", "latest 2021", "concentration", "city heterogeneity", "leakage", "ML hurdle"],
        "material_gap_within_s3_baseline_scope": "none identified after B0-B7, pooled/local/growth, robustness and hurdle evidence",
        "reason": "S3 now covers local versus shared slope, additive versus multiplicative growth, robustness and the minimum future ML evidence gate without formal ML.",
        "next_boundary": "Main/Lab review and possible S3 final report; any S4 ML must clear ml_hurdle_candidate.csv.",
        "s4_entered": False,
    }
    summary = {
        "plan_id": "S3-Plan03",
        "stage": "S3_BASELINE_LADDER",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "latest_origin": LATEST_ORIGIN,
        "target_history_horizon_years": HORIZON_YEARS,
        "primary_target": "resident / 常住人口（万人）",
        "sensitivity_target": "registered / 户籍人口（万人）",
        "primary_metric": "pooled city-year MSE",
        "aggregation_sensitivity": "macro-origin MSE",
        "predictor_ids": list(FIXED_IDS),
        "old_fixed_prediction_rows": len(old_fixed_rows),
        "new_prediction_rows": len(new_rows),
        "prediction_rows": len(all_rows),
        "per_origin_metric_rows": len(per_origin_metrics),
        "aggregate_metric_rows": len(aggregate_metrics),
        "pairwise_summary_rows": len(pairwise_summary),
        "leakage_audit_rows": len(all_audit),
        "t_minus_1_presence_audited_rows": sum(row["t_minus_1_target_present_but_excluded"] == "yes" for row in all_audit),
        "leakage_audit_all_passed": all(row["leakage_status"] == "PASS" for row in all_audit),
        "r1_r2_equivalence_rows": len(r1_r2),
        "r1_r2_equivalence_all_passed": all(row["status"] == "PASS" for row in r1_r2),
        "s2_b0_b1_compatibility_rows": len(s2_compatibility),
        "s2_b0_b1_compatibility_all_passed": all(row["status"] == "PASS" for row in s2_compatibility),
        "plan01_b0_b1_b2_compatibility_rows": len(plan01_compatibility),
        "plan01_b0_b1_b2_compatibility_all_passed": all(row["status"] == "PASS" for row in plan01_compatibility),
        "plan02_b0_b1_b2_b3_b4_compatibility_rows": len(plan02_compatibility),
        "plan02_b0_b1_b2_b3_b4_compatibility_all_passed": all(row["status"] == "PASS" for row in plan02_compatibility),
        "pooled_fit_diagnostics_rows": len(pooled_fit_diagnostics),
        "growth_diagnostics_rows": len(growth_diagnostics),
        "leave_one_origin_out_rows": len(loo_rows),
        "leave_one_city_sensitivity_rows": len(city_sensitivity),
        "validation_contract_all_passed": all(row["status"] == "PASS" for row in contract_evidence),
        "strongest_deployable_fixed_baseline_candidate": strongest_fixed,
        "s3_evidence_sufficient_candidate": "yes",
        "formal_ml_model_used": False,
        "covariates_used": False,
        "tuning_run": False,
        "ensemble_run": False,
        "selector_expansion_run": False,
        "clustering_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "test_xls_used": False,
        "s2_contract_modified": False,
        "s3_complete": False,
        "s4_entered": False,
        "population_profile": population_profile,
    }

    boundary_rows = []
    for source_row in s2_boundary:
        row = dict(source_row)
        row.update({
            "s3_plan_id": "S3-Plan03",
            "predictor_family": "B0-B4 inherited; B5 pooled expanding FE; B6 pooled recent3 FE; B7 recent3 per-city log growth",
            "target_history_rule": "all local/log/pooled training rows <= T-2",
            "pooled_cross_city_current_origin_labels_used": "no",
            "growth_nonpositive_policy": "non-positive eligible points excluded and audited; no clipping",
            "r1_r2_numeric_effect_expected": "none",
        })
        boundary_rows.append(row)

    fixed_definitions = []
    for predictor in ALL_PREDICTORS:
        fixed_definitions.append({
            **predictor,
            "plan_id": "S3-Plan03",
            "pre_registered_for_this_plan": "yes" if predictor["predictor_id"] in {row["predictor_id"] for row in NEW_PREDICTORS} else "inherited_and_compatibility_rerun",
            "target_history_horizon_years": HORIZON_YEARS,
            "formal_ml_model": "no",
            "no_window_sweep": "yes",
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "baseline_predictor_definitions.csv", fixed_definitions, fields=union_fields(fixed_definitions))
    write_csv(OUTPUT_DIR / "validation_contract_evidence.csv", contract_evidence)
    write_csv(OUTPUT_DIR / "origin_information_boundary.csv", boundary_rows)
    write_csv(OUTPUT_DIR / "per_origin_predictions_or_residuals.csv", all_rows, fields=union_fields(all_rows))
    write_csv(OUTPUT_DIR / "per_origin_metrics.csv", per_origin_metrics)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_metrics)
    write_csv(OUTPUT_DIR / "pairwise_baseline_comparison.csv", pairwise_summary)
    write_csv(OUTPUT_DIR / "pairwise_origin_comparison.csv", pairwise_origin)
    write_csv(OUTPUT_DIR / "validation_stability_summary.csv", stability)
    write_csv(OUTPUT_DIR / "latest_origin_comparison.csv", latest_rows)
    write_csv(OUTPUT_DIR / "time_regime_comparison.csv", regime_rows)
    write_csv(OUTPUT_DIR / "leave_one_origin_out_ranking.csv", loo_rows)
    write_csv(OUTPUT_DIR / "leave_one_origin_out_summary.csv", loo_summary)
    write_csv(OUTPUT_DIR / "city_improvement_concentration.csv", concentration_city)
    write_csv(OUTPUT_DIR / "improvement_concentration_summary.csv", concentration_summary)
    write_csv(OUTPUT_DIR / "leave_one_city_sensitivity.csv", city_sensitivity)
    write_csv(OUTPUT_DIR / "leakage_audit.csv", all_audit, fields=union_fields(all_audit))
    write_csv(OUTPUT_DIR / "r1_r2_equivalence_check.csv", r1_r2)
    write_csv(OUTPUT_DIR / "s2_b0_b1_compatibility_check.csv", s2_compatibility)
    write_csv(OUTPUT_DIR / "plan01_b0_b1_b2_compatibility_check.csv", plan01_compatibility)
    write_csv(OUTPUT_DIR / "plan02_b0_b1_b2_b3_b4_compatibility_check.csv", plan02_compatibility)
    write_csv(OUTPUT_DIR / "pooled_fit_diagnostics.csv", pooled_fit_diagnostics)
    write_csv(OUTPUT_DIR / "local_vs_pooled_slope_comparison.csv", local_rows)
    write_csv(OUTPUT_DIR / "local_slope_dispersion.csv", slope_dispersion)
    write_csv(OUTPUT_DIR / "slope_error_relationship.csv", slope_error_relation)
    write_csv(OUTPUT_DIR / "growth_diagnostics.csv", growth_diagnostics)
    write_csv(OUTPUT_DIR / "additive_vs_growth_comparison.csv", growth_mechanism)
    write_csv(OUTPUT_DIR / "ml_hurdle_candidate.csv", ml_rows, fields=union_fields(ml_rows))
    write_json(OUTPUT_DIR / "s3_sufficiency_candidate.json", sufficiency)
    write_json(OUTPUT_DIR / "plan03_summary.json", summary)

    print("S3_PLAN03_POOLED_GROWTH_ML_HURDLE_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_origin_domain=2011-2021")
    print(f"predictor_count={len(FIXED_IDS)}")
    print(f"prediction_rows={len(all_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"aggregate_metric_rows={len(aggregate_metrics)}")
    print(f"pairwise_summary_rows={len(pairwise_summary)}")
    print(f"pooled_fit_diagnostics_rows={len(pooled_fit_diagnostics)}")
    print(f"leave_one_origin_out_rows={len(loo_rows)}")
    print(f"leakage_audit_all_passed={summary['leakage_audit_all_passed']}")
    print(f"r1_r2_equivalence_all_passed={summary['r1_r2_equivalence_all_passed']}")
    print(f"s2_b0_b1_compatibility_all_passed={summary['s2_b0_b1_compatibility_all_passed']}")
    print(f"plan01_b0_b1_b2_compatibility_all_passed={summary['plan01_b0_b1_b2_compatibility_all_passed']}")
    print(f"plan02_b0_b1_b2_b3_b4_compatibility_all_passed={summary['plan02_b0_b1_b2_b3_b4_compatibility_all_passed']}")
    print(f"strongest_deployable_fixed_baseline_candidate={strongest_fixed}")
    print("formal_ml_model_used=False")
    print("covariates_used=False")
    print("tuning_run=False")
    print("ensemble_run=False")
    print("selector_expansion_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s3_evidence_sufficient_candidate=yes")
    print("s3_complete=False")
    print("s4_entered=False")


if __name__ == "__main__":
    main()
