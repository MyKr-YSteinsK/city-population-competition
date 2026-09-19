"""Screen pre-origin structure in frozen B3 residual risk.

This is a diagnostic-only S5 Plan01 runner.  It reuses the frozen S3
per-origin B1/B2/B3 predictions and reconstructs only target-history
diagnostics from canonical population observations at the T-2 boundary.
It does not fit a new model, search thresholds, or create a routing rule.
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


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan01"
BASE_SHA = "04e5cd469c45468a60b1f9b65a96e76311d5326b"
COMMON_ORIGINS = tuple(range(2011, 2022))
EARLY_ORIGINS = tuple(range(2011, 2016))
MIDDLE_ORIGINS = tuple(range(2016, 2019))
LATE_ORIGINS = tuple(range(2017, 2022))
REGIMES = {
    "early_2011_2015": EARLY_ORIGINS,
    "middle_2016_2018": MIDDLE_ORIGINS,
    "late_2019_2021": LATE_ORIGINS,
}
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
CITIES = tuple(f"city{i}" for i in range(1, 41))
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
PREDICTOR_IDS = {
    "b1": "B1_TWO_POINT_LINEAR_TREND",
    "b2": "B2_EXPANDING_PER_CITY_LINEAR_TREND",
    "b3": B3_ID,
}
TARGET_COLUMNS = {
    "resident": "常住人口（万人）",
    "registered": "户籍人口（万人）",
}
DIAGNOSTICS = (
    (
        "slope_disagreement_abs",
        "absolute gap between B3 recent-3 local slope and B2 expanding local slope",
    ),
    (
        "baseline_prediction_spread",
        "max minus min prediction among frozen B1, B2 and B3",
    ),
    (
        "history_difference_volatility",
        "population standard deviation of all available historical first differences",
    ),
    (
        "recent_acceleration_abs",
        "absolute change from the preceding first difference to the latest difference",
    ),
    (
        "preorigin_target_scale",
        "latest allowed target level before the pseudo-origin",
    ),
)
DIAGNOSTIC_IDS = tuple(item[0] for item in DIAGNOSTICS)
DIAGNOSTIC_DESCRIPTIONS = dict(DIAGNOSTICS)
TOP_QUARTILE_FRACTION = 0.25
TOP_QUARTILE_MIN_COUNT = 1
PRIMARY_MIN_MEDIAN_ABS_SPEARMAN = 0.25
PRIMARY_MIN_DIRECTION_ORIGINS = 8
PRIMARY_MIN_HIGH_RISK_RATIO = 1.50
PRIMARY_MIN_HIGHER_ERROR_ORIGINS = 8
REGISTERED_STRONG_REVERSAL_RHO = -0.10
REGISTERED_STRONG_REVERSAL_MAX_HIGHER_ERROR_ORIGINS = 3

S3_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
PREDICTION_PATH = S3_DIR / "per_origin_predictions_or_residuals.csv"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan01"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AssertionError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int:
    return int(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def population_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def rank_average(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        value = ordered[index][1]
        while end < len(ordered) and ordered[end][1] == value:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss == 0 or right_ss == 0:
        return 0.0
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    return numerator / math.sqrt(left_ss * right_ss)


def spearman(values: list[float], outcomes: list[float]) -> float | None:
    if len(values) != len(outcomes) or len(values) < 2:
        return None
    return pearson(rank_average(values), rank_average(outcomes))


def json_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_population() -> dict[tuple[str, str, int], float]:
    population: dict[tuple[str, str, int], float] = {}
    for row in read_csv(POPULATION_PATH):
        city = row["城市名称"].strip()
        year = int(row["年份"])
        for target_id, column in TARGET_COLUMNS.items():
            value = as_float(row.get(column))
            if value is not None:
                key = (target_id, city, year)
                if key in population:
                    raise AssertionError(f"duplicate population key: {key}")
                population[key] = value
    return population


def collect_prediction_rows() -> dict[tuple[str, str, int, str], dict[str, dict[str, str]]]:
    rows = read_csv(PREDICTION_PATH)
    grouped: dict[tuple[str, str, int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    allowed_predictors = set(PREDICTOR_IDS.values())
    for row in rows:
        target_id = row["target_id"]
        scenario_id = row["scenario_id"]
        origin_year = as_int(row["origin_year"])
        predictor_id = row["predictor_id"]
        city_id = row["city_id"]
        if (
            target_id not in TARGETS
            or scenario_id not in SCENARIOS
            or origin_year not in COMMON_ORIGINS
            or city_id not in CITIES
            or predictor_id not in allowed_predictors
        ):
            continue
        if row["scoreable"].lower() != "yes":
            raise AssertionError(f"unexpected non-scoreable frozen row: {row}")
        key = (target_id, scenario_id, origin_year, city_id)
        if predictor_id in grouped[key]:
            raise AssertionError(f"duplicate frozen predictor row: {key}, {predictor_id}")
        grouped[key][predictor_id] = row
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    if len(grouped) != expected:
        raise AssertionError(f"expected {expected} city-origin groups, got {len(grouped)}")
    for key, predictors in grouped.items():
        if set(predictors) != allowed_predictors:
            raise AssertionError(f"incomplete B1/B2/B3 group: {key}, {set(predictors)}")
    return grouped


def history_for(
    population: dict[tuple[str, str, int], float],
    target_id: str,
    city_id: str,
    cutoff_year: int,
) -> list[tuple[int, float]]:
    return [
        (year, population[(target_id, city_id, year)])
        for year in sorted(
            year
            for target, city, year in population
            if target == target_id and city == city_id and year <= cutoff_year
        )
    ]


def build_records(
    population: dict[tuple[str, str, int], float],
    grouped: dict[tuple[str, str, int, str], dict[str, dict[str, str]]],
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    boundary_violation_count = 0
    for (target_id, scenario_id, origin_year, city_id), predictors in sorted(grouped.items()):
        b1 = predictors[PREDICTOR_IDS["b1"]]
        b2 = predictors[PREDICTOR_IDS["b2"]]
        b3 = predictors[PREDICTOR_IDS["b3"]]
        cutoff_year = origin_year - 2
        history = history_for(population, target_id, city_id, cutoff_year)
        if len(history) < 3:
            raise AssertionError(f"insufficient history for B3 group: {target_id}, {city_id}, {origin_year}")
        if history[-1][0] > cutoff_year:
            boundary_violation_count += 1
        actual = as_float(b3["actual_target_value_for_scoring"])
        predictions = {
            "b1": as_float(b1["prediction_value"]),
            "b2": as_float(b2["prediction_value"]),
            "b3": as_float(b3["prediction_value"]),
        }
        if actual is None or any(value is None for value in predictions.values()):
            raise AssertionError(f"missing frozen outcome or prediction: {target_id}, {city_id}, {origin_year}")
        differences = [
            history[index][1] - history[index - 1][1]
            for index in range(1, len(history))
        ]
        b1_slope = slope(history[-2:])
        b2_slope = slope(history)
        b3_slope = slope(history[-3:])
        if b1_slope is None or b2_slope is None or b3_slope is None:
            raise AssertionError(f"missing slope diagnostic: {target_id}, {city_id}, {origin_year}")
        diagnostics = {
            "slope_disagreement_abs": abs(b3_slope - b2_slope),
            "baseline_prediction_spread": max(predictions.values()) - min(predictions.values()),
            "history_difference_volatility": population_std(differences),
            "recent_acceleration_abs": abs(differences[-1] - differences[-2]) if len(differences) >= 2 else None,
            "preorigin_target_scale": history[-1][1],
        }
        if any(value is None for value in diagnostics.values()):
            raise AssertionError(f"missing compact diagnostic: {target_id}, {city_id}, {origin_year}")
        squared_errors = {
            predictor: (prediction - actual) ** 2
            for predictor, prediction in predictions.items()
        }
        source_b3_squared_error = as_float(b3["squared_error"])
        if source_b3_squared_error is None or abs(source_b3_squared_error - squared_errors["b3"]) > 1e-8:
            raise AssertionError(f"B3 row squared-error mismatch: {target_id}, {city_id}, {origin_year}")
        records.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": origin_year,
                "city_id": city_id,
                "target_history_cutoff_year": cutoff_year,
                "history_max_year_used": history[-1][0],
                "history_observation_count": len(history),
                "actual_target_value": actual,
                "b1_prediction": predictions["b1"],
                "b2_prediction": predictions["b2"],
                "b3_prediction": predictions["b3"],
                "b1_squared_error": squared_errors["b1"],
                "b2_squared_error": squared_errors["b2"],
                "b3_squared_error": squared_errors["b3"],
                "b3_residual_prediction_minus_actual": predictions["b3"] - actual,
                "b1_slope": b1_slope,
                "b2_slope": b2_slope,
                "b3_slope": b3_slope,
                **diagnostics,
            }
        )
    return records, boundary_violation_count


def group_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    return mean(float(row[field]) for row in rows)


def b3_compatibility(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "resident": 872.452651515143,
        "registered": 225.421212121213,
    }
    observed: dict[str, dict[str, float]] = {}
    checks: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            rows = [
                row
                for row in records
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            by_origin = {
                origin: [row for row in rows if row["origin_year"] == origin]
                for origin in COMMON_ORIGINS
            }
            pooled = group_mean(rows, "b3_squared_error")
            macro = mean(
                group_mean(origin_rows, "b3_squared_error")
                for origin_rows in by_origin.values()
            )
            latest = group_mean(by_origin[2021], "b3_squared_error")
            if pooled is None or macro is None or latest is None:
                raise AssertionError(f"missing B3 aggregate: {target_id}, {scenario_id}")
            observed[f"{target_id}_{scenario_id}"] = {
                "pooled_mse": pooled,
                "macro_origin_mse": macro,
                "latest_2021_mse": latest,
                "scoreable_rows": len(rows),
            }
            checks.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "pooled_delta_vs_frozen": pooled - expected[target_id],
                    "macro_delta_vs_frozen": macro - expected[target_id],
                    "latest_2021_mse": latest,
                    "scoreable_rows": len(rows),
                    "status": "PASS"
                    if len(rows) == 440 and abs(pooled - expected[target_id]) <= 1e-9
                    else "FAIL",
                }
            )
    return {
        "expected_pooled_mse": expected,
        "observed": observed,
        "checks": checks,
        "pass": all(item["status"] == "PASS" for item in checks),
    }


def high_risk_rows(rows: list[dict[str, Any]], diagnostic_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = [row for row in rows if as_float(row[diagnostic_id]) is not None]
    ordered = sorted(
        available,
        key=lambda row: (-float(row[diagnostic_id]), row["city_id"]),
    )
    high_count = max(TOP_QUARTILE_MIN_COUNT, math.ceil(len(ordered) * TOP_QUARTILE_FRACTION))
    return ordered[:high_count], ordered[high_count:]


def tradeoff_row(
    rows: list[dict[str, Any]],
    target_id: str,
    scenario_id: str,
    diagnostic_id: str,
    risk_stratum: str,
    origin_year: int | str,
) -> dict[str, Any]:
    if origin_year == "ALL":
        selected: list[dict[str, Any]] = []
        for origin in COMMON_ORIGINS:
            origin_rows = [row for row in rows if row["origin_year"] == origin]
            high, low = high_risk_rows(origin_rows, diagnostic_id)
            selected.extend(high if risk_stratum == "high_risk_top_quartile" else low)
    else:
        selected = [row for row in rows if row["origin_year"] == origin_year]
        if risk_stratum == "high_risk_top_quartile":
            selected, _ = high_risk_rows(selected, diagnostic_id)
        else:
            _, selected = high_risk_rows(selected, diagnostic_id)
    b1 = group_mean(selected, "b1_squared_error")
    b2 = group_mean(selected, "b2_squared_error")
    b3 = group_mean(selected, "b3_squared_error")
    return {
        "target_id": target_id,
        "scenario_id": scenario_id,
        "diagnostic_id": diagnostic_id,
        "risk_stratum": risk_stratum,
        "origin_year": origin_year,
        "city_count": len(selected) if origin_year != "ALL" else len({row["city_id"] for row in selected}),
        "row_count": len(selected),
        "b1_mse": b1,
        "b2_mse": b2,
        "b3_mse": b3,
        "b1_minus_b3_mse_delta": None if b1 is None or b3 is None else b3 - b1,
        "b2_minus_b3_mse_delta": None if b2 is None or b3 is None else b3 - b2,
        "b1_beats_b3": None if b1 is None or b3 is None else b1 < b3,
        "b2_beats_b3": None if b2 is None or b3 is None else b2 < b3,
    }


def association_summary(
    records: list[dict[str, Any]],
    target_id: str,
    scenario_id: str,
    diagnostic_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        row
        for row in records
        if row["target_id"] == target_id and row["scenario_id"] == scenario_id
    ]
    origin_stats: list[dict[str, Any]] = []
    tradeoffs: list[dict[str, Any]] = []
    for origin_year in COMMON_ORIGINS:
        origin_rows = [row for row in rows if row["origin_year"] == origin_year]
        available = [row for row in origin_rows if as_float(row[diagnostic_id]) is not None]
        values = [float(row[diagnostic_id]) for row in available]
        errors = [float(row["b3_squared_error"]) for row in available]
        rho = spearman(values, errors)
        high, low = high_risk_rows(origin_rows, diagnostic_id)
        high_mse = group_mean(high, "b3_squared_error")
        low_mse = group_mean(low, "b3_squared_error")
        high_ratio = safe_ratio(high_mse, low_mse)
        high_worse = high_mse is not None and low_mse is not None and high_mse > low_mse
        origin_stats.append(
            {
                "origin_year": origin_year,
                "available_city_count": len(available),
                "high_risk_city_count": len(high),
                "remaining_city_count": len(low),
                "spearman_b3_squared_error": rho,
                "high_risk_b3_mse": high_mse,
                "remaining_b3_mse": low_mse,
                "high_to_remaining_b3_mse_ratio": high_ratio,
                "high_risk_worse": high_worse,
            }
        )
        tradeoffs.extend(
            [
                tradeoff_row(
                    origin_rows,
                    target_id,
                    scenario_id,
                    diagnostic_id,
                    "high_risk_top_quartile",
                    origin_year,
                ),
                tradeoff_row(
                    origin_rows,
                    target_id,
                    scenario_id,
                    diagnostic_id,
                    "remaining_rows",
                    origin_year,
                ),
            ]
        )
    rho_values = [
        float(item["spearman_b3_squared_error"])
        for item in origin_stats
        if item["spearman_b3_squared_error"] is not None
    ]
    ratios = [
        float(item["high_to_remaining_b3_mse_ratio"])
        for item in origin_stats
        if item["high_to_remaining_b3_mse_ratio"] is not None
    ]
    median_rho = statistics.median(rho_values) if rho_values else None
    median_abs_rho = statistics.median(abs(value) for value in rho_values) if rho_values else None
    positive_count = sum(value > 0 for value in rho_values)
    negative_count = sum(value < 0 for value in rho_values)
    high_worse_count = sum(bool(item["high_risk_worse"]) for item in origin_stats)
    high_all, low_all = high_risk_rows(rows, diagnostic_id)
    high_all_mse = group_mean(high_all, "b3_squared_error")
    low_all_mse = group_mean(low_all, "b3_squared_error")
    high_all_ratio = safe_ratio(high_all_mse, low_all_mse)
    primary_risk_gate_pass = (
        target_id == "resident"
        and median_abs_rho is not None
        and median_abs_rho >= PRIMARY_MIN_MEDIAN_ABS_SPEARMAN
        and positive_count >= PRIMARY_MIN_DIRECTION_ORIGINS
        and high_all_ratio is not None
        and high_all_ratio >= PRIMARY_MIN_HIGH_RISK_RATIO
        and high_worse_count >= PRIMARY_MIN_HIGHER_ERROR_ORIGINS
    )
    registered_not_reversed = not (
        median_rho is not None
        and median_rho < REGISTERED_STRONG_REVERSAL_RHO
        and high_worse_count <= REGISTERED_STRONG_REVERSAL_MAX_HIGHER_ERROR_ORIGINS
    )
    tradeoffs.append(
        tradeoff_row(
            rows,
            target_id,
            scenario_id,
            diagnostic_id,
            "high_risk_top_quartile",
            "ALL",
        )
    )
    tradeoffs.append(
        tradeoff_row(
            rows,
            target_id,
            scenario_id,
            diagnostic_id,
            "remaining_rows",
            "ALL",
        )
    )
    summary = {
        "diagnostic_id": diagnostic_id,
        "description": DIAGNOSTIC_DESCRIPTIONS[diagnostic_id],
        "target_id": target_id,
        "scenario_id": scenario_id,
        "origin_count": len(origin_stats),
        "available_city_origin_rows": len(rows),
        "pooled_high_risk_b3_mse": high_all_mse,
        "pooled_remaining_b3_mse": low_all_mse,
        "pooled_high_to_remaining_b3_mse_ratio": high_all_ratio,
        "median_spearman": median_rho,
        "median_abs_spearman": median_abs_rho,
        "positive_direction_origins": positive_count,
        "negative_direction_origins": negative_count,
        "high_risk_worse_origin_count": high_worse_count,
        "primary_resident_risk_gate_pass": primary_risk_gate_pass,
        "registered_not_materially_contradictory": registered_not_reversed,
        "per_origin_spearman": {str(item["origin_year"]): item["spearman_b3_squared_error"] for item in origin_stats},
        "per_origin_high_to_remaining_ratio": {
            str(item["origin_year"]): item["high_to_remaining_b3_mse_ratio"] for item in origin_stats
        },
        "per_origin_high_risk_worse": {str(item["origin_year"]): item["high_risk_worse"] for item in origin_stats},
    }
    return summary, origin_stats, tradeoffs


def add_risk_screen_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_id": summary["diagnostic_id"],
        "diagnostic_description": summary["description"],
        "target_id": summary["target_id"],
        "scenario_id": summary["scenario_id"],
        "origin_count": summary["origin_count"],
        "available_city_origin_rows": summary["available_city_origin_rows"],
        "high_risk_definition": "within-origin top ceil(25%) by diagnostic, ties broken by city_id",
        "pooled_high_risk_b3_mse": summary["pooled_high_risk_b3_mse"],
        "pooled_remaining_b3_mse": summary["pooled_remaining_b3_mse"],
        "pooled_high_to_remaining_b3_mse_ratio": summary["pooled_high_to_remaining_b3_mse_ratio"],
        "median_spearman": summary["median_spearman"],
        "median_abs_spearman": summary["median_abs_spearman"],
        "positive_direction_origins": summary["positive_direction_origins"],
        "negative_direction_origins": summary["negative_direction_origins"],
        "high_risk_worse_origin_count": summary["high_risk_worse_origin_count"],
        "primary_resident_risk_gate_pass": summary["primary_resident_risk_gate_pass"],
        "registered_not_materially_contradictory": summary["registered_not_materially_contradictory"],
        "per_origin_spearman_json": json.dumps(summary["per_origin_spearman"], ensure_ascii=False, sort_keys=True),
        "per_origin_ratio_json": json.dumps(
            summary["per_origin_high_to_remaining_ratio"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "per_origin_high_risk_worse_json": json.dumps(
            summary["per_origin_high_risk_worse"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def build_city_concentration(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            rows = [
                row
                for row in records
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_city[row["city_id"]].append(row)
            total_sse = sum(float(row["b3_squared_error"]) for row in rows)
            city_stats: list[dict[str, Any]] = []
            for city_id in CITIES:
                city_rows = by_city[city_id]
                cumulative_sse = sum(float(row["b3_squared_error"]) for row in city_rows)
                top_error_count = 0
                slope_high_count = 0
                volatility_high_count = 0
                overlap_count = 0
                for origin_year in COMMON_ORIGINS:
                    origin_rows = [row for row in rows if row["origin_year"] == origin_year]
                    error_order = sorted(
                        origin_rows,
                        key=lambda row: (-float(row["b3_squared_error"]), row["city_id"]),
                    )
                    top_error_cities = {
                        row["city_id"]
                        for row in error_order[: max(1, math.ceil(len(error_order) * TOP_QUARTILE_FRACTION))]
                    }
                    city_row = next(row for row in origin_rows if row["city_id"] == city_id)
                    if city_id in top_error_cities:
                        top_error_count += 1
                    slope_high, _ = high_risk_rows(origin_rows, "slope_disagreement_abs")
                    volatility_high, _ = high_risk_rows(origin_rows, "history_difference_volatility")
                    slope_cities = {row["city_id"] for row in slope_high}
                    volatility_cities = {row["city_id"] for row in volatility_high}
                    if city_id in slope_cities:
                        slope_high_count += 1
                    if city_id in volatility_cities:
                        volatility_high_count += 1
                    if city_id in top_error_cities and (
                        city_id in slope_cities or city_id in volatility_cities
                    ):
                        overlap_count += 1
                city_stats.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "city_id": city_id,
                        "cumulative_b3_sse": cumulative_sse,
                        "cumulative_b3_sse_share": cumulative_sse / total_sse if total_sse else None,
                        "top_error_quartile_origin_count": top_error_count,
                        "slope_disagreement_high_risk_origin_count": slope_high_count,
                        "volatility_high_risk_origin_count": volatility_high_count,
                        "top_error_and_preorigin_high_risk_overlap_count": overlap_count,
                        "preorigin_high_risk_overlap_share_of_top_error_origins": (
                            overlap_count / top_error_count if top_error_count else None
                        ),
                    }
                )
            city_stats.sort(key=lambda row: (-row["cumulative_b3_sse"], row["city_id"]))
            for rank, row in enumerate(city_stats, start=1):
                row["cumulative_sse_rank"] = rank
                row["is_top3_cumulative_sse_city"] = rank <= 3
                output.append(row)
            top3 = city_stats[:3]
            summaries[f"{target_id}_{scenario_id}"] = {
                "total_b3_sse": total_sse,
                "top1_city_id": city_stats[0]["city_id"],
                "top1_sse_share": city_stats[0]["cumulative_b3_sse_share"],
                "top3_city_ids": [row["city_id"] for row in top3],
                "top3_sse_share": sum(row["cumulative_b3_sse_share"] for row in top3),
                "top_error_quartile_recurrence_max": max(
                    row["top_error_quartile_origin_count"] for row in city_stats
                ),
                "top3_preorigin_overlap_shares": {
                    row["city_id"]: row["preorigin_high_risk_overlap_share_of_top_error_origins"]
                    for row in top3
                },
                "routing_rule_created": False,
            }
    return output, summaries


def regime_diagnostics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            rows = [
                row
                for row in records
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            for diagnostic_id in DIAGNOSTIC_IDS:
                key = f"{target_id}_{scenario_id}_{diagnostic_id}"
                output[key] = {}
                for regime_id, origins in REGIMES.items():
                    regime_rows = [row for row in rows if row["origin_year"] in origins]
                    output[key][regime_id] = {
                        "origin_domain": f"{min(origins)}-{max(origins)}",
                        "origin_count": len(set(row["origin_year"] for row in regime_rows)),
                        "mean_diagnostic": group_mean(regime_rows, diagnostic_id),
                        "median_diagnostic": (
                            statistics.median(float(row[diagnostic_id]) for row in regime_rows)
                            if regime_rows
                            else None
                        ),
                        "pooled_b3_mse": group_mean(regime_rows, "b3_squared_error"),
                    }
    return output


def tradeoff_decision(
    tradeoffs_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
    resident_summaries: dict[tuple[str, str], dict[str, Any]],
    registered_summaries: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def origin_rows(target: str, scenario: str, diagnostic: str, stratum: str) -> list[dict[str, Any]]:
        rows = tradeoffs_by_key[(target, scenario, diagnostic)]
        return [
            row
            for row in rows
            if row["risk_stratum"] == stratum and row["origin_year"] != "ALL"
        ]

    slope_id = "slope_disagreement_abs"
    volatility_id = "history_difference_volatility"
    slope_gate_both = all(
        resident_summaries[(slope_id, scenario)]["primary_resident_risk_gate_pass"]
        for scenario in SCENARIOS
    )
    volatility_gate_both = all(
        resident_summaries[(volatility_id, scenario)]["primary_resident_risk_gate_pass"]
        for scenario in SCENARIOS
    )
    registered_not_reversed = all(
        registered_summaries[(slope_id, scenario)]["registered_not_materially_contradictory"]
        for scenario in SCENARIOS
    )
    shrinkage_details: dict[str, Any] = {}
    blend_details: dict[str, Any] = {}
    for scenario in SCENARIOS:
        high = origin_rows("resident", scenario, slope_id, "high_risk_top_quartile")
        low = origin_rows("resident", scenario, slope_id, "remaining_rows")
        high_comparator_gain = sum(
            bool(row["b1_beats_b3"] or row["b2_beats_b3"]) for row in high
        )
        low_b3_advantage = sum(
            bool(
                row["b3_mse"] is not None
                and row["b1_mse"] is not None
                and row["b2_mse"] is not None
                and row["b3_mse"] < row["b1_mse"]
                and row["b3_mse"] < row["b2_mse"]
            )
            for row in low
        )
        registered_high = origin_rows("registered", scenario, slope_id, "high_risk_top_quartile")
        registered_comparator_gain = sum(
            bool(row["b1_beats_b3"] or row["b2_beats_b3"]) for row in registered_high
        )
        shrinkage_details[scenario] = {
            "resident_high_risk_comparator_gain_origins": high_comparator_gain,
            "registered_high_risk_comparator_gain_origins": registered_comparator_gain,
            "resident_low_risk_b3_advantage_origins": low_b3_advantage,
        }
        blend_details[scenario] = {
            "resident_high_risk_comparator_gain_origins": high_comparator_gain,
            "resident_low_risk_b3_advantage_origins": low_b3_advantage,
        }
    shrinkage_pass = (
        slope_gate_both
        and registered_not_reversed
        and all(
            details["resident_high_risk_comparator_gain_origins"] >= 7
            for details in shrinkage_details.values()
        )
        and all(
            details["registered_high_risk_comparator_gain_origins"] >= 4
            for details in shrinkage_details.values()
        )
    )
    blend_pass = (
        slope_gate_both
        and registered_not_reversed
        and all(
            details["resident_high_risk_comparator_gain_origins"] >= 7
            and details["resident_low_risk_b3_advantage_origins"] >= 7
            for details in blend_details.values()
        )
    )
    robust_pass = volatility_gate_both and all(
        resident_summaries[(volatility_id, scenario)]["primary_resident_risk_gate_pass"]
        for scenario in SCENARIOS
    )
    decisions = {
        "partial_pooling_shrinkage": {
            "risk_gate_pass": slope_gate_both,
            "decision_relevance_pass": shrinkage_pass,
            "registered_not_materially_reversed": registered_not_reversed,
            "details": shrinkage_details,
        },
        "fixed_low_capacity_blend": {
            "risk_gate_pass": slope_gate_both,
            "decision_relevance_pass": blend_pass,
            "registered_not_materially_reversed": registered_not_reversed,
            "details": blend_details,
        },
        "robust_local_trend": {
            "risk_gate_pass": volatility_gate_both,
            "decision_relevance_pass": robust_pass,
            "registered_not_materially_reversed": all(
                registered_summaries[(volatility_id, scenario)][
                    "registered_not_materially_contradictory"
                ]
                for scenario in SCENARIOS
            ),
            "details": {
                "diagnostic_id": volatility_id,
                "reason": "risk gate itself requires concentrated B3 excess error in the high-instability stratum",
            },
        },
    }
    active = [
        direction
        for direction, detail in decisions.items()
        if detail["risk_gate_pass"] and detail["decision_relevance_pass"]
    ]
    return {
        "direction_decisions": decisions,
        "activated_directions": active,
        "risk_gate_definition": {
            "median_abs_spearman_min": PRIMARY_MIN_MEDIAN_ABS_SPEARMAN,
            "positive_direction_origins_min": PRIMARY_MIN_DIRECTION_ORIGINS,
            "high_to_remaining_mse_ratio_min": PRIMARY_MIN_HIGH_RISK_RATIO,
            "high_risk_worse_origins_min": PRIMARY_MIN_HIGHER_ERROR_ORIGINS,
            "registered_strong_reversal_rule": (
                f"median Spearman < {REGISTERED_STRONG_REVERSAL_RHO} "
                f"and high-risk-worse origins <= {REGISTERED_STRONG_REVERSAL_MAX_HIGHER_ERROR_ORIGINS}"
            ),
        },
    }, active


def hypothesis_rows(
    decision: dict[str, Any],
    concentration: dict[str, Any],
    resident_summaries: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    directions = decision["direction_decisions"]
    rows = [
        {
            "hypothesis_id": "partial_pooling_shrinkage",
            "supporting_evidence": "slope disagreement high-risk stratum and B1/B2 tradeoff gates",
            "contradicting_evidence": "slope-risk gate passed, but comparator decision relevance was insufficient"
            if not directions["partial_pooling_shrinkage"]["decision_relevance_pass"]
            else "",
            "risk_gate_status": directions["partial_pooling_shrinkage"]["risk_gate_pass"],
            "decision_relevance_status": directions["partial_pooling_shrinkage"]["decision_relevance_pass"],
            "activation_recommendation": "active_candidate_direction"
            if directions["partial_pooling_shrinkage"]["decision_relevance_pass"]
            else "inactive",
        },
        {
            "hypothesis_id": "robust_local_trend",
            "supporting_evidence": "historical difference volatility risk gate and concentrated B3 excess error",
            "contradicting_evidence": "volatility did not pass the pre-registered resident risk gate"
            if not directions["robust_local_trend"]["risk_gate_pass"]
            else "",
            "risk_gate_status": directions["robust_local_trend"]["risk_gate_pass"],
            "decision_relevance_status": directions["robust_local_trend"]["decision_relevance_pass"],
            "activation_recommendation": "active_candidate_direction"
            if directions["robust_local_trend"]["decision_relevance_pass"]
            else "inactive",
        },
        {
            "hypothesis_id": "fixed_low_capacity_blend",
            "supporting_evidence": "slope-disagreement stratum separates comparator gain from B3 advantage",
            "contradicting_evidence": "high-risk comparator gain occurred in 6/11 origins and low-risk B3 advantage in 5/11, below the 7/11 rule"
            if not directions["fixed_low_capacity_blend"]["decision_relevance_pass"]
            else "",
            "risk_gate_status": directions["fixed_low_capacity_blend"]["risk_gate_pass"],
            "decision_relevance_status": directions["fixed_low_capacity_blend"]["decision_relevance_pass"],
            "activation_recommendation": "active_candidate_direction"
            if directions["fixed_low_capacity_blend"]["decision_relevance_pass"]
            else "inactive",
        },
        {
            "hypothesis_id": "stronger_boosting",
            "supporting_evidence": "none from this diagnostic-only plan",
            "contradicting_evidence": "S4 low-capacity HGB and multisource evidence already failed; no direct capacity signal here",
            "risk_gate_status": "not_screened",
            "decision_relevance_status": "not_screened",
            "activation_recommendation": "inactive",
        },
        {
            "hypothesis_id": "remaining_source_groups",
            "supporting_evidence": "none; no new source is in scope",
            "contradicting_evidence": "S4 external-source and multisource evidence did not establish stable gain",
            "risk_gate_status": "not_screened",
            "decision_relevance_status": "not_screened",
            "activation_recommendation": "inactive",
        },
        {
            "hypothesis_id": "tuning",
            "supporting_evidence": "none; threshold and parameter search are forbidden here",
            "contradicting_evidence": "no mechanism signal justifies tuning",
            "risk_gate_status": "not_screened",
            "decision_relevance_status": "not_screened",
            "activation_recommendation": "inactive",
        },
    ]
    return rows


def main() -> None:
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(PREDICTION_PATH)
    if not POPULATION_PATH.is_file():
        raise FileNotFoundError(POPULATION_PATH)
    population = load_population()
    grouped = collect_prediction_rows()
    records, boundary_violation_count = build_records(population, grouped)
    expected_rows = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    if len(records) != expected_rows:
        raise AssertionError(f"expected {expected_rows} diagnostic rows, got {len(records)}")
    compatibility = b3_compatibility(records)
    if not compatibility["pass"]:
        raise AssertionError(f"frozen B3 compatibility failed: {compatibility}")

    risk_rows: list[dict[str, Any]] = []
    origin_association_rows: list[dict[str, Any]] = []
    tradeoff_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    resident_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    registered_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    tradeoffs_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for diagnostic_id in DIAGNOSTIC_IDS:
                summary, origin_stats, tradeoffs = association_summary(
                    records,
                    target_id,
                    scenario_id,
                    diagnostic_id,
                )
                summaries[(target_id, scenario_id, diagnostic_id)] = summary
                if target_id == "resident":
                    resident_summaries[(diagnostic_id, scenario_id)] = summary
                else:
                    registered_summaries[(diagnostic_id, scenario_id)] = summary
                risk_rows.append(add_risk_screen_row(summary))
                for item in origin_stats:
                    origin_association_rows.append(
                        {
                            "diagnostic_id": diagnostic_id,
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            **item,
                        }
                    )
                tradeoff_rows.extend(tradeoffs)
                tradeoffs_by_key[(target_id, scenario_id, diagnostic_id)] = tradeoffs

    decision, active_directions = tradeoff_decision(
        tradeoffs_by_key,
        resident_summaries,
        registered_summaries,
    )
    concentration_rows, concentration_summary = build_city_concentration(records)
    regime_summary = regime_diagnostics(records)
    hypothesis_matrix = hypothesis_rows(decision, concentration_summary, resident_summaries)
    if active_directions:
        screening_decision = "mechanism_signal_found:" + ",".join(active_directions)
    else:
        screening_decision = "no_actionable_structure"

    late_mechanism_support = []
    for diagnostic_id in DIAGNOSTIC_IDS:
        for scenario_id in SCENARIOS:
            resident = resident_summaries[(diagnostic_id, scenario_id)]
            early = regime_summary[
                f"resident_{scenario_id}_{diagnostic_id}"
            ]["early_2011_2015"]["mean_diagnostic"]
            late = regime_summary[
                f"resident_{scenario_id}_{diagnostic_id}"
            ]["late_2019_2021"]["mean_diagnostic"]
            late_mechanism_support.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "scenario_id": scenario_id,
                    "late_mean_minus_early_mean": None if early is None or late is None else late - early,
                    "resident_primary_risk_gate_pass": resident["primary_resident_risk_gate_pass"],
                }
            )

    summary_json = {
        "plan_id": PLAN_ID,
        "stage": "S5_EVIDENCE_DRIVEN_ITERATION",
        "base_sha": BASE_SHA,
        "worker_branch": current_branch(),
        "target_ids": list(TARGETS),
        "scenario_ids": list(SCENARIOS),
        "common_origins": list(COMMON_ORIGINS),
        "city_count": len(CITIES),
        "diagnostic_row_count": len(records),
        "scoreable_rows_per_target_scenario": 440,
        "target_history_cutoff_rule": "all diagnostic source observations <= origin_year - 2",
        "diagnostics_screened": [
            {
                "diagnostic_id": diagnostic_id,
                "description": description,
                "source": "canonical population history only",
            }
            for diagnostic_id, description in DIAGNOSTICS
        ],
        "frozen_b3_compatibility": compatibility,
        "information_boundary": {
            "max_history_boundary_violation_count": boundary_violation_count,
            "outcome_used_only_for_b3_squared_error": True,
            "new_external_source_used": False,
            "test_xls_used": False,
            "threshold_search": False,
            "new_model_fit": False,
            "routing_rule_created": False,
        },
        "risk_gate_rules": decision["risk_gate_definition"],
        "risk_signal_rows": risk_rows,
        "origin_association_rows": origin_association_rows,
        "decision_relevance": decision,
        "regime_diagnostics": regime_summary,
        "late_regime_mechanism_screen": late_mechanism_support,
        "city_concentration": concentration_summary,
        "hypothesis_activation": hypothesis_matrix,
        "activated_directions": active_directions,
        "screening_decision": screening_decision,
        "s5_complete": False,
        "s6_entered": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "reused_artifacts": [
            "reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv",
            "reports/S3_HANDOFFS/S3-Plan03/per_origin_metrics.csv",
            "reports/S3_HANDOFFS/S3-Plan03/local_slope_dispersion.csv",
            "reports/S3_HANDOFFS/S3-Plan03/slope_error_relationship.csv",
            "reports/S3_HANDOFFS/S3-Plan03/time_regime_comparison.csv",
            "reports/S4_HANDOFFS/S4-Plan05A/plan05a_summary.json",
        ],
        "generated_artifacts": [
            "HANDOFF.md",
            "screening_summary.json",
            "preorigin_risk_signal_screen.csv",
            "model_tradeoff_by_risk.csv",
            "city_concentration_repeatability.csv",
            "hypothesis_activation_matrix.csv",
            "scripts/build_s5_plan01_residual_regime_screen.py",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "preorigin_risk_signal_screen.csv", risk_rows)
    write_csv(OUTPUT_DIR / "model_tradeoff_by_risk.csv", tradeoff_rows)
    write_csv(OUTPUT_DIR / "city_concentration_repeatability.csv", concentration_rows)
    write_csv(OUTPUT_DIR / "hypothesis_activation_matrix.csv", hypothesis_matrix)
    (OUTPUT_DIR / "screening_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "screening_decision": screening_decision,
                "active_directions": active_directions,
                "diagnostic_rows": len(records),
                "risk_rows": len(risk_rows),
                "tradeoff_rows": len(tradeoff_rows),
                "city_concentration_rows": len(concentration_rows),
                "b3_compatibility": compatibility["pass"],
                "boundary_violation_count": boundary_violation_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
