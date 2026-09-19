"""Screen deterministic robust local trends under the frozen S5 contract.

This runner is intentionally narrow.  It reuses the frozen S3 B1/B3/B4
city-origin outcomes, reconstructs the target-history-only diagnostics at
the T-2 boundary, and evaluates only the pre-registered RLT3/RLT5
median-slope estimators.  It does not fit a blend, route rows, search
thresholds, or introduce external data.
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
PLAN_ID = "S5-Plan02"
PLAN01_BASE_SHA = "04e5cd469c45468a60b1f9b65a96e76311d5326b"
PLAN01_TIP_SHA = "1a6770f763966b58dd7ae60fef34e332febcea09"
PLAN01_EVIDENCE_SHA = "db7bf05fa2447c6cdc316a054d08e1d248641a45"
INTEGRATED_MAIN_SHA = PLAN01_TIP_SHA
WORKER_BRANCH = "s5/plan02-robust-local-trend-screening"

COMMON_ORIGINS = tuple(range(2011, 2022))
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
CITIES = tuple(f"city{i}" for i in range(1, 41))

B1_ID = "B1_TWO_POINT_LINEAR_TREND"
B3_ID = "B3_RECENT_3POINT_OLS_TREND"
B4_ID = "B4_RECENT_5POINT_OLS_TREND"
RLT3_ID = "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND"
RLT5_ID = "RLT5_RECENT_5POINT_MEDIAN_SLOPE_TREND"
CANDIDATE_IDS = (B3_ID, B4_ID, RLT3_ID, RLT5_ID)
ROBUST_IDS = (RLT3_ID, RLT5_ID)
COMPARISON_IDS = (B4_ID, RLT3_ID, RLT5_ID)

TARGET_COLUMNS = {
    "resident": "常住人口（万人）",
    "registered": "户籍人口（万人）",
}

TOP_QUARTILE_FRACTION = 0.25
SCALE_NORMALIZATION_EPSILON = 1e-12
SCALE_OUTSIDE_MIN_RELATIVE_IMPROVEMENT = 0.01
MSE_TOLERANCE = 1e-9

S3_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan03"
S3_PREDICTIONS_PATH = S3_DIR / "per_origin_predictions_or_residuals.csv"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
PLAN01_SUMMARY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan01" / "screening_summary.json"
OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan02"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
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
            if value is None:
                continue
            key = (target_id, city, year)
            if key in population:
                raise AssertionError(f"duplicate population key: {key}")
            population[key] = value
    return population


def load_plan01_summary() -> dict[str, Any]:
    summary = json.loads(PLAN01_SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["plan_id"] != "S5-Plan01":
        raise AssertionError("unexpected Plan01 summary")
    if summary["base_sha"] != PLAN01_BASE_SHA:
        raise AssertionError("Plan01 base SHA drifted")
    if summary["screening_decision"] != "mechanism_signal_found:robust_local_trend":
        raise AssertionError("Plan02 is not authorized by the recorded Plan01 decision")
    if summary["activated_directions"] != ["robust_local_trend"]:
        raise AssertionError("unexpected Plan01 activation set")
    return summary


def load_source_rows() -> dict[tuple[str, str, int, str], dict[str, dict[str, str]]]:
    allowed = {B1_ID, B3_ID, B4_ID}
    grouped: dict[tuple[str, str, int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_csv(S3_PREDICTIONS_PATH):
        target_id = row["target_id"]
        scenario_id = row["scenario_id"]
        origin_year = as_int(row["origin_year"])
        city_id = row["city_id"]
        predictor_id = row["predictor_id"]
        if (
            target_id not in TARGETS
            or scenario_id not in SCENARIOS
            or origin_year not in COMMON_ORIGINS
            or city_id not in CITIES
            or predictor_id not in allowed
        ):
            continue
        if row["scoreable"].lower() != "yes":
            raise AssertionError(f"non-scoreable frozen source row: {row}")
        key = (target_id, scenario_id, origin_year, city_id)
        if predictor_id in grouped[key]:
            raise AssertionError(f"duplicate source row: {key}, {predictor_id}")
        grouped[key][predictor_id] = row
    expected = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    if len(grouped) != expected:
        raise AssertionError(f"expected {expected} city-origin groups, got {len(grouped)}")
    for key, predictors in grouped.items():
        if set(predictors) != allowed:
            raise AssertionError(f"incomplete B1/B3/B4 source group: {key}, {set(predictors)}")
    return grouped


def history_for(
    population: dict[tuple[str, str, int], float],
    target_id: str,
    city_id: str,
    cutoff_year: int,
) -> list[tuple[int, float]]:
    return [
        (year, population[(target_id, city_id, year)])
        for target, city, year in sorted(population)
        if target == target_id and city == city_id and year <= cutoff_year
    ]


def fit_median_slope(points: list[tuple[int, float]], target_year: int) -> dict[str, float]:
    if len(points) < 2:
        raise AssertionError("median-slope fit requires at least two points")
    pair_slopes = [
        (points[j][1] - points[i][1]) / (points[j][0] - points[i][0])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    robust_slope = statistics.median(pair_slopes)
    intercepts = [value - robust_slope * year for year, value in points]
    robust_intercept = statistics.median(intercepts)
    return {
        "slope": robust_slope,
        "intercept": robust_intercept,
        "prediction": robust_intercept + robust_slope * target_year,
    }


def fit_two_point(points: list[tuple[int, float]], target_year: int) -> float:
    left_year, left_value = points[-2]
    right_year, right_value = points[-1]
    slope = (right_value - left_value) / (right_year - left_year)
    return right_value + slope * (target_year - right_year)


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def build_records(
    population: dict[tuple[str, str, int], float],
    source: dict[tuple[str, str, int, str], dict[str, dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    records: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, int]] = {
        RLT3_ID: {"median_slope": 0, "b1_fallback": 0, "not_computable": 0},
        RLT5_ID: {"median_slope": 0, "b1_fallback": 0, "not_computable": 0},
    }
    for (target_id, scenario_id, origin_year, city_id), rows in sorted(source.items()):
        cutoff_year = origin_year - 2
        history = history_for(population, target_id, city_id, cutoff_year)
        if len(history) < 3:
            raise AssertionError(f"B3-compatible history missing: {target_id}, {city_id}, {origin_year}")
        if history[-1][0] > cutoff_year:
            raise AssertionError(f"history boundary violation: {target_id}, {city_id}, {origin_year}")

        b1_prediction = as_float(rows[B1_ID]["prediction_value"])
        b3_prediction = as_float(rows[B3_ID]["prediction_value"])
        b4_prediction = as_float(rows[B4_ID]["prediction_value"])
        actual = as_float(rows[B3_ID]["actual_target_value_for_scoring"])
        if any(value is None for value in (b1_prediction, b3_prediction, b4_prediction, actual)):
            raise AssertionError(f"missing source prediction/outcome: {target_id}, {city_id}, {origin_year}")
        if len(history) < 5 and abs(b4_prediction - b1_prediction) > MSE_TOLERANCE:
            raise AssertionError(f"frozen B4 fallback is not B1-compatible: {target_id}, {city_id}, {origin_year}")

        differences = [history[index][1] - history[index - 1][1] for index in range(1, len(history))]
        raw_volatility = population_std(differences)
        latest_allowed_target = history[-1][1]
        relative_volatility = raw_volatility / max(abs(latest_allowed_target), SCALE_NORMALIZATION_EPSILON)

        robust_predictions: dict[str, float] = {}
        robust_status: dict[str, str] = {}
        robust_window_years: dict[str, list[int]] = {}
        for candidate_id, window_size in ((RLT3_ID, 3), (RLT5_ID, 5)):
            if len(history) >= window_size:
                window = history[-window_size:]
                robust_predictions[candidate_id] = fit_median_slope(window, origin_year)["prediction"]
                robust_status[candidate_id] = "median_slope"
                robust_window_years[candidate_id] = [year for year, _ in window]
                coverage[candidate_id]["median_slope"] += 1
            elif len(history) >= 2:
                robust_predictions[candidate_id] = fit_two_point(history, origin_year)
                robust_status[candidate_id] = "fallback_to_B1_inherited_window_policy"
                robust_window_years[candidate_id] = [year for year, _ in history[-2:]]
                coverage[candidate_id]["b1_fallback"] += 1
            else:
                coverage[candidate_id]["not_computable"] += 1
                raise AssertionError(f"robust candidate not computable: {target_id}, {city_id}, {origin_year}")

        source_b3_squared_error = as_float(rows[B3_ID]["squared_error"])
        computed_b3_squared_error = (b3_prediction - actual) ** 2
        if source_b3_squared_error is None or abs(source_b3_squared_error - computed_b3_squared_error) > MSE_TOLERANCE:
            raise AssertionError(f"B3 source squared-error mismatch: {target_id}, {city_id}, {origin_year}")

        predictions = {
            B3_ID: b3_prediction,
            B4_ID: b4_prediction,
            RLT3_ID: robust_predictions[RLT3_ID],
            RLT5_ID: robust_predictions[RLT5_ID],
        }
        squared_errors = {candidate_id: (prediction - actual) ** 2 for candidate_id, prediction in predictions.items()}
        absolute_errors = {candidate_id: abs(prediction - actual) for candidate_id, prediction in predictions.items()}
        records.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": origin_year,
                "city_id": city_id,
                "target_history_cutoff_year": cutoff_year,
                "history_max_year_used": history[-1][0],
                "history_observation_count": len(history),
                "history_years": ";".join(str(year) for year, _ in history),
                "latest_allowed_target": latest_allowed_target,
                "raw_history_difference_volatility": raw_volatility,
                "relative_history_difference_volatility": relative_volatility,
                "predictions": predictions,
                "squared_errors": squared_errors,
                "absolute_errors": absolute_errors,
                "actual": actual,
                "robust_status": robust_status,
                "robust_window_years": robust_window_years,
            }
        )
    return records, coverage


def metric_summary(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, float | int | None]:
    squared = [row["squared_errors"][candidate_id] for row in rows]
    absolute = [row["absolute_errors"][candidate_id] for row in rows]
    signed = [row["predictions"][candidate_id] - row["actual"] for row in rows]
    return {
        "row_count": len(rows),
        "mse": mean(squared),
        "rmse": math.sqrt(mean(squared)) if squared else None,
        "mae": mean(absolute),
        "bias": mean(signed),
        "median_absolute_error": median(absolute),
    }


def group_by(rows: list[dict[str, Any]], *fields: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def aggregate_metric_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    grouped = group_by(records, "target_id", "scenario_id")
    for (target_id, scenario_id), target_rows in sorted(grouped.items()):
        by_origin = group_by(target_rows, "origin_year")
        for candidate_id in CANDIDATE_IDS:
            pooled = metric_summary(target_rows, candidate_id)
            origin_summaries = {origin_key[0]: metric_summary(rows, candidate_id) for origin_key, rows in sorted(by_origin.items())}
            origin_mse = [float(summary["mse"]) for summary in origin_summaries.values()]
            origin_bias = [float(summary["bias"]) for summary in origin_summaries.values()]
            latest = origin_summaries[2021]
            row = {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "candidate_id": candidate_id,
                "scoreable_rows": pooled["row_count"],
                "pooled_mse": pooled["mse"],
                "pooled_rmse": pooled["rmse"],
                "pooled_mae": pooled["mae"],
                "pooled_bias": pooled["bias"],
                "pooled_median_absolute_error": pooled["median_absolute_error"],
                "macro_origin_mse": mean(origin_mse),
                "macro_origin_rmse": math.sqrt(mean(origin_mse)) if origin_mse else None,
                "macro_origin_mae": mean(float(summary["mae"]) for summary in origin_summaries.values()),
                "macro_origin_bias": mean(origin_bias),
                "latest_2021_mse": latest["mse"],
                "latest_2021_rmse": latest["rmse"],
                "latest_2021_mae": latest["mae"],
                "latest_2021_bias": latest["bias"],
                "origin_mse_json": json.dumps({str(origin): summary["mse"] for origin, summary in origin_summaries.items()}, sort_keys=True),
            }
            output.append(row)
            lookup[(target_id, scenario_id, candidate_id)] = row
    return output, lookup


def comparison_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    grouped = group_by(records, "target_id", "scenario_id", "origin_year")
    summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (target_id, scenario_id, origin_year), rows in sorted(grouped.items()):
        for candidate_id in COMPARISON_IDS:
            b3 = metric_summary(rows, B3_ID)
            candidate = metric_summary(rows, candidate_id)
            delta = float(candidate["mse"]) - float(b3["mse"])
            output.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "origin_year": origin_year,
                    "candidate_id": candidate_id,
                    "b3_mse": b3["mse"],
                    "candidate_mse": candidate["mse"],
                    "candidate_minus_b3_mse_delta": delta,
                    "b3_mae": b3["mae"],
                    "candidate_mae": candidate["mae"],
                    "candidate_minus_b3_mae_delta": float(candidate["mae"]) - float(b3["mae"]),
                    "candidate_bias": candidate["bias"],
                    "b3_bias": b3["bias"],
                    "origin_result": "win" if delta < -MSE_TOLERANCE else "loss" if delta > MSE_TOLERANCE else "tie",
                }
            )
    by_key = group_by(output, "target_id", "scenario_id", "candidate_id")
    for key, rows in by_key.items():
        deltas = [float(row["candidate_minus_b3_mse_delta"]) for row in rows]
        summary[key] = {
            "origin_count": len(rows),
            "wins": sum(delta < -MSE_TOLERANCE for delta in deltas),
            "losses": sum(delta > MSE_TOLERANCE for delta in deltas),
            "ties": sum(abs(delta) <= MSE_TOLERANCE for delta in deltas),
            "median_delta": median(deltas),
            "worst_degradation": max(deltas) if deltas else None,
        }
    return output, summary


def top_quartile_city_ids(rows: list[dict[str, Any]], field: str) -> set[str]:
    count = max(1, math.ceil(len(CITIES) * TOP_QUARTILE_FRACTION))
    ordered = sorted(rows, key=lambda row: (-float(row[field]), row["city_id"]))
    return {row["city_id"] for row in ordered[:count]}


def stratum_rows(
    records: list[dict[str, Any]],
    diagnostic_field: str,
    candidate_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    grouped = group_by(records, "target_id", "scenario_id", "origin_year")
    for (target_id, scenario_id, origin_year), origin_rows in sorted(grouped.items()):
        high_ids = top_quartile_city_ids(origin_rows, diagnostic_field)
        strata = {
            "high_volatility_top_quartile" if diagnostic_field == "raw_history_difference_volatility" else "high_relative_volatility_top_quartile": [
                row for row in origin_rows if row["city_id"] in high_ids
            ],
            "remaining": [row for row in origin_rows if row["city_id"] not in high_ids],
        }
        for stratum, selected in strata.items():
            for candidate_id in candidate_ids:
                b3 = metric_summary(selected, B3_ID)
                candidate = metric_summary(selected, candidate_id)
                b3_mse = float(b3["mse"])
                candidate_mse = float(candidate["mse"])
                row = {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "diagnostic_id": "raw_history_difference_volatility" if diagnostic_field == "raw_history_difference_volatility" else "relative_history_difference_volatility",
                    "risk_stratum": stratum,
                    "origin_year": origin_year,
                    "city_count": len(selected),
                    "b3_mse": b3_mse,
                    "candidate_id": candidate_id,
                    "candidate_mse": candidate_mse,
                    "candidate_minus_b3_mse_delta": candidate_mse - b3_mse,
                    "relative_improvement_vs_b3": (b3_mse - candidate_mse) / b3_mse if b3_mse else None,
                }
                output.append(row)
                lookup[(target_id, scenario_id, candidate_id, f"{row['diagnostic_id']}::{stratum}::{origin_year}")] = row

    pooled_groups = group_by(output, "target_id", "scenario_id", "diagnostic_id", "risk_stratum", "candidate_id")
    for (target_id, scenario_id, diagnostic_id, stratum, candidate_id), rows in sorted(pooled_groups.items()):
        selected = []
        for record in records:
            if record["target_id"] != target_id or record["scenario_id"] != scenario_id:
                continue
            high_field = "raw_history_difference_volatility" if diagnostic_id == "raw_history_difference_volatility" else "relative_history_difference_volatility"
            origin_rows = [item for item in records if item["target_id"] == target_id and item["scenario_id"] == scenario_id and item["origin_year"] == record["origin_year"]]
            high_ids = top_quartile_city_ids(origin_rows, high_field)
            is_high = record["city_id"] in high_ids
            if (stratum.endswith("top_quartile") and is_high) or (stratum == "remaining" and not is_high):
                selected.append(record)
        b3 = metric_summary(selected, B3_ID)
        candidate = metric_summary(selected, candidate_id)
        b3_mse = float(b3["mse"])
        candidate_mse = float(candidate["mse"])
        pooled = {
            "target_id": target_id,
            "scenario_id": scenario_id,
            "diagnostic_id": diagnostic_id,
            "risk_stratum": stratum,
            "origin_year": "ALL",
            "city_count": len(selected),
            "b3_mse": b3_mse,
            "candidate_id": candidate_id,
            "candidate_mse": candidate_mse,
            "candidate_minus_b3_mse_delta": candidate_mse - b3_mse,
            "relative_improvement_vs_b3": (b3_mse - candidate_mse) / b3_mse if b3_mse else None,
        }
        output.append(pooled)
        lookup[(target_id, scenario_id, candidate_id, f"{diagnostic_id}::{stratum}::ALL")] = pooled
    return output, lookup


def scale_confounding_rows(
    records: list[dict[str, Any]],
    candidate_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    grouped = group_by(records, "target_id", "scenario_id", "origin_year")
    for (target_id, scenario_id, origin_year), origin_rows in sorted(grouped.items()):
        raw_high = top_quartile_city_ids(origin_rows, "raw_history_difference_volatility")
        scale_high = top_quartile_city_ids(origin_rows, "latest_allowed_target")
        cells = {
            "high_raw_high_scale": [row for row in origin_rows if row["city_id"] in raw_high and row["city_id"] in scale_high],
            "high_raw_not_high_scale": [row for row in origin_rows if row["city_id"] in raw_high and row["city_id"] not in scale_high],
            "not_high_raw_high_scale": [row for row in origin_rows if row["city_id"] not in raw_high and row["city_id"] in scale_high],
            "neither_high_raw_nor_high_scale": [row for row in origin_rows if row["city_id"] not in raw_high and row["city_id"] not in scale_high],
        }
        for cell, selected in cells.items():
            for candidate_id in candidate_ids:
                b3 = metric_summary(selected, B3_ID)
                candidate = metric_summary(selected, candidate_id)
                b3_mse = float(b3["mse"])
                candidate_mse = float(candidate["mse"])
                row = {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "cell": cell,
                    "origin_year": origin_year,
                    "city_count": len(selected),
                    "candidate_id": candidate_id,
                    "b3_mse": b3_mse,
                    "candidate_mse": candidate_mse,
                    "candidate_minus_b3_mse_delta": candidate_mse - b3_mse,
                    "relative_improvement_vs_b3": (b3_mse - candidate_mse) / b3_mse if b3_mse else None,
                }
                output.append(row)
    pooled_groups = group_by(output, "target_id", "scenario_id", "cell", "candidate_id")
    for (target_id, scenario_id, cell, candidate_id), _rows in sorted(pooled_groups.items()):
        selected: list[dict[str, Any]] = []
        for (row_target, row_scenario, row_origin), origin_rows in grouped.items():
            if row_target != target_id or row_scenario != scenario_id:
                continue
            raw_high = top_quartile_city_ids(origin_rows, "raw_history_difference_volatility")
            scale_high = top_quartile_city_ids(origin_rows, "latest_allowed_target")
            selected.extend(
                row
                for row in origin_rows
                if (
                    cell == "high_raw_high_scale" and row["city_id"] in raw_high and row["city_id"] in scale_high
                )
                or (
                    cell == "high_raw_not_high_scale" and row["city_id"] in raw_high and row["city_id"] not in scale_high
                )
                or (
                    cell == "not_high_raw_high_scale" and row["city_id"] not in raw_high and row["city_id"] in scale_high
                )
                or (
                    cell == "neither_high_raw_nor_high_scale" and row["city_id"] not in raw_high and row["city_id"] not in scale_high
                )
            )
        b3 = metric_summary(selected, B3_ID)
        candidate = metric_summary(selected, candidate_id)
        b3_mse = float(b3["mse"])
        candidate_mse = float(candidate["mse"])
        pooled = {
            "target_id": target_id,
            "scenario_id": scenario_id,
            "cell": cell,
            "origin_year": "ALL",
            "city_count": len(selected),
            "candidate_id": candidate_id,
            "b3_mse": b3_mse,
            "candidate_mse": candidate_mse,
            "candidate_minus_b3_mse_delta": candidate_mse - b3_mse,
            "relative_improvement_vs_b3": (b3_mse - candidate_mse) / b3_mse if b3_mse else None,
        }
        output.append(pooled)
        lookup[(target_id, scenario_id, candidate_id, cell)] = pooled
    return output, lookup


def b3_compatibility(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {"resident": 872.452651515143, "registered": 225.421212121213}
    observed: dict[str, dict[str, float | int]] = {}
    checks: list[dict[str, Any]] = []
    grouped = group_by(records, "target_id", "scenario_id")
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            rows = grouped[(target_id, scenario_id)]
            by_origin = group_by(rows, "origin_year")
            pooled = metric_summary(rows, B3_ID)
            origin_mses = [float(metric_summary(origin_rows, B3_ID)["mse"]) for origin_rows in by_origin.values()]
            macro = mean(origin_mses)
            latest = metric_summary(by_origin[(2021,)], B3_ID)["mse"]
            key = f"{target_id}_{scenario_id}"
            observed[key] = {
                "pooled_mse": float(pooled["mse"]),
                "macro_origin_mse": float(macro),
                "latest_2021_mse": float(latest),
                "scoreable_rows": len(rows),
            }
            checks.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "expected_pooled_mse": expected[target_id],
                    "observed_pooled_mse": pooled["mse"],
                    "pooled_delta_vs_frozen": float(pooled["mse"]) - expected[target_id],
                    "macro_origin_mse": macro,
                    "latest_2021_mse": latest,
                    "scoreable_rows": len(rows),
                    "status": "PASS" if abs(float(pooled["mse"]) - expected[target_id]) <= MSE_TOLERANCE else "FAIL",
                }
            )
    return {"expected_pooled_mse": expected, "observed": observed, "checks": checks, "pass": all(row["status"] == "PASS" for row in checks)}


def r1_r2_equivalence(records: list[dict[str, Any]]) -> dict[str, Any]:
    left = {(row["target_id"], row["origin_year"], row["city_id"]): row for row in records if row["scenario_id"] == SCENARIOS[0]}
    right = {(row["target_id"], row["origin_year"], row["city_id"]): row for row in records if row["scenario_id"] == SCENARIOS[1]}
    checks: list[dict[str, Any]] = []
    max_delta = 0.0
    for key in sorted(left):
        if key not in right:
            checks.append({"key": key, "status": "FAIL", "reason": "missing R2 row"})
            continue
        for candidate_id in CANDIDATE_IDS:
            delta = abs(left[key]["predictions"][candidate_id] - right[key]["predictions"][candidate_id])
            max_delta = max(max_delta, delta)
            if delta > MSE_TOLERANCE:
                checks.append({"key": key, "candidate_id": candidate_id, "delta": delta, "status": "FAIL"})
    return {"row_count": len(checks), "max_prediction_delta": max_delta, "pass": not checks, "failed_checks": checks[:20]}


def concentration_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped = group_by(records, "target_id", "scenario_id")
    for (target_id, scenario_id), rows in sorted(grouped.items()):
        for candidate_id in ROBUST_IDS:
            by_city = group_by(rows, "city_id")
            improvements = []
            for (city_id,), city_rows in sorted(by_city.items()):
                improvement = sum(row["squared_errors"][B3_ID] - row["squared_errors"][candidate_id] for row in city_rows)
                improvements.append((city_id, improvement))
            positive = sorted(((city, value) for city, value in improvements if value > 0), key=lambda item: (-item[1], item[0]))
            total_positive = sum(value for _, value in positive)
            pooled_b3 = metric_summary(rows, B3_ID)["mse"]
            pooled_candidate = metric_summary(rows, candidate_id)["mse"]
            output.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "candidate_id": candidate_id,
                    "candidate_improves_pooled": pooled_candidate < pooled_b3,
                    "pooled_b3_mse": pooled_b3,
                    "pooled_candidate_mse": pooled_candidate,
                    "top_absolute_sse_improvement_city": positive[0][0] if positive else None,
                    "top_absolute_sse_improvement": positive[0][1] if positive else 0.0,
                    "top3_positive_improvement_share": sum(value for _, value in positive[:3]) / total_positive if total_positive else None,
                    "improved_city_count": sum(value > 0 for _, value in improvements),
                    "improved_city_share": sum(value > 0 for _, value in improvements) / len(CITIES),
                }
            )
    return output


def candidate_definition_rows(records: list[dict[str, Any]], coverage: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": B3_ID,
            "family_role": "frozen_control",
            "formula": "OLS intercept + slope on latest 3 non-missing target observations <= T-2; predict T",
            "history_window": 3,
            "estimator": "ordinary least squares",
            "fallback_policy": "frozen B1 fallback only if fewer than 3; no new fallback used in this coverage",
            "missing_history_policy": "deterministic non-missing history; inherited S3 policy",
            "coverage_median_slope": "n/a",
            "coverage_b1_fallback": "n/a",
        },
        {
            "candidate_id": B4_ID,
            "family_role": "frozen_window_control",
            "formula": "OLS intercept + slope on latest 5 non-missing target observations <= T-2; predict T",
            "history_window": 5,
            "estimator": "ordinary least squares",
            "fallback_policy": "frozen B1 fallback if fewer than 5; no silent B3 fallback",
            "missing_history_policy": "deterministic non-missing history; inherited S3 policy",
            "coverage_median_slope": "n/a",
            "coverage_b1_fallback": "source artifact count",
        },
        {
            "candidate_id": RLT3_ID,
            "family_role": "pre_registered_robust_candidate",
            "formula": "median of all pairwise slopes; median(y_i - robust_slope * year_i) intercept; predict T",
            "history_window": 3,
            "estimator": "deterministic median-slope / median-intercept",
            "fallback_policy": "fixed B1 fallback only if fewer than 3; no silent B3 fallback",
            "missing_history_policy": "latest non-missing observations <= T-2",
            "coverage_median_slope": coverage[RLT3_ID]["median_slope"],
            "coverage_b1_fallback": coverage[RLT3_ID]["b1_fallback"],
        },
        {
            "candidate_id": RLT5_ID,
            "family_role": "pre_registered_robust_candidate",
            "formula": "median of all pairwise slopes on latest 5; median(y_i - robust_slope * year_i) intercept; predict T",
            "history_window": 5,
            "estimator": "deterministic median-slope / median-intercept",
            "fallback_policy": "inherit frozen B4 fixed B1 fallback if fewer than 5; never fall back to B3",
            "missing_history_policy": "latest non-missing observations <= T-2",
            "coverage_median_slope": coverage[RLT5_ID]["median_slope"],
            "coverage_b1_fallback": coverage[RLT5_ID]["b1_fallback"],
        },
        {
            "candidate_id": "relative_history_difference_volatility",
            "family_role": "diagnostic_only",
            "formula": "population_std(first_differences) / max(abs(latest_allowed_target), 1e-12)",
            "history_window": "all available <= T-2",
            "estimator": "fixed scale-normalized diagnostic",
            "fallback_policy": "epsilon only prevents division by zero",
            "missing_history_policy": "no imputation",
            "coverage_median_slope": "n/a",
            "coverage_b1_fallback": "n/a",
        },
    ]


def gate_details(
    records: list[dict[str, Any]],
    aggregate_lookup: dict[tuple[str, str, str], dict[str, Any]],
    comparison_summary: dict[tuple[str, str, str], dict[str, Any]],
    stratum_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    scale_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    strong: dict[str, Any] = {}
    complementary: dict[str, Any] = {}
    concentration = concentration_summary(records)
    for candidate_id in ROBUST_IDS:
        resident_agg = aggregate_lookup[("resident", SCENARIOS[0], candidate_id)]
        resident_b3 = aggregate_lookup[("resident", SCENARIOS[0], B3_ID)]
        resident_relative_gain = (float(resident_b3["pooled_mse"]) - float(resident_agg["pooled_mse"])) / float(resident_b3["pooled_mse"])
        resident_origin = comparison_summary[("resident", SCENARIOS[0], candidate_id)]
        registered_agg = [
            aggregate_lookup[("registered", scenario_id, candidate_id)] for scenario_id in SCENARIOS
        ]
        registered_b3 = [
            aggregate_lookup[("registered", scenario_id, B3_ID)] for scenario_id in SCENARIOS
        ]
        registered_relative_changes = [
            (float(candidate["pooled_mse"]) - float(b3["pooled_mse"])) / float(b3["pooled_mse"])
            for candidate, b3 in zip(registered_agg, registered_b3)
        ]
        registered_origin_summaries = [comparison_summary[("registered", scenario_id, candidate_id)] for scenario_id in SCENARIOS]
        raw_high = stratum_lookup[("resident", SCENARIOS[0], candidate_id, "raw_history_difference_volatility::high_volatility_top_quartile::ALL")]
        raw_remaining = stratum_lookup[("resident", SCENARIOS[0], candidate_id, "raw_history_difference_volatility::remaining::ALL")]
        relative_high = stratum_lookup[("resident", SCENARIOS[0], candidate_id, "relative_history_difference_volatility::high_relative_volatility_top_quartile::ALL")]
        scale_outside = scale_lookup[("resident", SCENARIOS[0], candidate_id, "high_raw_not_high_scale")]
        scale_outside_gain = float(scale_outside["relative_improvement_vs_b3"])
        rlt5_b4_pass = True
        rlt5_b4_delta = None
        if candidate_id == RLT5_ID:
            b4_agg = aggregate_lookup[("resident", SCENARIOS[0], B4_ID)]
            rlt5_b4_delta = float(resident_agg["pooled_mse"]) - float(b4_agg["pooled_mse"])
            rlt5_b4_pass = rlt5_b4_delta < 0

        registered_loss_ok = all(summary["losses"] <= 8 for summary in registered_origin_summaries)
        registered_raw_high_rows = []
        for scenario_id in SCENARIOS:
            raw_rows = [
                row
                for row in records
                if row["target_id"] == "registered" and row["scenario_id"] == scenario_id
            ]
            raw_strata, _ = stratum_rows(raw_rows, "raw_history_difference_volatility", (candidate_id,))
            high_rows = [
                row
                for row in raw_strata
                if row["origin_year"] != "ALL" and row["risk_stratum"] == "high_volatility_top_quartile"
            ]
            registered_raw_high_rows.append(
                {
                    "scenario_id": scenario_id,
                    "worse_origins": sum(float(row["candidate_minus_b3_mse_delta"]) > MSE_TOLERANCE for row in high_rows),
                    "improved_origins": sum(float(row["candidate_minus_b3_mse_delta"]) < -MSE_TOLERANCE for row in high_rows),
                }
            )
        registered_raw_high_no_reversal = all(item["worse_origins"] <= 8 for item in registered_raw_high_rows)
        raw_high_origin_rows = [
            row
            for row in stratum_rows(
                [item for item in records if item["target_id"] == "resident" and item["scenario_id"] == SCENARIOS[0]],
                "raw_history_difference_volatility",
                (candidate_id,),
            )[0]
            if row["origin_year"] != "ALL" and row["risk_stratum"] == "high_volatility_top_quartile"
        ]
        relative_high_origin_rows = [
            row
            for row in stratum_rows(
                [item for item in records if item["target_id"] == "resident" and item["scenario_id"] == SCENARIOS[0]],
                "relative_history_difference_volatility",
                (candidate_id,),
            )[0]
            if row["origin_year"] != "ALL" and row["risk_stratum"] == "high_relative_volatility_top_quartile"
        ]
        raw_high_wins = sum(float(row["candidate_minus_b3_mse_delta"]) < -MSE_TOLERANCE for row in raw_high_origin_rows)
        relative_high_wins = sum(float(row["candidate_minus_b3_mse_delta"]) < -MSE_TOLERANCE for row in relative_high_origin_rows)
        raw_remaining_ratio = float(raw_remaining["candidate_mse"]) / float(raw_remaining["b3_mse"])
        mechanism_alignment = float(raw_high["candidate_minus_b3_mse_delta"]) < float(raw_remaining["candidate_minus_b3_mse_delta"])
        scale_outside_pass = scale_outside_gain >= SCALE_OUTSIDE_MIN_RELATIVE_IMPROVEMENT
        common = {
            "candidate_id": candidate_id,
            "resident_pooled_mse": resident_agg["pooled_mse"],
            "resident_b3_pooled_mse": resident_b3["pooled_mse"],
            "resident_relative_pooled_gain": resident_relative_gain,
            "resident_origin_wins": resident_origin["wins"],
            "resident_origin_losses": resident_origin["losses"],
            "resident_latest_2021_mse": resident_agg["latest_2021_mse"],
            "registered_relative_pooled_changes": registered_relative_changes,
            "registered_origin_losses": [summary["losses"] for summary in registered_origin_summaries],
            "registered_overall_loss_gate_pass": registered_loss_ok,
            "registered_raw_high_stratum_direction": registered_raw_high_rows,
            "registered_raw_high_no_strong_reversal": registered_raw_high_no_reversal,
            "raw_high_volatility_pooled_delta": raw_high["candidate_minus_b3_mse_delta"],
            "raw_remaining_pooled_delta": raw_remaining["candidate_minus_b3_mse_delta"],
            "raw_high_volatility_relative_improvement": raw_high["relative_improvement_vs_b3"],
            "raw_remaining_mse_ratio": raw_remaining_ratio,
            "mechanism_alignment_pass": mechanism_alignment,
            "relative_high_volatility_wins": relative_high_wins,
            "raw_high_volatility_wins": raw_high_wins,
            "scale_outside_high_raw_not_high_scale_relative_improvement": scale_outside_gain,
            "scale_outside_nontrivial_pass": scale_outside_pass,
            "rlt5_beats_b4_resident_pooled": rlt5_b4_pass,
            "rlt5_minus_b4_resident_pooled_mse_delta": rlt5_b4_delta,
        }
        strong[candidate_id] = {
            **common,
            "gates": {
                "resident_pooled_gain_at_least_2_percent": resident_relative_gain >= 0.02,
                "resident_origin_wins_at_least_7_of_11": resident_origin["wins"] >= 7,
                "registered_pooled_worsening_not_over_2_percent": all(change <= 0.02 for change in registered_relative_changes),
                "registered_losses_not_over_8_of_11": registered_loss_ok,
                "mechanism_high_volatility_gain_larger_than_remaining": mechanism_alignment,
                "scale_confounding_nontrivial_outside_high_scale": scale_outside_pass,
                "rlt5_beats_b4_if_applicable": rlt5_b4_pass,
            },
        }
        strong[candidate_id]["pass"] = all(strong[candidate_id]["gates"].values())
        complementary[candidate_id] = {
            **common,
            "gates": {
                "raw_high_volatility_wins_at_least_8_of_11": raw_high_wins >= 8,
                "relative_high_volatility_wins_at_least_7_of_11": relative_high_wins >= 7,
                "raw_remaining_mse_not_over_10_percent_worse": raw_remaining_ratio <= 1.10,
                "registered_no_strong_directional_reversal": registered_raw_high_no_reversal,
                "scale_confounding_nontrivial_outside_high_scale": scale_outside_pass,
                "candidate_error_profile_is_complementary": (
                    float(raw_high["candidate_minus_b3_mse_delta"]) < 0
                    and raw_remaining_ratio <= 1.10
                    and float(raw_high["candidate_minus_b3_mse_delta"]) < float(raw_remaining["candidate_minus_b3_mse_delta"])
                ),
            },
        }
        complementary[candidate_id]["pass"] = all(complementary[candidate_id]["gates"].values())
    strong_candidates = [candidate_id for candidate_id in ROBUST_IDS if strong[candidate_id]["pass"]]
    complementary_candidates = [candidate_id for candidate_id in ROBUST_IDS if complementary[candidate_id]["pass"]]
    if strong_candidates:
        winner = min(
            strong_candidates,
            key=lambda candidate_id: (
                float(strong[candidate_id]["resident_pooled_mse"]),
                -int(strong[candidate_id]["resident_origin_wins"]),
                float(strong[candidate_id]["raw_high_volatility_pooled_delta"]),
            ),
        )
        decision = f"promising_overall:{winner}"
        next_direction = "winner_only_robust_local_trend_deep_dive"
    elif complementary_candidates:
        winner = min(
            complementary_candidates,
            key=lambda candidate_id: (
                float(complementary[candidate_id]["raw_high_volatility_pooled_delta"]),
                float(complementary[candidate_id]["resident_pooled_mse"]),
            ),
        )
        decision = f"complementary_robust_signal:{winner}"
        next_direction = "fixed_low_capacity_blend_screening_conditional"
    else:
        winner = None
        decision = "robust_local_trend_unsupported"
        next_direction = "stop_robust_local_trend_mechanism"
    return strong, complementary, decision, {
        "winner": winner,
        "next_direction": next_direction,
        "concentration": concentration,
    }


def activation_rows(decision: str, strong: dict[str, Any], complementary: dict[str, Any]) -> list[dict[str, Any]]:
    robust_active = decision.startswith("promising_overall:") or decision.startswith("complementary_robust_signal:")
    blend_active = decision.startswith("complementary_robust_signal:")
    return [
        {
            "hypothesis_id": "robust_local_trend",
            "supporting_evidence": "RLT3/RLT5 evaluated under raw and relative volatility strata",
            "contradicting_evidence": "none" if robust_active else "neither strong nor complementary robust signal passed",
            "gate_status": "pass" if robust_active else "fail",
            "activation_recommendation": "active_next_boundary" if robust_active else "inactive",
        },
        {
            "hypothesis_id": "partial_pooling_shrinkage",
            "supporting_evidence": "no new Plan02 evidence for shrinkage parameters",
            "contradicting_evidence": "Plan01 decision relevance for shrinkage was inactive",
            "gate_status": "not_screened",
            "activation_recommendation": "inactive",
        },
        {
            "hypothesis_id": "fixed_low_capacity_blend",
            "supporting_evidence": "robust candidate has complementary volatile-regime value" if blend_active else "no qualifying complementary robust signal",
            "contradicting_evidence": "blend weights/routing are forbidden in Plan02" if not blend_active else "blend itself not fitted in Plan02",
            "gate_status": "conditional_pass" if blend_active else "not_activated",
            "activation_recommendation": "conditional_next_boundary" if blend_active else "inactive",
        },
        {
            "hypothesis_id": "stronger_boosting",
            "supporting_evidence": "none",
            "contradicting_evidence": "outside Plan02 scope and no direct evidence",
            "gate_status": "not_screened",
            "activation_recommendation": "inactive",
        },
        {
            "hypothesis_id": "remaining_source_groups",
            "supporting_evidence": "none; no new source in scope",
            "contradicting_evidence": "external sources remain inactive from prior evidence",
            "gate_status": "not_screened",
            "activation_recommendation": "inactive",
        },
        {
            "hypothesis_id": "tuning",
            "supporting_evidence": "none; window and estimator are pre-registered",
            "contradicting_evidence": "parameter and threshold search forbidden",
            "gate_status": "not_screened",
            "activation_recommendation": "inactive",
        },
    ]


def main() -> None:
    plan01_summary = load_plan01_summary()
    if not S3_PREDICTIONS_PATH.is_file() or not POPULATION_PATH.is_file():
        raise FileNotFoundError("required frozen S3 predictions or canonical population file is missing")
    population = load_population()
    source = load_source_rows()
    records, coverage = build_records(population, source)
    expected_records = len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS) * len(CITIES)
    if len(records) != expected_records:
        raise AssertionError(f"expected {expected_records} records, got {len(records)}")

    compatibility = b3_compatibility(records)
    if not compatibility["pass"]:
        raise AssertionError(f"B3 compatibility failed: {compatibility}")
    equivalence = r1_r2_equivalence(records)
    if not equivalence["pass"]:
        raise AssertionError(f"R1/R2 target-history-only equivalence failed: {equivalence}")

    aggregate_rows, aggregate_lookup = aggregate_metric_rows(records)
    origin_rows, comparison_summary = comparison_rows(records)
    raw_stratum_rows, raw_stratum_lookup = stratum_rows(records, "raw_history_difference_volatility", COMPARISON_IDS)
    relative_stratum_rows, relative_stratum_lookup = stratum_rows(records, "relative_history_difference_volatility", COMPARISON_IDS)
    stratum_rows_all = raw_stratum_rows + relative_stratum_rows
    stratum_lookup = {**raw_stratum_lookup, **relative_stratum_lookup}
    scale_rows, scale_lookup = scale_confounding_rows(records, COMPARISON_IDS)
    strong, complementary, decision, decision_meta = gate_details(
        records,
        aggregate_lookup,
        comparison_summary,
        stratum_lookup,
        scale_lookup,
    )

    definitions = candidate_definition_rows(records, coverage)
    activation = activation_rows(decision, strong, complementary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "screening_metrics.csv", aggregate_rows)
    write_csv(OUTPUT_DIR / "per_origin_comparison.csv", origin_rows)
    write_csv(OUTPUT_DIR / "volatility_stratum_comparison.csv", stratum_rows_all)
    write_csv(OUTPUT_DIR / "scale_confounding_check.csv", scale_rows)
    write_csv(OUTPUT_DIR / "candidate_definitions.csv", definitions)
    write_csv(OUTPUT_DIR / "hypothesis_activation_matrix.csv", activation)

    summary = {
        "plan_id": PLAN_ID,
        "stage": "S5_EVIDENCE_DRIVEN_ITERATION",
        "base_sha": INTEGRATED_MAIN_SHA,
        "plan01_integration": {
            "plan01_base_sha": PLAN01_BASE_SHA,
            "plan01_tip_sha": PLAN01_TIP_SHA,
            "plan01_evidence_commit": PLAN01_EVIDENCE_SHA,
            "merge_base": PLAN01_BASE_SHA,
            "ahead_from_plan01_base": 2,
            "behind_from_plan01_base": 0,
            "integrated_main_sha": INTEGRATED_MAIN_SHA,
            "plan01_decision": plan01_summary["screening_decision"],
        },
        "worker_branch": current_branch(),
        "worker_head_at_run": current_commit(),
        "target_ids": list(TARGETS),
        "scenario_ids": list(SCENARIOS),
        "common_origins": list(COMMON_ORIGINS),
        "city_count": len(CITIES),
        "scoreable_rows_per_target_scenario": len(CITIES) * len(COMMON_ORIGINS),
        "target_history_cutoff_rule": "all candidate history observations <= origin_year - 2",
        "candidate_ids": list(CANDIDATE_IDS),
        "robust_candidate_ids": list(ROBUST_IDS),
        "rlt5_coverage_policy": "inherit frozen B4 fixed B1 fallback for fewer than 5 points; never fallback to B3",
        "rlt5_coverage": coverage[RLT5_ID],
        "diagnostic_definitions": {
            "raw_history_difference_volatility": "population standard deviation of all available first differences <= T-2",
            "relative_history_difference_volatility": "raw volatility / max(abs(latest allowed target), 1e-12); diagnostic only",
            "preorigin_target_scale": "latest allowed target level <= T-2",
            "within_origin_high_risk_rule": "top ceil(25%) by diagnostic, ties broken by city_id",
        },
        "frozen_b3_compatibility": compatibility,
        "r1_r2_equivalence": equivalence,
        "boundary_checks": {
            "max_history_boundary_violation_count": 0,
            "origin_outcome_used_only_for_scoring": True,
            "new_external_source_used": False,
            "test_xls_used": False,
            "threshold_search": False,
            "parameter_search": False,
            "routing_rule_created": False,
            "blend_fitted": False,
            "final_2023_forecast_run": False,
            "submission_run": False,
        },
        "strong_screening_gate": strong,
        "complementary_signal_gate": complementary,
        "concentration_sanity": decision_meta["concentration"],
        "final_decision": decision,
        "winner": decision_meta["winner"],
        "activated_next_direction": decision_meta["next_direction"],
        "hypothesis_activation": activation,
        "s5_complete": False,
        "s6_entered": False,
        "reused_artifacts": [
            "reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv",
            "reports/S3_HANDOFFS/S3-Plan03/per_origin_metrics.csv",
            "reports/S3_HANDOFFS/S3-Plan03/baseline_predictor_definitions.csv",
            "reports/S5_HANDOFFS/S5-Plan01/screening_summary.json",
        ],
        "generated_artifacts": [
            "HANDOFF.md",
            "screening_metrics.csv",
            "per_origin_comparison.csv",
            "volatility_stratum_comparison.csv",
            "scale_confounding_check.csv",
            "candidate_definitions.csv",
            "screening_summary.json",
            "hypothesis_activation_matrix.csv",
            "scripts/build_s5_plan02_robust_local_trend_screen.py",
        ],
    }
    (OUTPUT_DIR / "screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "final_decision": decision,
                "winner": decision_meta["winner"],
                "records": len(records),
                "aggregate_rows": len(aggregate_rows),
                "origin_rows": len(origin_rows),
                "stratum_rows": len(stratum_rows_all),
                "scale_rows": len(scale_rows),
                "b3_compatibility": compatibility["pass"],
                "r1_r2_equivalence": equivalence["pass"],
                "rlt5_coverage": coverage[RLT5_ID],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
