"""Build S3-Plan01 formal baseline-ladder evidence.

The experiment re-runs the two S2 target-history probes as formal S3
baselines and adds exactly one new mechanism: an expanding, per-city ordinary
least-squares linear trend.  It keeps the frozen S2 validation contract and
does not use covariates, tuning, ensembles, final forecasts, or submission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S3_HANDOFFS" / "S3-Plan01"
S2_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05"
PLAN02_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
S1_CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
S1_QUALITY_PATH = ROOT / "docs" / "s1_quality_issue_registry.csv"
RAW_TEST_PATH = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集" / "test.xls"
QUARANTINE_TEST_PATH = ROOT / "data" / "quarantine" / "source" / "test.xls"

STARTING_COMMIT = "1e896466c45e79b557ec0a0dc386f301b65f1954"
REFERENCE_TARGET_YEAR = 2023
HORIZON_YEARS = 2
LATEST_ORIGIN = 2021
COMMON_ORIGINS = tuple(range(2011, 2022))
EARLY_ORIGINS = tuple(range(2011, 2016))
LATE_ORIGINS = tuple(range(2017, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))

SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)

TARGETS = (
    {
        "target_id": "resident",
        "semantic_name": "常住人口",
        "canonical_column": "常住人口（万人）",
        "contract_role": "primary_proxy",
    },
    {
        "target_id": "registered",
        "semantic_name": "户籍人口",
        "canonical_column": "户籍人口（万人）",
        "contract_role": "sensitivity_target",
    },
)

PREDICTORS = (
    {
        "predictor_id": "B0_PERSISTENCE_LAST_OBSERVED",
        "s2_predictor_id": "PERSISTENCE_LAST_OBSERVED",
        "predictor_name": "persistence / last observed target",
        "mechanism": "reference",
        "formula": "y_hat(T) = y_last; latest non-missing target year <= T-2",
        "minimum_effective_history_points": 1,
        "uses_covariates": "no",
        "tuning": "no",
        "window_selection": "none",
    },
    {
        "predictor_id": "B1_TWO_POINT_LINEAR_TREND",
        "s2_predictor_id": "TWO_POINT_LINEAR_TREND",
        "predictor_name": "recent two-point linear trend",
        "mechanism": "recent_slope",
        "formula": "fit the slope between the two latest non-missing target points <= T-2 and extrapolate to T",
        "minimum_effective_history_points": 2,
        "uses_covariates": "no",
        "tuning": "no",
        "window_selection": "fixed two latest observations",
    },
    {
        "predictor_id": "B2_EXPANDING_PER_CITY_LINEAR_TREND",
        "s2_predictor_id": "",
        "predictor_name": "expanding per-city linear trend",
        "mechanism": "long_history_city_specific_trend",
        "formula": "fit target = intercept + slope * year on all non-missing city target history <= T-2; predict T",
        "minimum_effective_history_points": 2,
        "uses_covariates": "no",
        "tuning": "no",
        "window_selection": "all available history through T-2; no window search",
    },
)

PAIRWISE_COMPARISONS = (
    (
        "B1_vs_B0",
        "B0_PERSISTENCE_LAST_OBSERVED",
        "B1_TWO_POINT_LINEAR_TREND",
    ),
    (
        "B2_vs_B0",
        "B0_PERSISTENCE_LAST_OBSERVED",
        "B2_EXPANDING_PER_CITY_LINEAR_TREND",
    ),
    (
        "B2_vs_B1",
        "B1_TWO_POINT_LINEAR_TREND",
        "B2_EXPANDING_PER_CITY_LINEAR_TREND",
    ),
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


def rounded(value: Any, digits: int = 12) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            writer.writerow(
                {
                    field: "" if row.get(field) is None else rounded(row.get(field))
                    for field in fieldnames
                }
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    assert values
    return math.fsum(values) / len(values)


def mse(errors: list[float]) -> float:
    assert errors
    return math.fsum(error * error for error in errors) / len(errors)


def rmse(errors: list[float]) -> float:
    return math.sqrt(mse(errors))


def competition_score(mse_value: float) -> float:
    return 1.0 / (1.0 + mse_value)


def rank_values(values: dict[str, float]) -> dict[str, int]:
    return {
        key: 1 + sum(other < value - 1e-12 for other in values.values())
        for key, value in values.items()
    }


def value_delta(left: float, right: float) -> float:
    return left - right


def relative_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100 if baseline else 0.0


def json_metric_map(rows: list[dict[str, Any]], field: str) -> str:
    return stable_json(
        {str(row["origin_year"]): rounded(float(row[field])) for row in rows}
    )


def load_population() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    rows = read_csv(POPULATION_PATH)
    assert rows
    keys = [(row["城市名称"], as_int(row["年份"])) for row in rows]
    assert all(year is not None for _, year in keys)
    assert len(keys) == len(set(keys)), "population city-year keys must be unique"
    assert set(city for city, _ in keys) == set(CITIES)

    population = {
        target["target_id"]: {city: {} for city in CITIES}
        for target in TARGETS
    }
    for row in rows:
        city = row["城市名称"]
        year = as_int(row["年份"])
        assert year is not None
        for target in TARGETS:
            value = as_float(row.get(target["canonical_column"]))
            if value is not None:
                population[target["target_id"]][city][year] = value

    city40_2001 = next(
        row
        for row in rows
        if row["城市名称"] == "city40" and as_int(row["年份"]) == 2001
    )
    assert as_float(city40_2001["常住人口（万人）"]) == 284.0
    assert as_float(city40_2001["户籍人口（万人）"]) is None

    profile = {
        "canonical_row_count": len(rows),
        "unique_key_count": len(set(keys)),
        "duplicate_key_rows": len(rows) - len(set(keys)),
        "city_count": len(CITIES),
        "year_min": min(year for _, year in keys),
        "year_max": max(year for _, year in keys),
        "population_sha256": sha256_file(POPULATION_PATH),
    }
    for target in TARGETS:
        target_id = target["target_id"]
        profile[f"{target_id}_nonempty_count"] = sum(
            len(population[target_id][city]) for city in CITIES
        )
        profile[f"{target_id}_latest_observed_year"] = max(
            year
            for city in CITIES
            for year in population[target_id][city]
        )
    return population, profile


def assert_static_boundaries() -> None:
    contract = read_csv(S1_CONTRACT_PATH)
    quality = read_csv(S1_QUALITY_PATH)
    assert len(contract) == 14
    assert next(row for row in contract if row["dataset_id"] == "population")
    assert next(row for row in quality if row["issue_id"] == "QI-004")
    assert not RAW_TEST_PATH.exists()
    assert QUARANTINE_TEST_PATH.exists()


def load_s2_context() -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    dict[tuple[str, str, str, int, str], dict[str, str]],
]:
    summary = read_json(S2_DIR / "plan05_summary.json")
    assert summary["common_strict_origin_domain"] == "2011-2021"
    assert summary["target_history_horizon_years"] == HORIZON_YEARS
    assert summary["r1_r2_numeric_residuals_exactly_equal"] is True
    assert summary["s2_complete"] is False
    assert summary["s3_entered"] is False

    boundary = read_csv(S2_DIR / "origin_information_boundary.csv")
    assert len(boundary) == len(TARGETS) * len(SCENARIOS) * len(COMMON_ORIGINS)
    assert {
        row["scenario_id"] for row in boundary
    } == set(SCENARIOS)
    assert {
        int(row["origin_year"]) for row in boundary
    } == set(COMMON_ORIGINS)

    s2_prediction_rows = read_csv(
        S2_DIR / "per_origin_predictions_or_residuals.csv"
    )
    s2_index = {}
    for row in s2_prediction_rows:
        key = (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
            int(row["origin_year"]),
            row["city_id"],
        )
        s2_index[key] = row
    assert len(s2_index) == len(s2_prediction_rows)
    return summary, boundary, s2_index


def load_scenarios() -> list[dict[str, str]]:
    definitions = read_csv(PLAN02_DIR / "replay_scenario_definitions.csv")
    by_id = {row["scenario_id"]: row for row in definitions}
    assert set(by_id) == set(SCENARIOS)
    for row in by_id.values():
        assert row["target_history_lag_years"] == "2"
        assert row["publication_timing_proven"] == "no"
        assert row["model_scoring_run"] == "no"
    return [by_id[scenario_id] for scenario_id in SCENARIOS]


def predictor_by_id(predictor_id: str) -> dict[str, Any]:
    return next(
        predictor for predictor in PREDICTORS
        if predictor["predictor_id"] == predictor_id
    )


def fit_prediction(
    predictor_id: str,
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
) -> dict[str, Any]:
    history_years = sorted(year for year in series if year <= cutoff_year)
    history_values = [series[year] for year in history_years]
    base = {
        "history_years": history_years,
        "history_count": len(history_years),
        "history_min_year": history_years[0] if history_years else None,
        "history_max_year": history_years[-1] if history_years else None,
        "latest_target_year_used": history_years[-1] if history_years else None,
        "previous_target_year_used": (
            history_years[-2] if len(history_years) >= 2 else None
        ),
        "regression_fit_min_year": None,
        "regression_fit_max_year": None,
        "regression_fit_distinct_year_count": len(set(history_years)),
        "regression_slope": None,
        "regression_intercept": None,
        "regression_fit_status": "not_applicable",
        "prediction": None,
    }

    if predictor_id == "B0_PERSISTENCE_LAST_OBSERVED":
        if history_years:
            base["prediction"] = series[history_years[-1]]
            base["regression_fit_status"] = "not_applicable_reference"
        else:
            base["regression_fit_status"] = "insufficient_history"
        return base

    if predictor_id == "B1_TWO_POINT_LINEAR_TREND":
        if len(history_years) >= 2:
            previous_year, latest_year = history_years[-2:]
            slope = (series[latest_year] - series[previous_year]) / (
                latest_year - previous_year
            )
            base["regression_slope"] = slope
            base["regression_intercept"] = series[latest_year] - slope * latest_year
            base["prediction"] = series[latest_year] + slope * (
                target_year - latest_year
            )
            base["regression_fit_status"] = "two_point_slope"
        else:
            base["regression_fit_status"] = "insufficient_history"
        return base

    if predictor_id == "B2_EXPANDING_PER_CITY_LINEAR_TREND":
        if len(history_years) < 2 or len(set(history_years)) < 2:
            base["regression_fit_status"] = "insufficient_history_for_identifiable_slope"
            return base
        x_mean = mean([float(year) for year in history_years])
        y_mean = mean(history_values)
        denominator = math.fsum(
            (float(year) - x_mean) ** 2 for year in history_years
        )
        assert denominator > 0
        numerator = math.fsum(
            (float(year) - x_mean) * (value - y_mean)
            for year, value in zip(history_years, history_values)
        )
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        base["regression_fit_min_year"] = history_years[0]
        base["regression_fit_max_year"] = history_years[-1]
        base["regression_slope"] = slope
        base["regression_intercept"] = intercept
        base["prediction"] = intercept + slope * target_year
        base["regression_fit_status"] = "expanding_all_history_ols"
        return base

    raise AssertionError(predictor_id)


def build_predictions(
    population: dict[str, dict[str, dict[int, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIOS:
            for predictor in PREDICTORS:
                predictor_id = predictor["predictor_id"]
                for origin in COMMON_ORIGINS:
                    cutoff = origin - HORIZON_YEARS
                    for city in CITIES:
                        series = population[target_id][city]
                        fit = fit_prediction(
                            predictor_id,
                            series,
                            origin,
                            cutoff,
                        )
                        actual = series.get(origin)
                        prediction = fit["prediction"]
                        scoreable = prediction is not None and actual is not None
                        residual = (
                            prediction - actual if scoreable else None
                        )
                        future_years = sorted(
                            year for year in series if year > cutoff
                        )
                        latest_used = fit["latest_target_year_used"]
                        fit_max = fit["regression_fit_max_year"]
                        leakage_pass = (
                            (latest_used is None or latest_used <= cutoff)
                            and (fit_max is None or fit_max <= cutoff)
                        )
                        assert leakage_pass
                        row = {
                            "target_id": target_id,
                            "target_column": target["canonical_column"],
                            "target_contract_role": target["contract_role"],
                            "scenario_id": scenario_id,
                            "predictor_id": predictor_id,
                            "origin_year": origin,
                            "city_id": city,
                            "reference_target_year": REFERENCE_TARGET_YEAR,
                            "target_history_cutoff_year": cutoff,
                            "actual_target_year_for_scoring": (
                                origin if actual is not None else ""
                            ),
                            "actual_target_value_for_scoring": actual,
                            "history_observation_count": fit["history_count"],
                            "history_years_used": ";".join(
                                str(year) for year in fit["history_years"]
                            ),
                            "history_min_year": fit["history_min_year"],
                            "history_max_year": fit["history_max_year"],
                            "latest_target_year_used": latest_used,
                            "previous_target_year_used": fit[
                                "previous_target_year_used"
                            ],
                            "regression_fit_min_year": fit[
                                "regression_fit_min_year"
                            ],
                            "regression_fit_max_year": fit[
                                "regression_fit_max_year"
                            ],
                            "regression_fit_distinct_year_count": fit[
                                "regression_fit_distinct_year_count"
                            ],
                            "regression_slope": fit["regression_slope"],
                            "regression_intercept": fit["regression_intercept"],
                            "regression_fit_status": fit["regression_fit_status"],
                            "prediction_value": prediction,
                            "prediction_status": (
                                "scoreable"
                                if scoreable
                                else (
                                    "missing_actual_target"
                                    if actual is None
                                    else "insufficient_target_history"
                                )
                            ),
                            "scoreable": "yes" if scoreable else "no",
                            "residual_prediction_minus_actual": residual,
                            "squared_error": residual * residual
                            if residual is not None
                            else None,
                            "absolute_error": abs(residual)
                            if residual is not None
                            else None,
                        }
                        prediction_rows.append(row)
                        leakage_rows.append(
                            {
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "predictor_id": predictor_id,
                                "origin_year": origin,
                                "city_id": city,
                                "scoring_target_year": origin,
                                "allowed_target_history_cutoff_year": cutoff,
                                "history_rows_used": fit["history_count"],
                                "history_years_used": row["history_years_used"],
                                "latest_target_year_used": latest_used,
                                "regression_fit_max_year_used": fit_max
                                if fit_max is not None
                                else latest_used,
                                "t_minus_1_target_present_but_excluded": (
                                    "yes" if origin - 1 in series else "no"
                                ),
                                "origin_target_present_but_scoring_only": (
                                    "yes" if actual is not None else "not_available"
                                ),
                                "future_target_years_not_used": ";".join(
                                    str(year) for year in future_years
                                ),
                                "target_fit_history_max_within_T_minus_2": (
                                    "PASS"
                                    if latest_used is None or latest_used <= cutoff
                                    else "FAIL"
                                ),
                                "regression_fit_history_max_within_T_minus_2": (
                                    "PASS"
                                    if fit_max is None or fit_max <= cutoff
                                    else "FAIL"
                                ),
                                "future_labels_excluded_from_fit": "PASS",
                                "leakage_status": (
                                    "PASS" if leakage_pass else "FAIL"
                                ),
                                "audit_rule": (
                                    "all predictor history and regression fit rows "
                                    "must have year <= T-2; T is scoring-only"
                                ),
                            }
                        )
    assert len(prediction_rows) == (
        len(TARGETS)
        * len(SCENARIOS)
        * len(PREDICTORS)
        * len(COMMON_ORIGINS)
        * len(CITIES)
    )
    assert all(row["leakage_status"] == "PASS" for row in leakage_rows)
    return prediction_rows, leakage_rows


def build_s3_boundary_rows(
    s2_boundary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for source_row in s2_boundary:
        row = dict(source_row)
        row.update(
            {
                "s3_plan_id": "S3-Plan01",
                "s3_predictor_family": (
                    "B0 persistence; B1 recent two-point trend; "
                    "B2 expanding per-city linear trend"
                ),
                "s3_predictors_use_covariates": "no",
                "s3_target_history_rule": "all fit rows have year <= T-2",
                "s3_scenario_numeric_effect_expected": "none",
            }
        )
        rows.append(row)
    return rows


def build_per_origin_metrics(
    prediction_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str, int], dict[str, Any]],
]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[
            (
                row["target_id"],
                row["scenario_id"],
                row["predictor_id"],
                int(row["origin_year"]),
            )
        ].append(row)

    metrics: list[dict[str, Any]] = []
    metric_map = {}
    for key in sorted(grouped):
        target_id, scenario_id, predictor_id, origin = key
        rows = grouped[key]
        score_rows = [row for row in rows if row["scoreable"] == "yes"]
        errors = [
            float(row["residual_prediction_minus_actual"])
            for row in score_rows
        ]
        absolute_errors = [abs(error) for error in errors]
        squared_errors = [error * error for error in errors]
        assert errors
        target_label_count = sum(
            row["actual_target_year_for_scoring"] != "" for row in rows
        )
        predictor_count = sum(
            row["prediction_value"] is not None for row in rows
        )
        row = {
            "target_id": target_id,
            "scenario_id": scenario_id,
            "predictor_id": predictor_id,
            "origin_year": origin,
            "target_history_cutoff_year": origin - HORIZON_YEARS,
            "origin_score_scope": "city-year rows within this pseudo-origin",
            "all_city_count": len(CITIES),
            "target_label_available_city_count": target_label_count,
            "predictor_computable_city_count": predictor_count,
            "scoreable_city_count": len(score_rows),
            "scoreable_city_rate": len(score_rows) / len(CITIES),
            "sse": math.fsum(squared_errors),
            "mse": mean(squared_errors),
            "rmse": math.sqrt(mean(squared_errors)),
            "mae": mean(absolute_errors),
            "mean_signed_error_bias": mean(errors),
            "median_absolute_error": statistics.median(absolute_errors),
            "error_std_population": statistics.pstdev(errors)
            if len(errors) > 1
            else 0.0,
            "max_absolute_error": max(absolute_errors),
            "min_error": min(errors),
            "max_error": max(errors),
            "competition_score_from_origin_mse": competition_score(
                mean(squared_errors)
            ),
            "r1_r2_numeric_score_effect": "expected_none_target_history_only",
        }
        metrics.append(row)
        metric_map[key] = row

    rank_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        rank_groups[
            (row["target_id"], row["scenario_id"], int(row["origin_year"]))
        ].append(row)
    for key, rows in rank_groups.items():
        ranks = rank_values(
            {row["predictor_id"]: float(row["mse"]) for row in rows}
        )
        winner = min(rows, key=lambda row: float(row["mse"]))["predictor_id"]
        for row in rows:
            row["origin_mse_rank"] = ranks[row["predictor_id"]]
            row["origin_winner_predictor_id"] = winner
    expected_count = (
        len(TARGETS) * len(SCENARIOS) * len(PREDICTORS) * len(COMMON_ORIGINS)
    )
    assert len(metrics) == expected_count
    return metrics, metric_map


def dominant_city_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    city_sse: dict[str, float] = defaultdict(float)
    for row in rows:
        city_sse[row["city_id"]] += float(row["squared_error"])
    total_sse = math.fsum(city_sse.values())
    ranked = sorted(city_sse.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total_sse": total_sse,
        "dominant_city_id": ranked[0][0],
        "dominant_city_sse": ranked[0][1],
        "dominant_city_sse_share": ranked[0][1] / total_sse
        if total_sse
        else 0.0,
        "top_three_city_sse_share": (
            math.fsum(value for _, value in ranked[:3]) / total_sse
            if total_sse
            else 0.0
        ),
    }


def build_aggregate_metrics(
    prediction_rows: list[dict[str, Any]],
    per_origin_metrics: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    metric_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_origin_metrics:
        metric_grouped[
            (row["target_id"], row["scenario_id"], row["predictor_id"])
        ].append(row)
    prediction_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["scoreable"] == "yes":
            prediction_grouped[
                (row["target_id"], row["scenario_id"], row["predictor_id"])
            ].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_map = {}
    for key in sorted(metric_grouped):
        target_id, scenario_id, predictor_id = key
        origin_rows = sorted(
            metric_grouped[key], key=lambda row: int(row["origin_year"])
        )
        scored_rows = prediction_grouped[key]
        errors = [
            float(row["residual_prediction_minus_actual"])
            for row in scored_rows
        ]
        absolute_errors = [abs(error) for error in errors]
        squared_errors = [error * error for error in errors]
        pooled_mse = mean(squared_errors)
        macro_mse = mean([float(row["mse"]) for row in origin_rows])
        latest = next(
            row for row in origin_rows
            if int(row["origin_year"]) == LATEST_ORIGIN
        )
        origin_mses = [float(row["mse"]) for row in origin_rows]
        dominant = dominant_city_stats(scored_rows)
        early_mses = [
            float(row["mse"])
            for row in origin_rows
            if int(row["origin_year"]) in EARLY_ORIGINS
        ]
        late_mses = [
            float(row["mse"])
            for row in origin_rows
            if int(row["origin_year"]) in LATE_ORIGINS
        ]
        row = {
            "target_id": target_id,
            "scenario_id": scenario_id,
            "predictor_id": predictor_id,
            "origin_domain": "common_strict_2011-2021",
            "origin_count": len(origin_rows),
            "total_scoreable_city_year_count": len(scored_rows),
            "origin_scoreable_city_counts": stable_json(
                {
                    str(origin_row["origin_year"]): int(
                        origin_row["scoreable_city_count"]
                    )
                    for origin_row in origin_rows
                }
            ),
            "scoreable_city_count_equal_across_origins": (
                "yes"
                if len(
                    {
                        int(origin_row["scoreable_city_count"])
                        for origin_row in origin_rows
                    }
                )
                == 1
                else "no"
            ),
            "pooled_city_year_mse": pooled_mse,
            "pooled_city_year_rmse": math.sqrt(pooled_mse),
            "pooled_city_year_mae": mean(absolute_errors),
            "pooled_mean_signed_error_bias": mean(errors),
            "pooled_median_absolute_error": statistics.median(absolute_errors),
            "pooled_competition_score_from_mse": competition_score(pooled_mse),
            "macro_origin_mse": macro_mse,
            "sqrt_macro_origin_mse": math.sqrt(macro_mse),
            "mean_origin_rmse": mean(
                [float(origin_row["rmse"]) for origin_row in origin_rows]
            ),
            "macro_origin_mae": mean(
                [float(origin_row["mae"]) for origin_row in origin_rows]
            ),
            "macro_origin_mean_signed_error_bias": mean(
                [float(origin_row["mean_signed_error_bias"]) for origin_row in origin_rows]
            ),
            "macro_competition_score_from_mse": competition_score(macro_mse),
            "latest_origin_mse": latest["mse"],
            "latest_origin_rmse": latest["rmse"],
            "latest_origin_mae": latest["mae"],
            "latest_origin_mean_signed_error_bias": latest[
                "mean_signed_error_bias"
            ],
            "latest_origin_competition_score_from_mse": competition_score(
                float(latest["mse"])
            ),
            "latest_origin_mse_rank": latest["origin_mse_rank"],
            "mse_mean_across_origins": mean(origin_mses),
            "mse_median_across_origins": statistics.median(origin_mses),
            "mse_std_population_across_origins": statistics.pstdev(origin_mses),
            "mse_min_across_origins": min(origin_mses),
            "mse_max_across_origins": max(origin_mses),
            "best_origin_year": min(
                origin_rows, key=lambda item: float(item["mse"])
            )["origin_year"],
            "worst_origin_year": max(
                origin_rows, key=lambda item: float(item["mse"])
            )["origin_year"],
            "latest_origin_mse_rank_low_is_better": 1
            + sum(
                value < float(latest["mse"]) - 1e-12
                for value in origin_mses
            ),
            "latest_origin_mse_percentile_low_is_better": sum(
                value <= float(latest["mse"]) + 1e-12
                for value in origin_mses
            )
            / len(origin_mses),
            "latest_vs_historical_median_mse_ratio": float(latest["mse"])
            / statistics.median(origin_mses)
            if statistics.median(origin_mses)
            else "",
            "early_origin_mean_mse": mean(early_mses),
            "late_origin_mean_mse": mean(late_mses),
            "late_minus_early_mean_mse": mean(late_mses) - mean(early_mses),
            "origin_winner_count": sum(
                origin_row["origin_mse_rank"] == 1
                for origin_row in origin_rows
            ),
            "dominant_city_id_by_pooled_sse": dominant["dominant_city_id"],
            "dominant_city_pooled_sse_share": dominant[
                "dominant_city_sse_share"
            ],
            "top_three_city_pooled_sse_share": dominant[
                "top_three_city_sse_share"
            ],
            "r1_r2_numeric_score_effect": "none_for_target_history_only",
        }
        aggregate_rows.append(row)
        aggregate_map[key] = row
    assert len(aggregate_rows) == (
        len(TARGETS) * len(SCENARIOS) * len(PREDICTORS)
    )
    return aggregate_rows, aggregate_map


def build_stability_summary(
    per_origin_metrics: list[dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_origin_metrics:
        grouped[
            (row["target_id"], row["scenario_id"], row["predictor_id"])
        ].append(row)
    result = []
    for key in sorted(grouped):
        target_id, scenario_id, predictor_id = key
        rows = sorted(grouped[key], key=lambda row: int(row["origin_year"]))
        aggregate = aggregate_map[key]
        rank_order_by_origin = []
        winner_by_origin = []
        for origin in COMMON_ORIGINS:
            same_origin = [
                row
                for row in per_origin_metrics
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
                and int(row["origin_year"]) == origin
            ]
            ordered = tuple(
                row["predictor_id"]
                for row in sorted(
                    same_origin,
                    key=lambda row: (
                        int(row["origin_mse_rank"]),
                        row["predictor_id"],
                    ),
                )
            )
            rank_order_by_origin.append(f"{origin}:{'>'.join(ordered)}")
            winner_by_origin.append(ordered[0])
        full_flip_count = sum(
            left != right
            for left, right in zip(
                [item.split(":", 1)[1] for item in rank_order_by_origin[:-1]],
                [item.split(":", 1)[1] for item in rank_order_by_origin[1:]],
            )
        )
        winner_flip_count = sum(
            left != right
            for left, right in zip(winner_by_origin[:-1], winner_by_origin[1:])
        )
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "predictor_id": predictor_id,
                "origin_domain": "common_strict_2011-2021",
                "origin_count": len(rows),
                "per_origin_mse_json": json_metric_map(rows, "mse"),
                "per_origin_rmse_json": json_metric_map(rows, "rmse"),
                "per_origin_mae_json": json_metric_map(rows, "mae"),
                "per_origin_bias_json": json_metric_map(
                    rows, "mean_signed_error_bias"
                ),
                "per_origin_score_json": json_metric_map(
                    rows, "competition_score_from_origin_mse"
                ),
                "mse_mean": aggregate["mse_mean_across_origins"],
                "mse_median": aggregate["mse_median_across_origins"],
                "mse_std_population": aggregate[
                    "mse_std_population_across_origins"
                ],
                "mse_min": aggregate["mse_min_across_origins"],
                "mse_max": aggregate["mse_max_across_origins"],
                "best_origin_year": aggregate["best_origin_year"],
                "worst_origin_year": aggregate["worst_origin_year"],
                "latest_origin_year": LATEST_ORIGIN,
                "latest_origin_mse": aggregate["latest_origin_mse"],
                "latest_origin_rank_low_is_better": aggregate[
                    "latest_origin_mse_rank_low_is_better"
                ],
                "latest_origin_percentile_low_is_better": aggregate[
                    "latest_origin_mse_percentile_low_is_better"
                ],
                "latest_origin_is_worst": (
                    "yes"
                    if aggregate["latest_origin_mse_rank_low_is_better"]
                    == len(rows)
                    else "no"
                ),
                "latest_vs_historical_median_mse_ratio": aggregate[
                    "latest_vs_historical_median_mse_ratio"
                ],
                "early_origin_domain": "2011-2015",
                "late_origin_domain": "2017-2021",
                "early_origin_mean_mse": aggregate["early_origin_mean_mse"],
                "late_origin_mean_mse": aggregate["late_origin_mean_mse"],
                "late_minus_early_mean_mse": aggregate[
                    "late_minus_early_mean_mse"
                ],
                "late_vs_early_shift": (
                    "higher"
                    if aggregate["late_minus_early_mean_mse"] > 1e-12
                    else "lower"
                    if aggregate["late_minus_early_mean_mse"] < -1e-12
                    else "same"
                ),
                "predictor_ranking_order_by_origin": ";".join(
                    rank_order_by_origin
                ),
                "winner_by_origin": ";".join(
                    f"{origin}:{winner}"
                    for origin, winner in zip(COMMON_ORIGINS, winner_by_origin)
                ),
                "adjacent_full_ranking_flip_count": full_flip_count,
                "adjacent_winner_flip_count": winner_flip_count,
                "origin_winner_count": aggregate["origin_winner_count"],
                "dominant_city_id_by_pooled_sse": aggregate[
                    "dominant_city_id_by_pooled_sse"
                ],
                "dominant_city_pooled_sse_share": aggregate[
                    "dominant_city_pooled_sse_share"
                ],
                "top_three_city_pooled_sse_share": aggregate[
                    "top_three_city_pooled_sse_share"
                ],
                "r1_r2_numeric_score_effect": "none_for_target_history_only",
            }
        )
    return result


def build_pairwise_comparisons(
    prediction_rows: list[dict[str, Any]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
    per_origin_metric_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str, str], list[dict[str, Any]]],
]:
    prediction_map = {
        (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in prediction_rows
    }
    pairwise_origin_rows = []
    pairwise_summary_rows = []
    row_comparison_map: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)

    for target in TARGETS:
        for scenario_id in SCENARIOS:
            target_id = target["target_id"]
            for pair_id, baseline_id, candidate_id in PAIRWISE_COMPARISONS:
                baseline_aggregate = aggregate_map[
                    (target_id, scenario_id, baseline_id)
                ]
                candidate_aggregate = aggregate_map[
                    (target_id, scenario_id, candidate_id)
                ]
                for origin in COMMON_ORIGINS:
                    baseline_metric = per_origin_metric_map[
                        (target_id, scenario_id, baseline_id, origin)
                    ]
                    candidate_metric = per_origin_metric_map[
                        (target_id, scenario_id, candidate_id, origin)
                    ]
                    comparable_rows = []
                    for city in CITIES:
                        baseline = prediction_map[
                            (
                                target_id,
                                scenario_id,
                                baseline_id,
                                origin,
                                city,
                            )
                        ]
                        candidate = prediction_map[
                            (
                                target_id,
                                scenario_id,
                                candidate_id,
                                origin,
                                city,
                            )
                        ]
                        if (
                            baseline["scoreable"] != "yes"
                            or candidate["scoreable"] != "yes"
                        ):
                            continue
                        baseline_se = float(baseline["squared_error"])
                        candidate_se = float(candidate["squared_error"])
                        se_delta = baseline_se - candidate_se
                        row_result = {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "pair_id": pair_id,
                            "baseline_predictor_id": baseline_id,
                            "candidate_predictor_id": candidate_id,
                            "origin_year": origin,
                            "city_id": city,
                            "baseline_squared_error": baseline_se,
                            "candidate_squared_error": candidate_se,
                            "baseline_minus_candidate_squared_error": se_delta,
                            "row_outcome": (
                                "improved"
                                if se_delta > 1e-12
                                else "worsened"
                                if se_delta < -1e-12
                                else "tied"
                            ),
                        }
                        comparable_rows.append(row_result)
                        row_comparison_map[
                            (target_id, scenario_id, pair_id, city)
                        ].append(row_result)

                    candidate_wins = (
                        float(candidate_metric["mse"])
                        < float(baseline_metric["mse"]) - 1e-12
                    )
                    baseline_wins = (
                        float(baseline_metric["mse"])
                        < float(candidate_metric["mse"]) - 1e-12
                    )
                    pairwise_origin_rows.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "pair_id": pair_id,
                            "baseline_predictor_id": baseline_id,
                            "candidate_predictor_id": candidate_id,
                            "origin_year": origin,
                            "baseline_mse": baseline_metric["mse"],
                            "candidate_mse": candidate_metric["mse"],
                            "mse_absolute_delta_baseline_minus_candidate": (
                                float(baseline_metric["mse"])
                                - float(candidate_metric["mse"])
                            ),
                            "mse_relative_improvement_pct": relative_improvement(
                                float(baseline_metric["mse"]),
                                float(candidate_metric["mse"]),
                            ),
                            "baseline_rmse": baseline_metric["rmse"],
                            "candidate_rmse": candidate_metric["rmse"],
                            "rmse_delta_baseline_minus_candidate": (
                                float(baseline_metric["rmse"])
                                - float(candidate_metric["rmse"])
                            ),
                            "baseline_mae": baseline_metric["mae"],
                            "candidate_mae": candidate_metric["mae"],
                            "mae_delta_baseline_minus_candidate": (
                                float(baseline_metric["mae"])
                                - float(candidate_metric["mae"])
                            ),
                            "baseline_competition_score": baseline_metric[
                                "competition_score_from_origin_mse"
                            ],
                            "candidate_competition_score": candidate_metric[
                                "competition_score_from_origin_mse"
                            ],
                            "competition_score_delta_candidate_minus_baseline": (
                                float(candidate_metric[
                                    "competition_score_from_origin_mse"
                                ])
                                - float(baseline_metric[
                                    "competition_score_from_origin_mse"
                                ])
                            ),
                            "candidate_origin_outcome": (
                                "win"
                                if candidate_wins
                                else "loss"
                                if baseline_wins
                                else "tie"
                            ),
                            "comparable_city_origin_row_count": len(comparable_rows),
                            "improved_city_origin_row_count": sum(
                                row["row_outcome"] == "improved"
                                for row in comparable_rows
                            ),
                            "worsened_city_origin_row_count": sum(
                                row["row_outcome"] == "worsened"
                                for row in comparable_rows
                            ),
                            "tied_city_origin_row_count": sum(
                                row["row_outcome"] == "tied"
                                for row in comparable_rows
                            ),
                            "improved_city_origin_row_share": (
                                sum(
                                    row["row_outcome"] == "improved"
                                    for row in comparable_rows
                                )
                                / len(comparable_rows)
                                if comparable_rows
                                else ""
                            ),
                            "worsened_city_origin_row_share": (
                                sum(
                                    row["row_outcome"] == "worsened"
                                    for row in comparable_rows
                                )
                                / len(comparable_rows)
                                if comparable_rows
                                else ""
                            ),
                            "tied_city_origin_row_share": (
                                sum(
                                    row["row_outcome"] == "tied"
                                    for row in comparable_rows
                                )
                                / len(comparable_rows)
                                if comparable_rows
                                else ""
                            ),
                        }
                    )

                origin_rows = [
                    row
                    for row in pairwise_origin_rows
                    if row["target_id"] == target_id
                    and row["scenario_id"] == scenario_id
                    and row["pair_id"] == pair_id
                ]
                candidate_win_count = sum(
                    row["candidate_origin_outcome"] == "win"
                    for row in origin_rows
                )
                baseline_loss_count = sum(
                    row["candidate_origin_outcome"] == "loss"
                    for row in origin_rows
                )
                tie_count = sum(
                    row["candidate_origin_outcome"] == "tie"
                    for row in origin_rows
                )
                latest_origin_row = next(
                    row for row in origin_rows
                    if int(row["origin_year"]) == LATEST_ORIGIN
                )
                comparable_count = sum(
                    int(row["comparable_city_origin_row_count"])
                    for row in origin_rows
                )
                improved_count = sum(
                    int(row["improved_city_origin_row_count"])
                    for row in origin_rows
                )
                worsened_count = sum(
                    int(row["worsened_city_origin_row_count"])
                    for row in origin_rows
                )
                tied_count = sum(
                    int(row["tied_city_origin_row_count"])
                    for row in origin_rows
                )
                pairwise_summary_rows.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "pair_id": pair_id,
                        "baseline_predictor_id": baseline_id,
                        "candidate_predictor_id": candidate_id,
                        "origin_domain": "common_strict_2011-2021",
                        "baseline_pooled_mse": baseline_aggregate[
                            "pooled_city_year_mse"
                        ],
                        "candidate_pooled_mse": candidate_aggregate[
                            "pooled_city_year_mse"
                        ],
                        "pooled_mse_absolute_delta_baseline_minus_candidate": (
                            float(baseline_aggregate["pooled_city_year_mse"])
                            - float(candidate_aggregate["pooled_city_year_mse"])
                        ),
                        "pooled_mse_relative_improvement_pct": relative_improvement(
                            float(baseline_aggregate["pooled_city_year_mse"]),
                            float(candidate_aggregate["pooled_city_year_mse"]),
                        ),
                        "baseline_pooled_rmse": baseline_aggregate[
                            "pooled_city_year_rmse"
                        ],
                        "candidate_pooled_rmse": candidate_aggregate[
                            "pooled_city_year_rmse"
                        ],
                        "pooled_rmse_delta_baseline_minus_candidate": (
                            float(baseline_aggregate["pooled_city_year_rmse"])
                            - float(candidate_aggregate["pooled_city_year_rmse"])
                        ),
                        "baseline_pooled_mae": baseline_aggregate[
                            "pooled_city_year_mae"
                        ],
                        "candidate_pooled_mae": candidate_aggregate[
                            "pooled_city_year_mae"
                        ],
                        "pooled_mae_delta_baseline_minus_candidate": (
                            float(baseline_aggregate["pooled_city_year_mae"])
                            - float(candidate_aggregate["pooled_city_year_mae"])
                        ),
                        "baseline_pooled_competition_score": baseline_aggregate[
                            "pooled_competition_score_from_mse"
                        ],
                        "candidate_pooled_competition_score": candidate_aggregate[
                            "pooled_competition_score_from_mse"
                        ],
                        "pooled_competition_score_delta_candidate_minus_baseline": (
                            float(candidate_aggregate[
                                "pooled_competition_score_from_mse"
                            ])
                            - float(baseline_aggregate[
                                "pooled_competition_score_from_mse"
                            ])
                        ),
                        "baseline_macro_origin_mse": baseline_aggregate[
                            "macro_origin_mse"
                        ],
                        "candidate_macro_origin_mse": candidate_aggregate[
                            "macro_origin_mse"
                        ],
                        "macro_mse_absolute_delta_baseline_minus_candidate": (
                            float(baseline_aggregate["macro_origin_mse"])
                            - float(candidate_aggregate["macro_origin_mse"])
                        ),
                        "macro_mse_relative_improvement_pct": relative_improvement(
                            float(baseline_aggregate["macro_origin_mse"]),
                            float(candidate_aggregate["macro_origin_mse"]),
                        ),
                        "baseline_macro_competition_score": baseline_aggregate[
                            "macro_competition_score_from_mse"
                        ],
                        "candidate_macro_competition_score": candidate_aggregate[
                            "macro_competition_score_from_mse"
                        ],
                        "macro_competition_score_delta_candidate_minus_baseline": (
                            float(candidate_aggregate[
                                "macro_competition_score_from_mse"
                            ])
                            - float(baseline_aggregate[
                                "macro_competition_score_from_mse"
                            ])
                        ),
                        "baseline_latest_origin_mse": baseline_aggregate[
                            "latest_origin_mse"
                        ],
                        "candidate_latest_origin_mse": candidate_aggregate[
                            "latest_origin_mse"
                        ],
                        "latest_mse_absolute_delta_baseline_minus_candidate": (
                            float(baseline_aggregate["latest_origin_mse"])
                            - float(candidate_aggregate["latest_origin_mse"])
                        ),
                        "latest_mse_relative_improvement_pct": relative_improvement(
                            float(baseline_aggregate["latest_origin_mse"]),
                            float(candidate_aggregate["latest_origin_mse"]),
                        ),
                        "baseline_latest_competition_score": baseline_aggregate[
                            "latest_origin_competition_score_from_mse"
                        ],
                        "candidate_latest_competition_score": candidate_aggregate[
                            "latest_origin_competition_score_from_mse"
                        ],
                        "latest_competition_score_delta_candidate_minus_baseline": (
                            float(candidate_aggregate[
                                "latest_origin_competition_score_from_mse"
                            ])
                            - float(baseline_aggregate[
                                "latest_origin_competition_score_from_mse"
                            ])
                        ),
                        "origin_count": len(origin_rows),
                        "candidate_origin_win_count": candidate_win_count,
                        "candidate_origin_loss_count": baseline_loss_count,
                        "origin_tie_count": tie_count,
                        "candidate_origin_win_rate": candidate_win_count
                        / len(origin_rows),
                        "candidate_origin_loss_rate": baseline_loss_count
                        / len(origin_rows),
                        "candidate_origin_tie_rate": tie_count / len(origin_rows),
                        "latest_origin_candidate_outcome": latest_origin_row[
                            "candidate_origin_outcome"
                        ],
                        "comparable_city_origin_row_count": comparable_count,
                        "improved_city_origin_row_count": improved_count,
                        "worsened_city_origin_row_count": worsened_count,
                        "tied_city_origin_row_count": tied_count,
                        "improved_city_origin_row_share": improved_count
                        / comparable_count
                        if comparable_count
                        else "",
                        "worsened_city_origin_row_share": worsened_count
                        / comparable_count
                        if comparable_count
                        else "",
                        "tied_city_origin_row_share": tied_count / comparable_count
                        if comparable_count
                        else "",
                        "ranking_metric": "pooled_city_year_mse",
                        "score_is_secondary_transform_only": "yes",
                    }
                )
    return pairwise_summary_rows, pairwise_origin_rows, row_comparison_map


def build_improvement_concentration(
    pairwise_summary_rows: list[dict[str, Any]],
    row_comparison_map: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    aggregate_map: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for summary in pairwise_summary_rows:
        target_id = summary["target_id"]
        scenario_id = summary["scenario_id"]
        pair_id = summary["pair_id"]
        baseline_id = summary["baseline_predictor_id"]
        candidate_id = summary["candidate_predictor_id"]
        by_city = {}
        for city in CITIES:
            rows = row_comparison_map[(target_id, scenario_id, pair_id, city)]
            baseline_sse = math.fsum(
                float(row["baseline_squared_error"]) for row in rows
            )
            candidate_sse = math.fsum(
                float(row["candidate_squared_error"]) for row in rows
            )
            delta_sse = baseline_sse - candidate_sse
            by_city[city] = {
                "rows": rows,
                "baseline_sse": baseline_sse,
                "candidate_sse": candidate_sse,
                "delta_sse": delta_sse,
            }
        positive_delta_total = math.fsum(
            max(0.0, value["delta_sse"]) for value in by_city.values()
        )
        total_baseline_sse = math.fsum(
            value["baseline_sse"] for value in by_city.values()
        )
        total_candidate_sse = math.fsum(
            value["candidate_sse"] for value in by_city.values()
        )
        ranked_positive = sorted(
            by_city.items(),
            key=lambda item: (-max(0.0, item[1]["delta_sse"]), item[0]),
        )
        improved_city_count = sum(
            value["delta_sse"] > 1e-12 for value in by_city.values()
        )
        worsened_city_count = sum(
            value["delta_sse"] < -1e-12 for value in by_city.values()
        )
        tied_city_count = len(CITIES) - improved_city_count - worsened_city_count
        dominant_improvement_city = ranked_positive[0][0]
        dominant_improvement_sse = ranked_positive[0][1]["delta_sse"]

        baseline_without_top = math.fsum(
            value["baseline_sse"]
            for city, value in by_city.items()
            if city != dominant_improvement_city
        )
        candidate_without_top = math.fsum(
            value["candidate_sse"]
            for city, value in by_city.items()
            if city != dominant_improvement_city
        )
        comparable_rows_without_top = summary["comparable_city_origin_row_count"] - len(
            by_city[dominant_improvement_city]["rows"]
        )
        baseline_without_top_mse = (
            baseline_without_top / comparable_rows_without_top
            if comparable_rows_without_top
            else ""
        )
        candidate_without_top_mse = (
            candidate_without_top / comparable_rows_without_top
            if comparable_rows_without_top
            else ""
        )
        delta_without_top = (
            baseline_without_top_mse - candidate_without_top_mse
            if comparable_rows_without_top
            else ""
        )

        top1_share = (
            max(0.0, ranked_positive[0][1]["delta_sse"])
            / positive_delta_total
            if positive_delta_total
            else 0.0
        )
        top3_share = (
            math.fsum(
                max(0.0, item[1]["delta_sse"])
                for item in ranked_positive[:3]
            )
            / positive_delta_total
            if positive_delta_total
            else 0.0
        )
        improved_city_share = improved_city_count / len(CITIES)
        if float(summary["pooled_mse_absolute_delta_baseline_minus_candidate"]) <= 0:
            assessment = "no_net_pooled_improvement"
        elif improved_city_share >= 0.50 and top3_share <= 0.60:
            assessment = "broad_based_improvement"
        else:
            assessment = "concentrated_or_mixed_improvement"

        for city, values in sorted(by_city.items()):
            city_rows.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "pair_id": pair_id,
                    "baseline_predictor_id": baseline_id,
                    "candidate_predictor_id": candidate_id,
                    "city_id": city,
                    "comparable_city_origin_row_count": len(values["rows"]),
                    "baseline_city_sse": values["baseline_sse"],
                    "candidate_city_sse": values["candidate_sse"],
                    "baseline_minus_candidate_city_delta_sse": values[
                        "delta_sse"
                    ],
                    "city_outcome": (
                        "improved"
                        if values["delta_sse"] > 1e-12
                        else "worsened"
                        if values["delta_sse"] < -1e-12
                        else "tied"
                    ),
                    "improved_origin_count": sum(
                        row["row_outcome"] == "improved"
                        for row in values["rows"]
                    ),
                    "worsened_origin_count": sum(
                        row["row_outcome"] == "worsened"
                        for row in values["rows"]
                    ),
                    "tied_origin_count": sum(
                        row["row_outcome"] == "tied"
                        for row in values["rows"]
                    ),
                    "improved_origin_share": (
                        sum(
                            row["row_outcome"] == "improved"
                            for row in values["rows"]
                        )
                        / len(values["rows"])
                        if values["rows"]
                        else ""
                    ),
                    "positive_delta_sse_contribution_share": (
                        max(0.0, values["delta_sse"]) / positive_delta_total
                        if positive_delta_total
                        else 0.0
                    ),
                    "baseline_city_sse_share": (
                        values["baseline_sse"] / total_baseline_sse
                        if total_baseline_sse
                        else 0.0
                    ),
                    "candidate_city_sse_share": (
                        values["candidate_sse"] / total_candidate_sse
                        if total_candidate_sse
                        else 0.0
                    ),
                }
            )
        summary_rows.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "pair_id": pair_id,
                "baseline_predictor_id": baseline_id,
                "candidate_predictor_id": candidate_id,
                "pooled_baseline_sse": total_baseline_sse,
                "pooled_candidate_sse": total_candidate_sse,
                "pooled_delta_sse_baseline_minus_candidate": (
                    total_baseline_sse - total_candidate_sse
                ),
                "pooled_mse_absolute_delta_baseline_minus_candidate": summary[
                    "pooled_mse_absolute_delta_baseline_minus_candidate"
                ],
                "improved_city_count": improved_city_count,
                "worsened_city_count": worsened_city_count,
                "tied_city_count": tied_city_count,
                "improved_city_share": improved_city_share,
                "worsened_city_share": worsened_city_count / len(CITIES),
                "positive_delta_sse_total": positive_delta_total,
                "top_improvement_city_id": dominant_improvement_city,
                "top_improvement_city_delta_sse": dominant_improvement_sse,
                "top1_positive_delta_sse_contribution_share": top1_share,
                "top3_positive_delta_sse_contribution_share": top3_share,
                "broad_based_rule": (
                    "pooled improvement > 0 AND improved_city_share >= 0.50 "
                    "AND top3 positive delta share <= 0.60"
                ),
                "broad_vs_concentrated_assessment": assessment,
                "largest_improvement_city_removed_baseline_mse": (
                    baseline_without_top_mse
                ),
                "largest_improvement_city_removed_candidate_mse": (
                    candidate_without_top_mse
                ),
                "largest_improvement_city_removed_delta_mse": delta_without_top,
                "improvement_survives_removing_largest_city": (
                    "yes"
                    if delta_without_top != ""
                    and delta_without_top > 1e-12
                    else "no"
                ),
                "interpretation_scope": "city-origin residual rows; no reweighting",
            }
        )
    return city_rows, summary_rows


def build_r1_r2_equivalence(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_scenario = {
        scenario_id: {
            (
                row["target_id"],
                row["predictor_id"],
                int(row["origin_year"]),
                row["city_id"],
            ): row
            for row in prediction_rows
            if row["scenario_id"] == scenario_id
        }
        for scenario_id in SCENARIOS
    }
    result = []
    for target in TARGETS:
        for predictor in PREDICTORS:
            for origin in COMMON_ORIGINS:
                rows_r1 = [
                    row
                    for key, row in by_scenario[SCENARIOS[0]].items()
                    if key[0] == target["target_id"]
                    and key[1] == predictor["predictor_id"]
                    and key[2] == origin
                ]
                rows_r2 = [
                    row
                    for key, row in by_scenario[SCENARIOS[1]].items()
                    if key[0] == target["target_id"]
                    and key[1] == predictor["predictor_id"]
                    and key[2] == origin
                ]
                assert len(rows_r1) == len(CITIES)
                assert len(rows_r2) == len(CITIES)
                r2_map = {row["city_id"]: row for row in rows_r2}
                prediction_deltas = []
                residual_deltas = []
                squared_error_deltas = []
                exact = True
                for row_r1 in rows_r1:
                    row_r2 = r2_map[row_r1["city_id"]]
                    values_r1 = (
                        row_r1["prediction_value"],
                        row_r1["residual_prediction_minus_actual"],
                        row_r1["squared_error"],
                    )
                    values_r2 = (
                        row_r2["prediction_value"],
                        row_r2["residual_prediction_minus_actual"],
                        row_r2["squared_error"],
                    )
                    if values_r1 != values_r2:
                        exact = False
                    for left, right, output in (
                        (
                            row_r1["prediction_value"],
                            row_r2["prediction_value"],
                            prediction_deltas,
                        ),
                        (
                            row_r1["residual_prediction_minus_actual"],
                            row_r2["residual_prediction_minus_actual"],
                            residual_deltas,
                        ),
                        (
                            row_r1["squared_error"],
                            row_r2["squared_error"],
                            squared_error_deltas,
                        ),
                    ):
                        if left is not None and right is not None:
                            output.append(abs(float(left) - float(right)))
                result.append(
                    {
                        "target_id": target["target_id"],
                        "predictor_id": predictor["predictor_id"],
                        "origin_year": origin,
                        "rows_compared": len(rows_r1),
                        "max_abs_prediction_delta": max(prediction_deltas)
                        if prediction_deltas
                        else 0.0,
                        "max_abs_residual_delta": max(residual_deltas)
                        if residual_deltas
                        else 0.0,
                        "max_abs_squared_error_delta": max(
                            squared_error_deltas
                        )
                        if squared_error_deltas
                        else 0.0,
                        "exact_numeric_equality": "yes" if exact else "no",
                        "status": "PASS" if exact else "FAIL",
                        "scenario_left": SCENARIOS[0],
                        "scenario_right": SCENARIOS[1],
                        "interpretation": (
                            "target-history-only predictors should be numerically "
                            "scenario-invariant; no covariates are used"
                        ),
                    }
                )
    assert all(row["status"] == "PASS" for row in result)
    return result


def build_s2_compatibility(
    prediction_rows: list[dict[str, Any]],
    s2_index: dict[tuple[str, str, str, int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    current = {
        (
            row["target_id"],
            row["scenario_id"],
            row["predictor_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in prediction_rows
        if row["predictor_id"] in {
            "B0_PERSISTENCE_LAST_OBSERVED",
            "B1_TWO_POINT_LINEAR_TREND",
        }
    }
    result = []
    for predictor in PREDICTORS[:2]:
        for target in TARGETS:
            for scenario_id in SCENARIOS:
                for origin in COMMON_ORIGINS:
                    prediction_deltas = []
                    residual_deltas = []
                    squared_error_deltas = []
                    rows_compared = 0
                    exact = True
                    for city in CITIES:
                        s3_row = current[
                            (
                                target["target_id"],
                                scenario_id,
                                predictor["predictor_id"],
                                origin,
                                city,
                            )
                        ]
                        s2_row = s2_index[
                            (
                                target["target_id"],
                                scenario_id,
                                predictor["s2_predictor_id"],
                                origin,
                                city,
                            )
                        ]
                        rows_compared += 1
                        s3_values = (
                            s3_row["prediction_value"],
                            s3_row["residual_prediction_minus_actual"],
                            s3_row["squared_error"],
                        )
                        s2_values = (
                            as_float(s2_row["prediction_value"]),
                            as_float(
                                s2_row["residual_prediction_minus_actual"]
                            ),
                            as_float(s2_row["squared_error"]),
                        )
                        if any(
                            (
                                left is None and right is not None
                            )
                            or (
                                left is not None
                                and right is None
                            )
                            or (
                                left is not None
                                and right is not None
                                and float(left) != float(right)
                            )
                            for left, right in zip(s3_values, s2_values)
                        ):
                            exact = False
                        for left, right, output in (
                            (
                                s3_values[0],
                                s2_values[0],
                                prediction_deltas,
                            ),
                            (
                                s3_values[1],
                                s2_values[1],
                                residual_deltas,
                            ),
                            (
                                s3_values[2],
                                s2_values[2],
                                squared_error_deltas,
                            ),
                        ):
                            if left is not None and right is not None:
                                output.append(abs(float(left) - float(right)))
                    result.append(
                        {
                            "target_id": target["target_id"],
                            "scenario_id": scenario_id,
                            "origin_year": origin,
                            "s3_predictor_id": predictor["predictor_id"],
                            "s2_predictor_id": predictor["s2_predictor_id"],
                            "rows_compared": rows_compared,
                            "max_abs_prediction_delta": max(
                                prediction_deltas
                            )
                            if prediction_deltas
                            else 0.0,
                            "max_abs_residual_delta": max(residual_deltas)
                            if residual_deltas
                            else 0.0,
                            "max_abs_squared_error_delta": max(
                                squared_error_deltas
                            )
                            if squared_error_deltas
                            else 0.0,
                            "exact_numeric_compatibility": (
                                "yes" if exact else "no"
                            ),
                            "status": "PASS" if exact else "FAIL",
                        }
                    )
    assert all(row["status"] == "PASS" for row in result)
    return result


def build_validation_contract_evidence() -> list[dict[str, Any]]:
    return [
        {
            "contract_item": "origin_domain",
            "frozen_value": "2011-2021 common strict origins",
            "scope": "all B0/B1/B2 target x scenario comparisons",
            "status": "PASS",
            "evidence": "per_origin_metrics.csv and origin_information_boundary.csv",
        },
        {
            "contract_item": "target_history_cutoff",
            "frozen_value": "all predictor history rows have year <= T-2",
            "scope": "all prediction and regression fit rows",
            "status": "PASS",
            "evidence": "leakage_audit.csv",
        },
        {
            "contract_item": "primary_target",
            "frozen_value": "resident / 常住人口（万人）",
            "scope": "primary interpretation",
            "status": "PASS",
            "evidence": "aggregate_metrics.csv; target contract inherited from S2",
        },
        {
            "contract_item": "sensitivity_target",
            "frozen_value": "registered / 户籍人口（万人）",
            "scope": "mandatory separate sensitivity",
            "status": "PASS",
            "evidence": "aggregate_metrics.csv",
        },
        {
            "contract_item": "primary_aggregation",
            "frozen_value": "pooled city-year MSE",
            "scope": "ranking metric",
            "status": "PASS",
            "evidence": "aggregate_metrics.csv and pairwise_baseline_comparison.csv",
        },
        {
            "contract_item": "aggregation_sensitivity",
            "frozen_value": "macro-origin MSE",
            "scope": "mandatory sensitivity",
            "status": "PASS",
            "evidence": "aggregate_metrics.csv and pairwise_baseline_comparison.csv",
        },
        {
            "contract_item": "latest_diagnostic",
            "frozen_value": "2021 holdout reported separately",
            "scope": "proximity/stress diagnostic, not sole winner rule",
            "status": "PASS",
            "evidence": "aggregate_metrics.csv and stability_summary.csv",
        },
        {
            "contract_item": "replay_scenarios",
            "frozen_value": "R1 and R2 retained",
            "scope": "source-specific information-set identity",
            "status": "PASS",
            "evidence": "origin_information_boundary.csv and r1_r2_equivalence_check.csv",
        },
    ]


def build_plan_summary(
    population_profile: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    per_origin_metrics: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    pairwise_summary_rows: list[dict[str, Any]],
    concentration_summary_rows: list[dict[str, Any]],
    r1_r2_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "plan_id": "S3-Plan01",
        "stage": "S3_BASELINE_LADDER",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_horizon_years": HORIZON_YEARS,
        "common_origin_domain": "2011-2021",
        "origin_count": len(COMMON_ORIGINS),
        "latest_origin": LATEST_ORIGIN,
        "target_contract_status": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
        "primary_target": "resident / 常住人口（万人）",
        "sensitivity_target": "registered / 户籍人口（万人）",
        "primary_metric": "pooled city-year MSE",
        "aggregation_sensitivity": "macro-origin MSE",
        "population_profile": population_profile,
        "predictor_ids": [predictor["predictor_id"] for predictor in PREDICTORS],
        "predictor_count": len(PREDICTORS),
        "prediction_rows": len(prediction_rows),
        "leakage_audit_rows": len(leakage_rows),
        "leakage_audit_all_passed": all(
            row["leakage_status"] == "PASS" for row in leakage_rows
        ),
        "per_origin_metric_rows": len(per_origin_metrics),
        "aggregate_metric_rows": len(aggregate_rows),
        "pairwise_summary_rows": len(pairwise_summary_rows),
        "concentration_summary_rows": len(concentration_summary_rows),
        "r1_r2_equivalence_rows": len(r1_r2_rows),
        "r1_r2_equivalence_all_passed": all(
            row["status"] == "PASS" for row in r1_r2_rows
        ),
        "s2_b0_b1_compatibility_rows": len(compatibility_rows),
        "s2_b0_b1_compatibility_all_passed": all(
            row["status"] == "PASS" for row in compatibility_rows
        ),
        "validation_contract_all_passed": all(
            row["status"] == "PASS" for row in contract_rows
        ),
        "formal_ml_model_used": False,
        "covariates_used": False,
        "tuning_run": False,
        "ensemble_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "test_xls_used": False,
        "s2_contract_modified": False,
        "s3_complete": False,
        "s4_entered": False,
        "resulting_commit": "recorded_in_HANDOFF_after_artifact_commit",
    }


def main() -> None:
    assert_static_boundaries()
    population, population_profile = load_population()
    s2_summary, s2_boundary, s2_index = load_s2_context()
    load_scenarios()

    prediction_rows, leakage_rows = build_predictions(population)
    per_origin_metrics, per_origin_metric_map = build_per_origin_metrics(
        prediction_rows
    )
    aggregate_rows, aggregate_map = build_aggregate_metrics(
        prediction_rows,
        per_origin_metrics,
    )
    stability_rows = build_stability_summary(
        per_origin_metrics,
        aggregate_map,
    )
    pairwise_summary_rows, pairwise_origin_rows, row_comparison_map = (
        build_pairwise_comparisons(
            prediction_rows,
            aggregate_map,
            per_origin_metric_map,
        )
    )
    concentration_city_rows, concentration_summary_rows = (
        build_improvement_concentration(
            pairwise_summary_rows,
            row_comparison_map,
            aggregate_map,
        )
    )
    r1_r2_rows = build_r1_r2_equivalence(prediction_rows)
    compatibility_rows = build_s2_compatibility(prediction_rows, s2_index)
    boundary_rows = build_s3_boundary_rows(s2_boundary)
    contract_rows = build_validation_contract_evidence()
    summary = build_plan_summary(
        population_profile,
        prediction_rows,
        leakage_rows,
        per_origin_metrics,
        aggregate_rows,
        pairwise_summary_rows,
        concentration_summary_rows,
        r1_r2_rows,
        compatibility_rows,
        contract_rows,
    )

    assert summary["leakage_audit_all_passed"]
    assert summary["r1_r2_equivalence_all_passed"]
    assert summary["s2_b0_b1_compatibility_all_passed"]
    assert summary["validation_contract_all_passed"]
    assert summary["formal_ml_model_used"] is False
    assert summary["covariates_used"] is False
    assert summary["tuning_run"] is False
    assert summary["ensemble_run"] is False
    assert summary["final_2023_forecast_run"] is False
    assert summary["submission_run"] is False
    assert summary["test_xls_used"] is False
    assert summary["s2_contract_modified"] is False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "baseline_predictor_definitions.csv",
        [
            {
                **predictor,
                "plan_id": "S3-Plan01",
                "pre_registered_for_this_plan": "yes",
                "target_history_horizon_years": HORIZON_YEARS,
                "minimum_history_policy": (
                    "unscoreable if fewer than the declared minimum; no imputation"
                ),
                "formal_ml_model": "no",
                "feature_engineering": "no",
                "selection_after_scoring": "no",
            }
            for predictor in PREDICTORS
        ],
    )
    write_csv(
        OUTPUT_DIR / "validation_contract_evidence.csv",
        contract_rows,
    )
    write_csv(
        OUTPUT_DIR / "origin_information_boundary.csv",
        boundary_rows,
    )
    write_csv(
        OUTPUT_DIR / "per_origin_predictions_or_residuals.csv",
        prediction_rows,
    )
    write_csv(
        OUTPUT_DIR / "per_origin_metrics.csv",
        per_origin_metrics,
    )
    write_csv(
        OUTPUT_DIR / "aggregate_metrics.csv",
        aggregate_rows,
    )
    write_csv(
        OUTPUT_DIR / "pairwise_baseline_comparison.csv",
        pairwise_summary_rows,
    )
    write_csv(
        OUTPUT_DIR / "pairwise_origin_comparison.csv",
        pairwise_origin_rows,
    )
    write_csv(
        OUTPUT_DIR / "validation_stability_summary.csv",
        stability_rows,
    )
    write_csv(
        OUTPUT_DIR / "city_improvement_concentration.csv",
        concentration_city_rows,
    )
    write_csv(
        OUTPUT_DIR / "improvement_concentration_summary.csv",
        concentration_summary_rows,
    )
    write_csv(
        OUTPUT_DIR / "leakage_audit.csv",
        leakage_rows,
    )
    write_csv(
        OUTPUT_DIR / "r1_r2_equivalence_check.csv",
        r1_r2_rows,
    )
    write_csv(
        OUTPUT_DIR / "s2_b0_b1_compatibility_check.csv",
        compatibility_rows,
    )
    write_json(OUTPUT_DIR / "plan01_summary.json", summary)

    print("S3_PLAN01_BASELINE_LADDER_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_origin_domain=2011-2021")
    print(f"predictor_count={len(PREDICTORS)}")
    print(f"prediction_rows={len(prediction_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"aggregate_metric_rows={len(aggregate_rows)}")
    print(f"pairwise_summary_rows={len(pairwise_summary_rows)}")
    print(f"leakage_audit_all_passed={summary['leakage_audit_all_passed']}")
    print(f"r1_r2_equivalence_all_passed={summary['r1_r2_equivalence_all_passed']}")
    print(
        "s2_b0_b1_compatibility_all_passed="
        f"{summary['s2_b0_b1_compatibility_all_passed']}"
    )
    print("formal_ml_model_used=False")
    print("covariates_used=False")
    print("tuning_run=False")
    print("ensemble_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
