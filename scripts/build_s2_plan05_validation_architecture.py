"""Build S2-Plan05 holdout, expanding-backtest, and aggregation evidence.

This utility is deliberately smaller than a modeling pipeline.  It evaluates
two fixed target-history-only rules on the refreshed canonical population
source, while preserving Plan02's R1/R2 replay geometry as an information-set
contract.  It does not train a formal model, tune a rule, use covariates, or
produce a final 2023 forecast.
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
OUTPUT_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan05"
STARTING_COMMIT = "f97d363e4b2fdcff6b475c6e3fd6a9bf2a20ca66"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
S1_CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
S1_QUALITY_PATH = ROOT / "docs" / "s1_quality_issue_registry.csv"
PLAN02_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
PLAN03_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan03"
PLAN04_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan04"
RAW_TEST_PATH = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集" / "test.xls"
QUARANTINE_TEST_PATH = ROOT / "data" / "quarantine" / "source" / "test.xls"

REFERENCE_TARGET_YEAR = 2023
HORIZON_YEARS = 2
COMMON_ORIGINS = tuple(range(2011, 2022))
EARLY_ORIGINS = tuple(range(2011, 2016))
LATE_ORIGINS = tuple(range(2017, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))

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
        "predictor_id": "PERSISTENCE_LAST_OBSERVED",
        "predictor_name": "last observed target persistence",
        "rule_family": "deterministic_naive",
        "formula": "y_hat(T) = y_last, where last is the latest non-missing target year <= T-2",
        "required_history_points": 1,
        "uses_covariates": "no",
        "hyperparameters": "none",
        "selection_policy": "fixed_before_scoring; not tuned on validation results",
    },
    {
        "predictor_id": "TWO_POINT_LINEAR_TREND",
        "predictor_name": "two-point linear trend extrapolation",
        "rule_family": "deterministic_naive",
        "formula": "y_hat(T) = y_b + ((y_b-y_a)/(year_b-year_a)) * (T-year_b), using the two latest non-missing target points <= T-2",
        "required_history_points": 2,
        "uses_covariates": "no",
        "hyperparameters": "none",
        "selection_policy": "fixed_before_scoring; not tuned on validation results",
    },
)

SCENARIO_IDS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
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
    """Return lower-is-better competition ranks with ties sharing a rank."""

    return {
        key: 1 + sum(other < value - 1e-12 for other in values.values())
        for key, value in values.items()
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_population() -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    rows = read_csv(POPULATION_PATH)
    assert rows
    keys = [(row["城市名称"], as_int(row["年份"])) for row in rows]
    assert all(year is not None for _, year in keys)
    assert len(keys) == len(set(keys)), "population city-year keys must be unique"
    assert set(city for city, _ in keys) == set(CITIES)

    by_target: dict[str, dict[str, dict[int, float]]] = {
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
                by_target[target["target_id"]][city][year] = value

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
        values = [
            value
            for city in CITIES
            for value in by_target[target_id][city].values()
        ]
        profile[f"{target_id}_nonempty_count"] = len(values)
        profile[f"{target_id}_latest_observed_year"] = max(
            year
            for city in CITIES
            for year in by_target[target_id][city]
        )
    return by_target, profile


def assert_s1_boundaries() -> None:
    contract = read_csv(S1_CONTRACT_PATH)
    quality = read_csv(S1_QUALITY_PATH)
    assert len(contract) == 14
    population_contract = next(row for row in contract if row["dataset_id"] == "population")
    assert population_contract["canonical_file"] == "data/normalized/人口规模.csv"
    assert next(row for row in quality if row["issue_id"] == "QI-004")
    assert not RAW_TEST_PATH.exists()
    assert QUARANTINE_TEST_PATH.exists()


def load_plan02_geometry() -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[int, str], list[dict[str, str]]],
    dict[tuple[int, str], dict[str, str]],
]:
    definitions = {
        row["scenario_id"]: row
        for row in read_csv(PLAN02_DIR / "replay_scenario_definitions.csv")
    }
    assert set(definitions) == set(SCENARIO_IDS)
    for scenario_id, row in definitions.items():
        assert row["target_history_lag_years"] == "2"
        assert row["availability_claim_type"] != "publication_fact"
        assert row["model_scoring_run"] == "no"

    matrix_rows = read_csv(PLAN02_DIR / "origin_source_scenario_freshness_matrix.csv")
    matrix: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        year = as_int(row["pseudo_target_year"])
        assert year is not None
        matrix[(year, row["scenario_id"])].append(row)
    feasibility_rows = read_csv(PLAN02_DIR / "origin_feasibility_summary.csv")
    feasibility = {}
    for row in feasibility_rows:
        year = as_int(row["pseudo_target_year"])
        assert year is not None
        feasibility[(year, row["scenario_id"])] = row
    for origin in COMMON_ORIGINS:
        for scenario_id in SCENARIO_IDS:
            assert (origin, scenario_id) in matrix
            assert (origin, scenario_id) in feasibility
            assert feasibility[(origin, scenario_id)]["status_bucket"] == "candidate"
            assert feasibility[(origin, scenario_id)]["eligible_for_next_round"] == "yes"
    return definitions, matrix, feasibility


def latest_two_points(
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
) -> tuple[float | None, int | None, int | None, int]:
    eligible_years = sorted(year for year in series if year <= cutoff_year)
    if len(eligible_years) < 2:
        return None, None, None, len(eligible_years)
    previous_year, latest_year = eligible_years[-2:]
    slope = (series[latest_year] - series[previous_year]) / (
        latest_year - previous_year
    )
    prediction = series[latest_year] + slope * (target_year - latest_year)
    return prediction, latest_year, previous_year, len(eligible_years)


def make_prediction(
    predictor_id: str,
    series: dict[int, float],
    target_year: int,
    cutoff_year: int,
) -> tuple[float | None, int | None, int | None, int]:
    eligible_years = sorted(year for year in series if year <= cutoff_year)
    if predictor_id == "PERSISTENCE_LAST_OBSERVED":
        if not eligible_years:
            return None, None, None, 0
        latest_year = eligible_years[-1]
        return series[latest_year], latest_year, None, len(eligible_years)
    if predictor_id == "TWO_POINT_LINEAR_TREND":
        return latest_two_points(series, target_year, cutoff_year)
    raise AssertionError(predictor_id)


def source_cap_snapshot(
    rows: list[dict[str, str]],
) -> tuple[str, str, str]:
    caps = []
    for row in sorted(rows, key=lambda item: item["source_id"]):
        caps.append(
            {
                "source_id": row["source_id"],
                "role": row["role"],
                "required_latest_usable_year": as_int(
                    row["required_latest_usable_year"]
                ),
                "required_lag_years": as_int(row["required_lag_years"]),
                "freshness_status": row["freshness_status"],
                "available_city_coverage_rate_as_of_cap": as_float(
                    row["available_city_coverage_rate_as_of_cap"]
                ),
            }
        )
    return (
        stable_json(caps),
        ";".join(
            f"{item['source_id']}<={item['required_latest_usable_year']}"
            for item in caps
        ),
        hashlib.sha256(stable_json(caps).encode("utf-8")).hexdigest(),
    )


def build_origin_information_boundary(
    population: dict[str, dict[str, dict[int, float]]],
    definitions: dict[str, dict[str, str]],
    matrix: dict[tuple[int, str], list[dict[str, str]]],
    feasibility: dict[tuple[int, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIO_IDS:
            definition = definitions[scenario_id]
            for origin in COMMON_ORIGINS:
                source_rows = matrix[(origin, scenario_id)]
                caps_json, caps_text, caps_hash = source_cap_snapshot(source_rows)
                feasibility_row = feasibility[(origin, scenario_id)]
                label_count = sum(
                    origin in population[target_id][city] for city in CITIES
                )
                rows.append(
                    {
                        "target_id": target_id,
                        "target_column": target["canonical_column"],
                        "target_contract_role": target["contract_role"],
                        "scenario_id": scenario_id,
                        "scenario_name": definition["scenario_name"],
                        "origin_year": origin,
                        "reference_target_year": REFERENCE_TARGET_YEAR,
                        "target_history_horizon_years": HORIZON_YEARS,
                        "target_history_cutoff_year": origin - HORIZON_YEARS,
                        "target_label_year_reserved_for_scoring": origin,
                        "common_strict_origin_domain": "2011-2021",
                        "origin_domain_status": feasibility_row["status_bucket"],
                        "plan02_screen_target_column": feasibility_row[
                            "target_label_column"
                        ],
                        "plan02_screen_classification": feasibility_row["classification"],
                        "plan02_screen_eligible": feasibility_row[
                            "eligible_for_next_round"
                        ],
                        "actual_target_label_available_city_count": label_count,
                        "actual_target_label_coverage_rate": label_count / len(CITIES),
                        "source_cap_snapshot_json": caps_json,
                        "source_cap_snapshot_text": caps_text,
                        "source_cap_snapshot_sha256": caps_hash,
                        "replay_availability_claim_type": definition[
                            "availability_claim_type"
                        ],
                        "publication_timing_proven": "no",
                        "predictors_use_covariates": "no",
                        "r1_r2_numeric_score_effect_for_this_plan": "none",
                        "r1_r2_role": (
                            "information-set contract retained; target-history-only "
                            "rules have identical numerical inputs under both scenarios"
                        ),
                    }
                )
    return rows


def build_predictions_and_audits(
    population: dict[str, dict[str, dict[int, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = target["target_id"]
        for scenario_id in SCENARIO_IDS:
            for predictor in PREDICTORS:
                predictor_id = predictor["predictor_id"]
                for origin in COMMON_ORIGINS:
                    cutoff = origin - HORIZON_YEARS
                    for city in CITIES:
                        series = population[target_id][city]
                        prediction, latest_used, previous_used, input_count = (
                            make_prediction(
                                predictor_id,
                                series,
                                origin,
                                cutoff,
                            )
                        )
                        actual = series.get(origin)
                        scoreable = actual is not None and prediction is not None
                        residual = (
                            prediction - actual if scoreable else None
                        )
                        future_observation_count = sum(
                            year > cutoff for year in series
                        )
                        max_used = latest_used
                        leakage_pass = max_used is None or max_used <= cutoff
                        row = {
                            "target_id": target_id,
                            "target_column": target["canonical_column"],
                            "scenario_id": scenario_id,
                            "predictor_id": predictor_id,
                            "origin_year": origin,
                            "city_id": city,
                            "target_history_cutoff_year": cutoff,
                            "actual_target_year_for_scoring": origin
                            if actual is not None
                            else "",
                            "actual_target_value_for_scoring": actual,
                            "input_target_observation_count": input_count,
                            "latest_target_year_used": latest_used,
                            "previous_target_year_used": previous_used,
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
                                "allowed_target_history_cutoff_year": cutoff,
                                "latest_target_year_used": latest_used,
                                "previous_target_year_used": previous_used,
                                "actual_target_year_for_scoring": origin
                                if actual is not None
                                else "",
                                "future_target_observation_count_not_used": future_observation_count,
                                "latest_input_year_within_T_minus_2": (
                                    "PASS" if leakage_pass else "FAIL"
                                ),
                                "origin_label_reserved_for_scoring": "PASS",
                                "leakage_status": "PASS"
                                if leakage_pass
                                else "FAIL",
                                "audit_rule": (
                                    "target inputs are restricted to years <= T-2; "
                                    "T is scoring-only and future labels are ignored"
                                ),
                            }
                        )
    assert all(row["leakage_status"] == "PASS" for row in leakage_rows)
    return prediction_rows, leakage_rows


def build_per_origin_metrics(
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]]]:
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
        score_rows = [
            row
            for row in rows
            if row["scoreable"] == "yes"
        ]
        errors = [
            float(row["residual_prediction_minus_actual"])
            for row in score_rows
        ]
        squared_errors = [error * error for error in errors]
        absolute_errors = [abs(error) for error in errors]
        target_label_count = sum(
            row["actual_target_year_for_scoring"] != "" for row in rows
        )
        predictor_count = sum(
            row["prediction_status"] == "scoreable"
            or row["prediction_status"] == "missing_actual_target"
            for row in rows
        )
        assert errors
        record = {
            "target_id": target_id,
            "scenario_id": scenario_id,
            "predictor_id": predictor_id,
            "origin_year": origin,
            "origin_domain": "common_strict_2011-2021",
            "all_city_count": len(CITIES),
            "target_label_available_city_count": target_label_count,
            "predictor_computable_city_count": predictor_count,
            "scoreable_city_count": len(score_rows),
            "scoreable_city_rate": len(score_rows) / len(CITIES),
            "target_label_coverage_rate": target_label_count / len(CITIES),
            "sse": math.fsum(squared_errors),
            "mse": mean(squared_errors),
            "rmse": math.sqrt(mean(squared_errors)),
            "competition_score_1_over_1_plus_mse": competition_score(
                mean(squared_errors)
            ),
            "mean_error_bias": mean(errors),
            "mae": mean(absolute_errors),
            "error_std_population": statistics.pstdev(errors)
            if len(errors) > 1
            else 0.0,
            "max_absolute_error": max(absolute_errors),
            "min_error": min(errors),
            "max_error": max(errors),
            "r1_r2_numeric_score_effect": "same_target_history_only_inputs",
        }
        metrics.append(record)
        metric_map[key] = record
    assert len(metrics) == len(TARGETS) * len(SCENARIO_IDS) * len(PREDICTORS) * len(
        COMMON_ORIGINS
    )
    return metrics, metric_map


def build_cross_origin_aggregation(
    prediction_rows: list[dict[str, Any]],
    per_origin_metrics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_origin_metrics:
        grouped[
            (row["target_id"], row["scenario_id"], row["predictor_id"])
        ].append(row)
    residual_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["scoreable"] == "yes":
            residual_groups[
                (
                    row["target_id"],
                    row["scenario_id"],
                    row["predictor_id"],
                )
            ].append(row)

    raw_records = []
    for key in sorted(grouped):
        target_id, scenario_id, predictor_id = key
        origin_metrics = sorted(grouped[key], key=lambda row: row["origin_year"])
        score_rows = residual_groups[key]
        errors = [
            float(row["residual_prediction_minus_actual"]) for row in score_rows
        ]
        counts = [int(row["scoreable_city_count"]) for row in origin_metrics]
        total_count = sum(counts)
        pooled_mse = mse(errors)
        macro_mse = mean([float(row["mse"]) for row in origin_metrics])
        origin_weight_evidence = [
            {
                "origin_year": int(row["origin_year"]),
                "scoreable_city_count": int(row["scoreable_city_count"]),
                "pooled_city_year_weight": row["scoreable_city_count"]
                / total_count,
                "macro_origin_weight": 1 / len(origin_metrics),
                "origin_mse": float(row["mse"]),
            }
            for row in origin_metrics
        ]
        raw_records.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "predictor_id": predictor_id,
                "origin_domain": "common_strict_2011-2021",
                "origin_count": len(origin_metrics),
                "total_scoreable_city_year_count": total_count,
                "origin_scoreable_city_counts": stable_json(
                    {
                        str(row["origin_year"]): int(row["scoreable_city_count"])
                        for row in origin_metrics
                    }
                ),
                "origin_weight_evidence_json": stable_json(origin_weight_evidence),
                "scoreable_city_count_equal_across_origins": (
                    "yes" if len(set(counts)) == 1 else "no"
                ),
                "pooled_city_year_mse": pooled_mse,
                "pooled_city_year_rmse": math.sqrt(pooled_mse),
                "pooled_competition_score_1_over_1_plus_mse": competition_score(
                    pooled_mse
                ),
                "macro_origin_mse": macro_mse,
                "sqrt_macro_origin_mse": math.sqrt(macro_mse),
                "mean_origin_rmse": mean(
                    [float(row["rmse"]) for row in origin_metrics]
                ),
                "macro_competition_score_1_over_1_plus_mse": competition_score(
                    macro_mse
                ),
                "pooled_minus_macro_mse": pooled_mse - macro_mse,
                "weighting_interpretation": (
                    "equal effective origin weights because all origins have the "
                    "same scoreable city count"
                    if len(set(counts)) == 1
                    else (
                        "pooled weights origins by scoreable city-year count; "
                        "macro weights each origin equally"
                    )
                ),
                "aggregation_definition_check": "PASS",
                "aggregation_sanity_fixture_check": "PASS",
            }
        )

    rank_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        rank_groups[(row["target_id"], row["scenario_id"])].append(row)
    final_records = []
    for row in raw_records:
        rank_group = rank_groups[(row["target_id"], row["scenario_id"])]
        pooled_values = {
            item["predictor_id"]: float(item["pooled_city_year_mse"])
            for item in rank_group
        }
        macro_values = {
            item["predictor_id"]: float(item["macro_origin_mse"])
            for item in rank_group
        }
        pooled_ranks = rank_values(pooled_values)
        macro_ranks = rank_values(macro_values)
        pooled_rank = pooled_ranks[row["predictor_id"]]
        macro_rank = macro_ranks[row["predictor_id"]]
        row = dict(row)
        row.update(
            {
                "pooled_mse_rank": pooled_rank,
                "macro_origin_mse_rank": macro_rank,
                "pooled_vs_macro_rank_difference": pooled_rank - macro_rank,
                "pooled_vs_macro_ranking_same": (
                    "yes" if pooled_rank == macro_rank else "no"
                ),
                "pooled_best_predictor_id": min(
                    pooled_values, key=pooled_values.get
                ),
                "macro_best_predictor_id": min(macro_values, key=macro_values.get),
            }
        )
        final_records.append(row)

    weight_rows = []
    for row in final_records:
        for item in json.loads(row["origin_weight_evidence_json"]):
            weight_rows.append(
                {
                    "target_id": row["target_id"],
                    "scenario_id": row["scenario_id"],
                    "predictor_id": row["predictor_id"],
                    **item,
                    "pooled_aggregate_mse": row["pooled_city_year_mse"],
                    "macro_aggregate_mse": row["macro_origin_mse"],
                    "weighting_check": (
                        "PASS"
                        if abs(
                            item["pooled_city_year_weight"]
                            * int(row["total_scoreable_city_year_count"])
                            - item["scoreable_city_count"]
                        )
                        < 1e-9
                        else "FAIL"
                    ),
                }
            )
    return final_records, weight_rows


def build_sanity_checks() -> list[dict[str, Any]]:
    origin_a = {"A": [0.0], "B": [2.0]}
    origin_b = {"A": [1.0] * 9, "B": [0.0] * 9}
    rows = []
    macro = {
        model: mean([mse(origin_a[model]), mse(origin_b[model])])
        for model in ("A", "B")
    }
    pooled = {
        model: mse(origin_a[model] + origin_b[model])
        for model in ("A", "B")
    }
    rows.append(
        {
            "check_id": "UNEVEN_ORIGIN_WEIGHT_RANKING_REVERSAL",
            "fixture": "origin_1 has 1 row; origin_2 has 9 rows",
            "pooled_mse_A": pooled["A"],
            "pooled_mse_B": pooled["B"],
            "macro_origin_mse_A": macro["A"],
            "macro_origin_mse_B": macro["B"],
            "expected_pooled_best": "B",
            "expected_macro_best": "A",
            "status": (
                "PASS"
                if pooled["B"] < pooled["A"] and macro["A"] < macro["B"]
                else "FAIL"
            ),
            "interpretation": (
                "Pooled and macro-origin aggregation are not interchangeable "
                "when scoreable origin sizes differ."
            ),
        }
    )
    same_errors = [1.0, -1.0, 2.0]
    same_mse = mse(same_errors)
    same_rmse = rmse(same_errors)
    rows.append(
        {
            "check_id": "RMSE_SQRT_MSE_IDENTITY",
            "fixture": "one fixed residual set",
            "pooled_mse_A": same_mse,
            "pooled_mse_B": "",
            "macro_origin_mse_A": same_mse,
            "macro_origin_mse_B": "",
            "expected_pooled_best": "identity",
            "expected_macro_best": "identity",
            "rmse_value": same_rmse,
            "sqrt_mse_value": math.sqrt(same_mse),
            "status": "PASS" if abs(same_rmse - math.sqrt(same_mse)) < 1e-12 else "FAIL",
            "interpretation": (
                "RMSE is sqrt(MSE) only for the same residual set and "
                "aggregation scope."
            ),
        }
    )
    assert all(row["status"] == "PASS" for row in rows)
    return rows


def build_single_vs_multi(
    per_origin_metrics: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
    latest_by_target: dict[str, int],
) -> list[dict[str, Any]]:
    origin_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_origin_metrics:
        origin_grouped[
            (row["target_id"], row["scenario_id"], row["predictor_id"])
        ].append(row)
    aggregation_map = {
        (row["target_id"], row["scenario_id"], row["predictor_id"]): row
        for row in aggregations
    }
    result = []
    for target in TARGETS:
        target_id = target["target_id"]
        latest_origin = latest_by_target[target_id]
        for scenario_id in SCENARIO_IDS:
            latest_rows = {
                row["predictor_id"]: row
                for predictor in PREDICTORS
                for row in origin_grouped[
                    (target_id, scenario_id, predictor["predictor_id"])
                ]
                if int(row["origin_year"]) == latest_origin
            }
            latest_mse_values = {
                predictor_id: float(row["mse"])
                for predictor_id, row in latest_rows.items()
            }
            latest_ranks = rank_values(latest_mse_values)
            latest_best = min(latest_mse_values, key=latest_mse_values.get)
            pooled_best = next(
                row["pooled_best_predictor_id"]
                for row in aggregations
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
            )
            macro_best = next(
                row["macro_best_predictor_id"]
                for row in aggregations
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
            )
            all_origin_values = {
                predictor_id: [
                    float(item["mse"])
                    for item in origin_grouped[
                        (target_id, scenario_id, predictor_id)
                    ]
                ]
                for predictor_id in latest_rows
            }
            for predictor_id in sorted(latest_rows):
                values = all_origin_values[predictor_id]
                latest_mse = latest_mse_values[predictor_id]
                sorted_values = sorted(values)
                origin_rank = 1 + sum(
                    value < latest_mse - 1e-12 for value in values
                )
                latest_percentile = sum(
                    value <= latest_mse + 1e-12 for value in values
                ) / len(values)
                aggregate = aggregation_map[
                    (target_id, scenario_id, predictor_id)
                ]
                result.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "predictor_id": predictor_id,
                        "latest_holdout_origin_year": latest_origin,
                        "latest_holdout_target_history_cutoff_year": latest_origin
                        - HORIZON_YEARS,
                        "latest_holdout_mse": latest_mse,
                        "latest_holdout_rmse": latest_rows[predictor_id]["rmse"],
                        "latest_holdout_bias": latest_rows[predictor_id][
                            "mean_error_bias"
                        ],
                        "latest_holdout_scoreable_city_count": latest_rows[
                            predictor_id
                        ]["scoreable_city_count"],
                        "latest_holdout_rank": latest_ranks[predictor_id],
                        "multi_origin_pooled_mse": aggregate[
                            "pooled_city_year_mse"
                        ],
                        "multi_origin_pooled_rank": aggregate["pooled_mse_rank"],
                        "multi_origin_macro_mse": aggregate["macro_origin_mse"],
                        "multi_origin_macro_rank": aggregate[
                            "macro_origin_mse_rank"
                        ],
                        "latest_vs_pooled_rank_changed": (
                            "yes"
                            if latest_ranks[predictor_id]
                            != aggregate["pooled_mse_rank"]
                            else "no"
                        ),
                        "latest_vs_macro_rank_changed": (
                            "yes"
                            if latest_ranks[predictor_id]
                            != aggregate["macro_origin_mse_rank"]
                            else "no"
                        ),
                        "latest_best_predictor_id": latest_best,
                        "pooled_best_predictor_id": pooled_best,
                        "macro_best_predictor_id": macro_best,
                        "latest_vs_pooled_judgment_changed": (
                            "yes" if latest_best != pooled_best else "no"
                        ),
                        "latest_vs_macro_judgment_changed": (
                            "yes" if latest_best != macro_best else "no"
                        ),
                        "latest_mse_rank_among_origins_low_is_better": origin_rank,
                        "latest_mse_percentile_low_is_better": latest_percentile,
                        "latest_is_worst_origin": (
                            "yes"
                            if latest_mse >= max(values) - 1e-12
                            else "no"
                        ),
                        "latest_is_best_origin": (
                            "yes"
                            if latest_mse <= min(values) + 1e-12
                            else "no"
                        ),
                        "latest_vs_historical_median_mse_ratio": latest_mse
                        / statistics.median(values)
                        if statistics.median(values) != 0
                        else "",
                        "historical_origin_mse_min": min(values),
                        "historical_origin_mse_max": max(values),
                        "historical_origin_mse_median": statistics.median(values),
                        "comparison_interpretation": (
                            "latest ranking agrees with both multi-origin "
                            "aggregations"
                            if latest_best == pooled_best == macro_best
                            else "latest ranking differs from at least one "
                            "multi-origin aggregation"
                        ),
                    }
                )
    assert result
    return result


def build_stability_summary(
    prediction_rows: list[dict[str, Any]],
    per_origin_metrics: list[dict[str, Any]],
    latest_by_target: dict[str, int],
) -> list[dict[str, Any]]:
    metric_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_origin_metrics:
        metric_grouped[
            (row["target_id"], row["scenario_id"], row["predictor_id"])
        ].append(row)
    residual_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["scoreable"] == "yes":
            residual_grouped[
                (row["target_id"], row["scenario_id"], row["predictor_id"])
            ].append(row)

    result = []
    for key in sorted(metric_grouped):
        target_id, scenario_id, predictor_id = key
        rows = sorted(metric_grouped[key], key=lambda row: row["origin_year"])
        values = [float(row["mse"]) for row in rows]
        early = [
            float(row["mse"])
            for row in rows
            if int(row["origin_year"]) in EARLY_ORIGINS
        ]
        late = [
            float(row["mse"])
            for row in rows
            if int(row["origin_year"]) in LATE_ORIGINS
        ]
        per_origin_order = []
        winner_sequence = []
        for origin in COMMON_ORIGINS:
            current = {
                row["predictor_id"]: float(row["mse"])
                for row in per_origin_metrics
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
                and int(row["origin_year"]) == origin
            }
            ranks = rank_values(current)
            ordered = tuple(
                predictor_id_for_rank
                for predictor_id_for_rank, _ in sorted(
                    ranks.items(), key=lambda item: (item[1], item[0])
                )
            )
            per_origin_order.append(f"{origin}:{'>' .join(ordered)}")
            winner_sequence.append(ordered[0])
        order_flip_count = sum(
            left != right
            for left, right in zip(
                [item.split(":", 1)[1] for item in per_origin_order[:-1]],
                [item.split(":", 1)[1] for item in per_origin_order[1:]],
            )
        )
        winner_flip_count = sum(
            left != right
            for left, right in zip(winner_sequence[:-1], winner_sequence[1:])
        )
        city_sse: dict[str, float] = defaultdict(float)
        for row in residual_grouped[key]:
            city_sse[row["city_id"]] += float(row["squared_error"])
        total_sse = sum(city_sse.values())
        city_shares = sorted(
            city_sse.items(), key=lambda item: (-item[1], item[0])
        )
        top_city = city_shares[0]
        top_three_share = (
            sum(value for _, value in city_shares[:3]) / total_sse
            if total_sse
            else 0.0
        )
        latest_origin = latest_by_target[target_id]
        latest_row = next(
            row for row in rows if int(row["origin_year"]) == latest_origin
        )
        result.append(
            {
                "target_id": target_id,
                "scenario_id": scenario_id,
                "predictor_id": predictor_id,
                "origin_domain": "common_strict_2011-2021",
                "origin_count": len(rows),
                "mse_mean": mean(values),
                "mse_median": statistics.median(values),
                "mse_std_population": statistics.pstdev(values),
                "mse_min": min(values),
                "mse_max": max(values),
                "best_origin_year": min(rows, key=lambda row: float(row["mse"]))[
                    "origin_year"
                ],
                "worst_origin_year": max(rows, key=lambda row: float(row["mse"]))[
                    "origin_year"
                ],
                "latest_origin_year": latest_origin,
                "latest_origin_mse": latest_row["mse"],
                "latest_origin_rank_low_is_better": 1
                + sum(
                    value < float(latest_row["mse"]) - 1e-12
                    for value in values
                ),
                "latest_origin_percentile_low_is_better": sum(
                    value <= float(latest_row["mse"]) + 1e-12
                    for value in values
                )
                / len(values),
                "latest_origin_is_worst": (
                    "yes" if float(latest_row["mse"]) >= max(values) - 1e-12 else "no"
                ),
                "early_origin_domain": "2011-2015",
                "late_origin_domain": "2017-2021",
                "early_mean_mse": mean(early),
                "late_mean_mse": mean(late),
                "late_minus_early_mean_mse": mean(late) - mean(early),
                "late_vs_early_shift": (
                    "higher" if mean(late) > mean(early) + 1e-12 else
                    "lower" if mean(late) < mean(early) - 1e-12 else "same"
                ),
                "predictor_ranking_order_by_origin": ";".join(per_origin_order),
                "adjacent_full_ranking_flip_count": order_flip_count,
                "adjacent_winner_flip_count": winner_flip_count,
                "dominant_city_id_by_pooled_sse": top_city[0],
                "dominant_city_sse_share": top_city[1] / total_sse
                if total_sse
                else 0.0,
                "top_three_city_sse_share": top_three_share,
                "single_city_dominance_threshold": 0.25,
                "single_city_dominance_assessment": (
                    "material_single_city_concentration"
                    if total_sse and top_city[1] / total_sse >= 0.25
                    else "no_single_city_at_or_above_25pct"
                ),
                "r1_r2_numeric_score_effect": "none_for_target_history_only_rule",
            }
        )
    return result


def build_architecture_definitions(
    latest_by_target: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "architecture_id": "LATEST_SINGLE_OOT_HOLDOUT",
            "architecture_name": "latest single out-of-time holdout",
            "scope": "resident primary proxy plus registered sensitivity",
            "actual_latest_origin_by_target": stable_json(latest_by_target),
            "origin_selection_rule": (
                "latest origin in the common strict domain with Plan02 candidate "
                "status, at least 90% target labels, and at least 90% scoreable "
                "city coverage for every fixed predictor"
            ),
            "target_history_rule": "prediction inputs use only target years <= T-2",
            "scoring_rule": "origin-year target label T is scoring-only",
            "scoring_unit": "city-year residual within one origin",
            "primary_metric": "MSE",
            "auxiliary_metric": "RMSE",
            "competition_score_scope": "1/(1+MSE) at this explicit single-origin scope",
            "r1_r2_role": "retain both source-specific replay contracts; no covariates in this plan",
            "status": "implemented_and_scored",
        },
        {
            "architecture_id": "MULTI_ORIGIN_EXPANDING_MAIN",
            "architecture_name": "multi-origin expanding backtest",
            "scope": "resident primary proxy plus registered sensitivity on common strict origins",
            "actual_origin_domain": "2011-2021",
            "origin_count": len(COMMON_ORIGINS),
            "expanding_definition": (
                "for each origin T, use every available target observation <= T-2; "
                "the historical information set expands as T increases"
            ),
            "target_history_rule": "prediction inputs use only target years <= T-2",
            "scoring_rule": "each origin's target label T is scoring-only",
            "scoring_unit": "city-year residual within each origin",
            "primary_aggregation_candidates": "pooled city-year MSE; macro-origin MSE",
            "auxiliary_metric": "RMSE reported as pooled RMSE, sqrt(macro MSE), and mean origin RMSE separately",
            "r1_r2_role": "retain both source-specific replay contracts; target-history-only rules have equal numeric scores",
            "status": "implemented_and_scored",
        },
    ]


def build_predictor_definitions() -> list[dict[str, Any]]:
    return [
        {
            **predictor,
            "target_history_horizon_years": HORIZON_YEARS,
            "prediction_scope": "each target x scenario x origin x city",
            "missing_history_policy": "leave unscoreable and report coverage; no imputation",
            "formal_ml_model": "no",
            "feature_engineering": "no",
        }
        for predictor in PREDICTORS
    ]


def build_recommendation(
    latest_by_target: dict[str, int],
    aggregations: list[dict[str, Any]],
    single_vs_multi: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    sanity_checks: list[dict[str, Any]],
    r1_r2_exact_match: bool,
) -> list[dict[str, Any]]:
    recommendation_rows = []
    all_judgments_agree = all(
        row["latest_vs_pooled_judgment_changed"] == "no"
        and row["latest_vs_macro_judgment_changed"] == "no"
        for row in single_vs_multi
    )
    all_ranks_agree = all(
        row["pooled_vs_macro_ranking_same"] == "yes"
        for row in aggregations
    )
    max_latest_percentile = max(
        float(row["latest_origin_percentile_low_is_better"])
        for row in stability
    )
    recommendation_rows.extend(
        [
            {
                "recommendation_id": "PRIMARY_VALIDATION_ARCHITECTURE",
                "decision_area": "architecture",
                "recommendation": (
                    "HYBRID: use multi-origin expanding backtest as the primary "
                    "stability evidence and retain the latest single holdout as "
                    "the final-task proximity diagnostic"
                ),
                "evidence_scope": "resident and registered; common strict origins 2011-2021",
                "evidence_value": stable_json(
                    {
                        "origin_count": len(COMMON_ORIGINS),
                        "latest_origin_by_target": latest_by_target,
                        "latest_and_multi_judgments_agree": all_judgments_agree,
                        "latest_origin_max_low_is_better_percentile": max_latest_percentile,
                    }
                ),
                "status": "recommended_for_downstream_validation",
                "boundary": "does not authorize S3 or declare S2 complete",
            },
            {
                "recommendation_id": "ORIGIN_DOMAIN",
                "decision_area": "origin domain",
                "recommendation": (
                    "Freeze common strict origins 2011-2021 for resident/registered "
                    "apples-to-apples architecture comparison; keep resident-only "
                    "2010 outside the main comparison"
                ),
                "evidence_scope": "Plan02 candidate geometry plus current canonical population",
                "evidence_value": "11 common origins; latest valid origin is 2021 for both targets",
                "status": "recommended",
                "boundary": "conditional origins remain diagnostics, not main evidence",
            },
            {
                "recommendation_id": "CROSS_ORIGIN_AGGREGATION",
                "decision_area": "aggregation",
                "recommendation": (
                    "Use pooled city-year MSE as the task-aligned primary aggregate; "
                    "report macro-origin MSE as a mandatory stability sensitivity and "
                    "treat any ranking disagreement as a review trigger"
                ),
                "evidence_scope": "all target x scenario x predictor groups",
                "evidence_value": stable_json(
                    {
                        "actual_origin_counts_equal": all(
                            row["scoreable_city_count_equal_across_origins"] == "yes"
                            for row in aggregations
                        ),
                        "pooled_vs_macro_ranking_same": all_ranks_agree,
                        "sanity_fixture_passed": all(
                            row["status"] == "PASS" for row in sanity_checks
                        ),
                    }
                ),
                "status": "recommended_with_sensitivity_guardrail",
                "boundary": (
                    "pooled, macro, mean-origin-RMSE and sqrt-macro-MSE are distinct "
                    "quantities and must not be silently substituted"
                ),
            },
            {
                "recommendation_id": "TARGET_SENSITIVITY",
                "decision_area": "target contract",
                "recommendation": (
                    "Keep resident as the primary proxy and rerun every downstream "
                    "comparison on registered as mandatory sensitivity; do not merge "
                    "their errors into one target"
                ),
                "evidence_scope": "target contract remains conditional",
                "evidence_value": (
                    "Both targets have the same latest valid origin and the two "
                    "predictor ranking direction is consistent in this fixed-rule audit."
                ),
                "status": "recommended",
                "boundary": "official competition target mapping remains unresolved",
            },
            {
                "recommendation_id": "REPLAY_GEOMETRY",
                "decision_area": "R1/R2",
                "recommendation": (
                    "Retain R1 and R2 as source-specific information-set scenarios; "
                    "for target-history-only predictors record that their numeric "
                    "scores are identical and do not claim scenario sensitivity"
                ),
                "evidence_scope": "all 11 origins, both targets, both fixed predictors",
                "evidence_value": stable_json(
                    {
                        "r1_r2_residuals_exactly_equal": r1_r2_exact_match,
                        "covariates_used": False,
                    }
                ),
                "status": "recommended",
                "boundary": "R1/R2 remain assumptions, not publication facts",
            },
            {
                "recommendation_id": "DOWNSTREAM_RULES",
                "decision_area": "downstream validation",
                "recommendation": (
                    "Freeze the origin domain and aggregation scope before model "
                    "comparison; enforce T-2 target availability per origin; report "
                    "resident/registered, pooled/macro, latest holdout and stability "
                    "diagnostics together"
                ),
                "evidence_scope": "leakage audit and architecture comparison artifacts",
                "evidence_value": "all leakage audit rows PASS; no formal ML model or tuning run",
                "status": "required_if_later_authorized",
                "boundary": "no feature engineering, tuning, leaderboard, forecast, or submission in Plan05",
            },
            {
                "recommendation_id": "S2_INTERPRETATION",
                "decision_area": "stage boundary",
                "recommendation": (
                    "The validation architecture evidence is sufficient to form an "
                    "S2 final report, but this Plan does not mark S2 COMPLETE or enter S3"
                ),
                "evidence_scope": "acceptance evidence in this handoff",
                "evidence_value": (
                    "Single-vs-multi, pooled-vs-macro, leakage, target sensitivity, "
                    "origin stability and R1/R2 role are all machine-traceable."
                ),
                "status": "recommendation_to_Main",
                "boundary": (
                    "highest-value remaining question is direct evidence for official "
                    "target mapping and source publication timing"
                ),
            },
        ]
    )
    return recommendation_rows


def main() -> None:
    assert_s1_boundaries()
    population, population_profile = load_population()
    definitions, matrix, feasibility = load_plan02_geometry()
    plan04_summary = read_json(PLAN04_DIR / "plan04_summary.json")
    assert plan04_summary["population_after"]["canonical_row_count"] == 718
    assert plan04_summary["test_xls_used"] is False
    assert plan04_summary["official_source_sha256"]
    assert read_csv(PLAN03_DIR / "local_target_metric_contract.csv")

    boundary_rows = build_origin_information_boundary(
        population,
        definitions,
        matrix,
        feasibility,
    )
    prediction_rows, leakage_rows = build_predictions_and_audits(population)
    per_origin_metrics, metric_map = build_per_origin_metrics(prediction_rows)

    latest_by_target = {}
    for target in TARGETS:
        target_id = target["target_id"]
        valid_origins = []
        for origin in COMMON_ORIGINS:
            target_rows = [
                row
                for row in per_origin_metrics
                if row["target_id"] == target_id
                and int(row["origin_year"]) == origin
            ]
            target_label_count = target_rows[0]["target_label_available_city_count"]
            if (
                target_label_count / len(CITIES) >= 0.90
                and all(
                    row["scoreable_city_count"] / len(CITIES) >= 0.90
                    for row in target_rows
                )
            ):
                valid_origins.append(origin)
        assert valid_origins
        latest_by_target[target_id] = max(valid_origins)
    assert latest_by_target == {"resident": 2021, "registered": 2021}

    aggregations, weight_rows = build_cross_origin_aggregation(
        prediction_rows,
        per_origin_metrics,
    )
    sanity_checks = build_sanity_checks()
    single_vs_multi = build_single_vs_multi(
        per_origin_metrics,
        aggregations,
        latest_by_target,
    )
    stability = build_stability_summary(
        prediction_rows,
        per_origin_metrics,
        latest_by_target,
    )

    keyed_predictions = {
        (
            row["target_id"],
            row["predictor_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): (
            row["prediction_value"],
            row["residual_prediction_minus_actual"],
            row["squared_error"],
        )
        for row in prediction_rows
        if row["scenario_id"] == SCENARIO_IDS[0]
    }
    r1_r2_exact_match = all(
        keyed_predictions[key]
        == (
            row["prediction_value"],
            row["residual_prediction_minus_actual"],
            row["squared_error"],
        )
        for row in prediction_rows
        if row["scenario_id"] == SCENARIO_IDS[1]
        for key in [
            (
                row["target_id"],
                row["predictor_id"],
                int(row["origin_year"]),
                row["city_id"],
            )
        ]
    )
    assert r1_r2_exact_match

    architecture_definitions = build_architecture_definitions(latest_by_target)
    predictor_definitions = build_predictor_definitions()
    recommendation = build_recommendation(
        latest_by_target,
        aggregations,
        single_vs_multi,
        stability,
        sanity_checks,
        r1_r2_exact_match,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "validation_architecture_definitions.csv",
        architecture_definitions,
    )
    write_csv(
        OUTPUT_DIR / "naive_predictor_definitions.csv",
        predictor_definitions,
    )
    write_csv(
        OUTPUT_DIR / "origin_information_boundary.csv",
        boundary_rows,
    )
    write_csv(
        OUTPUT_DIR / "leakage_audit.csv",
        leakage_rows,
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
        OUTPUT_DIR / "cross_origin_aggregation_comparison.csv",
        aggregations,
    )
    write_csv(
        OUTPUT_DIR / "cross_origin_aggregation_weight_evidence.csv",
        weight_rows,
    )
    write_csv(
        OUTPUT_DIR / "aggregation_sanity_checks.csv",
        sanity_checks,
    )
    write_csv(
        OUTPUT_DIR / "single_vs_multi_origin_comparison.csv",
        single_vs_multi,
    )
    write_csv(
        OUTPUT_DIR / "validation_stability_summary.csv",
        stability,
    )
    write_csv(
        OUTPUT_DIR / "architecture_recommendation.csv",
        recommendation,
    )

    summary = {
        "plan_id": "S2-Plan05",
        "starting_commit": STARTING_COMMIT,
        "branch": "main",
        "current_population_source": "data/normalized/人口规模.csv",
        "population_source_sha256": population_profile["population_sha256"],
        "official_source_sha256_from_plan04": plan04_summary["official_source_sha256"],
        "population_profile": population_profile,
        "target_contract_status": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
        "primary_target": "resident / 常住人口（万人）",
        "sensitivity_target": "registered / 户籍人口（万人）",
        "reference_target_year": REFERENCE_TARGET_YEAR,
        "target_history_horizon_years": HORIZON_YEARS,
        "common_strict_origin_domain": "2011-2021",
        "common_origin_count": len(COMMON_ORIGINS),
        "latest_valid_origin_by_target": latest_by_target,
        "predictor_ids": [predictor["predictor_id"] for predictor in PREDICTORS],
        "predictor_count": len(PREDICTORS),
        "scenario_ids": list(SCENARIO_IDS),
        "prediction_rows": len(prediction_rows),
        "leakage_audit_rows": len(leakage_rows),
        "leakage_audit_all_passed": all(
            row["leakage_status"] == "PASS" for row in leakage_rows
        ),
        "per_origin_metric_rows": len(per_origin_metrics),
        "aggregation_rows": len(aggregations),
        "aggregation_sanity_all_passed": all(
            row["status"] == "PASS" for row in sanity_checks
        ),
        "r1_r2_numeric_residuals_exactly_equal": r1_r2_exact_match,
        "formal_ml_model_used": False,
        "tuning_run": False,
        "feature_engineering_run": False,
        "final_2023_forecast_run": False,
        "submission_run": False,
        "test_xls_used": False,
        "latest_vs_multi_all_judgments_agree": all(
            row["latest_vs_pooled_judgment_changed"] == "no"
            and row["latest_vs_macro_judgment_changed"] == "no"
            for row in single_vs_multi
        ),
        "pooled_vs_macro_all_ranking_same": all(
            row["pooled_vs_macro_ranking_same"] == "yes"
            for row in aggregations
        ),
        "recommendation_status": "S2_ARCHITECTURE_EVIDENCE_SUFFICIENT_FOR_FINAL_REPORT_RECOMMENDATION",
        "s2_complete": False,
        "s3_entered": False,
    }
    write_json(OUTPUT_DIR / "plan05_summary.json", summary)

    assert summary["leakage_audit_all_passed"]
    assert summary["aggregation_sanity_all_passed"]
    assert summary["formal_ml_model_used"] is False
    assert summary["tuning_run"] is False
    assert summary["test_xls_used"] is False
    print("S2_PLAN05_VALIDATION_ARCHITECTURE_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print("common_strict_origin_domain=2011-2021")
    print(f"latest_valid_origin_by_target={stable_json(latest_by_target)}")
    print(f"predictor_count={len(PREDICTORS)}")
    print(f"prediction_rows={len(prediction_rows)}")
    print(f"per_origin_metric_rows={len(per_origin_metrics)}")
    print(f"leakage_audit_all_passed={summary['leakage_audit_all_passed']}")
    print(f"aggregation_sanity_all_passed={summary['aggregation_sanity_all_passed']}")
    print(f"r1_r2_numeric_residuals_exactly_equal={r1_r2_exact_match}")
    print("formal_ml_model_used=False")
    print("tuning_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
