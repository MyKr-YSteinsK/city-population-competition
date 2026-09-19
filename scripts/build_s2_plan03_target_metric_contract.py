"""Build the S2-Plan03 target and metric contract evidence.

This script reads only committed S1/S2 canonical and handoff artifacts. It
does not read raw workbooks or test.xls, train a model, run formal backtests,
or create predictions/submissions.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PLAN02_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
OUTPUT_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan03"
CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
POPULATION_PATH = ROOT / "data" / "normalized" / "人口规模.csv"
AGE_PATH = ROOT / "data" / "normalized" / "年龄结构.csv"
AGE_QUARANTINE_PATH = ROOT / "data" / "quarantine" / "年龄结构_quarantine.csv"
EMPLOYEE_PROVENANCE_PATH = ROOT / "data" / "normalized" / "就业信息_从业人员数_provenance.csv"
QUARANTINE_TEST = ROOT / "data" / "quarantine" / "source" / "test.xls"
RAW_TEST = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集" / "test.xls"

CITIES = tuple(f"city{i}" for i in range(1, 41))
REFERENCE_YEAR = 2023
ORIGIN_YEARS = tuple(range(2001, REFERENCE_YEAR + 1))
MIN_HISTORY_YEARS = 3
MIN_CITY_COVERAGE = 0.90
CORE_SOURCES = (
    "density",
    "urbanization",
    "unemployment",
    "employees",
    "sector",
    "wage",
)
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
TARGETS = (
    {
        "target_id": "resident",
        "semantic_name": "常住人口",
        "canonical_column": "常住人口（万人）",
        "pdf_field": "pr_Population",
        "pdf_semantic": "常住人口规模",
    },
    {
        "target_id": "registered",
        "semantic_name": "户籍人口",
        "canonical_column": "户籍人口（万人）",
        "pdf_field": "r_Population",
        "pdf_semantic": "户籍人口规模",
    },
)

PDF_TARGET_EVIDENCE = (
    "PDF page 1 task wording says to predict each city's 2023 total population; "
    "page 2 embedded data table labels pr_Population as 常住人口规模 and "
    "r_Population as 户籍人口规模, but does not map total population to one field."
)
PDF_MSE_EVIDENCE = (
    "PDF page 4 formal 评测标准 states mean squared error, lower is better, "
    "and score = 1/(1+MSE)."
)
PDF_RMSE_EVIDENCE = (
    "PDF page 4 FAQ item 5 says model evaluation uses root mean squared error "
    "(RMSE), without replacing the formal score formula."
)


def present(value: str | None) -> bool:
    return value is not None and value.strip() not in {
        "",
        "NA",
        "N/A",
        "NaN",
        "nan",
        "None",
    }


def as_int(value: str | None) -> int | None:
    if not present(value):
        return None
    return int(float(value))


def as_float(value: str | None) -> float | None:
    if not present(value):
        return None
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: "" if row.get(field) is None else row.get(field) for field in fields}
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def median_or_blank(values: Iterable[float]) -> str:
    values = list(values)
    return fmt(statistics.median(values)) if values else ""


def rate(count: int, denominator: int = len(CITIES)) -> str:
    return f"{count / denominator:.6f}"


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def load_contract() -> list[dict[str, str]]:
    rows = read_csv(CONTRACT_PATH)
    assert len(rows) == 14, len(rows)
    assert len({row["dataset_id"] for row in rows}) == 14
    return rows


def load_population_rows() -> list[dict[str, str]]:
    rows = read_csv(POPULATION_PATH)
    keys = {(row["城市名称"], as_int(row["年份"])) for row in rows}
    assert None not in {year for _, year in keys}
    assert len(keys) == len(rows)
    return rows


def load_age_rows() -> list[dict[str, str]]:
    rows = read_csv(AGE_PATH)
    keys = {(row["城市名称"], as_int(row["年份"])) for row in rows}
    assert None not in {year for _, year in keys}
    assert len(keys) == len(rows)
    return rows


def assert_s1_boundaries(contract: list[dict[str, str]]) -> None:
    by_id = {row["dataset_id"]: row for row in contract}
    assert by_id["overall_income"]["source_status"] == "source_plus_invalidation"
    assert by_id["overall_income"]["repair_count"] == "0"
    assert by_id["overall_income"]["invalidated_count"] == "32"
    assert by_id["urban_income"]["source_status"] == "source_plus_label_repair"
    assert by_id["urban_income"]["repair_count"] == "2"
    assert by_id["age"]["year_domain"] == "2000;2010;2020"
    expected_age_keys = {
        (f"city{i}", year)
        for i in range(1, 41)
        for year in (2000, 2010, 2020)
    }
    age_keys = {
        (row["城市名称"], as_int(row["年份"]))
        for row in read_csv(AGE_PATH)
    }
    assert age_keys == expected_age_keys - {("city25", 2000), ("city26", 2000)}
    for spec in contract:
        key_columns = spec["key_columns"].split(";")
        canonical = read_csv(ROOT / spec["canonical_file"])
        keys = {
            (row[key_columns[0]], as_int(row[key_columns[1]]))
            for row in canonical
        }
        assert len(keys) == len(canonical), spec["dataset_id"]
        if present(spec.get("quarantine_file")):
            quarantine = read_csv(ROOT / spec["quarantine_file"])
            assert len(quarantine) == int(spec["invalidated_count"]), spec["dataset_id"]
    assert len(read_csv(EMPLOYEE_PROVENANCE_PATH)) == 91


def target_value_rows(
    population_rows: list[dict[str, str]],
    target: dict[str, str],
) -> list[tuple[str, int, float]]:
    result: list[tuple[str, int, float]] = []
    for row in population_rows:
        value = as_float(row.get(target["canonical_column"]))
        year = as_int(row.get("年份"))
        if value is not None and year is not None:
            result.append((row["城市名称"], year, value))
    return result


def target_geometry(
    population_rows: list[dict[str, str]],
    target: dict[str, str],
) -> dict[str, Any]:
    values = target_value_rows(population_rows, target)
    by_key = {(city, year): value for city, year, value in values}
    latest_year = max(year for _, year, _ in values)
    latest_cities = {city for city, year, _ in values if year == latest_year}
    rows_by_year = {
        year: {city for city, row_year, _ in values if row_year == year}
        for year in range(2001, REFERENCE_YEAR + 1)
    }
    missing_cell_count = sum(
        1
        for row in population_rows
        if not present(row.get(target["canonical_column"]))
    )
    return {
        **target,
        "observed_row_count": len(values),
        "canonical_row_count": len(population_rows),
        "canonical_missing_cell_count": missing_cell_count,
        "observed_year_min": min(year for _, year, _ in values),
        "observed_year_max": latest_year,
        "latest_observed_year": latest_year,
        "reference_year": REFERENCE_YEAR,
        "implied_reference_horizon_years": REFERENCE_YEAR - latest_year,
        "latest_city_count": len(latest_cities),
        "latest_city_coverage_rate": rate(len(latest_cities)),
        "latest_missing_cities": ";".join(sorted(set(CITIES) - latest_cities)),
        "values_by_key": by_key,
        "cities_by_year": rows_by_year,
    }


def complete_age_sums(age_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in age_rows:
        values = [as_float(row.get(column)) for column in ("0-14", "15-64", "65+")]
        year = as_int(row.get("年份"))
        if year is not None and all(value is not None for value in values):
            result.append(
                {
                    "city": row["城市名称"],
                    "year": year,
                    "age_sum": sum(values),
                }
            )
    return result


def consistency_detail(
    age_sums: list[dict[str, Any]],
    geometries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for age_row in age_sums:
        city = age_row["city"]
        year = age_row["year"]
        resident = geometries["resident"]["values_by_key"].get((city, year))
        registered = geometries["registered"]["values_by_key"].get((city, year))
        resident_diff = (
            abs(age_row["age_sum"] - resident) if resident is not None else None
        )
        registered_diff = (
            abs(age_row["age_sum"] - registered) if registered is not None else None
        )
        closer = ""
        if resident_diff is not None and registered_diff is not None:
            if resident_diff < registered_diff:
                closer = "resident"
            elif registered_diff < resident_diff:
                closer = "registered"
            else:
                closer = "tie"
        result.append(
            {
                "city": city,
                "age_year": year,
                "age_sum": fmt(age_row["age_sum"]),
                "resident_same_year": fmt(resident),
                "registered_same_year": fmt(registered),
                "resident_absolute_diff": fmt(resident_diff),
                "registered_absolute_diff": fmt(registered_diff),
                "resident_relative_absolute_diff": (
                    fmt(resident_diff / abs(resident))
                    if resident not in (None, 0) and resident_diff is not None
                    else ""
                ),
                "registered_relative_absolute_diff": (
                    fmt(registered_diff / abs(registered))
                    if registered not in (None, 0) and registered_diff is not None
                    else ""
                ),
                "closer_same_year_candidate": closer,
                "evidence_boundary": (
                    "Complete canonical age-band rows only; partial or invalidated "
                    "age cells are excluded, not filled."
                ),
            }
        )
    return result


def offset_consistency_rows(
    age_sums: list[dict[str, Any]],
    geometries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id in ("resident", "registered"):
        geometry = geometries[target_id]
        for offset in (-2, -1, 0, 1, 2):
            pairs: list[tuple[float, float]] = []
            for age_row in age_sums:
                population_year = age_row["year"] + offset
                value = geometry["values_by_key"].get(
                    (age_row["city"], population_year)
                )
                if value is not None:
                    pairs.append((age_row["age_sum"], value))
            absolute = [abs(age_sum - population) for age_sum, population in pairs]
            relative = [
                abs(age_sum - population) / abs(population)
                for age_sum, population in pairs
                if population != 0
            ]
            signed = [age_sum - population for age_sum, population in pairs]
            result.append(
                {
                    "target_id": target_id,
                    "population_year_minus_age_year": offset,
                    "overlap_n": len(pairs),
                    "mean_absolute_diff": fmt(statistics.mean(absolute) if absolute else None),
                    "median_absolute_diff": median_or_blank(absolute),
                    "mean_relative_absolute_diff": fmt(
                        statistics.mean(relative) if relative else None
                    ),
                    "mean_signed_diff": fmt(statistics.mean(signed) if signed else None),
                    "interpretation": (
                        "Temporal-offset diagnostic only; same-year alignment remains "
                        "the primary semantic comparison and no lag is declared official."
                    ),
                }
            )
    return result


def consistency_summary(
    details: list[dict[str, Any]],
    offsets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    both = [row for row in details if row["closer_same_year_candidate"]]
    for target_id in ("resident", "registered"):
        rows = [
            row
            for row in details
            if row[f"{target_id}_same_year"] != ""
        ]
        absolute = [
            float(row[f"{target_id}_absolute_diff"])
            for row in rows
        ]
        relative = [
            float(row[f"{target_id}_relative_absolute_diff"])
            for row in rows
            if row[f"{target_id}_relative_absolute_diff"] != ""
        ]
        signed = [
            float(row[f"{target_id}_same_year"]) - float(row["age_sum"])
            for row in rows
        ]
        closer_to_target = sum(
            row["closer_same_year_candidate"] == target_id
            for row in both
        )
        closer_to_other = sum(
            row["closer_same_year_candidate"] not in {"", target_id, "tie"}
            for row in both
        )
        target_offsets = [
            row
            for row in offsets
            if row["target_id"] == target_id and row["overlap_n"] != 0
        ]
        best_offset = min(
            target_offsets,
            key=lambda row: float(row["mean_absolute_diff"]),
        )
        result.append(
            {
                "target_id": target_id,
                "same_year_overlap_n": len(rows),
                "same_year_mean_absolute_diff": fmt(statistics.mean(absolute)),
                "same_year_median_absolute_diff": fmt(statistics.median(absolute)),
                "same_year_mean_relative_absolute_diff": fmt(statistics.mean(relative)),
                "same_year_mean_signed_population_minus_age": fmt(statistics.mean(signed)),
                "paired_both_target_overlap_n": len(both),
                "paired_closer_to_this_target_n": closer_to_target,
                "paired_closer_to_other_target_n": closer_to_other,
                "same_year_consistency_rank": (
                    "stronger_internal_consistency"
                    if target_id == "resident"
                    else "weaker_internal_consistency"
                ),
                "best_temporal_offset_diagnostic": best_offset[
                    "population_year_minus_age_year"
                ],
                "best_temporal_offset_mean_absolute_diff": best_offset[
                    "mean_absolute_diff"
                ],
                "interpretation": (
                    "Cross-source consistency strengthens or weakens a proxy; it does "
                    "not define the official competition target."
                ),
            }
        )
    return result


def load_plan02_matrix() -> list[dict[str, str]]:
    return read_csv(
        PLAN02_DIR / "origin_source_scenario_freshness_matrix.csv"
    )


def core_screen_by_origin(
    plan02_matrix: list[dict[str, str]],
    origin: int,
    scenario_id: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in plan02_matrix
        if int(row["pseudo_target_year"]) == origin
        and row["scenario_id"] == scenario_id
        and row["source_id"] in CORE_SOURCES
    ]
    assert len(rows) == len(CORE_SOURCES), (origin, scenario_id, len(rows))
    failures = [
        row["source_id"]
        for row in rows
        if row["structural_90pct_coverage_screen"] != "pass"
    ]
    return {
        "core_source_count": len(CORE_SOURCES),
        "core_sources_meeting_90pct_screen": len(CORE_SOURCES) - len(failures),
        "core_sources_below_90pct_screen": ";".join(failures),
        "core_screen_ok": not failures,
    }


def history_ready_count(
    target_values: list[tuple[str, int, float]],
    cutoff: int,
) -> int:
    counts = {
        city: sum(
            1
            for value_city, year, _ in target_values
            if value_city == city and year <= cutoff
        )
        for city in CITIES
    }
    return sum(count >= MIN_HISTORY_YEARS for count in counts.values())


def classify_origin(
    label_count: int,
    history_ready: int,
    core_ok: bool,
) -> tuple[str, str, str]:
    label_ok = label_count >= len(CITIES) * MIN_CITY_COVERAGE
    history_ok = history_ready >= len(CITIES) * MIN_CITY_COVERAGE
    if label_count == 0:
        return (
            "exclude_no_target_label",
            "exclude",
            "target label is not observed; no retrospective score can be formed",
        )
    if not label_ok:
        return (
            "exclude_target_label_coverage",
            "exclude",
            f"target label coverage is {label_count}/40, below the {MIN_CITY_COVERAGE:.0%} screen",
        )
    if not history_ok and not core_ok:
        return (
            "conditional_history_and_core_gap",
            "conditional",
            f"target history through T-horizon reaches only {history_ready}/40 cities with at least {MIN_HISTORY_YEARS} labels; core source gap remains",
        )
    if not history_ok:
        return (
            "conditional_target_horizon_history_gap",
            "conditional",
            f"target history through T-horizon reaches only {history_ready}/40 cities with at least {MIN_HISTORY_YEARS} labels",
        )
    if not core_ok:
        return (
            "conditional_core_source_coverage_gap",
            "conditional",
            "one or more provisional core sources fail the 90% as-of coverage screen",
        )
    return (
        "real_like_candidate",
        "candidate",
        "target label, T-horizon history and all provisional core source screens pass",
    )


def target_origin_geometry(
    population_rows: list[dict[str, str]],
    geometries: dict[str, dict[str, Any]],
    plan02_matrix: list[dict[str, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id, geometry in geometries.items():
        values = [
            (city, year, value)
            for (city, year), value in geometry["values_by_key"].items()
        ]
        horizon = geometry["implied_reference_horizon_years"]
        for scenario_id in SCENARIOS:
            for origin in ORIGIN_YEARS:
                label_count = sum(year == origin for _, year, _ in values)
                cutoff = origin - horizon
                history_ready = history_ready_count(values, cutoff)
                core = core_screen_by_origin(plan02_matrix, origin, scenario_id)
                classification, bucket, reason = classify_origin(
                    label_count,
                    history_ready,
                    core["core_screen_ok"],
                )
                result.append(
                    {
                        "target_id": target_id,
                        "target_semantic_name": geometry["semantic_name"],
                        "canonical_column": geometry["canonical_column"],
                        "scenario_id": scenario_id,
                        "pseudo_target_year": origin,
                        "target_observed_latest_year": geometry["latest_observed_year"],
                        "target_horizon_years": horizon,
                        "target_history_cutoff_year": cutoff,
                        "target_label_available_city_count": label_count,
                        "target_label_coverage_rate": rate(label_count),
                        "cities_with_min_3_target_labels_through_cutoff": history_ready,
                        "target_history_coverage_rate": rate(history_ready),
                        **core,
                        "classification": classification,
                        "status_bucket": bucket,
                        "eligible_for_next_round": "yes" if bucket == "candidate" else "no",
                        "screening_rules_are_final_contract": "no",
                        "reason": reason,
                    }
                )
    return result


def target_origin_comparison(
    geometry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = {
        (row["scenario_id"], int(row["pseudo_target_year"]), row["target_id"]): row
        for row in geometry_rows
    }
    result: list[dict[str, Any]] = []
    for scenario_id in SCENARIOS:
        for origin in ORIGIN_YEARS:
            resident = index[(scenario_id, origin, "resident")]
            registered = index[(scenario_id, origin, "registered")]
            result.append(
                {
                    "scenario_id": scenario_id,
                    "pseudo_target_year": origin,
                    "resident_status_bucket": resident["status_bucket"],
                    "registered_status_bucket": registered["status_bucket"],
                    "resident_candidate": resident["eligible_for_next_round"],
                    "registered_candidate": registered["eligible_for_next_round"],
                    "candidate_status_differs": (
                        "yes"
                        if resident["status_bucket"] != registered["status_bucket"]
                        else "no"
                    ),
                    "resident_history_ready": resident[
                        "cities_with_min_3_target_labels_through_cutoff"
                    ],
                    "registered_history_ready": registered[
                        "cities_with_min_3_target_labels_through_cutoff"
                    ],
                    "resident_target_label_count": resident[
                        "target_label_available_city_count"
                    ],
                    "registered_target_label_count": registered[
                        "target_label_available_city_count"
                    ],
                    "interpretation": (
                        "Target hypothesis comparison only; covariate R1/R2 policy is "
                        "inherited from Plan02 and no model score is computed."
                    ),
                }
            )
    return result


def target_hypothesis_summary(
    geometries: dict[str, dict[str, Any]],
    origin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id, geometry in geometries.items():
        for scenario_id in SCENARIOS:
            rows = [
                row
                for row in origin_rows
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
            ]
            result.append(
                {
                    "target_id": target_id,
                    "semantic_name": geometry["semantic_name"],
                    "canonical_column": geometry["canonical_column"],
                    "scenario_id": scenario_id,
                    "latest_observed_year": geometry["latest_observed_year"],
                    "implied_horizon_years": geometry[
                        "implied_reference_horizon_years"
                    ],
                    "candidate_years": ";".join(
                        str(row["pseudo_target_year"])
                        for row in rows
                        if row["status_bucket"] == "candidate"
                    ),
                    "conditional_years": ";".join(
                        str(row["pseudo_target_year"])
                        for row in rows
                        if row["status_bucket"] == "conditional"
                    ),
                    "excluded_years": ";".join(
                        str(row["pseudo_target_year"])
                        for row in rows
                        if row["status_bucket"] == "exclude"
                    ),
                    "candidate_count": sum(
                        row["status_bucket"] == "candidate" for row in rows
                    ),
                    "conditional_count": sum(
                        row["status_bucket"] == "conditional" for row in rows
                    ),
                    "excluded_count": sum(
                        row["status_bucket"] == "exclude" for row in rows
                    ),
                    "interpretation": (
                        "Structural feasibility comparison only; not a final target "
                        "contract or model-score result."
                    ),
                }
            )
    return result


def mse(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred) and y_true
    return sum((pred - truth) ** 2 for truth, pred in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mse(y_true, y_pred))


def competition_score_from_mse(value: float) -> float:
    return 1.0 / (1.0 + value)


def metric_sanity_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def record(
        test_id: str,
        actual: Any,
        expected: Any,
        passed: bool,
        interpretation: str,
    ) -> None:
        results.append(
            {
                "test_id": test_id,
                "actual": json.dumps(actual, ensure_ascii=False, sort_keys=True),
                "expected": json.dumps(expected, ensure_ascii=False, sort_keys=True),
                "status": "PASS" if passed else "FAIL",
                "interpretation": interpretation,
            }
        )
        assert passed, test_id

    truth = [1.0, 2.0, 4.0]
    prediction = [2.0, 2.0, 1.0]
    hand_mse = mse(truth, prediction)
    hand_rmse = rmse(truth, prediction)
    hand_score = competition_score_from_mse(hand_mse)
    record(
        "hand_check_mse",
        hand_mse,
        10.0 / 3.0,
        close(hand_mse, 10.0 / 3.0),
        "Residual squares are 1, 0 and 9.",
    )
    record(
        "hand_check_rmse",
        hand_rmse,
        math.sqrt(10.0 / 3.0),
        close(hand_rmse, math.sqrt(10.0 / 3.0)),
        "RMSE is the square root of the same-sample MSE.",
    )
    record(
        "hand_check_score",
        hand_score,
        3.0 / 13.0,
        close(hand_score, 3.0 / 13.0),
        "score = 1/(1+MSE) for the same hand-check residuals.",
    )

    truth_same = [0.0, 0.0, 0.0]
    model_a = [1.0, 1.0, 1.0]
    model_b = [0.0, 0.0, 2.0]
    mse_a = mse(truth_same, model_a)
    mse_b = mse(truth_same, model_b)
    rmse_a = rmse(truth_same, model_a)
    rmse_b = rmse(truth_same, model_b)
    score_a = competition_score_from_mse(mse_a)
    score_b = competition_score_from_mse(mse_b)
    record(
        "same_aggregation_ranking_equivalence",
        {
            "mse_a": mse_a,
            "mse_b": mse_b,
            "rmse_a": rmse_a,
            "rmse_b": rmse_b,
            "score_a": score_a,
            "score_b": score_b,
        },
        "MSE_A < MSE_B; RMSE_A < RMSE_B; score_A > score_B",
        mse_a < mse_b and rmse_a < rmse_b and score_a > score_b,
        "Same sample and same aggregation preserve model ordering.",
    )
    record(
        "rmse_sqrt_mse_identity",
        rmse_a,
        math.sqrt(mse_a),
        close(rmse_a, math.sqrt(mse_a)),
        "RMSE equals sqrt(MSE) for the same residual set.",
    )

    origin_errors = {
        "origin_1": {"A": [0.0], "B": [2.0]},
        "origin_2": {"A": [1.0] * 9, "B": [0.0] * 9},
    }
    macro_mse = {
        model: statistics.mean(
            mse([0.0] * len(errors), errors)
            for errors in (origin_errors["origin_1"][model], origin_errors["origin_2"][model])
        )
        for model in ("A", "B")
    }
    pooled_errors = {
        model: [
            *origin_errors["origin_1"][model],
            *origin_errors["origin_2"][model],
        ]
        for model in ("A", "B")
    }
    pooled_mse = {
        model: mse([0.0] * len(errors), errors)
        for model, errors in pooled_errors.items()
    }
    macro_rmse = {
        model: statistics.mean(
            rmse([0.0] * len(errors), errors)
            for errors in (origin_errors["origin_1"][model], origin_errors["origin_2"][model])
        )
        for model in ("A", "B")
    }
    pooled_rmse = {
        model: math.sqrt(pooled_mse[model])
        for model in ("A", "B")
    }
    record(
        "cross_origin_aggregation_is_not_interchangeable",
        {
            "macro_mse": macro_mse,
            "pooled_mse": pooled_mse,
            "macro_rmse": macro_rmse,
            "pooled_rmse": pooled_rmse,
        },
        "macro ranks A better; pooled ranks B better",
        macro_mse["A"] < macro_mse["B"]
        and pooled_mse["A"] > pooled_mse["B"]
        and macro_rmse["A"] < macro_rmse["B"]
        and pooled_rmse["A"] > pooled_rmse["B"],
        "Different origin weighting can change ranking; aggregation must be a separate contract.",
    )
    return results


def metric_evidence_rows() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "PDF_FORMAL_EVALUATION_MSE",
            "pdf_page": 4,
            "pdf_section": "评测标准",
            "wording": "采用均方误差（mean squared error）评估模型的预测精度；均方误差越小表示预测效果越好。",
            "metric": "MSE",
            "formula": "score = 1/(1+MSE)",
            "source_authority": "formal_evaluation_section",
            "evidence_class": "direct_official_wording",
            "conflict_type": "conflicts_with_FAQ_RMSE_wording",
            "downstream_implication": "MSE is the primary local competition metric candidate; score maps directly from MSE.",
        },
        {
            "evidence_id": "PDF_FORMAL_SCORE_TRANSFORM",
            "pdf_page": 4,
            "pdf_section": "评测标准",
            "wording": "MSE 为本赛题的评测参考标准，排行榜的分数计算依据为 score =1/(1+MSE)。",
            "metric": "competition_score",
            "formula": "1/(1+MSE)",
            "source_authority": "formal_evaluation_section",
            "evidence_class": "direct_official_wording",
            "conflict_type": "none",
            "downstream_implication": "Score is higher-is-better and should only be computed from an explicitly defined MSE aggregation.",
        },
        {
            "evidence_id": "PDF_FAQ_RMSE",
            "pdf_page": 4,
            "pdf_section": "常见问题说明 / 5、模型评估",
            "wording": "赛题要求选手使用均方根误差（RMSE）作为评价指标。",
            "metric": "RMSE",
            "formula": "none in FAQ text",
            "source_authority": "FAQ_guidance",
            "evidence_class": "direct_official_wording",
            "conflict_type": "formal_section_uses_MSE_and_score_formula",
            "downstream_implication": "Report RMSE as an auxiliary interpretable unit-scale metric; do not replace the formal score basis without new evidence.",
        },
        {
            "evidence_id": "MATHEMATICAL_RELATION",
            "pdf_page": "",
            "pdf_section": "derived sanity evidence",
            "wording": "For non-negative MSE, RMSE = sqrt(MSE).",
            "metric": "MSE/RMSE",
            "formula": "RMSE = sqrt(MSE)",
            "source_authority": "mathematical_derivation",
            "evidence_class": "inference_only",
            "conflict_type": "not a PDF authority decision",
            "downstream_implication": "Same residual set and same aggregation preserve ranking, but scale and cross-origin aggregation remain distinct.",
        },
    ]


def local_contract_rows() -> list[dict[str, Any]]:
    return [
        {
            "contract_item": "target_contract_status",
            "value": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
            "direction_or_scope": "conditional target contract",
            "evidence_basis": "PDF ambiguity plus resident-favoring age-sum consistency",
            "status": "proposed_for_next_S2",
            "deferred_question": "Official competition label remains unproven.",
        },
        {
            "contract_item": "primary_target_proxy",
            "value": "resident / 常住人口（万人）",
            "direction_or_scope": "label at pseudo-origin T for retrospective scoring",
            "evidence_basis": "same-year age-sum consistency is materially stronger than registered",
            "status": "primary_proxy",
            "deferred_question": "Recheck alongside registered sensitivity in later backtests.",
        },
        {
            "contract_item": "secondary_target_sensitivity",
            "value": "registered / 户籍人口（万人）",
            "direction_or_scope": "sensitivity target, not official truth claim",
            "evidence_basis": "PDF names the field but does not map total population to it",
            "status": "required_sensitivity",
            "deferred_question": "If rankings diverge, interpretation as official competition score is blocked.",
        },
        {
            "contract_item": "primary_metric",
            "value": "MSE",
            "direction_or_scope": "lower_is_better; per-origin city-year residuals",
            "evidence_basis": "formal PDF evaluation section and score formula",
            "status": "local_primary",
            "deferred_question": "Cross-origin aggregation is not frozen.",
        },
        {
            "contract_item": "auxiliary_metric",
            "value": "RMSE",
            "direction_or_scope": "lower_is_better; same residual set as MSE",
            "evidence_basis": "FAQ wording plus RMSE=sqrt(MSE) sanity evidence",
            "status": "local_auxiliary",
            "deferred_question": "Do not average or pool across origins without an explicit aggregation rule.",
        },
        {
            "contract_item": "competition_score_transform",
            "value": "1/(1+MSE)",
            "direction_or_scope": "higher_is_better; only after MSE aggregation is specified",
            "evidence_basis": "formal PDF evaluation section",
            "status": "reportable_per_defined_scope",
            "deferred_question": "No single multi-origin score is computed in this Plan.",
        },
        {
            "contract_item": "sample_unit",
            "value": "city-year evaluation row within one pseudo-origin",
            "direction_or_scope": "one target label per city at origin T",
            "evidence_basis": "submission schema and historical origin design",
            "status": "defined_for_single_origin",
            "deferred_question": "Origin weighting remains open.",
        },
        {
            "contract_item": "cross_origin_aggregation",
            "value": "UNRESOLVED_DEFERRED",
            "direction_or_scope": "do not mix pooled MSE, macro-origin MSE, average RMSE or pooled RMSE",
            "evidence_basis": "sanity test demonstrates ranking can change",
            "status": "must_be_decided_in_backtest_architecture",
            "deferred_question": "Compare pooled city-year weighting versus macro-origin weighting.",
        },
        {
            "contract_item": "formal_model_scoring",
            "value": "NOT_RUN",
            "direction_or_scope": "non-goal of S2-Plan03",
            "evidence_basis": "execution boundary",
            "status": "not_executed",
            "deferred_question": "Next Plan only.",
        },
    ]


def write_artifacts(
    geometries: dict[str, dict[str, Any]],
    target_matrix: list[dict[str, Any]],
    target_comparison: list[dict[str, Any]],
    target_summary: list[dict[str, Any]],
    consistency_details: list[dict[str, Any]],
    consistency_summary_rows: list[dict[str, Any]],
    consistency_offsets: list[dict[str, Any]],
    metric_sanity: list[dict[str, Any]],
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_evidence = []
    summary_by_target = {
        row["target_id"]: row
        for row in consistency_summary_rows
    }
    for target_id, geometry in geometries.items():
        consistency = summary_by_target[target_id]
        target_evidence.append(
            {
                "target_id": target_id,
                "semantic_name": geometry["semantic_name"],
                "pdf_field": geometry["pdf_field"],
                "pdf_field_semantic": geometry["pdf_semantic"],
                "canonical_column": geometry["canonical_column"],
                "pdf_target_wording": "2023 total population; no direct field mapping",
                "pdf_page_evidence": "page 1 task; page 2 embedded data table",
                "observed_year_domain": f"{geometry['observed_year_min']}-{geometry['observed_year_max']}",
                "observed_nonempty_row_count": geometry["observed_row_count"],
                "canonical_missing_cell_count": geometry["canonical_missing_cell_count"],
                "latest_observed_year": geometry["latest_observed_year"],
                "reference_horizon_years": geometry["implied_reference_horizon_years"],
                "latest_city_count": geometry["latest_city_count"],
                "latest_city_coverage_rate": geometry["latest_city_coverage_rate"],
                "same_year_age_overlap_n": consistency["same_year_overlap_n"],
                "same_year_age_mean_absolute_diff": consistency["same_year_mean_absolute_diff"],
                "same_year_age_mean_relative_absolute_diff": consistency[
                    "same_year_mean_relative_absolute_diff"
                ],
                "paired_both_closer_to_this_target_n": consistency[
                    "paired_closer_to_this_target_n"
                ],
                "direct_official_evidence": (
                    "Official task wording names total population for 2023, "
                    "but does not select this candidate field."
                ),
                "source_schema_semantics": (
                    f"PDF data table labels {geometry['pdf_field']} as "
                    f"{geometry['pdf_semantic']}."
                ),
                "cross_source_consistency_evidence": (
                    "Age-band sum is compared on complete canonical benchmark rows; "
                    "invalidated/partial rows are excluded."
                ),
                "inference_only": (
                    "Resident is preferred as primary proxy only because its same-year "
                    "internal consistency is materially stronger; this is not official wording."
                ),
                "ambiguity_or_counterevidence": (
                    "The PDF says total population and also lists registered population; "
                    "age-sum alignment is an internal consistency signal, not a definition."
                ),
                "evidence_class": (
                    "direct_official_wording;source_schema_semantics;"
                    "cross_source_consistency;inference_only"
                ),
                "target_contract_conclusion": (
                    "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET"
                    if target_id == "resident"
                    else "secondary_sensitivity_target"
                ),
            }
        )
    write_csv(
        OUTPUT_DIR / "target_evidence_matrix.csv",
        target_evidence,
        list(target_evidence[0]),
    )
    write_csv(
        OUTPUT_DIR / "age_population_consistency_detail.csv",
        consistency_details,
        list(consistency_details[0]),
    )
    write_csv(
        OUTPUT_DIR / "age_population_consistency_summary.csv",
        consistency_summary_rows,
        list(consistency_summary_rows[0]),
    )
    write_csv(
        OUTPUT_DIR / "age_population_consistency_offset.csv",
        consistency_offsets,
        list(consistency_offsets[0]),
    )
    write_csv(
        OUTPUT_DIR / "target_hypothesis_origin_geometry.csv",
        target_matrix,
        list(target_matrix[0]),
    )
    write_csv(
        OUTPUT_DIR / "target_hypothesis_origin_comparison.csv",
        target_comparison,
        list(target_comparison[0]),
    )
    write_csv(
        OUTPUT_DIR / "target_hypothesis_summary.csv",
        target_summary,
        list(target_summary[0]),
    )
    write_csv(
        OUTPUT_DIR / "metric_evidence_conflict_table.csv",
        metric_evidence_rows(),
        list(metric_evidence_rows()[0]),
    )
    write_csv(
        OUTPUT_DIR / "metric_sanity_results.csv",
        metric_sanity,
        list(metric_sanity[0]),
    )
    contract_rows = local_contract_rows()
    write_csv(
        OUTPUT_DIR / "local_target_metric_contract.csv",
        contract_rows,
        list(contract_rows[0]),
    )
    candidate_years = {
        f"{target_id}/{scenario_id}": [
            int(row["pseudo_target_year"])
            for row in target_matrix
            if row["target_id"] == target_id
            and row["scenario_id"] == scenario_id
            and row["status_bucket"] == "candidate"
        ]
        for target_id in geometries
        for scenario_id in SCENARIOS
    }
    summary = {
        "plan_id": "S2-Plan03",
        "reference_year": REFERENCE_YEAR,
        "target_contract_status": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
        "primary_target_proxy": "resident / 常住人口（万人）",
        "secondary_target_sensitivity": "registered / 户籍人口（万人）",
        "primary_metric": "MSE",
        "auxiliary_metric": "RMSE",
        "competition_score_transform": "1/(1+MSE)",
        "cross_origin_aggregation": "UNRESOLVED_DEFERRED",
        "target_hypothesis_candidate_years": candidate_years,
        "target_evidence_rows": len(target_evidence),
        "age_complete_rows_used": len(consistency_details),
        "age_quarantine_rows_not_reintroduced": len(read_csv(AGE_QUARANTINE_PATH)),
        "employee_provenance_rows_preserved": len(read_csv(EMPLOYEE_PROVENANCE_PATH)),
        "metric_sanity_tests": len(metric_sanity),
        "metric_sanity_all_passed": all(
            row["status"] == "PASS" for row in metric_sanity
        ),
        "formal_model_scoring_run": False,
        "test_xls_used": False,
        "publication_timing_resolved": False,
    }
    write_json(OUTPUT_DIR / "plan03_summary.json", summary)
    return summary


def main() -> None:
    assert not RAW_TEST.exists(), "test.xls must remain outside raw source inputs"
    assert QUARANTINE_TEST.exists(), "quarantine test.xls is missing"
    contract = load_contract()
    assert len(contract) == 14
    assert_s1_boundaries(contract)
    population_rows = load_population_rows()
    age_rows = load_age_rows()
    geometries = {
        target["target_id"]: target_geometry(population_rows, target)
        for target in TARGETS
    }
    assert all(
        geometry["latest_observed_year"] == 2021
        and geometry["implied_reference_horizon_years"] == 2
        for geometry in geometries.values()
    )
    age_sums = complete_age_sums(age_rows)
    assert len(age_sums) == 91
    assert len(read_csv(AGE_QUARANTINE_PATH)) == 6
    assert len(read_csv(EMPLOYEE_PROVENANCE_PATH)) == 91
    details = consistency_detail(age_sums, geometries)
    offsets = offset_consistency_rows(age_sums, geometries)
    summaries = consistency_summary(details, offsets)
    assert (
        float(next(row for row in summaries if row["target_id"] == "resident")[
            "same_year_mean_absolute_diff"
        ])
        < float(next(row for row in summaries if row["target_id"] == "registered")[
            "same_year_mean_absolute_diff"
        ])
    )
    plan02_matrix = load_plan02_matrix()
    target_matrix = target_origin_geometry(
        population_rows,
        geometries,
        plan02_matrix,
    )
    target_comparison = target_origin_comparison(target_matrix)
    target_summary = target_hypothesis_summary(geometries, target_matrix)
    metric_sanity = metric_sanity_results()
    summary = write_artifacts(
        geometries,
        target_matrix,
        target_comparison,
        target_summary,
        details,
        summaries,
        offsets,
        metric_sanity,
    )
    assert summary["target_contract_status"] == "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET"
    assert summary["metric_sanity_all_passed"] is True
    assert summary["target_hypothesis_candidate_years"]["resident/R1_REFERENCE_SNAPSHOT_LAG"] == list(range(2010, 2022))
    assert summary["target_hypothesis_candidate_years"]["registered/R1_REFERENCE_SNAPSHOT_LAG"] == list(range(2011, 2022))
    print("S2_PLAN03_TARGET_METRIC_PASS")
    print("target_evidence_rows=2")
    print(f"age_complete_rows_used={len(details)}")
    print(f"target_geometry_rows={len(target_matrix)}")
    print(f"target_comparison_rows={len(target_comparison)}")
    print(f"metric_sanity_tests={len(metric_sanity)}")
    print("target_contract_status=PRIMARY_PROXY_WITH_SENSITIVITY_TARGET")
    print("primary_target=resident")
    print("secondary_target=registered")
    print("primary_metric=MSE")
    print("auxiliary_metric=RMSE")
    print("cross_origin_aggregation=UNRESOLVED_DEFERRED")
    print("formal_model_scoring_run=False")
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
