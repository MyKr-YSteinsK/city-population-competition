"""Build the S2-Plan01 historical information-set feasibility matrix.

The script consumes the frozen S1 contract and canonical CSVs only.  It does
not read test.xls, rebuild S1 data, train a model, score a model, or create a
submission.  All availability statements produced here are assumptions for
historical pseudo-tests, not claims about the real publication calendar.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
OUTPUT_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan01"
QUARANTINE_SOURCE = ROOT / "data" / "quarantine" / "source" / "test.xls"
RAW_TEST = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集" / "test.xls"
CITIES = tuple(f"city{i}" for i in range(1, 41))
TARGET_SOURCE = "population"
TARGET_VALUE_COLUMN: str | None = None
TARGET_YEARS = tuple(range(2001, 2024))
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

POLICIES = (
    {
        "policy_id": "P1_ASOF_T_MINUS_1",
        "policy_name": "source-specific as-of T-1",
        "cutoff_lag_years": 1,
        "assumption": (
            "For a pseudo-target T, each source may contribute only values dated "
            "at or before T-1. Source-specific observed coverage asymmetry is "
            "retained; an observed max year is not treated as a proven publication date."
        ),
        "selection_reason": "Closest simple historical replay that preserves the observed source freshness asymmetry.",
    },
    {
        "policy_id": "P2_ASOF_T_MINUS_2",
        "policy_name": "conservative as-of T-2",
        "cutoff_lag_years": 2,
        "assumption": (
            "For a pseudo-target T, every source is held back through T-2. "
            "This is a one-year more conservative freshness stress than P1, while "
            "source-specific coverage, missingness, benchmark domains, and quality "
            "boundaries remain unchanged."
        ),
        "selection_reason": "Sensitivity comparison for an additional publication/availability lag; not a claim about the real calendar.",
    },
)


def present(value: str | None) -> bool:
    return value is not None and value.strip() not in {"", "NA", "NaN", "nan", "None"}


def as_int(value: str | None) -> int | None:
    if not present(value):
        return None
    return int(float(value))


def key(row: dict[str, str], city_col: str, year_col: str) -> tuple[str, int]:
    year = as_int(row.get(year_col))
    assert year is not None, row
    return row[city_col], year


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def median_or_blank(values: Iterable[int]) -> str | int:
    values = list(values)
    return int(statistics.median(values)) if values else ""


def load_contract() -> list[dict[str, str]]:
    rows = read_csv(CONTRACT_PATH)
    assert len(rows) == 14, len(rows)
    ids = [row["dataset_id"] for row in rows]
    assert len(ids) == len(set(ids)), ids
    return rows


def load_source_data(contract: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for spec in contract:
        source_id = spec["dataset_id"]
        city_col, year_col = spec["key_columns"].split(";")
        value_columns = spec["value_columns"].split(";")
        rows = read_csv(ROOT / spec["canonical_file"])
        keys = [key(row, city_col, year_col) for row in rows]
        assert len(keys) == len(set(keys)), f"duplicate canonical key: {source_id}"
        value_rows = [
            row
            for row in rows
            if any(present(row.get(column)) for column in value_columns)
        ]
        value_years = [key(row, city_col, year_col)[1] for row in value_rows]
        derived_keys: set[tuple[str, int]] = set()
        if present(spec.get("provenance_file")):
            provenance = read_csv(ROOT / spec["provenance_file"])
            derived_keys = {
                key(row, city_col, year_col)
                for row in provenance
            }
        quarantine_keys: set[tuple[str, int]] = set()
        if present(spec.get("quarantine_file")):
            quarantine = read_csv(ROOT / spec["quarantine_file"])
            quarantine_keys = {
                key(row, city_col, year_col)
                for row in quarantine
            }
        city_rows: dict[str, list[dict[str, str]]] = {city: [] for city in CITIES}
        for row in rows:
            city_rows.setdefault(row[city_col], []).append(row)
        city_value_rows: dict[str, list[dict[str, str]]] = {city: [] for city in CITIES}
        for row in value_rows:
            city_value_rows.setdefault(row[city_col], []).append(row)
        row_years = [key(row, city_col, year_col)[1] for row in rows]
        sources[source_id] = {
            "spec": spec,
            "rows": rows,
            "city_col": city_col,
            "year_col": year_col,
            "value_columns": value_columns,
            "keys": set(keys),
            "value_rows": value_rows,
            "value_years": value_years,
            "derived_keys": derived_keys,
            "quarantine_keys": quarantine_keys,
            "city_rows": city_rows,
            "city_value_rows": city_value_rows,
            "row_year_min": min(row_years),
            "row_year_max": max(row_years),
            "value_year_min": min(value_years) if value_years else "",
            "value_year_max": max(value_years) if value_years else "",
            "missing_value_cells": sum(
                1
                for row in rows
                for column in value_columns
                if not present(row.get(column))
            ),
            "value_cell_count": sum(
                1
                for row in rows
                for column in value_columns
                if present(row.get(column))
            ),
        }
    return sources


def assert_s1_boundaries(contract: list[dict[str, str]], sources: dict[str, dict[str, Any]]) -> None:
    by_id = {row["dataset_id"]: row for row in contract}
    assert by_id["overall_income"]["source_status"] == "source_plus_invalidation"
    assert by_id["overall_income"]["repair_count"] == "0"
    assert by_id["overall_income"]["invalidated_count"] == "32"
    assert by_id["urban_income"]["source_status"] == "source_plus_label_repair"
    assert by_id["urban_income"]["repair_count"] == "2"
    assert by_id["age"]["year_domain"] == "2000;2010;2020"
    age = sources["age"]
    age_keys = age["keys"]
    expected_age_keys = {
        (f"city{i}", year)
        for i in range(1, 41)
        for year in (2000, 2010, 2020)
    }
    assert age_keys == expected_age_keys - {("city25", 2000), ("city26", 2000)}
    assert len(age_keys) == 118
    for source_id, source in sources.items():
        spec = source["spec"]
        if present(spec.get("quarantine_file")):
            quarantine_rows = read_csv(ROOT / spec["quarantine_file"])
            assert len(quarantine_rows) == int(spec["invalidated_count"]), source_id
    assert len(sources["employees"]["derived_keys"]) == 91


def city_coverage_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        spec = source["spec"]
        domain_type = "benchmark_only" if source_id == "age" else "annual_or_source_defined"
        benchmark_years = {2000, 2010, 2020} if source_id == "age" else set()
        for city in CITIES:
            all_rows = source["city_rows"].get(city, [])
            value_rows = source["city_value_rows"].get(city, [])
            row_years = sorted({key(row, source["city_col"], source["year_col"])[1] for row in all_rows})
            value_years = sorted({key(row, source["city_col"], source["year_col"])[1] for row in value_rows})
            if domain_type == "benchmark_only":
                missing_domain = sorted(benchmark_years - set(row_years))
                internal_gap = ""
            elif row_years:
                missing_domain = sorted(set(range(min(row_years), max(row_years) + 1)) - set(row_years))
                internal_gap = len(missing_domain)
            else:
                missing_domain = []
                internal_gap = ""
            result.append(
                {
                    "source_id": source_id,
                    "role": "target_label" if source_id == TARGET_SOURCE else "covariate",
                    "city": city,
                    "domain_type": domain_type,
                    "canonical_row_count": len(all_rows),
                    "value_row_count": len(value_rows),
                    "value_cell_count": sum(
                        1
                        for row in value_rows
                        for column in source["value_columns"]
                        if present(row.get(column))
                    ),
                    "row_year_min": min(row_years) if row_years else "",
                    "row_year_max": max(row_years) if row_years else "",
                    "value_year_min": min(value_years) if value_years else "",
                    "value_year_max": max(value_years) if value_years else "",
                    "row_internal_gap_count": internal_gap,
                    "missing_domain_years": ";".join(str(year) for year in missing_domain),
                    "derived_value_rows": sum(
                        1
                        for row in value_rows
                        if key(row, source["city_col"], source["year_col"]) in source["derived_keys"]
                    ),
                    "invalidated_keys": sum(
                        1
                        for item in source["quarantine_keys"]
                        if item[0] == city
                    ),
                    "source_status": spec["source_status"],
                    "forecast_origin_availability": spec["forecast_origin_availability"],
                    "coverage_warning": spec["regime_or_warning"],
                }
            )
    return result


def source_inventory_rows(
    sources: dict[str, dict[str, Any]],
    city_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in city_rows:
        by_source.setdefault(row["source_id"], []).append(row)
    result: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        spec = source["spec"]
        coverage = by_source[source_id]
        result.append(
            {
                "source_id": source_id,
                "role": "target_label" if source_id == TARGET_SOURCE else "covariate",
                "target_value_columns": spec["value_columns"] if source_id == TARGET_SOURCE else "",
                "raw_file": spec["raw_file"],
                "raw_sheet": spec["raw_sheet"],
                "canonical_file": spec["canonical_file"],
                "domain_type": "benchmark_only" if source_id == "age" else "annual_or_source_defined",
                "canonical_row_count": len(source["rows"]),
                "value_row_count": len(source["value_rows"]),
                "value_cell_count": source["value_cell_count"],
                "missing_value_cells": source["missing_value_cells"],
                "city_count_with_any_value": sum(row["value_row_count"] > 0 for row in coverage),
                "city_value_row_count_min": min(row["value_row_count"] for row in coverage),
                "city_value_row_count_median": median_or_blank(row["value_row_count"] for row in coverage),
                "city_value_row_count_max": max(row["value_row_count"] for row in coverage),
                "row_year_min": source["row_year_min"],
                "row_year_max": source["row_year_max"],
                "value_year_min": source["value_year_min"],
                "value_year_max": source["value_year_max"],
                "year_domain": spec["year_domain"],
                "source_status": spec["source_status"],
                "repair_count": spec["repair_count"],
                "invalidated_count": spec["invalidated_count"],
                "derived_count": spec["derived_count"],
                "quarantine_file": spec["quarantine_file"],
                "provenance_file": spec["provenance_file"],
                "forecast_origin_availability": spec["forecast_origin_availability"],
                "quality_warning": spec["regime_or_warning"],
                "validation_information_set_warning": spec["downstream_note"],
            }
        )
    return result


def target_label_summary(source: dict[str, Any], target_year: int) -> dict[str, Any]:
    value_column = source["value_columns"][0]
    target_rows = [
        row
        for row in source["rows"]
        if as_int(row.get(source["year_col"])) == target_year and present(row.get(value_column))
    ]
    target_cities = {row[source["city_col"]] for row in target_rows}
    history_counts = {
        city: sum(
            1
            for row in source["rows"]
            if row[source["city_col"]] == city
            and (year := as_int(row.get(source["year_col"]))) is not None
            and year < target_year
            and present(row.get(value_column))
        )
        for city in CITIES
    }
    history_ready = sum(count >= MIN_HISTORY_YEARS for count in history_counts.values())
    return {
        "target_label_column": value_column,
        "target_label_available_city_count": len(target_cities),
        "target_label_coverage_rate": f"{len(target_cities) / len(CITIES):.3f}",
        "cities_with_min_prior_target_history": history_ready,
        "prior_target_history_coverage_rate": f"{history_ready / len(CITIES):.3f}",
        "prior_target_history_min": min(history_counts.values()),
        "prior_target_history_median": median_or_blank(history_counts.values()),
        "prior_target_history_max": max(history_counts.values()),
    }


def matrix_rows(
    sources: dict[str, dict[str, Any]],
    target_summaries: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_year in TARGET_YEARS:
        target_summary = target_summaries[target_year]
        for policy in POLICIES:
            policy_cutoff = target_year - policy["cutoff_lag_years"]
            for source_id, source in sources.items():
                spec = source["spec"]
                values = [
                    row
                    for row in source["value_rows"]
                    if (year := as_int(row.get(source["year_col"]))) is not None
                    and year <= policy_cutoff
                ]
                source_years = sorted({as_int(row.get(source["year_col"])) for row in values})
                source_years = [year for year in source_years if year is not None]
                effective_cap = max(source_years) if source_years else ""
                available_cities = {
                    row[source["city_col"]]
                    for row in values
                    if any(present(row.get(column)) for column in source["value_columns"])
                }
                boundary_cities = {
                    row[source["city_col"]]
                    for row in values
                    if as_int(row.get(source["year_col"])) == policy_cutoff
                }
                value_cells = sum(
                    1
                    for row in values
                    for column in source["value_columns"]
                    if present(row.get(column))
                )
                derived_rows = sum(
                    1
                    for row in values
                    if key(row, source["city_col"], source["year_col"]) in source["derived_keys"]
                )
                invalidated_rows = sum(
                    year <= policy_cutoff
                    for city, year in source["quarantine_keys"]
                )
                latest_by_city = {
                    city: max(
                        [
                            as_int(row.get(source["year_col"]))
                            for row in values
                            if row[source["city_col"]] == city
                            and as_int(row.get(source["year_col"])) is not None
                        ],
                        default=None,
                    )
                    for city in CITIES
                }
                latest_values = [year for year in latest_by_city.values() if year is not None]
                if source_id == "age":
                    availability_class = "irregular_benchmark_domain"
                elif not available_cities:
                    availability_class = "unavailable"
                elif len(available_cities) < len(CITIES):
                    availability_class = "partial_city_coverage"
                else:
                    availability_class = "available"
                gaps: list[str] = []
                if not available_cities:
                    gaps.append("no non-empty canonical value at or before policy cutoff")
                elif len(available_cities) < len(CITIES):
                    gaps.append(f"only {len(available_cities)}/40 cities have a non-empty value")
                if source_id == "age":
                    gaps.append("benchmark-only source; no annual publication series inferred")
                if derived_rows:
                    gaps.append(f"derived rows present={derived_rows}; provenance remains distinct")
                if invalidated_rows:
                    gaps.append(f"invalidated/quarantined keys retained={invalidated_rows}; not reintroduced")
                gaps.append("forecast-origin availability is unknown from source materials")
                result.append(
                    {
                        "pseudo_target_year": target_year,
                        "policy_id": policy["policy_id"],
                        "policy_name": policy["policy_name"],
                        "availability_assumption": policy["assumption"],
                        "policy_allowed_through_year": policy_cutoff,
                        "source_id": source_id,
                        "role": "target_label" if source_id == TARGET_SOURCE else "covariate",
                        "domain_type": "benchmark_only" if source_id == "age" else "annual_or_source_defined",
                        "observed_fact_year_domain": spec["year_domain"],
                        "availability_evidence": "non-empty canonical values with year <= policy cutoff; no imputation",
                        "canonical_row_year_min": source["row_year_min"],
                        "canonical_row_year_max": source["row_year_max"],
                        "value_year_min_at_or_before_cutoff": min(source_years) if source_years else "",
                        "effective_source_cap_year": effective_cap,
                        "available_canonical_rows_at_or_before_cutoff": len(values),
                        "available_value_cells_at_or_before_cutoff": value_cells,
                        "available_cities_with_any_value": len(available_cities),
                        "available_cities_at_exact_policy_cutoff": len(boundary_cities),
                        "latest_value_year_median_by_city": median_or_blank(latest_values),
                        "latest_value_year_max_by_city": max(latest_values) if latest_values else "",
                        "derived_rows_available": derived_rows,
                        "non_derived_rows_available": len(values) - derived_rows,
                        "invalidated_quarantine_keys_at_or_before_cutoff": invalidated_rows,
                        "availability_class": availability_class,
                        "target_label_column": target_summary["target_label_column"],
                        "target_label_available_city_count": target_summary["target_label_available_city_count"],
                        "target_label_coverage_rate": target_summary["target_label_coverage_rate"],
                        "target_label_status": (
                            "unavailable_for_scoring"
                            if target_summary["target_label_available_city_count"] == 0
                            else (
                                "available_for_scoring"
                                if target_summary["target_label_available_city_count"] == len(CITIES)
                                else "partial_for_scoring"
                            )
                        ),
                        "quality_boundary": spec["regime_or_warning"],
                        "information_set_warning": "; ".join(gaps),
                    }
                )
    return result


def candidate_rows(
    sources: dict[str, dict[str, Any]],
    matrix: list[dict[str, Any]],
    target_summaries: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    index = {
        (row["pseudo_target_year"], row["policy_id"], row["source_id"]): row
        for row in matrix
    }
    result: list[dict[str, Any]] = []
    for target_year in TARGET_YEARS:
        summary = target_summaries[target_year]
        for policy in POLICIES:
            core_rows = [
                index[(target_year, policy["policy_id"], source_id)]
                for source_id in CORE_SOURCES
            ]
            core_coverage = {
                row["source_id"]: int(row["available_cities_with_any_value"])
                for row in core_rows
            }
            core_ready = [source_id for source_id, count in core_coverage.items() if count / len(CITIES) >= MIN_CITY_COVERAGE]
            core_failures = [source_id for source_id in CORE_SOURCES if source_id not in core_ready]
            label_ok = summary["target_label_available_city_count"] / len(CITIES) >= MIN_CITY_COVERAGE
            history_ok = summary["cities_with_min_prior_target_history"] / len(CITIES) >= MIN_CITY_COVERAGE
            core_ok = len(core_failures) == 0
            if summary["target_label_available_city_count"] == 0:
                classification = "exclude_no_target_label"
                reason = "population resident label is not observed for this year; no retrospective score can be formed"
            elif not label_ok:
                classification = "exclude_target_label_coverage"
                reason = (
                    f"target label coverage is {summary['target_label_available_city_count']}/40, below the "
                    f"{MIN_CITY_COVERAGE:.0%} structural screen"
                )
            elif not history_ok and not core_ok:
                classification = "exclude_history_and_core_gap"
                reason = (
                    f"only {summary['cities_with_min_prior_target_history']}/40 cities have at least "
                    f"{MIN_HISTORY_YEARS} prior target labels; core gaps: {','.join(core_failures)}"
                )
            elif not history_ok:
                classification = "exclude_insufficient_prior_target_history"
                reason = (
                    f"only {summary['cities_with_min_prior_target_history']}/40 cities have at least "
                    f"{MIN_HISTORY_YEARS} prior target labels"
                )
            elif not core_ok:
                classification = "conditional_core_source_gap"
                reason = f"core source coverage below {MIN_CITY_COVERAGE:.0%}: {','.join(core_failures)}"
            else:
                classification = "candidate"
                reason = (
                    f"target label {summary['target_label_available_city_count']}/40; prior history "
                    f"{summary['cities_with_min_prior_target_history']}/40; all six core sources meet "
                    f"the {MIN_CITY_COVERAGE:.0%} structural screen"
                )
            result.append(
                {
                    "pseudo_target_year": target_year,
                    "policy_id": policy["policy_id"],
                    "policy_name": policy["policy_name"],
                    "target_label_column": summary["target_label_column"],
                    "target_label_available_city_count": summary["target_label_available_city_count"],
                    "target_label_coverage_rate": summary["target_label_coverage_rate"],
                    "cities_with_min_prior_target_history": summary["cities_with_min_prior_target_history"],
                    "prior_target_history_coverage_rate": summary["prior_target_history_coverage_rate"],
                    "prior_target_history_min": summary["prior_target_history_min"],
                    "prior_target_history_median": summary["prior_target_history_median"],
                    "core_source_count": len(CORE_SOURCES),
                    "core_sources_meeting_coverage_screen": len(core_ready),
                    "core_sources_below_coverage_screen": ";".join(core_failures),
                    "core_city_coverage_min": min(core_coverage.values()),
                    "classification": classification,
                    "eligible_for_next_round": "yes" if classification == "candidate" else "no",
                    "reason": reason,
                }
            )
    return result


def main() -> None:
    assert not RAW_TEST.exists(), "test.xls must remain outside raw source inputs"
    assert QUARANTINE_SOURCE.exists(), "quarantine test.xls is missing"
    contract = load_contract()
    sources = load_source_data(contract)
    assert set(sources) == {row["dataset_id"] for row in contract}
    assert_s1_boundaries(contract, sources)
    target_source = sources[TARGET_SOURCE]
    global TARGET_VALUE_COLUMN
    TARGET_VALUE_COLUMN = target_source["value_columns"][0]
    city_rows = city_coverage_rows(sources)
    inventory = source_inventory_rows(sources, city_rows)
    target_summaries = {
        year: target_label_summary(target_source, year)
        for year in TARGET_YEARS
    }
    matrix = matrix_rows(sources, target_summaries)
    candidates = candidate_rows(sources, matrix, target_summaries)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "source_temporal_inventory.csv",
        inventory,
        list(inventory[0]),
    )
    write_csv(
        OUTPUT_DIR / "source_city_coverage.csv",
        city_rows,
        list(city_rows[0]),
    )
    write_csv(
        OUTPUT_DIR / "policy_definitions.csv",
        list(POLICIES),
        ["policy_id", "policy_name", "cutoff_lag_years", "assumption", "selection_reason"],
    )
    write_csv(
        OUTPUT_DIR / "pseudo_target_source_policy_matrix.csv",
        matrix,
        list(matrix[0]),
    )
    write_csv(
        OUTPUT_DIR / "candidate_origin_summary.csv",
        candidates,
        list(candidates[0]),
    )
    candidate_years = {
        policy["policy_id"]: [
            row["pseudo_target_year"]
            for row in candidates
            if row["policy_id"] == policy["policy_id"] and row["classification"] == "candidate"
        ]
        for policy in POLICIES
    }
    summary = {
        "plan_id": "S2-Plan01",
        "target_source": TARGET_SOURCE,
        "target_value_column": TARGET_VALUE_COLUMN,
        "target_years_in_matrix": [TARGET_YEARS[0], TARGET_YEARS[-1]],
        "source_count": len(sources),
        "source_inventory_rows": len(inventory),
        "source_city_coverage_rows": len(city_rows),
        "matrix_rows": len(matrix),
        "candidate_summary_rows": len(candidates),
        "policies": [policy["policy_id"] for policy in POLICIES],
        "core_sources": list(CORE_SOURCES),
        "screening_rules": {
            "minimum_prior_target_years_per_city": MIN_HISTORY_YEARS,
            "minimum_city_coverage": MIN_CITY_COVERAGE,
            "note": "structural screening only; not a final validation contract",
        },
        "candidate_years_by_policy": candidate_years,
        "target_label_unavailable_years": sorted({
            row["pseudo_target_year"]
            for row in candidates
            if row["target_label_available_city_count"] == 0
        }),
        "test_xls_used": False,
        "s1_contract_rows_read": len(contract),
        "s1_canonical_key_uniqueness_checked": True,
        "s1_boundary_corrections_checked": True,
    }
    write_json(OUTPUT_DIR / "matrix_summary.json", summary)
    print("S2_PLAN01_MATRIX_PASS")
    print(f"source_inventory_rows={len(inventory)}")
    print(f"source_city_coverage_rows={len(city_rows)}")
    print(f"matrix_rows={len(matrix)}")
    print(f"candidate_summary_rows={len(candidates)}")
    print("candidate_years_by_policy=" + json.dumps(candidate_years, ensure_ascii=False, sort_keys=True))
    print("target_label_unavailable_years=" + json.dumps(summary["target_label_unavailable_years"]))
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
