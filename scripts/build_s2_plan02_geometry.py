"""Build the S2-Plan02 source-specific historical replay evidence.

This script consumes the frozen S1 contract, canonical CSVs, provenance CSVs,
quarantine CSVs, and the already committed Plan01 artifacts.  It does not read
raw workbooks or ``test.xls``, rebuild S1, train a model, score a model, or
create a submission.

The two Plan02 scenarios are explicit replay assumptions.  The latest
canonical value year in the current 2023 snapshot is used as a relative
freshness proxy only; it is not treated as a proven publication date.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "s1_data_contract.csv"
PLAN01_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan01"
OUTPUT_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
QUARANTINE_SOURCE = ROOT / "data" / "quarantine" / "source" / "test.xls"
RAW_TEST = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集" / "test.xls"

CITIES = tuple(f"city{i}" for i in range(1, 41))
REFERENCE_YEAR = 2023
TARGET_YEARS = tuple(range(2001, REFERENCE_YEAR + 1))
TARGET_SOURCE = "population"
TARGET_VALUE_COLUMN = "常住人口（万人）"
TARGET_HORIZON_YEARS = 2
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
    {
        "scenario_id": "R1_REFERENCE_SNAPSHOT_LAG",
        "scenario_name": "reference snapshot relative lag",
        "scenario_family": "plan02_source_specific",
        "target_history_lag_years": TARGET_HORIZON_YEARS,
        "source_lag_rule": (
            "For each source, required_latest_year = pseudo_target_year minus "
            "its observed 2023 snapshot lag. Population target history remains "
            "separately capped at T-2."
        ),
        "benchmark_rule": (
            "Age remains benchmark-only; values are filtered by the required cap "
            "without creating annual rows or interpolating between benchmarks."
        ),
        "availability_claim_type": "observed_snapshot_relative_freshness_proxy",
        "publication_timing_proven": "no",
        "selection_reason": (
            "Primary real-like replay: preserves each logical source's observed "
            "relative freshness while keeping the target horizon at two years."
        ),
    },
    {
        "scenario_id": "R2_CONSERVATIVE_PUBLICATION_STRESS",
        "scenario_name": "conservative unknown-publication stress",
        "scenario_family": "plan02_source_specific_sensitivity",
        "target_history_lag_years": TARGET_HORIZON_YEARS,
        "source_lag_rule": (
            "Keep the observed lag for sources already at least two years stale; "
            "add one year only to annual sources whose observed snapshot lag is "
            "one year. Population target history and benchmark-only age keep "
            "their separate geometry."
        ),
        "benchmark_rule": (
            "Age remains benchmark-only and retains its observed three-year "
            "snapshot lag; no annual publication schedule is inferred."
        ),
        "availability_claim_type": "publication_unknown_conservative_sensitivity",
        "publication_timing_proven": "no",
        "selection_reason": (
            "Stress the only sources that appear in the 2022 snapshot by one "
            "additional year, without silently imposing a universal T-2 rule on "
            "the target horizon or benchmark source."
        ),
    },
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


def key(row: dict[str, str], city_col: str, year_col: str) -> tuple[str, int]:
    year = as_int(row.get(year_col))
    assert year is not None, row
    return row[city_col], year


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
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def median_or_blank(values: Iterable[int]) -> str | int:
    values = list(values)
    return int(statistics.median(values)) if values else ""


def rate(count: int, denominator: int = len(CITIES)) -> str:
    return f"{count / denominator:.3f}"


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
        row_years = [key(row, city_col, year_col)[1] for row in rows]
        value_years = [key(row, city_col, year_col)[1] for row in value_rows]
        city_rows: dict[str, list[dict[str, str]]] = {city: [] for city in CITIES}
        city_value_rows: dict[str, list[dict[str, str]]] = {city: [] for city in CITIES}
        for row in rows:
            city_rows.setdefault(row[city_col], []).append(row)
        for row in value_rows:
            city_value_rows.setdefault(row[city_col], []).append(row)
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


def assert_s1_boundaries(
    contract: list[dict[str, str]],
    sources: dict[str, dict[str, Any]],
) -> None:
    by_id = {row["dataset_id"]: row for row in contract}
    assert by_id["overall_income"]["source_status"] == "source_plus_invalidation"
    assert by_id["overall_income"]["repair_count"] == "0"
    assert by_id["overall_income"]["invalidated_count"] == "32"
    assert by_id["urban_income"]["source_status"] == "source_plus_label_repair"
    assert by_id["urban_income"]["repair_count"] == "2"
    assert by_id["age"]["year_domain"] == "2000;2010;2020"
    age_keys = sources["age"]["keys"]
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


def reference_rows(source_id: str, source: dict[str, Any]) -> list[dict[str, str]]:
    if source_id == TARGET_SOURCE:
        return [
            row
            for row in source["value_rows"]
            if present(row.get(TARGET_VALUE_COLUMN))
        ]
    return source["value_rows"]


def source_geometry_rows(sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        spec = source["spec"]
        value_rows = reference_rows(source_id, source)
        latest_year = max(
            key(row, source["city_col"], source["year_col"])[1]
            for row in value_rows
        )
        latest_rows = [
            row
            for row in value_rows
            if as_int(row.get(source["year_col"])) == latest_year
        ]
        latest_cities = {row[source["city_col"]] for row in latest_rows}
        latest_derived = sum(
            1
            for row in latest_rows
            if key(row, source["city_col"], source["year_col"])
            in source["derived_keys"]
        )
        latest_quarantine = sum(
            1
            for city, year in source["quarantine_keys"]
            if year == latest_year
        )
        total_derived = len(source["derived_keys"])
        total_quarantine = len(source["quarantine_keys"])
        has_boundary = (
            total_derived > 0
            or total_quarantine > 0
            or spec["source_status"] != "source_preserved"
        )
        if latest_derived:
            latest_provenance_class = "derived_rows_at_latest"
        elif latest_quarantine:
            latest_provenance_class = "quarantine_boundary_at_latest"
        elif has_boundary:
            latest_provenance_class = "provenance_sensitive_elsewhere"
        else:
            latest_provenance_class = "source_observation_only"
        domain_type = "benchmark_only" if source_id == "age" else "annual_or_source_defined"
        domain_years = spec["year_domain"] if source_id == "age" else ""
        rows.append(
            {
                "source_id": source_id,
                "role": "target_label" if source_id == TARGET_SOURCE else "covariate",
                "reference_value_columns": (
                    TARGET_VALUE_COLUMN if source_id == TARGET_SOURCE
                    else spec["value_columns"]
                ),
                "all_canonical_value_columns": spec["value_columns"],
                "domain_type": domain_type,
                "domain_years": domain_years,
                "canonical_value_year_min": min(
                    key(row, source["city_col"], source["year_col"])[1]
                    for row in value_rows
                ),
                "canonical_value_year_max": max(
                    key(row, source["city_col"], source["year_col"])[1]
                    for row in value_rows
                ),
                "latest_usable_year": latest_year,
                "reference_year": REFERENCE_YEAR,
                "observed_snapshot_freshness_lag_years": REFERENCE_YEAR - latest_year,
                "lag_evidence_class": "observed_canonical_snapshot_boundary",
                "lag_interpretation": (
                    "Observed max non-empty canonical value year minus 2023; "
                    "heuristic proxy only, not a publication-date fact."
                ),
                "publication_timing_proven": "no",
                "publication_timing_status": "unknown_from_source_materials",
                "latest_year_city_count": len(latest_cities),
                "latest_year_city_coverage_rate": rate(len(latest_cities)),
                "latest_year_missing_cities": ";".join(sorted(set(CITIES) - latest_cities)),
                "latest_year_value_row_count": len(latest_rows),
                "latest_year_value_cell_count": sum(
                    sum(present(row.get(column)) for column in spec["value_columns"])
                    for row in latest_rows
                ),
                "latest_year_derived_row_count": latest_derived,
                "latest_year_quarantine_key_count": latest_quarantine,
                "latest_information_contains_derived_rows": "yes" if latest_derived else "no",
                "latest_information_provenance_class": latest_provenance_class,
                "source_has_provenance_sensitive_history": "yes" if has_boundary else "no",
                "source_derived_count": total_derived,
                "source_invalidated_quarantine_count": total_quarantine,
                "source_status": spec["source_status"],
                "repair_count": spec["repair_count"],
                "invalidated_count": spec["invalidated_count"],
                "derived_count": spec["derived_count"],
                "quarantine_file": spec["quarantine_file"],
                "provenance_file": spec["provenance_file"],
                "quality_warning": spec["regime_or_warning"],
                "forecast_origin_availability": spec["forecast_origin_availability"],
                "downstream_note": spec["downstream_note"],
            }
        )
    return rows


def scenario_definitions() -> list[dict[str, Any]]:
    return [
        {
            **scenario,
            "target_geometry": (
                "Population target history is capped at T-2; target label at T "
                "is reserved for retrospective scoring only."
            ),
            "uses_uniform_global_cutoff": "no",
            "is_final_backtest_contract": "no",
            "model_scoring_run": "no",
        }
        for scenario in SCENARIOS
    ]


def scenario_by_id(scenario_id: str) -> dict[str, Any]:
    return next(scenario for scenario in SCENARIOS if scenario["scenario_id"] == scenario_id)


def required_lag_years(
    source_id: str,
    geometry: dict[str, Any],
    scenario_id: str,
) -> int:
    if source_id == TARGET_SOURCE:
        return TARGET_HORIZON_YEARS
    base_lag = int(geometry["observed_snapshot_freshness_lag_years"])
    if (
        scenario_id == "R2_CONSERVATIVE_PUBLICATION_STRESS"
        and geometry["domain_type"] == "annual_or_source_defined"
        and base_lag == 1
    ):
        return 2
    return base_lag


def rows_as_of(source: dict[str, Any], cap: int) -> list[dict[str, str]]:
    return [
        row
        for row in source["value_rows"]
        if (year := as_int(row.get(source["year_col"]))) is not None and year <= cap
    ]


def city_freshness_rows(
    source_id: str,
    source: dict[str, Any],
    geometry: dict[str, Any],
    scenario_id: str,
    target_year: int,
) -> list[dict[str, Any]]:
    required_lag = required_lag_years(source_id, geometry, scenario_id)
    required_year = target_year - required_lag
    capped_rows = rows_as_of(source, required_year)
    by_city: dict[str, list[dict[str, str]]] = {city: [] for city in CITIES}
    for row in capped_rows:
        by_city.setdefault(row[source["city_col"]], []).append(row)
    result: list[dict[str, Any]] = []
    benchmark_years = {
        int(part)
        for part in geometry["domain_years"].split(";")
        if part.strip()
    }
    for city in CITIES:
        city_rows = by_city.get(city, [])
        latest = max(
            (as_int(row.get(source["year_col"])) for row in city_rows),
            default=None,
        )
        latest_rows = [
            row
            for row in city_rows
            if as_int(row.get(source["year_col"])) == latest
        ] if latest is not None else []
        latest_derived = any(
            key(row, source["city_col"], source["year_col"])
            in source["derived_keys"]
            for row in latest_rows
        )
        quarantine_count = sum(
            1
            for quarantine_city, quarantine_year in source["quarantine_keys"]
            if quarantine_city == city and quarantine_year <= required_year
        )
        if latest is None:
            availability_state = "missing_as_of_required_cap"
            staleness = ""
            freshness_satisfied = "no"
        elif geometry["domain_type"] == "benchmark_only":
            staleness = required_year - latest
            if latest in benchmark_years and latest == required_year:
                availability_state = "benchmark_exact_required_year"
            elif latest in benchmark_years:
                availability_state = "benchmark_observation_older_than_cap"
            else:
                availability_state = "benchmark_domain_value"
            freshness_satisfied = "domain_aware"
        else:
            staleness = required_year - latest
            if latest == required_year:
                availability_state = "exact_required_year"
                freshness_satisfied = "yes"
            else:
                availability_state = "stale_prior_value"
                freshness_satisfied = "no_stale"
        if latest_derived:
            quality_class = "derived_or_provenance_sensitive_latest"
        elif quarantine_count:
            quality_class = "quarantine_boundary_exists_before_cap"
        elif availability_state == "missing_as_of_required_cap":
            quality_class = "missing"
        elif availability_state == "stale_prior_value":
            quality_class = "stale"
        else:
            quality_class = "source_observation_or_benchmark"
        result.append(
            {
                "pseudo_target_year": target_year,
                "scenario_id": scenario_id,
                "source_id": source_id,
                "role": geometry["role"],
                "city": city,
                "required_lag_years": required_lag,
                "required_latest_usable_year": required_year,
                "actual_latest_year_as_of_required_cap": latest if latest is not None else "",
                "staleness_years": staleness,
                "availability_state": availability_state,
                "freshness_requirement_status": freshness_satisfied,
                "latest_row_contains_derived_value": "yes" if latest_derived else "no",
                "quarantine_keys_at_or_before_required_cap": quarantine_count,
                "quality_class": quality_class,
                "domain_type": geometry["domain_type"],
                "information_set_warning": (
                    "Publication timing is unknown; observed cap is a replay assumption."
                ),
            }
        )
    return result


def aggregate_freshness_row(
    source_id: str,
    geometry: dict[str, Any],
    scenario_id: str,
    target_year: int,
    city_rows: list[dict[str, Any]],
    source: dict[str, Any],
    target_summary: dict[str, Any],
) -> dict[str, Any]:
    required_lag = required_lag_years(source_id, geometry, scenario_id)
    required_year = target_year - required_lag
    available = [
        row for row in city_rows
        if row["availability_state"] != "missing_as_of_required_cap"
    ]
    exact = [row for row in city_rows if row["availability_state"] == "exact_required_year"]
    stale = [row for row in city_rows if row["availability_state"] == "stale_prior_value"]
    missing = [row for row in city_rows if row["availability_state"] == "missing_as_of_required_cap"]
    staleness = [
        int(row["staleness_years"])
        for row in city_rows
        if row["staleness_years"] != ""
    ]
    latest_years = [
        int(row["actual_latest_year_as_of_required_cap"])
        for row in city_rows
        if row["actual_latest_year_as_of_required_cap"] != ""
    ]
    derived_latest = sum(
        row["latest_row_contains_derived_value"] == "yes"
        for row in city_rows
    )
    quarantine_count = sum(
        int(row["quarantine_keys_at_or_before_required_cap"])
        for row in city_rows
    )
    if geometry["domain_type"] == "benchmark_only":
        freshness_status = "irregular_benchmark_domain" if available else "unavailable"
        freshness_satisfied_count = sum(
            row["freshness_requirement_status"] == "domain_aware"
            for row in city_rows
        )
    elif not available:
        freshness_status = "unavailable"
        freshness_satisfied_count = 0
    elif len(exact) == len(available):
        freshness_status = "all_available_at_required_year"
        freshness_satisfied_count = len(exact)
    elif exact:
        freshness_status = "partial_exact_year_with_stale_or_missing"
        freshness_satisfied_count = len(exact)
    else:
        freshness_status = "stale_only"
        freshness_satisfied_count = 0
    warnings = [
        "publication timing unknown; required cap is an availability assumption",
    ]
    if geometry["domain_type"] == "benchmark_only":
        warnings.append("benchmark-only source; no annual sequence inferred")
    if derived_latest or source["derived_keys"]:
        warnings.append("derived/provenance-sensitive rows remain distinguishable")
    if source["quarantine_keys"]:
        warnings.append("quarantine/invalidation stays unavailable and is not reintroduced")
    return {
        "pseudo_target_year": target_year,
        "scenario_id": scenario_id,
        "scenario_name": scenario_by_id(scenario_id)["scenario_name"],
        "source_id": source_id,
        "role": geometry["role"],
        "domain_type": geometry["domain_type"],
        "observed_reference_latest_year": geometry["latest_usable_year"],
        "observed_reference_lag_years": geometry["observed_snapshot_freshness_lag_years"],
        "required_lag_years": required_lag,
        "required_latest_usable_year": required_year,
        "actual_latest_year_as_of_required_cap": max(latest_years) if latest_years else "",
        "actual_latest_year_min_by_city": min(latest_years) if latest_years else "",
        "actual_latest_year_median_by_city": median_or_blank(latest_years),
        "actual_latest_year_max_by_city": max(latest_years) if latest_years else "",
        "available_cities_with_any_value_as_of_cap": len(available),
        "available_city_coverage_rate_as_of_cap": rate(len(available)),
        "cities_with_value_at_exact_required_year": len(exact),
        "exact_required_year_city_coverage_rate": rate(len(exact)),
        "cities_with_stale_prior_value": len(stale),
        "cities_missing_as_of_required_cap": len(missing),
        "staleness_min_years": min(staleness) if staleness else "",
        "staleness_median_years": median_or_blank(staleness),
        "staleness_max_years": max(staleness) if staleness else "",
        "freshness_status": freshness_status,
        "freshness_satisfied_city_count": freshness_satisfied_count,
        "freshness_satisfied_city_rate": rate(freshness_satisfied_count),
        "derived_rows_available_as_of_cap": sum(
            1
            for row in rows_as_of(source, required_year)
            if key(row, source["city_col"], source["year_col"])
            in source["derived_keys"]
        ),
        "derived_latest_city_count": derived_latest,
        "invalidated_quarantine_keys_at_or_before_cap": quarantine_count,
        "reference_latest_year_city_count": geometry["latest_year_city_count"],
        "reference_latest_year_city_coverage_rate": geometry["latest_year_city_coverage_rate"],
        "coverage_delta_vs_reference_latest_year": len(available) - int(geometry["latest_year_city_count"]),
        "structural_90pct_coverage_screen": "pass" if len(available) >= len(CITIES) * MIN_CITY_COVERAGE else "fail",
        "target_label_available_city_count": target_summary["target_label_available_city_count"],
        "target_label_coverage_rate": target_summary["target_label_coverage_rate"],
        "target_history_cutoff_year": target_summary["history_cutoff_year"],
        "target_history_cities_with_min_3_labels": target_summary["history_ready_city_count"],
        "target_history_coverage_rate": target_summary["history_ready_coverage_rate"],
        "information_set_warning": "; ".join(warnings),
        "quality_boundary": geometry["quality_warning"],
    }


def target_summary(sources: dict[str, dict[str, Any]], target_year: int) -> dict[str, Any]:
    source = sources[TARGET_SOURCE]
    target_rows = [
        row
        for row in source["value_rows"]
        if as_int(row.get(source["year_col"])) == target_year
        and present(row.get(TARGET_VALUE_COLUMN))
    ]
    target_cities = {row[source["city_col"]] for row in target_rows}
    cutoff = target_year - TARGET_HORIZON_YEARS
    history_counts = {
        city: sum(
            1
            for row in source["value_rows"]
            if row[source["city_col"]] == city
            and (year := as_int(row.get(source["year_col"]))) is not None
            and year <= cutoff
            and present(row.get(TARGET_VALUE_COLUMN))
        )
        for city in CITIES
    }
    ready = sum(count >= MIN_HISTORY_YEARS for count in history_counts.values())
    return {
        "target_label_column": TARGET_VALUE_COLUMN,
        "target_label_available_city_count": len(target_cities),
        "target_label_coverage_rate": rate(len(target_cities)),
        "history_cutoff_year": cutoff,
        "history_ready_city_count": ready,
        "history_ready_coverage_rate": rate(ready),
        "history_min_labels_per_city": min(history_counts.values()),
        "history_median_labels_per_city": median_or_blank(history_counts.values()),
        "history_max_labels_per_city": max(history_counts.values()),
    }


def build_matrix(
    sources: dict[str, dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    geometry_by_id = {row["source_id"]: row for row in geometry_rows}
    aggregate_rows: list[dict[str, Any]] = []
    city_rows_all: list[dict[str, Any]] = []
    target_summaries = {
        year: target_summary(sources, year)
        for year in TARGET_YEARS
    }
    for scenario in SCENARIOS:
        scenario_id = scenario["scenario_id"]
        for target_year in TARGET_YEARS:
            summary = target_summaries[target_year]
            for source_id, source in sources.items():
                geometry = geometry_by_id[source_id]
                city_rows = city_freshness_rows(
                    source_id,
                    source,
                    geometry,
                    scenario_id,
                    target_year,
                )
                city_rows_all.extend(city_rows)
                aggregate_rows.append(
                    aggregate_freshness_row(
                        source_id,
                        geometry,
                        scenario_id,
                        target_year,
                        city_rows,
                        source,
                        summary,
                    )
                )
    return aggregate_rows, city_rows_all, target_summaries


def build_feasibility_summary(
    matrix: list[dict[str, Any]],
    target_summaries: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    index = {
        (row["pseudo_target_year"], row["scenario_id"], row["source_id"]): row
        for row in matrix
    }
    result: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_id = scenario["scenario_id"]
        for target_year in TARGET_YEARS:
            summary = target_summaries[target_year]
            core_rows = [
                index[(target_year, scenario_id, source_id)]
                for source_id in CORE_SOURCES
            ]
            coverage_failures = [
                row["source_id"]
                for row in core_rows
                if row["structural_90pct_coverage_screen"] != "pass"
            ]
            exact_freshness_partial = [
                row["source_id"]
                for row in core_rows
                if row["freshness_status"] not in {"all_available_at_required_year"}
            ]
            label_ok = summary["target_label_available_city_count"] >= len(CITIES) * MIN_CITY_COVERAGE
            history_ok = summary["history_ready_city_count"] >= len(CITIES) * MIN_CITY_COVERAGE
            core_ok = not coverage_failures
            if summary["target_label_available_city_count"] == 0:
                classification = "exclude_no_target_label"
                bucket = "exclude"
                reason = (
                    "population resident label is not observed for this year; "
                    "no retrospective score can be formed"
                )
            elif not label_ok:
                classification = "exclude_target_label_coverage"
                bucket = "exclude"
                reason = (
                    f"target label coverage is {summary['target_label_available_city_count']}/40, "
                    f"below the {MIN_CITY_COVERAGE:.0%} structural screen"
                )
            elif not history_ok and not core_ok:
                classification = "conditional_history_and_core_gap"
                bucket = "conditional"
                reason = (
                    f"target history through T-2 reaches only {summary['history_ready_city_count']}/40 "
                    f"cities with at least {MIN_HISTORY_YEARS} labels; core gaps: "
                    f"{','.join(coverage_failures)}"
                )
            elif not history_ok:
                classification = "conditional_target_horizon_history_gap"
                bucket = "conditional"
                reason = (
                    f"target history through T-2 reaches only {summary['history_ready_city_count']}/40 "
                    f"cities with at least {MIN_HISTORY_YEARS} labels"
                )
            elif not core_ok:
                classification = "conditional_core_source_coverage_gap"
                bucket = "conditional"
                reason = (
                    f"core source as-of coverage below {MIN_CITY_COVERAGE:.0%}: "
                    f"{','.join(coverage_failures)}"
                )
            else:
                classification = "real_like_candidate"
                bucket = "candidate"
                reason = (
                    f"target label {summary['target_label_available_city_count']}/40; "
                    f"target history through T-2 {summary['history_ready_city_count']}/40; "
                    "all six provisional core sources meet the structural as-of screen"
                )
            result.append(
                {
                    "pseudo_target_year": target_year,
                    "scenario_id": scenario_id,
                    "scenario_name": scenario["scenario_name"],
                    "target_label_column": TARGET_VALUE_COLUMN,
                    "target_label_available_city_count": summary["target_label_available_city_count"],
                    "target_label_coverage_rate": summary["target_label_coverage_rate"],
                    "target_history_horizon_years": TARGET_HORIZON_YEARS,
                    "target_history_cutoff_year": summary["history_cutoff_year"],
                    "cities_with_min_3_target_labels_through_cutoff": summary["history_ready_city_count"],
                    "target_history_coverage_rate": summary["history_ready_coverage_rate"],
                    "core_source_count": len(CORE_SOURCES),
                    "core_sources_meeting_90pct_asof_screen": len(CORE_SOURCES) - len(coverage_failures),
                    "core_sources_below_90pct_asof_screen": ";".join(coverage_failures),
                    "core_sources_with_nonfull_exact_cap_freshness": ";".join(exact_freshness_partial),
                    "classification": classification,
                    "status_bucket": bucket,
                    "eligible_for_next_round": "yes" if bucket == "candidate" else "no",
                    "screening_rules_are_final_contract": "no",
                    "reason": reason,
                }
            )
    return result


def read_plan01_candidates() -> dict[tuple[int, str], dict[str, str]]:
    rows = read_csv(PLAN01_DIR / "candidate_origin_summary.csv")
    return {
        (int(row["pseudo_target_year"]), row["policy_id"]): row
        for row in rows
    }


def plan01_bucket(classification: str) -> str:
    if classification == "candidate":
        return "candidate"
    if classification.startswith("conditional_"):
        return "conditional"
    return "exclude"


def build_comparison(feasibility: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan01 = read_plan01_candidates()
    current = {
        (int(row["pseudo_target_year"]), row["scenario_id"]): row
        for row in feasibility
    }
    result: list[dict[str, Any]] = []
    for target_year in TARGET_YEARS:
        p1 = plan01[(target_year, "P1_ASOF_T_MINUS_1")]
        p2 = plan01[(target_year, "P2_ASOF_T_MINUS_2")]
        r1 = current[(target_year, "R1_REFERENCE_SNAPSHOT_LAG")]
        r2 = current[(target_year, "R2_CONSERVATIVE_PUBLICATION_STRESS")]
        p1_candidate = p1["classification"] == "candidate"
        p2_candidate = p2["classification"] == "candidate"
        p1_bucket = plan01_bucket(p1["classification"])
        p2_bucket = plan01_bucket(p2["classification"])
        r1_candidate = r1["status_bucket"] == "candidate"
        r2_candidate = r2["status_bucket"] == "candidate"
        if p1_bucket != r1["status_bucket"] or p2_bucket != r2["status_bucket"]:
            if (p1_candidate or p2_candidate) and target_year <= 2009 and r1_candidate is False:
                change_reason = "target history is now capped at T-2, removing the Plan01 one-step-history shortcut"
            else:
                change_reason = "source-specific lag, target horizon, or freshness-aware coverage changed the status bucket"
        elif r1["classification"] != r2["classification"]:
            change_reason = "reference and conservative source-specific scenarios disagree"
        else:
            change_reason = "candidate/conditional/exclude bucket is unchanged; freshness diagnostics still differ"
        result.append(
            {
                "pseudo_target_year": target_year,
                "plan01_p1_classification": p1["classification"],
                "plan01_p1_status_bucket": p1_bucket,
                "plan01_p1_candidate": "yes" if p1_candidate else "no",
                "plan01_p2_classification": p2["classification"],
                "plan01_p2_status_bucket": p2_bucket,
                "plan01_p2_candidate": "yes" if p2_candidate else "no",
                "plan02_r1_classification": r1["classification"],
                "plan02_r1_candidate": "yes" if r1_candidate else "no",
                "plan02_r2_classification": r2["classification"],
                "plan02_r2_candidate": "yes" if r2_candidate else "no",
                "r1_bucket": r1["status_bucket"],
                "r2_bucket": r2["status_bucket"],
                "candidate_set_change_r1_vs_plan01_p1": "yes" if p1_candidate != r1_candidate else "no",
                "candidate_set_change_r2_vs_plan01_p2": "yes" if p2_candidate != r2_candidate else "no",
                "scenario_disagreement": "yes" if r1["classification"] != r2["classification"] else "no",
                "change_reason": change_reason,
            }
        )
    return result


def heuristic_review_rows() -> list[dict[str, Any]]:
    return [
        {
            "heuristic_id": "minimum_city_coverage",
            "plan01_value": "90%",
            "plan02_use": "retained as a structural diagnostic screen for core sources",
            "status": "not_final_validation_contract",
            "reason": (
                "Coverage is necessary to expose missing information, but it does not "
                "prove publication timing, feature usefulness, independence, or model validity."
            ),
        },
        {
            "heuristic_id": "minimum_prior_target_years_per_city",
            "plan01_value": "3 labels per city",
            "plan02_use": "reapplied only to target labels at or before T-2",
            "status": "target-horizon_structural_screen_only",
            "reason": (
                "The horizon correction prevents a 2-year target task from silently using "
                "the T-1 label; the count still remains a feasibility heuristic, not a score contract."
            ),
        },
        {
            "heuristic_id": "core_source_set",
            "plan01_value": ";".join(CORE_SOURCES),
            "plan02_use": "provisional coverage screen only",
            "status": "not_final_source_contract",
            "reason": (
                "The six sources were inherited for comparability; no model evidence or "
                "independence analysis was run, and optional sources remain auditable."
            ),
        },
    ]


def write_artifacts(
    geometry_rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    city_matrix: list[dict[str, Any]],
    feasibility: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    target_summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "reference_forecast_geometry.csv", geometry_rows, list(geometry_rows[0]))
    write_csv(OUTPUT_DIR / "replay_scenario_definitions.csv", scenarios, list(scenarios[0]))
    write_csv(OUTPUT_DIR / "origin_source_scenario_freshness_matrix.csv", matrix, list(matrix[0]))
    write_csv(
        OUTPUT_DIR / "origin_city_source_scenario_freshness.csv",
        city_matrix,
        list(city_matrix[0]),
    )
    write_csv(OUTPUT_DIR / "origin_feasibility_summary.csv", feasibility, list(feasibility[0]))
    write_csv(
        OUTPUT_DIR / "plan01_vs_plan02_candidate_origin_comparison.csv",
        comparison,
        list(comparison[0]),
    )
    heuristics = heuristic_review_rows()
    write_csv(OUTPUT_DIR / "screening_heuristics_review.csv", heuristics, list(heuristics[0]))
    candidate_years = {
        scenario_id: sorted({
            int(row["pseudo_target_year"])
            for row in feasibility
            if row["scenario_id"] == scenario_id
            and row["status_bucket"] == "candidate"
        })
        for scenario_id in (scenario["scenario_id"] for scenario in SCENARIOS)
    }
    conditional_years = {
        scenario_id: sorted({
            int(row["pseudo_target_year"])
            for row in feasibility
            if row["scenario_id"] == scenario_id
            and row["status_bucket"] == "conditional"
        })
        for scenario_id in (scenario["scenario_id"] for scenario in SCENARIOS)
    }
    excluded_years = {
        scenario_id: sorted({
            int(row["pseudo_target_year"])
            for row in feasibility
            if row["scenario_id"] == scenario_id
            and row["status_bucket"] == "exclude"
        })
        for scenario_id in (scenario["scenario_id"] for scenario in SCENARIOS)
    }
    summary = {
        "plan_id": "S2-Plan02",
        "reference_year": REFERENCE_YEAR,
        "target_source": TARGET_SOURCE,
        "target_value_column": TARGET_VALUE_COLUMN,
        "target_proxy_status": "conditional_proxy_not_uniquely_resolved_by_source_materials",
        "target_history_horizon_years": TARGET_HORIZON_YEARS,
        "target_horizon_basis": (
            "Population resident proxy latest observed value year is 2021; "
            "2023 minus 2021 equals two-year snapshot horizon."
        ),
        "publication_timing_proven": False,
        "source_count": len(geometry_rows),
        "scenario_count": len(scenarios),
        "origin_count": len(TARGET_YEARS),
        "matrix_rows": len(matrix),
        "city_matrix_rows": len(city_matrix),
        "feasibility_rows": len(feasibility),
        "candidate_years_by_scenario": candidate_years,
        "conditional_years_by_scenario": conditional_years,
        "excluded_years_by_scenario": excluded_years,
        "screening_rules": {
            "minimum_city_coverage": MIN_CITY_COVERAGE,
            "minimum_prior_target_years_per_city": MIN_HISTORY_YEARS,
            "core_sources": list(CORE_SOURCES),
            "note": "All three remain structural heuristics, not final validation contracts.",
        },
        "s1_boundary_corrections_checked": True,
        "s1_canonical_key_uniqueness_checked": True,
        "age_benchmark_domain_preserved": True,
        "employee_provenance_preserved": True,
        "test_xls_used": False,
        "model_scoring_run": False,
        "target_summaries": target_summaries,
    }
    write_json(OUTPUT_DIR / "geometry_summary.json", summary)
    return summary


def main() -> None:
    assert not RAW_TEST.exists(), "test.xls must remain outside raw source inputs"
    assert QUARANTINE_SOURCE.exists(), "quarantine test.xls is missing"
    contract = load_contract()
    sources = load_source_data(contract)
    assert set(sources) == {row["dataset_id"] for row in contract}
    assert_s1_boundaries(contract, sources)
    geometry_rows = source_geometry_rows(sources)
    geometry_by_id = {row["source_id"]: row for row in geometry_rows}
    assert geometry_by_id[TARGET_SOURCE]["latest_usable_year"] == 2021
    assert geometry_by_id[TARGET_SOURCE]["observed_snapshot_freshness_lag_years"] == 2
    assert geometry_by_id["age"]["domain_type"] == "benchmark_only"
    scenarios = scenario_definitions()
    matrix, city_matrix, target_summaries = build_matrix(sources, geometry_rows)
    feasibility = build_feasibility_summary(matrix, target_summaries)
    comparison = build_comparison(feasibility)
    summary = write_artifacts(
        geometry_rows,
        scenarios,
        matrix,
        city_matrix,
        feasibility,
        comparison,
        target_summaries,
    )
    assert summary["candidate_years_by_scenario"]["R1_REFERENCE_SNAPSHOT_LAG"] == list(range(2010, 2022))
    assert summary["candidate_years_by_scenario"]["R2_CONSERVATIVE_PUBLICATION_STRESS"] == list(range(2010, 2022))
    print("S2_PLAN02_GEOMETRY_PASS")
    print(f"reference_geometry_rows={len(geometry_rows)}")
    print(f"scenario_rows={len(scenarios)}")
    print(f"aggregate_matrix_rows={len(matrix)}")
    print(f"city_matrix_rows={len(city_matrix)}")
    print(f"feasibility_rows={len(feasibility)}")
    print("candidate_years_by_scenario=" + json.dumps(summary["candidate_years_by_scenario"], ensure_ascii=False, sort_keys=True))
    print("conditional_years_by_scenario=" + json.dumps(summary["conditional_years_by_scenario"], ensure_ascii=False, sort_keys=True))
    print("target_history_horizon_years=2")
    print("test_xls_used=False")
    print("model_scoring_run=False")


if __name__ == "__main__":
    main()
