"""Build S2-Plan04 official population source reconciliation evidence.

The source workbook has already been replaced by the exact official file named
by the Plan.  This utility records the raw-source scope, compares the old
starting-commit snapshot with the refreshed S1 canonical layer, and recomputes
only the population-derived facts used by S2 Plan01-03.  It never rewrites the
historical Plan01-03 handoffs and never runs model scoring.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan04"
RAW_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
POP_RAW_REL = "data/raw/城市人口分析与预测/数据集/人口规模.xlsx"
POP_CANONICAL_REL = "data/normalized/人口规模.csv"
CONTRACT_REL = "docs/s1_data_contract.csv"
QUALITY_REL = "docs/s1_quality_issue_registry.csv"
STARTING_COMMIT = "a394ea204ddcc26a5767a34bf47ede37fa4c9e64"
EXPECTED_OFFICIAL_SHA256 = "9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0"
OFFICIAL_DOWNLOAD = Path(r"D:\Downloads\official_population.xlsx")
RAW_TEST = RAW_DIR / "test.xls"
QUARANTINE_TEST = ROOT / "data" / "quarantine" / "source" / "test.xls"
CITIES = tuple(f"city{i}" for i in range(1, 41))
POPULATION_COLUMNS = ("常住人口（万人）", "户籍人口（万人）")
HISTORICAL_HANDOFF_DIRS = (
    "reports/S2_HANDOFFS/S2-Plan01",
    "reports/S2_HANDOFFS/S2-Plan02",
    "reports/S2_HANDOFFS/S2-Plan03",
)


def load_module(name: str, relative_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN01 = load_module("s2_plan01_for_plan04", "scripts/build_s2_plan01_matrix.py")
PLAN02 = load_module("s2_plan02_for_plan04", "scripts/build_s2_plan02_geometry.py")
PLAN03 = load_module("s2_plan03_for_plan04", "scripts/build_s2_plan03_target_metric_contract.py")


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
    return float(value) if present(value) else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob(relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{STARTING_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_git_csv(relative_path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(git_blob(relative_path).decode("utf-8-sig"))))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_git_json(relative_path: str) -> dict[str, Any]:
    return json.loads(git_blob(relative_path).decode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    assert rows or fields
    fieldnames = list(fields or rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in sorted(value))
    return str(value)


def population_profile(rows: list[dict[str, str]]) -> dict[str, Any]:
    keys = [(row["城市名称"], as_int(row["年份"])) for row in rows]
    assert all(year is not None for _, year in keys)
    profile: dict[str, Any] = {
        "canonical_row_count": len(rows),
        "unique_key_count": len(set(keys)),
        "duplicate_key_rows": len(rows) - len(set(keys)),
        "city_count": len({city for city, _ in keys}),
        "year_min": min(year for _, year in keys),
        "year_max": max(year for _, year in keys),
    }
    for column in POPULATION_COLUMNS:
        values = [as_float(row.get(column)) for row in rows]
        observed = [value for value in values if value is not None]
        profile[f"{column}_nonempty_count"] = len(observed)
        profile[f"{column}_missing_count"] = len(values) - len(observed)
        profile[f"{column}_2001_label_count"] = sum(
            as_int(row["年份"]) == 2001 and present(row.get(column)) for row in rows
        )
        profile[f"{column}_latest_year"] = max(
            as_int(row["年份"]) for row in rows if present(row.get(column))
        )
        profile[f"{column}_latest_city_count"] = len({
            row["城市名称"]
            for row in rows
            if as_int(row["年份"]) == profile[f"{column}_latest_year"] and present(row.get(column))
        })
    profile["canonical_missing_cell_count"] = sum(
        not present(row.get(column)) for row in rows for column in POPULATION_COLUMNS
    )
    city40_2001 = next(
        row for row in rows if row["城市名称"] == "city40" and as_int(row["年份"]) == 2001
    ) if any(row["城市名称"] == "city40" and as_int(row["年份"]) == 2001 for row in rows) else None
    profile["city40_2001_resident"] = "" if city40_2001 is None else city40_2001.get(POPULATION_COLUMNS[0], "")
    profile["city40_2001_registered"] = "" if city40_2001 is None else city40_2001.get(POPULATION_COLUMNS[1], "")
    return profile


def history_ready_count(rows: list[dict[str, str]], column: str, cutoff: int) -> int:
    counts = {
        city: sum(
            as_int(row["年份"]) is not None
            and as_int(row["年份"]) <= cutoff
            and present(row.get(column))
            for row in rows
            if row["城市名称"] == city
        )
        for city in CITIES
    }
    return sum(count >= 3 for count in counts.values())


def current_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True
    ).stdout.strip()


def historical_handoffs_are_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *HISTORICAL_HANDOFF_DIRS],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() == ""


def raw_hash_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert OFFICIAL_DOWNLOAD.exists(), OFFICIAL_DOWNLOAD
    official_sha = sha256_file(OFFICIAL_DOWNLOAD)
    assert official_sha == EXPECTED_OFFICIAL_SHA256, official_sha
    rows: list[dict[str, Any]] = []
    changed: list[str] = []
    for path in sorted(RAW_DIR.glob("*.xlsx")):
        relative = path.relative_to(ROOT).as_posix()
        before_sha = sha256_bytes(git_blob(relative))
        after_sha = sha256_file(path)
        is_changed = before_sha != after_sha
        if is_changed:
            changed.append(relative)
        rows.append(
            {
                "path": relative,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "changed": "yes" if is_changed else "no",
                "expected_change": "yes" if relative == POP_RAW_REL else "no",
                "scope_result": "expected" if is_changed == (relative == POP_RAW_REL) else "unexpected",
            }
        )
    assert changed == [POP_RAW_REL], changed
    assert official_sha == sha256_file(RAW_DIR / "人口规模.xlsx")
    return rows, {
        "official_download_path": str(OFFICIAL_DOWNLOAD),
        "expected_official_sha256": EXPECTED_OFFICIAL_SHA256,
        "official_download_sha256": official_sha,
        "repository_population_sha256": sha256_file(RAW_DIR / "人口规模.xlsx"),
        "changed_raw_xlsx_paths": changed,
        "other_raw_xlsx_unchanged": all(row["scope_result"] == "expected" for row in rows),
    }


def population_reconciliation_row(download_info: dict[str, Any], official_frame: pd.DataFrame) -> dict[str, Any]:
    city40 = official_frame[(official_frame["城市名称"] == "city40") & (official_frame["年份"] == 2001)]
    assert len(city40) == 1
    row = city40.iloc[0]
    assert float(row[POPULATION_COLUMNS[0]]) == 284
    assert pd.isna(row[POPULATION_COLUMNS[1]])
    return {
        "source_file": POP_RAW_REL,
        "official_download_path": download_info["official_download_path"],
        "expected_sha256": download_info["expected_official_sha256"],
        "download_sha256": download_info["official_download_sha256"],
        "repository_after_sha256": download_info["repository_population_sha256"],
        "sha256_match": "yes",
        "official_sheet": "Sheet1",
        "official_raw_row_count": len(official_frame),
        "official_city_count": official_frame["城市名称"].nunique(),
        "city40_2001_resident": row[POPULATION_COLUMNS[0]],
        "city40_2001_registered": "missing",
        "source_scope_decision": "replace population workbook only; do not replace other six XLSX",
    }


def selected_contract_row(row: dict[str, str], snapshot: str) -> dict[str, Any]:
    fields = (
        "dataset_id",
        "raw_file",
        "raw_sheet",
        "canonical_file",
        "row_count",
        "raw_row_count",
        "city_count",
        "year_min",
        "year_max",
        "actual_key_count",
        "missing_cell_count",
        "raw_missing_cell_count",
        "source_status",
        "repair_count",
        "invalidated_count",
        "derived_count",
        "observed_coverage_status",
        "regime_or_warning",
    )
    return {"snapshot": snapshot, **{field: row.get(field, "") for field in fields}}


def quality_population_row(rows: list[dict[str, str]], snapshot: str) -> dict[str, Any]:
    row = next(item for item in rows if item["issue_id"] == "QI-004")
    fields = (
        "issue_id",
        "dataset",
        "affected_keys_or_coverage",
        "status",
        "evidence",
        "decision",
        "downstream_warning",
        "source_ref",
    )
    return {"snapshot": snapshot, **{field: row.get(field, "") for field in fields}}


def add_delta(
    rows: list[dict[str, Any]],
    fact_id: str,
    layer: str,
    old_value: Any,
    new_value: Any,
    impact_level: str,
    downstream_implication: str,
) -> None:
    old_text = display(old_value)
    new_text = display(new_value)
    rows.append(
        {
            "fact_id": fact_id,
            "layer": layer,
            "old_value": old_text,
            "new_value": new_text,
            "changed": "yes" if old_text != new_text else "no",
            "impact_level": impact_level,
            "downstream_implication": downstream_implication,
        }
    )


def bucket_years(rows: list[dict[str, Any]], scenario_id: str, bucket: str, year_field: str = "pseudo_target_year") -> str:
    return ";".join(
        str(row[year_field])
        for row in rows
        if row.get("scenario_id") == scenario_id and row.get("status_bucket") == bucket
    )


def plan03_bucket_years(rows: list[dict[str, Any]], target_id: str, scenario_id: str, bucket: str) -> str:
    return ";".join(
        str(row["pseudo_target_year"])
        for row in rows
        if row["target_id"] == target_id
        and row["scenario_id"] == scenario_id
        and row["status_bucket"] == bucket
    )


def find_row(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    assert len(matches) == 1, (criteria, len(matches))
    return matches[0]


def recompute_current_facts() -> dict[str, Any]:
    p1_contract = PLAN01.load_contract()
    p1_sources = PLAN01.load_source_data(p1_contract)
    PLAN01.assert_s1_boundaries(p1_contract, p1_sources)
    PLAN01.TARGET_VALUE_COLUMN = p1_sources["population"]["value_columns"][0]
    p1_city_rows = PLAN01.city_coverage_rows(p1_sources)
    p1_inventory = PLAN01.source_inventory_rows(p1_sources, p1_city_rows)
    p1_target_source = p1_sources["population"]
    p1_target_summaries = {
        year: PLAN01.target_label_summary(p1_target_source, year)
        for year in PLAN01.TARGET_YEARS
    }
    p1_matrix = PLAN01.matrix_rows(p1_sources, p1_target_summaries)
    p1_candidates = PLAN01.candidate_rows(p1_sources, p1_matrix, p1_target_summaries)

    p2_contract = PLAN02.load_contract()
    p2_sources = PLAN02.load_source_data(p2_contract)
    PLAN02.assert_s1_boundaries(p2_contract, p2_sources)
    PLAN02.TARGET_VALUE_COLUMN = p2_sources["population"]["value_columns"][0]
    p2_geometry_rows = PLAN02.source_geometry_rows(p2_sources)
    p2_matrix, p2_city_matrix, p2_target_summaries = PLAN02.build_matrix(p2_sources, p2_geometry_rows)
    p2_feasibility = PLAN02.build_feasibility_summary(p2_matrix, p2_target_summaries)

    p3_contract = PLAN03.load_contract()
    PLAN03.assert_s1_boundaries(p3_contract)
    p3_population_rows = PLAN03.load_population_rows()
    p3_age_rows = PLAN03.load_age_rows()
    p3_geometries = {
        target["target_id"]: PLAN03.target_geometry(p3_population_rows, target)
        for target in PLAN03.TARGETS
    }
    p3_age_sums = PLAN03.complete_age_sums(p3_age_rows)
    p3_details = PLAN03.consistency_detail(p3_age_sums, p3_geometries)
    p3_offsets = PLAN03.offset_consistency_rows(p3_age_sums, p3_geometries)
    p3_consistency = PLAN03.consistency_summary(p3_details, p3_offsets)
    p3_plan02_matrix = PLAN03.load_plan02_matrix()
    p3_target_matrix = PLAN03.target_origin_geometry(p3_population_rows, p3_geometries, p3_plan02_matrix)
    p3_target_comparison = PLAN03.target_origin_comparison(p3_target_matrix)
    p3_target_summary = PLAN03.target_hypothesis_summary(p3_geometries, p3_target_matrix)
    return {
        "p1_inventory": p1_inventory,
        "p1_target_summaries": p1_target_summaries,
        "p1_candidates": p1_candidates,
        "p2_geometry_rows": p2_geometry_rows,
        "p2_target_summaries": p2_target_summaries,
        "p2_feasibility": p2_feasibility,
        "p3_geometries": p3_geometries,
        "p3_details": p3_details,
        "p3_offsets": p3_offsets,
        "p3_consistency": p3_consistency,
        "p3_target_matrix": p3_target_matrix,
        "p3_target_comparison": p3_target_comparison,
        "p3_target_summary": p3_target_summary,
    }


def main() -> None:
    assert not RAW_TEST.exists(), "test.xls must remain outside raw source inputs"
    assert QUARANTINE_TEST.exists(), "quarantine test.xls is missing"
    assert historical_handoffs_are_clean(), "historical S2 handoff has unexpected working-tree changes"

    raw_hash_rows, download_info = raw_hash_evidence()
    official_frame = pd.read_excel(OFFICIAL_DOWNLOAD, sheet_name="Sheet1")
    assert list(official_frame.columns) == ["城市名称", "年份", *POPULATION_COLUMNS]
    source_row = population_reconciliation_row(download_info, official_frame)

    old_population_rows = read_git_csv(POP_CANONICAL_REL)
    current_population_rows = read_csv(ROOT / POP_CANONICAL_REL)
    old_profile = population_profile(old_population_rows)
    current_profile = population_profile(current_population_rows)
    city40_current = next(
        row for row in current_population_rows
        if row["城市名称"] == "city40" and as_int(row["年份"]) == 2001
    )
    assert as_float(city40_current[POPULATION_COLUMNS[0]]) == 284
    assert not present(city40_current[POPULATION_COLUMNS[1]])
    assert current_profile["duplicate_key_rows"] == 0
    assert current_profile["canonical_row_count"] == 718

    old_contract = next(row for row in read_git_csv(CONTRACT_REL) if row["dataset_id"] == "population")
    current_contract = next(row for row in read_csv(ROOT / CONTRACT_REL) if row["dataset_id"] == "population")
    old_quality = read_git_csv(QUALITY_REL)
    current_quality = read_csv(ROOT / QUALITY_REL)
    current_quality_population = quality_population_row(current_quality, "after")
    assert "resident missing=25" in current_quality_population["affected_keys_or_coverage"]
    assert "registered missing=11" in current_quality_population["affected_keys_or_coverage"]

    current = recompute_current_facts()
    old_p1_candidates = read_git_csv("reports/S2_HANDOFFS/S2-Plan01/candidate_origin_summary.csv")
    old_p1_summary = read_git_json("reports/S2_HANDOFFS/S2-Plan01/matrix_summary.json")
    old_p2_summary = read_git_json("reports/S2_HANDOFFS/S2-Plan02/geometry_summary.json")
    old_p3_summary = read_git_json("reports/S2_HANDOFFS/S2-Plan03/plan03_summary.json")
    old_p3_target_summary = read_git_csv("reports/S2_HANDOFFS/S2-Plan03/target_hypothesis_summary.csv")
    old_p3_consistency = read_git_csv("reports/S2_HANDOFFS/S2-Plan03/age_population_consistency_summary.csv")
    old_p3_offsets = read_git_csv("reports/S2_HANDOFFS/S2-Plan03/age_population_consistency_offset.csv")

    delta: list[dict[str, Any]] = []
    for field, label, level, implication in (
        ("canonical_row_count", "population canonical row count", "S1", "Refreshes the population canonical source; all downstream population-derived counts must use the new row."),
        ("常住人口（万人）_nonempty_count", "resident non-empty observation count", "S1/S2", "Adds one valid resident historical label; target label semantics do not become official by this change."),
        ("户籍人口（万人）_nonempty_count", "registered non-empty observation count", "S1/S2", "Registered target history is unchanged; its sensitivity role remains required."),
        ("常住人口（万人）_missing_count", "resident canonical missing count", "S1", "Source missingness remains explicit; no value is filled."),
        ("户籍人口（万人）_missing_count", "registered canonical missing count", "S1", "The new row has registered missingness; retain it as missing."),
        ("canonical_missing_cell_count", "population canonical missing-cell count", "S1", "The contract and quality registry must report current missing cells."),
        ("city40_2001_resident", "city40-2001 resident value", "S1", "Confirms the official source fact was absorbed by canonicalization."),
        ("city40_2001_registered", "city40-2001 registered value", "S1", "Confirms the official source leaves registered population missing."),
        ("常住人口（万人）_2001_label_count", "resident 2001 label count", "S2", "Plan01 early target-label coverage increases by one city but remains below the 90% screen."),
        ("户籍人口（万人）_2001_label_count", "registered 2001 label count", "S2", "Registered target-label coverage is unchanged."),
        ("unique_key_count", "population canonical unique-key count", "S1", "The city-year key remains unique after the refresh."),
    ):
        add_delta(delta, f"population.{field}", "population", old_profile[field], current_profile[field], level, implication)
    for field, label in (("year_min", "population year minimum"), ("year_max", "population year maximum")):
        add_delta(delta, f"population.{field}", "population", old_profile[field], current_profile[field], "boundary", "The observed source year domain remains unchanged.")

    add_delta(delta, "s1.population_contract_row_count", "S1 contract", old_contract["row_count"], current_contract["row_count"], "S1", "Refreshes the machine-readable population source contract.")
    add_delta(delta, "s1.population_contract_missing_cell_count", "S1 contract", old_contract["missing_cell_count"], current_contract["missing_cell_count"], "S1", "Contract now reflects the additional registered missing cell.")
    add_delta(delta, "s1.population_contract_raw_missing_cell_count", "S1 contract", old_contract["raw_missing_cell_count"], current_contract["raw_missing_cell_count"], "S1", "Raw missing-cell count reflects the official row before sentinel normalization.")
    old_quality_row = quality_population_row(old_quality, "before")
    add_delta(delta, "s1.QI-004.affected_keys_or_coverage", "S1 quality registry", old_quality_row["affected_keys_or_coverage"], current_quality_population["affected_keys_or_coverage"], "S1", "QI-004 now matches the refreshed canonical missingness while preserving the coverage-gap decision.")

    old_p1_population_inventory = next(row for row in read_git_csv("reports/S2_HANDOFFS/S2-Plan01/source_temporal_inventory.csv") if row["source_id"] == "population")
    current_p1_population_inventory = next(row for row in current["p1_inventory"] if row["source_id"] == "population")
    for field in ("canonical_row_count", "value_row_count", "missing_value_cells", "row_year_min", "row_year_max"):
        add_delta(delta, f"plan01.population_inventory.{field}", "S2-Plan01", old_p1_population_inventory[field], current_p1_population_inventory[field], "diagnostic", "Plan01 inventory is refreshed in delta evidence; historical Plan01 handoff remains unchanged.")
    old_p1_target_rows = read_git_csv("reports/S2_HANDOFFS/S2-Plan01/candidate_origin_summary.csv")
    for year in (2001, 2004, 2010):
        old_row = next(row for row in old_p1_target_rows if int(row["pseudo_target_year"]) == year and row["policy_id"] == "P1_ASOF_T_MINUS_1")
        new_row = next(row for row in current["p1_candidates"] if int(row["pseudo_target_year"]) == year and row["policy_id"] == "P1_ASOF_T_MINUS_1")
        add_delta(delta, f"plan01.resident_{year}_label_city_count", "S2-Plan01", old_row["target_label_available_city_count"], new_row["target_label_available_city_count"], "diagnostic", "Early target-label coverage is recomputed without changing the structural screen." if year == 2001 else "Target-label coverage remains comparable to the historical matrix.")
        add_delta(delta, f"plan01.resident_{year}_history_ready_city_count", "S2-Plan01", old_row["cities_with_min_prior_target_history"], new_row["cities_with_min_prior_target_history"], "diagnostic", "Prior target-history count is recomputed from the refreshed canonical population.")
    for policy_id in ("P1_ASOF_T_MINUS_1", "P2_ASOF_T_MINUS_2"):
        old_set = ";".join(str(row["pseudo_target_year"]) for row in old_p1_target_rows if row["policy_id"] == policy_id and row["classification"] == "candidate")
        new_set = ";".join(str(row["pseudo_target_year"]) for row in current["p1_candidates"] if row["policy_id"] == policy_id and row["classification"] == "candidate")
        add_delta(delta, f"plan01.{policy_id}.candidate_years", "S2-Plan01", old_set, new_set, "candidate geometry", "Candidate-origin set is explicitly compared; no historical handoff rewrite.")
    old_p1_by_key = {(int(row["pseudo_target_year"]), row["policy_id"]): row for row in old_p1_candidates}
    new_p1_by_key = {(int(row["pseudo_target_year"]), row["policy_id"]): row for row in current["p1_candidates"]}
    p1_changed_rows = sum(old_p1_by_key[key]["classification"] != new_p1_by_key[key]["classification"] for key in old_p1_by_key)
    add_delta(delta, "plan01.candidate_classification_changed_rows", "S2-Plan01", 0, p1_changed_rows, "candidate geometry", "No Plan01 candidate classification changed if this remains zero.")

    old_p2_bucket_key = {
        "candidate": "candidate_years_by_scenario",
        "conditional": "conditional_years_by_scenario",
        "exclude": "excluded_years_by_scenario",
    }
    for scenario_id in ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS"):
        for bucket in ("candidate", "conditional", "exclude"):
            old_value = ";".join(str(year) for year in old_p2_summary[old_p2_bucket_key[bucket]][scenario_id])
            new_value = bucket_years(current["p2_feasibility"], scenario_id, bucket)
            add_delta(delta, f"plan02.resident.{scenario_id}.{bucket}_years", "S2-Plan02", old_value, new_value, "candidate geometry", "Source-specific resident-origin geometry is compared against the historical Plan02 artifact.")
    old_p2_2023 = old_p2_summary["target_summaries"]["2023"]
    new_p2_2023 = current["p2_target_summaries"][2023]
    for field in ("target_label_available_city_count", "history_ready_city_count", "history_min_labels_per_city", "history_max_labels_per_city"):
        add_delta(delta, f"plan02.resident_2023.{field}", "S2-Plan02", old_p2_2023[field], new_p2_2023[field], "diagnostic", "The 2023 resident target horizon is recomputed without changing the two-year horizon rule.")
    population_geometry = next(row for row in current["p2_geometry_rows"] if row["source_id"] == "population")
    current_horizon = 2023 - int(population_geometry["latest_usable_year"])
    add_delta(delta, "plan02.resident_target_horizon_years", "S2-Plan02", 2, current_horizon, "contract boundary", "The resident 2023 target horizon remains two years because the latest observed target year remains 2021.")

    old_p3_evidence = read_git_csv("reports/S2_HANDOFFS/S2-Plan03/target_evidence_matrix.csv")
    old_p3_bucket_key = {
        "candidate": "candidate_years",
        "conditional": "conditional_years",
        "exclude": "excluded_years",
    }
    for target_id in ("resident", "registered"):
        old_row = find_row(old_p3_evidence, target_id=target_id)
        geometry = current["p3_geometries"][target_id]
        consistency = find_row(current["p3_consistency"], target_id=target_id)
        add_delta(delta, f"plan03.{target_id}.observed_nonempty_row_count", "S2-Plan03", old_row["observed_nonempty_row_count"], geometry["observed_row_count"], "target evidence", "Target evidence count is refreshed from the new canonical population.")
        add_delta(delta, f"plan03.{target_id}.same_year_age_overlap_n", "S2-Plan03", old_row["same_year_age_overlap_n"], consistency["same_year_overlap_n"], "target evidence", "Age benchmark semantics remain unchanged; only overlapping population history is recomputed.")
        for scenario_id in ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS"):
            for bucket in ("candidate", "conditional", "exclude"):
                old_value = next(row for row in old_p3_target_summary if row["target_id"] == target_id and row["scenario_id"] == scenario_id)[old_p3_bucket_key[bucket]]
                new_value = plan03_bucket_years(current["p3_target_matrix"], target_id, scenario_id, bucket)
                add_delta(delta, f"plan03.{target_id}.{scenario_id}.{bucket}_years", "S2-Plan03", old_value, new_value, "candidate geometry", "Target-origin comparison is refreshed while preserving resident/registered sensitivity.")
    for offset in (1,):
        for field in ("overlap_n", "mean_absolute_diff", "median_absolute_diff", "mean_relative_absolute_diff", "mean_signed_diff"):
            old_row = find_row(old_p3_offsets, target_id="resident", population_year_minus_age_year=str(offset))
            new_row = find_row(current["p3_offsets"], target_id="resident", population_year_minus_age_year=offset)
            add_delta(delta, f"plan03.resident_age_offset_{offset}.{field}", "S2-Plan03", old_row[field], new_row[field], "diagnostic", "The new 2001 resident label can affect the offset diagnostic only; same-year benchmark semantics remain primary.")
    old_consistency_resident = find_row(old_p3_consistency, target_id="resident")
    new_consistency_resident = find_row(current["p3_consistency"], target_id="resident")
    add_delta(delta, "plan03.resident_best_temporal_offset_diagnostic", "S2-Plan03", old_consistency_resident["best_temporal_offset_diagnostic"], new_consistency_resident["best_temporal_offset_diagnostic"], "diagnostic", "No temporal lag is promoted to an official target definition.")
    for fact_id, old_value, new_value, implication in (
        ("plan03.target_contract_status", old_p3_summary["target_contract_status"], "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET", "Target classification remains conditional, with resident primary and registered sensitivity."),
        ("plan03.primary_metric", old_p3_summary["primary_metric"], "MSE", "Metric contract is unaffected by a historical source-row addition."),
        ("plan03.auxiliary_metric", old_p3_summary["auxiliary_metric"], "RMSE", "Metric contract is unaffected by a historical source-row addition."),
        ("plan03.cross_origin_aggregation", old_p3_summary["cross_origin_aggregation"], "UNRESOLVED_DEFERRED", "Cross-origin aggregation remains deliberately unresolved."),
        ("plan03.age_complete_rows_used", old_p3_summary["age_complete_rows_used"], len(current["p3_details"]), "Age partial/invalidated rows are still excluded, not filled."),
        ("plan03.employee_provenance_rows_preserved", old_p3_summary["employee_provenance_rows_preserved"], 91, "Employee provenance is outside the population refresh and remains unchanged."),
    ):
        add_delta(delta, fact_id, "S2-Plan03", old_value, new_value, "contract boundary", implication)

    write_csv(OUTPUT_DIR / "raw_source_hash_evidence.csv", raw_hash_rows)
    write_csv(OUTPUT_DIR / "population_source_reconciliation.csv", [source_row])
    write_csv(OUTPUT_DIR / "s1_population_contract_evidence.csv", [selected_contract_row(old_contract, "before"), selected_contract_row(current_contract, "after")])
    write_csv(OUTPUT_DIR / "s1_population_quality_evidence.csv", [old_quality_row, current_quality_population])
    write_csv(OUTPUT_DIR / "population_before_after_delta.csv", [
        {
            "fact_id": row["fact_id"].replace("population.", ""),
            "old_value": row["old_value"],
            "new_value": row["new_value"],
            "changed": row["changed"],
            "impact_level": row["impact_level"],
            "downstream_implication": row["downstream_implication"],
        }
        for row in delta if row["layer"] == "population"
    ])
    write_csv(OUTPUT_DIR / "s2_population_impact_delta.csv", [row for row in delta if row["layer"].startswith("S2-") or row["layer"] == "S2-Plan03"])
    write_csv(OUTPUT_DIR / "s2_recomputed_facts.csv", [
        {"fact_id": "plan01_candidate_years_P1", "value": ";".join(str(row["pseudo_target_year"]) for row in current["p1_candidates"] if row["policy_id"] == "P1_ASOF_T_MINUS_1" and row["classification"] == "candidate")},
        {"fact_id": "plan01_candidate_years_P2", "value": ";".join(str(row["pseudo_target_year"]) for row in current["p1_candidates"] if row["policy_id"] == "P2_ASOF_T_MINUS_2" and row["classification"] == "candidate")},
        {"fact_id": "plan02_resident_2023_history_ready_city_count", "value": current["p2_target_summaries"][2023]["history_ready_city_count"]},
        {"fact_id": "plan02_R1_resident_candidate_years", "value": bucket_years(current["p2_feasibility"], "R1_REFERENCE_SNAPSHOT_LAG", "candidate")},
        {"fact_id": "plan02_R2_resident_candidate_years", "value": bucket_years(current["p2_feasibility"], "R2_CONSERVATIVE_PUBLICATION_STRESS", "candidate")},
        {"fact_id": "plan03_R1_resident_candidate_years", "value": plan03_bucket_years(current["p3_target_matrix"], "resident", "R1_REFERENCE_SNAPSHOT_LAG", "candidate")},
        {"fact_id": "plan03_R1_registered_candidate_years", "value": plan03_bucket_years(current["p3_target_matrix"], "registered", "R1_REFERENCE_SNAPSHOT_LAG", "candidate")},
        {"fact_id": "plan03_R2_resident_candidate_years", "value": plan03_bucket_years(current["p3_target_matrix"], "resident", "R2_CONSERVATIVE_PUBLICATION_STRESS", "candidate")},
        {"fact_id": "plan03_R2_registered_candidate_years", "value": plan03_bucket_years(current["p3_target_matrix"], "registered", "R2_CONSERVATIVE_PUBLICATION_STRESS", "candidate")},
        {"fact_id": "plan03_target_contract_status", "value": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET"},
        {"fact_id": "plan03_primary_metric", "value": "MSE"},
        {"fact_id": "plan03_auxiliary_metric", "value": "RMSE"},
        {"fact_id": "plan03_cross_origin_aggregation", "value": "UNRESOLVED_DEFERRED"},
    ])

    expected_current = {
        "population_canonical_rows": current_profile["canonical_row_count"],
        "resident_nonempty_rows": current_profile["常住人口（万人）_nonempty_count"],
        "registered_nonempty_rows": current_profile["户籍人口（万人）_nonempty_count"],
        "resident_2010_history_ready_city_count": history_ready_count(current_population_rows, POPULATION_COLUMNS[0], 2008),
        "resident_2023_horizon_years": 2,
        "plan02_R1_resident_candidate_years": bucket_years(current["p2_feasibility"], "R1_REFERENCE_SNAPSHOT_LAG", "candidate"),
        "plan02_R2_resident_candidate_years": bucket_years(current["p2_feasibility"], "R2_CONSERVATIVE_PUBLICATION_STRESS", "candidate"),
        "plan03_R1_resident_candidate_years": plan03_bucket_years(current["p3_target_matrix"], "resident", "R1_REFERENCE_SNAPSHOT_LAG", "candidate"),
        "plan03_R1_registered_candidate_years": plan03_bucket_years(current["p3_target_matrix"], "registered", "R1_REFERENCE_SNAPSHOT_LAG", "candidate"),
        "target_contract_status": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
        "primary_metric": "MSE",
        "auxiliary_metric": "RMSE",
        "cross_origin_aggregation": "UNRESOLVED_DEFERRED",
        "test_xls_used": False,
        "model_scoring_run": False,
    }
    assert expected_current["population_canonical_rows"] == 718
    assert expected_current["resident_nonempty_rows"] == 693
    assert expected_current["registered_nonempty_rows"] == 707
    assert expected_current["resident_2010_history_ready_city_count"] == 38
    assert expected_current["plan02_R1_resident_candidate_years"] == ";".join(str(year) for year in range(2010, 2022))
    assert expected_current["plan02_R2_resident_candidate_years"] == ";".join(str(year) for year in range(2010, 2022))
    assert expected_current["plan03_R1_resident_candidate_years"] == ";".join(str(year) for year in range(2010, 2022))
    assert expected_current["plan03_R1_registered_candidate_years"] == ";".join(str(year) for year in range(2011, 2022))

    write_json(OUTPUT_DIR / "reproducibility_evidence.json", {
        "plan_id": "S2-Plan04",
        "starting_commit": STARTING_COMMIT,
        "analysis_head": current_git_head(),
        "official_source_sha256": download_info["official_download_sha256"],
        "official_source_sha256_matches_expected": download_info["official_download_sha256"] == EXPECTED_OFFICIAL_SHA256,
        "only_population_raw_xlsx_changed": download_info["changed_raw_xlsx_paths"] == [POP_RAW_REL],
        "historical_plan01_plan02_plan03_handoffs_clean": historical_handoffs_are_clean(),
        "canonical_population_key_unique": current_profile["duplicate_key_rows"] == 0,
        "city40_2001_verified": True,
        "s1_population_contract_refreshed": True,
        "s1_population_quality_registry_refreshed": True,
        "employee_provenance_rows": 91,
        "age_benchmark_domain_preserved": True,
        "test_xls_used": False,
        "formal_model_scoring_run": False,
        "s3_entered": False,
        "delta_rows": len(delta),
    })
    write_json(OUTPUT_DIR / "plan04_summary.json", {
        "plan_id": "S2-Plan04",
        "starting_commit": STARTING_COMMIT,
        "official_source_sha256": EXPECTED_OFFICIAL_SHA256,
        "population_before": old_profile,
        "population_after": current_profile,
        "resident_2010_history_ready_city_count_before": history_ready_count(old_population_rows, POPULATION_COLUMNS[0], 2008),
        "resident_2010_history_ready_city_count_after": expected_current["resident_2010_history_ready_city_count"],
        "resident_target_horizon_years": 2,
        "registered_target_horizon_years": 2,
        "plan02_resident_candidate_years": {
            scenario: bucket_years(current["p2_feasibility"], scenario, "candidate")
            for scenario in ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS")
        },
        "plan03_target_candidate_years": {
            f"{target}/{scenario}": plan03_bucket_years(current["p3_target_matrix"], target, scenario, "candidate")
            for target in ("resident", "registered")
            for scenario in ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS")
        },
        "target_contract_status": "PRIMARY_PROXY_WITH_SENSITIVITY_TARGET",
        "primary_metric": "MSE",
        "auxiliary_metric": "RMSE",
        "cross_origin_aggregation": "UNRESOLVED_DEFERRED",
        "s2_core_conclusions_changed": any(row["changed"] == "yes" and row["impact_level"] == "contract boundary" for row in delta),
        "formal_model_scoring_run": False,
        "test_xls_used": False,
    })
    print("S2_PLAN04_OFFICIAL_POPULATION_RECONCILIATION_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"official_source_sha256={EXPECTED_OFFICIAL_SHA256}")
    print(f"population_rows={old_profile['canonical_row_count']}->{current_profile['canonical_row_count']}")
    print(f"resident_nonempty={old_profile['常住人口（万人）_nonempty_count']}->{current_profile['常住人口（万人）_nonempty_count']}")
    print(f"registered_nonempty={old_profile['户籍人口（万人）_nonempty_count']}->{current_profile['户籍人口（万人）_nonempty_count']}")
    print(f"resident_2010_history_ready={history_ready_count(old_population_rows, POPULATION_COLUMNS[0], 2008)}->{expected_current['resident_2010_history_ready_city_count']}")
    print(f"plan02_R1_resident_candidates={expected_current['plan02_R1_resident_candidate_years']}")
    print(f"plan03_R1_registered_candidates={expected_current['plan03_R1_registered_candidate_years']}")
    print("target_contract_status=PRIMARY_PROXY_WITH_SENSITIVITY_TARGET")
    print("primary_metric=MSE")
    print("auxiliary_metric=RMSE")
    print("cross_origin_aggregation=UNRESOLVED_DEFERRED")
    print("formal_model_scoring_run=False")
    print("test_xls_used=False")


if __name__ == "__main__":
    main()
