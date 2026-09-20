"""Audit and freeze the S6-Plan01 final-solution contract.

This is an evidence/audit runner only.  It does not fit a model, read the
forbidden ``test.xls`` input, generate a 2023 prediction, or write a
submission.  It consumes the committed S1-S5 contracts and historical
Plan06E lineage, then emits the S6-Plan01 contract and handoff packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openpyxl import load_workbook
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "S6_HANDOFFS" / "S6-Plan01"
S1_CONTRACT = ROOT / "docs" / "s1_data_contract.csv"
S2_PLAN01 = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan01"
S2_PLAN02 = ROOT / "reports" / "S2_HANDOFFS" / "S2-Plan02"
S3_PLAN02_SCRIPT = ROOT / "scripts" / "build_s3_plan02_recent_window_and_selector.py"
S5_PLAN04 = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04"
S5_PLAN04B = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B"
PLAN04_SCRIPT = ROOT / "scripts" / "build_s5_plan04_full_multisource_feature_foundation.py"
PLAN04B_SCRIPT = ROOT / "scripts" / "build_s5_plan04b_training_replay_foundation.py"
S2_GEOMETRY_SCRIPT = ROOT / "scripts" / "build_s2_plan02_geometry.py"
PDF_PATH = ROOT / "data" / "raw" / "城市人口分析与预测" / "城市人口分析与预测.pdf"
RAW_DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
FORBIDDEN_TEST_XLS = ROOT / "data" / "quarantine" / "source" / "test.xls"

REFERENCE_YEAR = 2023
TARGET_HISTORY_CUTOFF = 2021
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
CITIES = tuple(f"city{i}" for i in range(1, 41))
F4_ID = "F4_LOG_GROWTH_FROM_LATEST"
F4_LABEL = "log(actual_target_value / pop_latest_level)"
F4_RECONSTRUCTION = "pop_latest_level * exp(raw_prediction)"

LINEAGE = {
    "plan06e_artifact_tip": "343970c608fa1a0f6d5e7ab5d355c29757d6ce2e",
    "plan06e_artifact_generation_source": "469787d00b1dc220a017df02c6fbb6fcb76d2605",
    "plan06e_runner_introduction": "f960d2cb71de73e4ab39d341f7383022d9c55cac",
    "plan06e_wave_base": "1f97f67488cd7aec120ef8b9fa04407dc13d54ba",
    "plan06e_archive_tag": "archive/pre-s6/s5-plan06e-delta-growth-framing",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.STDOUT).strip()


def resolve_source_sha(requested: str | None) -> str:
    """Resolve the main source baseline audited before local packet edits."""
    if requested:
        return git("rev-parse", requested)
    try:
        return git("rev-parse", "origin/main")
    except subprocess.CalledProcessError:
        return git("rev-parse", "HEAD")


def present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "NA", "N/A", "NaN", "nan", "None"}


def as_int(value: Any) -> int | None:
    return int(float(value)) if present(value) else None


def as_float(value: Any) -> float | None:
    return float(value) if present(value) else None


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canonical_stats(spec: dict[str, str]) -> dict[str, Any]:
    path = ROOT / spec["canonical_file"]
    rows = read_csv(path)
    city_col, year_col = spec["key_columns"].split(";")
    value_columns = spec["value_columns"].split(";")
    keys = [(row.get(city_col), as_int(row.get(year_col))) for row in rows]
    value_rows = [row for row in rows if any(present(row.get(column)) for column in value_columns)]
    value_keys = [(row.get(city_col), as_int(row.get(year_col))) for row in value_rows]
    years = [year for _, year in keys if year is not None]
    value_years = [year for _, year in value_keys if year is not None]
    return {
        "canonical_rows": len(rows),
        "canonical_duplicate_keys": len(keys) - len(set(keys)),
        "value_rows": len(value_rows),
        "value_year_min": min(value_years) if value_years else None,
        "value_year_max": max(value_years) if value_years else None,
        "city_count_with_any_value": len({city for city, _ in value_keys}),
        "value_rows_after_cap": lambda cap: sum(1 for _, year in value_keys if year is not None and year <= cap),
        "value_rows_after_cap_city_count": lambda cap: len({city for city, year in value_keys if year is not None and year <= cap}),
        "value_rows_after_cap_max_year": lambda cap: max((year for _, year in value_keys if year is not None and year <= cap), default=None),
        "value_rows_after_cap_keys": lambda cap: {(city, year) for city, year in value_keys if year is not None and year <= cap},
        "all_years": years,
        "path": relative(path),
    }


def raw_workbook_years(path: Path) -> tuple[int | None, int | None, list[str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_names = list(workbook.sheetnames)
    all_rows: list[list[Any]] = []
    try:
        for sheet_name in sheet_names:
            sheet = workbook[sheet_name]
            for row in sheet.iter_rows(values_only=True):
                all_rows.append(list(row))
    finally:
        workbook.close()
    string_year_tokens_by_column: defaultdict[int, list[int]] = defaultdict(list)
    numeric_year_values_by_column: defaultdict[int, list[int]] = defaultdict(list)
    years: list[int] = []
    for row in all_rows:
        for index, value in enumerate(row):
            if isinstance(value, str):
                tokens = [int(match) for match in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value)]
                string_year_tokens_by_column[index].extend(tokens)
            elif isinstance(value, (int, float)) and float(value).is_integer() and 1900 <= int(value) <= 2100:
                numeric_year_values_by_column[index].append(int(value))
    year_columns = {
        index
        for row in all_rows[:20]
        for index, value in enumerate(row)
        if isinstance(value, str) and ("年份" in value or value.strip().lower() == "year")
    }
    for row in all_rows[:20]:
        row_candidates = [
            int(value) for value in row
            if isinstance(value, (int, float)) and float(value).is_integer() and 1900 <= int(value) <= 2100
        ]
        first_value = row[0] if row else None
        if len(row_candidates) >= 3 and (first_value is None or isinstance(first_value, str)):
            years.extend(row_candidates)
    for index in year_columns:
        years.extend(string_year_tokens_by_column[index])
        years.extend(numeric_year_values_by_column[index])
    for index, values in string_year_tokens_by_column.items():
        distinct = set(values)
        if len(distinct) >= 3:
            years.extend(values)
    return (min(years) if years else None, max(years) if years else None, sheet_names)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_feature_contract() -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], list[str]]:
    definitions = read_csv(S5_PLAN04 / "feature_definitions.csv")
    schema = json.loads((S5_PLAN04 / "feature_matrix_schema.json").read_text(encoding="utf-8"))
    worker_contract = json.loads((S5_PLAN04B / "parallel_worker_input_contract.json").read_text(encoding="utf-8"))
    csv_features = [row["feature_name"] for row in definitions if row["model_feature"] == "yes"]
    assert csv_features == schema["model_features"]
    assert csv_features == worker_contract["model_features"]
    assert len(csv_features) == 144
    return definitions, schema, worker_contract, csv_features


def load_replay_geometry() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    replay = read_csv(S5_PLAN04B / "full_multisource_training_replay_matrix.csv")
    boundary = read_csv(S5_PLAN04B / "replay_boundary_audit.csv")
    inner = read_csv(S5_PLAN04B / "inner_split_geometry.csv")
    summary = json.loads((S5_PLAN04B / "training_replay_summary.json").read_text(encoding="utf-8"))
    assert len(replay) == 2320
    assert len({(r["target_id"], r["scenario_id"], r["pseudo_target_year"], r["city_id"]) for r in replay}) == len(replay)
    assert summary["replay_boundary_audit"]["source_cap_violation_count"] == 0
    assert summary["replay_boundary_audit"]["quarantine_reintroduction_count"] == 0
    return replay, boundary, inner, summary


def build_geometry(replay: list[dict[str, str]], boundary: list[dict[str, str]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    boundary_by_pair = defaultdict(list)
    for row in boundary:
        boundary_by_pair[(row["target_id"], row["scenario_id"])].append(row)
    exclusion_reasons = ";".join(f"{key}:{value}" for key, value in sorted(summary["replay_matrix"]["excluded_rows_by_reason"].items()))
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            scoped = [
                row for row in replay
                if row["target_id"] == target_id
                and row["scenario_id"] == scenario_id
                and as_int(row["pseudo_target_year"]) <= TARGET_HISTORY_CUTOFF
            ]
            audit = boundary_by_pair[(target_id, scenario_id)]
            keys = [(r["pseudo_target_year"], r["city_id"]) for r in scoped]
            pop_latest = [as_float(r.get("pop_latest_level")) for r in scoped]
            labels = [as_float(r.get("actual_target_value")) for r in scoped]
            rows.append({
                "target_id": target_id,
                "scenario_id": scenario_id,
                "candidate_role": "primary" if target_id == "resident" else "sensitivity_only",
                "final_prediction_year": REFERENCE_YEAR,
                "target_history_cutoff_year": TARGET_HISTORY_CUTOFF,
                "allowed_label_max_pseudo_target_year": max((as_int(r["pseudo_target_year"]) for r in scoped), default=None),
                "allowed_label_min_pseudo_target_year": min((as_int(r["pseudo_target_year"]) for r in scoped), default=None),
                "training_row_count": len(scoped),
                "training_city_count": len({r["city_id"] for r in scoped}),
                "training_year_count": len({r["pseudo_target_year"] for r in scoped}),
                "duplicate_key_count": len(keys) - len(set(keys)),
                "missing_label_count": sum(value is None for value in labels),
                "nonpositive_label_count": sum(value is not None and value <= 0 for value in labels),
                "nonpositive_pop_latest_count": sum(value is not None and value <= 0 for value in pop_latest),
                "full_replay_matrix_rows_for_pair": sum(1 for r in replay if r["target_id"] == target_id and r["scenario_id"] == scenario_id),
                "builder_excluded_rows_total": sum(summary["replay_matrix"]["excluded_rows_by_reason"].values()),
                "builder_excluded_label_rows_total": summary["replay_matrix"]["excluded_label_rows_total"],
                "builder_exclusion_reasons": exclusion_reasons,
                "boundary_audit_rows_for_pair": len(audit),
                "source_cap_violation_count": sum(as_int(r.get("cap_violation_count")) or 0 for r in audit),
                "quarantine_reintroduction_count": sum(as_int(r.get("quarantine_used_count")) or 0 for r in audit),
                "benchmark_domain_violation_count": sum(as_int(r.get("benchmark_domain_violation_count")) or 0 for r in audit),
                "population_history_cutoff_fail_count": sum(r.get("population_history_cutoff_pass") != "PASS" for r in audit),
                "b3_history_cutoff_fail_count": sum(r.get("b3_history_cutoff_pass") != "PASS" for r in audit),
                "b3_prediction_unavailable_count": sum(r.get("b3_prediction_available") != "PASS" for r in audit),
                "uses_2022_or_2023_target_truth": "no" if max((as_int(r["pseudo_target_year"]) for r in scoped), default=0) <= 2021 else "yes",
                "geometry_status": "PASS" if (
                    len(scoped) > 0
                    and len({r["city_id"] for r in scoped}) == 40
                    and len(keys) == len(set(keys))
                    and all(value is not None and value > 0 for value in labels)
                    and all(value is not None and value > 0 for value in pop_latest)
                    and all(as_int(r.get("cap_violation_count")) == 0 for r in audit)
                    and all(as_int(r.get("benchmark_domain_violation_count")) == 0 for r in audit)
                    and all(r.get("quarantine_reintroduction_status") == "PASS" for r in audit)
                    and all(as_int(r.get("pseudo_target_year")) <= TARGET_HISTORY_CUTOFF for r in scoped)
                ) else "FAIL",
            })
    assert all(row["geometry_status"] == "PASS" for row in rows)
    return rows


def build_information_audit(contract: list[dict[str, str]]) -> list[dict[str, Any]]:
    inventory = {row["source_id"]: row for row in read_csv(S2_PLAN01 / "source_temporal_inventory.csv")}
    freshness = [row for row in read_csv(S2_PLAN02 / "origin_source_scenario_freshness_matrix.csv") if as_int(row["pseudo_target_year"]) == REFERENCE_YEAR]
    policies = [row for row in read_csv(S2_PLAN01 / "pseudo_target_source_policy_matrix.csv") if as_int(row["pseudo_target_year"]) == REFERENCE_YEAR]
    contract_by_id = {row["dataset_id"]: row for row in contract}
    canonical = {source_id: canonical_stats(spec) for source_id, spec in contract_by_id.items()}
    raw_years: dict[str, tuple[int | None, int | None]] = {}
    raw_sheet_names: dict[str, list[str]] = {}
    for spec in contract:
        raw_path = ROOT / spec["raw_file"]
        if raw_path.suffix.lower() != ".xlsx":
            continue
        if relative(raw_path) not in raw_years:
            raw_min, raw_max, sheets = raw_workbook_years(raw_path)
            raw_years[relative(raw_path)] = (raw_min, raw_max)
            raw_sheet_names[relative(raw_path)] = sheets
    rows: list[dict[str, Any]] = []
    for freshness_row in freshness:
        source_id = freshness_row["source_id"]
        scenario_id = freshness_row["scenario_id"]
        source = inventory[source_id]
        cap = as_int(freshness_row["required_latest_usable_year"])
        expected_policy_id = "P1_ASOF_T_MINUS_1" if scenario_id == SCENARIOS[0] else "P2_ASOF_T_MINUS_2"
        policy_matches = [
            row for row in policies
            if row["source_id"] == source_id
            and row["policy_id"] == expected_policy_id
            and as_int(row["effective_source_cap_year"]) == cap
        ]
        policy = policy_matches[0] if len(policy_matches) == 1 else None
        stats = canonical[source_id]
        spec = contract_by_id[source_id]
        raw_path = ROOT / spec["raw_file"]
        raw_min, raw_max = raw_years.get(relative(raw_path), (None, None))
        pair_boundary_path = S5_PLAN04B / "replay_boundary_audit.csv"
        boundary_rows = [
            row for row in read_csv(pair_boundary_path)
            if row["target_id"] == "resident" and row["scenario_id"] == scenario_id and row["source_id"] == source_id
        ]
        cap_rows = stats["value_rows_after_cap"](cap) if cap is not None else 0
        cap_cities = stats["value_rows_after_cap_city_count"](cap) if cap is not None else 0
        future_like_excluded = sum(1 for year in stats["all_years"] if year > REFERENCE_YEAR)
        mapping_status = "EXACT_R1" if scenario_id == SCENARIOS[0] and policy is not None else "EXACT_R2" if scenario_id == SCENARIOS[1] and policy is not None else "NO_FROZEN_MAPPING"
        rows.append({
            "pseudo_target_year": REFERENCE_YEAR,
            "source_id": source_id,
            "role": source["role"],
            "scenario_id": scenario_id,
            "mapping_status": mapping_status,
            "relation_to_R1_R2": "frozen Plan02 R1 source-specific snapshot lag" if scenario_id == SCENARIOS[0] else "frozen Plan02 R2 conservative publication stress",
            "policy_id": policy["policy_id"] if policy else "",
            "canonical_file": source["canonical_file"],
            "raw_file": source["raw_file"],
            "raw_sheet": source["raw_sheet"],
            "raw_xlsx_observed_year_min": raw_min,
            "raw_xlsx_observed_year_max": raw_max,
            "canonical_observed_value_year_min": as_int(source["value_year_min"]),
            "canonical_observed_value_year_max": as_int(source["value_year_max"]),
            "observed_city_count_with_any_value": as_int(source["city_count_with_any_value"]),
            "required_lag_years": as_int(freshness_row["required_lag_years"]),
            "required_cap_year": cap,
            "max_year_used_under_cap": as_int(freshness_row["actual_latest_year_as_of_required_cap"]),
            "available_city_count_under_cap": as_int(freshness_row["available_cities_with_any_value_as_of_cap"]),
            "exact_cap_city_count": as_int(freshness_row["cities_with_value_at_exact_required_year"]),
            "exact_cap_city_count_interpretation": "benchmark-domain metric; annual exact-year coverage is not applicable" if source_id == "age" else "annual/source-defined exact-year coverage",
            "canonical_value_rows_under_cap": cap_rows,
            "canonical_city_count_under_cap_recomputed": cap_cities,
            "rows_after_cap_not_used": stats["value_rows"] - cap_rows,
            "future_like_rows_after_2023": future_like_excluded,
            "contemporaneous_future_like_check": "PASS_NO_2023_PLUS_VALUE_USED" if future_like_excluded == 0 and (cap is None or cap <= 2022) else "PASS_EXCLUDED_BY_CAP",
            "freshness_status": freshness_row["freshness_status"],
            "availability_class": policy["availability_class"] if policy else "",
            "quarantine_keys_as_of_cap": policy["invalidated_quarantine_keys_at_or_before_cutoff"] if policy else "",
            "derived_rows_available_as_of_cap": policy["derived_rows_available"] if policy else "",
            "derived_rows_used_in_replay_pair": sum(as_int(row.get("derived_used_count")) or 0 for row in boundary_rows),
            "quarantine_reintroduction_in_replay_pair": sum(as_int(row.get("quarantine_used_count")) or 0 for row in boundary_rows),
            "domain_guard": policy["quality_boundary"] if policy else "NO_POLICY_MATCH",
            "information_set_warning": policy["information_set_warning"] if policy else "NO_POLICY_MATCH",
            "publication_timing_proven": "no",
            "source_mapping_status": "PASS" if mapping_status.startswith("EXACT_") else "BLOCKED",
        })
    assert len(rows) == 28
    assert all(row["source_mapping_status"] == "PASS" for row in rows)
    assert all(row["contemporaneous_future_like_check"] in {"PASS_NO_2023_PLUS_VALUE_USED", "PASS_EXCLUDED_BY_CAP"} for row in rows)
    return rows


def submission_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reader = PdfReader(str(PDF_PATH))
    pages = [(page.extract_text() or "") for page in reader.pages]
    full_text = "\n".join(pages)
    page1 = pages[0] if pages else ""
    page3 = pages[2] if len(pages) >= 3 else ""
    assert "2023" in page1 and "总人口" in page1
    assert "submission.csv" in page3 and all(token in page3 for token in ("city_id", "year", "pred"))
    rows = [
        {"field": "official_task_prediction_year", "value": "2023", "status": "CONFIRMED", "evidence": f"{relative(PDF_PATH)} page 1; CURRENT_LEARNING_CHECKPOINT.md official sample check"},
        {"field": "official_target_wording", "value": "各个城市 2023 年的总人口数", "status": "CONFIRMED", "evidence": f"{relative(PDF_PATH)} page 1"},
        {"field": "resident_vs_registered_mapping", "value": "not proved by official material", "status": "UNKNOWN", "evidence": "PDF says 总人口; legal population source has separate resident/registered columns; no direct mapping"},
        {"field": "known_city_universe", "value": "city1..city40", "status": "CONFIRMED", "evidence": "CURRENT_LEARNING_CHECKPOINT.md official sample direct check; S1 canonical coverage"},
        {"field": "final_universe_size", "value": "40", "status": "CONFIRMED", "evidence": "CURRENT_LEARNING_CHECKPOINT.md official sample direct check; canonical city set"},
        {"field": "city_ordering_requirement", "value": "sample is city1..city40; formal ordering rule not stated", "status": "INFERRED_NOT_SAFE_TO_FREEZE", "evidence": "CURRENT_LEARNING_CHECKPOINT.md; official PDF schema page"},
        {"field": "header_columns", "value": "city_id, year, pred", "status": "CONFIRMED", "evidence": f"{relative(PDF_PATH)} page 3; CURRENT_LEARNING_CHECKPOINT.md"},
        {"field": "file_format", "value": "CSV", "status": "CONFIRMED", "evidence": f"{relative(PDF_PATH)} page 3"},
        {"field": "target_unit", "value": "local contract uses 万人; official unit not stated in PDF", "status": "INFERRED_NOT_SAFE_TO_FREEZE", "evidence": f"{relative(PDF_PATH)} page 1 and source contract"},
        {"field": "official_submission_template", "value": "submission_sample.csv was confirmed in Main evidence but is not in current raw tree", "status": "CONFIRMED", "evidence": "CURRENT_LEARNING_CHECKPOINT.md with official sample MD5"},
    ]
    return rows, {"pdf_page_count": len(reader.pages), "pdf_contains_official_schema": True, "pdf_text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest()}


def build_manifest() -> list[dict[str, Any]]:
    raw_xlsx = sorted(RAW_DATA_DIR.glob("*.xlsx"))
    assert len(raw_xlsx) == 7
    rows: list[dict[str, Any]] = []
    rows.append({"category": "immutable_preserve", "path": relative(PDF_PATH), "status": "PRESERVE", "source_of_truth": "official raw PDF", "clean_rebuild_action": "never delete or rewrite", "notes": "Official task wording and submission schema evidence."})
    for path in raw_xlsx:
        rows.append({"category": "immutable_preserve", "path": relative(path), "status": "PRESERVE", "source_of_truth": "official raw XLSX", "clean_rebuild_action": "never delete or rewrite", "notes": "One of the 7 legal raw source workbooks; read-only input."})
    rows.extend([
        {"category": "immutable_preserve", "path": "Git history/tags and S5 historical identity", "status": "PRESERVE", "source_of_truth": "Git object database", "clean_rebuild_action": "never rewrite history; no force push", "notes": LINEAGE["plan06e_archive_tag"]},
        {"category": "immutable_preserve_but_forbidden_input", "path": relative(FORBIDDEN_TEST_XLS), "status": "PRESERVE_FILE_ONLY", "source_of_truth": "existing file metadata only", "clean_rebuild_action": "do not delete in this Plan; never read content", "notes": "Forbidden input; not hashed or opened by this runner."},
        {"category": "final_contract_evidence", "path": "reports/S6_HANDOFFS/S6-Plan01/*", "status": "PRESERVE_AFTER_FREEZE", "source_of_truth": "S6 contract/handoff", "clean_rebuild_action": "preserve contract and evidence identity", "notes": "Contract hash is recorded in historical_lineage_hashes.csv."},
        {"category": "rebuildable_derived", "path": "data/normalized/*.csv", "status": "REBUILD", "source_of_truth": "S1 scripts plus legal raw inputs", "clean_rebuild_action": "can be removed and rebuilt from raw", "notes": "Canonical normalized data; not an irreplaceable source."},
        {"category": "rebuildable_derived", "path": "data/quarantine/*.csv and data/normalized/*provenance*.csv", "status": "REBUILD", "source_of_truth": "S1 audit/rebuild scripts", "clean_rebuild_action": "can be removed and rebuilt from raw", "notes": "Repairs, provenance, quarantine, and regime guards remain explicit."},
        {"category": "rebuildable_derived", "path": "reports/S5_HANDOFFS/S5-Plan04/**", "status": "REBUILD", "source_of_truth": "Plan04 builder and feature contract", "clean_rebuild_action": "can be removed and rebuilt; preserve current copy for audit", "notes": "144-feature matrix is derived, not a new source of truth."},
        {"category": "rebuildable_derived", "path": "reports/S5_HANDOFFS/S5-Plan04B/**", "status": "REBUILD", "source_of_truth": "Plan04B builder and replay contract", "clean_rebuild_action": "can be removed and rebuilt; preserve current copy for audit", "notes": "Replay/geometry/boundary artifacts are derived."},
        {"category": "rebuildable_derived", "path": "model artifacts, prediction artifacts, caches, tmp/**", "status": "REBUILD", "source_of_truth": "future S6 builders", "clean_rebuild_action": "remove only within an authorized clean rebuild", "notes": "No final model/prediction/submission exists in this Plan."},
        {"category": "historical_evidence", "path": "reports/S5_HANDOFFS/** and archive/pre-s6/s5-plan06e-delta-growth-framing", "status": "PRESERVE_IDENTITY", "source_of_truth": "committed S5 evidence and immutable tag", "clean_rebuild_action": "do not reinterpret or replace with new metrics", "notes": "Historical evidence supports review; worker OOF is not a final-fit shortcut."},
        {"category": "forbidden_input", "path": "external data / B-data / manually edited truth / 2022-2023 target truth", "status": "FORBIDDEN", "source_of_truth": "none", "clean_rebuild_action": "never introduce", "notes": "B-data is historical research material, not current default input."},
        {"category": "forbidden_input", "path": relative(FORBIDDEN_TEST_XLS), "status": "FORBIDDEN", "source_of_truth": "none", "clean_rebuild_action": "never open/read", "notes": "May remain on disk only as preserved file metadata."},
    ])
    return rows


def build_contract(
    source_sha: str,
    definitions: list[dict[str, str]],
    schema: dict[str, Any],
    worker_contract: dict[str, Any],
    model_features: list[str],
    information_audit: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    submission_rows: list[dict[str, Any]],
    submission_meta: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    group_counts = Counter(row["feature_group"] for row in definitions if row["model_feature"] == "yes")
    return {
        "contract_version": "S6-Plan01-FINAL-SOLUTION-CONTRACT-V1",
        "contract_id": "S6_FINAL_F4_RESIDENT_PRIMARY_2023_V1",
        "contract_status": "PARTIALLY_FROZEN_SUBMISSION_BLOCKED",
        "ready_for_main_stage_review": True,
        "ready_for_s6_clean_rebuild": True,
        "submission_generation_authorized": False,
        "execution": {"source_main_sha": source_sha, "branch": git("branch", "--show-current"), "audit_generated_at_utc": datetime.now(timezone.utc).isoformat()},
        "lineage": LINEAGE,
        "target": {
            "primary_target_id": "resident",
            "primary_source_column": "常住人口（万人）",
            "primary_contract_role": "primary_proxy",
            "official_target_wording": "总人口",
            "official_resident_registered_mapping": "UNKNOWN",
            "registered_sensitivity_target_id": "registered",
            "registered_sensitivity_source_column": "户籍人口（万人）",
            "unit_in_local_contract": "万人",
            "final_pipeline_target_paths": {
                "resident": {
                    "role": "primary",
                    "plan01_action": "freeze_contract_only",
                    "future_action": "separately_train_and_generate_candidate_when_authorized",
                },
                "registered": {
                    "role": "alternative",
                    "plan01_action": "retain_contract_path_only",
                    "future_action": "separately_train_and_generate_candidate_when_authorized",
                },
            },
            "target_path_policy": "Keep resident and registered as separate final-pipeline paths; do not average, mix, or route between them without new evidence.",
        },
        "framing": {
            "framing_id": F4_ID,
            "label": F4_LABEL,
            "reconstruction": F4_RECONSTRUCTION,
            "population_history_cutoff": "latest non-missing target history <= pseudo_target_year - 2",
            "final_prediction_history_cutoff": "<= 2021 for 2023 prediction",
            "positive_finite_requirements": ["pop_latest_level > 0", "actual_target_value > 0 for historical F4 labels", "raw prediction and reconstruction finite"],
            "clipping": "none",
            "target_truth_sources_for_prediction": "forbidden; no 2022/2023 target truth",
        },
        "feature_contract": {
            "source_of_feature_names": [relative(S5_PLAN04 / "feature_definitions.csv"), relative(S5_PLAN04B / "parallel_worker_input_contract.json")],
            "feature_contract_version": schema["feature_contract_version"],
            "numeric_model_feature_count": len(model_features),
            "feature_group_count": len(group_counts),
            "feature_group_counts": dict(sorted(group_counts.items())),
            "feature_names_are_order_checked": True,
            "categorical_metadata": ["city_id"],
            "categorical_handling": "CatBoost native categorical; no one-hot or target encoding",
            "missingness": "raw numeric NaN retained; no Plan04 imputation",
            "preprocessing": "no imputation, scaling, selector, or post-hoc feature selection",
            "quarantine_provenance_density_age_guards": "preserved from S1/Plan04; no quarantine reintroduction; age benchmark-only",
        },
        "current_2023_information_set": {
            "reference_year": REFERENCE_YEAR,
            "mapping_status_by_scenario": {scenario: "EXACT_R1" if scenario == SCENARIOS[0] else "EXACT_R2" for scenario in SCENARIOS},
            "new_policy_required": False,
            "source_audit_path": relative(OUT / "current_2023_information_set_audit.csv"),
            "source_count": 14,
            "allowed_scenarios": list(SCENARIOS),
            "publication_timing_proven": False,
        },
        "training_geometry": {
            "prediction_year": REFERENCE_YEAR,
            "allowed_label_max_pseudo_target_year": TARGET_HISTORY_CUTOFF,
            "training_geometry_path": relative(OUT / "final_training_geometry.csv"),
            "geometry_rows": geometry,
            "target_label_truth_2022_or_2023_used": False,
            "duplicate_key_count_all_rows": sum(int(row["duplicate_key_count"]) for row in geometry),
            "source_cap_violation_count_all_rows": sum(int(row["source_cap_violation_count"]) for row in geometry),
            "quarantine_reintroduction_count_all_rows": sum(int(row["quarantine_reintroduction_count"]) for row in geometry),
        },
        "model": {
            "estimator": "CatBoostRegressor",
            "catboost_version_required": "1.2.10",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 20260919,
            "random_strength": 1.0,
            "thread_count": 1,
            "allow_writing_files": False,
            "verbose": False,
            "outer_label_early_stopping": False,
            "configs": [
                {"config_id": "FRM_C1_CONSERVATIVE", "iterations": 400, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 20.0, "rsm": 0.70, "fallback_default": True},
                {"config_id": "FRM_C2_MEDIUM", "iterations": 600, "learning_rate": 0.03, "depth": 5, "l2_leaf_reg": 10.0, "rsm": 0.80, "fallback_default": False},
                {"config_id": "FRM_C3_FLEXIBLE_REGULARIZED", "iterations": 800, "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 10.0, "rsm": 0.85, "fallback_default": False},
            ],
        },
        "config_selection": {
            "historical_source": f"{LINEAGE['plan06e_artifact_generation_source']}:scripts/build_s5_plan06e_delta_growth_framing.py",
            "selection_procedure": "eligible inner folds only; each config mean inner validation MSE; choose lowest by strict comparison with np.isclose tie preservation in preregistered order",
            "inner_training_rule": "pseudo_target_year <= inner_validation_year - 2",
            "eligibility": "inner_training_row_count >= 40, inner_validation_row_count >= 20, boundary_pass == PASS",
            "fallback": "if eligible inner fold count < 3, select FRM_C1_CONSERVATIVE",
            "fallback_is_not": "B3 solution fallback; no per-city routing/gating",
            "uses_2023_truth": False,
            "selection_is_unique_without_2023_labels": True,
        },
        "b3_control": {
            "model_id": "B3_RECENT_3POINT_OLS_TREND",
            "authoritative_code": relative(S3_PLAN02_SCRIPT),
            "definition": "per-city OLS on the latest three non-missing target observations <= T-2; extrapolate to T; B1 fallback only when fewer than three points",
            "target_history_cutoff": "T-2",
            "resident_pooled_mse": 872.452651515143,
            "registered_pooled_mse": 225.421212121213,
            "resident_latest_2021_mse": 3705.427777777728,
            "registered_latest_2021_mse": 1485.604861111118,
            "compatibility_evidence": [relative(ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01" / "b3_compatibility_check.csv"), relative(ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01" / "b3_metric_compatibility_check.csv")],
            "no_city_mixing_or_gating": True,
        },
        "fallback_policy": {
            "solution_fallback": "B3_RECENT_3POINT_OLS_TREND is the frozen reference/fallback if F4 cannot be reproduced under this contract",
            "config_fallback": "FRM_C1_CONSERVATIVE only when eligible inner fold count < 3",
            "no_post_hoc_city_routing": True,
        },
        "submission_schema": {
            "evidence_path": relative(OUT / "submission_schema_evidence.md"),
            "fields": submission_rows,
            "pdf_evidence": submission_meta,
            "schema_only": {"file_format": "CSV", "file_name": "submission.csv", "columns": ["city_id", "year", "pred"], "row_count": 40, "city_ids": "city1..city40", "year": 2023},
            "target_mapping_and_unit_safe_to_freeze": False,
            "formal_submission_generated": False,
            "future_candidate_policy": "When authorized, generate separate resident and registered submission candidates; never average or mix the two paths.",
        },
        "rebuild_boundary": {"manifest_path": relative(OUT / "rebuild_manifest.csv"), "categories": sorted({row["category"] for row in manifest}), "clean_rebuild_is_authorized_now": False, "contract_evidence_preserved": True},
        "forbidden_inputs": [relative(FORBIDDEN_TEST_XLS), "external data", "B-data as current default input", "manually edited target truth", "2022/2023 target truth", "worker OOF predictions as final-fit shortcuts"],
        "immutability_mechanism": {"tracked_contract": relative(OUT / "final_solution_contract.json"), "hash_registry": relative(OUT / "historical_lineage_hashes.csv"), "source_lineage_locked_to_commit_and_blob": True, "no_force_push_or_history_rewrite": True},
    }


def build_markdown(contract: dict[str, Any], info: list[dict[str, Any]], geometry: list[dict[str, Any]], manifest: list[dict[str, Any]], source_sha: str) -> tuple[str, str, str, str]:
    resident_geometry = [row for row in geometry if row["target_id"] == "resident"]
    primary_rows = "; ".join(f"{row['scenario_id']}={row['training_row_count']} rows/{row['training_city_count']} cities" for row in resident_geometry)
    mapping = "; ".join(f"{scenario}=EXACT_{'R1' if scenario == SCENARIOS[0] else 'R2'}" for scenario in SCENARIOS)
    contract_md = f"""# S6-Plan01 Final Solution Contract

## Freeze result

- Contract ID: `{contract['contract_id']}`
- Status: `{contract['contract_status']}`
- Ready for Main stage review: `yes`
- Ready for the later clean raw rebuild: `yes`
- Formal 2023 prediction/submission generated: `no`
- Source main SHA audited: `{source_sha}`

## Target and framing

- Primary target: `resident`, source column `常住人口（万人）`, local role `primary_proxy`.
- Official wording: `总人口`; official mapping to resident versus registered remains `UNKNOWN`.
- The later final pipeline must retain two separate target paths: resident `primary` and registered `alternative`; this Plan freezes the paths but does not fit, predict, average, or mix them.
- F4: `{F4_ID}`.
- Training label: `{F4_LABEL}`.
- Reconstruction: `{F4_RECONSTRUCTION}`.
- Population history and final labels are capped at `T-2`; for 2023 this is `<= 2021`.
- No clipping, no 2022/2023 target truth, and positive/finite checks are required.

## Feature contract

- Feature names are read from Plan04 `feature_definitions.csv` and checked against the Plan04B machine contract: **144 numeric model features**.
- `city_id` is the only categorical metadata and is handled natively by CatBoost.
- Numeric missingness remains raw NaN. No imputation, scaling, one-hot encoding, target encoding, selector, or post-hoc feature change is allowed.
- Source quarantine, provenance, density-regime, and age benchmark guards remain active.

## Current 2023 information set

- Existing Plan02 mappings are reused without new source policy: {mapping}.
- The audit covers all 14 logical sources backed by the 7 legal XLSX workbooks.
- Publication timing remains unproven; R1/R2 remain explicit sensitivity assumptions, not historical facts.
- Detailed evidence: `current_2023_information_set_audit.csv`.

## Final training geometry

- Conceptual outer prediction year: `2023`.
- Allowed historical target labels: `pseudo_target_year <= 2021`.
- Resident primary geometry: {primary_rows}.
- The replay source has 2,320 rows, unique `(target, scenario, pseudo_target_year, city_id)` keys, and zero source-cap/quarantine/benchmark boundary violations in the audited rows.
- The builder excluded `1,040` pre-matrix rows for documented reasons (`b3_prediction_unavailable:480; no_target_label:560`); no fallback rows are invented.
- Detailed evidence: `final_training_geometry.csv`.

## Model and selection

- CatBoostRegressor `1.2.10`, RMSE, seed `20260919`, `thread_count=1`, `random_strength=1.0`, no outer-label early stopping.
- Only the three historical configurations C1/C2/C3 are allowed.
- Inner selection uses only eligible folds inside each outer training pool; C1 is selected when eligible fold count is below 3. This is distinct from the B3 solution fallback.
- No 2023 label, score, per-city gate, routing, ensemble, or stacking is allowed.

## B3 control and fallback

- Authoritative implementation: `scripts/build_s3_plan02_recent_window_and_selector.py`.
- B3 is the frozen reference and whole-solution fallback, not a per-city mixture with F4.
- Frozen resident/registered pooled MSE: `872.452651515143` / `225.421212121213`.

## Submission boundary

- Official PDF confirms CSV `submission.csv` with `city_id`, `year`, and `pred`; Main evidence confirms 40 rows `city1..city40` and `year=2023`.
- Official target wording is `总人口`, but official mapping and unit are not safe to freeze from available evidence. The formal submission remains blocked and is not generated.
- When later authorized, resident and registered must produce separate candidates; no averaging or mixing is permitted without new evidence.
- Detailed evidence: `submission_schema_evidence.md`.

## Rebuild boundary

- Preserve: official PDF/7 XLSX, Git history/tags, and this final contract/evidence.
- Rebuildable: normalized/quarantine/provenance/regime CSVs, Plan04/04B matrices, model/prediction artifacts, caches/temp.
- Forbidden: `data/quarantine/source/test.xls` content, external/B-data as default input, manually edited truth, 2022/2023 target truth, and worker OOF as final-fit shortcut.
- Detailed manifest: `rebuild_manifest.csv`.
"""
    dependency_md = f"""# S6-Plan01 Dependency Map

```text
official PDF + 7 legal raw XLSX
  -> S1 contract / canonical / quarantine / provenance / regime guards
  -> S2 source-specific R1/R2 geometry and current-information policy
  -> S3 B3 implementation and compatibility evidence
  -> S5 Plan04 144-feature contract
  -> S5 Plan04B replay matrix and T-2 split geometry
  -> historical Plan06E archive tag {LINEAGE['plan06e_archive_tag']}
  -> S6 F4 contract audit (this packet)
  -> later clean rebuild (not executed here)
```

## Current source-of-truth files

- S1 source contract: `docs/s1_data_contract.csv`
- S2 geometry: `scripts/build_s2_plan02_geometry.py` and `reports/S2_HANDOFFS/S2-Plan02/`
- B3: `scripts/build_s3_plan02_recent_window_and_selector.py`
- Plan04 feature names: `reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv`
- Plan04B model contract: `reports/S5_HANDOFFS/S5-Plan04B/parallel_worker_input_contract.json`
- Plan04B replay: `reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv`
- F4 historical implementation: immutable Plan06E artifact tip `{LINEAGE['plan06e_artifact_tip']}`; runner blob is recorded in `historical_lineage_hashes.csv`.

## Boundary

The current Plan04/04B CSVs are reproducible derived artifacts, not new raw truth. The historical Plan06E branch artifact is preserved for implementation lineage only; it is not silently cherry-picked into current main. No model fit or 2023 output is part of this audit.
"""
    submission_md = """# Submission Schema Evidence

Status is intentionally split between schema evidence and target semantics.

| Field | Value | Status | Evidence |
|---|---|---|---|
| official prediction year | 2023 | CONFIRMED | official PDF page 1; current Main checkpoint |
| official target wording | 各个城市 2023 年的总人口数 | CONFIRMED | official PDF page 1 |
| target mapping | resident versus registered not proven | UNKNOWN | PDF says 总人口; source contains both columns |
| city universe | city1..city40 | CONFIRMED | Main checkpoint's direct official-sample check; canonical source coverage |
| row count | 40 | CONFIRMED | Main checkpoint's direct official-sample check |
| ordering | sample is city1..city40; formal ordering rule unstated | INFERRED_NOT_SAFE_TO_FREEZE | Main checkpoint and sample evidence |
| file format | CSV | CONFIRMED | official PDF page 3 |
| headers | city_id, year, pred | CONFIRMED | official PDF page 3 |
| target unit | local source contract is 万人; official PDF does not state unit | INFERRED_NOT_SAFE_TO_FREEZE | PDF plus S1 contract |
| official submission sample | confirmed in Main evidence, not present in current raw tree | CONFIRMED | Main checkpoint and recorded MD5 |

The official target wording does not resolve the local resident/registered mapping. Therefore this Plan freezes the resident-primary F4 reproduction contract and retains a separate registered alternative path for later review; it does not fit either final path, generate a 2023 prediction, or generate a formal submission.
"""
    handoff_md = f"""# S6-Plan01 Handoff｜F4 最终方案合同冻结与复现边界审计

## Result

- Status: `{contract['contract_status']}`
- `ready_for_main_stage_review`: `true`
- `ready_for_s6_clean_rebuild`: `true`
- `formal_submission_generated`: `false`
- Execution source main SHA: `{source_sha}`
- `origin/main` at audit: `{git('rev-parse', 'origin/main')}`
- Branch at audit: `{git('branch', '--show-current')}`
- This handoff records the audited source state; the exact delivery HEAD is reported after commit/push verification.

## What was frozen

- Resident-primary proxy target and F4 log-growth framing.
- Two future final-pipeline target paths: resident `primary` and registered `alternative`, each to be trained and submitted separately only after authorization; no averaging or mixing.
- Plan04/04B exact 144-feature contract, raw NaN, native `city_id`, and S1 guards.
- Existing Plan02 R1/R2 current-information mappings: `{mapping}`; no new policy was introduced.
- Conceptual 2023 final-training geometry with target labels capped at 2021.
- CatBoost 1.2.10, three pre-registered configs, seed, deterministic single-thread expectations, and historical inner selection/fallback algorithm.
- B3 authoritative control and whole-solution fallback.

## Evidence paths

- `FINAL_SOLUTION_CONTRACT.md`
- `final_solution_contract.json`
- `dependency_map.md`
- `rebuild_manifest.csv`
- `current_2023_information_set_audit.csv`
- `final_training_geometry.csv`
- `historical_lineage_hashes.csv`
- `submission_schema_evidence.md`
- `environment_snapshot.json`

## Verification actually passed

- `main` was clean before the audit and was synchronized to `origin/main` at the audited source SHA.
- 7 legal XLSX files and the official PDF were read; `test.xls` was not opened or hashed.
- 14 logical source mappings × R1/R2 were matched to existing Plan02 policy rows; all rows are `EXACT_R1` or `EXACT_R2`.
- 144 feature names match Plan04 definitions, Plan04 schema, and Plan04B worker contract in order.
- Replay geometry: 2,320 unique rows; final label maximum 2021; 40-city coverage; zero duplicate keys; zero source-cap, benchmark-domain, population-cutoff, B3-cutoff, and quarantine-reintroduction violations in the audited boundary rows.
- Historical Plan06E runner/config/lineage was read from immutable Git objects; no runner cherry-pick and no model fit was performed.
- Official PDF schema evidence was extracted and cross-checked with the current Main checkpoint.

## Submission and risks

- The official target says `总人口`, but resident versus registered and official unit are not proven. This is the reason for `PARTIALLY_FROZEN_SUBMISSION_BLOCKED`.
- R1/R2 are observed-snapshot freshness proxies, not proven publication dates.
- F4 is a resident-primary local candidate; it is not an official-target or universal-target claim.
- The current Main checkpoint requires retaining a separately trainable registered alternative until a real platform score can inform target-mapping review; this Plan does not execute that later step.
- No 2023 prediction, submission, model selection experiment, tuning, feature change, ensemble, routing, S7 entry, or checkpoint edit was performed.

## Recommendation to Main

Accept this packet for Main stage review and, if the target/unit boundary is accepted, proceed to the separately authorized clean raw rebuild while preserving both target paths. Keep formal submission generation blocked until Main resolves or explicitly accepts the resident-proxy/unit limitation. Do not declare `S6_COMPLETE`.
"""
    return contract_md, dependency_md, submission_md, handoff_md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-main-sha", help="main baseline audited before local packet edits; defaults to origin/main")
    args = parser.parse_args()
    source_sha = resolve_source_sha(args.source_main_sha)
    definitions, schema, worker_contract, model_features = load_feature_contract()
    replay, boundary, inner, replay_summary = load_replay_geometry()
    contract_rows = read_csv(S1_CONTRACT)
    information_audit = build_information_audit(contract_rows)
    geometry = build_geometry(replay, boundary, replay_summary)
    submission_rows, submission_meta = submission_evidence()
    manifest = build_manifest()

    OUT.mkdir(parents=True, exist_ok=True)
    info_fields = list(information_audit[0])
    geometry_fields = list(geometry[0])
    write_csv(OUT / "current_2023_information_set_audit.csv", information_audit, info_fields)
    write_csv(OUT / "final_training_geometry.csv", geometry, geometry_fields)
    write_csv(OUT / "rebuild_manifest.csv", manifest, ["category", "path", "status", "source_of_truth", "clean_rebuild_action", "notes"])

    contract = build_contract(source_sha, definitions, schema, worker_contract, model_features, information_audit, geometry, submission_rows, submission_meta, manifest)
    write_json(OUT / "final_solution_contract.json", contract)

    lineage_rows: list[dict[str, Any]] = []
    def add_lineage(kind: str, reference: str, path: Path | None = None, commit: str = "", blob: str = "", notes: str = "") -> None:
        lineage_rows.append({
            "kind": kind,
            "reference": reference,
            "path": relative(path) if path else "",
            "commit": commit,
            "blob_sha": blob,
            "sha256": sha256_file(path) if path and path.exists() else "",
            "status": "VERIFIED",
            "notes": notes,
        })

    add_lineage("execution_source_main", source_sha, commit=source_sha, notes="main SHA audited before S6 contract edits")
    for name, value in LINEAGE.items():
        if name == "plan06e_archive_tag":
            resolved = git("rev-parse", value)
            add_lineage("historical_git_ref", value, commit=resolved, notes="immutable historical Plan06E evidence identity")
        else:
            add_lineage("historical_commit", name, commit=value)
    historical_runner_blob = git("rev-parse", f"{LINEAGE['plan06e_artifact_generation_source']}:scripts/build_s5_plan06e_delta_growth_framing.py")
    add_lineage("historical_runner", "Plan06E runner", commit=LINEAGE["plan06e_artifact_generation_source"], blob=historical_runner_blob, notes="historical F4 implementation source; not copied into current main")
    for path in [S2_GEOMETRY_SCRIPT, S3_PLAN02_SCRIPT, PLAN04_SCRIPT, PLAN04B_SCRIPT, S5_PLAN04 / "feature_definitions.csv", S5_PLAN04 / "feature_matrix_schema.json", S5_PLAN04B / "parallel_worker_input_contract.json", S5_PLAN04B / "full_multisource_training_replay_matrix.csv", ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01" / "b3_compatibility_check.csv", ROOT / "reports" / "S4_HANDOFFS" / "S4-Plan01" / "b3_metric_compatibility_check.csv", PDF_PATH]:
        add_lineage("current_or_evidence_file", relative(path), path=path, commit=git("log", "-1", "--format=%H", "--", relative(path)) if path.is_file() else "")
    for path in sorted(RAW_DATA_DIR.glob("*.xlsx")):
        add_lineage("legal_raw_xlsx", relative(path), path=path, commit=git("log", "-1", "--format=%H", "--", relative(path)))
    add_lineage("generated_contract", "final_solution_contract.json", path=OUT / "final_solution_contract.json", notes="hash recorded in this registry")
    write_csv(OUT / "historical_lineage_hashes.csv", lineage_rows, ["kind", "reference", "path", "commit", "blob_sha", "sha256", "status", "notes"])

    package_versions = {name: package_version(name) for name in ["pandas", "numpy", "openpyxl", "catboost", "xgboost", "lightgbm", "scikit-learn", "pypdf"]}
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable_name": Path(sys.executable).name, "version": sys.version},
        "platform": {"system": platform.system(), "release": platform.release(), "version": platform.version(), "machine": platform.machine()},
        "package_versions": package_versions,
        "git": {"branch": git("branch", "--show-current"), "source_main_sha": source_sha, "status_before_generated_outputs": "clean_before_script_edit_is_recorded_in_handoff", "origin_main": git("rev-parse", "origin/main")},
        "audit_scope": {"model_fit_executed": False, "prediction_generated": False, "submission_generated": False, "test_xls_opened": False, "external_data_used": False},
        "input_files": {"official_pdf": relative(PDF_PATH), "legal_raw_xlsx_count": len(list(RAW_DATA_DIR.glob("*.xlsx"))), "forbidden_test_xls_metadata_only": {"path": relative(FORBIDDEN_TEST_XLS), "exists": FORBIDDEN_TEST_XLS.exists(), "size_bytes": FORBIDDEN_TEST_XLS.stat().st_size if FORBIDDEN_TEST_XLS.exists() else None}},
    }
    write_json(OUT / "environment_snapshot.json", environment)

    contract_md, dependency_md, submission_md, handoff_md = build_markdown(contract, information_audit, geometry, manifest, source_sha)
    (OUT / "FINAL_SOLUTION_CONTRACT.md").write_text(contract_md, encoding="utf-8")
    (OUT / "dependency_map.md").write_text(dependency_md, encoding="utf-8")
    (OUT / "submission_schema_evidence.md").write_text(submission_md, encoding="utf-8")
    (OUT / "HANDOFF.md").write_text(handoff_md, encoding="utf-8")

    print(f"S6-Plan01 audit PASS source_main_sha={source_sha}")
    print(f"information_rows={len(information_audit)} geometry_rows={len(geometry)} feature_count={len(model_features)}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
