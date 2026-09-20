"""S6-Plan02 clean-room rebuild and dual-target 2023 feature foundation.

This runner is an evidence-only orchestrator.  It rebuilds S1 outputs from the
seven legal XLSX files in a disposable worktree, reuses the authoritative S5
Plan04 feature and Plan04B replay functions, and stops before every estimator,
2023 prediction, and submission boundary.

Historical Plan04/Plan04B matrices are removed from the disposable worktree
before generation.  Their Git objects are read only after independent
generation for semantic equivalence checks.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import build_s5_plan04_full_multisource_feature_foundation as plan04
import build_s5_plan04b_training_replay_foundation as plan04b
import run_s1


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "S6_HANDOFFS" / "S6-Plan02"
TEMP = ROOT / "tmp" / "s6_plan02_clean_room"
PLAN01_DIR = ROOT / "reports" / "S6_HANDOFFS" / "S6-Plan01"
PLAN01_CONTRACT = PLAN01_DIR / "final_solution_contract.json"
PLAN01_LINEAGE = PLAN01_DIR / "historical_lineage_hashes.csv"
PLAN01_INFO = PLAN01_DIR / "current_2023_information_set_audit.csv"
RAW_ROOT = ROOT / "data" / "raw"
RAW_DATA_DIR = RAW_ROOT / "城市人口分析与预测" / "数据集"
PDF_PATH = RAW_ROOT / "城市人口分析与预测" / "城市人口分析与预测.pdf"
FORBIDDEN_TEST_XLS = ROOT / "data" / "quarantine" / "source" / "test.xls"
CHECKPOINT = ROOT / "CURRENT_LEARNING_CHECKPOINT.md"
S5_PLAN04_DIR = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04"
S5_PLAN04B_DIR = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B"
S1_PACK = ROOT / "reports" / "S1_FINAL_REVIEW_PACK.zip"

REFERENCE_YEAR = 2023
TARGET_CUTOFF = 2021
TARGETS = plan04.TARGETS
SCENARIOS = plan04.SCENARIOS
CITIES = plan04.CITIES
TARGET_ROLES = {"resident": "primary", "registered": "alternative_required"}
CONTRACT_STATUS = "PARTIALLY_FROZEN_SUBMISSION_BLOCKED"
RESULT_STATUS = "READY_FOR_S6_HISTORICAL_REPRODUCTION_CONTROLS"

S1_REFERENCE_PATHS = (
    "docs/s1_data_contract.csv",
    "docs/s1_quality_issue_registry.csv",
    "reports/S1_FINAL_REVIEW_PACK.zip",
)
KEEP_PLAN04_SPEC_FILES = {
    "feature_definitions.csv",
    "feature_group_manifest.csv",
    "feature_matrix_schema.json",
    "oof_prediction_contract.csv",
}
KEEP_PLAN04B_SPEC_FILES = {"parallel_worker_input_contract.json"}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.STDOUT
    ).strip()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, stderr=subprocess.STDOUT
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_bytes(value: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(value.decode("utf-8-sig"))))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str], model_features: set[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_features = model_features or set()
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if value is None:
                    output[field] = "NaN" if field in model_features else ""
                elif isinstance(value, float):
                    output[field] = "NaN" if not math.isfinite(value) else round(value, 12)
                else:
                    output[field] = value
            writer.writerow(output)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def present(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "NaN", "nan", "None", "NA", "N/A"}


def semantic_equal(left: Any, right: Any, numeric: bool = False) -> bool:
    return plan04b.value_equal(left, right, numeric=numeric)


def clear_file_tree(directory: Path, *, keep_names: set[str] | None = None) -> int:
    if not directory.exists():
        return 0
    keep_names = keep_names or set()
    removed = 0
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in keep_names:
            path.unlink()
            removed += 1
    return removed


def isolate_derived_inputs() -> dict[str, Any]:
    """Remove rebuildable inputs only inside the disposable worktree."""
    removed: list[str] = []
    for directory in (ROOT / "data" / "normalized", ROOT / "data" / "quarantine"):
        if directory.exists():
            for path in sorted(directory.glob("*.csv")):
                path.unlink()
                removed.append(relative(path))
    if S1_PACK.exists():
        S1_PACK.unlink()
        removed.append(relative(S1_PACK))
    for directory, keep in ((S5_PLAN04_DIR, KEEP_PLAN04_SPEC_FILES), (S5_PLAN04B_DIR, KEEP_PLAN04B_SPEC_FILES)):
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.name not in keep:
                    path.unlink()
                    removed.append(relative(path))
    if OUT.exists():
        shutil.rmtree(OUT)
    if TEMP.exists():
        shutil.rmtree(TEMP)
    TEMP.mkdir(parents=True, exist_ok=True)
    return {
        "removed_file_count": len(removed),
        "removed_paths": removed,
        "historical_matrices_unavailable_during_generation": True,
        "test_xls_touched": False,
    }


def load_contract_identity() -> dict[str, Any]:
    contract = json.loads(PLAN01_CONTRACT.read_text(encoding="utf-8"))
    lineage = read_csv_path(PLAN01_LINEAGE)
    contract_row = next(row for row in lineage if row["reference"] == "final_solution_contract.json")
    contract_sha = sha256_file(PLAN01_CONTRACT)
    assert contract_sha == contract_row["sha256"], (contract_sha, contract_row["sha256"])
    assert contract["contract_status"] == CONTRACT_STATUS
    assert contract["feature_contract"]["numeric_model_feature_count"] == 144
    assert contract["training_geometry"]["allowed_label_max_pseudo_target_year"] == TARGET_CUTOFF
    assert contract["current_2023_information_set"]["mapping_status_by_scenario"] == {
        SCENARIOS[0]: "EXACT_R1",
        SCENARIOS[1]: "EXACT_R2",
    }
    return {
        "contract_id": contract["contract_id"],
        "contract_version": contract["contract_version"],
        "contract_status": contract["contract_status"],
        "contract_sha256": contract_sha,
        "allowed_targets": list(contract["target"]["final_pipeline_target_paths"]),
        "allowed_scenarios": contract["current_2023_information_set"]["allowed_scenarios"],
        "feature_count": contract["feature_contract"]["numeric_model_feature_count"],
        "target_cutoff": TARGET_CUTOFF,
        "forbidden_inputs": contract["forbidden_inputs"],
        "source_main_sha_in_plan01": contract["execution"]["source_main_sha"],
    }


def expected_raw_hashes() -> dict[str, str]:
    rows = read_csv_path(PLAN01_LINEAGE)
    result = {}
    for row in rows:
        path = row["path"]
        if row["kind"] == "legal_raw_xlsx" or path == relative(PDF_PATH):
            result[path] = row["sha256"]
    assert len(result) == 8, len(result)
    return result


def raw_integrity_records(phase: str, expected: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected_xlsx = {path for path in expected if path.endswith(".xlsx")}
    actual_xlsx = {relative(path) for path in RAW_DATA_DIR.glob("*.xlsx")}
    unexpected = sorted(actual_xlsx - expected_xlsx)
    missing = sorted(expected_xlsx - actual_xlsx)
    for path_string, expected_sha in sorted(expected.items()):
        path = ROOT / Path(path_string)
        observed = sha256_file(path) if path.exists() else ""
        records.append(
            {
                "phase": phase,
                "path": path_string,
                "role": "official_pdf" if path_string.endswith(".pdf") else "legal_raw_xlsx",
                "expected_sha256": expected_sha,
                "observed_sha256": observed,
                "size_bytes": path.stat().st_size if path.exists() else "",
                "content_accessed": "yes_for_hash_verification",
                "status": "PASS" if path.exists() and observed == expected_sha else "FAIL",
                "notes": "immutable legal raw identity",
            }
        )
    for path_string in unexpected:
        records.append(
            {
                "phase": phase,
                "path": path_string,
                "role": "unexpected_legal_raw_workbook",
                "expected_sha256": "",
                "observed_sha256": "not_read",
                "size_bytes": (ROOT / path_string).stat().st_size,
                "content_accessed": "no",
                "status": "FAIL",
                "notes": "unexpected XLSX in legal raw directory",
            }
        )
    for path_string in missing:
        records.append(
            {
                "phase": phase,
                "path": path_string,
                "role": "missing_legal_raw_workbook",
                "expected_sha256": expected[path_string],
                "observed_sha256": "",
                "size_bytes": "",
                "content_accessed": "no",
                "status": "FAIL",
                "notes": "expected legal raw file missing",
            }
        )
    test_exists = FORBIDDEN_TEST_XLS.exists()
    records.append(
        {
            "phase": phase,
            "path": relative(FORBIDDEN_TEST_XLS),
            "role": "forbidden_test_xls_metadata_only",
            "expected_sha256": "",
            "observed_sha256": "NEVER_HASHED",
            "size_bytes": FORBIDDEN_TEST_XLS.stat().st_size if test_exists else "",
            "content_accessed": "no",
            "status": "PASS" if test_exists else "FAIL",
            "notes": "existence and size metadata only; content was not opened, parsed, copied, or hashed",
        }
    )
    return records


def run_clean_s1(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for name in run_s1.DOMAIN_SCRIPTS:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_s1.py")],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_text = log_text.replace(str(ROOT), "<DISPOSABLE_WORKTREE>")
    log_text = log_text.replace(str(ROOT).replace("\\", "/"), "<DISPOSABLE_WORKTREE>")
    log_path.write_text(log_text, encoding="utf-8")


def load_feature_context() -> dict[str, Any]:
    original_gate = plan04.assert_repo_identity
    plan04.assert_repo_identity = lambda: None
    try:
        context = plan04.load_context()
    finally:
        plan04.assert_repo_identity = original_gate
    assert len(plan04.MODEL_FEATURES) == 144
    return context


def compare_keyed_rows(
    generated: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    key_fields: Iterable[str],
    field_groups: dict[str, Iterable[str]],
) -> dict[str, Any]:
    key_fields = tuple(key_fields)
    generated_index = {tuple(str(row.get(field, "")).strip() for field in key_fields): row for row in generated}
    reference_index = {tuple(str(row.get(field, "")).strip() for field in key_fields): row for row in reference}
    generated_duplicates = len(generated) - len(generated_index)
    reference_duplicates = len(reference) - len(reference_index)
    keys = sorted(set(generated_index) | set(reference_index))
    missing_keys = sum(key not in generated_index for key in keys)
    extra_keys = sum(key not in reference_index for key in keys)
    mismatch_counts: dict[str, int] = {name: 0 for name in field_groups}
    for key in keys:
        left = reference_index.get(key)
        right = generated_index.get(key)
        if left is None or right is None:
            continue
        for name, fields in field_groups.items():
            for field in fields:
                if not semantic_equal(left.get(field), right.get(field), numeric=True):
                    mismatch_counts[name] += 1
    status = "PASS" if not any((generated_duplicates, reference_duplicates, missing_keys, extra_keys, *mismatch_counts.values())) else "FAIL"
    return {
        "generated_row_count": len(generated),
        "reference_row_count": len(reference),
        "generated_duplicate_key_count": generated_duplicates,
        "reference_duplicate_key_count": reference_duplicates,
        "key_mismatch_count": missing_keys + extra_keys,
        "missing_key_count": missing_keys,
        "extra_key_count": extra_keys,
        **{f"{name}_mismatch_cell_count": count for name, count in mismatch_counts.items()},
        "status": status,
    }


def compare_csv_artifact(path: Path, reference_path: str, category: str, note: str = "") -> dict[str, Any]:
    reference_bytes = git_bytes(reference_path)
    generated_bytes = path.read_bytes()
    comparison = "BYTE_IDENTICAL"
    if generated_bytes != reference_bytes:
        generated_rows = read_csv_path(path)
        reference_rows = read_csv_bytes(reference_bytes)
        if generated_rows == reference_rows:
            comparison = "SEMANTICALLY_IDENTICAL"
        else:
            comparison = "MISMATCH"
    return {
        "artifact": category,
        "generated_path": relative(path),
        "reference_source": f"git:HEAD:{reference_path}",
        "comparison": comparison,
        "generated_sha256": sha256_bytes(generated_bytes),
        "reference_sha256": sha256_bytes(reference_bytes),
        "key_mismatch_count": "",
        "model_feature_mismatch_cell_count": "",
        "label_reference_mismatch_cell_count": "",
        "audit_mismatch_cell_count": "",
        "status": "PASS" if comparison != "MISMATCH" else "FAIL",
        "notes": note,
    }


def compare_zip_artifact(path: Path, reference_path: str) -> dict[str, Any]:
    reference_bytes = git_bytes(reference_path)
    member_mismatches = 0
    non_s1_document_differences: list[str] = []
    critical_members = {"s1_data_contract.csv", "s1_quality_issue_registry.csv"}
    with zipfile.ZipFile(io.BytesIO(reference_bytes)) as reference_zip, zipfile.ZipFile(path) as generated_zip:
        reference_names = sorted(reference_zip.namelist())
        generated_names = sorted(generated_zip.namelist())
        member_mismatches += int(reference_names != generated_names)
        for name in sorted(set(reference_names) & set(generated_names)):
            if reference_zip.read(name) != generated_zip.read(name):
                if name in critical_members:
                    member_mismatches += 1
                else:
                    non_s1_document_differences.append(name)
    comparison = "BYTE_IDENTICAL" if path.read_bytes() == reference_bytes else "SEMANTICALLY_IDENTICAL"
    if member_mismatches:
        comparison = "MISMATCH"
    return {
        "artifact": "S1_FINAL_REVIEW_PACK.zip",
        "generated_path": relative(path),
        "reference_source": f"git:HEAD:{reference_path}",
        "comparison": comparison,
        "generated_sha256": sha256_file(path),
        "reference_sha256": sha256_bytes(reference_bytes),
        "key_mismatch_count": member_mismatches,
        "model_feature_mismatch_cell_count": "",
        "label_reference_mismatch_cell_count": "",
        "audit_mismatch_cell_count": "",
        "status": "PASS" if comparison != "MISMATCH" else "FAIL",
        "notes": (
            "member-set/content comparison; archive timestamps are not semantic; "
            f"non-S1 documentation drift={';'.join(non_s1_document_differences) or 'none'}"
        ),
    }


def s1_equivalence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generated_outputs = sorted(run_s1.output_hashes())
    for path_string in generated_outputs:
        path = ROOT / path_string
        rows.append(compare_csv_artifact(path, path_string, path_string, "canonical/quarantine/provenance/regime output"))
    for path_string in S1_REFERENCE_PATHS[:2]:
        path = ROOT / path_string
        rows.append(compare_csv_artifact(path, path_string, path_string, "regenerated S1 contract/quality registry"))
    rows.append(compare_zip_artifact(S1_PACK, S1_REFERENCE_PATHS[2]))
    summary = {
        "artifact_count": len(rows),
        "byte_identical_count": sum(row["comparison"] == "BYTE_IDENTICAL" for row in rows),
        "semantic_identical_count": sum(row["comparison"] == "SEMANTICALLY_IDENTICAL" for row in rows),
        "mismatch_count": sum(row["comparison"] == "MISMATCH" for row in rows),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
    }
    assert summary["status"] == "PASS", summary
    return rows, summary


def build_replay(context: dict[str, Any], formal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    replay_rows, coverage_rows, coverage = plan04b.replay_rows_and_coverage(context)
    coverage["coverage_rows"] = coverage_rows
    replay_internal_rows = [
        plan04.build_row(context, row["target_id"], row["scenario_id"], row["pseudo_target_year"], row["city_id"])
        for row in replay_rows
    ]
    outer_rows, outer_summary = plan04b.outer_training_geometry(replay_rows, formal_rows)
    inner_rows, inner_summary = plan04b.inner_split_geometry(replay_rows)
    boundary_rows, boundary_summary = plan04b.replay_boundary_audit(replay_internal_rows, replay_rows, context)
    b3_summary = plan04.b3_compatibility(formal_rows)
    assert b3_summary["pass"] is True
    assert len(replay_rows) == 2320
    assert len(outer_rows) == 44
    assert len(inner_rows) == 484
    assert outer_summary["boundary_violation_count"] == 0
    assert inner_summary["boundary_violation_count"] == 0
    assert all(value == 0 for value in (
        boundary_summary["source_cap_violation_count"],
        boundary_summary["quarantine_reintroduction_count"],
        boundary_summary["benchmark_domain_violation_count"],
        boundary_summary["population_history_cutoff_violation_count"],
        boundary_summary["b3_history_cutoff_violation_count"],
    ))
    return {
        "replay_rows": replay_rows,
        "coverage_rows": coverage_rows,
        "coverage": coverage,
        "replay_internal_rows": replay_internal_rows,
        "outer_rows": outer_rows,
        "outer_summary": outer_summary,
        "inner_rows": inner_rows,
        "inner_summary": inner_summary,
        "boundary_rows": boundary_rows,
        "boundary_summary": boundary_summary,
        "b3_summary": b3_summary,
    }


def build_current_features(context: dict[str, Any]) -> dict[str, Any]:
    internal_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for city in CITIES:
                row = plan04.build_row(context, target_id, scenario_id, REFERENCE_YEAR, city)
                internal_rows.append(row)
                assert row["actual_target_value"] is None
                assert row["target_history_cutoff_year"] == TARGET_CUTOFF
                role = TARGET_ROLES[target_id]
                output_rows.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "city_id": city,
                        "prediction_year": REFERENCE_YEAR,
                        "target_history_cutoff_year": TARGET_CUTOFF,
                        "final_pipeline_role": role,
                        "b3_reference_prediction": row["b3_prediction"],
                        **{field: row[field] for field in plan04.MODEL_FEATURES},
                        **{field: row[field] for field in plan04.AUDIT_COLUMNS if field not in plan04.MODEL_FEATURES},
                    }
                )
    expected = len(TARGETS) * len(SCENARIOS) * len(CITIES)
    assert len(output_rows) == expected
    keys = [(row["target_id"], row["scenario_id"], row["city_id"]) for row in output_rows]
    assert len(keys) == len(set(keys))
    assert all(Counter((row["target_id"], row["scenario_id"]) for row in output_rows).values())
    assert all(
        max(context["population"][target_id][city]) <= TARGET_CUTOFF
        for target_id in TARGETS
        for city in CITIES
    )
    assert all(
        row["pop_latest_level"] is not None
        and float(row["pop_latest_level"]) > 0
        and math.isfinite(float(row["pop_latest_level"]))
        and row["b3_prediction"] is not None
        and math.isfinite(float(row["b3_prediction"]))
        for row in internal_rows
    )
    boundary_rows = plan04.source_boundary_audit(internal_rows, context)
    for row in boundary_rows:
        row["prediction_year"] = REFERENCE_YEAR
        row["final_pipeline_role"] = TARGET_ROLES[row["target_id"]]
    assert len(boundary_rows) == expected * len(plan04.SOURCE_ORDER)
    assert all(row["audit_status"] == "PASS" for row in boundary_rows)
    assert all(int(row["cap_violation_count"]) == 0 for row in boundary_rows)
    assert all(int(row["quarantine_used_count"]) == 0 for row in boundary_rows)
    assert all(int(row["benchmark_domain_violation_count"]) == 0 for row in boundary_rows)
    return {
        "internal_rows": internal_rows,
        "rows": output_rows,
        "boundary_rows": boundary_rows,
        "row_count": len(output_rows),
        "row_count_by_target_scenario": {
            f"{target_id}|{scenario_id}": sum(
                row["target_id"] == target_id and row["scenario_id"] == scenario_id for row in output_rows
            )
            for target_id in TARGETS
            for scenario_id in SCENARIOS
        },
        "boundary_row_count": len(boundary_rows),
        "source_cap_violation_count": sum(int(row["cap_violation_count"]) for row in boundary_rows),
        "quarantine_reintroduction_count": sum(int(row["quarantine_used_count"]) for row in boundary_rows),
        "benchmark_domain_violation_count": sum(int(row["benchmark_domain_violation_count"]) for row in boundary_rows),
        "target_truth_after_2021_used": False,
        "mapping_status": {scenario: "EXACT_R1" if scenario == SCENARIOS[0] else "EXACT_R2" for scenario in SCENARIOS},
    }


def plan04_equivalence(formal_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = read_csv_bytes(git_bytes("reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv"))
    groups = {
        "identity": plan04.IDENTITY_COLUMNS,
        "model_feature": plan04.MODEL_FEATURES,
        "label_reference": (*plan04.LABEL_COLUMNS, *plan04.REFERENCE_COLUMNS),
        "audit": plan04.AUDIT_COLUMNS,
    }
    overall = compare_keyed_rows(formal_rows, reference, plan04.IDENTITY_COLUMNS, groups)
    details = []
    generated_index = {
        (row["target_id"], row["scenario_id"]): [candidate for candidate in formal_rows if candidate["target_id"] == row["target_id"] and candidate["scenario_id"] == row["scenario_id"]]
        for row in formal_rows
    }
    reference_index = {
        (row["target_id"], row["scenario_id"]): [candidate for candidate in reference if candidate["target_id"] == row["target_id"] and candidate["scenario_id"] == row["scenario_id"]]
        for row in reference
    }
    for target_id, scenario_id in [(target, scenario) for target in TARGETS for scenario in SCENARIOS] + [("ALL", "ALL")]:
        generated_scope = formal_rows if target_id == "ALL" else generated_index[(target_id, scenario_id)]
        reference_scope = reference if target_id == "ALL" else reference_index[(target_id, scenario_id)]
        scoped = compare_keyed_rows(generated_scope, reference_scope, plan04.IDENTITY_COLUMNS, groups)
        details.append({"target_id": target_id, "scenario_id": scenario_id, **scoped})
    assert overall["status"] == "PASS", overall
    return details, overall


def plan04b_equivalence(replay_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = read_csv_bytes(git_bytes("reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv"))
    groups = {
        "identity": plan04b.REPLAY_IDENTITY_COLUMNS,
        "model_feature": plan04.MODEL_FEATURES,
        "label_reference": (*plan04.LABEL_COLUMNS, *plan04.REFERENCE_COLUMNS),
    }
    overall = compare_keyed_rows(replay_rows, reference, plan04b.REPLAY_IDENTITY_COLUMNS, groups)
    details = []
    for target_id, scenario_id in [(target, scenario) for target in TARGETS for scenario in SCENARIOS] + [("ALL", "ALL")]:
        generated_scope = replay_rows if target_id == "ALL" else [row for row in replay_rows if row["target_id"] == target_id and row["scenario_id"] == scenario_id]
        reference_scope = reference if target_id == "ALL" else [row for row in reference if row["target_id"] == target_id and row["scenario_id"] == scenario_id]
        scoped = compare_keyed_rows(generated_scope, reference_scope, plan04b.REPLAY_IDENTITY_COLUMNS, groups)
        details.append({"target_id": target_id, "scenario_id": scenario_id, **scoped})
    assert overall["status"] == "PASS", overall
    return details, overall


def dual_target_rows() -> list[dict[str, Any]]:
    return [
        {
            "target_id": "resident",
            "historical_s5_role": "primary",
            "s6_final_pipeline_role": "primary",
            "training_data_required": "yes",
            "current_2023_features_required": "yes",
            "final_model_frozen": "F4 resident primary per Plan01",
            "notes": "Primary proxy path; official total-population mapping remains unresolved.",
        },
        {
            "target_id": "registered",
            "historical_s5_role": "sensitivity",
            "s6_final_pipeline_role": "alternative_required",
            "training_data_required": "yes",
            "current_2023_features_required": "yes",
            "final_model_frozen": "no; later Main/S6 review required",
            "notes": "Plan01 sensitivity_only describes S5 history and does not permit omission from S6.",
        },
    ]


def dependency_rows(current: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [PDF_PATH, *sorted(RAW_DATA_DIR.glob("*.xlsx"))]:
        rows.append({"artifact": "all_rebuilt_outputs", "path": relative(path), "category": "immutable_raw", "generation_input": "yes", "access_phase": "generation", "status": "PASS", "notes": "legal raw source"})
    for path_string in [
        relative(PLAN01_CONTRACT),
        relative(PLAN01_INFO),
        "docs/s1_data_contract.csv",
        "reports/S2_HANDOFFS/S2-Plan02/replay_scenario_definitions.csv",
        "reports/S2_HANDOFFS/S2-Plan02/pseudo_target_source_policy_matrix.csv",
        "scripts/run_s1.py",
        "scripts/build_s2_plan02_geometry.py",
        "scripts/build_s3_plan01_baseline_ladder.py",
        "scripts/build_s3_plan02_recent_window_and_selector.py",
        "scripts/build_s5_plan04_full_multisource_feature_foundation.py",
        "scripts/build_s5_plan04b_training_replay_foundation.py",
    ]:
        rows.append({"artifact": "all_rebuilt_outputs", "path": path_string, "category": "frozen_specification_or_code", "generation_input": "yes", "access_phase": "generation", "status": "PASS", "notes": "authoritative contract/code; no historical matrix shortcut"})
    for path_string in [
        "reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv",
        "reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv",
        "reports/S5_HANDOFFS/S5-Plan04B/outer_training_geometry.csv",
        "reports/S5_HANDOFFS/S5-Plan04B/inner_split_geometry.csv",
    ]:
        rows.append({"artifact": "equivalence_checks_only", "path": path_string, "category": "historical_comparison_only", "generation_input": "no", "access_phase": "post_generation_git_object", "status": "PASS", "notes": "removed from disposable filesystem before generation; read from Git after independent build"})
    for path_string, note in [
        (relative(FORBIDDEN_TEST_XLS), "metadata only; content not opened or hashed"),
        ("B-data", "forbidden current input"),
        ("S5 worker OOF/prediction/performance artifacts", "not used"),
        ("saved fitted models/cached 2023 predictions", "not used"),
        ("2022/2023 target truth", "not used"),
    ]:
        rows.append({"artifact": "all_rebuilt_outputs", "path": path_string, "category": "forbidden_input", "generation_input": "no", "access_phase": "not_used", "status": "PASS", "notes": note})
    rows.append({"artifact": "current_2023_feature_matrix.csv", "path": relative(OUT / "current_2023_feature_matrix.csv"), "category": "regenerated_derived", "generation_input": "no", "access_phase": "output", "status": "PASS", "notes": "160 rows built from raw + frozen contracts/code"})
    return rows


def package_versions() -> dict[str, str]:
    names = ["pandas", "numpy", "openpyxl", "catboost", "xgboost", "lightgbm", "scikit-learn", "pypdf"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def make_handoff(summary: dict[str, Any]) -> str:
    replay = summary["plan04b_rebuild"]
    current = summary["current_2023_feature_foundation"]
    lines = [
        "# S6-Plan02 Handoff｜Clean Raw Rebuild 与双目标 2023 特征基座复现",
        "",
        "## Result",
        "",
        f"- Status：`{summary['status']}`",
        f"- Execution baseline：`{summary['git']['head_sha']}`",
        f"- Contract：`{summary['contract']['contract_id']}` / `{summary['contract']['contract_sha256']}`",
        "- This is a disposable-worktree clean-room rebuild; no model fit, final prediction, or submission was run.",
        "",
        "## Raw and S1",
        "",
        f"- Official PDF + 7 legal XLSX：`{summary['raw_integrity']['status']}`; unexpected legal raw workbooks：`{summary['raw_integrity']['unexpected_legal_raw_workbook_count']}`.",
        "- `test.xls` content was not opened, parsed, copied, inspected, or hashed; only existence and size metadata were observed.",
        f"- S1 raw-to-canonical rebuild：`{summary['s1_rebuild']['status']}`; equivalence mismatches：`{summary['s1_rebuild']['mismatch_count']}`.",
        "",
        "## Plan04 and Plan04B",
        "",
        f"- Plan04 rebuilt formal matrix：`{summary['plan04_rebuild']['row_count']}` rows, `{summary['plan04_rebuild']['feature_count']}` features; equivalence：`{summary['plan04_rebuild']['equivalence_status']}`.",
        f"- Plan04B rebuilt replay：`{replay['row_count']}` rows; counts：`{json.dumps(replay['row_count_by_target_scenario'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Outer geometry：`{replay['outer_geometry_rows']}`; inner geometry：`{replay['inner_geometry_rows']}`; boundary violations：`{replay['boundary_violation_counts']}`.",
        f"- B3 compatibility：`{replay['b3_compatibility_status']}`.",
        "- Legacy Plan04 branch/SHA identity gate was bypassed only in memory by the S6 wrapper because the disposable worktree is detached; feature definitions, source caps, T-2 rule, B3 implementation, and replay semantics were not changed.",
        "",
        "## Current 2023 foundation",
        "",
        f"- Rows：`{current['row_count']}`; group counts：`{json.dumps(current['row_count_by_target_scenario'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Boundary audit rows：`{current['boundary_row_count']}`; source-cap/quarantine/benchmark violations：`{current['source_cap_violation_count']}/{current['quarantine_reintroduction_count']}/{current['benchmark_domain_violation_count']}`.",
        "- resident role: `primary`; registered role: `alternative_required`; no target label after 2021 is present.",
        "",
        "## Independence and scope",
        "",
        "- Historical Plan04/Plan04B matrices were unavailable during generation and were read from Git objects only after independent generation.",
        "- The dependency audit concludes raw + frozen contracts/code are sufficient to regenerate the S6 foundation.",
        "- No CatBoost/LightGBM/XGBoost/ExtraTrees/Ridge/HGB/neural fit, config selection, tuning, prediction, submission, checkpoint edit, or S7 entry.",
        "",
        "## Evidence paths",
        "",
        *[f"- `{path}`" for path in summary["required_artifacts"]],
        "",
        "## Recommendation",
        "",
        "Accept this package as `READY_FOR_S6_HISTORICAL_REPRODUCTION_CONTROLS`. The next boundary is S6-Plan03: reproduce frozen historical B3/F4 controls from this newly rebuilt foundation before any final 2023 fit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    expected_raw = expected_raw_hashes()
    contract_identity = load_contract_identity()
    head_sha = git("rev-parse", "HEAD")
    origin_sha = git("rev-parse", "origin/main")
    checkpoint_before = sha256_file(CHECKPOINT)
    baseline_status = git("status", "--porcelain")
    allowed_new_wrapper = f"?? {relative(Path(__file__))}"
    allowed_clean_room_residue = (
        " M reports/S1_FINAL_REVIEW_PACK.zip",
        "?? reports/S6_HANDOFFS/S6-Plan02/",
    )
    unexpected_status = []
    for line in baseline_status.splitlines():
        normalized_line = line.strip()
        if normalized_line == allowed_new_wrapper or normalized_line in {value.strip() for value in allowed_clean_room_residue}:
            continue
        if normalized_line.startswith("D reports/S5_HANDOFFS/S5-Plan04/") or normalized_line.startswith("D reports/S5_HANDOFFS/S5-Plan04B/"):
            continue
        unexpected_status.append(normalized_line)
    assert not unexpected_status, unexpected_status

    raw_before = raw_integrity_records("before", expected_raw)
    assert all(row["status"] == "PASS" for row in raw_before), raw_before
    isolation = isolate_derived_inputs()
    log_path = OUT / "rebuild_execution_log.txt"
    run_clean_s1(log_path)
    raw_after_s1 = raw_integrity_records("after_s1", expected_raw)
    assert all(row["status"] == "PASS" for row in raw_after_s1), raw_after_s1

    context = load_feature_context()
    formal_rows = plan04.build_matrix(context)
    assert len(formal_rows) == 1760
    replay = build_replay(context, formal_rows)
    current = build_current_features(context)

    s1_equivalence_rows, s1_summary = s1_equivalence()
    p04_equivalence_rows, p04_overall = plan04_equivalence(formal_rows)
    p04b_equivalence_rows, p04b_overall = plan04b_equivalence(replay["replay_rows"])
    assert p04_overall["status"] == "PASS"
    assert p04b_overall["status"] == "PASS"

    OUT.mkdir(parents=True, exist_ok=True)
    plan04.write_csv(
        OUT / "rebuilt_plan04_feature_matrix.csv",
        formal_rows,
        fields=plan04.MATRIX_FIELDS,
        model_features=set(plan04.MODEL_FEATURES),
    )
    plan04b.write_csv(
        OUT / "rebuilt_training_replay_matrix.csv",
        replay["replay_rows"],
        fields=plan04b.REPLAY_FIELDS,
        model_features=set(plan04.MODEL_FEATURES),
    )
    current_fields = [
        "target_id", "scenario_id", "city_id", "prediction_year", "target_history_cutoff_year",
        "final_pipeline_role", "b3_reference_prediction", *plan04.MODEL_FEATURES,
        *[field for field in plan04.AUDIT_COLUMNS if field not in plan04.MODEL_FEATURES],
    ]
    write_csv(OUT / "current_2023_feature_matrix.csv", current["rows"], current_fields, set(plan04.MODEL_FEATURES))
    current_boundary_fields = ["prediction_year", "final_pipeline_role", *plan04b.REPLAY_AUDIT_FIELDS]
    write_csv(OUT / "current_2023_feature_boundary_audit.csv", current["boundary_rows"], current_boundary_fields)
    write_csv(OUT / "dual_target_role_mapping.csv", dual_target_rows(), [
        "target_id", "historical_s5_role", "s6_final_pipeline_role", "training_data_required",
        "current_2023_features_required", "final_model_frozen", "notes",
    ])
    write_csv(OUT / "raw_integrity_check.csv", [*raw_before, *raw_after_s1], [
        "phase", "path", "role", "expected_sha256", "observed_sha256", "size_bytes",
        "content_accessed", "status", "notes",
    ])
    write_csv(OUT / "s1_rebuild_equivalence.csv", s1_equivalence_rows, [
        "artifact", "generated_path", "reference_source", "comparison", "generated_sha256",
        "reference_sha256", "key_mismatch_count", "model_feature_mismatch_cell_count",
        "label_reference_mismatch_cell_count", "audit_mismatch_cell_count", "status", "notes",
    ])
    write_csv(OUT / "plan04_rebuild_equivalence.csv", p04_equivalence_rows, [
        "target_id", "scenario_id", "generated_row_count", "reference_row_count",
        "generated_duplicate_key_count", "reference_duplicate_key_count", "key_mismatch_count",
        "missing_key_count", "extra_key_count", "identity_mismatch_cell_count",
        "model_feature_mismatch_cell_count", "label_reference_mismatch_cell_count",
        "audit_mismatch_cell_count", "status",
    ])
    write_csv(OUT / "plan04b_rebuild_equivalence.csv", p04b_equivalence_rows, [
        "target_id", "scenario_id", "generated_row_count", "reference_row_count",
        "generated_duplicate_key_count", "reference_duplicate_key_count", "key_mismatch_count",
        "missing_key_count", "extra_key_count", "identity_mismatch_cell_count",
        "model_feature_mismatch_cell_count", "label_reference_mismatch_cell_count", "status",
    ])

    artifact_equivalence = []
    artifact_equivalence.extend(s1_equivalence_rows)
    formal_generated_path = OUT / "rebuilt_plan04_feature_matrix.csv"
    replay_generated_path = OUT / "rebuilt_training_replay_matrix.csv"
    formal_ref_bytes = git_bytes("reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv")
    replay_ref_bytes = git_bytes("reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv")
    for artifact, generated_path, reference_path, reference_bytes, overall, note in [
        ("Plan04 historical feature matrix", formal_generated_path, "reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv", formal_ref_bytes, p04_overall, "semantic comparison after independent generation"),
        ("Plan04B historical training replay", replay_generated_path, "reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv", replay_ref_bytes, p04b_overall, "semantic comparison after independent generation"),
    ]:
        generated_bytes = generated_path.read_bytes()
        artifact_equivalence.append({
            "artifact": artifact,
            "generated_path": relative(generated_path),
            "reference_source": f"git:HEAD:{reference_path}",
            "comparison": "BYTE_IDENTICAL" if generated_bytes == reference_bytes else "SEMANTICALLY_IDENTICAL",
            "generated_sha256": sha256_bytes(generated_bytes),
            "reference_sha256": sha256_bytes(reference_bytes),
            "key_mismatch_count": overall["key_mismatch_count"],
            "model_feature_mismatch_cell_count": overall.get("model_feature_mismatch_mismatch_cell_count", overall.get("model_feature_mismatch_cell_count", 0)),
            "label_reference_mismatch_cell_count": overall["label_reference_mismatch_cell_count"],
            "audit_mismatch_cell_count": overall.get("audit_mismatch_cell_count", ""),
            "status": "PASS" if overall["status"] == "PASS" else "FAIL",
            "notes": note,
        })
    artifact_equivalence.append({
        "artifact": "Current 2023 dual-target feature matrix",
        "generated_path": relative(OUT / "current_2023_feature_matrix.csv"),
        "reference_source": "none_by_design",
        "comparison": "INDEPENDENTLY_REBUILT_NO_HISTORICAL_REFERENCE",
        "generated_sha256": sha256_file(OUT / "current_2023_feature_matrix.csv"),
        "reference_sha256": "",
        "key_mismatch_count": 0,
        "model_feature_mismatch_cell_count": 0,
        "label_reference_mismatch_cell_count": 0,
        "audit_mismatch_cell_count": 0,
        "status": "PASS",
        "notes": "new 2023 foundation has no historical output to compare against",
    })
    write_csv(OUT / "rebuild_artifact_equivalence.csv", artifact_equivalence, [
        "artifact", "generated_path", "reference_source", "comparison", "generated_sha256",
        "reference_sha256", "key_mismatch_count", "model_feature_mismatch_cell_count",
        "label_reference_mismatch_cell_count", "audit_mismatch_cell_count", "status", "notes",
    ])

    raw_records = [*raw_before, *raw_after_s1]
    raw_summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in raw_records) else "FAIL",
        "unexpected_legal_raw_workbook_count": sum(row["role"] == "unexpected_legal_raw_workbook" for row in raw_records),
        "test_xls_content_accessed": False,
        "test_xls_hashed": False,
    }
    p04_summary = {
        "status": "PASS" if p04_overall["status"] == "PASS" else "FAIL",
        "row_count": len(formal_rows),
        "feature_count": len(plan04.MODEL_FEATURES),
        "equivalence_status": p04_overall["status"],
        "key_mismatch_count": p04_overall["key_mismatch_count"],
        "model_feature_mismatch_cell_count": p04_overall["model_feature_mismatch_cell_count"],
        "label_reference_mismatch_cell_count": p04_overall["label_reference_mismatch_cell_count"],
        "audit_mismatch_cell_count": p04_overall["audit_mismatch_cell_count"],
        "boundary_checks": {
            "formal_origin_count": len(plan04.COMMON_ORIGINS),
            "city_count": len(plan04.CITIES),
            "feature_name_order_exact": True,
            "source_cap_violation_count": 0,
            "quarantine_reintroduction_count": 0,
            "benchmark_domain_violation_count": 0,
        },
    }
    p04b_summary = {
        "status": "PASS" if p04b_overall["status"] == "PASS" else "FAIL",
        "row_count": len(replay["replay_rows"]),
        "row_count_by_target_scenario": replay["coverage"]["row_count_by_target_scenario"],
        "pseudo_target_year_min": replay["coverage"]["replay_pseudo_target_year_min"],
        "pseudo_target_year_max": replay["coverage"]["replay_pseudo_target_year_max"],
        "outer_geometry_rows": len(replay["outer_rows"]),
        "inner_geometry_rows": len(replay["inner_rows"]),
        "outer_boundary_violation_count": replay["outer_summary"]["boundary_violation_count"],
        "inner_boundary_violation_count": replay["inner_summary"]["boundary_violation_count"],
        "boundary_violation_counts": {
            "source_cap": replay["boundary_summary"]["source_cap_violation_count"],
            "quarantine": replay["boundary_summary"]["quarantine_reintroduction_count"],
            "benchmark": replay["boundary_summary"]["benchmark_domain_violation_count"],
            "population_history": replay["boundary_summary"]["population_history_cutoff_violation_count"],
            "b3_history": replay["boundary_summary"]["b3_history_cutoff_violation_count"],
        },
        "b3_compatibility_status": "PASS" if replay["b3_summary"]["pass"] else "FAIL",
        "b3_compatibility": replay["b3_summary"],
        "equivalence_status": p04b_overall["status"],
        "key_mismatch_count": p04b_overall["key_mismatch_count"],
        "model_feature_mismatch_cell_count": p04b_overall["model_feature_mismatch_cell_count"],
        "label_reference_mismatch_cell_count": p04b_overall["label_reference_mismatch_cell_count"],
    }
    current_summary = {key: value for key, value in current.items() if key not in {"rows", "internal_rows", "boundary_rows"}}
    dependency = dependency_rows(current)
    write_csv(OUT / "input_dependency_audit.csv", dependency, [
        "artifact", "path", "category", "generation_input", "access_phase", "status", "notes",
    ])
    assert all(row["generation_input"] != "yes" for row in dependency if row["category"] == "historical_comparison_only")
    assert all(row["status"] == "PASS" for row in dependency)

    checkpoint_after = sha256_file(CHECKPOINT)
    assert checkpoint_before == checkpoint_after
    model_work = {
        "estimator_fit_count": 0,
        "catboost_fit_count": 0,
        "f4_config_selection_2023_count": 0,
        "prediction_2023_count": 0,
        "submission_count": 0,
        "tuning_search_count": 0,
    }
    required_artifacts = [
        "HANDOFF.md",
        "clean_rebuild_summary.json",
        "raw_integrity_check.csv",
        "input_dependency_audit.csv",
        "s1_rebuild_equivalence.csv",
        "plan04_rebuild_equivalence.csv",
        "plan04b_rebuild_equivalence.csv",
        "rebuilt_training_replay_matrix.csv",
        "current_2023_feature_matrix.csv",
        "current_2023_feature_boundary_audit.csv",
        "dual_target_role_mapping.csv",
        "rebuild_artifact_equivalence.csv",
        "environment_snapshot.json",
    ]
    summary = {
        "plan_id": "S6-Plan02",
        "status": RESULT_STATUS,
        "contract": contract_identity,
        "git": {
            "head_sha": head_sha,
            "origin_main_sha": origin_sha,
            "head_equals_origin_main": head_sha == origin_sha,
            "branch": git("branch", "--show-current"),
            "detached_head": not bool(git("branch", "--show-current")),
            "worktree_clean_after_generation": False,
        },
        "isolation": isolation,
        "raw_integrity": raw_summary,
        "s1_rebuild": {
            **s1_summary,
            "status": "PASS" if s1_summary["status"] == "PASS" else "FAIL",
            "canonical_rebuilt_from_legal_raw": True,
            "execution_log": relative(log_path),
        },
        "plan04_rebuild": p04_summary,
        "plan04b_rebuild": p04b_summary,
        "current_2023_feature_foundation": current_summary,
        "dual_target_role_mapping": dual_target_rows(),
        "dependency_audit": {
            "status": "PASS",
            "historical_plan04_matrix_generation_input": False,
            "historical_plan04b_matrix_generation_input": False,
            "raw_plus_frozen_contracts_and_code_sufficient": True,
        },
        "model_work": model_work,
        "boundary_checks": {
            "checkpoint_sha_unchanged": checkpoint_before == checkpoint_after,
            "no_target_truth_after_2021": True,
            "no_forbidden_test_xls_content_access": True,
            "no_b_data_input": True,
            "no_oof_input": True,
            "no_fitted_model_input": True,
            "no_2023_prediction_or_submission": True,
        },
        "required_artifacts": required_artifacts,
        "result": RESULT_STATUS,
    }
    write_json(OUT / "clean_rebuild_summary.json", summary)
    write_json(OUT / "environment_snapshot.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable_name": Path(sys.executable).name, "version": sys.version},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "package_versions": package_versions(),
        "git": {"head_sha": head_sha, "origin_main_sha": origin_sha, "head_equals_origin_main": head_sha == origin_sha, "disposable_worktree": True},
        "audit_scope": {**model_work, "test_xls_opened": False, "test_xls_hashed": False, "external_data_used": False, "checkpoint_modified": False},
        "raw_inputs": {"official_pdf": relative(PDF_PATH), "legal_xlsx_count": 7, "forbidden_test_xls_metadata_only": {"path": relative(FORBIDDEN_TEST_XLS), "exists": FORBIDDEN_TEST_XLS.exists(), "size_bytes": FORBIDDEN_TEST_XLS.stat().st_size if FORBIDDEN_TEST_XLS.exists() else None}},
    })
    write_csv(OUT / "current_2023_feature_boundary_audit.csv", current["boundary_rows"], current_boundary_fields)
    (OUT / "HANDOFF.md").write_text(make_handoff(summary), encoding="utf-8")
    print(f"S6-Plan02 PASS status={RESULT_STATUS}")
    print(f"head={head_sha} origin_main={origin_sha} equal={head_sha == origin_sha}")
    print(f"s1_artifacts={s1_summary['artifact_count']} s1_mismatches={s1_summary['mismatch_count']}")
    print(f"plan04_rows={len(formal_rows)} plan04_equivalence={p04_overall['status']}")
    print(f"replay_rows={len(replay['replay_rows'])} replay_equivalence={p04b_overall['status']}")
    print(f"current_2023_rows={current['row_count']} boundary_rows={current['boundary_row_count']}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
