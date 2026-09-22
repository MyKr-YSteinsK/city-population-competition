"""Read-only structural and provenance validator for S8 truth artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
FINAL_FOUR_IDS = {"city16", "city36", "city38", "city39"}
FINAL_FOUR_ALLOWED_STATUSES = {
    "TIER1_DIRECT_OFFICIAL",
    "TIER2_OFFICIAL_REPOST",
    "TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK",
    "TIER3_FALLBACK_ONLY",
    "CONFLICTING_VALUES",
    "UNRESOLVED",
}
UPGRADE_CITY_IDS = {
    "city1", "city4", "city6", "city7", "city9", "city10", "city11", "city14",
    "city16", "city19", "city20", "city21", "city23", "city25", "city27", "city29",
    "city33", "city35", "city36", "city38", "city39",
}
HISTORICAL_SOURCE_IDS = {
    "SRC_ZJ_2022_POP_BULLETIN_REPRO",
    "SRC_GD_2022_YEARBOOK_REPRO",
    "SRC_ZZ_2022_BULLETIN_REPRO",
    "SRC_DY_2022_BULLETIN_REPRO",
}
ALLOWED_FREEZE_STATUSES = {
    "S8_TRUTH_V1_PROVISIONAL_FREEZE",
    "S8_TRUTH_PROVISIONAL_FREEZE_RETAINED",
    "S8_TRUTH_V1_FROZEN_WAITING_FOR_S9",
    "S8_TRUTH_V2_FROZEN_WAITING_FOR_S9",
}

COMMON_REQUIRED_FILES = [
    "competition_city_fingerprints.csv",
    "source_ledger.csv",
    "city_identity_evidence.csv",
    "city_identity_mapping.csv",
    "year_alignment_evidence.csv",
    "year_alignment_summary.csv",
    "target_semantics_evidence.csv",
    "target_semantics_summary.json",
    "statistical_revision_ledger.csv",
    "truth_conflicts.csv",
    "rounding_unit_evidence.csv",
    "truth_source_upgrade_queue.csv",
    "truth_v1_source_upgrade_delta.csv",
    "city40_identity_resolution.csv",
    "city40_registered_anomaly_audit.md",
    "city23_truth_resolution.csv",
    "S8_TRUTH_RECONSTRUCTION_REPORT.md",
    "HANDOFF.md",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decimal(value: str) -> Decimal:
    return Decimal(value)


def render_truth(value: str) -> str:
    return str(decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def ordered_city_rows(rows: list[dict[str, str]], label: str) -> None:
    city_ids = [row.get("city_id", "") for row in rows]
    if city_ids != EXPECTED_CITY_IDS:
        fail(f"{label} city order is not city1..city40")
    if len(set(city_ids)) != 40:
        fail(f"{label} contains duplicate city IDs")


def counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "") for row in rows).items()))


def counts_with_zero(rows: list[dict[str, str]], field: str, categories: set[str]) -> dict[str, int]:
    actual = counts(rows, field)
    return {category: actual.get(category, 0) for category in sorted(categories)}


def require_files(truth_dir: Path, version: str, manifest: dict[str, object]) -> list[str]:
    truth_file = str(manifest.get("truth_file", ""))
    full_truth_file = str(manifest.get("full_truth_file", ""))
    required = list(COMMON_REQUIRED_FILES) + [truth_file, full_truth_file]
    if version == "v1":
        required += ["candidate_truth_2023_raw.csv", "candidate_truth_2023_v1.csv", "candidate_truth_2023_v1_evaluator.csv", "truth_manifest_v1.json"]
    else:
        required += [
            "candidate_truth_2023_v2_raw.csv", "candidate_truth_2023_v2.csv", "candidate_truth_2023_v2_evaluator.csv",
            "truth_v1_to_v2_delta.csv", "truth_manifest_v2.json", "final_four_source_resolution_queue.csv",
            "city16_source_resolution.csv", "city36_source_resolution.csv", "city38_source_resolution.csv",
            "city39_source_resolution.csv", "final_four_truth_delta.csv",
        ]
    required = sorted(set(required))
    missing = [name for name in required if not (truth_dir / name).is_file()]
    if missing:
        fail(f"Missing required artifacts: {missing}")
    return required


def validate_source_ledger(raw_truth: list[dict[str, str]], source_ledger: list[dict[str, str]]) -> dict[str, int]:
    source_ids = [row.get("source_id", "") for row in source_ledger]
    if len(source_ids) != len(set(source_ids)):
        fail("Source ledger contains duplicate source IDs")
    if not HISTORICAL_SOURCE_IDS.issubset(set(source_ids)):
        fail("Historical reproduced sources were removed from source ledger")
    valid_tiers = {"TIER1", "TIER2", "TIER3", "TIER4", "TIER1_COMPETITION"}
    if any(row.get("source_tier") not in valid_tiers for row in source_ledger):
        fail("Source ledger contains an unknown source tier")
    ledger_by_id = {row["source_id"]: row for row in source_ledger}
    missing = sorted({row.get("source_id", "") for row in raw_truth} - set(ledger_by_id))
    if missing:
        fail(f"Truth source IDs missing from source ledger: {missing}")
    mismatches = [
        row["city_id"] for row in raw_truth
        if row.get("source_tier") != ledger_by_id[row["source_id"]].get("source_tier")
    ]
    if mismatches:
        fail(f"Raw truth/source ledger tier mismatch: {mismatches}")
    return {
        "source_count": len(source_ledger),
        "tier1_source_count": sum(row["source_tier"] == "TIER1" for row in source_ledger),
        "fallback_source_count": sum(row["source_tier"] in {"TIER3", "TIER4"} for row in source_ledger),
    }


def validate_truth_tables(
    evaluator: list[dict[str, str]],
    full_truth: list[dict[str, str]],
    raw_truth: list[dict[str, str]],
    mapping: list[dict[str, str]],
    manifest: dict[str, object],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    for label, rows in (("Evaluator", evaluator), ("Full truth", full_truth), ("Raw truth", raw_truth), ("Mapping", mapping)):
        if len(rows) != 40:
            fail(f"{label} coverage is not exactly 40 rows")
        ordered_city_rows(rows, label)
    if [(row["city_id"], row.get("year")) for row in evaluator] != [(city_id, "2023") for city_id in EXPECTED_CITY_IDS]:
        fail("Evaluator keys are not unique 2023 rows")
    if any(not row.get("truth", "").strip() for row in evaluator):
        fail("Evaluator contains an empty truth")
    mapping_by_city = {row["city_id"]: row for row in mapping}
    full_by_city = {row["city_id"]: row for row in full_truth}
    raw_by_city = {row["city_id"]: row for row in raw_truth}
    evaluator_by_city = {row["city_id"]: row for row in evaluator}
    for city_id in EXPECTED_CITY_IDS:
        raw = raw_by_city[city_id]
        full = full_by_city[city_id]
        mapping_row = mapping_by_city[city_id]
        evaluator_row = evaluator_by_city[city_id]
        if raw.get("target_semantics") != "resident_population":
            fail(f"Unexpected raw target semantics for {city_id}")
        if full.get("primary_source_id") != raw.get("source_id"):
            fail(f"Full truth primary source does not match raw source for {city_id}")
        if full.get("candidate_truth") != render_truth(raw["raw_value"]):
            fail(f"Full truth rendered value mismatch for {city_id}")
        if evaluator_row.get("truth") != full.get("candidate_truth"):
            fail(f"Evaluator/full truth mismatch for {city_id}")
        if full.get("mapping_confidence") != mapping_row.get("mapping_confidence"):
            fail(f"Mapping confidence mismatch for {city_id}")
    raw_tier_counts = counts_with_zero(raw_truth, "source_tier", {"TIER1", "TIER2", "TIER3", "TIER4"})
    full_status_counts = counts_with_zero(full_truth, "truth_status", {"EXACT_OFFICIAL", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"})
    mapping_status_counts = counts_with_zero(mapping, "mapping_status", {"CONFIRMED", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"})
    if manifest.get("truth_tier_counts") != raw_tier_counts:
        fail(f"Manifest truth tier counts mismatch: {manifest.get('truth_tier_counts')} != {raw_tier_counts}")
    if manifest.get("truth_status_counts") != full_status_counts:
        fail(f"Manifest truth status counts mismatch: {manifest.get('truth_status_counts')} != {full_status_counts}")
    if manifest.get("mapping_status_counts") != mapping_status_counts:
        fail(f"Manifest mapping status counts mismatch: {manifest.get('mapping_status_counts')} != {mapping_status_counts}")
    if int(manifest.get("city_count", -1)) != 40:
        fail("Manifest city_count is not 40")
    if int(manifest.get("resolved_city_count", -1)) != 40 - full_status_counts.get("UNRESOLVED", 0):
        fail("Manifest resolved_city_count mismatch")
    if int(manifest.get("provisional_city_count", -1)) != full_status_counts.get("PROVISIONAL", 0):
        fail("Manifest provisional_city_count mismatch")
    if int(manifest.get("unresolved_city_count", -1)) != full_status_counts.get("UNRESOLVED", 0):
        fail("Manifest unresolved_city_count mismatch")
    return raw_tier_counts, full_status_counts, mapping_status_counts


def validate_upgrade_artifacts(truth_dir: Path, version: str, raw_truth: list[dict[str, str]], evaluator: list[dict[str, str]]) -> None:
    queue = read_csv(truth_dir / "truth_source_upgrade_queue.csv")
    if {row.get("city_id") for row in queue} != UPGRADE_CITY_IDS or len(queue) != 21:
        fail("Truth-source upgrade queue is not the expected 21 reproduced rows")
    if any(not row.get("upgrade_status", "").strip() for row in queue):
        fail("Truth-source upgrade queue contains an empty status")
    delta = read_csv(truth_dir / "truth_v1_source_upgrade_delta.csv")
    ordered_city_rows(delta, "Truth v1 source-upgrade delta")
    raw_by_city = {row["city_id"]: row for row in raw_truth}
    for row in delta:
        if version == "v1" and row.get("hardened_source_id") != raw_by_city[row["city_id"]].get("source_id"):
            fail(f"Source-upgrade delta does not match v1 raw source for {row['city_id']}")
    resolution = read_csv(truth_dir / "city40_identity_resolution.csv")
    if len(resolution) != 1 or resolution[0].get("status_after") != "HIGH_CONFIDENCE":
        fail("city40 identity resolution did not reach HIGH_CONFIDENCE")
    if "ANOMALY_CAUSE_UNRESOLVED" not in resolution[0].get("registered_65_interpretation", ""):
        fail("city40 anomaly cause is not explicitly unresolved")
    audit = (truth_dir / "city40_registered_anomaly_audit.md").read_text(encoding="utf-8")
    for marker in ("ANOMALY_CAUSE_UNRESOLVED", "test.xls", "No competition source value is corrected"):
        if marker not in audit:
            fail(f"city40 audit missing marker: {marker}")
    city23 = read_csv(truth_dir / "city23_truth_resolution.csv")
    if len(city23) != 1 or city23[0].get("official_2022_value") != "247.72" or city23[0].get("old_chat_value_ignored") != "248":
        fail("city23 source/version resolution is incomplete")
    if version == "v1":
        if next(row["truth"] for row in evaluator if row["city_id"] == "city23") != "245":
            fail("v1 city23 evaluator value changed unexpectedly")
    else:
        if next(row["truth"] for row in evaluator if row["city_id"] == "city23") != "248":
            fail("v2 city23 evaluator value is not 248")
        v2_delta = read_csv(truth_dir / "truth_v1_to_v2_delta.csv")
        ordered_city_rows(v2_delta, "Truth v1-to-v2 delta")
        changed = [row for row in v2_delta if row.get("rendered_changed") == "YES"]
        if len(changed) != 1 or changed[0].get("city_id") != "city23" or changed[0].get("delta_raw") != "2.72":
            fail("v2 delta is not an exact city23-only rendered change")
        if sum(row.get("delta_raw") not in {"", "0", "0.0", "0.00"} for row in v2_delta) != 1:
            fail("v2 delta has unexpected raw changes")


def validate_final_four_artifacts(
    truth_dir: Path,
    raw_truth: list[dict[str, str]],
    evaluator: list[dict[str, str]],
    source_ledger: list[dict[str, str]],
    manifest: dict[str, object],
) -> None:
    queue = read_csv(truth_dir / "final_four_source_resolution_queue.csv")
    if {row.get("city_id") for row in queue} != FINAL_FOUR_IDS or len(queue) != 4:
        fail("Final-four resolution queue is not exactly city16/city36/city38/city39")
    queue_by_city = {row["city_id"]: row for row in queue}
    raw_by_city = {row["city_id"]: row for row in raw_truth}
    evaluator_by_city = {row["city_id"]: row for row in evaluator}
    ledger_by_id = {row["source_id"]: row for row in source_ledger}
    for city_id in sorted(FINAL_FOUR_IDS):
        row = queue_by_city[city_id]
        if row.get("resolution_status") not in FINAL_FOUR_ALLOWED_STATUSES:
            fail(f"Invalid final-four status for {city_id}: {row.get('resolution_status')}")
        if row.get("new_source_tier") not in {"TIER1", "TIER2"}:
            fail(f"Final-four source remains fallback for {city_id}")
        if row.get("new_raw_value") != row.get("current_raw_value") or row.get("delta_raw") not in {"0", "0.0", "0.00"}:
            fail(f"Final-four raw value changed unexpectedly for {city_id}")
        if row.get("rendered_changed") != "NO" or row.get("new_rendered_truth") != evaluator_by_city[city_id].get("truth"):
            fail(f"Final-four evaluator delta mismatch for {city_id}")
        raw = raw_by_city[city_id]
        if raw.get("source_id") != row.get("new_source_id") or raw.get("source_tier") != row.get("new_source_tier"):
            fail(f"Final-four raw/source resolution mismatch for {city_id}")
        if row.get("new_source_id") not in ledger_by_id:
            fail(f"Final-four source missing from ledger for {city_id}")
        resolution_file = read_csv(truth_dir / f"{city_id}_source_resolution.csv")
        if len(resolution_file) != 1 or resolution_file[0] != row:
            fail(f"Per-city final-four resolution does not match queue for {city_id}")
    delta = read_csv(truth_dir / "final_four_truth_delta.csv")
    if {row.get("city_id") for row in delta} != FINAL_FOUR_IDS or len(delta) != 4:
        fail("Final-four truth delta is not exactly four rows")
    if any(row.get("rendered_changed") != "NO" or row.get("new_truth_version_required") != "NO" for row in delta):
        fail("Final-four delta indicates an unexpected evaluator/version change")
    status_counts = Counter(row["resolution_status"] for row in queue)
    expected_status_counts = {status: status_counts.get(status, 0) for status in sorted(FINAL_FOUR_ALLOWED_STATUSES)}
    if manifest.get("final_four_resolution_status_counts") != expected_status_counts:
        fail("Manifest final-four resolution status counts mismatch")
    blocking = sum(status_counts.get(status, 0) for status in {"TIER3_FALLBACK_ONLY", "CONFLICTING_VALUES", "UNRESOLVED"})
    if int(manifest.get("remaining_blocking_truth_rows", -1)) != blocking:
        fail("Manifest remaining_blocking_truth_rows mismatch")
    if int(manifest.get("remaining_fallback_count", -1)) != 0:
        fail("Manifest remaining_fallback_count is not zero")
    if int(manifest.get("material_truth_value_conflicts", -1)) != 0:
        fail("Manifest material_truth_value_conflicts is not zero")
    if int(manifest.get("plan03_raw_value_rows_changed", -1)) != 0 or int(manifest.get("plan03_rendered_evaluator_rows_changed", -1)) != 0:
        fail("Plan03 reports unexpected value/evaluator changes")
    if manifest.get("new_truth_version_required") is not False or manifest.get("successor_truth_version") is not None:
        fail("Plan03 versioning decision is inconsistent with unchanged evaluator values")
    handoff = (truth_dir / "HANDOFF.md").read_text(encoding="utf-8")
    for marker in ("Artifact Freeze Commit:", "Metadata Finalization Commit:", "Final Branch HEAD:"):
        if marker not in handoff or "PENDING" in handoff.split(marker, 1)[1].splitlines()[0]:
            fail(f"Final handoff metadata is missing or pending: {marker}")


def validate_version_linkage(truth_dir: Path, version: str, manifest: dict[str, object], evaluator_path: Path, full_path: Path) -> None:
    evaluator_sha = sha256_file(evaluator_path)
    full_sha = sha256_file(full_path)
    if evaluator_sha != manifest.get("truth_sha256"):
        fail("Evaluator SHA256 does not match manifest")
    if full_sha != manifest.get("full_truth_sha256"):
        fail("Full truth SHA256 does not match manifest")
    if version == "v1":
        if manifest.get("freeze_status") != "S8_TRUTH_V1_PROVISIONAL_FREEZE":
            fail("Unexpected v1 freeze status")
        if manifest.get("successor_truth_version") != "candidate_truth_2023_v2":
            fail("v1 successor version is not recorded")
        if manifest.get("v1_evaluator_sha256") != evaluator_sha:
            fail("v1 evaluator self-hash is inconsistent")
    else:
        sibling = truth_dir.parent / "truth_v1"
        if not sibling.is_dir():
            fail("v2 validation requires sibling truth_v1 artifacts")
        v1_manifest = json.loads((sibling / "truth_manifest_v1.json").read_text(encoding="utf-8"))
        if manifest.get("freeze_status") not in {"S8_TRUTH_PROVISIONAL_FREEZE_RETAINED", "S8_TRUTH_V2_FROZEN_WAITING_FOR_S9"}:
            fail("Unexpected v2 freeze status")
        if manifest.get("supersedes_truth_version") != "candidate_truth_2023_v1":
            fail("v2 supersedes version is not recorded")
        if manifest.get("v1_evaluator_sha256") != v1_manifest.get("truth_sha256"):
            fail("v2 does not link to hardened v1 evaluator SHA")
        if manifest.get("v1_full_truth_sha256_after_hardening") != v1_manifest.get("full_truth_sha256"):
            fail("v2 does not link to hardened v1 full-truth SHA")
        if int(manifest.get("rendered_evaluator_rows_changed", -1)) != 1 or int(manifest.get("raw_value_rows_changed", -1)) != 1:
            fail("v2 manifest change counts are not exactly one")
        if manifest.get("freeze_status") == "S8_TRUTH_V2_FROZEN_WAITING_FOR_S9":
            if manifest.get("validator_status") != "PASS":
                fail("Full-freeze manifest validator_status is not PASS")
            if int(manifest.get("remaining_blocking_truth_rows", -1)) != 0:
                fail("Full-freeze manifest still reports blocking rows")
            for key in ("artifact_freeze_commit", "metadata_finalization_commit", "final_branch_head"):
                value = str(manifest.get(key, ""))
                if not value or "PENDING" in value:
                    fail(f"Full-freeze manifest metadata is missing: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", type=Path, default=Path("reports/S8_TRUTH/truth_v1"))
    args = parser.parse_args()
    truth_dir = args.truth_dir.resolve()
    version = "v2" if truth_dir.name == "truth_v2" else "v1"
    manifest_name = f"truth_manifest_{version}.json"
    manifest = json.loads((truth_dir / manifest_name).read_text(encoding="utf-8"))
    required_files = require_files(truth_dir, version, manifest)
    evaluator = read_csv(truth_dir / str(manifest["truth_file"]))
    full_truth = read_csv(truth_dir / str(manifest["full_truth_file"]))
    raw_name = "candidate_truth_2023_v2_raw.csv" if version == "v2" else "candidate_truth_2023_raw.csv"
    raw_truth = read_csv(truth_dir / raw_name)
    mapping = read_csv(truth_dir / "city_identity_mapping.csv")
    identity_evidence = read_csv(truth_dir / "city_identity_evidence.csv")
    source_ledger = read_csv(truth_dir / "source_ledger.csv")
    if list(dict.fromkeys(row.get("city_id", "") for row in identity_evidence)) != EXPECTED_CITY_IDS:
        fail("Identity evidence does not cover city1..city40 in order")
    if any(not row.get("positive_evidence_count", "").strip() for row in mapping):
        fail("Mapping row lacks positive evidence count")
    raw_tier_counts, truth_status_counts, mapping_status_counts = validate_truth_tables(evaluator, full_truth, raw_truth, mapping, manifest)
    ledger_counts = validate_source_ledger(raw_truth, source_ledger)
    for field, actual in ledger_counts.items():
        if int(manifest.get(field, -1)) != actual:
            fail(f"Manifest {field} mismatch: {manifest.get(field)} != {actual}")
    validate_upgrade_artifacts(truth_dir, version, raw_truth, evaluator)
    if version == "v2":
        validate_final_four_artifacts(truth_dir, raw_truth, evaluator, source_ledger, manifest)
    validate_version_linkage(truth_dir, version, manifest, truth_dir / str(manifest["truth_file"]), truth_dir / str(manifest["full_truth_file"]))
    if manifest.get("target_semantics") != "RESIDENT_POPULATION":
        fail("Unexpected target semantics")
    if manifest.get("usage_scope") is None or "POST_HOC_HIDDEN_EVALUATOR_ONLY" not in manifest["usage_scope"]:
        fail("Evaluator usage scope is not restricted")
    if any(manifest.get("isolation", {}).values()):
        fail(f"Isolation audit has an unexpected true flag: {manifest.get('isolation')}")
    if manifest.get("freeze_status") not in ALLOWED_FREEZE_STATUSES:
        fail(f"Unexpected freeze status: {manifest.get('freeze_status')}")
    if manifest.get("city40_mapping_status") != "HIGH_CONFIDENCE":
        fail("city40 mapping is not HIGH_CONFIDENCE")
    if truth_status_counts.get("UNRESOLVED", 0) != 0 or truth_status_counts.get("PROVISIONAL", 0) != 0:
        fail("Truth table still contains unresolved/provisional rows")
    print(json.dumps({
        "truth_dir": str(truth_dir),
        "version": version,
        "required_file_count": len(required_files),
        "city_count": len(evaluator),
        "truth_sha256": sha256_file(truth_dir / str(manifest["truth_file"])),
        "full_truth_sha256": sha256_file(truth_dir / str(manifest["full_truth_file"])),
        "truth_tier_counts": raw_tier_counts,
        "mapping_status_counts": mapping_status_counts,
        "truth_status_counts": truth_status_counts,
        "source_count": len(source_ledger),
        "validation": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
