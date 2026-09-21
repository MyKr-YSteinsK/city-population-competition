"""Read-only structural validator for the S8 truth reconstruction artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
REQUIRED_FILES = [
    "competition_city_fingerprints.csv",
    "source_ledger.csv",
    "city_identity_evidence.csv",
    "city_identity_mapping.csv",
    "year_alignment_evidence.csv",
    "year_alignment_summary.csv",
    "target_semantics_evidence.csv",
    "target_semantics_summary.json",
    "statistical_revision_ledger.csv",
    "candidate_truth_2023_raw.csv",
    "truth_conflicts.csv",
    "rounding_unit_evidence.csv",
    "candidate_truth_2023_v1.csv",
    "candidate_truth_2023_v1_evaluator.csv",
    "truth_manifest_v1.json",
    "S8_TRUTH_RECONSTRUCTION_REPORT.md",
    "HANDOFF.md",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", type=Path, default=Path("reports/S8_TRUTH/truth_v1"))
    args = parser.parse_args()
    truth_dir = args.truth_dir.resolve()

    missing = [name for name in REQUIRED_FILES if not (truth_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")

    manifest = json.loads((truth_dir / "truth_manifest_v1.json").read_text(encoding="utf-8"))
    evaluator = read_csv(truth_dir / manifest["truth_file"])
    full_truth = read_csv(truth_dir / manifest["full_truth_file"])
    mapping = read_csv(truth_dir / "city_identity_mapping.csv")
    identity_evidence = read_csv(truth_dir / "city_identity_evidence.csv")
    source_ledger = read_csv(truth_dir / "source_ledger.csv")
    raw_truth = read_csv(truth_dir / "candidate_truth_2023_raw.csv")

    evaluator_keys = [(row["city_id"], row["year"]) for row in evaluator]
    if [row["city_id"] for row in evaluator] != EXPECTED_CITY_IDS:
        raise RuntimeError("Evaluator city order is not city1..city40")
    if evaluator_keys != list(zip(EXPECTED_CITY_IDS, ["2023"] * 40)):
        raise RuntimeError("Evaluator keys are not unique 2023 rows")
    if any(not row.get("truth", "").strip() for row in evaluator):
        raise RuntimeError("Evaluator contains an empty truth")
    if [row["city_id"] for row in full_truth] != EXPECTED_CITY_IDS:
        raise RuntimeError("Full truth city order is not city1..city40")
    if [row["city_id"] for row in mapping] != EXPECTED_CITY_IDS:
        raise RuntimeError("Mapping city order is not city1..city40")
    evidence_city_ids = list(dict.fromkeys(row["city_id"] for row in identity_evidence))
    if evidence_city_ids != EXPECTED_CITY_IDS:
        raise RuntimeError("Identity evidence does not cover city1..city40 in order")
    if any(not row.get("positive_evidence_count", "").strip() for row in mapping):
        raise RuntimeError("Mapping row lacks positive evidence count")

    source_ids = {row["source_id"] for row in source_ledger}
    missing_truth_sources = sorted({row["source_id"] for row in raw_truth} - source_ids)
    if missing_truth_sources:
        raise RuntimeError(f"Truth source IDs missing from source ledger: {missing_truth_sources}")

    evaluator_sha = sha256_file(truth_dir / manifest["truth_file"])
    full_truth_sha = sha256_file(truth_dir / manifest["full_truth_file"])
    if evaluator_sha != manifest["truth_sha256"]:
        raise RuntimeError("Evaluator SHA256 does not match manifest")
    if full_truth_sha != manifest["full_truth_sha256"]:
        raise RuntimeError("Full truth SHA256 does not match manifest")
    if int(manifest["city_count"]) != 40 or len(evaluator) != 40 or len(raw_truth) != 40:
        raise RuntimeError("Truth coverage is not exactly 40 cities")
    if manifest["target_semantics"] != "RESIDENT_POPULATION":
        raise RuntimeError("Unexpected target semantics")
    if manifest["freeze_status"] != "S8_TRUTH_V1_PROVISIONAL_FREEZE":
        raise RuntimeError("Unexpected freeze status")
    if "POST_HOC_HIDDEN_EVALUATOR_ONLY" not in manifest["usage_scope"]:
        raise RuntimeError("Evaluator usage scope is not restricted")
    isolation = manifest["isolation"]
    if any(isolation.values()):
        raise RuntimeError(f"Isolation audit has an unexpected true flag: {isolation}")

    status_counts: dict[str, int] = {}
    for row in mapping:
        status = row["mapping_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    truth_status_counts: dict[str, int] = {}
    for row in full_truth:
        status = row["truth_status"]
        truth_status_counts[status] = truth_status_counts.get(status, 0) + 1

    print(json.dumps({
        "truth_dir": str(truth_dir),
        "required_file_count": len(REQUIRED_FILES),
        "city_count": len(evaluator),
        "truth_sha256": evaluator_sha,
        "mapping_status_counts": status_counts,
        "truth_status_counts": truth_status_counts,
        "source_count": len(source_ledger),
        "validation": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
