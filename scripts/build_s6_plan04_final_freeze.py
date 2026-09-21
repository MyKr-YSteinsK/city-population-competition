"""Freeze the already-selected S6 resident F4-R1 solution.

This is a closeout/provenance runner only.  It does not import a modelling
library, fit a model, read held-out truth, or search for a candidate.  The
high-precision Plan03 artifact is recovered from its immutable Git commit
because a later, explicitly authorized formatting step changed the working
copy to two decimal places.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


PLAN_ID = "S6-Plan04"
STATUS = "FINAL_SOLUTION_FROZEN_FOR_MAIN_CLOSEOUT"
SOURCE_COMMIT = "e7040aff56eac45d3cbaadc8045ba446d9b9f02d"
SOURCE_RELATIVE_PATH = "reports/S6_HANDOFFS/S6-Plan03/submission_resident_f4_r1.csv"
PLAN03_FORMATTED_RELATIVE_PATH = "reports/S6_HANDOFFS/S6-Plan03/submission_resident_f4_r1.csv"
PLAN04_RELATIVE_DIR = "reports/S6_HANDOFFS/S6-Plan04"
HIGH_PRECISION_RELATIVE_PATH = f"{PLAN04_RELATIVE_DIR}/submission_resident_f4_r1_high_precision.csv"
FINAL_RELATIVE_PATH = f"{PLAN04_RELATIVE_DIR}/submission_final.csv"
REPORT_RELATIVE_PATH = "S6_FINAL_SOLUTION_REPORT.md"
MANIFEST_RELATIVE_PATH = f"{PLAN04_RELATIVE_DIR}/final_solution_manifest.json"
HANDOFF_RELATIVE_PATH = f"{PLAN04_RELATIVE_DIR}/HANDOFF.md"
EXPECTED_COLUMNS = ["city_id", "year", "pred"]
EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
EXPECTED_YEAR = 2023
TWOPLACES = Decimal("0.01")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git_show(repo_root: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{SOURCE_RELATIVE_PATH}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def read_rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {reader.fieldnames!r}")
    rows = list(reader)
    if any(row is None for row in rows):
        raise ValueError("Encountered an empty submission row")
    return rows


def validate_source_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 40:
        raise ValueError(f"Expected 40 source rows, got {len(rows)}")
    if [row["city_id"] for row in rows] != EXPECTED_CITY_IDS:
        raise ValueError("Source city order is not city1 through city40")
    if {int(row["year"]) for row in rows} != {EXPECTED_YEAR}:
        raise ValueError("Source year is not uniformly 2023")
    for row in rows:
        Decimal(row["pred"])


def round_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], Decimal]:
    output: list[dict[str, str]] = []
    max_delta = Decimal("0")
    for row in rows:
        original = Decimal(row["pred"])
        rounded = original.quantize(TWOPLACES, rounding=ROUND_HALF_UP)
        max_delta = max(max_delta, abs(original - rounded))
        output.append(
            {
                "city_id": row["city_id"],
                "year": row["year"],
                "pred": format(rounded, ".2f"),
            }
        )
    return output, max_delta


def write_rows(path: Path, rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    path.write_bytes(data)
    return data


def validate_final_rows(rows: list[dict[str, str]]) -> None:
    validate_source_rows(rows)
    if any(not re.fullmatch(r"-?\d+\.\d{2}", row["pred"]) for row in rows):
        raise ValueError("Final predictions are not formatted with exactly two decimals")


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan04_dir = repo_root / PLAN04_RELATIVE_DIR
    plan04_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = run_git_show(repo_root)
    source_rows = read_rows(source_bytes)
    validate_source_rows(source_rows)

    high_precision_path = repo_root / HIGH_PRECISION_RELATIVE_PATH
    high_precision_path.write_bytes(source_bytes)

    final_rows, max_rounding_delta = round_rows(source_rows)
    final_bytes = write_rows(repo_root / FINAL_RELATIVE_PATH, final_rows)
    validate_final_rows(read_rows(final_bytes))

    plan03_formatted_path = repo_root / PLAN03_FORMATTED_RELATIVE_PATH
    plan03_formatted_rows = read_rows(plan03_formatted_path.read_bytes())
    validate_final_rows(plan03_formatted_rows)
    if plan03_formatted_rows != final_rows:
        raise ValueError("Plan03 formatted R1 candidate differs from final submission")

    source_sha = sha256_bytes(source_bytes)
    final_sha = sha256_bytes(final_bytes)
    plan03_formatted_sha = sha256_bytes(plan03_formatted_path.read_bytes())
    manifest = {
        "plan_id": PLAN_ID,
        "status": STATUS,
        "final_solution": {
            "target": "resident",
            "framing": "F4_LOG_GROWTH_FROM_LATEST",
            "scenario": "R1_REFERENCE_SNAPSHOT_LAG",
            "model_config": "FRM_C3_FLEXIBLE_REGULARIZED",
            "seed": 20260919,
            "feature_count": 144,
            "training_rows": 573,
            "max_label_year": 2021,
            "prediction_year": 2023,
        },
        "artifacts": {
            "final_submission_path": FINAL_RELATIVE_PATH,
            "high_precision_source_path": HIGH_PRECISION_RELATIVE_PATH,
            "plan03_formatted_candidate_path": PLAN03_FORMATTED_RELATIVE_PATH,
            "report_path": REPORT_RELATIVE_PATH,
            "manifest_path": MANIFEST_RELATIVE_PATH,
            "handoff_path": HANDOFF_RELATIVE_PATH,
        },
        "submission_verification": {
            "columns": EXPECTED_COLUMNS,
            "row_count": len(final_rows),
            "city_order": "city1..city40",
            "year_values": [EXPECTED_YEAR],
            "prediction_format": "fixed_point_2_decimal",
            "rounding": "Decimal ROUND_HALF_UP",
            "max_absolute_rounding_delta": str(max_rounding_delta),
            "final_sha256": final_sha,
            "high_precision_source_sha256": source_sha,
            "plan03_formatted_candidate_sha256": plan03_formatted_sha,
            "plan03_formatted_candidate_matches_final": True,
        },
        "provenance": {
            "high_precision_source_commit": SOURCE_COMMIT,
            "high_precision_source_git_path": SOURCE_RELATIVE_PATH,
            "source_recovery_reason": (
                "The authorized S6-Plan03 formatting commit changed the working copy at the "
                "historical source path to two decimals; this exact high-precision blob was "
                "recovered from the immutable Plan03 commit without retraining."
            ),
            "original_plan03_working_copy_modified": True,
        },
        "evidence_boundaries": {
            "leaderboard_score_available": False,
            "official_total_population_mapping_externally_verified": False,
            "resident_target_frozen_by_project_decision": True,
            "full_historical_f4_replay": "deferred",
            "registered_city40_abnormal_drop_in_final_submission": False,
            "test_xls_accessed": False,
            "b_data_accessed": False,
            "external_2022_2023_truth_accessed": False,
        },
        "execution": {
            "model_fit_count_in_plan04": 0,
            "new_model_experiments": 0,
            "generated_by": "scripts/build_s6_plan04_final_freeze.py",
        },
    }
    json_dump(repo_root / MANIFEST_RELATIVE_PATH, manifest)

    report = f"""# S6 Final Solution Report

**Status:** `{STATUS}`

## Final frozen solution

The local S6 solution is frozen for Main closeout as a 2023 resident-population
submission for 40 cities:

- target: `resident`
- framing: `F4_LOG_GROWTH_FROM_LATEST`
- scenario: `R1_REFERENCE_SNAPSHOT_LAG`
- model configuration: `FRM_C3_FLEXIBLE_REGULARIZED`
- seed: `20260919`
- training rows: `573`
- maximum label year: `2021`
- prediction year: `2023`

No new fitting or candidate search was performed in Plan04. The final file is
the exact two-decimal, standard-rounded rendering of the previously selected
resident F4-R1 predictions. It has 40 rows, columns `city_id,year,pred`,
`city1` through `city40` in order, and year `2023` throughout.

## Data and contract boundary

Plan02 established the clean-raw foundation from the legal official PDF and
seven legal XLSX sources, with canonicalization and quarantine checks. The
final foundation uses the project T-2 information boundary, 144 features, and
the clean replay/training contract. `test.xls`, B-data, external sources, and
2022-2023 target truth were not read, copied, hashed, or used.

## Why F4-R1 was frozen

B3 remains the resident reference baseline (pooled historical MSE
`872.452651515143`). S4/S5 attempts with Ridge corrections, urbanization and
other external signals, low-capacity boosted trees, stronger model families,
hierarchical fitting, and city-local fitting did not establish a stable,
reproducible resident replacement. F4 was the strongest resident framing in
the completed evidence, with resident pooled MSE approximately `739.600665`
for R1 versus approximately `802.445747` for R2. R1 therefore has the better
historical scenario behavior.

Within the current S6 resident R1 configuration selection, C3 had inner
validation MSE `673.646095001205`, versus `688.5431302040054` for C1 and
`681.7136425379174` for C2. The selected C3 replay uses 573 training rows and
does not require any post-hoc clipping or manual prediction adjustment.

## Routes not adopted

The final solution does not use the registered target, registered F4
alternatives, Ridge/urbanization residual corrections, low-capacity GBDT,
stronger Plan06A-D model families, Plan06G partial pooling, Plan06H city-local
models, blending, routing, or gating. The registered B3/RLT3 exploratory
submissions also exposed an abnormal historical city40 drop and negative
extrapolated predictions; this was not repaired by clipping and is not part of
the final resident submission.

## Remaining boundaries and risks

- No leaderboard score was available before freeze.
- The official “总人口” semantic mapping was not externally verified; resident
  was intentionally frozen as the project decision.
- Full historical F4 replay was deferred after the fast first submission path.
- The final artifact is a local candidate; completion status remains with Main.
  The checkpoint remains unchanged and the next stage is not declared.

## Output artifacts

- Final submission: `{FINAL_RELATIVE_PATH}`
- High-precision provenance source: `{HIGH_PRECISION_RELATIVE_PATH}`
- Machine-readable manifest: `{MANIFEST_RELATIVE_PATH}`
- Main handoff: `{HANDOFF_RELATIVE_PATH}`

The high-precision provenance file is byte-recovered from Git commit
`{SOURCE_COMMIT}`. This is necessary because the separately authorized Plan03
formatting step changed the original working copy to two decimals; it does not
alter the Plan03 output or introduce a new model run. The maximum absolute
rounding delta in the final rendering is `{max_rounding_delta}`.
"""
    (repo_root / REPORT_RELATIVE_PATH).write_text(report, encoding="utf-8")

    handoff = f"""# S6-Plan04 Handoff — Final Freeze & Closeout

## Result

- Status: `{STATUS}`
- Final local solution: resident / F4 / R1 / CatBoost C3
- Final submission: `{FINAL_RELATIVE_PATH}`
- High-precision provenance: `{HIGH_PRECISION_RELATIVE_PATH}`
- Manifest: `{MANIFEST_RELATIVE_PATH}`
- Model fits in Plan04: `0`

The final local solution is frozen as resident F4-R1. There was no leaderboard
score before freeze. Per the closeout decision, no further optimization or
full reproduction was started; the deferred reproduction boundary is recorded
in the final report and manifest.

The earlier Plan03 two-decimal formatting commit changed the working copy at
the historical source path. For traceability, the exact high-precision R1
artifact was recovered from immutable commit `{SOURCE_COMMIT}` and used only
as provenance for the final rendering. The current Plan03 formatted candidate
and `submission_final.csv` contain the same values at two-decimal precision.

Registered alternatives are no longer selected. The incomplete or unsupported
families and the registered city40 anomaly remain explicitly documented; no
clipping, routing, ensemble, or manual correction was introduced.

## Main review request

Lab recommends that Main close the S6 implementation phase and decide the
project/learning-state handling. This handoff does not set a completed-stage
status, does not rewrite `CURRENT_LEARNING_CHECKPOINT.md`, and does not declare
the next stage.
"""
    (repo_root / HANDOFF_RELATIVE_PATH).write_text(handoff, encoding="utf-8")

    print(json.dumps({
        "status": STATUS,
        "final_submission": FINAL_RELATIVE_PATH,
        "high_precision_source": HIGH_PRECISION_RELATIVE_PATH,
        "rows": len(final_rows),
        "columns": EXPECTED_COLUMNS,
        "max_absolute_rounding_delta": str(max_rounding_delta),
        "model_fit_count": 0,
        "plan03_formatted_candidate_matches_final": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
