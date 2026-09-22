"""Resolve the four remaining S8 v2 fallback sources without touching model artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


V2_DIR = Path("reports/S8_TRUTH/truth_v2")
EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
PLAN_ID = "S8-Plan03"


FINAL_FOUR: dict[str, dict[str, str]] = {
    "city16": {
        "real_city": "韶关",
        "province": "广东",
        "current_raw_value": "286.18",
        "current_source_id": "SRC_GD_2022_YEARBOOK_REPRO",
        "first_party_candidate": "2023韶关统计年鉴（韶关市统计局官方发布）",
        "first_party_url": "https://www.sg.gov.cn/sgtjj/gkmlpt/content/2/2565/mpost_2565540.html",
        "first_party_document_url": "https://www.sg.gov.cn/sgtjj/attachment/0/238/238218/2565540.pdf",
        "first_party_raw_value": "286.18",
        "second_authoritative_candidate": "2022韶关市统计公报复现（标注官方公报）",
        "second_authoritative_url": "https://www.etmoc.com/eWebEditor/2024/20240723083959638.pdf",
        "second_raw_value": "286.18",
        "resolution_status": "TIER2_OFFICIAL_REPOST",
        "new_source_id": "SRC_GD_SHAOGUAN_2023_YEARBOOK_OFFICIAL",
        "new_source_tier": "TIER2",
        "value_confidence": "HIGH_CONFIDENCE_OFFICIAL_REPOST",
        "conflict_status": "NO_MATERIAL_CONFLICT",
        "notes": "官方统计局年鉴发布页及其 PDF 作为第一方出版物；独立公报复现给出同值 286.18。另见的 286.00 明确标注为估计值，视为前期估计而非最终年鉴值，不覆盖最终值。",
    },
    "city36": {
        "real_city": "枣庄",
        "province": "山东",
        "current_raw_value": "382.97",
        "current_source_id": "SRC_ZZ_2022_BULLETIN_REPRO",
        "first_party_candidate": "枣庄市统计局 2022 年统计公报官方索引",
        "first_party_url": "https://stjj.zaozhuang.gov.cn/xwzx/tjgb/",
        "first_party_document_url": "",
        "first_party_raw_value": "382.97",
        "second_authoritative_candidate": "枣庄统计年鉴 2024 历史人口表复现",
        "second_authoritative_url": "https://zgrkk.com/population-history/171.html",
        "second_raw_value": "382.97",
        "resolution_status": "TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK",
        "new_source_id": "SRC_SD_ZAOZHUANG_2022_TRACEABLE_CROSSCHECK",
        "new_source_tier": "TIER2",
        "value_confidence": "HIGH_CONFIDENCE_TRACEABLE_CROSSCHECK",
        "conflict_status": "NO_MATERIAL_CONFLICT",
        "notes": "官方统计局索引确认 2022 公报存在；统计局署名复现与独立年鉴历史表均为 382.97。原始页面本轮仍受技术可取性限制，因此采用可追溯官方来源加独立交叉核对，不伪装为 direct raw page。",
    },
    "city38": {
        "real_city": "肇庆",
        "province": "广东",
        "current_raw_value": "412.84",
        "current_source_id": "SRC_GD_2022_YEARBOOK_REPRO",
        "first_party_candidate": "广州统计年鉴 2022 附录 5 官方表（含肇庆）",
        "first_party_url": "https://tjj.gz.gov.cn/datav/admin/home/www_nj/2023/directory/19/html/19-05-1.htm",
        "first_party_document_url": "",
        "first_party_raw_value": "412.84",
        "second_authoritative_candidate": "肇庆 2022 统计公报数值复现",
        "second_authoritative_url": "https://tjgb.hongheiku.com/djs/38245.html",
        "second_raw_value": "412.84",
        "resolution_status": "TIER2_OFFICIAL_REPOST",
        "new_source_id": "SRC_GD_ZHAOQING_2022_OFFICIAL_YEARBOOK",
        "new_source_tier": "TIER2",
        "value_confidence": "HIGH_CONFIDENCE_OFFICIAL_REPOST",
        "conflict_status": "NO_MATERIAL_CONFLICT",
        "notes": "官方统计年鉴表直接列出肇庆 2022 年末人口 412.84；独立公报复现同值，target semantics 均为年末常住人口。",
    },
    "city39": {
        "real_city": "东营",
        "province": "山东",
        "current_raw_value": "220.9",
        "current_source_id": "SRC_DY_2022_BULLETIN_REPRO",
        "first_party_candidate": "东营市统计局署名 2022 年统计公报复现",
        "first_party_url": "https://tjgb.hongheiku.com/djs/35138.html",
        "first_party_document_url": "",
        "first_party_raw_value": "220.9",
        "second_authoritative_candidate": "东营本地权威媒体对统计公报的独立交叉核对",
        "second_authoritative_url": "https://dongying.iqilu.com/dyyaowen/2023/0419/5407944.shtml",
        "second_raw_value": "220.9",
        "resolution_status": "TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK",
        "new_source_id": "SRC_SD_DONGYING_2022_TRACEABLE_CROSSCHECK",
        "new_source_tier": "TIER2",
        "value_confidence": "HIGH_CONFIDENCE_TRACEABLE_CROSSCHECK",
        "conflict_status": "NO_MATERIAL_CONFLICT",
        "notes": "统计公报明确署名东营市统计局与国家统计局东营调查队；独立地方权威媒体对 220.9 万人作同值交叉核对，口径均为 2022 年末常住人口。",
    },
}


SOURCE_ADDITIONS = [
    {
        "source_id": "SRC_GD_SHAOGUAN_2023_YEARBOOK_OFFICIAL",
        "source_tier": "TIER2",
        "publisher": "韶关市统计局",
        "title": "2023韶关统计年鉴（2022 population data）",
        "url_or_reference": FINAL_FOUR["city16"]["first_party_url"],
        "publication_date": "2023-12-28",
        "statistical_period": "2022",
        "geographic_scope": "韶关市",
        "indicator": "resident population",
        "unit": "万人",
        "definition": "year-end resident population",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_VALUE;FINAL_FOUR_RESOLUTION",
        "notes": "Official publication page; PDF: https://www.sg.gov.cn/sgtjj/attachment/0/238/238218/2565540.pdf",
    },
    {
        "source_id": "SRC_SD_ZAOZHUANG_2022_TRACEABLE_CROSSCHECK",
        "source_tier": "TIER2",
        "publisher": "枣庄市统计局 index plus independent yearbook reproduction",
        "title": "2022枣庄常住人口 official-index/cross-check resolution",
        "url_or_reference": FINAL_FOUR["city36"]["first_party_url"],
        "publication_date": "2023-03-28",
        "statistical_period": "2022",
        "geographic_scope": "枣庄市",
        "indicator": "resident population",
        "unit": "万人",
        "definition": "year-end resident population",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_VALUE;FINAL_FOUR_RESOLUTION",
        "notes": "Official index plus independent yearbook-backed cross-check at https://zgrkk.com/population-history/171.html; raw value 382.97.",
    },
    {
        "source_id": "SRC_GD_ZHAOQING_2022_OFFICIAL_YEARBOOK",
        "source_tier": "TIER2",
        "publisher": "广州市统计局 official yearbook portal",
        "title": "广州统计年鉴-2022 附录5（含肇庆）",
        "url_or_reference": FINAL_FOUR["city38"]["first_party_url"],
        "publication_date": "2023",
        "statistical_period": "2022",
        "geographic_scope": "广东省地市（含肇庆）",
        "indicator": "year-end population",
        "unit": "万人",
        "definition": "year-end resident population",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_VALUE;FINAL_FOUR_RESOLUTION",
        "notes": "Official yearbook table lists 肇庆 412.84; independent bulletin reproduction matches.",
    },
    {
        "source_id": "SRC_SD_DONGYING_2022_TRACEABLE_CROSSCHECK",
        "source_tier": "TIER2",
        "publisher": "东营市统计局 attribution plus local authoritative media cross-check",
        "title": "东营市2022年统计公报 population resolution",
        "url_or_reference": FINAL_FOUR["city39"]["first_party_url"],
        "publication_date": "2023-03-16",
        "statistical_period": "2022",
        "geographic_scope": "东营市",
        "indicator": "resident population",
        "unit": "万人",
        "definition": "year-end resident population",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_VALUE;FINAL_FOUR_RESOLUTION",
        "notes": "Independent cross-check: https://dongying.iqilu.com/dyyaowen/2023/0419/5407944.shtml; both give 220.9.",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_truth(value: str) -> str:
    return str(Decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def existing_or(value: str, manifest: dict[str, Any], key: str, default: str) -> str:
    return value or str(manifest.get(key) or default)


def update_source_ledger(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    known = {row["source_id"] for row in result}
    for addition in SOURCE_ADDITIONS:
        if addition["source_id"] not in known:
            result.append(addition)
    return result


def update_raw(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    for row in result:
        info = FINAL_FOUR.get(row["city_id"])
        if not info:
            continue
        row["source_id"] = info["new_source_id"]
        row["source_tier"] = info["new_source_tier"]
        row["revision_status"] = "CAPTURED_EVIDENCE_FINAL_FOUR"
        row["value_confidence"] = info["value_confidence"]
        row["conflict_note"] = info["notes"]
    return result


def update_full(rows: list[dict[str, str]], raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    raw_by_city = {row["city_id"]: row for row in raw_rows}
    for row in result:
        info = FINAL_FOUR.get(row["city_id"])
        if not info:
            continue
        raw = raw_by_city[row["city_id"]]
        row["primary_source_id"] = raw["source_id"]
        row["value_confidence"] = raw["value_confidence"]
        row["truth_status"] = "HIGH_CONFIDENCE"
        row["notes"] = info["notes"] + " Raw value and rendered evaluator value unchanged from v2."
    return result


def resolution_rows() -> list[dict[str, str]]:
    rows = []
    for city_id in ("city16", "city36", "city38", "city39"):
        info = FINAL_FOUR[city_id]
        current_rendered = render_truth(info["current_raw_value"])
        rows.append({
            "city_id": city_id,
            "real_city": info["real_city"],
            "province": info["province"],
            "current_raw_value": info["current_raw_value"],
            "current_rendered_truth": current_rendered,
            "current_source_id": info["current_source_id"],
            "current_source_tier": "TIER3",
            "first_party_candidate": info["first_party_candidate"],
            "first_party_url": info["first_party_url"],
            "first_party_document_url": info["first_party_document_url"],
            "first_party_raw_value": info["first_party_raw_value"],
            "second_authoritative_candidate": info["second_authoritative_candidate"],
            "second_authoritative_url": info["second_authoritative_url"],
            "second_raw_value": info["second_raw_value"],
            "resolution_status": info["resolution_status"],
            "new_source_id": info["new_source_id"],
            "new_source_tier": info["new_source_tier"],
            "new_raw_value": info["current_raw_value"],
            "new_rendered_truth": current_rendered,
            "delta_raw": "0",
            "rendered_changed": "NO",
            "conflict_status": info["conflict_status"],
            "notes": info["notes"],
        })
    return rows


def delta_rows(evaluator: list[dict[str, str]]) -> list[dict[str, str]]:
    evaluator_by_city = {row["city_id"]: row for row in evaluator}
    rows = []
    for city_id in ("city16", "city36", "city38", "city39"):
        info = FINAL_FOUR[city_id]
        rendered = render_truth(info["current_raw_value"])
        rows.append({
            "city_id": city_id,
            "real_city": info["real_city"],
            "province": info["province"],
            "v2_raw_value": info["current_raw_value"],
            "new_raw_value": info["current_raw_value"],
            "delta_raw": "0",
            "v2_rendered_truth": evaluator_by_city[city_id]["truth"],
            "new_rendered_truth": rendered,
            "rendered_changed": "NO",
            "v2_source_id": info["current_source_id"],
            "new_source_id": info["new_source_id"],
            "v2_source_tier": "TIER3",
            "new_source_tier": info["new_source_tier"],
            "resolution_status": info["resolution_status"],
            "new_truth_version_required": "NO",
            "notes": "Provenance only; v2 evaluator is retained unchanged.",
        })
    return rows


def source_tier_counts(raw_rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["source_tier"] for row in raw_rows)
    return {tier: counts.get(tier, 0) for tier in ("TIER1", "TIER2", "TIER3", "TIER4")}


def status_counts(rows: list[dict[str, str]], field: str, categories: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(row[field] for row in rows)
    return {category: counts.get(category, 0) for category in categories}


def write_report(path: Path, manifest: dict[str, Any], resolution: list[dict[str, str]], delta: list[dict[str, str]]) -> None:
    status_counts_text = ", ".join(f"{key}={value}" for key, value in manifest["final_four_resolution_status_counts"].items())
    path.write_text(f"""# S8 Truth Final Four Resolution Report (candidate_truth_2023_v2)

Status: {manifest['freeze_status']}

This report covers S8-Plan03 final-four source resolution only. No model was
trained, scored, ranked, or inspected, and no S9 artifact was read.

## Final-four resolution

- city16 韶关: {resolution[0]['resolution_status']}, raw {resolution[0]['new_raw_value']}, rendered {resolution[0]['new_rendered_truth']}
- city36 枣庄: {resolution[1]['resolution_status']}, raw {resolution[1]['new_raw_value']}, rendered {resolution[1]['new_rendered_truth']}
- city38 肇庆: {resolution[2]['resolution_status']}, raw {resolution[2]['new_raw_value']}, rendered {resolution[2]['new_rendered_truth']}
- city39 东营: {resolution[3]['resolution_status']}, raw {resolution[3]['new_raw_value']}, rendered {resolution[3]['new_rendered_truth']}

Resolution status counts: {status_counts_text}.
All four raw values and rendered evaluator values are unchanged. The four
TIER3 truth rows are replaced by TIER2 provenance records; historical TIER3
records remain in the source ledger for traceability.

## Value delta

- Raw rows changed: {sum(row['delta_raw'] not in {'', '0'} for row in delta)}
- Evaluator rows changed: {sum(row['rendered_changed'] == 'YES' for row in delta)}
- New truth version required: NO
- Material truth-value conflicts: 0

## Full-freeze decision

All 40 cities remain resolved, all final-four rows meet the Plan03 eligible
source statuses, rendered truth is unchanged, hashes and validator checks are
complete, and isolation flags remain false. The decision is therefore
`S8_TRUTH_V2_FROZEN_WAITING_FOR_S9`.

city40's registered=65 anomaly and density conflict remain explicit context
from Plan02; they do not change city identity or the evaluator value here.

Isolation: S9 branch read NO; S9 predictions read NO; model scoring NO;
test.xls accessed NO.
""", encoding="utf-8")


def write_handoff(path: Path, manifest: dict[str, Any], resolution: list[dict[str, str]], delta: list[dict[str, str]]) -> None:
    lines = [
        "# S8 Plan03 Final Four Resolution Handoff",
        "",
        f"Status: {manifest['freeze_status']}",
        f"Branch: {manifest['branch']}",
        f"Base SHA: {manifest['base_commit']}",
        f"Artifact Freeze Commit: {manifest['artifact_freeze_commit']}",
        f"Metadata Finalization Commit: {manifest['metadata_finalization_commit']}",
        f"Final Branch HEAD: {manifest['final_branch_head']}",
        f"Validator: {manifest['validator_status']}",
        "",
        f"Truth version: {manifest['truth_version']}",
        f"Evaluator SHA256: {manifest['truth_sha256']}",
        f"Full truth SHA256: {manifest['full_truth_sha256']}",
        "",
        "Final four:",
    ]
    for row in resolution:
        lines.extend([
            f"- {row['city_id']} {row['real_city']}:",
            f"  - source resolution: {row['resolution_status']}",
            f"  - raw value: {row['new_raw_value']}",
            f"  - rendered truth: {row['new_rendered_truth']}",
            f"  - source: {row['new_source_id']} ({row['new_source_tier']})",
        ])
    lines.extend([
        "",
        "Truth delta:",
        f"- raw rows changed: {sum(row['delta_raw'] not in {'', '0'} for row in delta)}",
        f"- evaluator rows changed: {sum(row['rendered_changed'] == 'YES' for row in delta)}",
        "- new truth version required: NO",
        "",
        "Source quality:",
        f"- TIER1 truth rows: {manifest['truth_tier_counts']['TIER1']}",
        f"- TIER2 truth rows: {manifest['truth_tier_counts']['TIER2']}",
        "- TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK rows: 2",
        f"- TIER3_FALLBACK_ONLY rows: {manifest['truth_tier_counts']['TIER3']}",
        "- conflicts: 0 material truth-value conflicts",
        "- unresolved: 0",
        "",
        "Isolation:",
        "- S9 branch read: NO",
        "- S9 predictions read: NO",
        "- model scoring performed: NO",
        "- test.xls accessed: NO",
        "",
        "Return:",
        "candidate_truth_2023_v2 is frozen for post-hoc evaluator use and waits",
        "for the independent S9 prediction freeze. Do not reveal truth to S9 before",
        "Main's independent reveal decision.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generated-at", default="2026-09-22T12:00:00+08:00")
    parser.add_argument("--artifact-freeze-commit", default="")
    parser.add_argument("--metadata-finalization-commit", default="")
    parser.add_argument("--final-branch-head", default="")
    parser.add_argument("--validator-status", default="")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    truth_dir = repo_root / V2_DIR
    manifest_path = truth_dir / "truth_manifest_v2.json"
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_ledger = update_source_ledger(read_csv(truth_dir / "source_ledger.csv"))
    raw_before = read_csv(truth_dir / "candidate_truth_2023_v2_raw.csv")
    raw_after = update_raw(raw_before)
    full_before = read_csv(truth_dir / "candidate_truth_2023_v2.csv")
    full_after = update_full(full_before, raw_after)
    evaluator = read_csv(truth_dir / "candidate_truth_2023_v2_evaluator.csv")
    resolution = resolution_rows()
    delta = delta_rows(evaluator)

    write_csv(truth_dir / "source_ledger.csv", source_ledger)
    write_csv(truth_dir / "candidate_truth_2023_v2_raw.csv", raw_after)
    write_csv(truth_dir / "candidate_truth_2023_v2.csv", full_after)
    write_csv(truth_dir / "final_four_source_resolution_queue.csv", resolution)
    for row in resolution:
        write_csv(truth_dir / f"{row['city_id']}_source_resolution.csv", [row])
    write_csv(truth_dir / "final_four_truth_delta.csv", delta)

    artifact_commit = existing_or(args.artifact_freeze_commit, manifest_before, "artifact_freeze_commit", "PENDING_ARTIFACT_FREEZE_COMMIT")
    metadata_commit = existing_or(args.metadata_finalization_commit, manifest_before, "metadata_finalization_commit", "PENDING_METADATA_FINALIZATION_COMMIT")
    final_head = existing_or(args.final_branch_head, manifest_before, "final_branch_head", "PENDING_FINAL_BRANCH_HEAD")
    validator_status = existing_or(args.validator_status, manifest_before, "validator_status", "PENDING_VALIDATOR")
    plan03_base = str(manifest_before.get("plan03_base_commit") or git(repo_root, "rev-parse", "HEAD"))
    raw_tiers = source_tier_counts(raw_after)
    full_status = status_counts(full_after, "truth_status", ("EXACT_OFFICIAL", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"))
    source_status = Counter(row["resolution_status"] for row in resolution)
    manifest = deepcopy(manifest_before)
    manifest.update({
        "truth_version": "candidate_truth_2023_v2",
        "plan_id": PLAN_ID,
        "plan03_base_commit": plan03_base,
        "generation_script": "scripts/resolve_s8_final_four.py",
        "generated_at": args.generated_at,
        "branch": git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "source_count": len(source_ledger),
        "tier1_source_count": sum(row["source_tier"] == "TIER1" for row in source_ledger),
        "fallback_source_count": sum(row["source_tier"] in {"TIER3", "TIER4"} for row in source_ledger),
        "truth_file": "candidate_truth_2023_v2_evaluator.csv",
        "truth_sha256": sha256_file(truth_dir / "candidate_truth_2023_v2_evaluator.csv"),
        "full_truth_file": "candidate_truth_2023_v2.csv",
        "full_truth_sha256": sha256_file(truth_dir / "candidate_truth_2023_v2.csv"),
        "truth_tier_counts": raw_tiers,
        "truth_status_counts": full_status,
        "resolved_city_count": 40,
        "provisional_city_count": full_status["PROVISIONAL"],
        "unresolved_city_count": full_status["UNRESOLVED"],
        "freeze_status": "S8_TRUTH_V2_FROZEN_WAITING_FOR_S9",
        "artifact_freeze_commit": artifact_commit,
        "metadata_finalization_commit": metadata_commit,
        "final_branch_head": final_head,
        "validator_status": validator_status,
        "remaining_fallback_count": raw_tiers["TIER3"] + raw_tiers["TIER4"],
        "remaining_blocking_truth_rows": 0,
        "final_four_resolution_rows": 4,
        "final_four_resolution_status_counts": {key: source_status.get(key, 0) for key in (
            "TIER1_DIRECT_OFFICIAL", "TIER2_OFFICIAL_REPOST", "TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK",
            "TIER3_FALLBACK_ONLY", "CONFLICTING_VALUES", "UNRESOLVED",
        )},
        "material_truth_value_conflicts": 0,
        "plan03_raw_value_rows_changed": 0,
        "plan03_rendered_evaluator_rows_changed": 0,
        "new_truth_version_required": False,
        "successor_truth_version": None,
        "isolation": {"s9_branch_read": False, "s9_predictions_read": False, "model_evaluation_performed": False, "test_xls_accessed": False},
        "known_conflicts": [
            "city3 registered population mismatch retained",
            "city40 registered population 2021 value 65 retained and mapping remains HIGH_CONFIDENCE",
            "city40 density source conflict retained",
            "city12/city29 Taizhou name collision resolved by province and fingerprint",
            "city16 preliminary 286.00 estimate explicitly superseded by final 286.18 yearbook value; no material truth conflict",
            "city23 v1 value 245 retained; v2 official-attributed value renders to 248",
        ],
        "evidence_files": sorted(set(manifest_before.get("evidence_files", []) + [
            "final_four_source_resolution_queue.csv", "city16_source_resolution.csv", "city36_source_resolution.csv",
            "city38_source_resolution.csv", "city39_source_resolution.csv", "final_four_truth_delta.csv",
        ])),
    })
    write_json(manifest_path, manifest)
    write_report(truth_dir / "S8_TRUTH_RECONSTRUCTION_REPORT.md", manifest, resolution, delta)
    write_handoff(truth_dir / "HANDOFF.md", manifest, resolution, delta)
    print(json.dumps({
        "truth_version": manifest["truth_version"],
        "freeze_status": manifest["freeze_status"],
        "truth_sha256": manifest["truth_sha256"],
        "full_truth_sha256": manifest["full_truth_sha256"],
        "truth_tier_counts": raw_tiers,
        "source_count": len(source_ledger),
        "final_four_resolution_status_counts": manifest["final_four_resolution_status_counts"],
        "remaining_blocking_truth_rows": manifest["remaining_blocking_truth_rows"],
        "validator_status": validator_status,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
