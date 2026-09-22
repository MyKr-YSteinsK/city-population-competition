"""Harden S8 truth provenance and produce a versioned v2 when needed.

This script only reads competition data and statistical-source evidence. It
does not read S9 artifacts, train models, score predictions, or inspect
test.xls. The v1 evaluator is preserved when a rendered value changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
PLAN_ID = "S8-Plan02"
V1_DIR = Path("reports/S8_TRUTH/truth_v1")
V2_DIR = Path("reports/S8_TRUTH/truth_v2")
RAW_SOURCE_BY_CITY = {
    **{city_id: "SRC_ZJ_2022_POP_BULLETIN_REPRO" for city_id in ["city1", "city4", "city7", "city10", "city11", "city19", "city20", "city21", "city25", "city29", "city33"]},
    **{city_id: "SRC_GD_2022_YEARBOOK_REPRO" for city_id in ["city6", "city9", "city14", "city16", "city23", "city27", "city35", "city38"]},
    "city36": "SRC_ZZ_2022_BULLETIN_REPRO",
    "city39": "SRC_DY_2022_BULLETIN_REPRO",
}

SOURCE_ADDITIONS = [
    {
        "source_id": "SRC_ZJ_2022_POP_BULLETIN_GOV_REPOST",
        "source_tier": "TIER2",
        "publisher": "绍兴市人民政府门户网站转载浙江省统计局公报",
        "title": "2022年浙江省人口主要数据公布",
        "url_or_reference": "https://www.sx.gov.cn/art/2023/2/23/art_1462940_59443513.html",
        "publication_date": "2023-02-23",
        "statistical_period": "2022",
        "geographic_scope": "浙江省11设区市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口；官方公报表格的政府门户转载",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE;CITY23_CROSS_CHECK",
        "notes": "supersedes_source_id=SRC_ZJ_2022_POP_BULLETIN_REPRO; verification_role=official-government-repost-cross-check; source image/table values are retained without silent correction.",
    },
    {
        "source_id": "SRC_GD_HEYUAN_2022_BULLETIN",
        "source_tier": "TIER1",
        "publisher": "河源市人民政府/河源市统计局",
        "title": "2022年河源市国民经济和社会发展统计公报",
        "url_or_reference": "https://www.heyuan.gov.cn/zwgk/tjsj/tjnb/content/post_545677.html",
        "publication_date": "2023-04-04",
        "statistical_period": "2022",
        "geographic_scope": "河源市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=first-party-city-bulletin.",
    },
    {
        "source_id": "SRC_GD_SHANTOU_2022_BULLETIN",
        "source_tier": "TIER1",
        "publisher": "汕头市统计局",
        "title": "2022年汕头国民经济和社会发展统计公报",
        "url_or_reference": "https://www.shantou.gov.cn/tjj/tjzl/content/post_2203963.html",
        "publication_date": "2023-03-31",
        "statistical_period": "2022",
        "geographic_scope": "汕头市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=first-party-city-bulletin.",
    },
    {
        "source_id": "SRC_GD_ZHANJIANG_2022_BULLETIN",
        "source_tier": "TIER1",
        "publisher": "湛江市统计局",
        "title": "2022年湛江市常住人口情况",
        "url_or_reference": "https://www.zhanjiang.gov.cn/zjsfw/bmdh/tjxxw/zwgk/tjsjzl/tjgb/content/post_1757222.html",
        "publication_date": "2023-04-26",
        "statistical_period": "2022",
        "geographic_scope": "湛江市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=first-party-city-statistics-page.",
    },
    {
        "source_id": "SRC_GD_FOSHAN_2022_OFFICIAL_MEDIA",
        "source_tier": "TIER2",
        "publisher": "佛山日报/佛山市统计局",
        "title": "2022年佛山市国民经济和社会发展统计公报",
        "url_or_reference": "https://epaper1.fsonline.com.cn/fsrb/resfile/2023-04-16/A03/fsrb20230416A03.pdf",
        "publication_date": "2023-04-16",
        "statistical_period": "2022",
        "geographic_scope": "佛山市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口；市统计局公报的官方地方媒体刊载",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=official-local-media-repost; not promoted to TIER1 because the statistics-bureau original page was not directly retrieved.",
    },
    {
        "source_id": "SRC_GD_MAOMING_2022_OFFICIAL_MEDIA",
        "source_tier": "TIER2",
        "publisher": "茂名日报/茂名市统计局",
        "title": "2022年茂名市国民经济和社会发展统计公报",
        "url_or_reference": "https://www.mm111.net/mpaper/2023-04/04/content_991311633.html",
        "publication_date": "2023-04-04",
        "statistical_period": "2022",
        "geographic_scope": "茂名市",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口；市统计局公报的官方地方媒体刊载",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=official-local-media-repost; not promoted to TIER1 because the statistics-bureau original page was not directly retrieved.",
    },
    {
        "source_id": "SRC_GD_ZHUHAI_2022_OFFICIAL_MEDIA",
        "source_tier": "TIER2",
        "publisher": "珠海特区报/珠海市统计局",
        "title": "247.72万人！珠海2022年末常住人口数据公布",
        "url_or_reference": "https://pub-zhtb.hizh.cn/a/202304/10/AP6433bdb3e4b07175af433fcd.html",
        "publication_date": "2023-04-10",
        "statistical_period": "2022",
        "geographic_scope": "珠海市（含横琴粤澳深度合作区）",
        "indicator": "年末常住人口",
        "unit": "万人",
        "definition": "常住人口；报道明确署名珠海市统计局发布",
        "revision_status": "CAPTURED_EVIDENCE",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE;CITY23_RESOLUTION",
        "notes": "supersedes_source_id=SRC_GD_2022_YEARBOOK_REPRO; verification_role=official-statistics-attributed-media-repost; raw value 247.72 differs from v1 245 and therefore requires truth_v2.",
    },
    {
        "source_id": "SRC_SD_ZAOZHUANG_2022_OFFICIAL_INDEX",
        "source_tier": "TIER2",
        "publisher": "枣庄市统计局",
        "title": "统计公报目录（2022年枣庄市国民经济和社会发展统计公报）",
        "url_or_reference": "https://stjj.zaozhuang.gov.cn/xwzx/tjgb/",
        "publication_date": "2023-03-28",
        "statistical_period": "2022",
        "geographic_scope": "枣庄市",
        "indicator": "来源存在性核验",
        "unit": "N/A",
        "definition": "官方统计局目录列出公报，但本轮未直接取得原文表值",
        "revision_status": "INDEX_ONLY",
        "retrieved_at": "2026-09-22",
        "used_for": "TRUTH_SOURCE_UPGRADE_QUEUE",
        "notes": "supersedes_source_id=SRC_ZZ_2022_BULLETIN_REPRO; verification_role=official-index-cross-check; raw truth remains TIER3 until direct value retrieval.",
    },
]

ROW_UPGRADES = {
    **{city_id: {"source_id": "SRC_ZJ_2022_POP_BULLETIN_GOV_REPOST", "source_tier": "TIER2", "official_value": value, "status": "UPGRADED_TIER2_MATCH", "url": SOURCE_ADDITIONS[0]["url_or_reference"], "apply_v1": True, "notes": "Government repost of the provincial official population bulletin; value matches v1 reproduced table."} for city_id, value in {
        "city1": "341.3", "city4": "117.0", "city7": "535.3", "city10": "251.5", "city11": "555.1", "city19": "712.7", "city20": "229.0", "city21": "1237.6", "city25": "961.8", "city29": "667.8", "city33": "967.9",
    }.items()},
    "city6": {"source_id": "SRC_GD_FOSHAN_2022_OFFICIAL_MEDIA", "source_tier": "TIER2", "official_value": "955.23", "status": "UPGRADED_TIER2_MATCH", "url": SOURCE_ADDITIONS[4]["url_or_reference"], "apply_v1": True, "notes": "Official local-media reproduction of the city statistics bulletin; not promoted to TIER1."},
    "city9": {"source_id": "SRC_GD_MAOMING_2022_OFFICIAL_MEDIA", "source_tier": "TIER2", "official_value": "623.82", "status": "UPGRADED_TIER2_MATCH", "url": SOURCE_ADDITIONS[5]["url_or_reference"], "apply_v1": True, "notes": "Official local-media reproduction of the city statistics bulletin; not promoted to TIER1."},
    "city14": {"source_id": "SRC_GD_HEYUAN_2022_BULLETIN", "source_tier": "TIER1", "official_value": "284.17", "status": "UPGRADED_TIER1_MATCH", "url": SOURCE_ADDITIONS[1]["url_or_reference"], "apply_v1": True, "notes": "Direct first-party city government/statistics bulletin."},
    "city16": {"source_id": "", "source_tier": "TIER3", "official_value": "", "status": "RETAIN_TIER3_NO_DIRECT_FIRST_PARTY", "url": "", "apply_v1": False, "notes": "Search found reproductions attributed to the statistics bureau, but no directly retrievable first-party page in this pass."},
    "city23": {"source_id": "SRC_GD_ZHUHAI_2022_OFFICIAL_MEDIA", "source_tier": "TIER2", "official_value": "247.72", "status": "MATERIAL_VALUE_CHANGE_REQUIRES_V2", "url": SOURCE_ADDITIONS[6]["url_or_reference"], "apply_v1": False, "notes": "Official-statistics-attributed report gives 247.72 versus v1 245; v1 is preserved and v2 is required."},
    "city27": {"source_id": "SRC_GD_SHANTOU_2022_BULLETIN", "source_tier": "TIER1", "official_value": "554.19", "status": "UPGRADED_TIER1_MATCH", "url": SOURCE_ADDITIONS[2]["url_or_reference"], "apply_v1": True, "notes": "Direct first-party city statistics bulletin."},
    "city35": {"source_id": "SRC_GD_ZHANJIANG_2022_BULLETIN", "source_tier": "TIER1", "official_value": "703.54", "status": "UPGRADED_TIER1_MATCH", "url": SOURCE_ADDITIONS[3]["url_or_reference"], "apply_v1": True, "notes": "Direct first-party city statistics page."},
    "city36": {"source_id": "SRC_SD_ZAOZHUANG_2022_OFFICIAL_INDEX", "source_tier": "TIER2", "official_value": "", "status": "OFFICIAL_INDEX_FOUND_RAW_NOT_RETRIEVED", "url": SOURCE_ADDITIONS[7]["url_or_reference"], "apply_v1": False, "notes": "Official index confirms the 2022 bulletin exists, but the raw value was not directly retrieved; retain the traceable reproduction."},
    "city38": {"source_id": "", "source_tier": "TIER3", "official_value": "", "status": "RETAIN_TIER3_NO_DIRECT_FIRST_PARTY", "url": "", "apply_v1": False, "notes": "Search found reproductions attributed to the statistics bureau, but no directly retrievable first-party page in this pass."},
    "city39": {"source_id": "", "source_tier": "TIER3", "official_value": "", "status": "RETAIN_TIER3_NO_DIRECT_FIRST_PARTY", "url": "", "apply_v1": False, "notes": "Search found a traceable reproduction, but no directly retrievable city statistics page in this pass."},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decimal(value: str | int | float | None) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def display_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def render_truth(value: str | Decimal | None) -> str:
    parsed = decimal(value)
    if parsed is None:
        return ""
    return str(parsed.quantize(Decimal("1"), rounding=ROUND_HALF_UP).to_integral())


def git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return ""
    return result.stdout.strip()


def source_tier_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row.get("source_tier", "") for row in rows)
    return {tier: counts.get(tier, 0) for tier in ["TIER1", "TIER2", "TIER3", "TIER4"]}


def mapping_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row.get("mapping_status", "") for row in rows)
    return {status: counts.get(status, 0) for status in ["CONFIRMED", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"]}


def truth_status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row.get("truth_status", "") for row in rows)
    return {status: counts.get(status, 0) for status in ["EXACT_OFFICIAL", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"]}


def original_raw_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    for row in result:
        city_id = row["city_id"]
        if city_id in RAW_SOURCE_BY_CITY:
            row["source_id"] = RAW_SOURCE_BY_CITY[city_id]
            row["source_tier"] = "TIER3"
            row["value_confidence"] = "HIGH_CONFIDENCE_REPRODUCED"
            if city_id == "city36":
                row["conflict_note"] = ""
        if city_id == "city40":
            row["value_confidence"] = "PROVISIONAL"
            row["conflict_note"] = "Identity provisional because of retained registered-field anomaly."
    return result


def append_note(existing: str, note: str) -> str:
    if not note or note in existing:
        return existing
    return f"{existing} {note}".strip()


def apply_source_upgrade(raw_rows: list[dict[str, str]], include_city23: bool) -> list[dict[str, str]]:
    result = deepcopy(raw_rows)
    for row in result:
        city_id = row["city_id"]
        upgrade = ROW_UPGRADES.get(city_id)
        should_apply = bool(upgrade and upgrade["source_id"] and upgrade["official_value"] and (include_city23 or upgrade["apply_v1"]))
        if should_apply:
            old_value = decimal(row["raw_value"])
            new_value = decimal(upgrade["official_value"])
            if not include_city23 and old_value != new_value:
                raise RuntimeError(f"A v1 provenance update would change {city_id}; v2 is required")
            row["raw_value"] = display_decimal(new_value)
            row["normalized_value_wan"] = display_decimal(new_value)
            row["source_id"] = upgrade["source_id"]
            row["source_tier"] = upgrade["source_tier"]
            row["value_confidence"] = "EXACT_OFFICIAL" if upgrade["source_tier"] == "TIER1" else "HIGH_CONFIDENCE_OFFICIAL_REPOST"
            row["conflict_note"] = append_note(row.get("conflict_note", ""), "Provenance hardened; raw source retained in source ledger.")
        if city_id == "city40":
            row["value_confidence"] = "EXACT_OFFICIAL"
            row["conflict_note"] = "Registered 2021=65 and density-field conflict are retained; identity upgraded using independent non-anomalous evidence."
    return result


def update_mapping(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    for row in result:
        if row["city_id"] != "city40":
            continue
        row["mapping_status"] = "HIGH_CONFIDENCE"
        row["mapping_confidence"] = "HIGH"
        row["positive_evidence_count"] = "6"
        row["independent_source_count"] = "3"
        row["conflict_count"] = "3"
        row["main_conflict"] = "registered 2021=65; density source values 2017/2019/2020 conflict with Zhenjiang anchor"
        row["notes"] = "Resident trajectory, non-anomalous registered trajectory, urbanization, province fingerprint, and 2020 resident anchor independently support Zhenjiang; raw anomalies remain unresolved."
    return result


def update_full_truth(full_rows: list[dict[str, str]], raw_rows: list[dict[str, str]], mapping_rows: list[dict[str, str]], version: str) -> list[dict[str, str]]:
    raw_by_city = {row["city_id"]: row for row in raw_rows}
    mapping_by_city = {row["city_id"]: row for row in mapping_rows}
    result = deepcopy(full_rows)
    for row in result:
        raw = raw_by_city[row["city_id"]]
        mapping = mapping_by_city[row["city_id"]]
        prior_source_id = row.get("primary_source_id", "")
        row["candidate_truth"] = render_truth(raw["raw_value"])
        row["raw_real_value"] = raw["raw_value"]
        row["primary_source_id"] = raw["source_id"]
        row["mapping_confidence"] = mapping["mapping_confidence"]
        row["value_confidence"] = raw["value_confidence"]
        if raw["source_tier"] == "TIER1" and mapping["mapping_status"] in {"CONFIRMED", "HIGH_CONFIDENCE"}:
            row["truth_status"] = "EXACT_OFFICIAL"
        elif mapping["mapping_status"] == "UNRESOLVED":
            row["truth_status"] = "UNRESOLVED"
        elif mapping["mapping_status"] == "PROVISIONAL":
            row["truth_status"] = "PROVISIONAL"
        else:
            row["truth_status"] = "HIGH_CONFIDENCE"
        if row["city_id"] == "city40":
            row["notes"] = "Identity upgraded to HIGH_CONFIDENCE from independent non-anomalous evidence; registered=65 and density conflicts are retained without correction."
        elif version == "v2" and row["city_id"] == "city23":
            row["notes"] = "Truth v2 adopts the 2022 official-statistics-attributed value 247.72; v1 value 245 is preserved for versioned comparison."
        elif raw["source_id"] != prior_source_id:
            row["notes"] = append_note(row.get("notes", ""), "Source provenance hardened without changing the rendered value.")
        value = decimal(raw["raw_value"])
        row["uncertainty_lower"] = display_decimal(value - Decimal("0.5")) if value is not None else ""
        row["uncertainty_upper"] = display_decimal(value + Decimal("0.5")) if value is not None else ""
    return result


def add_conflicts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    if not any(row.get("city_id") == "city40" and row.get("conflict_type") == "DENSITY_FIELD_MISMATCH" for row in result):
        result.append({
            "city_id": "city40", "real_city": "镇江", "province": "江苏",
            "conflict_type": "DENSITY_FIELD_MISMATCH", "indicator": "population_density",
            "competition_year": "2017/2019/2020", "real_year": "2020",
            "primary_value": "1600", "alternate_value": "835",
            "primary_source_id": "SRC_COMP_DENSITY", "alternate_source_id": "SRC_JS_CENSUS_2020",
            "severity": "MATERIAL", "resolution_status": "UNRESOLVED_RETAIN_RAW",
            "resolution_note": "Density source values are retained; resident, non-anomalous registered and urbanization evidence still uniquely support Zhenjiang.",
        })
    return result


def build_queue(original_raw: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in original_raw:
        if row.get("value_confidence") != "HIGH_CONFIDENCE_REPRODUCED":
            continue
        city_id = row["city_id"]
        upgrade = ROW_UPGRADES[city_id]
        old_value = decimal(row["raw_value"])
        candidate = decimal(upgrade["official_value"])
        delta = display_decimal(candidate - old_value) if candidate is not None and old_value is not None else ""
        rows.append({
            "city_id": city_id,
            "real_city": row["real_city"],
            "province": row["province"],
            "raw_value": row["raw_value"],
            "current_source_id": row["source_id"],
            "current_source_tier": row["source_tier"],
            "upgrade_status": upgrade["status"],
            "official_candidate_source": upgrade["source_id"],
            "official_candidate_url": upgrade["url"],
            "official_candidate_value": upgrade["official_value"],
            "value_delta": delta,
            "rendered_v1": render_truth(row["raw_value"]),
            "rendered_candidate": render_truth(upgrade["official_value"]),
            "notes": upgrade["notes"],
        })
    if len(rows) != 21:
        raise RuntimeError(f"Expected 21 reproduced rows, found {len(rows)}")
    return rows


def build_upgrade_delta(original_raw: list[dict[str, str]], hardened_raw: list[dict[str, str]], mapping_before: list[dict[str, str]], mapping_after: list[dict[str, str]]) -> list[dict[str, str]]:
    old_by = {row["city_id"]: row for row in original_raw}
    new_by = {row["city_id"]: row for row in hardened_raw}
    before_map = {row["city_id"]: row for row in mapping_before}
    after_map = {row["city_id"]: row for row in mapping_after}
    rows = []
    for city_id in EXPECTED_CITY_IDS:
        old = old_by[city_id]
        new = new_by[city_id]
        upgrade = ROW_UPGRADES.get(city_id, {})
        old_value = decimal(old["raw_value"])
        candidate = decimal(upgrade.get("official_value"))
        rows.append({
            "city_id": city_id,
            "real_city": old["real_city"],
            "province": old["province"],
            "v1_raw_value": old["raw_value"],
            "hardened_v1_raw_value": new["raw_value"],
            "new_official_candidate_raw_value": upgrade.get("official_value", ""),
            "v1_source_id": old["source_id"],
            "hardened_source_id": new["source_id"],
            "official_candidate_source": upgrade.get("source_id", ""),
            "source_tier_after": new["source_tier"],
            "delta_raw_if_adopted": display_decimal(candidate - old_value) if candidate is not None and old_value is not None else "",
            "v1_rendered_truth": render_truth(old["raw_value"]),
            "hardened_v1_rendered_truth": render_truth(new["raw_value"]),
            "rendered_changed_if_adopted": "YES" if candidate is not None and render_truth(candidate) != render_truth(old["raw_value"]) else "NO",
            "provenance_changed": "YES" if old["source_id"] != new["source_id"] else "NO",
            "mapping_status_before": before_map[city_id]["mapping_status"],
            "mapping_status_after": after_map[city_id]["mapping_status"],
            "notes": upgrade.get("notes", "") if upgrade else "No truth-source upgrade required.",
        })
    return rows


def build_v1_to_v2_delta(v1_raw: list[dict[str, str]], v2_raw: list[dict[str, str]], v1_eval: list[dict[str, str]], v2_eval: list[dict[str, str]]) -> list[dict[str, str]]:
    v1_by = {row["city_id"]: row for row in v1_raw}
    v2_by = {row["city_id"]: row for row in v2_raw}
    e1 = {row["city_id"]: row for row in v1_eval}
    e2 = {row["city_id"]: row for row in v2_eval}
    rows = []
    for city_id in EXPECTED_CITY_IDS:
        raw1 = v1_by[city_id]
        raw2 = v2_by[city_id]
        d1 = decimal(raw1["raw_value"])
        d2 = decimal(raw2["raw_value"])
        rows.append({
            "city_id": city_id,
            "real_city": raw1["real_city"],
            "province": raw1["province"],
            "v1_raw_value": raw1["raw_value"],
            "v2_raw_value": raw2["raw_value"],
            "delta_raw": display_decimal(d2 - d1) if d1 is not None and d2 is not None else "",
            "v1_source_id": raw1["source_id"],
            "v2_source_id": raw2["source_id"],
            "v1_rendered_truth": e1[city_id]["truth"],
            "v2_rendered_truth": e2[city_id]["truth"],
            "rendered_changed": "YES" if e1[city_id]["truth"] != e2[city_id]["truth"] else "NO",
            "old_chat_value_ignored": "248" if city_id == "city23" else "",
            "resolution": "Official-statistics-attributed 2022 value 247.72 requires v2; old chat 248 is not evidence." if city_id == "city23" else "No evaluator change.",
        })
    return rows


def build_city40_resolution(identity_evidence: list[dict[str, str]], mapping_before: list[dict[str, str]], mapping_after: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    for row in identity_evidence:
        if row.get("province") != "江苏":
            continue
        city = row["candidate_real_city"]
        candidates.setdefault(city, {})
        etype = row["evidence_type"]
        if etype == "CENSUS_POPULATION_ANCHOR":
            candidates[city]["resident"] = row["real_value"]
        elif etype == "REGISTERED_POPULATION_STRUCTURE":
            candidates[city]["registered"] = row["real_value"]
        elif etype == "URBANIZATION_STRUCTURE":
            candidates[city]["urbanization"] = row["real_value"]
        elif etype == "DENSITY_STRUCTURE":
            candidates[city]["density"] = row["real_value"]
    comparison = []
    target = {"resident": Decimal("320"), "registered": Decimal("270"), "urbanization": Decimal("80"), "density": Decimal("1600")}
    for city, values in candidates.items():
        parsed = {key: decimal(values.get(key)) for key in target}
        deltas = {key: display_decimal(abs(target[key] - parsed[key])) if parsed[key] is not None else "" for key in target}
        comparison.append({"city": city, **values, **{f"{key}_abs_delta": value for key, value in deltas.items()}})
    comparison.sort(key=lambda row: (decimal(row.get("resident_abs_delta")) or Decimal("999999"), decimal(row.get("registered_abs_delta")) or Decimal("999999")))
    alternates = [row["city"] for row in comparison if row["city"] != "镇江" and decimal(row.get("resident_abs_delta")) is not None and decimal(row["resident_abs_delta"]) <= Decimal("5") and decimal(row.get("registered_abs_delta")) is not None and decimal(row["registered_abs_delta"]) <= Decimal("5")]
    alternate = ";".join(alternates) if alternates else "NONE"
    status_before = next(row["mapping_status"] for row in mapping_before if row["city_id"] == "city40")
    return [{
        "city_id": "city40",
        "candidate_city": "镇江",
        "province": "江苏",
        "status_before": status_before,
        "status_after": next(row["mapping_status"] for row in mapping_after if row["city_id"] == "city40"),
        "evidence_for_zhenjiang": "2020 resident=320 versus official 321.0418; 2017-2020 registered trajectory=269-272; 2021 urbanization=80 versus official 79.5; no other Jiangsu city jointly explains resident, registered and urbanization.",
        "evidence_against_zhenjiang": "Registered 2021=65 versus official anchor 269.25; density source has 2017/2019/2020 values around 1600 versus official anchor 835.",
        "alternate_candidate": alternate,
        "registered_65_interpretation": "ANOMALY_CAUSE_UNRESOLVED; raw XLSX and normalized CSV both contain literal 65 in the 2021 registered field; no unit conversion or field-shift evidence found.",
        "resolution": "Upgrade to HIGH_CONFIDENCE while retaining the registered and density conflicts; do not correct competition raw values.",
        "confidence": "HIGH_CONFIDENCE",
        "alternate_falsification": json.dumps(comparison, ensure_ascii=False, separators=(",", ":")),
    }]


def raw_source_audit(data_root: Path) -> dict[str, str]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for the read-only raw XLSX anomaly audit") from exc
    source_dir = data_root / "data" / "raw" / "城市人口分析与预测" / "数据集"
    matched = None
    for path in sorted(source_dir.glob("*.xlsx")):
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    values = [cell.value for cell in row]
                    if len(values) >= 4 and values[0] == "city40" and values[1] == 2021 and values[2] == 321 and values[3] == 65:
                        matched = {"path": str(path.relative_to(data_root)).replace("\\", "/"), "sheet": sheet.title, "row": str(row[0].row), "cells": json.dumps({cell.coordinate: cell.value for cell in row[:4]}, ensure_ascii=False)}
                        break
                if matched:
                    break
        finally:
            workbook.close()
        if matched:
            break
    if not matched:
        raise RuntimeError("Could not locate literal city40 2021 registered value 65 in legal XLSX sources")
    normalized_path = data_root / "data" / "normalized" / "人口规模.csv"
    normalized_line = ""
    with normalized_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("city40,2021,"):
                normalized_line = f"line {line_no}: {line.strip()}"
                break
    if not normalized_line:
        raise RuntimeError("Could not locate city40 2021 row in normalized population CSV")
    matched.update({
        "raw_sha256": sha256_file(data_root / matched["path"]),
        "normalized_path": "data/normalized/人口规模.csv",
        "normalized_sha256": sha256_file(normalized_path),
        "normalized_row": normalized_line,
    })
    return matched


def write_city40_audit(path: Path, audit: dict[str, str]) -> None:
    path.write_text(f"""# city40 registered-field anomaly audit

Status: `ANOMALY_CAUSE_UNRESOLVED`

## Raw-to-normalized propagation

- Legal raw workbook: `{audit['path']}`
- Sheet / row: `{audit['sheet']}` / `{audit['row']}`
- Raw cells: `{audit['cells']}`
- Raw workbook SHA256: `{audit['raw_sha256']}`
- Normalized file: `{audit['normalized_path']}`
- Normalized file SHA256: `{audit['normalized_sha256']}`
- Normalized row: `{audit['normalized_row']}`

The literal `65` is present in the legal source workbook and is propagated
unchanged into the normalized CSV. This audit found no transformation-layer
unit conversion, column shift, interpolation, or replacement that created the
value. `test.xls` was not opened, parsed, hashed, or used.

## Independent identity evidence

- Resident trajectory: city40 is approximately 318–321万人 in 2017–2021,
  matching Zhenjiang's official 2020 census anchor of 321.0418万人.
- Registered trajectory excluding the anomalous row: 2017–2020 is 272, 272,
  271, 270万人, matching the official Zhenjiang anchor of 269.25万人.
- Urbanization: competition 2021 is 80%, versus the Zhenjiang 2020 anchor of
  79.5%.
- Density: competition source values around 1600 are inconsistent with the
  Zhenjiang anchor around 835 and are retained as a separate source conflict.
- Alternate-city falsification found no other Jiangsu city jointly explaining
  resident, non-anomalous registered, and urbanization trajectories.

## Resolution

`city40 -> 镇江` is upgraded from `PROVISIONAL` to `HIGH_CONFIDENCE` using
independent evidence. The registered value 65 and density conflict remain
raw, explicit, and unresolved. No competition source value is corrected.
""", encoding="utf-8")


def update_ledger(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = deepcopy(rows)
    known = {row["source_id"] for row in result}
    for addition in SOURCE_ADDITIONS:
        if addition["source_id"] not in known:
            result.append(addition)
    return result


def write_city23_resolution(path: Path) -> None:
    write_csv(path, [{
        "city_id": "city23", "real_city": "珠海", "official_2022_value": "247.72",
        "source_id": "SRC_GD_ZHUHAI_2022_OFFICIAL_MEDIA",
        "source_url": ROW_UPGRADES["city23"]["url"], "current_v1_value": "245",
        "old_chat_value_ignored": "248",
        "resolution": "The official-statistics-attributed 2022 value is 247.72; old chat 248 is ignored as evidence.",
        "rendered_truth_v1": "245", "rendered_truth_v2": "248",
        "changed_vs_v1": "YES", "version_decision": "CREATE_TRUTH_V2",
    }])


def copy_common_evidence(v1_dir: Path, v2_dir: Path) -> None:
    names = [
        "competition_city_fingerprints.csv", "city_identity_evidence.csv", "year_alignment_evidence.csv",
        "year_alignment_summary.csv", "target_semantics_evidence.csv", "target_semantics_summary.json",
        "statistical_revision_ledger.csv", "rounding_unit_evidence.csv",
    ]
    for name in names:
        shutil.copyfile(v1_dir / name, v2_dir / name)


def manifest_base(manifest: dict[str, Any], repo_root: Path, generated_at: str, source_ledger: list[dict[str, str]], raw_rows: list[dict[str, str]], full_rows: list[dict[str, str]], evaluator_path: Path, full_path: Path, mapping_rows: list[dict[str, str]], truth_version: str) -> dict[str, Any]:
    result = deepcopy(manifest)
    result.update({
        "truth_version": truth_version,
        "plan_id": PLAN_ID,
        "plan02_base_commit": git(repo_root, "rev-parse", "HEAD"),
        "branch": git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "generation_script": "scripts/harden_s8_truth_v1.py",
        "generated_at": generated_at,
        "city_count": len(raw_rows),
        "resolved_city_count": sum(row["truth_status"] != "UNRESOLVED" for row in full_rows),
        "provisional_city_count": sum(row["truth_status"] == "PROVISIONAL" for row in full_rows),
        "unresolved_city_count": sum(row["truth_status"] == "UNRESOLVED" for row in full_rows),
        "source_count": len(source_ledger),
        "tier1_source_count": sum(row["source_tier"] == "TIER1" for row in source_ledger),
        "fallback_source_count": sum(row["source_tier"] in {"TIER3", "TIER4"} for row in source_ledger),
        "truth_file": evaluator_path.name,
        "truth_sha256": sha256_file(evaluator_path),
        "full_truth_file": full_path.name,
        "full_truth_sha256": sha256_file(full_path),
        "truth_tier_counts": source_tier_counts(raw_rows),
        "truth_status_counts": truth_status_counts(full_rows),
        "mapping_status_counts": mapping_counts(mapping_rows),
        "city40_mapping_status": next(row["mapping_status"] for row in mapping_rows if row["city_id"] == "city40"),
        "target_semantics": "RESIDENT_POPULATION",
        "usage_scope": ["POST_HOC_HIDDEN_EVALUATOR_ONLY", "NOT_FOR_MODEL_SELECTION", "NOT_SHARED_WITH_S9"],
        "isolation": {"s9_branch_read": False, "s9_predictions_read": False, "model_evaluation_performed": False, "test_xls_accessed": False},
    })
    return result


def write_report(path: Path, version: str, manifest: dict[str, Any], queue: list[dict[str, str]], delta: list[dict[str, str]], v2_delta: list[dict[str, str]] | None = None) -> None:
    tier = manifest["truth_tier_counts"]
    status = manifest["freeze_status"]
    changed = sum(row["rendered_changed"] == "YES" for row in v2_delta or [])
    path.write_text(f"""# S8 Truth Provenance Hardening Report ({version})

Status: {status}

This report covers S8-Plan02 truth provenance hardening only. No model was
trained, scored, ranked, or inspected, and no S9 artifact was read.

## Source hardening

- Reproduced truth rows before: 21
- Tier1 truth rows after hardening: {tier['TIER1']}
- Tier2 truth rows after hardening: {tier['TIER2']}
- Tier3 fallback truth rows after hardening: {tier['TIER3']}
- Tier4 fallback truth rows after hardening: {tier['TIER4']}
- Direct source upgrade queue rows: {sum(row['upgrade_status'].startswith('UPGRADED') for row in queue)}
- Rows retained without direct first-party value: {sum('RETAIN' in row['upgrade_status'] or 'NOT_RETRIEVED' in row['upgrade_status'] for row in queue)}

The remaining fallback rows are explicit in `truth_source_upgrade_queue.csv`.
The old reproduced source rows remain in `source_ledger.csv` as historical
fallback/cross-check records.

## City40

`city40 -> 镇江` moves from `PROVISIONAL` to `HIGH_CONFIDENCE`. The raw
registered value 65 is present in the legal XLSX and normalized CSV. It is not
corrected. The non-anomalous registered trajectory, resident trajectory,
urbanization and province fingerprint support Zhenjiang; the density conflict
and cause of 65 remain unresolved.

## City23 and versioning

The current v1 value is 245. A 2022 official-statistics-attributed Zhuhai
source gives 247.72, which renders to 248. The old chat value 248 is recorded
only as ignored trigger context, not evidence. This material rendered change
requires `candidate_truth_2023_v2`; v1 remains preserved.

- v1 evaluator SHA256: {manifest.get('v1_evaluator_sha256', manifest.get('truth_sha256', ''))}
- {version} evaluator SHA256: {manifest['truth_sha256']}
- Rendered rows changed from v1: {changed if v2_delta is not None else 0}

## Freeze gate

The gate is not promoted to full freeze because four truth rows still lack a
directly retrieved Tier1 source in this pass, and the city40 source anomalies
remain explicit. No value was changed to improve any model result.

Isolation: S9 branch read NO; S9 predictions read NO; model scoring NO;
test.xls accessed NO.
""", encoding="utf-8")


def write_handoff(path: Path, version: str, manifest: dict[str, Any], artifact_commit: str, final_branch_head: str) -> None:
    tier = manifest["truth_tier_counts"]
    mapping = manifest["mapping_status_counts"]
    path.write_text(f"""# S8 Plan02 Truth Provenance Hardening Handoff

Status: {manifest['freeze_status']}
Branch: {manifest['branch']}
Base SHA: {manifest['base_commit']}
Artifact Freeze Commit: {artifact_commit or 'PENDING_ARTIFACT_FREEZE_COMMIT'}
Final Branch HEAD: {final_branch_head or 'PENDING_FINAL_BRANCH_HEAD'}
Working tree at generation: metadata may be pending after artifact commit

Truth version: {version}
Evaluator SHA256: {manifest['truth_sha256']}
Full truth SHA256: {manifest['full_truth_sha256']}

Source upgrade:
- Tier1 truth rows before: 19
- Tier1 truth rows after: {tier['TIER1']}
- Tier2 truth rows after: {tier['TIER2']}
- fallback truth rows after: {tier['TIER3'] + tier['TIER4']}
- unresolved source rows: 4

City40:
- mapping before: PROVISIONAL
- mapping after: {mapping['HIGH_CONFIDENCE'] and 'HIGH_CONFIDENCE' or 'PROVISIONAL'}
- registered=65 explanation: ANOMALY_CAUSE_UNRESOLVED; literal raw value propagated unchanged; no correction
- remaining uncertainty: density conflict and anomaly cause

City23:
- official 2022 value: 247.72
- rendered truth: v1=245, v2=248
- changed vs v1: YES; v2 required
- old chat 248: ignored as evidence

Truth delta:
- raw value rows changed in v2: 1
- evaluator rows changed: 1
- v2 required: YES

Isolation:
- S9 branch read: NO
- S9 predictions read: NO
- model scoring performed: NO
- test.xls accessed: NO

Return:
{version} remains provisional and waits for Main review. Do not reveal truth
to S9 before the independent prediction freeze.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, required=True, help="Read-only root containing legal XLSX and normalized CSV")
    parser.add_argument("--generated-at", default="2026-09-22T12:00:00+08:00")
    parser.add_argument("--artifact-commit", default="")
    parser.add_argument("--final-branch-head", default="")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    data_root = args.input_root.resolve()
    v1_dir = repo_root / V1_DIR
    v2_dir = repo_root / V2_DIR
    v2_dir.mkdir(parents=True, exist_ok=True)

    source_ledger = update_ledger(read_csv(v1_dir / "source_ledger.csv"))
    current_raw = read_csv(v1_dir / "candidate_truth_2023_raw.csv")
    original_raw = original_raw_rows(current_raw)
    raw_v1 = apply_source_upgrade(current_raw, include_city23=False)
    raw_v2 = apply_source_upgrade(raw_v1, include_city23=True)
    mapping_before = read_csv(v1_dir / "city_identity_mapping.csv")
    mapping_after = update_mapping(mapping_before)
    full_before = read_csv(v1_dir / "candidate_truth_2023_v1.csv")
    full_v1 = update_full_truth(full_before, raw_v1, mapping_after, "v1")
    full_v2 = update_full_truth(full_v1, raw_v2, mapping_after, "v2")
    conflicts = add_conflicts(read_csv(v1_dir / "truth_conflicts.csv"))
    evaluator_v1 = read_csv(v1_dir / "candidate_truth_2023_v1_evaluator.csv")
    evaluator_v2 = deepcopy(evaluator_v1)
    for row in evaluator_v2:
        if row["city_id"] == "city23":
            row["truth"] = "248"

    queue = build_queue(original_raw)
    upgrade_delta = build_upgrade_delta(original_raw, raw_v1, mapping_before, mapping_after)
    v2_delta = build_v1_to_v2_delta(raw_v1, raw_v2, evaluator_v1, evaluator_v2)
    resolution = build_city40_resolution(read_csv(v1_dir / "city_identity_evidence.csv"), mapping_before, mapping_after)
    audit = raw_source_audit(data_root)

    write_csv(v1_dir / "source_ledger.csv", source_ledger)
    write_csv(v1_dir / "candidate_truth_2023_raw.csv", raw_v1)
    write_csv(v1_dir / "candidate_truth_2023_v1.csv", full_v1)
    write_csv(v1_dir / "candidate_truth_2023_v1_evaluator.csv", evaluator_v1)
    write_csv(v1_dir / "city_identity_mapping.csv", mapping_after)
    write_csv(v1_dir / "truth_conflicts.csv", conflicts)
    write_csv(v1_dir / "truth_source_upgrade_queue.csv", queue)
    write_csv(v1_dir / "truth_v1_source_upgrade_delta.csv", upgrade_delta)
    write_csv(v1_dir / "city40_identity_resolution.csv", resolution)
    write_city40_audit(v1_dir / "city40_registered_anomaly_audit.md", audit)
    write_city23_resolution(v1_dir / "city23_truth_resolution.csv")

    base_manifest = json.loads((v1_dir / "truth_manifest_v1.json").read_text(encoding="utf-8"))
    v1_eval_path = v1_dir / "candidate_truth_2023_v1_evaluator.csv"
    v1_full_path = v1_dir / "candidate_truth_2023_v1.csv"
    manifest_v1 = manifest_base(base_manifest, repo_root, args.generated_at, source_ledger, raw_v1, full_v1, v1_eval_path, v1_full_path, mapping_after, "candidate_truth_2023_v1")
    manifest_v1["freeze_status"] = "S8_TRUTH_V1_PROVISIONAL_FREEZE"
    manifest_v1["successor_truth_version"] = "candidate_truth_2023_v2"
    manifest_v1["v1_evaluator_sha256_before_hardening"] = base_manifest["truth_sha256"]
    manifest_v1["v1_evaluator_sha256"] = sha256_file(v1_eval_path)
    manifest_v1["provenance_hardening"] = {"queue_rows": len(queue), "rows_with_source_upgrade": sum(row["provenance_changed"] == "YES" for row in upgrade_delta), "city40_status_before": "PROVISIONAL", "city40_status_after": "HIGH_CONFIDENCE", "city23_rendered_change": True}
    manifest_v1["known_conflicts"] = [
        "city3 registered population mismatch retained",
        "city40 registered population 2021 value 65 retained and mapping upgraded only to HIGH_CONFIDENCE",
        "city40 density source conflict retained",
        "city12/city29 Taizhou name collision resolved by province and fingerprint",
        "four 2022 truth rows retain Tier-3 fallback pending direct first-party retrieval",
        "city23 v1 value 245 retained; v2 official-attributed value renders to 248",
    ]
    manifest_v1["evidence_files"] = sorted(set(manifest_v1.get("evidence_files", []) + [
        "truth_source_upgrade_queue.csv", "truth_v1_source_upgrade_delta.csv", "city40_identity_resolution.csv",
        "city40_registered_anomaly_audit.md", "city23_truth_resolution.csv",
    ]))
    write_json(v1_dir / "truth_manifest_v1.json", manifest_v1)
    write_report(v1_dir / "S8_TRUTH_RECONSTRUCTION_REPORT.md", "candidate_truth_2023_v1", manifest_v1, queue, upgrade_delta)
    write_handoff(v1_dir / "HANDOFF.md", "candidate_truth_2023_v1 (successor v2 required)", manifest_v1, args.artifact_commit, args.final_branch_head)

    copy_common_evidence(v1_dir, v2_dir)
    for name in ["source_ledger.csv", "city_identity_mapping.csv", "truth_conflicts.csv", "truth_source_upgrade_queue.csv", "truth_v1_source_upgrade_delta.csv", "city40_identity_resolution.csv", "city23_truth_resolution.csv"]:
        shutil.copyfile(v1_dir / name, v2_dir / name)
    shutil.copyfile(v1_dir / "city40_registered_anomaly_audit.md", v2_dir / "city40_registered_anomaly_audit.md")
    write_csv(v2_dir / "candidate_truth_2023_v2_raw.csv", raw_v2)
    write_csv(v2_dir / "candidate_truth_2023_v2.csv", full_v2)
    write_csv(v2_dir / "candidate_truth_2023_v2_evaluator.csv", evaluator_v2)
    write_csv(v2_dir / "truth_v1_to_v2_delta.csv", v2_delta)
    manifest_v2 = manifest_base(manifest_v1, repo_root, args.generated_at, source_ledger, raw_v2, full_v2, v2_dir / "candidate_truth_2023_v2_evaluator.csv", v2_dir / "candidate_truth_2023_v2.csv", mapping_after, "candidate_truth_2023_v2")
    manifest_v2.update({
        "freeze_status": "S8_TRUTH_PROVISIONAL_FREEZE_RETAINED",
        "supersedes_truth_version": "candidate_truth_2023_v1",
        "v1_evaluator_sha256": sha256_file(v1_eval_path),
        "v1_full_truth_sha256_after_hardening": sha256_file(v1_full_path),
        "rendered_evaluator_rows_changed": sum(row["rendered_changed"] == "YES" for row in v2_delta),
        "raw_value_rows_changed": sum(row["delta_raw"] not in {"", "0"} for row in v2_delta),
        "versioning_decision": "truth_v2 required because city23 247.72 renders to 248 versus v1 245",
        "known_conflicts": manifest_v1["known_conflicts"],
        "evidence_files": sorted(set(manifest_v1.get("evidence_files", []) + [
            "candidate_truth_2023_v2_raw.csv", "candidate_truth_2023_v2.csv", "candidate_truth_2023_v2_evaluator.csv", "truth_v1_to_v2_delta.csv",
        ])),
    })
    write_json(v2_dir / "truth_manifest_v2.json", manifest_v2)
    write_report(v2_dir / "S8_TRUTH_RECONSTRUCTION_REPORT.md", "candidate_truth_2023_v2", manifest_v2, queue, upgrade_delta, v2_delta)
    write_handoff(v2_dir / "HANDOFF.md", "candidate_truth_2023_v2", manifest_v2, args.artifact_commit, args.final_branch_head)

    print(json.dumps({
        "v1_evaluator_sha256": manifest_v1["truth_sha256"],
        "v1_full_truth_sha256": manifest_v1["full_truth_sha256"],
        "v2_evaluator_sha256": manifest_v2["truth_sha256"],
        "v2_full_truth_sha256": manifest_v2["full_truth_sha256"],
        "v1_truth_tier_counts": manifest_v1["truth_tier_counts"],
        "v2_truth_tier_counts": manifest_v2["truth_tier_counts"],
        "mapping_status_counts": manifest_v2["mapping_status_counts"],
        "city23_v1": "245",
        "city23_v2": "248",
        "city40_status": manifest_v2["city40_mapping_status"],
        "freeze_status": manifest_v2["freeze_status"],
        "raw_audit_source": audit["path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
