"""Build the S8 target-forensics and truth-reconstruction artifacts.

This is a non-modelling pipeline. It reads only competition-normalized tables
through input-root and a source-ledger-backed set of manually transcribed
official evidence values. It never reads test.xls, model outputs, scoreboards,
or any S9 worktree/content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


PLAN_ID = "S8-Plan01"
TARGET_YEAR = 2023
REAL_YEAR = 2022
OUTPUT_RELATIVE_DIR = Path("reports/S8_TRUTH/truth_v1")
EXPECTED_CITY_IDS = [f"city{i}" for i in range(1, 41)]
MAPPING_STATUSES = {"CONFIRMED", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"}


CITY_MAP = [
    ("city1", "湖州", "浙江"), ("city2", "无锡", "江苏"),
    ("city3", "深圳", "广东"), ("city4", "舟山", "浙江"),
    ("city5", "苏州", "江苏"), ("city6", "佛山", "广东"),
    ("city7", "绍兴", "浙江"), ("city8", "淮安", "江苏"),
    ("city9", "茂名", "广东"), ("city10", "丽水", "浙江"),
    ("city11", "嘉兴", "浙江"), ("city12", "泰州", "江苏"),
    ("city13", "宿迁", "江苏"), ("city14", "河源", "广东"),
    ("city15", "徐州", "江苏"), ("city16", "韶关", "广东"),
    ("city17", "广州", "广东"), ("city18", "常州", "江苏"),
    ("city19", "金华", "浙江"), ("city20", "衢州", "浙江"),
    ("city21", "杭州", "浙江"), ("city22", "南京", "江苏"),
    ("city23", "珠海", "广东"), ("city24", "济南", "山东"),
    ("city25", "宁波", "浙江"), ("city26", "南通", "江苏"),
    ("city27", "汕头", "广东"), ("city28", "青岛", "山东"),
    ("city29", "台州", "浙江"), ("city30", "连云港", "江苏"),
    ("city31", "江门", "广东"), ("city32", "淄博", "山东"),
    ("city33", "温州", "浙江"), ("city34", "盐城", "江苏"),
    ("city35", "湛江", "广东"), ("city36", "枣庄", "山东"),
    ("city37", "扬州", "江苏"), ("city38", "肇庆", "广东"),
    ("city39", "东营", "山东"), ("city40", "镇江", "江苏"),
]


RESIDENT_2020_PERSONS = {
    "南京": 9314685, "无锡": 7462135, "徐州": 9083790, "常州": 5278121,
    "苏州": 12748262, "南通": 7726635, "连云港": 4599360, "淮安": 4556230,
    "盐城": 6709629, "扬州": 4559797, "镇江": 3210418, "泰州": 4512762,
    "宿迁": 4986192, "杭州": 11936010, "宁波": 9404283, "温州": 9572903,
    "嘉兴": 5400868, "湖州": 3367579, "绍兴": 5270977, "金华": 7050683,
    "衢州": 2276184, "舟山": 1157817, "台州": 6622888, "丽水": 2507396,
    "广州": 18676605, "深圳": 17560061, "珠海": 2439585, "汕头": 5502031,
    "佛山": 9498863, "韶关": 2855131, "河源": 2837686, "江门": 4798090,
    "湛江": 6981236, "茂名": 6174050, "肇庆": 4113594,
    "济南": 9202432, "青岛": 10071722, "淄博": 4704138, "枣庄": 3855601,
    "东营": 2193518,
}


STRUCTURE_2020 = {
    "南京": (722.57, 86.8, 1412), "无锡": (508.97, 82.8, 1612),
    "徐州": (1038.05, 65.6, 771), "常州": (386.63, 77.1, 1207),
    "苏州": (744.33, 81.7, 1470), "南通": (756.06, 70.4, 732),
    "连云港": (534.48, 61.5, 603), "淮安": (557.97, 65.7, 459),
    "盐城": (814.49, 64.1, 396), "扬州": (454.71, 71.0, 691),
    "镇江": (269.25, 79.5, 835), "泰州": (497.15, 68.1, 781),
    "宿迁": (592.10, 62.2, 584), "杭州": (813.83, 83.3, 710),
    "宁波": (613.66, 78.0, 959.7), "温州": (833.75, 72.2, 789.4),
    "嘉兴": (367.38, 71.3, 1281.3), "湖州": (268.06, 65.6, 579.4),
    "绍兴": (447.64, 71.0, 639.1), "金华": (493.90, 68.2, 645.4),
    "衢州": (256.87, 57.6, 257.5), "舟山": (96.20, 71.9, 794.4),
    "台州": (606.98, 62.0, 659.4), "丽水": (270.74, 61.8, 145.2),
    "广州": (985.1142, 86.19, 2512), "深圳": (584.5791, 99.54, 8760),
    "珠海": (139.2176, 90.47, 1405), "汕头": (575.5585, 70.70, 2503),
    "佛山": (473.7736, 95.20, 2501), "韶关": (336.5996, 57.33, 157),
    "河源": (372.1774, 48.50, 181), "江门": (401.5866, 67.63, 505),
    "湛江": (859.5758, 45.46, 528), "茂名": (823.2969, 43.56, 539),
    "肇庆": (455.4056, 51.02, 273), "济南": (806.7252, 73.46, 902.1),
    "青岛": (836.8395, 76.34, 895.7), "淄博": (434.4877, 74.27, 788.9),
    "枣庄": (426.1114, 59.32, 845.5), "东营": (197.9605, 71.45, 266.1),
}


HISTORICAL_RESIDENT_WAN = {
    "南京": {2017: 833.50, 2018: 843.62}, "无锡": {2017: 655.30, 2018: 657.45},
    "徐州": {2017: 876.35, 2018: 880.20}, "常州": {2017: 471.73, 2018: 472.86},
    "苏州": {2017: 1068.36, 2018: 1072.17}, "南通": {2017: 730.50, 2018: 731.00},
    "连云港": {2017: 451.84, 2018: 452.00}, "淮安": {2017: 491.40, 2018: 492.50},
    "盐城": {2017: 724.22, 2018: 720.00}, "扬州": {2017: 450.82, 2018: 453.10},
    "镇江": {2017: 318.63, 2018: 319.64}, "泰州": {2017: 465.19, 2018: 463.57},
    "宿迁": {2017: 491.46, 2018: 492.59}, "杭州": {2017: 946.80, 2018: 980.60},
    "宁波": {2017: 800.50, 2018: 820.20}, "温州": {2017: 921.50, 2018: 925.00},
    "嘉兴": {2017: 465.60, 2018: 472.60}, "湖州": {2017: 299.50, 2018: 302.70},
    "绍兴": {2017: 501.00, 2018: 503.50}, "金华": {2017: 556.40, 2018: 560.40},
    "衢州": {2017: 218.50, 2018: 220.90}, "舟山": {2017: 116.80, 2018: 117.30},
    "台州": {2017: 611.80, 2018: 613.90}, "丽水": {2017: 218.60, 2018: 219.90},
    "广州": {2017: 1449.84, 2018: 1490.44}, "深圳": {2017: 1252.83, 2018: 1302.66},
    "珠海": {2017: 176.54, 2018: 189.11}, "汕头": {2017: 560.82, 2018: 563.85},
    "佛山": {2017: 765.67, 2018: 790.57}, "韶关": {2017: 297.92, 2018: 299.76},
    "河源": {2017: 309.11, 2018: 309.39}, "江门": {2017: 456.17, 2018: 459.82},
    "湛江": {2017: 730.50, 2018: 733.20}, "茂名": {2017: 620.41, 2018: 631.32},
    "肇庆": {2017: 411.54, 2018: 415.17}, "济南": {2017: 732.12, 2018: 746.04},
    "青岛": {2017: 929.05, 2018: 939.48}, "淄博": {2017: 470.84, 2018: 470.18},
    "枣庄": {2017: 392.03, 2018: 392.73}, "东营": {2017: 215.46, 2018: 217.21},
}


TRUTH_2022_WAN = {
    "南京": 949.11, "无锡": 749.08, "徐州": 901.85, "常州": 536.62,
    "苏州": 1291.06, "南通": 774.35, "连云港": 460.05, "淮安": 455.31,
    "盐城": 668.97, "扬州": 458.29, "镇江": 322.22, "泰州": 450.56,
    "宿迁": 497.53, "杭州": 1237.6, "宁波": 961.8, "温州": 967.9,
    "嘉兴": 555.1, "湖州": 341.3, "绍兴": 535.3, "金华": 712.7,
    "衢州": 229.0, "舟山": 117.0, "台州": 667.8, "丽水": 251.5,
    "广州": 1873.41, "深圳": 1766.18, "珠海": 245.0, "汕头": 554.19,
    "佛山": 955.23, "韶关": 286.18, "河源": 284.17, "江门": 482.22,
    "湛江": 703.54, "茂名": 623.82, "肇庆": 412.84, "济南": 941.5,
    "青岛": 1034.21, "淄博": 470.59, "枣庄": 382.97, "东营": 220.9,
}


TRUTH_SOURCE = {
    **{name: "SRC_JS_2022_NJ0306" for name in ["南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁"]},
    **{name: "SRC_ZJ_2022_POP_BULLETIN_REPRO" for name in ["杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水"]},
    "广州": "SRC_GZ_2022_BULLETIN", "深圳": "SRC_SZ_2022_BULLETIN", "江门": "SRC_JM_2022_BULLETIN",
    **{name: "SRC_GD_2022_YEARBOOK_REPRO" for name in ["珠海", "汕头", "佛山", "韶关", "河源", "湛江", "茂名", "肇庆"]},
    "济南": "SRC_JN_2022_BULLETIN", "青岛": "SRC_QD_2022_BULLETIN", "淄博": "SRC_ZB_2022_BULLETIN",
    "枣庄": "SRC_ZZ_2022_BULLETIN_REPRO", "东营": "SRC_DY_2022_BULLETIN_REPRO",
}


def source(source_id: str, tier: str, publisher: str, title: str, url: str, period: str, indicator: str, used_for: str, notes: str = "") -> dict[str, str]:
    return {
        "source_id": source_id, "source_tier": tier, "publisher": publisher,
        "title": title, "url_or_reference": url, "publication_date": "",
        "statistical_period": period, "geographic_scope": "mapped cities",
        "indicator": indicator, "unit": "source unit", "definition": "",
        "revision_status": "CAPTURED_EVIDENCE", "retrieved_at": "",
        "used_for": used_for, "notes": notes,
    }


SOURCE_ROWS = [
    source("SRC_COMP_PDF", "TIER1_COMPETITION", "Competition organizer", "城市人口分析与预测 PDF", "data/raw/城市人口分析与预测/城市人口分析与预测.pdf", "competition schema", "target and auxiliary fields", "TARGET_SEMANTICS", "Local read-only source."),
    source("SRC_COMP_POP", "TIER1_COMPETITION", "Competition tables", "人口规模 normalized table", "data/normalized/人口规模.csv", "2001-2022", "resident and registered population", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS;ROUNDING", "Competition-only evidence."),
    source("SRC_COMP_DENSITY", "TIER1_COMPETITION", "Competition tables", "人口密度 normalized table", "data/normalized/人口密度.csv", "2001-2022", "density", "IDENTITY;YEAR_ALIGNMENT", "Competition-only evidence."),
    source("SRC_COMP_URBAN", "TIER1_COMPETITION", "Competition tables", "城镇化率 normalized table", "data/normalized/城镇化率.csv", "2007-2022", "urbanization", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS", "Competition-only evidence."),
    source("SRC_COMP_OTHER", "TIER1_COMPETITION", "Competition tables", "age employment wage living normalized tables", "data/normalized/*.csv", "2001-2021", "age/employment/wage/living", "IDENTITY", "Auxiliary fingerprint only."),
    source("SRC_JS_CENSUS_2020", "TIER1", "江苏省统计局", "江苏省第七次全国人口普查公报（第二号）", "https://tj.jiangsu.gov.cn/art/2021/5/18/art_85767_10520836.html", "2020-11-01", "resident population", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS;ROUNDING", "Primary census anchor."),
    source("SRC_JS_YEARBOOK_2020", "TIER1", "江苏省统计局", "19-2 三大区域经济社会基本情况（2020年）", "https://tj.jiangsu.gov.cn/2021/nj19/nj1902.htm", "2020", "registered/resident/urbanization", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS", "Rounded annual table."),
    source("SRC_JS_2022_NJ0306", "TIER1", "江苏省统计局", "3-6 按地区分常住人口", "https://tj.jiangsu.gov.cn/2023/nj03/nj0306.htm", "2022", "year-end resident population", "TRUTH_VALUE;YEAR_ALIGNMENT;ROUNDING", "Primary 2022 truth source."),
    source("SRC_ZJ_CENSUS_2020", "TIER1", "浙江省统计局", "浙江省第七次全国人口普查公报", "https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud.zj.gov.cn/jcms_files/jcms1/web3077/site/flash/tjj/Reports1/%E6%B5%99%E6%B1%9F%E4%BA%BA%E6%99%AE2020%E5%85%89%E7%9B%98/zk/html/fu04.pdf", "2020-11-01", "resident population", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS;ROUNDING", "Primary census PDF."),
    source("SRC_ZJ_2018_BULLETIN", "TIER1", "浙江省统计局", "2018年浙江省人口主要数据公报", "https://tjj.zj.gov.cn/art/2019/1/30/art_1525539_30164273.html", "2018", "resident population/urbanization", "IDENTITY;YEAR_ALIGNMENT", "Primary historical annual source."),
    source("SRC_ZJ_2022_POP_BULLETIN_REPRO", "TIER3", "中国经济周刊", "2022年浙江省人口主要数据公报 reproduced", "https://www.ceweekly.cn/economic/industry/2023/0224/406403.html", "2022", "resident population/urbanization", "TRUTH_VALUE", "Official bulletin reported in an embedded table image; fallback source."),
    source("SRC_GD_CENSUS_2020", "TIER1", "广东省统计局", "广东省第七次全国人口普查公报", "https://stats.gd.gov.cn/attachment/0/421/421309/3283428.pdf", "2020-11-01", "resident population", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS;ROUNDING", "Primary census PDF."),
    source("SRC_GD_2022_YEARBOOK_REPRO", "TIER3", "Statistics yearbook reproduction", "2022广东统计年鉴 3-7 各市年末常住人口数", "https://www.tjnj.net/navipage-n3022111617000177.html", "2022", "resident population", "TRUTH_VALUE;YEAR_ALIGNMENT", "Traceable reproduction of original Guangdong yearbook."),
    source("SRC_GZ_2022_BULLETIN", "TIER1", "广州市统计局", "2022年广州市国民经济和社会发展统计公报", "https://tjj.gz.gov.cn/stats_newtjyw/tjsj/tjgb/qstjgb/content/post_8972753.html", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin."),
    source("SRC_SZ_2022_BULLETIN", "TIER1", "深圳市人民政府", "深圳市2022年国民经济和社会发展统计公报", "https://www.sz.gov.cn/cn/xxgk/zfxxgj/tjsj/tjgb/content/post_10578003.html", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin."),
    source("SRC_JM_2022_BULLETIN", "TIER1", "江门市人民政府", "2022年江门市国民经济和社会发展统计公报", "https://www.jiangmen.gov.cn/zzzq/tzgg/content/post_2852860.html", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin."),
    source("SRC_SD_CENSUS_2020", "TIER1", "山东省统计局", "山东省第七次全国人口普查公报（第二号）", "https://tjj.shandong.gov.cn/art/2021/5/21/art_6293_10287511.html", "2020-11-01", "resident population", "IDENTITY;YEAR_ALIGNMENT;TARGET_SEMANTICS;ROUNDING", "Primary census source."),
    source("SRC_SD_2022_PROV_BULLETIN", "TIER1", "山东省统计局", "2022年山东省国民经济和社会发展统计公报", "https://tjj.shandong.gov.cn/art/2023/3/2/art_6196_10303466.html", "2022", "resident population/urbanization", "YEAR_ALIGNMENT", "Primary aggregate cross-check."),
    source("SRC_JN_2022_BULLETIN", "TIER1", "济南市统计局", "2022年济南市国民经济和社会发展统计公报", "https://jntj.jinan.gov.cn/col57208/art/2023/art_57208_4748588.html", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin."),
    source("SRC_QD_2022_BULLETIN", "TIER1", "青岛市统计局", "2022年青岛市国民经济和社会发展统计公报", "https://www.qingdao.gov.cn/zwgk/xxgk/fgw/ywfl/jjfz/202304/t20230420_7153787.shtml", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin."),
    source("SRC_ZB_2022_BULLETIN", "TIER1", "淄博市统计局", "2022年淄博市国民经济和社会发展统计公报", "https://tj.zibo.gov.cn/gongkai/channel_c_5f9fa491ab327f36e4c13077_n_1605682680.3204/doc_6423a0bf9a872c3f275a8cc8.html", "2022", "resident population", "TRUTH_VALUE", "Primary city bulletin page."),
    source("SRC_ZZ_2022_BULLETIN_REPRO", "TIER3", "Traceable reproduction of 枣庄市统计局", "2022年枣庄市国民经济和社会发展统计公报", "https://www.zgrkk.com/reports/959.html", "2022", "resident population", "TRUTH_VALUE", "Official bulletin list: https://stjj.zaozhuang.gov.cn/xwzx/tjgb/."),
    source("SRC_DY_2022_BULLETIN_REPRO", "TIER3", "中国统计信息网", "东营市2022年国民经济和社会发展统计公报", "https://www.tjcn.org/tjgb/15sd/37307.html", "2022", "resident population", "TRUTH_VALUE", "Page identifies original Dongying Statistics Bureau source."),
    source("SRC_CITY_YEARBOOK_2023_T3", "TIER3", "Zeeny-lin/vibe-coding-gis-", "中国城市统计年鉴2023 地级市截面数据", "https://github.com/Zeeny-lin/vibe-coding-gis-/tree/main/yearbook_data/2023", "2022", "registered population", "IDENTITY;TARGET_SEMANTICS", "Registered cross-check only; not truth target."),
    source("SRC_NBS_SEMANTICS", "TIER1", "国家统计局", "城市统计年鉴人口指标口径答复", "https://www.stats.gov.cn/hd/lyzx/zxgk/202405/t20240524_1954107.html", "methodological", "population definitions", "TARGET_SEMANTICS", "Definition cross-check; no numeric value."),
    source("SRC_HIST_JS", "TIER4", "维基百科 cross-check", "江苏省各地级市人口列表", "https://zh.wikipedia.org/wiki/江苏省各地级市人口列表", "2017-2018", "resident population", "IDENTITY;YEAR_ALIGNMENT", "Lead/cross-check only."),
    source("SRC_HIST_ZJ", "TIER4", "维基百科 cross-check", "浙江省各地级市人口列表", "https://zh.wikipedia.org/wiki/浙江省各地级市人口列表", "2017-2018", "resident population", "IDENTITY;YEAR_ALIGNMENT", "Lead/cross-check only."),
    source("SRC_HIST_GD", "TIER4", "维基百科 cross-check", "广东省各地级市人口列表", "https://zh.wikipedia.org/wiki/广东省各地级市人口列表", "2017-2018", "resident population", "IDENTITY;YEAR_ALIGNMENT", "Lead/cross-check only."),
    source("SRC_HIST_SD", "TIER4", "维基百科 cross-check", "山东省各地级市人口列表", "https://zh.wikipedia.org/wiki/山东省各地级市人口列表", "2017-2018", "resident population", "IDENTITY;YEAR_ALIGNMENT", "Lead/cross-check only."),
]

SOURCE_TIER = {row["source_id"]: row["source_tier"] for row in SOURCE_ROWS}


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "..."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def display_number(value: Any) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def json_value(value: Any) -> int | float | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(round(number)) if abs(number - round(number)) < 1e-9 else round(number, 6)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_table(rows: list[dict[str, str]], city_field: str, year_field: str, value_fields: list[str]) -> dict[str, dict[int, dict[str, float | None]]]:
    indexed: dict[str, dict[int, dict[str, float | None]]] = defaultdict(dict)
    for row in rows:
        city = str(row.get(city_field, "")).strip()
        year = parse_float(row.get(year_field))
        if city and year is not None:
            indexed[city][int(year)] = {field: parse_float(row.get(field)) for field in value_fields}
    return indexed


def load_tables(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "data" / "normalized"
    specs = {
        "population": ("人口规模.csv", "城市名称", "年份", ["常住人口（万人）", "户籍人口（万人）"]),
        "density": ("人口密度.csv", "城市名称", "年份", ["人口密度（人/平方公里）"]),
        "urbanization": ("城镇化率.csv", "城市名称", "年份", ["urbanizationRate"]),
        "age": ("年龄结构.csv", "城市名称", "年份", ["0-14", "15-64", "65+"]),
        "employees": ("就业信息_从业人员数.csv", "城市名称", "年份", ["employeesNumber"]),
        "sectors": ("就业信息_第一二三产业就业人数.csv", "城市名称", "年份", ["pi_Employment", "si_Employment", "ti_Employment"]),
        "wage": ("工资水平_职工平均工资.csv", "城市名称", "年份", ["averageWage"]),
        "income": ("生活水平_人均可支配收入.csv", "城市名称", "年份", ["disposableIncome"]),
    }
    result = {}
    for key, (filename, city_field, year_field, fields) in specs.items():
        path = base / filename
        result[key] = index_table(read_csv(path), city_field, year_field, fields) if path.exists() else {}
    return result


def value(table: dict[str, Any], city_id: str, year: int, field: str) -> float | None:
    return table.get(city_id, {}).get(year, {}).get(field)


def series(table: dict[str, Any], city_id: str, field: str) -> dict[str, int | float]:
    return {str(year): json_value(row[field]) for year, row in sorted(table.get(city_id, {}).items()) if row.get(field) is not None}


def differences(values: dict[str, int | float]) -> dict[str, int | float]:
    result = {}
    previous = None
    for year, current in sorted(values.items(), key=lambda item: int(item[0])):
        if previous is not None:
            result[year] = json_value(float(current) - previous)
        previous = float(current)
    return result


def growth(values: dict[str, int | float]) -> dict[str, int | float]:
    result = {}
    previous = None
    for year, current in sorted(values.items(), key=lambda item: int(item[0])):
        current_float = float(current)
        if previous not in (None, 0):
            result[year] = json_value(current_float / previous - 1)
        previous = current_float
    return result


def normalized_2020_wan(city: str) -> float:
    return RESIDENT_2020_PERSONS[city] / 10000


def anchor_source(province: str) -> str:
    return {"江苏": "SRC_JS_CENSUS_2020", "浙江": "SRC_ZJ_CENSUS_2020", "广东": "SRC_GD_CENSUS_2020", "山东": "SRC_SD_CENSUS_2020"}[province]


def history_source(province: str) -> str:
    return {"江苏": "SRC_HIST_JS", "浙江": "SRC_HIST_ZJ", "广东": "SRC_HIST_GD", "山东": "SRC_HIST_SD"}[province]


def near(a: float | None, b: float | None, tolerance: float) -> bool:
    return a is not None and b is not None and abs(a - b) <= tolerance


def relative_difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b in (None, 0) else abs(a - b) / abs(b)


def build_fingerprints(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for city_id in EXPECTED_CITY_IDS:
        resident = series(tables["population"], city_id, "常住人口（万人）")
        registered = series(tables["population"], city_id, "户籍人口（万人）")
        density = series(tables["density"], city_id, "人口密度（人/平方公里）")
        urban = series(tables["urbanization"], city_id, "urbanizationRate")
        age = {str(year): {field: json_value(v) for field, v in values.items()} for year, values in sorted(tables["age"].get(city_id, {}).items())}
        employees = series(tables["employees"], city_id, "employeesNumber")
        sectors = {str(year): {field: json_value(v) for field, v in values.items()} for year, values in sorted(tables["sectors"].get(city_id, {}).items())}
        wages = series(tables["wage"], city_id, "averageWage")
        income = series(tables["income"], city_id, "disposableIncome")
        flags = []
        r21 = value(tables["population"], city_id, 2021, "户籍人口（万人）")
        r20 = value(tables["population"], city_id, 2020, "户籍人口（万人）")
        if r21 is not None and r20 and r21 < r20 * 0.5:
            flags.append("registered_2021_large_drop")
        if len(density) < 5:
            flags.append("density_sparse_or_missing")
        rows.append({
            "city_id": city_id,
            "resident_series_wan": compact_json(resident),
            "resident_difference_series_wan": compact_json(differences(resident)),
            "resident_relative_growth_series": compact_json(growth(resident)),
            "registered_series_wan": compact_json(registered),
            "registered_difference_series_wan": compact_json(differences(registered)),
            "density_series": compact_json(density),
            "density_available_years": ",".join(density),
            "urbanization_series": compact_json(urban),
            "urbanization_available_years": ",".join(urban),
            "age_structure": compact_json(age),
            "employees_series": compact_json(employees),
            "sector_employment_series": compact_json(sectors),
            "wage_series": compact_json(wages),
            "disposable_income_series": compact_json(income),
            "anomaly_flags": ";".join(flags),
            "source_scope": "competition_normalized_only",
        })
    return rows


def build_identity(tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = []
    summaries = []
    for city_id, real_city, province in CITY_MAP:
        p21 = value(tables["population"], city_id, 2021, "常住人口（万人）")
        r21 = value(tables["population"], city_id, 2021, "户籍人口（万人）")
        u21 = value(tables["urbanization"], city_id, 2021, "urbanizationRate")
        d21 = value(tables["density"], city_id, 2021, "人口密度（人/平方公里）")
        source_ids = set()
        positives = 0
        conflicts = []
        for comp_year, real_year in ((2018, 2017), (2019, 2018)):
            comp = value(tables["population"], city_id, comp_year, "常住人口（万人）")
            real = HISTORICAL_RESIDENT_WAN[real_city][real_year]
            ok = near(comp, real, 1.0)
            sid = history_source(province)
            source_ids.add(sid)
            positives += int(ok)
            evidence.append({
                "city_id": city_id, "candidate_real_city": real_city, "province": province,
                "evidence_type": "MULTIYEAR_POPULATION_TRAJECTORY", "competition_year": comp_year,
                "competition_value": display_number(comp), "real_year": real_year,
                "real_value": display_number(real), "unit": "万人", "source_id": sid,
                "match_description": "historical annual trajectory cross-check",
                "supports_or_conflicts": "SUPPORTS" if ok else "CONFLICTS",
                "evidence_strength": "MEDIUM",
                "notes": "Tier-4 trajectory cross-check; not used alone.",
            })
        real_p = normalized_2020_wan(real_city)
        ok = near(p21, real_p, 1.0)
        sid = anchor_source(province)
        source_ids.add(sid)
        positives += int(ok)
        evidence.append({
            "city_id": city_id, "candidate_real_city": real_city, "province": province,
            "evidence_type": "CENSUS_POPULATION_ANCHOR", "competition_year": 2021,
            "competition_value": display_number(p21), "real_year": 2020,
            "real_value": display_number(real_p), "unit": "万人", "source_id": sid,
            "match_description": "2021 competition resident value rounds to the 2020 census anchor",
            "supports_or_conflicts": "SUPPORTS" if ok else "CONFLICTS",
            "evidence_strength": "HIGH",
            "notes": "Persons converted to 万人 for comparison.",
        })
        registered, urban, density = STRUCTURE_2020[real_city]
        checks = [
            ("REGISTERED_POPULATION_STRUCTURE", r21, registered, "万人", "SRC_JS_YEARBOOK_2020" if province == "江苏" else sid, near(r21, registered, 1.0), "registered is retained as an independent comparator"),
            ("URBANIZATION_STRUCTURE", u21, urban, "%", "SRC_JS_YEARBOOK_2020" if province == "江苏" else sid, near(u21, urban, 1.0), "urbanization is independent structural evidence"),
            ("DENSITY_STRUCTURE", d21, density, "人/平方公里", sid, near(d21, density, max(5, density * 0.02)), "density is independent structural evidence"),
        ]
        for etype, comp, real, unit, source_id, ok, note in checks:
            source_ids.add(source_id)
            positives += int(ok)
            if not ok:
                conflicts.append(etype)
            evidence.append({
                "city_id": city_id, "candidate_real_city": real_city, "province": province,
                "evidence_type": etype, "competition_year": 2021,
                "competition_value": display_number(comp), "real_year": 2020,
                "real_value": display_number(real), "unit": unit, "source_id": source_id,
                "match_description": "2021 competition structure compared with 2020 mapped-city source",
                "supports_or_conflicts": "SUPPORTS" if ok else "CONFLICTS",
                "evidence_strength": "HIGH" if etype != "DENSITY_STRUCTURE" else "MEDIUM",
                "notes": note,
            })
        if city_id == "city40":
            conflicts.append("registered_2021_large_drop")
        status = "PROVISIONAL" if city_id == "city40" else ("HIGH_CONFIDENCE" if conflicts else "CONFIRMED")
        confidence = "MEDIUM" if city_id == "city40" else ("HIGH" if conflicts else "VERY_HIGH")
        main_conflict = ""
        if city_id == "city40":
            main_conflict = "registered 2021 is 65 versus Zhenjiang 2020 registered 269.25"
        elif conflicts:
            main_conflict = "; ".join(conflicts)
        alternate = ""
        if city_id == "city12":
            alternate = "浙江台州; province and fingerprint resolve collision"
        if city_id == "city29":
            alternate = "江苏泰州; province and fingerprint resolve collision"
        summaries.append({
            "city_id": city_id, "real_city": real_city, "province": province,
            "mapping_status": status, "mapping_confidence": confidence,
            "positive_evidence_count": positives, "independent_source_count": len(source_ids),
            "conflict_count": len(conflicts), "main_conflict": main_conflict,
            "alternate_candidate": alternate,
            "notes": "Multi-year population plus registered, urbanization, and density evidence; no model evidence.",
        })
    return evidence, summaries


def build_alignment(tables: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []

    def add(mapping, indicator, comp_year, comp, real_year, real, offset, sid, tolerance, note, forced=None):
        abs_diff = None if comp is None or real is None else abs(comp - real)
        rel_diff = relative_difference(comp, real)
        ok = forced if forced is not None else (abs_diff is not None and abs_diff <= tolerance)
        rows.append({
            "city_id": mapping[0], "real_city": mapping[1], "indicator": indicator,
            "competition_year": comp_year, "competition_value": display_number(comp),
            "candidate_real_year": real_year, "real_value": display_number(real),
            "alignment_offset": offset, "source_id": sid,
            "absolute_difference": display_number(abs_diff),
            "relative_difference": display_number(rel_diff),
            "revision_note": note,
            "supports_alignment": "YES" if ok else ("NO" if real is not None else "NOT_AVAILABLE"),
        })

    for mapping in CITY_MAP:
        city_id, real_city, province = mapping
        p19 = value(tables["population"], city_id, 2019, "常住人口（万人）")
        p18 = value(tables["population"], city_id, 2018, "常住人口（万人）")
        p20 = value(tables["population"], city_id, 2020, "常住人口（万人）")
        p21 = value(tables["population"], city_id, 2021, "常住人口（万人）")
        h = HISTORICAL_RESIDENT_WAN[real_city]
        p2020 = normalized_2020_wan(real_city)
        add(mapping, "resident_population", 2019, p19, 2017, h[2017], -2, history_source(province), 1.0, "historical negative control")
        add(mapping, "resident_population", 2021, p21, 2020, p2020, -1, anchor_source(province), 0.75, "CENSUS_BENCHMARK")
        add(mapping, "resident_population", 2018, p18, 2018, h[2018], 0, history_source(province), 1.0, "same-label negative control")
        add(mapping, "resident_population", 2020, p20, 2020, p2020, 0, anchor_source(province), 0.75, "same-label census negative control")
        add(mapping, "resident_population", 2021, p21, 2022, TRUTH_2022_WAN[real_city], 1, TRUTH_SOURCE[real_city], 0.75, "future-year change check, not alignment evidence", False)
        registered, urban, density = STRUCTURE_2020[real_city]
        add(mapping, "registered_population", 2021, value(tables["population"], city_id, 2021, "户籍人口（万人）"), 2020, registered, -1, "SRC_JS_YEARBOOK_2020" if province == "江苏" else anchor_source(province), 1.0, "registered comparator")
        add(mapping, "urbanization_rate", 2021, value(tables["urbanization"], city_id, 2021, "urbanizationRate"), 2020, urban, -1, "SRC_JS_YEARBOOK_2020" if province == "江苏" else anchor_source(province), 1.0, "structural comparator")
        add(mapping, "population_density", 2021, value(tables["density"], city_id, 2021, "人口密度（人/平方公里）"), 2020, density, -1, anchor_source(province), max(5, density * 0.02), "structural comparator")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["alignment_offset"]), row["indicator"])].append(row)
    summaries = []
    for offset in (-2, -1, 0, 1):
        indicators = sorted({indicator for off, indicator in grouped if off == offset} | {"resident_population"})
        for indicator in indicators:
            group = grouped.get((offset, indicator), [])
            diffs = [float(row["absolute_difference"]) for row in group if row["absolute_difference"]]
            rels = [float(row["relative_difference"]) for row in group if row["relative_difference"]]
            yes = [row for row in group if row["supports_alignment"] == "YES"]
            no = [row["city_id"] for row in group if row["supports_alignment"] == "NO"]
            summaries.append({
                "alignment_offset": offset, "indicator": indicator,
                "city_count": len({row["city_id"] for row in group}), "row_count": len(group),
                "exact_or_near_match_rate": display_number(len(yes) / len(group)) if group else "",
                "median_absolute_difference": display_number(median(diffs)) if diffs else "",
                "median_relative_difference": display_number(median(rels)) if rels else "",
                "major_exceptions": ",".join(no[:12]) + ("..." if len(no) > 12 else ""),
            })
    return rows, summaries


def build_semantics(tables: dict[str, Any], identity_summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    conflicts = []
    counts = defaultdict(int)
    for mapping, identity in zip(CITY_MAP, identity_summary):
        city_id, real_city, province = mapping
        p = value(tables["population"], city_id, 2021, "常住人口（万人）")
        r = value(tables["population"], city_id, 2021, "户籍人口（万人）")
        u = value(tables["urbanization"], city_id, 2021, "urbanizationRate")
        d = value(tables["density"], city_id, 2021, "人口密度（人/平方公里）")
        p_real = normalized_2020_wan(real_city)
        r_real, u_real, d_real = STRUCTURE_2020[real_city]
        checks = [
            ("RESIDENT_HIGH_CONFIDENCE", p, p_real, "万人", anchor_source(province), near(p, p_real, 1.0), "resident candidate"),
            ("REGISTERED_HIGH_CONFIDENCE" if near(r, r_real, 1.0) else "REGISTERED_CONFLICT", r, r_real, "万人", "SRC_JS_YEARBOOK_2020" if province == "江苏" else anchor_source(province), near(r, r_real, 1.0), "registered comparator retained separately"),
            ("URBANIZATION_STRUCTURE", u, u_real, "%", "SRC_JS_YEARBOOK_2020" if province == "江苏" else anchor_source(province), near(u, u_real, 1.0), "non-population structure"),
            ("DENSITY_STRUCTURE", d, d_real, "人/平方公里", anchor_source(province) if d is not None else "SRC_COMP_DENSITY", near(d, d_real, max(5, d_real * 0.02)), "non-population structure"),
        ]
        for etype, comp, real, unit, sid, ok, note in checks:
            counts[etype] += int(ok)
            rows.append({
                "city_id": city_id, "real_city": real_city, "province": province,
                "evidence_type": etype, "competition_year": 2021,
                "competition_value": display_number(comp), "real_year": 2020,
                "real_value": display_number(real), "unit": unit, "source_id": sid,
                "supports_or_conflicts": "SUPPORTS" if ok else "CONFLICTS",
                "evidence_strength": "HIGH" if "RESIDENT" in etype or "REGISTERED" in etype else "MEDIUM",
                "notes": note,
            })
        if not near(r, r_real, 1.0):
            conflicts.append({"city_id": city_id, "real_city": real_city, "registered_competition": r, "registered_anchor": r_real})
    summary = {
        "selected_semantics": "RESIDENT_POPULATION",
        "confidence": "HIGH_WITH_EXPLICIT_EXCEPTIONS",
        "target_year_in_competition": TARGET_YEAR,
        "real_statistical_year_for_semantics": 2020,
        "city_count": 40,
        "resident_match_count": counts["RESIDENT_HIGH_CONFIDENCE"],
        "registered_match_count": counts["REGISTERED_HIGH_CONFIDENCE"],
        "urbanization_match_count": counts["URBANIZATION_STRUCTURE"],
        "density_match_count": counts["DENSITY_STRUCTURE"],
        "registered_conflict_count": len(conflicts),
        "registered_conflicts": conflicts,
        "forbidden_model_evidence_used": False,
        "notes": [
            "Resident 2021 competition rows align to 2020 census anchors under the selected +1 label lag.",
            "Registered population remains a separate field; city3 and city40 anomalies are not corrected.",
            "No model performance, MSE, ranking, or S9 artifact was used.",
        ],
    }
    return rows, summary


def truth_source_for(city: str) -> str:
    js = ["南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁"]
    zj = ["杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水"]
    gd_repro = ["珠海", "汕头", "佛山", "韶关", "河源", "湛江", "茂名", "肇庆"]
    if city in js:
        return "SRC_JS_2022_NJ0306"
    if city in zj:
        return "SRC_ZJ_2022_POP_BULLETIN_REPRO"
    if city == "广州":
        return "SRC_GZ_2022_BULLETIN"
    if city == "深圳":
        return "SRC_SZ_2022_BULLETIN"
    if city == "江门":
        return "SRC_JM_2022_BULLETIN"
    if city in gd_repro:
        return "SRC_GD_2022_YEARBOOK_REPRO"
    return {
        "济南": "SRC_JN_2022_BULLETIN", "青岛": "SRC_QD_2022_BULLETIN",
        "淄博": "SRC_ZB_2022_BULLETIN", "枣庄": "SRC_ZZ_2022_BULLETIN_REPRO",
        "东营": "SRC_DY_2022_BULLETIN_REPRO",
    }[city]


def build_truth_raw() -> list[dict[str, Any]]:
    rows = []
    for city_id, city, province in CITY_MAP:
        sid = truth_source_for(city)
        tier = SOURCE_TIER[sid]
        confidence = "EXACT_OFFICIAL" if tier == "TIER1" else "HIGH_CONFIDENCE_REPRODUCED"
        note = ""
        if city_id == "city40":
            confidence = "PROVISIONAL"
            note = "Identity provisional because of retained registered-field anomaly."
        rows.append({
            "city_id": city_id, "real_city": city, "province": province,
            "competition_target_year": TARGET_YEAR, "real_statistical_year": REAL_YEAR,
            "target_semantics": "resident_population", "raw_value": display_number(TRUTH_2022_WAN[city]),
            "raw_unit": "万人", "normalized_value_wan": display_number(TRUTH_2022_WAN[city]),
            "source_id": sid, "source_tier": tier,
            "revision_status": SOURCE_ROWS[[row["source_id"] for row in SOURCE_ROWS].index(sid)]["revision_status"],
            "value_confidence": confidence, "alternate_value": "", "alternate_source_id": "",
            "conflict_note": note,
        })
    return rows


def build_revision_ledger() -> list[dict[str, Any]]:
    rows = []
    js_yearbook = {
        "南京": 931.97, "无锡": 746.40, "徐州": 908.39, "常州": 527.96,
        "苏州": 1274.96, "南通": 772.80, "连云港": 460.10, "淮安": 455.92,
        "盐城": 671.06, "扬州": 456.10, "镇江": 321.10, "泰州": 451.68, "宿迁": 498.82,
    }
    for city_id, city, province in CITY_MAP:
        census = normalized_2020_wan(city)
        rows.append({
            "city": city, "real_year": 2020, "indicator": "resident_population",
            "initial_value": display_number(census), "revised_value": display_number(census),
            "census_value": display_number(census), "difference": "0",
            "revision_type": "CENSUS_BENCHMARK", "source_initial": anchor_source(province),
            "source_revised": anchor_source(province), "selected_value_for_truth": display_number(census),
            "selection_reason": "Exact census benchmark retained for identity and alignment.",
        })
        if province == "江苏":
            revised = js_yearbook[city]
            rows.append({
                "city": city, "real_year": 2020, "indicator": "resident_population",
                "initial_value": display_number(census), "revised_value": display_number(revised),
                "census_value": display_number(census), "difference": display_number(revised - census),
                "revision_type": "REVISED_YEARBOOK_ROUNDING", "source_initial": "SRC_JS_CENSUS_2020",
                "source_revised": "SRC_JS_YEARBOOK_2020", "selected_value_for_truth": display_number(census),
                "selection_reason": "Rounded annual presentation is logged but does not replace the census value.",
            })
        sid = truth_source_for(city)
        rows.append({
            "city": city, "real_year": 2022, "indicator": "resident_population",
            "initial_value": display_number(TRUTH_2022_WAN[city]), "revised_value": display_number(TRUTH_2022_WAN[city]),
            "census_value": "", "difference": "0", "revision_type": SOURCE_ROWS[[row["source_id"] for row in SOURCE_ROWS].index(sid)]["revision_status"],
            "source_initial": sid, "source_revised": sid, "selected_value_for_truth": display_number(TRUTH_2022_WAN[city]),
            "selection_reason": "Captured candidate value; tier and residual uncertainty remain explicit in raw truth and manifest.",
        })
    return rows


def build_conflicts(tables: dict[str, Any]) -> list[dict[str, Any]]:
    conflicts = []
    for city_id, city, province in CITY_MAP:
        comp = value(tables["population"], city_id, 2021, "户籍人口（万人）")
        anchor = STRUCTURE_2020[city][0]
        if comp is not None and abs(comp - anchor) > 15:
            conflicts.append({
                "city_id": city_id, "real_city": city, "province": province,
                "conflict_type": "REGISTERED_FIELD_MISMATCH", "indicator": "registered_population",
                "competition_year": 2021, "real_year": 2020, "primary_value": display_number(comp),
                "alternate_value": display_number(anchor), "primary_source_id": "SRC_COMP_POP",
                "alternate_source_id": "SRC_JS_YEARBOOK_2020" if province == "江苏" else anchor_source(province),
                "severity": "MATERIAL", "resolution_status": "UNRESOLVED_RETAIN_RAW",
                "resolution_note": "Raw competition value is preserved; no city correction is applied.",
            })
    conflicts.extend([
        {
            "city_id": "city12", "real_city": "泰州", "province": "江苏",
            "conflict_type": "CHINESE_NAME_COLLISION", "indicator": "city_name",
            "competition_year": 2021, "real_year": 2020, "primary_value": "泰州",
            "alternate_value": "台州", "primary_source_id": "SRC_COMP_POP",
            "alternate_source_id": "SRC_ZJ_CENSUS_2020", "severity": "MINOR",
            "resolution_status": "RESOLVED_BY_PROVINCE_AND_FINGERPRINT",
            "resolution_note": "Jiangsu Taizhou selected; Zhejiang Taizhou is city29.",
        },
        {
            "city_id": "city29", "real_city": "台州", "province": "浙江",
            "conflict_type": "CHINESE_NAME_COLLISION", "indicator": "city_name",
            "competition_year": 2021, "real_year": 2020, "primary_value": "台州",
            "alternate_value": "泰州", "primary_source_id": "SRC_COMP_POP",
            "alternate_source_id": "SRC_JS_CENSUS_2020", "severity": "MINOR",
            "resolution_status": "RESOLVED_BY_PROVINCE_AND_FINGERPRINT",
            "resolution_note": "Zhejiang Taizhou selected; Jiangsu Taizhou is city12.",
        },
    ])
    return conflicts


def build_rounding(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for city_id, city, province in CITY_MAP:
        raw = normalized_2020_wan(city)
        comp = value(tables["population"], city_id, 2021, "常住人口（万人）")
        nearest = int(math.floor(raw + 0.5))
        trunc = int(math.floor(raw))
        one_decimal = round(raw, 1)
        rows.append({
            "city_id": city_id, "real_city": city, "competition_year": 2021, "real_year": 2020,
            "raw_official_value_persons": RESIDENT_2020_PERSONS[city],
            "raw_official_value_wan": display_number(raw), "competition_value_wan": display_number(comp),
            "nearest_integer_wan": nearest, "truncate_integer_wan": trunc,
            "one_decimal_wan": display_number(one_decimal),
            "nearest_match": "YES" if comp == nearest else "NO",
            "truncate_match": "YES" if comp == trunc else "NO",
            "one_decimal_match": "YES" if comp == one_decimal else "NO",
            "selected_rule": "ROUND_HALF_UP_TO_INTEGER_WAN",
            "source_id": anchor_source(province),
            "notes": "Historical competition rendering evidence; no model metric used.",
        })
    return rows


def build_truth_v1(raw_rows: list[dict[str, Any]], identity_summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity_by_city = {row["city_id"]: row for row in identity_summary}
    full = []
    evaluator = []
    for raw in raw_rows:
        identity = identity_by_city[raw["city_id"]]
        truth = int(math.floor(float(raw["normalized_value_wan"]) + 0.5))
        if identity["mapping_status"] == "PROVISIONAL":
            status, confidence, note = "PROVISIONAL", "PROVISIONAL", "Mapping is provisional because a material registered conflict is retained."
        elif raw["source_tier"] == "TIER1":
            status, confidence, note = "EXACT_OFFICIAL", "EXACT_OFFICIAL", "Primary official 2022 source; integer 万人 rendering is derived."
        else:
            status, confidence, note = "HIGH_CONFIDENCE", "HIGH_CONFIDENCE_REPRODUCED", "Traceable reproduced source; remaining source-tier uncertainty is explicit."
        full.append({
            "city_id": raw["city_id"], "year": TARGET_YEAR, "candidate_truth": truth,
            "raw_real_value": raw["normalized_value_wan"], "real_city": raw["real_city"],
            "province": raw["province"], "truth_status": status,
            "mapping_confidence": identity["mapping_confidence"], "value_confidence": confidence,
            "primary_source_id": raw["source_id"], "alternate_candidate": identity["alternate_candidate"],
            "uncertainty_lower": f"{truth - 0.5:.1f}", "uncertainty_upper": f"{truth + 0.5:.1f}",
            "notes": note,
        })
        evaluator.append({"city_id": raw["city_id"], "year": TARGET_YEAR, "truth": truth})
    return full, evaluator


def git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def report(identity, alignment, semantics, raw, truth, conflicts, commit) -> str:
    mapping_counts = {status: sum(row["mapping_status"] == status for row in identity) for status in sorted(MAPPING_STATUSES)}
    tier_counts = defaultdict(int)
    for row in raw:
        tier_counts[row["source_tier"]] += 1
    return f"""# S8 Truth Reconstruction Report

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE

This is the post-hoc target-forensics track requested by S8-Plan01. It does
not train, select, score, or compare models. The plan is task input; it is
kept separate from the user's short request to continue this line.

## City identity

All 40 competition IDs are mapped with resident trajectory anchors plus
registered population, urbanization, density, and province evidence.

Mapping counts: CONFIRMED={mapping_counts["CONFIRMED"]},
HIGH_CONFIDENCE={mapping_counts["HIGH_CONFIDENCE"]},
PROVISIONAL={mapping_counts["PROVISIONAL"]},
UNRESOLVED={mapping_counts["UNRESOLVED"]}.

city3 has a retained Shenzhen registered-field mismatch. city40 maps to
Zhenjiang, but its 2021 registered value is 65 versus the 2020 anchor 269.25.
No competition raw value is corrected.

## Year alignment

The selected hypothesis is competition year = real statistical year + 1.
Therefore the candidate target year 2023 is reconstructed from real statistical
year 2022. Resident, registered, urbanization, and density evidence supports
the 2021 to 2020 comparison. Offsets -2, -1, 0, and +1 are all recorded.

## Target semantics

Selected semantics: RESIDENT_POPULATION.
Confidence: HIGH_WITH_EXPLICIT_EXCEPTIONS.
Resident matches: {semantics["resident_match_count"]}/40.
Registered matches: {semantics["registered_match_count"]}/40.
Urbanization matches: {semantics["urbanization_match_count"]}/40.
Density matches: {semantics["density_match_count"]}/40.

Registered is retained as an independent comparator. No model performance,
MSE, ranking, or S9 artifact was used.

## Candidate truth and uncertainty

candidate_truth_2023_raw.csv preserves 2022 raw values in 万人. The v1
candidate uses ROUND_HALF_UP_TO_INTEGER_WAN, and the evaluator file is
derived from the full provenance table. Primary raw rows: {tier_counts["TIER1"]};
reproduced-source raw rows: {tier_counts["TIER3"]}.

The freeze is provisional because some 2022 values are Tier-3 reproductions
and because the city40 identity/registered conflict remains explicit.

## Isolation audit

- S9 branch read: NO
- S9 predictions or scoreboard read: NO
- model evaluation performed: NO
- test.xls accessed: NO
- Main worktree modified: NO

## Reproduction

Run python scripts/build_s8_truth_v1.py --input-root <data-root> --repo-root <worktree-root>.
Outputs are under reports/S8_TRUTH/truth_v1.

Artifact commit at generation time: {commit or "not yet committed"}.
"""


def handoff(repo_root: Path, identity, truth, evaluator_path: Path, commit) -> str:
    counts = {status: sum(row["mapping_status"] == status for row in identity) for status in sorted(MAPPING_STATUSES)}
    truth_counts = defaultdict(int)
    for row in truth:
        truth_counts[row["truth_status"]] += 1
    tier1 = sum(row["source_tier"] == "TIER1" for row in SOURCE_ROWS)
    fallback = sum(row["source_tier"] in {"TIER3", "TIER4"} for row in SOURCE_ROWS)
    base = git(repo_root, "merge-base", "HEAD", "origin/main") or git(repo_root, "rev-parse", "HEAD")
    return f"""# S8 Truth Track Handoff

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE
Branch: {git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")}
Base SHA: {base}
Final Commit SHA: {commit or "PENDING_ARTIFACT_COMMIT"}
Working tree at generation: {git(repo_root, "status", "--short") or "clean"}

Truth version: candidate_truth_2023_v1
Truth SHA256 evaluator: {sha256_file(evaluator_path)}

City mapping:
- confirmed: {counts["CONFIRMED"]}
- high confidence: {counts["HIGH_CONFIDENCE"]}
- provisional: {counts["PROVISIONAL"]}
- unresolved: {counts["UNRESOLVED"]}

Year alignment:
- selected hypothesis: competition year = real year + 1
- evidence scope: 40 cities; resident, registered, urbanization, density; four province groups
- exceptions: city3 and city40 registered conflicts; census/yearbook presentation differences

Target semantics:
- result: RESIDENT_POPULATION
- confidence: HIGH_WITH_EXPLICIT_EXCEPTIONS
- conflicts: retained in truth_conflicts.csv

Sources:
- total source ledger rows: {len(SOURCE_ROWS)}
- Tier 1 sources: {tier1}
- fallback source rows: {fallback}

Truth reconstruction:
- resolved rows: {sum(truth_counts.values())}
- provisional rows: {truth_counts["PROVISIONAL"]}
- unresolved rows: {truth_counts["UNRESOLVED"]}
- unit: raw 万人; evaluator integer 万人
- rendering rule: ROUND_HALF_UP_TO_INTEGER_WAN

Isolation:
- S9 branch read: NO
- S9 predictions read: NO
- model evaluation performed: NO
- test.xls accessed: NO

Project documentation drift:
- CURRENT_LEARNING_CHECKPOINT.md and learning documents were not modified.
- Main was not modified or merged.

Return:
S8 truth track remains provisional and waits for Main review. Do not feed the
truth values to S9 from this worker.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path.cwd(), help="Read-only repository/data root containing data/normalized")
    parser.add_argument("--repo-root", type=Path, default=None, help="Git/output root; defaults to this script's worktree")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--final-commit", default="")
    args = parser.parse_args()
    data_root = args.input_root.resolve()
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    output_dir = (args.output_dir or repo_root / OUTPUT_RELATIVE_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(data_root)
    missing = [city_id for city_id in EXPECTED_CITY_IDS if city_id not in tables["population"]]
    if missing:
        raise RuntimeError(f"Missing competition population rows: {missing}")

    fingerprints = build_fingerprints(tables)
    identity_evidence, identity_summary = build_identity(tables)
    alignment_evidence, alignment_summary = build_alignment(tables)
    semantics_evidence, semantics_summary = build_semantics(tables, identity_summary)
    raw_truth = build_truth_raw()
    revision_ledger = build_revision_ledger()
    conflicts = build_conflicts(tables)
    rounding = build_rounding(tables)
    truth_v1, evaluator = build_truth_v1(raw_truth, identity_summary)

    write_csv(output_dir / "competition_city_fingerprints.csv", list(fingerprints[0]), fingerprints)
    write_csv(output_dir / "source_ledger.csv", list(SOURCE_ROWS[0]), SOURCE_ROWS)
    write_csv(output_dir / "city_identity_evidence.csv", list(identity_evidence[0]), identity_evidence)
    write_csv(output_dir / "city_identity_mapping.csv", list(identity_summary[0]), identity_summary)
    write_csv(output_dir / "year_alignment_evidence.csv", list(alignment_evidence[0]), alignment_evidence)
    write_csv(output_dir / "year_alignment_summary.csv", list(alignment_summary[0]), alignment_summary)
    write_csv(output_dir / "target_semantics_evidence.csv", list(semantics_evidence[0]), semantics_evidence)
    (output_dir / "target_semantics_summary.json").write_text(json.dumps(semantics_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "statistical_revision_ledger.csv", list(revision_ledger[0]), revision_ledger)
    write_csv(output_dir / "candidate_truth_2023_raw.csv", list(raw_truth[0]), raw_truth)
    write_csv(output_dir / "truth_conflicts.csv", list(conflicts[0]), conflicts)
    write_csv(output_dir / "rounding_unit_evidence.csv", list(rounding[0]), rounding)
    write_csv(output_dir / "candidate_truth_2023_v1.csv", list(truth_v1[0]), truth_v1)
    evaluator_path = output_dir / "candidate_truth_2023_v1_evaluator.csv"
    write_csv(evaluator_path, list(evaluator[0]), evaluator)

    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    manifest = {
        "truth_version": "candidate_truth_2023_v1",
        "plan_id": PLAN_ID,
        "base_commit": git(repo_root, "merge-base", "HEAD", "origin/main") or git(repo_root, "rev-parse", "HEAD"),
        "branch": git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "generation_script": "scripts/build_s8_truth_v1.py",
        "generated_at": generated_at,
        "competition_target_year": TARGET_YEAR,
        "real_statistical_year": REAL_YEAR,
        "target_semantics": "RESIDENT_POPULATION",
        "unit": "raw 万人; evaluator integer 万人",
        "rounding_rule": "ROUND_HALF_UP_TO_INTEGER_WAN",
        "city_count": 40,
        "resolved_city_count": sum(row["truth_status"] != "UNRESOLVED" for row in truth_v1),
        "provisional_city_count": sum(row["truth_status"] == "PROVISIONAL" for row in truth_v1),
        "unresolved_city_count": sum(row["truth_status"] == "UNRESOLVED" for row in truth_v1),
        "source_count": len(SOURCE_ROWS),
        "tier1_source_count": sum(row["source_tier"] == "TIER1" for row in SOURCE_ROWS),
        "fallback_source_count": sum(row["source_tier"] in {"TIER3", "TIER4"} for row in SOURCE_ROWS),
        "truth_file": "candidate_truth_2023_v1_evaluator.csv",
        "truth_sha256": sha256_file(evaluator_path),
        "full_truth_file": "candidate_truth_2023_v1.csv",
        "full_truth_sha256": sha256_file(output_dir / "candidate_truth_2023_v1.csv"),
        "evidence_files": [
            "competition_city_fingerprints.csv", "source_ledger.csv",
            "city_identity_evidence.csv", "city_identity_mapping.csv",
            "year_alignment_evidence.csv", "year_alignment_summary.csv",
            "target_semantics_evidence.csv", "target_semantics_summary.json",
            "statistical_revision_ledger.csv", "candidate_truth_2023_raw.csv",
            "truth_conflicts.csv", "rounding_unit_evidence.csv",
            "candidate_truth_2023_v1.csv", "candidate_truth_2023_v1_evaluator.csv",
        ],
        "known_conflicts": [
            "city3 registered population mismatch retained",
            "city40 registered population 2021 value 65 retained and mapping provisional",
            "city12/city29 Taizhou name collision resolved by province and fingerprint",
            "Tier-3 2022 values keep freeze provisional",
        ],
        "usage_scope": ["POST_HOC_HIDDEN_EVALUATOR_ONLY", "NOT_FOR_MODEL_SELECTION", "NOT_SHARED_WITH_S9"],
        "freeze_status": "S8_TRUTH_V1_PROVISIONAL_FREEZE",
        "isolation": {"s9_branch_read": False, "s9_predictions_read": False, "model_evaluation_performed": False, "test_xls_accessed": False},
    }
    (output_dir / "truth_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "S8_TRUTH_RECONSTRUCTION_REPORT.md").write_text(report(identity_summary, alignment_summary, semantics_summary, raw_truth, truth_v1, conflicts, args.final_commit), encoding="utf-8")
    (output_dir / "HANDOFF.md").write_text(handoff(repo_root, identity_summary, truth_v1, evaluator_path, args.final_commit), encoding="utf-8")

    if [row["city_id"] for row in evaluator] != EXPECTED_CITY_IDS:
        raise RuntimeError("Evaluator city order is not city1..city40")
    if len({(row["city_id"], row["year"]) for row in evaluator}) != 40:
        raise RuntimeError("Evaluator has duplicate keys")
    if any(row["truth"] == "" for row in evaluator):
        raise RuntimeError("Evaluator has missing truth")
    if any(row["mapping_status"] not in MAPPING_STATUSES for row in identity_summary):
        raise RuntimeError("Invalid mapping status")
    if any(row["source_id"] not in SOURCE_TIER for row in raw_truth):
        raise RuntimeError("Truth source missing from source ledger")
    print(json.dumps({
        "output_dir": str(output_dir),
        "city_count": 40,
        "truth_sha256": manifest["truth_sha256"],
        "mapping_status_counts": {status: sum(row["mapping_status"] == status for row in identity_summary) for status in sorted(MAPPING_STATUSES)},
        "truth_status_counts": {status: sum(row["truth_status"] == status for row in truth_v1) for status in ["EXACT_OFFICIAL", "HIGH_CONFIDENCE", "PROVISIONAL", "UNRESOLVED"]},
        "freeze_status": manifest["freeze_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
