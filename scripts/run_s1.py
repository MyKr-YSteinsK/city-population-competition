"""Rebuild and freeze the source-level S1 data contract.

This runner deliberately stops at source coverage, canonicalization,
quarantine/provenance, reproducibility, and handoff.  It does not perform
feature engineering, validation design, modeling, prediction, or submission.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
RAW_ROOT = ROOT / "data" / "raw"
NORMALIZED_DIR = ROOT / "data" / "normalized"
QUARANTINE_DIR = ROOT / "data" / "quarantine"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "reports"
PACK_PATH = REPORTS_DIR / "S1_FINAL_REVIEW_PACK.zip"

DOMAIN_SCRIPTS = [
    "audit_population.py",
    "audit_population_density.py",
    "audit_urbanization.py",
    "audit_employment.py",
    "audit_wage.py",
    "audit_age_structure.py",
    "audit_living_standard.py",
]

PRIMARY_REPAIR_CITIES = {
    "city1",
    "city24",
    "city28",
    "city32",
    "city36",
    "city39",
}
KNOWN_SECTOR_CONFLICT_KEYS = {
    ("city3", 2018),
    ("city3", 2019),
    ("city6", 2018),
    ("city6", 2019),
    ("city9", 2018),
    ("city9", 2019),
    ("city16", 2019),
    ("city17", 2019),
    ("city23", 2019),
    ("city23", 2020),
    ("city27", 2019),
    ("city31", 2019),
    ("city31", 2020),
    ("city35", 2019),
    ("city35", 2020),
    ("city38", 2019),
    ("city38", 2020),
    ("city22", 2020),
}

SPECS: list[dict[str, Any]] = [
    {
        "dataset_id": "population",
        "raw_file": "人口规模.xlsx",
        "raw_sheet": "Sheet1",
        "canonical_file": "人口规模.csv",
        "value_columns": ["常住人口（万人）", "户籍人口（万人）"],
        "grain": "city × year × population indicators",
        "status": "source_plus_sentinel_repair",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "Raw zero sentinels are canonicalized to missing; source-specific coverage gaps are preserved.",
        "downstream": "Do not infer missing population magnitudes from other tables in S1.",
    },
    {
        "dataset_id": "density",
        "raw_file": "人口密度.xlsx",
        "raw_sheet": "Sheet1",
        "canonical_file": "人口密度.csv",
        "value_columns": ["人口密度（人/平方公里）"],
        "grain": "city × year × density",
        "status": "source_plus_repair_and_invalidation",
        "repair_count": 1,
        "invalidated_count": 13,
        "derived_count": 0,
        "quarantine_file": "人口密度_quarantine.csv",
        "provenance_file": "",
        "warning": "13 pre-regime observations are quarantined; 13 persistent regime reports are retained.",
        "downstream": "Keep regime boundaries and missing invalidations explicit; do not treat every local jump as an error.",
    },
    {
        "dataset_id": "urbanization",
        "raw_file": "城镇化率.xlsx",
        "raw_sheet": "Sheet1",
        "canonical_file": "城镇化率.csv",
        "value_columns": ["urbanizationRate"],
        "grain": "city × year × urbanization rate",
        "status": "source_preserved",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "city3 has a persistent 99-to-100 plateau; no source-level invalidation was justified.",
        "downstream": "Use the observed rate and its coverage; publication timing is unknown.",
    },
    {
        "dataset_id": "unemployment",
        "raw_file": "就业信息.xlsx",
        "raw_sheet": "城镇失业率",
        "canonical_file": "就业信息_城镇失业率.csv",
        "value_columns": ["unemploymentRate"],
        "grain": "city × year × unemployment rate",
        "status": "source_preserved",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "Ten low-cardinality raw source-block similarities remain unresolved but are not semantically unique.",
        "downstream": "Do not relabel blocks from sequence similarity alone.",
    },
    {
        "dataset_id": "employees",
        "raw_file": "就业信息.xlsx",
        "raw_sheet": "从业人员数",
        "canonical_file": "就业信息_从业人员数.csv",
        "value_columns": ["employeesNumber"],
        "grain": "city × year × employees number",
        "status": "source_plus_remap_quarantine_and_derived",
        "repair_count": 4,
        "invalidated_count": 80,
        "derived_count": 91,
        "quarantine_file": "就业信息_从业人员数_quarantine.csv",
        "provenance_file": "就业信息_从业人员数_provenance.csv",
        "warning": "91 rows are explicit sector-sum reconstructions with provenance; 80 source rows remain quarantined.",
        "downstream": "Keep source-derived and reconstructed rows distinguishable in any later information set.",
    },
    {
        "dataset_id": "sector",
        "raw_file": "就业信息.xlsx",
        "raw_sheet": "第一、二、三产业就业人数",
        "canonical_file": "就业信息_第一二三产业就业人数.csv",
        "value_columns": ["pi_Employment", "si_Employment", "ti_Employment"],
        "grain": "city × year × three sector employment indicators",
        "status": "source_plus_repair_and_invalidation",
        "repair_count": 103,
        "invalidated_count": 18,
        "derived_count": 0,
        "quarantine_file": "就业信息_第一二三产业就业人数_quarantine.csv",
        "provenance_file": "",
        "warning": "103 primary-sector components are repaired by an invariant; 18 conflicting component rows are quarantined.",
        "downstream": "Do not reuse the repair relation as independent validation without excluding its repair boundary.",
    },
    {
        "dataset_id": "wage",
        "raw_file": "工资水平.xlsx",
        "raw_sheet": "职工平均工资",
        "canonical_file": "工资水平_职工平均工资.csv",
        "value_columns": ["averageWage"],
        "grain": "city × year × average wage",
        "status": "source_plus_invalidation",
        "repair_count": 0,
        "invalidated_count": 1,
        "derived_count": 0,
        "quarantine_file": "工资水平_职工平均工资_quarantine.csv",
        "provenance_file": "",
        "warning": "One high-confidence isolated observation is invalidated while its true value remains unknown.",
        "downstream": "Preserve missingness; do not fill the invalidated value from a model in S1.",
    },
    {
        "dataset_id": "age",
        "raw_file": "年龄结构.xlsx",
        "raw_sheet": "Sheet1",
        "canonical_file": "年龄结构.csv",
        "value_columns": ["0-14", "15-64", "65+"],
        "grain": "city × benchmark year × age-band indicators",
        "status": "source_plus_invalidation",
        "repair_count": 0,
        "invalidated_count": 6,
        "derived_count": 0,
        "quarantine_file": "年龄结构_quarantine.csv",
        "provenance_file": "",
        "warning": "The domain is benchmark years {2000, 2010, 2020}; only city25-2000 and city26-2000 are missing keys; six 2000 observations are invalidated at cell level.",
        "downstream": "Do not create annual age rows or interpret non-benchmark years as missing observations.",
    },
    {
        "dataset_id": "overall_income",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "人均可支配收入",
        "canonical_file": "生活水平_人均可支配收入.csv",
        "value_columns": ["disposableIncome"],
        "grain": "city × year × overall disposable income",
        "status": "source_plus_invalidation",
        "repair_count": 0,
        "invalidated_count": 32,
        "derived_count": 0,
        "quarantine_file": "生活水平_人均可支配收入_quarantine.csv",
        "provenance_file": "",
        "warning": "32 values are quarantined as block contamination with unknown true values; city3 shift semantics remain unresolved.",
        "downstream": "Do not treat overall and urban income as two independent signals without an information-set review.",
    },
    {
        "dataset_id": "overall_consumption",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "人均消费支出",
        "canonical_file": "生活水平_人均消费支出.csv",
        "value_columns": ["consumptionExpenditures"],
        "grain": "city × year × overall consumption expenditure",
        "status": "source_plus_repair",
        "repair_count": 1,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "city21-2022 is repaired from 446007 to 44607; weighted residual candidates remain preserved.",
        "downstream": "Retain the repair report and do not assume a universal urban/rural accounting identity.",
    },
    {
        "dataset_id": "urban_consumption",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "城镇居民消费支出",
        "canonical_file": "生活水平_城镇居民消费支出.csv",
        "value_columns": ["urbanConsumptionExpenditures"],
        "grain": "city × year × urban resident consumption expenditure",
        "status": "source_preserved",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "Coverage is source-defined and asymmetric across cities.",
        "downstream": "Do not equate missing rural/urban rows with zero.",
    },
    {
        "dataset_id": "rural_consumption",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "农村居民消费支出",
        "canonical_file": "生活水平_农村居民消费支出.csv",
        "value_columns": ["ruralConsumptionExpenditures"],
        "grain": "city × year × rural resident consumption expenditure",
        "status": "source_preserved",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "city3 has no rural consumption row in the raw source.",
        "downstream": "Keep absent coverage distinct from a numeric zero or an imputed rural value.",
    },
    {
        "dataset_id": "urban_income",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "城镇居民人均收入",
        "canonical_file": "生活水平_城镇居民人均收入.csv",
        "value_columns": ["urbanDisposableIncome"],
        "grain": "city × year × urban resident disposable income",
        "status": "source_plus_label_repair",
        "repair_count": 2,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "Two city-label repairs are documented; city3 has a blank 2022 cell and a shifted-copy-like sequence.",
        "downstream": "The observed sequence is not proof of publication-time availability or independent signal content.",
    },
    {
        "dataset_id": "rural_income",
        "raw_file": "生活水平.xlsx",
        "raw_sheet": "农村居民人均收入",
        "canonical_file": "生活水平_农村居民人均收入.csv",
        "value_columns": ["ruralDisposableIncome"],
        "grain": "city × year × rural resident disposable income",
        "status": "source_preserved",
        "repair_count": 0,
        "invalidated_count": 0,
        "derived_count": 0,
        "quarantine_file": "",
        "provenance_file": "",
        "warning": "city3 has no rural income row in the raw source.",
        "downstream": "Do not infer absent rural coverage from overall or urban income.",
    },
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes_under(directory: Path, suffixes: set[str]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    }


def output_hashes() -> dict[str, str]:
    return hashes_under(NORMALIZED_DIR, {".csv"}) | hashes_under(QUARANTINE_DIR, {".csv"})


def read_canonical(spec: dict[str, Any]) -> pd.DataFrame:
    path = NORMALIZED_DIR / spec["canonical_file"]
    assert path.exists(), path
    return pd.read_csv(path)


def raw_profile(spec: dict[str, Any], canonical: pd.DataFrame) -> dict[str, Any]:
    path = RAW_DIR / spec["raw_file"]
    raw = pd.read_excel(path, sheet_name=spec["raw_sheet"])
    direct = {"城市名称", "年份"}.issubset(raw.columns)
    if direct:
        city_count = int(raw["城市名称"].nunique())
        year_min = int(raw["年份"].min())
        year_max = int(raw["年份"].max())
        missing_cells = int(raw[spec["value_columns"]].isna().sum().sum())
        duplicate_keys = int(raw.duplicated(["城市名称", "年份"]).sum())
    else:
        city_count = int(canonical["城市名称"].nunique())
        year_min = int(canonical["年份"].min())
        year_max = int(canonical["年份"].max())
        missing_cells = ""
        duplicate_keys = ""
    return {
        "row_count": int(len(raw)),
        "city_count": city_count,
        "year_min": year_min,
        "year_max": year_max,
        "missing_cells": missing_cells,
        "duplicate_keys": duplicate_keys,
    }


def canonical_profile(spec: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_keys = int(frame.duplicated(["城市名称", "年份"]).sum())
    return {
        "row_count": int(len(frame)),
        "city_count": int(frame["城市名称"].nunique()),
        "year_min": int(frame["年份"].min()),
        "year_max": int(frame["年份"].max()),
        "missing_cells": int(frame[spec["value_columns"]].isna().sum().sum()),
        "duplicate_keys": duplicate_keys,
        "key_count": int(len(set(zip(frame["城市名称"], frame["年份"].astype(int))))),
    }


def contract_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        canonical = read_canonical(spec)
        raw = raw_profile(spec, canonical)
        profile = canonical_profile(spec, canonical)
        if spec["dataset_id"] == "age":
            expected_years = {2000, 2010, 2020}
            expected_keys = {(f"city{i}", year) for i in range(1, 41) for year in expected_years}
            actual_keys = set(zip(canonical["城市名称"], canonical["年份"].astype(int)))
            missing = sorted(expected_keys - actual_keys)
            assert missing == [("city25", 2000), ("city26", 2000)]
            year_domain = "2000;2010;2020"
            expected_key_count = 120
            missing_summary = (
                "benchmark years={2000,2010,2020}; expected keys=120; actual keys=118; "
                "missing=city25-2000,city26-2000; non-benchmark years are outside domain"
            )
        else:
            year_domain = f"observed source range {profile['year_min']}-{profile['year_max']}"
            expected_key_count = ""
            missing_summary = (
                f"observed canonical keys={profile['key_count']}; source-defined coverage; "
                "no complete city-year grid assumed"
            )
        rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "raw_file": f"data/raw/城市人口分析与预测/数据集/{spec['raw_file']}",
                "raw_sheet": spec["raw_sheet"],
                "canonical_file": f"data/normalized/{spec['canonical_file']}",
                "key_columns": "城市名称;年份",
                "value_columns": ";".join(spec["value_columns"]),
                "grain": spec["grain"],
                "row_count": profile["row_count"],
                "raw_row_count": raw["row_count"],
                "city_count": profile["city_count"],
                "year_min": profile["year_min"],
                "year_max": profile["year_max"],
                "year_domain": year_domain,
                "expected_key_count": expected_key_count,
                "actual_key_count": profile["key_count"],
                "missing_key_summary": missing_summary,
                "canonical_missing_key_summary": missing_summary,
                "missing_cell_count": profile["missing_cells"],
                "raw_missing_cell_count": raw["missing_cells"],
                "source_status": spec["status"],
                "repair_count": spec["repair_count"],
                "invalidated_count": spec["invalidated_count"],
                "derived_count": spec["derived_count"],
                "quarantine_file": (
                    f"data/quarantine/{spec['quarantine_file']}"
                    if spec["quarantine_file"]
                    else ""
                ),
                "provenance_file": (
                    f"data/normalized/{spec['provenance_file']}"
                    if spec["provenance_file"]
                    else ""
                ),
                "regime_or_warning": spec["warning"],
                "observed_coverage_status": (
                    f"raw_rows={raw['row_count']}; raw_cities={raw['city_count']}; "
                    f"raw_years={raw['year_min']}-{raw['year_max']}; "
                    f"raw_duplicate_key_rows={raw['duplicate_keys']}; "
                    f"canonical_rows={profile['row_count']}; "
                    f"canonical_duplicate_key_rows={profile['duplicate_keys']}"
                ),
                "forecast_origin_availability": "unknown from source materials",
                "downstream_note": spec["downstream"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_rows() -> list[dict[str, str]]:
    return [
        {
            "issue_id": "QI-001",
            "scope": "source contamination / shift semantics",
            "dataset": "overall_income + urban_income",
            "sheet": "人均可支配收入 ↔ 城镇居民人均收入",
            "affected_keys_or_coverage": "city3; overall 2015-2022; urban 2014-2021 observed; urban 2022 blank",
            "status": "high-priority unresolved",
            "evidence": "overall(t) vs urban(t-1): shift=-1, overlap=8, within-5=8/8",
            "decision": "canonical preserved; no repair, invalidation, derive, or quarantine change",
            "downstream_warning": "Do not treat overall and urban income as independent trusted signals without an information-set review.",
            "source_ref": "raw 生活水平.xlsx; data/normalized/生活水平_人均可支配收入.csv",
        },
        {
            "issue_id": "QI-002",
            "scope": "regime semantics",
            "dataset": "density",
            "sheet": "人口密度",
            "affected_keys_or_coverage": "13 retained regime cities; known invalidation boundary is 13 keys",
            "status": "preserved quality warning",
            "evidence": "A-B-A candidates and persistent post-shift regimes were rechecked; regime report remains retained.",
            "decision": "retain regime report and current canonical boundary",
            "downstream_warning": "Do not convert persistent shifts or all extra candidates into new invalidations without independent evidence.",
            "source_ref": "data/normalized/人口密度_regime_report.csv",
        },
        {
            "issue_id": "QI-003",
            "scope": "invalidated truth",
            "dataset": "density; employees; sector; wage; age; overall_income",
            "sheet": "multiple",
            "affected_keys_or_coverage": "13 + 80 + 18 + 1 + 6 + 32 quarantined observations",
            "status": "unresolved true values",
            "evidence": "Confirmed invalid or conflicting observations remain in quarantine with source context.",
            "decision": "canonical values remain missing where invalidated",
            "downstream_warning": "Do not silently interpolate, backfill, or treat invalidated values as zero.",
            "source_ref": "data/quarantine/*.csv",
        },
        {
            "issue_id": "QI-004",
            "scope": "source coverage",
            "dataset": "population",
            "sheet": "人口规模",
            "affected_keys_or_coverage": "40-city source grid with resident missing=25 and registered missing=10",
            "status": "preserved coverage gap",
            "evidence": "Population canonical rows and zero-sentinel assertion were reproduced.",
            "decision": "preserve source-specific missingness",
            "downstream_warning": "Do not reconstruct population values at source-normalization stage.",
            "source_ref": "data/normalized/人口规模.csv",
        },
        {
            "issue_id": "QI-005",
            "scope": "low-cardinality sequence similarity",
            "dataset": "unemployment",
            "sheet": "城镇失业率",
            "affected_keys_or_coverage": "10 qualifying raw source-block matches",
            "status": "preserved quality warning",
            "evidence": "Repeated 0/1/2/3/4 patterns do not identify a unique semantic relabeling.",
            "decision": "preserve current canonical values",
            "downstream_warning": "Do not relabel or drop blocks from sequence similarity alone.",
            "source_ref": "raw 就业信息.xlsx[城镇失业率]",
        },
        {
            "issue_id": "QI-006",
            "scope": "provenance / derived rows",
            "dataset": "employees",
            "sheet": "从业人员数",
            "affected_keys_or_coverage": "91 derived rows with provenance",
            "status": "documented derived input",
            "evidence": "Independent employee-sector invariant is 484/484 within absolute difference 1 after exclusions.",
            "decision": "retain derived rows and provenance; exclude them from circular independent validation",
            "downstream_warning": "Later modeling must decide whether source-derived rows belong in the forecast information set.",
            "source_ref": "data/normalized/就业信息_从业人员数_provenance.csv",
        },
        {
            "issue_id": "QI-007",
            "scope": "weighted residual context",
            "dataset": "overall_consumption",
            "sheet": "人均消费支出",
            "affected_keys_or_coverage": "city14-2021",
            "status": "preserved boundary candidate",
            "evidence": "overall=16000, urban=18320, rural=16020, urbanization=49; weighted residual is not a universal identity.",
            "decision": "preserve; no canonical change",
            "downstream_warning": "Weighted urban/rural arithmetic is a control, not a universal accounting identity.",
            "source_ref": "data/normalized/生活水平_repair_report.csv",
        },
    ]


def assert_contract_facts(rows: list[dict[str, Any]]) -> None:
    by_id = {row["dataset_id"]: row for row in rows}
    assert by_id["overall_income"]["repair_count"] == 0
    assert by_id["overall_income"]["source_status"] == "source_plus_invalidation"
    assert by_id["overall_income"]["invalidated_count"] == 32
    assert by_id["urban_income"]["repair_count"] == 2
    assert by_id["urban_income"]["source_status"] == "source_plus_label_repair"
    age = by_id["age"]
    assert age["year_domain"] == "2000;2010;2020"
    assert age["expected_key_count"] == 120
    assert age["actual_key_count"] == 118
    assert "city25-2000,city26-2000" in age["missing_key_summary"]
    assert "non-benchmark years are outside domain" in age["missing_key_summary"]


def assert_data_state() -> dict[str, Any]:
    expected_canonical = {spec["canonical_file"] for spec in SPECS}
    actual_canonical = {path.name for path in NORMALIZED_DIR.glob("*.csv") if "report" not in path.name and "provenance" not in path.name and "regime" not in path.name}
    assert actual_canonical == expected_canonical, sorted(actual_canonical ^ expected_canonical)
    assert not list(NORMALIZED_DIR.glob("*.zip")), "generated normalized zip must not remain"
    assert not list(QUARANTINE_DIR.glob("*.zip")), "generated quarantine zip must not remain"
    assert not (RAW_DIR / "test.xls").exists()
    assert (QUARANTINE_DIR / "source" / "test.xls").exists()
    assert {path.name for path in RAW_DIR.glob("*.xlsx")} == {spec["raw_file"] for spec in SPECS if spec["raw_file"] != "就业信息.xlsx" and spec["raw_file"] != "生活水平.xlsx"} | {"就业信息.xlsx", "生活水平.xlsx"}

    for spec in SPECS:
        frame = read_canonical(spec)
        assert not frame.duplicated(["城市名称", "年份"]).any(), spec["canonical_file"]

    quarantine_expected = {
        "人口密度_quarantine.csv": 13,
        "就业信息_从业人员数_quarantine.csv": 80,
        "就业信息_第一二三产业就业人数_quarantine.csv": 18,
        "工资水平_职工平均工资_quarantine.csv": 1,
        "年龄结构_quarantine.csv": 6,
        "生活水平_人均可支配收入_quarantine.csv": 32,
    }
    for name, count in quarantine_expected.items():
        frame = pd.read_csv(QUARANTINE_DIR / name)
        assert len(frame) == count, (name, len(frame), count)

    provenance = pd.read_csv(NORMALIZED_DIR / "就业信息_从业人员数_provenance.csv")
    assert len(provenance) == 91
    derived_error = (
        provenance["employeesNumber"]
        - provenance[["pi_Employment", "si_Employment", "ti_Employment"]].sum(axis=1)
    ).abs()
    assert derived_error.max() == 0

    population = pd.read_csv(NORMALIZED_DIR / "人口规模.csv")
    population_values = population[["常住人口（万人）", "户籍人口（万人）"]]
    assert int((population_values == 0).sum().sum()) == 0

    consumption = pd.read_csv(NORMALIZED_DIR / "生活水平_人均消费支出.csv")
    city21 = consumption[(consumption["城市名称"] == "city21") & (consumption["年份"] == 2022)]
    assert len(city21) == 1 and float(city21.iloc[0]["consumptionExpenditures"]) == 44607

    age = pd.read_csv(NORMALIZED_DIR / "年龄结构.csv")
    age_keys = set(zip(age["城市名称"], age["年份"].astype(int)))
    expected_age_keys = {(f"city{i}", year) for i in range(1, 41) for year in {2000, 2010, 2020}}
    assert age_keys == expected_age_keys - {("city25", 2000), ("city26", 2000)}
    assert len(age_keys) == 118

    employees = pd.read_csv(NORMALIZED_DIR / "就业信息_从业人员数.csv")
    provenance_keys = set(zip(provenance["城市名称"], provenance["年份"].astype(int)))
    source_employees = employees.loc[
        ~employees.apply(
            lambda row: (row["城市名称"], int(row["年份"])) in provenance_keys,
            axis=1,
        )
    ].copy()
    raw_sector = pd.read_excel(RAW_DIR / "就业信息.xlsx", sheet_name="第一、二、三产业就业人数")
    raw_sector["sector_total"] = raw_sector[["pi_Employment", "si_Employment", "ti_Employment"]].sum(axis=1, min_count=3)
    merged = source_employees.merge(raw_sector, on=["城市名称", "年份"], how="inner", validate="one_to_one")
    merged["key"] = list(zip(merged["城市名称"], merged["年份"].astype(int)))
    merged["difference"] = merged["employeesNumber"] - merged["sector_total"]
    independent = merged[
        ~merged["城市名称"].isin(PRIMARY_REPAIR_CITIES)
        & ~merged["key"].isin(KNOWN_SECTOR_CONFLICT_KEYS)
    ]
    assert len(merged) == 605 and len(independent) == 484
    assert (independent["difference"].abs() <= 1).all()

    rows = contract_rows()
    assert_contract_facts(rows)
    return {
        "canonical_tables": len(SPECS),
        "employee_provenance_rows": len(provenance),
        "employee_invariant": "484/484 within absolute difference 1",
        "density_invalidated": 13,
        "density_regime_rows": len(pd.read_csv(NORMALIZED_DIR / "人口密度_regime_report.csv")),
        "wage_invalidated": 1,
        "age_invalidated": 6,
        "living_income_invalidated": 32,
        "population_zero_sentinels": int((population_values == 0).sum().sum()),
        "city21_consumption_2022": float(city21.iloc[0]["consumptionExpenditures"]),
        "contract_rows": rows,
    }


def run_domain_audits() -> None:
    for name in DOMAIN_SCRIPTS:
        path = ROOT / "scripts" / name
        assert path.exists(), path
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


def build_pack() -> list[str]:
    members = {
        "README.md": ROOT / "README.md",
        "CODEX_INSTRUCTIONS.md": DOCS_DIR / "CODEX_INSTRUCTIONS.md",
        "PROJECT_CONTEXT.md": DOCS_DIR / "PROJECT_CONTEXT.md",
        "S1_FINAL_STATE.md": DOCS_DIR / "S1_FINAL_STATE.md",
        "S1_DATA_CONTRACT.md": DOCS_DIR / "S1_DATA_CONTRACT.md",
        "S1_MAIN_HANDOFF.md": DOCS_DIR / "S1_MAIN_HANDOFF.md",
        "PROJECT_FILE_MAP.md": DOCS_DIR / "PROJECT_FILE_MAP.md",
        "s1_data_contract.csv": DOCS_DIR / "s1_data_contract.csv",
        "s1_quality_issue_registry.csv": DOCS_DIR / "s1_quality_issue_registry.csv",
    }
    for path in members.values():
        assert path.exists(), path
    payloads = {name: path.read_bytes() for name, path in sorted(members.items())}
    manifest_lines = [
        "# S1 Final Review Pack Manifest",
        "",
        "This pack contains only current S1 truth and machine-readable contract/quality registries.",
        "It excludes historical Plan01-04 reports and all modeling or validation artifacts.",
        "",
        "| member | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for name, data in payloads.items():
        manifest_lines.append(f"| `{name}` | {len(data)} | `{hashlib.sha256(data).hexdigest()}` |")
    payloads["MANIFEST.md"] = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if PACK_PATH.exists():
        PACK_PATH.unlink()
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(PACK_PATH) as archive:
        assert archive.namelist() == sorted(payloads)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    return sorted(payloads)


def main() -> None:
    raw_before = hashes_under(RAW_ROOT, {".xlsx", ".pdf"})
    output_before = output_hashes()
    state = assert_data_state()
    run_domain_audits()
    raw_after = hashes_under(RAW_ROOT, {".xlsx", ".pdf"})
    output_after = output_hashes()
    assert raw_before == raw_after, "raw source changed during rebuild"
    assert output_before == output_after, "canonical/quarantine/provenance/regime output changed during rebuild"

    write_csv(DOCS_DIR / "s1_data_contract.csv", state["contract_rows"])
    write_csv(DOCS_DIR / "s1_quality_issue_registry.csv", quality_rows())
    pack_members = build_pack()
    print("S1_REBUILD_PASS")
    print(f"canonical_tables={state['canonical_tables']}")
    print(f"employee_provenance_rows={state['employee_provenance_rows']}")
    print(f"employee_invariant={state['employee_invariant']}")
    print(f"density_invalidated={state['density_invalidated']}")
    print(f"density_regime_rows={state['density_regime_rows']}")
    print(f"wage_invalidated={state['wage_invalidated']}")
    print(f"age_invalidated={state['age_invalidated']}")
    print(f"living_income_invalidated={state['living_income_invalidated']}")
    print(f"population_zero_sentinels={state['population_zero_sentinels']}")
    print(f"city21_consumption_2022={state['city21_consumption_2022']:.0f}")
    print("raw_hash_equal=True")
    print("output_hash_equal=True")
    print(f"pack_members={len(pack_members)}")
    print(f"pack_sha256={sha256_file(PACK_PATH)}")


if __name__ == "__main__":
    main()
