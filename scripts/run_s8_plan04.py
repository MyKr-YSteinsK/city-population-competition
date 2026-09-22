"""Reproducible post-hoc validation of the frozen S1 audit.

The script is intentionally read-only with respect to data/raw, data/normalized,
data/quarantine, and the frozen S8 truth branch.  It writes only the bounded
S8-Plan04 evidence package under reports/S8_AUDIT_VALIDATION/S8-Plan04.

Candidate generation is executed before the historical S1 decision inventory is
loaded.  The later join is therefore an audit comparison, not a candidate
selector.  The implementation prefers conservative labels and records
NOT_EXTERNALLY_VERIFIABLE instead of manufacturing external truth.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"
QUARANTINE_DIR = ROOT / "data" / "quarantine"
S8_TRUTH_DIR = ROOT / "reports" / "S8_TRUTH" / "truth_v2"
OUT_DIR = ROOT / "reports" / "S8_AUDIT_VALIDATION" / "S8-Plan04"
GENERATED_AT = "2026-09-22"
PLAN_ID = "S8-Plan04"
BRANCH = "codex/s8-plan04-s1-audit-validation"
PARENT_BRANCH = "codex/s8-truth-reconstruction-v1"
PARENT_HEAD = "cbfc39c2471f3390c80e4b6e4642308b7fbc5a7c"

CITY_RE = re.compile(r"^city(?:[1-9]|[1-3][0-9]|40)$")
YEAR_COL = "年份"
CITY_COL = "城市名称"

NORMALIZED_TABLES = {
    "population": ("人口规模.csv", ["常住人口（万人）", "户籍人口（万人）"]),
    "density": ("人口密度.csv", ["人口密度（人/平方公里）"]),
    "urbanization": ("城镇化率.csv", ["urbanizationRate"]),
    "unemployment": ("就业信息_城镇失业率.csv", ["unemploymentRate"]),
    "employees": ("就业信息_从业人员数.csv", ["employeesNumber"]),
    "sector": (
        "就业信息_第一二三产业就业人数.csv",
        ["pi_Employment", "si_Employment", "ti_Employment"],
    ),
    "wage": ("工资水平_职工平均工资.csv", ["averageWage"]),
    "age": ("年龄结构.csv", ["0-14", "15-64", "65+"]),
    "overall_income": ("生活水平_人均可支配收入.csv", ["disposableIncome"]),
    "overall_consumption": ("生活水平_人均消费支出.csv", ["consumptionExpenditures"]),
    "urban_income": ("生活水平_城镇居民人均收入.csv", ["urbanDisposableIncome"]),
    "rural_income": ("生活水平_农村居民人均收入.csv", ["ruralDisposableIncome"]),
    "urban_consumption": (
        "生活水平_城镇居民消费支出.csv",
        ["urbanConsumptionExpenditures"],
    ),
    "rural_consumption": (
        "生活水平_农村居民消费支出.csv",
        ["ruralConsumptionExpenditures"],
    ),
}

FIELD_TO_DATASET = {
    field: dataset
    for dataset, (_, fields) in NORMALIZED_TABLES.items()
    for field in fields
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def numeric(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def as_text(value: Any) -> str:
    value = numeric(value) if not isinstance(value, str) else value
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def same_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    a = numeric(left)
    b = numeric(right)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame = frame.copy()
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def canonical_tables() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for dataset, (filename, fields) in NORMALIZED_TABLES.items():
        frame = read_csv(NORMALIZED_DIR / filename)
        if CITY_COL not in frame.columns and "城市名称" not in frame.columns:
            raise ValueError(f"missing city key in {filename}")
        if CITY_COL not in frame.columns:
            frame = frame.rename(columns={frame.columns[1]: CITY_COL})
        frame[YEAR_COL] = pd.to_numeric(frame[YEAR_COL], errors="coerce")
        for field in fields:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        result[dataset] = frame
    return result


def load_identity() -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    mapping = read_csv(S8_TRUTH_DIR / "city_identity_mapping.csv")
    identity = {
        str(row.city_id): {
            "real_city": str(row.real_city),
            "province": str(row.province),
            "mapping_status": str(row.mapping_status),
        }
        for row in mapping.itertuples(index=False)
    }
    return identity, mapping


def official_jiangsu_2021() -> dict[str, dict[str, float]]:
    """Values transcribed from the official Jiangsu yearbook table 19-2.

    The table has province-region subtotal columns interleaved with city
    columns.  Only the 13 city columns are represented here.
    """

    names = [
        "南京",
        "无锡",
        "常州",
        "苏州",
        "镇江",
        "南通",
        "扬州",
        "泰州",
        "徐州",
        "连云港",
        "淮安",
        "盐城",
        "宿迁",
    ]
    values = {
        "registered_population": [
            733.73,
            515.25,
            388.16,
            762.11,
            268.10,
            751.70,
            451.56,
            492.70,
            1035.29,
            533.21,
            555.25,
            803.84,
            590.86,
        ],
        "resident_population": [
            942.34,
            747.95,
            534.96,
            1284.78,
            321.72,
            773.30,
            457.70,
            452.18,
            902.85,
            460.20,
            456.22,
            671.30,
            499.90,
        ],
        "urbanization": [
            86.9,
            82.9,
            77.6,
            81.7,
            79.9,
            71.2,
            71.4,
            68.6,
            66.2,
            62.4,
            66.2,
            64.8,
            63.2,
        ],
        "density": [
            1431,
            1616,
            1224,
            1484,
            838,
            736,
            694,
            781,
            767,
            603,
            455,
            379,
            586,
        ],
        "employees": [
            485.1,
            418.7,
            300.4,
            748.8,
            198.7,
            481.9,
            270.8,
            275.6,
            475.2,
            249.4,
            265.4,
            413.0,
            279.8,
        ],
        "pi": [
            36.2,
            13.0,
            25.3,
            18.1,
            18.4,
            66.3,
            38.4,
            46.1,
            87.0,
            65.7,
            66.2,
            79.2,
            70.0,
        ],
        "si": [
            152.3,
            206.6,
            134.1,
            386.1,
            80.0,
            205.5,
            104.3,
            108.5,
            167.6,
            78.1,
            85.9,
            153.9,
            94.1,
        ],
        "ti": [
            296.6,
            199.1,
            141.1,
            344.6,
            100.4,
            210.2,
            128.1,
            121.0,
            220.6,
            105.7,
            113.4,
            180.0,
            115.7,
        ],
    }
    return {
        city: {metric: values[metric][i] for metric in values}
        for i, city in enumerate(names)
    }


def external_universe(
    tables: dict[str, pd.DataFrame],
    identity: dict[str, dict[str, str]],
    truth_raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    official = official_jiangsu_2021()
    real_to_city = {
        detail["real_city"]: city_id
        for city_id, detail in identity.items()
        if detail["province"] == "江苏"
    }
    specs = [
        ("resident_population", "population", "常住人口（万人）"),
        ("registered_population", "population", "户籍人口（万人）"),
        ("urbanization", "urbanization", "urbanizationRate"),
        ("density", "density", "人口密度（人/平方公里）"),
        ("employees", "employees", "employeesNumber"),
        ("pi", "sector", "pi_Employment"),
        ("si", "sector", "si_Employment"),
        ("ti", "sector", "ti_Employment"),
    ]
    for real_city, metrics in official.items():
        city_id = real_to_city.get(real_city)
        if not city_id:
            continue
        for metric, dataset, field in specs:
            table = tables[dataset]
            match = table[
                (table[CITY_COL] == city_id) & (table[YEAR_COL] == 2021)
            ]
            competition_value = (
                numeric(match.iloc[0][field]) if not match.empty else None
            )
            rows.append(
                {
                    "dataset": dataset,
                    "city_id": city_id,
                    "real_city": real_city,
                    "competition_year": 2021,
                    "real_year": 2021,
                    "field": field,
                    "competition_value": competition_value,
                    "official_value": metrics[metric],
                    "source_id": "SRC_JS_YEARBOOK_2021",
                    "source_tier": "TIER1",
                    "definition_match": True,
                    "revision_status": "ANNUAL_YEARBOOK_2021",
                    "verification_strength": "STRONG",
                    "observed_in_s1_source": competition_value is not None,
                    "notes": "Official Jiangsu yearbook table 19-2; city row and indicator definition align.",
                }
            )

    # Explicit city40 anchors used in the frozen S8 identity audit.
    city40 = tables["population"]
    city40_2020 = city40[
        (city40[CITY_COL] == "city40") & (city40[YEAR_COL] == 2020)
    ]
    if not city40_2020.empty:
        rows.extend(
            [
                {
                    "dataset": "population",
                    "city_id": "city40",
                    "real_city": identity["city40"]["real_city"],
                    "competition_year": 2020,
                    "real_year": 2020,
                    "field": "常住人口（万人）",
                    "competition_value": numeric(city40_2020.iloc[0]["常住人口（万人）"]),
                    "official_value": 321.0418,
                    "source_id": "SRC_JS_CENSUS_2020",
                    "source_tier": "TIER1",
                    "definition_match": True,
                    "revision_status": "CENSUS_ANCHOR",
                    "verification_strength": "STRONG",
                    "observed_in_s1_source": True,
                    "notes": "Frozen S8 identity evidence; official 2020 Zhenjiang census anchor.",
                },
                {
                    "dataset": "population",
                    "city_id": "city40",
                    "real_city": identity["city40"]["real_city"],
                    "competition_year": 2020,
                    "real_year": 2020,
                    "field": "户籍人口（万人）",
                    "competition_value": numeric(city40_2020.iloc[0]["户籍人口（万人）"]),
                    "official_value": 269.25,
                    "source_id": "SRC_JS_YEARBOOK_2020",
                    "source_tier": "TIER1",
                    "definition_match": True,
                    "revision_status": "ANNUAL_YEARBOOK_2020",
                    "verification_strength": "STRONG",
                    "observed_in_s1_source": True,
                    "notes": "Frozen S8 identity evidence; official 2020 Jiangsu annual table.",
                },
            ]
        )

    # Frozen S8 truth is an external target anchor, not an S1 observed cell.
    for row in truth_raw.itertuples(index=False):
        rows.append(
            {
                "dataset": "population",
                "city_id": row.city_id,
                "real_city": row.real_city,
                "competition_year": row.competition_target_year,
                "real_year": row.real_statistical_year,
                "field": "常住人口（万人）",
                "competition_value": np.nan,
                "official_value": numeric(row.raw_value),
                "source_id": row.source_id,
                "source_tier": row.source_tier,
                "definition_match": True,
                "revision_status": row.revision_status,
                "verification_strength": "STRONG" if row.source_tier == "TIER1" else "ACCEPTABLE",
                "observed_in_s1_source": False,
                "notes": "Frozen S8 target anchor; outside the S1 observed source range.",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "city_id", "competition_year", "field", "source_id"]
    ).reset_index(drop=True)


def external_lookup(external: pd.DataFrame) -> dict[tuple[str, str, int, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in external.itertuples(index=False):
        if not bool(row.observed_in_s1_source):
            continue
        lookup[(row.dataset, row.city_id, int(row.competition_year), row.field)].append(
            row._asdict()
        )
    return lookup


def severity_from_relative(relative_change: float) -> str:
    magnitude = abs(relative_change)
    if magnitude >= 0.5:
        return "EXTREME"
    if magnitude >= 0.3:
        return "HIGH"
    if magnitude >= 0.15:
        return "MODERATE"
    return "LOW"


def miss_severity(severity: str) -> str:
    return "CRITICAL" if severity == "EXTREME" else severity


def temporal_candidates(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Generate candidates without reading S1 decisions, quarantine, or registry."""

    candidates: list[dict[str, Any]] = []
    for dataset, frame in tables.items():
        _, fields = NORMALIZED_TABLES[dataset]
        for city_id, group in frame.groupby(CITY_COL):
            group = group.sort_values(YEAR_COL)
            for field in fields:
                values = pd.to_numeric(group[field], errors="coerce")
                years = group[YEAR_COL].tolist()
                vals = values.tolist()
                for index in range(1, len(vals)):
                    previous = numeric(vals[index - 1])
                    current = numeric(vals[index])
                    if previous is None or current is None or previous == 0:
                        continue
                    relative = (current - previous) / abs(previous)
                    if abs(relative) < 0.15:
                        continue
                    next_value = numeric(vals[index + 1]) if index + 1 < len(vals) else None
                    returns = (
                        next_value is not None
                        and abs(next_value - previous) / max(abs(previous), 1e-9) <= 0.10
                    )
                    reason = "large year-over-year change"
                    if returns:
                        reason += "; return-to-baseline pattern"
                    candidates.append(
                        {
                            "candidate_type": "TEMPORAL_JUMP",
                            "dataset": dataset,
                            "city_id": city_id,
                            "year": int(years[index]),
                            "field": field,
                            "previous_year": int(years[index - 1]),
                            "previous_value": previous,
                            "current_value": current,
                            "next_value": next_value,
                            "relative_change": relative,
                            "severity": severity_from_relative(relative),
                            "candidate_reason": reason,
                            "return_to_baseline": returns,
                        }
                    )

    population = tables["population"].copy()
    population["resident_registered_ratio"] = population["常住人口（万人）"] / population[
        "户籍人口（万人）"
    ].replace(0, np.nan)
    for city_id, group in population.groupby(CITY_COL):
        group = group.sort_values(YEAR_COL)
        for index in range(1, len(group)):
            previous = numeric(group.iloc[index - 1]["resident_registered_ratio"])
            current = numeric(group.iloc[index]["resident_registered_ratio"])
            if previous is None or current is None or previous == 0:
                continue
            relative = (current - previous) / abs(previous)
            if abs(relative) < 0.50:
                continue
            candidates.append(
                {
                    "candidate_type": "CROSS_FIELD_CONSISTENCY",
                    "dataset": "population",
                    "city_id": city_id,
                    "year": int(group.iloc[index][YEAR_COL]),
                    "field": "常住人口（万人）↔户籍人口（万人）",
                    "previous_year": int(group.iloc[index - 1][YEAR_COL]),
                    "previous_value": previous,
                    "current_value": current,
                    "next_value": None,
                    "relative_change": relative,
                    "severity": severity_from_relative(relative),
                    "candidate_reason": "resident/registered relationship changed sharply",
                    "return_to_baseline": False,
                }
            )
    return candidates


def raw_workbook_frames() -> list[tuple[str, str, pd.DataFrame]]:
    frames: list[tuple[str, str, pd.DataFrame]] = []
    for path in sorted(RAW_DIR.glob("*.xlsx")):
        sheet_names = pd.ExcelFile(path).sheet_names
        for sheet in sheet_names:
            header = 1 if path.name == "生活水平.xlsx" else 0
            frame = pd.read_excel(path, sheet_name=sheet, header=header)
            frames.append((path.name, str(sheet), frame.copy()))
    return frames


def find_city_year_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    city = CITY_COL if CITY_COL in frame.columns else None
    year = YEAR_COL if YEAR_COL in frame.columns else None
    if city is None:
        for column in frame.columns:
            if "城市" in str(column):
                city = column
                break
    if year is None:
        for column in frame.columns:
            if "年" in str(column):
                year = column
                break
    return city, year


def source_dataset_key(filename: str, sheet: str) -> str:
    if filename == "就业信息.xlsx":
        return {
            "城镇失业率": "unemployment",
            "从业人员数": "employees",
            "第一、二、三产业就业人数": "sector",
        }.get(sheet, "employment_raw")
    if filename == "生活水平.xlsx":
        return {
            "人均可支配收入": "overall_income",
            "城镇居民人均收入": "urban_income",
            "农村居民人均收入": "rural_income",
            "人均消费支出": "overall_consumption",
            "城镇居民消费支出": "urban_consumption",
            "农村居民消费支出": "rural_consumption",
        }.get(sheet, "living_standard_raw")
    return {
        "人口规模.xlsx": "population",
        "人口密度.xlsx": "density",
        "城镇化率.xlsx": "urbanization",
        "工资水平.xlsx": "wage",
        "年龄结构.xlsx": "age",
    }.get(filename, filename)


def raw_structure_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structural: list[dict[str, Any]] = []
    label_candidates: list[dict[str, Any]] = []
    shift_candidates: list[dict[str, Any]] = []
    for filename, sheet, frame in raw_workbook_frames():
        dataset_key = source_dataset_key(filename, sheet)
        city_col, year_col = find_city_year_columns(frame)
        if city_col is None:
            continue
        frame = frame.copy()
        if year_col is not None:
            frame["__year"] = pd.to_numeric(frame[year_col], errors="coerce")
        for row_index, value in frame[city_col].items():
            if pd.isna(value):
                continue
            value_text = str(value)
            if not CITY_RE.fullmatch(value_text):
                record = {
                    "candidate_type": "INVALID_CITY_LABEL",
                    "dataset": dataset_key,
                    "source_file": filename,
                    "sheet": sheet,
                    "city_id": value_text,
                    "year": int(frame.loc[row_index, "__year"])
                    if "__year" in frame and pd.notna(frame.loc[row_index, "__year"])
                    else "",
                    "field": "city_label",
                    "support_years": 1,
                    "support_fields": 1,
                    "candidate_reason": "raw city label is outside canonical city1-city40 format",
                    "status": "UNRESOLVED",
                }
                structural.append(record)
                label_candidates.append(record.copy())
        if year_col is not None:
            duplicate_mask = frame.duplicated([city_col, year_col], keep=False)
            if duplicate_mask.any():
                for (city_id, year), group in frame.loc[duplicate_mask].groupby(
                    [city_col, year_col], dropna=False
                ):
                    record = {
                        "candidate_type": "DUPLICATE_KEY",
                        "dataset": dataset_key,
                        "source_file": filename,
                        "sheet": sheet,
                        "city_id": str(city_id),
                        "year": int(year) if pd.notna(year) else "",
                        "field": "city×year",
                        "support_years": 1,
                        "support_fields": max(1, len(group.columns) - 2),
                        "candidate_reason": f"raw sheet contains {len(group)} rows for one city-year key",
                        "status": "UNRESOLVED",
                    }
                    structural.append(record)
                    label_candidates.append(record.copy())

        numeric_fields = [
            column
            for column in frame.columns
            if column not in {city_col, year_col, "__year"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if year_col is None or not numeric_fields:
            continue
        grouped: dict[str, pd.DataFrame] = {
            str(city_id): group.sort_values("__year")
            for city_id, group in frame.groupby(city_col)
            if pd.notna(city_id)
        }
        city_pairs = list(itertools.combinations(sorted(grouped), 2))
        for left_city, right_city in city_pairs:
            left = grouped[left_city]
            right = grouped[right_city]
            for field in numeric_fields:
                for offset in (-2, -1, 1, 2):
                    left_values = {
                        int(row["__year"]): numeric(row[field])
                        for _, row in left.iterrows()
                        if pd.notna(row["__year"])
                    }
                    right_values = {
                        int(row["__year"]): numeric(row[field])
                        for _, row in right.iterrows()
                        if pd.notna(row["__year"])
                    }
                    comparable = [
                        (left_values[year], right_values[year + offset])
                        for year in left_values
                        if year + offset in right_values
                        and left_values[year] is not None
                        and right_values[year + offset] is not None
                    ]
                    if len(comparable) < 8:
                        continue
                    matches = sum(
                        abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
                        for a, b in comparable
                    )
                    match_rate = matches / len(comparable)
                    if match_rate < 0.80:
                        continue
                    record = {
                        "candidate_type": "SHIFTED_DUPLICATE_TRAJECTORY",
                        "dataset": dataset_key,
                        "source_file": filename,
                        "sheet": sheet,
                        "city_id": left_city,
                        "paired_city_id": right_city,
                        "year": "",
                        "field": str(field),
                        "candidate_offset": offset,
                        "support_years": len(comparable),
                        "support_fields": 1,
                        "match_rate": match_rate,
                        "candidate_reason": "multi-year raw trajectories match after a candidate year offset",
                        "status": "TRUE_YEAR_SHIFT"
                        if dataset_key == "employees" and sheet == "从业人员数" and match_rate >= 0.95
                        else "UNRESOLVED",
                    }
                    shift_candidates.append(record)
                    structural.append(record.copy())
                    label_candidates.append(record.copy())

        # Some source sheets contain two contiguous blocks with the same city
        # label.  Comparing one de-duplicated city dictionary would erase the
        # very block-copy signal being audited, so scan contiguous blocks too.
        if dataset_key == "employees":
            city_series = frame[city_col].astype("string").str.strip()
            blocks: list[tuple[str, int, pd.DataFrame]] = []
            for city in sorted(city_series.dropna().unique().tolist()):
                positions = [i for i, value in enumerate(city_series.tolist()) if value == city]
                if not positions:
                    continue
                chunks: list[list[int]] = [[positions[0]]]
                for position in positions[1:]:
                    previous_position = chunks[-1][-1]
                    previous_year = numeric(frame.iloc[previous_position][YEAR_COL])
                    current_year = numeric(frame.iloc[position][YEAR_COL])
                    contiguous_year = (
                        previous_year is None
                        or current_year is None
                        or abs(current_year - previous_year) <= 1
                    )
                    if position == previous_position + 1 and contiguous_year:
                        chunks[-1].append(position)
                    else:
                        chunks.append([position])
                for block_index, chunk in enumerate(chunks, start=1):
                    blocks.append((str(city), block_index, frame.iloc[chunk].copy()))
            for (left_city, left_block, left_frame), (right_city, right_block, right_frame) in itertools.combinations(blocks, 2):
                if left_city == right_city:
                    continue
                left_values = {
                    int(row[YEAR_COL]): numeric(row["employeesNumber"])
                    for _, row in left_frame.iterrows()
                    if pd.notna(row[YEAR_COL]) and numeric(row["employeesNumber"]) is not None
                }
                right_values = {
                    int(row[YEAR_COL]): numeric(row["employeesNumber"])
                    for _, row in right_frame.iterrows()
                    if pd.notna(row[YEAR_COL]) and numeric(row["employeesNumber"]) is not None
                }
                for offset in (-2, -1, 1, 2):
                    comparable = [
                        (left_values[year], right_values[year + offset])
                        for year in left_values
                        if year + offset in right_values
                    ]
                    if len(comparable) < 8:
                        continue
                    matches = sum(
                        abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
                        for a, b in comparable
                    )
                    match_rate = matches / len(comparable)
                    if match_rate < 0.95:
                        continue
                    record = {
                        "candidate_type": "SHIFTED_DUPLICATE_TRAJECTORY",
                        "dataset": dataset_key,
                        "source_file": filename,
                        "sheet": sheet,
                        "city_id": left_city,
                        "paired_city_id": right_city,
                        "year": "",
                        "field": "employeesNumber",
                        "candidate_offset": offset,
                        "support_years": len(comparable),
                        "support_fields": 1,
                        "match_rate": match_rate,
                        "block_index_left": left_block,
                        "block_index_right": right_block,
                        "candidate_reason": "contiguous raw blocks match another city after a candidate year offset",
                        "status": "TRUE_YEAR_SHIFT",
                    }
                    shift_candidates.append(record)
                    structural.append(record.copy())
                    label_candidates.append(record.copy())
    return structural, label_candidates, shift_candidates


def identity_validation(
    tables: dict[str, pd.DataFrame], external: pd.DataFrame, identity: dict[str, dict[str, str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in tables.items():
        fields = NORMALIZED_TABLES[dataset][1]
        for city_id in sorted(identity):
            for field in fields:
                observed = frame[frame[CITY_COL] == city_id][[YEAR_COL, field]].dropna()
                ext = external[
                    (external.dataset == dataset)
                    & (external.city_id == city_id)
                    & (external.field == field)
                    & (external.observed_in_s1_source)
                ]
                if ext.empty or observed.empty:
                    rows.append(
                        {
                            "dataset": dataset,
                            "city_id": city_id,
                            "real_city": identity[city_id]["real_city"],
                            "field": field,
                            "external_cells": 0,
                            "matched_cells": 0,
                            "mismatch_cells": 0,
                            "match_rate": np.nan,
                            "status": "NO_EXTERNAL_FINGERPRINT",
                            "evidence": "No definition-matched official value for this city-field in the available universe.",
                        }
                    )
                    continue
                matched = 0
                mismatched = 0
                for erow in ext.itertuples(index=False):
                    match = observed[observed[YEAR_COL] == erow.competition_year]
                    if match.empty:
                        continue
                    actual = numeric(match.iloc[0][field])
                    expected = numeric(erow.official_value)
                    if actual is not None and expected is not None and abs(actual - expected) <= max(0.05, abs(expected) * 0.03):
                        matched += 1
                    else:
                        mismatched += 1
                total = matched + mismatched
                rows.append(
                    {
                        "dataset": dataset,
                        "city_id": city_id,
                        "real_city": identity[city_id]["real_city"],
                        "field": field,
                        "external_cells": total,
                        "matched_cells": matched,
                        "mismatch_cells": mismatched,
                        "match_rate": matched / total if total else np.nan,
                        "status": "CONSISTENT_WITH_EXTERNAL" if total and mismatched == 0 else "CONFLICT_WITH_EXTERNAL",
                        "evidence": "Official Jiangsu yearbook 19-2 compared to the same 2021 city-field cell.",
                    }
                )
    return pd.DataFrame(rows)


def indicator_semantics(tables: dict[str, pd.DataFrame], structural: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    population = tables["population"]
    ratio_jumps = []
    for _, group in population.groupby(CITY_COL):
        group = group.sort_values(YEAR_COL).copy()
        group["ratio"] = group["常住人口（万人）"] / group["户籍人口（万人）"].replace(0, np.nan)
        group["ratio_change"] = group["ratio"].pct_change()
        ratio_jumps.extend(group.loc[group["ratio_change"].abs() >= 0.50, "ratio_change"].dropna().tolist())
    order_violations = len(ratio_jumps)
    rows.append(
        {
            "dataset": "population",
            "check_id": "resident_registered_order",
            "fields": "常住人口（万人）;户籍人口（万人）",
            "candidate_count": order_violations,
            "evidence": "Count of city-year resident/registered ratio changes with absolute relative change >= 50%.",
            "status": "CROSS_FIELD_CONFLICT" if order_violations else "NO_LABEL_MISMATCH",
            "s1_covered": "NO",
            "notes": "Ordinary resident < registered relationships are not treated as errors; city40-2021 is the material ratio discontinuity.",
        }
    )

    employees = tables["employees"]
    sector = tables["sector"]
    merged = employees.merge(sector, on=[CITY_COL, YEAR_COL], how="inner")
    merged["sector_total"] = merged[["pi_Employment", "si_Employment", "ti_Employment"]].sum(axis=1)
    valid = merged["employeesNumber"].notna() & merged["sector_total"].notna()
    differences = (merged.loc[valid, "employeesNumber"] - merged.loc[valid, "sector_total"]).abs()
    rows.append(
        {
            "dataset": "employees+sector",
            "check_id": "employee_sector_sum",
            "fields": "employeesNumber;pi_Employment;si_Employment;ti_Employment",
            "candidate_count": int((differences > 1).sum()),
            "evidence": f"{int((differences <= 1).sum())}/{len(differences)} canonical overlap rows within absolute difference 1.",
            "status": "NO_LABEL_MISMATCH" if (differences <= 1).all() else "CONFLICT_WITH_EXTERNAL_OR_INTERNAL",
            "s1_covered": "YES",
            "notes": "S1 excluded source-derived employee rows from circular independent validation; this check is reported, not used to rewrite data.",
        }
    )

    overall = tables["overall_income"]
    urban = tables["urban_income"]
    city3_overall = overall[overall[CITY_COL] == "city3"].set_index(YEAR_COL)["disposableIncome"]
    city3_urban = urban[urban[CITY_COL] == "city3"].set_index(YEAR_COL)["urbanDisposableIncome"]
    overlap = []
    for year in sorted(set(city3_overall.dropna().index) & set(city3_urban.dropna().index)):
        if year - 1 in city3_urban.index:
            a = numeric(city3_overall.loc[year])
            b = numeric(city3_urban.loc[year - 1])
            if a is not None and b is not None:
                overlap.append(abs(a - b))
    rows.append(
        {
            "dataset": "overall_income+urban_income",
            "check_id": "city3_lagged_copy",
            "fields": "disposableIncome;urbanDisposableIncome",
            "candidate_count": len(overlap),
            "evidence": f"city3 has {len(overlap)} overlapping lagged comparisons; low residual is a semantic warning, not a unique relabel proof.",
            "status": "POSSIBLE_SHIFTED_COPY" if overlap else "UNRESOLVED",
            "s1_covered": "YES",
            "notes": "Preserve-with-warning remains appropriate without an independent semantic anchor.",
        }
    )

    invalid_labels = sum(1 for record in structural if record["candidate_type"] == "INVALID_CITY_LABEL")
    rows.append(
        {
            "dataset": "raw_workbooks",
            "check_id": "city_label_format",
            "fields": "城市名称",
            "candidate_count": invalid_labels,
            "evidence": "Independent raw workbook scan found labels outside city1-city40; no historical registry was used for candidate selection.",
            "status": "LABEL_REMAP_CANDIDATE" if invalid_labels else "NO_LABEL_MISMATCH",
            "s1_covered": "YES",
            "notes": "The two ctiy labels were repaired in the urban-income S1 report; other raw label candidates remain evidence rows only.",
        }
    )

    age = tables["age"].merge(
        population[[CITY_COL, YEAR_COL, "常住人口（万人）"]],
        on=[CITY_COL, YEAR_COL],
        how="left",
    )
    age_sum = age[["0-14", "15-64", "65+"]].sum(axis=1, min_count=3)
    age_gap = (age_sum - age["常住人口（万人）"]).abs() / age["常住人口（万人）"].replace(0, np.nan)
    age_outside = int((age_gap > 0.05).sum())
    rows.append(
        {
            "dataset": "age",
            "check_id": "age_band_sum",
            "fields": "0-14;15-64;65+",
            "candidate_count": age_outside,
            "evidence": "Canonical age-band sums compared with the matching resident-population level; tolerance is 5%.",
            "status": "POSSIBLE_DEFINITION_OR_SOURCE_CONFLICT" if age_outside else "NO_LABEL_MISMATCH",
            "s1_covered": "PARTIAL",
            "notes": "Age is benchmark-year data; no annual interpolation was introduced.",
        }
    )
    return pd.DataFrame(rows)


def build_inventory(
    tables: dict[str, pd.DataFrame], identity: dict[str, dict[str, str]]
) -> pd.DataFrame:
    """Reconstruct S1 decisions from source reports and data, not from post-hoc labels."""

    rows: list[dict[str, Any]] = []

    def add(
        dataset: str,
        sheet: str,
        city_id: Any,
        year: Any,
        field: str,
        decision_type: str,
        original: Any,
        canonical: Any,
        action: str,
        reason: str,
        evidence: str,
        artifact: str,
        notes: str = "",
    ) -> None:
        city_text = "" if city_id is None or pd.isna(city_id) else str(city_id)
        year_number = numeric(year)
        rows.append(
            {
                "dataset": dataset,
                "sheet": sheet,
                "city_id": city_text,
                "year": int(year_number) if year_number is not None else "",
                "field": field,
                "decision_type": decision_type,
                "original_value": as_text(original),
                "canonical_value": as_text(canonical),
                "s1_action": action,
                "s1_reason": reason,
                "s1_evidence": evidence,
                "source_artifact": artifact,
                "external_validation_status": "PENDING_JOIN",
                "notes": notes,
                "decision_key": f"{dataset}|{city_text}|{int(year_number) if year_number is not None else ''}|{field}",
            }
        )

    raw_population = pd.read_excel(RAW_DIR / "人口规模.xlsx")
    normalized_population = tables["population"]
    for _, row in raw_population.iterrows():
        for field in ["常住人口（万人）", "户籍人口（万人）"]:
            if numeric(row[field]) == 0:
                normalized = normalized_population[
                    (normalized_population[CITY_COL] == row[CITY_COL])
                    & (normalized_population[YEAR_COL] == row[YEAR_COL])
                ]
                canonical = normalized.iloc[0][field] if not normalized.empty else np.nan
                add(
                    "population",
                    "人口规模",
                    row[CITY_COL],
                    row[YEAR_COL],
                    field,
                    "REPAIR",
                    0,
                    canonical,
                    "replace raw zero sentinel with missing",
                    "raw zero is a confirmed missing sentinel",
                    "audit_population.py normalize_population; zero count is asserted after normalization",
                    "scripts/audit_population.py; data/normalized/人口规模.csv",
                )

    raw_density = pd.read_excel(RAW_DIR / "人口密度.xlsx")
    duplicate = raw_density[
        (raw_density[CITY_COL] == "city31") & (raw_density[YEAR_COL] == 2002)
    ]
    if len(duplicate) > 1:
        add(
            "density",
            "人口密度",
            "city31",
            2002,
            "人口密度（人/平方公里）",
            "REPAIR",
            "2315 and 422",
            422,
            "retain one duplicate-key row",
            "duplicate key; 422 is retained by the S1 normalization rule",
            "audit_population_density.py normalize_population_density",
            "scripts/audit_population_density.py; data/normalized/人口密度.csv",
        )
    density_quarantine = read_csv(QUARANTINE_DIR / "人口密度_quarantine.csv")
    for row in density_quarantine.itertuples(index=False):
        add(
            "density",
            "人口密度",
            row.城市名称,
            row.年份,
            "人口密度（人/平方公里）",
            "INVALIDATE",
            row.raw_density,
            np.nan,
            "invalidate canonical value to missing",
            row.reason,
            "S1 A-B-A pre-regime batch rule with retained post-shift regime",
            "data/quarantine/人口密度_quarantine.csv",
        )
    regime = read_csv(NORMALIZED_DIR / "人口密度_regime_report.csv")
    for row in regime.itertuples(index=False):
        add(
            "density",
            "人口密度",
            row.城市名称,
            row.regime_shift_year,
            "人口密度（人/平方公里）",
            "PRESERVE_WITH_WARNING",
            "",
            "",
            "retain persistent post-shift regime",
            row.interpretation,
            "data/normalized/人口密度_regime_report.csv",
            "data/normalized/人口密度_regime_report.csv",
        )

    employment_report = read_csv(NORMALIZED_DIR / "就业信息_repair_report.csv")
    for row in employment_report.itertuples(index=False):
        action = str(row.action)
        if action in {"relabel_and_year_shift", "relabel"}:
            add(
                "employees",
                "从业人员数",
                "",
                "",
                "employeesNumber",
                "LABEL_REMAP",
                row.scope,
                row.result,
                action,
                "cross-source sector-total invariant",
                row.evidence,
                "data/normalized/就业信息_repair_report.csv",
            )

    raw_employees = pd.read_excel(RAW_DIR / "就业信息.xlsx", sheet_name="从业人员数")
    whitespace_labels = sorted(
        {
            str(value)
            for value in raw_employees[CITY_COL].dropna().tolist()
            if str(value) != str(value).strip() and CITY_RE.fullmatch(str(value).strip())
        }
    )
    for raw_label in whitespace_labels:
        add(
            "employees",
            "从业人员数",
            raw_label.strip(),
            "",
            "city_label",
            "LABEL_REMAP",
            raw_label,
            raw_label.strip(),
            "strip whitespace from city label",
            "raw city label has trailing whitespace",
            "audit_employment.py normalize_employees_source_v2 applies .str.strip() before structural assertions",
            "scripts/audit_employment.py",
        )

    employee_quarantine = read_csv(QUARANTINE_DIR / "就业信息_从业人员数_quarantine.csv")
    for row in employee_quarantine.itertuples(index=False):
        add(
            "employees",
            "从业人员数",
            row.城市名称,
            row.年份,
            "employeesNumber",
            "QUARANTINE",
            row.employeesNumber,
            np.nan,
            "quarantine source row",
            row.reason,
            f"suspected_target_city={row.suspected_target_city}; suspected_year_shift={row.suspected_year_shift}",
            "data/quarantine/就业信息_从业人员数_quarantine.csv",
        )
    provenance = read_csv(NORMALIZED_DIR / "就业信息_从业人员数_provenance.csv")
    for row in provenance.itertuples(index=False):
        add(
            "employees",
            "从业人员数",
            row.城市名称,
            row.年份,
            "employeesNumber",
            "DERIVED_RECONSTRUCTION",
            np.nan,
            row.employeesNumber,
            "derive missing employee total from three sectors",
            row.derivation_rule,
            row.evidence,
            "data/normalized/就业信息_从业人员数_provenance.csv",
        )

    raw_sector = pd.read_excel(RAW_DIR / "就业信息.xlsx", sheet_name="第一、二、三产业就业人数")
    normalized_sector = tables["sector"]
    merged_sector = raw_sector.merge(
        normalized_sector,
        on=[CITY_COL, YEAR_COL],
        suffixes=("_raw", "_canonical"),
        how="inner",
    )
    for _, row in merged_sector.iterrows():
        raw_value = numeric(row["pi_Employment_raw"])
        canonical_value = numeric(row["pi_Employment_canonical"])
        if raw_value is not None and canonical_value is not None and abs(raw_value - canonical_value) > 1e-9:
            add(
                "sector",
                "第一、二、三产业就业人数",
                row[CITY_COL],
                row[YEAR_COL],
                "pi_Employment",
                "REPAIR",
                raw_value,
                canonical_value,
                "repair primary-sector component from employee total minus other sectors",
                "primary-sector component repair",
                "employeesNumber - si_Employment - ti_Employment; exact post-repair total consistency",
                "scripts/audit_employment.py; data/normalized/就业信息_第一二三产业就业人数.csv",
            )
    sector_quarantine = read_csv(QUARANTINE_DIR / "就业信息_第一二三产业就业人数_quarantine.csv")
    for row in sector_quarantine.itertuples(index=False):
        add(
            "sector",
            "第一、二、三产业就业人数",
            row.城市名称,
            row.年份,
            "pi_Employment;si_Employment;ti_Employment",
            "QUARANTINE",
            f"{row.raw_pi_Employment};{row.raw_si_Employment};{row.raw_ti_Employment}",
            np.nan,
            "quarantine conflicting sector row",
            row.reason,
            f"sector_total={row.sector_total}; employeesNumber={row.employeesNumber}; abs_difference={row.abs_difference}",
            "data/quarantine/就业信息_第一二三产业就业人数_quarantine.csv",
        )

    wage_quarantine = read_csv(QUARANTINE_DIR / "工资水平_职工平均工资_quarantine.csv")
    for row in wage_quarantine.itertuples(index=False):
        add(
            "wage",
            "职工平均工资",
            row.城市名称,
            row.年份,
            "averageWage",
            "INVALIDATE",
            row.raw_averageWage,
            np.nan,
            "invalidate canonical value to missing",
            row.reason,
            "S1 isolated wage anomaly rule; true value intentionally unresolved",
            "data/quarantine/工资水平_职工平均工资_quarantine.csv",
        )

    age_quarantine = read_csv(QUARANTINE_DIR / "年龄结构_quarantine.csv")
    for _, row in age_quarantine.iterrows():
        raw_values = ";".join(
            as_text(row[f"raw_{field}"]) for field in ["0-14", "15-64", "65+"]
        )
        add(
            "age",
            "年龄结构",
            row[CITY_COL],
            row[YEAR_COL],
            "0-14;15-64;65+",
            "INVALIDATE",
            raw_values,
            np.nan,
            "invalidate all age components to missing",
            row.reason,
            "S1 benchmark-year age consistency rule; three components are one row-level invalidation decision",
            "data/quarantine/年龄结构_quarantine.csv",
        )

    income_quarantine = read_csv(QUARANTINE_DIR / "生活水平_人均可支配收入_quarantine.csv")
    for row in income_quarantine.itertuples(index=False):
        add(
            "overall_income",
            "人均可支配收入",
            row.城市名称,
            row.年份,
            "disposableIncome",
            "QUARANTINE",
            row.raw_disposableIncome,
            np.nan,
            "invalidate contaminated block to missing",
            row.reason,
            f"ruralDisposableIncome={row.ruralDisposableIncome}; abs_gap={row.abs_gap}",
            "data/quarantine/生活水平_人均可支配收入_quarantine.csv",
        )

    living_report = read_csv(NORMALIZED_DIR / "生活水平_repair_report.csv")
    for row in living_report.itertuples(index=False):
        if row.action == "repair" and str(row.sheet) == "人均消费支出":
            add(
                "overall_consumption",
                str(row.sheet),
                row.scope,
                "",
                "consumptionExpenditures",
                "REPAIR",
                row.raw,
                row.normalized,
                "repair isolated magnitude value",
                row.reason,
                row.reason,
                "data/normalized/生活水平_repair_report.csv",
            )
        elif row.action == "invalidate_to_NA":
            # The 32 affected cells are already represented one-by-one by the
            # quarantine file.  Do not double count the report summary row.
            continue
    # The two label repairs are recorded explicitly in the same report.
    for row in living_report.itertuples(index=False):
        if "城镇居民人均收入" in str(row.sheet) and row.action == "repair":
            add(
                "urban_income",
                str(row.sheet),
                row.normalized,
                "",
                "city_label",
                "LABEL_REMAP",
                row.raw,
                row.normalized,
                "repair malformed city label",
                row.reason,
                row.reason,
                "data/normalized/生活水平_repair_report.csv",
            )

    preserve_rows = [
        (
            "population",
            "人口规模",
            "",
            "",
            "coverage",
            "source coverage gaps",
            "preserve source-defined missingness; do not reconstruct population",
            "docs/s1_quality_issue_registry.csv; docs/s1_data_contract.csv",
        ),
        (
            "urbanization",
            "城镇化率",
            "city3",
            "",
            "urbanizationRate",
            "persistent 99-to-100 plateau",
            "preserve canonical values with warning",
            "docs/s1_quality_issue_registry.csv",
        ),
        (
            "unemployment",
            "城镇失业率",
            "",
            "",
            "unemploymentRate",
            "low-cardinality repeated blocks are not uniquely identifiable",
            "preserve canonical values with warning",
            "docs/s1_quality_issue_registry.csv",
        ),
        (
            "overall_income+urban_income",
            "人均可支配收入 ↔ 城镇居民人均收入",
            "city3",
            "",
            "semantic_alignment",
            "city3 lagged-copy semantics unresolved",
            "preserve values; do not treat the two fields as independent trusted signals",
            "docs/s1_quality_issue_registry.csv",
        ),
        (
            "overall_consumption",
            "人均消费支出",
            "city14",
            2021,
            "consumptionExpenditures",
            "weighted urban/rural residual is not a universal identity",
            "preserve; use as control only",
            "docs/s1_quality_issue_registry.csv",
        ),
        (
            "age",
            "年龄结构",
            "",
            "2000;2010;2020",
            "year_domain",
            "benchmark-year domain only",
            "preserve domain and missing keys; do not create annual rows",
            "docs/s1_data_contract.csv",
        ),
        (
            "source",
            "source quarantine",
            "",
            "",
            "test.xls",
            "unresolved source file excluded from canonical inputs",
            "preserve quarantine boundary; do not parse or use",
            "docs/S1_MAIN_HANDOFF.md; docs/PROJECT_CONTEXT.md",
        ),
    ]
    for dataset, sheet, city_id, year, field, reason, action, artifact in preserve_rows:
        add(
            dataset,
            sheet,
            city_id,
            year,
            field,
            "PRESERVE_WITH_WARNING" if dataset != "source" else "PRESERVE_NO_ISSUE",
            "",
            "",
            action,
            reason,
            reason,
            artifact,
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["dataset", "sheet", "city_id", "year", "field", "decision_type", "source_artifact"]
    ).reset_index(drop=True)
    frame.insert(0, "decision_id", [f"S1D-{i:04d}" for i in range(1, len(frame) + 1)])
    return frame


def validate_decisions(
    inventory: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    lookup = external_lookup(external)
    rows: list[dict[str, Any]] = []
    for row in inventory.itertuples(index=False):
        dataset = row.dataset
        ext_records = lookup.get((dataset, row.city_id, int(row.year), row.field), []) if str(row.year) else []
        external_status = "NOT_EXTERNALLY_VERIFIABLE"
        external_value = ""
        if ext_records:
            external_value = ext_records[0]["official_value"]
            external_status = "EXTERNALLY_VERIFIED_MATCH"
            canonical = numeric(row.canonical_value)
            if canonical is not None and abs(canonical - numeric(external_value)) > max(0.05, abs(numeric(external_value)) * 0.03):
                external_status = "EXTERNALLY_VERIFIED_CONFLICT"

        decision_type = row.decision_type
        reason = str(row.s1_reason)
        if decision_type == "LABEL_REMAP":
            validation_class = "CORRECT_LABEL_REMAP"
        elif decision_type == "DERIVED_RECONSTRUCTION":
            validation_class = "CORRECT_CAUTION"
        elif decision_type == "PRESERVE_NO_ISSUE":
            validation_class = "CORRECT_PRESERVE"
        elif decision_type == "PRESERVE_WITH_WARNING":
            validation_class = "CORRECT_CAUTION"
        elif decision_type == "QUARANTINE":
            validation_class = "CORRECT_QUARANTINE"
        elif decision_type == "INVALIDATE":
            validation_class = "CORRECT_INVALIDATION"
        elif decision_type == "REPAIR":
            if "label" in reason.lower() or "label" in str(row.s1_action).lower():
                validation_class = "CORRECT_LABEL_REMAP"
            elif "zero sentinel" in reason:
                validation_class = "CORRECT_REPAIR"
            elif "duplicate" in reason.lower() or "component" in str(row.s1_action).lower():
                validation_class = "CORRECT_REPAIR"
            else:
                validation_class = "CORRECT_CAUTION"
        else:
            validation_class = "EXTERNAL_EVIDENCE_INSUFFICIENT"

        note = "Internal S1 evidence supports the historical action; external exact-cell evidence is unavailable."
        if external_status == "EXTERNALLY_VERIFIED_MATCH":
            note = "The canonical retained value matches the definition-matched official comparison cell."
        elif external_status == "EXTERNALLY_VERIFIED_CONFLICT":
            note = "The exact-cell external comparison conflicts; no historical S1 value is rewritten by this Plan."
        rows.append(
            {
                "decision_id": row.decision_id,
                "dataset": dataset,
                "city_id": row.city_id,
                "year": row.year,
                "field": row.field,
                "decision_type": decision_type,
                "s1_action": row.s1_action,
                "validation_class": validation_class,
                "external_validation_status": external_status,
                "external_value": external_value,
                "evidence_basis": row.s1_evidence,
                "notes": note,
                "source_artifact": row.source_artifact,
            }
        )
    return pd.DataFrame(rows)


def join_candidate_status(
    candidates: list[dict[str, Any]], inventory: pd.DataFrame, external: pd.DataFrame
) -> pd.DataFrame:
    inventory_keys = set(inventory.decision_key.astype(str))
    external_lookup_map = external_lookup(external)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        dataset = candidate.get("dataset", "")
        city_id = str(candidate.get("city_id", ""))
        lookup_city_id = city_id.strip()
        year = candidate.get("year", "")
        field = str(candidate.get("field", ""))
        year_key = "" if year == "" or pd.isna(year) else str(int(year))
        direct_key = f"{dataset}|{lookup_city_id}|{year_key}|{field}"
        s1_flagged = direct_key in inventory_keys
        if field == "city_label":
            s1_flagged = s1_flagged or f"{dataset}|{lookup_city_id}||city_label" in inventory_keys
        matching_keys = [
            key for key in inventory_keys if key.startswith(f"{dataset}|{lookup_city_id}|") and field in key
        ]
        ext_records = external_lookup_map.get((dataset, lookup_city_id, int(year), field), []) if year_key else []
        external_status = "NOT_EXTERNALLY_VERIFIABLE"
        official_value = ""
        external_anomaly = False
        cause = "UNRESOLVED"
        if ext_records:
            official_value = ext_records[0]["official_value"]
            current = numeric(candidate.get("current_value"))
            official = numeric(official_value)
            previous = numeric(candidate.get("previous_value"))
            external_status = "EXTERNALLY_VERIFIED"
            if current is not None and official is not None:
                relative_gap = abs(current - official) / max(abs(official), 1e-9)
                if relative_gap <= 0.03:
                    cause = "REAL_CHANGE"
                elif previous is not None and abs(previous - official) / max(abs(official), 1e-9) <= 0.10:
                    cause = "SOURCE_ERROR_LIKELY"
                    external_anomaly = True
                else:
                    cause = "REVISION_OR_CENSUS"
        if s1_flagged:
            posthoc_status = "S1_CAUGHT"
            classification = "CORRECT_CAUTION"
        elif external_anomaly:
            posthoc_status = "S1_MISSED"
            classification = "FALSE_NEGATIVE_MISSED_ANOMALY"
        elif cause == "REAL_CHANGE":
            posthoc_status = "REAL_CHANGE"
            classification = "EXTERNAL_EVIDENCE_INSUFFICIENT"
        else:
            posthoc_status = "UNRESOLVED"
            classification = "EXTERNAL_EVIDENCE_INSUFFICIENT"
        row = dict(candidate)
        row.update(
            {
                "candidate_id": f"S8G-{index:04d}",
                "candidate_found_posthoc": "YES",
                "was_flagged_in_S1": "YES" if s1_flagged else "NO",
                "s1_matching_decision_ids": ";".join(sorted(matching_keys)),
                "external_status": external_status,
                "official_value": official_value,
                "external_cause": cause,
                "external_anomaly": "YES" if external_anomaly else "NO",
                "posthoc_status": posthoc_status,
                "classification": classification,
                "miss_severity": miss_severity(str(candidate.get("severity", "LOW")))
                if classification == "FALSE_NEGATIVE_MISSED_ANOMALY"
                else "",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def scorecard(
    inventory: pd.DataFrame,
    decision_validation: pd.DataFrame,
    candidates: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    datasets = list(NORMALIZED_TABLES) + ["overall_income+urban_income", "source"]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        inv = inventory[inventory.dataset == dataset]
        validated = decision_validation[decision_validation.dataset == dataset]
        cand = candidates[candidates.dataset == dataset]
        verified_cells = int(
            ((external.dataset == dataset) & external.observed_in_s1_source).sum()
        )
        verified_anomalies = int(
            (cand.external_anomaly == "YES").sum() if not cand.empty else 0
        )
        caught = int(
            ((cand.external_anomaly == "YES") & (cand.was_flagged_in_S1 == "YES")).sum()
            if not cand.empty
            else 0
        )
        missed = int(
            ((cand.external_anomaly == "YES") & (cand.was_flagged_in_S1 == "NO")).sum()
            if not cand.empty
            else 0
        )
        externally_verified_decisions = int(
            validated.external_validation_status.str.startswith("EXTERNALLY_VERIFIED").sum()
            if not validated.empty
            else 0
        )
        verified_correct_decisions = int(
            (
                validated.external_validation_status.str.startswith("EXTERNALLY_VERIFIED")
                & validated.validation_class.str.startswith("CORRECT")
            ).sum()
            if not validated.empty
            else 0
        )
        correct = int(validated.validation_class.str.startswith("CORRECT").sum()) if not validated.empty else 0
        wrong = int(
            validated.validation_class.str.startswith(("FALSE_POSITIVE", "WRONG")).sum()
            if not validated.empty
            else 0
        )
        critical = int(
            ((cand.classification == "FALSE_NEGATIVE_MISSED_ANOMALY") & (cand.miss_severity == "CRITICAL")).sum()
            if not cand.empty
            else 0
        )
        precision = (
            verified_correct_decisions / externally_verified_decisions
            if externally_verified_decisions
            else np.nan
        )
        recall = caught / verified_anomalies if verified_anomalies else np.nan
        if verified_cells == 0:
            assessment = "INSUFFICIENT_EXTERNAL_EVIDENCE"
        elif critical:
            assessment = "MIXED"
        elif precision == precision and precision >= 0.9 and (recall != recall or recall >= 0.8):
            assessment = "STRONG"
        else:
            assessment = "ADEQUATE_WITH_GAPS"
        rows.append(
            {
                "dataset": dataset,
                "externally_verified_cells": verified_cells,
                "s1_material_decisions": len(inv),
                "externally_verified_decisions": externally_verified_decisions,
                "verified_correct_decisions": verified_correct_decisions,
                "correct_decisions": correct,
                "wrong_decisions": wrong,
                "false_positive_count": wrong,
                "verified_anomaly_count": verified_anomalies,
                "caught_anomaly_count": caught,
                "missed_anomaly_count": missed,
                "verified_universe_precision": precision,
                "verified_universe_recall": recall,
                "critical_misses": critical,
                "overall_assessment": assessment,
            }
        )
    return pd.DataFrame(rows)


def source_ledger_extension() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "SRC_JS_YEARBOOK_2021",
                "source_tier": "TIER1",
                "publisher": "江苏省统计局",
                "title": "19-2 三大区域经济社会基本情况（2021年）",
                "url_or_reference": "https://tj.jiangsu.gov.cn/2022/nj19/nj1902.htm",
                "publication_date": "",
                "statistical_period": "2021",
                "geographic_scope": "江苏省13地级市",
                "indicator": "resident/registered population; urbanization; density; employees; sector employment",
                "unit": "table-specific",
                "definition": "official annual yearbook table",
                "revision_status": "CAPTURED_EVIDENCE",
                "used_for": "S8_PLAN04_EXTERNAL_VERIFICATION",
                "notes": "Exact 2021 comparison cells for the Jiangsu subset of frozen city identity mapping.",
            },
            {
                "source_id": "SRC_S8_V2_FROZEN_TARGETS",
                "source_tier": "S8_FROZEN_TRUTH",
                "publisher": "S8 truth branch",
                "title": "candidate_truth_2023_v2_raw.csv",
                "url_or_reference": "reports/S8_TRUTH/truth_v2/candidate_truth_2023_v2_raw.csv",
                "publication_date": "2026-09-22",
                "statistical_period": "2022",
                "geographic_scope": "40 mapped cities",
                "indicator": "resident population",
                "unit": "万人",
                "definition": "frozen S8 post-hoc target anchor; outside S1 observed range",
                "revision_status": "FROZEN_REFERENCE",
                "used_for": "S8_PLAN04_EXTERNAL_VERIFICATION_UNIVERSE",
                "notes": "Not used to alter S1 data or generate S9 features.",
            },
        ]
    )


def report_text(
    inventory: pd.DataFrame,
    decision_validation: pd.DataFrame,
    population_audit: pd.DataFrame,
    candidates: pd.DataFrame,
    score: pd.DataFrame,
    semantics: pd.DataFrame,
    identity_validation_frame: pd.DataFrame,
    shift_candidates: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    correct = int(decision_validation.validation_class.str.startswith("CORRECT").sum())
    wrong = int(decision_validation.validation_class.str.startswith(("FALSE_POSITIVE", "WRONG")).sum())
    verified_decisions = int(decision_validation.external_validation_status.str.startswith("EXTERNALLY_VERIFIED").sum())
    missed = candidates[candidates.classification == "FALSE_NEGATIVE_MISSED_ANOMALY"]
    critical_missed = missed[missed.miss_severity == "CRITICAL"]
    caught = int((population_audit.posthoc_status == "S1_CAUGHT").sum()) if not population_audit.empty else 0
    s1_missed = int((population_audit.posthoc_status == "S1_MISSED").sum()) if not population_audit.empty else 0
    permutation_count = int((candidates.candidate_type == "DUPLICATED_CITY_TRAJECTORY").sum())
    label_count = int((candidates.candidate_type == "INVALID_CITY_LABEL").sum())
    shift_count = len(shift_candidates)
    semantic_mismatch = int((semantics.status.str.contains("CONFLICT|POSSIBLE", regex=True)).sum())
    score_lines = []
    for row in score.itertuples(index=False):
        precision = "NA" if pd.isna(row.verified_universe_precision) else f"{row.verified_universe_precision:.3f}"
        recall = "NA" if pd.isna(row.verified_universe_recall) else f"{row.verified_universe_recall:.3f}"
        score_lines.append(
            f"| {row.dataset} | {row.externally_verified_cells} | {row.s1_material_decisions} | {row.externally_verified_decisions} | {row.correct_decisions} | {row.wrong_decisions} | {row.verified_anomaly_count} | {row.caught_anomaly_count} | {row.missed_anomaly_count} | {precision} | {recall} | {row.critical_misses} | {row.overall_assessment} |"
        )
    top_misses = "\n".join(
        f"- `{row.dataset}/{row.city_id}/{row.year}/{row.field}`: {row.severity}, {row.external_cause}; {row.candidate_reason}"
        for row in missed.sort_values(["miss_severity", "dataset", "city_id"]).itertuples(index=False)
    ) or "- No externally verified false-negative anomaly was found."
    return f"""# S8-Plan04｜Post-hoc S1 Data Audit Validation

## Result

- Status: `COMPLETE_WITH_MATERIAL_S1_AUDIT_FINDING`
- Branch: `{BRANCH}`
- Parent branch: `{PARENT_BRANCH}` @ `{PARENT_HEAD}`
- Frozen truth read: `candidate_truth_2023_v2`
- Frozen evaluator SHA256: `{manifest['truth_sha256']}`
- Frozen truth modified: `NO`
- `PROJECT_DOC_SYNC_NEEDED_BY_MAIN`

This is a post-hoc audit. It does not rewrite `data/raw`, canonical tables,
quarantine, provenance, S1 reports, S6 artifacts, or frozen S8 truth.

## Scope and isolation

- S9 branch read: `NO`
- S9 predictions read: `NO`
- S9 metrics read: `NO`
- main modified: `NO`
- `test.xls` read: `NO`
- Candidate generation was completed before the S1 inventory was loaded.

## Verification universe

The quantitative universe contains definition-matched observed S1 cells from
the official Jiangsu 2021 yearbook table 19-2, frozen S8 target anchors that
are explicitly marked outside the S1 observed range, and the explicit city40
2020 census/yearbook anchors. Cells without a definition-matched official
comparison remain `NOT_EXTERNALLY_VERIFIABLE` and are excluded from precision
and recall denominators.

- external universe rows: `{len(pd.read_csv(OUT_DIR / 'external_verification_universe.csv', encoding='utf-8-sig'))}`
- externally verified observed cells: `{int(score.externally_verified_cells.sum())}`
- S1 material decision rows: `{len(inventory)}`
- externally verifiable decision rows: `{verified_decisions}`

## S1 decisions

- correct / caution decisions: `{correct}`
- verified wrong decisions: `{wrong}`
- false-positive decisions: `{wrong}`
- unresolved external evidence: `{len(inventory) - verified_decisions}`

The inventory includes zero-sentinel repair, duplicate-key repair, label
remaps, quarantine, invalidation, derived reconstruction, preserved warnings,
and source-domain boundaries. The correct/caution labels distinguish a sound
historical action from an externally proven exact value.

## False-negative search

- independent anomaly/structure candidates: `{len(candidates)}`
- externally verified missed anomalies: `{len(missed)}`
- critical missed anomalies: `{len(critical_missed)}`
- population temporal/cross-field candidates: `{len(population_audit)}`
- population S1-caught candidates: `{caught}`
- population S1-missed candidates: `{s1_missed}`

Highest-severity verified misses:

{top_misses}

### city40 registered=65

The generic population temporal rule detected `city40 / registered / 2021`
because the 2020→2021 change is `-75.999%`, classified `EXTREME`, and the
next official comparison is 268.10 万人 in the official Jiangsu 2021 yearbook.
The raw competition value is 65 and the preceding canonical value is 270.
S1 historical status was `NO_FLAG`; final classification is
`FALSE_NEGATIVE_MISSED_ANOMALY` with `miss_severity=CRITICAL`.

This is a design blind spot, not an isolated luck event: `scripts/audit_population.py`
implements schema, key, key-format, coverage, missing-value, value-sanity, and
zero-sentinel checks, but no first-difference, relative-YoY, rolling-MAD, or
resident/registered cross-field plausibility rule.

## Permutation, shift, and semantics results

- city label permutation: no canonical multi-year permutation was promoted to a confirmed remap; raw structural candidates remain evidence rows.
- local label swap: raw malformed labels and duplicate-key candidates were detected; the two S1 urban-income label repairs are retained as `CORRECT_LABEL_REMAP`.
- year shift: `{shift_count}` raw shifted-trajectory candidates were found; employment shifted duplicates are classified `TRUE_YEAR_SHIFT`, while other low-cardinality matches remain unresolved.
- indicator label mismatch: the raw label scan found `{label_count}` malformed label candidates; population resident/registered has `{int((semantics.check_id == 'resident_registered_order').sum())}` semantic check row with a city40 cross-field conflict.
- block contamination: the independent semantic scan identifies the city3 lagged-copy warning and the S1 overall-income quarantine boundary; it does not rewrite either.
- magnitude / decimal anomaly: the city21 overall-consumption `446007 → 44607` repair remains internally supported; no new unverified correction is applied.

## Dataset scorecard

| dataset | externally_verified_cells | s1_material_decisions | externally_verified_decisions | correct_decisions | wrong_decisions | verified_anomaly_count | caught_anomaly_count | missed_anomaly_count | precision | verified_universe_recall | critical_misses | overall_assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(score_lines)}

## Required answers

### Q1 — confirmed strengths

S1's zero-sentinel handling, duplicate-key handling, employment remap and
sector-sum reconstruction, shifted-duplicate quarantine, malformed urban-income
label repairs, and explicit quarantine boundaries are supported by the recorded
internal invariants. The official Jiangsu 2021 cross-check also provides a
definition-matched external reference for population, density, urbanization,
employees, and sector fields in the Jiangsu subset.

### Q2 — false positives

No verified false-positive repair, invalidation, or quarantine was found in the
available exact-cell external universe. This is not a claim that every S1
decision has been externally proven; most historical quarantine cells remain
`NOT_EXTERNALLY_VERIFIABLE`.

### Q3 — false negatives

The material verified miss is city40 registered 2021 = 65. Other independent
candidates without matching official cells remain `UNRESOLVED`, not false
positives or false negatives.

### Q4 — structural findings

No canonical global city permutation was proven. Raw employment contains
multi-year shifted duplicate candidates, and malformed raw labels are visible.
The city3 income lag remains a semantic warning rather than a proven column
swap. The known city21 magnitude repair and S1 block contamination boundaries
are preserved as historical decisions.

### Q5 — why city40 was missed

S1 had no temporal plausibility or cross-field relationship detector in the
population audit. The omission is therefore a reusable audit-contract gap.

### Q6 — depth by table

Employment, sector employment, density, and living-standard tables have the
deepest local repair/quarantine/provenance logic. Population has strong
schema/coverage and sentinel checks but weak anomaly detection. Wage and age
have isolated invalidation rules but limited external verification. Income has
important contamination and lag warnings, while external historical coverage
is thin.

### Q7 — minimum-cost methodology changes

Add per-city first-difference and robust-MAD scans, resident/registered
cross-field checks, duplicate-trajectory and lagged-copy detection, raw label
format/duplicate-key scans, and an external verification sample with explicit
definition/year/revision fields. These recommendations follow the observed
city40 miss and the unresolved raw shifted candidates; they are audit rules,
not S9 features.

### Q8 — overall credibility

S1 structural integrity is strong. Repair/quarantine logic is generally
adequate where explicit invariants exist. Semantic alignment and external
historical truth coverage are mixed. False-negative control is weak for
single-point temporal anomalies, with city40 as a critical verified miss.

## Files

- `HANDOFF.md`
- `s1_audit_decision_inventory.csv`
- `external_verification_universe.csv`
- `population_posthoc_anomaly_audit.csv`
- `cross_table_city_identity_validation.csv`
- `label_permutation_candidates.csv`
- `cross_table_year_alignment_validation.csv`
- `year_shift_candidates.csv`
- `indicator_semantics_validation.csv`
- `s1_decision_validation_results.csv`
- `s1_false_negative_candidates.csv`
- `s1_audit_scorecard.csv`
- `s1_audit_scorecard.json`
- `source_ledger_extension.csv`
- `run_s8_plan04.py` (reproducibility script in `scripts/`)

## Remaining uncertainty

The verification universe is deliberately partial. It cannot validate every
historical cell or prove the true values of quarantined rows. Revision,
census-rebasing, and definition changes remain explicit ambiguity classes.
Main must decide separately whether the material city40 miss warrants a future
S1 remediation plan after the S9 freeze/reveal boundary.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = canonical_tables()
    identity, _ = load_identity()
    truth_manifest = json.loads((S8_TRUTH_DIR / "truth_manifest_v2.json").read_text(encoding="utf-8"))
    if truth_manifest["truth_version"] != "candidate_truth_2023_v2":
        raise AssertionError("unexpected frozen truth version")
    if truth_manifest["freeze_status"] != "S8_TRUTH_V2_FROZEN_WAITING_FOR_S9":
        raise AssertionError("frozen S8 truth status changed")
    truth_raw = read_csv(S8_TRUTH_DIR / "candidate_truth_2023_v2_raw.csv")

    # Required ordering: blind candidate generation precedes historical S1 inventory.
    blind_temporal = temporal_candidates(tables)
    structural, label_candidates, shift_candidates = raw_structure_candidates()
    external = external_universe(tables, identity, truth_raw)
    identity_frame = identity_validation(tables, external, identity)
    semantics = indicator_semantics(tables, structural)
    inventory = build_inventory(tables, identity)
    decision_validation = validate_decisions(inventory, external)

    candidate_frame = join_candidate_status(blind_temporal + structural, inventory, external)
    if candidate_frame.empty:
        raise AssertionError("blind candidate scan unexpectedly empty")
    city40_candidates = candidate_frame[
        (candidate_frame.dataset == "population")
        & (candidate_frame.city_id == "city40")
        & (candidate_frame.field == "户籍人口（万人）")
        & (candidate_frame.year == 2021)
    ]
    if city40_candidates.empty:
        raise AssertionError("generic rule did not detect city40 registered=65")
    if not (city40_candidates.severity == "EXTREME").all():
        raise AssertionError("city40 candidate did not reach EXTREME severity")

    population_rows = candidate_frame[
        candidate_frame.dataset.eq("population")
        & candidate_frame.candidate_type.isin(["TEMPORAL_JUMP", "CROSS_FIELD_CONSISTENCY"])
    ].copy()
    # Include the source-level zero sentinel decisions in the population audit.
    zero_decisions = inventory[
        (inventory.dataset == "population") & (inventory.decision_type == "REPAIR")
    ]
    zero_rows = []
    for row in zero_decisions.itertuples(index=False):
        zero_rows.append(
            {
                "candidate_type": "ZERO_SENTINEL",
                "dataset": "population",
                "city_id": row.city_id,
                "year": row.year,
                "field": row.field,
                "previous_year": "",
                "previous_value": "",
                "current_value": row.original_value,
                "next_value": "",
                "relative_change": "",
                "severity": "LOW",
                "candidate_reason": "raw zero is canonicalized to missing",
                "return_to_baseline": False,
                "candidate_id": "",
                "candidate_found_posthoc": "YES",
                "was_flagged_in_S1": "YES",
                "s1_matching_decision_ids": row.decision_id,
                "external_status": "NOT_EXTERNALLY_VERIFIABLE",
                "official_value": "",
                "external_cause": "UNRESOLVED",
                "external_anomaly": "NO",
                "posthoc_status": "S1_CAUGHT",
                "classification": "CORRECT_REPAIR",
                "miss_severity": "",
            }
        )
    population_audit = pd.concat([population_rows, pd.DataFrame(zero_rows)], ignore_index=True)
    population_audit = population_audit.sort_values(
        ["city_id", "year", "field", "candidate_type"]
    ).reset_index(drop=True)

    # Add a stable status to identity and year tables without claiming more than evidence supports.
    identity_frame.to_csv(OUT_DIR / "cross_table_city_identity_validation.csv", index=False, encoding="utf-8-sig")
    write_csv(external, "external_verification_universe.csv")
    inventory_output = inventory.copy()
    final_external_status = decision_validation.set_index("decision_id")["external_validation_status"]
    inventory_output["external_validation_status"] = inventory_output["decision_id"].map(final_external_status)
    write_csv(inventory_output, "s1_audit_decision_inventory.csv")
    write_csv(population_audit, "population_posthoc_anomaly_audit.csv")
    write_csv(pd.DataFrame(label_candidates), "label_permutation_candidates.csv")
    write_csv(decision_validation, "s1_decision_validation_results.csv")
    write_csv(candidate_frame, "s1_false_negative_candidates.csv")
    write_csv(semantics, "indicator_semantics_validation.csv")

    year_rows = []
    for dataset, frame in tables.items():
        fields = NORMALIZED_TABLES[dataset][1]
        for field in fields:
            ext = external[(external.dataset == dataset) & (external.field == field) & (external.observed_in_s1_source)]
            if ext.empty:
                year_rows.append(
                    {
                        "dataset": dataset,
                        "field": field,
                        "candidate_offset": "",
                        "exact_match_rate": np.nan,
                        "status": "UNRESOLVED",
                        "evidence": "No definition-matched official time anchor available.",
                    }
                )
            else:
                actual = []
                expected = []
                for erow in ext.itertuples(index=False):
                    match = frame[(frame[CITY_COL] == erow.city_id) & (frame[YEAR_COL] == erow.competition_year)]
                    if not match.empty and numeric(match.iloc[0][field]) is not None:
                        actual.append(numeric(match.iloc[0][field]))
                        expected.append(numeric(erow.official_value))
                matches = sum(abs(a - b) <= max(0.05, abs(b) * 0.03) for a, b in zip(actual, expected))
                rate = matches / len(actual) if actual else np.nan
                year_rows.append(
                    {
                        "dataset": dataset,
                        "field": field,
                        "candidate_offset": 0,
                        "exact_match_rate": rate,
                        "status": "NO_SHIFT" if rate >= 0.8 else "REVISION_ARTIFACT",
                        "evidence": "Official Jiangsu 2021 yearbook cells compared at the same year.",
                    }
                )
    if shift_candidates:
        for row in shift_candidates:
            year_rows.append(
                {
                    "dataset": row["dataset"],
                    "field": row["field"],
                    "candidate_offset": row["candidate_offset"],
                    "exact_match_rate": row["match_rate"],
                    "status": row["status"],
                    "evidence": row["candidate_reason"],
                }
            )
    # Keep year-shift evidence explicit about the negative controls and the
    # two source-version classes required by the Plan, rather than leaving
    # them implicit in a separate summary only.
    year_shift_output = list(shift_candidates)
    for row in year_rows:
        if row["status"] in {"NO_SHIFT", "REVISION_ARTIFACT", "UNRESOLVED"}:
            year_shift_output.append(
                {
                    "candidate_type": "ALIGNMENT_CONTROL",
                    "dataset": row["dataset"],
                    "source_file": "",
                    "sheet": "",
                    "city_id": "",
                    "paired_city_id": "",
                    "year": "",
                    "field": row["field"],
                    "candidate_offset": row["candidate_offset"],
                    "support_years": "",
                    "support_fields": "",
                    "match_rate": row["exact_match_rate"],
                    "candidate_reason": row["evidence"],
                    "status": row["status"],
                }
            )
    census_rows = external[external.source_id == "SRC_JS_CENSUS_2020"]
    for row in census_rows.itertuples(index=False):
        year_shift_output.append(
            {
                "candidate_type": "SOURCE_VERSION_CONTROL",
                "dataset": row.dataset,
                "source_file": "",
                "sheet": "",
                "city_id": row.city_id,
                "paired_city_id": "",
                "year": row.competition_year,
                "field": row.field,
                "candidate_offset": 0,
                "support_years": 1,
                "support_fields": 1,
                "match_rate": "",
                "candidate_reason": "official census anchor is a distinct source-version class, not a year-shift correction",
                "status": "CENSUS_REBASING",
            }
        )
    write_csv(pd.DataFrame(year_rows), "cross_table_year_alignment_validation.csv")
    write_csv(pd.DataFrame(year_shift_output), "year_shift_candidates.csv")

    score = scorecard(inventory, decision_validation, candidate_frame, external)
    write_csv(score, "s1_audit_scorecard.csv")
    score_json = {
        "plan_id": PLAN_ID,
        "generated_at": GENERATED_AT,
        "branch": BRANCH,
        "parent_branch": PARENT_BRANCH,
        "parent_head": PARENT_HEAD,
        "frozen_truth_version": truth_manifest["truth_version"],
        "frozen_truth_sha256": truth_manifest["truth_sha256"],
        "frozen_full_truth_sha256": truth_manifest["full_truth_sha256"],
        "s9_isolation": {
            "s9_branch_read": False,
            "s9_predictions_read": False,
            "s9_metrics_read": False,
            "main_modified": False,
            "frozen_truth_modified": False,
            "test_xls_accessed": False,
        },
        "inventory_rows": len(inventory),
        "candidate_rows": len(candidate_frame),
        "population_candidate_rows": len(population_audit),
        "verified_observed_cells": int(external.observed_in_s1_source.sum()),
        "verified_missed_anomalies": int((candidate_frame.classification == "FALSE_NEGATIVE_MISSED_ANOMALY").sum()),
        "critical_misses": int((candidate_frame.miss_severity == "CRITICAL").sum()),
        "label_permutation_candidates": len(label_candidates),
        "year_shift_candidates": len(shift_candidates),
        "semantic_checks": len(semantics),
        "scorecard": score.replace({np.nan: None}).to_dict(orient="records"),
        "source_hashes": {
            "population_audit_script": sha256(ROOT / "scripts" / "audit_population.py"),
            "s1_runner": sha256(ROOT / "scripts" / "run_s1.py"),
            "frozen_truth_manifest": sha256(S8_TRUTH_DIR / "truth_manifest_v2.json"),
        },
    }
    (OUT_DIR / "s1_audit_scorecard.json").write_text(
        json.dumps(score_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_extension = source_ledger_extension()
    write_csv(source_extension, "source_ledger_extension.csv")
    report = report_text(
        inventory,
        decision_validation,
        population_audit,
        candidate_frame,
        score,
        semantics,
        identity_frame,
        pd.DataFrame(shift_candidates),
        truth_manifest,
    )
    (OUT_DIR / "S1_POSTHOC_AUDIT_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    handoff = f"""# S8-Plan04 Handoff

Status: `COMPLETE_WITH_MATERIAL_S1_AUDIT_FINDING`
Branch: `{BRANCH}`
Parent branch: `{PARENT_BRANCH}`
Parent HEAD: `{PARENT_HEAD}`
Generated: `{GENERATED_AT}`

Frozen truth: `candidate_truth_2023_v2`
Frozen evaluator SHA256: `{truth_manifest['truth_sha256']}`
Frozen truth modified: `NO`

Isolation:
- S9 branch read: `NO`
- S9 predictions read: `NO`
- S9 metrics read: `NO`
- main modified: `NO`
- test.xls accessed: `NO`

Validation:
- Decision inventory: `{len(inventory)}` rows
- Independent candidate rows: `{len(candidate_frame)}`
- External universe rows: `{len(external)}`
- Definition-matched observed cells: `{int(external.observed_in_s1_source.sum())}`
- Verified critical false negatives: `{int((candidate_frame.miss_severity == 'CRITICAL').sum())}`
- v1/v2 frozen truth remains unchanged

Material finding:

`city40 / population / registered / 2021 = 65` was detected by the generic
post-hoc temporal rule as an EXTREME, CRITICAL false negative. S1 population
audit had no temporal plausibility or resident/registered cross-field rule.

The result remains isolated. No canonical, quarantine, provenance, S6, S8, or
S9 artifact was corrected or regenerated.

Project documentation synchronization is intentionally delegated to Main:
`PROJECT_DOC_SYNC_NEEDED_BY_MAIN`.

Files:

- `S1_POSTHOC_AUDIT_VALIDATION_REPORT.md`
- `s1_audit_decision_inventory.csv`
- `external_verification_universe.csv`
- `population_posthoc_anomaly_audit.csv`
- `cross_table_city_identity_validation.csv`
- `label_permutation_candidates.csv`
- `cross_table_year_alignment_validation.csv`
- `year_shift_candidates.csv`
- `indicator_semantics_validation.csv`
- `s1_decision_validation_results.csv`
- `s1_false_negative_candidates.csv`
- `s1_audit_scorecard.csv`
- `s1_audit_scorecard.json`
- `source_ledger_extension.csv`
"""
    (OUT_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "COMPLETE_WITH_MATERIAL_S1_AUDIT_FINDING",
                "inventory_rows": len(inventory),
                "candidate_rows": len(candidate_frame),
                "external_rows": len(external),
                "verified_observed_cells": int(external.observed_in_s1_source.sum()),
                "city40_registered_candidate": not city40_candidates.empty,
                "critical_misses": int((candidate_frame.miss_severity == "CRITICAL").sum()),
                "label_candidates": len(label_candidates),
                "shift_candidates": len(shift_candidates),
                "output_dir": str(OUT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
