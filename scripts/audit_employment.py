from pathlib import Path

import pandas as pd


# =========================================================
# Paths / Constants
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"
QUARANTINE_DIR = ROOT / "data" / "quarantine"

EMPLOYMENT_FILE = DATA_DIR / "就业信息.xlsx"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"
EXPECTED_CITIES = {f"city{i}" for i in range(1, 41)}

UNEMPLOYMENT_COLUMN = "unemploymentRate"
EMPLOYEES_COLUMN = "employeesNumber"

SECTOR_COLUMNS = [
    "pi_Employment",
    "si_Employment",
    "ti_Employment",
]

PRIMARY_REPAIR_CITIES = {
    "city1",
    "city24",
    "city28",
    "city32",
    "city36",
    "city39",
}
# =========================================================
# Common Helpers
# =========================================================

def load_employment_sheet(sheet_name):
    """读取就业信息.xlsx 中指定 Sheet。"""
    return pd.read_excel(
        EMPLOYMENT_FILE,
        sheet_name=sheet_name,
    )


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_schema(df):
    print("\n=== SCHEMA ===")
    print(f"shape = {df.shape}")
    print(f"columns = {df.columns.tolist()}")
    print(df.dtypes)


def check_key(df):
    """检查 (城市名称, 年份) 唯一性。"""

    duplicate_mask = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    )

    print("\n=== KEY CHECK ===")
    print(f"rows = {len(df)}")

    print(
        "unique_city_year_keys =",
        df[
            ["城市名称", "年份"]
        ].drop_duplicates().shape[0],
    )

    print(
        f"duplicate_rows = "
        f"{duplicate_mask.sum()}"
    )

    if duplicate_mask.any():
        print("\n=== DUPLICATE CITY COUNTS ===")
        print(
            df.loc[
                duplicate_mask,
                "城市名称",
            ]
            .value_counts()
            .to_string()
        )


def check_key_format(df):
    """检查城市标签是否符合 city1~city40。"""

    valid_city = (
        df["城市名称"]
        .str.fullmatch(CITY_PATTERN)
    )

    print("\n=== KEY FORMAT ===")
    print(
        f"invalid_city_rows = "
        f"{(~valid_city).sum()}"
    )

    if (~valid_city).any():
        print(
            df.loc[
                ~valid_city,
                "城市名称",
            ]
            .drop_duplicates()
            .map(repr)
            .to_string(index=False)
        )


def check_entity_coverage(df):
    """检查 city1~city40 实体覆盖。"""

    observed_cities = set(
        df["城市名称"].unique()
    )

    print("\n=== ENTITY COVERAGE ===")
    print(
        f"expected_cities = "
        f"{len(EXPECTED_CITIES)}"
    )
    print(
        f"observed_cities = "
        f"{len(observed_cities)}"
    )

    print(
        "missing_cities =",
        sorted(
            EXPECTED_CITIES - observed_cities
        ),
    )

    print(
        "unexpected_cities =",
        sorted(
            observed_cities - EXPECTED_CITIES
        ),
    )


def check_coverage(df):
    """检查整体年份范围与城市内部缺年。"""

    print("\n=== COVERAGE ===")
    print(f"year_min = {df['年份'].min()}")
    print(f"year_max = {df['年份'].max()}")
    print(
        f"years = "
        f"{df['年份'].nunique()}"
    )

    gaps = []

    for city, city_df in df.groupby(
        "城市名称"
    ):
        observed_years = set(
            city_df["年份"]
        )

        year_min = min(observed_years)
        year_max = max(observed_years)

        expected_years = set(
            range(
                year_min,
                year_max + 1,
            )
        )

        missing_years = sorted(
            expected_years
            - observed_years
        )

        if missing_years:
            gaps.append(
                (city, missing_years)
            )

    print(
        f"cities_with_internal_gaps = "
        f"{len(gaps)}"
    )

    for city, missing_years in gaps:
        print(
            f"{city}: "
            f"{missing_years}"
        )


def check_missing(df):
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())


# =========================================================
# 1. 城镇失业率
# =========================================================

def check_unemployment_value_domain(df):
    values = (
        df[UNEMPLOYMENT_COLUMN]
        .dropna()
    )

    print(
        "\n=== UNEMPLOYMENT "
        "VALUE DOMAIN ==="
    )

    print(f"count = {len(values)}")
    print(f"min = {values.min()}")
    print(f"max = {values.max()}")

    print(
        f"zero_count = "
        f"{(values == 0).sum()}"
    )

    print(
        f"negative_count = "
        f"{(values < 0).sum()}"
    )

    print(
        f"above_100 = "
        f"{(values > 100).sum()}"
    )

    zero_rows = df[
        df[UNEMPLOYMENT_COLUMN] == 0
    ]

    if not zero_rows.empty:
        print(
            "\n=== ZERO "
            "UNEMPLOYMENT ROWS ==="
        )
        print(
            zero_rows.to_string(
                index=False
            )
        )

    missing_rows = df[
        df[
            UNEMPLOYMENT_COLUMN
        ].isna()
    ]

    if not missing_rows.empty:
        print(
            "\n=== MISSING "
            "UNEMPLOYMENT ROWS ==="
        )
        print(
            missing_rows.to_string(
                index=False
            )
        )


def normalize_unemployment(df):
    """
    当前没有 evidence-backed repair。

    保留：
    - city13 / city34 整实体缺失
    - internal year gaps
    - city35-2022 NaN
    - city21-2011 unemploymentRate=0
    """
    return df.copy()


def verify_unemployment(df):
    print(
        "\n=== NORMALIZED "
        "VERIFICATION ==="
    )

    check_key(df)

    assert (
        df.duplicated(
            ["城市名称", "年份"]
        ).sum()
        == 0
    )


def save_unemployment(df):
    output_path = (
        NORMALIZED_DIR
        / "就业信息_城镇失业率.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nsaved: {output_path}"
    )
    print(f"shape: {df.shape}")


def audit_unemployment():
    print_section(
        "就业信息.xlsx / 城镇失业率"
    )

    raw_df = load_employment_sheet(
        "城镇失业率"
    )

    check_schema(raw_df)
    check_key(raw_df)
    check_key_format(raw_df)
    check_entity_coverage(raw_df)
    check_coverage(raw_df)
    check_missing(raw_df)

    check_unemployment_value_domain(
        raw_df
    )

    normalized_df = (
        normalize_unemployment(
            raw_df
        )
    )

    verify_unemployment(
        normalized_df
    )

    save_unemployment(
        normalized_df
    )


# =========================================================
# 2. 从业人员数 v2
# =========================================================

UNTRUSTED_SECTOR_KEYS_FOR_EMPLOYEE_DERIVATION = {
    ("city14", 2018),
    ("city14", 2019),
}

EXPECTED_SECTOR_CONFLICT_KEYS = {
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


def normalize_employees_source_v2(df):
    """
    第二轮 cross-source audit 后得到的 source-level 修订。

    高置信度 relabel / year shift：
        raw city14 -> city16, year +1
        raw city16 -> city6,  year -1
        raw city6  -> city31, year +1
        raw city33 第二 block -> city34

    高置信度 quarantine：
        city25 第二 block：city2 的 year-shifted duplicate
        city29 第二 block：city8 的 year-shifted duplicate
        raw city31 两套候选：都不再作为 canonical city31

    此函数只生成独立 source-derived employees。
    """

    working = df.copy()
    working["城市名称"] = (
        working["城市名称"]
        .astype("string")
        .str.strip()
    )
    working["source_index"] = working.index

    # ---------- fail-closed source structure ----------
    assert working.loc[94:115, "城市名称"].eq("city14").all()
    assert working.loc[94:115, "年份"].tolist() == list(range(2000, 2022))

    assert working.loc[196:217, "城市名称"].eq("city6").all()
    assert working.loc[196:217, "年份"].tolist() == list(range(2000, 2022))

    assert working.loc[318:338, "城市名称"].eq("city16").all()
    assert working.loc[318:338, "年份"].tolist() == list(range(2001, 2022))

    assert working.loc[486:504, "城市名称"].eq("city25").all()
    assert working.loc[486:504, "年份"].tolist() == list(range(2021, 2002, -1))

    assert working.loc[563:581, "城市名称"].eq("city29").all()
    assert working.loc[563:581, "年份"].tolist() == list(range(2021, 2002, -1))

    city31_raw = working.loc[582:623]
    assert len(city31_raw) == 42
    assert city31_raw["城市名称"].eq("city31").all()
    assert city31_raw.groupby("年份").size().eq(2).all()
    assert set(city31_raw["年份"]) == set(range(2001, 2022))

    assert working.loc[661:679, "城市名称"].eq("city33").all()
    assert working.loc[661:679, "年份"].tolist() == list(range(2021, 2002, -1))

    # ---------- shifted-duplicate evidence ----------
    city2_lookup = (
        working[working["城市名称"] == "city2"]
        .set_index("年份")[EMPLOYEES_COLUMN]
    )
    city25_b = working.loc[486:504].copy()
    city25_b["suspected_target_year"] = city25_b["年份"] - 1
    city25_b["target_value"] = city25_b["suspected_target_year"].map(city2_lookup)
    assert (city25_b[EMPLOYEES_COLUMN] == city25_b["target_value"]).all()

    city8_lookup = (
        working[working["城市名称"] == "city8"]
        .set_index("年份")[EMPLOYEES_COLUMN]
    )
    city29_b = working.loc[563:581].copy()
    city29_b["suspected_target_year"] = city29_b["年份"] - 1
    city29_b["target_value"] = city29_b["suspected_target_year"].map(city8_lookup)
    assert (city29_b[EMPLOYEES_COLUMN] == city29_b["target_value"]).all()

    # ---------- quarantine ----------
    parts = []

    q_city25 = working.loc[
        486:504,
        ["source_index", "城市名称", "年份", EMPLOYEES_COLUMN],
    ].copy()
    q_city25["reason"] = (
        "shifted duplicate block: matches city2 at source_year-1"
    )
    q_city25["suspected_target_city"] = "city2"
    q_city25["suspected_year_shift"] = -1
    parts.append(q_city25)

    q_city29 = working.loc[
        563:581,
        ["source_index", "城市名称", "年份", EMPLOYEES_COLUMN],
    ].copy()
    q_city29["reason"] = (
        "shifted duplicate block: matches city8 at source_year-1"
    )
    q_city29["suspected_target_city"] = "city8"
    q_city29["suspected_year_shift"] = -1
    parts.append(q_city29)

    q_city31 = working.loc[
        582:623,
        ["source_index", "城市名称", "年份", EMPLOYEES_COLUMN],
    ].copy()
    q_city31["reason"] = (
        "original city31 duplicate block: neither candidate series is "
        "supported as canonical city31 after cross-source remapping"
    )
    q_city31["suspected_target_city"] = pd.NA
    q_city31["suspected_year_shift"] = pd.NA
    parts.append(q_city31)

    quarantine = pd.concat(parts, ignore_index=True)
    quarantine_indices = set(quarantine["source_index"])

    # ---------- remove quarantined rows ----------
    normalized = working[
        ~working["source_index"].isin(quarantine_indices)
    ].copy()

    # raw city14 -> canonical city16, year +1
    mask = normalized["source_index"].between(94, 115)
    normalized.loc[mask, "城市名称"] = "city16"
    normalized.loc[mask, "年份"] = normalized.loc[mask, "年份"] + 1

    # raw city6 -> canonical city31, year +1
    mask = normalized["source_index"].between(196, 217)
    normalized.loc[mask, "城市名称"] = "city31"
    normalized.loc[mask, "年份"] = normalized.loc[mask, "年份"] + 1

    # raw city16 -> canonical city6, year -1
    mask = normalized["source_index"].between(318, 338)
    normalized.loc[mask, "城市名称"] = "city6"
    normalized.loc[mask, "年份"] = normalized.loc[mask, "年份"] - 1

    # raw city33 second block -> canonical city34
    mask = normalized["source_index"].between(661, 679)
    normalized.loc[mask, "城市名称"] = "city34"

    normalized = normalized[
        ["城市名称", "年份", EMPLOYEES_COLUMN]
    ].copy()

    return normalized, quarantine


def verify_employees_source_mapping(employees_source_df, quarantine_df, sector_df):
    """验证 source mapping；sector 只作为独立外部约束，不参与生成 source rows。"""

    duplicate_rows = employees_source_df.duplicated(
        ["城市名称", "年份"], keep=False
    ).sum()
    observed_cities = set(employees_source_df["城市名称"])

    print("\n=== EMPLOYEES V2 SOURCE VERIFICATION ===")
    print(f"source_rows = {len(employees_source_df)}")
    print(f"unique_keys = {employees_source_df[['城市名称', '年份']].drop_duplicates().shape[0]}")
    print(f"duplicate_rows = {duplicate_rows}")
    print(f"cities = {len(observed_cities)}")
    print("missing_cities =", sorted(EXPECTED_CITIES - observed_cities))
    print(f"quarantine_rows = {len(quarantine_df)}")

    assert len(employees_source_df) == 682
    assert duplicate_rows == 0
    assert len(observed_cities) == 35
    assert EXPECTED_CITIES - observed_cities == {
        "city14", "city26", "city30", "city37", "city40"
    }
    assert len(quarantine_df) == 80

    # remapped coverage
    expected_ranges = {
        "city6": (21, 2000, 2020),
        "city16": (22, 2001, 2022),
        "city31": (22, 2001, 2022),
        "city34": (19, 2003, 2021),
    }
    for city, (expected_n, expected_min, expected_max) in expected_ranges.items():
        city_df = employees_source_df[employees_source_df["城市名称"] == city]
        assert len(city_df) == expected_n
        assert city_df["年份"].min() == expected_min
        assert city_df["年份"].max() == expected_max

    # compact cross-source proof for remapped blocks
    sector_check = sector_df.copy()
    sector_check["sector_total"] = (
        sector_check["pi_Employment"]
        + sector_check["si_Employment"]
        + sector_check["ti_Employment"]
    )
    merged = employees_source_df.merge(
        sector_check[["城市名称", "年份", "sector_total"]],
        on=["城市名称", "年份"],
        how="inner",
        validate="one_to_one",
    )
    merged["abs_difference"] = (
        merged[EMPLOYEES_COLUMN] - merged["sector_total"]
    ).abs()

    excluded_years = {
        "city6": {2018, 2019},
        "city16": {2019},
        "city31": {2019, 2020},
        "city34": set(),
    }
    expected_good = {
        "city6": 13,
        "city16": 14,
        "city31": 13,
        "city34": 19,
    }
    for city in expected_good:
        check = merged[merged["城市名称"] == city].copy()
        check = check[~check["年份"].isin(excluded_years[city])]
        assert len(check) == expected_good[city]
        assert (check["abs_difference"] <= 1).all()


def derive_missing_employees_from_sector(employees_source_df, sector_df):
    """
    用可信三产业总和重建 source employee 缺失 key。

    仅填 source 不存在的 key；不覆盖 source-derived value。
    已知不可信 sector keys 不参与推导。
    """

    sector = sector_df.copy()
    sector["sector_total"] = (
        sector["pi_Employment"]
        + sector["si_Employment"]
        + sector["ti_Employment"]
    )
    assert sector.duplicated(["城市名称", "年份"]).sum() == 0

    source_keys = set(zip(
        employees_source_df["城市名称"],
        employees_source_df["年份"],
    ))
    sector["key"] = list(zip(sector["城市名称"], sector["年份"]))

    candidates = sector[~sector["key"].isin(source_keys)].copy()
    assert len(candidates) == 93

    candidates["trusted_for_derivation"] = ~candidates["key"].isin(
        UNTRUSTED_SECTOR_KEYS_FOR_EMPLOYEE_DERIVATION
    )
    rejected = candidates[~candidates["trusted_for_derivation"]].copy()
    trusted = candidates[candidates["trusted_for_derivation"]].copy()

    assert len(rejected) == 2
    assert set(zip(rejected["城市名称"], rejected["年份"])) == (
        UNTRUSTED_SECTOR_KEYS_FOR_EMPLOYEE_DERIVATION
    )
    assert len(trusted) == 91

    derived = trusted[
        [
            "城市名称", "年份", "sector_total",
            "pi_Employment", "si_Employment", "ti_Employment",
        ]
    ].copy()
    derived = derived.rename(columns={"sector_total": EMPLOYEES_COLUMN})
    derived["source_type"] = "sector_sum_reconstruction"
    derived["derivation_rule"] = (
        "employeesNumber = pi_Employment + si_Employment + ti_Employment"
    )
    derived["evidence"] = (
        "cross-source invariant validated on trusted overlapping observations"
    )

    return derived, rejected


def combine_employees_source_and_derived(employees_source_df, derived_df):
    source_keys = set(zip(
        employees_source_df["城市名称"], employees_source_df["年份"]
    ))
    derived_keys = set(zip(
        derived_df["城市名称"], derived_df["年份"]
    ))
    assert source_keys & derived_keys == set()

    derived_core = derived_df[["城市名称", "年份", EMPLOYEES_COLUMN]].copy()
    combined = pd.concat(
        [employees_source_df, derived_core],
        ignore_index=True,
    )
    return (
        combined
        .sort_values(["城市名称", "年份"])
        .reset_index(drop=True)
    )


def get_internal_year_gaps(df):
    gaps = {}
    for city, city_df in df.groupby("城市名称"):
        years = set(city_df["年份"])
        expected = set(range(min(years), max(years) + 1))
        missing = sorted(expected - years)
        if missing:
            gaps[city] = missing
    return gaps


def verify_employees_final(combined_df, derived_df, rejected_df):
    duplicate_rows = combined_df.duplicated(
        ["城市名称", "年份"], keep=False
    ).sum()
    observed_cities = set(combined_df["城市名称"])
    gaps = get_internal_year_gaps(combined_df)

    print("\n=== EMPLOYEES V2 FINAL VERIFICATION ===")
    print(f"rows = {len(combined_df)}")
    print(f"unique_keys = {combined_df[['城市名称', '年份']].drop_duplicates().shape[0]}")
    print(f"duplicate_rows = {duplicate_rows}")
    print(f"cities = {len(observed_cities)}")
    print("missing_cities =", sorted(EXPECTED_CITIES - observed_cities))
    print(f"derived_rows = {len(derived_df)}")
    print(f"rejected_sector_rows = {len(rejected_df)}")
    print("internal_gaps =", gaps)

    assert len(combined_df) == 773
    assert duplicate_rows == 0
    assert observed_cities == EXPECTED_CITIES
    assert len(derived_df) == 91
    assert len(rejected_df) == 2
    assert gaps == {
        "city1": [2020],
        "city4": [2020],
        "city7": [2020],
        "city10": [2020],
        "city11": [2020],
        "city14": [2018, 2019],
    }

    expected_counts = {
        "city6": 1,
        "city14": 14,
        "city26": 19,
        "city30": 19,
        "city37": 19,
        "city40": 19,
    }
    actual_counts = derived_df.groupby("城市名称").size().to_dict()
    assert actual_counts == expected_counts

    assert (
        derived_df[EMPLOYEES_COLUMN]
        - (
            derived_df["pi_Employment"]
            + derived_df["si_Employment"]
            + derived_df["ti_Employment"]
        )
    ).abs().le(1e-9).all()


def check_employees_value_domain(df):
    values = df[EMPLOYEES_COLUMN].dropna()
    print("\n=== EMPLOYEES VALUE DOMAIN ===")
    print(f"count = {len(values)}")
    print(f"min = {values.min()}")
    print(f"max = {values.max()}")
    print(f"zero_count = {(values == 0).sum()}")
    print(f"negative_count = {(values < 0).sum()}")


def save_employees_v2(normalized_df, quarantine_df, provenance_df):
    normalized_path = NORMALIZED_DIR / "就业信息_从业人员数.csv"
    quarantine_path = QUARANTINE_DIR / "就业信息_从业人员数_quarantine.csv"
    provenance_path = NORMALIZED_DIR / "就业信息_从业人员数_provenance.csv"

    normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    quarantine_df.to_csv(quarantine_path, index=False, encoding="utf-8-sig")
    provenance_df.to_csv(provenance_path, index=False, encoding="utf-8-sig")

    print(f"\nnormalized: {normalized_path} {normalized_df.shape}")
    print(f"quarantine: {quarantine_path} {quarantine_df.shape}")
    print(f"provenance: {provenance_path} {provenance_df.shape}")


# =========================================================
# 3. 第一、二、三产业就业人数 v2
# =========================================================


def check_sector_value_domain(df):
    print("\n=== SECTOR VALUE DOMAIN ===")
    for column in SECTOR_COLUMNS:
        values = df[column].dropna()
        print(f"\n--- {column} ---")
        print(f"count = {len(values)}")
        print(f"min = {values.min()}")
        print(f"max = {values.max()}")
        print(f"zero_count = {(values == 0).sum()}")
        print(f"negative_count = {(values < 0).sum()}")


def build_sector_v2(sector_raw_df, employees_source_df):
    """
    1. 六城 primary repair；
    2. 只用独立 source-derived employees 验证 sector；
    3. 明确冲突行保留 key，但三个 component 全部失效为 NA；
    4. raw component 值进入 quarantine。
    """

    sector = sector_raw_df.copy().reset_index(names="source_index")
    sector["source_position"] = sector.groupby("城市名称").cumcount() + 1

    for column in SECTOR_COLUMNS:
        sector[f"raw_{column}"] = sector[column]

    sector = sector.merge(
        employees_source_df[["城市名称", "年份", EMPLOYEES_COLUMN]],
        on=["城市名称", "年份"],
        how="left",
        validate="one_to_one",
    )

    primary_mask = (
        sector["城市名称"].isin(PRIMARY_REPAIR_CITIES)
        & sector[EMPLOYEES_COLUMN].notna()
    )
    assert primary_mask.sum() == 103

    sector.loc[primary_mask, "pi_Employment"] = (
        sector.loc[primary_mask, EMPLOYEES_COLUMN]
        - sector.loc[primary_mask, "si_Employment"]
        - sector.loc[primary_mask, "ti_Employment"]
    )

    sector["sector_total"] = (
        sector["pi_Employment"]
        + sector["si_Employment"]
        + sector["ti_Employment"]
    )
    sector["abs_difference"] = (
        sector["sector_total"] - sector[EMPLOYEES_COLUMN]
    ).abs()

    assert (sector.loc[primary_mask, "abs_difference"] <= 1e-9).all()

    comparable = sector[sector[EMPLOYEES_COLUMN].notna()].copy()
    conflicts = comparable[comparable["abs_difference"] > 1].copy()
    conflict_keys = set(zip(conflicts["城市名称"], conflicts["年份"]))

    assert len(conflicts) == 18
    assert conflict_keys == EXPECTED_SECTOR_CONFLICT_KEYS

    quarantine = conflicts[
        [
            "source_index", "source_position", "城市名称", "年份",
            "raw_pi_Employment", "raw_si_Employment", "raw_ti_Employment",
            "sector_total", EMPLOYEES_COLUMN, "abs_difference",
        ]
    ].copy()
    quarantine["reason"] = (
        "cross-source sector conflict: |pi+si+ti-employeesNumber| > 1; "
        "component-level true values unresolved"
    )

    normalized = sector[
        [
            "source_index", "城市名称", "年份",
            "pi_Employment", "si_Employment", "ti_Employment",
        ]
    ].copy()
    normalized[SECTOR_COLUMNS] = normalized[SECTOR_COLUMNS].astype("Float64")

    conflict_mask = normalized["source_index"].isin(set(conflicts["source_index"]))
    assert conflict_mask.sum() == 18
    normalized.loc[conflict_mask, SECTOR_COLUMNS] = pd.NA

    normalized = normalized[
        ["城市名称", "年份", *SECTOR_COLUMNS]
    ].copy()

    return normalized, quarantine, conflicts


def verify_sector_v2(normalized_df, quarantine_df, employees_source_df):
    duplicate_rows = normalized_df.duplicated(
        ["城市名称", "年份"], keep=False
    ).sum()

    print("\n=== SECTOR V2 FINAL VERIFICATION ===")
    print(f"rows = {len(normalized_df)}")
    print(f"unique_keys = {normalized_df[['城市名称', '年份']].drop_duplicates().shape[0]}")
    print(f"duplicate_rows = {duplicate_rows}")
    print(f"cities = {normalized_df['城市名称'].nunique()}")
    print("missing values:")
    print(normalized_df[SECTOR_COLUMNS].isna().sum().to_string())
    print(f"quarantine_rows = {len(quarantine_df)}")

    assert len(normalized_df) == 698
    assert duplicate_rows == 0
    assert normalized_df["城市名称"].nunique() == 40
    assert len(quarantine_df) == 18

    for column in SECTOR_COLUMNS:
        assert normalized_df[column].isna().sum() == 18

    missing_keys = set(zip(
        normalized_df.loc[normalized_df["pi_Employment"].isna(), "城市名称"],
        normalized_df.loc[normalized_df["pi_Employment"].isna(), "年份"],
    ))
    assert missing_keys == EXPECTED_SECTOR_CONFLICT_KEYS

    # primary repair 仍必须满足独立 employee 总量约束
    repaired_rows = (
        normalized_df[
            normalized_df["城市名称"].isin(PRIMARY_REPAIR_CITIES)
        ]
        .merge(
            employees_source_df[["城市名称", "年份", EMPLOYEES_COLUMN]],
            on=["城市名称", "年份"],
            how="inner",
            validate="one_to_one",
        )
    )
    repaired_rows["sector_total"] = (
        repaired_rows["pi_Employment"]
        + repaired_rows["si_Employment"]
        + repaired_rows["ti_Employment"]
    )
    assert len(repaired_rows) == 103
    assert (
        repaired_rows["sector_total"] == repaired_rows[EMPLOYEES_COLUMN]
    ).all()

    # 剩余极小 primary 值仍只允许来自 city3
    small_primary = normalized_df[
        normalized_df["pi_Employment"].isin([0, 1])
    ]
    assert set(small_primary["城市名称"]) == {"city3"}


def save_sector_v2(normalized_df, quarantine_df):
    normalized_path = NORMALIZED_DIR / "就业信息_第一二三产业就业人数.csv"
    quarantine_path = QUARANTINE_DIR / "就业信息_第一二三产业就业人数_quarantine.csv"

    normalized_df.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    quarantine_df.to_csv(quarantine_path, index=False, encoding="utf-8-sig")

    print(f"\nnormalized: {normalized_path} {normalized_df.shape}")
    print(f"quarantine: {quarantine_path} {quarantine_df.shape}")


def save_employment_repair_report():
    report = pd.DataFrame(
        [
            {
                "table": "从业人员数",
                "scope": "raw city14 block",
                "action": "relabel_and_year_shift",
                "result": "city16, year+1",
                "evidence": "cross-source sector-total invariant",
            },
            {
                "table": "从业人员数",
                "scope": "raw city16 block",
                "action": "relabel_and_year_shift",
                "result": "city6, year-1",
                "evidence": "cross-source sector-total invariant",
            },
            {
                "table": "从业人员数",
                "scope": "raw city6 block",
                "action": "relabel_and_year_shift",
                "result": "city31, year+1",
                "evidence": "cross-source sector-total invariant",
            },
            {
                "table": "从业人员数",
                "scope": "raw city33 second block",
                "action": "relabel",
                "result": "city34",
                "evidence": "cross-source sector-total invariant",
            },
            {
                "table": "从业人员数",
                "scope": "city25 second block",
                "action": "quarantine",
                "result": "19 shifted-duplicate rows",
                "evidence": "19/19 exact match to city2 at source_year-1",
            },
            {
                "table": "从业人员数",
                "scope": "city29 second block",
                "action": "quarantine",
                "result": "19 shifted-duplicate rows",
                "evidence": "19/19 exact match to city8 at source_year-1",
            },
            {
                "table": "从业人员数",
                "scope": "raw city31 duplicate block",
                "action": "quarantine",
                "result": "42 rows",
                "evidence": "canonical city31 supported by remapped raw city6 block",
            },
            {
                "table": "从业人员数",
                "scope": "source-missing keys",
                "action": "sector_sum_reconstruction",
                "result": "91 rows",
                "evidence": "validated employees ≈ pi+si+ti invariant",
            },
            {
                "table": "第一二三产业就业人数",
                "scope": "primary employment",
                "action": "repair pi_Employment",
                "result": "103 rows across 6 cities",
                "evidence": "employees-si-ti with exact post-repair total consistency",
            },
            {
                "table": "第一二三产业就业人数",
                "scope": "cross-source conflict rows",
                "action": "invalidate_components_to_NA_and_quarantine",
                "result": "18 rows",
                "evidence": "|pi+si+ti-employeesNumber| > 1 against independent source employees",
            },
        ]
    )

    output_path = NORMALIZED_DIR / "就业信息_repair_report.csv"
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"saved repair report: {output_path}")


# =========================================================
# Combined audits
# =========================================================


def audit_employees_and_sector():
    print_section("就业信息.xlsx / 从业人员数 v2 + 三产业就业 v2")

    employees_raw = load_employment_sheet("从业人员数")
    sector_raw = load_employment_sheet("第一、二、三产业就业人数")

    print("\n--- RAW EMPLOYEES ---")
    check_schema(employees_raw)
    check_key(employees_raw)
    check_key_format(employees_raw.assign(
        城市名称=employees_raw["城市名称"].astype("string").str.strip()
    ))

    employees_source, employees_quarantine = normalize_employees_source_v2(
        employees_raw
    )
    verify_employees_source_mapping(
        employees_source,
        employees_quarantine,
        sector_raw,
    )

    derived_df, rejected_df = derive_missing_employees_from_sector(
        employees_source,
        sector_raw,
    )
    employees_final = combine_employees_source_and_derived(
        employees_source,
        derived_df,
    )
    verify_employees_final(
        employees_final,
        derived_df,
        rejected_df,
    )

    print("\n--- FINAL EMPLOYEES ---")
    check_key_format(employees_final)
    check_entity_coverage(employees_final)
    check_coverage(employees_final)
    check_missing(employees_final)
    check_employees_value_domain(employees_final)

    save_employees_v2(
        employees_final,
        employees_quarantine,
        derived_df,
    )

    print("\n--- RAW SECTOR ---")
    check_schema(sector_raw)
    check_key(sector_raw)
    check_key_format(sector_raw)
    check_entity_coverage(sector_raw)
    check_coverage(sector_raw)
    check_missing(sector_raw)
    check_sector_value_domain(sector_raw)

    sector_final, sector_quarantine, conflicts = build_sector_v2(
        sector_raw,
        employees_source,
    )
    verify_sector_v2(
        sector_final,
        sector_quarantine,
        employees_source,
    )

    print("\n=== SECTOR CONFLICTS BY SOURCE POSITION ===")
    print(
        conflicts["source_position"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n--- FINAL SECTOR ---")
    check_key(sector_final)
    check_key_format(sector_final)
    check_entity_coverage(sector_final)
    check_coverage(sector_final)
    check_missing(sector_final)
    check_sector_value_domain(sector_final)

    save_sector_v2(
        sector_final,
        sector_quarantine,
    )

    save_employment_repair_report()


# =========================================================
# Entry
# =========================================================


def main():
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    audit_unemployment()
    audit_employees_and_sector()


if __name__ == "__main__":
    main()
