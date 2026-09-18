from pathlib import Path

import pandas as pd


# =========================================================
# Paths / Constants
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"
QUARANTINE_DIR = ROOT / "data" / "quarantine"

LIVING_FILE = DATA_DIR / "生活水平.xlsx"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"

EXPECTED_CITIES = {
    f"city{i}"
    for i in range(1, 41)
}


# =========================================================
# Sheet metadata
# =========================================================

SHEET_META = {
    "人均可支配收入": {
        "raw_indicator": "disposableIcome",
        "value_name": "disposableIncome",
        "output": "生活水平_人均可支配收入.csv",
    },

    "人均消费支出": {
        "raw_indicator": "consumptionExpenditures",
        "value_name": "consumptionExpenditures",
        "output": "生活水平_人均消费支出.csv",
    },

    "城镇居民消费支出": {
        "raw_indicator": "towner_ ConsumptionExpenditures",
        "value_name": "urbanConsumptionExpenditures",
        "output": "生活水平_城镇居民消费支出.csv",
    },

    "农村居民消费支出": {
        "raw_indicator": "rural_ConsumptionExpenditures",
        "value_name": "ruralConsumptionExpenditures",
        "output": "生活水平_农村居民消费支出.csv",
    },

    "城镇居民人均收入": {
        "raw_indicator": "towner_disposableIcome",
        "value_name": "urbanDisposableIncome",
        "output": "生活水平_城镇居民人均收入.csv",
    },

    "农村居民人均收入": {
        "raw_indicator": "rural_disposableIcome",
        "value_name": "ruralDisposableIncome",
        "output": "生活水平_农村居民人均收入.csv",
    },
}


# =========================================================
# Known coverage
# =========================================================

MISSING_CITIES = {
    "人均可支配收入": {
        "city2", "city5", "city8", "city12", "city13",
        "city15", "city18", "city22", "city24", "city26",
        "city28", "city30", "city32", "city34", "city36",
        "city37", "city39", "city40",
    },

    "人均消费支出": {
        "city2", "city5", "city8", "city12", "city13",
        "city15", "city22", "city24", "city26", "city28",
        "city30", "city32", "city34", "city36", "city37",
        "city39", "city40",
    },

    "城镇居民消费支出": {
        "city2", "city5", "city12", "city13", "city15",
        "city18", "city26", "city30", "city33", "city34",
        "city37", "city40",
    },

    "农村居民消费支出": {
        "city2", "city3", "city5", "city8", "city12",
        "city13", "city15", "city18", "city24", "city26",
        "city28", "city30", "city32", "city33", "city34",
        "city36", "city37", "city39", "city40",
    } - {"city8"},  # city8 实际存在，只是覆盖窗口窄

    "城镇居民人均收入": {
        "city5", "city15", "city26", "city33",
        "city34", "city37", "city40",
    },

    "农村居民人均收入": {
        "city3", "city5", "city15", "city24",
        "city26", "city28", "city32", "city34",
        "city36", "city37", "city39", "city40",
    },
}


# =========================================================
# Loading
# =========================================================

def load_living_sheet(sheet_name):
    """
    生活水平.xlsx 的 6 个 Sheet 均为：

    row 0: 原始指标名
    row 1: 城市名称 + 年份
    row 2+: 数据

    所以 row 1 才是真正 header。
    """

    meta = SHEET_META[sheet_name]

    raw_top = pd.read_excel(
        LIVING_FILE,
        sheet_name=sheet_name,
        header=None,
        nrows=2,
    )

    raw_indicator = raw_top.iloc[0, 0]

    # Fail closed:
    # 如果源文件结构变化，不静默继续。
    assert raw_indicator == meta["raw_indicator"], (
        f"{sheet_name}: unexpected raw indicator: "
        f"{raw_indicator!r}"
    )

    df = pd.read_excel(
        LIVING_FILE,
        sheet_name=sheet_name,
        header=1,
    )

    return df


# =========================================================
# Generic audit
# =========================================================

def get_year_columns(df):
    return [
        column
        for column in df.columns
        if column != "城市名称"
    ]


def audit_wide_sheet(
    sheet_name,
    df,
):
    """
    长期保留的核心 source audit。

    raw row order 默认没有时间语义。
    所有时间运算必须先 canonicalize / sort。
    """

    meta = SHEET_META[sheet_name]
    value_name = meta["value_name"]

    year_columns = get_year_columns(df)

    city_series = (
        df["城市名称"]
        .astype("string")
    )

    observed_cities = set(
        city_series.dropna()
    )

    valid_city = city_series.str.fullmatch(
        CITY_PATTERN
    )

    print(
        f"\n{'=' * 70}\n"
        f"{sheet_name}\n"
        f"{'=' * 70}"
    )

    print("\n=== STRUCTURE ===")
    print(f"shape = {df.shape}")

    print(
        f"unique_cities = "
        f"{city_series.nunique()}"
    )

    print(
        f"duplicate_city_rows = "
        f"{city_series.duplicated(keep=False).sum()}"
    )

    print(
        f"invalid_city_rows = "
        f"{(~valid_city.fillna(False)).sum()}"
    )

    print(
        "missing_cities =",
        sorted(
            EXPECTED_CITIES
            - observed_cities
        ),
    )

    print(
        "years =",
        year_columns,
    )

    print("\n=== MISSING VALUES ===")
    print(
        df.isna()
        .sum()
        .to_string()
    )

    # -------------------------
    # Value domain
    # -------------------------

    long_df = df.melt(
        id_vars="城市名称",
        value_vars=year_columns,
        var_name="年份",
        value_name=value_name,
    )

    values = long_df[
        value_name
    ].dropna()

    print("\n=== VALUE DOMAIN ===")
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

    # -------------------------
    # Small temporal drift check
    # -------------------------

    long_df["年份"] = (
        long_df["年份"]
        .astype(int)
    )

    long_df = long_df.sort_values(
        ["城市名称", "年份"]
    ).copy()

    long_df["prev_year"] = (
        long_df
        .groupby("城市名称")["年份"]
        .shift(1)
    )

    long_df["prev_value"] = (
        long_df
        .groupby("城市名称")[value_name]
        .shift(1)
    )

    adjacent = long_df[
        (
            long_df["年份"]
            - long_df["prev_year"]
            == 1
        )
        & long_df[value_name].notna()
        & long_df["prev_value"].notna()
    ].copy()

    adjacent["change_rate"] = (
        adjacent[value_name]
        / adjacent["prev_value"]
        - 1
    )

    print("\n=== TEMPORAL EXTREMES ===")

    if adjacent.empty:
        print("No adjacent valid year pairs.")
        return

    print("\ntop increases:")

    print(
        adjacent[
            [
                "城市名称",
                "年份",
                "prev_value",
                value_name,
                "change_rate",
            ]
        ]
        .sort_values(
            "change_rate",
            ascending=False,
        )
        .head(3)
        .to_string(
            index=False,
            formatters={
                "change_rate":
                    lambda x: f"{x:.2%}"
            },
        )
    )

    print("\ntop decreases:")

    print(
        adjacent[
            [
                "城市名称",
                "年份",
                "prev_value",
                value_name,
                "change_rate",
            ]
        ]
        .sort_values("change_rate")
        .head(3)
        .to_string(
            index=False,
            formatters={
                "change_rate":
                    lambda x: f"{x:.2%}"
            },
        )
    )


# =========================================================
# Generic normalization
# =========================================================

def normalize_wide_sheet(
    df,
    value_name,
):
    year_columns = get_year_columns(df)

    normalized = df.melt(
        id_vars="城市名称",
        value_vars=year_columns,
        var_name="年份",
        value_name=value_name,
    )

    normalized["年份"] = (
        normalized["年份"]
        .astype(int)
    )

    normalized = (
        normalized[
            [
                "城市名称",
                "年份",
                value_name,
            ]
        ]
        .sort_values(
            ["城市名称", "年份"]
        )
        .reset_index(drop=True)
    )

    return normalized


# =========================================================
# Repair 1:
# 人均消费支出 city21-2022
# =========================================================

def apply_consumption_repairs(df):
    """
    city21-2022:

        446007 -> 44607

    Evidence:
    - raw Excel cell is hard-coded numeric 446007;
    - isolated +1066.49% YoY;
    - consumption / disposable-income ratio
      becomes 6.587 instead of historical ~0.62-0.73;
    - urban/rural consumption + urbanization relationship
      gives local residual:
        2020 ~ -1039
        2021 ~ -1080
        candidate 2022 ~ -1076

    因此属于 evidence-backed source typo repair。
    """

    repaired = df.copy()

    mask = (
        repaired["城市名称"]
        == "city21"
    )

    assert mask.sum() == 1

    raw_value = repaired.loc[
        mask,
        2022,
    ].iloc[0]

    assert raw_value == 446007, (
        "Unexpected city21-2022 "
        f"consumption value: {raw_value}"
    )

    repaired.loc[
        mask,
        2022,
    ] = 44607

    return repaired


# =========================================================
# Repair 2:
# 城镇居民人均收入 city labels
# =========================================================

def apply_urban_income_label_repairs(df):
    """
    Confirmed source label typos:

        ctiy22 -> city22
        ctiy31 -> city31
    """

    repaired = df.copy()

    assert (
        repaired["城市名称"]
        == "ctiy22"
    ).sum() == 1

    assert (
        repaired["城市名称"]
        == "ctiy31"
    ).sum() == 1

    assert (
        repaired["城市名称"]
        == "city22"
    ).sum() == 0

    assert (
        repaired["城市名称"]
        == "city31"
    ).sum() == 0

    repaired["城市名称"] = (
        repaired["城市名称"]
        .replace(
            {
                "ctiy22": "city22",
                "ctiy31": "city31",
            }
        )
    )

    return repaired


# =========================================================
# Invalidation:
# 人均可支配收入 block contamination
# =========================================================

DISPOSABLE_CONTAMINATED_CITIES = {
    "city21",
    "city25",
    "city29",
    "city33",
}

DISPOSABLE_CONTAMINATED_YEARS = set(
    range(2015, 2023)
)


def build_disposable_income():
    """
    人均可支配收入末尾四个连续 city block：

        city21
        city25
        city29
        city33

    2015-2022 共 32 个值与农村居民人均收入
    同 city-year 的绝对差全部 <= 3。

    农村收入从 2014 起自身连续，
    因此更可能是 overall Sheet 的末尾 block
    被 rural-income 数据污染。

    正确 overall value 未知。

    所以：
        保留 canonical key
        contaminated value -> NA
        raw value -> quarantine

    不用 weighted formula 重建。
    """

    overall_raw = load_living_sheet(
        "人均可支配收入"
    )

    rural_raw = load_living_sheet(
        "农村居民人均收入"
    )

    # -------------------------
    # Source block evidence
    # -------------------------

    expected_tail_block = [
        "city21",
        "city25",
        "city29",
        "city33",
    ]

    actual_tail_block = (
        overall_raw["城市名称"]
        .tail(4)
        .tolist()
    )

    assert (
        actual_tail_block
        == expected_tail_block
    ), (
        "Unexpected disposable-income "
        f"tail block: {actual_tail_block}"
    )

    overall_long = normalize_wide_sheet(
        overall_raw,
        "disposableIncome",
    )

    rural_long = normalize_wide_sheet(
        rural_raw,
        "ruralDisposableIncome",
    )

    evidence = overall_long.merge(
        rural_long,
        on=["城市名称", "年份"],
        how="inner",
    )

    evidence = evidence[
        evidence["城市名称"].isin(
            DISPOSABLE_CONTAMINATED_CITIES
        )
        & evidence["年份"].isin(
            DISPOSABLE_CONTAMINATED_YEARS
        )
    ].copy()

    evidence["abs_gap"] = (
        evidence["disposableIncome"]
        - evidence["ruralDisposableIncome"]
    ).abs()

    assert len(evidence) == 32
    assert (
        evidence["abs_gap"] <= 3
    ).all()

    # -------------------------
    # Quarantine
    # -------------------------

    quarantine = evidence[
        [
            "城市名称",
            "年份",
            "disposableIncome",
            "ruralDisposableIncome",
            "abs_gap",
        ]
    ].copy()

    quarantine = quarantine.rename(
        columns={
            "disposableIncome":
                "raw_disposableIncome",
        }
    )

    quarantine["reason"] = (
        "high-confidence block-level "
        "cross-sheet contamination; "
        "correct overall value unknown"
    )

    # -------------------------
    # Invalidate contaminated values
    # -------------------------

    normalized = overall_long.copy()

    normalized["disposableIncome"] = (
        normalized["disposableIncome"]
        .astype("Float64")
    )

    mask = (
        normalized["城市名称"].isin(
            DISPOSABLE_CONTAMINATED_CITIES
        )
        & normalized["年份"].isin(
            DISPOSABLE_CONTAMINATED_YEARS
        )
    )

    assert mask.sum() == 32

    normalized.loc[
        mask,
        "disposableIncome",
    ] = pd.NA

    return normalized, quarantine


# =========================================================
# Generic verification
# =========================================================

def verify_normalized(
    sheet_name,
    df,
    expected_rows,
    expected_city_count,
    expected_years,
    expected_missing,
):
    meta = SHEET_META[sheet_name]
    value_name = meta["value_name"]

    duplicate_rows = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    ).sum()

    observed_cities = set(
        df["城市名称"]
    )

    missing_cities = (
        EXPECTED_CITIES
        - observed_cities
    )

    print(
        f"\n=== VERIFY: {sheet_name} ==="
    )

    print(f"rows = {len(df)}")

    print(
        "unique_keys =",
        df[
            ["城市名称", "年份"]
        ].drop_duplicates().shape[0],
    )

    print(
        f"duplicate_rows = "
        f"{duplicate_rows}"
    )

    print(
        f"cities = "
        f"{df['城市名称'].nunique()}"
    )

    print(
        "years =",
        sorted(df["年份"].unique()),
    )

    print(
        f"missing_values = "
        f"{df[value_name].isna().sum()}"
    )

    print(
        "missing_cities =",
        sorted(missing_cities),
    )

    assert len(df) == expected_rows
    assert duplicate_rows == 0

    assert (
        df["城市名称"].nunique()
        == expected_city_count
    )

    assert (
        set(df["年份"])
        == set(expected_years)
    )

    assert (
        df[value_name]
        .isna()
        .sum()
        == expected_missing
    )

    assert (
        missing_cities
        == MISSING_CITIES[sheet_name]
    )

    valid_city = (
        df["城市名称"]
        .astype("string")
        .str.fullmatch(CITY_PATTERN)
    )

    assert valid_city.all()


def assert_missing_pattern(
    df,
    value_name,
    expected,
):
    missing_by_city = (
        df[
            df[value_name].isna()
        ]
        .groupby("城市名称")["年份"]
        .apply(list)
        .to_dict()
    )

    assert missing_by_city == expected, (
        f"Unexpected missing pattern for "
        f"{value_name}:\n"
        f"{missing_by_city}"
    )


# =========================================================
# Sheet-specific verification
# =========================================================

def verify_disposable_income(
    df,
    quarantine,
):
    verify_normalized(
        "人均可支配收入",
        df,
        expected_rows=176,
        expected_city_count=22,
        expected_years=range(2015, 2023),
        expected_missing=32,
    )

    assert len(quarantine) == 32
    assert quarantine["abs_gap"].max() <= 3

    expected_missing = {
        city: list(range(2015, 2023))
        for city in
        DISPOSABLE_CONTAMINATED_CITIES
    }

    assert_missing_pattern(
        df,
        "disposableIncome",
        expected_missing,
    )


def verify_consumption(df):
    verify_normalized(
        "人均消费支出",
        df,
        expected_rows=184,
        expected_city_count=23,
        expected_years=range(2015, 2023),
        expected_missing=0,
    )

    value = df.loc[
        (df["城市名称"] == "city21")
        & (df["年份"] == 2022),
        "consumptionExpenditures",
    ].iloc[0]

    assert value == 44607

    assert (
        df["consumptionExpenditures"]
        .max()
        < 100000
    )


def verify_urban_consumption(df):
    verify_normalized(
        "城镇居民消费支出",
        df,
        expected_rows=336,
        expected_city_count=28,
        expected_years=range(2011, 2023),
        expected_missing=39,
    )

    expected = {
        "city1": [2011, 2012, 2013],
        "city4": [2011, 2012, 2013],
        "city7": [2011, 2012, 2013],
        "city10": [2011, 2012, 2013],
        "city11": [2011, 2012, 2013],
        "city19": [2011, 2012, 2013],
        "city20": [2011, 2012, 2013],
        "city21": [2011, 2012, 2013],
        "city25": [2011, 2012, 2013],
        "city29": [2011, 2012, 2013],

        "city8": list(
            range(2014, 2023)
        ),
    }

    assert_missing_pattern(
        df,
        "urbanConsumptionExpenditures",
        expected,
    )


def verify_rural_consumption(df):
    verify_normalized(
        "农村居民消费支出",
        df,
        expected_rows=264,
        expected_city_count=22,
        expected_years=range(2011, 2023),
        expected_missing=40,
    )

    expected = {
        "city1": [2011, 2012],
        "city4": [2011, 2012],
        "city7": [2011, 2012],
        "city10": [2011, 2012],
        "city11": [2011, 2012],

        "city19": [2011, 2012, 2013],
        "city20": [2011, 2012, 2013],
        "city21": [2011, 2012, 2013],
        "city22": [2011, 2012, 2013],
        "city25": [2011, 2012, 2013],
        "city29": [2011, 2012, 2013],

        "city6": [2022],
        "city9": [2022],
        "city14": [2022],

        "city8": [
            2011, 2012, 2013,
            2017, 2018, 2019,
            2020, 2021, 2022,
        ],
    }

    assert_missing_pattern(
        df,
        "ruralConsumptionExpenditures",
        expected,
    )


def verify_urban_income(df):
    verify_normalized(
        "城镇居民人均收入",
        df,
        expected_rows=396,
        expected_city_count=33,
        expected_years=range(2011, 2023),
        expected_missing=35,
    )

    expected = {
        "city1": [2011, 2012],
        "city4": [2011, 2012],
        "city7": [2011, 2012],
        "city10": [2011, 2012],
        "city11": [2011, 2012],
        "city30": [2011, 2012],

        "city2": [2011],
        "city13": [2011],

        "city19": [2011, 2012, 2013],
        "city20": [2011, 2012, 2013],
        "city21": [2011, 2012, 2013],
        "city25": [2011, 2012, 2013],
        "city29": [2011, 2012, 2013],

        "city3": [2022],
        "city6": [2022],
        "city9": [2022],
        "city14": [2022],

        # real internal gaps
        "city8": [2013],
        "city18": [2014],
    }

    assert_missing_pattern(
        df,
        "urbanDisposableIncome",
        expected,
    )

    # Cross-source-supported temporal anomaly:
    # preserve raw values.
    assert df.loc[
        (df["城市名称"] == "city8")
        & (df["年份"] == 2017),
        "urbanDisposableIncome",
    ].iloc[0] == 24880

    assert df.loc[
        (df["城市名称"] == "city8")
        & (df["年份"] == 2018),
        "urbanDisposableIncome",
    ].iloc[0] == 35825


def verify_rural_income(df):
    verify_normalized(
        "农村居民人均收入",
        df,
        expected_rows=336,
        expected_city_count=28,
        expected_years=range(2011, 2023),
        expected_missing=48,
    )

    expected = {
        "city30": [
            2011, 2012, 2013, 2014
        ],
        "city8": [
            2011, 2012, 2013, 2014
        ],

        "city13": [2011, 2012, 2013],
        "city19": [2011, 2012, 2013],
        "city20": [2011, 2012, 2013],
        "city21": [2011, 2012, 2013],
        "city22": [2011, 2012, 2013],
        "city25": [2011, 2012, 2013],
        "city29": [2011, 2012, 2013],
        "city33": [2011, 2012, 2013],

        "city1": [2011, 2012],
        "city4": [2011, 2012],
        "city7": [2011, 2012],
        "city10": [2011, 2012],
        "city11": [2011, 2012],

        "city2": [2011],

        "city6": [2022],
        "city9": [2022],
        "city14": [2022],

        # real internal gaps
        "city18": [2013, 2014],
    }

    assert_missing_pattern(
        df,
        "ruralDisposableIncome",
        expected,
    )

    # city8 V-shaped change is supported by
    # the corresponding urban-income series.
    assert df.loc[
        (df["城市名称"] == "city8")
        & (df["年份"] == 2016),
        "ruralDisposableIncome",
    ].iloc[0] == 14316

    assert df.loc[
        (df["城市名称"] == "city8")
        & (df["年份"] == 2017),
        "ruralDisposableIncome",
    ].iloc[0] == 11928

    assert df.loc[
        (df["城市名称"] == "city8")
        & (df["年份"] == 2018),
        "ruralDisposableIncome",
    ].iloc[0] == 17056


# =========================================================
# Save
# =========================================================

def save_normalized(
    sheet_name,
    df,
):
    output_path = (
        NORMALIZED_DIR
        / SHEET_META[sheet_name]["output"]
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"saved: {output_path} "
        f"{df.shape}"
    )


def save_disposable_quarantine(
    quarantine,
):
    output_path = (
        QUARANTINE_DIR
        / "生活水平_人均可支配收入_quarantine.csv"
    )

    quarantine.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"saved quarantine: "
        f"{output_path} "
        f"{quarantine.shape}"
    )


# =========================================================
# Repair report
# =========================================================

def save_repair_report():
    report = pd.DataFrame(
        [
            {
                "sheet": "人均消费支出",
                "scope": "city21-2022",
                "action": "repair",
                "raw": "446007",
                "normalized": "44607",
                "reason":
                    "isolated magnitude error + "
                    "cross-source mechanism evidence",
            },
            {
                "sheet": "城镇居民人均收入",
                "scope": "city label",
                "action": "repair",
                "raw": "ctiy22",
                "normalized": "city22",
                "reason":
                    "invalid label with unique "
                    "missing canonical counterpart",
            },
            {
                "sheet": "城镇居民人均收入",
                "scope": "city label",
                "action": "repair",
                "raw": "ctiy31",
                "normalized": "city31",
                "reason":
                    "invalid label with unique "
                    "missing canonical counterpart",
            },
            {
                "sheet": "人均可支配收入",
                "scope":
                    "city21/city25/city29/city33 "
                    "2015-2022",
                "action": "invalidate_to_NA",
                "raw": "32 source values",
                "normalized": "NA",
                "reason":
                    "high-confidence block-level "
                    "cross-sheet contamination; "
                    "correct values unknown",
            },
        ]
    )

    output_path = (
        NORMALIZED_DIR
        / "生活水平_repair_report.csv"
    )

    report.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"saved repair report: "
        f"{output_path}"
    )


# =========================================================
# Main
# =========================================================

def main():
    NORMALIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUARANTINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 1. 人均可支配收入
    # -----------------------------------------------------

    disposable_raw = load_living_sheet(
        "人均可支配收入"
    )

    audit_wide_sheet(
        "人均可支配收入",
        disposable_raw,
    )

    disposable_df, quarantine_df = (
        build_disposable_income()
    )

    verify_disposable_income(
        disposable_df,
        quarantine_df,
    )

    save_normalized(
        "人均可支配收入",
        disposable_df,
    )

    save_disposable_quarantine(
        quarantine_df
    )

    # -----------------------------------------------------
    # 2. 人均消费支出
    # -----------------------------------------------------

    consumption_raw = load_living_sheet(
        "人均消费支出"
    )

    audit_wide_sheet(
        "人均消费支出",
        consumption_raw,
    )

    consumption_repaired = (
        apply_consumption_repairs(
            consumption_raw
        )
    )

    consumption_df = normalize_wide_sheet(
        consumption_repaired,
        "consumptionExpenditures",
    )

    verify_consumption(
        consumption_df
    )

    save_normalized(
        "人均消费支出",
        consumption_df,
    )

    # -----------------------------------------------------
    # 3. 城镇居民消费支出
    # -----------------------------------------------------

    urban_consumption_raw = load_living_sheet(
        "城镇居民消费支出"
    )

    audit_wide_sheet(
        "城镇居民消费支出",
        urban_consumption_raw,
    )

    urban_consumption_df = (
        normalize_wide_sheet(
            urban_consumption_raw,
            "urbanConsumptionExpenditures",
        )
    )

    verify_urban_consumption(
        urban_consumption_df
    )

    save_normalized(
        "城镇居民消费支出",
        urban_consumption_df,
    )

    # -----------------------------------------------------
    # 4. 农村居民消费支出
    # -----------------------------------------------------

    rural_consumption_raw = load_living_sheet(
        "农村居民消费支出"
    )

    audit_wide_sheet(
        "农村居民消费支出",
        rural_consumption_raw,
    )

    rural_consumption_df = (
        normalize_wide_sheet(
            rural_consumption_raw,
            "ruralConsumptionExpenditures",
        )
    )

    verify_rural_consumption(
        rural_consumption_df
    )

    save_normalized(
        "农村居民消费支出",
        rural_consumption_df,
    )

    # -----------------------------------------------------
    # 5. 城镇居民人均收入
    # -----------------------------------------------------

    urban_income_raw = load_living_sheet(
        "城镇居民人均收入"
    )

    audit_wide_sheet(
        "城镇居民人均收入",
        urban_income_raw,
    )

    urban_income_repaired = (
        apply_urban_income_label_repairs(
            urban_income_raw
        )
    )

    urban_income_df = normalize_wide_sheet(
        urban_income_repaired,
        "urbanDisposableIncome",
    )

    verify_urban_income(
        urban_income_df
    )

    save_normalized(
        "城镇居民人均收入",
        urban_income_df,
    )

    # -----------------------------------------------------
    # 6. 农村居民人均收入
    # -----------------------------------------------------

    rural_income_raw = load_living_sheet(
        "农村居民人均收入"
    )

    audit_wide_sheet(
        "农村居民人均收入",
        rural_income_raw,
    )

    rural_income_df = normalize_wide_sheet(
        rural_income_raw,
        "ruralDisposableIncome",
    )

    verify_rural_income(
        rural_income_df
    )

    save_normalized(
        "农村居民人均收入",
        rural_income_df,
    )

    # -----------------------------------------------------
    # Repair / provenance report
    # -----------------------------------------------------

    save_repair_report()

    print(
        "\n=== LIVING STANDARD COMPLETE ==="
    )
    print(
        "6 normalized tables verified and saved."
    )


if __name__ == "__main__":
    main()
