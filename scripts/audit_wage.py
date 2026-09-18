from pathlib import Path

import pandas as pd


# =========================================================
# Paths / Constants
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"

WAGE_FILE = DATA_DIR / "工资水平.xlsx"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"
EXPECTED_CITIES = {f"city{i}" for i in range(1, 41)}
QUARANTINE_DIR = ROOT / "data" / "quarantine"

# =========================================================
# Load
# =========================================================

def load_wage():
    return pd.read_excel(
        WAGE_FILE,
        sheet_name="职工平均工资",
    )


# =========================================================
# Raw Audit
# =========================================================

def check_schema(df):
    print("=== SCHEMA ===")
    print(f"shape = {df.shape}")
    print(f"columns = {df.columns.tolist()}")
    print("\n=== DTYPES ===")
    print(df.dtypes)


def check_axes(df):
    """检查宽表年份轴与城市轴。"""

    year_labels = df["averageWage"]

    year_valid = year_labels.str.fullmatch(
        r"\d{4}年"
    )

    city_columns = [
        column
        for column in df.columns
        if column != "averageWage"
    ]

    observed_cities = set(city_columns)

    invalid_city_columns = [
        city
        for city in city_columns
        if not pd.Series([city])
        .str.fullmatch(CITY_PATTERN)
        .iloc[0]
    ]

    print("\n=== AXES ===")
    print(f"year_rows = {len(year_labels)}")
    print(
        f"unique_year_labels = "
        f"{year_labels.nunique()}"
    )
    print(
        f"invalid_year_rows = "
        f"{(~year_valid).sum()}"
    )

    print(
        f"city_columns = "
        f"{len(city_columns)}"
    )
    print(
        f"unique_city_columns = "
        f"{len(observed_cities)}"
    )
    print(
        "missing_cities =",
        sorted(
            EXPECTED_CITIES
            - observed_cities
        ),
    )
    print(
        "unexpected_cities =",
        sorted(
            observed_cities
            - EXPECTED_CITIES
        ),
    )
    print(
        "invalid_city_columns =",
        invalid_city_columns,
    )


def check_missing_pattern(df):
    """
    检查宽表 NaN，并区分：
    leading / internal / trailing missing。
    """

    wage = df.drop(
        columns="averageWage"
    )

    years = (
        df["averageWage"]
        .str.removesuffix("年")
        .astype(int)
    )

    print("\n=== MISSING SUMMARY ===")

    print(
        f"total_cells = {wage.size}"
    )
    print(
        f"missing_cells = "
        f"{wage.isna().sum().sum()}"
    )
    print(
        f"observed_cells = "
        f"{wage.notna().sum().sum()}"
    )

    print(
        "\n=== MISSING YEARS BY CITY ==="
    )

    for city in wage.columns:
        missing_mask = wage[city].isna()

        if not missing_mask.any():
            continue

        missing_years = years[
            missing_mask
        ].tolist()

        observed_years = years[
            ~missing_mask
        ]

        first_observed = (
            observed_years.min()
        )
        last_observed = (
            observed_years.max()
        )

        leading = [
            year
            for year in missing_years
            if year < first_observed
        ]

        internal = [
            year
            for year in missing_years
            if first_observed
            < year
            < last_observed
        ]

        trailing = [
            year
            for year in missing_years
            if year > last_observed
        ]

        print(
            f"{city}: "
            f"leading={leading}, "
            f"internal={internal}, "
            f"trailing={trailing}"
        )


def check_value_domain(df):
    wage = (
        df.drop(
            columns="averageWage"
        )
        .melt(
            var_name="城市名称",
            value_name="averageWage",
        )
    )

    values = wage[
        "averageWage"
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


def check_temporal_anomalies(df):
    """
    保留最核心的时间异常检查。

    当前已知最突出的异常：
    city31 2011→2012 大幅下降。
    这里只报告，不自动 repair。
    """

    long_df = (
        df.rename(
            columns={
                "averageWage": "年份"
            }
        )
        .melt(
            id_vars="年份",
            var_name="城市名称",
            value_name="averageWage",
        )
    )

    long_df["年份"] = (
        long_df["年份"]
        .str.removesuffix("年")
        .astype(int)
    )

    long_df = long_df.sort_values(
        ["城市名称", "年份"]
    ).copy()

    long_df["prev_year"] = (
        long_df.groupby(
            "城市名称"
        )["年份"].shift(1)
    )

    long_df["prev_wage"] = (
        long_df.groupby(
            "城市名称"
        )["averageWage"].shift(1)
    )

    adjacent = long_df[
        (long_df["年份"] - long_df["prev_year"] == 1)
        & long_df["averageWage"].notna()
        & long_df["prev_wage"].notna()
    ].copy()

    adjacent["change_rate"] = (
        adjacent["averageWage"]
        / adjacent["prev_wage"]
        - 1
    )

    negative = adjacent[
        adjacent["change_rate"] < 0
    ]

    print(
        "\n=== NEGATIVE YEAR-TO-YEAR CHANGES ==="
    )

    if negative.empty:
        print("none")
        return

    print(
        negative[
            [
                "城市名称",
                "年份",
                "prev_wage",
                "averageWage",
                "change_rate",
            ]
        ]
        .sort_values("change_rate")
        .to_string(
            index=False,
            formatters={
                "change_rate":
                    lambda x: f"{x:.2%}"
            },
        )
    )


# =========================================================
# Normalization
# =========================================================

def normalize_wage(df):
    """
    Wide → canonical long table.

    输出粒度：
        (城市名称, 年份) -> averageWage

    当前没有 evidence-backed value repair：
    - 49 个 NaN 保留；
    - city31-2011/2012 异常值保留。
    """

    normalized = (
        df.rename(
            columns={
                "averageWage": "年份"
            }
        )
        .melt(
            id_vars="年份",
            var_name="城市名称",
            value_name="averageWage",
        )
    )

    normalized["年份"] = (
        normalized["年份"]
        .str.removesuffix("年")
        .astype(int)
    )

    return normalized[
        [
            "城市名称",
            "年份",
            "averageWage",
        ]
    ]


# =========================================================
# Verification
# =========================================================

def verify_wage(df):
    duplicate_rows = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    ).sum()

    observed_cities = set(
        df["城市名称"]
    )

    print(
        "\n=== NORMALIZED VERIFICATION ==="
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
        f"year_min = "
        f"{df['年份'].min()}"
    )
    print(
        f"year_max = "
        f"{df['年份'].max()}"
    )
    print(
        "missing_averageWage =",
        df[
            "averageWage"
        ].isna().sum(),
    )

    assert len(df) == 840
    assert duplicate_rows == 0
    assert observed_cities == EXPECTED_CITIES
    assert df["年份"].nunique() == 21
    assert df["年份"].min() == 2001
    assert df["年份"].max() == 2021
    assert (
        df["averageWage"]
        .isna()
        .sum()
        == 49
    )

    # 已调查的 source anomaly 必须原样保留。
    def get_value(city, year):
        return df.loc[
            (df["城市名称"] == city)
            & (df["年份"] == year),
            "averageWage",
        ].iloc[0]

    assert pd.isna(
        get_value("city31", 2010)
    )
    assert (
        get_value(
            "city31",
            2011,
        )
        == 37492
    )
    assert (
        get_value(
            "city31",
            2012,
        )
        == 31908
    )


# =========================================================
def build_wage_v2(normalized_df):
    """
    第二轮 cross-table audit 后的工资处理。

    city31-2011 = 37492：
        高置信度异常，但无法恢复真实值。

    因此：
        canonical value -> NA
        raw value -> quarantine

    注意：
        不猜测 27492 等候选值。
    """

    df = normalized_df.copy()

    target_mask = (
        (df["城市名称"] == "city31")
        & (df["年份"] == 2011)
    )

    assert target_mask.sum() == 1

    raw_value = df.loc[
        target_mask,
        "averageWage",
    ].iloc[0]

    # fail-closed:
    # 如果原始 source 已经变化，则停止执行。
    assert raw_value == 37492

    # 2012 已经在第二轮审计中被重新判定为可信。
    value_2012 = df.loc[
        (df["城市名称"] == "city31")
        & (df["年份"] == 2012),
        "averageWage",
    ].iloc[0]

    assert value_2012 == 31908

    quarantine = df.loc[
        target_mask,
        [
            "城市名称",
            "年份",
            "averageWage",
        ],
    ].copy()

    quarantine = quarantine.rename(
        columns={
            "averageWage":
                "raw_averageWage"
        }
    )

    quarantine["reason"] = (
        "high-confidence isolated wage anomaly; "
        "true value unresolved"
    )

    quarantine["action"] = (
        "invalidate canonical value to NA"
    )

    df["averageWage"] = (
        df["averageWage"]
        .astype("Float64")
    )

    df.loc[
        target_mask,
        "averageWage",
    ] = pd.NA

    return df, quarantine
def verify_wage_v2(
    wage_v2,
    quarantine,
):
    print(
        "\n=== WAGE V2 "
        "INVALIDATION VERIFICATION ==="
    )

    duplicate_rows = (
        wage_v2
        .duplicated(
            ["城市名称", "年份"],
            keep=False,
        )
        .sum()
    )

    print(
        f"rows = {len(wage_v2)}"
    )

    print(
        f"unique_keys = "
        f"{wage_v2[['城市名称', '年份']].drop_duplicates().shape[0]}"
    )

    print(
        f"duplicate_rows = "
        f"{duplicate_rows}"
    )

    print(
        f"missing_averageWage = "
        f"{wage_v2['averageWage'].isna().sum()}"
    )

    print(
        f"quarantine_rows = "
        f"{len(quarantine)}"
    )

    city31 = (
        wage_v2[
            wage_v2[
                "城市名称"
            ] == "city31"
        ]
        .sort_values("年份")
    )

    print(
        "\n=== CITY31 CONTEXT ==="
    )

    print(
        city31[
            city31["年份"].between(
                2008,
                2014,
            )
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== QUARANTINE ==="
    )

    print(
        quarantine.to_string(
            index=False
        )
    )

    assert len(wage_v2) == 840

    assert duplicate_rows == 0

    # 原来 49 个 NaN，现在增加一个 invalidated value。
    assert (
        wage_v2[
            "averageWage"
        ].isna().sum()
        == 50
    )

    assert len(quarantine) == 1

    value_2011 = wage_v2.loc[
        (wage_v2["城市名称"] == "city31")
        & (wage_v2["年份"] == 2011),
        "averageWage",
    ].iloc[0]

    value_2012 = wage_v2.loc[
        (wage_v2["城市名称"] == "city31")
        & (wage_v2["年份"] == 2012),
        "averageWage",
    ].iloc[0]

    assert pd.isna(
        value_2011
    )

    assert value_2012 == 31908

def save_wage_v2(
    wage_v2,
    quarantine,
):
    normalized_path = (
        NORMALIZED_DIR
        / "工资水平_职工平均工资.csv"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "工资水平_职工平均工资_quarantine.csv"
    )

    wage_v2.to_csv(
        normalized_path,
        index=False,
        encoding="utf-8-sig",
    )

    quarantine.to_csv(
        quarantine_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SAVE RESULT ===")

    print(
        f"normalized: "
        f"{normalized_path}"
    )
    print(
        f"normalized shape: "
        f"{wage_v2.shape}"
    )

    print(
        f"quarantine: "
        f"{quarantine_path}"
    )
    print(
        f"quarantine shape: "
        f"{quarantine.shape}"
    )
# =========================================================
# Entry
# =========================================================

def main():
    raw_df = load_wage()

    # =============================================
    # 1. Raw audit
    # =============================================

    print("===== RAW AUDIT =====")

    check_schema(raw_df)
    check_axes(raw_df)
    check_missing_pattern(raw_df)
    check_value_domain(raw_df)
    check_temporal_anomalies(raw_df)

    # =============================================
    # 2. Wide -> canonical long
    # =============================================

    normalized_df = normalize_wage(
        raw_df
    )

    # 先验证原始 normalization。
    # 这里仍然应该是：
    # 49 missing，city31-2011 = 37492。
    verify_wage(
        normalized_df
    )

    # =============================================
    # 3. Second-round evidence-backed invalidation
    # =============================================

    wage_v2, quarantine = (
        build_wage_v2(
            normalized_df
        )
    )

    verify_wage_v2(
        wage_v2,
        quarantine,
    )

    # =============================================
    # 4. Save final v2
    # =============================================

    save_wage_v2(
        wage_v2,
        quarantine,
    )


if __name__ == "__main__":
    main()
