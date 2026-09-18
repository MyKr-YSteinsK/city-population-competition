from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"

RATE_COLUMN = "urbanizationRate"


def load_urbanization():
    """读取城镇化率原始表。"""
    return pd.read_excel(DATA_DIR / "城镇化率.xlsx")


def check_schema(df):
    print("=== SCHEMA ===")
    print(f"shape = {df.shape}")
    print(f"columns = {df.columns.tolist()}")
    print(df.dtypes)


def check_key(df):
    duplicate_mask = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    )

    print("\n=== KEY CHECK ===")
    print(f"unique_city_year_keys = {df[['城市名称', '年份']].drop_duplicates().shape[0]}")
    print(f"duplicate_rows = {duplicate_mask.sum()}")

    if duplicate_mask.any():
        print(
            df.loc[duplicate_mask]
            .sort_values(["城市名称", "年份"])
            .to_string(index=False)
        )


def check_key_format(df):
    valid_city = df["城市名称"].str.fullmatch(CITY_PATTERN)

    print("\n=== KEY FORMAT ===")
    print(f"invalid_city_rows = {(~valid_city).sum()}")

    if (~valid_city).any():
        print(
            df.loc[~valid_city, "城市名称"]
            .drop_duplicates()
            .map(repr)
            .to_string(index=False)
        )


def check_coverage(df):
    coverage = (
        df.groupby("城市名称")["年份"]
        .agg(
            year_min="min",
            year_max="max",
            observed_years="nunique",
        )
        .reset_index()
    )

    coverage["expected_years"] = (
        coverage["year_max"] - coverage["year_min"] + 1
    )

    coverage["internal_missing_years"] = (
        coverage["expected_years"] - coverage["observed_years"]
    )

    gaps = coverage[
        coverage["internal_missing_years"] > 0
    ]

    print("\n=== COVERAGE ===")
    print(f"cities = {df['城市名称'].nunique()}")
    print(f"years = {df['年份'].min()}-{df['年份'].max()}")
    print(f"cities_with_internal_gaps = {len(gaps)}")

    if not gaps.empty:
        print(gaps.to_string(index=False))


def check_missing(df):
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())


def check_value_domain(df):
    values = df[RATE_COLUMN]

    print("\n=== VALUE DOMAIN ===")
    print(f"min = {values.min()}")
    print(f"max = {values.max()}")
    print(f"outside_0_100 = {((values < 0) | (values > 100)).sum()}")
    print(f"zero_count = {(values == 0).sum()}")
    print(f"hundred_count = {(values == 100).sum()}")


def normalize_urbanization(df):
    """
    当前无 evidence-backed repair rule。
    normalized table 与 raw values 保持一致。
    """
    return df.copy()


def verify_normalized(df):
    print("\n=== NORMALIZED CHECK ===")
    print(f"rows = {len(df)}")
    print(f"duplicate_rows = {df.duplicated(['城市名称', '年份'], keep=False).sum()}")


def save_normalized(df):
    output_path = NORMALIZED_DIR / "城镇化率.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nsaved: {output_path}")
    print(f"shape: {df.shape}")


def main():
    raw_df = load_urbanization()

    check_schema(raw_df)
    check_key(raw_df)
    check_key_format(raw_df)
    check_coverage(raw_df)
    check_missing(raw_df)
    check_value_domain(raw_df)

    normalized_df = normalize_urbanization(raw_df)

    verify_normalized(normalized_df)
    save_normalized(normalized_df)


if __name__ == "__main__":
    main()
