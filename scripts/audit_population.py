from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"

POPULATION_COLUMNS = [
    "常住人口（万人）",
    "户籍人口（万人）",
]


def load_population():
    """读取人口规模原始表。"""
    return pd.read_excel(DATA_DIR / "人口规模.xlsx")


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
    """检查城市 key 是否符合 city1~city40 的 canonical 格式。"""
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
    ].copy()

    print("\n=== COVERAGE ===")
    print(f"cities = {df['城市名称'].nunique()}")
    print(f"years = {df['年份'].min()}-{df['年份'].max()}")
    print(f"cities_with_internal_gaps = {len(gaps)}")

    if not gaps.empty:
        print(gaps.to_string(index=False))


def check_missing(df):
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())


def check_value_sanity(df):
    print("\n=== VALUE SANITY ===")

    for column in POPULATION_COLUMNS:
        values = df[column].dropna()

        print(
            f"{column}: "
            f"min={values.min()}, "
            f"max={values.max()}, "
            f"zero={int((values == 0).sum())}, "
            f"negative={int((values < 0).sum())}"
        )


def normalize_population(df):
    """
    已确认 repair：
    两个人口指标中的 0 是 missing sentinel。
    """
    normalized_df = df.copy()

    normalized_df[POPULATION_COLUMNS] = (
        normalized_df[POPULATION_COLUMNS]
        .replace(0, pd.NA)
    )

    return normalized_df


def verify_normalized(df):
    print("\n=== NORMALIZED CHECK ===")
    print("null_counts:")
    print(df[POPULATION_COLUMNS].isna().sum())

    print("zero_counts:")
    print((df[POPULATION_COLUMNS] == 0).sum())


def save_normalized(df):
    output_path = NORMALIZED_DIR / "人口规模.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nsaved: {output_path}")
    print(f"shape: {df.shape}")


def main():
    raw_df = load_population()

    check_schema(raw_df)
    check_key(raw_df)
    check_key_format(raw_df)
    check_coverage(raw_df)
    check_missing(raw_df)
    check_value_sanity(raw_df)

    normalized_df = normalize_population(raw_df)

    verify_normalized(normalized_df)
    save_normalized(normalized_df)


if __name__ == "__main__":
    main()
