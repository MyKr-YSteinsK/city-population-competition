from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"
QUARANTINE_DIR = ROOT / "data" / "quarantine"
CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"

DENSITY_COLUMN = "人口密度（人/平方公里）"

POPULATION_NORMALIZED_PATH = (
    NORMALIZED_DIR / "人口规模.csv"
)
DENSITY_BATCH_ANOMALY_VALUES = {
    ("city2", 2013): 1401,
    ("city5", 2013): 1246,
    ("city8", 2013): 479,
    ("city12", 2013): 801,
    ("city13", 2013): 565,
    ("city15", 2013): 760,
    ("city18", 2013): 1070,

    ("city22", 2014): 1243,
    ("city26", 2014): 912,
    ("city30", 2014): 582,
    ("city34", 2014): 425,
    ("city37", 2014): 643,
    ("city40", 2014): 823,
}


DENSITY_REGIME_SHIFT = {
    "city2": 2016,
    "city5": 2016,
    "city8": 2016,
    "city12": 2016,
    "city13": 2016,
    "city15": 2016,
    "city18": 2016,

    "city22": 2017,
    "city26": 2017,
    "city30": 2017,
    "city34": 2017,
    "city37": 2017,
    "city40": 2017,
}
def load_population_density():
    """读取人口密度原始表。"""
    return pd.read_excel(DATA_DIR / "人口密度.xlsx")


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

    print("\n=== RAW KEY CHECK ===")
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


def normalize_population_density(df):
    """
    已确认 repair：
    city31-2002 同时存在 2315 和 422。
    根据邻近年份序列，删除冲突值 2315，保留 422。
    """
    normalized_df = df.copy()

    bad_row = (
        (normalized_df["城市名称"] == "city31")
        & (normalized_df["年份"] == 2002)
        & (normalized_df[DENSITY_COLUMN] == 2315)
    )

    return normalized_df.loc[~bad_row].copy()


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


def check_value_sanity(df):
    values = df[DENSITY_COLUMN]

    print("\n=== VALUE SANITY ===")
    print(f"min = {values.min()}")
    print(f"max = {values.max()}")
    print(f"zero_count = {(values == 0).sum()}")
    print(f"negative_count = {(values < 0).sum()}")


def verify_normalized(df):
    duplicate_rows = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    ).sum()

    print("\n=== NORMALIZED CHECK ===")
    print(f"rows = {len(df)}")
    print(f"unique_city_year_keys = {df[['城市名称', '年份']].drop_duplicates().shape[0]}")
    print(f"duplicate_rows = {duplicate_rows}")


def save_normalized(df):
    output_path = NORMALIZED_DIR / "人口密度.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nsaved: {output_path}")
    print(f"shape: {df.shape}")

def investigate_density_regime_changes(
    raw_df,
    normalized_density_df,
):
    """
    第二轮 cross-table investigation。

    目标：
    1. 找出多个城市同步发生的 sustained density jump；
    2. 利用常住人口计算 implied area（隐含面积）；
    3. 判断跳变更像人口变化，还是 density 分母/统计范围变化；
    4. 检查这些城市是否与其他表中的 source-pattern 重合。

    本函数只调查，不修改数据。
    """

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    # =====================================================
    # A. Density × resident population
    # =====================================================

    merged = normalized_density_df.merge(
        population[
            [
                "城市名称",
                "年份",
                "常住人口（万人）",
            ]
        ],
        on=["城市名称", "年份"],
        how="inner",
        validate="one_to_one",
    )

    # 无常住人口时不能计算隐含面积。
    merged = merged.dropna(
        subset=["常住人口（万人）"]
    ).copy()

    merged = merged.sort_values(
        ["城市名称", "年份"]
    )

    # 常住人口单位为“万人”：
    #
    # area =
    # resident_population(person)
    # / density(person / km²)
    merged["implied_area_km2"] = (
        merged["常住人口（万人）"]
        * 10000
        / merged[DENSITY_COLUMN]
    )

    grouped = merged.groupby(
        "城市名称"
    )

    merged["prev_year"] = (
        grouped["年份"].shift(1)
    )

    merged["prev_density"] = (
        grouped[DENSITY_COLUMN]
        .shift(1)
    )

    merged["density_ratio"] = (
        merged[DENSITY_COLUMN]
        / merged["prev_density"]
    )

    # 只比较真正连续的两个年份。
    consecutive = (
        merged["年份"]
        - merged["prev_year"]
        == 1
    )

    # 大于等于 1.4：
    # 不是把 40% 定义成“错误”，
    # 只是作为 investigation trigger。
    large_jump = merged[
        consecutive
        & (
            merged[
                "density_ratio"
            ] >= 1.4
        )
    ].copy()

    print(
        "\n=== LARGE CONSECUTIVE "
        "DENSITY JUMPS ==="
    )

    print(
        large_jump[
            [
                "城市名称",
                "年份",
                "prev_density",
                DENSITY_COLUMN,
                "density_ratio",
            ]
        ]
        .sort_values(
            ["年份", "城市名称"]
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # B. Detect cohort-level jump years
    # =====================================================

    jump_counts = (
        large_jump
        .groupby("年份")
        .size()
        .sort_index()
    )

    print(
        "\n=== LARGE JUMPS BY YEAR ==="
    )

    print(
        jump_counts.to_string()
    )

    # 至少 3 个城市在同一年一起跳变，
    # 才提升为 cohort-level candidate。
    regime_years = (
        jump_counts[
            jump_counts >= 3
        ]
        .index
        .tolist()
    )

    assert regime_years == [
        2016,
        2017,
    ]

    cohorts = {}

    for year in regime_years:
        cities = sorted(
            large_jump.loc[
                large_jump["年份"]
                == year,
                "城市名称",
            ].tolist(),
            key=lambda x: int(
                x.removeprefix("city")
            ),
        )

        cohorts[year] = cities

    print(
        "\n=== REGIME COHORTS ==="
    )

    for year, cities in (
        cohorts.items()
    ):
        print(
            f"{year}: "
            f"{cities}"
        )

    expected_2016 = {
        "city2",
        "city5",
        "city8",
        "city12",
        "city13",
        "city15",
        "city18",
    }

    expected_2017 = {
        "city22",
        "city26",
        "city30",
        "city34",
        "city37",
        "city40",
    }

    assert (
        set(cohorts[2016])
        == expected_2016
    )

    assert (
        set(cohorts[2017])
        == expected_2017
    )

    # =====================================================
    # C. Compare implied area before / after shift
    # =====================================================

    def relative_mad(series):
        """
        Relative median absolute deviation.

        用中位数而不是 mean/std，
        减少单点异常对稳定性判断的影响。
        """
        median = series.median()

        return (
            (series - median)
            .abs()
            .median()
            / median
        )

    area_rows = []

    for shift_year, cities in (
        cohorts.items()
    ):
        for city in cities:
            city_df = merged[
                merged["城市名称"]
                == city
            ].sort_values(
                "年份"
            )

            # shift 前最近 5 个有效年份。
            pre = (
                city_df[
                    city_df["年份"]
                    < shift_year
                ]
                .tail(5)
            )

            # shift 后最早 3 个有效年份。
            post = (
                city_df[
                    city_df["年份"]
                    >= shift_year
                ]
                .head(3)
            )

            pre_area = (
                pre[
                    "implied_area_km2"
                ].median()
            )

            post_area = (
                post[
                    "implied_area_km2"
                ].median()
            )

            area_rows.append(
                {
                    "城市名称": city,
                    "shift_year":
                        shift_year,
                    "pre_area_median":
                        pre_area,
                    "pre_relative_MAD":
                        relative_mad(
                            pre[
                                "implied_area_km2"
                            ]
                        ),
                    "post_area_median":
                        post_area,
                    "post_relative_MAD":
                        relative_mad(
                            post[
                                "implied_area_km2"
                            ]
                        ),
                    "post_pre_ratio":
                        post_area
                        / pre_area,
                }
            )

    area_summary = pd.DataFrame(
        area_rows
    )

    print(
        "\n=== IMPLIED AREA "
        "BEFORE / AFTER REGIME ==="
    )

    display = area_summary.copy()

    for column in [
        "pre_area_median",
        "post_area_median",
    ]:
        display[column] = (
            display[column]
            .round(1)
        )

    for column in [
        "pre_relative_MAD",
        "post_relative_MAD",
        "post_pre_ratio",
    ]:
        display[column] = (
            display[column]
            .round(3)
        )

    print(
        display.to_string(
            index=False
        )
    )

    # =====================================================
    # D. Raw source-layout evidence
    # =====================================================

    source = raw_df.copy()

    # 一个 run：
    # 原始 Excel 中同一 city 连续出现的一段。
    #
    # 注意：
    # 这里不是赋予 row order 时间语义，
    # 而是在调查 source construction。
    source["new_run"] = (
        source["城市名称"]
        != source["城市名称"]
        .shift(1)
    )

    source["run_id"] = (
        source["new_run"]
        .cumsum()
    )

    run_counts = (
        source.groupby(
            "城市名称"
        )["run_id"]
        .nunique()
    )

    print(
        "\n=== RAW SOURCE LAYOUT "
        "FOR REGIME COHORTS ==="
    )

    for year, cities in (
        cohorts.items()
    ):
        print(
            f"\nshift_year = {year}"
        )

        for city in cities:
            print(
                f"{city}: "
                f"raw_runs="
                f"{run_counts[city]}"
            )

    # =====================================================
    # E. Cross-table source-pattern evidence
    # =====================================================

    population_gap_cities = set()

    for city, group in (
        population.groupby(
            "城市名称"
        )
    ):
        years = sorted(
            group["年份"]
            .dropna()
            .astype(int)
            .unique()
        )

        expected = set(
            range(
                min(years),
                max(years) + 1,
            )
        )

        missing = (
            expected
            - set(years)
        )

        if missing:
            population_gap_cities.add(
                city
            )

    density_regime_cities = (
        set(cohorts[2016])
        | set(cohorts[2017])
    )

    overlap = (
        density_regime_cities
        & population_gap_cities
    )

    print(
        "\n=== CROSS-TABLE SOURCE "
        "PATTERN ==="
    )

    print(
        "population_internal_gap_cities =",
        sorted(
            population_gap_cities,
            key=lambda x: int(
                x.removeprefix("city")
            ),
        ),
    )

    print(
        "density_regime_cities =",
        sorted(
            density_regime_cities,
            key=lambda x: int(
                x.removeprefix("city")
            ),
        ),
    )

    print(
        "overlap =",
        sorted(
            overlap,
            key=lambda x: int(
                x.removeprefix("city")
            ),
        ),
    )

    print(
        f"overlap_count = "
        f"{len(overlap)} / "
        f"{len(population_gap_cities)} "
        f"population-gap cities"
    )

    assert (
        len(population_gap_cities)
        == 10
    )

    assert (
        overlap
        == population_gap_cities
    )
def investigate_isolated_density_anomalies(
    normalized_density_df,
):
    """
    找“单年跳出去、下一年又恢复”的 density anomaly。

    和 regime change 区别：
    - regime change：跳变以后持续处于新水平
    - isolated anomaly：中间一年异常，前后邻年仍处于同一水平
    """

    df = (
        normalized_density_df
        .sort_values(
            ["城市名称", "年份"]
        )
        .copy()
    )

    grouped = df.groupby(
        "城市名称"
    )

    df["prev_year"] = (
        grouped["年份"].shift(1)
    )

    df["next_year"] = (
        grouped["年份"].shift(-1)
    )

    df["prev_density"] = (
        grouped[DENSITY_COLUMN]
        .shift(1)
    )

    df["next_density"] = (
        grouped[DENSITY_COLUMN]
        .shift(-1)
    )

    # 只比较真正连续的三年：
    # t-1, t, t+1
    three_consecutive = (
        (df["年份"] - df["prev_year"] == 1)
        & (
            df["next_year"]
            - df["年份"]
            == 1
        )
    )

    # 前后两年自己的差异。
    #
    # 如果前后邻年很接近，
    # 但中间一年明显偏离，
    # 就很像 isolated spike/dip。
    neighbor_mean = (
        df["prev_density"]
        + df["next_density"]
    ) / 2

    df[
        "neighbor_relative_difference"
    ] = (
        (
            df["next_density"]
            - df["prev_density"]
        ).abs()
        / neighbor_mean
    )

    df[
        "center_relative_deviation"
    ] = (
        (
            df[DENSITY_COLUMN]
            - neighbor_mean
        ).abs()
        / neighbor_mean
    )

    # Investigation trigger：
    #
    # 前后两年相差 <= 10%
    # 中间年份偏离前后平均 >= 30%
    #
    # 这里只是找候选，
    # 不把阈值本身当“错误定义”。
    candidates = df[
        three_consecutive
        & (
            df[
                "neighbor_relative_difference"
            ]
            <= 0.10
        )
        & (
            df[
                "center_relative_deviation"
            ]
            >= 0.30
        )
    ].copy()

    print(
        "\n=== ISOLATED DENSITY "
        "ANOMALY CANDIDATES ==="
    )

    display = candidates[
        [
            "城市名称",
            "年份",
            "prev_density",
            DENSITY_COLUMN,
            "next_density",
            "neighbor_relative_difference",
            "center_relative_deviation",
        ]
    ].copy()

    display[
        "neighbor_relative_difference"
    ] = (
        display[
            "neighbor_relative_difference"
        ]
        .map(
            lambda x: f"{x:.2%}"
        )
    )

    display[
        "center_relative_deviation"
    ] = (
        display[
            "center_relative_deviation"
        ]
        .map(
            lambda x: f"{x:.2%}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    # =====================================================
    # Cross-table population check
    # =====================================================

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    comparison = candidates.merge(
        population[
            [
                "城市名称",
                "年份",
                "常住人口（万人）",
            ]
        ],
        on=["城市名称", "年份"],
        how="left",
        validate="one_to_one",
    )

    # 前后常住人口也一起取出。
    population_sorted = (
        population
        .sort_values(
            ["城市名称", "年份"]
        )
        .copy()
    )

    pop_grouped = (
        population_sorted.groupby(
            "城市名称"
        )
    )

    population_sorted[
        "prev_population"
    ] = (
        pop_grouped[
            "常住人口（万人）"
        ].shift(1)
    )

    population_sorted[
        "next_population"
    ] = (
        pop_grouped[
            "常住人口（万人）"
        ].shift(-1)
    )

    comparison = comparison.merge(
        population_sorted[
            [
                "城市名称",
                "年份",
                "prev_population",
                "next_population",
            ]
        ],
        on=["城市名称", "年份"],
        how="left",
        validate="one_to_one",
    )

    print(
        "\n=== POPULATION CONTEXT "
        "FOR ISOLATED CANDIDATES ==="
    )

    print(
        comparison[
            [
                "城市名称",
                "年份",
                "prev_density",
                DENSITY_COLUMN,
                "next_density",
                "prev_population",
                "常住人口（万人）",
                "next_population",
            ]
        ].to_string(
            index=False
        )
    )

    # =====================================================
    # Implied area context
    # =====================================================

    comparison[
        "prev_implied_area"
    ] = (
        comparison[
            "prev_population"
        ]
        * 10000
        / comparison[
            "prev_density"
        ]
    )

    comparison[
        "current_implied_area"
    ] = (
        comparison[
            "常住人口（万人）"
        ]
        * 10000
        / comparison[
            DENSITY_COLUMN
        ]
    )

    comparison[
        "next_implied_area"
    ] = (
        comparison[
            "next_population"
        ]
        * 10000
        / comparison[
            "next_density"
        ]
    )

    print(
        "\n=== IMPLIED AREA CONTEXT ==="
    )

    area_display = comparison[
        [
            "城市名称",
            "年份",
            "prev_implied_area",
            "current_implied_area",
            "next_implied_area",
        ]
    ].copy()

    for col in [
        "prev_implied_area",
        "current_implied_area",
        "next_implied_area",
    ]:
        area_display[col] = (
            area_display[col]
            .round(1)
        )

    print(
        area_display.to_string(
            index=False
        )
    )
DENSITY_BATCH_ANOMALY_KEYS = {
    # 2016 regime cohort 的 pre-regime batch anomaly
    ("city2", 2013),
    ("city5", 2013),
    ("city8", 2013),
    ("city12", 2013),
    ("city13", 2013),
    ("city15", 2013),
    ("city18", 2013),

    # 2017 regime cohort
    ("city22", 2014),
    ("city26", 2014),
    ("city30", 2014),
    ("city34", 2014),
    ("city37", 2014),
    ("city40", 2014),
}
def verify_density_batch_anomalies(
    raw_df,
    normalized_density_df,
):
    """
    验证两个 density regime cohort 在 regime change 之前
    是否存在共同的单年 batch anomaly。

    这里验证的是：
    1. 中间一年 density 明显偏离前后邻年；
    2. 这种偏离远大于该城市平时的局部波动；
    3. 同时常住人口保持平稳；
    4. anomaly 在 cohort/year 上具有明显结构。

    本函数只验证，不修改数据。
    """

    df = (
        normalized_density_df
        .sort_values(
            ["城市名称", "年份"]
        )
        .copy()
    )

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    # =====================================================
    # A. Build local three-year context
    # =====================================================

    grouped = df.groupby(
        "城市名称"
    )

    df["prev_year"] = (
        grouped["年份"].shift(1)
    )

    df["next_year"] = (
        grouped["年份"].shift(-1)
    )

    df["prev_density"] = (
        grouped[DENSITY_COLUMN]
        .shift(1)
    )

    df["next_density"] = (
        grouped[DENSITY_COLUMN]
        .shift(-1)
    )

    consecutive = (
        (df["年份"] - df["prev_year"] == 1)
        & (
            df["next_year"]
            - df["年份"]
            == 1
        )
    )

    df = df[
        consecutive
    ].copy()

    df["neighbor_mean"] = (
        df["prev_density"]
        + df["next_density"]
    ) / 2

    df["local_deviation"] = (
        (
            df[DENSITY_COLUMN]
            - df["neighbor_mean"]
        ).abs()
        / df["neighbor_mean"]
    )

    # =====================================================
    # B. Estimate each city's ordinary local variation
    # =====================================================

    anomaly_year_by_city = {
        city: year
        for city, year
        in DENSITY_BATCH_ANOMALY_KEYS
    }

    typical_rows = []

    for city, anomaly_year in (
        anomaly_year_by_city.items()
    ):
        city_df = df[
            df["城市名称"]
            == city
        ].copy()

        # 只使用 regime change 之前的数据。
        shift_year = (
            2016
            if anomaly_year == 2013
            else 2017
        )

        city_df = city_df[
            city_df["年份"]
            < shift_year - 1
        ]

        # candidate 会污染 t-1 和 t+1 的三点局部统计，
        # 因此三个年份都排除。
        clean_baseline = city_df[
            ~city_df["年份"].isin(
                {
                    anomaly_year - 1,
                    anomaly_year,
                    anomaly_year + 1,
                }
            )
        ]

        typical_deviation = (
            clean_baseline[
                "local_deviation"
            ].median()
        )

        candidate = df[
            (df["城市名称"] == city)
            & (
                df["年份"]
                == anomaly_year
            )
        ]

        assert len(candidate) == 1

        candidate_deviation = (
            candidate[
                "local_deviation"
            ].iloc[0]
        )

        typical_rows.append(
            {
                "城市名称": city,
                "年份": anomaly_year,
                "prev_density":
                    candidate[
                        "prev_density"
                    ].iloc[0],
                "density":
                    candidate[
                        DENSITY_COLUMN
                    ].iloc[0],
                "next_density":
                    candidate[
                        "next_density"
                    ].iloc[0],
                "local_deviation":
                    candidate_deviation,
                "typical_local_deviation":
                    typical_deviation,
                "deviation_multiple":
                    candidate_deviation
                    / typical_deviation,
            }
        )

    evidence = pd.DataFrame(
        typical_rows
    )

    # =====================================================
    # C. Attach population evidence
    # =====================================================

    pop = (
        population
        .sort_values(
            ["城市名称", "年份"]
        )
        .copy()
    )

    pop_grouped = pop.groupby(
        "城市名称"
    )

    pop["prev_population"] = (
        pop_grouped[
            "常住人口（万人）"
        ].shift(1)
    )

    pop["next_population"] = (
        pop_grouped[
            "常住人口（万人）"
        ].shift(-1)
    )

    pop["neighbor_population"] = (
        pop["prev_population"]
        + pop["next_population"]
    ) / 2

    pop[
        "population_local_deviation"
    ] = (
        (
            pop["常住人口（万人）"]
            - pop[
                "neighbor_population"
            ]
        ).abs()
        / pop[
            "neighbor_population"
        ]
    )

    evidence = evidence.merge(
        pop[
            [
                "城市名称",
                "年份",
                "prev_population",
                "常住人口（万人）",
                "next_population",
                "population_local_deviation",
            ]
        ],
        on=["城市名称", "年份"],
        how="left",
        validate="one_to_one",
    )

    # =====================================================
    # D. Display evidence
    # =====================================================

    print(
        "\n=== DENSITY BATCH "
        "ANOMALY VERIFICATION ==="
    )

    display = evidence.copy()

    for col in [
        "local_deviation",
        "typical_local_deviation",
        "population_local_deviation",
    ]:
        display[col] = (
            display[col]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

    display[
        "deviation_multiple"
    ] = (
        display[
            "deviation_multiple"
        ].round(1)
    )

    print(
        display[
            [
                "城市名称",
                "年份",
                "prev_density",
                "density",
                "next_density",
                "local_deviation",
                "typical_local_deviation",
                "deviation_multiple",
                "prev_population",
                "常住人口（万人）",
                "next_population",
                "population_local_deviation",
            ]
        ]
        .sort_values(
            ["年份", "城市名称"]
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # E. Raw source-layout evidence
    # =====================================================

    source = raw_df.copy()

    source["source_index"] = (
        source.index
    )

    print(
        "\n=== RAW ROWS FOR "
        "BATCH ANOMALIES ==="
    )

    source["key"] = list(
        zip(
            source["城市名称"],
            source["年份"],
        )
    )

    anomaly_raw = source[
        source["key"].isin(
            DENSITY_BATCH_ANOMALY_KEYS
        )
    ].copy()

    print(
        anomaly_raw[
            [
                "source_index",
                "城市名称",
                "年份",
                DENSITY_COLUMN,
            ]
        ]
        .sort_values(
            "source_index"
        )
        .to_string(
            index=False
        )
    )

    # 2017 cohort 的 2014 六行，
    # 在 raw source 中应形成连续物理 block。
    cohort_2014 = anomaly_raw[
        anomaly_raw["年份"]
        == 2014
    ].sort_values(
        "source_index"
    )

    indexes = (
        cohort_2014[
            "source_index"
        ]
        .tolist()
    )

    print(
        "\n2014 cohort raw indexes =",
        indexes,
    )

    assert len(indexes) == 6

    assert indexes == list(
        range(
            min(indexes),
            min(indexes) + 6,
        )
    )

    # =====================================================
    # F. Final verification
    # =====================================================

    keys = set(
        zip(
            evidence["城市名称"],
            evidence["年份"],
        )
    )

    assert (
        keys
        == DENSITY_BATCH_ANOMALY_KEYS
    )

    assert len(evidence) == 13

    print(
        "\nminimum anomaly / "
        "typical multiple =",
        evidence[
            "deviation_multiple"
        ].min(),
    )

    print(
        "maximum population "
        "local deviation =",
        evidence[
            "population_local_deviation"
        ].max(),
    )

    # 这是 verification threshold，
    # 不是最初用于发现 anomaly 的规则。
    assert (
        evidence[
            "deviation_multiple"
        ].min()
        > 40
    )

    assert (
        evidence[
            "population_local_deviation"
        ].max()
        < 0.01
    )
def build_population_density_v2(
    normalized_df,
):
    """
    第二轮 cross-table audit 后的人口密度处理。

    已确认两类现象：

    1. 13 个 pre-regime 单年异常点：
       - density 呈 A -> B -> A
       - 常住人口同期基本不变
       - 异常幅度至少为该城市平常局部波动的 43 倍
       - 严格按 source cohort / year 聚集

       这些点：
       raw value -> quarantine
       canonical value -> NA

    2. 2016 / 2017 后持续的新 density regime：
       - 切换前后各自内部稳定
       - 但 implied area 明显改变
       - 更像来源/统计分母口径改变

       这些值保留，不进行数值修复。

    注意：
    知道异常值不可信 != 知道正确值。
    因此不做线性插值或人工估值。
    """

    df = normalized_df.copy()

    # density 原来是整数列；
    # 为容纳 pd.NA 转为 nullable Float64。
    df[DENSITY_COLUMN] = (
        df[DENSITY_COLUMN]
        .astype("Float64")
    )

    quarantine_rows = []

    for (
        city,
        year,
    ), expected_raw_value in (
        DENSITY_BATCH_ANOMALY_VALUES.items()
    ):
        mask = (
            (df["城市名称"] == city)
            & (df["年份"] == year)
        )

        # key 必须存在且唯一。
        assert mask.sum() == 1

        raw_value = df.loc[
            mask,
            DENSITY_COLUMN,
        ].iloc[0]

        # fail-closed：
        # 如果 source 已变化，不继续套用旧判断。
        assert (
            raw_value
            == expected_raw_value
        )

        quarantine_rows.append(
            {
                "城市名称": city,
                "年份": year,
                "raw_density":
                    raw_value,
                "regime_shift_year":
                    DENSITY_REGIME_SHIFT[
                        city
                    ],
                "reason":
                    (
                        "high-confidence "
                        "pre-regime batch anomaly; "
                        "A-B-A density pattern with "
                        "stable resident population"
                    ),
                "action":
                    (
                        "invalidate canonical "
                        "density to NA"
                    ),
            }
        )

        df.loc[
            mask,
            DENSITY_COLUMN,
        ] = pd.NA

    quarantine = pd.DataFrame(
        quarantine_rows
    )

    return df, quarantine
def build_density_regime_report():
    rows = []

    for city, shift_year in (
        DENSITY_REGIME_SHIFT.items()
    ):
        rows.append(
            {
                "城市名称": city,
                "regime_shift_year":
                    shift_year,
                "pre_regime_anomaly_year":
                    (
                        2013
                        if shift_year == 2016
                        else 2014
                    ),
                "status":
                    "retained",
                "interpretation":
                    (
                        "suspected source/denominator "
                        "regime change; values retained "
                        "because post-shift regime is persistent"
                    ),
            }
        )

    return pd.DataFrame(rows)
def verify_population_density_v2(
    density_v2,
    quarantine,
):
    print(
        "\n=== POPULATION DENSITY "
        "V2 VERIFICATION ==="
    )

    duplicate_rows = (
        density_v2
        .duplicated(
            ["城市名称", "年份"],
            keep=False,
        )
        .sum()
    )

    missing_density = (
        density_v2[
            DENSITY_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"rows = {len(density_v2)}"
    )

    print(
        f"unique_keys = "
        f"{density_v2[['城市名称', '年份']].drop_duplicates().shape[0]}"
    )

    print(
        f"duplicate_rows = "
        f"{duplicate_rows}"
    )

    print(
        f"cities = "
        f"{density_v2['城市名称'].nunique()}"
    )

    print(
        f"missing_density = "
        f"{missing_density}"
    )

    print(
        f"quarantine_rows = "
        f"{len(quarantine)}"
    )

    assert len(density_v2) == 757
    assert duplicate_rows == 0
    assert (
        density_v2[
            "城市名称"
        ].nunique()
        == 40
    )

    assert missing_density == 13
    assert len(quarantine) == 13

    missing_keys = set(
        zip(
            density_v2.loc[
                density_v2[
                    DENSITY_COLUMN
                ].isna(),
                "城市名称",
            ],
            density_v2.loc[
                density_v2[
                    DENSITY_COLUMN
                ].isna(),
                "年份",
            ],
        )
    )

    assert (
        missing_keys
        == set(
            DENSITY_BATCH_ANOMALY_VALUES
        )
    )

    # 原来 city31-2002 的 duplicate repair
    # 仍然必须存在且只剩 422。
    city31_2002 = density_v2[
        (density_v2["城市名称"] == "city31")
        & (density_v2["年份"] == 2002)
    ]

    assert len(city31_2002) == 1

    assert (
        city31_2002[
            DENSITY_COLUMN
        ].iloc[0]
        == 422
    )

    print(
        "\n=== DENSITY QUARANTINE ==="
    )

    print(
        quarantine
        .sort_values(
            ["年份", "城市名称"]
        )
        .to_string(index=False)
    )
def save_population_density_v2(
    density_v2,
    quarantine,
    regime_report,
):
    normalized_path = (
        NORMALIZED_DIR
        / "人口密度.csv"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "人口密度_quarantine.csv"
    )

    regime_report_path = (
        NORMALIZED_DIR
        / "人口密度_regime_report.csv"
    )

    density_v2.to_csv(
        normalized_path,
        index=False,
        encoding="utf-8-sig",
    )

    quarantine.to_csv(
        quarantine_path,
        index=False,
        encoding="utf-8-sig",
    )

    regime_report.to_csv(
        regime_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nsaved: "
        f"{normalized_path}"
    )

    print(
        f"shape: "
        f"{density_v2.shape}"
    )

    print(
        f"saved quarantine: "
        f"{quarantine_path}"
    )

    print(
        f"quarantine shape: "
        f"{quarantine.shape}"
    )

    print(
        f"saved regime report: "
        f"{regime_report_path}"
    )
def main():
    raw_df = load_population_density()

    # =============================================
    # 1. Raw audit
    # =============================================

    check_schema(raw_df)
    check_key(raw_df)
    check_key_format(raw_df)

    # =============================================
    # 2. First-round confirmed repair
    # =============================================

    base_normalized = (
        normalize_population_density(
            raw_df
        )
    )

    # 验证 duplicate repair：
    # city31-2002 2315 删除、422 保留。
    verify_normalized(
        base_normalized
    )

    # =============================================
    # 3. Second-round batch invalidation
    # =============================================

    density_v2, quarantine = (
        build_population_density_v2(
            base_normalized
        )
    )

    regime_report = (
        build_density_regime_report()
    )

    verify_population_density_v2(
        density_v2,
        quarantine,
    )

    # =============================================
    # 4. Final audits
    # =============================================

    check_coverage(
        density_v2
    )

    check_missing(
        density_v2
    )

    check_value_sanity(
        density_v2
    )

    # =============================================
    # 5. Save
    # =============================================

    save_population_density_v2(
        density_v2,
        quarantine,
        regime_report,
    )


if __name__ == "__main__":
    main()
