from pathlib import Path

import pandas as pd


# =========================================================
# Paths / Constants
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "城市人口分析与预测" / "数据集"
NORMALIZED_DIR = ROOT / "data" / "normalized"

AGE_FILE = DATA_DIR / "年龄结构.xlsx"
QUARANTINE_DIR = ROOT / "data" / "quarantine"

CITY_PATTERN = r"city(?:[1-9]|[1-3][0-9]|40)"
EXPECTED_CITIES = {f"city{i}" for i in range(1, 41)}
EXPECTED_YEARS = {2000, 2010, 2020}

AGE_COLUMNS = [
    "0-14",
    "15-64",
    "65+",
]

KNOWN_MISSING_KEYS = {
    ("city25", 2000),
    ("city26", 2000),
}

KNOWN_SUSPICIOUS_2000 = {
    "city3",
    "city16",
    "city23",
    "city27",
    "city35",
    "city38",
}

POPULATION_NORMALIZED_PATH = (
    NORMALIZED_DIR / "人口规模.csv"
)
# =========================================================
# Load
# =========================================================

def load_age_structure():
    return pd.read_excel(
        AGE_FILE,
        sheet_name="Sheet1",
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


def check_key_and_coverage(df):
    duplicate_rows = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    ).sum()

    valid_city = (
        df["城市名称"]
        .str.fullmatch(CITY_PATTERN)
    )

    observed_cities = set(
        df["城市名称"]
    )

    print("\n=== KEY / COVERAGE ===")
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
        "years =",
        sorted(df["年份"].unique()),
    )

    print(
        f"invalid_city_rows = "
        f"{(~valid_city).sum()}"
    )

    print(
        "missing_cities =",
        sorted(
            EXPECTED_CITIES
            - observed_cities
        ),
    )

    print("\nrows_by_year:")
    print(
        df["年份"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def check_missing_structure(df):
    actual_keys = set(
        zip(
            df["城市名称"],
            df["年份"],
        )
    )

    expected_keys = {
        (city, year)
        for city in EXPECTED_CITIES
        for year in EXPECTED_YEARS
    }

    missing_keys = (
        expected_keys - actual_keys
    )

    print("\n=== MISSING STRUCTURE ===")
    print(
        "missing_keys =",
        sorted(missing_keys),
    )

    print("\ncell_missing:")
    print(
        df[AGE_COLUMNS]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "rows_with_missing_age_value =",
        df[
            AGE_COLUMNS
        ]
        .isna()
        .any(axis=1)
        .sum(),
    )


def check_value_domain(df):
    print("\n=== VALUE DOMAIN ===")

    for column in AGE_COLUMNS:
        values = df[column].dropna()

        print(f"\n--- {column} ---")
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


def check_known_temporal_anomalies(df):
    """
    报告 raw source 中已识别的 severe temporal anomalies。

    这些值在基础 normalization 阶段原样保留，
    随后由 second-round cross-table audit
    决定是否 invalidation。

    注意：
    raw row order 不具有时间语义。
    时间比较前显式按 key 选取 / 排序。
    """
    complete = df[
        df[AGE_COLUMNS]
        .notna()
        .all(axis=1)
        & df["城市名称"].isin(
            KNOWN_SUSPICIOUS_2000
        )
        & df["年份"].isin(
            [2000, 2010]
        )
    ].copy()

    complete["age_total"] = (
        complete["0-14"]
        + complete["15-64"]
        + complete["65+"]
    )

    complete = complete.sort_values(
        ["城市名称", "年份"]
    )

    print(
        "\n=== KNOWN TEMPORAL ANOMALIES ==="
    )

    print(
        complete[
            [
                "城市名称",
                "年份",
                "0-14",
                "15-64",
                "65+",
                "age_total",
            ]
        ].to_string(
            index=False
        )
    )


# =========================================================
# Normalization
# =========================================================

def normalize_age_structure(df):
    """
    基础 canonicalization。

    本阶段只忠实保留 raw source：
    - city25-2000 / city26-2000 缺失 key；
    - 原始 cell-level NaN；
    - raw severe temporal anomalies。

    second-round cross-table invalidation
    在后续 build_age_structure_v2() 中执行。

    这样可以区分：
    source representation
    与 evidence-backed cleaning decision。
    """

    return df[
        [
            "城市名称",
            "年份",
            "0-14",
            "15-64",
            "65+",
        ]
    ].copy()


# =========================================================
# Verification
# =========================================================

def verify_age_structure(df):
    duplicate_rows = df.duplicated(
        ["城市名称", "年份"],
        keep=False,
    ).sum()

    actual_keys = set(
        zip(
            df["城市名称"],
            df["年份"],
        )
    )

    expected_keys = {
        (city, year)
        for city in EXPECTED_CITIES
        for year in EXPECTED_YEARS
    }

    missing_keys = (
        expected_keys - actual_keys
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
        "years =",
        sorted(df["年份"].unique()),
    )

    print(
        "missing_keys =",
        sorted(missing_keys),
    )

    print("\nmissing values:")
    print(
        df[AGE_COLUMNS]
        .isna()
        .sum()
        .to_string()
    )

    assert len(df) == 118
    assert duplicate_rows == 0
    assert df["城市名称"].nunique() == 40
    assert set(df["年份"]) == EXPECTED_YEARS

    assert missing_keys == KNOWN_MISSING_KEYS

    assert df["0-14"].isna().sum() == 0
    assert df["15-64"].isna().sum() == 20
    assert df["65+"].isna().sum() == 15

    # 关键 unresolved anomaly 必须原样保留
    known_cases = {
        ("city3", 2000): (66, 174, 18),
        ("city16", 2000): (160, 745, 60),
        ("city23", 2000): (60, 629, 9),
        ("city27", 2000): (21, 98, 5),
    }

    for (city, year), expected in known_cases.items():
        row = df[
            (df["城市名称"] == city)
            & (df["年份"] == year)
        ].iloc[0]

        actual = (
            row["0-14"],
            row["15-64"],
            row["65+"],
        )

        assert actual == expected


# =========================================================
# Save
# =========================================================

def save_age_structure(df):
    output_path = (
        NORMALIZED_DIR
        / "年龄结构.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SAVE RESULT ===")
    print(
        f"normalized: {output_path}"
    )
    print(f"shape: {df.shape}")

def investigate_age_2000_cross_table(df):
    """
    第二轮 cross-table investigation。

    目标：
    用两个独立锚点检查 2000 年年龄结构总量：

    1. 同城市最早可用的常住人口；
    2. 同城市 2010 年年龄结构总量。

    注意：
    这里的比例阈值只用于筛选调查对象，
    不是“超过阈值就一定错误”的业务规则。
    """

    age = df.copy()

    age["age_total"] = (
        age["0-14"]
        + age["15-64"]
        + age["65+"]
    )

    # =====================================================
    # A. 2000 / 2010 age totals
    # =====================================================

    age_2000 = age[
        (age["年份"] == 2000)
        & age[
            AGE_COLUMNS
        ].notna().all(axis=1)
    ][
        [
            "城市名称",
            "age_total",
        ]
    ].rename(
        columns={
            "age_total":
                "age_total_2000"
        }
    )

    age_2010 = age[
        (age["年份"] == 2010)
        & age[
            AGE_COLUMNS
        ].notna().all(axis=1)
    ][
        [
            "城市名称",
            "age_total",
        ]
    ].rename(
        columns={
            "age_total":
                "age_total_2010"
        }
    )

    evidence = age_2000.merge(
        age_2010,
        on="城市名称",
        how="left",
        validate="one_to_one",
    )

    # =====================================================
    # B. Earliest valid resident population
    # =====================================================

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    valid_population = (
        population
        .dropna(
            subset=[
                "常住人口（万人）"
            ]
        )
        .sort_values(
            ["城市名称", "年份"]
        )
    )

    first_population = (
        valid_population
        .groupby(
            "城市名称",
            as_index=False,
        )
        .first()
        [
            [
                "城市名称",
                "年份",
                "常住人口（万人）",
            ]
        ]
        .rename(
            columns={
                "年份":
                    "first_population_year",
                "常住人口（万人）":
                    "first_resident_population",
            }
        )
    )

    evidence = evidence.merge(
        first_population,
        on="城市名称",
        how="left",
        validate="one_to_one",
    )

    # =====================================================
    # C. Multiplicative mismatch
    # =====================================================

    evidence[
        "ratio_to_first_population"
    ] = (
        evidence[
            "age_total_2000"
        ]
        / evidence[
            "first_resident_population"
        ]
    )

    evidence[
        "ratio_to_2010_age_total"
    ] = (
        evidence[
            "age_total_2000"
        ]
        / evidence[
            "age_total_2010"
        ]
    )

    def mismatch_factor(series):
        """
        例如：
        ratio = 0.25 -> mismatch factor = 4
        ratio = 4.00 -> mismatch factor = 4

        比直接看正负百分比更适合比较
        “过大”和“过小”两种异常。
        """

        return pd.concat(
            [
                series,
                1 / series,
            ],
            axis=1,
        ).max(axis=1)

    evidence[
        "first_population_mismatch"
    ] = mismatch_factor(
        evidence[
            "ratio_to_first_population"
        ]
    )

    evidence[
        "age_2010_mismatch"
    ] = mismatch_factor(
        evidence[
            "ratio_to_2010_age_total"
        ]
    )

    # =====================================================
    # D. Show all complete 2000 observations
    # =====================================================

    print(
        "\n=== AGE 2000 "
        "CROSS-TABLE CONSISTENCY ==="
    )

    display = evidence.copy()

    for col in [
        "ratio_to_first_population",
        "ratio_to_2010_age_total",
    ]:
        display[col] = (
            display[col]
            .round(3)
        )

    for col in [
        "first_population_mismatch",
        "age_2010_mismatch",
    ]:
        display[col] = (
            display[col]
            .round(2)
        )

    print(
        display[
            [
                "城市名称",
                "age_total_2000",
                "first_population_year",
                "first_resident_population",
                "ratio_to_first_population",
                "first_population_mismatch",
                "age_total_2010",
                "ratio_to_2010_age_total",
                "age_2010_mismatch",
            ]
        ]
        .sort_values(
            [
                "first_population_mismatch",
                "age_2010_mismatch",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # E. Investigation trigger
    # =====================================================

    # 这两个 threshold 只是筛选器：
    #
    # - 对最近人口规模偏差 >= 1.45 倍
    # 或
    # - 与 2010 年年龄总量偏差 >= 1.8 倍
    #
    # 后面仍需要进一步寻找 cross-city mapping、
    # source layout 和独立反证。
    suspicious = evidence[
        (
            evidence[
                "first_population_mismatch"
            ] >= 1.45
        )
        |
        (
            evidence[
                "age_2010_mismatch"
            ] >= 1.8
        )
    ].copy()

    print(
        "\n=== HIGH-PRIORITY "
        "2000 INVESTIGATION CANDIDATES ==="
    )

    print(
        suspicious[
            [
                "城市名称",
                "age_total_2000",
                "first_population_year",
                "first_resident_population",
                "first_population_mismatch",
                "age_total_2010",
                "age_2010_mismatch",
            ]
        ]
        .sort_values(
            [
                "first_population_mismatch",
                "age_2010_mismatch",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )
def rank_age_2000_cross_city_candidates(
    age_df,
):
    """
    对 6 个 suspicious 2000 age rows 做跨城市候选匹配。

    第一阶段只使用：
        age_total_2000
        vs
        2001 resident population

    为什么只用 2001：
    - 2000 -> 2001 只隔一年；
    - 可把人口自然变化压到很小；
    - 不把 2006/2007 才出现的人口记录混进来。

    本阶段只做 candidate ranking，
    不做 relabel。
    """

    age = age_df.copy()

    age["age_total"] = (
        age["0-14"]
        + age["15-64"]
        + age["65+"]
    )

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    # =====================================================
    # A. Build strict 2001 population anchors
    # =====================================================

    anchors = population[
        (population["年份"] == 2001)
        & population[
            "常住人口（万人）"
        ].notna()
    ][
        [
            "城市名称",
            "年份",
            "常住人口（万人）",
        ]
    ].copy()

    assert (
        anchors[
            "城市名称"
        ].duplicated().sum()
        == 0
    )

    print(
        "\n=== 2001 POPULATION ANCHORS ==="
    )

    print(
        f"anchor_cities = "
        f"{anchors['城市名称'].nunique()}"
    )

    # =====================================================
    # B. Suspicious complete 2000 rows
    # =====================================================

    suspicious = age[
        (age["年份"] == 2000)
        & age[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
        & age[
            AGE_COLUMNS
        ].notna()
        .all(axis=1)
    ].copy()

    assert len(suspicious) == 6

    ranking_rows = []
    summary_rows = []

    for _, source in (
        suspicious.iterrows()
    ):
        source_city = (
            source["城市名称"]
        )

        age_total = (
            source["age_total"]
        )

        candidates = (
            anchors.copy()
        )

        candidates[
            "relative_error"
        ] = (
            (
                candidates[
                    "常住人口（万人）"
                ]
                - age_total
            )
            .abs()
            / candidates[
                "常住人口（万人）"
            ]
        )

        candidates[
            "age_to_population_ratio"
        ] = (
            age_total
            / candidates[
                "常住人口（万人）"
            ]
        )

        candidates = (
            candidates
            .sort_values(
                "relative_error"
            )
            .reset_index(drop=True)
        )

        candidates[
            "rank"
        ] = (
            candidates.index
            + 1
        )

        top5 = candidates.head(
            5
        )

        for _, row in (
            top5.iterrows()
        ):
            ranking_rows.append(
                {
                    "source_city":
                        source_city,
                    "age_total_2000":
                        age_total,
                    "rank":
                        int(
                            row["rank"]
                        ),
                    "candidate_city":
                        row[
                            "城市名称"
                        ],
                    "resident_2001":
                        row[
                            "常住人口（万人）"
                        ],
                    "age/pop_ratio":
                        row[
                            "age_to_population_ratio"
                        ],
                    "relative_error":
                        row[
                            "relative_error"
                        ],
                }
            )

        best = candidates.iloc[0]
        second = candidates.iloc[1]

        own = candidates[
            candidates[
                "城市名称"
            ] == source_city
        ]

        if len(own) == 1:
            own_rank = int(
                own.iloc[0][
                    "rank"
                ]
            )

            own_error = (
                own.iloc[0][
                    "relative_error"
                ]
            )
        else:
            # source city 本身没有 2001 population anchor
            own_rank = pd.NA
            own_error = pd.NA

        summary_rows.append(
            {
                "source_city":
                    source_city,
                "age_total_2000":
                    age_total,

                "best_candidate":
                    best[
                        "城市名称"
                    ],
                "best_population":
                    best[
                        "常住人口（万人）"
                    ],
                "best_relative_error":
                    best[
                        "relative_error"
                    ],

                "second_candidate":
                    second[
                        "城市名称"
                    ],
                "second_relative_error":
                    second[
                        "relative_error"
                    ],

                "margin_second_minus_best":
                    (
                        second[
                            "relative_error"
                        ]
                        - best[
                            "relative_error"
                        ]
                    ),

                "own_rank":
                    own_rank,
                "own_relative_error":
                    own_error,
            }
        )

    ranking = pd.DataFrame(
        ranking_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    # =====================================================
    # C. Output
    # =====================================================

    print(
        "\n=== TOP-5 CROSS-CITY "
        "POPULATION MATCHES ==="
    )

    display = ranking.copy()

    display[
        "age/pop_ratio"
    ] = (
        display[
            "age/pop_ratio"
        ].round(3)
    )

    display[
        "relative_error"
    ] = (
        display[
            "relative_error"
        ].map(
            lambda x:
                f"{x:.2%}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n=== MATCH SUMMARY ==="
    )

    summary_display = (
        summary.copy()
    )

    for col in [
        "best_relative_error",
        "second_relative_error",
        "margin_second_minus_best",
        "own_relative_error",
    ]:
        summary_display[col] = (
            summary_display[col]
            .map(
                lambda x:
                    (
                        "NA"
                        if pd.isna(x)
                        else f"{x:.2%}"
                    )
            )
        )

    print(
        summary_display.to_string(
            index=False
        )
    )

    return ranking, summary
KNOWN_SUSPICIOUS_2000 = {
    "city3",
    "city16",
    "city23",
    "city27",
    "city35",
    "city38",
}
def evaluate_age_profile_candidates(
    age_df,
    ranking_df,
):
    """
    第二层 cross-city validation：

    先用非 suspicious 城市建立正常的
    2000 -> 2010 年龄结构变化参考分布；

    再测试：
        suspicious 2000 row
        ->
        candidate city's 2010 structure

    是否形成合理的人口结构演化。

    这里只用于 candidate validation，
    不直接 relabel。
    """

    age = age_df.copy()

    # =====================================================
    # A. Build age shares
    # =====================================================

    age["age_total"] = (
        age["0-14"]
        + age["15-64"]
        + age["65+"]
    )

    for col in AGE_COLUMNS:
        age[f"{col}_share"] = (
            age[col]
            / age["age_total"]
        )

    # 只使用三个年龄段都完整的记录。
    complete = age[
        age[
            AGE_COLUMNS
        ].notna().all(axis=1)
    ].copy()

    # =====================================================
    # B. Normal-city 2000 -> 2010 evolution
    # =====================================================

    age_2000 = complete[
        complete["年份"] == 2000
    ][
        [
            "城市名称",
            "0-14_share",
            "15-64_share",
            "65+_share",
        ]
    ].rename(
        columns={
            "0-14_share": "child_share_2000",
            "15-64_share": "working_share_2000",
            "65+_share": "elderly_share_2000",
        }
    )

    age_2010 = complete[
        complete["年份"] == 2010
    ][
        [
            "城市名称",
            "0-14_share",
            "15-64_share",
            "65+_share",
        ]
    ].rename(
        columns={
            "0-14_share": "child_share_2010",
            "15-64_share": "working_share_2010",
            "65+_share": "elderly_share_2010",
        }
    )

    transitions = age_2000.merge(
        age_2010,
        on="城市名称",
        how="inner",
        validate="one_to_one",
    )

    # suspicious 城市不能参与建立“正常参考分布”。
    controls = transitions[
        ~transitions[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
    ].copy()

    controls["delta_child"] = (
        controls["child_share_2010"]
        - controls["child_share_2000"]
    )

    controls["delta_elderly"] = (
        controls["elderly_share_2010"]
        - controls["elderly_share_2000"]
    )

    print(
        "\n=== CONTROL 2000->2010 "
        "AGE-SHARE EVOLUTION ==="
    )

    print(
        f"control_cities = "
        f"{len(controls)}"
    )

    def describe_delta(series):
        return {
            "median": series.median(),
            "q10": series.quantile(0.10),
            "q90": series.quantile(0.90),
            "mad": (
                series
                - series.median()
            ).abs().median(),
        }

    child_stats = describe_delta(
        controls["delta_child"]
    )

    elderly_stats = describe_delta(
        controls["delta_elderly"]
    )

    print(
        "child share change:",
        {
            k: round(v, 4)
            for k, v in child_stats.items()
        },
    )

    print(
        "elderly share change:",
        {
            k: round(v, 4)
            for k, v in elderly_stats.items()
        },
    )

    # =====================================================
    # C. Candidate target structures in 2010
    # =====================================================

    target_2010 = age_2010.set_index(
        "城市名称"
    )

    source_2000 = (
        complete[
            (complete["年份"] == 2000)
            & complete[
                "城市名称"
            ].isin(
                KNOWN_SUSPICIOUS_2000
            )
        ]
        .set_index("城市名称")
    )

    results = []

    # ranking_df 来自上一轮：
    # 每个 suspicious source 的人口 top-5 candidate。
    for _, candidate_row in (
        ranking_df.iterrows()
    ):
        source_city = (
            candidate_row[
                "source_city"
            ]
        )

        candidate_city = (
            candidate_row[
                "candidate_city"
            ]
        )

        if candidate_city not in (
            target_2010.index
        ):
            continue

        source = source_2000.loc[
            source_city
        ]

        target = target_2010.loc[
            candidate_city
        ]

        delta_child = (
            target[
                "child_share_2010"
            ]
            - source[
                "0-14_share"
            ]
        )

        delta_elderly = (
            target[
                "elderly_share_2010"
            ]
            - source[
                "65+_share"
            ]
        )

        # =================================================
        # Robust distance from normal demographic evolution
        # =================================================
        #
        # MAD 很小时加一个很小的 floor，
        # 防止除零。
        child_scale = max(
            child_stats["mad"] * 1.4826,
            0.005,
        )

        elderly_scale = max(
            elderly_stats["mad"] * 1.4826,
            0.005,
        )

        child_z = (
            delta_child
            - child_stats["median"]
        ) / child_scale

        elderly_z = (
            delta_elderly
            - elderly_stats["median"]
        ) / elderly_scale

        # 0-14 和 65+ 是两个独立方向；
        # 15-64 因三者和为1，不再重复计分。
        profile_distance = (
            abs(child_z)
            + abs(elderly_z)
        )

        results.append(
            {
                "source_city":
                    source_city,

                "candidate_city":
                    candidate_city,

                "population_rank":
                    candidate_row["rank"],

                "population_relative_error":
                    candidate_row[
                        "relative_error"
                    ],

                "source_child_share_2000":
                    source[
                        "0-14_share"
                    ],

                "target_child_share_2010":
                    target[
                        "child_share_2010"
                    ],

                "delta_child":
                    delta_child,

                "source_elderly_share_2000":
                    source[
                        "65+_share"
                    ],

                "target_elderly_share_2010":
                    target[
                        "elderly_share_2010"
                    ],

                "delta_elderly":
                    delta_elderly,

                "child_robust_z":
                    child_z,

                "elderly_robust_z":
                    elderly_z,

                "profile_distance":
                    profile_distance,
            }
        )

    result = pd.DataFrame(
        results
    )

    # =====================================================
    # D. Print all population top-5 candidates
    # =====================================================

    print(
        "\n=== POPULATION TOP-5 + "
        "AGE-PROFILE EVIDENCE ==="
    )

    display = result.copy()

    percent_cols = [
        "population_relative_error",
        "source_child_share_2000",
        "target_child_share_2010",
        "delta_child",
        "source_elderly_share_2000",
        "target_elderly_share_2010",
        "delta_elderly",
    ]

    for col in percent_cols:
        display[col] = (
            display[col]
            .map(
                lambda x:
                    f"{x:.2%}"
            )
        )

    for col in [
        "child_robust_z",
        "elderly_robust_z",
        "profile_distance",
    ]:
        display[col] = (
            display[col]
            .round(2)
        )

    print(
        display[
            [
                "source_city",
                "candidate_city",
                "population_rank",
                "population_relative_error",
                "delta_child",
                "delta_elderly",
                "child_robust_z",
                "elderly_robust_z",
                "profile_distance",
            ]
        ]
        .sort_values(
            [
                "source_city",
                "profile_distance",
            ]
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # E. Best profile candidate within population top-5
    # =====================================================

    best_profile = (
        result
        .sort_values(
            [
                "source_city",
                "profile_distance",
            ]
        )
        .groupby(
            "source_city",
            as_index=False,
        )
        .first()
    )

    print(
        "\n=== BEST PROFILE MATCH "
        "AMONG POPULATION TOP-5 ==="
    )

    best_display = (
        best_profile[
            [
                "source_city",
                "candidate_city",
                "population_rank",
                "population_relative_error",
                "profile_distance",
                "child_robust_z",
                "elderly_robust_z",
            ]
        ]
        .copy()
    )

    best_display[
        "population_relative_error"
    ] = (
        best_display[
            "population_relative_error"
        ]
        .map(
            lambda x:
                f"{x:.2%}"
        )
    )

    print(
        best_display.to_string(
            index=False
        )
    )

    return result
def inspect_suspicious_age_source_layout(
    raw_df,
):
    """
    检查 suspicious 2000 rows 在原始 Excel 中的物理位置。

    目标：
    判断是否存在：
    - 整行错位
    - 相邻城市交换
    - 某个 2000 block 整体 shift

    row order 本身不代表时间语义，
   这里只作为 source provenance evidence。
    """

    source = raw_df.reset_index(
        names="source_index"
    )

    print(
        "\n=== SUSPICIOUS AGE SOURCE LAYOUT ==="
    )

    for city in sorted(
        KNOWN_SUSPICIOUS_2000,
        key=lambda x: int(
            x.removeprefix("city")
        ),
    ):
        city_rows = source[
            source["城市名称"] == city
        ][
            [
                "source_index",
                "城市名称",
                "年份",
                "0-14",
                "15-64",
                "65+",
            ]
        ]

        print(
            f"\n--- {city} ---"
        )

        print(
            city_rows.to_string(
                index=False
            )
        )

        row_2000 = source[
            (source["城市名称"] == city)
            & (source["年份"] == 2000)
        ]

        assert len(row_2000) == 1

        idx = int(
            row_2000[
                "source_index"
            ].iloc[0]
        )

        start = max(
            0,
            idx - 2,
        )

        end = min(
            len(source),
            idx + 3,
        )

        print(
            "local raw context:"
        )

        print(
            source.iloc[
                start:end
            ][
                [
                    "source_index",
                    "城市名称",
                    "年份",
                    "0-14",
                    "15-64",
                    "65+",
                ]
            ].to_string(
                index=False
            )
        )
def verify_age_2000_control_envelope(
    age_df,
):
    """
    用“非 suspicious、完整”的城市建立经验正常范围。

    两个独立比较：

    A. age_total_2000 / resident_population_2001
    B. age_total_2000 / age_total_2010

    不假设二者严格等于 1；
    只问 suspicious rows 是否超出
    正常城市实际出现过的经验范围。
    """

    age = age_df.copy()

    age["age_total"] = (
        age["0-14"]
        + age["15-64"]
        + age["65+"]
    )

    complete = age[
        age[
            AGE_COLUMNS
        ].notna().all(axis=1)
    ].copy()

    age_2000 = complete[
        complete["年份"] == 2000
    ][
        [
            "城市名称",
            "age_total",
        ]
    ].rename(
        columns={
            "age_total":
                "age_total_2000"
        }
    )

    age_2010 = complete[
        complete["年份"] == 2010
    ][
        [
            "城市名称",
            "age_total",
        ]
    ].rename(
        columns={
            "age_total":
                "age_total_2010"
        }
    )

    population = pd.read_csv(
        POPULATION_NORMALIZED_PATH
    )

    population_2001 = population[
        (population["年份"] == 2001)
        & population[
            "常住人口（万人）"
        ].notna()
    ][
        [
            "城市名称",
            "常住人口（万人）",
        ]
    ].rename(
        columns={
            "常住人口（万人）":
                "resident_2001"
        }
    )

    # =====================================================
    # A. 2000 age total vs 2001 resident
    # =====================================================

    pop_compare = age_2000.merge(
        population_2001,
        on="城市名称",
        how="inner",
        validate="one_to_one",
    )

    pop_compare[
        "ratio_age2000_to_pop2001"
    ] = (
        pop_compare[
            "age_total_2000"
        ]
        / pop_compare[
            "resident_2001"
        ]
    )

    pop_controls = pop_compare[
        ~pop_compare[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
    ].copy()

    pop_suspicious = pop_compare[
        pop_compare[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
    ].copy()

    pop_low = (
        pop_controls[
            "ratio_age2000_to_pop2001"
        ].min()
    )

    pop_high = (
        pop_controls[
            "ratio_age2000_to_pop2001"
        ].max()
    )

    print(
        "\n=== CONTROL ENVELOPE A ==="
    )

    print(
        "age_total_2000 / "
        "resident_2001"
    )

    print(
        f"control_cities = "
        f"{len(pop_controls)}"
    )

    print(
        f"control range = "
        f"[{pop_low:.3f}, "
        f"{pop_high:.3f}]"
    )

    print(
        "\nsuspicious:"
    )

    print(
        pop_suspicious[
            [
                "城市名称",
                "age_total_2000",
                "resident_2001",
                "ratio_age2000_to_pop2001",
            ]
        ]
        .sort_values(
            "ratio_age2000_to_pop2001"
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # B. 2000 age total vs 2010 age total
    # =====================================================

    age_compare = age_2000.merge(
        age_2010,
        on="城市名称",
        how="inner",
        validate="one_to_one",
    )

    age_compare[
        "ratio_age2000_to_age2010"
    ] = (
        age_compare[
            "age_total_2000"
        ]
        / age_compare[
            "age_total_2010"
        ]
    )

    age_controls = age_compare[
        ~age_compare[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
    ].copy()

    age_suspicious = age_compare[
        age_compare[
            "城市名称"
        ].isin(
            KNOWN_SUSPICIOUS_2000
        )
    ].copy()

    age_low = (
        age_controls[
            "ratio_age2000_to_age2010"
        ].min()
    )

    age_high = (
        age_controls[
            "ratio_age2000_to_age2010"
        ].max()
    )

    print(
        "\n=== CONTROL ENVELOPE B ==="
    )

    print(
        "age_total_2000 / "
        "age_total_2010"
    )

    print(
        f"control_cities = "
        f"{len(age_controls)}"
    )

    print(
        f"control range = "
        f"[{age_low:.3f}, "
        f"{age_high:.3f}]"
    )

    print(
        "\nsuspicious:"
    )

    print(
        age_suspicious[
            [
                "城市名称",
                "age_total_2000",
                "age_total_2010",
                "ratio_age2000_to_age2010",
            ]
        ]
        .sort_values(
            "ratio_age2000_to_age2010"
        )
        .to_string(
            index=False
        )
    )

    # =====================================================
    # C. Check whether all 6 lie outside both envelopes
    # =====================================================

    pop_suspicious[
        "outside_pop_envelope"
    ] = (
        (
            pop_suspicious[
                "ratio_age2000_to_pop2001"
            ]
            < pop_low
        )
        |
        (
            pop_suspicious[
                "ratio_age2000_to_pop2001"
            ]
            > pop_high
        )
    )

    age_suspicious[
        "outside_age_envelope"
    ] = (
        (
            age_suspicious[
                "ratio_age2000_to_age2010"
            ]
            < age_low
        )
        |
        (
            age_suspicious[
                "ratio_age2000_to_age2010"
            ]
            > age_high
        )
    )

    combined = (
        pop_suspicious[
            [
                "城市名称",
                "outside_pop_envelope",
            ]
        ]
        .merge(
            age_suspicious[
                [
                    "城市名称",
                    "outside_age_envelope",
                ]
            ],
            on="城市名称",
            how="inner",
            validate="one_to_one",
        )
    )

    print(
        "\n=== ENVELOPE SUMMARY ==="
    )

    print(
        combined.to_string(
            index=False
        )
    )

    assert len(combined) == 6

    assert (
        combined[
            "outside_pop_envelope"
        ].all()
    )

    assert (
        combined[
            "outside_age_envelope"
        ].all()
    )
AGE_2000_INVALID_VALUES = {
    ("city3", 2000): {
        "0-14": 66,
        "15-64": 174,
        "65+": 18,
    },
    ("city16", 2000): {
        "0-14": 160,
        "15-64": 745,
        "65+": 60,
    },
    ("city23", 2000): {
        "0-14": 60,
        "15-64": 629,
        "65+": 9,
    },
    ("city27", 2000): {
        "0-14": 21,
        "15-64": 98,
        "65+": 5,
    },
    ("city35", 2000): {
        "0-14": 89,
        "15-64": 263,
        "65+": 34,
    },
    ("city38", 2000): {
        "0-14": 189,
        "15-64": 279,
        "65+": 40,
    },
}
def build_age_structure_v2(
    normalized_df,
):
    """
    第二轮 cross-table audit：

    6 条 2000 年 observation 已被确认不可信。

    证据来自：
    1. 与同城 2001 常住人口严重冲突；
    2. 与同城 2010 年年龄总量严重冲突；
    3. 6/6 同时落在两个 control envelope 之外；
    4. cross-city relabel 候选并不形成一致规则；
    5. raw source layout 不支持统一 row/label shift。

    因此：
        3 个年龄 component 全部 -> NA
        原始值 -> quarantine

    不 relabel，不猜测正确 component。
    """

    df = normalized_df.copy()

    for column in AGE_COLUMNS:
        df[column] = (
            df[column]
            .astype("Float64")
        )

    quarantine_rows = []

    for (
        city,
        year,
    ), expected_values in (
        AGE_2000_INVALID_VALUES.items()
    ):
        mask = (
            (df["城市名称"] == city)
            & (df["年份"] == year)
        )

        assert mask.sum() == 1

        # =========================================
        # Fail-closed source-value verification
        # =========================================

        for column in AGE_COLUMNS:
            actual = df.loc[
                mask,
                column,
            ].iloc[0]

            assert (
                actual
                == expected_values[column]
            ), (
                f"{city}-{year} "
                f"{column}: "
                f"expected "
                f"{expected_values[column]}, "
                f"got {actual}"
            )

        quarantine_rows.append(
            {
                "城市名称": city,
                "年份": year,

                "raw_0-14":
                    expected_values[
                        "0-14"
                    ],

                "raw_15-64":
                    expected_values[
                        "15-64"
                    ],

                "raw_65+":
                    expected_values[
                        "65+"
                    ],

                "reason":
                    (
                        "high-confidence invalid "
                        "2000 age-structure observation; "
                        "inconsistent with resident "
                        "population and later age totals; "
                        "relabel target unresolved"
                    ),

                "action":
                    (
                        "invalidate all age "
                        "components to NA"
                    ),
            }
        )

        df.loc[
            mask,
            AGE_COLUMNS,
        ] = pd.NA

    quarantine = pd.DataFrame(
        quarantine_rows
    )

    return df, quarantine
def verify_age_structure_v2(
    age_v2,
    quarantine,
):
    print(
        "\n=== AGE STRUCTURE V2 "
        "INVALIDATION VERIFICATION ==="
    )

    duplicate_rows = (
        age_v2
        .duplicated(
            ["城市名称", "年份"],
            keep=False,
        )
        .sum()
    )

    print(
        f"rows = {len(age_v2)}"
    )

    print(
        f"unique_keys = "
        f"{age_v2[['城市名称', '年份']].drop_duplicates().shape[0]}"
    )

    print(
        f"duplicate_rows = "
        f"{duplicate_rows}"
    )

    print(
        f"cities = "
        f"{age_v2['城市名称'].nunique()}"
    )

    print(
        "\n=== MISSING CELLS ==="
    )

    print(
        age_v2[
            AGE_COLUMNS
        ]
        .isna()
        .sum()
        .to_string()
    )

    print(
        f"\nquarantine_rows = "
        f"{len(quarantine)}"
    )

    # =============================================
    # Structure
    # =============================================

    assert len(age_v2) == 118
    assert duplicate_rows == 0
    assert (
        age_v2[
            "城市名称"
        ].nunique()
        == 40
    )

    # 第一轮：
    # 0-14  = 0 missing
    # 15-64 = 20 missing
    # 65+   = 15 missing
    #
    # 新增 6 个完整 observation × 3 components。
    expected_missing = {
        "0-14": 6,
        "15-64": 26,
        "65+": 21,
    }

    actual_missing = (
        age_v2[
            AGE_COLUMNS
        ]
        .isna()
        .sum()
        .to_dict()
    )

    assert (
        actual_missing
        == expected_missing
    )

    assert len(quarantine) == 6

    # =============================================
    # Six invalidated observations
    # =============================================

    invalidated_keys = set()

    for (
        city,
        year,
    ) in AGE_2000_INVALID_VALUES:
        row = age_v2[
            (age_v2["城市名称"] == city)
            & (age_v2["年份"] == year)
        ]

        assert len(row) == 1

        assert (
            row[
                AGE_COLUMNS
            ]
            .isna()
            .all(axis=None)
        )

        invalidated_keys.add(
            (city, year)
        )

    print(
        "\n=== INVALIDATED KEYS ==="
    )

    print(
        sorted(
            invalidated_keys
        )
    )

    # =============================================
    # Original key-level absence must remain absence
    # =============================================

    expected_grid = {
        (f"city{i}", year)
        for i in range(1, 41)
        for year in [
            2000,
            2010,
            2020,
        ]
    }

    observed_keys = set(
        zip(
            age_v2[
                "城市名称"
            ],
            age_v2[
                "年份"
            ],
        )
    )

    missing_keys = (
        expected_grid
        - observed_keys
    )

    print(
        "\n=== MISSING KEYS ==="
    )

    print(
        sorted(
            missing_keys
        )
    )

    assert missing_keys == {
        ("city25", 2000),
        ("city26", 2000),
    }

    print(
        "\n=== QUARANTINE ==="
    )

    print(
        quarantine
        .sort_values(
            "城市名称"
        )
        .to_string(
            index=False
        )
    )
def save_age_structure_v2(
    age_v2,
    quarantine,
):
    normalized_path = (
        NORMALIZED_DIR
        / "年龄结构.csv"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "年龄结构_quarantine.csv"
    )

    age_v2.to_csv(
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
        f"{age_v2.shape}"
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
    raw_df = load_age_structure()

    # =============================================
    # 1. Raw audit
    # =============================================

    print("===== RAW AUDIT =====")

    check_schema(raw_df)
    check_key_and_coverage(raw_df)
    check_missing_structure(raw_df)
    check_value_domain(raw_df)
    check_known_temporal_anomalies(
        raw_df
    )

    # =============================================
    # 2. Base canonicalization
    # =============================================

    normalized_df = (
        normalize_age_structure(
            raw_df
        )
    )

    # 验证忠实读取：
    # 此时仍然是原始 35 个 cell NaN，
    # 六个 2000 anomaly 仍存在。
    verify_age_structure(
        normalized_df
    )

    # =============================================
    # 3. Second-round cross-table invalidation
    # =============================================

    age_v2, quarantine = (
        build_age_structure_v2(
            normalized_df
        )
    )

    verify_age_structure_v2(
        age_v2,
        quarantine,
    )

    # =============================================
    # 4. Save final v2
    # =============================================

    save_age_structure_v2(
        age_v2,
        quarantine,
    )


if __name__ == "__main__":
    main()
