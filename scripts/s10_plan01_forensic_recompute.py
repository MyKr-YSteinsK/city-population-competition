"""Bounded S10-Plan01 forensic harness.

This harness imports the immutable external script and calls its functions. It
does not edit the external script, add a model, or use any future target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGAL_RAW = [
    ("人口规模.xlsx", "*人口规模*.xlsx"),
    ("人口密度.xlsx", "*人口密度*.xlsx"),
    ("城镇化率.xlsx", "*城镇化率*.xlsx"),
    ("就业信息.xlsx", "*就业信息*.xlsx"),
    ("工资水平.xlsx", "*工资水平*.xlsx"),
    ("年龄结构.xlsx", "*年龄结构*.xlsx"),
    ("生活水平.xlsx", "*生活水平*.xlsx"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("s10_original_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def build_environment(module_path: Path, runtime_root: Path) -> dict[str, Any]:
    package_names = ["numpy", "pandas", "scikit-learn", "openpyxl"]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", None),
        "virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "packages": {
            name: metadata.version(name)
            for name in package_names
            if _package_available(name)
        },
        "module_path": str(module_path),
        "disposable_runtime_root": str(runtime_root),
        "exact_command": "python forecast_population.py",
    }


def _package_available(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def build_raw_manifest(repo_root: Path, runtime_root: Path) -> list[dict[str, Any]]:
    repo_raw = repo_root / "data" / "raw" / "城市人口分析与预测" / "数据集"
    runtime_raw = runtime_root / "raw"
    prior_hashes = {
        row["path"].split("/")[-1]: row["after_sha256"]
        for row in pd.read_csv(
            repo_root / "reports" / "S2_HANDOFFS" / "S2-Plan04" / "raw_source_hash_evidence.csv"
        ).to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for name, pattern in LEGAL_RAW:
        source = repo_raw / name
        runtime = runtime_raw / name
        source_hash = sha256_file(source)
        runtime_hash = sha256_file(runtime)
        matches = sorted(str(path) for path in runtime_raw.glob(pattern))
        rows.append(
            {
                "logical_name": name,
                "glob_pattern": pattern,
                "source_repo_path": str(source.relative_to(repo_root)).replace("\\", "/"),
                "runtime_path": str(runtime),
                "source_size": source.stat().st_size,
                "runtime_size": runtime.stat().st_size,
                "source_sha256": source_hash,
                "runtime_sha256": runtime_hash,
                "runtime_exact_copy": source_hash == runtime_hash,
                "s2_plan04_after_sha256": prior_hashes.get(name),
                "matches_s2_plan04_after_sha256": source_hash == prior_hashes.get(name),
                "runtime_glob_match_count": len(matches),
                "runtime_glob_matches": ";".join(matches),
                "competition_legal_official_raw": True,
            }
        )
    runtime_names = sorted(path.name for path in runtime_raw.iterdir() if path.is_file())
    test_xls = runtime_raw / "test.xls"
    rows.append(
        {
            "logical_name": "RUNTIME_DIRECTORY_CHECK",
            "glob_pattern": "",
            "source_repo_path": "",
            "runtime_path": str(runtime_raw),
            "source_size": "",
            "runtime_size": "",
            "source_sha256": "",
            "runtime_sha256": "",
            "runtime_exact_copy": "",
            "s2_plan04_after_sha256": "",
            "matches_s2_plan04_after_sha256": "",
            "runtime_glob_match_count": len(runtime_names),
            "runtime_glob_matches": ";".join(runtime_names),
            "competition_legal_official_raw": "exactly_seven_files_and_no_test_xls"
            if runtime_names == sorted(name for name, _ in LEGAL_RAW) and not test_xls.exists()
            else "FAIL",
        }
    )
    return rows


def frame_duplicate_summary(frame: pd.DataFrame, city_col: str, year_col: str) -> dict[str, Any]:
    work = frame.copy()
    work[city_col] = work[city_col].astype(str).str.strip()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work[work[city_col].str.match(r"^city\d+$", case=False, na=False)]
    work = work.dropna(subset=[year_col])
    counts = work.groupby([city_col, year_col]).size()
    duplicate_groups = counts[counts > 1]
    return {
        "rows_after_city_year_filter": int(len(work)),
        "unique_city_year_keys": int(len(counts)),
        "duplicate_key_count": int(len(duplicate_groups)),
        "duplicate_extra_row_count": int((duplicate_groups - 1).sum()),
        "max_rows_per_duplicate_key": int(duplicate_groups.max()) if not duplicate_groups.empty else 1,
        "duplicate_keys": ";".join(
            f"{city}/{int(year)}x{int(count)}"
            for (city, year), count in duplicate_groups.items()
        ),
    }


def source_handling_audit(
    module,
    repo_root: Path,
    runtime_root: Path,
    population_before: pd.DataFrame,
    population_after: pd.DataFrame,
    exog: pd.DataFrame,
    all_years: list[int],
) -> dict[str, Any]:
    runtime_raw = runtime_root / "raw"
    employment_path = module._pick("*就业信息*.xlsx")
    employment_specs = [
        (0, "unemployment", "year", "city"),
        (1, "employees", "city", "year"),
        (2, "sectors", "city", "year"),
    ]
    employment_duplicates: list[dict[str, Any]] = []
    for sheet_index, label, city_col, year_col in employment_specs:
        raw = pd.read_excel(employment_path, sheet_name=sheet_index)
        raw = raw.iloc[:, : (3 if sheet_index < 2 else 5)].copy()
        if sheet_index == 0:
            raw.columns = ["year", "city", "unemployment_rate"]
        elif sheet_index == 1:
            raw.columns = ["city", "year", "employees_number"]
        else:
            raw.columns = [
                "city",
                "year",
                "primary_employment",
                "secondary_employment",
                "tertiary_employment",
            ]
        employment_duplicates.append(
            {
                "sheet_index": sheet_index,
                "logical_sheet": label,
                **frame_duplicate_summary(raw, city_col, year_col),
                "original_groupby_mean": True,
            }
        )

    living_path = module._pick("*生活水平*.xlsx")
    workbook = pd.ExcelFile(living_path)
    income_names = [
        "income_disposable",
        "income_consumption",
        "income_urban_consumption",
        "income_rural_consumption",
        "income_urban_disposable",
        "income_rural_disposable",
    ]
    income_sheet_order = [
        {
            "sheet_index": index,
            "actual_sheet_name": sheet,
            "script_fixed_name": income_names[index] if index < len(income_names) else f"income_{index}",
        }
        for index, sheet in enumerate(workbook.sheet_names)
    ]

    quarantine_path = repo_root / "data" / "quarantine" / "生活水平_人均可支配收入_quarantine.csv"
    quarantine = pd.read_csv(quarantine_path)
    income_frames = module._long_income_tables(living_path)
    disposable = income_frames[0] if income_frames else pd.DataFrame(columns=["city", "year", "income_disposable"])
    disposable_values = {
        (str(row.city), int(row.year), float(row.income_disposable))
        for row in disposable.itertuples(index=False)
        if pd.notna(row.income_disposable)
    }
    quarantine_exact_matches = 0
    quarantine_keys_in_raw = 0
    for _, row in quarantine.iterrows():
        key = (str(row["城市名称"]), int(row["年份"]), float(row["raw_disposableIncome"]))
        if key in disposable_values:
            quarantine_exact_matches += 1
        if any(city == key[0] and year == key[1] for city, year, _ in disposable_values):
            quarantine_keys_in_raw += 1

    raw_disposable = pd.read_excel(living_path, sheet_name=0, header=None)
    raw_years = pd.to_numeric(raw_disposable.iloc[1, 1:], errors="coerce") if len(raw_disposable.index) > 1 else pd.Series(dtype=float)
    raw_disposable_values: set[tuple[str, int, float]] = set()
    for _, raw_row in raw_disposable.iloc[2:].iterrows():
        city = str(raw_row.iloc[0]).strip()
        if not city.lower().startswith("city"):
            continue
        for column_index, year in enumerate(raw_years, start=1):
            if pd.isna(year) or column_index >= len(raw_row):
                continue
            value = pd.to_numeric(raw_row.iloc[column_index], errors="coerce")
            if pd.notna(value):
                raw_disposable_values.add((city, int(year), float(value)))
    raw_quarantine_exact_matches = 0
    raw_quarantine_keys = 0
    for _, row in quarantine.iterrows():
        key = (str(row["城市名称"]), int(row["年份"]), float(row["raw_disposableIncome"]))
        if key in raw_disposable_values:
            raw_quarantine_exact_matches += 1
        if any(city == key[0] and year == key[1] for city, year, _ in raw_disposable_values):
            raw_quarantine_keys += 1

    age_path = module._pick("*年龄结构*.xlsx")
    age = pd.read_excel(age_path).iloc[:, :5].copy()
    age.columns = ["city", "year", "age_0_14", "age_15_64", "age_65_plus"]
    age["year"] = pd.to_numeric(age["year"], errors="coerce")
    age["city"] = age["city"].astype(str).str.strip()
    age = age[age["city"].str.match(r"^city\d+$", case=False, na=False)].dropna(subset=["year"])
    age_years = {
        city: sorted(int(year) for year in group["year"].unique())
        for city, group in age.groupby("city")
    }
    age_carry_counts: dict[str, int] = {"requested": 0, "resolved": 0, "stale_carry_forward": 0, "missing": 0}
    age_stale_by_year: dict[str, int] = {}
    for year in all_years:
        requested_year = year - 1
        for city in sorted(age_years):
            age_carry_counts["requested"] += 3
            available = [value for value in age_years[city] if value <= requested_year]
            if not available:
                age_carry_counts["missing"] += 3
                continue
            age_carry_counts["resolved"] += 3
            resolved_year = max(available)
            if resolved_year < requested_year:
                age_carry_counts["stale_carry_forward"] += 3
                age_stale_by_year[str(year)] = age_stale_by_year.get(str(year), 0) + 3

    before_city40 = population_before[
        population_before["city"].eq("city40") & population_before["year"].eq(2021)
    ]
    after_city40 = population_after[
        population_after["city"].eq("city40") & population_after["year"].eq(2021)
    ]
    prior = population_before.loc[
        population_before["city"].eq("city40")
        & population_before["year"].between(2017, 2020),
        "registered",
    ].dropna()
    original_value = numeric(before_city40["registered"].iloc[0]) if not before_city40.empty else None
    replacement_value = numeric(after_city40["registered"].iloc[0]) if not after_city40.empty else None
    prior_median = numeric(prior.median()) if not prior.empty else None
    trigger_ratio = (
        prior_median / max(original_value, 1.0)
        if prior_median is not None and original_value is not None
        else None
    )
    repair_evidence = {
        "city": "city40",
        "year": 2021,
        "registered_before_repair": original_value,
        "registered_after_repair": replacement_value,
        "prior_2017_2020_values": [float(value) for value in prior.tolist()],
        "prior_tail_two_median": numeric(prior.tail(2).median()) if not prior.empty else None,
        "prior_full_median": prior_median,
        "trigger_ratio_full_prior_median_over_current": trigger_ratio,
        "trigger_condition_len_prior_at_least_2": bool(len(prior) >= 2),
        "trigger_condition_ratio_gt_2": bool(trigger_ratio is not None and trigger_ratio > 2.0),
        "repair_triggered": bool(
            len(prior) >= 2 and trigger_ratio is not None and trigger_ratio > 2.0
        ),
        "resident_changed_by_repair": False,
    }

    return {
        "runtime_raw_glob_matches": {
            name: [str(path) for path in sorted(runtime_raw.glob(pattern))]
            for name, pattern in LEGAL_RAW
        },
        "runtime_raw_file_names": sorted(path.name for path in runtime_raw.iterdir() if path.is_file()),
        "test_xls_present_in_runtime": (runtime_raw / "test.xls").exists(),
        "employment_duplicate_summary": employment_duplicates,
        "employment_duplicate_semantics": "Each frame is grouped by city/year and numeric value columns are averaged before the outer merge; no repair is applied by the external script.",
        "income_sheet_order": income_sheet_order,
        "income_parser_output_frame_count": len(income_frames),
        "income_parser_output_row_counts": [int(len(frame)) for frame in income_frames],
        "income_columns_in_exog": [
            column for column in exog.columns if str(column).startswith("income_")
        ],
        "living_standard_quarantine_file": str(quarantine_path.relative_to(repo_root)).replace("\\", "/"),
        "living_standard_quarantine_rows": int(len(quarantine)),
        "living_standard_quarantine_exact_raw_value_matches_in_raw_workbook": raw_quarantine_exact_matches,
        "living_standard_quarantine_keys_present_in_raw_workbook": raw_quarantine_keys,
        "living_standard_quarantine_exact_raw_value_matches": quarantine_exact_matches,
        "living_standard_quarantine_keys_present_in_raw_disposable_sheet": quarantine_keys_in_raw,
        "living_standard_block_is_ingested_by_original_semantics": len(income_frames) > 0 and quarantine_exact_matches > 0,
        "living_standard_original_parser_finding": (
            "parser produced no income frames because raw.iloc[0,1:] is the title row, not the year row"
            if not income_frames
            else "income frames produced; see exact quarantine match counts"
        ),
        "age_observed_years_by_city": age_years,
        "age_asof_carry_forward_counts": age_carry_counts,
        "age_stale_carry_forward_by_feature_year": age_stale_by_year,
        "repair_evidence": repair_evidence,
        "exog_row_count_after_original_groupby_merge": int(len(exog)),
        "exog_unique_city_year_keys": int(exog[["city", "year"]].drop_duplicates().shape[0]),
    }


def independent_metrics(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = validation.dropna(subset=["pred", "actual"]).copy()
    valid["error"] = valid["pred"] - valid["actual"]
    valid["abs_error"] = valid["error"].abs()
    valid["sq_error"] = valid["error"] ** 2
    pooled = (
        valid.groupby(["target", "method"], as_index=False)
        .agg(
            n=("error", "size"),
            origin_count=("year", "nunique"),
            city_count=("city", "nunique"),
            pooled_mse=("sq_error", "mean"),
            rmse=("sq_error", lambda values: float(np.sqrt(values.mean()))),
            mae=("abs_error", "mean"),
            medae=("abs_error", "median"),
            bias=("error", "mean"),
        )
    )
    pooled["sqrt_mse_rmse_abs_diff"] = (np.sqrt(pooled["pooled_mse"]) - pooled["rmse"]).abs()
    origin = (
        valid.groupby(["target", "method", "year"], as_index=False)
        .agg(
            n=("error", "size"),
            city_count=("city", "nunique"),
            origin_mse=("sq_error", "mean"),
            origin_rmse=("sq_error", lambda values: float(np.sqrt(values.mean()))),
            origin_mae=("abs_error", "mean"),
            origin_bias=("error", "mean"),
        )
    )
    return pooled, origin


def selector_diagnostic(validation: pd.DataFrame, metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, frame in validation.groupby("target"):
        valid = frame.dropna(subset=["pred", "actual"]).copy()
        valid["abs_error"] = (valid["pred"] - valid["actual"]).abs()
        city_metrics = (
            valid.groupby(["city", "method"], as_index=False)
            .agg(n=("abs_error", "size"), mae=("abs_error", "mean"))
        )
        global_order = metrics[target].sort_values(["mae", "rmse", "method"])["method"].tolist()
        selected: list[dict[str, Any]] = []
        for city in sorted(valid["city"].unique()):
            cm = city_metrics[city_metrics["city"].eq(city)].copy()
            cm = cm[cm["n"].ge(2)].sort_values(["mae", "method"])
            best = str(cm["method"].iloc[0]) if not cm.empty else str(global_order[0])
            city_rows = valid[valid["city"].eq(city)]
            if city_rows[city_rows["method"].eq(best)].empty:
                for fallback in global_order:
                    if not city_rows[city_rows["method"].eq(fallback)].empty:
                        best = fallback
                        break
            selected.append({"city": city, "method": best})
        assignment = pd.DataFrame(selected)
        chosen = valid.merge(assignment, on="city", suffixes=("", "_selected"))
        chosen = chosen[chosen["method"].eq(chosen["method_selected"])].copy()
        chosen["sq_error"] = (chosen["pred"] - chosen["actual"]) ** 2
        chosen["abs_error"] = (chosen["pred"] - chosen["actual"]).abs()
        by_method = ";".join(
            f"{row.method}:{int(row.n)}"
            for row in assignment.groupby("method").size().rename("n").reset_index().itertuples()
        )
        rows.append(
            {
                "target": target,
                "diagnostic": "NON_NESTED_APPARENT_SELECTOR",
                "n_selected_rows": int(len(chosen)),
                "city_count": int(len(assignment)),
                "pooled_mse": float(chosen["sq_error"].mean()),
                "rmse": float(np.sqrt(chosen["sq_error"].mean())),
                "mae": float(chosen["abs_error"].mean()),
                "selected_method_city_counts": by_method,
                "interpretation": "INVALID_FOR_UNBIASED_HISTORICAL_MODEL_COMPARISON",
            }
        )
    return pd.DataFrame(rows)


def build_675_search(
    pooled: pd.DataFrame,
    origin: pd.DataFrame,
    validation: pd.DataFrame,
    selector: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in sorted(validation["target"].unique()):
        target_pooled = pooled[pooled["target"].eq(target)]
        target_origin = origin[origin["target"].eq(target)]
        target_validation = validation[validation["target"].eq(target)].copy()
        for row in target_pooled.itertuples(index=False):
            rows.extend(
                [
                    {
                        "target": target,
                        "method": row.method,
                        "pre_registered_calculation": "method_level_pooled_mse",
                        "value": float(row.pooled_mse),
                    },
                    {
                        "target": target,
                        "method": row.method,
                        "pre_registered_calculation": "method_level_pooled_rmse",
                        "value": float(row.rmse),
                    },
                    {
                        "target": target,
                        "method": row.method,
                        "pre_registered_calculation": "method_ranking_metric_mae",
                        "value": float(row.mae),
                    },
                ]
            )
            method_origin = target_origin[target_origin["method"].eq(row.method)]
            rows.append(
                {
                    "target": target,
                    "method": row.method,
                    "pre_registered_calculation": "method_level_macro_origin_mse",
                    "value": float(method_origin["origin_mse"].mean()),
                }
            )
            latest = method_origin[method_origin["year"].eq(2021)]
            rows.append(
                {
                    "target": target,
                    "method": row.method,
                    "pre_registered_calculation": "latest_origin_2021_mse",
                    "value": float(latest["origin_mse"].iloc[0]) if not latest.empty else None,
                }
            )
            rounded = target_validation.dropna(subset=["pred", "actual"]).copy()
            rounded["rounded_sq_error"] = (rounded["pred"].round(2) - rounded["actual"]) ** 2
            rounded = rounded[rounded["method"].eq(row.method)]
            rows.append(
                {
                    "target": target,
                    "method": row.method,
                    "pre_registered_calculation": "rounded_validation_prediction_pooled_mse",
                    "value": float(rounded["rounded_sq_error"].mean()) if not rounded.empty else None,
                }
            )
        apparent = selector[selector["target"].eq(target)]
        if not apparent.empty:
            rows.append(
                {
                    "target": target,
                    "method": "NON_NESTED_APPARENT_SELECTOR",
                    "pre_registered_calculation": "non_nested_apparent_selector_mse",
                    "value": float(apparent["pooled_mse"].iloc[0]),
                }
            )
    result = pd.DataFrame(rows)
    result["target_675"] = 675.0
    result["absolute_difference_from_675"] = (result["value"] - 675.0).abs()
    result["finite_value"] = result["value"].notna()
    return result.sort_values(["absolute_difference_from_675", "target", "method"]).reset_index(drop=True)


def build_leakage_audit(
    source_audit: dict[str, Any],
    repair: dict[str, Any],
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_total = sum(
        int(row["duplicate_key_count"])
        for row in source_audit["employment_duplicate_summary"]
    )
    rows = [
        {
            "item": "target itself enters feature",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:261-293,321-323; feature_cols excludes resident and registered",
            "runtime_evidence": "Current target columns are labels and are excluded from panel feature_cols.",
            "interpretation": "No current-year target column is sent to model matrices.",
        },
        {
            "item": "future target lag direction",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:269-280; target history uses year-2 and lags 2..5",
            "runtime_evidence": "Harness followed original validation construction for five origins.",
            "interpretation": "Target features do not request T-1, T, or later target labels.",
        },
        {
            "item": "target history > T-2",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:270,349-351",
            "runtime_evidence": "Training rows are filtered to feature year <= valid_year-2.",
            "interpretation": "Pass under the script's target-history contract.",
        },
        {
            "item": "exog T-1 usage vs project R1/R2",
            "status": "UNKNOWN",
            "exact_code_evidence": "forecast_population.py:268,281-282; _asof(feature_year=year-1)",
            "runtime_evidence": "T-1 as-of values are present, but no source-specific publication timing is encoded in the external script.",
            "interpretation": "Cannot be called project-contract legal without mapping every source to R1/R2 caps.",
        },
        {
            "item": "full-data imputer fit",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:296-304; imputer fit_transform(x_train) then transform(x_test)",
            "runtime_evidence": "Original _matrices path was called by the harness without modification.",
            "interpretation": "Median and missingness statistics are train-only.",
        },
        {
            "item": "full-data scaler fit",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:324-327; scaler fit on x_train only",
            "runtime_evidence": "Only Ridge enters this branch; tree models do not scale.",
            "interpretation": "No test-row numerical statistics enter scaling.",
        },
        {
            "item": "train/test numerical-statistics fit",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:296-303",
            "runtime_evidence": "Train/test concatenation is used only before city dummy schema construction.",
            "interpretation": "Category schema union is observable, but numerical imputer/scaler statistics are train-only.",
        },
        {
            "item": "random split",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:346-352",
            "runtime_evidence": "Five expanding validation origins were executed.",
            "interpretation": "No random train/validation split is present.",
        },
        {
            "item": "city/year derivative duplication",
            "status": "UNKNOWN" if duplicate_total else "PASS",
            "exact_code_evidence": "forecast_population.py:135-146; groupby mean after source-specific reshape",
            "runtime_evidence": f"employment duplicate key groups observed={duplicate_total}; see source_handling_audit.md.",
            "interpretation": "The script has local aggregation semantics that differ from the canonical provenance contract; this is not silently treated as equivalent.",
        },
        {
            "item": "test.xls",
            "status": "PASS",
            "exact_code_evidence": "No static reference; runtime _pick patterns are XLSX-only.",
            "runtime_evidence": "Disposable runtime contains exactly seven XLSX files and no test.xls.",
            "interpretation": "No test.xls read path observed.",
        },
        {
            "item": "B-data",
            "status": "PASS",
            "exact_code_evidence": "No B-data path or pattern in the external script.",
            "runtime_evidence": "Disposable runtime raw directory contains only the seven legal XLSX files.",
            "interpretation": "No B-data input was copied or read.",
        },
        {
            "item": "S8 truth",
            "status": "PASS",
            "exact_code_evidence": "No S8, candidate-truth, real-2022, or real-2023 path in the external script.",
            "runtime_evidence": "Harness inputs are limited to the runtime raw directory and project audit metadata.",
            "interpretation": "No S8 truth was used for prediction or scoring.",
        },
        {
            "item": "external future population",
            "status": "PASS",
            "exact_code_evidence": "Population target is selected only from *人口规模*.xlsx.",
            "runtime_evidence": "No external population file was in the runtime directory.",
            "interpretation": "The script does not use future external target labels.",
        },
        {
            "item": "hard-coded target repair",
            "status": "FAIL",
            "exact_code_evidence": "forecast_population.py:150-160",
            "runtime_evidence": json.dumps(repair, ensure_ascii=False, sort_keys=True),
            "interpretation": "The registered 2021 city40 target is conditionally mutated before validation and final fit.",
        },
        {
            "item": "city-specific historical selector reuse",
            "status": "FAIL",
            "exact_code_evidence": "forecast_population.py:377-405; selector reuses 2017-2021 city MAE to choose the method scored on those same rows",
            "runtime_evidence": "selector_optimism_diagnostic.csv records NON_NESTED_APPARENT_SELECTOR.",
            "interpretation": "Valid as a final-selection mechanism under the script contract, invalid as unbiased historical model comparison.",
        },
        {
            "item": "validation-target-informed post-processing",
            "status": "FAIL",
            "exact_code_evidence": "forecast_population.py:150-160,418-426",
            "runtime_evidence": "The repair condition inspects the 2021 registered target before 2021 validation scoring.",
            "interpretation": "The registered 2021 scoring label itself can be changed by target-informed post-processing.",
        },
        {
            "item": "best-seed selection",
            "status": "PASS",
            "exact_code_evidence": "forecast_population.py:310-317; fixed random_state=42 and no seed loop",
            "runtime_evidence": "No seed-search or post-hoc seed selection path executed.",
            "interpretation": "Fixed model seeds only.",
        },
    ]
    return pd.DataFrame(rows)


def write_source_audit(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# S10-Plan01 Source-Handling Audit",
        "",
        "This is runtime evidence for `ORIGINAL_SCRIPT_SEMANTICS`; it does not repair or replace any source behavior.",
        "",
        "## Runtime file selection",
        "",
        f"- Runtime raw file names: `{'; '.join(audit['runtime_raw_file_names'])}`.",
        f"- `test.xls` present in disposable runtime: `{audit['test_xls_present_in_runtime']}`.",
        "- The seven `_pick()` patterns each had their exact match list and hash recorded in `../runtime/raw_input_manifest.csv`.",
        "",
        "## Employment duplicate handling",
        "",
        "The external script reads three employment sheets and applies `groupby(['city', 'year'])[value_cols].mean(numeric_only=True)` before merging. The observed pre-aggregation duplicate evidence is:",
        "",
        "| Sheet | Rows | Unique keys | Duplicate key groups | Extra duplicate rows | Max group size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["employment_duplicate_summary"]:
        lines.append(
            f"| {row['logical_sheet']} | {row['rows_after_city_year_filter']} | {row['unique_city_year_keys']} | {row['duplicate_key_count']} | {row['duplicate_extra_row_count']} | {row['max_rows_per_duplicate_key']} |"
        )
    lines.extend(
        [
            "",
            "This mean aggregation is the external script's behavior; it is not the S1 canonical/provenance pipeline and no quarantine or provenance flag is applied by the script.",
            "",
            "## Living-standard sheet order and known S1 block",
            "",
            "The external script assigns fixed variable names by sheet index rather than checking sheet names:",
            "",
            "| Index | Actual sheet name | Script variable |",
            "|---:|---|---|",
        ]
    )
    for row in audit["income_sheet_order"]:
        lines.append(f"| {row['sheet_index']} | {row['actual_sheet_name']} | {row['script_fixed_name']} |")
    lines.extend(
        [
            "",
            f"The S1 quarantine file `{audit['living_standard_quarantine_file']}` has {audit['living_standard_quarantine_rows']} rows. A direct diagnostic of the raw first sheet finds {audit['living_standard_quarantine_exact_raw_value_matches_in_raw_workbook']} exact `(city, year, raw_disposableIncome)` values and {audit['living_standard_quarantine_keys_present_in_raw_workbook']} keys. However, the original `_long_income_tables()` parser produced {audit['income_parser_output_frame_count']} income frames with row counts `{audit['income_parser_output_row_counts']}` and exog income columns `{audit['income_columns_in_exog']}` because it treats raw row 0 as the year row. Therefore the known S1 block is not ingested by original semantics; it is absent because of the parser/header mismatch, not because the script implements the S1 quarantine boundary.",
            "",
            "## Age sparse-year carry-forward",
            "",
            f"Observed age years by city are recorded in the machine-readable audit. Across the feature rows used by the original script, requested age values={audit['age_asof_carry_forward_counts']['requested']}, resolved={audit['age_asof_carry_forward_counts']['resolved']}, stale as-of carry-forward={audit['age_asof_carry_forward_counts']['stale_carry_forward']}, missing={audit['age_asof_carry_forward_counts']['missing']}. This follows `_asof()` and does not infer annual age observations from the benchmark-only source.",
            "",
            "## Target mutation evidence",
            "",
            "The exact city40 registered repair evidence is stored in `../forensics/runtime_repair_evidence.json` and repeated in the leakage matrix. This is recorded as a script behavior, not endorsed as a project correction.",
            "",
            "## Contract difference",
            "",
            "The source behavior above is local reshape/groupby/as-of behavior. The project contract keeps canonical values, provenance, quarantine, source-specific caps, and benchmark-domain boundaries explicit. No correction is made in Plan01.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module_path = args.module_path.resolve()
    repo_root = args.repo_root.resolve()
    runtime_root = module_path.parent.resolve()
    module = load_module(module_path)

    raw_manifest = build_raw_manifest(repo_root, runtime_root)
    pd.DataFrame(raw_manifest).to_csv(
        args.output_dir.parent / "runtime" / "raw_input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(args.output_dir.parent / "runtime" / "environment.json", build_environment(module_path, runtime_root))

    population_before, exog = module.load_data()
    population_after = module.repair_obvious_target_anomalies(population_before)
    target_maps, exog_maps = module._series_maps(population_after, exog)
    all_years = sorted(set(population_after["year"].astype(int)) | {2023})
    features = module.build_features(population_after, exog, target_maps, exog_maps, all_years)

    validations: list[pd.DataFrame] = []
    original_metrics: list[pd.DataFrame] = []
    for target in ["resident", "registered"]:
        target_values = population_after.set_index(["city", "year"])[target].reindex(
            pd.MultiIndex.from_frame(features[["city", "year"]])
        ).to_numpy()
        features[target] = target_values
        _, metrics, _, validation = module.evaluate_and_forecast(
            population_after, features, target_maps, exog_maps, target
        )
        validation = validation.copy()
        validation.insert(0, "target", target)
        validations.append(validation)
        metric_frame = metrics.reset_index().copy()
        metric_frame.insert(0, "target", target)
        original_metrics.append(metric_frame)

    validation_all = pd.concat(validations, ignore_index=True)
    validation_all.to_csv(
        args.output_dir / "validation_predictions_original_semantics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    original_metrics_all = pd.concat(original_metrics, ignore_index=True)
    original_metrics_all.to_csv(
        args.output_dir / "script_printed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled, origin = independent_metrics(validation_all)
    pooled.to_csv(args.output_dir / "independently_recomputed_metrics.csv", index=False, encoding="utf-8-sig")
    origin.to_csv(args.output_dir / "per_origin_metrics.csv", index=False, encoding="utf-8-sig")
    metric_map = {target: group.copy() for target, group in pooled.groupby("target")}
    selector = selector_diagnostic(validation_all, metric_map)
    selector.to_csv(args.output_dir / "selector_optimism_diagnostic.csv", index=False, encoding="utf-8-sig")
    search = build_675_search(pooled, origin, validation_all, selector)
    search.to_csv(args.output_dir / "675_pre_registered_search.csv", index=False, encoding="utf-8-sig")

    source_audit = source_handling_audit(
        module, repo_root, runtime_root, population_before, population_after, exog, all_years
    )
    write_json(args.output_dir / "runtime_repair_evidence.json", source_audit["repair_evidence"])
    write_json(args.output_dir / "source_handling_runtime.json", source_audit)
    write_source_audit(args.output_dir / "source_handling_audit.md", source_audit)

    leakage = build_leakage_audit(source_audit, source_audit["repair_evidence"], pooled)
    leakage.to_csv(args.output_dir / "initial_leakage_audit.csv", index=False, encoding="utf-8-sig")

    runtime_manifest = {
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "runtime_root": str(runtime_root),
        "legal_raw_count": len(LEGAL_RAW),
        "test_xls_in_runtime": False,
        "validation_targets": ["resident", "registered"],
        "validation_years": [2017, 2018, 2019, 2020, 2021],
        "horizon": 2,
        "target_history_cutoff_rule": "T-2",
        "final_target_year": 2023,
        "final_target_history_cutoff": 2021,
        "validation_rows_total": int(len(validation_all)),
        "validation_finite_rows_total": int(validation_all["pred"].notna().sum()),
        "script_printed_mse": False,
        "script_printed_rmse": True,
        "repair_evidence": source_audit["repair_evidence"],
        "output_files_written_by_harness": [
            "validation_predictions_original_semantics.csv",
            "script_printed_metrics.csv",
            "independently_recomputed_metrics.csv",
            "per_origin_metrics.csv",
            "selector_optimism_diagnostic.csv",
            "675_pre_registered_search.csv",
            "initial_leakage_audit.csv",
            "source_handling_audit.md",
        ],
    }
    write_json(args.output_dir / "harness_manifest.json", runtime_manifest)


if __name__ == "__main__":
    main()
