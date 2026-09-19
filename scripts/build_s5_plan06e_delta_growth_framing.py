#!/usr/bin/env python3
"""Run the frozen S5 Plan06E target-framing screening.

The worker keeps the Plan04/04B feature and replay contracts fixed while
comparing two target reconstructions with the same CatBoost capacity family:
delta from the latest legal population and log growth from that level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor, Pool


PLAN_ID = "S5-Plan06E"
BASE_SHA = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
FEATURE_CONTRACT = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT = "S5-Plan04B-TRAINING-REPLAY-V1"
SEED = 20260919
DEFAULT_FIT_WORKERS = min(8, os.cpu_count() or 1)
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
FRAMINGS = ("F3_DELTA_FROM_LATEST", "F4_LOG_GROWTH_FROM_LATEST")
MODEL_IDS = {
    "F3_DELTA_FROM_LATEST": "CAT_DELTA_SELECTED",
    "F4_LOG_GROWTH_FROM_LATEST": "CAT_LOG_GROWTH_SELECTED",
}
OUTER_ORIGINS = tuple(range(2011, 2022))
EXPECTED_B3_MSE = {"resident": 872.452651515143, "registered": 225.421212121213}
CONFIGS = (
    {
        "config_id": "FRM_C1_CONSERVATIVE",
        "config_order": 1,
        "iterations": 400,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 20.0,
        "rsm": 0.70,
        "fallback_default": True,
    },
    {
        "config_id": "FRM_C2_MEDIUM",
        "config_order": 2,
        "iterations": 600,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 10.0,
        "rsm": 0.80,
        "fallback_default": False,
    },
    {
        "config_id": "FRM_C3_FLEXIBLE_REGULARIZED",
        "config_order": 3,
        "iterations": 800,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 10.0,
        "rsm": 0.85,
        "fallback_default": False,
    },
)
COMMON_CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 1,
    "random_strength": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=DEFAULT_FIT_WORKERS,
        help="Concurrent independent fits; each CatBoost fit remains thread_count=1.",
    )
    return parser.parse_args()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def assert_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_matrix(
    path: Path,
    model_features: list[str],
    name: str,
    year_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "target_id",
        "scenario_id",
        year_column,
        "city_id",
        "actual_target_value",
        "b3_prediction",
        "b3_residual_actual_minus_prediction",
        "b3_squared_error",
        "pop_latest_level",
        *model_features,
    ]
    assert_columns(frame, required, name)
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} has duplicate columns after CSV parsing: {duplicates}")
    if frame["city_id"].isna().any():
        raise ValueError(f"{name}.city_id contains missing values")
    frame["city_id"] = frame["city_id"].astype(str)
    for column in model_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{name}.{column} contains infinity")
    for column in (
        "actual_target_value",
        "b3_prediction",
        "b3_residual_actual_minus_prediction",
        "b3_squared_error",
        "pop_latest_level",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}.{column} contains non-finite values")
    frame[year_column] = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    return frame


def prepare_geometry(path: Path, year_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in year_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        frame[column] = numeric.astype("Int64") if numeric.isna().any() else numeric.astype(int)
    return frame


def make_pool(frame: pd.DataFrame, model_input_columns: list[str], labels: np.ndarray) -> Pool:
    features = frame.loc[:, model_input_columns].copy()
    features["city_id"] = features["city_id"].astype(str)
    return Pool(features, label=np.asarray(labels, dtype=float), cat_features=["city_id"])


def framing_labels(frame: pd.DataFrame, framing_id: str) -> np.ndarray:
    latest = frame["pop_latest_level"].to_numpy(dtype=float)
    actual = frame["actual_target_value"].to_numpy(dtype=float)
    if (latest <= 0).any():
        raise ValueError(f"{framing_id} encountered non-positive pop_latest_level in labels")
    if (actual <= 0).any():
        raise ValueError(f"{framing_id} encountered non-positive actual target in log labels")
    if framing_id == "F3_DELTA_FROM_LATEST":
        labels = actual - latest
    elif framing_id == "F4_LOG_GROWTH_FROM_LATEST":
        with np.errstate(divide="raise", invalid="raise"):
            labels = np.log(actual / latest)
    else:
        raise ValueError(f"Unknown framing: {framing_id}")
    if not np.isfinite(labels).all():
        raise ValueError(f"{framing_id} generated a non-finite training label")
    return labels


def reconstruct_prediction(
    frame: pd.DataFrame, raw_prediction: np.ndarray, framing_id: str
) -> np.ndarray:
    latest = frame["pop_latest_level"].to_numpy(dtype=float)
    if framing_id == "F3_DELTA_FROM_LATEST":
        prediction = latest + raw_prediction
    elif framing_id == "F4_LOG_GROWTH_FROM_LATEST":
        with np.errstate(over="raise", invalid="raise"):
            prediction = latest * np.exp(raw_prediction)
    else:
        raise ValueError(f"Unknown framing: {framing_id}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{framing_id} reconstruction generated a non-finite prediction")
    return prediction


def fit_and_predict(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    framing_id: str,
    config: dict[str, Any],
    model_input_columns: list[str],
) -> np.ndarray:
    params = dict(COMMON_CATBOOST_PARAMS)
    params.update(
        {
            key: config[key]
            for key in ("iterations", "learning_rate", "depth", "l2_leaf_reg", "rsm")
        }
    )
    try:
        model = CatBoostRegressor(**params)
        model.fit(make_pool(train_frame, model_input_columns, framing_labels(train_frame, framing_id)))
        raw_prediction = np.asarray(
            model.predict(make_pool(predict_frame, model_input_columns, np.zeros(len(predict_frame)))),
            dtype=float,
        )
    except (TypeError, CatBoostError) as exc:
        raise RuntimeError(
            "CatBoost fixed-parameter compatibility failure; no parameter was substituted: "
            f"{exc}"
        ) from exc
    return reconstruct_prediction(predict_frame, raw_prediction, framing_id)


def evaluate_inner_task(
    task: tuple[dict[str, Any], str, list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]
) -> dict[str, Any]:
    config, framing_id, model_input_columns, fold_data, inner_training, inner_validation = task
    prediction = fit_and_predict(
        inner_training, inner_validation, framing_id, config, model_input_columns
    )
    actual = inner_validation["actual_target_value"].to_numpy(dtype=float)
    return {
        "config": config,
        "fold": fold_data,
        "inner_training_row_count_actual": int(len(inner_training)),
        "inner_validation_row_count_actual": int(len(inner_validation)),
        "mse": float(np.mean((prediction - actual) ** 2)),
    }


def feature_group_map(contract: dict[str, Any], model_features: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    flattened: list[str] = []
    for group_id, features in contract["feature_groups"].items():
        for feature in features:
            if feature in mapping:
                raise ValueError(f"Feature appears in multiple groups: {feature}")
            mapping[feature] = group_id
            flattened.append(feature)
    if len(flattened) != len(model_features) or set(flattened) != set(model_features):
        raise ValueError("Feature-group manifest does not cover exactly the 144 model features")
    return mapping


def validate_contract(
    schema: dict[str, Any],
    contract: dict[str, Any],
    formal: pd.DataFrame,
    replay: pd.DataFrame,
    outer_geometry: pd.DataFrame,
    inner_geometry: pd.DataFrame,
    model_features: list[str],
) -> None:
    if schema["feature_contract_version"] != FEATURE_CONTRACT:
        raise ValueError("Unexpected feature contract version")
    if contract["contract_version"] != REPLAY_CONTRACT:
        raise ValueError("Unexpected replay contract version")
    if contract["base_sha"] != "fac7ed9bf11459bc5d7c6772857683417bf70eac":
        raise ValueError("Unexpected Plan04B contract base SHA")
    if len(model_features) != 144 or contract["model_feature_count"] != 144:
        raise ValueError("Frozen model feature count is not 144")
    if len(formal) != 1760 or len(replay) != 2320:
        raise ValueError(f"Frozen matrix row counts changed: formal={len(formal)}, replay={len(replay)}")
    if len(outer_geometry) != 44 or len(inner_geometry) != 484:
        raise ValueError("Frozen geometry row counts changed")
    if sorted(formal["origin_year"].unique().tolist()) != list(OUTER_ORIGINS):
        raise ValueError("Formal outer origin domain changed")
    if sorted(formal["city_id"].unique().tolist()) != sorted(replay["city_id"].unique().tolist()):
        raise ValueError("Formal and replay city vocabularies differ")
    if formal["city_id"].nunique() != 40:
        raise ValueError("Frozen city count changed")
    formal_key = ["target_id", "scenario_id", "origin_year", "city_id"]
    replay_key = ["target_id", "scenario_id", "pseudo_target_year", "city_id"]
    if formal.duplicated(formal_key).any() or replay.duplicated(replay_key).any():
        raise ValueError("Frozen matrices contain duplicate identity rows")
    if set(formal["target_id"].unique()) != set(TARGETS) or set(replay["target_id"].unique()) != set(TARGETS):
        raise ValueError("Frozen target domain changed")
    if set(formal["scenario_id"].unique()) != set(SCENARIOS) or set(replay["scenario_id"].unique()) != set(SCENARIOS):
        raise ValueError("Frozen scenario domain changed")
    if set(outer_geometry["outer_origin_year"].unique()) != set(OUTER_ORIGINS):
        raise ValueError("Frozen outer geometry origin domain changed")
    eligible_count = int(
        (
            (inner_geometry["inner_training_row_count"] >= 40)
            & (inner_geometry["inner_validation_row_count"] >= 20)
            & (inner_geometry["boundary_pass"] == "PASS")
        ).sum()
    )
    if eligible_count != 264:
        raise ValueError(f"Frozen eligible inner-fold count changed: {eligible_count}")
    for target_id, expected in EXPECTED_B3_MSE.items():
        actual = float(formal.loc[formal["target_id"] == target_id, "b3_squared_error"].mean())
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-9):
            raise ValueError(f"B3 reference drift for {target_id}: {actual} != {expected}")
    for name, frame in (("formal", formal), ("replay", replay)):
        nonpositive_latest = int((frame["pop_latest_level"] <= 0).sum())
        if nonpositive_latest:
            raise ValueError(
                f"{name} has {nonpositive_latest} rows with non-positive pop_latest_level; "
                "the frozen contract requires reporting and stopping, not repair"
            )
        nonpositive_actual = int((frame["actual_target_value"] <= 0).sum())
        if nonpositive_actual:
            raise ValueError(f"{name} has {nonpositive_actual} non-positive actual targets; log-growth is undefined")


def select_config(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    framing_id: str,
    outer_pool: pd.DataFrame,
    inner_rows: pd.DataFrame,
    model_input_columns: list[str],
    fit_counter: Counter,
    inner_executor: ProcessPoolExecutor,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, float], int]:
    eligible = inner_rows.loc[
        (inner_rows["inner_training_row_count"] >= 40)
        & (inner_rows["inner_validation_row_count"] >= 20)
        & (inner_rows["boundary_pass"] == "PASS")
    ].sort_values("inner_validation_year", kind="stable")
    evidence: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    task_rows: list[tuple[dict[str, Any], str, list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]] = []
    for config in CONFIGS:
        for fold in eligible.itertuples(index=False):
            inner_training = outer_pool.loc[
                outer_pool["pseudo_target_year"] <= int(fold.inner_validation_year) - 2
            ]
            inner_validation = outer_pool.loc[
                outer_pool["pseudo_target_year"] == int(fold.inner_validation_year)
            ]
            if len(inner_training) != int(fold.inner_training_row_count):
                raise ValueError("Inner training row count disagrees with frozen geometry")
            if len(inner_validation) != int(fold.inner_validation_row_count):
                raise ValueError("Inner validation row count disagrees with frozen geometry")
            task_rows.append(
                (
                    config,
                    framing_id,
                    model_input_columns,
                    fold._asdict(),
                    inner_training,
                    inner_validation,
                )
            )
    results = list(inner_executor.map(evaluate_inner_task, task_rows)) if task_rows else []
    fit_counter["inner"] += len(results)
    by_config: dict[str, list[dict[str, Any]]] = {config["config_id"]: [] for config in CONFIGS}
    for result in results:
        by_config[result["config"]["config_id"]].append(result)
    for config in CONFIGS:
        config_scores: list[float] = []
        for result in by_config[config["config_id"]]:
            fold = result["fold"]
            mse = result["mse"]
            config_scores.append(mse)
            evidence.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "outer_origin_year": outer_origin,
                    "framing_id": framing_id,
                    "config_id": config["config_id"],
                    "config_order": config["config_order"],
                    "iterations": config["iterations"],
                    "learning_rate": config["learning_rate"],
                    "depth": config["depth"],
                    "l2_leaf_reg": config["l2_leaf_reg"],
                    "rsm": config["rsm"],
                    "inner_validation_year": int(fold["inner_validation_year"]),
                    "inner_training_row_count_geometry": int(fold["inner_training_row_count"]),
                    "inner_validation_row_count_geometry": int(fold["inner_validation_row_count"]),
                    "inner_training_row_count_actual": int(result["inner_training_row_count_actual"]),
                    "inner_validation_row_count_actual": int(result["inner_validation_row_count_actual"]),
                    "inner_training_max_year_geometry": int(fold["inner_training_max_year"]),
                    "inner_validation_mse": mse,
                    "mean_inner_validation_mse": None,
                    "boundary_pass": fold["boundary_pass"],
                    "fold_eligible": True,
                }
            )
        scores[config["config_id"]] = float(np.mean(config_scores)) if config_scores else float("nan")
    if len(eligible) < 3:
        selected = CONFIGS[0]
        selection_method = "fallback_conservative_default"
        if eligible.empty:
            for config in CONFIGS:
                evidence.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": outer_origin,
                        "framing_id": framing_id,
                        "config_id": config["config_id"],
                        "config_order": config["config_order"],
                        "iterations": config["iterations"],
                        "learning_rate": config["learning_rate"],
                        "depth": config["depth"],
                        "l2_leaf_reg": config["l2_leaf_reg"],
                        "rsm": config["rsm"],
                        "inner_validation_year": None,
                        "inner_training_row_count_geometry": 0,
                        "inner_validation_row_count_geometry": 0,
                        "inner_training_row_count_actual": 0,
                        "inner_validation_row_count_actual": 0,
                        "inner_training_max_year_geometry": None,
                        "inner_validation_mse": None,
                        "mean_inner_validation_mse": None,
                        "boundary_pass": "PASS",
                        "fold_eligible": False,
                    }
                )
    else:
        selected = CONFIGS[0]
        best_score = scores[selected["config_id"]]
        for candidate in CONFIGS[1:]:
            candidate_score = scores[candidate["config_id"]]
            if candidate_score < best_score and not np.isclose(
                candidate_score, best_score, rtol=1e-12, atol=1e-12
            ):
                selected = candidate
                best_score = candidate_score
        selection_method = "inner_mean_validation_mse"
    for row in evidence:
        row["mean_inner_validation_mse"] = scores.get(row["config_id"])
        row["selected_config_id"] = selected["config_id"]
        row["selection_method"] = selection_method
        row["selected"] = row["config_id"] == selected["config_id"]
    return selected, selection_method, evidence, scores, int(len(eligible))


def selected_config_row(
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    framing_id: str,
    selected: dict[str, Any],
    selection_method: str,
    scores: dict[str, float],
    eligible_count: int,
    total_inner_count: int,
    outer_pool: pd.DataFrame,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": MODEL_IDS[framing_id],
        "target_id": target_id,
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "framing_id": framing_id,
        "selected_config_id": selected["config_id"],
        "selected_config_order": selected["config_order"],
        "selection_method": selection_method,
        "fallback_conservative_default": selection_method == "fallback_conservative_default",
        "eligible_inner_fold_count": eligible_count,
        "total_inner_fold_count": total_inner_count,
        "outer_training_row_count": int(len(outer_pool)),
        "outer_training_city_count": int(outer_pool["city_id"].nunique()),
        "outer_training_year_count": int(outer_pool["pseudo_target_year"].nunique()),
        "training_max_label_year": int(outer_origin - 2),
        "iterations": selected["iterations"],
        "learning_rate": selected["learning_rate"],
        "depth": selected["depth"],
        "l2_leaf_reg": selected["l2_leaf_reg"],
        "rsm": selected["rsm"],
        "city_id_handling": "native categorical",
        "numeric_missing_handling": "raw NaN retained; CatBoost native",
    }
    for config in CONFIGS:
        row[f"{config['config_id']}_mean_inner_validation_mse"] = scores.get(config["config_id"])
    return row


def build_metrics(
    oof: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    for (target_id, scenario_id, framing_id), group in oof.groupby(
        ["target_id", "scenario_id", "framing_id"], sort=True
    ):
        origin_mse = group.groupby("origin_year")["squared_error"].mean()
        origin_b3_mse = group.groupby("origin_year")["b3_squared_error"].mean()
        model_mse = float(group["squared_error"].mean())
        b3_mse = float(group["b3_squared_error"].mean())
        total_sse = float(group["squared_error"].sum())
        city_summary = (
            group.groupby("city_id")
            .agg(
                city_sse=("squared_error", "sum"),
                city_mse=("squared_error", "mean"),
                city_b3_sse=("b3_squared_error", "sum"),
                city_b3_mse=("b3_squared_error", "mean"),
                city_bias=("residual_prediction_minus_actual", "mean"),
                city_mae=("residual_prediction_minus_actual", lambda values: float(np.mean(np.abs(values)))),
                city_row_count=("squared_error", "size"),
            )
            .reset_index()
        )
        city_top = city_summary.sort_values("city_sse", ascending=False, kind="stable")
        top1_share = float(city_top.head(1)["city_sse"].sum() / total_sse) if total_sse else 0.0
        top3_share = float(city_top.head(3)["city_sse"].sum() / total_sse) if total_sse else 0.0
        selected = selections.loc[
            (selections["target_id"] == target_id)
            & (selections["scenario_id"] == scenario_id)
            & (selections["framing_id"] == framing_id)
        ]
        aggregate_rows.append(
            {
                "model_id": MODEL_IDS[framing_id],
                "target_id": target_id,
                "scenario_id": scenario_id,
                "framing_id": framing_id,
                "row_count": int(len(group)),
                "origin_count": int(group["origin_year"].nunique()),
                "city_count": int(group["city_id"].nunique()),
                "pooled_city_year_mse": model_mse,
                "b3_pooled_city_year_mse": b3_mse,
                "delta_vs_b3_mse": model_mse - b3_mse,
                "improvement_vs_b3_pct": 1.0 - model_mse / b3_mse,
                "macro_origin_mse": float(origin_mse.mean()),
                "origin_win_count_vs_b3": int((origin_mse < origin_b3_mse).sum()),
                "origin_win_rate_vs_b3": float((origin_mse < origin_b3_mse).mean()),
                "latest_2021_mse": float(origin_mse.loc[2021]),
                "rmse": float(np.sqrt(model_mse)),
                "mae": float(np.mean(np.abs(group["residual_prediction_minus_actual"]))),
                "bias": float(group["residual_prediction_minus_actual"].mean()),
                "bias_prediction_minus_actual": float(group["residual_prediction_minus_actual"].mean()),
                "total_sse": total_sse,
                "top1_city_sse_concentration": top1_share,
                "top3_city_sse_concentration": top3_share,
                "improved_city_count_vs_b3": int((city_summary["city_mse"] < city_summary["city_b3_mse"]).sum()),
                "fallback_fit_count": int(selected["fallback_conservative_default"].sum()),
                "outer_fit_count": int(len(selected)),
                "training_row_count_min": int(selected["outer_training_row_count"].min()),
                "training_row_count_max": int(selected["outer_training_row_count"].max()),
                "training_row_count_mean": float(selected["outer_training_row_count"].mean()),
            }
        )
        for origin_year, origin_group in group.groupby("origin_year", sort=True):
            selected_origin = selected.loc[selected["outer_origin_year"] == origin_year].iloc[0]
            model_origin_mse = float(origin_group["squared_error"].mean())
            b3_origin_mse = float(origin_group["b3_squared_error"].mean())
            origin_rows.append(
                {
                    "model_id": MODEL_IDS[framing_id],
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "framing_id": framing_id,
                    "origin_year": int(origin_year),
                    "row_count": int(len(origin_group)),
                    "pooled_city_year_mse": model_origin_mse,
                    "b3_mse": b3_origin_mse,
                    "delta_vs_b3_mse": model_origin_mse - b3_origin_mse,
                    "improvement_vs_b3_pct": 1.0 - model_origin_mse / b3_origin_mse,
                    "origin_win_vs_b3": model_origin_mse < b3_origin_mse,
                    "rmse": float(np.sqrt(model_origin_mse)),
                    "mae": float(np.mean(np.abs(origin_group["residual_prediction_minus_actual"]))),
                    "bias": float(origin_group["residual_prediction_minus_actual"].mean()),
                    "selected_config_id": selected_origin["selected_config_id"],
                    "selection_method": selected_origin["selection_method"],
                    "eligible_inner_fold_count": int(selected_origin["eligible_inner_fold_count"]),
                    "outer_training_row_count": int(selected_origin["outer_training_row_count"]),
                    "training_max_label_year": int(selected_origin["training_max_label_year"]),
                }
            )
        city_summary["model_id"] = MODEL_IDS[framing_id]
        city_summary["target_id"] = target_id
        city_summary["scenario_id"] = scenario_id
        city_summary["framing_id"] = framing_id
        city_summary["delta_vs_b3_mse"] = city_summary["city_mse"] - city_summary["city_b3_mse"]
        city_summary["improvement_vs_b3_pct"] = 1.0 - city_summary["city_mse"] / city_summary["city_b3_mse"]
        city_summary["city_sse_share"] = city_summary["city_sse"] / total_sse if total_sse else 0.0
        b3_total_sse = float(city_summary["city_b3_sse"].sum())
        city_summary["b3_city_sse_share"] = city_summary["city_b3_sse"] / b3_total_sse if b3_total_sse else 0.0
        city_summary["improved_vs_b3"] = city_summary["city_mse"] < city_summary["city_b3_mse"]
        city_summary["model_sse_rank"] = city_summary["city_sse"].rank(method="first", ascending=False).astype(int)
        city_summary["top1_model_sse"] = city_summary["model_sse_rank"] == 1
        city_summary["top3_model_sse"] = city_summary["model_sse_rank"] <= 3
        city_rows.extend(city_summary.to_dict(orient="records"))
    aggregate = pd.DataFrame(aggregate_rows)
    if not aggregate.empty:
        pair = aggregate.pivot_table(
            index=["target_id", "scenario_id"], columns="framing_id", values="pooled_city_year_mse"
        )
        pair = pair.rename(
            columns={
                "F3_DELTA_FROM_LATEST": "f3_pooled_city_year_mse",
                "F4_LOG_GROWTH_FROM_LATEST": "f4_pooled_city_year_mse",
            }
        ).reset_index()
        aggregate = aggregate.merge(pair, on=["target_id", "scenario_id"], how="left")
        aggregate["f3_vs_f4_pooled_mse_delta"] = (
            aggregate["f3_pooled_city_year_mse"] - aggregate["f4_pooled_city_year_mse"]
        )
    return aggregate, pd.DataFrame(origin_rows), pd.DataFrame(city_rows)


def compute_scale_bounds(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame["pop_latest_level"].to_numpy(dtype=float)
    return float(np.quantile(values, 1 / 3)), float(np.quantile(values, 2 / 3))


def assign_scale_bin(value: float, bounds: tuple[float, float]) -> str:
    if value <= bounds[0]:
        return "low"
    if value <= bounds[1]:
        return "medium"
    return "high"


def build_scale_diagnostics(
    formal: pd.DataFrame,
    oof: pd.DataFrame,
    scale_bounds: dict[tuple[str, str, int], tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["target_id", "scenario_id", "origin_year", "city_id"]
    base = formal.loc[:, key + ["actual_target_value", "pop_latest_level", "b3_prediction", "b3_squared_error"]].copy()
    base["absolute_scale"] = base["pop_latest_level"]
    base["absolute_delta"] = base["actual_target_value"] - base["pop_latest_level"]
    base["absolute_delta_magnitude"] = base["absolute_delta"].abs()
    base["relative_growth"] = base["actual_target_value"] / base["pop_latest_level"] - 1.0
    base["log_growth"] = np.log(base["actual_target_value"] / base["pop_latest_level"])
    predictions = oof.pivot_table(index=key, columns="framing_id", values="prediction_value").reset_index()
    errors = oof.pivot_table(index=key, columns="framing_id", values="squared_error").reset_index()
    predictions.columns.name = None
    errors.columns.name = None
    predictions = predictions.rename(
        columns={
            "F3_DELTA_FROM_LATEST": "f3_prediction",
            "F4_LOG_GROWTH_FROM_LATEST": "f4_prediction",
        }
    )
    errors = errors.rename(
        columns={
            "F3_DELTA_FROM_LATEST": "f3_squared_error",
            "F4_LOG_GROWTH_FROM_LATEST": "f4_squared_error",
        }
    )
    diagnostics = base.merge(predictions, on=key, validate="one_to_one").merge(errors, on=key, validate="one_to_one")
    diagnostics["scale_bin"] = [
        assign_scale_bin(
            float(row.pop_latest_level), scale_bounds[(row.target_id, row.scenario_id, int(row.origin_year))]
        )
        for row in diagnostics.itertuples(index=False)
    ]
    diagnostics = diagnostics.sort_values(key, kind="stable")

    scale_rows: list[dict[str, Any]] = []
    for (target_id, scenario_id, framing_id), group in oof.groupby(
        ["target_id", "scenario_id", "framing_id"], sort=True
    ):
        group_model = MODEL_IDS[framing_id]
        for scale_bin in ("low", "medium", "high"):
            selected = group.loc[group["scale_bin"] == scale_bin]
            latest_bounds = selected[["scale_bin_train_q33", "scale_bin_train_q67"]].drop_duplicates()
            q33 = float(latest_bounds.iloc[0]["scale_bin_train_q33"]) if len(latest_bounds) else None
            q67 = float(latest_bounds.iloc[0]["scale_bin_train_q67"]) if len(latest_bounds) else None
            row_count = len(selected)
            if row_count:
                mse = float(selected["squared_error"].mean())
                b3_mse = float(selected["b3_squared_error"].mean())
                values = selected["pop_latest_level"]
                scale_rows.append(
                    {
                        "model_id": group_model,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "framing_id": framing_id,
                        "scale_bin": scale_bin,
                        "bin_definition": "outer_training_pop_latest_level_tertiles",
                        "training_q33": q33,
                        "training_q67": q67,
                        "row_count": int(row_count),
                        "mean_latest_population": float(values.mean()),
                        "min_latest_population": float(values.min()),
                        "max_latest_population": float(values.max()),
                        "pooled_city_year_mse": mse,
                        "b3_mse": b3_mse,
                        "delta_vs_b3_mse": mse - b3_mse,
                        "improvement_vs_b3_pct": 1.0 - mse / b3_mse,
                        "rmse": float(np.sqrt(mse)),
                        "mae": float(np.mean(np.abs(selected["residual_prediction_minus_actual"]))),
                        "bias": float(selected["residual_prediction_minus_actual"].mean()),
                        "model_sse": float(selected["squared_error"].sum()),
                        "b3_sse": float(selected["b3_squared_error"].sum()),
                    }
                )
            else:
                scale_rows.append(
                    {
                        "model_id": group_model,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "framing_id": framing_id,
                        "scale_bin": scale_bin,
                        "bin_definition": "outer_training_pop_latest_level_tertiles",
                        "training_q33": q33,
                        "training_q67": q67,
                        "row_count": 0,
                        "mean_latest_population": None,
                        "min_latest_population": None,
                        "max_latest_population": None,
                        "pooled_city_year_mse": None,
                        "b3_mse": None,
                        "delta_vs_b3_mse": None,
                        "improvement_vs_b3_pct": None,
                        "rmse": None,
                        "mae": None,
                        "bias": None,
                        "model_sse": None,
                        "b3_sse": None,
                    }
                )
    return diagnostics, pd.DataFrame(scale_rows)


def classify_framings(aggregate: pd.DataFrame) -> tuple[dict[str, str], dict[str, Any], str]:
    classifications: dict[str, str] = {}
    evidence_by_framing: dict[str, Any] = {}
    rank = {
        "framing_unsupported": 0,
        "framing_competitive_but_unstable": 1,
        "framing_signal_found": 2,
    }
    for framing_id in FRAMINGS:
        rows = aggregate.loc[
            (aggregate["target_id"] == "resident") & (aggregate["framing_id"] == framing_id)
        ].set_index("scenario_id")
        if set(SCENARIOS) != set(rows.index):
            classification = "framing_unsupported"
            evidence_by_framing[framing_id] = {"missing_scenario_evidence": True}
            classifications[framing_id] = classification
            continue
        r1, r2 = rows.loc[SCENARIOS[0]], rows.loc[SCENARIOS[1]]
        r1_improvement = float(r1["improvement_vs_b3_pct"])
        r2_improvement = float(r2["improvement_vs_b3_pct"])
        stable_both = bool(
            r1["origin_win_count_vs_b3"] >= 2
            and r2["origin_win_count_vs_b3"] >= 2
            and r1["improved_city_count_vs_b3"] >= 2
            and r2["improved_city_count_vs_b3"] >= 2
        )
        strong_scenario = (
            SCENARIOS[0] if r1_improvement >= 0.02 else SCENARIOS[1] if r2_improvement >= 0.02 else None
        )
        strong_row = rows.loc[strong_scenario] if strong_scenario else None
        stable_single = bool(
            strong_row is not None
            and strong_row["origin_win_count_vs_b3"] >= 2
            and strong_row["improved_city_count_vs_b3"] >= 2
        )
        signal_both = bool(r1_improvement > 0 and r2_improvement > 0 and stable_both)
        signal_one = bool(
            strong_scenario is not None
            and max(r1_improvement, r2_improvement) >= 0.02
            and min(r1_improvement, r2_improvement) >= -0.01
            and stable_single
        )
        meaningful = bool(
            r1_improvement > 0
            or r2_improvement > 0
            or r1["origin_win_count_vs_b3"] > 0
            or r2["origin_win_count_vs_b3"] > 0
        )
        if signal_both or signal_one:
            classification = "framing_signal_found"
        elif meaningful:
            classification = "framing_competitive_but_unstable"
        else:
            classification = "framing_unsupported"
        classifications[framing_id] = classification
        evidence_by_framing[framing_id] = {
            "r1_improvement_vs_b3_pct": r1_improvement,
            "r2_improvement_vs_b3_pct": r2_improvement,
            "r1_origin_wins": int(r1["origin_win_count_vs_b3"]),
            "r2_origin_wins": int(r2["origin_win_count_vs_b3"]),
            "r1_improved_city_count": int(r1["improved_city_count_vs_b3"]),
            "r2_improved_city_count": int(r2["improved_city_count_vs_b3"]),
            "stable_for_both_scenarios": stable_both,
            "stable_for_strong_scenario": stable_single,
            "signal_both_scenarios": signal_both,
            "signal_one_scenario_with_other_within_minus_1pct": signal_one,
            "meaningful_local_or_scenario_improvement": meaningful,
            "classification": classification,
        }
    overall = max(classifications.values(), key=lambda value: rank[value])
    return classifications, evidence_by_framing, overall


def build_handoff(
    summary: dict[str, Any], aggregate: pd.DataFrame, scale_bins: pd.DataFrame
) -> str:
    lines = [
        f"# {PLAN_ID} Handoff — Delta/Growth target framing screening",
        "",
        "## Result",
        "",
        "- Status: `COMPLETE` (framing screening only)",
        f"- F3 classification: `{summary['classifications']['F3_DELTA_FROM_LATEST']}`",
        f"- F4 classification: `{summary['classifications']['F4_LOG_GROWTH_FROM_LATEST']}`",
        f"- Overall classification: `{summary['overall_classification']}`",
        f"- Worker branch: `{summary['git']['branch']}`",
        f"- Wave base SHA: `{BASE_SHA}`",
        f"- Artifact-generation source commit: `{summary['git']['worker_commit']}`",
        "- S5 is not declared complete; S6 is not entered.",
        "",
        "## Frozen contract and method",
        "",
        "- Formal validation: 1,760 rows; replay: 2,320 rows; 144 numeric Plan04 features plus raw string `city_id` categorical metadata.",
        "- CatBoostRegressor uses only the three pre-registered configurations, seed `20260919`, raw numeric NaN, CPU `thread_count=1`, and no outer-label early stopping.",
        "- F3 label is `actual_target_value - pop_latest_level`; F4 label is `log(actual_target_value / pop_latest_level)`; reconstruction is un-clipped.",
        "- Configuration selection uses only eligible inner folds inside each outer training pool; no outer labels enter selection.",
        "",
        "## Aggregate metrics",
        "",
        "| target | scenario | framing | pooled MSE | macro-origin MSE | origin wins vs B3 | latest-2021 MSE | RMSE | MAE | bias | top1 city SSE | top3 city SSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.sort_values(["target_id", "scenario_id", "framing_id"]).to_dict(orient="records"):
        lines.append(
            "| {target_id} | {scenario_id} | {framing_id} | {mse} | {macro} | {wins} | {latest} | {rmse} | {mae} | {bias} | {top1}% | {top3}% |".format(
                target_id=row["target_id"],
                scenario_id=row["scenario_id"],
                framing_id=row["framing_id"],
                mse=fmt(row["pooled_city_year_mse"]),
                macro=fmt(row["macro_origin_mse"]),
                wins=int(row["origin_win_count_vs_b3"]),
                latest=fmt(row["latest_2021_mse"]),
                rmse=fmt(row["rmse"]),
                mae=fmt(row["mae"]),
                bias=fmt(row["bias"]),
                top1=fmt(100 * row["top1_city_sse_concentration"], 2),
                top3=fmt(100 * row["top3_city_sse_concentration"], 2),
            )
        )
    lines.extend(["", "## Configuration selections", ""])
    for row in summary["inner_selection"]["selection_counts"]:
        lines.append(
            f"- `{row['framing_id']}` / `{row['selected_config_id']}`: {row['outer_fit_count']} outer fits"
        )
    lines.append(
        f"- Fallback conservative selections: {summary['inner_selection']['fallback_outer_fit_count_all_framings']} total, "
        f"{summary['inner_selection']['fallback_outer_fit_count_per_framing']} per framing."
    )
    lines.extend(["", "## Scale-bin behavior", ""])
    lines.append("Bins are low/medium/high tertiles of `pop_latest_level` computed inside each relevant outer training universe.")
    lines.append("| target | scenario | framing | bin | rows | model MSE | B3 MSE | improvement |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for row in scale_bins.sort_values(["target_id", "scenario_id", "framing_id", "scale_bin"]).to_dict(orient="records"):
        lines.append(
            f"| {row['target_id']} | {row['scenario_id']} | {row['framing_id']} | {row['scale_bin']} | {row['row_count']} | {fmt(row['pooled_city_year_mse'])} | {fmt(row['b3_mse'])} | {fmt(100 * row['improvement_vs_b3_pct'], 2) if row['improvement_vs_b3_pct'] is not None else 'NA'}% |"
        )
    lines.extend(
        [
            "",
            "## Reconstruction and boundary statement",
            "",
            f"- Pathology check: {summary['diagnostics']['reconstruction_pathology']}; all formal/replay `pop_latest_level` values were positive and no clipping was applied.",
            "- No Plan04 feature change, replay-row change, post-hoc city routing, gating, ensemble, stacking, 2023 prediction, or submission was performed.",
            "- This worker does not declare S5 complete and does not enter S6.",
            "",
            "## Artifacts",
            "",
            "- `oof_predictions.csv`, `aggregate_metrics.csv`, `per_origin_metrics.csv`, `configuration_selection.csv`, `inner_selection_evidence.csv`, `city_model_error.csv`.",
            "- `scale_error_diagnostics.csv`, `scale_bin_metrics.csv`, and `framing_summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(root: Path, fit_workers: int) -> dict[str, Any]:
    if fit_workers < 1:
        raise ValueError("--fit-workers must be >= 1")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    handoff_dir = root / "reports" / "S5_HANDOFFS" / PLAN_ID
    handoff_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(
        (root / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json").read_text(
            encoding="utf-8"
        )
    )
    contract = json.loads(
        (root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json").read_text(
            encoding="utf-8"
        )
    )
    model_features = list(contract["model_features"])
    feature_group_map(contract, model_features)
    model_input_columns = ["city_id", *model_features]
    formal = prepare_matrix(
        root / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv",
        model_features,
        "formal_validation_matrix",
        "origin_year",
    )
    replay = prepare_matrix(
        root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv",
        model_features,
        "training_replay_matrix",
        "pseudo_target_year",
    )
    outer_geometry = prepare_geometry(
        root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "outer_training_geometry.csv",
        ["outer_origin_year", "training_max_pseudo_target_year", "training_min_pseudo_target_year"],
    )
    inner_geometry = prepare_geometry(
        root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "inner_split_geometry.csv",
        ["outer_origin_year", "inner_validation_year", "inner_training_max_year", "inner_training_min_year"],
    )
    validate_contract(schema, contract, formal, replay, outer_geometry, inner_geometry, model_features)
    print(
        f"[{PLAN_ID}] validated frozen matrices: formal={len(formal)}, replay={len(replay)}, features={len(model_features)}",
        flush=True,
    )
    import catboost

    print(f"[{PLAN_ID}] CatBoost {catboost.__version__}; fit-workers={fit_workers}; thread_count=1", flush=True)

    fit_counter: Counter = Counter()
    selection_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    scale_bounds: dict[tuple[str, str, int], tuple[float, float]] = {}
    with ProcessPoolExecutor(max_workers=fit_workers) as inner_executor:
        for target_id in TARGETS:
            for scenario_id in SCENARIOS:
                for outer_origin in OUTER_ORIGINS:
                    outer_pool = replay.loc[
                        (replay["target_id"] == target_id)
                        & (replay["scenario_id"] == scenario_id)
                        & (replay["pseudo_target_year"] <= outer_origin - 2)
                    ].copy()
                    outer_geo = outer_geometry.loc[
                        (outer_geometry["target_id"] == target_id)
                        & (outer_geometry["scenario_id"] == scenario_id)
                        & (outer_geometry["outer_origin_year"] == outer_origin)
                    ].iloc[0]
                    if len(outer_pool) != int(outer_geo.training_row_count):
                        raise ValueError("Outer training row count disagrees with frozen geometry")
                    if outer_pool["pseudo_target_year"].max() != outer_origin - 2:
                        raise ValueError("Outer training maximum year violates the frozen cutoff")
                    formal_group = formal.loc[
                        (formal["target_id"] == target_id)
                        & (formal["scenario_id"] == scenario_id)
                        & (formal["origin_year"] == outer_origin)
                    ].copy()
                    if len(formal_group) != 40:
                        raise ValueError("Formal outer validation group does not contain 40 cities")
                    bounds = compute_scale_bounds(outer_pool)
                    scale_bounds[(target_id, scenario_id, outer_origin)] = bounds
                    formal_group["scale_bin"] = formal_group["pop_latest_level"].map(
                        lambda value: assign_scale_bin(float(value), bounds)
                    )
                    inner_rows = inner_geometry.loc[
                        (inner_geometry["target_id"] == target_id)
                        & (inner_geometry["scenario_id"] == scenario_id)
                        & (inner_geometry["outer_origin_year"] == outer_origin)
                    ].copy()
                    for framing_id in FRAMINGS:
                        selected, selection_method, evidence, scores, eligible_count = select_config(
                            target_id,
                            scenario_id,
                            outer_origin,
                            framing_id,
                            outer_pool,
                            inner_rows,
                            model_input_columns,
                            fit_counter,
                            inner_executor,
                        )
                        evidence_rows.extend(evidence)
                        selection_rows.append(
                            selected_config_row(
                                target_id,
                                scenario_id,
                                outer_origin,
                                framing_id,
                                selected,
                                selection_method,
                                scores,
                                eligible_count,
                                len(inner_rows),
                                outer_pool,
                            )
                        )
                        prediction = fit_and_predict(
                            outer_pool, formal_group, framing_id, selected, model_input_columns
                        )
                        fit_counter["outer"] += 1
                        actual = formal_group["actual_target_value"].to_numpy(dtype=float)
                        residual = prediction - actual
                        for row, pred, error in zip(formal_group.itertuples(index=False), prediction, residual):
                            oof_rows.append(
                                {
                                    "model_id": MODEL_IDS[framing_id],
                                    "framing_id": framing_id,
                                    "target_id": target_id,
                                    "scenario_id": scenario_id,
                                    "origin_year": int(row.origin_year),
                                    "city_id": row.city_id,
                                    "actual_target_value": float(row.actual_target_value),
                                    "prediction_value": float(pred),
                                    "residual_prediction_minus_actual": float(error),
                                    "squared_error": float(error * error),
                                    "training_max_label_year": int(outer_origin - 2),
                                    "feature_contract_version": FEATURE_CONTRACT,
                                    "worker_commit": git_value(root, "rev-parse", "HEAD"),
                                    "selected_config_id": selected["config_id"],
                                    "selection_method": selection_method,
                                    "outer_training_row_count": int(len(outer_pool)),
                                    "b3_prediction": float(row.b3_prediction),
                                    "b3_squared_error": float(row.b3_squared_error),
                                    "pop_latest_level": float(row.pop_latest_level),
                                    "scale_bin": row.scale_bin,
                                    "scale_bin_train_q33": bounds[0],
                                    "scale_bin_train_q67": bounds[1],
                                }
                            )
                        print(
                            f"[{PLAN_ID}] {target_id}/{scenario_id}/{outer_origin}/{framing_id}: "
                            f"selected={selected['config_id']} method={selection_method} eligible={eligible_count}",
                            flush=True,
                        )

    selections = pd.DataFrame(selection_rows)
    inner_evidence = pd.DataFrame(evidence_rows)
    oof = pd.DataFrame(oof_rows)
    expected_outer_fit_count = len(TARGETS) * len(SCENARIOS) * len(OUTER_ORIGINS) * len(FRAMINGS)
    if len(selections) != expected_outer_fit_count or fit_counter["outer"] != expected_outer_fit_count:
        raise ValueError("Unexpected outer fit count")
    if len(oof) != expected_outer_fit_count * 40:
        raise ValueError("Unexpected OOF row count")
    for framing_id in FRAMINGS:
        if len(oof.loc[oof["framing_id"] == framing_id]) != 1760:
            raise ValueError(f"OOF coverage is not 1760 rows for {framing_id}")
    if int(inner_evidence["fold_eligible"].sum()) != 264 * len(CONFIGS) * len(FRAMINGS):
        raise ValueError("Inner evidence eligible-fold count disagrees with frozen contract")
    if int(selections["fallback_conservative_default"].sum()) != 8 * len(FRAMINGS):
        raise ValueError("Fallback count disagrees with frozen contract")
    if oof.duplicated(["framing_id", "target_id", "scenario_id", "origin_year", "city_id"]).any():
        raise ValueError("OOF contains duplicate promoted-candidate rows")
    if (oof["training_max_label_year"] > oof["origin_year"] - 2).any():
        raise ValueError("OOF training cutoff violates the frozen outer boundary")
    if not np.allclose(
        oof["squared_error"].to_numpy(), oof["residual_prediction_minus_actual"].to_numpy() ** 2,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError("OOF squared-error column is inconsistent")

    aggregate, per_origin, city_errors = build_metrics(oof, selections)
    scale_diagnostics, scale_bins = build_scale_diagnostics(formal, oof, scale_bounds)
    classifications, classification_evidence, overall = classify_framings(aggregate)
    selection_counts = (
        selections.groupby(["framing_id", "selected_config_id"], sort=True)
        .size()
        .reset_index(name="outer_fit_count")
        .to_dict(orient="records")
    )
    r1_r2_comparison = []
    for framing_id in FRAMINGS:
        rows = aggregate.loc[aggregate["framing_id"] == framing_id]
        for target_id in TARGETS:
            pair = rows.loc[rows["target_id"] == target_id].set_index("scenario_id")
            if set(SCENARIOS) == set(pair.index):
                r1_r2_comparison.append(
                    {
                        "framing_id": framing_id,
                        "target_id": target_id,
                        "r1_pooled_city_year_mse": float(pair.loc[SCENARIOS[0], "pooled_city_year_mse"]),
                        "r2_pooled_city_year_mse": float(pair.loc[SCENARIOS[1], "pooled_city_year_mse"]),
                        "r2_minus_r1_pooled_city_year_mse": float(
                            pair.loc[SCENARIOS[1], "pooled_city_year_mse"]
                            - pair.loc[SCENARIOS[0], "pooled_city_year_mse"]
                        ),
                        "r1_origin_wins_vs_b3": int(pair.loc[SCENARIOS[0], "origin_win_count_vs_b3"]),
                        "r2_origin_wins_vs_b3": int(pair.loc[SCENARIOS[1], "origin_win_count_vs_b3"]),
                    }
                )
    worker_commit = git_value(root, "rev-parse", "HEAD")
    summary: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "status": "COMPLETE",
        "base_sha": BASE_SHA,
        "git": {"branch": git_value(root, "branch", "--show-current"), "worker_commit": worker_commit},
        "classifications": classifications,
        "overall_classification": overall,
        "classification_evidence": classification_evidence,
        "seed": SEED,
        "library_versions": {"catboost": catboost.__version__},
        "catboost_parameters": {"common": COMMON_CATBOOST_PARAMS, "pre_registered_configs": CONFIGS},
        "data_contract": {
            "feature_contract_version": FEATURE_CONTRACT,
            "replay_contract_version": REPLAY_CONTRACT,
            "formal_validation_rows": int(len(formal)),
            "replay_rows": int(len(replay)),
            "numeric_model_features": int(len(model_features)),
            "model_input_columns": int(len(model_input_columns)),
            "city_count": int(formal["city_id"].nunique()),
            "outer_origins": list(OUTER_ORIGINS),
            "targets": list(TARGETS),
            "scenarios": list(SCENARIOS),
            "framings": list(FRAMINGS),
            "formal_oof_rows_per_framing": 1760,
            "b3_reference_pooled_mse": EXPECTED_B3_MSE,
        },
        "inner_selection": {
            "eligible_inner_folds": 264,
            "total_inner_folds": int(len(inner_geometry)),
            "outer_fit_groups": int(len(outer_geometry)),
            "fallback_outer_fit_count_per_framing": int(
                selections.loc[selections["framing_id"] == FRAMINGS[0], "fallback_conservative_default"].sum()
            ),
            "fallback_outer_fit_count_all_framings": int(selections["fallback_conservative_default"].sum()),
            "selection_method_values": sorted(selections["selection_method"].unique().tolist()),
            "selection_counts": selection_counts,
        },
        "metrics": {
            "aggregate": aggregate.to_dict(orient="records"),
            "r1_vs_r2": r1_r2_comparison,
            "framing_pooled_mse_comparison": aggregate[
                ["target_id", "scenario_id", "framing_id", "pooled_city_year_mse", "f3_vs_f4_pooled_mse_delta"]
            ].to_dict(orient="records"),
        },
        "scale_bins": scale_bins.to_dict(orient="records"),
        "diagnostics": {
            "reconstruction_pathology": "none observed",
            "nonpositive_pop_latest_level_formal": int((formal["pop_latest_level"] <= 0).sum()),
            "nonpositive_pop_latest_level_replay": int((replay["pop_latest_level"] <= 0).sum()),
            "log_growth_clipping": False,
            "city_routing": False,
            "ensemble_or_blend": False,
            "feature_change": False,
            "prediction_2023_or_submission": False,
        },
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "inner_fit_count": int(fit_counter["inner"]),
            "outer_fit_count": int(fit_counter["outer"]),
            "total_fit_count": int(fit_counter["inner"] + fit_counter["outer"]),
            "parallel_fit_workers": int(fit_workers),
            "thread_count_per_fit": 1,
        },
        "forbidden_actions_run": {
            "feature_change": False,
            "routing_or_gating": False,
            "ensemble_or_blend": False,
            "outer_label_tuning": False,
            "2023_prediction_or_submission": False,
            "s5_complete_or_s6_entered": False,
        },
    }
    oof_columns = [
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "origin_year",
        "city_id",
        "actual_target_value",
        "prediction_value",
        "residual_prediction_minus_actual",
        "squared_error",
        "training_max_label_year",
        "feature_contract_version",
        "worker_commit",
        "selected_config_id",
        "selection_method",
        "outer_training_row_count",
        "b3_prediction",
        "b3_squared_error",
        "pop_latest_level",
        "scale_bin",
        "scale_bin_train_q33",
        "scale_bin_train_q67",
    ]
    oof = oof.loc[:, oof_columns].sort_values(
        ["framing_id", "target_id", "scenario_id", "origin_year", "city_id"], kind="stable"
    )
    aggregate.to_csv(handoff_dir / "aggregate_metrics.csv", index=False)
    per_origin.to_csv(handoff_dir / "per_origin_metrics.csv", index=False)
    selections.to_csv(handoff_dir / "configuration_selection.csv", index=False)
    inner_evidence.to_csv(handoff_dir / "inner_selection_evidence.csv", index=False)
    city_errors.to_csv(handoff_dir / "city_model_error.csv", index=False)
    scale_diagnostics.to_csv(handoff_dir / "scale_error_diagnostics.csv", index=False)
    scale_bins.to_csv(handoff_dir / "scale_bin_metrics.csv", index=False)
    oof.to_csv(handoff_dir / "oof_predictions.csv", index=False)
    (handoff_dir / "framing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (handoff_dir / "HANDOFF.md").write_text(
        build_handoff(summary, aggregate, scale_bins), encoding="utf-8"
    )
    print(
        f"[{PLAN_ID}] wrote {len(oof)} OOF rows, {fit_counter['inner']} inner fits, "
        f"{fit_counter['outer']} outer fits; overall={overall}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    run(root, args.fit_workers)


if __name__ == "__main__":
    main()
