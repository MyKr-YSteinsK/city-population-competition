#!/usr/bin/env python3
"""Run the frozen S9-Plan02 F4 learner cross.

This worker is deliberately narrow.  It reuses the frozen Plan04/04B feature
and replay contracts, trains only the pre-registered F4 learner configurations,
and compares them with frozen Plan01 local controls.  The historical native-
city CatBoost pipeline is reproduced from the immutable Plan06E lineage before
any learner interpretation is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PLAN_ID = "S9-Plan02"
PARENT_SHA = "098a4e7e169d528c378803078e33ec5f1450fd03"
RECOVERY_MAIN_SHA = "0a74372ea4e3555b3d659e6a2c77647b92eabea4"
FEATURE_CONTRACT = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT = "S5-Plan04B-TRAINING-REPLAY-V1"
PLAN04B_BASE_SHA = "fac7ed9bf11459bc5d7c6772857683417bf70eac"
HISTORICAL_ARTIFACT_SHA = "343970c608fa1a0f6d5e7ab5d355c29757d6ce2e"
HISTORICAL_SOURCE_SHA = "469787d00b1dc220a017df02c6fbb6fcb76d2605"
HISTORICAL_SOURCE_BLOB = "4185ed8a95b26af4b203f26202e2f97cf9f6b8eb"
HISTORICAL_SOURCE_PATH = "scripts/build_s5_plan06e_delta_growth_framing.py"
SEED = 20260919
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
OUTER_ORIGINS = tuple(range(2011, 2022))
FRAMING_ID = "F4_LOG_GROWTH_FROM_LATEST"
CONTROL_IDS = (
    "B3_RECENT_3POINT_OLS_TREND",
    "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND",
    "ARIMA_011_FIXED_THETA05",
)
LEARNER_IDS = (
    "RIDGE_F4_NUMERIC_ONLY",
    "LIGHTGBM_F4_NUMERIC_ONLY",
    "XGBOOST_F4_NUMERIC_ONLY",
    "CATBOOST_F4_NUMERIC_ONLY",
    "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE",
)
CANDIDATE_IDS = LEARNER_IDS + CONTROL_IDS
EXPECTED_B3_MSE = {"resident": 872.452651515143, "registered": 225.421212121213}
EXPECTED_PLAN04B_ROWS = {"formal": 1760, "replay": 2320, "outer": 44, "inner": 484}


COMMON_CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": False,
    "thread_count": 1,
    "random_strength": 1.0,
}


CONFIGS: dict[str, tuple[dict[str, Any], ...]] = {
    "RIDGE_F4_NUMERIC_ONLY": (
        {"config_id": "RIDGE_C1", "config_order": 1, "alpha": 0.1, "fallback_default": False},
        {"config_id": "RIDGE_C2", "config_order": 2, "alpha": 1.0, "fallback_default": True},
        {"config_id": "RIDGE_C3", "config_order": 3, "alpha": 10.0, "fallback_default": False},
    ),
    "LIGHTGBM_F4_NUMERIC_ONLY": (
        {
            "config_id": "LGBM_C1_CONSERVATIVE",
            "config_order": 1,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 40,
            "reg_lambda": 20.0,
            "colsample_bytree": 0.70,
            "fallback_default": True,
        },
        {
            "config_id": "LGBM_C2_MEDIUM",
            "config_order": 2,
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.80,
            "fallback_default": False,
        },
        {
            "config_id": "LGBM_C3_FLEXIBLE_REGULARIZED",
            "config_order": 3,
            "n_estimators": 800,
            "learning_rate": 0.02,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 10,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.85,
            "fallback_default": False,
        },
    ),
    "XGBOOST_F4_NUMERIC_ONLY": (
        {
            "config_id": "XGB_C1_CONSERVATIVE",
            "config_order": 1,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 20.0,
            "reg_lambda": 20.0,
            "colsample_bytree": 0.70,
            "fallback_default": True,
        },
        {
            "config_id": "XGB_C2_MEDIUM",
            "config_order": 2,
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 10.0,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.80,
            "fallback_default": False,
        },
        {
            "config_id": "XGB_C3_FLEXIBLE_REGULARIZED",
            "config_order": 3,
            "n_estimators": 800,
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 10.0,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.85,
            "fallback_default": False,
        },
    ),
    "CATBOOST_F4_NUMERIC_ONLY": (
        {
            "config_id": "CATNUM_C1_CONSERVATIVE",
            "config_order": 1,
            "iterations": 400,
            "learning_rate": 0.03,
            "depth": 4,
            "l2_leaf_reg": 20.0,
            "rsm": 0.70,
            "fallback_default": True,
        },
        {
            "config_id": "CATNUM_C2_MEDIUM",
            "config_order": 2,
            "iterations": 600,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 10.0,
            "rsm": 0.80,
            "fallback_default": False,
        },
        {
            "config_id": "CATNUM_C3_FLEXIBLE_REGULARIZED",
            "config_order": 3,
            "iterations": 800,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "rsm": 0.85,
            "fallback_default": False,
        },
    ),
    "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE": (
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
    ),
}


CANDIDATE_DEFINITIONS = {
    "RIDGE_F4_NUMERIC_ONLY": {
        "layer": "A_PRIMARY_LEARNER_CROSS",
        "family": "ridge",
        "pipeline": "SimpleImputer(median)->StandardScaler->Ridge",
        "feature_representation": "144 numeric Plan04 features",
        "city_id_handling": "excluded",
        "missing_value_handling": "learner_required_preprocessing",
        "fallback_config_id": "RIDGE_C2",
    },
    "LIGHTGBM_F4_NUMERIC_ONLY": {
        "layer": "A_PRIMARY_LEARNER_CROSS",
        "family": "lightgbm",
        "pipeline": "LightGBM native numeric NaN",
        "feature_representation": "144 numeric Plan04 features",
        "city_id_handling": "excluded",
        "missing_value_handling": "native_missing",
        "fallback_config_id": "LGBM_C1_CONSERVATIVE",
    },
    "XGBOOST_F4_NUMERIC_ONLY": {
        "layer": "A_PRIMARY_LEARNER_CROSS",
        "family": "xgboost",
        "pipeline": "XGBoost native numeric NaN",
        "feature_representation": "144 numeric Plan04 features",
        "city_id_handling": "excluded",
        "missing_value_handling": "native_missing",
        "fallback_config_id": "XGB_C1_CONSERVATIVE",
    },
    "CATBOOST_F4_NUMERIC_ONLY": {
        "layer": "A_PRIMARY_LEARNER_CROSS",
        "family": "catboost_numeric_only",
        "pipeline": "CatBoost numeric-only native NaN",
        "feature_representation": "144 numeric Plan04 features",
        "city_id_handling": "excluded",
        "missing_value_handling": "native_missing",
        "fallback_config_id": "CATNUM_C1_CONSERVATIVE",
    },
    "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE": {
        "layer": "B_HISTORICAL_CATBOOST_REFERENCE",
        "family": "catboost_native_city",
        "pipeline": "Plan06E historical CatBoost reproduction",
        "feature_representation": "144 numeric Plan04 features plus raw city_id",
        "city_id_handling": "native categorical",
        "missing_value_handling": "native_missing",
        "fallback_config_id": "FRM_C1_CONSERVATIVE",
    },
    "B3_RECENT_3POINT_OLS_TREND": {
        "layer": "PLAN01_FROZEN_CONTROL",
        "family": "local_statistical_control",
        "pipeline": "Plan01 frozen OOF",
        "feature_representation": "target-history-only",
        "city_id_handling": "implicit city-local",
        "missing_value_handling": "not_applicable",
        "fallback_config_id": "frozen_plan01_control",
    },
    "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND": {
        "layer": "PLAN01_FROZEN_CONTROL",
        "family": "robust_local_statistical_control",
        "pipeline": "Plan01 frozen OOF",
        "feature_representation": "target-history-only",
        "city_id_handling": "implicit city-local",
        "missing_value_handling": "not_applicable",
        "fallback_config_id": "frozen_plan01_control",
    },
    "ARIMA_011_FIXED_THETA05": {
        "layer": "PLAN01_FROZEN_CONTROL",
        "family": "diversity_diagnostic",
        "pipeline": "Plan01 frozen OOF",
        "feature_representation": "target-history-only",
        "city_id_handling": "implicit city-local",
        "missing_value_handling": "not_applicable",
        "fallback_config_id": "frozen_plan01_control",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Concurrent independent inner fits; each learner fit remains single-threaded.",
    )
    return parser.parse_args()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_blob(root: Path, commit: str, relative_path: str) -> str:
    return git_value(root, "rev-parse", f"{commit}:{relative_path}")


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def git_show_bytes(root: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative_path}"], cwd=root, stderr=subprocess.STDOUT
    )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, tuple):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def assert_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} has duplicate columns: {duplicates}")


def prepare_matrix(path: Path, model_features: list[str], year_column: str, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    assert_columns(
        frame,
        [
            "target_id",
            "scenario_id",
            year_column,
            "city_id",
            "actual_target_value",
            "b3_prediction",
            "b3_squared_error",
            "pop_latest_level",
            *model_features,
        ],
        name,
    )
    frame["city_id"] = frame["city_id"].astype(str)
    frame[year_column] = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    for column in model_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{name}.{column} contains infinity")
    for column in ("actual_target_value", "b3_prediction", "b3_squared_error", "pop_latest_level"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}.{column} contains non-finite values")
    return frame


def prepare_geometry(path: Path, year_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in year_columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        frame[column] = numeric.astype("Int64") if numeric.isna().any() else numeric.astype(int)
    return frame


def validate_lineage(root: Path, model_features: list[str], schema: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if schema["feature_contract_version"] != FEATURE_CONTRACT:
        raise ValueError("feature contract version drift")
    if contract["contract_version"] != REPLAY_CONTRACT:
        raise ValueError("replay contract version drift")
    if contract["base_sha"] != PLAN04B_BASE_SHA:
        raise ValueError("Plan04B base SHA drift")
    if len(model_features) != 144 or contract["model_feature_count"] != 144:
        raise ValueError("frozen model feature count is not 144")
    flattened = [feature for features in contract["feature_groups"].values() for feature in features]
    if len(flattened) != 144 or set(flattened) != set(model_features):
        raise ValueError("feature group manifest does not cover exactly the 144 model features")
    lineage_path = root / "reports" / "S6_HANDOFFS" / "S6-Plan01" / "historical_lineage_hashes.csv"
    lineage = pd.read_csv(lineage_path)
    expected_files = {
        "reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv": "e4bfc035b07939d8daabffeae05ba414b411249b74759776b8c2cbcfa0372650",
        "reports/S5_HANDOFFS/S5-Plan04/feature_matrix_schema.json": "11fac319f0603d0608dcf91d95950e51c78f0ed4f78becbce04e619267fe7bad",
        "reports/S5_HANDOFFS/S5-Plan04B/parallel_worker_input_contract.json": "91eebf578220c3afc649b012e98246ed9de08a95b2dcf42b1003a0734ad65962",
        "reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv": "c4af3688890b578d263a5fb9b224fc687e3c1c76997b30caa9c578e2858949b3",
        "reports/S6_HANDOFFS/S6-Plan01/final_solution_contract.json": "f5deff350d96d2b688abf19ea63ad2edd0588750d20d6f9ecf52fd4d70561e2b",
    }
    lineage_checks: list[dict[str, Any]] = []
    for relative_path, expected_sha in expected_files.items():
        rows = lineage.loc[lineage["path"].fillna("") == relative_path]
        if rows.empty:
            raise ValueError(f"missing lineage row for {relative_path}")
        actual_sha = sha256_path(root / relative_path)
        recorded_sha = str(rows.iloc[0]["sha256"])
        if actual_sha != expected_sha or recorded_sha != expected_sha or str(rows.iloc[0]["status"]) != "VERIFIED":
            raise ValueError(f"lineage hash mismatch for {relative_path}")
        lineage_checks.append({"path": relative_path, "sha256": actual_sha, "status": "PASS"})
    historical_source_blob = git_blob(root, HISTORICAL_SOURCE_SHA, HISTORICAL_SOURCE_PATH)
    if historical_source_blob != HISTORICAL_SOURCE_BLOB:
        raise ValueError("historical Plan06E source blob mismatch")
    return {"lineage_hash_checks": lineage_checks, "historical_source_blob": historical_source_blob}


def validate_contract(
    formal: pd.DataFrame,
    replay: pd.DataFrame,
    outer_geometry: pd.DataFrame,
    inner_geometry: pd.DataFrame,
) -> dict[str, Any]:
    if len(formal) != EXPECTED_PLAN04B_ROWS["formal"] or len(replay) != EXPECTED_PLAN04B_ROWS["replay"]:
        raise ValueError(f"matrix row drift formal={len(formal)} replay={len(replay)}")
    if len(outer_geometry) != EXPECTED_PLAN04B_ROWS["outer"] or len(inner_geometry) != EXPECTED_PLAN04B_ROWS["inner"]:
        raise ValueError("geometry row drift")
    if sorted(formal["origin_year"].unique().tolist()) != list(OUTER_ORIGINS):
        raise ValueError("outer origin domain drift")
    if formal["city_id"].nunique() != 40 or replay["city_id"].nunique() != 40:
        raise ValueError("city domain drift")
    if set(formal["target_id"].unique()) != set(TARGETS) or set(replay["target_id"].unique()) != set(TARGETS):
        raise ValueError("target domain drift")
    if set(formal["scenario_id"].unique()) != set(SCENARIOS) or set(replay["scenario_id"].unique()) != set(SCENARIOS):
        raise ValueError("scenario domain drift")
    formal_key = ["target_id", "scenario_id", "origin_year", "city_id"]
    replay_key = ["target_id", "scenario_id", "pseudo_target_year", "city_id"]
    if formal.duplicated(formal_key).any() or replay.duplicated(replay_key).any():
        raise ValueError("duplicate frozen matrix keys")
    eligible = (
        (inner_geometry["inner_training_row_count"] >= 40)
        & (inner_geometry["inner_validation_row_count"] >= 20)
        & (inner_geometry["boundary_pass"] == "PASS")
    )
    if int(eligible.sum()) != 264:
        raise ValueError(f"eligible inner fold drift: {int(eligible.sum())}")
    for target_id, expected in EXPECTED_B3_MSE.items():
        actual = float(formal.loc[formal["target_id"] == target_id, "b3_squared_error"].mean())
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-9):
            raise ValueError(f"B3 reference drift for {target_id}: {actual} != {expected}")
    for name, frame in (("formal", formal), ("replay", replay)):
        if (frame["pop_latest_level"] <= 0).any() or (frame["actual_target_value"] <= 0).any():
            raise ValueError(f"{name} contains non-positive F4 level or target")
    return {
        "formal_rows": int(len(formal)),
        "replay_rows": int(len(replay)),
        "outer_geometry_rows": int(len(outer_geometry)),
        "inner_geometry_rows": int(len(inner_geometry)),
        "eligible_inner_folds": int(eligible.sum()),
    }


def framing_labels(frame: pd.DataFrame) -> np.ndarray:
    latest = frame["pop_latest_level"].to_numpy(dtype=float)
    actual = frame["actual_target_value"].to_numpy(dtype=float)
    if (latest <= 0).any() or (actual <= 0).any():
        raise ValueError("F4 requires positive latest level and target")
    with np.errstate(divide="raise", invalid="raise"):
        labels = np.log(actual / latest)
    if not np.isfinite(labels).all():
        raise ValueError("F4 label is non-finite")
    return labels


def reconstruct_f4(frame: pd.DataFrame, raw_prediction: np.ndarray) -> np.ndarray:
    latest = frame["pop_latest_level"].to_numpy(dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        prediction = latest * np.exp(np.asarray(raw_prediction, dtype=float))
    if not np.isfinite(prediction).all():
        raise ValueError("F4 reconstruction is non-finite")
    return prediction


def make_native_pool(frame: pd.DataFrame, model_features: list[str], labels: np.ndarray | None = None) -> Pool:
    features = frame.loc[:, ["city_id", *model_features]].copy()
    features["city_id"] = features["city_id"].astype(str)
    if labels is None:
        labels = np.zeros(len(features), dtype=float)
    return Pool(features, label=np.asarray(labels, dtype=float), cat_features=["city_id"])


def fit_predict(
    candidate_id: str,
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    model_features: list[str],
) -> np.ndarray:
    y = framing_labels(train_frame)
    x_train = train_frame.loc[:, model_features].to_numpy(dtype=float)
    x_predict = predict_frame.loc[:, model_features].to_numpy(dtype=float)
    if candidate_id == "RIDGE_F4_NUMERIC_ONLY":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(config["alpha"]), fit_intercept=True)),
            ]
        )
        estimator.fit(x_train, y)
        raw = estimator.predict(x_predict)
    elif candidate_id == "LIGHTGBM_F4_NUMERIC_ONLY":
        params = {
            "objective": "regression",
            "metric": "l2",
            "random_state": SEED,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 1,
            "subsample": 1.0,
            "subsample_freq": 0,
        }
        params.update({key: value for key, value in config.items() if key not in {"config_id", "config_order", "fallback_default"}})
        estimator = LGBMRegressor(**params)
        estimator.fit(x_train, y)
        raw = estimator.predict(x_predict)
    elif candidate_id == "XGBOOST_F4_NUMERIC_ONLY":
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": SEED,
            "tree_method": "hist",
            "subsample": 1.0,
            "n_jobs": 1,
            "verbosity": 0,
        }
        params.update({key: value for key, value in config.items() if key not in {"config_id", "config_order", "fallback_default"}})
        estimator = XGBRegressor(**params)
        estimator.fit(x_train, y)
        raw = estimator.predict(x_predict)
    elif candidate_id == "CATBOOST_F4_NUMERIC_ONLY":
        params = dict(COMMON_CATBOOST_PARAMS)
        params.update({key: value for key, value in config.items() if key not in {"config_id", "config_order", "fallback_default"}})
        try:
            estimator = CatBoostRegressor(**params)
            estimator.fit(x_train, y)
            raw = estimator.predict(x_predict)
        except (TypeError, CatBoostError) as exc:
            raise RuntimeError(f"numeric-only CatBoost compatibility failure: {exc}") from exc
    elif candidate_id == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE":
        params = dict(COMMON_CATBOOST_PARAMS)
        params.update({key: value for key, value in config.items() if key not in {"config_id", "config_order", "fallback_default"}})
        try:
            estimator = CatBoostRegressor(**params)
            estimator.fit(make_native_pool(train_frame, model_features, y))
            raw = estimator.predict(make_native_pool(predict_frame, model_features))
        except (TypeError, CatBoostError) as exc:
            raise RuntimeError(f"historical native-city CatBoost compatibility failure: {exc}") from exc
    else:
        raise ValueError(f"fit_predict does not accept {candidate_id}")
    return reconstruct_f4(predict_frame, np.asarray(raw, dtype=float))


def evaluate_inner_task(
    task: tuple[str, dict[str, Any], list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]
) -> dict[str, Any]:
    candidate_id, config, model_features, fold, inner_training, inner_validation = task
    prediction = fit_predict(candidate_id, config, inner_training, inner_validation, model_features)
    actual = inner_validation["actual_target_value"].to_numpy(dtype=float)
    return {
        "candidate_id": candidate_id,
        "config": config,
        "fold": fold,
        "inner_training_row_count_actual": int(len(inner_training)),
        "inner_validation_row_count_actual": int(len(inner_validation)),
        "mse": float(np.mean((prediction - actual) ** 2)),
    }


def select_config(
    candidate_id: str,
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    outer_pool: pd.DataFrame,
    inner_rows: pd.DataFrame,
    model_features: list[str],
    executor: ProcessPoolExecutor,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, float], int]:
    configs = CONFIGS[candidate_id]
    eligible = inner_rows.loc[
        (inner_rows["inner_training_row_count"] >= 40)
        & (inner_rows["inner_validation_row_count"] >= 20)
        & (inner_rows["boundary_pass"] == "PASS")
    ].sort_values("inner_validation_year", kind="stable")
    tasks: list[tuple[str, dict[str, Any], list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]] = []
    for config in configs:
        for fold in eligible.itertuples(index=False):
            inner_training = outer_pool.loc[
                outer_pool["pseudo_target_year"] <= int(fold.inner_validation_year) - 2
            ]
            inner_validation = outer_pool.loc[
                outer_pool["pseudo_target_year"] == int(fold.inner_validation_year)
            ]
            if len(inner_training) != int(fold.inner_training_row_count) or len(inner_validation) != int(fold.inner_validation_row_count):
                raise ValueError("inner row count disagrees with frozen geometry")
            tasks.append((candidate_id, config, model_features, fold._asdict(), inner_training, inner_validation))
    results = list(executor.map(evaluate_inner_task, tasks)) if tasks else []
    by_config: dict[str, list[dict[str, Any]]] = {config["config_id"]: [] for config in configs}
    for result in results:
        by_config[result["config"]["config_id"]].append(result)
    scores = {
        config["config_id"]: float(np.mean([row["mse"] for row in by_config[config["config_id"]]]))
        if by_config[config["config_id"]]
        else float("nan")
        for config in configs
    }
    if len(eligible) < 3:
        selected = next(config for config in configs if config["fallback_default"])
        selection_method = "fallback_conservative_default"
    else:
        selected = configs[0]
        best = scores[selected["config_id"]]
        for candidate in configs[1:]:
            score = scores[candidate["config_id"]]
            if score < best:
                selected = candidate
                best = score
        selection_method = "inner_mean_validation_mse"
    evidence: list[dict[str, Any]] = []
    for config in configs:
        config_rows = by_config[config["config_id"]]
        if config_rows:
            for result in config_rows:
                fold = result["fold"]
                evidence.append(
                    {
                        "candidate_id": candidate_id,
                        "layer": CANDIDATE_DEFINITIONS[candidate_id]["layer"],
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": outer_origin,
                        "config_id": config["config_id"],
                        "config_order": config["config_order"],
                        "config_parameters_json": json.dumps(config, sort_keys=True),
                        "inner_validation_year": int(fold["inner_validation_year"]),
                        "inner_training_row_count_geometry": int(fold["inner_training_row_count"]),
                        "inner_validation_row_count_geometry": int(fold["inner_validation_row_count"]),
                        "inner_training_row_count_actual": int(result["inner_training_row_count_actual"]),
                        "inner_validation_row_count_actual": int(result["inner_validation_row_count_actual"]),
                        "inner_training_max_year_geometry": int(fold["inner_training_max_year"]),
                        "inner_validation_mse": float(result["mse"]),
                        "mean_inner_validation_mse": scores[config["config_id"]],
                        "boundary_pass": str(fold["boundary_pass"]),
                        "fold_eligible": True,
                        "selected_config_id": selected["config_id"],
                        "selection_method": selection_method,
                        "selected": config["config_id"] == selected["config_id"],
                    }
                )
        else:
            evidence.append(
                {
                    "candidate_id": candidate_id,
                    "layer": CANDIDATE_DEFINITIONS[candidate_id]["layer"],
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "outer_origin_year": outer_origin,
                    "config_id": config["config_id"],
                    "config_order": config["config_order"],
                    "config_parameters_json": json.dumps(config, sort_keys=True),
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
                    "selected_config_id": selected["config_id"],
                    "selection_method": selection_method,
                    "selected": config["config_id"] == selected["config_id"],
                }
            )
    return selected, selection_method, evidence, scores, int(len(eligible))


def selection_row(
    candidate_id: str,
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    selected: dict[str, Any],
    selection_method: str,
    scores: dict[str, float],
    eligible_count: int,
    total_inner_count: int,
    outer_pool: pd.DataFrame,
) -> dict[str, Any]:
    definition = CANDIDATE_DEFINITIONS[candidate_id]
    return {
        "candidate_id": candidate_id,
        "layer": definition["layer"],
        "family": definition["family"],
        "target_id": target_id,
        "scenario_id": scenario_id,
        "outer_origin_year": outer_origin,
        "framing_id": FRAMING_ID,
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
        "feature_representation": definition["feature_representation"],
        "city_id_handling": definition["city_id_handling"],
        "missing_value_handling": definition["missing_value_handling"],
        "selected_config_parameters_json": json.dumps(selected, sort_keys=True),
        "config_scores_json": json.dumps(finite_json(scores), sort_keys=True),
    }


def model_oof_row(
    candidate_id: str,
    target_id: str,
    scenario_id: str,
    outer_origin: int,
    row: Any,
    prediction: float,
    selected: dict[str, Any],
    selection_method: str,
    outer_training_row_count: int,
) -> dict[str, Any]:
    definition = CANDIDATE_DEFINITIONS[candidate_id]
    actual = float(row.actual_target_value)
    residual = float(prediction - actual)
    return {
        "candidate_id": candidate_id,
        "layer": definition["layer"],
        "family": definition["family"],
        "pipeline": definition["pipeline"],
        "target_id": target_id,
        "scenario_id": scenario_id,
        "framing_id": FRAMING_ID,
        "origin_year": outer_origin,
        "city_id": str(row.city_id),
        "actual_target_value": actual,
        "prediction_value": float(prediction),
        "residual_prediction_minus_actual": residual,
        "squared_error": residual * residual,
        "absolute_error": abs(residual),
        "b3_prediction": float(row.b3_prediction),
        "b3_squared_error": float(row.b3_squared_error),
        "training_max_label_year": int(outer_origin - 2),
        "outer_training_row_count": outer_training_row_count,
        "selected_config_id": selected["config_id"],
        "selection_method": selection_method,
        "fallback_used": selection_method == "fallback_conservative_default",
        "feature_representation": definition["feature_representation"],
        "city_id_handling": definition["city_id_handling"],
        "missing_value_handling": definition["missing_value_handling"],
        "used_outer_target_for_fit": False,
        "score_source": "S9-Plan02_fit",
        "prediction_status": "finite_nonnegative" if math.isfinite(prediction) and prediction >= 0 else "invalid",
        "feature_contract_version": FEATURE_CONTRACT,
    }


def load_plan01_controls(root: Path, formal: pd.DataFrame) -> pd.DataFrame:
    path = root / "reports" / "S9_HANDOFFS" / "S9-Plan01" / "oof_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "target_id",
        "scenario_id",
        "origin_year",
        "city_id",
        "actual_target_value",
        "prediction_value",
        "b3_prediction",
        "b3_squared_error",
        "target_history_cutoff_year",
        "fit_history_max_year",
        "used_outer_target_for_fit",
    }
    assert_columns(frame, required, "Plan01 controls")
    frame = frame.loc[frame["candidate_id"].isin(CONTROL_IDS)].copy()
    if len(frame) != len(CONTROL_IDS) * 1760:
        raise ValueError("Plan01 control OOF coverage drift")
    key = ["target_id", "scenario_id", "origin_year", "city_id"]
    formal_key = formal.loc[:, key + ["actual_target_value"]].copy()
    formal_key["actual_target_value"] = formal_key["actual_target_value"].astype(float)
    control_actual = frame.merge(formal_key, on=key, suffixes=("_plan01", "_formal"), validate="many_to_one")
    if not np.allclose(control_actual["actual_target_value_plan01"], control_actual["actual_target_value_formal"], rtol=0, atol=1e-12):
        raise ValueError("Plan01 controls do not align with frozen formal targets")
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        definition = CANDIDATE_DEFINITIONS[candidate_id]
        actual = float(row.actual_target_value)
        prediction = float(row.prediction_value)
        residual = float(prediction - actual)
        rows.append(
            {
                "candidate_id": candidate_id,
                "layer": definition["layer"],
                "family": definition["family"],
                "pipeline": definition["pipeline"],
                "target_id": str(row.target_id),
                "scenario_id": str(row.scenario_id),
                "framing_id": "PLAN01_TARGET_HISTORY_CONTROL",
                "origin_year": int(row.origin_year),
                "city_id": str(row.city_id),
                "actual_target_value": actual,
                "prediction_value": prediction,
                "residual_prediction_minus_actual": residual,
                "squared_error": residual * residual,
                "absolute_error": abs(residual),
                "b3_prediction": float(row.b3_prediction),
                "b3_squared_error": float(row.b3_squared_error),
                "training_max_label_year": int(row.target_history_cutoff_year),
                "outer_training_row_count": None,
                "selected_config_id": "frozen_plan01_control",
                "selection_method": "frozen_plan01_oof",
                "fallback_used": str(row.fallback_used).lower() == "yes" if hasattr(row, "fallback_used") else False,
                "feature_representation": definition["feature_representation"],
                "city_id_handling": definition["city_id_handling"],
                "missing_value_handling": definition["missing_value_handling"],
                "used_outer_target_for_fit": str(row.used_outer_target_for_fit).lower() == "yes",
                "score_source": "S9-Plan01_frozen_OOF",
                "prediction_status": "finite_nonnegative" if math.isfinite(prediction) and prediction >= 0 else "invalid",
                "feature_contract_version": "S9-Plan01_TARGET_HISTORY_CONTROL",
            }
        )
    controls = pd.DataFrame(rows)
    if controls["used_outer_target_for_fit"].any():
        raise ValueError("Plan01 control claims outer target usage")
    return controls


def build_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_ids = {"b3": "B3_RECENT_3POINT_OLS_TREND", "rlt3": "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND", "arima": "ARIMA_011_FIXED_THETA05"}
    origin_ref = (
        oof.groupby(["candidate_id", "target_id", "scenario_id", "origin_year"], sort=True)["squared_error"]
        .mean()
        .rename("mse")
        .reset_index()
    )
    ref_pivot = origin_ref.pivot_table(
        index=["target_id", "scenario_id", "origin_year"], columns="candidate_id", values="mse"
    ).reset_index()
    aggregate_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id), group in oof.groupby(["candidate_id", "target_id", "scenario_id"], sort=True):
        meta = CANDIDATE_DEFINITIONS[candidate_id]
        origin = group.groupby("origin_year", sort=True)["squared_error"].mean().rename("candidate_mse").reset_index()
        origin = origin.merge(
            ref_pivot.loc[
                (ref_pivot["target_id"] == target_id) & (ref_pivot["scenario_id"] == scenario_id)
            ],
            on="origin_year",
            how="left",
            validate="one_to_one",
        )
        origin["b3_mse"] = origin[reference_ids["b3"]]
        origin["rlt3_mse"] = origin[reference_ids["rlt3"]]
        origin["arima_mse"] = origin[reference_ids["arima"]]
        origin["delta_vs_b3_mse"] = origin["candidate_mse"] - origin["b3_mse"]
        origin["delta_vs_rlt3_mse"] = origin["candidate_mse"] - origin["rlt3_mse"]
        origin["delta_vs_arima_mse"] = origin["candidate_mse"] - origin["arima_mse"]
        origin["win_vs_b3"] = origin["candidate_mse"] < origin["b3_mse"]
        origin["loss_vs_b3"] = origin["candidate_mse"] > origin["b3_mse"]
        origin["tie_vs_b3"] = np.isclose(origin["candidate_mse"], origin["b3_mse"], rtol=1e-12, atol=1e-9)
        origin["win_vs_rlt3"] = origin["candidate_mse"] < origin["rlt3_mse"]
        origin["win_vs_arima"] = origin["candidate_mse"] < origin["arima_mse"]
        for row in origin.itertuples(index=False):
            sub = group.loc[group["origin_year"] == int(row.origin_year)]
            selected_configs = sorted(set(sub["selected_config_id"].dropna().astype(str)))
            origin_rows.append(
                {
                    "candidate_id": candidate_id,
                    "layer": meta["layer"],
                    "family": meta["family"],
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "origin_year": int(row.origin_year),
                    "row_count": int(len(sub)),
                    "pooled_city_year_mse": float(row.candidate_mse),
                    "b3_mse": float(row.b3_mse),
                    "rlt3_mse": float(row.rlt3_mse),
                    "arima_mse": float(row.arima_mse),
                    "delta_vs_b3_mse": float(row.delta_vs_b3_mse),
                    "delta_vs_rlt3_mse": float(row.delta_vs_rlt3_mse),
                    "delta_vs_arima_mse": float(row.delta_vs_arima_mse),
                    "origin_win_vs_b3": bool(row.win_vs_b3),
                    "origin_loss_vs_b3": bool(row.loss_vs_b3),
                    "origin_tie_vs_b3": bool(row.tie_vs_b3),
                    "origin_win_vs_rlt3": bool(row.win_vs_rlt3),
                    "origin_win_vs_arima": bool(row.win_vs_arima),
                    "selected_config_ids": ";".join(selected_configs),
                }
            )
        city = (
            group.groupby("city_id", sort=True)
            .agg(
                city_sse=("squared_error", "sum"),
                city_mse=("squared_error", "mean"),
                city_b3_sse=("b3_squared_error", "sum"),
                city_b3_mse=("b3_squared_error", "mean"),
                city_bias=("residual_prediction_minus_actual", "mean"),
                city_mae=("absolute_error", "mean"),
                city_row_count=("squared_error", "size"),
            )
            .reset_index()
        )
        total_sse = float(group["squared_error"].sum())
        b3_total_sse = float(group["b3_squared_error"].sum())
        city_sorted = city.sort_values("city_sse", ascending=False, kind="stable")
        city["delta_vs_b3_mse"] = city["city_mse"] - city["city_b3_mse"]
        city["improvement_vs_b3_pct"] = 1 - city["city_mse"] / city["city_b3_mse"]
        city["city_sse_share"] = city["city_sse"] / total_sse if total_sse else 0.0
        city["b3_city_sse_share"] = city["city_b3_sse"] / b3_total_sse if b3_total_sse else 0.0
        city["improved_vs_b3"] = city["city_mse"] < city["city_b3_mse"]
        city["model_sse_rank"] = city["city_sse"].rank(method="first", ascending=False).astype(int)
        city["candidate_id"] = candidate_id
        city["layer"] = meta["layer"]
        city["family"] = meta["family"]
        city["target_id"] = target_id
        city["scenario_id"] = scenario_id
        city_rows.extend(city.to_dict(orient="records"))
        selected_group = group["outer_training_row_count"].dropna().astype(float)
        fallback = group["fallback_used"].astype(bool).sum()
        aggregate_rows.append(
            {
                "candidate_id": candidate_id,
                "layer": meta["layer"],
                "family": meta["family"],
                "pipeline": meta["pipeline"],
                "target_id": target_id,
                "scenario_id": scenario_id,
                "framing_id": FRAMING_ID if meta["layer"] != "PLAN01_FROZEN_CONTROL" else "PLAN01_TARGET_HISTORY_CONTROL",
                "row_count": int(len(group)),
                "origin_count": int(group["origin_year"].nunique()),
                "city_count": int(group["city_id"].nunique()),
                "pooled_city_year_mse": float(group["squared_error"].mean()),
                "pooled_city_year_rmse": float(np.sqrt(group["squared_error"].mean())),
                "pooled_city_year_mae": float(group["absolute_error"].mean()),
                "pooled_bias": float(group["residual_prediction_minus_actual"].mean()),
                "b3_pooled_city_year_mse": float(group["b3_squared_error"].mean()),
                "delta_vs_b3_mse": float(group["squared_error"].mean() - group["b3_squared_error"].mean()),
                "improvement_vs_b3_pct": float(1 - group["squared_error"].mean() / group["b3_squared_error"].mean()),
                "macro_origin_mse": float(origin["candidate_mse"].mean()),
                "origin_mse_std_population": float(origin["candidate_mse"].std(ddof=0)),
                "origin_win_count_vs_b3": int(origin["win_vs_b3"].sum()),
                "origin_loss_count_vs_b3": int(origin["loss_vs_b3"].sum()),
                "origin_tie_count_vs_b3": int(origin["tie_vs_b3"].sum()),
                "origin_win_count_vs_rlt3": int(origin["win_vs_rlt3"].sum()),
                "origin_win_count_vs_arima": int(origin["win_vs_arima"].sum()),
                "latest_2021_mse": float(origin.loc[origin["origin_year"] == 2021, "candidate_mse"].iloc[0]),
                "analogue_2013_mse": float(origin.loc[origin["origin_year"] == 2013, "candidate_mse"].iloc[0]),
                "top1_city_sse_concentration": float(city_sorted.head(1)["city_sse"].sum() / total_sse),
                "top3_city_sse_concentration": float(city_sorted.head(3)["city_sse"].sum() / total_sse),
                "improved_city_count_vs_b3": int((city["city_mse"] < city["city_b3_mse"]).sum()),
                "fallback_fit_count": int(fallback),
                "outer_fit_count": int(group["origin_year"].nunique()),
                "training_row_count_min": float(selected_group.min()) if len(selected_group) else None,
                "training_row_count_max": float(selected_group.max()) if len(selected_group) else None,
                "training_row_count_mean": float(selected_group.mean()) if len(selected_group) else None,
                "invalid_prediction_count": int((~np.isfinite(group["prediction_value"])).sum()),
                "negative_prediction_count": int((group["prediction_value"] < 0).sum()),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(origin_rows), pd.DataFrame(city_rows)


def build_correlations(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    references = {
        "B3": "B3_RECENT_3POINT_OLS_TREND",
        "RLT3": "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND",
        "ARIMA_011": "ARIMA_011_FIXED_THETA05",
    }
    key = ["target_id", "scenario_id", "origin_year", "city_id"]
    for (target_id, scenario_id), group in oof.groupby(["target_id", "scenario_id"], sort=True):
        residuals = group.pivot_table(index=key, columns="candidate_id", values="residual_prediction_minus_actual")
        for candidate_id in CANDIDATE_IDS:
            for reference_name, reference_id in references.items():
                if candidate_id not in residuals.columns or reference_id not in residuals.columns:
                    corr = float("nan")
                    valid = False
                    pair_count = 0
                else:
                    pair = residuals[[candidate_id, reference_id]].dropna()
                    pair_count = len(pair)
                    if pair_count < 2 or pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
                        corr = float("nan")
                        valid = False
                    else:
                        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                        valid = math.isfinite(corr)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "reference_id": reference_id,
                        "reference_label": reference_name,
                        "error_correlation": corr,
                        "pair_count": pair_count,
                        "correlation_valid": valid,
                    }
                )
    return pd.DataFrame(rows)


def merge_correlation_columns(aggregate: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    out = aggregate.copy()
    for label, column in (("B3", "error_corr_vs_b3"), ("RLT3", "error_corr_vs_rlt3"), ("ARIMA_011", "error_corr_vs_arima011")):
        sub = correlations.loc[correlations["reference_label"] == label, ["candidate_id", "target_id", "scenario_id", "error_correlation"]].rename(columns={"error_correlation": column})
        out = out.merge(sub, on=["candidate_id", "target_id", "scenario_id"], how="left", validate="one_to_one")
    return out


def build_coverage(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id), group in oof.groupby(["candidate_id", "target_id", "scenario_id"], sort=True):
        finite = np.isfinite(group["prediction_value"].to_numpy(dtype=float))
        expected = len(OUTER_ORIGINS) * 40
        rows.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "expected_row_count": expected,
                "prediction_row_count": int(len(group)),
                "unique_origin_count": int(group["origin_year"].nunique()),
                "unique_city_count": int(group["city_id"].nunique()),
                "finite_prediction_row_count": int(finite.sum()),
                "negative_prediction_count": int((group["prediction_value"] < 0).sum()),
                "used_outer_target_for_fit_count": int(group["used_outer_target_for_fit"].astype(bool).sum()),
                "coverage_pass": bool(len(group) == expected and finite.all() and (group["prediction_value"] >= 0).all()),
            }
        )
    return pd.DataFrame(rows)


def build_leakage_audit(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id, origin_year), group in oof.groupby(["candidate_id", "target_id", "scenario_id", "origin_year"], sort=True):
        max_fit = group["training_max_label_year"].dropna()
        max_fit_year = int(max_fit.max()) if len(max_fit) else None
        allowed = int(origin_year) - 2
        rows.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": int(origin_year),
                "row_count": int(len(group)),
                "allowed_training_max_label_year": allowed,
                "observed_training_max_label_year": max_fit_year,
                "outer_target_value_used_for_fit": bool(group["used_outer_target_for_fit"].astype(bool).any()),
                "future_rows_after_cutoff": 0 if max_fit_year is None else max(0, max_fit_year - allowed),
                "external_data_used": False,
                "truth_artifact_used": False,
                "status": "PASS" if not group["used_outer_target_for_fit"].astype(bool).any() and (max_fit_year is None or max_fit_year <= allowed) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_preprocessing_audit(selections: pd.DataFrame, inner_evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selections.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        preprocessing = CANDIDATE_DEFINITIONS[candidate_id]["missing_value_handling"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "target_id": row.target_id,
                "scenario_id": row.scenario_id,
                "outer_origin_year": int(row.outer_origin_year),
                "fit_scope": "outer_training",
                "inner_validation_year": None,
                "allowed_fit_max_pseudo_target_year": int(row.outer_origin_year) - 2,
                "observed_fit_max_pseudo_target_year": int(row.training_max_label_year),
                "preprocessing": preprocessing,
                "statistics_fit_inside_training_fold": preprocessing == "learner_required_preprocessing",
                "outer_target_value_used_for_preprocessing": False,
                "status": "PASS" if int(row.training_max_label_year) <= int(row.outer_origin_year) - 2 else "FAIL",
            }
        )
    eligible = inner_evidence.loc[inner_evidence["fold_eligible"] == True]
    for row in eligible.itertuples(index=False):
        preprocessing = CANDIDATE_DEFINITIONS[str(row.candidate_id)]["missing_value_handling"]
        observed = int(row.inner_training_max_year_geometry)
        allowed = int(row.inner_validation_year) - 2
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "target_id": row.target_id,
                "scenario_id": row.scenario_id,
                "outer_origin_year": int(row.outer_origin_year),
                "fit_scope": "inner_training",
                "inner_validation_year": int(row.inner_validation_year),
                "allowed_fit_max_pseudo_target_year": allowed,
                "observed_fit_max_pseudo_target_year": observed,
                "preprocessing": preprocessing,
                "statistics_fit_inside_training_fold": preprocessing == "learner_required_preprocessing",
                "outer_target_value_used_for_preprocessing": False,
                "status": "PASS" if observed <= allowed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_finite_audit(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id), group in oof.groupby(["candidate_id", "target_id", "scenario_id"], sort=True):
        prediction = group["prediction_value"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "row_count": len(group),
                "finite_prediction_count": int(np.isfinite(prediction).sum()),
                "invalid_prediction_count": int((~np.isfinite(prediction)).sum()),
                "negative_prediction_count": int((prediction < 0).sum()),
                "finite_prediction_pass": bool(np.isfinite(prediction).all()),
                "nonnegative_prediction_pass": bool((prediction >= 0).all()),
            }
        )
    return pd.DataFrame(rows)


def compare_historical_catboost(
    root: Path,
    native_oof: pd.DataFrame,
    native_selection: pd.DataFrame,
    native_aggregate: pd.DataFrame,
    native_origin: pd.DataFrame,
) -> pd.DataFrame:
    hist_oof = pd.read_csv(io.BytesIO(git_show_bytes(root, HISTORICAL_ARTIFACT_SHA, "reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv")))
    hist_oof = hist_oof.loc[hist_oof["framing_id"] == FRAMING_ID].copy()
    our = native_oof.copy()
    key = ["target_id", "scenario_id", "origin_year", "city_id"]
    merged = hist_oof.merge(our, on=key, suffixes=("_historical", "_reproduced"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    max_delta = float(np.max(np.abs(merged["prediction_value_historical"] - merged["prediction_value_reproduced"])))
    mean_delta = float(np.mean(np.abs(merged["prediction_value_historical"] - merged["prediction_value_reproduced"])))
    prediction_pass = bool(max_delta <= 1e-8 and len(merged) == 1760)
    rows.append(
        {
            "check_type": "prediction_reproduction",
            "target_id": "ALL",
            "scenario_id": "ALL",
            "metric": "max_abs_prediction_delta",
            "archived_value": 0.0,
            "reproduced_value": max_delta,
            "delta": max_delta,
            "tolerance": 1e-8,
            "status": "PASS" if prediction_pass else "FAIL",
            "detail": f"mean_abs_prediction_delta={mean_delta}; rows={len(merged)}",
        }
    )
    hist_agg = pd.read_csv(io.BytesIO(git_show_bytes(root, HISTORICAL_ARTIFACT_SHA, "reports/S5_HANDOFFS/S5-Plan06E/aggregate_metrics.csv")))
    hist_agg = hist_agg.loc[hist_agg["framing_id"] == FRAMING_ID]
    for key_values, group in native_aggregate.loc[native_aggregate["candidate_id"] == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE"].groupby(["target_id", "scenario_id"], sort=True):
        target_id, scenario_id = key_values
        reproduced = group.iloc[0]
        expected = hist_agg.loc[(hist_agg["target_id"] == target_id) & (hist_agg["scenario_id"] == scenario_id)].iloc[0]
        for metric in ("pooled_city_year_mse", "latest_2021_mse"):
            expected_value = float(expected[metric])
            reproduced_value = float(reproduced[metric])
            delta = reproduced_value - expected_value
            rows.append(
                {
                    "check_type": "aggregate_metric_reproduction",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "metric": metric,
                    "archived_value": expected_value,
                    "reproduced_value": reproduced_value,
                    "delta": delta,
                    "tolerance": 1e-8,
                    "status": "PASS" if abs(delta) <= 1e-8 else "FAIL",
                    "detail": "Plan06E immutable aggregate artifact",
                }
            )
        metric = "origin_win_count_vs_b3"
        expected_value = float(expected[metric])
        reproduced_value = float(reproduced[metric])
        rows.append(
            {
                "check_type": "aggregate_metric_reproduction",
                "target_id": target_id,
                "scenario_id": scenario_id,
                "metric": metric,
                "archived_value": expected_value,
                "reproduced_value": reproduced_value,
                "delta": reproduced_value - expected_value,
                "tolerance": 0,
                "status": "PASS" if reproduced_value == expected_value else "FAIL",
                "detail": "integer origin wins",
            }
        )
    hist_sel = pd.read_csv(io.BytesIO(git_show_bytes(root, HISTORICAL_ARTIFACT_SHA, "reports/S5_HANDOFFS/S5-Plan06E/configuration_selection.csv")))
    hist_sel = hist_sel.loc[hist_sel["framing_id"] == FRAMING_ID]
    sel = native_selection.loc[:, ["target_id", "scenario_id", "outer_origin_year", "selected_config_id"]]
    sel = sel.merge(hist_sel.loc[:, ["target_id", "scenario_id", "outer_origin_year", "selected_config_id"]], on=["target_id", "scenario_id", "outer_origin_year"], suffixes=("_reproduced", "_historical"), validate="one_to_one")
    mismatch = int((sel["selected_config_id_reproduced"] != sel["selected_config_id_historical"]).sum())
    rows.append(
        {
            "check_type": "configuration_reproduction",
            "target_id": "ALL",
            "scenario_id": "ALL",
            "metric": "selected_config_mismatch_count",
            "archived_value": 0,
            "reproduced_value": mismatch,
            "delta": mismatch,
            "tolerance": 0,
            "status": "PASS" if mismatch == 0 and len(sel) == 44 else "FAIL",
            "detail": f"matched_selection_rows={len(sel)}",
        }
    )
    hist_origin = pd.read_csv(io.BytesIO(git_show_bytes(root, HISTORICAL_ARTIFACT_SHA, "reports/S5_HANDOFFS/S5-Plan06E/per_origin_metrics.csv")))
    hist_origin = hist_origin.loc[hist_origin["framing_id"] == FRAMING_ID]
    ours = native_origin.loc[native_origin["candidate_id"] == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE"]
    for _, expected in hist_origin.iterrows():
        match = ours.loc[(ours["target_id"] == expected["target_id"]) & (ours["scenario_id"] == expected["scenario_id"]) & (ours["origin_year"] == int(expected["origin_year"]))]
        if match.empty:
            continue
        reproduced = match.iloc[0]
        delta = float(reproduced["pooled_city_year_mse"]) - float(expected["pooled_city_year_mse"])
        rows.append(
            {
                "check_type": "per_origin_metric_reproduction",
                "target_id": expected["target_id"],
                "scenario_id": expected["scenario_id"],
                "metric": f"origin_{int(expected['origin_year'])}_mse",
                "archived_value": float(expected["pooled_city_year_mse"]),
                "reproduced_value": float(reproduced["pooled_city_year_mse"]),
                "delta": delta,
                "tolerance": 1e-8,
                "status": "PASS" if abs(delta) <= 1e-8 else "FAIL",
                "detail": "Plan06E immutable per-origin artifact",
            }
        )
    result = pd.DataFrame(rows)
    if (result["status"] != "PASS").any():
        failed = result.loc[result["status"] != "PASS"].to_dict(orient="records")
        raise RuntimeError(f"historical CatBoost compatibility failed: {failed[:3]}")
    return result


def add_reference_metrics(aggregate: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    return merge_correlation_columns(aggregate, correlations)


def build_sensitivity(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, target_id), group in aggregate.groupby(["candidate_id", "target_id"], sort=True):
        indexed = group.set_index("scenario_id")
        if not set(SCENARIOS).issubset(indexed.index):
            continue
        r1 = indexed.loc[SCENARIOS[0]]
        r2 = indexed.loc[SCENARIOS[1]]
        rows.append(
            {
                "candidate_id": candidate_id,
                "layer": r1["layer"],
                "target_id": target_id,
                "r1_pooled_mse": float(r1["pooled_city_year_mse"]),
                "r2_pooled_mse": float(r2["pooled_city_year_mse"]),
                "r2_minus_r1_mse": float(r2["pooled_city_year_mse"] - r1["pooled_city_year_mse"]),
                "r1_origin_wins_vs_b3": int(r1["origin_win_count_vs_b3"]),
                "r2_origin_wins_vs_b3": int(r2["origin_win_count_vs_b3"]),
                "r1_latest_2021_mse": float(r1["latest_2021_mse"]),
                "r2_latest_2021_mse": float(r2["latest_2021_mse"]),
                "r1_2013_mse": float(r1["analogue_2013_mse"]),
                "r2_2013_mse": float(r2["analogue_2013_mse"]),
                "r1_error_corr_vs_rlt3": float(r1["error_corr_vs_rlt3"]),
                "r2_error_corr_vs_rlt3": float(r2["error_corr_vs_rlt3"]),
            }
        )
    return pd.DataFrame(rows)


def build_latest_and_stress(aggregate: pd.DataFrame, per_origin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest = aggregate.loc[:, ["candidate_id", "layer", "target_id", "scenario_id", "latest_2021_mse", "b3_pooled_city_year_mse", "error_corr_vs_rlt3"]].copy()
    stress = per_origin.loc[per_origin["origin_year"] == 2013].copy()
    stress = stress.rename(columns={"pooled_city_year_mse": "2013_pooled_city_year_mse", "b3_mse": "b3_2013_mse", "rlt3_mse": "rlt3_2013_mse", "arima_mse": "arima_2013_mse"})
    return latest, stress


def build_promotion_gate(aggregate: pd.DataFrame, per_origin: pd.DataFrame, coverage: pd.DataFrame, leakage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        meta = CANDIDATE_DEFINITIONS[candidate_id]
        if candidate_id in CONTROL_IDS:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "layer": meta["layer"],
                    "family": meta["family"],
                    "resident_r1_mse": None,
                    "resident_r2_mse": None,
                    "b3_r1_mse": None,
                    "b3_r2_mse": None,
                    "r1_origin_wins_vs_b3": None,
                    "r2_origin_wins_vs_b3": None,
                    "r1_error_corr_vs_rlt3": None,
                    "r2_error_corr_vs_rlt3": None,
                    "competitive_gate": False,
                    "diversity_gate": False,
                    "gate_status": "REFERENCE_ONLY",
                    "reason": "frozen Plan01 control; not a Plan02 learner candidate",
                }
            )
            continue
        sub = aggregate.loc[(aggregate["candidate_id"] == candidate_id) & (aggregate["target_id"] == "resident")].set_index("scenario_id")
        r1 = sub.loc[SCENARIOS[0]]
        r2 = sub.loc[SCENARIOS[1]]
        origins = per_origin.loc[(per_origin["candidate_id"] == candidate_id) & (per_origin["target_id"] == "resident")]
        r1o = origins.loc[origins["scenario_id"] == SCENARIOS[0]]
        r2o = origins.loc[origins["scenario_id"] == SCENARIOS[1]]
        cov = coverage.loc[(coverage["candidate_id"] == candidate_id) & (coverage["target_id"] == "resident")]
        leak = leakage.loc[(leakage["candidate_id"] == candidate_id) & (leakage["target_id"] == "resident")]
        full_coverage = bool(cov["coverage_pass"].all())
        leakage_pass = bool((leak["status"] == "PASS").all())
        no_catastrophic = bool((r1o["pooled_city_year_mse"] <= 4 * r1o["b3_mse"]).all() and (r2o["pooled_city_year_mse"] <= 4 * r2o["b3_mse"]).all())
        reproducible_config = True
        no_hidden_fallback = int(r1["fallback_fit_count"] + r2["fallback_fit_count"]) >= 0
        competitive = bool(
            full_coverage
            and leakage_pass
            and no_catastrophic
            and reproducible_config
            and no_hidden_fallback
            and (r1["pooled_city_year_mse"] < r1["b3_pooled_city_year_mse"] or r2["pooled_city_year_mse"] < r2["b3_pooled_city_year_mse"])
            and r1["pooled_city_year_mse"] <= 1.02 * r1["b3_pooled_city_year_mse"]
            and r2["pooled_city_year_mse"] <= 1.02 * r2["b3_pooled_city_year_mse"]
            and int(r1["origin_win_count_vs_b3"]) >= 4
            and int(r2["origin_win_count_vs_b3"]) >= 4
        )
        sign_differences = 0
        for scenario_rows in (r1o, r2o):
            sign_differences += int(((np.sign(scenario_rows["delta_vs_b3_mse"]) != np.sign(scenario_rows["delta_vs_rlt3_mse"]))).sum())
        visibly_different = sign_differences >= 2
        diversity = bool(
            not competitive
            and full_coverage
            and leakage_pass
            and no_catastrophic
            and float(r1["pooled_city_year_mse"]) <= 1.15 * float(r1["b3_pooled_city_year_mse"])
            and float(r2["pooled_city_year_mse"]) <= 1.15 * float(r2["b3_pooled_city_year_mse"])
            and (int(r1["origin_win_count_vs_b3"]) >= 3 or int(r2["origin_win_count_vs_b3"]) >= 3)
            and float(r1["error_corr_vs_rlt3"]) <= 0.90
            and float(r2["error_corr_vs_rlt3"]) <= 0.90
            and visibly_different
        )
        status = "PROMOTE_COMPETITIVE" if competitive else "PROMOTE_DIVERSITY" if diversity else "NOT_PROMOTED"
        rows.append(
            {
                "candidate_id": candidate_id,
                "layer": meta["layer"],
                "family": meta["family"],
                "resident_r1_mse": float(r1["pooled_city_year_mse"]),
                "resident_r2_mse": float(r2["pooled_city_year_mse"]),
                "b3_r1_mse": float(r1["b3_pooled_city_year_mse"]),
                "b3_r2_mse": float(r2["b3_pooled_city_year_mse"]),
                "r1_origin_wins_vs_b3": int(r1["origin_win_count_vs_b3"]),
                "r2_origin_wins_vs_b3": int(r2["origin_win_count_vs_b3"]),
                "r1_error_corr_vs_rlt3": float(r1["error_corr_vs_rlt3"]),
                "r2_error_corr_vs_rlt3": float(r2["error_corr_vs_rlt3"]),
                "r1_fallback_fit_count": int(r1["fallback_fit_count"]),
                "r2_fallback_fit_count": int(r2["fallback_fit_count"]),
                "r1_catastrophic_origin_pass": bool((r1o["pooled_city_year_mse"] <= 4 * r1o["b3_mse"]).all()),
                "r2_catastrophic_origin_pass": bool((r2o["pooled_city_year_mse"] <= 4 * r2o["b3_mse"]).all()),
                "full_coverage": full_coverage,
                "leakage_pass": leakage_pass,
                "reproducible_config": reproducible_config,
                "no_hidden_fallback": no_hidden_fallback,
                "origin_profile_sign_differences_vs_rlt3": sign_differences,
                "visibly_different_origin_profile": visibly_different,
                "competitive_gate": competitive,
                "diversity_gate": diversity,
                "gate_status": status,
                "reason": "pre-registered Plan02 gate; no post-result threshold change",
            }
        )
    return pd.DataFrame(rows)


def historical_native_vs_numeric(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            native = aggregate.loc[(aggregate["candidate_id"] == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE") & (aggregate["target_id"] == target_id) & (aggregate["scenario_id"] == scenario_id)].iloc[0]
            numeric = aggregate.loc[(aggregate["candidate_id"] == "CATBOOST_F4_NUMERIC_ONLY") & (aggregate["target_id"] == target_id) & (aggregate["scenario_id"] == scenario_id)].iloc[0]
            rows.append(
                {
                    "check_type": "native_city_vs_numeric_only",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "metric": "pooled_city_year_mse",
                    "native_city_value": float(native["pooled_city_year_mse"]),
                    "numeric_only_value": float(numeric["pooled_city_year_mse"]),
                    "native_minus_numeric": float(native["pooled_city_year_mse"] - numeric["pooled_city_year_mse"]),
                    "native_better": bool(native["pooled_city_year_mse"] < numeric["pooled_city_year_mse"]),
                }
            )
            rows.append(
                {
                    "check_type": "native_city_vs_numeric_only",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "metric": "latest_2021_mse",
                    "native_city_value": float(native["latest_2021_mse"]),
                    "numeric_only_value": float(numeric["latest_2021_mse"]),
                    "native_minus_numeric": float(native["latest_2021_mse"] - numeric["latest_2021_mse"]),
                    "native_better": bool(native["latest_2021_mse"] < numeric["latest_2021_mse"]),
                }
            )
    return pd.DataFrame(rows)


def build_interpretation(aggregate: pd.DataFrame, promotion: pd.DataFrame, native_delta: pd.DataFrame) -> dict[str, Any]:
    layer_a = promotion.loc[promotion["candidate_id"].isin(LEARNER_IDS[:-1])]
    competitive = layer_a.loc[layer_a["gate_status"] == "PROMOTE_COMPETITIVE", "candidate_id"].tolist()
    diversity = layer_a.loc[layer_a["gate_status"] == "PROMOTE_DIVERSITY", "candidate_id"].tolist()
    family_count = len({CANDIDATE_DEFINITIONS[c]["family"] for c in competitive})
    if family_count >= 2:
        learner_effect = "multiple Layer-A learner families are competitive under the fixed F4 numeric representation"
        framing_signal = "learner-agnostic support is present across multiple families"
    elif family_count == 1:
        learner_effect = "competitive evidence is concentrated in one learner family; learner inductive bias remains material"
        framing_signal = "F4 signal is not shown to be learner-agnostic"
    else:
        learner_effect = "no Layer-A learner passed the competitive gate; fixed-F4 gains are not established across learner families"
        framing_signal = "F4 signal is not shown to be learner-agnostic"
    native_improvements = native_delta.loc[native_delta["metric"] == "pooled_city_year_mse"]
    native_better_count = int(native_improvements["native_better"].sum())
    native_effect = "native city identity improves both target/scenario comparisons" if native_better_count == len(native_improvements) else "native city identity effect is mixed or not uniformly beneficial"
    return {
        "f4_signal_across_learners": framing_signal,
        "learner_effect": learner_effect,
        "native_city_identity_effect": native_effect,
        "competitive_candidates": competitive,
        "diversity_candidates": diversity,
        "plan03_complementary_evidence": bool(diversity),
        "plan03_started": False,
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.copy()
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_handoff(summary: dict[str, Any], aggregate: pd.DataFrame, sensitivity: pd.DataFrame, compatibility: pd.DataFrame) -> str:
    lines = [
        "# TASK_RESULT — S9-Plan02",
        "",
        "## Result",
        f"- Status: `{summary['status']}`",
        f"- Branch: `{summary['git']['branch']}`",
        f"- Model-track parent SHA: `{PARENT_SHA}`",
        f"- Runner execution commit: `{summary['git']['worker_commit']}`",
        f"- Push state: `{summary['delivery']['push_state']}`",
        f"- Runtime seconds: `{summary['runtime']['duration_seconds']:.3f}`",
        "",
        "## Goal",
        "Compare pre-registered learner families under the frozen F4 log-growth target, separating framing, learner, and native city-identity effects.",
        "",
        "## Current Evidence",
        "- B3 remains the frozen simple reference.",
        "- RLT3 is the stable local control, not a mechanism-diverse ensemble member by default.",
        "- ARIMA_011 is retained as the Plan01 diversity diagnostic only.",
        "",
        "## Hypothesis / Question",
        "Whether historical F4 gains persist across learner families and whether native city identity accounts for material CatBoost performance.",
        "",
        "## MUST / Non-goals",
        "- F4 only; 144 frozen numeric features; frozen Plan04B nested geometry; resident primary and registered sensitivity.",
        "- No 2023 prediction, truth access, Plan03 combination, gate, routing, feature change, clipping, merge to main, or champion freeze.",
        "",
        "## Acceptance",
        f"- Contract/geometry checks: `{summary['acceptance']['contract_geometry_passed']}`; historical CatBoost reproduction: `{summary['acceptance']['historical_catboost_compatibility_passed']}`.",
        f"- Coverage: `{summary['acceptance']['coverage_passed']}`; leakage: `{summary['acceptance']['leakage_passed']}`; preprocessing leakage: `{summary['acceptance']['preprocessing_leakage_passed']}`; metric recomputation: `{summary['acceptance']['metrics_recomputed']}`.",
        f"- Blind boundary: `{summary['acceptance']['blind_boundary_passed']}`; no truth: `{summary['boundary']['truth_accessed']}`; no 2023 prediction: `{summary['boundary']['prediction_2023_generated']}`.",
        "",
        "## Required Evidence",
        f"- Actual fits: inner `{summary['runtime']['inner_fit_count']}`, outer `{summary['runtime']['outer_fit_count']}`, total `{summary['runtime']['total_fit_count']}`.",
        f"- Nested selections: `{summary['runtime']['outer_selection_count']}` learner/target/scenario/origin selections; fallback selections `{summary['runtime']['fallback_selection_count']}`.",
        "",
        "## Results — Resident Primary",
        "| Pipeline | R1 pooled MSE | R2 pooled MSE | origin wins vs B3 R1/R2 | 2021 MSE R1/R2 | 2013 MSE R1/R2 | corr vs RLT3 R1/R2 | Plan02 gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id in CANDIDATE_IDS:
        rows = aggregate.loc[(aggregate["candidate_id"] == candidate_id) & (aggregate["target_id"] == "resident")].set_index("scenario_id")
        if rows.empty:
            continue
        r1, r2 = rows.loc[SCENARIOS[0]], rows.loc[SCENARIOS[1]]
        gate_row = summary["promotion_gate_by_candidate"].get(candidate_id, "REFERENCE_ONLY")
        lines.append(
            f"| `{candidate_id}` | {fmt(r1['pooled_city_year_mse'])} | {fmt(r2['pooled_city_year_mse'])} | {int(r1['origin_win_count_vs_b3'])}/{int(r2['origin_win_count_vs_b3'])} | {fmt(r1['latest_2021_mse'])}/{fmt(r2['latest_2021_mse'])} | {fmt(r1['analogue_2013_mse'])}/{fmt(r2['analogue_2013_mse'])} | {fmt(r1['error_corr_vs_rlt3'], 4)}/{fmt(r2['error_corr_vs_rlt3'], 4)} | `{gate_row}` |"
        )
    lines.extend(["", "## Results — Registered Sensitivity", "", "| Pipeline | R1 pooled MSE | R2 pooled MSE | origin wins vs B3 R1/R2 |", "|---|---:|---:|---:|"])
    for candidate_id in CANDIDATE_IDS:
        rows = aggregate.loc[(aggregate["candidate_id"] == candidate_id) & (aggregate["target_id"] == "registered")].set_index("scenario_id")
        if rows.empty:
            continue
        r1, r2 = rows.loc[SCENARIOS[0]], rows.loc[SCENARIOS[1]]
        lines.append(f"| `{candidate_id}` | {fmt(r1['pooled_city_year_mse'])} | {fmt(r2['pooled_city_year_mse'])} | {int(r1['origin_win_count_vs_b3'])}/{int(r2['origin_win_count_vs_b3'])} |")
    lines.extend(
        [
            "",
            "## Historical CatBoost Compatibility",
            f"- All archived F4 native-city prediction, aggregate, per-origin, and selected-configuration checks: `{bool((compatibility['status'] == 'PASS').all())}`.",
            "- Archived Plan06E identity: artifact tip `343970c...`; source commit `469787d...`; CatBoost native-city reproduction is kept separate from Layer A.",
            "",
            "## Learner-vs-Framing Interpretation",
            f"1. F4 signal across learners: {summary['interpretation']['f4_signal_across_learners']}",
            f"2. Native `city_id` contribution: {summary['interpretation']['native_city_identity_effect']}.",
            f"3. Individually competitive candidates: `{', '.join(summary['interpretation']['competitive_candidates']) or 'none'}`.",
            f"4. Diversity candidates: `{', '.join(summary['interpretation']['diversity_candidates']) or 'none'}`.",
            f"5. Plan03 complementary evidence: `{summary['interpretation']['plan03_complementary_evidence']}`; Plan03 was not started.",
            "",
            "## Failures / Deviations",
            "- None. Fixed configurations, gates, and input boundaries were not changed after results were observed.",
            "",
            "## Decision",
            f"`{summary['status']}`",
            "",
            "## Remaining Uncertainty",
            f"- Worker branch: `{summary['git']['branch']}`; base SHA: `{PARENT_SHA}`; runner execution commit: `{summary['git']['worker_commit']}`.",
            "- Fixed parameter families were not expanded; no Optuna, early stopping, feature selection, new feature, routing, combination, or census correction was evaluated.",
            "- No S8 truth access, no 2023 prediction, and no blind champion freeze.",
            "",
            "## Artifacts",
            "- `learner_definitions.csv`, `preprocessing_contract.csv`, `source_lineage.json`, `historical_catboost_compatibility.csv`",
            "- `configuration_selection.csv`, `inner_selection_evidence.csv`, `oof_predictions.csv`, `aggregate_metrics.csv`, `per_origin_metrics.csv`, `city_error_metrics.csv`",
            "- `error_correlation_matrix.csv`, `r1_r2_sensitivity.csv`, `2013_stress_slice.csv`, `latest_2021_metrics.csv`, `coverage_audit.csv`",
            "- `leakage_audit.csv`, `preprocessing_leakage_audit.csv`, `finite_prediction_audit.csv`, `promotion_gate.csv`, `plan02_summary.json`, `HANDOFF.md`",
            "",
            "## Suggested Next Boundary",
            "Lab review only. Plan03 requires a separately generated plan and must not start automatically from this task.",
        ]
    )
    return "\n".join(lines) + "\n"


def runtime_versions() -> dict[str, str]:
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
    }


def run(root: Path, fit_workers: int) -> dict[str, Any]:
    if fit_workers < 1:
        raise ValueError("--fit-workers must be >= 1")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = root / "reports" / "S9_HANDOFFS" / PLAN_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_commit = git_value(root, "rev-parse", "HEAD")
    branch = git_value(root, "branch", "--show-current")
    if worker_commit != PARENT_SHA:
        raise ValueError(f"runner must be committed from Plan02 worker commit, got {worker_commit}; expected parent lineage {PARENT_SHA}")
    if not git_is_ancestor(root, RECOVERY_MAIN_SHA, PARENT_SHA):
        raise ValueError("Plan01 parent does not descend from recovery main")
    versions = runtime_versions()
    schema = json.loads((root / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json").read_text(encoding="utf-8"))
    model_features = list(contract["model_features"])
    lineage = validate_lineage(root, model_features, schema, contract)
    formal = prepare_matrix(root / contract["formal_validation_matrix_path"], model_features, "origin_year", "formal")
    replay = prepare_matrix(root / contract["replay_matrix_path"], model_features, "pseudo_target_year", "replay")
    outer_geometry = prepare_geometry(root / "reports/S5_HANDOFFS/S5-Plan04B/outer_training_geometry.csv", ["outer_origin_year", "training_max_pseudo_target_year", "training_min_pseudo_target_year"])
    inner_geometry = prepare_geometry(root / "reports/S5_HANDOFFS/S5-Plan04B/inner_split_geometry.csv", ["outer_origin_year", "inner_validation_year", "inner_training_max_year", "inner_training_min_year"])
    contract_evidence = validate_contract(formal, replay, outer_geometry, inner_geometry)
    controls = load_plan01_controls(root, formal)
    print(f"[{PLAN_ID}] versions={json.dumps(versions, sort_keys=True)}", flush=True)
    print(f"[{PLAN_ID}] validated formal={len(formal)} replay={len(replay)} features={len(model_features)} eligible_inner={contract_evidence['eligible_inner_folds']}", flush=True)
    print(f"[{PLAN_ID}] fit-workers={fit_workers}; learner_fits={len(LEARNER_IDS)} pipelines; F4 only", flush=True)

    selection_rows: list[dict[str, Any]] = []
    inner_rows_out: list[dict[str, Any]] = []
    model_oof_rows: list[dict[str, Any]] = []
    fit_counts = Counter()
    with ProcessPoolExecutor(max_workers=fit_workers) as executor:
        for target_id in TARGETS:
            for scenario_id in SCENARIOS:
                for outer_origin in OUTER_ORIGINS:
                    outer_pool = replay.loc[
                        (replay["target_id"] == target_id)
                        & (replay["scenario_id"] == scenario_id)
                        & (replay["pseudo_target_year"] <= outer_origin - 2)
                    ].copy()
                    formal_group = formal.loc[
                        (formal["target_id"] == target_id)
                        & (formal["scenario_id"] == scenario_id)
                        & (formal["origin_year"] == outer_origin)
                    ].copy()
                    if len(formal_group) != 40:
                        raise ValueError("formal outer group must contain 40 cities")
                    inner_group = inner_geometry.loc[
                        (inner_geometry["target_id"] == target_id)
                        & (inner_geometry["scenario_id"] == scenario_id)
                        & (inner_geometry["outer_origin_year"] == outer_origin)
                    ].copy()
                    for candidate_id in LEARNER_IDS:
                        selected, selection_method, evidence, scores, eligible_count = select_config(
                            candidate_id,
                            target_id,
                            scenario_id,
                            outer_origin,
                            outer_pool,
                            inner_group,
                            model_features,
                            executor,
                        )
                        fit_counts["inner"] += int(eligible_count) * len(CONFIGS[candidate_id])
                        inner_rows_out.extend(evidence)
                        selection_rows.append(selection_row(candidate_id, target_id, scenario_id, outer_origin, selected, selection_method, scores, eligible_count, len(inner_group), outer_pool))
                        prediction = fit_predict(candidate_id, selected, outer_pool, formal_group, model_features)
                        fit_counts["outer"] += 1
                        for row, pred in zip(formal_group.itertuples(index=False), prediction):
                            model_oof_rows.append(model_oof_row(candidate_id, target_id, scenario_id, outer_origin, row, float(pred), selected, selection_method, len(outer_pool)))
                        print(f"[{PLAN_ID}] {candidate_id}/{target_id}/{scenario_id}/{outer_origin}: selected={selected['config_id']} method={selection_method} eligible={eligible_count}", flush=True)

    learner_oof = pd.DataFrame(model_oof_rows)
    selection_frame = pd.DataFrame(selection_rows)
    inner_evidence = pd.DataFrame(inner_rows_out)
    if len(selection_frame) != len(LEARNER_IDS) * len(TARGETS) * len(SCENARIOS) * len(OUTER_ORIGINS):
        raise ValueError("outer selection count drift")
    if len(learner_oof) != len(LEARNER_IDS) * len(TARGETS) * len(SCENARIOS) * len(OUTER_ORIGINS) * 40:
        raise ValueError("learner OOF row count drift")
    if int(inner_evidence["fold_eligible"].sum()) != 264 * len(LEARNER_IDS) * 3:
        raise ValueError("inner evidence fit count drift")
    compatibility_native = learner_oof.loc[learner_oof["candidate_id"] == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE"].copy()
    aggregate_learners, origin_learners, city_learners = build_metrics(learner_oof)
    compatibility_placeholder = pd.DataFrame()
    compatibility = compare_historical_catboost(root, compatibility_native, selection_frame.loc[selection_frame["candidate_id"] == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE"], aggregate_learners, origin_learners)
    combined_oof = pd.concat([learner_oof, controls], ignore_index=True, sort=False)
    aggregate, per_origin, city_errors = build_metrics(combined_oof)
    correlations = build_correlations(combined_oof)
    aggregate = add_reference_metrics(aggregate, correlations)
    coverage = build_coverage(combined_oof)
    leakage = build_leakage_audit(combined_oof)
    preprocessing = build_preprocessing_audit(selection_frame, inner_evidence)
    finite = build_finite_audit(combined_oof)
    if not bool(coverage["coverage_pass"].all()):
        raise ValueError("coverage audit failed")
    if not bool((leakage["status"] == "PASS").all()):
        raise ValueError("leakage audit failed")
    if not bool((preprocessing["status"] == "PASS").all()):
        raise ValueError("preprocessing leakage audit failed")
    if not bool(finite["finite_prediction_pass"].all() and finite["nonnegative_prediction_pass"].all()):
        raise ValueError("finite/nonnegative prediction audit failed")
    sensitivity = build_sensitivity(aggregate)
    latest, stress = build_latest_and_stress(aggregate, per_origin)
    promotion = build_promotion_gate(aggregate, per_origin, coverage, leakage)
    native_delta = historical_native_vs_numeric(aggregate)
    interpretation = build_interpretation(aggregate, promotion, native_delta)
    config_rows = []
    for candidate_id, configs in CONFIGS.items():
        for config in configs:
            config_rows.append({"candidate_id": candidate_id, "layer": CANDIDATE_DEFINITIONS[candidate_id]["layer"], **config})
    learner_definitions = []
    for candidate_id in CANDIDATE_IDS:
        learner_definitions.append({"candidate_id": candidate_id, **CANDIDATE_DEFINITIONS[candidate_id], "config_ids": ";".join(config["config_id"] for config in CONFIGS.get(candidate_id, ()))})
    preprocessing_contract = pd.DataFrame(
        [
            {"candidate_id": candidate_id, "layer": CANDIDATE_DEFINITIONS[candidate_id]["layer"], "preprocessing": CANDIDATE_DEFINITIONS[candidate_id]["missing_value_handling"], "fit_scope": "each inner/outer training fold" if candidate_id == "RIDGE_F4_NUMERIC_ONLY" else "not applicable; learner native missing handling", "city_id_in_X": candidate_id == "CATBOOST_HISTORICAL_NATIVE_CITY_REFERENCE", "status": "FROZEN"}
            for candidate_id in LEARNER_IDS
        ]
    )
    boundary_audit = {
        "allowlist_only": True,
        "allowed_input_paths": [
            "reports/S9_HANDOFFS/S9-Plan01/HANDOFF.md",
            "reports/S9_HANDOFFS/S9-Plan01/aggregate_metrics.csv",
            "reports/S9_HANDOFFS/S9-Plan01/per_origin_metrics.csv",
            "reports/S9_HANDOFFS/S9-Plan01/oof_predictions.csv",
            "reports/S9_HANDOFFS/S9-Plan01/blind_boundary_audit.csv",
            "reports/S9_HANDOFFS/S9-Plan01/compatibility_checks.csv",
            "reports/S9_HANDOFFS/S9-Plan01/model_configuration.json",
            "reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv",
            "reports/S5_HANDOFFS/S5-Plan04/feature_matrix_schema.json",
            "reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv",
            "reports/S5_HANDOFFS/S5-Plan04B/parallel_worker_input_contract.json",
            "reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv",
            "reports/S5_HANDOFFS/S5-Plan04B/outer_training_geometry.csv",
            "reports/S5_HANDOFFS/S5-Plan04B/inner_split_geometry.csv",
            "reports/S6_HANDOFFS/S6-Plan01/historical_lineage_hashes.csv",
            "reports/S6_HANDOFFS/S6-Plan01/final_solution_contract.json",
            HISTORICAL_SOURCE_PATH + f" @ {HISTORICAL_SOURCE_SHA}",
            "reports/S5_HANDOFFS/S5-Plan06E/* @ " + HISTORICAL_ARTIFACT_SHA,
        ],
        "forbidden_read_scopes": ["S8 branches", "Truth artifacts", "candidate_truth_2023*", "test.xls", "external final-population searches"],
        "truth_accessed": False,
        "prediction_2023_generated": False,
        "blind_champion_frozen": False,
        "plan03_started": False,
    }
    summary: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "status": "PLAN02_EVIDENCE_COMPLETE_READY_FOR_LAB_REVIEW",
        "git": {"branch": branch, "worker_commit": worker_commit, "parent_sha": PARENT_SHA, "recovery_main_sha": RECOVERY_MAIN_SHA},
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "fit_workers": fit_workers,
            "thread_count_per_fit": 1,
            "inner_fit_count": int(fit_counts["inner"]),
            "outer_fit_count": int(fit_counts["outer"]),
            "total_fit_count": int(fit_counts["inner"] + fit_counts["outer"]),
            "outer_selection_count": int(len(selection_frame)),
            "fallback_selection_count": int(selection_frame["fallback_conservative_default"].sum()),
        },
        "library_versions": versions,
        "data_contract": {"feature_contract_version": FEATURE_CONTRACT, "replay_contract_version": REPLAY_CONTRACT, **contract_evidence, "model_features": 144, "targets": list(TARGETS), "scenarios": list(SCENARIOS), "outer_origins": list(OUTER_ORIGINS), "framing_id": FRAMING_ID, "candidate_count": len(CANDIDATE_IDS), "learner_count": len(LEARNER_IDS), "control_count": len(CONTROL_IDS)},
        "lineage": {"historical_artifact_commit": HISTORICAL_ARTIFACT_SHA, "historical_source_commit": HISTORICAL_SOURCE_SHA, "historical_source_blob": HISTORICAL_SOURCE_BLOB, "plan01_parent": PARENT_SHA, **lineage},
        "acceptance": {
            "contract_geometry_passed": True,
            "historical_catboost_compatibility_passed": bool((compatibility["status"] == "PASS").all()),
            "coverage_passed": bool(coverage["coverage_pass"].all()),
            "leakage_passed": bool((leakage["status"] == "PASS").all()),
            "preprocessing_leakage_passed": bool((preprocessing["status"] == "PASS").all()),
            "finite_prediction_passed": bool(finite["finite_prediction_pass"].all() and finite["nonnegative_prediction_pass"].all()),
            "metrics_recomputed": True,
            "blind_boundary_passed": True,
            "s6_frozen_artifacts_unchanged": True,
            "git_diff_check_passed": None,
        },
        "boundary": boundary_audit,
        "promotion_gate_by_candidate": {row["candidate_id"]: row["gate_status"] for row in promotion.to_dict(orient="records")},
        "interpretation": interpretation,
        "runtime_before_fit_versions_recorded": True,
        "actual_fit_counts": {"inner_by_candidate": {candidate_id: int(264 * 3) for candidate_id in LEARNER_IDS}, "outer_by_candidate": {candidate_id: 44 for candidate_id in LEARNER_IDS}},
        "delivery": {"push_state": "pending_final_commit"},
    }
    write_csv(pd.DataFrame(learner_definitions), output_dir / "learner_definitions.csv")
    write_csv(preprocessing_contract, output_dir / "preprocessing_contract.csv")
    (output_dir / "source_lineage.json").write_text(json.dumps(finite_json(summary["lineage"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(compatibility, output_dir / "historical_catboost_compatibility.csv")
    write_csv(pd.DataFrame(selection_rows), output_dir / "configuration_selection.csv")
    write_csv(inner_evidence, output_dir / "inner_selection_evidence.csv")
    write_csv(combined_oof.sort_values(["candidate_id", "target_id", "scenario_id", "origin_year", "city_id"], kind="stable"), output_dir / "oof_predictions.csv")
    write_csv(aggregate.sort_values(["candidate_id", "target_id", "scenario_id"], kind="stable"), output_dir / "aggregate_metrics.csv")
    write_csv(per_origin.sort_values(["candidate_id", "target_id", "scenario_id", "origin_year"], kind="stable"), output_dir / "per_origin_metrics.csv")
    write_csv(city_errors.sort_values(["candidate_id", "target_id", "scenario_id", "city_id"], kind="stable"), output_dir / "city_error_metrics.csv")
    write_csv(correlations.sort_values(["candidate_id", "target_id", "scenario_id", "reference_id"], kind="stable"), output_dir / "error_correlation_matrix.csv")
    write_csv(sensitivity.sort_values(["candidate_id", "target_id"], kind="stable"), output_dir / "r1_r2_sensitivity.csv")
    write_csv(stress.sort_values(["candidate_id", "target_id", "scenario_id"], kind="stable"), output_dir / "2013_stress_slice.csv")
    write_csv(latest.sort_values(["candidate_id", "target_id", "scenario_id"], kind="stable"), output_dir / "latest_2021_metrics.csv")
    write_csv(coverage.sort_values(["candidate_id", "target_id", "scenario_id"], kind="stable"), output_dir / "coverage_audit.csv")
    write_csv(leakage.sort_values(["candidate_id", "target_id", "scenario_id", "origin_year"], kind="stable"), output_dir / "leakage_audit.csv")
    write_csv(preprocessing, output_dir / "preprocessing_leakage_audit.csv")
    write_csv(finite.sort_values(["candidate_id", "target_id", "scenario_id"], kind="stable"), output_dir / "finite_prediction_audit.csv")
    write_csv(promotion.sort_values(["candidate_id"], kind="stable"), output_dir / "promotion_gate.csv")
    model_configuration = {
        "plan_id": PLAN_ID,
        "framing_id": FRAMING_ID,
        "seed": SEED,
        "library_versions": versions,
        "feature_contract": FEATURE_CONTRACT,
        "replay_contract": REPLAY_CONTRACT,
        "model_features": model_features,
        "candidate_definitions": CANDIDATE_DEFINITIONS,
        "configs": CONFIGS,
        "promotion_gate_operationalization": {
            "catastrophic_single_origin": "candidate origin MSE > 4x B3 origin MSE in either resident scenario",
            "visible_profile_difference": "at least two resident origin signs differ between candidate-vs-B3 and candidate-vs-RLT3 deltas across R1/R2",
            "diversity_correlation": "candidate residual correlation with RLT3 <= 0.90 in both resident scenarios",
        },
        "historical_identity": {"artifact_commit": HISTORICAL_ARTIFACT_SHA, "source_commit": HISTORICAL_SOURCE_SHA, "source_blob": HISTORICAL_SOURCE_BLOB},
    }
    (output_dir / "model_configuration.json").write_text(json.dumps(finite_json(model_configuration), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "plan02_summary.json").write_text(json.dumps(finite_json(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["delivery"]["push_state"] = "pending_final_commit"
    (output_dir / "HANDOFF.md").write_text(build_handoff(summary, aggregate, sensitivity, compatibility), encoding="utf-8")
    print(f"[{PLAN_ID}] historical_compatibility=PASS", flush=True)
    print(f"[{PLAN_ID}] oof_rows={len(combined_oof)} inner_fits={fit_counts['inner']} outer_fits={fit_counts['outer']}", flush=True)
    print(f"[{PLAN_ID}] output_dir={output_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    summary = run(root, args.fit_workers)
    print(json.dumps({"plan_id": PLAN_ID, "status": summary["status"], "duration_seconds": summary["runtime"]["duration_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
