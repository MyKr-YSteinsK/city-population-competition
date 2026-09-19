"""Run the preregistered S5-Plan06D ExtraTrees and Ridge control screening.

This worker consumes the frozen Plan04/Plan04B matrices and geometry.  It
does not change the feature contract, replay universe, or validation design.
All preprocessing is fitted inside the corresponding inner or outer training
fold.  The output is an auditable screening package, not a final forecast.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan06D"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
BASE_SHA = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
BRANCH = "s5/plan06d-extratrees-linear-controls"
SEED = 20260919
FEATURE_CONTRACT_VERSION = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT_VERSION = "S5-Plan04B-TRAINING-REPLAY-V1"
OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
FORMAL_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv"
SCHEMA_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
REPLAY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv"
INPUT_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json"
OUTER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "outer_training_geometry.csv"
INNER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "inner_split_geometry.csv"
WAVE_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "strong_model_wave_contract.json"

TARGETS = ["resident", "registered"]
SCENARIOS = ["R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS"]
FRAMINGS = ["F1_DIRECT", "F2_B3_RESIDUAL"]
ORIGINS = list(range(2011, 2022))
REQUIRED_CITY_COUNT = 40
REPLAY_MAX_GAP = 2
MIN_ELIGIBLE_FOLDS = 3

ET_CONFIGS = [
    {
        "config_id": "ET_C1_CONSERVATIVE",
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": 0.50,
        "criterion": "squared_error",
        "bootstrap": False,
        "random_state": SEED,
        "n_jobs": 1,
    },
    {
        "config_id": "ET_C2_MEDIUM",
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_leaf": 3,
        "max_features": 0.70,
        "criterion": "squared_error",
        "bootstrap": False,
        "random_state": SEED,
        "n_jobs": 1,
    },
    {
        "config_id": "ET_C3_FLEXIBLE_REGULARIZED",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.70,
        "criterion": "squared_error",
        "bootstrap": False,
        "random_state": SEED,
        "n_jobs": 1,
    },
]

RIDGE_CONFIGS = [
    {"config_id": "RIDGE_C1_STRONG", "alpha": 1000.0},
    {"config_id": "RIDGE_C2_MEDIUM", "alpha": 100.0},
    {"config_id": "RIDGE_C3_LIGHT", "alpha": 10.0},
]


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def py_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): py_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py_value(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(py_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: py_value(row.get(field)) for field in fields})


def assert_gate() -> None:
    assert git_text("branch", "--show-current") == BRANCH
    assert git_text("merge-base", BASE_SHA, "HEAD") == BASE_SHA
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode == 0
    assert git_text("status", "--porcelain") == "", "worker must start clean"


def load_contract() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    contract = json.loads(INPUT_CONTRACT_PATH.read_text(encoding="utf-8"))
    wave = json.loads(WAVE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert schema["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert contract["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert contract["contract_version"] == REPLAY_CONTRACT_VERSION
    assert contract["model_feature_count"] == 144
    assert len(contract["model_features"]) == 144
    assert wave["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert wave["replay_contract_version"] == REPLAY_CONTRACT_VERSION
    return {
        "schema": schema,
        "contract": contract,
        "wave": wave,
        "features": list(contract["model_features"]),
        "groups": dict(contract["feature_groups"]),
    }


def load_inputs(context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = context["features"]
    replay = pd.read_csv(REPLAY_PATH, low_memory=False)
    formal = pd.read_csv(FORMAL_PATH, low_memory=False)
    outer = pd.read_csv(OUTER_GEOMETRY_PATH)
    inner = pd.read_csv(INNER_GEOMETRY_PATH)
    assert len(replay) == 2320
    assert len(formal) == 1760
    assert len(features) == 144
    assert set(features).issubset(replay.columns)
    assert set(features).issubset(formal.columns)
    assert set(["target_id", "scenario_id", "pseudo_target_year", "city_id",
                "actual_target_value", "b3_prediction",
                "b3_residual_actual_minus_prediction"]).issubset(replay.columns)
    assert set(["target_id", "scenario_id", "origin_year", "city_id",
                "actual_target_value", "b3_prediction",
                "b3_residual_actual_minus_prediction"]).issubset(formal.columns)
    assert set(replay["target_id"]) == set(TARGETS)
    assert set(replay["scenario_id"]) == set(SCENARIOS)
    assert set(formal["origin_year"]) == set(ORIGINS)
    assert formal.groupby(["target_id", "scenario_id", "origin_year"]).size().eq(40).all()
    assert replay["city_id"].nunique() == REQUIRED_CITY_COUNT
    assert inner.shape[0] == 484
    assert outer.shape[0] == 44
    eligible = inner[
        (inner["inner_training_row_count"] >= 40)
        & (inner["inner_validation_row_count"] >= 20)
        & (inner["boundary_pass"] == "PASS")
    ]
    assert len(eligible) == 264
    replay["pseudo_target_year"] = replay["pseudo_target_year"].astype(int)
    formal["origin_year"] = formal["origin_year"].astype(int)
    for frame in (replay, formal):
        frame["city_id"] = frame["city_id"].astype(str)
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame["actual_target_value"] = pd.to_numeric(frame["actual_target_value"], errors="coerce")
        frame["b3_prediction"] = pd.to_numeric(frame["b3_prediction"], errors="coerce")
        frame["b3_residual_actual_minus_prediction"] = pd.to_numeric(
            frame["b3_residual_actual_minus_prediction"], errors="coerce"
        )
    return replay, formal, outer, inner


def check_geometry(replay: pd.DataFrame, formal: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame) -> None:
    for row in outer.itertuples(index=False):
        pool = replay[
            (replay.target_id == row.target_id)
            & (replay.scenario_id == row.scenario_id)
            & (replay.pseudo_target_year <= int(row.outer_origin_year) - REPLAY_MAX_GAP)
        ]
        validation = formal[
            (formal.target_id == row.target_id)
            & (formal.scenario_id == row.scenario_id)
            & (formal.origin_year == int(row.outer_origin_year))
        ]
        assert len(pool) == int(row.training_row_count)
        assert len(validation) == int(row.validation_row_count) == 40
        assert pool["city_id"].nunique() == int(row.training_city_count)
        assert validation["city_id"].nunique() == int(row.validation_city_count)
        assert len(pool[pool.pseudo_target_year > int(row.outer_origin_year) - REPLAY_MAX_GAP]) == 0
    for row in inner.itertuples(index=False):
        pool = replay[
            (replay.target_id == row.target_id)
            & (replay.scenario_id == row.scenario_id)
            & (replay.pseudo_target_year <= int(row.outer_origin_year) - REPLAY_MAX_GAP)
        ]
        inner_train = pool[pool.pseudo_target_year <= int(row.inner_validation_year) - REPLAY_MAX_GAP]
        inner_val = pool[pool.pseudo_target_year == int(row.inner_validation_year)]
        assert len(pool) == int(row.outer_training_pool_row_count)
        assert len(inner_train) == int(row.inner_training_row_count)
        assert len(inner_val) == int(row.inner_validation_row_count)
        assert inner_train["city_id"].nunique() == int(row.inner_training_city_count)
        assert inner_val["city_id"].nunique() == int(row.inner_validation_city_count)


def city_order(replay: pd.DataFrame, formal: pd.DataFrame) -> list[str]:
    cities = sorted(set(replay["city_id"]) | set(formal["city_id"]))
    assert len(cities) == REQUIRED_CITY_COUNT
    return cities


def make_preprocessor(features: list[str], cities: list[str], family: str) -> ColumnTransformer | Pipeline:
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
    )
    encoded = OneHotEncoder(
        categories=[cities],
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric, features),
            ("city", encoded, ["city_id"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if family == "RIDGE":
        return Pipeline([("transform", transformer), ("scale", StandardScaler())])
    return transformer


def transformed_feature_names(preprocessor: ColumnTransformer | Pipeline) -> list[str]:
    if isinstance(preprocessor, Pipeline):
        transformer = preprocessor.named_steps["transform"]
    else:
        transformer = preprocessor
    return [str(value) for value in transformer.get_feature_names_out()]


def transform(preprocessor: ColumnTransformer | Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = preprocessor.transform(frame[features + ["city_id"]])
    return np.asarray(values, dtype=float)


def fit_transform(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    cities: list[str],
) -> tuple[ColumnTransformer | Pipeline, np.ndarray, np.ndarray, list[str]]:
    preprocessor = make_preprocessor(features, cities, family)
    preprocessor.fit(train[features + ["city_id"]])
    return (
        preprocessor,
        transform(preprocessor, train, features),
        transform(preprocessor, validation, features),
        transformed_feature_names(preprocessor),
    )


def make_estimator(family: str, config: dict[str, Any]) -> Any:
    if family == "EXTRATREES":
        estimator_config = {
            key: value for key, value in config.items() if key != "config_id"
        }
        return ExtraTreesRegressor(**estimator_config)
    return Ridge(alpha=float(config["alpha"]), fit_intercept=True)


def label_values(frame: pd.DataFrame, framing: str) -> np.ndarray:
    column = "actual_target_value" if framing == "F1_DIRECT" else "b3_residual_actual_minus_prediction"
    return frame[column].to_numpy(dtype=float)


def final_prediction(model: Any, x: np.ndarray, frame: pd.DataFrame, framing: str) -> np.ndarray:
    prediction = np.asarray(model.predict(x), dtype=float)
    if framing == "F2_B3_RESIDUAL":
        prediction = frame["b3_prediction"].to_numpy(dtype=float) + prediction
    return prediction


def feature_group(feature_name: str, groups: dict[str, list[str]]) -> tuple[str, str]:
    if feature_name.startswith("city_id_"):
        return "CITY_ONE_HOT", "city_one_hot"
    if feature_name.endswith("_missing") or "_missing_" in feature_name:
        return "MISSING_INDICATOR", "missing_indicator"
    for group, members in groups.items():
        if feature_name in members:
            return group, "numeric_feature"
    return "OTHER", "transformed_feature"


def run_family(
    family: str,
    replay: pd.DataFrame,
    formal: pd.DataFrame,
    inner: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    cities: list[str],
    worker_commit: str,
) -> dict[str, Any]:
    configs = ET_CONFIGS if family == "EXTRATREES" else RIDGE_CONFIGS
    model_id = "EXTRATREES_SELECTED" if family == "EXTRATREES" else "RIDGE_SELECTED"
    oof_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    fit_count = 0
    fallback_count = 0
    selected_config_counts: Counter[str] = Counter()

    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for framing_id in FRAMINGS:
                for origin_year in ORIGINS:
                    outer_train = replay[
                        (replay.target_id == target_id)
                        & (replay.scenario_id == scenario_id)
                        & (replay.pseudo_target_year <= origin_year - REPLAY_MAX_GAP)
                    ].copy()
                    outer_validation = formal[
                        (formal.target_id == target_id)
                        & (formal.scenario_id == scenario_id)
                        & (formal.origin_year == origin_year)
                    ].copy()
                    eligible = inner[
                        (inner.target_id == target_id)
                        & (inner.scenario_id == scenario_id)
                        & (inner.outer_origin_year == origin_year)
                        & (inner.inner_training_row_count >= 40)
                        & (inner.inner_validation_row_count >= 20)
                        & (inner.boundary_pass == "PASS")
                    ].sort_values("inner_validation_year")
                    group_key = {
                        "family": family,
                        "framing_id": framing_id,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": origin_year,
                    }
                    config_means: dict[str, float | None] = {}
                    transform_cache: dict[int, tuple[Any, np.ndarray, np.ndarray, list[str]]] = {}
                    for config_index, config in enumerate(configs):
                        config_id = str(config["config_id"])
                        fold_mses: list[float] = []
                        if len(eligible) < MIN_ELIGIBLE_FOLDS:
                            evidence_rows.append(
                                {
                                    **group_key,
                                    "config_id": config_id,
                                    "config_order": config_index + 1,
                                    "inner_validation_year": "",
                                    "inner_training_row_count": "",
                                    "inner_validation_row_count": "",
                                    "validation_mse": "",
                                    "eligible_fold_count": len(eligible),
                                    "selection_status": "fallback_not_evaluated",
                                    "selection_method": "fallback_conservative_default",
                                }
                            )
                            config_means[config_id] = None
                            continue
                        for fold in eligible.itertuples(index=False):
                            validation_year = int(fold.inner_validation_year)
                            if validation_year not in transform_cache:
                                train = outer_train[
                                    outer_train.pseudo_target_year <= validation_year - REPLAY_MAX_GAP
                                ]
                                validation = outer_train[
                                    outer_train.pseudo_target_year == validation_year
                                ]
                                cache_value = fit_transform(
                                    family, train, validation, features, cities
                                )
                                transform_cache[validation_year] = cache_value
                                assert len(train) == int(fold.inner_training_row_count)
                                assert len(validation) == int(fold.inner_validation_row_count)
                            preprocessor, x_train, x_validation, _ = transform_cache[validation_year]
                            train = outer_train[
                                outer_train.pseudo_target_year <= validation_year - REPLAY_MAX_GAP
                            ]
                            validation = outer_train[
                                outer_train.pseudo_target_year == validation_year
                            ]
                            model = make_estimator(family, config)
                            model.fit(x_train, label_values(train, framing_id))
                            prediction = final_prediction(model, x_validation, validation, framing_id)
                            mse = float(
                                np.mean((prediction - validation["actual_target_value"].to_numpy(dtype=float)) ** 2)
                            )
                            fold_mses.append(mse)
                            evidence_rows.append(
                                {
                                    **group_key,
                                    "config_id": config_id,
                                    "config_order": config_index + 1,
                                    "inner_validation_year": validation_year,
                                    "inner_training_row_count": len(train),
                                    "inner_validation_row_count": len(validation),
                                    "validation_mse": mse,
                                    "eligible_fold_count": len(eligible),
                                    "selection_status": "evaluated",
                                    "selection_method": "nested_inner_mean_mse",
                                }
                            )
                        config_means[config_id] = float(np.mean(fold_mses))

                    if len(eligible) < MIN_ELIGIBLE_FOLDS:
                        selected_index = 0
                        selection_method = "fallback_conservative_default"
                        fallback_count += 1
                    else:
                        selected_index = 0
                        selected_mean = config_means[str(configs[0]["config_id"])]
                        assert selected_mean is not None
                        for candidate_index, candidate in enumerate(configs[1:], start=1):
                            candidate_mean = config_means[str(candidate["config_id"])]
                            assert candidate_mean is not None
                            if candidate_mean < selected_mean - 1e-12:
                                selected_index = candidate_index
                                selected_mean = candidate_mean
                        selection_method = "nested_inner_mean_mse"
                    selected = configs[selected_index]
                    selected_id = str(selected["config_id"])
                    selected_mean = config_means[selected_id]
                    selected_config_counts[selected_id] += 1
                    selection_rows.append(
                        {
                            **group_key,
                            "selected_config_id": selected_id,
                            "selection_method": selection_method,
                            "eligible_inner_fold_count": len(eligible),
                            "config_1_id": configs[0]["config_id"],
                            "config_1_mean_inner_mse": config_means[str(configs[0]["config_id"])],
                            "config_2_id": configs[1]["config_id"],
                            "config_2_mean_inner_mse": config_means[str(configs[1]["config_id"])],
                            "config_3_id": configs[2]["config_id"],
                            "config_3_mean_inner_mse": config_means[str(configs[2]["config_id"])],
                            "selected_mean_inner_mse": selected_mean,
                        }
                    )

                    outer_preprocessor, x_outer_train, x_outer_validation, transformed_names = fit_transform(
                        family, outer_train, outer_validation, features, cities
                    )
                    outer_model = make_estimator(family, selected)
                    outer_model.fit(x_outer_train, label_values(outer_train, framing_id))
                    prediction = final_prediction(
                        outer_model, x_outer_validation, outer_validation, framing_id
                    )
                    fit_count += 1
                    assert len(prediction) == 40
                    for row, predicted in zip(outer_validation.to_dict("records"), prediction):
                        actual = float(row["actual_target_value"])
                        oof_rows.append(
                            {
                                "model_id": model_id,
                                "family": family,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "origin_year": origin_year,
                                "city_id": row["city_id"],
                                "actual_target_value": actual,
                                "prediction_value": float(predicted),
                                "residual_prediction_minus_actual": float(predicted - actual),
                                "squared_error": float((predicted - actual) ** 2),
                                "training_max_label_year": int(outer_train["pseudo_target_year"].max()),
                                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                                "worker_commit": worker_commit,
                                "selected_config_id": selected_id,
                                "selection_method": selection_method,
                            }
                        )

                    if family == "EXTRATREES":
                        diagnostic_values = np.asarray(outer_model.feature_importances_, dtype=float)
                        diagnostic_type = "feature_importance"
                    else:
                        diagnostic_values = np.asarray(outer_model.coef_, dtype=float).reshape(-1)
                        diagnostic_type = "coefficient"
                    assert len(diagnostic_values) == len(transformed_names)
                    for name, value in zip(transformed_names, diagnostic_values):
                        group, role = feature_group(name, groups)
                        diagnostic_rows.append(
                            {
                                "family": family,
                                "model_id": model_id,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin_year,
                                "selected_config_id": selected_id,
                                "feature_name": name,
                                "feature_group": group,
                                "feature_role": role,
                                "diagnostic_type": diagnostic_type,
                                "importance_or_coefficient": float(value),
                                "absolute_magnitude": float(abs(value)),
                            }
                        )
                    if family == "EXTRATREES":
                        depths = [int(estimator.tree_.max_depth) for estimator in outer_model.estimators_]
                        nodes = [int(estimator.tree_.node_count) for estimator in outer_model.estimators_]
                        capacity = {
                            "n_estimators": len(outer_model.estimators_),
                            "mean_tree_depth": float(np.mean(depths)),
                            "max_tree_depth": max(depths),
                            "mean_tree_nodes": float(np.mean(nodes)),
                            "max_tree_nodes": max(nodes),
                        }
                    else:
                        coefficients = np.asarray(outer_model.coef_, dtype=float).reshape(-1)
                        capacity = {
                            "alpha": float(selected["alpha"]),
                            "nonzero_coefficient_count": int(np.count_nonzero(np.abs(coefficients) > 1e-12)),
                            "coefficient_l1": float(np.sum(np.abs(coefficients))),
                            "coefficient_abs_max": float(np.max(np.abs(coefficients))),
                        }
                    for metric_name, metric_value in capacity.items():
                        diagnostic_rows.append(
                            {
                                "family": family,
                                "model_id": model_id,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin_year,
                                "selected_config_id": selected_id,
                                "feature_name": "",
                                "feature_group": "MODEL_CAPACITY",
                                "feature_role": "capacity",
                                "diagnostic_type": "capacity_summary",
                                "importance_or_coefficient": "",
                                "absolute_magnitude": "",
                                "capacity_metric": metric_name,
                                "capacity_value": metric_value,
                            }
                        )

    expected_rows = 2 * 2 * 11 * 40
    assert len(oof_rows) == expected_rows
    assert len(selection_rows) == 2 * 2 * 2 * 11
    assert len(set((r["target_id"], r["scenario_id"], r["framing_id"], r["origin_year"], r["city_id"]) for r in oof_rows)) == expected_rows
    return {
        "family": family,
        "model_id": model_id,
        "oof_rows": oof_rows,
        "selection_rows": selection_rows,
        "evidence_rows": evidence_rows,
        "diagnostic_rows": diagnostic_rows,
        "fit_count": fit_count,
        "fallback_count": fallback_count,
        "selected_config_counts": dict(selected_config_counts),
    }


def baseline_frame(formal: pd.DataFrame) -> pd.DataFrame:
    return formal[
        [
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "actual_target_value",
            "b3_prediction",
        ]
    ].copy()


def calculate_metrics(
    all_oof: pd.DataFrame,
    formal: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference = baseline_frame(formal)
    reference["model_id"] = "B3_REFERENCE"
    reference["family"] = "BASELINE"
    reference["framing_id"] = "B3_REFERENCE"
    reference["prediction_value"] = reference["b3_prediction"]
    reference["residual_prediction_minus_actual"] = (
        reference["prediction_value"] - reference["actual_target_value"]
    )
    reference["squared_error"] = reference["residual_prediction_minus_actual"] ** 2
    combined = pd.concat(
        [
            all_oof[
                [
                    "model_id",
                    "family",
                    "framing_id",
                    "target_id",
                    "scenario_id",
                    "origin_year",
                    "city_id",
                    "actual_target_value",
                    "prediction_value",
                    "residual_prediction_minus_actual",
                    "squared_error",
                ]
            ],
            reference[
                [
                    "model_id",
                    "family",
                    "framing_id",
                    "target_id",
                    "scenario_id",
                    "origin_year",
                    "city_id",
                    "actual_target_value",
                    "prediction_value",
                    "residual_prediction_minus_actual",
                    "squared_error",
                ]
            ],
        ],
        ignore_index=True,
    )
    aggregate_rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    group_columns = ["model_id", "family", "framing_id", "target_id", "scenario_id"]
    origin_group_columns = group_columns + ["origin_year"]
    city_group_columns = group_columns + ["city_id"]
    origin_reference: dict[tuple[str, str, int], float] = {}
    for keys, frame in combined.groupby(origin_group_columns, sort=True):
        key = dict(zip(origin_group_columns, keys))
        mse = float(frame["squared_error"].mean())
        row = {
            **key,
            "row_count": len(frame),
            "origin_mse": mse,
            "origin_rmse": float(np.sqrt(mse)),
            "origin_mae": float(frame["residual_prediction_minus_actual"].abs().mean()),
            "origin_bias": float(frame["residual_prediction_minus_actual"].mean()),
        }
        if key["model_id"] == "B3_REFERENCE":
            origin_reference[(key["target_id"], key["scenario_id"], int(key["origin_year"]))] = mse
            row["origin_win_vs_b3"] = ""
        else:
            b3_mse = origin_reference[(key["target_id"], key["scenario_id"], int(key["origin_year"]))]
            row["b3_origin_mse"] = b3_mse
            row["origin_win_vs_b3"] = bool(mse < b3_mse)
        origin_rows.append(row)

    for keys, frame in combined.groupby(city_group_columns, sort=True):
        key = dict(zip(city_group_columns, keys))
        city_rows.append(
            {
                **key,
                "row_count": len(frame),
                "city_sse": float(frame["squared_error"].sum()),
                "city_mse": float(frame["squared_error"].mean()),
                "city_rmse": float(np.sqrt(frame["squared_error"].mean())),
                "city_bias": float(frame["residual_prediction_minus_actual"].mean()),
            }
        )

    metric_groups = list(combined.groupby(group_columns, sort=True))
    for keys, frame in metric_groups:
        key = dict(zip(group_columns, keys))
        mse = float(frame["squared_error"].mean())
        origin_frame = pd.DataFrame(
            [
                row
                for row in origin_rows
                if all(row[col] == key[col] for col in group_columns)
            ]
        )
        city_frame = pd.DataFrame(
            [
                row
                for row in city_rows
                if all(row[col] == key[col] for col in group_columns)
            ]
        )
        b3_key = ("resident", key["scenario_id"]) if key["target_id"] == "resident" else (key["target_id"], key["scenario_id"])
        b3_match = combined[
            (combined.model_id == "B3_REFERENCE")
            & (combined.target_id == key["target_id"])
            & (combined.scenario_id == key["scenario_id"])
        ]
        b3_mse = float(b3_match["squared_error"].mean())
        if key["model_id"] == "B3_REFERENCE":
            origin_wins = ""
            improvement = ""
        else:
            origin_wins = int(origin_frame["origin_win_vs_b3"].fillna(False).sum())
            improvement = float((b3_mse - mse) / b3_mse * 100.0) if b3_mse else None
        city_sse = city_frame["city_sse"].to_numpy(dtype=float)
        city_sse = np.sort(city_sse)[::-1]
        total_city_sse = float(city_sse.sum())
        top1 = float(city_sse[:1].sum() / total_city_sse) if total_city_sse else None
        top3 = float(city_sse[:3].sum() / total_city_sse) if total_city_sse else None
        latest = frame[frame.origin_year == 2021]
        aggregate_rows.append(
            {
                **key,
                "row_count": len(frame),
                "pooled_city_year_mse": mse,
                "macro_origin_mse": float(origin_frame["origin_mse"].mean()),
                "latest_2021_mse": float(latest["squared_error"].mean()) if len(latest) else None,
                "rmse": float(np.sqrt(mse)),
                "mae": float(frame["residual_prediction_minus_actual"].abs().mean()),
                "bias": float(frame["residual_prediction_minus_actual"].mean()),
                "origin_win_count_vs_b3": origin_wins,
                "origin_count": len(origin_frame),
                "city_sse_total": total_city_sse,
                "top1_city_sse_share": top1,
                "top3_city_sse_share": top3,
                "b3_pooled_city_year_mse": b3_mse,
                "improvement_vs_b3_pct": improvement,
            }
        )

    return aggregate_rows, origin_rows, city_rows, {
        "combined_rows": len(combined),
        "reference_rows": len(reference),
        "candidate_rows": len(all_oof),
    }


def family_classification(
    aggregate_rows: list[dict[str, Any]],
    family: str,
    framing: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in aggregate_rows
        if row["family"] == family
        and row["framing_id"] == framing
        and row["target_id"] == "resident"
    ]
    by_scenario = {row["scenario_id"]: row for row in rows}
    b3_rows = {
        row["scenario_id"]: row
        for row in aggregate_rows
        if row["model_id"] == "B3_REFERENCE" and row["target_id"] == "resident"
    }
    scenario_improvements = {}
    valid_signal_scenarios = []
    competitive_scenarios = []
    for scenario in SCENARIOS:
        row = by_scenario[scenario]
        b3 = b3_rows[scenario]["pooled_city_year_mse"]
        improvement = float((b3 - row["pooled_city_year_mse"]) / b3 * 100.0)
        scenario_improvements[scenario] = improvement
        if (
            improvement >= 2.0
            and row["origin_win_count_vs_b3"] >= 2
            and (row["top1_city_sse_share"] is None or row["top1_city_sse_share"] < 0.75)
        ):
            valid_signal_scenarios.append(scenario)
        if improvement >= -1.0:
            competitive_scenarios.append(scenario)
    both_better = all(scenario_improvements[s] > 0 for s in SCENARIOS)
    stable_rule = both_better or len(valid_signal_scenarios) >= 1
    if stable_rule and all(
        by_scenario[s]["origin_win_count_vs_b3"] >= 2
        and (by_scenario[s]["top1_city_sse_share"] is None or by_scenario[s]["top1_city_sse_share"] < 0.75)
        for s in SCENARIOS
        if scenario_improvements[s] > 0
    ):
        classification = "family_signal_found"
    elif competitive_scenarios:
        classification = "family_competitive_but_unstable"
    else:
        classification = "family_unsupported"
    return {
        "family": family,
        "framing_id": framing,
        "classification": classification,
        "scenario_improvements_pct": scenario_improvements,
        "valid_signal_scenarios": valid_signal_scenarios,
        "competitive_scenarios": competitive_scenarios,
        "primary_target": "resident",
        "rule": "both R1/R2 below B3 or >=2% in one with other no worse than -1%; gain requires >=2 origin wins and top1 city SSE share <75%",
    }


def missingness_rows(
    replay: pd.DataFrame,
    formal: pd.DataFrame,
    inner: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for origin_year in ORIGINS:
                train = replay[
                    (replay.target_id == target_id)
                    & (replay.scenario_id == scenario_id)
                    & (replay.pseudo_target_year <= origin_year - REPLAY_MAX_GAP)
                ]
                validation = formal[
                    (formal.target_id == target_id)
                    & (formal.scenario_id == scenario_id)
                    & (formal.origin_year == origin_year)
                ]
                for split, frame in (("outer_training", train), ("outer_validation", validation)):
                    missing = frame[features].isna()
                    rows.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": origin_year,
                            "split": split,
                            "row_count": len(frame),
                            "feature_count": len(features),
                            "missing_cell_count": int(missing.to_numpy().sum()),
                            "missing_cell_rate": float(missing.to_numpy().mean()),
                            "rows_with_any_missing": int(missing.any(axis=1).sum()),
                            "rows_with_any_missing_rate": float(missing.any(axis=1).mean()),
                            "features_with_missing": int(missing.any(axis=0).sum()),
                            "all_missing_feature_count": int(missing.all(axis=0).sum()),
                        }
                    )
    assert len(rows) == 88
    return rows


def metric_lookup(
    aggregate_rows: list[dict[str, Any]],
    family: str,
    framing: str,
    target: str,
    scenario: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in aggregate_rows
        if row["family"] == family
        and row["framing_id"] == framing
        and row["target_id"] == target
        and row["scenario_id"] == scenario
    ]
    assert len(matches) == 1
    return matches[0]


def handoff_text(
    context: dict[str, Any],
    summary: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    worker_commit: str,
) -> str:
    lines = [
        "# S5-Plan06D Handoff — ExtraTrees 与正则线性全量多表对照 screening",
        "",
        "## Result",
        "",
        f"- Plan ID：{PLAN_ID}",
        f"- Stage：{STAGE}",
        f"- Worker branch：{BRANCH}",
        f"- Exact base SHA：{BASE_SHA}",
        f"- Runner commit used for training：{worker_commit}",
        f"- Final evidence commit：{summary['final_evidence_commit']}",
        f"- Seed：{SEED}；CPU single-thread estimator：n_jobs=1",
        "",
        "本 worker 只比较预注册的 ExtraTrees 与 Ridge 控制族；没有改动 144 个特征、replay universe、outer/inner geometry 或 Plan04/04B/05 证据。",
        "",
        "## Coverage and selection",
        "",
        f"- Formal OOF：每个 family 1760 行；两 family 合计 {summary['candidate_oof_rows']} 行。",
        "- Framings：F1_DIRECT 与 F2_B3_RESIDUAL；targets：resident / registered；scenarios：R1 / R2。",
        f"- Nested selection groups：每个 family 88；合计 {summary['selection_group_count']}。",
        f"- Eligible inner folds：{summary['eligible_inner_fold_count']} / {summary['inner_geometry_row_count']}。",
        f"- Fallback groups：ExtraTrees {summary['families']['EXTRATREES']['fallback_count']}；Ridge {summary['families']['RIDGE']['fallback_count']}。",
        "- Selection objective：eligible inner folds 的 mean validation MSE；tie tolerance 1e-12，按注册配置顺序优先。",
        "- Fallback：eligible folds 少于 3 时使用预注册 conservative default，未扩大边界。",
        "",
        "## Resident primary metrics",
        "",
        "| family | framing | scenario | pooled MSE | B3 MSE | improvement % | origin wins | top1 city SSE share |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("EXTRATREES", "RIDGE"):
        for framing in FRAMINGS:
            for scenario in SCENARIOS:
                row = metric_lookup(aggregate_rows, family, framing, "resident", scenario)
                lines.append(
                    f"| {family} | {framing} | {scenario[:2]} | "
                    f"{row['pooled_city_year_mse']:.6f} | {row['b3_pooled_city_year_mse']:.6f} | "
                    f"{row['improvement_vs_b3_pct']:.3f} | {row['origin_win_count_vs_b3']} | "
                    f"{row['top1_city_sse_share']:.3f} |"
                )
    lines.extend(
        [
            "",
            "Registered-target metrics、macro-origin MSE、latest-2021 MSE、RMSE、MAE、bias、per-origin 与 per-city SSE/MSE 均在 CSV evidence 中提供。",
            "",
            "## Classification",
            "",
        ]
    )
    for item in summary["family_classifications"]:
        lines.append(
            f"- {item['family']} / {item['framing_id']}：{item['classification']}；"
            f" R1/R2 improvement={json.dumps(item['scenario_improvements_pct'], ensure_ascii=False)}"
        )
    lines.extend(
        [
            f"- Family-level screening label — ExtraTrees：{summary['family_overall_classification']['EXTRATREES']}；Ridge：{summary['family_overall_classification']['RIDGE']}。",
            f"- Overall lower resident pooled-MSE family across the registered screening cells：{summary['overall_stronger_family']}。",
            "",
            "## Diagnostics",
            "",
            "- ExtraTrees：outer-selected feature importance、feature group、city one-hot、tree capacity 均写入 feature_importance.csv。",
            "- Ridge：outer-selected coefficient magnitude、feature group、city one-hot、missing indicators、alpha 与稀疏度均写入同一诊断表。",
            "- Fold-local median imputation + missing indicators；Ridge 仅在同一 fold 内追加 StandardScaler；ExtraTrees 不缩放。",
            "- Library/runtime limitations：使用项目已冻结的 Python / numpy / pandas / scikit-learn 环境；ExtraTrees 与 Ridge 均为 CPU、single-thread；没有 GPU 或并行随机性。",
            "",
            "## Boundary",
            "",
            "- No Optuna、Bayesian optimization、random search、large grid、city routing、gating、blend、2023 forecast、submission、S5 complete 或 S6。",
            "- No checkpoint update；本 handoff 不是全局 winner 宣告。下一步由主线在所有 Plan06A/B/C/D worker evidence 汇总后统一判断。",
            "",
            "## Artifacts",
            "",
            "- oof_predictions.csv",
            "- aggregate_metrics.csv",
            "- per_origin_metrics.csv",
            "- configuration_selection.csv",
            "- inner_selection_evidence.csv",
            "- city_model_error.csv",
            "- missingness_diagnostics.csv",
            "- feature_importance.csv",
            "- model_summary.json",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    assert_gate()
    context = load_contract()
    replay, formal, outer, inner = load_inputs(context)
    check_geometry(replay, formal, outer, inner)
    cities = city_order(replay, formal)
    worker_commit = git_text("rev-parse", "HEAD")
    results = {}
    for family in ("EXTRATREES", "RIDGE"):
        print(f"[Plan06D] running {family}", flush=True)
        results[family] = run_family(
            family,
            replay,
            formal,
            inner,
            context["features"],
            context["groups"],
            cities,
            worker_commit,
        )
        print(
            f"[Plan06D] {family} complete: {len(results[family]['oof_rows'])} OOF rows, "
            f"{results[family]['fit_count']} outer fits, "
            f"{results[family]['fallback_count']} fallbacks",
            flush=True,
        )

    all_oof_rows = results["EXTRATREES"]["oof_rows"] + results["RIDGE"]["oof_rows"]
    all_oof = pd.DataFrame(all_oof_rows)
    assert len(all_oof) == 3520
    aggregate_rows, origin_rows, city_rows, metric_meta = calculate_metrics(all_oof, formal)
    family_classifications = [
        family_classification(aggregate_rows, family, framing)
        for family in ("EXTRATREES", "RIDGE")
        for framing in FRAMINGS
    ]
    overall_classification: dict[str, str] = {}
    for family in ("EXTRATREES", "RIDGE"):
        labels = [
            item["classification"]
            for item in family_classifications
            if item["family"] == family
        ]
        if "family_signal_found" in labels:
            overall_classification[family] = "family_signal_found"
        elif "family_competitive_but_unstable" in labels:
            overall_classification[family] = "family_competitive_but_unstable"
        else:
            overall_classification[family] = "family_unsupported"
    resident_rows = [
        row
        for row in aggregate_rows
        if row["family"] in ("EXTRATREES", "RIDGE")
        and row["target_id"] == "resident"
    ]
    family_scores = {
        family: float(
            np.mean(
                [
                    row["pooled_city_year_mse"]
                    for row in resident_rows
                    if row["family"] == family
                ]
            )
        )
        for family in ("EXTRATREES", "RIDGE")
    }
    overall_stronger = min(family_scores, key=lambda family: (family_scores[family], family))
    env = {}
    env_path = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "environment_versions.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "runner_commit_at_execution": worker_commit,
        "final_evidence_commit": git_text("rev-parse", "HEAD"),
        "seed": SEED,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "formal_row_count": len(formal),
        "replay_row_count": len(replay),
        "model_feature_count": len(context["features"]),
        "candidate_oof_rows": len(all_oof),
        "selection_group_count": sum(len(results[family]["selection_rows"]) for family in results),
        "inner_geometry_row_count": len(inner),
        "eligible_inner_fold_count": int(
            (
                (inner["inner_training_row_count"] >= 40)
                & (inner["inner_validation_row_count"] >= 20)
                & (inner["boundary_pass"] == "PASS")
            ).sum()
        ),
        "families": {
            family: {
                "model_id": results[family]["model_id"],
                "fit_count": results[family]["fit_count"],
                "fallback_count": results[family]["fallback_count"],
                "selected_config_counts": results[family]["selected_config_counts"],
                "inner_evidence_row_count": len(results[family]["evidence_rows"]),
                "diagnostic_row_count": len(results[family]["diagnostic_rows"]),
            }
            for family in results
        },
        "family_classifications": family_classifications,
        "family_overall_classification": overall_classification,
        "family_mean_resident_pooled_mse": family_scores,
        "overall_stronger_family": overall_stronger,
        "metric_meta": metric_meta,
        "environment": env,
        "preprocessing": {
            "numeric": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True), fold-local",
            "city_id": "fixed deterministic 40-city one-hot, no target encoding",
            "extratrees": "no scaling after imputation; n_jobs=1",
            "ridge": "StandardScaler after fold-local imputation and one-hot",
        },
        "forbidden_actions_not_run": [
            "optuna",
            "bayesian_optimization",
            "random_search",
            "large_grid_search",
            "routing",
            "gating",
            "blend",
            "2023_forecast",
            "submission",
            "s5_complete",
            "s6_entered",
        ],
        "elapsed_seconds_before_report_write": float(time.time() - started),
    }

    output_fields_oof = [
        "model_id", "family", "framing_id", "target_id", "scenario_id",
        "origin_year", "city_id", "actual_target_value", "prediction_value",
        "residual_prediction_minus_actual", "squared_error",
        "training_max_label_year", "feature_contract_version", "worker_commit",
        "selected_config_id", "selection_method",
    ]
    output_fields_aggregate = [
        "model_id", "family", "framing_id", "target_id", "scenario_id",
        "row_count", "pooled_city_year_mse", "macro_origin_mse",
        "latest_2021_mse", "rmse", "mae", "bias", "origin_win_count_vs_b3",
        "origin_count", "city_sse_total", "top1_city_sse_share",
        "top3_city_sse_share", "b3_pooled_city_year_mse", "improvement_vs_b3_pct",
    ]
    output_fields_origin = [
        "model_id", "family", "framing_id", "target_id", "scenario_id",
        "origin_year", "row_count", "origin_mse", "origin_rmse", "origin_mae",
        "origin_bias", "b3_origin_mse", "origin_win_vs_b3",
    ]
    output_fields_selection = [
        "family", "framing_id", "target_id", "scenario_id", "outer_origin_year",
        "selected_config_id", "selection_method", "eligible_inner_fold_count",
        "config_1_id", "config_1_mean_inner_mse", "config_2_id",
        "config_2_mean_inner_mse", "config_3_id", "config_3_mean_inner_mse",
        "selected_mean_inner_mse",
    ]
    output_fields_evidence = [
        "family", "framing_id", "target_id", "scenario_id", "outer_origin_year",
        "config_id", "config_order", "inner_validation_year",
        "inner_training_row_count", "inner_validation_row_count",
        "validation_mse", "eligible_fold_count", "selection_status",
        "selection_method",
    ]
    output_fields_city = [
        "model_id", "family", "framing_id", "target_id", "scenario_id", "city_id",
        "row_count", "city_sse", "city_mse", "city_rmse", "city_bias",
    ]
    output_fields_missing = [
        "target_id", "scenario_id", "outer_origin_year", "split", "row_count",
        "feature_count", "missing_cell_count", "missing_cell_rate",
        "rows_with_any_missing", "rows_with_any_missing_rate",
        "features_with_missing", "all_missing_feature_count",
    ]
    output_fields_diagnostic = [
        "family", "model_id", "framing_id", "target_id", "scenario_id",
        "outer_origin_year", "selected_config_id", "feature_name",
        "feature_group", "feature_role", "diagnostic_type",
        "importance_or_coefficient", "absolute_magnitude",
        "capacity_metric", "capacity_value",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT_DIR / "oof_predictions.csv", all_oof_rows, output_fields_oof)
    write_rows(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_rows, output_fields_aggregate)
    write_rows(OUTPUT_DIR / "per_origin_metrics.csv", origin_rows, output_fields_origin)
    write_rows(
        OUTPUT_DIR / "configuration_selection.csv",
        results["EXTRATREES"]["selection_rows"] + results["RIDGE"]["selection_rows"],
        output_fields_selection,
    )
    write_rows(
        OUTPUT_DIR / "inner_selection_evidence.csv",
        results["EXTRATREES"]["evidence_rows"] + results["RIDGE"]["evidence_rows"],
        output_fields_evidence,
    )
    write_rows(OUTPUT_DIR / "city_model_error.csv", city_rows, output_fields_city)
    write_rows(
        OUTPUT_DIR / "missingness_diagnostics.csv",
        missingness_rows(replay, formal, inner, context["features"]),
        output_fields_missing,
    )
    write_rows(
        OUTPUT_DIR / "feature_importance.csv",
        results["EXTRATREES"]["diagnostic_rows"] + results["RIDGE"]["diagnostic_rows"],
        output_fields_diagnostic,
    )
    summary["final_evidence_commit"] = git_text("rev-parse", "HEAD")
    summary["elapsed_seconds"] = float(time.time() - started)
    write_json(OUTPUT_DIR / "model_summary.json", summary)
    (OUTPUT_DIR / "HANDOFF.md").write_text(
        handoff_text(context, summary, aggregate_rows, worker_commit),
        encoding="utf-8",
    )
    print(f"[Plan06D] reports written to {OUTPUT_DIR}", flush=True)
    print(f"[Plan06D] elapsed seconds: {summary['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
