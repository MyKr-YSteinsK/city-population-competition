"""S5 Plan06A: pre-registered LightGBM screening on the frozen 144-feature contract."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan06A"
BRANCH = "s5/plan06a-lightgbm-screening"
STARTING_COMMIT = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
MODEL_ID = "LGBM_PLAN06A"
FEATURE_CONTRACT_VERSION = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT_VERSION = "S5-Plan04B-TRAINING-REPLAY-V1"
RANDOM_SEED = 20260919
TIE_ATOL = 1e-12
TIE_RTOL = 1e-12
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
FRAMINGS = ("F1_DIRECT", "F2_B3_RESIDUAL")
ORIGINS = tuple(range(2011, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))
CITY_FEATURES = tuple(f"city_onehot_{city}" for city in CITIES)
FEATURE_COLUMNS: tuple[str, ...] = ()

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "config_id": "LGB_C1_CONSERVATIVE",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "subsample": 0.90,
        "colsample_bytree": 0.70,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    },
    {
        "config_id": "LGB_C2_MEDIUM",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 15,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    {
        "config_id": "LGB_C3_FLEXIBLE_REGULARIZED",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.25,
        "reg_lambda": 3.0,
    },
)
CONFIG_BY_ID = {config["config_id"]: config for config in CONFIGS}

OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
SCHEMA_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
FORMAL_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv"
REPLAY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv"
WAVE_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "strong_model_wave_contract.json"
INPUT_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json"
OUTER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "outer_training_geometry.csv"
INNER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "inner_split_geometry.csv"
OOF_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "oof_prediction_contract.csv"
ENVIRONMENT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "environment_versions.json"

OOF_FIELDS = (
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
    "eligible_inner_fold_count",
)
SELECTION_FIELDS = (
    "model_id",
    "target_id",
    "scenario_id",
    "outer_origin_year",
    "framing_id",
    "selected_config_id",
    "selection_method",
    "eligible_inner_fold_count",
    "outer_training_row_count",
    "outer_validation_row_count",
    "training_max_label_year",
    "selected_inner_mean_validation_mse",
    "LGB_C1_CONSERVATIVE_mean_validation_mse",
    "LGB_C2_MEDIUM_mean_validation_mse",
    "LGB_C3_FLEXIBLE_REGULARIZED_mean_validation_mse",
    "selected_num_leaves",
    "selected_max_depth",
)
INNER_FIELDS = (
    "model_id",
    "target_id",
    "scenario_id",
    "outer_origin_year",
    "framing_id",
    "inner_validation_year",
    "inner_training_max_year",
    "inner_training_row_count",
    "inner_validation_row_count",
    "eligible_inner_fold_count",
    "config_id",
    "evaluation_status",
    "validation_mse",
    "mean_validation_mse_for_config",
    "selected_config_id",
    "selection_method",
    "outer_labels_used_for_selection",
)


def git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell_text(row.get(field)) for field in fieldnames})


def parse_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def city_number(value: str) -> int:
    if not value.startswith("city"):
        raise AssertionError(f"unexpected city identity: {value}")
    return int(value[4:])


def sort_rows(frame: pd.DataFrame, year_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["_city_number_internal"] = result["city_id"].map(city_number)
    result = result.sort_values(
        ["target_id", "scenario_id", year_column, "_city_number_internal"],
        kind="mergesort",
    )
    return result.drop(columns=["_city_number_internal"]).reset_index(drop=True)


def ensure_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() != frame[column].notna().sum():
            raise AssertionError(f"non-numeric values in {column}")
        frame[column] = converted


def feature_group_map(schema: dict[str, Any], model_features: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group_id, group in schema["feature_groups"].items():
        for feature in group["features"]:
            if feature in mapping:
                raise AssertionError(f"feature appears in multiple groups: {feature}")
            mapping[feature] = group_id
    missing = [feature for feature in model_features if feature not in mapping]
    if missing:
        raise AssertionError(f"model features missing from feature groups: {missing}")
    return mapping


def load_environment() -> dict[str, str]:
    import sklearn

    versions = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    recorded = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    for name in ("numpy", "pandas", "scikit_learn", "lightgbm"):
        if versions[name] != recorded[name]:
            raise AssertionError(
                f"environment drift for {name}: {versions[name]} != {recorded[name]}"
            )
    if versions["lightgbm"] != "4.7.0":
        raise AssertionError("Plan06A requires LightGBM 4.7.0")
    return versions


def build_features(frame: pd.DataFrame, model_features: tuple[str, ...]) -> np.ndarray:
    numeric = frame.loc[:, list(model_features)].to_numpy(dtype=np.float64)
    city_onehot = np.zeros((len(frame), len(CITIES)), dtype=np.float64)
    for row_index, city in enumerate(frame["city_id"].tolist()):
        city_onehot[row_index, city_number(city) - 1] = 1.0
    result = np.concatenate([numeric, city_onehot], axis=1)
    if result.shape[1] != 184:
        raise AssertionError(f"unexpected LightGBM feature width: {result.shape[1]}")
    return result


def validate_geometry(
    formal: pd.DataFrame,
    replay: pd.DataFrame,
    outer_geometry: pd.DataFrame,
    inner_geometry: pd.DataFrame,
) -> dict[str, Any]:
    outer_mismatch = 0
    outer_boundary = 0
    inner_mismatch = 0
    inner_boundary = 0
    eligible_count = 0
    eligible_by_outer: Counter[str] = Counter()

    for row in outer_geometry.to_dict("records"):
        target = row["target_id"]
        scenario = row["scenario_id"]
        origin = int(row["outer_origin_year"])
        train_mask = (
            (replay["target_id"] == target)
            & (replay["scenario_id"] == scenario)
            & (replay["pseudo_target_year"] <= origin - 2)
        )
        validation_mask = (
            (formal["target_id"] == target)
            & (formal["scenario_id"] == scenario)
            & (formal["origin_year"] == origin)
        )
        train = replay.loc[train_mask]
        validation = formal.loc[validation_mask]
        if len(train) != int(row["training_row_count"]):
            outer_mismatch += 1
        if len(validation) != int(row["validation_row_count"]):
            outer_mismatch += 1
        if len(train) and int(train["pseudo_target_year"].max()) > origin - 2:
            outer_boundary += 1
        if int(row["forbidden_training_rows_after_cutoff"]) != 0:
            outer_boundary += 1
        if row["boundary_pass"] != "PASS":
            outer_boundary += 1

    for row in inner_geometry.to_dict("records"):
        target = row["target_id"]
        scenario = row["scenario_id"]
        origin = int(row["outer_origin_year"])
        validation_year = int(row["inner_validation_year"])
        outer_mask = (
            (replay["target_id"] == target)
            & (replay["scenario_id"] == scenario)
            & (replay["pseudo_target_year"] <= origin - 2)
        )
        validation_mask = outer_mask & (replay["pseudo_target_year"] == validation_year)
        training_max = parse_int(row["inner_training_max_year"])
        training_mask = (
            outer_mask & (replay["pseudo_target_year"] <= training_max)
            if training_max is not None
            else outer_mask & False
        )
        training = replay.loc[training_mask]
        validation = replay.loc[validation_mask]
        if len(training) != int(row["inner_training_row_count"]):
            inner_mismatch += 1
        if len(validation) != int(row["inner_validation_row_count"]):
            inner_mismatch += 1
        if int(row["inner_training_rows_after_cutoff"]) != 0:
            inner_boundary += 1
        if row["boundary_pass"] != "PASS":
            inner_boundary += 1
        if len(training) and int(training["pseudo_target_year"].max()) > origin - 2:
            inner_boundary += 1
        if validation_year > origin - 2:
            inner_boundary += 1
        eligible = (
            int(row["inner_training_row_count"]) >= 40
            and int(row["inner_validation_row_count"]) >= 20
            and row["boundary_pass"] == "PASS"
        )
        if eligible:
            eligible_count += 1
            eligible_by_outer[f"{target}|{scenario}|{origin}"] += 1

    fallback = sum(count < 3 for count in eligible_by_outer.values())
    if (
        outer_mismatch
        or outer_boundary
        or inner_mismatch
        or inner_boundary
        or eligible_count != 264
        or len(eligible_by_outer) != 44
        or fallback != 8
    ):
        raise AssertionError(
            "geometry validation failed: "
            f"outer_mismatch={outer_mismatch}, outer_boundary={outer_boundary}, "
            f"inner_mismatch={inner_mismatch}, inner_boundary={inner_boundary}, "
            f"eligible={eligible_count}, fallback={fallback}"
        )
    return {
        "outer_geometry_rows": int(len(outer_geometry)),
        "inner_geometry_rows": int(len(inner_geometry)),
        "eligible_inner_fold_count": eligible_count,
        "outer_fit_group_count": len(eligible_by_outer),
        "outer_fit_groups_with_at_least_3_eligible_folds": sum(
            count >= 3 for count in eligible_by_outer.values()
        ),
        "fallback_outer_fit_count": fallback,
        "outer_mismatch_count": outer_mismatch,
        "outer_boundary_violation_count": outer_boundary,
        "inner_mismatch_count": inner_mismatch,
        "inner_boundary_violation_count": inner_boundary,
        "eligible_folds_by_outer_fit": dict(sorted(eligible_by_outer.items())),
    }


def validate_b3_reference(formal: pd.DataFrame, wave_contract: dict[str, Any]) -> dict[str, Any]:
    expected = wave_contract["outer_evaluation_contract"]["b3_reference_pooled_mse"]
    result: dict[str, Any] = {}
    for target in TARGETS:
        result[target] = {}
        for scenario in SCENARIOS:
            group = formal[
                (formal["target_id"] == target)
                & (formal["scenario_id"] == scenario)
            ]
            error = group["b3_prediction"].to_numpy(dtype=float) - group[
                "actual_target_value"
            ].to_numpy(dtype=float)
            mse = float(np.mean(error**2))
            result[target][scenario] = {
                "pooled_city_year_mse": mse,
                "absolute_delta_vs_contract": abs(mse - float(expected[target])),
                "row_count": int(len(group)),
            }
            if len(group) != 440 or abs(mse - float(expected[target])) > 1e-9:
                raise AssertionError(f"B3 reference mismatch for {target}/{scenario}")
    return result


def load_context() -> dict[str, Any]:
    if git_text("branch", "--show-current") != BRANCH:
        raise AssertionError(f"runner must execute on {BRANCH}")
    if git_text("merge-base", STARTING_COMMIT, "HEAD") != STARTING_COMMIT:
        raise AssertionError("worker branch is not based on the exact Wave base")
    required = (
        SCHEMA_PATH,
        FORMAL_PATH,
        REPLAY_PATH,
        WAVE_CONTRACT_PATH,
        INPUT_CONTRACT_PATH,
        OUTER_GEOMETRY_PATH,
        INNER_GEOMETRY_PATH,
        OOF_CONTRACT_PATH,
        ENVIRONMENT_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    wave_contract = json.loads(WAVE_CONTRACT_PATH.read_text(encoding="utf-8"))
    input_contract = json.loads(INPUT_CONTRACT_PATH.read_text(encoding="utf-8"))
    oof_contract = pd.read_csv(OOF_CONTRACT_PATH)
    formal = pd.read_csv(FORMAL_PATH)
    replay = pd.read_csv(REPLAY_PATH)
    outer_geometry = pd.read_csv(OUTER_GEOMETRY_PATH)
    inner_geometry = pd.read_csv(INNER_GEOMETRY_PATH)
    model_features = tuple(schema["model_features"])
    if len(model_features) != 144 or len(set(model_features)) != 144:
        raise AssertionError("model feature contract is not exactly 144 unique features")
    group_map = feature_group_map(schema, model_features)
    if schema["feature_contract_version"] != FEATURE_CONTRACT_VERSION:
        raise AssertionError("feature contract version mismatch")
    if input_contract["contract_version"] != REPLAY_CONTRACT_VERSION:
        raise AssertionError("replay contract version mismatch")
    if wave_contract["feature_contract_version"] != FEATURE_CONTRACT_VERSION:
        raise AssertionError("Wave feature contract mismatch")
    if wave_contract["replay_contract_version"] != REPLAY_CONTRACT_VERSION:
        raise AssertionError("Wave replay contract mismatch")
    if tuple(wave_contract["data_contract"]["target_ids"]) != TARGETS:
        raise AssertionError("target contract mismatch")
    if tuple(wave_contract["data_contract"]["scenario_ids"]) != SCENARIOS:
        raise AssertionError("scenario contract mismatch")
    required_oof = set(wave_contract["oof_contract"]["required_columns"])
    if not required_oof.issubset(set(oof_contract.columns)):
        raise AssertionError("OOF required columns changed")

    ensure_numeric(
        formal,
        (
            "origin_year",
            "target_history_cutoff_year",
            "actual_target_value",
            "b3_prediction",
            "b3_residual_actual_minus_prediction",
            "b3_squared_error",
        ),
    )
    ensure_numeric(
        replay,
        (
            "pseudo_target_year",
            "target_history_cutoff_year",
            "actual_target_value",
            "b3_prediction",
            "b3_residual_actual_minus_prediction",
            "b3_squared_error",
        ),
    )
    for frame in (formal, replay):
        ensure_numeric(frame, model_features)
        if not set(frame["city_id"]).issubset(set(CITIES)):
            raise AssertionError("unexpected city outside fixed vocabulary")
        if np.isinf(frame.loc[:, list(model_features)].to_numpy(dtype=np.float64)).sum():
            raise AssertionError("infinite numeric feature value is not allowed")
    formal = sort_rows(formal, "origin_year")
    replay = sort_rows(replay, "pseudo_target_year")
    outer_geometry = outer_geometry.sort_values(
        ["target_id", "scenario_id", "outer_origin_year"], kind="mergesort"
    ).reset_index(drop=True)
    inner_geometry = inner_geometry.sort_values(
        ["target_id", "scenario_id", "outer_origin_year", "inner_validation_year"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(formal) != 1760 or len(replay) != 2320:
        raise AssertionError("frozen matrix row counts changed")
    if len(outer_geometry) != 44 or len(inner_geometry) != 484:
        raise AssertionError("frozen geometry row counts changed")
    if formal[["target_id", "scenario_id", "origin_year", "city_id"]].duplicated().any():
        raise AssertionError("formal validation key is not unique")
    if replay[["target_id", "scenario_id", "pseudo_target_year", "city_id"]].duplicated().any():
        raise AssertionError("replay key is not unique")
    if sorted(formal["origin_year"].unique()) != list(ORIGINS):
        raise AssertionError("formal origins changed")
    if sorted(replay["pseudo_target_year"].unique()) != list(range(2004, 2022)):
        raise AssertionError("replay pseudo-target years changed")
    if set(formal["feature_contract_version"]) != {FEATURE_CONTRACT_VERSION}:
        raise AssertionError("formal feature contract labels changed")
    if set(replay["feature_contract_version"]) != {FEATURE_CONTRACT_VERSION}:
        raise AssertionError("replay feature contract labels changed")
    environment = load_environment()
    source_commit = os.environ.get("S5_WORKER_COMMIT") or git_text("rev-parse", "HEAD")
    return {
        "schema": schema,
        "wave_contract": wave_contract,
        "input_contract": input_contract,
        "formal": formal,
        "replay": replay,
        "outer_geometry": outer_geometry,
        "inner_geometry": inner_geometry,
        "model_features": model_features,
        "group_map": group_map,
        "formal_x": build_features(formal, model_features),
        "replay_x": build_features(replay, model_features),
        "geometry_audit": validate_geometry(formal, replay, outer_geometry, inner_geometry),
        "b3_reference": validate_b3_reference(formal, wave_contract),
        "environment": environment,
        "source_commit": source_commit,
        "contract_declared_wave_base": wave_contract.get("wave_base_sha"),
    }


def outer_training_mask(replay: pd.DataFrame, target: str, scenario: str, origin: int) -> pd.Series:
    return (
        (replay["target_id"] == target)
        & (replay["scenario_id"] == scenario)
        & (replay["pseudo_target_year"] <= origin - 2)
    )


def formal_validation_mask(formal: pd.DataFrame, target: str, scenario: str, origin: int) -> pd.Series:
    return (
        (formal["target_id"] == target)
        & (formal["scenario_id"] == scenario)
        & (formal["origin_year"] == origin)
    )


def fit_model(config_id: str, features: np.ndarray, labels: np.ndarray) -> LGBMRegressor:
    config = CONFIG_BY_ID[config_id]
    params = {
        "objective": "regression",
        "random_state": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "data_random_seed": RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "subsample_freq": 1,
        **{key: value for key, value in config.items() if key != "config_id"},
    }
    estimator = LGBMRegressor(**params)
    estimator.fit(features, labels)
    return estimator


def prediction_for_framing(
    estimator: LGBMRegressor,
    features: np.ndarray,
    frame: pd.DataFrame,
    framing_id: str,
) -> np.ndarray:
    prediction = np.asarray(estimator.predict(features), dtype=float)
    if framing_id == "F2_B3_RESIDUAL":
        prediction = prediction + frame["b3_prediction"].to_numpy(dtype=float)
    return prediction


def close_enough(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, atol=TIE_ATOL, rtol=TIE_RTOL))


def select_config(
    context: dict[str, Any],
    target: str,
    scenario: str,
    origin: int,
    framing_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int, int]:
    replay = context["replay"]
    replay_x = context["replay_x"]
    outer_mask = outer_training_mask(replay, target, scenario, origin)
    outer_rows = replay.loc[outer_mask]
    geometry = context["inner_geometry"]
    folds = geometry[
        (geometry["target_id"] == target)
        & (geometry["scenario_id"] == scenario)
        & (geometry["outer_origin_year"] == origin)
    ]
    eligible = folds[
        (folds["inner_training_row_count"] >= 40)
        & (folds["inner_validation_row_count"] >= 20)
        & (folds["boundary_pass"] == "PASS")
    ]
    eligible_count = int(len(eligible))
    evidence: list[dict[str, Any]] = []
    mean_scores: dict[str, float] = {}
    if eligible_count < 3:
        selected_id = CONFIGS[0]["config_id"]
        selection_method = "fallback_conservative_default"
        for config in CONFIGS:
            evidence.append(
                {
                    "model_id": MODEL_ID,
                    "target_id": target,
                    "scenario_id": scenario,
                    "outer_origin_year": origin,
                    "framing_id": framing_id,
                    "inner_validation_year": "",
                    "inner_training_max_year": "",
                    "inner_training_row_count": "",
                    "inner_validation_row_count": "",
                    "eligible_inner_fold_count": eligible_count,
                    "config_id": config["config_id"],
                    "evaluation_status": "fallback_not_evaluated",
                    "validation_mse": "",
                    "mean_validation_mse_for_config": "",
                    "selected_config_id": selected_id,
                    "selection_method": selection_method,
                    "outer_labels_used_for_selection": "false",
                }
            )
    else:
        fold_scores: dict[str, list[float]] = {config["config_id"]: [] for config in CONFIGS}
        evaluated_rows: list[dict[str, Any]] = []
        for fold in eligible.to_dict("records"):
            validation_year = int(fold["inner_validation_year"])
            training_max = int(fold["inner_training_max_year"])
            training_mask = outer_mask & (replay["pseudo_target_year"] <= training_max)
            validation_mask = outer_mask & (replay["pseudo_target_year"] == validation_year)
            train_indices = np.flatnonzero(training_mask.to_numpy())
            validation_indices = np.flatnonzero(validation_mask.to_numpy())
            train_frame = replay.loc[training_mask]
            validation_frame = replay.loc[validation_mask]
            if len(train_frame) != int(fold["inner_training_row_count"]):
                raise AssertionError("inner training geometry changed")
            if len(validation_frame) != int(fold["inner_validation_row_count"]):
                raise AssertionError("inner validation geometry changed")
            labels = train_frame[
                "actual_target_value"
                if framing_id == "F1_DIRECT"
                else "b3_residual_actual_minus_prediction"
            ].to_numpy(dtype=float)
            for config in CONFIGS:
                estimator = fit_model(config["config_id"], replay_x[train_indices], labels)
                prediction = prediction_for_framing(
                    estimator,
                    replay_x[validation_indices],
                    validation_frame,
                    framing_id,
                )
                actual = validation_frame["actual_target_value"].to_numpy(dtype=float)
                mse = float(np.mean((prediction - actual) ** 2))
                fold_scores[config["config_id"]].append(mse)
                evaluated_rows.append(
                    {
                        "model_id": MODEL_ID,
                        "target_id": target,
                        "scenario_id": scenario,
                        "outer_origin_year": origin,
                        "framing_id": framing_id,
                        "inner_validation_year": validation_year,
                        "inner_training_max_year": training_max,
                        "inner_training_row_count": len(train_frame),
                        "inner_validation_row_count": len(validation_frame),
                        "eligible_inner_fold_count": eligible_count,
                        "config_id": config["config_id"],
                        "evaluation_status": "evaluated",
                        "validation_mse": mse,
                        "mean_validation_mse_for_config": "",
                        "selected_config_id": "",
                        "selection_method": "inner_mean_mse",
                        "outer_labels_used_for_selection": "false",
                    }
                )
        mean_scores = {
            config_id: float(np.mean(scores))
            for config_id, scores in fold_scores.items()
        }
        selected_id = CONFIGS[0]["config_id"]
        for config in CONFIGS[1:]:
            candidate_id = config["config_id"]
            if not close_enough(mean_scores[candidate_id], mean_scores[selected_id]):
                if mean_scores[candidate_id] < mean_scores[selected_id]:
                    selected_id = candidate_id
        selection_method = "inner_mean_mse"
        for row in evaluated_rows:
            row["mean_validation_mse_for_config"] = mean_scores[row["config_id"]]
            row["selected_config_id"] = selected_id
        evidence.extend(evaluated_rows)

    selected = CONFIG_BY_ID[selected_id]
    training_max_label_year = int(outer_rows["pseudo_target_year"].max())
    if training_max_label_year > origin - 2:
        raise AssertionError("outer labels cross the two-year boundary")
    selection = {
        "model_id": MODEL_ID,
        "target_id": target,
        "scenario_id": scenario,
        "outer_origin_year": origin,
        "framing_id": framing_id,
        "selected_config_id": selected_id,
        "selection_method": selection_method,
        "eligible_inner_fold_count": eligible_count,
        "outer_training_row_count": len(outer_rows),
        "outer_validation_row_count": 40,
        "training_max_label_year": training_max_label_year,
        "selected_inner_mean_validation_mse": mean_scores.get(selected_id),
        "LGB_C1_CONSERVATIVE_mean_validation_mse": mean_scores.get("LGB_C1_CONSERVATIVE"),
        "LGB_C2_MEDIUM_mean_validation_mse": mean_scores.get("LGB_C2_MEDIUM"),
        "LGB_C3_FLEXIBLE_REGULARIZED_mean_validation_mse": mean_scores.get(
            "LGB_C3_FLEXIBLE_REGULARIZED"
        ),
        "selected_num_leaves": selected["num_leaves"],
        "selected_max_depth": selected["max_depth"],
    }
    return selected, selection, evidence, eligible_count, (
        eligible_count * len(CONFIGS) if eligible_count >= 3 else 0
    )


def train_outer_models(context: dict[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    formal = context["formal"]
    replay = context["replay"]
    formal_x = context["formal_x"]
    replay_x = context["replay_x"]
    oof_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    inner_fit_count = 0
    for target in TARGETS:
        for scenario in SCENARIOS:
            for origin in ORIGINS:
                training_mask = outer_training_mask(replay, target, scenario, origin)
                training_indices = np.flatnonzero(training_mask.to_numpy())
                training_frame = replay.loc[training_mask]
                validation_mask = formal_validation_mask(formal, target, scenario, origin)
                validation_indices = np.flatnonzero(validation_mask.to_numpy())
                validation = formal.loc[validation_mask].reset_index(drop=True)
                if len(validation) != 40:
                    raise AssertionError("formal validation coverage changed")
                for framing_id in FRAMINGS:
                    selected, selection, evidence, eligible_count, fit_count = select_config(
                        context, target, scenario, origin, framing_id
                    )
                    selection_rows.append(selection)
                    inner_rows.extend(evidence)
                    inner_fit_count += fit_count
                    labels = training_frame[
                        "actual_target_value"
                        if framing_id == "F1_DIRECT"
                        else "b3_residual_actual_minus_prediction"
                    ].to_numpy(dtype=float)
                    estimator = fit_model(
                        selected["config_id"], replay_x[training_indices], labels
                    )
                    predictions = prediction_for_framing(
                        estimator, formal_x[validation_indices], validation, framing_id
                    )
                    if len(predictions) != 40 or not np.isfinite(predictions).all():
                        raise AssertionError("outer prediction coverage/value failure")
                    fit_records.append(
                        {
                            "target_id": target,
                            "scenario_id": scenario,
                            "outer_origin_year": origin,
                            "framing_id": framing_id,
                            "selected_config_id": selected["config_id"],
                            "estimator": estimator,
                        }
                    )
                    actual = validation["actual_target_value"].to_numpy(dtype=float)
                    for row_index, row in validation.iterrows():
                        prediction = float(predictions[row_index])
                        actual_value = float(actual[row_index])
                        oof_rows.append(
                            {
                                "model_id": MODEL_ID,
                                "framing_id": framing_id,
                                "target_id": target,
                                "scenario_id": scenario,
                                "origin_year": origin,
                                "city_id": row["city_id"],
                                "actual_target_value": actual_value,
                                "prediction_value": prediction,
                                "residual_prediction_minus_actual": prediction - actual_value,
                                "squared_error": (prediction - actual_value) ** 2,
                                "training_max_label_year": selection["training_max_label_year"],
                                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                                "worker_commit": context["source_commit"],
                                "selected_config_id": selected["config_id"],
                                "selection_method": selection["selection_method"],
                                "eligible_inner_fold_count": eligible_count,
                            }
                        )
    if len(oof_rows) != 3520 or len(selection_rows) != 88 or len(fit_records) != 88:
        raise AssertionError("unexpected OOF or outer-fit count")
    return oof_rows, selection_rows, inner_rows, fit_records, inner_fit_count


def city_stats(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(
            _sse=frame["residual_prediction_minus_actual"] ** 2,
            _absolute_error=frame["residual_prediction_minus_actual"].abs(),
        )
        .groupby("city_id", sort=False)
        .agg(
            row_count=("city_id", "size"),
            sse=("_sse", "sum"),
            mse=("_sse", "mean"),
            mae=("_absolute_error", "mean"),
            bias=("residual_prediction_minus_actual", "mean"),
        )
        .reset_index()
    )


def metric_row(
    frame: pd.DataFrame,
    model_id: str,
    framing_id: str,
    target: str,
    scenario: str,
    b3_metrics: dict[str, Any] | None,
    is_reference: bool = False,
) -> dict[str, Any]:
    error = frame["residual_prediction_minus_actual"].to_numpy(dtype=float)
    mse = float(np.mean(error**2))
    per_origin = {
        int(origin): float(
            np.mean(
                frame.loc[
                    frame["origin_year"] == origin,
                    "residual_prediction_minus_actual",
                ].to_numpy(dtype=float)
                ** 2
            )
        )
        for origin in sorted(frame["origin_year"].unique())
    }
    b3_origin = (b3_metrics or {}).get("_per_origin_mse", {})
    wins = (
        sum(per_origin[origin] < float(b3_origin[origin]) - 1e-12 for origin in per_origin)
        if not is_reference
        else None
    )
    city = city_stats(frame).sort_values(["sse", "city_id"], ascending=[False, True])
    total_sse = float(city["sse"].sum())
    b3_mse = None if is_reference or b3_metrics is None else b3_metrics["pooled_city_year_mse"]
    return {
        "model_id": model_id,
        "framing_id": framing_id,
        "target_id": target,
        "scenario_id": scenario,
        "pooled_city_year_mse": mse,
        "macro_origin_mse": float(np.mean(list(per_origin.values()))),
        "origin_win_count_vs_b3": wins,
        "latest_2021_mse": per_origin.get(2021),
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "city_level_sse": total_sse,
        "top1_city_sse_concentration": float(city.iloc[0]["sse"] / total_sse) if total_sse else None,
        "top3_city_sse_concentration": float(city.head(3)["sse"].sum() / total_sse) if total_sse else None,
        "top1_city": city.iloc[0]["city_id"] if len(city) else "",
        "top3_cities": ";".join(city.head(3)["city_id"].tolist()),
        "b3_pooled_city_year_mse": b3_mse,
        "pooled_delta_vs_b3": mse - b3_mse if b3_mse is not None else None,
        "relative_improvement_pct_vs_b3": (
            (b3_mse - mse) / b3_mse * 100.0 if b3_mse not in (None, 0) else None
        ),
        "row_count": len(frame),
        "is_reference": "true" if is_reference else "false",
        "_per_origin_mse": per_origin,
    }


def baseline_frame(formal: pd.DataFrame, target: str, scenario: str) -> pd.DataFrame:
    result = formal[
        (formal["target_id"] == target) & (formal["scenario_id"] == scenario)
    ].copy()
    result["prediction_value"] = result["b3_prediction"].astype(float)
    result["residual_prediction_minus_actual"] = (
        result["prediction_value"] - result["actual_target_value"]
    )
    result["squared_error"] = result["residual_prediction_minus_actual"] ** 2
    return result


def metric_bundle(
    oof: pd.DataFrame,
    formal: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    per_origin: list[dict[str, Any]] = []
    city_error: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario in SCENARIOS:
            baseline = baseline_frame(formal, target, scenario)
            baseline_metric = metric_row(
                baseline, "B3_BASELINE", "B3_REFERENCE", target, scenario, None, True
            )
            baseline_origin = baseline_metric["_per_origin_mse"]
            aggregate.append(baseline_metric)
            baseline_city = city_stats(baseline).sort_values(
                ["sse", "city_id"], ascending=[False, True]
            )
            for rank, city_row in enumerate(baseline_city.to_dict("records"), 1):
                city_error.append(
                    {
                        "model_id": "B3_BASELINE",
                        "framing_id": "B3_REFERENCE",
                        "target_id": target,
                        "scenario_id": scenario,
                        "city_id": city_row["city_id"],
                        **city_row,
                        "sse_share": city_row["sse"] / baseline_metric["city_level_sse"],
                        "rank_by_sse": rank,
                    }
                )
            for origin in ORIGINS:
                origin_frame = baseline[baseline["origin_year"] == origin]
                error = origin_frame["residual_prediction_minus_actual"].to_numpy(dtype=float)
                mse = float(np.mean(error**2))
                per_origin.append(
                    {
                        "model_id": "B3_BASELINE",
                        "framing_id": "B3_REFERENCE",
                        "target_id": target,
                        "scenario_id": scenario,
                        "origin_year": origin,
                        "mse": mse,
                        "rmse": math.sqrt(mse),
                        "mae": float(np.mean(np.abs(error))),
                        "bias": float(np.mean(error)),
                        "city_count": len(origin_frame),
                        "b3_mse": "",
                        "delta_vs_b3": "",
                        "relative_improvement_pct_vs_b3": "",
                        "win_vs_b3": "",
                    }
                )
            for framing in FRAMINGS:
                candidate = oof[
                    (oof["target_id"] == target)
                    & (oof["scenario_id"] == scenario)
                    & (oof["framing_id"] == framing)
                ]
                candidate_metric = metric_row(
                    candidate, MODEL_ID, framing, target, scenario, baseline_metric
                )
                aggregate.append(candidate_metric)
                candidate_origin = candidate_metric["_per_origin_mse"]
                candidate_city = city_stats(candidate).sort_values(
                    ["sse", "city_id"], ascending=[False, True]
                )
                for rank, city_row in enumerate(candidate_city.to_dict("records"), 1):
                    city_error.append(
                        {
                            "model_id": MODEL_ID,
                            "framing_id": framing,
                            "target_id": target,
                            "scenario_id": scenario,
                            "city_id": city_row["city_id"],
                            **city_row,
                            "sse_share": city_row["sse"] / candidate_metric["city_level_sse"],
                            "rank_by_sse": rank,
                        }
                    )
                for origin in ORIGINS:
                    origin_frame = candidate[candidate["origin_year"] == origin]
                    error = origin_frame["residual_prediction_minus_actual"].to_numpy(dtype=float)
                    mse = float(np.mean(error**2))
                    b3_mse = baseline_origin[origin]
                    per_origin.append(
                        {
                            "model_id": MODEL_ID,
                            "framing_id": framing,
                            "target_id": target,
                            "scenario_id": scenario,
                            "origin_year": origin,
                            "mse": mse,
                            "rmse": math.sqrt(mse),
                            "mae": float(np.mean(np.abs(error))),
                            "bias": float(np.mean(error)),
                            "city_count": len(origin_frame),
                            "b3_mse": b3_mse,
                            "delta_vs_b3": mse - b3_mse,
                            "relative_improvement_pct_vs_b3": (b3_mse - mse) / b3_mse * 100.0,
                            "win_vs_b3": "true" if mse < b3_mse - 1e-12 else "false",
                        }
                    )
    for row in aggregate:
        row.pop("_per_origin_mse", None)
    return aggregate, per_origin, city_error


def classification_from_metrics(
    aggregate: list[dict[str, Any]],
    per_origin: list[dict[str, Any]],
    city_error: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row for row in aggregate if row["model_id"] == MODEL_ID and row["target_id"] == "resident"
    ]
    city_index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    baseline_city_index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in city_error:
        if row["model_id"] == MODEL_ID and row["target_id"] == "resident":
            city_index[(row["framing_id"], row["scenario_id"])][row["city_id"]] = row
        if row["model_id"] == "B3_BASELINE" and row["target_id"] == "resident":
            baseline_city_index[row["scenario_id"]][row["city_id"]] = row
    signal_checks: list[dict[str, Any]] = []
    local_evidence = False
    for row in candidates:
        framing = row["framing_id"]
        scenario = row["scenario_id"]
        origin_rows = [
            item
            for item in per_origin
            if item["model_id"] == MODEL_ID
            and item["target_id"] == "resident"
            and item["framing_id"] == framing
            and item["scenario_id"] == scenario
        ]
        latest = next(item for item in origin_rows if item["origin_year"] == 2021)
        city_wins = sum(
            city_index[(framing, scenario)][city]["sse"]
            < baseline_city_index[scenario][city]["sse"] - 1e-12
            for city in CITIES
        )
        relative = float(row["relative_improvement_pct_vs_b3"] or 0.0)
        wins = int(row["origin_win_count_vs_b3"] or 0)
        latest_relative = float(latest["relative_improvement_pct_vs_b3"] or 0.0)
        stable_support = wins >= 2 and city_wins >= 2
        local_evidence = local_evidence or relative > 0.0 or wins >= 1 or latest_relative > 0.0
        signal_checks.append(
            {
                "framing_id": framing,
                "scenario_id": scenario,
                "pooled_relative_improvement_pct": relative,
                "origin_win_count": wins,
                "city_win_count": city_wins,
                "latest_2021_relative_improvement_pct": latest_relative,
                "stable_support_not_one_origin_or_city": stable_support,
            }
        )
    stable_signal = False
    for framing in FRAMINGS:
        r1 = next(row for row in candidates if row["framing_id"] == framing and row["scenario_id"] == SCENARIOS[0])
        r2 = next(row for row in candidates if row["framing_id"] == framing and row["scenario_id"] == SCENARIOS[1])
        p1 = float(r1["relative_improvement_pct_vs_b3"] or 0.0)
        p2 = float(r2["relative_improvement_pct_vs_b3"] or 0.0)
        s1 = next(item for item in signal_checks if item["framing_id"] == framing and item["scenario_id"] == SCENARIOS[0])
        s2 = next(item for item in signal_checks if item["framing_id"] == framing and item["scenario_id"] == SCENARIOS[1])
        if p1 > 0.0 and p2 > 0.0:
            stable_signal = stable_signal or (
                s1["stable_support_not_one_origin_or_city"]
                and s2["stable_support_not_one_origin_or_city"]
            )
        one_scenario = (p1 >= 2.0 and p2 >= -1.0) or (p2 >= 2.0 and p1 >= -1.0)
        stable_for_two_percent = (
            (p1 >= 2.0 and s1["stable_support_not_one_origin_or_city"])
            or (p2 >= 2.0 and s2["stable_support_not_one_origin_or_city"])
        )
        stable_signal = stable_signal or (one_scenario and stable_for_two_percent)
    if stable_signal:
        classification = "family_signal_found"
    elif local_evidence:
        classification = "family_competitive_but_unstable"
    else:
        classification = "family_unsupported"
    best_framing = min(
        FRAMINGS,
        key=lambda framing: np.mean(
            [
                row["pooled_city_year_mse"]
                for row in candidates
                if row["framing_id"] == framing
            ]
        ),
    )
    return {
        "classification": classification,
        "best_framing_by_resident_pooled_mse": best_framing,
        "stable_signal": stable_signal,
        "local_evidence": local_evidence,
        "family_signal_rule": {
            "both_r1_r2_below_b3": True,
            "one_scenario_improvement_pct_at_least": 2.0,
            "other_scenario_not_worse_than_pct": 1.0,
            "stable_support_requires_at_least_two_winning_origins_and_cities": True,
            "local_evidence_definition": "positive pooled improvement, at least one origin win, or positive latest-2021 improvement",
        },
        "signal_checks": signal_checks,
    }


def feature_importance_rows(
    context: dict[str, Any],
    fit_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_map = context["group_map"]
    feature_names = list(FEATURE_COLUMNS)
    scopes = {
        "all_selected_outer_fits": fit_records,
        "early_origins_2011_2014": [row for row in fit_records if row["outer_origin_year"] <= 2014],
        "middle_origins_2015_2018": [row for row in fit_records if 2015 <= row["outer_origin_year"] <= 2018],
        "late_origins_2019_2021": [row for row in fit_records if row["outer_origin_year"] >= 2019],
    }
    output: list[dict[str, Any]] = []
    top_groups: dict[str, list[dict[str, Any]]] = {}
    for scope, records in scopes.items():
        normalized_values: list[np.ndarray] = []
        raw_values: list[np.ndarray] = []
        for record in records:
            raw = np.asarray(
                record["estimator"].booster_.feature_importance(importance_type="gain"),
                dtype=float,
            )
            if len(raw) != len(feature_names):
                raise AssertionError("LightGBM importance width changed")
            normalized_values.append(raw / raw.sum() if raw.sum() else np.zeros_like(raw))
            raw_values.append(raw)
        normalized = np.vstack(normalized_values)
        raw = np.vstack(raw_values)
        groups: dict[str, dict[str, Any]] = {}
        for index, feature_name in enumerate(feature_names):
            group = "CITY_ID_ONE_HOT" if feature_name in CITY_FEATURES else group_map[feature_name]
            groups.setdefault(group, {"normalized": 0.0, "raw": 0.0, "features": []})
            groups[group]["normalized"] += float(normalized[:, index].mean())
            groups[group]["raw"] += float(raw[:, index].mean())
            groups[group]["features"].append(feature_name)
            output.append(
                {
                    "aggregation_scope": scope,
                    "importance_level": "feature",
                    "feature_name": feature_name,
                    "feature_group": group,
                    "feature_kind": "city_one_hot" if feature_name in CITY_FEATURES else "numeric_model_feature",
                    "normalized_gain_mean": float(normalized[:, index].mean()),
                    "mean_gain": float(raw[:, index].mean()),
                    "sum_gain": float(raw[:, index].sum()),
                    "selected_fit_count": len(records),
                    "nonzero_gain_fit_count": int(np.count_nonzero(raw[:, index] > 0)),
                    "feature_count_in_group": 1,
                }
            )
        group_rows: list[dict[str, Any]] = []
        for group, values in sorted(groups.items()):
            group_rows.append(
                {
                    "aggregation_scope": scope,
                    "importance_level": "feature_group",
                    "feature_name": "__GROUP_AGGREGATE__",
                    "feature_group": group,
                    "feature_kind": "city_one_hot_aggregate" if group == "CITY_ID_ONE_HOT" else "numeric_feature_group",
                    "normalized_gain_mean": values["normalized"],
                    "mean_gain": values["raw"],
                    "sum_gain": values["raw"] * len(records),
                    "selected_fit_count": len(records),
                    "nonzero_gain_fit_count": "",
                    "feature_count_in_group": len(values["features"]),
                }
            )
        output.extend(group_rows)
        top_groups[scope] = sorted(
            group_rows, key=lambda row: row["normalized_gain_mean"], reverse=True
        )
    summary = {
        scope: [
            {
                "feature_group": row["feature_group"],
                "normalized_gain_mean": row["normalized_gain_mean"],
                "feature_count": row["feature_count_in_group"],
            }
            for row in rows[:10]
        ]
        for scope, rows in top_groups.items()
    }
    return output, summary


def missingness_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    group_map = context["group_map"]
    all_features = list(context["model_features"]) + list(CITY_FEATURES)
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for scenario in SCENARIOS:
            for origin in ORIGINS:
                train_mask = outer_training_mask(context["replay"], target, scenario, origin)
                validation_mask = formal_validation_mask(context["formal"], target, scenario, origin)
                for split, values in (
                    ("outer_training", context["replay_x"][train_mask.to_numpy()]),
                    ("outer_validation", context["formal_x"][validation_mask.to_numpy()]),
                ):
                    missing = np.isnan(values).sum(axis=0)
                    for index, feature in enumerate(all_features):
                        is_city = feature in CITY_FEATURES
                        rows.append(
                            {
                                "target_id": target,
                                "scenario_id": scenario,
                                "outer_origin_year": origin,
                                "dataset_split": split,
                                "feature_name": feature,
                                "feature_group": "CITY_ID_ONE_HOT" if is_city else group_map[feature],
                                "feature_kind": "city_one_hot" if is_city else "numeric_model_feature",
                                "row_count": len(values),
                                "missing_count": int(missing[index]),
                                "missing_rate": float(missing[index] / len(values)) if len(values) else None,
                                "all_missing": "true" if len(values) and missing[index] == len(values) else "false",
                                "handling": "fixed_vocabulary_one_hot" if is_city else "lightgbm_native_nan",
                            }
                        )
    return rows


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    config_counts = Counter((row["framing_id"], row["selected_config_id"]) for row in rows)
    fallback = Counter(
        row["framing_id"]
        for row in rows
        if row["selection_method"] == "fallback_conservative_default"
    )
    return {
        "outer_fit_selection_row_count": len(rows),
        "selected_config_counts_by_framing": {
            f"{framing}|{config}": count for (framing, config), count in sorted(config_counts.items())
        },
        "fallback_fit_count_by_framing": dict(sorted(fallback.items())),
        "fallback_fit_count_total": int(sum(fallback.values())),
        "fallback_fit_count_per_framing_expected": 8,
    }


def city_concentration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        if row["model_id"] != MODEL_ID:
            continue
        key = f"{row['framing_id']}|{row['target_id']}|{row['scenario_id']}"
        result.setdefault(
            key,
            {
                "top1_city": row["city_id"],
                "top1_sse_share": row["sse_share"],
                "top3_sse_share": None,
            },
        )
        if int(row["rank_by_sse"]) == 3:
            result[key]["top3_sse_share"] = float(
                sum(
                    item["sse_share"]
                    for item in rows
                    if item["model_id"] == MODEL_ID
                    and item["framing_id"] == row["framing_id"]
                    and item["target_id"] == row["target_id"]
                    and item["scenario_id"] == row["scenario_id"]
                    and int(item["rank_by_sse"]) <= 3
                )
            )
    return result


def summary_object(
    context: dict[str, Any],
    aggregate: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    inner: list[dict[str, Any]],
    city_error: list[dict[str, Any]],
    top_groups: dict[str, Any],
    classification: dict[str, Any],
    inner_fit_count: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    metric_table = sorted(
        [row for row in aggregate if row["model_id"] == MODEL_ID],
        key=lambda row: (
            FRAMINGS.index(row["framing_id"]),
            TARGETS.index(row["target_id"]),
            SCENARIOS.index(row["scenario_id"]),
        ),
    )
    return {
        "plan_id": PLAN_ID,
        "stage": "S5_EVIDENCE_DRIVEN_ITERATION",
        "status": "screening_complete",
        "branch": BRANCH,
        "starting_commit": STARTING_COMMIT,
        "source_commit_for_oof": context["source_commit"],
        "contract_declared_wave_base_sha_at_execution_base": context["contract_declared_wave_base"],
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "model_id": MODEL_ID,
        "classification": classification["classification"],
        "best_framing_by_resident_pooled_mse": classification["best_framing_by_resident_pooled_mse"],
        "classification_evidence": classification,
        "data_contract": {
            "formal_validation_row_count": len(context["formal"]),
            "replay_row_count": len(context["replay"]),
            "numeric_model_feature_count": len(context["model_features"]),
            "lightgbm_input_feature_count": len(FEATURE_COLUMNS),
            "city_one_hot_feature_count": len(CITY_FEATURES),
            "formal_outer_origins": list(ORIGINS),
            "replay_pseudo_target_year_min": 2004,
            "replay_pseudo_target_year_max": 2021,
            "city_count": len(CITIES),
            "targets": list(TARGETS),
            "scenarios": list(SCENARIOS),
            "framings": list(FRAMINGS),
            "outer_training_rule": "pseudo_target_year <= outer_origin_year - 2",
            "inner_eligibility_rule": "inner_training_row_count >= 40 AND inner_validation_row_count >= 20 AND boundary_pass == PASS",
        },
        "geometry_audit": context["geometry_audit"],
        "b3_reference": context["b3_reference"],
        "selection_summary": selection_summary(selection),
        "inner_selection_evidence_row_count": len(inner),
        "inner_model_fit_count": inner_fit_count,
        "outer_model_fit_count": len(selection),
        "missing_value_handling": {
            "numeric_features": "raw NaN retained; LightGBM native missing-value handling",
            "city_id": "fixed vocabulary city1..city40 deterministic one-hot",
            "imputer_used": False,
            "scaler_used": False,
            "global_imputation_before_outer_split": False,
        },
        "fixed_randomness": {
            "random_seed": RANDOM_SEED,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
        },
        "pre_registered_configs": [dict(config) for config in CONFIGS],
        "top_feature_groups": top_groups,
        "city_concentration": city_concentration_summary(city_error),
        "metric_table": metric_table,
        "runtime": {
            "seconds": runtime_seconds,
            "library_versions": context["environment"],
            "library_limitations": [],
        },
        "screening_compliance": {
            "outer_labels_used_for_selection": False,
            "tuning_beyond_preregistered_configs": False,
            "feature_changes": False,
            "city_routing_or_gating": False,
            "ensemble_or_blend": False,
            "prediction_2023": False,
            "submission_generated": False,
            "s5_complete": False,
            "s6_entered": False,
            "checkpoint_modified": False,
        },
        "required_output_files": [
            "HANDOFF.md",
            "oof_predictions.csv",
            "aggregate_metrics.csv",
            "per_origin_metrics.csv",
            "configuration_selection.csv",
            "inner_selection_evidence.csv",
            "city_model_error.csv",
            "missingness_diagnostics.csv",
            "feature_importance.csv",
            "model_summary.json",
        ],
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def handoff_text(summary: dict[str, Any]) -> str:
    lines = [
        f"# {PLAN_ID} Handoff - LightGBM full multi-source screening",
        "",
        "## Result",
        "",
        "- Status: Complete",
        f"- Branch: {summary['branch']}",
        f"- Exact execution base: {summary['starting_commit']}",
        f"- Runner source commit for OOF: {summary['source_commit_for_oof']}",
        f"- Family classification: {summary['classification']}",
        f"- Best resident framing by pooled MSE: {summary['best_framing_by_resident_pooled_mse']}",
        "",
        "## Frozen data and nested selection",
        "",
        f"- Formal matrix: {summary['data_contract']['formal_validation_row_count']} rows; replay matrix: {summary['data_contract']['replay_row_count']} rows; numeric features: {summary['data_contract']['numeric_model_feature_count']}.",
        f"- Outer origins: {summary['data_contract']['formal_outer_origins']}; cities: {summary['data_contract']['city_count']}.",
        f"- Eligible inner folds: {summary['geometry_audit']['eligible_inner_fold_count']} / {summary['geometry_audit']['inner_geometry_rows']}; fallback outer fits: {summary['geometry_audit']['fallback_outer_fit_count']} per framing.",
        "- Selection used only eligible inner replay folds; outer-origin labels were not used for configuration selection.",
        "- Numeric NaN was kept raw for LightGBM native missing handling; city identity used fixed-vocabulary one-hot encoding.",
        "",
        "## Pooled metrics versus B3",
        "",
        "framing | target | scenario | pooled_mse | b3_mse | improvement_pct | origin_wins | latest_2021_mse",
    ]
    for row in summary["metric_table"]:
        lines.append(
            f"{row['framing_id']} | {row['target_id']} | {row['scenario_id']} | "
            f"{fmt(row['pooled_city_year_mse'])} | {fmt(row['b3_pooled_city_year_mse'])} | "
            f"{fmt(row['relative_improvement_pct_vs_b3'], 3)} | "
            f"{row['origin_win_count_vs_b3']} | {fmt(row['latest_2021_mse'])}"
        )
    lines.extend(
        [
            "",
            "## Configuration and diagnostics",
            "",
            f"- Selected configurations: {stable_json(summary['selection_summary']['selected_config_counts_by_framing'])}.",
            f"- Fallback fits: {stable_json(summary['selection_summary']['fallback_fit_count_by_framing'])}.",
            f"- Top feature groups by normalized LightGBM gain: {stable_json(summary['top_feature_groups']['all_selected_outer_fits'][:5])}.",
            "- City error concentration is recorded in city_model_error.csv; origin-band feature importance is recorded in feature_importance.csv.",
            "",
            "## Boundary and governance",
            "",
            "- No tuning beyond the three pre-registered configurations.",
            "- No Optuna, Bayesian optimization, random search, large grid search, routing, gating, ensemble or blend.",
            "- No feature changes; no Plan04, Plan04B or Plan05 artifact changes; no requirements change.",
            "- No 2023 prediction, no submission, S5 not declared complete, S6 not entered.",
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
            "",
            "## Next boundary",
            "",
            "- Main/Lab should compare this family evidence with the other Plan06 workers under the frozen Wave contract. This worker does not select a global winner or start tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global FEATURE_COLUMNS
    started = time.perf_counter()
    context = load_context()
    FEATURE_COLUMNS = tuple(context["model_features"]) + CITY_FEATURES
    oof_rows, selection_rows, inner_rows, fit_records, inner_fit_count = train_outer_models(context)
    oof = pd.DataFrame(oof_rows)
    aggregate, per_origin, city_error = metric_bundle(oof, context["formal"])
    classification = classification_from_metrics(aggregate, per_origin, city_error)
    importance, top_groups = feature_importance_rows(context, fit_records)
    missingness = missingness_rows(context)
    runtime = time.perf_counter() - started
    summary = summary_object(
        context,
        aggregate,
        selection_rows,
        inner_rows,
        city_error,
        top_groups,
        classification,
        inner_fit_count,
        runtime,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "oof_predictions.csv", oof_rows, OOF_FIELDS)
    aggregate_fields = (
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "pooled_city_year_mse",
        "macro_origin_mse",
        "origin_win_count_vs_b3",
        "latest_2021_mse",
        "rmse",
        "mae",
        "bias",
        "city_level_sse",
        "top1_city_sse_concentration",
        "top3_city_sse_concentration",
        "top1_city",
        "top3_cities",
        "b3_pooled_city_year_mse",
        "pooled_delta_vs_b3",
        "relative_improvement_pct_vs_b3",
        "row_count",
        "is_reference",
    )
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv", aggregate, aggregate_fields)
    write_csv(
        OUTPUT_DIR / "per_origin_metrics.csv",
        per_origin,
        (
            "model_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "mse",
            "rmse",
            "mae",
            "bias",
            "city_count",
            "b3_mse",
            "delta_vs_b3",
            "relative_improvement_pct_vs_b3",
            "win_vs_b3",
        ),
    )
    write_csv(OUTPUT_DIR / "configuration_selection.csv", selection_rows, SELECTION_FIELDS)
    write_csv(OUTPUT_DIR / "inner_selection_evidence.csv", inner_rows, INNER_FIELDS)
    write_csv(
        OUTPUT_DIR / "city_model_error.csv",
        city_error,
        (
            "model_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "city_id",
            "row_count",
            "sse",
            "mse",
            "mae",
            "bias",
            "sse_share",
            "rank_by_sse",
        ),
    )
    write_csv(
        OUTPUT_DIR / "missingness_diagnostics.csv",
        missingness,
        (
            "target_id",
            "scenario_id",
            "outer_origin_year",
            "dataset_split",
            "feature_name",
            "feature_group",
            "feature_kind",
            "row_count",
            "missing_count",
            "missing_rate",
            "all_missing",
            "handling",
        ),
    )
    write_csv(
        OUTPUT_DIR / "feature_importance.csv",
        importance,
        (
            "aggregation_scope",
            "importance_level",
            "feature_name",
            "feature_group",
            "feature_kind",
            "normalized_gain_mean",
            "mean_gain",
            "sum_gain",
            "selected_fit_count",
            "nonzero_gain_fit_count",
            "feature_count_in_group",
        ),
    )
    write_json(OUTPUT_DIR / "model_summary.json", summary)
    (OUTPUT_DIR / "HANDOFF.md").write_text(handoff_text(summary), encoding="utf-8")

    print("S5_PLAN06A_LIGHTGBM_SCREENING_PASS")
    print(f"starting_commit={STARTING_COMMIT}")
    print(f"source_commit_for_oof={context['source_commit']}")
    print(f"oof_rows={len(oof_rows)}")
    print(f"outer_fit_count={len(selection_rows)}")
    print(f"inner_model_fit_count={inner_fit_count}")
    print(f"eligible_inner_fold_count={context['geometry_audit']['eligible_inner_fold_count']}")
    print(f"fallback_outer_fit_count_per_framing={context['geometry_audit']['fallback_outer_fit_count']}")
    print(f"classification={classification['classification']}")
    print(f"best_framing={classification['best_framing_by_resident_pooled_mse']}")
    print("outer_labels_used_for_selection=False")
    print("tuning_beyond_preregistered_configs=False")
    print("prediction_2023=False")
    print("submission_generated=False")
    print("s5_complete=False")
    print("s6_entered=False")


if __name__ == "__main__":
    main()
