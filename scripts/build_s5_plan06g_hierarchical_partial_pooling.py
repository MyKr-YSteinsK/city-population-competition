"""Run the preregistered S5-Plan06G hierarchical partial-pooling screen.

The worker consumes the frozen Plan04/Plan04B matrices and geometry.  It
keeps the 144 Plan04 numeric features unchanged, fits preprocessing inside
each legal training fold, and evaluates a small, fixed MixedLM configuration
set.  The outputs are a screening handoff, not a forecast or submission.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan06G"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
BASE_SHA = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
BRANCH = "s5/plan06g-hierarchical-partial-pooling"
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
SLOPE_FEATURE = "pop_recent3_ols_slope"
FAMILY = "HIERARCHICAL_PARTIAL_POOLING"
MODEL_ID = "HIERARCHICAL_SELECTED"

CONFIGS = [
    {
        "config_id": "HIER_C1_RI_PCA8",
        "config_order": 1,
        "n_components": 8,
        "random_effects": "random_intercept",
        "reml": False,
    },
    {
        "config_id": "HIER_C2_RI_PCA16",
        "config_order": 2,
        "n_components": 16,
        "random_effects": "random_intercept",
        "reml": False,
    },
    {
        "config_id": "HIER_C3_RI_RS_PCA12",
        "config_order": 3,
        "n_components": 12,
        "random_effects": "random_intercept_and_slope",
        "reml": False,
    },
]

MixedLM: Any = None


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def py_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): py_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py_value(v) for v in value]
    return value


def short_error(error: Any) -> str:
    return str(error).replace("\r", " ").replace("\n", " ")[:500]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(py_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def verify_worker_environment() -> dict[str, Any]:
    """Verify the plan-authorized dependency and record the actual runtime."""

    global MixedLM
    import importlib.metadata as metadata

    try:
        import statsmodels
        from statsmodels.regression.mixed_linear_model import MixedLM as MixedLMClass

        MixedLM = MixedLMClass
        import_status = "PASS"
        import_error = ""
    except Exception as error:  # pragma: no cover - exercised only when blocked
        import_status = "FAIL"
        import_error = short_error(error)
        MixedLM = None

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    pip_check_output = (pip_check.stdout + pip_check.stderr).strip()
    pip_check_status = "PASS" if pip_check.returncode == 0 else "FAIL"
    try:
        statsmodels_version = metadata.version("statsmodels")
    except metadata.PackageNotFoundError:
        statsmodels_version = None
    package_names = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "patsy", "formulaic"]
    package_versions: dict[str, str | None] = {}
    for package in package_names:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None

    status = (
        "PASS"
        if import_status == "PASS"
        and pip_check_status == "PASS"
        and statsmodels_version == "0.15.0"
        else "FAIL"
    )
    environment = {
        "plan_id": PLAN_ID,
        "python": sys.version.split()[0],
        "requested_dependency": "statsmodels==0.15.0",
        "install_command": f"{Path(sys.executable).name} -m pip install statsmodels==0.15.0",
        "installation_status": status,
        "statsmodels_version": statsmodels_version,
        "import_smoke_tests": {
            "import statsmodels": import_status,
            "from statsmodels.regression.mixed_linear_model import MixedLM": import_status,
        },
        "import_error": import_error,
        "pip_check_status": pip_check_status,
        "pip_check_output": pip_check_output,
        "package_versions": package_versions,
        "shared_requirements_modified": False,
    }
    write_json(OUTPUT_DIR / "worker_environment.json", environment)
    if status != "PASS":
        raise RuntimeError("worker_dependency_blocked: statsmodels environment verification failed")
    return environment


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
    assert wave["data_contract"]["formal_validation_row_count"] == 1760
    assert wave["data_contract"]["replay_row_count"] == 2320
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
    replay_required = {
        "target_id",
        "scenario_id",
        "pseudo_target_year",
        "city_id",
        "actual_target_value",
        "b3_prediction",
        "b3_residual_actual_minus_prediction",
    }
    formal_required = {
        "target_id",
        "scenario_id",
        "origin_year",
        "city_id",
        "actual_target_value",
        "b3_prediction",
        "b3_residual_actual_minus_prediction",
    }
    assert replay_required.issubset(replay.columns)
    assert formal_required.issubset(formal.columns)
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
        assert frame["actual_target_value"].notna().all()
        assert frame["b3_prediction"].notna().all()
        assert frame["b3_residual_actual_minus_prediction"].notna().all()
    return replay, formal, outer, inner


def check_geometry(
    replay: pd.DataFrame,
    formal: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
) -> None:
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
        assert len(inner_train[inner_train.pseudo_target_year > int(row.inner_validation_year) - REPLAY_MAX_GAP]) == 0


def city_order(replay: pd.DataFrame, formal: pd.DataFrame) -> list[str]:
    cities = sorted(set(replay["city_id"]) | set(formal["city_id"]))
    assert len(cities) == REQUIRED_CITY_COUNT
    return cities


def label_values(frame: pd.DataFrame, framing: str) -> np.ndarray:
    column = "actual_target_value" if framing == "F1_DIRECT" else "b3_residual_actual_minus_prediction"
    values = frame[column].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    return values


def fit_fold_transform(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit all numeric preprocessing and the C3 random-slope scaling on train only."""

    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train[features])
    validation_imputed = imputer.transform(validation[features])
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    validation_scaled = scaler.transform(validation_imputed)
    max_rank = min(train_scaled.shape[0], train_scaled.shape[1])
    requested = int(config["n_components"])
    effective = max(1, min(requested, max_rank))
    pca = PCA(n_components=effective, svd_solver="full", random_state=SEED)
    train_fixed = pca.fit_transform(train_scaled)
    validation_fixed = pca.transform(validation_scaled)

    slope_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_slope = slope_imputer.fit_transform(train[[SLOPE_FEATURE]])
    validation_slope = slope_imputer.transform(validation[[SLOPE_FEATURE]])
    slope_scaler = StandardScaler()
    train_slope = slope_scaler.fit_transform(train_slope).reshape(-1)
    validation_slope = slope_scaler.transform(validation_slope).reshape(-1)
    assert np.isfinite(train_fixed).all()
    assert np.isfinite(validation_fixed).all()
    assert np.isfinite(train_slope).all()
    assert np.isfinite(validation_slope).all()
    return {
        "train_fixed": np.asarray(train_fixed, dtype=float),
        "validation_fixed": np.asarray(validation_fixed, dtype=float),
        "train_slope": np.asarray(train_slope, dtype=float),
        "validation_slope": np.asarray(validation_slope, dtype=float),
        "requested_components": requested,
        "effective_components": effective,
        "pca_input_feature_count": int(train_imputed.shape[1]),
        "pca_adjusted": bool(requested != effective),
    }


def make_design(prepared: dict[str, Any], config: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.asarray(prepared[f"{split}_fixed"], dtype=float)
    exog = np.column_stack([np.ones(len(fixed), dtype=float), fixed])
    if config["random_effects"] == "random_intercept":
        exog_re = np.ones((len(fixed), 1), dtype=float)
    else:
        slope = np.asarray(prepared[f"{split}_slope"], dtype=float)
        exog_re = np.column_stack([np.ones(len(fixed), dtype=float), slope])
    return exog, exog_re


def random_effect_vector(random_effects: Any, city_id: str) -> np.ndarray | None:
    if random_effects is None:
        return None
    for key, value in random_effects.items():
        if str(key) == str(city_id):
            array = np.asarray(value, dtype=float).reshape(-1)
            return array
    return None


def fit_mixed_model(
    train: pd.DataFrame,
    framing: str,
    prepared: dict[str, Any],
    config: dict[str, Any],
    fit_stats: Counter[str],
) -> dict[str, Any]:
    """Fit the registered statsmodels default path without adding rescue logic.

    ``MixedLM.fit(method=None)`` is the library's default fitting path.  In
    statsmodels 0.15 it tries the documented internal sequence bfgs, lbfgs,
    then cg.  This is treated as the single primary/default path.  Because
    that path already includes lbfgs, the plan's optional lbfgs fallback is
    not run a second time.
    """

    y = label_values(train, framing)
    exog, exog_re = make_design(prepared, config, "train")
    groups = train["city_id"].astype(str).to_numpy()
    model = MixedLM(endog=y, exog=exog, groups=groups, exog_re=exog_re)
    attempts = [("default", False)]
    errors: list[str] = []
    fit_stats["model_fit_units"] += 1
    for method, fallback in attempts:
        if fallback:
            fit_stats["fallback_attempts"] += 1
        else:
            fit_stats["primary_attempts"] += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=bool(config["reml"]), disp=False)
            if not bool(getattr(result, "converged", False)):
                fit_stats["fallback_nonconverged_units" if fallback else "primary_nonconverged_units"] += 1
                errors.append(f"{method}:not_converged")
                continue
            beta = np.asarray(result.fe_params, dtype=float).reshape(-1)
            covariance = np.asarray(result.cov_re, dtype=float)
            scale = float(result.scale)
            random_effects = result.random_effects
            random_dimension = exog_re.shape[1]
            if (
                len(beta) != exog.shape[1]
                or not np.isfinite(beta).all()
                or covariance.shape != (random_dimension, random_dimension)
                or not np.isfinite(covariance).all()
                or not np.isfinite(scale)
                or scale < 0
            ):
                raise ValueError("non_finite_or_wrong_shape_fit_result")
            for city_id in sorted(set(groups)):
                vector = random_effect_vector(random_effects, city_id)
                if vector is None or len(vector) != random_dimension or not np.isfinite(vector).all():
                    raise ValueError(f"invalid_random_effect:{city_id}")
            fit_stats["fit_success_units"] += 1
            if fallback:
                fit_stats["fallback_success_units"] += 1
            return {
                "success": True,
                "result": result,
                "random_effects": random_effects,
                "optimizer": method,
                "fallback_used": fallback,
                "error": "; ".join(errors),
            }
        except Exception as error:
            if fallback:
                fit_stats["fallback_exception_units"] += 1
            else:
                fit_stats["primary_exception_units"] += 1
            errors.append(f"{method}:{short_error(error)}")
    fit_stats["fit_failed_units"] += 1
    return {
        "success": False,
        "result": None,
        "random_effects": None,
        "optimizer": "",
        "fallback_used": False,
        "error": "; ".join(errors),
    }


def predict_from_fit(
    fit: dict[str, Any],
    prepared: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
    split: str,
    include_random_effects: bool = True,
) -> np.ndarray:
    assert fit["success"]
    exog, exog_re = make_design(prepared, config, split)
    beta = np.asarray(fit["result"].fe_params, dtype=float).reshape(-1)
    prediction = exog @ beta
    if include_random_effects:
        for index, city_id in enumerate(frame["city_id"].astype(str)):
            vector = random_effect_vector(fit["random_effects"], city_id)
            if vector is not None:
                prediction[index] += float(exog_re[index] @ vector)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("non_finite_prediction")
    return prediction


def config_by_id(config_id: str) -> dict[str, Any]:
    for config in CONFIGS:
        if config["config_id"] == config_id:
            return config
    raise KeyError(config_id)


def _metric_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target_id"],
        row["scenario_id"],
        row["framing_id"],
        int(row["origin_year"]),
        str(row["city_id"]),
    )


def baseline_frame(formal: pd.DataFrame) -> pd.DataFrame:
    reference = formal[
        [
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "actual_target_value",
            "b3_prediction",
        ]
    ].copy()
    reference["model_id"] = "B3_REFERENCE"
    reference["family"] = "BASELINE"
    reference["framing_id"] = "B3_REFERENCE"
    reference["prediction_value"] = reference["b3_prediction"]
    reference["residual_prediction_minus_actual"] = (
        reference["prediction_value"] - reference["actual_target_value"]
    )
    reference["squared_error"] = reference["residual_prediction_minus_actual"] ** 2
    return reference


def calculate_metrics(
    all_oof: pd.DataFrame,
    formal: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference = baseline_frame(formal)
    candidate_columns = [
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
    if len(all_oof):
        candidate = all_oof[candidate_columns].copy()
    else:
        candidate = pd.DataFrame(columns=candidate_columns)
    combined = pd.concat([candidate, reference[candidate_columns]], ignore_index=True)
    group_columns = ["model_id", "family", "framing_id", "target_id", "scenario_id"]
    origin_group_columns = group_columns + ["origin_year"]
    city_group_columns = group_columns + ["city_id"]
    b3_origin_mse: dict[tuple[str, str, int], float] = {}
    for keys, frame in reference.groupby(["target_id", "scenario_id", "origin_year"], sort=True):
        b3_origin_mse[(str(keys[0]), str(keys[1]), int(keys[2]))] = float(frame["squared_error"].mean())

    origin_rows: list[dict[str, Any]] = []
    for keys, frame in combined.groupby(origin_group_columns, sort=True):
        key = dict(zip(origin_group_columns, keys))
        mse = float(frame["squared_error"].mean())
        row: dict[str, Any] = {
            **key,
            "row_count": len(frame),
            "origin_mse": mse,
            "origin_rmse": float(np.sqrt(mse)),
            "origin_mae": float(frame["residual_prediction_minus_actual"].abs().mean()),
            "origin_bias": float(frame["residual_prediction_minus_actual"].mean()),
            "b3_origin_mse": None,
            "origin_win_vs_b3": None,
        }
        if key["model_id"] != "B3_REFERENCE":
            reference_mse = b3_origin_mse[(str(key["target_id"]), str(key["scenario_id"]), int(key["origin_year"]))]
            row["b3_origin_mse"] = reference_mse
            row["origin_win_vs_b3"] = bool(mse < reference_mse)
        origin_rows.append(row)

    city_rows: list[dict[str, Any]] = []
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

    aggregate_rows: list[dict[str, Any]] = []
    for keys, frame in combined.groupby(group_columns, sort=True):
        key = dict(zip(group_columns, keys))
        mse = float(frame["squared_error"].mean())
        origin_frame = pd.DataFrame(
            [row for row in origin_rows if all(row[column] == key[column] for column in group_columns)]
        )
        city_frame = pd.DataFrame(
            [row for row in city_rows if all(row[column] == key[column] for column in group_columns)]
        )
        b3_match = combined[
            (combined.model_id == "B3_REFERENCE")
            & (combined.target_id == key["target_id"])
            & (combined.scenario_id == key["scenario_id"])
        ]
        b3_mse = float(b3_match["squared_error"].mean())
        city_sse = np.sort(city_frame["city_sse"].to_numpy(dtype=float))[::-1]
        total_city_sse = float(city_sse.sum())
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
                "origin_win_count_vs_b3": (
                    int(origin_frame["origin_win_vs_b3"].fillna(False).sum())
                    if key["model_id"] != "B3_REFERENCE"
                    else None
                ),
                "origin_count": len(origin_frame),
                "city_sse_total": total_city_sse,
                "top1_city_sse_share": float(city_sse[:1].sum() / total_city_sse) if total_city_sse else None,
                "top3_city_sse_share": float(city_sse[:3].sum() / total_city_sse) if total_city_sse else None,
                "b3_pooled_city_year_mse": b3_mse,
                "improvement_vs_b3_pct": (
                    float((b3_mse - mse) / b3_mse * 100.0)
                    if key["model_id"] != "B3_REFERENCE" and b3_mse
                    else None
                ),
            }
        )
    return aggregate_rows, origin_rows, city_rows, {
        "combined_rows": len(combined),
        "reference_rows": len(reference),
        "candidate_rows": len(candidate),
    }


def family_classification(
    aggregate_rows: list[dict[str, Any]],
    framing: str,
    candidate_row_count: int,
) -> dict[str, Any]:
    rows = [
        row
        for row in aggregate_rows
        if row["family"] == FAMILY
        and row["framing_id"] == framing
        and row["target_id"] == "resident"
    ]
    if len(rows) != 2 or any(int(row["row_count"]) != 440 for row in rows) or candidate_row_count != 1760:
        return {
            "family": FAMILY,
            "framing_id": framing,
            "classification": "coverage_incomplete",
            "scenario_improvements_pct": {},
            "valid_signal_scenarios": [],
            "competitive_scenarios": [],
            "primary_target": "resident",
            "rule": "full 2 x 2 x 11 x 40 OOF coverage is required before performance classification",
        }
    by_scenario = {row["scenario_id"]: row for row in rows}
    scenario_improvements: dict[str, float] = {}
    valid_signal_scenarios: list[str] = []
    competitive_scenarios: list[str] = []
    for scenario in SCENARIOS:
        row = by_scenario[scenario]
        improvement = float(row["improvement_vs_b3_pct"])
        scenario_improvements[scenario] = improvement
        if (
            improvement >= 2.0
            and int(row["origin_win_count_vs_b3"]) >= 2
            and (row["top1_city_sse_share"] is None or row["top1_city_sse_share"] < 0.75)
        ):
            valid_signal_scenarios.append(scenario)
        if improvement >= -1.0:
            competitive_scenarios.append(scenario)
    both_better = all(scenario_improvements[scenario] > 0 for scenario in SCENARIOS)
    stable_rule = both_better or len(valid_signal_scenarios) >= 1
    if stable_rule and all(
        int(by_scenario[scenario]["origin_win_count_vs_b3"]) >= 2
        and (
            by_scenario[scenario]["top1_city_sse_share"] is None
            or by_scenario[scenario]["top1_city_sse_share"] < 0.75
        )
        for scenario in SCENARIOS
        if scenario_improvements[scenario] > 0
    ):
        classification = "family_signal_found"
    elif competitive_scenarios:
        classification = "family_competitive_but_unstable"
    else:
        classification = "family_unsupported"
    return {
        "family": FAMILY,
        "framing_id": framing,
        "classification": classification,
        "scenario_improvements_pct": scenario_improvements,
        "valid_signal_scenarios": valid_signal_scenarios,
        "competitive_scenarios": competitive_scenarios,
        "primary_target": "resident",
        "rule": "both R1/R2 below B3 or >=2% in one with other no worse than -1%; gain requires >=2 origin wins and top1 city SSE share <75%",
    }


def safe_variance(values: list[float] | np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return None
    value = float(np.var(array, ddof=1))
    return value if math.isfinite(value) else None


def random_effect_diagnostics(
    fit: dict[str, Any],
    train: pd.DataFrame,
    cities: list[str],
    config: dict[str, Any],
    group_key: dict[str, Any],
) -> list[dict[str, Any]]:
    counts = train.groupby("city_id").size().to_dict()
    seen = set(str(city) for city in counts)
    rows: list[dict[str, Any]] = []
    for city in cities:
        city_seen = city in seen
        vector = random_effect_vector(fit.get("random_effects"), city) if fit["success"] and city_seen else None
        if fit["success"] and not city_seen:
            intercept = 0.0
            slope = 0.0 if config["random_effects"] == "random_intercept_and_slope" else None
        elif vector is not None:
            intercept = float(vector[0])
            slope = float(vector[1]) if len(vector) > 1 else None
        else:
            intercept = None
            slope = None
        rows.append(
            {
                **group_key,
                "config_id": config["config_id"],
                "city_id": city,
                "random_intercept": intercept,
                "random_slope": slope,
                "city_seen_in_training": city_seen,
                "city_training_row_count": int(counts.get(city, 0)),
                "fit_status": "converged" if fit["success"] else "fit_failed",
                "optimizer": fit.get("optimizer", ""),
                "fallback_used": fit.get("fallback_used", False),
            }
        )
    return rows


def variance_component_row(
    fit: dict[str, Any],
    train: pd.DataFrame,
    config: dict[str, Any],
    group_key: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **group_key,
        "config_id": config["config_id"],
        "random_intercept_variance": None,
        "random_slope_variance": None,
        "random_intercept_slope_covariance": None,
        "residual_variance": None,
        "random_effect_group_count": int(train["city_id"].nunique()),
        "convergence_status": "converged" if fit["success"] else "fit_failed",
        "optimizer": fit.get("optimizer", ""),
        "fallback_used": fit.get("fallback_used", False),
        "fit_error": fit.get("error", ""),
    }
    if fit["success"]:
        covariance = np.asarray(fit["result"].cov_re, dtype=float)
        row["random_intercept_variance"] = float(covariance[0, 0])
        if covariance.shape[0] > 1:
            row["random_slope_variance"] = float(covariance[1, 1])
            row["random_intercept_slope_covariance"] = float(covariance[0, 1])
        row["residual_variance"] = float(fit["result"].scale)
    return row


def pooling_record(
    fit: dict[str, Any],
    prepared: dict[str, Any],
    train: pd.DataFrame,
    framing: str,
    config: dict[str, Any],
    group_key: dict[str, Any],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        **group_key,
        "summary_scope": "outer_fit",
        "config_id": config["config_id"],
        "fit_status": "converged" if fit["success"] else "fit_failed",
        "raw_between_city_mean_variance": None,
        "fixed_effect_residual_between_city_mean_variance": None,
        "random_intercept_variance": None,
        "random_intercept_variance_fraction_of_raw": None,
        "mean_abs_random_intercept": None,
        "mean_abs_raw_city_mean_deviation": None,
        "empirical_shrinkage_proxy": None,
        "low_history_mean_abs_random_intercept": None,
        "high_history_mean_abs_random_intercept": None,
        "low_history_mean_abs_raw_deviation": None,
        "high_history_mean_abs_raw_deviation": None,
        "random_slope_variance": None,
        "random_slope_sd": None,
        "random_slope_meaningfully_nonzero": None,
        "city_group_count": int(train["city_id"].nunique()),
        "low_history_definition": "city training rows <= median among seen training cities",
        "interpretation": "",
    }
    if not fit["success"]:
        return base

    y = label_values(train, framing)
    fixed_exog, _ = make_design(prepared, config, "train")
    fixed_prediction = fixed_exog @ np.asarray(fit["result"].fe_params, dtype=float)
    records = pd.DataFrame(
        {
            "city_id": train["city_id"].astype(str).to_numpy(),
            "y": y,
            "fixed_residual": y - fixed_prediction,
        }
    )
    raw_means = records.groupby("city_id")["y"].mean()
    fixed_residual_means = records.groupby("city_id")["fixed_residual"].mean()
    counts = records.groupby("city_id").size()
    random_intercepts = pd.Series(
        {
            city: float(random_effect_vector(fit["random_effects"], city)[0])
            for city in raw_means.index
        }
    )
    raw_deviation = raw_means - float(records["y"].mean())
    covariance = np.asarray(fit["result"].cov_re, dtype=float)
    random_intercept_variance = float(covariance[0, 0])
    base["raw_between_city_mean_variance"] = safe_variance(raw_means.to_numpy())
    base["fixed_effect_residual_between_city_mean_variance"] = safe_variance(
        fixed_residual_means.to_numpy()
    )
    base["random_intercept_variance"] = random_intercept_variance
    if base["raw_between_city_mean_variance"]:
        base["random_intercept_variance_fraction_of_raw"] = float(
            random_intercept_variance / base["raw_between_city_mean_variance"]
        )
    base["mean_abs_random_intercept"] = float(random_intercepts.abs().mean())
    base["mean_abs_raw_city_mean_deviation"] = float(raw_deviation.abs().mean())
    usable = raw_deviation.abs() > 1e-12
    if usable.any():
        shrinkage_ratio = (random_intercepts.abs()[usable] / raw_deviation.abs()[usable]).clip(upper=1.0)
        base["empirical_shrinkage_proxy"] = float(1.0 - shrinkage_ratio.mean())
    median_count = float(counts.median())
    low = counts <= median_count
    high = counts > median_count
    if low.any():
        base["low_history_mean_abs_random_intercept"] = float(random_intercepts[low].abs().mean())
        base["low_history_mean_abs_raw_deviation"] = float(raw_deviation[low].abs().mean())
    if high.any():
        base["high_history_mean_abs_random_intercept"] = float(random_intercepts[high].abs().mean())
        base["high_history_mean_abs_raw_deviation"] = float(raw_deviation[high].abs().mean())
    if covariance.shape[0] > 1:
        slope_variance = float(covariance[1, 1])
        base["random_slope_variance"] = slope_variance
        base["random_slope_sd"] = float(np.sqrt(max(slope_variance, 0.0)))
        base["random_slope_meaningfully_nonzero"] = bool(slope_variance > 1e-8)
    return base


def aggregate_pooling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    success = frame[frame["fit_status"] == "converged"].copy()
    if success.empty:
        return []
    group_columns = ["target_id", "scenario_id", "framing_id"]
    numeric_columns = [
        "raw_between_city_mean_variance",
        "fixed_effect_residual_between_city_mean_variance",
        "random_intercept_variance",
        "random_intercept_variance_fraction_of_raw",
        "mean_abs_random_intercept",
        "mean_abs_raw_city_mean_deviation",
        "empirical_shrinkage_proxy",
        "low_history_mean_abs_random_intercept",
        "high_history_mean_abs_random_intercept",
        "low_history_mean_abs_raw_deviation",
        "high_history_mean_abs_raw_deviation",
        "random_slope_variance",
        "random_slope_sd",
    ]
    aggregate: list[dict[str, Any]] = []
    for keys, group in success.groupby(group_columns, sort=True):
        row: dict[str, Any] = {
            "target_id": keys[0],
            "scenario_id": keys[1],
            "framing_id": keys[2],
            "outer_origin_year": "ALL",
            "summary_scope": "target_scenario_framing",
            "config_id": ";".join(sorted(set(group["config_id"].astype(str)))),
            "fit_status": "converged",
            "city_group_count": None,
            "low_history_definition": "city training rows <= median among seen training cities",
            "interpretation": "",
            "outer_fit_count": int(len(group)),
            "random_slope_meaningfully_nonzero_fit_fraction": float(
                group["random_slope_meaningfully_nonzero"].fillna(False).astype(bool).mean()
            ),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[column] = float(values.mean()) if len(values) else None
        aggregate.append(row)
    return aggregate


def pooling_interpretations(aggregate_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in aggregate_rows:
        label = f"{row['target_id']}/{row['scenario_id']}/{row['framing_id']}"
        remaining = row.get("fixed_effect_residual_between_city_mean_variance")
        raw = row.get("raw_between_city_mean_variance")
        if raw is not None and remaining is not None:
            lines.append(
                f"{label}: fixed-effect residual city-mean variance averaged {remaining:.6g} "
                f"versus raw between-city variance {raw:.6g}."
            )
        low = row.get("low_history_mean_abs_random_intercept")
        high = row.get("high_history_mean_abs_random_intercept")
        if low is not None and high is not None:
            direction = "smaller" if low < high else "not smaller"
            lines.append(
                f"{label}: low-history mean absolute random intercept was {direction} than high-history "
                f"({low:.6g} vs {high:.6g}); this is a diagnostic, not routing evidence."
            )
        slope_fraction = row.get("random_slope_meaningfully_nonzero_fit_fraction")
        if slope_fraction is not None:
            lines.append(
                f"{label}: selected-fit C3/non-C3 slope-nonzero fraction was {slope_fraction:.3f}; "
                "stability must be judged with the variance and convergence evidence."
            )
    if not lines:
        lines.append("No converged selected outer fit was available for pooling interpretation.")
    return lines


def run_screening(
    replay: pd.DataFrame,
    formal: pd.DataFrame,
    inner: pd.DataFrame,
    features: list[str],
    cities: list[str],
    worker_commit: str,
) -> dict[str, Any]:
    oof_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    pooling_rows: list[dict[str, Any]] = []
    missingness_rows: list[dict[str, Any]] = []
    model_failures: list[dict[str, Any]] = []
    selected_config_counts: Counter[str] = Counter()
    selected_config_counts_by_framing: dict[str, Counter[str]] = {
        framing: Counter() for framing in FRAMINGS
    }
    fit_stats: Counter[str] = Counter()

    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
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
                for framing_id in FRAMINGS:
                    group_key = {
                        "family": FAMILY,
                        "model_id": MODEL_ID,
                        "framing_id": framing_id,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": origin_year,
                    }
                    config_summaries: dict[str, dict[str, Any]] = {}
                    transform_cache: dict[tuple[int, str], dict[str, Any]] = {}
                    for config in CONFIGS:
                        config_id = config["config_id"]
                        valid_mses: list[float] = []
                        valid_count = 0
                        failed_count = 0
                        if len(eligible) < MIN_ELIGIBLE_FOLDS:
                            evidence_rows.append(
                                {
                                    **group_key,
                                    "config_id": config_id,
                                    "config_order": config["config_order"],
                                    "inner_validation_year": None,
                                    "inner_training_row_count": None,
                                    "inner_validation_row_count": None,
                                    "pca_components_requested": config["n_components"],
                                    "pca_components_effective": None,
                                    "fit_status": "fallback_not_evaluated",
                                    "optimizer": "",
                                    "fallback_used": False,
                                    "converged": None,
                                    "validation_mse": None,
                                    "eligible_fold_count": len(eligible),
                                    "valid_prediction_count": 0,
                                    "fit_error": "eligible_inner_fold_count_below_3",
                                    "selection_method": "fallback_conservative_default",
                                }
                            )
                        else:
                            for fold in eligible.itertuples(index=False):
                                validation_year = int(fold.inner_validation_year)
                                fold_train = outer_train[
                                    outer_train.pseudo_target_year <= validation_year - REPLAY_MAX_GAP
                                ].copy()
                                fold_validation = outer_train[
                                    outer_train.pseudo_target_year == validation_year
                                ].copy()
                                assert len(fold_train) == int(fold.inner_training_row_count)
                                assert len(fold_validation) == int(fold.inner_validation_row_count)
                                cache_key = (validation_year, config_id)
                                if cache_key not in transform_cache:
                                    transform_cache[cache_key] = fit_fold_transform(
                                        fold_train, fold_validation, features, config
                                    )
                                prepared = transform_cache[cache_key]
                                fit = fit_mixed_model(
                                    fold_train, framing_id, prepared, config, fit_stats
                                )
                                mse: float | None = None
                                fit_status = "fit_failed"
                                error = fit.get("error", "")
                                if fit["success"]:
                                    try:
                                        prediction = predict_from_fit(
                                            fit, prepared, fold_validation, config, "validation"
                                        )
                                        mse = float(
                                            np.mean(
                                                (
                                                    prediction
                                                    - fold_validation["actual_target_value"].to_numpy(dtype=float)
                                                )
                                                ** 2
                                            )
                                        )
                                        if not math.isfinite(mse):
                                            raise ValueError("non_finite_validation_mse")
                                        valid_count += 1
                                        valid_mses.append(mse)
                                        fit_status = "valid"
                                    except Exception as prediction_error:
                                        fit_stats["prediction_failure_units"] += 1
                                        error = short_error(prediction_error)
                                        fit_status = "prediction_failed"
                                if fit_status != "valid":
                                    failed_count += 1
                                evidence_rows.append(
                                    {
                                        **group_key,
                                        "config_id": config_id,
                                        "config_order": config["config_order"],
                                        "inner_validation_year": validation_year,
                                        "inner_training_row_count": len(fold_train),
                                        "inner_validation_row_count": len(fold_validation),
                                        "pca_components_requested": prepared["requested_components"],
                                        "pca_components_effective": prepared["effective_components"],
                                        "fit_status": fit_status,
                                        "optimizer": fit.get("optimizer", ""),
                                        "fallback_used": fit.get("fallback_used", False),
                                        "converged": bool(fit["success"]) if fit_status == "valid" else False,
                                        "validation_mse": mse,
                                        "eligible_fold_count": len(eligible),
                                        "valid_prediction_count": valid_count,
                                        "fit_error": error,
                                        "selection_method": "nested_inner_mean_mse",
                                    }
                                )
                        mean_mse = float(np.mean(valid_mses)) if valid_count >= MIN_ELIGIBLE_FOLDS else None
                        config_summaries[config_id] = {
                            "mean_mse": mean_mse,
                            "valid_count": valid_count,
                            "failed_count": failed_count,
                        }

                    valid_config_indices = [
                        index
                        for index, config in enumerate(CONFIGS)
                        if config_summaries[config["config_id"]]["mean_mse"] is not None
                    ]
                    fallback_reason = ""
                    if len(eligible) < MIN_ELIGIBLE_FOLDS:
                        selected_index = 0
                        selection_method = "fallback_conservative_default"
                        fallback_reason = "eligible_inner_fold_count_below_3"
                    elif not valid_config_indices:
                        selected_index = 0
                        selection_method = "fallback_conservative_default"
                        fallback_reason = "no_config_with_at_least_3_valid_inner_folds"
                    else:
                        selected_index = min(
                            valid_config_indices,
                            key=lambda index: (
                                float(config_summaries[CONFIGS[index]["config_id"]]["mean_mse"]),
                                CONFIGS[index]["config_order"],
                            ),
                        )
                        selection_method = "nested_inner_mean_mse"
                    selected = CONFIGS[selected_index]
                    selected_id = selected["config_id"]
                    selected_config_counts[selected_id] += 1
                    selected_config_counts_by_framing[framing_id][selected_id] += 1

                    selection_row: dict[str, Any] = {
                        **group_key,
                        "selected_config_id": selected_id,
                        "selection_method": selection_method,
                        "fallback_reason": fallback_reason,
                        "eligible_inner_fold_count": len(eligible),
                        "selected_mean_inner_mse": config_summaries[selected_id]["mean_mse"],
                        "outer_fit_status": "pending",
                        "outer_fit_optimizer": "",
                        "outer_fit_fallback_used": False,
                        "outer_fit_error": "",
                        "outer_pca_components_requested": selected["n_components"],
                        "outer_pca_components_effective": None,
                        "unseen_city_count": None,
                    }
                    for index, config in enumerate(CONFIGS, start=1):
                        summary = config_summaries[config["config_id"]]
                        selection_row[f"config_{index}_id"] = config["config_id"]
                        selection_row[f"config_{index}_mean_inner_mse"] = summary["mean_mse"]
                        selection_row[f"config_{index}_valid_fold_count"] = summary["valid_count"]
                        selection_row[f"config_{index}_failed_fold_count"] = summary["failed_count"]

                    outer_prepared = fit_fold_transform(
                        outer_train, outer_validation, features, selected
                    )
                    selection_row["outer_pca_components_effective"] = outer_prepared["effective_components"]
                    outer_fit = fit_mixed_model(
                        outer_train, framing_id, outer_prepared, selected, fit_stats
                    )
                    unseen_city_count = len(set(cities) - set(outer_train["city_id"].astype(str)))
                    selection_row["unseen_city_count"] = unseen_city_count
                    selection_row["outer_fit_status"] = "converged" if outer_fit["success"] else "fit_failed"
                    selection_row["outer_fit_optimizer"] = outer_fit.get("optimizer", "")
                    selection_row["outer_fit_fallback_used"] = outer_fit.get("fallback_used", False)
                    selection_row["outer_fit_error"] = outer_fit.get("error", "")
                    selection_rows.append(selection_row)

                    missing_group_key = {
                        "family": FAMILY,
                        "model_id": MODEL_ID,
                        "framing_id": framing_id,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": origin_year,
                        "config_id": selected_id,
                        "pca_components_requested": selected["n_components"],
                        "pca_components_effective": outer_prepared["effective_components"],
                        "fit_status": "converged" if outer_fit["success"] else "fit_failed",
                    }
                    for split, frame in (("outer_training", outer_train), ("outer_validation", outer_validation)):
                        missing = frame[features].isna()
                        missingness_rows.append(
                            {
                                **missing_group_key,
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

                    random_rows.extend(
                        random_effect_diagnostics(
                            outer_fit, outer_train, cities, selected, {
                                "family": FAMILY,
                                "model_id": MODEL_ID,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin_year,
                            }
                        )
                    )
                    variance_rows.append(
                        variance_component_row(
                            outer_fit,
                            outer_train,
                            selected,
                            {
                                "family": FAMILY,
                                "model_id": MODEL_ID,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin_year,
                            },
                        )
                    )
                    pooling_rows.append(
                        pooling_record(
                            outer_fit,
                            outer_prepared,
                            outer_train,
                            framing_id,
                            selected,
                            {
                                "family": FAMILY,
                                "model_id": MODEL_ID,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "outer_origin_year": origin_year,
                            },
                        )
                    )

                    if not outer_fit["success"]:
                        fit_stats["outer_fit_failure_count"] += 1
                        model_failures.append(
                            {
                                **group_key,
                                "config_id": selected_id,
                                "failure_type": "outer_fit_failed",
                                "error": outer_fit.get("error", ""),
                            }
                        )
                        continue

                    try:
                        prediction = predict_from_fit(
                            outer_fit, outer_prepared, outer_validation, selected, "validation"
                        )
                    except Exception as error:
                        fit_stats["outer_prediction_failure_count"] += 1
                        model_failures.append(
                            {
                                **group_key,
                                "config_id": selected_id,
                                "failure_type": "outer_prediction_failed",
                                "error": short_error(error),
                            }
                        )
                        continue
                    assert len(prediction) == 40
                    for row, predicted in zip(outer_validation.to_dict("records"), prediction):
                        actual = float(row["actual_target_value"])
                        oof_rows.append(
                            {
                                "model_id": MODEL_ID,
                                "family": FAMILY,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "origin_year": origin_year,
                                "city_id": str(row["city_id"]),
                                "actual_target_value": actual,
                                "prediction_value": float(predicted),
                                "residual_prediction_minus_actual": float(predicted - actual),
                                "squared_error": float((predicted - actual) ** 2),
                                "training_max_label_year": int(outer_train["pseudo_target_year"].max()),
                                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                                "worker_commit": worker_commit,
                                "selected_config_id": selected_id,
                                "selection_method": selection_method,
                                "unseen_city_count": unseen_city_count,
                            }
                        )

    return {
        "oof_rows": oof_rows,
        "selection_rows": selection_rows,
        "evidence_rows": evidence_rows,
        "random_rows": random_rows,
        "variance_rows": variance_rows,
        "pooling_rows": pooling_rows,
        "missingness_rows": missingness_rows,
        "model_failures": model_failures,
        "selected_config_counts": dict(selected_config_counts),
        "selected_config_counts_by_framing": {
            framing: dict(counter) for framing, counter in selected_config_counts_by_framing.items()
        },
        "fit_stats": dict(fit_stats),
    }


def summarize_unseen(selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(selection_rows)
    if frame.empty:
        return {"total_unseen_city_counts": 0, "by_target_scenario": {}}
    grouped: dict[str, Any] = {}
    for keys, group in frame.groupby(["target_id", "scenario_id"], sort=True):
        values = pd.to_numeric(group["unseen_city_count"], errors="coerce").dropna()
        label = f"{keys[0]}|{keys[1]}"
        grouped[label] = {
            "outer_fit_count": int(len(values)),
            "total_unseen_city_counts": int(values.sum()),
            "mean_unseen_city_count": float(values.mean()),
            "max_unseen_city_count": int(values.max()) if len(values) else None,
        }
    return {
        "total_unseen_city_counts": int(pd.to_numeric(frame["unseen_city_count"], errors="coerce").sum()),
        "by_target_scenario": grouped,
    }


def variance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    successful = frame[frame["convergence_status"] == "converged"].copy()
    if successful.empty:
        return {"successful_outer_fit_count": 0}
    result: dict[str, Any] = {
        "successful_outer_fit_count": int(len(successful)),
        "random_intercept_variance_mean": float(successful["random_intercept_variance"].mean()),
        "random_intercept_variance_median": float(successful["random_intercept_variance"].median()),
        "residual_variance_mean": float(successful["residual_variance"].mean()),
        "random_slope_variance_nonnull_count": int(successful["random_slope_variance"].notna().sum()),
    }
    slope = pd.to_numeric(successful["random_slope_variance"], errors="coerce").dropna()
    if len(slope):
        result["random_slope_variance_mean"] = float(slope.mean())
        result["random_slope_variance_median"] = float(slope.median())
        result["random_slope_variance_meaningfully_nonzero_fraction"] = float((slope > 1e-8).mean())
    else:
        result["random_slope_variance_mean"] = None
        result["random_slope_variance_median"] = None
        result["random_slope_variance_meaningfully_nonzero_fraction"] = None
    return result


def metric_lookup(
    aggregate_rows: list[dict[str, Any]],
    framing: str,
    target: str,
    scenario: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in aggregate_rows
        if row["family"] == FAMILY
        and row["framing_id"] == framing
        and row["target_id"] == target
        and row["scenario_id"] == scenario
    ]
    if len(matches) != 1:
        raise KeyError((framing, target, scenario))
    return matches[0]


def handoff_text(
    summary: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    environment: dict[str, Any],
    worker_commit: str,
) -> str:
    lines = [
        f"# {PLAN_ID} Handoff — Hierarchical / Partial Pooling 城市异质性 screening",
        "",
        "## Result",
        "",
        f"- Plan ID：{PLAN_ID}",
        f"- Stage：{STAGE}",
        f"- Worker branch：{BRANCH}",
        f"- Exact base SHA：{BASE_SHA}",
        f"- Runner commit used for execution：{worker_commit}",
        f"- Official OOF coverage：{summary['candidate_oof_rows']} / {summary['required_oof_rows']} rows — {summary['oof_coverage_status']}",
        f"- Classification：{summary['classification']}",
        "",
        "## Dependency",
        "",
        f"- `statsmodels==0.15.0` install/import/pip-check：{environment['installation_status']} / {environment['import_smoke_tests']['import statsmodels']} / {environment['pip_check_status']}",
        f"- Python：{environment['python']}；actual statsmodels：{environment['statsmodels_version']}",
        "- `requirements.txt` 未修改；未替换为 PyMC 或其他 Bayesian stack。",
        "",
        "## Method and boundary",
        "",
        "- F1_DIRECT 使用 `actual_target_value`；F2_B3_RESIDUAL 使用 `b3_residual_actual_minus_prediction`，最终预测为 B3 + residual prediction。",
        "- 每个 inner/outer training fold 独立执行 median imputation + missing indicators + standardization + PCA；PCA 不进入 `city_id`，随机效应分组为 raw `city_id`。",
        "- C1/C2 是 city random intercept；C3 是 city random intercept + standardized `pop_recent3_ols_slope` random slope；REML=false。",
        "- 未见过的验证城市使用随机效应 0；未使用 held-out target 估计验证城市随机效应。",
        "",
        "## Coverage and convergence",
        "",
        f"- Outer fit groups：{summary['selection_group_count']} selected framing groups；model failures：{len(summary['model_failures'])}。",
        f"- Fit counters：{json.dumps(summary['convergence_failure_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- Selected config counts：{json.dumps(summary['selected_config_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- Unseen-city counts：{json.dumps(summary['unseen_city_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Pooled metrics",
        "",
        "| framing | target | scenario | pooled MSE | B3 MSE | improvement % | origin wins | latest-2021 MSE | top-1 city SSE share |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for framing in FRAMINGS:
        for target in TARGETS:
            for scenario in SCENARIOS:
                try:
                    row = metric_lookup(aggregate_rows, framing, target, scenario)
                except KeyError:
                    continue
                lines.append(
                    f"| {framing} | {target} | {scenario[:2]} | {row['pooled_city_year_mse']:.6f} | "
                    f"{row['b3_pooled_city_year_mse']:.6f} | {row['improvement_vs_b3_pct']:.3f} | "
                    f"{row['origin_win_count_vs_b3']} | {row['latest_2021_mse']:.6f} | "
                    f"{row['top1_city_sse_share']:.3f} |"
                )
    lines.extend(
        [
            "",
            "Resident/registered R1/R2 的 macro-origin MSE、RMSE、MAE、bias、per-origin、city-level SSE/MSE 与 top-3 集中度均在 CSV artifacts 中提供。",
            "",
            "## F1/F2 comparison",
            "",
        ]
    )
    for target in TARGETS:
        for scenario in SCENARIOS:
            f1 = metric_lookup(aggregate_rows, "F1_DIRECT", target, scenario)
            f2 = metric_lookup(aggregate_rows, "F2_B3_RESIDUAL", target, scenario)
            better = "F1_DIRECT" if f1["pooled_city_year_mse"] <= f2["pooled_city_year_mse"] else "F2_B3_RESIDUAL"
            lines.append(
                f"- {target}/{scenario[:2]}：{better} pooled MSE 更低 "
                f"(F1={f1['pooled_city_year_mse']:.6f}, F2={f2['pooled_city_year_mse']:.6f})。"
            )
    lines.extend(
        [
            "",
            "## Variance and pooling interpretation",
            "",
            f"- Variance summary：{json.dumps(summary['variance_component_summary'], ensure_ascii=False, sort_keys=True)}",
        ]
    )
    lines.extend(f"- {line}" for line in summary["pooling_strength_interpretation"])
    lines.extend(
        [
            "",
            "## Boundary and deviations",
            "",
            "- 未修改 Plan04 的 144 个特征、replay rows、validation geometry、Plan04/04B/05 artifacts 或 `CURRENT_LEARNING_CHECKPOINT.md`。",
            "- 未执行 city routing、post-hoc selector、ensemble/stacking/gating、2023 prediction/submission；未声明 S5 complete，未进入 S6。",
            "- pooling strength 仅为诊断，不转化为城市路由或后验配置选择。",
            "",
            "## Artifacts",
            "",
            "- `HANDOFF.md`",
            "- `worker_environment.json`",
            "- `oof_predictions.csv`",
            "- `aggregate_metrics.csv`",
            "- `per_origin_metrics.csv`",
            "- `configuration_selection.csv`",
            "- `inner_selection_evidence.csv`",
            "- `city_model_error.csv`",
            "- `missingness_diagnostics.csv`",
            "- `random_effects_diagnostics.csv`",
            "- `variance_components.csv`",
            "- `pooling_strength_summary.csv`",
            "- `model_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    assert_gate()
    try:
        environment = verify_worker_environment()
    except Exception as error:
        print(str(error), flush=True)
        raise
    context = load_contract()
    replay, formal, outer, inner = load_inputs(context)
    check_geometry(replay, formal, outer, inner)
    cities = city_order(replay, formal)
    worker_commit = git_text("rev-parse", "HEAD")
    result = run_screening(
        replay,
        formal,
        inner,
        context["features"],
        cities,
        worker_commit,
    )
    oof_fields = [
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
        "training_max_label_year",
        "feature_contract_version",
        "worker_commit",
        "selected_config_id",
        "selection_method",
        "unseen_city_count",
    ]
    all_oof = pd.DataFrame(result["oof_rows"], columns=oof_fields)
    if len(all_oof):
        all_oof = all_oof.sort_values(
            ["framing_id", "target_id", "scenario_id", "origin_year", "city_id"]
        ).reset_index(drop=True)
    expected_keys = {
        (framing, target, scenario, origin, city)
        for framing in FRAMINGS
        for target in TARGETS
        for scenario in SCENARIOS
        for origin in ORIGINS
        for city in cities
    }
    actual_keys = {
        (
            str(row.framing_id),
            str(row.target_id),
            str(row.scenario_id),
            int(row.origin_year),
            str(row.city_id),
        )
        for row in all_oof.itertuples(index=False)
    }
    coverage_status = "PASS" if len(all_oof) == len(expected_keys) and actual_keys == expected_keys else "FAIL"
    aggregate_rows, origin_rows, city_rows, metric_meta = calculate_metrics(all_oof, formal)
    classifications = [
        family_classification(aggregate_rows, framing, len(all_oof)) for framing in FRAMINGS
    ]
    labels = [item["classification"] for item in classifications]
    if coverage_status != "PASS":
        classification = "coverage_incomplete"
    elif "family_signal_found" in labels:
        classification = "family_signal_found"
    elif "family_competitive_but_unstable" in labels:
        classification = "family_competitive_but_unstable"
    else:
        classification = "family_unsupported"

    pooling_aggregate = aggregate_pooling_rows(result["pooling_rows"])
    pooling_strength_rows = result["pooling_rows"] + pooling_aggregate
    pooling_interpretation = pooling_interpretations(pooling_aggregate)
    variance_component_summary = variance_summary(result["variance_rows"])
    summary: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "runner_commit_at_execution": worker_commit,
        "final_evidence_commit_at_report_generation": worker_commit,
        "seed": SEED,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "formal_row_count": len(formal),
        "replay_row_count": len(replay),
        "model_feature_count": len(context["features"]),
        "candidate_oof_rows": len(all_oof),
        "required_oof_rows": len(expected_keys),
        "oof_coverage_status": coverage_status,
        "selection_group_count": len(result["selection_rows"]),
        "inner_geometry_row_count": len(inner),
        "eligible_inner_fold_count": int(
            (
                (inner["inner_training_row_count"] >= 40)
                & (inner["inner_validation_row_count"] >= 20)
                & (inner["boundary_pass"] == "PASS")
            ).sum()
        ),
        "selected_config_counts": result["selected_config_counts"],
        "selected_config_counts_by_framing": result["selected_config_counts_by_framing"],
        "convergence_failure_counts": result["fit_stats"],
        "model_failures": result["model_failures"],
        "classification": classification,
        "framings": classifications,
        "metric_meta": metric_meta,
        "unseen_city_counts": summarize_unseen(result["selection_rows"]),
        "variance_component_summary": variance_component_summary,
        "pooling_strength_interpretation": pooling_interpretation,
        "city_concentration": [
            {
                "framing_id": row["framing_id"],
                "target_id": row["target_id"],
                "scenario_id": row["scenario_id"],
                "top1_city_sse_share": row["top1_city_sse_share"],
                "top3_city_sse_share": row["top3_city_sse_share"],
            }
            for row in aggregate_rows
            if row["family"] == FAMILY
        ],
        "preprocessing": {
            "numeric": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True), StandardScaler, PCA; fold-local",
            "pca_components": [8, 16, 12],
            "city_id": "raw grouping variable only; not PCA input",
            "random_slope": f"standardized {SLOPE_FEATURE} using each training fold only",
            "reml": False,
            "primary_optimizer": "statsmodels MixedLM default (bfgs -> lbfgs -> cg)",
            "fallback_optimizer": "lbfgs only when not already used by the primary path",
        },
        "forbidden_actions_not_run": [
            "alter_plan04_144_features",
            "alter_replay_rows",
            "alter_validation_geometry",
            "outer_label_driven_rescue",
            "city_routing",
            "ensemble",
            "stacking",
            "gating",
            "2023_forecast",
            "submission",
            "s5_complete",
            "s6_entered",
        ],
        "elapsed_seconds_before_report_write": float(time.time() - started),
    }

    aggregate_fields = [
        "model_id",
        "family",
        "framing_id",
        "target_id",
        "scenario_id",
        "row_count",
        "pooled_city_year_mse",
        "macro_origin_mse",
        "latest_2021_mse",
        "rmse",
        "mae",
        "bias",
        "origin_win_count_vs_b3",
        "origin_count",
        "city_sse_total",
        "top1_city_sse_share",
        "top3_city_sse_share",
        "b3_pooled_city_year_mse",
        "improvement_vs_b3_pct",
    ]
    origin_fields = [
        "model_id",
        "family",
        "framing_id",
        "target_id",
        "scenario_id",
        "origin_year",
        "row_count",
        "origin_mse",
        "origin_rmse",
        "origin_mae",
        "origin_bias",
        "b3_origin_mse",
        "origin_win_vs_b3",
    ]
    city_fields = [
        "model_id",
        "family",
        "framing_id",
        "target_id",
        "scenario_id",
        "city_id",
        "row_count",
        "city_sse",
        "city_mse",
        "city_rmse",
        "city_bias",
    ]
    selection_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "selected_config_id",
        "selection_method",
        "fallback_reason",
        "eligible_inner_fold_count",
        "selected_mean_inner_mse",
        "config_1_id",
        "config_1_mean_inner_mse",
        "config_1_valid_fold_count",
        "config_1_failed_fold_count",
        "config_2_id",
        "config_2_mean_inner_mse",
        "config_2_valid_fold_count",
        "config_2_failed_fold_count",
        "config_3_id",
        "config_3_mean_inner_mse",
        "config_3_valid_fold_count",
        "config_3_failed_fold_count",
        "outer_fit_status",
        "outer_fit_optimizer",
        "outer_fit_fallback_used",
        "outer_fit_error",
        "outer_pca_components_requested",
        "outer_pca_components_effective",
        "unseen_city_count",
    ]
    evidence_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "config_id",
        "config_order",
        "inner_validation_year",
        "inner_training_row_count",
        "inner_validation_row_count",
        "pca_components_requested",
        "pca_components_effective",
        "fit_status",
        "optimizer",
        "fallback_used",
        "converged",
        "validation_mse",
        "eligible_fold_count",
        "valid_prediction_count",
        "fit_error",
        "selection_method",
    ]
    missingness_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "config_id",
        "pca_components_requested",
        "pca_components_effective",
        "fit_status",
        "split",
        "row_count",
        "feature_count",
        "missing_cell_count",
        "missing_cell_rate",
        "rows_with_any_missing",
        "rows_with_any_missing_rate",
        "features_with_missing",
        "all_missing_feature_count",
    ]
    random_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "config_id",
        "city_id",
        "random_intercept",
        "random_slope",
        "city_seen_in_training",
        "city_training_row_count",
        "fit_status",
        "optimizer",
        "fallback_used",
    ]
    variance_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "config_id",
        "random_intercept_variance",
        "random_slope_variance",
        "random_intercept_slope_covariance",
        "residual_variance",
        "random_effect_group_count",
        "convergence_status",
        "optimizer",
        "fallback_used",
        "fit_error",
    ]
    pooling_fields = [
        "family",
        "model_id",
        "framing_id",
        "target_id",
        "scenario_id",
        "outer_origin_year",
        "summary_scope",
        "config_id",
        "fit_status",
        "raw_between_city_mean_variance",
        "fixed_effect_residual_between_city_mean_variance",
        "random_intercept_variance",
        "random_intercept_variance_fraction_of_raw",
        "mean_abs_random_intercept",
        "mean_abs_raw_city_mean_deviation",
        "empirical_shrinkage_proxy",
        "low_history_mean_abs_random_intercept",
        "high_history_mean_abs_random_intercept",
        "low_history_mean_abs_raw_deviation",
        "high_history_mean_abs_raw_deviation",
        "random_slope_variance",
        "random_slope_sd",
        "random_slope_meaningfully_nonzero",
        "random_slope_meaningfully_nonzero_fit_fraction",
        "outer_fit_count",
        "city_group_count",
        "low_history_definition",
        "interpretation",
    ]

    write_rows(OUTPUT_DIR / "oof_predictions.csv", all_oof.to_dict("records"), oof_fields)
    write_rows(OUTPUT_DIR / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)
    write_rows(OUTPUT_DIR / "per_origin_metrics.csv", origin_rows, origin_fields)
    write_rows(OUTPUT_DIR / "configuration_selection.csv", result["selection_rows"], selection_fields)
    write_rows(OUTPUT_DIR / "inner_selection_evidence.csv", result["evidence_rows"], evidence_fields)
    write_rows(OUTPUT_DIR / "city_model_error.csv", city_rows, city_fields)
    write_rows(OUTPUT_DIR / "missingness_diagnostics.csv", result["missingness_rows"], missingness_fields)
    write_rows(OUTPUT_DIR / "random_effects_diagnostics.csv", result["random_rows"], random_fields)
    write_rows(OUTPUT_DIR / "variance_components.csv", result["variance_rows"], variance_fields)
    write_rows(OUTPUT_DIR / "pooling_strength_summary.csv", pooling_strength_rows, pooling_fields)
    write_json(OUTPUT_DIR / "model_summary.json", summary)
    (OUTPUT_DIR / "HANDOFF.md").write_text(
        handoff_text(summary, aggregate_rows, environment, worker_commit),
        encoding="utf-8",
    )
    print(
        f"[{PLAN_ID}] complete: {len(all_oof)} OOF rows, coverage={coverage_status}, "
        f"outer failures={len(result['model_failures'])}, elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    if coverage_status != "PASS":
        raise RuntimeError("hierarchical_worker_oof_coverage_incomplete")


if __name__ == "__main__":
    main()
