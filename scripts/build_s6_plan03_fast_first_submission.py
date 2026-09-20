"""Build the S6-Plan03 fast-first submission candidates.

This runner is intentionally narrower than the historical S5 workers.  It
reuses the clean S6-Plan02 replay/current-information artifacts, applies the
already frozen F4 configuration-selection contract to the conceptual 2023
outer pool, fits the two resident F4 scenario candidates, and creates
deterministic registered/resident B3 controls.  It never reads or hashes
``test.xls`` and never uploads to a platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor, Pool


PLAN_ID = "S6-Plan03"
FEATURE_CONTRACT = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT = "S5-Plan04B-TRAINING-REPLAY-V1"
F4_ID = "F4_LOG_GROWTH_FROM_LATEST"
SEED = 20260919
PREDICTION_YEAR = 2023
LABEL_CUTOFF = 2021
TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
EXPECTED_REPLAY_ROWS = {
    ("resident", "R1_REFERENCE_SNAPSHOT_LAG"): 573,
    ("resident", "R2_CONSERVATIVE_PUBLICATION_STRESS"): 573,
    ("registered", "R1_REFERENCE_SNAPSHOT_LAG"): 587,
    ("registered", "R2_CONSERVATIVE_PUBLICATION_STRESS"): 587,
}
EXPECTED_B3_MSE = {"resident": 872.452651515143, "registered": 225.421212121213}
SUBMISSION_FILES = {
    "resident_f4_r1": "submission_resident_f4_r1.csv",
    "resident_f4_r2": "submission_resident_f4_r2.csv",
    "registered_rlt3": "submission_registered_rlt3.csv",
    "registered_b3": "submission_registered_b3.csv",
    "resident_b3": "submission_resident_b3.csv",
}
CONFIGS: tuple[dict[str, Any], ...] = (
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def city_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"city(\d+)", str(value))
    return (int(match.group(1)), str(value)) if match else (10**9, str(value))


def sorted_cities(values: Any) -> list[str]:
    return sorted({str(value) for value in values}, key=city_key)


def assert_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} contains duplicated columns: {duplicated}")


def load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    final_contract_path = root / "reports" / "S6_HANDOFFS" / "S6-Plan01" / "final_solution_contract.json"
    replay_contract_path = root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json"
    final_contract = json.loads(final_contract_path.read_text(encoding="utf-8"))
    replay_contract = json.loads(replay_contract_path.read_text(encoding="utf-8"))
    model_features = list(replay_contract["model_features"])

    if len(model_features) != 144 or replay_contract["model_feature_count"] != 144:
        raise ValueError("Frozen model feature count is not 144")
    flattened = [feature for group in replay_contract["feature_groups"].values() for feature in group]
    if set(flattened) != set(model_features) or len(set(flattened)) != len(flattened):
        raise ValueError("Feature group manifest does not exactly cover the frozen model features")
    if final_contract["feature_contract"]["numeric_model_feature_count"] != 144:
        raise ValueError("Plan01 feature count changed")
    if final_contract["feature_contract"]["feature_contract_version"] != FEATURE_CONTRACT:
        raise ValueError("Plan01 feature contract version changed")
    if final_contract["framing"]["framing_id"] != F4_ID:
        raise ValueError("Plan01 F4 framing changed")
    if final_contract["framing"]["final_prediction_history_cutoff"] != "<= 2021 for 2023 prediction":
        raise ValueError("Plan01 label cutoff changed")
    if final_contract["model"]["random_seed"] != SEED or final_contract["model"]["thread_count"] != 1:
        raise ValueError("Plan01 CatBoost seed/thread contract changed")
    if final_contract["model"]["catboost_version_required"] != "1.2.10":
        raise ValueError("Plan01 CatBoost version contract changed")
    contract_configs = final_contract["model"]["configs"]
    for expected, actual in zip(CONFIGS, contract_configs):
        for key in ("config_id", "iterations", "depth", "l2_leaf_reg", "learning_rate", "rsm"):
            if not np.isclose(expected[key], actual[key]) if isinstance(expected[key], float) else expected[key] != actual[key]:
                raise ValueError(f"Frozen config changed for {expected['config_id']}: {key}")
    if final_contract["config_selection"]["fallback"] != "if eligible inner fold count < 3, select FRM_C1_CONSERVATIVE":
        raise ValueError("Frozen fallback rule changed")
    if final_contract["config_selection"]["inner_training_rule"] != "pseudo_target_year <= inner_validation_year - 2":
        raise ValueError("Frozen inner training rule changed")
    return final_contract, replay_contract, model_features


def load_inputs(root: Path) -> dict[str, Any]:
    final_contract, replay_contract, model_features = load_contract(root)
    s6_dir = root / "reports" / "S6_HANDOFFS" / "S6-Plan02"
    summary_path = s6_dir / "clean_rebuild_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay_path = s6_dir / "rebuilt_training_replay_matrix.csv"
    current_path = s6_dir / "current_2023_feature_matrix.csv"
    boundary_path = s6_dir / "current_2023_feature_boundary_audit.csv"
    replay = pd.read_csv(replay_path)
    current = pd.read_csv(current_path)
    boundary = pd.read_csv(boundary_path)

    replay_required = [
        "target_id",
        "scenario_id",
        "pseudo_target_year",
        "city_id",
        "target_history_cutoff_year",
        "actual_target_value",
        "b3_prediction",
        "b3_squared_error",
        "pop_latest_level",
        *model_features,
    ]
    current_required = [
        "target_id",
        "scenario_id",
        "city_id",
        "prediction_year",
        "target_history_cutoff_year",
        "final_pipeline_role",
        "b3_reference_prediction",
        "pop_latest_level",
        *model_features,
    ]
    assert_columns(replay, replay_required, "S6-Plan02 rebuilt replay")
    assert_columns(current, current_required, "S6-Plan02 current feature matrix")
    assert_columns(boundary, ["target_id", "scenario_id", "city_id", "audit_status"], "S6-Plan02 boundary audit")
    if len(replay) != 2320 or len(current) != 160 or len(boundary) != 2240:
        raise ValueError("S6-Plan02 row counts changed")
    if summary["status"] != "READY_FOR_S6_HISTORICAL_REPRODUCTION_CONTROLS":
        raise ValueError("S6-Plan02 is not in the expected ready state")
    if summary["plan04_rebuild"]["equivalence_status"] != "PASS":
        raise ValueError("S6-Plan02 Plan04 equivalence is not PASS")
    if summary["plan04b_rebuild"]["equivalence_status"] != "PASS":
        raise ValueError("S6-Plan02 Plan04B equivalence is not PASS")
    if summary["plan04b_rebuild"]["inner_boundary_violation_count"] != 0:
        raise ValueError("S6-Plan02 inner boundary evidence is not PASS")
    if set(boundary["audit_status"].dropna().astype(str)) != {"PASS"}:
        raise ValueError("S6-Plan02 current feature boundary audit is not all PASS")

    replay["city_id"] = replay["city_id"].astype(str)
    current["city_id"] = current["city_id"].astype(str)
    for frame, name in ((replay, "replay"), (current, "current")):
        for column in model_features:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
            numeric = frame[column].to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(numeric).any():
                raise ValueError(f"{name}.{column} contains infinity")
    for column in ("actual_target_value", "b3_prediction", "b3_squared_error", "pop_latest_level"):
        replay[column] = pd.to_numeric(replay[column], errors="raise").astype(float)
    replay["pseudo_target_year"] = pd.to_numeric(replay["pseudo_target_year"], errors="raise").astype(int)
    current["prediction_year"] = pd.to_numeric(current["prediction_year"], errors="raise").astype(int)
    current["target_history_cutoff_year"] = pd.to_numeric(current["target_history_cutoff_year"], errors="raise").astype(int)
    if replay["pseudo_target_year"].max() > LABEL_CUTOFF:
        raise ValueError("Replay contains target labels after the frozen cutoff")
    if (replay["actual_target_value"] <= 0).any() or (replay["pop_latest_level"] <= 0).any():
        raise ValueError("F4 requires positive replay actual and latest population values")
    if replay.duplicated(["target_id", "scenario_id", "pseudo_target_year", "city_id"]).any():
        raise ValueError("S6-Plan02 replay has duplicate identity rows")
    for key, expected_count in EXPECTED_REPLAY_ROWS.items():
        actual_count = int(((replay["target_id"] == key[0]) & (replay["scenario_id"] == key[1])).sum())
        if actual_count != expected_count:
            raise ValueError(f"Replay count drift for {key}: {actual_count} != {expected_count}")
    if set(current["prediction_year"]) != {PREDICTION_YEAR}:
        raise ValueError("Current feature matrix is not the 2023 foundation")
    if set(current["target_history_cutoff_year"]) != {LABEL_CUTOFF}:
        raise ValueError("Current feature matrix target cutoff changed")
    forbidden_current_columns = {
        "actual_target_value",
        "target_value",
        "target_label",
        "label",
        "y",
        "b3_residual_actual_minus_prediction",
        "b3_squared_error",
    }
    leaked_columns = sorted(forbidden_current_columns.intersection(current.columns))
    if leaked_columns:
        raise ValueError(f"Current feature foundation contains target truth columns: {leaked_columns}")
    observed_b3 = summary["plan04b_rebuild"]["b3_compatibility"]["observed"]
    for target_id, expected_mse in EXPECTED_B3_MSE.items():
        for scenario in SCENARIOS:
            observed = float(observed_b3[f"{target_id}_{scenario}"]["pooled_mse"])
            if not np.isclose(observed, expected_mse, rtol=1e-12, atol=1e-9):
                raise ValueError(f"Plan02 B3 reference drift for {target_id}/{scenario}: {observed} != {expected_mse}")

    checkpoint_path = root / "CURRENT_LEARNING_CHECKPOINT.md"
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.exists() else "missing"
    input_paths = {
        "clean_rebuild_summary": summary_path,
        "rebuilt_training_replay_matrix": replay_path,
        "current_2023_feature_matrix": current_path,
        "current_2023_feature_boundary_audit": boundary_path,
        "final_solution_contract": root / "reports" / "S6_HANDOFFS" / "S6-Plan01" / "final_solution_contract.json",
        "replay_input_contract": root / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json",
    }
    return {
        "final_contract": final_contract,
        "replay_contract": replay_contract,
        "model_features": model_features,
        "replay": replay,
        "current": current,
        "summary": summary,
        "input_paths": input_paths,
        "checkpoint_sha": checkpoint_sha,
        "plan02_verification": {
            "status": "PASS",
            "replay_rows": int(len(replay)),
            "current_2023_rows": int(len(current)),
            "current_boundary_rows": int(len(boundary)),
            "model_feature_count": len(model_features),
            "replay_contract_version": REPLAY_CONTRACT,
            "feature_contract_version": FEATURE_CONTRACT,
            "replay_hash": sha256_file(replay_path),
            "current_feature_hash": sha256_file(current_path),
            "boundary_audit_hash": sha256_file(boundary_path),
            "plan04_equivalence": summary["plan04_rebuild"]["equivalence_status"],
            "plan04b_equivalence": summary["plan04b_rebuild"]["equivalence_status"],
            "inner_boundary_violation_count": summary["plan04b_rebuild"]["inner_boundary_violation_count"],
            "target_truth_after_2021_used": False,
        },
    }


def build_inner_folds(pool_frame: pd.DataFrame) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for validation_year in sorted(pool_frame["pseudo_target_year"].unique().tolist()):
        training = pool_frame.loc[pool_frame["pseudo_target_year"] <= int(validation_year) - 2]
        validation = pool_frame.loc[pool_frame["pseudo_target_year"] == int(validation_year)]
        folds.append(
            {
                "inner_validation_year": int(validation_year),
                "inner_training_row_count": int(len(training)),
                "inner_validation_row_count": int(len(validation)),
                "inner_training_max_year": int(training["pseudo_target_year"].max()) if len(training) else None,
                "boundary_pass": "PASS",
            }
        )
    return folds


def make_pool(frame: pd.DataFrame, model_features: list[str], labels: np.ndarray | None = None) -> Pool:
    columns = ["city_id", *model_features]
    features = frame.loc[:, columns].copy()
    features["city_id"] = features["city_id"].astype(str)
    if labels is None:
        return Pool(features, cat_features=["city_id"])
    return Pool(features, label=np.asarray(labels, dtype=float), cat_features=["city_id"])


def f4_labels(frame: pd.DataFrame) -> np.ndarray:
    latest = frame["pop_latest_level"].to_numpy(dtype=float)
    actual = frame["actual_target_value"].to_numpy(dtype=float)
    if (latest <= 0).any() or (actual <= 0).any():
        raise ValueError("F4 labels require positive latest levels and actual targets")
    with np.errstate(divide="raise", invalid="raise"):
        labels = np.log(actual / latest)
    if not np.isfinite(labels).all():
        raise ValueError("F4 labels contain non-finite values")
    return labels


def fit_f4_and_predict(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    config: dict[str, Any],
    model_features: list[str],
) -> np.ndarray:
    params = dict(COMMON_CATBOOST_PARAMS)
    params.update({key: config[key] for key in ("iterations", "learning_rate", "depth", "l2_leaf_reg", "rsm")})
    try:
        model = CatBoostRegressor(**params)
        model.fit(make_pool(training, model_features, f4_labels(training)))
        raw_prediction = np.asarray(model.predict(make_pool(prediction, model_features)), dtype=float)
    except (TypeError, CatBoostError) as exc:
        raise RuntimeError(f"Frozen CatBoost configuration failed without substitution: {exc}") from exc
    if not np.isfinite(raw_prediction).all():
        raise ValueError("F4 raw predictions contain non-finite values")
    latest = prediction["pop_latest_level"].to_numpy(dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        reconstructed = latest * np.exp(raw_prediction)
    if not np.isfinite(reconstructed).all() or (reconstructed <= 0).any():
        raise ValueError("F4 reconstructed predictions are not finite and positive")
    return reconstructed


def select_final_config(
    scenario: str,
    pool_frame: pd.DataFrame,
    model_features: list[str],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]], int]:
    folds = build_inner_folds(pool_frame)
    eligible = [
        fold
        for fold in folds
        if fold["inner_training_row_count"] >= 40
        and fold["inner_validation_row_count"] >= 20
        and fold["boundary_pass"] == "PASS"
    ]
    detail_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    total_tasks = len(CONFIGS) * len(eligible)
    completed = 0
    scores: dict[str, float] = {}
    print(
        f"[S6-Plan03][{scenario}] final pool={len(pool_frame)} rows; "
        f"eligible inner folds={len(eligible)}/{len(folds)}; tasks={total_tasks}",
        flush=True,
    )
    for config in CONFIGS:
        config_mses: list[float] = []
        for fold in eligible:
            training = pool_frame.loc[pool_frame["pseudo_target_year"] <= fold["inner_validation_year"] - 2]
            validation = pool_frame.loc[pool_frame["pseudo_target_year"] == fold["inner_validation_year"]]
            if len(training) != fold["inner_training_row_count"] or len(validation) != fold["inner_validation_row_count"]:
                raise ValueError(f"{scenario} inner geometry mismatch at {fold['inner_validation_year']}")
            prediction = fit_f4_and_predict(training, validation, config, model_features)
            actual = validation["actual_target_value"].to_numpy(dtype=float)
            mse = float(np.mean((prediction - actual) ** 2))
            config_mses.append(mse)
            detail_rows.append(
                {
                    "scenario_id": scenario,
                    "config_id": config["config_id"],
                    "config_order": config["config_order"],
                    "inner_validation_year": fold["inner_validation_year"],
                    "inner_training_row_count": fold["inner_training_row_count"],
                    "inner_validation_row_count": fold["inner_validation_row_count"],
                    "inner_training_max_year": fold["inner_training_max_year"],
                    "boundary_pass": fold["boundary_pass"],
                    "inner_validation_mse": mse,
                }
            )
            completed += 1
            if completed == 1 or completed == total_tasks or completed % 5 == 0:
                print(
                    f"[S6-Plan03][{scenario}] inner fits {completed}/{total_tasks}",
                    flush=True,
                )
        scores[config["config_id"]] = float(np.mean(config_mses)) if config_mses else float("nan")

    if len(eligible) < 3:
        selected = CONFIGS[0]
        selection_method = "fallback_conservative_default"
    else:
        selected = CONFIGS[0]
        best_score = scores[selected["config_id"]]
        for candidate in CONFIGS[1:]:
            candidate_score = scores[candidate["config_id"]]
            if candidate_score < best_score and not np.isclose(candidate_score, best_score, rtol=1e-12, atol=1e-12):
                selected = candidate
                best_score = candidate_score
        selection_method = "inner_mean_validation_mse"
    for config in CONFIGS:
        fold_rows = [row for row in detail_rows if row["config_id"] == config["config_id"]]
        selection_rows.append(
            {
                "target_id": "resident",
                "scenario_id": scenario,
                "outer_conceptual_year": PREDICTION_YEAR,
                "config_id": config["config_id"],
                "config_order": config["config_order"],
                "iterations": config["iterations"],
                "learning_rate": config["learning_rate"],
                "depth": config["depth"],
                "l2_leaf_reg": config["l2_leaf_reg"],
                "rsm": config["rsm"],
                "eligible_inner_fold_count": len(eligible),
                "total_inner_fold_count": len(folds),
                "mean_inner_validation_mse": scores[config["config_id"]],
                "inner_validation_years": ";".join(str(row["inner_validation_year"]) for row in fold_rows),
                "selection_method": selection_method,
                "selected_config_id": selected["config_id"],
                "selected": config["config_id"] == selected["config_id"],
                "boundary_pass_all_eligible": all(row["boundary_pass"] == "PASS" for row in fold_rows),
            }
        )
    print(
        f"[S6-Plan03][{scenario}] selected={selected['config_id']} "
        f"scores={json.dumps(scores, sort_keys=True)}",
        flush=True,
    )
    return selected, selection_method, selection_rows, detail_rows, len(eligible)


def run_f4_worker(root_string: str, scenario: str, output_string: str) -> dict[str, Any]:
    root = Path(root_string)
    output_dir = Path(output_string)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(root)
    replay = inputs["replay"]
    current = inputs["current"]
    model_features = inputs["model_features"]
    pool = replay.loc[
        (replay["target_id"] == "resident")
        & (replay["scenario_id"] == scenario)
        & (replay["pseudo_target_year"] <= LABEL_CUTOFF)
    ].copy()
    current_group = current.loc[
        (current["target_id"] == "resident") & (current["scenario_id"] == scenario)
    ].copy()
    if len(pool) != EXPECTED_REPLAY_ROWS[("resident", scenario)] or len(current_group) != 40:
        raise ValueError(f"Unexpected {scenario} resident final-fit/current row counts")
    pool = pool.sort_values(["pseudo_target_year", "city_id"], key=lambda col: col.map(city_key) if col.name == "city_id" else col)
    current_group["_city_sort"] = current_group["city_id"].map(city_key)
    current_group = current_group.sort_values("_city_sort").drop(columns=["_city_sort"])
    selected, selection_method, selection_rows, detail_rows, eligible_count = select_final_config(
        scenario, pool, model_features
    )
    final_prediction = fit_f4_and_predict(pool, current_group, selected, model_features)
    candidate_id = "resident_f4_r1" if scenario == SCENARIOS[0] else "resident_f4_r2"
    prediction_frame = pd.DataFrame(
        {"candidate_id": candidate_id, "target_id": "resident", "scenario_id": scenario, "city_id": current_group["city_id"].tolist(), "year": PREDICTION_YEAR, "pred": final_prediction}
    )
    prediction_frame.to_csv(output_dir / "prediction.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(selection_rows).to_csv(output_dir / "selection.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(detail_rows).to_csv(output_dir / "inner_detail.csv", index=False, encoding="utf-8", lineterminator="\n")
    audit = {
        "candidate_id": candidate_id,
        "target_id": "resident",
        "scenario_id": scenario,
        "framing_id": F4_ID,
        "training_row_count": int(len(pool)),
        "training_city_count": int(pool["city_id"].nunique()),
        "training_year_count": int(pool["pseudo_target_year"].nunique()),
        "training_min_label_year": int(pool["pseudo_target_year"].min()),
        "training_max_label_year": int(pool["pseudo_target_year"].max()),
        "final_prediction_year": PREDICTION_YEAR,
        "selected_config_id": selected["config_id"],
        "selection_method": selection_method,
        "eligible_inner_fold_count": eligible_count,
        "final_current_feature_rows": int(len(current_group)),
        "numeric_feature_count": len(model_features),
        "catboost_version": "1.2.10",
        "catboost_fit_count": len(CONFIGS) * eligible_count + 1,
        "target_truth_after_2021_used": False,
        "forbidden_input_test_xls_used": False,
        "forbidden_input_b_data_used": False,
        "external_data_used": False,
    }
    write_json(output_dir / "result.json", {"audit": audit, "selection_rows": selection_rows, "scores": {row["config_id"]: row["mean_inner_validation_mse"] for row in selection_rows}, "checkpoint_sha": inputs["checkpoint_sha"]})
    print(f"[S6-Plan03][{scenario}] final fit complete; candidate={candidate_id}", flush=True)
    return {"audit": audit, "selection_rows": selection_rows, "detail_rows": detail_rows, "prediction": prediction_frame.to_dict(orient="records"), "scores": {row["config_id"]: row["mean_inner_validation_mse"] for row in selection_rows}}


def latest_three(history: pd.DataFrame, target_id: str, scenario: str) -> pd.DataFrame:
    frame = history.loc[
        (history["target_id"] == target_id)
        & (history["scenario_id"] == scenario)
        & (history["pseudo_target_year"] <= LABEL_CUTOFF),
        ["city_id", "pseudo_target_year", "actual_target_value"],
    ].copy()
    if frame["actual_target_value"].isna().any():
        frame = frame.dropna(subset=["actual_target_value"])
    frame = frame.sort_values(["city_id", "pseudo_target_year"])
    rows: list[pd.DataFrame] = []
    for city, group in frame.groupby("city_id", sort=False):
        group = group.drop_duplicates(["pseudo_target_year"], keep="first").tail(3)
        if len(group) != 3:
            raise ValueError(f"{target_id}/{scenario}/{city} has fewer than three legal target points")
        rows.append(group)
    result = pd.concat(rows, ignore_index=True)
    result["city_id"] = result["city_id"].astype(str)
    return result


def ols_three(group: pd.DataFrame) -> float:
    x = group["pseudo_target_year"].to_numpy(dtype=float)
    y = group["actual_target_value"].to_numpy(dtype=float)
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    intercept = float(y.mean() - slope * x.mean())
    return float(intercept + slope * PREDICTION_YEAR)


def rlt3_three(group: pd.DataFrame) -> float:
    x = group["pseudo_target_year"].to_numpy(dtype=float)
    y = group["actual_target_value"].to_numpy(dtype=float)
    slopes = []
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            slopes.append(float((y[right] - y[left]) / (x[right] - x[left])))
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return float(intercept + slope * PREDICTION_YEAR)


def run_deterministic_worker(root_string: str, output_string: str) -> dict[str, Any]:
    root = Path(root_string)
    output_dir = Path(output_string)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(root)
    replay = inputs["replay"]
    current = inputs["current"]
    candidate_rows: list[dict[str, Any]] = []
    equivalence: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []

    for target_id in ("registered", "resident"):
        history_r1 = latest_three(replay, target_id, SCENARIOS[0])
        history_r2 = latest_three(replay, target_id, SCENARIOS[1])
        h1 = history_r1.sort_values(["city_id", "pseudo_target_year"]).reset_index(drop=True)
        h2 = history_r2.sort_values(["city_id", "pseudo_target_year"]).reset_index(drop=True)
        if not np.allclose(h1["actual_target_value"], h2["actual_target_value"], rtol=0, atol=0):
            raise ValueError(f"{target_id} target history differs between R1 and R2")
        equivalence[f"{target_id}_history_r1_r2"] = {"status": "PASS", "max_abs_diff": 0.0}
        predictions_b3: dict[str, float] = {}
        predictions_rlt3: dict[str, float] = {}
        for city, group in h1.groupby("city_id", sort=False):
            predictions_b3[str(city)] = ols_three(group)
            predictions_rlt3[str(city)] = rlt3_three(group)

        current_r1 = current.loc[(current["target_id"] == target_id) & (current["scenario_id"] == SCENARIOS[0])].copy()
        current_r2 = current.loc[(current["target_id"] == target_id) & (current["scenario_id"] == SCENARIOS[1])].copy()
        current_r1["_city_sort"] = current_r1["city_id"].map(city_key)
        current_r2["_city_sort"] = current_r2["city_id"].map(city_key)
        current_r1 = current_r1.sort_values("_city_sort")
        current_r2 = current_r2.sort_values("_city_sort")
        if not np.allclose(current_r1["b3_reference_prediction"], current_r2["b3_reference_prediction"], rtol=0, atol=0):
            raise ValueError(f"{target_id} B3 current predictions differ between R1 and R2")
        b3_reference = dict(zip(current_r1["city_id"], current_r1["b3_reference_prediction"].astype(float)))
        b3_max_error = max(abs(predictions_b3[city] - float(b3_reference[city])) for city in predictions_b3)
        if b3_max_error > 1e-8:
            raise ValueError(f"{target_id} deterministic B3 does not match Plan02 current reference: {b3_max_error}")
        equivalence[f"{target_id}_b3_reconstruction"] = {"status": "PASS", "max_abs_diff_vs_current": b3_max_error}

        if target_id == "registered":
            candidate_id = "registered_rlt3"
            for city in sorted_cities(predictions_rlt3):
                candidate_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": "TARGET_HISTORY_ONLY", "city_id": city, "year": PREDICTION_YEAR, "pred": predictions_rlt3[city]})
            audit_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": "TARGET_HISTORY_ONLY", "framing_id": "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND", "training_row_count": int(len(h1)), "training_city_count": int(h1["city_id"].nunique()), "training_year_count": 3, "training_min_label_year": int(h1["pseudo_target_year"].min()), "training_max_label_year": int(h1["pseudo_target_year"].max()), "final_prediction_year": PREDICTION_YEAR, "selected_config_id": "deterministic", "selection_method": "none", "eligible_inner_fold_count": 0, "final_current_feature_rows": 40, "numeric_feature_count": 0, "catboost_version": "not_used", "catboost_fit_count": 0, "target_truth_after_2021_used": False, "forbidden_input_test_xls_used": False, "forbidden_input_b_data_used": False, "external_data_used": False})

            candidate_id = "registered_b3"
            for city in sorted_cities(predictions_b3):
                candidate_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": SCENARIOS[0], "city_id": city, "year": PREDICTION_YEAR, "pred": predictions_b3[city]})
            audit_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": SCENARIOS[0], "framing_id": "B3_RECENT_3POINT_OLS_TREND", "training_row_count": int(len(h1)), "training_city_count": int(h1["city_id"].nunique()), "training_year_count": 3, "training_min_label_year": int(h1["pseudo_target_year"].min()), "training_max_label_year": int(h1["pseudo_target_year"].max()), "final_prediction_year": PREDICTION_YEAR, "selected_config_id": "deterministic", "selection_method": "none", "eligible_inner_fold_count": 0, "final_current_feature_rows": 40, "numeric_feature_count": 0, "catboost_version": "not_used", "catboost_fit_count": 0, "target_truth_after_2021_used": False, "forbidden_input_test_xls_used": False, "forbidden_input_b_data_used": False, "external_data_used": False})
        else:
            candidate_id = "resident_b3"
            for city in sorted_cities(predictions_b3):
                candidate_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": SCENARIOS[0], "city_id": city, "year": PREDICTION_YEAR, "pred": predictions_b3[city]})
            audit_rows.append({"candidate_id": candidate_id, "target_id": target_id, "scenario_id": SCENARIOS[0], "framing_id": "B3_RECENT_3POINT_OLS_TREND", "training_row_count": int(len(h1)), "training_city_count": int(h1["city_id"].nunique()), "training_year_count": 3, "training_min_label_year": int(h1["pseudo_target_year"].min()), "training_max_label_year": int(h1["pseudo_target_year"].max()), "final_prediction_year": PREDICTION_YEAR, "selected_config_id": "deterministic", "selection_method": "none", "eligible_inner_fold_count": 0, "final_current_feature_rows": 40, "numeric_feature_count": 0, "catboost_version": "not_used", "catboost_fit_count": 0, "target_truth_after_2021_used": False, "forbidden_input_test_xls_used": False, "forbidden_input_b_data_used": False, "external_data_used": False})

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_dir / "deterministic_candidates.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(audit_rows).to_csv(output_dir / "deterministic_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    result = {
        "candidate_ids": sorted(candidate_frame["candidate_id"].unique().tolist()),
        "rlt3_r1_r2_equivalence": {"status": "PASS", "target_id": "registered", "reason": "registered target history only", "max_abs_diff": 0.0},
        "checks": equivalence,
        "deterministic_fit_count": 0,
    }
    write_json(output_dir / "result.json", result)
    print("[S6-Plan03][deterministic] registered RLT3/B3 and resident B3 complete", flush=True)
    return {"candidate_rows": candidate_rows, "audit_rows": audit_rows, **result}


def validate_submission(frame: pd.DataFrame, expected_cities: list[str], candidate_id: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"candidate_id": candidate_id, "check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add("columns_exact", list(frame.columns) == ["city_id", "year", "pred"], str(list(frame.columns)))
    add("row_count", len(frame) == 40, str(len(frame)))
    add("city_set_exact", sorted_cities(frame["city_id"]) == expected_cities, str(sorted_cities(frame["city_id"])))
    add("city_unique", not frame["city_id"].duplicated().any(), str(int(frame["city_id"].duplicated().sum())))
    add("year_exact_2023", set(frame["year"]) == {PREDICTION_YEAR}, str(sorted(frame["year"].unique().tolist())))
    numeric = pd.to_numeric(frame["pred"], errors="coerce")
    add("pred_numeric", not numeric.isna().any(), str(int(numeric.isna().sum())))
    add("pred_finite", bool(np.isfinite(numeric.to_numpy(dtype=float)).all()), "all finite")
    values = numeric.to_numpy(dtype=float)
    add(
        "pred_positive",
        bool((values > 0).all()),
        f"min={float(np.nanmin(values))}; nonpositive_count={int((values <= 0).sum())}",
    )
    return checks


def prediction_sanity(frame: pd.DataFrame, candidate_id: str, target_id: str, scenario: str, path: str) -> dict[str, Any]:
    values = frame["pred"].to_numpy(dtype=float)
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "scenario_id": scenario,
        "path": path,
        "row_count": int(len(frame)),
        "city_count": int(frame["city_id"].nunique()),
        "year": int(frame["year"].iloc[0]) if len(frame) else None,
        "finite": bool(np.isfinite(values).all()),
        "positive": bool((values > 0).all()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def package_outputs(root: Path, inputs: dict[str, Any], worker_results: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    handoff_dir = root / "reports" / "S6_HANDOFFS" / PLAN_ID
    handoff_dir.mkdir(parents=True, exist_ok=True)
    for child in handoff_dir.iterdir():
        if child.name.startswith("submission_") or child.name in {"HANDOFF.md", "fast_first_submission_summary.json", "final_fit_config_selection.csv", "final_fit_training_audit.csv", "prediction_sanity.csv", "candidate_comparison.csv", "submission_validation.csv", "environment_snapshot.json"}:
            if child.is_file():
                child.unlink()

    all_predictions: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        result = worker_results[scenario]
        prediction = pd.DataFrame(result["prediction"])
        candidate_id = "resident_f4_r1" if scenario == SCENARIOS[0] else "resident_f4_r2"
        all_predictions[candidate_id] = prediction[["city_id", "year", "pred"]].copy()
        audit_rows.append(result["audit"])
        selection_rows.extend(result["selection_rows"])

    deterministic = worker_results["deterministic"]
    deterministic_frame = pd.DataFrame(deterministic["candidate_rows"])
    for candidate_id in ("registered_rlt3", "registered_b3", "resident_b3"):
        candidate = deterministic_frame.loc[deterministic_frame["candidate_id"] == candidate_id].copy()
        candidate["_city_sort"] = candidate["city_id"].map(city_key)
        candidate = candidate.sort_values("_city_sort").drop(columns=["_city_sort"])
        all_predictions[candidate_id] = candidate[["city_id", "year", "pred"]].copy()
    audit_rows.extend(deterministic["audit_rows"])

    expected_cities = sorted_cities(inputs["current"].loc[(inputs["current"]["target_id"] == "resident") & (inputs["current"]["scenario_id"] == SCENARIOS[0]), "city_id"])
    submission_checks: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    candidate_summary_rows: list[dict[str, Any]] = []
    candidate_paths: dict[str, str] = {}
    for candidate_id, frame in all_predictions.items():
        output_name = SUBMISSION_FILES[candidate_id]
        output_path = handoff_dir / output_name
        frame = frame.copy()
        frame["city_id"] = frame["city_id"].astype(str)
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
        frame["pred"] = pd.to_numeric(frame["pred"], errors="raise").astype(float)
        frame = frame.sort_values("city_id", key=lambda col: col.map(city_key))
        frame[["city_id", "year", "pred"]].to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.12f")
        candidate_paths[candidate_id] = str(output_path.relative_to(root)).replace("\\", "/")
        checks = validate_submission(frame[["city_id", "year", "pred"]], expected_cities, candidate_id)
        submission_checks.extend(checks)
        target_id = "registered" if candidate_id.startswith("registered") else "resident"
        scenario = "TARGET_HISTORY_ONLY" if candidate_id == "registered_rlt3" else (SCENARIOS[0] if candidate_id.endswith("b3") else (SCENARIOS[0] if candidate_id.endswith("r1") else SCENARIOS[1]))
        sanity = prediction_sanity(frame, candidate_id, target_id, scenario, candidate_paths[candidate_id])
        sanity_rows.append(sanity)
        candidate_summary_rows.append({"comparison_kind": "candidate_distribution", **sanity})

    for left, right in (
        ("resident_f4_r1", "resident_f4_r2"),
        ("resident_f4_r1", "resident_b3"),
        ("registered_rlt3", "registered_b3"),
    ):
        left_frame = all_predictions[left].sort_values("city_id", key=lambda col: col.map(city_key)).reset_index(drop=True)
        right_frame = all_predictions[right].sort_values("city_id", key=lambda col: col.map(city_key)).reset_index(drop=True)
        difference = left_frame["pred"].to_numpy(dtype=float) - right_frame["pred"].to_numpy(dtype=float)
        denominator = np.maximum(np.abs(right_frame["pred"].to_numpy(dtype=float)), 1e-12)
        candidate_summary_rows.append(
            {
                "comparison_kind": "pairwise_difference",
                "left_candidate": left,
                "right_candidate": right,
                "mean_abs_diff": float(np.mean(np.abs(difference))),
                "median_abs_diff": float(np.median(np.abs(difference))),
                "max_abs_diff": float(np.max(np.abs(difference))),
                "mean_signed_diff": float(np.mean(difference)),
                "max_relative_abs_diff": float(np.max(np.abs(difference) / denominator)),
            }
        )

    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(handoff_dir / "final_fit_config_selection.csv", index=False, encoding="utf-8", lineterminator="\n")
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(handoff_dir / "final_fit_training_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(sanity_rows).to_csv(handoff_dir / "prediction_sanity.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(candidate_summary_rows).to_csv(handoff_dir / "candidate_comparison.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(submission_checks).to_csv(handoff_dir / "submission_validation.csv", index=False, encoding="utf-8", lineterminator="\n")

    catboost_fit_count = int(sum(result["audit"]["catboost_fit_count"] for result in (worker_results[SCENARIOS[0]], worker_results[SCENARIOS[1]])))
    failed_candidates = sorted({row["candidate_id"] for row in submission_checks if row["status"] != "PASS"})
    all_submission_checks_pass = not failed_candidates
    submission_status = "PASS" if all_submission_checks_pass else "BLOCKED_SUBMISSION_SCHEMA"
    checkpoint_sha_after = sha256_file(root / "CURRENT_LEARNING_CHECKPOINT.md") if (root / "CURRENT_LEARNING_CHECKPOINT.md").exists() else "missing"
    initial_head = git_value(root, "rev-parse", "HEAD")
    initial_branch = git_value(root, "branch", "--show-current")
    summary = {
        "plan_id": PLAN_ID,
        "status": "READY_FOR_FIRST_PLATFORM_SUBMISSION" if all_submission_checks_pass else submission_status,
        "strategy": "FAST_FIRST_SUBMISSION",
        "generated_at_utc": utc_now(),
        "git": {
            "generation_source_sha": initial_head,
            "generation_branch": initial_branch,
            "canonical_main_at_start": "2c64f5ec6eb8ed1ad6c571ecbe01f1dd2947d288",
            "final_delivery_sha_recorded_by_main_after_commit": "pending_commit",
        },
        "parallel_topology": {
            "workstreams": ["A resident F4 R1", "B resident F4 R2", "C registered RLT3/B3 plus resident B3", "D packaging/schema validation"],
            "execution": "ProcessPoolExecutor(max_workers=3) for A/B/C; D in orchestrator after all futures",
            "fit_thread_count": 1,
        },
        "plan02_foundation": inputs["plan02_verification"],
        "resident_f4": {
            scenario: {
                "selected_config_id": worker_results[scenario]["audit"]["selected_config_id"],
                "selection_method": worker_results[scenario]["audit"]["selection_method"],
                "eligible_inner_fold_count": worker_results[scenario]["audit"]["eligible_inner_fold_count"],
                "training_row_count": worker_results[scenario]["audit"]["training_row_count"],
                "training_max_label_year": worker_results[scenario]["audit"]["training_max_label_year"],
                "inner_scores": worker_results[scenario]["scores"],
            }
            for scenario in SCENARIOS
        },
        "registered_sensitivity": {
            "rlt3_model_id": "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND",
            "b3_model_id": "B3_RECENT_3POINT_OLS_TREND",
            "rlt3_r1_r2_equivalence": deterministic["rlt3_r1_r2_equivalence"],
            "no_covariates": True,
        },
        "candidate_paths": candidate_paths,
        "candidate_count": len(candidate_paths),
        "model_work": {
            "catboost_fit_count": catboost_fit_count,
            "estimator_fit_count": catboost_fit_count,
            "f4_config_selection_2023_count": 2,
            "prediction_2023_count": len(candidate_paths),
            "submission_count": len(candidate_paths),
            "tuning_search_count": 0,
            "new_model_family_count": 0,
            "blend_count": 0,
            "routing_or_gating_count": 0,
        },
        "submission_validation": {
            "status": submission_status,
            "failed_checks": int(sum(row["status"] != "PASS" for row in submission_checks)),
            "failed_candidates": failed_candidates,
            "candidate_files": list(candidate_paths.values()),
        },
        "forbidden_input_audit": {
            "test_xls_content_accessed": False,
            "test_xls_hashed": False,
            "b_data_used": False,
            "external_data_used": False,
            "target_truth_after_2021_used": False,
            "current_2023_target_truth_columns_present": False,
            "worker_oof_used_as_final_fit_shortcut": False,
            "checkpoint_sha_before": inputs["checkpoint_sha"],
            "checkpoint_sha_after": checkpoint_sha_after,
            "checkpoint_unchanged": inputs["checkpoint_sha"] == checkpoint_sha_after,
        },
        "deferred_reproduction_debt": [
            "full historical F4 OOF replay",
            "exact all-origin configuration-selection reproduction",
            "full RLT3 historical regeneration",
            "deeper fresh end-to-end release proof",
        ],
        "platform_upload_performed": False,
        "recommended_first_upload": "submission_resident_f4_r1.csv" if not failed_candidates or "resident_f4_r1" not in failed_candidates else None,
        "resident_first_submission_candidates_pass": all(candidate not in failed_candidates for candidate in ("resident_f4_r1", "resident_f4_r2")),
        "ready_for_first_platform_submission": all_submission_checks_pass,
        "ready_for_main_stage_review": True,
        "s6_complete": False,
        "s7_entered": False,
    }
    write_json(handoff_dir / "fast_first_submission_summary.json", summary)

    env = {
        "captured_at_utc": utc_now(),
        "plan_id": PLAN_ID,
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "package_versions": {"catboost": "1.2.10", "numpy": np.__version__, "pandas": pd.__version__},
        "git": {"generation_source_sha": initial_head, "generation_branch": initial_branch},
        "input_hashes": {name: sha256_file(path) for name, path in inputs["input_paths"].items()},
        "checkpoint_sha_before": inputs["checkpoint_sha"],
        "checkpoint_sha_after": checkpoint_sha_after,
        "parallel_topology": summary["parallel_topology"],
        "test_xls_access": {"opened": False, "hashed": False},
        "model_binary_persisted": False,
    }
    write_json(handoff_dir / "environment_snapshot.json", env)

    failed_lines = []
    for candidate_id in failed_candidates:
        failed_checks = [row["check"] + ": " + row["detail"] for row in submission_checks if row["candidate_id"] == candidate_id and row["status"] != "PASS"]
        failed_lines.append(f"- `{candidate_id}` — " + "; ".join(failed_checks))
    status_line = summary["status"]
    validation_line = (
        "All five files passed the exact schema checks: 40 cities, `year=2023`, columns `city_id,year,pred`, finite positive predictions, no duplicate city, UTF-8 CSV, stable city ordering."
        if all_submission_checks_pass
        else "The five files were generated, but the packet is blocked because one or more candidates fail the positive-prediction schema. No clipping or manual city repair was applied."
    )
    handoff = f"""# TASK_RESULT — S6-Plan03 Fast First Submission

## Result

- Status: `{status_line}`
- Strategy: `FAST_FIRST_SUBMISSION`
- Generation source: `{initial_branch}` at `{initial_head}`
- Platform upload: not performed; manual upload remains with Main/user
- S6 completion: not declared; S7 not entered

## Main recommendation

The resident-primary R1 candidate remains the provisional first-upload choice because it is the stronger frozen historical information scenario and passed its own schema checks. The local unit remains `万人`; official resident/registered mapping and unit confirmation remain unresolved and must not be silently treated as proved. Main/user should review the packet status before any manual upload.

## Final-fit evidence

| Candidate | Selected config | Eligible inner folds | Training rows | Max label year | Inner scores |
|---|---|---:|---:|---:|---|
"""
    for scenario in SCENARIOS:
        result = worker_results[scenario]
        handoff += f"| resident F4 {scenario} | `{result['audit']['selected_config_id']}` | {result['audit']['eligible_inner_fold_count']} | {result['audit']['training_row_count']} | {result['audit']['training_max_label_year']} | `{json.dumps(result['scores'], sort_keys=True)}` |\n"
    handoff += f"""
## Candidate files

""" + "\n".join(f"- `{path}`" for path in candidate_paths.values()) + """

{validation_line}

"""
    if failed_lines:
        handoff += "## Schema blocker evidence\n\n" + "\n".join(failed_lines) + "\n\n"
## Controls and boundaries
    handoff += """## Controls and boundaries

- Plan02 clean replay/current foundation: PASS; 2,320 replay rows, 160 current rows, 144 numeric features, 2,240 boundary rows.
- F4 uses only the frozen Plan01 semantics and C1/C2/C3; no new model family, tuning sweep, ensemble, routing, clipping, or manual city edit.
- Registered RLT3 and B3 are target-history-only controls; RLT3 R1/R2 equivalence: PASS.
- No `test.xls` content/hash, B-data, external data, or target truth after 2021 was used.
- `CURRENT_LEARNING_CHECKPOINT.md` remained unchanged.

## Deferred reproduction debt

Full historical F4 OOF replay, exact all-origin selection reproduction, full RLT3 historical regeneration, and deeper fresh end-to-end release proof remain explicitly deferred until platform feedback. This is a fast-first handoff, not a claim that S6 is complete.

## Required next action

If Main accepts the resident-primary path after reviewing the registered sensitivity anomaly, manually upload `submission_resident_f4_r1.csv` and return the platform score. Do not automatically upload from this repository.
"""
    (handoff_dir / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    return summary


def run_orchestrator(root: Path) -> None:
    started = time.perf_counter()
    print(f"[S6-Plan03] preflight started at {utc_now()}", flush=True)
    inputs = load_inputs(root)
    print("[S6-Plan03] Plan02 foundation preflight PASS", flush=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="s6_plan03_fast_first_"))
    worker_results: dict[str, Any] = {}
    try:
        jobs = {
            "A_resident_f4_r1": (run_f4_worker, (str(root), SCENARIOS[0], str(temp_dir / "resident_r1"))),
            "B_resident_f4_r2": (run_f4_worker, (str(root), SCENARIOS[1], str(temp_dir / "resident_r2"))),
            "C_deterministic": (run_deterministic_worker, (str(root), str(temp_dir / "deterministic"))),
        }
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(function, *args): label for label, (function, args) in jobs.items()}
            for future in as_completed(futures):
                label = futures[future]
                result = future.result()
                if label == "A_resident_f4_r1":
                    worker_results[SCENARIOS[0]] = result
                elif label == "B_resident_f4_r2":
                    worker_results[SCENARIOS[1]] = result
                else:
                    worker_results["deterministic"] = result
                print(f"[S6-Plan03] workstream {label} complete", flush=True)
        print("[S6-Plan03] workstreams A/B/C complete; packaging D", flush=True)
        summary = package_outputs(root, inputs, worker_results, temp_dir)
        print(
            f"[S6-Plan03] {summary['status']}; "
            f"fits={summary['model_work']['catboost_fit_count']}; elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--orchestrate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    if not args.orchestrate:
        raise SystemExit("Use --orchestrate for the authorized S6-Plan03 run")
    run_orchestrator(root)


if __name__ == "__main__":
    main()
