"""Run the frozen S5-Plan06C XGBoost screening worker.

This worker consumes the Plan04/Plan04B artifacts without rebuilding or
changing their feature, replay, or temporal geometry contracts.  It evaluates
only the three pre-registered XGBoost configurations, selects configurations
with the frozen inner temporal folds, fits the selected configuration on each
legal outer training pool, and emits the worker-local handoff artifacts.

The runner intentionally does not create a 2023 forecast, submission, city
router, gate, blend, or ensemble.  Numeric NaN values remain NaN for native
XGBoost missing-value handling; city_id is represented by a fixed city1..city40
one-hot block.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan06C"
BRANCH = "s5/plan06c-xgboost-screening"
BASE_SHA = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
FEATURE_CONTRACT_VERSION = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT_VERSION = "S5-Plan04B-TRAINING-REPLAY-V1"
SEED = 20260919
HORIZON_YEARS = 2

OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
FORMAL_MATRIX_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv"
FORMAL_SCHEMA_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
OOF_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "oof_prediction_contract.csv"
REPLAY_MATRIX_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv"
OUTER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "outer_training_geometry.csv"
INNER_GEOMETRY_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "inner_split_geometry.csv"
WORKER_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "strong_model_wave_contract.json"
PARALLEL_INPUT_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json"

TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
FRAMINGS = ("F1_DIRECT", "F2_B3_RESIDUAL")
ORIGINS = tuple(range(2011, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))
CITY_INDEX = {city: index for index, city in enumerate(CITIES)}
MODEL_ID = "S5-Plan06C_XGB_SCREENING"

B3_REFERENCE_POOLED_MSE = {
    "resident": 872.452651515143,
    "registered": 225.421212121213,
}

CONFIGS: list[dict[str, Any]] = [
    {
        "config_name": "XGB_C1_CONSERVATIVE",
        "order": 1,
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 10.0,
            "subsample": 0.90,
            "colsample_bytree": 0.70,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "gamma": 0.0,
        },
    },
    {
        "config_name": "XGB_C2_MEDIUM",
        "order": 2,
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 5.0,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "gamma": 0.0,
        },
    },
    {
        "config_name": "XGB_C3_FLEXIBLE_REGULARIZED",
        "order": 3,
        "params": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3.0,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.25,
            "reg_lambda": 3.0,
            "gamma": 0.1,
        },
    },
]
CONFIG_BY_NAME = {config["config_name"]: config for config in CONFIGS}
DESIGN_FEATURES: tuple[str, ...] = ()
FEATURE_GROUP_BY_NAME: dict[str, str] = {}

MISSING_TEXT = {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "NULL"}
NUMERIC_TOLERANCE = 1e-12


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return stable_json(json_safe(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else round(value, 12)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    assert fieldnames, path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fieldnames})


def read_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, list[int]]]:
    """Read CSV while preserving the first value of duplicate headers.

    Plan04's formal matrix intentionally contains audit columns whose names
    overlap model-feature names.  The writer emits identical values for those
    duplicate columns.  We verify that invariant instead of allowing a normal
    DictReader to silently overwrite one value.
    """

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - contract failure
            raise AssertionError(f"empty CSV: {path}") from exc
        positions: dict[str, list[int]] = defaultdict(list)
        for index, name in enumerate(header):
            positions[name].append(index)
        rows: list[dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(header):
                raise AssertionError(
                    f"CSV width mismatch: {path}:{line_number} expected {len(header)} got {len(values)}"
                )
            row: dict[str, str] = {}
            for name, indexes in positions.items():
                selected = values[indexes[0]]
                if len(indexes) > 1 and any(values[index] != selected for index in indexes[1:]):
                    raise AssertionError(f"duplicate CSV headers disagree: {path}:{line_number}:{name}")
                row[name] = selected
            rows.append(row)
    return header, rows, dict(positions)


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() in MISSING_TEXT:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    return None if number is None else int(number)


def require_fields(header: list[str], required: Iterable[str], label: str) -> None:
    missing = [field for field in required if field not in header]
    assert not missing, f"{label} missing fields: {missing}"


def close_enough(left: float, right: float) -> bool:
    return abs(left - right) <= NUMERIC_TOLERANCE or abs(left - right) <= NUMERIC_TOLERANCE * max(
        1.0, abs(left), abs(right)
    )


def compare_metrics(left: float, right: float) -> int:
    """Return -1 if left wins, 1 if right wins, 0 for the frozen tie band."""

    tolerance = NUMERIC_TOLERANCE * max(1.0, abs(left), abs(right))
    if left < right - tolerance:
        return -1
    if right < left - tolerance:
        return 1
    return 0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    assert values
    return math.fsum(values) / len(values)


def mse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    return mean((float(prediction) - float(target)) ** 2 for target, prediction in zip(actual, predicted))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    return math.sqrt(mse(actual, predicted))


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    return mean(abs(float(prediction) - float(target)) for target, prediction in zip(actual, predicted))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return math.fsum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def assert_worker_identity() -> str:
    assert git_text("branch", "--show-current") == BRANCH
    assert git_text("merge-base", BASE_SHA, "HEAD") == BASE_SHA
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "exact Plan05 wave base must be an ancestor of worker tip"
    return git_text("rev-parse", "HEAD")


def load_contracts() -> dict[str, Any]:
    schema = json.loads(FORMAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    wave = json.loads(WORKER_CONTRACT_PATH.read_text(encoding="utf-8"))
    parallel = json.loads(PARALLEL_INPUT_CONTRACT_PATH.read_text(encoding="utf-8"))
    oof_header, _, _ = read_csv_table(OOF_CONTRACT_PATH)

    assert schema["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert len(schema["model_features"]) == 144
    assert schema["categorical_features"] == ["city_id"]
    assert wave["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert wave["replay_contract_version"] == REPLAY_CONTRACT_VERSION
    assert wave["data_contract"]["formal_validation_row_count"] == 1760
    assert wave["data_contract"]["replay_row_count"] == 2320
    assert wave["data_contract"]["model_feature_count"] == 144
    assert parallel["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    # The parallel-worker contract names its replay version ``contract_version``;
    # the wave contract uses the more explicit ``replay_contract_version``.
    assert parallel["contract_version"] == REPLAY_CONTRACT_VERSION
    assert oof_header == [
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
    ]
    return {"schema": schema, "wave": wave, "parallel": parallel, "oof_fields": oof_header}


def validate_rows(
    formal_header: list[str],
    formal_rows: list[dict[str, str]],
    replay_header: list[str],
    replay_rows: list[dict[str, str]],
    schema: dict[str, Any],
) -> None:
    model_features = tuple(schema["model_features"])
    common_required = (
        "target_id",
        "scenario_id",
        "city_id",
        "actual_target_value",
        "b3_prediction",
        "b3_residual_actual_minus_prediction",
        "b3_squared_error",
        "feature_contract_version",
        *model_features,
    )
    require_fields(formal_header, (*common_required, "origin_year"), "formal matrix")
    require_fields(replay_header, (*common_required, "pseudo_target_year"), "replay matrix")
    assert len(formal_rows) == 1760, len(formal_rows)
    assert len(replay_rows) == 2320, len(replay_rows)

    formal_keys = set()
    for row in formal_rows:
        key = (row["target_id"], row["scenario_id"], row["origin_year"], row["city_id"])
        assert key not in formal_keys, key
        formal_keys.add(key)
        assert row["target_id"] in TARGETS
        assert row["scenario_id"] in SCENARIOS
        assert parse_int(row["origin_year"]) in ORIGINS
        assert row["city_id"] in CITY_INDEX
        assert row["feature_contract_version"] == FEATURE_CONTRACT_VERSION
        assert parse_float(row["actual_target_value"]) is not None
        assert parse_float(row["b3_prediction"]) is not None
        assert parse_float(row["b3_squared_error"]) is not None

    replay_keys = set()
    for row in replay_rows:
        key = (row["target_id"], row["scenario_id"], row["pseudo_target_year"], row["city_id"])
        assert key not in replay_keys, key
        replay_keys.add(key)
        assert row["target_id"] in TARGETS
        assert row["scenario_id"] in SCENARIOS
        assert 2004 <= parse_int(row["pseudo_target_year"]) <= 2021
        assert row["city_id"] in CITY_INDEX
        assert row["feature_contract_version"] == FEATURE_CONTRACT_VERSION
        assert parse_float(row["actual_target_value"]) is not None
        assert parse_float(row["b3_prediction"]) is not None
        assert parse_float(row["b3_squared_error"]) is not None

    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            formal_scoped = [
                row
                for row in formal_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            assert len(formal_scoped) == 440
            for origin_year in ORIGINS:
                origin_rows = [row for row in formal_scoped if parse_int(row["origin_year"]) == origin_year]
                assert len(origin_rows) == 40
                assert {row["city_id"] for row in origin_rows} == set(CITIES)
            replay_scoped = [
                row
                for row in replay_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            assert len(replay_scoped) in (573, 587), len(replay_scoped)
            b3_mse = mean(float(row["b3_squared_error"]) for row in formal_scoped)
            assert close_enough(b3_mse, B3_REFERENCE_POOLED_MSE[target_id]), (target_id, scenario_id, b3_mse)


def eligible_fold(row: dict[str, str]) -> bool:
    return (
        row["boundary_pass"] == "PASS"
        and int(row["inner_training_row_count"]) >= 40
        and int(row["inner_validation_row_count"]) >= 20
    )


def load_geometry(
    formal_rows: list[dict[str, str]], replay_rows: list[dict[str, str]]
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[tuple[str, str, int], list[dict[str, str]]]]:
    outer_header, outer_rows, _ = read_csv_table(OUTER_GEOMETRY_PATH)
    inner_header, inner_rows, _ = read_csv_table(INNER_GEOMETRY_PATH)
    require_fields(
        outer_header,
        (
            "target_id",
            "scenario_id",
            "outer_origin_year",
            "training_max_pseudo_target_year",
            "training_min_pseudo_target_year",
            "training_row_count",
            "training_city_count",
            "training_year_count",
            "validation_row_count",
            "validation_city_count",
            "forbidden_training_rows_after_cutoff",
            "boundary_pass",
        ),
        "outer geometry",
    )
    require_fields(
        inner_header,
        (
            "target_id",
            "scenario_id",
            "outer_origin_year",
            "inner_validation_year",
            "inner_training_max_year",
            "inner_training_min_year",
            "inner_training_row_count",
            "inner_validation_row_count",
            "inner_training_city_count",
            "inner_validation_city_count",
            "inner_training_year_count",
            "outer_training_pool_row_count",
            "inner_training_rows_after_cutoff",
            "boundary_pass",
        ),
        "inner geometry",
    )
    assert len(outer_rows) == 44
    assert len(inner_rows) == 484

    outer_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in outer_rows:
        key = (row["target_id"], row["scenario_id"], int(row["outer_origin_year"]))
        assert key not in outer_by_key
        assert row["boundary_pass"] == "PASS"
        target_id, scenario_id, origin_year = key
        training = [
            item
            for item in replay_rows
            if item["target_id"] == target_id
            and item["scenario_id"] == scenario_id
            and int(item["pseudo_target_year"]) <= origin_year - HORIZON_YEARS
        ]
        validation = [
            item
            for item in formal_rows
            if item["target_id"] == target_id
            and item["scenario_id"] == scenario_id
            and int(item["origin_year"]) == origin_year
        ]
        assert int(row["training_row_count"]) == len(training)
        assert int(row["training_city_count"]) == len({item["city_id"] for item in training})
        assert int(row["training_year_count"]) == len({item["pseudo_target_year"] for item in training})
        assert int(row["validation_row_count"]) == len(validation) == 40
        assert int(row["validation_city_count"]) == len({item["city_id"] for item in validation}) == 40
        assert int(row["forbidden_training_rows_after_cutoff"]) == 0
        outer_by_key[key] = row

    inner_by_key: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in inner_rows:
        key = (row["target_id"], row["scenario_id"], int(row["outer_origin_year"]))
        assert key in outer_by_key
        target_id, scenario_id, outer_origin_year = key
        inner_validation_year = int(row["inner_validation_year"])
        outer_training = [
            item
            for item in replay_rows
            if item["target_id"] == target_id
            and item["scenario_id"] == scenario_id
            and int(item["pseudo_target_year"]) <= outer_origin_year - HORIZON_YEARS
        ]
        inner_training_max = int(row["inner_training_max_year"])
        inner_training = [
            item for item in outer_training if int(item["pseudo_target_year"]) <= inner_training_max
        ]
        inner_validation = [
            item for item in outer_training if int(item["pseudo_target_year"]) == inner_validation_year
        ]
        assert int(row["inner_training_row_count"]) == len(inner_training)
        assert int(row["inner_validation_row_count"]) == len(inner_validation)
        assert int(row["inner_training_city_count"]) == len({item["city_id"] for item in inner_training})
        assert int(row["inner_validation_city_count"]) == len({item["city_id"] for item in inner_validation})
        assert int(row["outer_training_pool_row_count"]) == len(outer_training)
        assert int(row["inner_training_rows_after_cutoff"]) == 0
        assert row["boundary_pass"] == "PASS"
        inner_by_key[key].append(row)

    eligible_count = sum(eligible_fold(row) for rows in inner_by_key.values() for row in rows)
    fallback_group_count = sum(
        sum(eligible_fold(row) for row in rows) < 3 for rows in inner_by_key.values()
    )
    assert eligible_count == 264
    assert fallback_group_count == 8
    assert all(len(rows) > 0 for rows in inner_by_key.values())
    return outer_by_key, dict(inner_by_key)


def design_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.vstack([row["_x"] for row in rows]).astype(np.float64, copy=False)
    assert matrix.shape[1] == len(DESIGN_FEATURES)
    assert not np.isinf(matrix).any()
    return matrix


def attach_design_vectors(rows: list[dict[str, Any]], model_features: tuple[str, ...]) -> None:
    global DESIGN_FEATURES
    DESIGN_FEATURES = (*model_features, *(f"city_onehot_{city}" for city in CITIES))
    for row in rows:
        # Numeric model features retain raw NaN; one-hot categories are an
        # ordinary dense indicator block and therefore must use explicit 0s.
        vector = np.zeros(len(DESIGN_FEATURES), dtype=np.float64)
        vector[: len(model_features)] = np.nan
        for index, feature_name in enumerate(model_features):
            value = parse_float(row[feature_name])
            if value is not None:
                vector[index] = value
        vector[len(model_features) + CITY_INDEX[row["city_id"]]] = 1.0
        row["_x"] = vector


def make_model(config: dict[str, Any]) -> XGBRegressor:
    params = dict(config["params"])
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
        **params,
    )


def framing_label(row: dict[str, Any], framing_id: str) -> float:
    if framing_id == "F1_DIRECT":
        value = parse_float(row["actual_target_value"])
    else:
        value = parse_float(row["b3_residual_actual_minus_prediction"])
    assert value is not None
    return value


def actual_value(row: dict[str, Any]) -> float:
    value = parse_float(row["actual_target_value"])
    assert value is not None
    return value


def b3_value(row: dict[str, Any]) -> float:
    value = parse_float(row["b3_prediction"])
    assert value is not None
    return value


def fit_and_predict(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    framing_id: str,
    fit_counter: Counter[str],
) -> tuple[XGBRegressor, np.ndarray, float]:
    started = time.perf_counter()
    model = make_model(config)
    x_train = design_matrix(train_rows)
    y_train = np.asarray([framing_label(row, framing_id) for row in train_rows], dtype=np.float64)
    assert np.isfinite(y_train).all()
    model.fit(x_train, y_train, verbose=False)
    raw_prediction = np.asarray(model.predict(design_matrix(validation_rows)), dtype=np.float64)
    assert np.isfinite(raw_prediction).all()
    if framing_id == "F2_B3_RESIDUAL":
        prediction = raw_prediction + np.asarray([b3_value(row) for row in validation_rows], dtype=np.float64)
    else:
        prediction = raw_prediction
    assert np.isfinite(prediction).all()
    fit_counter["total_fit_count"] += 1
    fit_counter["inner_fit_count"] += 1
    fit_counter["fit_runtime_seconds"] += time.perf_counter() - started
    return model, prediction, time.perf_counter() - started


def gain_by_feature(model: XGBRegressor) -> np.ndarray:
    gains = np.zeros(len(DESIGN_FEATURES), dtype=np.float64)
    score = model.get_booster().get_score(importance_type="gain")
    feature_index = {name: index for index, name in enumerate(DESIGN_FEATURES)}
    for key, value in score.items():
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
        else:
            assert key in feature_index, key
            index = feature_index[key]
        if 0 <= index < len(gains):
            gains[index] = float(value)
    assert np.isfinite(gains).all()
    return gains


def collect_missingness(
    rows_by_scope: list[tuple[str, str, str, str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    numeric_count = len(DESIGN_FEATURES) - len(CITIES)
    for matrix_id, target_id, scenario_id, origin_label, rows in rows_by_scope:
        for index, feature_name in enumerate(DESIGN_FEATURES[:numeric_count]):
            missing_count = sum(np.isnan(row["_x"][index]) for row in rows)
            result.append(
                {
                    "matrix_id": matrix_id,
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "origin_year": origin_label,
                    "feature_name": feature_name,
                    "feature_group": FEATURE_GROUP_BY_NAME.get(feature_name, "UNKNOWN"),
                    "row_count": len(rows),
                    "missing_count": int(missing_count),
                    "missing_rate": missing_count / len(rows) if rows else None,
                    "handling": "native_xgboost_missing_nan",
                    "numeric_imputer": "none",
                    "scaler": "none",
                }
            )
        city_missing = sum(np.isnan(row["_x"][numeric_count:]).any() for row in rows)
        result.append(
            {
                "matrix_id": matrix_id,
                "target_id": target_id,
                "scenario_id": scenario_id,
                "origin_year": origin_label,
                "feature_name": "city_onehot_block",
                "feature_group": "CITY_ONE_HOT",
                "row_count": len(rows),
                "missing_count": int(city_missing),
                "missing_rate": city_missing / len(rows) if rows else None,
                "handling": "fixed_city1_to_city40_one_hot",
                "numeric_imputer": "not_applicable",
                "scaler": "none",
            }
        )
    return result


def aggregate_gain(gain_records: list[np.ndarray]) -> list[dict[str, Any]]:
    assert gain_records
    gain_sum = np.sum(np.vstack(gain_records), axis=0)
    normalized_per_fit = np.vstack(
        [record / record.sum() if record.sum() > 0 else np.zeros_like(record) for record in gain_records]
    )
    mean_normalized = normalized_per_fit.mean(axis=0)
    nonzero = np.sum(np.vstack([record > 0 for record in gain_records]), axis=0)
    rows: list[dict[str, Any]] = []
    feature_order = sorted(range(len(DESIGN_FEATURES)), key=lambda i: (-mean_normalized[i], i))
    rank_by_index = {index: rank for rank, index in enumerate(feature_order, start=1)}
    for index, feature_name in enumerate(DESIGN_FEATURES):
        group_id = FEATURE_GROUP_BY_NAME.get(feature_name, "CITY_ONE_HOT")
        rows.append(
            {
                "record_type": "feature",
                "feature_name": feature_name,
                "feature_group": group_id,
                "is_city_onehot": "yes" if feature_name.startswith("city_onehot_") else "no",
                "raw_gain_sum": float(gain_sum[index]),
                "mean_normalized_gain": float(mean_normalized[index]),
                "selected_fit_count": len(gain_records),
                "nonzero_fit_count": int(nonzero[index]),
                "rank": rank_by_index[index],
            }
        )

    group_gain: dict[str, float] = defaultdict(float)
    group_normalized: dict[str, float] = defaultdict(float)
    group_nonzero: Counter[str] = Counter()
    for index, feature_name in enumerate(DESIGN_FEATURES):
        group_id = FEATURE_GROUP_BY_NAME.get(feature_name, "CITY_ONE_HOT")
        group_gain[group_id] += float(gain_sum[index])
        group_normalized[group_id] += float(mean_normalized[index])
        group_nonzero[group_id] += int(nonzero[index])
    group_order = sorted(group_gain, key=lambda group: (-group_normalized[group], group))
    for rank, group_id in enumerate(group_order, start=1):
        rows.append(
            {
                "record_type": "feature_group",
                "feature_name": "",
                "feature_group": group_id,
                "is_city_onehot": "yes" if group_id == "CITY_ONE_HOT" else "no",
                "raw_gain_sum": group_gain[group_id],
                "mean_normalized_gain": group_normalized[group_id],
                "selected_fit_count": len(gain_records),
                "nonzero_fit_count": group_nonzero[group_id],
                "rank": rank,
            }
        )
    return rows


def build_metrics(
    oof_rows: list[dict[str, Any]], formal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    per_origin_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []

    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            formal_scope = [
                row
                for row in formal_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            b3_by_origin = {
                origin_year: [row for row in formal_scope if int(row["origin_year"]) == origin_year]
                for origin_year in ORIGINS
            }
            b3_by_city = {
                city: [row for row in formal_scope if row["city_id"] == city]
                for city in CITIES
            }
            for framing_id in FRAMINGS:
                scope = [
                    row
                    for row in oof_rows
                    if row["target_id"] == target_id
                    and row["scenario_id"] == scenario_id
                    and row["framing_id"] == framing_id
                ]
                assert len(scope) == 440
                actual = [float(row["actual_target_value"]) for row in scope]
                prediction = [float(row["prediction_value"]) for row in scope]
                per_origin: list[dict[str, Any]] = []
                for origin_year in ORIGINS:
                    model_origin = [row for row in scope if int(row["origin_year"]) == origin_year]
                    b3_origin = b3_by_origin[origin_year]
                    model_mse = mean(float(row["squared_error"]) for row in model_origin)
                    b3_mse = mean(
                        (b3_value(row) - actual_value(row)) ** 2
                        for row in b3_origin
                    )
                    comparison = compare_metrics(model_mse, b3_mse)
                    per_origin.append(
                        {
                            "model_id": MODEL_ID,
                            "framing_id": framing_id,
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "origin_year": origin_year,
                            "row_count": len(model_origin),
                            "pooled_city_year_mse": model_mse,
                            "b3_mse": b3_mse,
                            "delta_mse_vs_b3": model_mse - b3_mse,
                            "rmse": math.sqrt(model_mse),
                            "mae": mean(abs(float(row["residual_prediction_minus_actual"])) for row in model_origin),
                            "bias": mean(float(row["residual_prediction_minus_actual"]) for row in model_origin),
                            "origin_result_vs_b3": "win" if comparison < 0 else "loss" if comparison > 0 else "tie",
                            "selected_config": ";".join(sorted({row["selected_config"] for row in model_origin})),
                        }
                    )
                per_origin_rows.extend(per_origin)

                city_summaries: list[dict[str, Any]] = []
                for city in CITIES:
                    model_city = [row for row in scope if row["city_id"] == city]
                    b3_city = b3_by_city[city]
                    model_sse = math.fsum(float(row["squared_error"]) for row in model_city)
                    b3_sse = math.fsum((b3_value(row) - actual_value(row)) ** 2 for row in b3_city)
                    city_summaries.append(
                        {
                            "model_id": MODEL_ID,
                            "framing_id": framing_id,
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "city_id": city,
                            "row_count": len(model_city),
                            "model_sse": model_sse,
                            "model_mse": model_sse / len(model_city),
                            "model_mae": mean(abs(float(row["residual_prediction_minus_actual"])) for row in model_city),
                            "model_bias": mean(float(row["residual_prediction_minus_actual"]) for row in model_city),
                            "b3_sse": b3_sse,
                            "b3_mse": b3_sse / len(b3_city),
                            "delta_sse_vs_b3": model_sse - b3_sse,
                        }
                    )
                total_city_sse = math.fsum(row["model_sse"] for row in city_summaries)
                ordered_cities = sorted(city_summaries, key=lambda row: (-row["model_sse"], row["city_id"]))
                top1_sse = ordered_cities[0]["model_sse"]
                top3_sse = math.fsum(row["model_sse"] for row in ordered_cities[:3])
                for rank, row in enumerate(ordered_cities, start=1):
                    row["city_sse_rank"] = rank
                    row["city_sse_share"] = row["model_sse"] / total_city_sse if total_city_sse else None
                    row["top_1_flag"] = "yes" if rank == 1 else "no"
                    row["top_3_flag"] = "yes" if rank <= 3 else "no"
                city_rows.extend(city_summaries)

                latest = [row for row in scope if int(row["origin_year"]) == 2021]
                b3_latest = b3_by_origin[2021]
                pooled = mean(float(row["squared_error"]) for row in scope)
                b3_pooled = mean((b3_value(row) - actual_value(row)) ** 2 for row in formal_scope)
                win_count = sum(row["origin_result_vs_b3"] == "win" for row in per_origin)
                loss_count = sum(row["origin_result_vs_b3"] == "loss" for row in per_origin)
                tie_count = sum(row["origin_result_vs_b3"] == "tie" for row in per_origin)
                aggregate_rows.append(
                    {
                        "model_id": MODEL_ID,
                        "framing_id": framing_id,
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "row_count": len(scope),
                        "pooled_city_year_mse": pooled,
                        "macro_origin_mse": mean(row["pooled_city_year_mse"] for row in per_origin),
                        "latest_2021_mse": mean(float(row["squared_error"]) for row in latest),
                        "rmse": math.sqrt(pooled),
                        "mae": mean(abs(float(row["residual_prediction_minus_actual"])) for row in scope),
                        "bias": mean(float(row["residual_prediction_minus_actual"]) for row in scope),
                        "origin_win_count_vs_b3": win_count,
                        "origin_loss_count_vs_b3": loss_count,
                        "origin_tie_count_vs_b3": tie_count,
                        "b3_pooled_city_year_mse": b3_pooled,
                        "b3_latest_2021_mse": mean((b3_value(row) - actual_value(row)) ** 2 for row in b3_latest),
                        "delta_pooled_mse_vs_b3": pooled - b3_pooled,
                        "delta_latest_2021_mse_vs_b3": mean(float(row["squared_error"]) for row in latest)
                        - mean((b3_value(row) - actual_value(row)) ** 2 for row in b3_latest),
                        "total_city_sse": total_city_sse,
                        "top_1_city_id": ordered_cities[0]["city_id"],
                        "top_1_city_sse": top1_sse,
                        "top_1_city_sse_concentration": top1_sse / total_city_sse if total_city_sse else None,
                        "top_3_city_ids": ";".join(row["city_id"] for row in ordered_cities[:3]),
                        "top_3_city_sse": top3_sse,
                        "top_3_city_sse_concentration": top3_sse / total_city_sse if total_city_sse else None,
                        "selected_config_distribution": stable_json(
                            dict(sorted(Counter(row["selected_config"] for row in scope).items()))
                        ),
                    }
                )

                for unit_scope in ("origin", "city"):
                    if unit_scope == "origin":
                        unit_keys = list(ORIGINS)
                        model_unit_values = {
                            key: [row for row in scope if int(row["origin_year"]) == key] for key in unit_keys
                        }
                        b3_unit_values = {key: b3_by_origin[key] for key in unit_keys}
                    else:
                        unit_keys = list(CITIES)
                        model_unit_values = {
                            key: [row for row in scope if row["city_id"] == key] for key in unit_keys
                        }
                        b3_unit_values = {key: b3_by_city[key] for key in unit_keys}
                    model_signed: list[float] = []
                    b3_signed: list[float] = []
                    model_abs: list[float] = []
                    b3_abs: list[float] = []
                    model_squared: list[float] = []
                    b3_squared: list[float] = []
                    for key in unit_keys:
                        model_unit = model_unit_values[key]
                        b3_unit = b3_unit_values[key]
                        model_errors = [float(row["residual_prediction_minus_actual"]) for row in model_unit]
                        b3_errors = [b3_value(row) - actual_value(row) for row in b3_unit]
                        model_signed.append(mean(model_errors))
                        b3_signed.append(mean(b3_errors))
                        model_abs.append(mean(abs(value) for value in model_errors))
                        b3_abs.append(mean(abs(value) for value in b3_errors))
                        model_squared.append(mean(value * value for value in model_errors))
                        b3_squared.append(mean(value * value for value in b3_errors))
                    comparisons = [compare_metrics(a, b) for a, b in zip(model_squared, b3_squared)]
                    complementarity_rows.append(
                        {
                            "model_id": MODEL_ID,
                            "framing_id": framing_id,
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "unit_scope": unit_scope,
                            "unit_count": len(unit_keys),
                            "signed_error_correlation": pearson(model_signed, b3_signed),
                            "absolute_error_correlation": pearson(model_abs, b3_abs),
                            "squared_error_correlation": pearson(model_squared, b3_squared),
                            "model_better_unit_count": comparisons.count(-1),
                            "b3_better_unit_count": comparisons.count(1),
                            "tie_unit_count": comparisons.count(0),
                            "mean_model_unit_squared_error": mean(model_squared),
                            "mean_b3_unit_squared_error": mean(b3_squared),
                        }
                    )
    return aggregate_rows, per_origin_rows, city_rows, complementarity_rows


def comparison_rows(aggregate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_rows: list[dict[str, Any]] = []
    framing_rows: list[dict[str, Any]] = []
    by_key = {(row["target_id"], row["framing_id"], row["scenario_id"]): row for row in aggregate_rows}
    for target_id in TARGETS:
        for framing_id in FRAMINGS:
            r1 = by_key[(target_id, framing_id, SCENARIOS[0])]
            r2 = by_key[(target_id, framing_id, SCENARIOS[1])]
            scenario_rows.append(
                {
                    "comparison": "R2_MINUS_R1",
                    "target_id": target_id,
                    "framing_id": framing_id,
                    "r1_pooled_mse": r1["pooled_city_year_mse"],
                    "r2_pooled_mse": r2["pooled_city_year_mse"],
                    "delta_pooled_mse": r2["pooled_city_year_mse"] - r1["pooled_city_year_mse"],
                    "r1_latest_2021_mse": r1["latest_2021_mse"],
                    "r2_latest_2021_mse": r2["latest_2021_mse"],
                    "delta_latest_2021_mse": r2["latest_2021_mse"] - r1["latest_2021_mse"],
                    "r1_origin_wins": r1["origin_win_count_vs_b3"],
                    "r2_origin_wins": r2["origin_win_count_vs_b3"],
                }
            )
        for scenario_id in SCENARIOS:
            direct = by_key[(target_id, "F1_DIRECT", scenario_id)]
            residual = by_key[(target_id, "F2_B3_RESIDUAL", scenario_id)]
            framing_rows.append(
                {
                    "comparison": "F2_RESIDUAL_MINUS_F1_DIRECT",
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "direct_pooled_mse": direct["pooled_city_year_mse"],
                    "residual_pooled_mse": residual["pooled_city_year_mse"],
                    "delta_pooled_mse": residual["pooled_city_year_mse"] - direct["pooled_city_year_mse"],
                    "direct_latest_2021_mse": direct["latest_2021_mse"],
                    "residual_latest_2021_mse": residual["latest_2021_mse"],
                    "delta_latest_2021_mse": residual["latest_2021_mse"] - direct["latest_2021_mse"],
                    "direct_origin_wins": direct["origin_win_count_vs_b3"],
                    "residual_origin_wins": residual["origin_win_count_vs_b3"],
                }
            )
    return scenario_rows, framing_rows


def classify_family(aggregate_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    by_key = {(row["target_id"], row["framing_id"], row["scenario_id"]): row for row in aggregate_rows}
    framing_checks: dict[str, Any] = {}
    signal_framings: list[str] = []
    competitive_framings: list[str] = []
    for framing_id in FRAMINGS:
        resident_rows = [by_key[("resident", framing_id, scenario_id)] for scenario_id in SCENARIOS]
        beats_b3_both = all(
            compare_metrics(row["pooled_city_year_mse"], row["b3_pooled_city_year_mse"]) < 0
            for row in resident_rows
        )
        majority_wins_both = all(row["origin_win_count_vs_b3"] >= 6 for row in resident_rows)
        within_one_percent = any(
            row["pooled_city_year_mse"] <= row["b3_pooled_city_year_mse"] * 1.01
            for row in resident_rows
        )
        stable_signal = beats_b3_both and majority_wins_both
        if stable_signal:
            signal_framings.append(framing_id)
        elif within_one_percent:
            competitive_framings.append(framing_id)
        framing_checks[framing_id] = {
            "resident_beats_b3_pooled_in_both_scenarios": beats_b3_both,
            "resident_majority_origin_wins_in_both_scenarios": majority_wins_both,
            "resident_within_one_percent_of_b3_in_at_least_one_scenario": within_one_percent,
            "classification_signal": stable_signal,
        }
    if signal_framings:
        classification = "family_signal_found"
    elif competitive_framings:
        classification = "family_competitive_but_unstable"
    else:
        classification = "family_unsupported"
    return classification, {
        "rule": (
            "family_signal_found requires one fixed framing to beat B3 on resident pooled MSE in both R1 and R2 "
            "and have at least 6/11 origin wins in both; within 1% without that stability is competitive_but_unstable."
        ),
        "framing_checks": framing_checks,
        "signal_framings": signal_framings,
        "competitive_framings": competitive_framings,
    }


def make_handoff(
    worker_commit: str,
    aggregate_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    framing_rows: list[dict[str, Any]],
    complementarity_rows: list[dict[str, Any]],
    configuration_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    model_summary: dict[str, Any],
) -> str:
    classification = model_summary["classification"]
    by_key = {(row["target_id"], row["framing_id"], row["scenario_id"]): row for row in aggregate_rows}
    lines = [
        f"# {PLAN_ID} Handoff — XGBoost 全量多表强模型 screening",
        "",
        "## TASK_RESULT",
        "",
        f"- 分支：`{BRANCH}`。",
        f"- exact Wave base：`{BASE_SHA}`。",
        f"- 本次 runner 读取并生成 artifacts 时的代码 tip：`{worker_commit}`。最终 worker tip 以该分支提交后的 git tip 与主结果回复为准。",
        f"- worker classification：`{classification}`。这只是 XGBoost family-level screening 结论，不是全局 winner。",
        "",
        "## Frozen contract and execution",
        "",
        "- Feature contract：`S5-Plan04-FULL-MULTISOURCE-V1`；144 numeric features；city_id 使用固定 `city1..city40` one-hot。",
        "- Replay contract：`S5-Plan04B-TRAINING-REPLAY-V1`；replay 2320 rows；formal validation 1760 rows；每个 framing 的 OOF coverage 为 1760，合并 F1/F2 后文件为 3520 rows。",
        "- Outer training：`pseudo_target_year <= outer_origin_year - 2`；inner eligible fold：training >= 40、validation >= 20、`boundary_pass=PASS`。",
        "- 264 eligible inner folds / 484 geometry rows；8 个 outer fit group 使用 C1 conservative fallback。",
        "- XGBoost：`objective=reg:squarederror`、`tree_method=hist`、`random_state=20260919`、`n_jobs=1`、无 early stopping；原始 NaN 交给 native missing handling。",
        "",
        "## Resident / registered metrics",
        "",
        "| target | framing | scenario | pooled MSE | B3 MSE | origin wins | latest-2021 MSE | RMSE | MAE | bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['target_id']} | {row['framing_id']} | {row['scenario_id']} | "
            f"{row['pooled_city_year_mse']:.6f} | {row['b3_pooled_city_year_mse']:.6f} | "
            f"{row['origin_win_count_vs_b3']}/{row['origin_loss_count_vs_b3']}/{row['origin_tie_count_vs_b3']} | "
            f"{row['latest_2021_mse']:.6f} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['bias']:.6f} |"
        )
    lines.extend([
        "",
        "胜负顺序为 `wins/losses/ties`；pooled MSE 与 latest-2021 MSE 越低越好。",
        "",
        "## Direct / residual and R1 / R2 comparisons",
        "",
    ])
    for row in framing_rows:
        lines.append(
            f"- `{row['target_id']}` / `{row['scenario_id']}`：F2 residual - F1 direct pooled MSE "
            f"`{row['delta_pooled_mse']:.6f}`；latest-2021 delta `{row['delta_latest_2021_mse']:.6f}`；"
            f"origin wins F1/F2 `{row['direct_origin_wins']}/{row['residual_origin_wins']}`。"
        )
    for row in scenario_rows:
        lines.append(
            f"- `{row['target_id']}` / `{row['framing_id']}`：R2 - R1 pooled MSE "
            f"`{row['delta_pooled_mse']:.6f}`；latest-2021 delta `{row['delta_latest_2021_mse']:.6f}`；"
            f"origin wins R1/R2 `{row['r1_origin_wins']}/{row['r2_origin_wins']}`。"
        )
    lines.extend(["", "## Configuration and fallback distribution", ""])
    config_counts = Counter(row["selected_config"] for row in configuration_rows)
    fallback_count = sum(row["selection_method"] == "fallback_conservative_default" for row in configuration_rows)
    lines.append(
        f"- Selected outer fits：`{len(configuration_rows)}`；fallback fit count：`{fallback_count}`；"
        f"selection fit count：`{len(configuration_rows) - fallback_count}`。"
    )
    lines.append("- Config counts：" + ", ".join(f"`{name}`={config_counts.get(name, 0)}" for name in CONFIG_BY_NAME))
    lines.extend(["", "## City concentration", ""])
    for row in aggregate_rows:
        lines.append(
            f"- `{row['target_id']}` / `{row['framing_id']}` / `{row['scenario_id']}`："
            f"top-1 `{row['top_1_city_id']}` concentration `{row['top_1_city_sse_concentration']:.4f}`；"
            f"top-3 `{row['top_3_city_ids']}` concentration `{row['top_3_city_sse_concentration']:.4f}`。"
        )
    lines.extend(["", "## Top feature groups by normalized gain", ""])
    group_rows = [row for row in importance_rows if row["record_type"] == "feature_group"]
    for row in sorted(group_rows, key=lambda item: int(item["rank"]))[:8]:
        lines.append(
            f"- `{row['feature_group']}` rank `{row['rank']}`，mean normalized gain `{row['mean_normalized_gain']:.6f}` "
            f"（raw gain sum `{row['raw_gain_sum']:.6f}`）。"
        )
    lines.extend(["", "## B3 error complementarity diagnostic", ""])
    for row in complementarity_rows:
        lines.append(
            f"- `{row['target_id']}` / `{row['framing_id']}` / `{row['scenario_id']}` / `{row['unit_scope']}`："
            f"signed-error corr `{row['signed_error_correlation'] if row['signed_error_correlation'] is not None else 'NA'}`；"
            f"model/B3 better units `{row['model_better_unit_count']}/{row['b3_better_unit_count']}/{row['tie_unit_count']}`。"
        )
    lines.extend([
        "",
        "## Deviations and explicit boundaries",
        "",
        "- 未修改 requirements.txt、Plan04/04B/05 artifacts、CURRENT_LEARNING_CHECKPOINT.md 或其他 worker 文件。",
        "- 未运行 Optuna、Bayesian optimization、random search 或 large grid search；未根据 outer labels 选 config。",
        "- 未创建 city routing/gating/ensemble/blend；未生成 2023 prediction/submission；未声明 S5 完成；未进入 S6。",
        "- XGBoost 使用固定 CPU 单线程参数；本环境未强制额外 GPU/外部 deterministic plugin。",
        "",
        "## Worker artifacts",
        "",
        "- `scripts/build_s5_plan06c_xgboost_screening.py`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/oof_predictions.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/aggregate_metrics.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/per_origin_metrics.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/configuration_selection.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/inner_selection_evidence.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/city_model_error.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/missingness_diagnostics.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/feature_importance.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/model_summary.json`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/scenario_comparison.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/framing_comparison.csv`",
        "- `reports/S5_HANDOFFS/S5-Plan06C/error_complementarity.csv`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started_at = time.perf_counter()
    worker_commit = assert_worker_identity()
    contracts = load_contracts()
    schema = contracts["schema"]
    global FEATURE_GROUP_BY_NAME
    FEATURE_GROUP_BY_NAME = {
        feature_name: group_id
        for group_id, group in schema["feature_groups"].items()
        for feature_name in group["features"]
    }
    model_features = tuple(schema["model_features"])

    formal_header, formal_raw, _ = read_csv_table(FORMAL_MATRIX_PATH)
    replay_header, replay_raw, _ = read_csv_table(REPLAY_MATRIX_PATH)
    validate_rows(formal_header, formal_raw, replay_header, replay_raw, schema)
    outer_by_key, inner_by_key = load_geometry(formal_raw, replay_raw)

    formal_rows: list[dict[str, Any]] = [dict(row) for row in formal_raw]
    replay_rows: list[dict[str, Any]] = [dict(row) for row in replay_raw]
    attach_design_vectors(formal_rows, model_features)
    attach_design_vectors(replay_rows, model_features)

    fit_counter: Counter[str] = Counter()
    fit_counter["total_fit_count"] = 0
    fit_counter["inner_fit_count"] = 0
    fit_counter["outer_fit_count"] = 0
    fit_counter["fit_runtime_seconds"] = 0.0
    oof_rows: list[dict[str, Any]] = []
    configuration_rows: list[dict[str, Any]] = []
    inner_evidence_rows: list[dict[str, Any]] = []
    gain_records: list[np.ndarray] = []
    selected_depths: Counter[str] = Counter()
    selection_started = time.perf_counter()

    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            scoped_replay = [
                row
                for row in replay_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            for outer_origin_year in ORIGINS:
                fit_key = (target_id, scenario_id, outer_origin_year)
                outer_geometry = outer_by_key[fit_key]
                outer_training = [
                    row
                    for row in scoped_replay
                    if int(row["pseudo_target_year"]) <= outer_origin_year - HORIZON_YEARS
                ]
                validation_rows = [
                    row
                    for row in formal_rows
                    if row["target_id"] == target_id
                    and row["scenario_id"] == scenario_id
                    and int(row["origin_year"]) == outer_origin_year
                ]
                assert len(outer_training) == int(outer_geometry["training_row_count"])
                assert len(validation_rows) == 40
                inner_folds = [row for row in inner_by_key[fit_key] if eligible_fold(row)]
                for framing_id in FRAMINGS:
                    config_scores: dict[str, float] = {}
                    config_evidence: list[dict[str, Any]] = []
                    fold_payloads: list[tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]] = []
                    for fold in inner_folds:
                        inner_training_max = int(fold["inner_training_max_year"])
                        inner_validation_year = int(fold["inner_validation_year"])
                        inner_training = [
                            row
                            for row in outer_training
                            if int(row["pseudo_target_year"]) <= inner_training_max
                        ]
                        inner_validation = [
                            row
                            for row in outer_training
                            if int(row["pseudo_target_year"]) == inner_validation_year
                        ]
                        assert len(inner_training) == int(fold["inner_training_row_count"])
                        assert len(inner_validation) == int(fold["inner_validation_row_count"])
                        fold_payloads.append((fold, inner_training, inner_validation))
                    for config in CONFIGS:
                        fold_mse: list[float] = []
                        for fold, inner_training, inner_validation in fold_payloads:
                            _, prediction, runtime_seconds = fit_and_predict(
                                config, inner_training, inner_validation, framing_id, fit_counter
                            )
                            actual = np.asarray([actual_value(row) for row in inner_validation], dtype=np.float64)
                            validation_mse = float(np.mean((prediction - actual) ** 2))
                            fold_mse.append(validation_mse)
                            config_evidence.append(
                                {
                                    "target_id": target_id,
                                    "scenario_id": scenario_id,
                                    "outer_origin_year": outer_origin_year,
                                    "framing_id": framing_id,
                                    "inner_validation_year": int(fold["inner_validation_year"]),
                                    "inner_training_max_year": int(fold["inner_training_max_year"]),
                                    "inner_training_row_count": len(inner_training),
                                    "inner_validation_row_count": len(inner_validation),
                                    "config_name": config["config_name"],
                                    "config_order": config["order"],
                                    "validation_mse": validation_mse,
                                    "fit_runtime_seconds": runtime_seconds,
                                    "selection_method": "",
                                    "selected_for_outer_fit": "",
                                }
                            )
                        assert fold_mse
                        config_scores[config["config_name"]] = mean(fold_mse)

                    if len(inner_folds) < 3:
                        selected_config = CONFIGS[0]
                        selection_method = "fallback_conservative_default"
                    else:
                        selected_config = CONFIGS[0]
                        for candidate in CONFIGS[1:]:
                            selected_score = config_scores[selected_config["config_name"]]
                            candidate_score = config_scores[candidate["config_name"]]
                            if candidate_score < selected_score and not close_enough(candidate_score, selected_score):
                                selected_config = candidate
                        selection_method = "mean_inner_validation_mse"
                    for evidence in config_evidence:
                        evidence["selection_method"] = selection_method
                        evidence["selected_for_outer_fit"] = (
                            "yes" if evidence["config_name"] == selected_config["config_name"] else "no"
                        )
                    inner_evidence_rows.extend(config_evidence)

                    final_started = time.perf_counter()
                    final_model = make_model(selected_config)
                    x_outer = design_matrix(outer_training)
                    y_outer = np.asarray([framing_label(row, framing_id) for row in outer_training], dtype=np.float64)
                    assert np.isfinite(y_outer).all()
                    final_model.fit(x_outer, y_outer, verbose=False)
                    raw_prediction = np.asarray(final_model.predict(design_matrix(validation_rows)), dtype=np.float64)
                    if framing_id == "F2_B3_RESIDUAL":
                        predictions = raw_prediction + np.asarray([b3_value(row) for row in validation_rows])
                    else:
                        predictions = raw_prediction
                    assert np.isfinite(predictions).all()
                    final_runtime = time.perf_counter() - final_started
                    fit_counter["total_fit_count"] += 1
                    fit_counter["outer_fit_count"] += 1
                    fit_counter["fit_runtime_seconds"] += final_runtime
                    gain_records.append(gain_by_feature(final_model))
                    selected_depths[str(selected_config["params"]["max_depth"])] += 1
                    training_max_label_year = max(int(row["pseudo_target_year"]) for row in outer_training)
                    assert training_max_label_year == int(outer_geometry["training_max_pseudo_target_year"])
                    for row, prediction in zip(validation_rows, predictions):
                        actual = actual_value(row)
                        error = float(prediction) - actual
                        oof_rows.append(
                            {
                                "model_id": MODEL_ID,
                                "framing_id": framing_id,
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "origin_year": outer_origin_year,
                                "city_id": row["city_id"],
                                "actual_target_value": actual,
                                "prediction_value": float(prediction),
                                "residual_prediction_minus_actual": error,
                                "squared_error": error * error,
                                "training_max_label_year": training_max_label_year,
                                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                                "worker_commit": worker_commit,
                                "selected_config": selected_config["config_name"],
                            }
                        )
                    config_row = {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": outer_origin_year,
                        "framing_id": framing_id,
                        "outer_training_row_count": len(outer_training),
                        "outer_training_min_year": min(int(row["pseudo_target_year"]) for row in outer_training),
                        "training_max_label_year": training_max_label_year,
                        "eligible_inner_fold_count": len(inner_folds),
                        "selection_method": selection_method,
                        "selected_config": selected_config["config_name"],
                        "selected_config_order": selected_config["order"],
                        "selected_config_params": selected_config["params"],
                        "selected_inner_mean_mse": config_scores[selected_config["config_name"]],
                        "fallback_default": "yes" if selection_method == "fallback_conservative_default" else "no",
                        "final_fit_runtime_seconds": final_runtime,
                        "seed": SEED,
                        "tree_method": "hist",
                        "n_jobs": 1,
                    }
                    for config in CONFIGS:
                        config_row[f"{config['config_name']}_mean_inner_mse"] = config_scores[config["config_name"]]
                    configuration_rows.append(config_row)
                print(
                    f"completed target={target_id} scenario={scenario_id} origin={outer_origin_year} "
                    f"fits={fit_counter['total_fit_count']}",
                    flush=True,
                )

    # The frozen coverage is 1760 rows per framing.  This worker keeps both
    # required framings in one file, so the combined artifact has 3520 rows.
    assert len(oof_rows) == len(TARGETS) * len(SCENARIOS) * len(ORIGINS) * len(CITIES) * len(FRAMINGS)
    assert len(configuration_rows) == 88
    assert len(inner_evidence_rows) == 1584
    assert len(gain_records) == 88
    aggregate_rows, per_origin_rows, city_rows, complementarity_rows = build_metrics(oof_rows, formal_rows)
    scenario_rows, framing_rows = comparison_rows(aggregate_rows)
    classification, classification_detail = classify_family(aggregate_rows)
    importance_rows = aggregate_gain(gain_records)

    scope_rows: list[tuple[str, str, str, str, list[dict[str, Any]]]] = [
        ("replay_all", "ALL", "ALL", "ALL", replay_rows),
        ("formal_all", "ALL", "ALL", "ALL", formal_rows),
    ]
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            scope_rows.append(
                (
                    "replay_pair",
                    target_id,
                    scenario_id,
                    "ALL",
                    [
                        row
                        for row in replay_rows
                        if row["target_id"] == target_id and row["scenario_id"] == scenario_id
                    ],
                )
            )
            scope_rows.append(
                (
                    "formal_pair",
                    target_id,
                    scenario_id,
                    "ALL",
                    [
                        row
                        for row in formal_rows
                        if row["target_id"] == target_id and row["scenario_id"] == scenario_id
                    ],
                )
            )
    missingness_rows = collect_missingness(scope_rows)

    model_summary: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "branch": BRANCH,
        "wave_base_sha": BASE_SHA,
        "runner_commit": worker_commit,
        "classification": classification,
        "classification_detail": classification_detail,
        "contracts": {
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "formal_row_count": len(formal_rows),
            "replay_row_count": len(replay_rows),
            "model_numeric_feature_count": len(model_features),
            "design_feature_count": len(DESIGN_FEATURES),
            "oof_row_count": len(oof_rows),
            "oof_row_count_per_framing": len(oof_rows) // len(FRAMINGS),
            "configuration_selection_row_count": len(configuration_rows),
            "inner_evidence_row_count": len(inner_evidence_rows),
        },
        "randomness_and_runtime": {
            "seed": SEED,
            "tree_method": "hist",
            "n_jobs": 1,
            "device": "CPU default; no GPU parameter supplied",
            "determinism_limitation": "No additional host-level OpenMP determinism guarantee was available; XGBoost seed and single-thread fitting were fixed.",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "total_runtime_seconds": time.perf_counter() - started_at,
            "selection_runtime_seconds": time.perf_counter() - selection_started,
            "fit_counter": dict(fit_counter),
        },
        "configurations": [
            {
                "config_name": config["config_name"],
                "order": config["order"],
                "params": config["params"],
                "selected_count": sum(row["selected_config"] == config["config_name"] for row in configuration_rows),
            }
            for config in CONFIGS
        ],
        "fallback_fit_count": sum(
            row["selection_method"] == "fallback_conservative_default" for row in configuration_rows
        ),
        "selected_depth_distribution": dict(sorted(selected_depths.items())),
        "missing_value_handling": {
            "numeric_nan_retained": True,
            "numeric_imputer": "none",
            "scaler": "none",
            "xgboost_native_missing": True,
            "city_onehot_vocabulary": list(CITIES),
        },
        "aggregate_metrics": aggregate_rows,
        "direct_residual_comparison": framing_rows,
        "r1_r2_comparison": scenario_rows,
        "b3_error_complementarity": complementarity_rows,
        "top_feature_groups": sorted(
            [row for row in importance_rows if row["record_type"] == "feature_group"],
            key=lambda row: int(row["rank"]),
        )[:8],
        "forbidden_actions_status": {
            "tuning_beyond_preregistered_configs": False,
            "outer_label_driven_config_selection": False,
            "city_routing": False,
            "ensemble_or_blend": False,
            "prediction_2023": False,
            "submission": False,
            "s5_complete": False,
            "s6_entered": False,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        oof_rows,
        contracts["oof_fields"],
    )
    write_csv(
        OUTPUT_DIR / "aggregate_metrics.csv",
        aggregate_rows,
        (
            "model_id",
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
            "origin_loss_count_vs_b3",
            "origin_tie_count_vs_b3",
            "b3_pooled_city_year_mse",
            "b3_latest_2021_mse",
            "delta_pooled_mse_vs_b3",
            "delta_latest_2021_mse_vs_b3",
            "total_city_sse",
            "top_1_city_id",
            "top_1_city_sse",
            "top_1_city_sse_concentration",
            "top_3_city_ids",
            "top_3_city_sse",
            "top_3_city_sse_concentration",
            "selected_config_distribution",
        ),
    )
    write_csv(
        OUTPUT_DIR / "per_origin_metrics.csv",
        per_origin_rows,
        (
            "model_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "row_count",
            "pooled_city_year_mse",
            "b3_mse",
            "delta_mse_vs_b3",
            "rmse",
            "mae",
            "bias",
            "origin_result_vs_b3",
            "selected_config",
        ),
    )
    write_csv(
        OUTPUT_DIR / "configuration_selection.csv",
        configuration_rows,
        (
            "target_id",
            "scenario_id",
            "outer_origin_year",
            "framing_id",
            "outer_training_row_count",
            "outer_training_min_year",
            "training_max_label_year",
            "eligible_inner_fold_count",
            "selection_method",
            "selected_config",
            "selected_config_order",
            "selected_config_params",
            "selected_inner_mean_mse",
            "fallback_default",
            "final_fit_runtime_seconds",
            "seed",
            "tree_method",
            "n_jobs",
            "XGB_C1_CONSERVATIVE_mean_inner_mse",
            "XGB_C2_MEDIUM_mean_inner_mse",
            "XGB_C3_FLEXIBLE_REGULARIZED_mean_inner_mse",
        ),
    )
    write_csv(
        OUTPUT_DIR / "inner_selection_evidence.csv",
        inner_evidence_rows,
        (
            "target_id",
            "scenario_id",
            "outer_origin_year",
            "framing_id",
            "inner_validation_year",
            "inner_training_max_year",
            "inner_training_row_count",
            "inner_validation_row_count",
            "config_name",
            "config_order",
            "validation_mse",
            "fit_runtime_seconds",
            "selection_method",
            "selected_for_outer_fit",
        ),
    )
    write_csv(
        OUTPUT_DIR / "city_model_error.csv",
        city_rows,
        (
            "model_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "city_id",
            "row_count",
            "model_sse",
            "model_mse",
            "model_mae",
            "model_bias",
            "b3_sse",
            "b3_mse",
            "delta_sse_vs_b3",
            "city_sse_rank",
            "city_sse_share",
            "top_1_flag",
            "top_3_flag",
        ),
    )
    write_csv(
        OUTPUT_DIR / "missingness_diagnostics.csv",
        missingness_rows,
        (
            "matrix_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "feature_name",
            "feature_group",
            "row_count",
            "missing_count",
            "missing_rate",
            "handling",
            "numeric_imputer",
            "scaler",
        ),
    )
    write_csv(
        OUTPUT_DIR / "feature_importance.csv",
        importance_rows,
        (
            "record_type",
            "feature_name",
            "feature_group",
            "is_city_onehot",
            "raw_gain_sum",
            "mean_normalized_gain",
            "selected_fit_count",
            "nonzero_fit_count",
            "rank",
        ),
    )
    write_csv(
        OUTPUT_DIR / "scenario_comparison.csv",
        scenario_rows,
        (
            "comparison",
            "target_id",
            "framing_id",
            "r1_pooled_mse",
            "r2_pooled_mse",
            "delta_pooled_mse",
            "r1_latest_2021_mse",
            "r2_latest_2021_mse",
            "delta_latest_2021_mse",
            "r1_origin_wins",
            "r2_origin_wins",
        ),
    )
    write_csv(
        OUTPUT_DIR / "framing_comparison.csv",
        framing_rows,
        (
            "comparison",
            "target_id",
            "scenario_id",
            "direct_pooled_mse",
            "residual_pooled_mse",
            "delta_pooled_mse",
            "direct_latest_2021_mse",
            "residual_latest_2021_mse",
            "delta_latest_2021_mse",
            "direct_origin_wins",
            "residual_origin_wins",
        ),
    )
    write_csv(
        OUTPUT_DIR / "error_complementarity.csv",
        complementarity_rows,
        (
            "model_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "unit_scope",
            "unit_count",
            "signed_error_correlation",
            "absolute_error_correlation",
            "squared_error_correlation",
            "model_better_unit_count",
            "b3_better_unit_count",
            "tie_unit_count",
            "mean_model_unit_squared_error",
            "mean_b3_unit_squared_error",
        ),
    )
    write_json(OUTPUT_DIR / "model_summary.json", model_summary)
    (OUTPUT_DIR / "HANDOFF.md").write_text(
        make_handoff(
            worker_commit,
            aggregate_rows,
            scenario_rows,
            framing_rows,
            complementarity_rows,
            configuration_rows,
            importance_rows,
            model_summary,
        ),
        encoding="utf-8",
    )

    print(f"classification={classification}")
    print(f"oof_rows={len(oof_rows)}")
    print(f"outer_fit_count={fit_counter['outer_fit_count']}")
    print(f"inner_fit_count={fit_counter['inner_fit_count']}")
    print(f"fallback_fit_count={model_summary['fallback_fit_count']}")
    print(f"worker_commit={worker_commit}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
