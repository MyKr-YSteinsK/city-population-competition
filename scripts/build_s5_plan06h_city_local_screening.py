"""Run the frozen S5-Plan06H city-local screening worker.

This worker evaluates only city-local, independently fitted corrections to the
frozen B3 baseline.  Every estimator, imputer, scaler, residual statistic,
and model comparison is scoped to one (target, scenario, city) key and to
replay rows with ``pseudo_target_year <= outer_origin_year - 2``.  The formal
Plan04 matrix is used only for outer validation and B3 references.

The worker intentionally does not create a router, gate, blend, ensemble,
2023 forecast, submission, S5 completion signal, or S6 artifact.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan06H"
BRANCH = "s5/plan06h-city-local-screening"
BASE_SHA = "1f97f67488cd7aec120ef8b9fa04407dc13d54ba"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
FEATURE_CONTRACT_VERSION = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT_VERSION = "S5-Plan04B-TRAINING-REPLAY-V1"
HORIZON_YEARS = 2
RIDGE_ALPHA = 1000.0
EWMA_DECAY = 0.7
MIN_MEAN_ROWS = 3
MIN_RIDGE_ROWS = 5
# Contract CSVs store rounded predictions and squared errors.  The input
# identity check therefore uses a slightly wider tolerance, while metric
# win/loss comparisons retain the strict frozen tie band.
NUMERIC_TOLERANCE = 1e-12
INPUT_NUMERIC_TOLERANCE = 1e-10

OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
FORMAL_MATRIX_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv"
FORMAL_SCHEMA_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
FEATURE_DEFINITIONS_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_definitions.csv"
OOF_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "oof_prediction_contract.csv"
REPLAY_MATRIX_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv"
WORKER_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan05" / "strong_model_wave_contract.json"
PARALLEL_INPUT_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "parallel_worker_input_contract.json"

TARGETS = ("resident", "registered")
SCENARIOS = (
    "R1_REFERENCE_SNAPSHOT_LAG",
    "R2_CONSERVATIVE_PUBLICATION_STRESS",
)
ORIGINS = tuple(range(2011, 2022))
CITIES = tuple(f"city{i}" for i in range(1, 41))

B3_REFERENCE_POOLED_MSE = {
    "resident": 872.452651515143,
    "registered": 225.421212121213,
}

FGP_FEATURES = (
    "pop_latest_level",
    "pop_log1p_latest",
    "pop_latest_observation_year",
    "pop_history_observation_count",
    "pop_source_age_years",
    "pop_latest_first_difference",
    "pop_latest_relative_growth",
    "pop_recent3_ols_slope",
    "pop_expanding_ols_slope",
    "pop_recent_vs_expanding_slope_disagreement_abs",
    "pop_history_difference_volatility",
    "pop_relative_history_difference_volatility",
    "pop_recent_acceleration_abs",
    "pop_b1_prediction",
    "pop_b2_prediction",
    "pop_b3_prediction",
    "pop_baseline_prediction_spread",
)

CANDIDATES = (
    {
        "candidate_id": "LOCAL_RESIDUAL_MEAN",
        "framing_id": "F2_B3_RESIDUAL",
        "kind": "residual_mean",
        "definition": "same-city legal B3 residual mean; minimum 3 residual rows; otherwise B3 unchanged",
    },
    {
        "candidate_id": "LOCAL_RESIDUAL_EWMA",
        "framing_id": "F2_B3_RESIDUAL",
        "kind": "residual_ewma",
        "definition": "same-city legal B3 residual EWMA with fixed decay 0.7; minimum 3 residual rows; otherwise B3 unchanged",
    },
    {
        "candidate_id": "LOCAL_RESIDUAL_TIME_TREND",
        "framing_id": "F2_B3_RESIDUAL",
        "kind": "residual_time_trend",
        "definition": "same-city legal residual OLS trend; minimum 4 rows; otherwise Candidate A mean fallback",
    },
    {
        "candidate_id": "LOCAL_FGP_RIDGE_RESIDUAL",
        "framing_id": "F2_B3_RESIDUAL",
        "kind": "fgp_ridge_residual",
        "definition": "same-city 17-feature FG-P Ridge residual with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows",
    },
    {
        "candidate_id": "LOCAL_ALL144_RIDGE_RESIDUAL",
        "framing_id": "F2_B3_RESIDUAL",
        "kind": "all144_ridge_residual",
        "definition": "same-city all-144-feature Ridge residual with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows",
    },
    {
        "candidate_id": "LOCAL_FGP_RIDGE_DIRECT",
        "framing_id": "F1_DIRECT",
        "kind": "fgp_ridge_direct",
        "definition": "diagnostic same-city 17-feature direct-actual Ridge with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows",
    },
)
CANDIDATE_BY_ID = {candidate["candidate_id"]: candidate for candidate in CANDIDATES}

MISSING_TEXT = {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "NULL"}


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
    """Read a CSV and verify duplicate headers have identical values.

    Plan04's formal matrix intentionally carries audit columns overlapping
    model-feature names.  A normal DictReader would silently overwrite one of
    those values, so the first value is retained only after all duplicates are
    checked row by row.
    """

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
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
    return abs(left - right) <= INPUT_NUMERIC_TOLERANCE * max(1.0, abs(left), abs(right))


def compare_metrics(left: float, right: float) -> int:
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
    assert result.returncode == 0, "the required exact base must be an ancestor of worker tip"
    return git_text("rev-parse", "HEAD")


def load_contracts() -> tuple[dict[str, Any], tuple[str, ...]]:
    schema = json.loads(FORMAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    wave = json.loads(WORKER_CONTRACT_PATH.read_text(encoding="utf-8"))
    parallel = json.loads(PARALLEL_INPUT_CONTRACT_PATH.read_text(encoding="utf-8"))
    oof_header, _, _ = read_csv_table(OOF_CONTRACT_PATH)
    definition_header, definition_rows, _ = read_csv_table(FEATURE_DEFINITIONS_PATH)

    model_features = tuple(schema["model_features"])
    assert schema["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert len(model_features) == 144
    assert schema["categorical_features"] == ["city_id"]
    assert wave["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert wave["replay_contract_version"] == REPLAY_CONTRACT_VERSION
    assert wave["data_contract"]["formal_validation_row_count"] == 1760
    assert wave["data_contract"]["replay_row_count"] == 2320
    assert wave["data_contract"]["model_feature_count"] == 144
    assert parallel["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert parallel["contract_version"] == REPLAY_CONTRACT_VERSION
    assert parallel["model_feature_count"] == 144
    assert tuple(parallel["model_features"]) == model_features
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

    require_fields(
        definition_header,
        (
            "feature_name",
            "feature_group",
            "model_feature",
            "feature_contract_version",
        ),
        "feature definitions",
    )
    definition_by_name = {
        row["feature_name"]: row for row in definition_rows if row["model_feature"].lower() == "yes"
    }
    assert tuple(definition_by_name) == model_features
    assert all(row["feature_contract_version"] == FEATURE_CONTRACT_VERSION for row in definition_by_name.values())
    defined_fgp = tuple(
        row["feature_name"] for row in definition_rows if row["feature_group"] == "FG-P" and row["model_feature"].lower() == "yes"
    )
    assert defined_fgp == FGP_FEATURES
    return {"schema": schema, "wave": wave, "parallel": parallel, "oof_fields": oof_header}, model_features


def validate_and_index_rows(
    formal_header: list[str],
    formal_rows: list[dict[str, str]],
    replay_header: list[str],
    replay_rows: list[dict[str, str]],
    model_features: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str, int, str], dict[str, str]],
    dict[tuple[str, str, str], list[dict[str, str]]],
]:
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

    formal_by_key: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in formal_rows:
        target_id = row["target_id"]
        scenario_id = row["scenario_id"]
        origin_year = parse_int(row["origin_year"])
        city_id = row["city_id"]
        assert target_id in TARGETS and scenario_id in SCENARIOS and origin_year in ORIGINS and city_id in CITIES
        assert row["feature_contract_version"] == FEATURE_CONTRACT_VERSION
        actual = parse_float(row["actual_target_value"])
        b3 = parse_float(row["b3_prediction"])
        b3_residual = parse_float(row["b3_residual_actual_minus_prediction"])
        b3_error = parse_float(row["b3_squared_error"])
        assert actual is not None and b3 is not None and b3_residual is not None and b3_error is not None
        assert close_enough(b3_residual, actual - b3)
        assert close_enough(b3_error, (b3 - actual) ** 2)
        key = (target_id, scenario_id, origin_year, city_id)
        assert key not in formal_by_key, key
        formal_by_key[key] = row

    replay_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    replay_keys: set[tuple[str, str, int, str]] = set()
    for row in replay_rows:
        target_id = row["target_id"]
        scenario_id = row["scenario_id"]
        pseudo_year = parse_int(row["pseudo_target_year"])
        city_id = row["city_id"]
        assert target_id in TARGETS and scenario_id in SCENARIOS and pseudo_year is not None
        assert 2004 <= pseudo_year <= 2021 and city_id in CITIES
        assert row["feature_contract_version"] == FEATURE_CONTRACT_VERSION
        actual = parse_float(row["actual_target_value"])
        b3 = parse_float(row["b3_prediction"])
        b3_residual = parse_float(row["b3_residual_actual_minus_prediction"])
        b3_error = parse_float(row["b3_squared_error"])
        assert actual is not None and b3 is not None and b3_residual is not None and b3_error is not None
        assert close_enough(b3_residual, actual - b3)
        assert close_enough(b3_error, (b3 - actual) ** 2)
        key = (target_id, scenario_id, pseudo_year, city_id)
        assert key not in replay_keys, key
        replay_keys.add(key)
        replay_by_key[(target_id, scenario_id, city_id)].append(row)

    assert len(formal_by_key) == 1760
    assert len(replay_keys) == 2320
    assert len(replay_by_key) == len(TARGETS) * len(SCENARIOS) * len(CITIES)
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            formal_scoped = [
                row
                for (target, scenario, _origin, _city), row in formal_by_key.items()
                if target == target_id and scenario == scenario_id
            ]
            assert len(formal_scoped) == 440
            b3_mse = mean(
                (parse_float(row["b3_prediction"]) - parse_float(row["actual_target_value"])) ** 2
                for row in formal_scoped
            )
            assert close_enough(b3_mse, B3_REFERENCE_POOLED_MSE[target_id]), (target_id, scenario_id, b3_mse)
            for origin_year in ORIGINS:
                scoped_origin = [
                    row
                    for (target, scenario, origin, _city), row in formal_by_key.items()
                    if target == target_id and scenario == scenario_id and origin == origin_year
                ]
                assert len(scoped_origin) == len(CITIES)
    return formal_by_key, replay_by_key


def legal_local_rows(
    replay_by_key: dict[tuple[str, str, str], list[dict[str, str]]],
    target_id: str,
    scenario_id: str,
    city_id: str,
    origin_year: int,
) -> list[dict[str, str]]:
    cutoff = origin_year - HORIZON_YEARS
    rows = [
        row
        for row in replay_by_key[(target_id, scenario_id, city_id)]
        if parse_int(row["pseudo_target_year"]) <= cutoff
    ]
    assert all(row["city_id"] == city_id for row in rows)
    assert all(parse_int(row["pseudo_target_year"]) <= cutoff for row in rows)
    return sorted(rows, key=lambda row: parse_int(row["pseudo_target_year"]))


def feature_matrix(rows: list[dict[str, str]], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [
            [np.nan if parse_float(row[feature]) is None else parse_float(row[feature]) for feature in features]
            for row in rows
        ],
        dtype=np.float64,
    )


def ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def train_residual_values(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for row in rows:
        actual = parse_float(row["actual_target_value"])
        b3 = parse_float(row["b3_prediction"])
        year = parse_int(row["pseudo_target_year"])
        assert actual is not None and b3 is not None and year is not None
        values.append((year, actual - b3))
    return values


def fit_candidate(
    candidate: dict[str, str],
    train_rows: list[dict[str, str]],
    validation_row: dict[str, str],
    model_features: tuple[str, ...],
) -> tuple[float, float, bool, dict[str, Any]]:
    """Return prediction, B3-relative correction, fallback, and fit evidence."""

    candidate_id = candidate["candidate_id"]
    kind = candidate["kind"]
    b3 = parse_float(validation_row["b3_prediction"])
    assert b3 is not None
    residual_values = train_residual_values(train_rows)
    valid_residual_count = len(residual_values)
    fallback = False
    fit_method = ""
    correction = 0.0

    if kind == "residual_mean":
        if valid_residual_count >= MIN_MEAN_ROWS:
            correction = mean(value for _year, value in residual_values)
            fit_method = "local_mean"
        else:
            fallback = True
            fit_method = "b3_unchanged_under_minimum"
        prediction = b3 + correction
    elif kind == "residual_ewma":
        if valid_residual_count >= MIN_MEAN_ROWS:
            latest_year = max(year for year, _value in residual_values)
            weights = [EWMA_DECAY ** (latest_year - year) for year, _value in residual_values]
            correction = math.fsum(weight * value for weight, (_year, value) in zip(weights, residual_values)) / math.fsum(weights)
            fit_method = "local_ewma_decay_0.7"
        else:
            fallback = True
            fit_method = "b3_unchanged_under_minimum"
        prediction = b3 + correction
    elif kind == "residual_time_trend":
        if valid_residual_count >= 4:
            years = np.asarray([year for year, _value in residual_values], dtype=np.float64)
            residuals = np.asarray([value for _year, value in residual_values], dtype=np.float64)
            year_mean = float(np.mean(years))
            residual_mean = float(np.mean(residuals))
            denominator = float(np.sum((years - year_mean) ** 2))
            if denominator == 0:
                correction = residual_mean
                fit_method = "local_mean_due_to_zero_time_variance"
            else:
                slope = float(np.sum((years - year_mean) * (residuals - residual_mean)) / denominator)
                intercept = residual_mean - slope * year_mean
                correction = intercept + slope * parse_int(validation_row["origin_year"])
                fit_method = "local_ols_residual_time_trend"
            prediction = b3 + correction
        elif valid_residual_count >= MIN_MEAN_ROWS:
            fallback = True
            correction = mean(value for _year, value in residual_values)
            fit_method = "candidate_a_local_mean_fallback"
            prediction = b3 + correction
        else:
            fallback = True
            fit_method = "b3_unchanged_under_minimum"
            prediction = b3
    elif kind in {"fgp_ridge_residual", "all144_ridge_residual", "fgp_ridge_direct"}:
        if len(train_rows) < MIN_RIDGE_ROWS:
            fallback = True
            fit_method = "b3_unchanged_under_minimum"
            prediction = b3
        else:
            features = FGP_FEATURES if kind in {"fgp_ridge_residual", "fgp_ridge_direct"} else model_features
            x_train = feature_matrix(train_rows, features)
            x_validation = feature_matrix([validation_row], features)
            pipeline = ridge_pipeline()
            if kind == "fgp_ridge_direct":
                labels = np.asarray(
                    [parse_float(row["actual_target_value"]) for row in train_rows], dtype=np.float64
                )
            else:
                labels = np.asarray(
                    [
                        parse_float(row["actual_target_value"]) - parse_float(row["b3_prediction"])
                        for row in train_rows
                    ],
                    dtype=np.float64,
                )
            assert np.isfinite(labels).all()
            pipeline.fit(x_train, labels)
            raw_prediction = float(np.asarray(pipeline.predict(x_validation), dtype=np.float64)[0])
            assert math.isfinite(raw_prediction)
            if kind == "fgp_ridge_direct":
                prediction = raw_prediction
                correction = prediction - b3
                fit_method = "local_fgp_ridge_direct_actual"
            else:
                correction = raw_prediction
                prediction = b3 + correction
                fit_method = "local_ridge_residual"
    else:  # pragma: no cover - candidate registry is frozen above
        raise AssertionError(candidate_id)

    assert math.isfinite(prediction) and math.isfinite(correction)
    return prediction, correction, fallback, {
        "fit_method": fit_method,
        "local_training_row_count": len(train_rows),
        "valid_residual_row_count": valid_residual_count,
        "training_min_year": min((year for year, _value in residual_values), default=None),
        "training_max_year": max((year for year, _value in residual_values), default=None),
    }


def build_geometry(
    formal_by_key: dict[tuple[str, str, int, str], dict[str, str]],
    replay_by_key: dict[tuple[str, str, str], list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, str], list[dict[str, str]]]]:
    geometry_rows: list[dict[str, Any]] = []
    local_rows_by_validation: dict[tuple[str, str, int, str], list[dict[str, str]]] = {}
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for origin_year in ORIGINS:
                for city_id in CITIES:
                    key = (target_id, scenario_id, origin_year, city_id)
                    assert key in formal_by_key
                    local_rows = legal_local_rows(replay_by_key, target_id, scenario_id, city_id, origin_year)
                    local_rows_by_validation[key] = local_rows
                    residual_values = train_residual_values(local_rows)
                    local_cities = {row["city_id"] for row in local_rows}
                    row = {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "origin_year": origin_year,
                        "city_id": city_id,
                        "outer_cutoff_year": origin_year - HORIZON_YEARS,
                        "local_training_row_count": len(local_rows),
                        "valid_residual_row_count": len(residual_values),
                        "local_training_min_year": min((year for year, _value in residual_values), default=None),
                        "local_training_max_year": max((year for year, _value in residual_values), default=None),
                        "local_city_count": len(local_cities),
                        "local_city_ids": sorted(local_cities),
                        "city_only_pass": "yes" if local_cities in ({city_id}, set()) else "no",
                        "cutoff_pass": "yes"
                        if all(year <= origin_year - HORIZON_YEARS for year, _value in residual_values)
                        else "no",
                        "mean_threshold_pass": "yes" if len(residual_values) >= MIN_MEAN_ROWS else "no",
                        "trend_threshold_pass": "yes" if len(residual_values) >= 4 else "no",
                        "ridge_threshold_pass": "yes" if len(local_rows) >= MIN_RIDGE_ROWS else "no",
                    }
                    assert row["city_only_pass"] == "yes"
                    assert row["cutoff_pass"] == "yes"
                    geometry_rows.append(row)
    assert len(geometry_rows) == 1760
    return geometry_rows, local_rows_by_validation


def build_oof(
    formal_by_key: dict[tuple[str, str, int, str], dict[str, str]],
    local_rows_by_validation: dict[tuple[str, str, int, str], list[dict[str, str]]],
    model_features: tuple[str, ...],
    worker_commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    oof_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        print(f"running candidate={candidate_id}", flush=True)
        for target_id in TARGETS:
            for scenario_id in SCENARIOS:
                for origin_year in ORIGINS:
                    for city_id in CITIES:
                        key = (target_id, scenario_id, origin_year, city_id)
                        validation_row = formal_by_key[key]
                        local_rows = local_rows_by_validation[key]
                        prediction, correction, fallback, evidence = fit_candidate(
                            candidate, local_rows, validation_row, model_features
                        )
                        actual = parse_float(validation_row["actual_target_value"])
                        b3 = parse_float(validation_row["b3_prediction"])
                        assert actual is not None and b3 is not None
                        error = prediction - actual
                        b3_error = b3 - actual
                        row = {
                            "model_id": "S5-Plan06H_CITY_LOCAL_SCREENING",
                            "framing_id": candidate["framing_id"],
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "origin_year": origin_year,
                            "city_id": city_id,
                            "actual_target_value": actual,
                            "prediction_value": prediction,
                            "residual_prediction_minus_actual": error,
                            "squared_error": error * error,
                            "training_max_label_year": evidence["training_max_year"],
                            "feature_contract_version": FEATURE_CONTRACT_VERSION,
                            "worker_commit": worker_commit,
                            "candidate_id": candidate_id,
                            "local_training_row_count": evidence["local_training_row_count"],
                            "local_fallback_used": "yes" if fallback else "no",
                            "local_correction_value": correction,
                            "b3_prediction": b3,
                            "b3_squared_error": b3_error * b3_error,
                            "valid_residual_row_count": evidence["valid_residual_row_count"],
                            "local_training_min_year": evidence["training_min_year"],
                            "local_training_max_year": evidence["training_max_year"],
                            "fit_method": evidence["fit_method"],
                        }
                        oof_rows.append(row)
                        correction_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "framing_id": candidate["framing_id"],
                                "target_id": target_id,
                                "scenario_id": scenario_id,
                                "origin_year": origin_year,
                                "city_id": city_id,
                                "local_training_row_count": evidence["local_training_row_count"],
                                "valid_residual_row_count": evidence["valid_residual_row_count"],
                                "local_training_min_year": evidence["training_min_year"],
                                "local_training_max_year": evidence["training_max_year"],
                                "outer_cutoff_year": origin_year - HORIZON_YEARS,
                                "local_fallback_used": "yes" if fallback else "no",
                                "fit_method": evidence["fit_method"],
                                "local_correction_value": correction,
                                "absolute_correction_value": abs(correction),
                                "actual_target_value": actual,
                                "b3_prediction": b3,
                                "prediction_value": prediction,
                                "model_squared_error": error * error,
                                "b3_squared_error": b3_error * b3_error,
                                "delta_squared_error_vs_b3": error * error - b3_error * b3_error,
                            }
                        )
    expected_rows = len(CANDIDATES) * len(TARGETS) * len(SCENARIOS) * len(ORIGINS) * len(CITIES)
    assert len(oof_rows) == expected_rows
    assert len(correction_rows) == expected_rows
    return oof_rows, correction_rows


def metric_core(rows: list[dict[str, Any]], prediction_key: str = "prediction_value") -> dict[str, float]:
    actual = [float(row["actual_target_value"]) for row in rows]
    predicted = [float(row[prediction_key]) for row in rows]
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    return {
        "row_count": len(rows),
        "mse": mean(error * error for error in errors),
        "rmse": math.sqrt(mean(error * error for error in errors)),
        "mae": mean(abs(error) for error in errors),
        "bias": mean(errors),
    }


def metric_group_rows(
    oof_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in oof_rows:
        grouped[(row["candidate_id"], row["target_id"], row["scenario_id"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    per_origin_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    for (candidate_id, target_id, scenario_id), rows in grouped.items():
        candidate = CANDIDATE_BY_ID[candidate_id]
        origin_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        city_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            origin_groups[int(row["origin_year"])].append(row)
            city_groups[row["city_id"]].append(row)
        core = metric_core(rows)
        origin_mses: list[float] = []
        origin_win_count = 0
        origin_loss_count = 0
        origin_tie_count = 0
        for origin_year in ORIGINS:
            origin_rows = origin_groups[origin_year]
            assert len(origin_rows) == len(CITIES)
            model_metric = metric_core(origin_rows)
            b3_mse = mean(float(row["b3_squared_error"]) for row in origin_rows)
            result = compare_metrics(model_metric["mse"], b3_mse)
            result_text = {-1: "win", 0: "tie", 1: "loss"}[result]
            origin_win_count += result < 0
            origin_loss_count += result > 0
            origin_tie_count += result == 0
            origin_mses.append(model_metric["mse"])
            per_origin_rows.append(
                {
                    "model_id": "S5-Plan06H_CITY_LOCAL_SCREENING",
                    "candidate_id": candidate_id,
                    "framing_id": candidate["framing_id"],
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "origin_year": origin_year,
                    "row_count": model_metric["row_count"],
                    "pooled_city_year_mse": model_metric["mse"],
                    "b3_mse": b3_mse,
                    "delta_mse_vs_b3": model_metric["mse"] - b3_mse,
                    "rmse": model_metric["rmse"],
                    "mae": model_metric["mae"],
                    "bias": model_metric["bias"],
                    "origin_result_vs_b3": result_text,
                    "fallback_count": sum(row["local_fallback_used"] == "yes" for row in origin_rows),
                    "mean_abs_correction": mean(abs(float(row["local_correction_value"])) for row in origin_rows),
                }
            )

        city_sse: dict[str, float] = {}
        city_stats: dict[str, dict[str, Any]] = {}
        for city_id, city_group in city_groups.items():
            model_metric = metric_core(city_group)
            b3_sse = math.fsum(float(row["b3_squared_error"]) for row in city_group)
            wins = losses = ties = 0
            deltas: list[float] = []
            for row in city_group:
                result = compare_metrics(float(row["squared_error"]), float(row["b3_squared_error"]))
                wins += result < 0
                losses += result > 0
                ties += result == 0
                deltas.append(float(row["squared_error"]) - float(row["b3_squared_error"]))
            city_sse[city_id] = math.fsum(float(row["squared_error"]) for row in city_group)
            city_stats[city_id] = {
                "metric": model_metric,
                "b3_sse": b3_sse,
                "b3_mse": b3_sse / len(city_group),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mean_delta": mean(deltas),
            }
        total_city_sse = math.fsum(city_sse.values())
        ranked_cities = sorted(city_sse, key=lambda city: (-city_sse[city], city))
        top1 = ranked_cities[:1]
        top3 = ranked_cities[:3]
        latest_rows = origin_groups[2021]
        latest_mse = metric_core(latest_rows)["mse"]
        b3_pooled = mean(float(row["b3_squared_error"]) for row in rows)
        b3_latest = mean(float(row["b3_squared_error"]) for row in latest_rows)
        aggregate_rows.append(
            {
                "model_id": "S5-Plan06H_CITY_LOCAL_SCREENING",
                "candidate_id": candidate_id,
                "framing_id": candidate["framing_id"],
                "target_id": target_id,
                "scenario_id": scenario_id,
                "row_count": core["row_count"],
                "pooled_city_year_mse": core["mse"],
                "macro_origin_mse": mean(origin_mses),
                "latest_2021_mse": latest_mse,
                "rmse": core["rmse"],
                "mae": core["mae"],
                "bias": core["bias"],
                "origin_win_count_vs_b3": origin_win_count,
                "origin_loss_count_vs_b3": origin_loss_count,
                "origin_tie_count_vs_b3": origin_tie_count,
                "b3_pooled_city_year_mse": b3_pooled,
                "b3_latest_2021_mse": b3_latest,
                "delta_pooled_mse_vs_b3": core["mse"] - b3_pooled,
                "delta_latest_2021_mse_vs_b3": latest_mse - b3_latest,
                "total_city_sse": total_city_sse,
                "top_1_city_id": top1[0] if top1 else None,
                "top_1_city_sse": city_sse[top1[0]] if top1 else None,
                "top_1_city_sse_concentration": city_sse[top1[0]] / total_city_sse if top1 else None,
                "top_3_city_ids": top3,
                "top_3_city_sse": math.fsum(city_sse[city] for city in top3),
                "top_3_city_sse_concentration": math.fsum(city_sse[city] for city in top3) / total_city_sse,
                "fallback_count": sum(row["local_fallback_used"] == "yes" for row in rows),
                "mean_abs_correction": mean(abs(float(row["local_correction_value"])) for row in rows),
                "max_abs_correction": max(abs(float(row["local_correction_value"])) for row in rows),
            }
        )
        for city_id in CITIES:
            stats = city_stats[city_id]
            city_rows.append(
                {
                    "model_id": "S5-Plan06H_CITY_LOCAL_SCREENING",
                    "candidate_id": candidate_id,
                    "framing_id": candidate["framing_id"],
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "city_id": city_id,
                    "row_count": stats["metric"]["row_count"],
                    "model_sse": city_sse[city_id],
                    "model_mse": stats["metric"]["mse"],
                    "model_mae": stats["metric"]["mae"],
                    "model_bias": stats["metric"]["bias"],
                    "b3_sse": stats["b3_sse"],
                    "b3_mse": stats["b3_mse"],
                    "delta_sse_vs_b3": city_sse[city_id] - stats["b3_sse"],
                    "mean_delta_mse_vs_b3": stats["mean_delta"],
                    "city_win_count_vs_b3": stats["wins"],
                    "city_loss_count_vs_b3": stats["losses"],
                    "city_tie_count_vs_b3": stats["ties"],
                    "city_sse_rank": ranked_cities.index(city_id) + 1,
                    "city_sse_share": city_sse[city_id] / total_city_sse,
                    "top_1_flag": "yes" if city_id in top1 else "no",
                    "top_3_flag": "yes" if city_id in top3 else "no",
                }
            )
    assert len(aggregate_rows) == len(CANDIDATES) * len(TARGETS) * len(SCENARIOS)
    assert len(per_origin_rows) == len(CANDIDATES) * len(TARGETS) * len(SCENARIOS) * len(ORIGINS)
    assert len(city_rows) == len(CANDIDATES) * len(TARGETS) * len(SCENARIOS) * len(CITIES)
    return aggregate_rows, per_origin_rows, city_rows


def persistence_rows(
    replay_by_key: dict[tuple[str, str, str], list[dict[str, str]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            for city_id in CITIES:
                source = sorted(
                    replay_by_key[(target_id, scenario_id, city_id)],
                    key=lambda item: parse_int(item["pseudo_target_year"]),
                )
                points = [
                    (parse_int(item["pseudo_target_year"]), parse_float(item["actual_target_value"]) - parse_float(item["b3_prediction"]))
                    for item in source
                ]
                points = [(year, residual) for year, residual in points if year is not None and residual is not None]
                assert points
                lag_left: list[float] = []
                lag_right: list[float] = []
                for (previous_year, previous_residual), (year, residual) in zip(points, points[1:]):
                    if year - previous_year == 1:
                        lag_left.append(previous_residual)
                        lag_right.append(residual)
                nonzero_pairs = [
                    (left, right)
                    for left, right in zip(lag_left, lag_right)
                    if left != 0.0 and right != 0.0
                ]
                same_sign_count = sum((left > 0) == (right > 0) for left, right in nonzero_pairs)
                positive_count = sum(residual > 0 for _year, residual in points)
                negative_count = sum(residual < 0 for _year, residual in points)
                zero_count = len(points) - positive_count - negative_count
                rows.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "city_id": city_id,
                        "residual_row_count": len(points),
                        "residual_year_min": min(year for year, _residual in points),
                        "residual_year_max": max(year for year, _residual in points),
                        "historical_mean_residual": mean(residual for _year, residual in points),
                        "historical_residual_std": float(np.std([residual for _year, residual in points], ddof=0)),
                        "latest_residual": points[-1][1],
                        "latest_residual_year": points[-1][0],
                        "positive_residual_share": positive_count / len(points),
                        "negative_residual_share": negative_count / len(points),
                        "zero_residual_share": zero_count / len(points),
                        "lag1_pair_count": len(lag_left),
                        "lag1_residual_correlation": pearson(lag_left, lag_right),
                        "nonzero_sign_pair_count": len(nonzero_pairs),
                        "lag1_same_sign_count": same_sign_count,
                        "lag1_sign_persistence": same_sign_count / len(nonzero_pairs) if nonzero_pairs else None,
                    }
                )
    assert len(rows) == len(TARGETS) * len(SCENARIOS) * len(CITIES)
    return rows


def classify_candidates(aggregate_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    by_key = {
        (row["candidate_id"], row["target_id"], row["scenario_id"]): row for row in aggregate_rows
    }
    classifications: dict[str, str] = {}
    details: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        resident_rows = [
            by_key[(candidate_id, "resident", scenario_id)] for scenario_id in SCENARIOS
        ]
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
            classification = "city_local_signal_found"
        elif within_one_percent:
            classification = "city_local_competitive_but_unstable"
        else:
            classification = "city_local_unsupported"
        classifications[candidate_id] = classification
        details[candidate_id] = {
            "resident_beats_b3_pooled_in_both_scenarios": beats_b3_both,
            "resident_majority_origin_wins_in_both_scenarios": majority_wins_both,
            "resident_within_one_percent_of_b3_in_at_least_one_scenario": within_one_percent,
            "classification_signal": stable_signal,
            "resident_scenarios": [
                {
                    "scenario_id": row["scenario_id"],
                    "pooled_city_year_mse": row["pooled_city_year_mse"],
                    "b3_pooled_city_year_mse": row["b3_pooled_city_year_mse"],
                    "origin_win_count_vs_b3": row["origin_win_count_vs_b3"],
                }
                for row in resident_rows
            ],
        }
    return classifications, {
        "rule": (
            "city_local_signal_found requires one fixed candidate to beat B3 on resident pooled MSE in both R1 and R2 "
            "and have at least 6/11 origin wins in both; within 1% without that stability is competitive_but_unstable."
        ),
        "candidate_checks": details,
    }


def all144_status(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in aggregate_rows
        if row["candidate_id"] == "LOCAL_ALL144_RIDGE_RESIDUAL" and row["target_id"] == "resident"
    ]
    deltas = {row["scenario_id"]: row["delta_pooled_mse_vs_b3"] for row in rows}
    if all(delta < 0 for delta in deltas.values()):
        status = "helps"
    elif all(delta > 0 for delta in deltas.values()):
        status = "collapses"
    else:
        status = "mixed"
    return {
        "status": status,
        "reason": "resident R1/R2 pooled MSE delta vs B3; the local n is much smaller than the 144 raw features",
        "resident_scenario_deltas": deltas,
        "feature_count": 144,
    }


def choose_best_candidate(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in aggregate_rows:
        if row["target_id"] == "resident":
            scores[row["candidate_id"]].append(row["pooled_city_year_mse"])
    ranked = sorted((mean(values), candidate_id) for candidate_id, values in scores.items())
    best_score, best_candidate = ranked[0]
    return {
        "candidate_id": best_candidate,
        "mean_resident_r1_r2_pooled_mse": best_score,
        "ranking": [
            {"candidate_id": candidate_id, "mean_resident_r1_r2_pooled_mse": score}
            for score, candidate_id in ranked
        ],
        "status": "diagnostic_best_only_no_global_winner_or_routing",
    }


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def make_handoff(
    worker_commit: str,
    aggregate_rows: list[dict[str, Any]],
    classifications: dict[str, str],
    best_candidate: dict[str, Any],
    all144: dict[str, Any],
    persistence: list[dict[str, Any]],
    fallback_counts: dict[str, int],
) -> str:
    lines = [
        f"# {PLAN_ID} handoff",
        "",
        "## 结论",
        "",
        "本 worker 完成严格 City-Local 独立训练与 B3 残差校准筛选。每个外层验证单元只使用同一 `target_id + scenario_id + city_id` 的合法历史 replay 行；所有统计量、imputer、scaler、Ridge 和模型标签均未跨城市共享。结果仅用于证据筛选，不产生全局 winner、路由、融合或提交预测。",
        "",
        f"- 分支：`{BRANCH}`；精确基线：`{BASE_SHA}`；runner commit：`{worker_commit}`。",
        f"- 覆盖：6 candidates × 2 targets × 2 scenarios × 11 origins × 40 cities = {len(aggregate_rows) * 440} OOF rows；每 candidate 为 1760 行。",
        f"- 诊断最优（resident R1/R2 pooled MSE 均值，仅诊断）：`{best_candidate['candidate_id']}`，均值 `{fmt(best_candidate['mean_resident_r1_r2_pooled_mse'])}`。",
        f"- all-144 Ridge 在 n 远小于 p 的本地几何下：`{all144['status']}`；详细 delta 见 `aggregate_metrics.csv` 与 `model_summary.json`。",
        "",
        "## 固定候选与规则",
        "",
        "| candidate | framing | 固定规则 | fallback 行数 |",
        "|---|---|---|---:|",
    ]
    for candidate in CANDIDATES:
        lines.append(
            f"| `{candidate['candidate_id']}` | `{candidate['framing_id']}` | {candidate['definition']} | {fallback_counts[candidate['candidate_id']]} |"
        )
    lines.extend(
        [
            "",
            "Ridge 候选严格使用 `SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)` + `StandardScaler()` + `Ridge(alpha=1000.0)`；FG-P 为冻结的 17 个特征，all-144 为冻结的 144 个数值特征。没有 inner tuning、per-city tuning 或其他城市样本。",
            "",
            "## 全量 pooled 结果",
            "",
            "| candidate | target | scenario | MSE | B3 MSE | delta vs B3 | origin wins/11 | latest 2021 MSE | top-3 city SSE share |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(aggregate_rows, key=lambda item: (item["candidate_id"], item["target_id"], item["scenario_id"])):
        lines.append(
            f"| `{row['candidate_id']}` | {row['target_id']} | {row['scenario_id']} | {fmt(row['pooled_city_year_mse'])} | {fmt(row['b3_pooled_city_year_mse'])} | {fmt(row['delta_pooled_mse_vs_b3'])} | {row['origin_win_count_vs_b3']} | {fmt(row['latest_2021_mse'])} | {fmt(row['top_3_city_sse_concentration'])} |"
        )
    mean_lag_corr = [row["lag1_residual_correlation"] for row in persistence if row["lag1_residual_correlation"] is not None]
    mean_sign_persistence = [row["lag1_sign_persistence"] for row in persistence if row["lag1_sign_persistence"] is not None]
    lines.extend(
        [
            "",
            "## 信号分类与残差持久性",
            "",
            f"分类规则：{classify_candidates.__name__} 中冻结的 Plan05 family rule；`city_local_signal_found` 要求同一 candidate 在 resident 的 R1/R2 均优于 B3 且各自至少 6/11 个 origin 胜出；仅在任一场景距 B3 不超过 1% 但不满足稳定条件时记为 `city_local_competitive_but_unstable`。",
            "",
            "| candidate | classification |",
            "|---|---|",
        ]
    )
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        lines.append(f"| `{candidate_id}` | `{classifications[candidate_id]}` |")
    lines.extend(
        [
            "",
            f"B3 residual persistence diagnostics 覆盖 {len(persistence)} 个 target/scenario/city 单元；可估计 lag-1 相关的单元平均相关系数为 `{fmt(mean(mean_lag_corr) if mean_lag_corr else None)}`，可估计非零符号持久性的单元平均值为 `{fmt(mean(mean_sign_persistence) if mean_sign_persistence else None)}`。这些值仅作诊断，不用于 routing 或 candidate selection。",
            "",
            "## 证据边界",
            "",
            "- `city_local_training_geometry.csv` 逐 outer city-year 记录 cutoff、训练行数、同城检查和阈值；`correction_diagnostics.csv` 记录校准量、fallback 与 B3 对照。",
            "- 明确验证：任何 city-specific fit/preprocessing 都只接收同城且不晚于 `origin_year-2` 的 replay 行；没有外层验证标签进入训练。",
            "- 本计划未执行 routing、ensemble/blend/gating、2023、submission、checkpoint、S6，也未修改 requirements。",
            "",
            "## 产物",
            "",
            "- `oof_predictions.csv`",
            "- `aggregate_metrics.csv`",
            "- `per_origin_metrics.csv`",
            "- `city_model_error.csv`",
            "- `city_local_training_geometry.csv`",
            "- `residual_persistence_diagnostics.csv`",
            "- `correction_diagnostics.csv`",
            "- `model_summary.json`",
            "- `HANDOFF.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    started_at = time.perf_counter()
    worker_commit = assert_worker_identity()
    contracts, model_features = load_contracts()
    formal_header, formal_rows, _ = read_csv_table(FORMAL_MATRIX_PATH)
    replay_header, replay_rows, _ = read_csv_table(REPLAY_MATRIX_PATH)
    formal_by_key, replay_by_key = validate_and_index_rows(
        formal_header,
        formal_rows,
        replay_header,
        replay_rows,
        model_features,
    )
    geometry_rows, local_rows_by_validation = build_geometry(formal_by_key, replay_by_key)
    oof_rows, correction_rows = build_oof(
        formal_by_key,
        local_rows_by_validation,
        model_features,
        worker_commit,
    )
    aggregate_rows, per_origin_rows, city_rows = metric_group_rows(oof_rows)
    persistence = persistence_rows(replay_by_key)
    classifications, classification_details = classify_candidates(aggregate_rows)
    best_candidate = choose_best_candidate(aggregate_rows)
    all144 = all144_status(aggregate_rows)
    fallback_counts = {
        candidate["candidate_id"]: sum(
            row["candidate_id"] == candidate["candidate_id"] and row["local_fallback_used"] == "yes"
            for row in oof_rows
        )
        for candidate in CANDIDATES
    }

    oof_fields = contracts["oof_fields"] + [
        "candidate_id",
        "local_training_row_count",
        "local_fallback_used",
        "local_correction_value",
    ]
    write_csv(OUTPUT_DIR / "oof_predictions.csv", oof_rows, oof_fields)
    write_csv(
        OUTPUT_DIR / "aggregate_metrics.csv",
        aggregate_rows,
        (
            "model_id",
            "candidate_id",
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
            "fallback_count",
            "mean_abs_correction",
            "max_abs_correction",
        ),
    )
    write_csv(
        OUTPUT_DIR / "per_origin_metrics.csv",
        per_origin_rows,
        (
            "model_id",
            "candidate_id",
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
            "fallback_count",
            "mean_abs_correction",
        ),
    )
    write_csv(
        OUTPUT_DIR / "city_model_error.csv",
        city_rows,
        (
            "model_id",
            "candidate_id",
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
            "mean_delta_mse_vs_b3",
            "city_win_count_vs_b3",
            "city_loss_count_vs_b3",
            "city_tie_count_vs_b3",
            "city_sse_rank",
            "city_sse_share",
            "top_1_flag",
            "top_3_flag",
        ),
    )
    write_csv(
        OUTPUT_DIR / "city_local_training_geometry.csv",
        geometry_rows,
        (
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "outer_cutoff_year",
            "local_training_row_count",
            "valid_residual_row_count",
            "local_training_min_year",
            "local_training_max_year",
            "local_city_count",
            "local_city_ids",
            "city_only_pass",
            "cutoff_pass",
            "mean_threshold_pass",
            "trend_threshold_pass",
            "ridge_threshold_pass",
        ),
    )
    write_csv(
        OUTPUT_DIR / "residual_persistence_diagnostics.csv",
        persistence,
        (
            "target_id",
            "scenario_id",
            "city_id",
            "residual_row_count",
            "residual_year_min",
            "residual_year_max",
            "historical_mean_residual",
            "historical_residual_std",
            "latest_residual",
            "latest_residual_year",
            "positive_residual_share",
            "negative_residual_share",
            "zero_residual_share",
            "lag1_pair_count",
            "lag1_residual_correlation",
            "nonzero_sign_pair_count",
            "lag1_same_sign_count",
            "lag1_sign_persistence",
        ),
    )
    write_csv(
        OUTPUT_DIR / "correction_diagnostics.csv",
        correction_rows,
        (
            "candidate_id",
            "framing_id",
            "target_id",
            "scenario_id",
            "origin_year",
            "city_id",
            "local_training_row_count",
            "valid_residual_row_count",
            "local_training_min_year",
            "local_training_max_year",
            "outer_cutoff_year",
            "local_fallback_used",
            "fit_method",
            "local_correction_value",
            "absolute_correction_value",
            "actual_target_value",
            "b3_prediction",
            "prediction_value",
            "model_squared_error",
            "b3_squared_error",
            "delta_squared_error_vs_b3",
        ),
    )

    model_summary = {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "branch": BRANCH,
        "required_exact_base_sha": BASE_SHA,
        "runner_commit": worker_commit,
        "contracts": {
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "formal_row_count": len(formal_rows),
            "replay_row_count": len(replay_rows),
            "model_numeric_feature_count": len(model_features),
            "fgp_feature_count": len(FGP_FEATURES),
            "oof_row_count": len(oof_rows),
            "oof_row_count_per_candidate": len(oof_rows) // len(CANDIDATES),
            "candidate_count": len(CANDIDATES),
            "official_coverage": "2 targets x 2 scenarios x 11 origins x 40 cities",
        },
        "candidate_definitions": list(CANDIDATES),
        "frozen_constants": {
            "outer_cutoff_rule": "pseudo_target_year <= origin_year - 2",
            "ewma_decay": EWMA_DECAY,
            "minimum_mean_residual_rows": MIN_MEAN_ROWS,
            "minimum_ridge_rows": MIN_RIDGE_ROWS,
            "ridge_alpha": RIDGE_ALPHA,
            "fgp_features": list(FGP_FEATURES),
        },
        "preprocessing": {
            "imputer": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)",
            "scaler": "StandardScaler()",
            "estimator": "Ridge(alpha=1000.0)",
            "fit_scope": "same target/scenario/city and legal replay rows only",
            "city_id_as_feature": False,
        },
        "classification": classifications,
        "classification_details": classification_details,
        "best_city_local_candidate_diagnostic": best_candidate,
        "all144_ridge_assessment": all144,
        "fallback_counts": fallback_counts,
        "aggregate_metrics": aggregate_rows,
        "residual_persistence_summary": {
            "diagnostic_row_count": len(persistence),
            "mean_estimable_lag1_correlation": mean(
                [row["lag1_residual_correlation"] for row in persistence if row["lag1_residual_correlation"] is not None]
            )
            if any(row["lag1_residual_correlation"] is not None for row in persistence)
            else None,
            "mean_estimable_sign_persistence": mean(
                [row["lag1_sign_persistence"] for row in persistence if row["lag1_sign_persistence"] is not None]
            )
            if any(row["lag1_sign_persistence"] is not None for row in persistence)
            else None,
        },
        "invariants": {
            "no_other_city_row_entered_city_specific_fit_or_preprocessing": True,
            "no_outer_validation_label_entered_training": True,
            "all_geometry_city_only_pass": all(row["city_only_pass"] == "yes" for row in geometry_rows),
            "all_geometry_cutoff_pass": all(row["cutoff_pass"] == "yes" for row in geometry_rows),
            "no_global_imputation": True,
        },
        "forbidden_actions_status": {
            "routing": False,
            "ensemble_or_blend": False,
            "gating": False,
            "prediction_2023": False,
            "submission": False,
            "checkpoint": False,
            "requirements_modified": False,
            "s5_complete": False,
            "s6_entered": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "total_runtime_seconds": time.perf_counter() - started_at,
        },
        "artifacts": [
            "HANDOFF.md",
            "oof_predictions.csv",
            "aggregate_metrics.csv",
            "per_origin_metrics.csv",
            "city_model_error.csv",
            "city_local_training_geometry.csv",
            "residual_persistence_diagnostics.csv",
            "correction_diagnostics.csv",
            "model_summary.json",
        ],
    }
    write_json(OUTPUT_DIR / "model_summary.json", model_summary)
    (OUTPUT_DIR / "HANDOFF.md").write_text(
        make_handoff(
            worker_commit,
            aggregate_rows,
            classifications,
            best_candidate,
            all144,
            persistence,
            fallback_counts,
        ),
        encoding="utf-8",
    )

    print(f"classification={stable_json(classifications)}")
    print(f"oof_rows={len(oof_rows)}")
    print(f"fallback_counts={stable_json(fallback_counts)}")
    print(f"worker_commit={worker_commit}")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"runtime_seconds={time.perf_counter() - started_at:.3f}")


if __name__ == "__main__":
    main()
