"""Build the S5-Plan04B historical training-replay foundation.

This runner replays the exact S5-Plan04 feature contract for every historical
pseudo-target event whose target label and frozen B3 reference are available.
It then emits deterministic outer-training and inner-selection split geometry.

The runner deliberately stops before model work: it does not fit an imputer,
scaler, encoder, selector, estimator, gate, blend, 2023 forecast, or
submission.  Plan04 remains the single implementation of the 144 feature
definitions and source-specific as-of boundaries.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import build_s5_plan04_full_multisource_feature_foundation as plan04


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S5-Plan04B"
STAGE = "S5_EVIDENCE_DRIVEN_ITERATION"
BASE_SHA = "fac7ed9bf11459bc5d7c6772857683417bf70eac"
BRANCH = "s5/plan04b-training-replay-foundation"
FEATURE_CONTRACT_VERSION = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT_VERSION = "S5-Plan04B-TRAINING-REPLAY-V1"
OUTPUT_DIR = ROOT / "reports" / "S5_HANDOFFS" / PLAN_ID
FORMAL_MATRIX_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "full_multisource_feature_matrix.csv"
FORMAL_SCHEMA_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
OOF_CONTRACT_PATH = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "oof_prediction_contract.csv"

TARGETS = plan04.TARGETS
SCENARIOS = plan04.SCENARIOS
CITIES = plan04.CITIES
FORMAL_ORIGINS = plan04.COMMON_ORIGINS
HORIZON_YEARS = plan04.HORIZON_YEARS

REPLAY_IDENTITY_COLUMNS = (
    "target_id",
    "scenario_id",
    "pseudo_target_year",
    "city_id",
    "target_history_cutoff_year",
)
REPLAY_FIELDS = (
    *REPLAY_IDENTITY_COLUMNS,
    *plan04.LABEL_COLUMNS,
    *plan04.REFERENCE_COLUMNS,
    *plan04.MODEL_FEATURES,
    "feature_contract_version",
)

OUTER_FIELDS = (
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
)

INNER_FIELDS = (
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
)

REPLAY_AUDIT_FIELDS = (
    "target_id",
    "scenario_id",
    "pseudo_target_year",
    "city_id",
    "source_id",
    "required_lag_years",
    "required_cap_year",
    "domain_type",
    "allowed_domain_years",
    "latest_used_year",
    "source_age_years",
    "usable_observation_count",
    "used_years",
    "used_year_max",
    "cap_violation_count",
    "benchmark_domain_violation_count",
    "quarantine_keys_as_of_cap",
    "quarantine_used_count",
    "derived_used_count",
    "provenance_state",
    "availability_state",
    "missing_reason",
    "used_years_within_cap",
    "quarantine_reintroduction_status",
    "benchmark_domain_status",
    "population_history_cutoff_year",
    "population_history_used_year_max",
    "population_history_cutoff_pass",
    "b3_history_years",
    "b3_history_used_year_max",
    "b3_history_cutoff_pass",
    "b3_prediction_available",
    "audit_status",
)


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]] | None = None,
    fields: Iterable[str] | None = None,
    model_features: set[str] | None = None,
) -> None:
    plan04.write_csv(path, rows or [], fields=fields, model_features=model_features)


def parse_int(value: Any) -> int | None:
    return plan04.as_int(value)


def parse_float(value: Any) -> float | None:
    return plan04.as_float(value)


def value_equal(left: Any, right: Any, *, numeric: bool = False) -> bool:
    if numeric:
        try:
            left_number = parse_float(left)
            right_number = parse_float(right)
        except (TypeError, ValueError):
            # Some Plan04 audit fields are structured text (for example
            # ``used_years``) even though the surrounding audit block is
            # mostly numeric.  Preserve exact text comparison for those
            # fields instead of weakening the compatibility check.
            return value_equal(left, right, numeric=False)
        if left_number is None or right_number is None:
            return left_number is None and right_number is None
        return abs(left_number - right_number) <= 1e-12
    if not plan04.present(left) or not plan04.present(right):
        return not plan04.present(left) and not plan04.present(right)
    return str(left) == str(right)


def assert_integration_gate() -> None:
    assert git_text("branch", "--show-current") == BRANCH
    assert git_text("rev-parse", "origin/main") == BASE_SHA
    assert git_text("merge-base", BASE_SHA, "HEAD") == BASE_SHA
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Plan04 integrated SHA must be an ancestor of the worker tip"
    schema = json.loads(FORMAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert len(schema["model_features"]) == 144
    assert OOF_CONTRACT_PATH.exists()


def load_context() -> dict[str, Any]:
    # Plan04 owns the implementation and its loader checks the current worker
    # identity.  Override only those runtime identity constants; no Plan04
    # source or feature definition is copied or edited.
    plan04.BASE_SHA = BASE_SHA
    plan04.BRANCH = BRANCH
    context = plan04.load_context()
    assert len(plan04.MODEL_FEATURES) == 144
    return context


def target_year_domain(context: dict[str, Any], target_id: str) -> list[int]:
    years = {
        year
        for city in CITIES
        for year in context["population"][target_id][city]
    }
    assert years
    return sorted(years)


def candidate_to_replay_row(candidate: dict[str, Any]) -> dict[str, Any]:
    row = {
        field: candidate[field]
        for field in REPLAY_FIELDS
        if field != "pseudo_target_year"
    }
    row["pseudo_target_year"] = candidate["origin_year"]
    assert set(row) == set(REPLAY_FIELDS), sorted(set(row) ^ set(REPLAY_FIELDS))
    return {field: row[field] for field in REPLAY_FIELDS}


def b3_history_years(
    context: dict[str, Any], target_id: str, city: str, pseudo_target_year: int
) -> list[int]:
    series = context["population"][target_id][city]
    eligible = sorted(year for year in series if year <= pseudo_target_year - HORIZON_YEARS)
    return eligible[-3:] if len(eligible) >= 3 else eligible[-2:]


def replay_rows_and_coverage(
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_years = sorted(
        {
            year
            for target_id in TARGETS
            for year in target_year_domain(context, target_id)
        }
    )
    eligible_internal: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    excluded_total = Counter()
    excluded_by_pair = Counter()
    row_count_by_pair: Counter[tuple[str, str]] = Counter()
    row_count_by_pair_year: Counter[tuple[str, str, int]] = Counter()
    label_count_by_pair_year: Counter[tuple[str, str, int]] = Counter()
    city_coverage_by_year: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    for target_id in TARGETS:
        target_series = context["population"][target_id]
        target_years = set(target_year_domain(context, target_id))
        for scenario_id in SCENARIOS:
            for pseudo_target_year in all_years:
                excluded: list[tuple[str, str]] = []
                eligible_for_year: list[dict[str, Any]] = []
                fallback_count = 0
                recent3_count = 0
                feature_generation_failure_count = 0
                for city in CITIES:
                    actual = target_series[city].get(pseudo_target_year)
                    if actual is None:
                        excluded.append((city, "no_target_label"))
                        continue
                    label_count_by_pair_year[(target_id, scenario_id, pseudo_target_year)] += 1
                    try:
                        candidate = plan04.build_row(
                            context,
                            target_id,
                            scenario_id,
                            pseudo_target_year,
                            city,
                        )
                        if set(candidate) != set(plan04.MATRIX_FIELDS):
                            raise AssertionError("feature_vector_schema_mismatch")
                    except Exception as exc:  # pragma: no cover - failure evidence is reported
                        feature_generation_failure_count += 1
                        reason = f"feature_generation_error:{type(exc).__name__}"
                        excluded.append((city, reason))
                        continue
                    if candidate["b3_prediction"] is None:
                        excluded.append((city, "b3_prediction_unavailable"))
                        continue
                    eligible_for_year.append(candidate)
                    history_count = int(candidate["pop_history_observation_count"] or 0)
                    if history_count >= 3:
                        recent3_count += 1
                    else:
                        fallback_count += 1

                eligible_internal.extend(eligible_for_year)
                row_count_by_pair[(target_id, scenario_id)] += len(eligible_for_year)
                row_count_by_pair_year[(target_id, scenario_id, pseudo_target_year)] += len(
                    eligible_for_year
                )
                reasons = Counter(reason for _, reason in excluded)
                excluded_total.update(reasons)
                excluded_by_pair.update(
                    {(target_id, scenario_id, reason): count for reason, count in reasons.items()}
                )
                coverage_rows.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "pseudo_target_year": pseudo_target_year,
                        "target_label_city_count": sum(
                            target_series[city].get(pseudo_target_year) is not None for city in CITIES
                        ),
                        "replay_row_count": len(eligible_for_year),
                        "eligible_city_count": len(eligible_for_year),
                        "excluded_target_label_city_count": sum(
                            reason != "no_target_label" for _, reason in excluded
                        ),
                        "excluded_no_target_label_city_count": reasons.get("no_target_label", 0),
                        "excluded_reason_counts": stable_json(dict(sorted(reasons.items()))),
                        "excluded_city_reasons": ";".join(
                            f"{city}:{reason}" for city, reason in excluded if reason != "no_target_label"
                        ),
                        "b3_recent3_city_count": recent3_count,
                        "b3_fallback_city_count": fallback_count,
                        "feature_generation_failure_count": feature_generation_failure_count,
                        "city_coverage_rate": len(eligible_for_year) / len(CITIES),
                        "replay_eligibility_status": (
                            "eligible_rows_present" if eligible_for_year else "no_eligible_rows"
                        ),
                    }
                )
                city_coverage_by_year[target_id].setdefault(
                    scenario_id,
                    {},
                )[str(pseudo_target_year)] = len(eligible_for_year)

    replay_rows = [candidate_to_replay_row(row) for row in eligible_internal]
    keys = [
        (row["target_id"], row["scenario_id"], row["pseudo_target_year"], row["city_id"])
        for row in replay_rows
    ]
    assert len(keys) == len(set(keys))
    assert replay_rows
    assert all(row["feature_contract_version"] == FEATURE_CONTRACT_VERSION for row in replay_rows)

    pair_keys = {
        (row["target_id"], row["scenario_id"], row["pseudo_target_year"], row["city_id"])
        for row in replay_rows
    }
    shared_universe = all(
        {
            (year, city)
            for t, s, year, city in pair_keys
            if t == target_id and s == scenario_id
        }
        == {
            (year, city)
            for t, s, year, city in pair_keys
            if t == target_id and s == SCENARIOS[0]
        }
        for target_id in TARGETS
        for scenario_id in SCENARIOS[1:]
    )
    assert shared_universe

    coverage = {
        "all_candidate_years": all_years,
        "target_year_domains": {
            target_id: target_year_domain(context, target_id) for target_id in TARGETS
        },
        "row_count_by_target_scenario": {
            f"{target_id}|{scenario_id}": row_count_by_pair[(target_id, scenario_id)]
            for target_id in TARGETS
            for scenario_id in SCENARIOS
        },
        "row_count_by_target_scenario_year": {
            f"{target_id}|{scenario_id}|{year}": row_count_by_pair_year[
                (target_id, scenario_id, year)
            ]
            for target_id in TARGETS
            for scenario_id in SCENARIOS
            for year in all_years
        },
        "city_coverage_by_target_scenario_year": city_coverage_by_year,
        "excluded_label_rows_total": sum(
            count for reason, count in excluded_total.items() if reason != "no_target_label"
        ),
        "excluded_rows_by_reason": dict(sorted(excluded_total.items())),
        "excluded_rows_by_target_scenario_reason": {
            f"{target_id}|{scenario_id}|{reason}": count
            for (target_id, scenario_id, reason), count in sorted(excluded_by_pair.items())
        },
        "direct_residual_shared_replay_universe": shared_universe,
        "replay_pseudo_target_year_min": min(row["pseudo_target_year"] for row in replay_rows),
        "replay_pseudo_target_year_max": max(row["pseudo_target_year"] for row in replay_rows),
        "replay_pseudo_target_years": sorted({row["pseudo_target_year"] for row in replay_rows}),
    }
    return replay_rows, coverage_rows, coverage


def formal_compatibility(
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    existing = plan04.read_csv(FORMAL_MATRIX_PATH)
    generated = [
        plan04.build_row(context, target_id, scenario_id, origin_year, city)
        for target_id in TARGETS
        for scenario_id in SCENARIOS
        for origin_year in FORMAL_ORIGINS
        for city in CITIES
    ]
    assert len(existing) == 1760
    assert len(generated) == 1760
    existing_index = {
        (
            row["target_id"],
            row["scenario_id"],
            int(row["origin_year"]),
            row["city_id"],
        ): row
        for row in existing
    }
    generated_index = {
        (row["target_id"], row["scenario_id"], row["origin_year"], row["city_id"]): row
        for row in generated
    }
    keys = sorted(set(existing_index) | set(generated_index))
    compatibility_rows: list[dict[str, Any]] = []
    for target_id in [*TARGETS, "ALL"]:
        target_scenarios = SCENARIOS if target_id != "ALL" else ["ALL"]
        for scenario_id in target_scenarios:
            scoped_keys = [
                key
                for key in keys
                if target_id == "ALL"
                or (key[0] == target_id and key[1] == scenario_id)
            ]
            key_mismatch = sum(
                key not in existing_index or key not in generated_index for key in scoped_keys
            )
            model_mismatch = 0
            label_reference_mismatch = 0
            audit_mismatch = 0
            identity_mismatch = 0
            for key in scoped_keys:
                left = existing_index.get(key)
                right = generated_index.get(key)
                if left is None or right is None:
                    continue
                for field in plan04.MODEL_FEATURES:
                    if not value_equal(left.get(field), right.get(field), numeric=True):
                        model_mismatch += 1
                for field in (*plan04.LABEL_COLUMNS, *plan04.REFERENCE_COLUMNS):
                    if not value_equal(left.get(field), right.get(field), numeric=True):
                        label_reference_mismatch += 1
                for field in plan04.IDENTITY_COLUMNS:
                    if field == "origin_year":
                        expected = int(left[field])
                        observed = int(right[field])
                        same = expected == observed
                    elif field == "target_history_cutoff_year":
                        same = int(left[field]) == int(right[field])
                    else:
                        same = value_equal(left.get(field), right.get(field))
                    if not same:
                        identity_mismatch += 1
                for field in plan04.AUDIT_COLUMNS:
                    numeric = field != "feature_contract_version"
                    if not value_equal(left.get(field), right.get(field), numeric=numeric):
                        audit_mismatch += 1
            compatibility_rows.append(
                {
                    "target_id": target_id,
                    "scenario_id": scenario_id,
                    "existing_row_count": sum(
                        key in existing_index for key in scoped_keys
                    ),
                    "generated_row_count": sum(
                        key in generated_index for key in scoped_keys
                    ),
                    "key_mismatch_count": key_mismatch,
                    "identity_mismatch_count": identity_mismatch,
                    "model_feature_mismatch_cell_count": model_mismatch,
                    "label_reference_mismatch_cell_count": label_reference_mismatch,
                    "audit_mismatch_cell_count": audit_mismatch,
                    "feature_contract_version": FEATURE_CONTRACT_VERSION,
                    "status": "PASS"
                    if key_mismatch == 0
                    and identity_mismatch == 0
                    and model_mismatch == 0
                    and label_reference_mismatch == 0
                    and audit_mismatch == 0
                    else "FAIL",
                }
            )
    overall = next(
        row
        for row in compatibility_rows
        if row["target_id"] == "ALL" and row["scenario_id"] == "ALL"
    )
    assert overall["status"] == "PASS"
    assert len([row for row in compatibility_rows if row["status"] == "PASS"]) == 5
    return compatibility_rows, overall, generated


def outer_training_geometry(
    replay_rows: list[dict[str, Any]],
    formal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            validation_rows = [
                row
                for row in formal_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            assert len(validation_rows) == len(FORMAL_ORIGINS) * len(CITIES)
            for outer_origin_year in FORMAL_ORIGINS:
                training_max = outer_origin_year - HORIZON_YEARS
                training = [
                    row
                    for row in replay_rows
                    if row["target_id"] == target_id
                    and row["scenario_id"] == scenario_id
                    and row["pseudo_target_year"] <= training_max
                ]
                validation = [
                    row
                    for row in validation_rows
                    if int(row["origin_year"]) == outer_origin_year
                ]
                forbidden_count = sum(
                    row["pseudo_target_year"] > training_max for row in training
                )
                boundary_pass = (
                    len(validation) == len(CITIES)
                    and len({row["city_id"] for row in validation}) == len(CITIES)
                    and forbidden_count == 0
                )
                result.append(
                    {
                        "target_id": target_id,
                        "scenario_id": scenario_id,
                        "outer_origin_year": outer_origin_year,
                        "training_max_pseudo_target_year": training_max,
                        "training_min_pseudo_target_year": min(
                            (row["pseudo_target_year"] for row in training), default=None
                        ),
                        "training_row_count": len(training),
                        "training_city_count": len({row["city_id"] for row in training}),
                        "training_year_count": len(
                            {row["pseudo_target_year"] for row in training}
                        ),
                        "validation_row_count": len(validation),
                        "validation_city_count": len({row["city_id"] for row in validation}),
                        "forbidden_training_rows_after_cutoff": forbidden_count,
                        "boundary_pass": "PASS" if boundary_pass else "FAIL",
                    }
                )
    assert len(result) == len(TARGETS) * len(SCENARIOS) * len(FORMAL_ORIGINS)
    summary = {
        "row_count": len(result),
        "all_training_pools_nonempty": all(row["training_row_count"] > 0 for row in result),
        "all_validation_groups_have_40_rows": all(
            row["validation_row_count"] == 40 for row in result
        ),
        "all_validation_groups_have_40_cities": all(
            row["validation_city_count"] == 40 for row in result
        ),
        "boundary_violation_count": sum(row["boundary_pass"] != "PASS" for row in result),
        "row_count_by_outer_origin": {
            str(origin): sum(row["training_row_count"] for row in result if row["outer_origin_year"] == origin)
            for origin in FORMAL_ORIGINS
        },
    }
    assert summary["all_training_pools_nonempty"]
    assert summary["all_validation_groups_have_40_rows"]
    assert summary["all_validation_groups_have_40_cities"]
    assert summary["boundary_violation_count"] == 0
    return result, summary


def inner_split_geometry(
    replay_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id in TARGETS:
        for scenario_id in SCENARIOS:
            scoped = [
                row
                for row in replay_rows
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            ]
            available_years = sorted({row["pseudo_target_year"] for row in scoped})
            for outer_origin_year in FORMAL_ORIGINS:
                outer_training = [
                    row
                    for row in scoped
                    if row["pseudo_target_year"] <= outer_origin_year - HORIZON_YEARS
                ]
                for inner_validation_year in available_years:
                    if inner_validation_year > outer_origin_year - HORIZON_YEARS:
                        continue
                    inner_training_max = inner_validation_year - HORIZON_YEARS
                    inner_training = [
                        row
                        for row in outer_training
                        if row["pseudo_target_year"] <= inner_training_max
                    ]
                    inner_validation = [
                        row
                        for row in outer_training
                        if row["pseudo_target_year"] == inner_validation_year
                    ]
                    rows_after_cutoff = sum(
                        row["pseudo_target_year"] > inner_training_max for row in inner_training
                    )
                    boundary_pass = (
                        inner_validation
                        and all(
                            row["pseudo_target_year"] <= inner_training_max
                            for row in inner_training
                        )
                        and all(
                            row["pseudo_target_year"] == inner_validation_year
                            for row in inner_validation
                        )
                    )
                    result.append(
                        {
                            "target_id": target_id,
                            "scenario_id": scenario_id,
                            "outer_origin_year": outer_origin_year,
                            "inner_validation_year": inner_validation_year,
                            "inner_training_max_year": inner_training_max,
                            "inner_training_min_year": min(
                                (row["pseudo_target_year"] for row in inner_training), default=None
                            ),
                            "inner_training_row_count": len(inner_training),
                            "inner_validation_row_count": len(inner_validation),
                            "inner_training_city_count": len(
                                {row["city_id"] for row in inner_training}
                            ),
                            "inner_validation_city_count": len(
                                {row["city_id"] for row in inner_validation}
                            ),
                            "inner_training_year_count": len(
                                {row["pseudo_target_year"] for row in inner_training}
                            ),
                            "outer_training_pool_row_count": len(outer_training),
                            "inner_training_rows_after_cutoff": rows_after_cutoff,
                            "boundary_pass": "PASS" if boundary_pass else "FAIL",
                        }
                    )
    assert result
    summary = {
        "row_count": len(result),
        "boundary_violation_count": sum(row["boundary_pass"] != "PASS" for row in result),
        "zero_training_fold_count": sum(row["inner_training_row_count"] == 0 for row in result),
        "earliest_inner_validation_year_by_target_scenario": {
            f"{target_id}|{scenario_id}": min(
                row["inner_validation_year"]
                for row in result
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            )
            for target_id in TARGETS
            for scenario_id in SCENARIOS
        },
        "latest_inner_validation_year_by_target_scenario": {
            f"{target_id}|{scenario_id}": max(
                row["inner_validation_year"]
                for row in result
                if row["target_id"] == target_id and row["scenario_id"] == scenario_id
            )
            for target_id in TARGETS
            for scenario_id in SCENARIOS
        },
    }
    assert summary["boundary_violation_count"] == 0
    return result, summary


def replay_boundary_audit(
    replay_internal_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_audit = plan04.source_boundary_audit(replay_internal_rows, context)
    internal_index = {
        (
            row["target_id"],
            row["scenario_id"],
            row["origin_year"],
            row["city_id"],
        ): row
        for row in replay_internal_rows
    }
    result: list[dict[str, Any]] = []
    for row in base_audit:
        key = (row["target_id"], row["scenario_id"], row["origin_year"], row["city_id"])
        replay_row = internal_index[key]
        history = b3_history_years(
            context,
            row["target_id"],
            row["city_id"],
            int(row["origin_year"]),
        )
        cutoff = int(row["origin_year"]) - HORIZON_YEARS
        output = dict(row)
        output["pseudo_target_year"] = output.pop("origin_year")
        output.update(
            {
                "population_history_cutoff_year": cutoff,
                "population_history_used_year_max": replay_row.get("population_used_year_max"),
                "population_history_cutoff_pass": replay_row["population_history_cutoff_pass"],
                "b3_history_years": ";".join(str(year) for year in history),
                "b3_history_used_year_max": max(history) if history else None,
                "b3_history_cutoff_pass": "PASS"
                if all(year <= cutoff for year in history)
                else "FAIL",
                "b3_prediction_available": "PASS"
                if replay_row["b3_prediction"] is not None
                else "FAIL",
            }
        )
        output["audit_status"] = (
            "PASS"
            if output["audit_status"] == "PASS"
            and output["population_history_cutoff_pass"] == "PASS"
            and output["b3_history_cutoff_pass"] == "PASS"
            and output["b3_prediction_available"] == "PASS"
            else "FAIL"
        )
        result.append({field: output.get(field) for field in REPLAY_AUDIT_FIELDS})
    assert len(result) == len(replay_rows) * len(plan04.SOURCE_ORDER)
    assert all(row["audit_status"] == "PASS" for row in result)
    summary = {
        "row_count": len(result),
        "source_cap_violation_count": sum(int(row["cap_violation_count"]) for row in result),
        "quarantine_reintroduction_count": sum(
            int(row["quarantine_used_count"]) for row in result
        ),
        "benchmark_domain_violation_count": sum(
            int(row["benchmark_domain_violation_count"]) for row in result
        ),
        "population_history_cutoff_violation_count": sum(
            row["population_history_cutoff_pass"] != "PASS" for row in result
        ),
        "b3_history_cutoff_violation_count": sum(
            row["b3_history_cutoff_pass"] != "PASS" for row in result
        ),
        "audit_failure_count": sum(row["audit_status"] != "PASS" for row in result),
    }
    assert summary["source_cap_violation_count"] == 0
    assert summary["quarantine_reintroduction_count"] == 0
    assert summary["benchmark_domain_violation_count"] == 0
    assert summary["population_history_cutoff_violation_count"] == 0
    assert summary["b3_history_cutoff_violation_count"] == 0
    assert summary["audit_failure_count"] == 0
    return result, summary


def parallel_worker_contract() -> dict[str, Any]:
    metadata_columns = [
        *plan04.LABEL_COLUMNS,
        *plan04.REFERENCE_COLUMNS,
        "target_id",
        "scenario_id",
        "pseudo_target_year",
        "target_history_cutoff_year",
        *plan04.AUDIT_COLUMNS,
    ]
    overlapping_audit_names = sorted(set(plan04.AUDIT_COLUMNS) & set(plan04.MODEL_FEATURES))
    forbidden = sorted(set(metadata_columns) - set(plan04.MODEL_FEATURES) - {"city_id"})
    return {
        "contract_version": REPLAY_CONTRACT_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "base_sha": BASE_SHA,
        "replay_matrix_path": "reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv",
        "formal_validation_matrix_path": "reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv",
        "formal_validation_schema_path": "reports/S5_HANDOFFS/S5-Plan04/feature_matrix_schema.json",
        "identity_columns": list(REPLAY_IDENTITY_COLUMNS[:4]),
        "identity_columns_with_cutoff": list(REPLAY_IDENTITY_COLUMNS),
        "categorical_identity_columns": ["city_id"],
        "model_features": list(plan04.MODEL_FEATURES),
        "model_feature_count": len(plan04.MODEL_FEATURES),
        "replay_matrix_columns": list(REPLAY_FIELDS),
        "feature_groups": {
            group["group_id"]: list(group["feature_names"]) for group in plan04.GROUPS
        },
        "direct_label": "actual_target_value",
        "residual_label": "b3_residual_actual_minus_prediction",
        "b3_reference_prediction_column": "b3_prediction",
        "b3_reference_error_column": "b3_squared_error",
        "outer_origin_domain": list(FORMAL_ORIGINS),
        "outer_training_rule": "pseudo_target_year <= outer_origin_year - 2",
        "outer_validation_rule": "pseudo_target_year == outer_origin_year in the frozen Plan04 formal matrix",
        "inner_split_rule": "inner_training pseudo_target_year <= inner_validation_year - 2; inner validation year belongs to outer training pool",
        "target_ids": list(TARGETS),
        "scenario_ids": list(SCENARIOS),
        "oof_output_schema_path": "reports/S5_HANDOFFS/S5-Plan04/oof_prediction_contract.csv",
        "forbidden_columns_for_X": forbidden,
        "inherited_audit_names_overlapping_model_features": overlapping_audit_names,
        "overlap_resolution": "When an inherited Plan04 audit name overlaps a model-feature name, the model_features list is authoritative; full audit values live in replay_boundary_audit.csv.",
        "allowed_categorical_metadata": ["city_id"],
        "preprocessing_policy": "fit imputer/scaler/encoder/selector only inside the corresponding training fold; Plan04B fits none",
        "raw_missingness_policy": "preserve model-feature NaN; do not globally fill or target-encode city_id",
        "direct_and_residual_share_replay_universe": True,
        "strong_model_training_run": False,
        "2023_prediction_or_submission_run": False,
    }


def make_summary(
    context: dict[str, Any],
    replay_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    outer_summary: dict[str, Any],
    inner_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    compatibility_summary: dict[str, Any],
    b3_summary: dict[str, Any],
) -> dict[str, Any]:
    success_conditions = {
        "plan04_integrated_by_ff_only_gate": git_text("rev-parse", "origin/main") == BASE_SHA,
        "historical_replay_exists_before_earliest_formal_origin": coverage["replay_pseudo_target_year_min"] < min(FORMAL_ORIGINS),
        "exact_144_feature_contract": len(plan04.MODEL_FEATURES) == 144
        and compatibility_summary["model_feature_mismatch_cell_count"] == 0,
        "direct_and_residual_share_replay_universe": coverage["direct_residual_shared_replay_universe"],
        "all_outer_training_pools_nonempty": outer_summary["all_training_pools_nonempty"],
        "all_outer_validation_groups_have_40_cities": outer_summary["all_validation_groups_have_40_cities"],
        "outer_year_boundary_violations_zero": outer_summary["boundary_violation_count"] == 0,
        "inner_year_boundary_violations_zero": inner_summary["boundary_violation_count"] == 0,
        "source_boundary_quarantine_benchmark_zero": all(
            boundary_summary[key] == 0
            for key in (
                "source_cap_violation_count",
                "quarantine_reintroduction_count",
                "benchmark_domain_violation_count",
            )
        ),
        "formal_plan04_compatibility_pass": compatibility_summary["status"] == "PASS",
        "b3_compatibility_pass": b3_summary["pass"],
        "feature_generation_failures_zero": all(
            row["feature_generation_failure_count"] == 0
            for row in coverage["coverage_rows"]
        ),
    }
    ready = all(success_conditions.values())
    return {
        "plan_id": PLAN_ID,
        "stage": STAGE,
        "base_sha": BASE_SHA,
        "worker_branch": BRANCH,
        "build_commit": git_text("rev-parse", "HEAD"),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "plan04_integration": {
            "expected_integrated_sha": BASE_SHA,
            "origin_main_sha": git_text("rev-parse", "origin/main"),
            "fast_forward_gate": True,
        },
        "replay_matrix": {
            "row_count": len(replay_rows),
            "unique_key_count": len(
                {
                    (
                        row["target_id"],
                        row["scenario_id"],
                        row["pseudo_target_year"],
                        row["city_id"],
                    )
                    for row in replay_rows
                }
            ),
            "pseudo_target_year_min": coverage["replay_pseudo_target_year_min"],
            "pseudo_target_year_max": coverage["replay_pseudo_target_year_max"],
            "pseudo_target_year_count": len(coverage["replay_pseudo_target_years"]),
            "pseudo_target_years": coverage["replay_pseudo_target_years"],
            "row_count_by_target_scenario": coverage["row_count_by_target_scenario"],
            "row_count_by_target_scenario_year": coverage["row_count_by_target_scenario_year"],
            "target_year_domains": coverage["target_year_domains"],
            "excluded_label_rows_total": coverage["excluded_label_rows_total"],
            "excluded_rows_by_reason": coverage["excluded_rows_by_reason"],
            "direct_residual_shared_replay_universe": coverage[
                "direct_residual_shared_replay_universe"
            ],
        },
        "outer_training_geometry": outer_summary,
        "inner_split_geometry": inner_summary,
        "formal_plan04_feature_compatibility": compatibility_summary,
        "b3_compatibility": b3_summary,
        "replay_boundary_audit": boundary_summary,
        "success_conditions": success_conditions,
        "boundary_checks": {
            **boundary_summary,
            "outer_year_boundary_violation_count": outer_summary["boundary_violation_count"],
            "inner_year_boundary_violation_count": inner_summary["boundary_violation_count"],
            "annual_interpolation_used": False,
            "global_fill_used": False,
            "imputer_fitted": False,
            "scaler_fitted": False,
            "encoder_fitted": False,
            "feature_selection_by_target_performance": False,
            "strong_model_training_run": False,
            "routing_or_gate_model_run": False,
            "blend_run": False,
            "final_2023_forecast_run": False,
            "submission_run": False,
            "checkpoint_modified": False,
        },
        "feature_count": len(plan04.MODEL_FEATURES),
        "logical_source_group_count": len(plan04.SOURCE_ORDER),
        "semantic_feature_group_count": len(plan04.GROUPS),
        "required_artifacts": [
            "HANDOFF.md",
            "full_multisource_training_replay_matrix.csv",
            "training_replay_summary.json",
            "training_replay_year_coverage.csv",
            "outer_training_geometry.csv",
            "inner_split_geometry.csv",
            "plan04_feature_compatibility.csv",
            "replay_boundary_audit.csv",
            "parallel_worker_input_contract.json",
        ],
        "context_source_status": {
            source_id: context["geometry"][source_id]["source_status"]
            for source_id in plan04.SOURCE_ORDER
        },
        "result": "training_replay_foundation_ready" if ready else "training_geometry_insufficient",
        "s5_complete": False,
        "s6_entered": False,
    }


def handoff_text(summary: dict[str, Any]) -> str:
    replay = summary["replay_matrix"]
    outer = summary["outer_training_geometry"]
    inner = summary["inner_split_geometry"]
    compatibility = summary["formal_plan04_feature_compatibility"]
    boundary = summary["replay_boundary_audit"]
    lines = [
        f"# {PLAN_ID} Handoff — 多表训练回放矩阵与时间切分基座",
        "",
        "## Result",
        "",
        f"- Plan ID：`{PLAN_ID}`",
        f"- Stage：`{STAGE}`",
        f"- Plan04 integrated base：`{BASE_SHA}`（pre-integration gate 为 ff-only）",
        f"- Worker branch：`{BRANCH}`",
        f"- Build commit：`{summary['build_commit']}`",
        f"- Result：`{summary['result']}`",
        "- 本 worker 只建立历史训练回放和时间切分契约；未训练强模型，S5 未声明完成，S6 未进入。",
        "",
        "## Replay matrix",
        "",
        f"- Rows：`{replay['row_count']}`；unique keys：`{replay['unique_key_count']}`。",
        f"- Pseudo-target year：`{replay['pseudo_target_year_min']}`–`{replay['pseudo_target_year_max']}`，共 `{replay['pseudo_target_year_count']}` 个年份。",
        f"- Row counts by target/scenario：`{stable_json(replay['row_count_by_target_scenario'])}`。",
        f"- Excluded target-label rows：`{replay['excluded_label_rows_total']}`；reasons：`{stable_json(replay['excluded_rows_by_reason'])}`。",
        "- Replay eligibility requires actual target label, complete generated feature schema, and computable frozen B3 reference; individual feature cells remain missing when unavailable.",
        "- Direct and residual framings share the same replay-row universe.",
        "",
        "## Temporal split geometry",
        "",
        f"- Outer rows：`{outer['row_count']}`；all training pools non-empty：`{outer['all_training_pools_nonempty']}`。",
        f"- Outer validation groups retain 40 rows/cities：`{outer['all_validation_groups_have_40_rows']}` / `{outer['all_validation_groups_have_40_cities']}`。",
        f"- Outer boundary violations：`{outer['boundary_violation_count']}`。",
        f"- Inner split rows：`{inner['row_count']}`；earliest available inner validation years：`{stable_json(inner['earliest_inner_validation_year_by_target_scenario'])}`。",
        f"- Inner boundary violations：`{inner['boundary_violation_count']}`；zero-training folds reported but not removed：`{inner['zero_training_fold_count']}`。",
        "- Frozen rules: outer training `pseudo_target_year <= outer_origin - 2`; inner training `pseudo_target_year <= inner_validation_year - 2`.",
        "",
        "## Contract and leakage verification",
        "",
        f"- Plan04 formal compatibility：`{compatibility['status']}`; 1760 rows; model-feature mismatch cells：`{compatibility['model_feature_mismatch_cell_count']}`; label/reference mismatch cells：`{compatibility['label_reference_mismatch_cell_count']}`。",
        f"- B3 compatibility：`{summary['b3_compatibility']['pass']}`; resident expected pooled MSE `872.452651515143`; registered expected pooled MSE `225.421212121213`. ",
        f"- Replay boundary rows：`{boundary['row_count']}`; source-cap violations：`{boundary['source_cap_violation_count']}`; quarantine reintroductions：`{boundary['quarantine_reintroduction_count']}`; benchmark violations：`{boundary['benchmark_domain_violation_count']}`。",
        f"- Population-history cutoff violations：`{boundary['population_history_cutoff_violation_count']}`; B3-history cutoff violations：`{boundary['b3_history_cutoff_violation_count']}`。",
        "- Plan04's exact 144-feature contract, source-specific R1/R2 caps, benchmark-only age domain, S1 quarantine, and explicit missingness are retained.",
        "",
        "## Required artifacts",
        "",
        *[f"- `{path}`" for path in summary["required_artifacts"]],
        "",
        "## Forbidden actions not run",
        "",
        "- No LightGBM/CatBoost/XGBoost/ExtraTrees/Ridge/HGB/neural training, tuning, feature selection, routing, gating, blending, 2023 prediction, or submission.",
        "- `CURRENT_LEARNING_CHECKPOINT.md` was not modified.",
        "",
        "## Next boundary",
        "",
        "- Main/Lab should review and integrate this worker tip before creating Plan05A/B/C/D from one exact frozen base.",
        "- Future workers must consume this replay matrix and split geometry; feature changes require `feature_contract_change_required` / Main review.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    assert_integration_gate()
    context = load_context()
    replay_rows, coverage_rows, coverage = replay_rows_and_coverage(context)
    coverage["coverage_rows"] = coverage_rows

    compatibility_rows, compatibility_summary, formal_rows = formal_compatibility(context)
    assert compatibility_summary["status"] == "PASS"
    b3_summary = plan04.b3_compatibility(formal_rows)
    assert b3_summary["pass"] is True

    replay_internal_rows = [
        plan04.build_row(
            context,
            row["target_id"],
            row["scenario_id"],
            row["pseudo_target_year"],
            row["city_id"],
        )
        for row in replay_rows
    ]
    outer_rows, outer_summary = outer_training_geometry(replay_rows, formal_rows)
    inner_rows, inner_summary = inner_split_geometry(replay_rows)
    boundary_rows, boundary_summary = replay_boundary_audit(
        replay_internal_rows,
        replay_rows,
        context,
    )
    contract = parallel_worker_contract()
    summary = make_summary(
        context,
        replay_rows,
        coverage,
        outer_summary,
        inner_summary,
        boundary_summary,
        compatibility_summary,
        b3_summary,
    )
    assert summary["result"] == "training_replay_foundation_ready", summary["success_conditions"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "full_multisource_training_replay_matrix.csv",
        replay_rows,
        fields=REPLAY_FIELDS,
        model_features=set(plan04.MODEL_FEATURES),
    )
    write_json(OUTPUT_DIR / "training_replay_summary.json", summary)
    write_csv(OUTPUT_DIR / "training_replay_year_coverage.csv", coverage_rows)
    write_csv(OUTPUT_DIR / "outer_training_geometry.csv", outer_rows, fields=OUTER_FIELDS)
    write_csv(OUTPUT_DIR / "inner_split_geometry.csv", inner_rows, fields=INNER_FIELDS)
    write_csv(
        OUTPUT_DIR / "plan04_feature_compatibility.csv",
        compatibility_rows,
    )
    write_csv(OUTPUT_DIR / "replay_boundary_audit.csv", boundary_rows, fields=REPLAY_AUDIT_FIELDS)
    write_json(OUTPUT_DIR / "parallel_worker_input_contract.json", contract)
    (OUTPUT_DIR / "HANDOFF.md").write_text(handoff_text(summary), encoding="utf-8")

    print("S5_PLAN04B_TRAINING_REPLAY_FOUNDATION_READY")
    print(f"base_sha={BASE_SHA}")
    print(f"worker_branch={BRANCH}")
    print(f"replay_rows={len(replay_rows)}")
    print(f"pseudo_target_year_min={coverage['replay_pseudo_target_year_min']}")
    print(f"pseudo_target_year_max={coverage['replay_pseudo_target_year_max']}")
    print(f"model_feature_count={len(plan04.MODEL_FEATURES)}")
    print(f"formal_compatibility_pass={compatibility_summary['status'] == 'PASS'}")
    print(f"b3_compatibility_pass={b3_summary['pass']}")
    print(f"outer_rows={len(outer_rows)}")
    print(f"inner_rows={len(inner_rows)}")
    print(f"outer_boundary_violations={outer_summary['boundary_violation_count']}")
    print(f"inner_boundary_violations={inner_summary['boundary_violation_count']}")
    print(f"replay_boundary_audit_rows={len(boundary_rows)}")
    print("strong_model_training_run=False")
    print("final_2023_forecast_run=False")
    print("submission_run=False")
    print("s5_complete=False")
    print("s6_entered=False")
    print("result=training_replay_foundation_ready")


if __name__ == "__main__":
    main()
