"""Build the S5 final Wave consolidation evidence package.

This script is intentionally evidence-only.  It reads committed artifacts from
the frozen main line and the named worker branches with ``git show``.  It does
not fit a model, alter validation geometry, or create predictions for 2023.
"""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
MAIN_REF = "origin/main"
PLAN04_FEATURE_CONTRACT = "S5-Plan04-FULL-MULTISOURCE-V1"
VALIDATION_SCOPE = "common_strict_origins_2011-2021_40cities_440_rows_per_target_scenario"
TARGETS = ("resident", "registered")
SCENARIOS = ("R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS")


def git_show(ref: str, path: str) -> str:
    """Read a committed file without checking out or mutating another branch."""

    try:
        completed = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - evidence guard
        detail = exc.stderr.strip() or "git show failed"
        raise RuntimeError(f"Cannot read {ref}:{path}: {detail}") from exc
    return completed.stdout


def csv_from_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))


def git_csv(ref: str, path: str) -> list[dict[str, str]]:
    return csv_from_text(git_show(ref, path))


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value: object, digits: int = 6) -> str:
    number = to_float(value)
    if number is None:
        return "UNAVAILABLE"
    return f"{number:.{digits}f}"


def percentile(values: list[float], q: float) -> float:
    """Linear percentile using the (n - 1) interpolation convention."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def pick(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def normalize_oof(rows: list[dict[str, str]], model_id: str, framing_id: str | None) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        if row.get("model_id") != model_id:
            continue
        if framing_id is not None and row.get("framing_id") != framing_id:
            continue
        if row.get("scoreable") not in (None, "", "yes", "True", "true", "1"):
            continue
        actual = to_float(pick(row, "actual_target_value", "actual_target_value_for_scoring"))
        prediction = to_float(pick(row, "prediction_value", "prediction"))
        origin = pick(row, "origin_year")
        city = pick(row, "city_id")
        target = pick(row, "target_id")
        scenario = pick(row, "scenario_id")
        if None in (actual, prediction, origin, city, target, scenario):
            continue
        normalized.append(
            {
                "target_id": str(target),
                "scenario_id": str(scenario),
                "origin_year": int(float(str(origin))),
                "city_id": str(city),
                "actual": float(actual),
                "prediction": float(prediction),
            }
        )
    return normalized


def key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row["target_id"]),
        str(row["scenario_id"]),
        int(row["origin_year"]),
        str(row["city_id"]),
    )


def group_scope(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["scenario_id"]))].append(row)
    return grouped


def make_metric_row(
    *,
    candidate: dict[str, object],
    target: str,
    scenario: str,
    rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
) -> dict[str, object]:
    if len(rows) != len(baseline_rows):
        raise RuntimeError(
            f"OOF scope mismatch for {candidate['candidate_id']} {target}/{scenario}: "
            f"candidate={len(rows)} baseline={len(baseline_rows)}"
        )
    baseline_by_key = {key(row): row for row in baseline_rows}
    errors: list[float] = []
    squared: list[float] = []
    absolute: list[float] = []
    origin_errors: dict[int, list[float]] = defaultdict(list)
    city_errors: dict[str, list[float]] = defaultdict(list)
    baseline_origin_errors: dict[int, list[float]] = defaultdict(list)
    baseline_city_errors: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        baseline = baseline_by_key.get(key(row))
        if baseline is None:
            raise RuntimeError(f"Missing B3 key for {candidate['candidate_id']} {key(row)}")
        error = float(row["prediction"]) - float(row["actual"])
        base_error = float(baseline["prediction"]) - float(baseline["actual"])
        errors.append(error)
        squared.append(error * error)
        absolute.append(abs(error))
        origin_errors[int(row["origin_year"])].append(error * error)
        city_errors[str(row["city_id"])].append(error * error)
        baseline_origin_errors[int(baseline["origin_year"])].append(base_error * base_error)
        baseline_city_errors[str(baseline["city_id"])].append(base_error * base_error)

    pooled_mse = mean(squared)
    baseline_mse = mean([float(row["prediction"]) - float(row["actual"]) for row in []]) if False else mean(
        [v for values in baseline_origin_errors.values() for v in values]
    )
    origin_mse = {origin: mean(values) for origin, values in origin_errors.items()}
    baseline_origin_mse = {origin: mean(values) for origin, values in baseline_origin_errors.items()}
    origin_wins = sum(origin_mse[o] < baseline_origin_mse[o] - 1e-12 for o in origin_mse)
    origin_losses = sum(origin_mse[o] > baseline_origin_mse[o] + 1e-12 for o in origin_mse)
    origin_ties = len(origin_mse) - origin_wins - origin_losses

    city_sse = {city: sum(values) for city, values in city_errors.items()}
    baseline_city_sse = {city: sum(values) for city, values in baseline_city_errors.items()}
    city_order = sorted(city_sse, key=lambda city: (-city_sse[city], city))
    total_sse = sum(city_sse.values())
    top1 = city_sse[city_order[0]] / total_sse if city_order else None
    top3 = sum(city_sse[city] for city in city_order[:3]) / total_sse if city_order else None
    improved_cities = sum(city_sse[city] < baseline_city_sse.get(city, float("inf")) for city in city_sse)

    return {
        "candidate_id": candidate["candidate_id"],
        "display_name": candidate["display_name"],
        "family": candidate["family"],
        "classification": candidate["classification"],
        "panel_role": candidate["panel_role"],
        "include_in_final_comparison": candidate["include_in_final_comparison"],
        "target_id": target,
        "scenario_id": scenario,
        "validation_scope": VALIDATION_SCOPE,
        "row_count": len(rows),
        "origin_count": len(origin_mse),
        "city_count": len(city_sse),
        "pooled_mse": pooled_mse,
        "relative_improvement_vs_b3_pct": (baseline_mse - pooled_mse) / baseline_mse * 100.0,
        "rmse": math.sqrt(pooled_mse),
        "mae": mean(absolute),
        "median_absolute_error": percentile(absolute, 0.50),
        "p90_absolute_error": percentile(absolute, 0.90),
        "p95_absolute_error": percentile(absolute, 0.95),
        "signed_bias_mean_error": mean(errors),
        "macro_origin_mse": mean(list(origin_mse.values())),
        "origin_wins_vs_b3": origin_wins,
        "origin_losses_vs_b3": origin_losses,
        "origin_ties_vs_b3": origin_ties,
        "latest_2021_mse": origin_mse.get(2021),
        "worst_cities_by_sse": ";".join(city_order[:3]),
        "worst_city_sse": city_sse[city_order[0]] if city_order else None,
        "top1_city_sse_concentration": top1,
        "top3_city_sse_concentration": top3,
        "improved_city_count_vs_b3": improved_cities,
        "scale_bin_behavior": candidate.get("scale_bin_behavior", "UNAVAILABLE_NOT_PERSISTED"),
        "r1_r2_pooled_mse_delta_r2_minus_r1": "PENDING",
        "r1_r2_relative_improvement_range_pct": "PENDING",
        "auxiliary_metric_source": candidate["auxiliary_metric_source"],
        "selection_decision": candidate["selection_decision"],
        "evidence_mode": "recomputed_from_frozen_oof_predictions",
    }


def unavailable_metric_row(
    *,
    candidate: dict[str, object],
    target: str,
    scenario: str,
    aggregate: dict[str, str],
    origin_comparison: list[dict[str, str]],
) -> dict[str, object]:
    pooled_mse = to_float(aggregate.get("pooled_mse"))
    baseline_mse = 872.452651515143 if target == "resident" else 225.421212121213
    comparisons = [
        row
        for row in origin_comparison
        if row.get("target_id") == target
        and row.get("scenario_id") == scenario
        and row.get("candidate_id") == "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND"
    ]
    wins = sum(row.get("origin_result") == "win" for row in comparisons)
    losses = sum(row.get("origin_result") == "loss" for row in comparisons)
    ties = sum(row.get("origin_result") == "tie" for row in comparisons)
    return {
        "candidate_id": candidate["candidate_id"],
        "display_name": candidate["display_name"],
        "family": candidate["family"],
        "classification": candidate["classification"],
        "panel_role": candidate["panel_role"],
        "include_in_final_comparison": candidate["include_in_final_comparison"],
        "target_id": target,
        "scenario_id": scenario,
        "validation_scope": VALIDATION_SCOPE,
        "row_count": int(float(aggregate.get("scoreable_rows", "440"))),
        "origin_count": 11,
        "city_count": 40,
        "pooled_mse": pooled_mse,
        "relative_improvement_vs_b3_pct": (baseline_mse - pooled_mse) / baseline_mse * 100.0,
        "rmse": to_float(aggregate.get("pooled_rmse")),
        "mae": to_float(aggregate.get("pooled_mae")),
        "median_absolute_error": to_float(aggregate.get("pooled_median_absolute_error")),
        "p90_absolute_error": "UNAVAILABLE_NOT_PERSISTED",
        "p95_absolute_error": "UNAVAILABLE_NOT_PERSISTED",
        "signed_bias_mean_error": to_float(aggregate.get("pooled_bias")),
        "macro_origin_mse": to_float(aggregate.get("macro_origin_mse")),
        "origin_wins_vs_b3": wins,
        "origin_losses_vs_b3": losses,
        "origin_ties_vs_b3": ties,
        "latest_2021_mse": to_float(aggregate.get("latest_2021_mse")),
        "worst_cities_by_sse": "UNAVAILABLE_NOT_PERSISTED",
        "worst_city_sse": "UNAVAILABLE_NOT_PERSISTED",
        "top1_city_sse_concentration": "UNAVAILABLE_NOT_PERSISTED",
        "top3_city_sse_concentration": "UNAVAILABLE_NOT_PERSISTED",
        "improved_city_count_vs_b3": "UNAVAILABLE_NOT_PERSISTED",
        "scale_bin_behavior": candidate.get("scale_bin_behavior", "UNAVAILABLE_NOT_PERSISTED"),
        "r1_r2_pooled_mse_delta_r2_minus_r1": "PENDING",
        "r1_r2_relative_improvement_range_pct": "PENDING",
        "auxiliary_metric_source": candidate["auxiliary_metric_source"],
        "selection_decision": candidate["selection_decision"],
        "evidence_mode": "frozen_aggregate_only_no_rerun",
    }


def scale_behavior_for_e(scale_rows: list[dict[str, str]], target: str, scenario: str, framing: str) -> str:
    selected = [
        row
        for row in scale_rows
        if row.get("target_id") == target
        and row.get("scenario_id") == scenario
        and row.get("framing_id") == framing
    ]
    values: dict[str, str] = {}
    for row in selected:
        improvement = to_float(row.get("improvement_vs_b3_pct"))
        if improvement is not None:
            values[row.get("scale_bin", "unknown")] = f"{improvement * 100.0:.2f}%"
    return ";".join(f"{name}={values[name]}" for name in ("low", "medium", "high") if name in values) or "UNAVAILABLE_NOT_PERSISTED"


def scale_behavior_for_rlt3(scale_rows: list[dict[str, str]], target: str, scenario: str) -> str:
    selected = [
        row
        for row in scale_rows
        if row.get("target_id") == target
        and row.get("scenario_id") == scenario
        and row.get("candidate_id") == "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND"
        and row.get("origin_year") == "ALL"
    ]
    values: dict[str, str] = {}
    for row in selected:
        improvement = to_float(row.get("relative_improvement_vs_b3"))
        if improvement is not None:
            values[row.get("cell", "unknown")] = f"{improvement * 100.0:.2f}%"
    order = ("high_raw_high_scale", "high_raw_not_high_scale", "not_high_raw_high_scale", "neither_high_raw_nor_high_scale")
    return ";".join(f"{name}={values[name]}" for name in order if name in values) or "UNAVAILABLE_NOT_PERSISTED"


def build_candidates() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    scale_e = git_csv("s5/plan06e-delta-growth-framing", "reports/S5_HANDOFFS/S5-Plan06E/scale_bin_metrics.csv")
    scale_rlt3 = git_csv(MAIN_REF, "reports/S5_HANDOFFS/S5-Plan02/scale_confounding_check.csv")
    candidates: list[dict[str, object]] = [
        {
            "candidate_id": "B3_RECENT_3POINT_OLS_TREND",
            "display_name": "B3 frozen recent-3-point OLS trend",
            "family": "frozen_baseline",
            "classification": "reference",
            "panel_role": "frozen_primary_reference",
            "include_in_final_comparison": "yes",
            "selection_decision": "official_primary_reference",
            "auxiliary_metric_source": "reports/S4_HANDOFFS/S4-Plan02/per_origin_predictions_or_residuals.csv; auxiliary metrics recomputed here",
            "main_model_id": "B3_RECENT_3POINT_OLS_TREND",
            "main_path": "reports/S4_HANDOFFS/S4-Plan02/per_origin_predictions_or_residuals.csv",
        },
        {
            "candidate_id": "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND",
            "display_name": "RLT3 recent-3-point median-slope trend",
            "family": "robust_local_trend",
            "classification": "robust_local_trend_unsupported",
            "panel_role": "legacy_valid_candidate",
            "include_in_final_comparison": "yes",
            "selection_decision": "legacy_candidate_not_promoted",
            "auxiliary_metric_source": "reports/S5_HANDOFFS/S5-Plan02/screening_metrics.csv; no row-level RLT3 OOF was persisted",
            "main_path": "reports/S5_HANDOFFS/S5-Plan02/screening_metrics.csv",
            "rlt3_path": "reports/S5_HANDOFFS/S5-Plan02/per_origin_comparison.csv",
            "scale_path": "reports/S5_HANDOFFS/S5-Plan02/scale_confounding_check.csv",
        },
        {
            "candidate_id": "LGBM_F2_B3_RESIDUAL",
            "display_name": "LightGBM F2 B3-residual",
            "family": "LightGBM",
            "classification": "family_competitive_but_unstable",
            "panel_role": "valid_residual_family_control",
            "include_in_final_comparison": "no_best_family_control",
            "selection_decision": "valid_control_not_selected",
            "auxiliary_metric_source": "s5/plan06a-lightgbm-screening:reports/S5_HANDOFFS/S5-Plan06A/oof_predictions.csv; recomputed here",
            "branch": "s5/plan06a-lightgbm-screening",
            "path": "reports/S5_HANDOFFS/S5-Plan06A/oof_predictions.csv",
            "model_id": "LGBM_PLAN06A",
            "framing_id": "F2_B3_RESIDUAL",
        },
        {
            "candidate_id": "CATBOOST_F2_B3_RESIDUAL",
            "display_name": "CatBoost F2 B3-residual",
            "family": "CatBoost",
            "classification": "family_competitive_but_unstable",
            "panel_role": "best_valid_residual_family_control",
            "include_in_final_comparison": "yes",
            "selection_decision": "strongest_valid_plan06a_d_residual_on_resident_primary",
            "auxiliary_metric_source": "s5/plan06b-catboost-screening:reports/S5_HANDOFFS/S5-Plan06B/oof_predictions.csv; recomputed here",
            "branch": "s5/plan06b-catboost-screening",
            "path": "reports/S5_HANDOFFS/S5-Plan06B/oof_predictions.csv",
            "model_id": "S5-Plan06B-CATBOOST",
            "framing_id": "F2_B3_RESIDUAL",
        },
        {
            "candidate_id": "XGB_F2_B3_RESIDUAL",
            "display_name": "XGBoost F2 B3-residual",
            "family": "XGBoost",
            "classification": "family_unsupported",
            "panel_role": "valid_residual_family_control",
            "include_in_final_comparison": "no_best_family_control",
            "selection_decision": "valid_control_not_selected",
            "auxiliary_metric_source": "s5/plan06c-xgboost-screening:reports/S5_HANDOFFS/S5-Plan06C/oof_predictions.csv; recomputed here",
            "branch": "s5/plan06c-xgboost-screening",
            "path": "reports/S5_HANDOFFS/S5-Plan06C/oof_predictions.csv",
            "model_id": "S5-Plan06C_XGB_SCREENING",
            "framing_id": "F2_B3_RESIDUAL",
        },
        {
            "candidate_id": "EXTRATREES_F2_B3_RESIDUAL",
            "display_name": "ExtraTrees F2 B3-residual",
            "family": "ExtraTrees",
            "classification": "family_unsupported",
            "panel_role": "valid_residual_family_control",
            "include_in_final_comparison": "no_best_family_control",
            "selection_decision": "valid_control_not_selected",
            "auxiliary_metric_source": "s5/plan06d-extratrees-linear-controls:reports/S5_HANDOFFS/S5-Plan06D/oof_predictions.csv; recomputed here",
            "branch": "s5/plan06d-extratrees-linear-controls",
            "path": "reports/S5_HANDOFFS/S5-Plan06D/oof_predictions.csv",
            "model_id": "EXTRATREES_SELECTED",
            "framing_id": "F2_B3_RESIDUAL",
        },
        {
            "candidate_id": "RIDGE_F2_B3_RESIDUAL",
            "display_name": "Ridge control F2 B3-residual",
            "family": "Ridge control",
            "classification": "family_unsupported",
            "panel_role": "valid_residual_family_control",
            "include_in_final_comparison": "no_best_family_control",
            "selection_decision": "valid_control_not_selected",
            "auxiliary_metric_source": "s5/plan06d-extratrees-linear-controls:reports/S5_HANDOFFS/S5-Plan06D/oof_predictions.csv; recomputed here",
            "branch": "s5/plan06d-extratrees-linear-controls",
            "path": "reports/S5_HANDOFFS/S5-Plan06D/oof_predictions.csv",
            "model_id": "RIDGE_SELECTED",
            "framing_id": "F2_B3_RESIDUAL",
        },
        {
            "candidate_id": "F3_DELTA_FROM_LATEST",
            "display_name": "Plan06E F3 delta-from-latest",
            "family": "target framing",
            "classification": "framing_signal_found",
            "panel_role": "framing_candidate",
            "include_in_final_comparison": "yes",
            "selection_decision": "resident_candidate_under_review",
            "auxiliary_metric_source": "s5/plan06e-delta-growth-framing:reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv and scale_bin_metrics.csv; recomputed here",
            "branch": "s5/plan06e-delta-growth-framing",
            "path": "reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv",
            "model_id": "CAT_DELTA_SELECTED",
            "framing_id": "F3_DELTA_FROM_LATEST",
            "scale_rows": scale_e,
        },
        {
            "candidate_id": "F4_LOG_GROWTH_FROM_LATEST",
            "display_name": "Plan06E F4 log-growth-from-latest",
            "family": "target framing",
            "classification": "framing_signal_found",
            "panel_role": "framing_candidate",
            "include_in_final_comparison": "yes",
            "selection_decision": "strongest_resident_candidate_under_review",
            "auxiliary_metric_source": "s5/plan06e-delta-growth-framing:reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv and scale_bin_metrics.csv; recomputed here",
            "branch": "s5/plan06e-delta-growth-framing",
            "path": "reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv",
            "model_id": "CAT_LOG_GROWTH_SELECTED",
            "framing_id": "F4_LOG_GROWTH_FROM_LATEST",
            "scale_rows": scale_e,
        },
    ]

    b3_rows = normalize_oof(
        git_csv(MAIN_REF, "reports/S4_HANDOFFS/S4-Plan02/per_origin_predictions_or_residuals.csv"),
        "B3_RECENT_3POINT_OLS_TREND",
        None,
    )
    oof_by_candidate: dict[str, list[dict[str, object]]] = {"B3_RECENT_3POINT_OLS_TREND": b3_rows}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if "branch" in candidate:
            oof_by_candidate[candidate_id] = normalize_oof(
                git_csv(str(candidate["branch"]), str(candidate["path"])),
                str(candidate["model_id"]),
                str(candidate["framing_id"]),
            )
    return candidates, oof_by_candidate


def finish_r1_r2_fields(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["candidate_id"]), str(row["target_id"]))][str(row["scenario_id"])] = row
    for (candidate_id, target), scenarios in grouped.items():
        r1 = scenarios.get("R1_REFERENCE_SNAPSHOT_LAG")
        r2 = scenarios.get("R2_CONSERVATIVE_PUBLICATION_STRESS")
        if not r1 or not r2:
            delta = "UNAVAILABLE"
            improvement_range = "UNAVAILABLE"
        else:
            delta = float(r2["pooled_mse"]) - float(r1["pooled_mse"])
            values = [float(r1["relative_improvement_vs_b3_pct"]), float(r2["relative_improvement_vs_b3_pct"])]
            improvement_range = f"{min(values):.6f}..{max(values):.6f}"
        for scenario_row in scenarios.values():
            scenario_row["r1_r2_pooled_mse_delta_r2_minus_r1"] = delta
            scenario_row["r1_r2_relative_improvement_range_pct"] = improvement_range


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_panel() -> list[dict[str, object]]:
    candidates, oof_by_candidate = build_candidates()
    baseline_grouped = group_scope(oof_by_candidate["B3_RECENT_3POINT_OLS_TREND"])
    aggregate_rows = git_csv(MAIN_REF, "reports/S5_HANDOFFS/S5-Plan02/screening_metrics.csv")
    origin_rows = git_csv(MAIN_REF, "reports/S5_HANDOFFS/S5-Plan02/per_origin_comparison.csv")
    panel: list[dict[str, object]] = []

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id == "RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND":
            for target in TARGETS:
                for scenario in SCENARIOS:
                    aggregate = next(
                        row
                        for row in aggregate_rows
                        if row.get("target_id") == target
                        and row.get("scenario_id") == scenario
                        and row.get("candidate_id") == candidate_id
                    )
                    candidate["scale_bin_behavior"] = scale_behavior_for_rlt3(
                        git_csv(MAIN_REF, str(candidate["scale_path"])), target, scenario
                    )
                    panel.append(
                        unavailable_metric_row(
                            candidate=candidate,
                            target=target,
                            scenario=scenario,
                            aggregate=aggregate,
                            origin_comparison=origin_rows,
                        )
                    )
            continue

        candidate_rows = oof_by_candidate[candidate_id]
        for target in TARGETS:
            for scenario in SCENARIOS:
                selected = [
                    row
                    for row in candidate_rows
                    if row["target_id"] == target and row["scenario_id"] == scenario
                ]
                baseline = baseline_grouped[(target, scenario)]
                if "scale_rows" in candidate:
                    candidate["scale_bin_behavior"] = scale_behavior_for_e(
                        list(candidate["scale_rows"]),
                        target,
                        scenario,
                        str(candidate["framing_id"]),
                    )
                panel.append(
                    make_metric_row(
                        candidate=candidate,
                        target=target,
                        scenario=scenario,
                        rows=selected,
                        baseline_rows=baseline,
                    )
                )
    finish_r1_r2_fields(panel)
    return panel


def write_registry() -> None:
    header = [
        "experiment_id",
        "hypothesis",
        "validation_contract",
        "change",
        "primary_result",
        "stability_result",
        "interpretation",
        "decision",
        "next_direction",
        "evidence_path",
    ]
    rows = [
        {
            "experiment_id": "S5-Plan01",
            "hypothesis": "Can pre-origin diagnostics identify B3 failure risk and a repairable mechanism?",
            "validation_contract": "Origins 2011-2021; history <= T-2; resident primary; registered sensitivity; pooled city-year MSE",
            "change": "Reused frozen B1/B2/B3 predictions and constructed five pre-origin diagnostics; no new model/source",
            "primary_result": "Historical first-difference volatility passed risk and decision-relevance gates; median abs Spearman 0.4281",
            "stability_result": "Risk direction positive in 11/11 origins; shrinkage comparator gain 6/11; registered did not reverse",
            "interpretation": "Risk identification exists but does not establish repairability",
            "decision": "mechanism_signal_found:robust_local_trend",
            "next_direction": "robust_local_trend_screening",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan01/HANDOFF.md;reports/S5_HANDOFFS/S5-Plan01/screening_summary.json",
        },
        {
            "experiment_id": "S5-Plan02",
            "hypothesis": "Can deterministic robust local trend repair B3 error in unstable history?",
            "validation_contract": "Same frozen contract; R1/R2 retained; 40-city full coverage; no routing/blend/tuning",
            "change": "Evaluated B3/B4/RLT3/RLT5 with raw/relative volatility and scale strata",
            "primary_result": "RLT3 resident pooled MSE 861.4909, 1.256% below B3; registered 223.6909",
            "stability_result": "Resident RLT3 wins 6/11; registered RLT3 wins 8/11; scale-outside gain 0.1595%",
            "interpretation": "Direction exists but fails strong and complementary gates",
            "decision": "robust_local_trend_unsupported",
            "next_direction": "No further robust-local-trend expansion; superseded as final comparison by Plan07 panel",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan02/HANDOFF.md;reports/S5_HANDOFFS/S5-Plan02/screening_metrics.csv",
        },
        {
            "experiment_id": "S5-Plan03",
            "hypothesis": "Can prior S5 negative evidence be consolidated into a stage report?",
            "validation_contract": "Inherited S5 frozen contract; report-only boundary",
            "change": "Prior final evidence report covering Plan01/02 only",
            "primary_result": "Historical report was valid for its scope but did not include the later Plan04-06 Wave",
            "stability_result": "No new model or validation evidence",
            "interpretation": "Report is superseded by the complete Wave consolidation",
            "decision": "superseded_by_S5-Plan07",
            "next_direction": "Main reviews Plan07 evidence package",
            "evidence_path": "S5_EVIDENCE_DRIVEN_ITERATION_FINAL_REPORT.md (superseded by this consolidation)",
        },
        {
            "experiment_id": "S5-Plan04",
            "hypothesis": "Can a full multi-source feature base expose city heterogeneity without changing the contract?",
            "validation_contract": PLAN04_FEATURE_CONTRACT,
            "change": "Built 1760-row, 144-feature model-agnostic matrix with source/audit metadata",
            "primary_result": "Foundation ready; B3 compatibility passed; source-cap violations 0",
            "stability_result": "R1/R2 differences are source-cap driven; target-history cutoff violations 0",
            "interpretation": "Safe shared feature foundation, not a model win",
            "decision": "full_multisource_foundation_ready",
            "next_direction": "Replay foundation and common strong-model Wave contract",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan04/HANDOFF.md;reports/S5_HANDOFFS/S5-Plan04/foundation_summary.json",
        },
        {
            "experiment_id": "S5-Plan04B",
            "hypothesis": "Can historical replay and time-split geometry support comparable parallel workers?",
            "validation_contract": "S5-Plan04-FULL-MULTISOURCE-V1; outer/inner <= origin-2 boundaries",
            "change": "Built 2320-row replay matrix, 44 outer groups, 484 inner split rows",
            "primary_result": "Replay foundation ready; all outer/inner boundary audits passed",
            "stability_result": "264/484 inner folds eligible; 8 pre-registered conservative fallback groups",
            "interpretation": "Contract infrastructure is reusable; no model evidence by itself",
            "decision": "training_replay_foundation_ready",
            "next_direction": "Plan05 common strong-model Wave",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan04B/HANDOFF.md;reports/S5_HANDOFFS/S5-Plan04B/training_replay_summary.json",
        },
        {
            "experiment_id": "S5-Plan05",
            "hypothesis": "Can a frozen common contract support independent strong-model screening?",
            "validation_contract": "Feature S5-Plan04-FULL-MULTISOURCE-V1; replay S5-Plan04B-TRAINING-REPLAY-V1",
            "change": "Froze the parallel worker input/OOF/metric contract; no model fit in this plan",
            "primary_result": "Wave base and mandatory evidence contract frozen",
            "stability_result": "264 eligible inner folds; 8 fallback groups; B3 references preserved",
            "interpretation": "Governance boundary for Plans06A-H",
            "decision": "strong_model_wave_contract_frozen",
            "next_direction": "Wave consolidation and Main review",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan05/HANDOFF.md;reports/S5_HANDOFFS/S5-Plan05/strong_model_wave_contract.json",
        },
        {
            "experiment_id": "S5-Plan06A",
            "hypothesis": "Can LightGBM improve B3 under the frozen multi-table residual/direct contract?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F1 direct + F2 residual; 1760 rows per framing",
            "change": "LightGBM screening, no tuning/routing/blending",
            "primary_result": "F2 residual was closest but resident pooled MSE remained above B3",
            "stability_result": "family_competitive_but_unstable; direct framing failed",
            "interpretation": "No stable resident replacement",
            "decision": "family_competitive_but_unstable",
            "next_direction": "Consolidate only; no new LightGBM experiment",
            "evidence_path": "branch s5/plan06a-lightgbm-screening;reports/S5_HANDOFFS/S5-Plan06A/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06B",
            "hypothesis": "Can CatBoost improve B3 under the frozen multi-table residual/direct contract?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F1 direct + F2 residual; 1760 rows per framing",
            "change": "CatBoost screening with frozen native categorical city metadata; no tuning/routing/blending",
            "primary_result": "F2 residual was the strongest valid A-D resident control but still above B3",
            "stability_result": "Registered F2 stayed close to B3 but did not replace it",
            "interpretation": "Best valid A-D residual control, not a promotion",
            "decision": "family_competitive_but_unstable",
            "next_direction": "Retain as comparison control only",
            "evidence_path": "branch s5/plan06b-catboost-screening;reports/S5_HANDOFFS/S5-Plan06B/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06C",
            "hypothesis": "Can XGBoost improve B3 under the frozen multi-table residual/direct contract?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F1 direct + F2 residual; 1760 rows per framing",
            "change": "XGBoost screening; corrected dense city-indicator handling; no tuning/routing",
            "primary_result": "Direct failed; residual remained above B3 on resident and registered",
            "stability_result": "Residual origin wins only 3/11 resident and 5/11 registered in the aggregate evidence",
            "interpretation": "No admissible replacement",
            "decision": "family_unsupported",
            "next_direction": "No XGBoost expansion",
            "evidence_path": "branch s5/plan06c-xgboost-screening;reports/S5_HANDOFFS/S5-Plan06C/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06D",
            "hypothesis": "Can ExtraTrees or regularized linear residual controls improve B3?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F1 direct + F2 residual; 1760 rows per candidate/framing",
            "change": "ExtraTrees and Ridge controls; no tuning/routing/blending",
            "primary_result": "Both residual controls remained above resident B3; registered controls remained non-promoting",
            "stability_result": "Complete OOF and auxiliary evidence; family classification unsupported",
            "interpretation": "Useful negative controls; no replacement",
            "decision": "family_unsupported",
            "next_direction": "No ExtraTrees/Ridge expansion",
            "evidence_path": "branch s5/plan06d-extratrees-linear-controls;reports/S5_HANDOFFS/S5-Plan06D/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06E",
            "hypothesis": "Can delta/growth target framing stabilize the B3 residual signal?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F3 delta + F4 log growth; 440 rows per target/scenario",
            "change": "CatBoost F3/F4 reconstruction under frozen replay and un-clipped positive level",
            "primary_result": "Resident F4 R1 739.601/R2 802.446; F3 R1 788.841/R2 818.448",
            "stability_result": "F4 wins 5/11 and 7/11 origins; all resident scale bins improve; registered F3/F4 do not beat B3 stably",
            "interpretation": "Real resident framing signal; strongest candidate remains under Main review",
            "decision": "framing_signal_found",
            "next_direction": "Consolidate F4/F3 in the final panel; do not tune or route",
            "evidence_path": "branch s5/plan06e-delta-growth-framing;reports/S5_HANDOFFS/S5-Plan06E/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06F",
            "hypothesis": "Can EBM additive residual/direct models improve B3?",
            "validation_contract": "S5-Plan04/S5-Plan04B; 88 outer fits; F1/F2 selection",
            "change": "Recovery-only archival; no rerun in Plan07",
            "primary_result": "execution_completed_but_core_performance_evidence_unrecoverable",
            "stability_result": "88/88 outer fits and selection counts recoverable; OOF/metrics/inner MSE not recoverable",
            "interpretation": "Selection counts are not performance evidence",
            "decision": "recovery_only_excluded",
            "next_direction": "No Plan06F rerun",
            "evidence_path": "reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md; local Codex session record cited there",
        },
        {
            "experiment_id": "S5-Plan06G",
            "hypothesis": "Can hierarchical/partial pooling stabilize city heterogeneity?",
            "validation_contract": "S5-Plan04/S5-Plan04B; F1/F2; full 3520 combined OOF required",
            "change": "MixedLM-style partial pooling screening with PCA/random effects",
            "primary_result": "Combined OOF 2760/3520; F2 only 1000/1760; 19 outer failures",
            "stability_result": "coverage_incomplete; registered/R1 F2 low MSE is inadmissible",
            "interpretation": "No candidate promotion under incomplete coverage",
            "decision": "coverage_incomplete_excluded",
            "next_direction": "No rerun in this consolidation",
            "evidence_path": "branch s5/plan06g-hierarchical-partial-pooling;reports/S5_HANDOFFS/S5-Plan06G/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan06H",
            "hypothesis": "Can city-local residual/direct fits exploit city heterogeneity without cross-city sharing?",
            "validation_contract": "S5-Plan04/S5-Plan04B; same-city legal history only",
            "change": "Six city-local candidates including FG-P/all-144 Ridge",
            "primary_result": "All candidates classified city_local_unsupported; all-144 local Ridge collapsed under n << p",
            "stability_result": "Diagnostic best resident local FGP Ridge still exceeded B3; no stable origin gate",
            "interpretation": "City-local family is unsupported; no routing inference",
            "decision": "city_local_unsupported",
            "next_direction": "No local routing/gating/expansion",
            "evidence_path": "branch s5/plan06h-city-local-screening;reports/S5_HANDOFFS/S5-Plan06H/HANDOFF.md",
        },
        {
            "experiment_id": "S5-Plan07",
            "hypothesis": "Can the complete S5 Wave be closed into a Main-review evidence package?",
            "validation_contract": "Frozen S5 Plan04/04B/05 contract; no contract changes",
            "change": "Evidence-only consolidation of Plan01-06H; no model fit",
            "primary_result": "Unified panel identifies F4 as strongest resident candidate under review; B3 remains official reference",
            "stability_result": "Registered sensitivity remains non-promoting; 06F recovery and 06G coverage defects isolated",
            "interpretation": "Ready for Main Stage Completion Review, not S5 completion",
            "decision": "ready_for_main_stage_review",
            "next_direction": "Main review; do not enter S6 from this branch",
            "evidence_path": "reports/S5_FINAL_CANDIDATE_COMPARISON.csv;reports/S5_MAIN_HANDOFF.md;S5_EVIDENCE_DRIVEN_ITERATION_FINAL_REPORT.md",
        },
    ]
    write_csv(ROOT / "reports/S5_EXPERIMENT_REGISTRY.csv", rows, header)


def write_recovery_note() -> None:
    path = ROOT / "reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = """# S5-Plan06F Recovery Note

Status: `execution_completed_but_core_performance_evidence_unrecoverable`

This note is a recovery archive only. Plan06F was not rerun during S5-Plan07 consolidation, and no missing performance metric is inferred or fabricated.

## Verified local session evidence

- Local Codex session record: `C:\\Users\\Yukis\\.codex\\sessions\\2026\\09\\19\\rollout-2026-09-19T22-28-10-01a0ba11-4a43-7662-89d1-0aacd29e45a1.jsonl`.
- The session record contains the completed first-run stdout and the final recovery statement (record line 5841); the first run reported a unique continuous `outer_fit=1..88` sequence.
- First run duration was approximately `23,398` seconds (about 6 hours 30 minutes).
- Recovered final selections: `C1=64/88`, `C2=24/88`; `F1=44/44 C1`; `F2=20 C1 / 24 C2`.
- Independently recoverable geometry: `44` outer groups, `264` eligible folds, `8` fallback groups. Correct inner evidence row count is `1560`; the runner assertion expected `1632` and failed before final artifact writing.
- The worker branch has no committed Plan06F performance artifacts beyond environment evidence; the session record states that no OOF/aggregate/per-origin/inner-MSE/term-importance/city-error files were written before the assertion.

## Evidence boundary

The following are **unavailable**, not zero and not reconstructed:

- EBM OOF predictions;
- outer/per-origin/aggregate metrics;
- inner per-configuration MSE and mean validation MSE;
- term importance, interaction summaries, and city errors.

Selection counts are execution diagnostics only and are not model-performance evidence. Plan06F is excluded from the final candidate comparison and no rerun is authorized by this consolidation.

## Reproducibility pointers

- Worker branch: `s5/plan06f-ebm-additive-screening`.
- Frozen Wave base: `1f97f67488cd7aec120ef8b9fa04407dc13d54ba`.
- Environment evidence, if needed for audit, is retained in final main at `reports/S5_HANDOFFS/S5-Plan06F/worker_environment.json`.
"""
    path.write_text(content, encoding="utf-8")


def write_main_handoff(panel: list[dict[str, object]]) -> None:
    def row(candidate_id: str, target: str, scenario: str) -> dict[str, object]:
        return next(
            item
            for item in panel
            if item["candidate_id"] == candidate_id
            and item["target_id"] == target
            and item["scenario_id"] == scenario
        )

    f4_r1 = row("F4_LOG_GROWTH_FROM_LATEST", "resident", "R1_REFERENCE_SNAPSHOT_LAG")
    f4_r2 = row("F4_LOG_GROWTH_FROM_LATEST", "resident", "R2_CONSERVATIVE_PUBLICATION_STRESS")
    f3_r1 = row("F3_DELTA_FROM_LATEST", "resident", "R1_REFERENCE_SNAPSHOT_LAG")
    f3_r2 = row("F3_DELTA_FROM_LATEST", "resident", "R2_CONSERVATIVE_PUBLICATION_STRESS")
    rlt3_reg = row("RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND", "registered", "R1_REFERENCE_SNAPSHOT_LAG")
    b3_reg = row("B3_RECENT_3POINT_OLS_TREND", "registered", "R1_REFERENCE_SNAPSHOT_LAG")
    cat_res = row("CATBOOST_F2_B3_RESIDUAL", "resident", "R1_REFERENCE_SNAPSHOT_LAG")

    content = f"""# S5-Plan07 Main Handoff — Final Wave Consolidation

## Review status

- `ready_for_main_stage_review: true`
- `S5_COMPLETE: false`
- `S6_ENTERED: false`
- Scope: evidence consolidation only; no new model, tuning, routing, ensemble, 2023 prediction, or submission.
- Validation and feature contracts remain frozen.

## Strongest valid resident candidate

`F4_LOG_GROWTH_FROM_LATEST` is the strongest resident candidate in the unified panel under review:

- R1 pooled MSE `{fmt(f4_r1['pooled_mse'])}`, relative improvement `{fmt(f4_r1['relative_improvement_vs_b3_pct'], 3)}%`, origin wins `{f4_r1['origin_wins_vs_b3']}/11`, latest-2021 MSE `{fmt(f4_r1['latest_2021_mse'])}`.
- R2 pooled MSE `{fmt(f4_r2['pooled_mse'])}`, relative improvement `{fmt(f4_r2['relative_improvement_vs_b3_pct'], 3)}%`, origin wins `{f4_r2['origin_wins_vs_b3']}/11`, latest-2021 MSE `{fmt(f4_r2['latest_2021_mse'])}`.
- F3 remains a valid framing signal but is weaker than F4 on resident pooled MSE in both scenarios: R1 `{fmt(f3_r1['pooled_mse'])}`, R2 `{fmt(f3_r2['pooled_mse'])}`.
- CatBoost F2 is the strongest valid Plan06A-D residual control on the resident primary, but remains above B3: R1 `{fmt(cat_res['pooled_mse'])}`.

This is a Main-review recommendation, not a final winner or deployment promotion. B3 remains the official frozen competition reference until Main completes its stage review.

## Registered sensitivity conclusion

The small legacy RLT3 signal is real but narrow: registered R1 pooled MSE `{fmt(rlt3_reg['pooled_mse'])}` versus frozen B3 `{fmt(b3_reg['pooled_mse'])}`, with 8/11 origin wins in the Plan02 artifact. Plan06E does not reproduce that registered gain: F3/F4 are above B3 in both scenarios. Therefore the resident-primary contract supports reviewing F4 as a resident candidate, while registered remains a sensitivity constraint and does not support a global replacement claim.

## Unsupported or incomplete families

- Plan06A LightGBM and Plan06B CatBoost: complete but competitive/unstable; residual controls do not replace B3 on resident primary.
- Plan06C XGBoost and Plan06D ExtraTrees/Ridge: unsupported negative controls.
- Plan06F EBM: recovery-only; execution/selection facts preserved, core performance evidence unavailable; not compared.
- Plan06G hierarchical/partial pooling: `coverage_incomplete` (`2760/3520` combined OOF; F2 `1000/1760`; 19 outer failures); low registered/R1 F2 MSE is inadmissible.
- Plan06H city-local: all candidates `city_local_unsupported`; all-144 local Ridge shows n << p collapse; no routing inference.

## Unresolved risks

- F4's resident improvement is a framing signal, not yet a deployment decision; Main must review tail errors, city concentration, bias, and R1/R2 behavior together.
- RLT3 auxiliary quantiles and city-tail fields are unavailable because Plan02 did not persist row-level RLT3 OOF; no rerun was used to fill them.
- 06F and 06G cannot support performance claims beyond their explicit recovery/coverage facts.

## Evidence package

- `reports/S5_FINAL_CANDIDATE_COMPARISON.csv`
- `reports/S5_FINAL_CANDIDATE_COMPARISON.json`
- `reports/S5_EXPERIMENT_REGISTRY.csv`
- `reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md`
- `S5_EVIDENCE_DRIVEN_ITERATION_FINAL_REPORT.md`
"""
    (ROOT / "reports/S5_MAIN_HANDOFF.md").write_text(content, encoding="utf-8")


def main() -> None:
    panel = build_panel()
    fields = [
        "candidate_id",
        "display_name",
        "family",
        "classification",
        "panel_role",
        "include_in_final_comparison",
        "target_id",
        "scenario_id",
        "validation_scope",
        "row_count",
        "origin_count",
        "city_count",
        "pooled_mse",
        "relative_improvement_vs_b3_pct",
        "rmse",
        "mae",
        "median_absolute_error",
        "p90_absolute_error",
        "p95_absolute_error",
        "signed_bias_mean_error",
        "macro_origin_mse",
        "origin_wins_vs_b3",
        "origin_losses_vs_b3",
        "origin_ties_vs_b3",
        "latest_2021_mse",
        "worst_cities_by_sse",
        "worst_city_sse",
        "top1_city_sse_concentration",
        "top3_city_sse_concentration",
        "improved_city_count_vs_b3",
        "scale_bin_behavior",
        "r1_r2_pooled_mse_delta_r2_minus_r1",
        "r1_r2_relative_improvement_range_pct",
        "auxiliary_metric_source",
        "selection_decision",
        "evidence_mode",
    ]
    write_csv(ROOT / "reports/S5_FINAL_CANDIDATE_COMPARISON.csv", panel, fields)
    payload = {
        "package": "S5-Plan07-final-wave-consolidation",
        "status": "ready_for_main_stage_review",
        "s5_complete": False,
        "s6_entered": False,
        "validation_scope": VALIDATION_SCOPE,
        "metric_method": {
            "primary": "pooled city-year MSE",
            "auxiliary": ["RMSE", "MAE", "median absolute error", "P90/P95 absolute error", "signed bias", "macro-origin MSE", "origin wins/losses", "latest-2021 MSE", "city SSE concentration", "scale-bin behavior", "R1/R2 sensitivity"],
            "percentile": "linear interpolation at (n-1)*q over absolute errors",
            "missing_policy": "UNAVAILABLE is retained when the committed source artifact did not persist row-level predictions; no retraining or synthetic OOF",
        },
        "frozen_b3": {"resident_pooled_mse": 872.452651515143, "registered_pooled_mse": 225.421212121213, "resident_latest_2021_mse": 3705.427777777728, "registered_latest_2021_mse": 1485.604861111118},
        "rows": panel,
    }
    (ROOT / "reports/S5_FINAL_CANDIDATE_COMPARISON.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_registry()
    write_recovery_note()
    write_main_handoff(panel)
    print(f"wrote {len(panel)} panel rows")
    print("wrote reports/S5_FINAL_CANDIDATE_COMPARISON.csv")
    print("wrote reports/S5_FINAL_CANDIDATE_COMPARISON.json")
    print("wrote reports/S5_EXPERIMENT_REGISTRY.csv")
    print("wrote reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md")
    print("wrote reports/S5_MAIN_HANDOFF.md")


if __name__ == "__main__":
    main()
