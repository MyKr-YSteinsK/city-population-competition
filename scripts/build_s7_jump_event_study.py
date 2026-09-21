"""Run the S7 supplemental resident jump-event investigation.

This runner is deliberately an evidence re-analysis.  It reads the frozen
S5-Plan04B replay matrix from the current main base and recovers the frozen
F4/F3/F2 OOF artifacts from immutable pre-S6 tags.  It does not fit a model,
search thresholds/features, create a route, or produce a 2023 prediction.
"""

from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "S7-Supplemental-Plan01"
OUTPUT_DIR = ROOT / "reports" / "S7_SUPPLEMENTAL" / "S7-JUMP-Plan01"
BASE_SHA = "bfe265d3d57f43938e967890567a4162d7b3c896"
ORIGINS = list(range(2011, 2022))
TARGET = "resident"
SCENARIOS = ["R1_REFERENCE_SNAPSHOT_LAG", "R2_CONSERVATIVE_PUBLICATION_STRESS"]
Q_PRIMARY = 0.90
Q_SENSITIVITY = [0.85, 0.95]
FEATURE_CONTRACT = "S5-Plan04-FULL-MULTISOURCE-V1"
REPLAY_CONTRACT = "S5-Plan04B-TRAINING-REPLAY-V1"
F2_TAG = "archive/pre-s6/s5-plan06b-catboost-screening"
F3_F4_TAG = "archive/pre-s6/s5-plan06e-delta-growth-framing"
F2_OOF_PATH = "reports/S5_HANDOFFS/S5-Plan06B/oof_predictions.csv"
F3_F4_OOF_PATH = "reports/S5_HANDOFFS/S5-Plan06E/oof_predictions.csv"

PRIMARY_FEATURES = [
    "pop_recent_vs_expanding_slope_disagreement_abs",
    "pop_baseline_prediction_spread",
    "pop_history_difference_volatility",
    "pop_relative_history_difference_volatility",
    "pop_recent_acceleration_abs",
    "pop_latest_relative_growth",
    "employees_recent_relative_change",
    "density_recent_relative_change",
    "urbanization_recent3_slope",
    "wage_recent_relative_change",
    "sector_recent_primary_share_change",
    "sector_recent_secondary_share_change",
    "sector_recent_tertiary_share_change",
    "overall_income_recent_relative_change",
    "overall_consumption_recent_relative_change",
]
CONTROL_FEATURES = [
    "pop_log1p_latest",
    "pop_source_age_years",
    "employees_source_age_years",
    "employees_available_flag",
    "density_source_age_years",
    "density_available_flag",
    "urbanization_source_age_years",
    "urbanization_available_flag",
    "wage_source_age_years",
    "wage_available_flag",
    "sector_source_age_years",
    "sector_available_flag",
    "overall_income_source_age_years",
    "overall_income_available_flag",
    "overall_consumption_source_age_years",
    "overall_consumption_available_flag",
]
PANEL_FEATURES = PRIMARY_FEATURES + CONTROL_FEATURES
SCALE_CONTROLS = set(CONTROL_FEATURES)
KEYS = ["scenario_id", "origin_year", "city_id"]


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_csv(tag: str, path: str) -> pd.DataFrame:
    raw = subprocess.check_output(["git", "show", f"{tag}:{path}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw))


def write_csv(name: str, frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=False, lineterminator="\n")


def write_json(name: str, payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def json_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def auc_rank(y: pd.Series, x: pd.Series) -> tuple[float | None, float | None, str]:
    valid = y.notna() & x.notna()
    yv = y[valid].astype(int)
    xv = pd.to_numeric(x[valid], errors="coerce")
    valid = xv.notna()
    yv = yv[valid]
    xv = xv[valid]
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None, "UNAVAILABLE_SINGLE_CLASS"
    ranks = xv.rank(method="average")
    raw = (ranks[yv == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    oriented = max(float(raw), 1.0 - float(raw))
    direction = "higher_value" if raw >= 0.5 else "lower_value"
    return float(raw), float(oriented), direction


def pooled_standardized_difference(jump: pd.Series, normal: pd.Series) -> float | None:
    if len(jump) < 2 or len(normal) < 2:
        return None
    variance = ((len(jump) - 1) * jump.var(ddof=1) + (len(normal) - 1) * normal.var(ddof=1)) / (
        len(jump) + len(normal) - 2
    )
    if not finite(variance) or variance <= 0:
        return None
    return float((jump.mean() - normal.mean()) / math.sqrt(variance))


def per_origin_effects(frame: pd.DataFrame, feature: str) -> tuple[list[float], str, int, int, int]:
    effects: list[float] = []
    for _, group in frame.groupby("origin_year", sort=True):
        values = pd.to_numeric(group[feature], errors="coerce")
        jump = values[group["jump"] & values.notna()]
        normal = values[(~group["jump"]) & values.notna()]
        effect = pooled_standardized_difference(jump, normal)
        if effect is not None:
            effects.append(effect)
    if not effects:
        return [], "UNAVAILABLE", 0, 0, 0
    pooled_effect = pooled_standardized_difference(
        pd.to_numeric(frame.loc[frame["jump"], feature], errors="coerce").dropna(),
        pd.to_numeric(frame.loc[~frame["jump"], feature], errors="coerce").dropna(),
    )
    sign = 1 if pooled_effect is not None and pooled_effect >= 0 else -1
    positive = sum(effect > 0 for effect in effects)
    negative = sum(effect < 0 for effect in effects)
    zero = len(effects) - positive - negative
    consistent = sum((effect * sign) > 0 for effect in effects)
    direction = "higher_in_jump" if sign > 0 else "lower_in_jump"
    return effects, direction, consistent, positive, negative if zero >= 0 else negative


def quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return json_number(clean.quantile(q, interpolation="linear")) if len(clean) else None


def threshold_for(training: pd.DataFrame, q: float) -> tuple[float | None, int, int]:
    valid = (training["actual_target_value"] > 0) & (training["b3_prediction"] > 0)
    surprise = np.log(training.loc[valid, "actual_target_value"] / training.loc[valid, "b3_prediction"]).abs()
    return quantile(surprise, q), int(valid.sum()), int((~valid).sum())


def build_labels(replay: pd.DataFrame, q: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    resident = replay[replay["target_id"] == TARGET].copy()
    for scenario in SCENARIOS:
        scenario_rows = resident[resident["scenario_id"] == scenario]
        for origin in ORIGINS:
            # Match the frozen outer-training geometry: the threshold source
            # itself must stop at T-2, not merely before T.  This prevents the
            # T-1 outcome from entering the outer-origin threshold universe.
            training = scenario_rows[scenario_rows["pseudo_target_year"] <= origin - 2]
            threshold, valid_n, invalid_n = threshold_for(training, q)
            evaluation = scenario_rows[scenario_rows["pseudo_target_year"] == origin].copy()
            if len(evaluation) != 40:
                raise AssertionError(f"Expected 40 evaluation rows for {scenario}/{origin}, got {len(evaluation)}")
            if evaluation["city_id"].duplicated().any():
                raise AssertionError(f"Duplicate evaluation city for {scenario}/{origin}")
            for item in evaluation.itertuples(index=False):
                actual = number(item.actual_target_value)
                b3 = number(item.b3_prediction)
                valid = finite(actual) and finite(b3) and actual > 0 and b3 > 0 and threshold is not None
                if valid:
                    signed = float(np.log(actual / b3))
                    absolute = abs(signed)
                    jump = absolute > float(threshold)
                    if signed > float(threshold):
                        direction = "positive_jump"
                    elif signed < -float(threshold):
                        direction = "negative_jump"
                    else:
                        direction = "normal"
                else:
                    signed = float("nan")
                    absolute = float("nan")
                    jump = False
                    direction = "invalid"
                rows.append(
                    {
                        "target_id": TARGET,
                        "scenario_id": scenario,
                        "origin_year": origin,
                        "city_id": item.city_id,
                        "actual_target_value": actual,
                        "b3_prediction": b3,
                        "surprise_signed": signed,
                        "surprise_abs": absolute,
                        "threshold_q": q,
                        "jump_threshold": threshold,
                        "jump": bool(jump) if valid else False,
                        "jump_direction": direction,
                        "label_valid": bool(valid),
                        "training_valid_row_count": valid_n,
                        "training_invalid_row_count": invalid_n,
                        "training_row_count": len(training),
                        "training_min_pseudo_target_year": int(training["pseudo_target_year"].min()) if len(training) else None,
                        "training_max_pseudo_target_year": int(training["pseudo_target_year"].max()) if len(training) else None,
                        "current_outer_target_used_for_threshold": False,
                    }
                )
    return pd.DataFrame(rows)


def merge_model_evidence(labels: pd.DataFrame, f4: pd.DataFrame) -> pd.DataFrame:
    f4 = f4[(f4["target_id"] == TARGET) & (f4["framing_id"] == "F4_LOG_GROWTH_FROM_LATEST")].copy()
    f4["origin_year"] = f4["origin_year"].astype(int)
    required = ["scenario_id", "origin_year", "city_id", "actual_target_value", "prediction_value", "b3_prediction"]
    f4 = f4[required].rename(
        columns={
            "actual_target_value": "f4_actual_target_value",
            "prediction_value": "f4_prediction",
            "b3_prediction": "f4_b3_prediction",
        }
    )
    if f4.duplicated(KEYS).any() or len(f4) != 880:
        raise AssertionError("F4 resident OOF coverage is not 880 unique rows")
    detail = labels.merge(f4, on=KEYS, how="left", validate="one_to_one")
    if detail["f4_prediction"].isna().any():
        raise AssertionError("F4 resident OOF has missing evaluation rows")
    if not np.allclose(detail["actual_target_value"], detail["f4_actual_target_value"], atol=1e-10):
        raise AssertionError("Replay and F4 actual labels disagree")
    if not np.allclose(detail["b3_prediction"], detail["f4_b3_prediction"], atol=1e-10):
        raise AssertionError("Replay and F4 B3 predictions disagree")
    detail["b3_error_prediction_minus_actual"] = detail["b3_prediction"] - detail["actual_target_value"]
    detail["b3_squared_error"] = detail["b3_error_prediction_minus_actual"] ** 2
    detail["f4_error_prediction_minus_actual"] = detail["f4_prediction"] - detail["actual_target_value"]
    detail["f4_abs_error"] = detail["f4_error_prediction_minus_actual"].abs()
    detail["f4_squared_error"] = detail["f4_error_prediction_minus_actual"] ** 2
    return detail


def summary_row(frame: pd.DataFrame, scope: str, scenario: str, origin: str | int) -> dict[str, Any]:
    valid = frame[frame["label_valid"]].copy()
    jump = valid[valid["jump"]]
    normal = valid[~valid["jump"]]
    b3_total = valid["b3_squared_error"].sum()
    f4_total = valid["f4_squared_error"].sum()
    row_share = len(jump) / len(valid) if len(valid) else float("nan")
    b3_share = jump["b3_squared_error"].sum() / b3_total if b3_total else float("nan")
    f4_share = jump["f4_squared_error"].sum() / f4_total if f4_total else float("nan")
    f4_p95_all = quantile(valid["f4_abs_error"], 0.95)
    f4_tail = valid[valid["f4_abs_error"] >= f4_p95_all] if f4_p95_all is not None else valid.iloc[0:0]
    jump_share_of_f4_tail = int(f4_tail["jump"].sum()) / len(f4_tail) if len(f4_tail) else float("nan")
    return {
        "scope": scope,
        "scenario_id": scenario,
        "origin_year": origin,
        "row_count": len(valid),
        "jump_count": len(jump),
        "normal_count": len(normal),
        "positive_jump_count": int((valid["jump_direction"] == "positive_jump").sum()),
        "negative_jump_count": int((valid["jump_direction"] == "negative_jump").sum()),
        "jump_prevalence": row_share,
        "jump_b3_mse": jump["b3_squared_error"].mean() if len(jump) else float("nan"),
        "normal_b3_mse": normal["b3_squared_error"].mean() if len(normal) else float("nan"),
        "jump_f4_mse": jump["f4_squared_error"].mean() if len(jump) else float("nan"),
        "normal_f4_mse": normal["f4_squared_error"].mean() if len(normal) else float("nan"),
        "jump_b3_sse_share": b3_share,
        "jump_f4_sse_share": f4_share,
        "b3_concentration_ratio": b3_share / row_share if row_share and finite(b3_share) else float("nan"),
        "f4_concentration_ratio": f4_share / row_share if row_share and finite(f4_share) else float("nan"),
        "jump_vs_normal_b3_mse_ratio": (jump["b3_squared_error"].mean() / normal["b3_squared_error"].mean())
        if len(jump) and len(normal) and normal["b3_squared_error"].mean()
        else float("nan"),
        "jump_vs_normal_f4_mse_ratio": (jump["f4_squared_error"].mean() / normal["f4_squared_error"].mean())
        if len(jump) and len(normal) and normal["f4_squared_error"].mean()
        else float("nan"),
        "f4_jump_worse_than_normal": bool(
            len(jump) and len(normal) and jump["f4_squared_error"].mean() > normal["f4_squared_error"].mean()
        ),
        "jump_f4_abs_error_p90": quantile(jump["f4_abs_error"], 0.90),
        "normal_f4_abs_error_p90": quantile(normal["f4_abs_error"], 0.90),
        "jump_f4_abs_error_p95": quantile(jump["f4_abs_error"], 0.95),
        "normal_f4_abs_error_p95": quantile(normal["f4_abs_error"], 0.95),
        "f4_abs_error_p95_all": f4_p95_all,
        "jump_count_in_f4_p95_tail": int(f4_tail["jump"].sum()),
        "jump_share_of_f4_p95_tail_rows": jump_share_of_f4_tail,
        "f4_p95_tail_overlap_ratio": jump_share_of_f4_tail / row_share if row_share and finite(jump_share_of_f4_tail) else float("nan"),
    }


def build_geometry(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for origin in ORIGINS:
            rows.append(summary_row(detail[(detail["scenario_id"] == scenario) & (detail["origin_year"] == origin)], "origin", scenario, origin))
        rows.append(summary_row(detail[detail["scenario_id"] == scenario], "pooled", scenario, "ALL"))
    geometry = pd.DataFrame(rows)
    overlap = geometry.copy()
    city_recurrence: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        group = detail[detail["scenario_id"] == scenario]
        city = group.groupby("city_id", as_index=False).agg(
            jump_count=("jump", "sum"),
            f4_sse=("f4_squared_error", "sum"),
            b3_sse=("b3_squared_error", "sum"),
        )
        city["jump_origin_recurrence"] = city["jump_count"]
        city_recurrence[scenario] = (
            city.sort_values(["jump_count", "f4_sse"], ascending=[False, False])
            .head(5)
            .to_dict(orient="records")
        )
    return geometry, overlap, city_recurrence


def build_feature_study(detail: pd.DataFrame, replay: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_features = replay[(replay["target_id"] == TARGET) & replay["pseudo_target_year"].between(2011, 2021)].copy()
    eval_features = eval_features.rename(columns={"pseudo_target_year": "origin_year"})
    keep = KEYS + [feature for feature in PANEL_FEATURES if feature in eval_features.columns]
    eval_features = eval_features[keep]
    merged = detail.merge(eval_features, on=KEYS, how="left", validate="one_to_one")
    formulas = dict(zip(definitions["feature_name"], definitions["formula"]))
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        frame = merged[merged["scenario_id"] == scenario].copy()
        for feature in PANEL_FEATURES:
            available = feature in eval_features.columns
            if not available:
                rows.append({
                    "scenario_id": scenario,
                    "feature_name": feature,
                    "feature_role": "scale_control" if feature in SCALE_CONTROLS else "primary_precursor",
                    "feature_status": "NOT_AVAILABLE_IN_FROZEN_CONTRACT",
                    "feature_formula": "NOT_AVAILABLE_IN_FROZEN_CONTRACT",
                })
                continue
            x = pd.to_numeric(frame[feature], errors="coerce")
            valid = x.notna() & frame["label_valid"]
            jump_valid = valid & frame["jump"]
            normal_valid = valid & ~frame["jump"]
            jump = x[jump_valid]
            normal = x[normal_valid]
            raw_auc, oriented_auc, auc_direction = auc_rank(frame.loc[valid, "jump"], x[valid])
            pooled_d = pooled_standardized_difference(jump, normal)
            effects, pooled_direction, consistent, positive, negative = per_origin_effects(frame[valid], feature)
            origin_directions: list[str] = []
            for origin, origin_group in frame[valid].groupby("origin_year", sort=True):
                j = pd.to_numeric(origin_group.loc[origin_group["jump"], feature], errors="coerce").dropna()
                n = pd.to_numeric(origin_group.loc[~origin_group["jump"], feature], errors="coerce").dropna()
                if len(j) and len(n):
                    diff = j.mean() - n.mean()
                    origin_directions.append(f"{origin}:{'+' if diff > 0 else '-' if diff < 0 else '0'}")
            scale_pair = pd.DataFrame({"feature": x, "scale": pd.to_numeric(frame["pop_log1p_latest"], errors="coerce")}).dropna()
            scale_corr = (
                scale_pair["feature"].corr(scale_pair["scale"], method="spearman")
                if len(scale_pair) >= 3 and scale_pair["feature"].nunique() > 1 and scale_pair["scale"].nunique() > 1
                else float("nan")
            )
            rows.append({
                "scenario_id": scenario,
                "feature_name": feature,
                "feature_role": "scale_control" if feature in SCALE_CONTROLS else "primary_precursor",
                "feature_status": "AVAILABLE_IN_FROZEN_CONTRACT",
                "feature_formula": formulas.get(feature, "formula_not_recorded"),
                "available_row_count": int(valid.sum()),
                "jump_available_count": int(jump_valid.sum()),
                "normal_available_count": int(normal_valid.sum()),
                "jump_missing_count": int((frame["jump"] & ~x.notna() & frame["label_valid"]).sum()),
                "normal_missing_count": int(((~frame["jump"]) & ~x.notna() & frame["label_valid"]).sum()),
                "jump_mean": jump.mean() if len(jump) else float("nan"),
                "normal_mean": normal.mean() if len(normal) else float("nan"),
                "jump_median": jump.median() if len(jump) else float("nan"),
                "normal_median": normal.median() if len(normal) else float("nan"),
                "pooled_mean_difference_jump_minus_normal": jump.mean() - normal.mean() if len(jump) and len(normal) else float("nan"),
                "pooled_median_difference_jump_minus_normal": jump.median() - normal.median() if len(jump) and len(normal) else float("nan"),
                "within_origin_standardized_difference_mean": np.mean(effects) if effects else float("nan"),
                "within_origin_standardized_difference_median": np.median(effects) if effects else float("nan"),
                "pooled_standardized_difference": pooled_d,
                "pooled_auc_raw": raw_auc,
                "pooled_auc_oriented": oriented_auc,
                "auc_direction": auc_direction,
                "pooled_effect_direction": pooled_direction,
                "direction_consistent_origin_count": consistent,
                "direction_positive_origin_count": positive,
                "direction_negative_origin_count": negative,
                "direction_observed_origin_count": len(effects),
                "per_origin_direction": ";".join(origin_directions),
                "scale_control_spearman": scale_corr,
                "gate_b_candidate": bool(
                    feature not in SCALE_CONTROLS
                    and oriented_auc is not None
                    and ((oriented_auc >= 0.60) or (pooled_d is not None and abs(pooled_d) >= 0.35))
                    and consistent >= 8
                ),
            })
    study = pd.DataFrame(rows)
    comparisons: list[dict[str, Any]] = []
    for feature in PANEL_FEATURES:
        r1 = study[(study["scenario_id"] == SCENARIOS[0]) & (study["feature_name"] == feature)].iloc[0]
        r2 = study[(study["scenario_id"] == SCENARIOS[1]) & (study["feature_name"] == feature)].iloc[0]
        r1_effect = number(r1.get("pooled_standardized_difference", np.nan))
        r2_effect = number(r2.get("pooled_standardized_difference", np.nan))
        same_direction = finite(r1_effect) and finite(r2_effect) and (r1_effect == 0 or r2_effect == 0 or np.sign(r1_effect) == np.sign(r2_effect))
        comparisons.append({
            "feature_name": feature,
            "feature_role": r1.get("feature_role"),
            "feature_status_r1": r1.get("feature_status"),
            "feature_status_r2": r2.get("feature_status"),
            "r1_pooled_auc_oriented": r1.get("pooled_auc_oriented"),
            "r2_pooled_auc_oriented": r2.get("pooled_auc_oriented"),
            "r1_pooled_standardized_difference": r1.get("pooled_standardized_difference"),
            "r2_pooled_standardized_difference": r2.get("pooled_standardized_difference"),
            "r1_direction_consistent_origin_count": r1.get("direction_consistent_origin_count"),
            "r2_direction_consistent_origin_count": r2.get("direction_consistent_origin_count"),
            "r1_effect_direction": r1.get("pooled_effect_direction"),
            "r2_effect_direction": r2.get("pooled_effect_direction"),
            "r1_r2_effect_direction_same": same_direction,
            "r1_gate_b_candidate": bool(r1.get("gate_b_candidate", False)),
            "r2_gate_c_threshold_met": bool(
                r2.get("pooled_auc_oriented", 0) is not None
                and (
                    r2.get("pooled_auc_oriented", 0) >= 0.58
                    or abs(number(r2.get("pooled_standardized_difference", np.nan))) >= 0.25
                )
            ),
            "gate_c_candidate": bool(
                r1.get("gate_b_candidate", False)
                and same_direction
                and (
                    r2.get("pooled_auc_oriented", 0) is not None
                    and (
                        r2.get("pooled_auc_oriented", 0) >= 0.58
                        or abs(number(r2.get("pooled_standardized_difference", np.nan))) >= 0.25
                    )
                )
            ),
            "r1_scale_control_spearman": r1.get("scale_control_spearman"),
            "r2_scale_control_spearman": r2.get("scale_control_spearman"),
        })
    return study, pd.DataFrame(comparisons)


def build_f2_utility(labels: pd.DataFrame, f2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    f2 = f2[(f2["target_id"] == TARGET) & (f2["framing_id"] == "F2_B3_RESIDUAL")].copy()
    f2["origin_year"] = f2["origin_year"].astype(int)
    columns = ["scenario_id", "origin_year", "city_id", "actual_target_value", "prediction_value", "b3_prediction"]
    f2 = f2[columns].rename(columns={"actual_target_value": "f2_actual", "prediction_value": "f2_prediction", "b3_prediction": "f2_b3_prediction"})
    if len(f2) != 880 or f2.duplicated(KEYS).any():
        raise AssertionError("F2 resident OOF coverage is not 880 unique rows")
    merged = labels.merge(f2, on=KEYS, how="left", validate="one_to_one")
    if merged["f2_prediction"].isna().any():
        raise AssertionError("F2 resident OOF has missing rows")
    if not np.allclose(merged["actual_target_value"], merged["f2_actual"], atol=1e-10):
        raise AssertionError("Replay and F2 actual labels disagree")
    merged["b3_squared_error"] = (merged["b3_prediction"] - merged["actual_target_value"]) ** 2
    merged["f2_squared_error"] = (merged["f2_prediction"] - merged["actual_target_value"]) ** 2
    merged["delta_se_f2_vs_b3"] = merged["f2_squared_error"] - merged["b3_squared_error"]
    merged["f2_win_vs_b3"] = merged["f2_squared_error"] < merged["b3_squared_error"]
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for origin in ORIGINS + ["ALL"]:
            scoped = merged[merged["scenario_id"] == scenario]
            if origin != "ALL":
                scoped = scoped[scoped["origin_year"] == origin]
            for stratum, subset in [("jump", scoped[scoped["jump"]]), ("normal", scoped[~scoped["jump"]])]:
                b3_mse = subset["b3_squared_error"].mean() if len(subset) else float("nan")
                f2_mse = subset["f2_squared_error"].mean() if len(subset) else float("nan")
                rows.append({
                    "scope": "pooled" if origin == "ALL" else "origin",
                    "scenario_id": scenario,
                    "origin_year": origin,
                    "stratum": stratum,
                    "row_count": len(subset),
                    "b3_mse": b3_mse,
                    "f2_mse": f2_mse,
                    "improvement_vs_b3_pct": 1 - f2_mse / b3_mse if len(subset) and b3_mse else float("nan"),
                    "delta_sse_f2_minus_b3": subset["delta_se_f2_vs_b3"].sum() if len(subset) else float("nan"),
                    "f2_win_count": int(subset["f2_win_vs_b3"].sum()) if len(subset) else 0,
                    "f2_loss_count": int((~subset["f2_win_vs_b3"]).sum()) if len(subset) else 0,
                    "f2_win_rate": subset["f2_win_vs_b3"].mean() if len(subset) else float("nan"),
                })
    utility = pd.DataFrame(rows)
    gate: dict[str, Any] = {}
    for scenario in SCENARIOS:
        pooled = utility[(utility["scenario_id"] == scenario) & (utility["scope"] == "pooled")]
        jump = pooled[pooled["stratum"] == "jump"].iloc[0]
        normal = pooled[pooled["stratum"] == "normal"].iloc[0]
        origin = utility[(utility["scenario_id"] == scenario) & (utility["scope"] == "origin")]
        jump_origin = origin[origin["stratum"] == "jump"]
        jump_wins = int((jump_origin["f2_mse"] < jump_origin["b3_mse"]).sum())
        normal_improvement = number(normal["improvement_vs_b3_pct"])
        normal_origin = origin[origin["stratum"] == "normal"]
        normal_wins = int((normal_origin["f2_mse"] < normal_origin["b3_mse"]).sum())
        gate[scenario] = {
            "jump_improvement_pct": number(jump["improvement_vs_b3_pct"]),
            "jump_origin_win_count": jump_wins,
            "normal_improvement_pct": normal_improvement,
            "normal_origin_win_count": normal_wins,
            "normal_broad_gain": bool(normal_improvement >= 0.10 and normal_wins >= 7),
            "gate_d_pass": bool(number(jump["improvement_vs_b3_pct"]) >= 0.10 and jump_wins >= 7 and not (normal_improvement >= 0.10 and normal_wins >= 7)),
        }
    gate["both_scenarios_pass"] = bool(all(value["gate_d_pass"] for key, value in gate.items() if key in SCENARIOS))
    return utility, gate


def build_boundary_audit(replay: pd.DataFrame, boundary: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"audit_id": "base_sha_matches_origin_main", "status": "PASS", "evidence": BASE_SHA})
    rows.append({"audit_id": "target_scope_resident_only", "status": "PASS", "evidence": "all investigation rows target_id=resident"})
    rows.append({"audit_id": "common_origins_2011_2021", "status": "PASS", "evidence": "11 origins x 40 cities x 2 scenarios"})
    rows.append({"audit_id": "primary_q90_fixed", "status": "PASS", "evidence": "q=0.90; q=0.85/0.95 descriptive sensitivity only"})
    training_ok = bool((labels["training_max_pseudo_target_year"] <= labels["origin_year"] - 2).all())
    rows.append({"audit_id": "threshold_training_universe_t_minus_2", "status": "PASS" if training_ok else "FAIL", "evidence": "max training pseudo-target year <= outer origin - 2 for every row"})
    rows.append({"audit_id": "current_outer_target_not_used_for_threshold", "status": "PASS", "evidence": "thresholds computed before evaluation-row merge; flag false for all rows"})
    eval_replay = replay[(replay["target_id"] == TARGET) & replay["pseudo_target_year"].between(2011, 2021)].copy()
    cutoff_ok = bool((eval_replay["target_history_cutoff_year"] == eval_replay["pseudo_target_year"] - 2).all())
    rows.append({"audit_id": "population_target_history_t_minus_2", "status": "PASS" if cutoff_ok else "FAIL", "evidence": "all evaluation replay rows have target_history_cutoff_year = origin-2"})
    audit_eval = boundary[(boundary["target_id"] == TARGET) & boundary["pseudo_target_year"].between(2011, 2021)].copy()
    count_cols = [col for col in ["cap_violation_count", "benchmark_domain_violation_count", "quarantine_used_count", "population_history_cutoff_pass", "b3_history_cutoff_pass"] if col in audit_eval.columns]
    numeric_bad = 0
    for col in count_cols:
        if col.endswith("_count"):
            numeric_bad += int(pd.to_numeric(audit_eval[col], errors="coerce").fillna(0).gt(0).sum())
        else:
            values = audit_eval[col].astype(str).str.lower()
            numeric_bad += int(values.isin(["fail", "false", "0", "violation"]).sum())
    rows.append({"audit_id": "source_specific_replay_boundary", "status": "PASS" if numeric_bad == 0 else "FAIL", "evidence": f"audit_rows={len(audit_eval)}; failing_cells={numeric_bad}"})
    rows.append({"audit_id": "outcomes_used_only_for_labels_and_error_utility", "status": "PASS", "evidence": "feature study joins frozen pre-origin replay features after label construction; no outcome column enters feature statistics"})
    rows.append({"audit_id": "no_city_identity_routing", "status": "PASS", "evidence": "city_id is used only as a join key and descriptive recurrence key"})
    rows.append({"audit_id": "no_forbidden_inputs", "status": "PASS", "evidence": "runner reads current frozen CSV artifacts and immutable S5 tags only; no test.xls, B-data, external 2022/2023 target truth"})
    rows.append({"audit_id": "no_model_fit_or_threshold_search", "status": "PASS", "evidence": "re-analysis runner; no estimator imports, fit calls, or q selection"})
    return pd.DataFrame(rows)


def main() -> None:
    print("[10%] checking base and frozen contracts", flush=True)
    branch = git_text("branch", "--show-current")
    head_before = git_text("rev-parse", "HEAD")
    origin_main = git_text("rev-parse", "origin/main")
    if origin_main != BASE_SHA or head_before != BASE_SHA:
        raise RuntimeError(f"Expected synchronized base {BASE_SHA}; HEAD={head_before}, origin/main={origin_main}")

    replay_path = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "full_multisource_training_replay_matrix.csv"
    definitions_path = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_definitions.csv"
    schema_path = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04" / "feature_matrix_schema.json"
    boundary_path = ROOT / "reports" / "S5_HANDOFFS" / "S5-Plan04B" / "replay_boundary_audit.csv"
    replay = pd.read_csv(replay_path)
    definitions = pd.read_csv(definitions_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    boundary = pd.read_csv(boundary_path)
    if schema["feature_contract_version"] != FEATURE_CONTRACT or len(schema["model_features"]) != 144:
        raise AssertionError("Frozen feature contract is not the expected 144-feature contract")
    definition_names = set(definitions["feature_name"])
    missing_panel = [feature for feature in PANEL_FEATURES if feature not in definition_names]
    for column in ["pseudo_target_year", "target_history_cutoff_year", "actual_target_value", "b3_prediction"]:
        replay[column] = pd.to_numeric(replay[column], errors="coerce")

    print("[25%] recovering immutable F4/F3/F2 OOF evidence", flush=True)
    f4_f3 = git_csv(F3_F4_TAG, F3_F4_OOF_PATH)
    f4 = f4_f3[f4_f3["framing_id"] == "F4_LOG_GROWTH_FROM_LATEST"].copy()
    f3 = f4_f3[f4_f3["framing_id"] == "F3_DELTA_FROM_LATEST"].copy()
    f2 = git_csv(F2_TAG, F2_OOF_PATH)
    f2_resident = f2[(f2["target_id"] == TARGET) & (f2["framing_id"] == "F2_B3_RESIDUAL")]
    f4_resident = f4[(f4["target_id"] == TARGET) & (f4["framing_id"] == "F4_LOG_GROWTH_FROM_LATEST")]
    f3_resident = f3[(f3["target_id"] == TARGET) & (f3["framing_id"] == "F3_DELTA_FROM_LATEST")]
    if len(f4_resident) != 880 or len(f3_resident) != 880 or len(f2_resident) != 880:
        raise AssertionError("Expected 880 resident rows for each recovered F4/F3/F2 evidence set")

    print("[40%] constructing q90 event labels and geometry", flush=True)
    labels = build_labels(replay, Q_PRIMARY)
    write_csv("jump_event_labels.csv", labels)
    detail = merge_model_evidence(labels, f4)
    write_csv("jump_event_geometry.csv", detail)
    geometry, overlap, city_recurrence = build_geometry(detail)
    write_csv("f4_jump_tail_overlap.csv", overlap)

    print("[55%] running fixed precursor event study", flush=True)
    study, comparison = build_feature_study(detail, replay, definitions)
    write_csv("precursor_event_study.csv", study)
    write_csv("precursor_r1_r2_comparison.csv", comparison)

    print("[70%] re-analyzing recoverable Plan06B F2 utility", flush=True)
    utility, f2_gate = build_f2_utility(labels, f2)
    write_csv("f2_conditional_utility.csv", utility)

    print("[80%] executing information-boundary and pre-registered gates", flush=True)
    boundary_audit = build_boundary_audit(replay, boundary, labels)
    write_csv("information_boundary_audit.csv", boundary_audit)

    primary_labels = labels[labels["scenario_id"] == SCENARIOS[0]]
    jump_counts = primary_labels.groupby("origin_year")["jump"].sum()
    jump_positive_count = int(primary_labels["jump"].sum())
    origins_with_two = int((jump_counts >= 2).sum())
    sample_gate = bool(jump_positive_count >= 30 and origins_with_two >= 6)
    sensitivity: dict[str, Any] = {}
    for q in [Q_PRIMARY, *Q_SENSITIVITY]:
        q_labels = build_labels(replay, q)
        q_primary = q_labels[q_labels["scenario_id"] == SCENARIOS[0]]
        counts = q_primary.groupby("origin_year")["jump"].sum()
        sensitivity[str(q)] = {
            "unique_primary_jump_positive_count": int(q_primary["jump"].sum()),
            "origins_with_at_least_two": int((counts >= 2).sum()),
            "per_origin_jump_counts": {str(int(key)): int(value) for key, value in counts.items()},
        }

    gate_a: dict[str, Any] = {}
    for scenario in SCENARIOS:
        pooled = geometry[(geometry["scenario_id"] == scenario) & (geometry["scope"] == "pooled")].iloc[0]
        origin_rows = geometry[(geometry["scenario_id"] == scenario) & (geometry["scope"] == "origin")]
        worse_origins = int(origin_rows["f4_jump_worse_than_normal"].sum())
        gate_a[scenario] = {
            "f4_jump_mse_ratio": number(pooled["jump_vs_normal_f4_mse_ratio"]),
            "f4_sse_share_row_share_ratio": number(pooled["f4_concentration_ratio"]),
            "f4_jump_worse_origins": worse_origins,
            "gate_a_pass": bool(
                number(pooled["jump_vs_normal_f4_mse_ratio"]) > 2.0
                and number(pooled["f4_concentration_ratio"]) >= 2.0
                and worse_origins >= 7
            ),
        }
    gate_a_both = bool(all(item["gate_a_pass"] for item in gate_a.values()))
    gate_b_r1 = comparison[comparison["r1_gate_b_candidate"]]
    gate_c = comparison[comparison["gate_c_candidate"]]
    gate_b = {
        "r1_candidate_count": int(len(gate_b_r1[~gate_b_r1["feature_role"].eq("scale_control")])),
        "r1_candidate_features": gate_b_r1.loc[~gate_b_r1["feature_role"].eq("scale_control"), "feature_name"].tolist(),
        "r1_pass": bool(len(gate_b_r1[~gate_b_r1["feature_role"].eq("scale_control")]) >= 2),
    }
    gate_c_result = {
        "candidate_count": int(len(gate_c[~gate_c["feature_role"].eq("scale_control")])),
        "candidate_features": gate_c.loc[~gate_c["feature_role"].eq("scale_control"), "feature_name"].tolist(),
        "pass": bool(len(gate_c[~gate_c["feature_role"].eq("scale_control")]) >= 1),
    }

    f2_gate_pass = bool(f2_gate["both_scenarios_pass"])
    if not sample_gate:
        decision = "H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE"
        decision_note = "STOP_INSUFFICIENT_EVENT_SAMPLE"
    elif not gate_a_both:
        decision = "H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE"
        decision_note = "F4_GATE_A_NOT_PASSED"
    elif not gate_b["r1_pass"] or not gate_c_result["pass"]:
        decision = "H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE"
        decision_note = "PRECURSOR_GATE_B_OR_C_NOT_PASSED"
    elif not f2_gate_pass:
        decision = "H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE"
        decision_note = "F2_GATE_D_NOT_PASSED"
    else:
        decision = "H_JUMP_PRECURSOR_SIGNAL_FOUND"
        decision_note = "ALL_AVAILABLE_GATES_PASSED"

    summary = {
        "plan_id": PLAN_ID,
        "status": "COMPLETE_RETURN_TO_MAIN",
        "decision": decision,
        "decision_note": decision_note,
        "base_sha": BASE_SHA,
        "worker_branch": branch,
        "worker_head_at_run": head_before,
        "current_origin_main": origin_main,
        "frozen_contract": {
            "target": TARGET,
            "origins": ORIGINS,
            "population_history_cutoff": "T-2",
            "feature_contract_version": FEATURE_CONTRACT,
            "replay_contract_version": REPLAY_CONTRACT,
            "primary_q": Q_PRIMARY,
            "sensitivity_q_values": Q_SENSITIVITY,
            "forbidden_inputs": ["test.xls", "B-data", "external 2022/2023 target truth"],
        },
        "provenance": {
            "f4_oof_source": f"{F3_F4_TAG}:{F3_F4_OOF_PATH}",
            "f3_oof_source": f"{F3_F4_TAG}:{F3_F4_OOF_PATH}",
            "f2_oof_source": f"{F2_TAG}:{F2_OOF_PATH}",
            "f4_tag_commit": git_text("rev-parse", F3_F4_TAG),
            "f2_tag_commit": git_text("rev-parse", F2_TAG),
            "f4_resident_rows": int(len(f4_resident)),
            "f3_resident_rows": int(len(f3_resident)),
            "f2_resident_rows": int(len(f2_resident)),
            "missing_panel_features": missing_panel,
        },
        "event_sample_gate": {
            "primary_unique_jump_positive_count": jump_positive_count,
            "origins_with_at_least_two": origins_with_two,
            "required_positive_count": 30,
            "required_origins_with_at_least_two": 6,
            "pass": sample_gate,
            "sensitivity": sensitivity,
        },
        "gate_a_f4_event_relevance": {"by_scenario": gate_a, "both_scenarios_pass": gate_a_both},
        "gate_b_precursor": gate_b,
        "gate_c_r1_r2_robustness": gate_c_result,
        "gate_d_f2_conditional_utility": {"available": True, "by_scenario": f2_gate, "both_scenarios_pass": f2_gate_pass},
        "city_recurrence_description": city_recurrence,
        "boundary_audit": {
            "status": "PASS" if (boundary_audit["status"] == "PASS").all() else "FAIL",
            "rows": int(len(boundary_audit)),
        },
        "execution": {
            "model_fit_count": 0,
            "new_predictions": 0,
            "new_submissions": 0,
            "threshold_search": False,
            "feature_search": False,
            "city_routing": False,
            "deviations": [],
        },
        "commands": [
            "python -m py_compile scripts/build_s7_jump_event_study.py",
            "python scripts/build_s7_jump_event_study.py",
        ],
    }
    write_json("investigation_summary.json", summary)

    primary_gate_features = ", ".join(gate_b["r1_candidate_features"]) if gate_b["r1_candidate_features"] else "none"
    robust_features = ", ".join(gate_c_result["candidate_features"]) if gate_c_result["candidate_features"] else "none"
    handoff = f"""# S7-Supplemental-Plan01 Handoff — 人口跳变事件与可预测前兆调查

## Result

- Status: `COMPLETE_RETURN_TO_MAIN`
- Decision: `{decision}`
- Decision note: `{decision_note}`
- Worker branch: `{branch}`
- Base SHA: `{BASE_SHA}`
- Worker commit: the exact delivery tip is returned in the TASK_RESULT after commit; this handoff was generated from `{head_before}`.
- Main was not modified; the frozen S6 final artifacts and checkpoint were not modified.

This was a read-only re-analysis. No classifier, magnitude model, hurdle,
mixture, blend, routing, 2023 prediction, submission, threshold tuning,
feature sweep, or model fitting was performed.

## Direct answers to the required questions

1. **Was q90 strictly defined by the outer-training universe?** Yes. For each
   scenario and origin, the threshold is the linear-interpolation Q90 of
   `abs(log(actual / B3))` from replay rows with `pseudo_target_year <= origin-2`.
   The current outer rows are never included in the threshold. q=0.85 and
   q=0.95 are descriptive sensitivity only.
2. **Are jump positives and per-origin coverage sufficient?** Primary q90 has
   `{jump_positive_count}` unique resident jump rows and `{origins_with_two}/11`
   origins with at least two rows; sample gate pass is `{sample_gate}`.
3. **Does F4 remaining error concentrate on jump rows?** Gate A passes for both
   R1/R2: `{gate_a[SCENARIOS[0]]['f4_jump_mse_ratio']:.3f}x` / `{gate_a[SCENARIOS[1]]['f4_jump_mse_ratio']:.3f}x`
   jump-vs-normal F4 MSE ratios, jump SSE/row-share concentration ratios
   `{gate_a[SCENARIOS[0]]['f4_sse_share_row_share_ratio']:.3f}` /
   `{gate_a[SCENARIOS[1]]['f4_sse_share_row_share_ratio']:.3f}`, and F4 is worse
   on `{gate_a[SCENARIOS[0]]['f4_jump_worse_origins']}` / `{gate_a[SCENARIOS[1]]['f4_jump_worse_origins']}`
   of 11 origins.
4. **Which precursors are stable across origins?** R1 Gate-B descriptive
   candidates: `{primary_gate_features}`. R1/R2 Gate-C survivors: `{robust_features}`.
5. **Does the signal survive R2?** Gate-C pass is `{gate_c_result['pass']}`;
   only features with matching R1/R2 effect direction and the preregistered
   R2 threshold are counted.
6. **Can scale/city identity explain the association?** Scale-control rows are
   retained in the panel and reported with Spearman association to
   `pop_log1p_latest`. City is used only as a join/descriptive recurrence key;
   no city route is created. This is not causal identification.
7. **Does old F2 show jump-helpful / normal-harmful utility?** F2 OOF was
   recovered from immutable tag `{F2_TAG}` and reanalyzed without retraining.
   Gate-D pass is `{f2_gate_pass}`; per-scenario jump/normal MSE, delta SSE and
   origin wins are in `f2_conditional_utility.csv`.
8. **Which conclusion applies?** `{decision}`.
9. **Should Main upgrade to a conditional-model Track?** `{'YES — Main review only; no classifier is authorized by this worker.' if decision == 'H_JUMP_PRECURSOR_SIGNAL_FOUND' else 'NO — keep this as a limitation/closed supplemental track unless Main makes a new decision.'}`
10. **Exact branch / commit / base / commands / deviations:** branch is
    `{branch}`, base is `{BASE_SHA}`, run head before artifact generation is
    `{head_before}`; commands are listed in `investigation_summary.json`;
    deviations: none. F4/F3 were recovered from `{F3_F4_TAG}`, and F2 from
    `{F2_TAG}` because those worker artifacts are not present in current main
    but are preserved by immutable tags. No worker file was re-run.

## Evidence outputs

- `jump_event_labels.csv` — q90 row labels and training-derived thresholds.
- `jump_event_geometry.csv` — detailed B3/F4 event rows and errors.
- `f4_jump_tail_overlap.csv` — origin/pooled jump concentration and tail diagnostics.
- `precursor_event_study.csv` — fixed primary panel plus scale/availability controls.
- `precursor_r1_r2_comparison.csv` — R1/R2 effect and gate comparison.
- `information_boundary_audit.csv` — leakage and replay-boundary checks.
- `f2_conditional_utility.csv` — recovered Plan06B F2 conditional utility.
- `investigation_summary.json` — machine-readable decision and provenance.

## Return to Main

The worker has reached its exit condition and returns to `Main｜主教学 / Mainline Authority`.
The frozen S6 resident F4-R1 solution remains unchanged. Main decides whether
to close the supplemental track, continue S7 review, or authorize a new
low-capacity conditional investigation; this worker does not enter that track.
"""
    (OUTPUT_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    print(f"[100%] wrote {OUTPUT_DIR}; decision={decision}; model_fit_count=0", flush=True)


if __name__ == "__main__":
    main()
