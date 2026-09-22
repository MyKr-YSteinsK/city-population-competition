"""Finalize the S1 post-hoc audit after fixing Plan04 evidence matching.

This worker is deliberately bounded to the immutable Plan04 evidence package.
It does not read raw/quarantine data, frozen S8 truth, S9 artifacts, or the
Plan04 output directory for writing. It writes a new Plan05 package only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAN04_DIR = ROOT / "reports" / "S8_AUDIT_VALIDATION" / "S8-Plan04"
OUT_DIR = ROOT / "reports" / "S8_AUDIT_VALIDATION" / "S8-Plan05"

PLAN_ID = "S8-Plan05"
GENERATED_AT = "2026-09-22"
BRANCH = "codex/s8-plan05-audit-validation-freeze"
BASE_COMMIT = "6aa4c88c98ad5e526466291541c21b71b4057fc8"
MATCHING_CONTRACT = "S8_PLAN05_MATCHING_V2"

REQUIRED_MATCHING_STATUSES = {
    "S1_CAUGHT_EXACT",
    "S1_CAUGHT_NORMALIZED_LABEL",
    "S1_CAUGHT_BLOCK_RELATION",
    "S1_CAUGHT_WARNING",
    "S1_MISSED",
    "NO_COMPARABLE_S1_RULE",
    "UNRESOLVED_MATCH",
}
CAUGHT_STATUSES = {
    "S1_CAUGHT_EXACT",
    "S1_CAUGHT_NORMALIZED_LABEL",
    "S1_CAUGHT_BLOCK_RELATION",
    "S1_CAUGHT_WARNING",
}
ALLOWED_BLOCK_DECISIONS = {
    "QUARANTINE",
    "LABEL_REMAP",
    "PRESERVE_WITH_WARNING",
    "PRESERVE_NO_ISSUE",
    "REPAIR",
    "INVALIDATE",
    "DERIVED_RECONSTRUCTION",
}

INPUT_FILES = {
    "inventory": "s1_audit_decision_inventory.csv",
    "candidates": "s1_false_negative_candidates.csv",
    "decision_validation": "s1_decision_validation_results.csv",
    "external": "external_verification_universe.csv",
    "year_shift": "year_shift_candidates.csv",
    "semantics": "indicator_semantics_validation.csv",
    "plan04_scorecard": "s1_audit_scorecard.csv",
    "plan04_scorecard_json": "s1_audit_scorecard.json",
}


def read_csv(name: str) -> pd.DataFrame:
    path = PLAN04_DIR / name
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def as_int(value: Any) -> int | None:
    text = clean(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def as_float(value: Any) -> float | None:
    text = clean(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(
        OUT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def stable_join(values: Iterable[Any]) -> str:
    cleaned = {clean(value) for value in values if clean(value)}
    return ";".join(sorted(cleaned))


def build_raw_label_normalization_map(
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    """Derive label mappings from S1 LABEL_REMAP evidence."""

    rows: list[dict[str, Any]] = []
    city_map: dict[tuple[str, str], str] = {}
    for row in inventory.itertuples(index=False):
        if clean(row.decision_type) != "LABEL_REMAP":
            continue
        raw = clean(row.original_value)
        canonical = clean(row.canonical_value)
        if not raw or not canonical:
            continue
        field = clean(row.field)
        scope = "CITY_LABEL" if field == "city_label" else "BLOCK_IDENTITY"
        map_type = (
            "TRAILING_WHITESPACE"
            if scope == "CITY_LABEL" and raw != raw.strip() and raw.strip() == canonical
            else "MALFORMED_LABEL"
            if scope == "CITY_LABEL"
            else "BLOCK_REMAP"
        )
        rows.append(
            {
                "dataset": clean(row.dataset),
                "mapping_scope": scope,
                "raw_label": raw,
                "canonical_label": canonical,
                "map_type": map_type,
                "field": field,
                "decision_id": clean(row.decision_id),
                "decision_type": clean(row.decision_type),
                "s1_action": clean(row.s1_action),
                "s1_reason": clean(row.s1_reason),
                "s1_evidence": clean(row.s1_evidence),
                "source_artifact": clean(row.source_artifact),
            }
        )
        if scope == "CITY_LABEL":
            city_map[(clean(row.dataset), raw)] = canonical

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AssertionError("S1 LABEL_REMAP evidence did not produce a map")
    frame = frame.sort_values(
        ["dataset", "mapping_scope", "raw_label", "canonical_label", "decision_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame, city_map


def canonical_city_id(
    raw_value: Any,
    dataset: str,
    city_map: dict[tuple[str, str], str],
    canonical_ids: set[str],
) -> str:
    raw = clean(raw_value)
    if not raw:
        return ""
    mapped = city_map.get((dataset, raw))
    if mapped:
        return mapped
    mapped = city_map.get(("", raw))
    if mapped:
        return mapped
    if raw in canonical_ids:
        return raw
    stripped = raw.strip()
    if (dataset, stripped) in city_map:
        return city_map[(dataset, stripped)]
    if stripped in canonical_ids:
        return stripped
    return raw


def parse_relation(text: str) -> tuple[str, float | None]:
    target_match = re.search(r"suspected_target_city=([^;]+)", text)
    shift_match = re.search(r"suspected_year_shift=([+-]?\d+(?:\.\d+)?)", text)
    target = target_match.group(1).strip() if target_match else ""
    shift = as_float(shift_match.group(1)) if shift_match else None
    return target, shift


def decision_is_allowed(row: dict[str, Any]) -> bool:
    return clean(row.get("decision_type")) in ALLOWED_BLOCK_DECISIONS


def sort_decision_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: clean(row.get("decision_id")))


def match_candidate(
    candidate: dict[str, Any],
    inventory_records: list[dict[str, Any]],
    city_map: dict[tuple[str, str], str],
    canonical_ids: set[str],
    comparable_fields: dict[str, set[str]],
) -> dict[str, Any]:
    dataset = clean(candidate.get("dataset"))
    raw_city = clean(candidate.get("city_id"))
    raw_paired_city = clean(candidate.get("paired_city_id"))
    canonical_city = canonical_city_id(raw_city, dataset, city_map, canonical_ids)
    canonical_paired_city = canonical_city_id(
        raw_paired_city, dataset, city_map, canonical_ids
    )
    year = as_int(candidate.get("year"))
    field = clean(candidate.get("field"))
    candidate_type = clean(candidate.get("candidate_type"))

    def same_year(row: dict[str, Any]) -> bool:
        row_year = as_int(row.get("year"))
        return row_year is not None and year is not None and row_year == year

    exact_matches = [
        row
        for row in inventory_records
        if clean(row.get("dataset")) == dataset
        and clean(row.get("city_id")) == canonical_city
        and same_year(row)
        and clean(row.get("field")) == field
        and decision_is_allowed(row)
    ]

    normalized_label_matches: list[dict[str, Any]] = []
    if field == "city_label" and (raw_city != canonical_city or raw_city not in canonical_ids):
        normalized_label_matches = [
            row
            for row in inventory_records
            if clean(row.get("dataset")) == dataset
            and clean(row.get("city_id")) == canonical_city
            and clean(row.get("field")) == "city_label"
            and clean(row.get("decision_type")) == "LABEL_REMAP"
        ]

    pair_matches: list[dict[str, Any]] = []
    if candidate_type == "SHIFTED_DUPLICATE_TRAJECTORY":
        offset = as_float(candidate.get("candidate_offset"))
        left = canonical_city
        right = canonical_paired_city
        for row in inventory_records:
            if (
                clean(row.get("dataset")) != dataset
                or clean(row.get("field")) != field
                or not decision_is_allowed(row)
            ):
                continue
            decision_city = clean(row.get("city_id"))
            if decision_city not in {left, right}:
                continue
            evidence = f"{clean(row.get('s1_reason'))};{clean(row.get('s1_evidence'))}"
            target_raw, decision_shift = parse_relation(evidence)
            target = canonical_city_id(target_raw, dataset, city_map, canonical_ids)
            if target not in {left, right} or target == decision_city:
                continue
            if {decision_city, target} != {left, right}:
                continue
            if offset is None or decision_shift is None:
                continue
            if abs(decision_shift - offset) <= 1e-9 or abs(decision_shift + offset) <= 1e-9:
                pair_matches.append(row)
        pair_matches = sort_decision_records(pair_matches)

    block_matches: list[dict[str, Any]] = []
    if candidate_type == "DUPLICATE_KEY" and canonical_city:
        block_matches = [
            row
            for row in inventory_records
            if clean(row.get("dataset")) == dataset
            and clean(row.get("city_id")) == canonical_city
            and same_year(row)
            and decision_is_allowed(row)
        ]

    warning_matches = [
        row
        for row in inventory_records
        if clean(row.get("dataset")) == dataset
        and clean(row.get("city_id")) == canonical_city
        and clean(row.get("field")) == field
        and clean(row.get("decision_type"))
        in {"PRESERVE_WITH_WARNING", "PRESERVE_NO_ISSUE"}
    ]

    matches: list[dict[str, Any]] = []
    match_basis = ""
    normalized = False
    if pair_matches:
        matches = pair_matches
        match_basis = "pairwise shifted-trajectory relation"
    elif block_matches:
        matches = block_matches
        match_basis = "same city-year block decision"
    elif normalized_label_matches:
        matches = normalized_label_matches
        match_basis = "S1 label-remap evidence after raw-label normalization"
        normalized = True
    elif exact_matches:
        matches = exact_matches
        match_basis = "exact canonical dataset-city-year-field decision"
    elif warning_matches:
        matches = warning_matches
        match_basis = "city-field warning decision with open year scope"

    matches = sort_decision_records(matches)
    matched_ids = [clean(row.get("decision_id")) for row in matches]
    matched_years = [
        as_int(row.get("year")) for row in matches if as_int(row.get("year")) is not None
    ]
    matched_types = [clean(row.get("decision_type")) for row in matches]
    matched_actions = [clean(row.get("s1_action")) for row in matches]
    has_warning = any(
        clean(row.get("decision_type"))
        in {"PRESERVE_WITH_WARNING", "PRESERVE_NO_ISSUE"}
        for row in matches
    )

    if matches:
        if pair_matches or block_matches:
            matching_status = "S1_CAUGHT_BLOCK_RELATION"
        elif normalized:
            matching_status = "S1_CAUGHT_NORMALIZED_LABEL"
        elif has_warning:
            matching_status = "S1_CAUGHT_WARNING"
        else:
            matching_status = "S1_CAUGHT_EXACT"
    else:
        comparable = field in comparable_fields.get(dataset, set())
        if clean(candidate.get("external_anomaly")) == "YES":
            matching_status = "S1_MISSED"
        elif not comparable:
            matching_status = "NO_COMPARABLE_S1_RULE"
        else:
            matching_status = "UNRESOLVED_MATCH"

    external_anomaly = clean(candidate.get("external_anomaly")) == "YES"
    if matching_status in CAUGHT_STATUSES:
        posthoc_status = "S1_CAUGHT"
        classification = "S1_CAUGHT"
        resolution_class = "S1_CAUGHT"
    elif external_anomaly:
        posthoc_status = "S1_MISSED"
        classification = "FALSE_NEGATIVE_MISSED_ANOMALY"
        resolution_class = "VERIFIED_FALSE_NEGATIVE"
    else:
        old_status = clean(candidate.get("posthoc_status"))
        posthoc_status = "REAL_CHANGE" if old_status == "REAL_CHANGE" else "UNRESOLVED"
        classification = "UNVERIFIED_CANDIDATE"
        resolution_class = "UNVERIFIED_CANDIDATE"

    support_years = as_float(candidate.get("support_years"))
    denominator = support_years if support_years and support_years > 0 else 1.0
    coverage = len(matched_ids) / denominator if matched_ids else 0.0
    coverage = min(coverage, 1.0)

    row = dict(candidate)
    row["plan04_s1_matching_decision_ids"] = clean(
        candidate.get("s1_matching_decision_ids")
    )
    row.update(
        {
            "candidate_raw_city_id": raw_city,
            "candidate_canonical_city_id": canonical_city,
            "paired_raw_city_id": raw_paired_city,
            "paired_canonical_city_id": canonical_paired_city,
            "matching_contract": MATCHING_CONTRACT,
            "matching_status": matching_status,
            "match_basis": match_basis,
            "matched_decision_count": len(matched_ids),
            "matched_decision_ids": ";".join(matched_ids),
            "matched_decision_types": stable_join(matched_types),
            "matched_s1_actions": stable_join(matched_actions),
            "matched_year_min": min(matched_years) if matched_years else "",
            "matched_year_max": max(matched_years) if matched_years else "",
            "match_coverage": round(coverage, 6),
            "candidate_found_posthoc": "YES",
            "was_flagged_in_S1": "YES" if matching_status in CAUGHT_STATUSES else "NO",
            "s1_matching_decision_ids": ";".join(matched_ids),
            "posthoc_status": posthoc_status,
            "classification": classification,
            "resolution_class": resolution_class,
            "miss_severity": clean(candidate.get("miss_severity"))
            if resolution_class == "VERIFIED_FALSE_NEGATIVE"
            else "",
        }
    )
    return row


def matching_audit(
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map_frame, city_map = build_raw_label_normalization_map(inventory)
    canonical_ids = {
        clean(value)
        for value in inventory.city_id.tolist()
        if re.fullmatch(r"city(?:[1-9]|[1-3][0-9]|40)", clean(value))
    }
    canonical_ids.update(f"city{i}" for i in range(1, 41))
    inventory_records = inventory.to_dict(orient="records")
    comparable_fields: dict[str, set[str]] = {}
    for row in inventory_records:
        dataset = clean(row.get("dataset"))
        field = clean(row.get("field"))
        if dataset and field:
            comparable_fields.setdefault(dataset, set()).add(field)

    rows = [
        match_candidate(
            candidate.to_dict(),
            inventory_records,
            city_map,
            canonical_ids,
            comparable_fields,
        )
        for _, candidate in candidates.iterrows()
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AssertionError("Plan04 candidate inventory is unexpectedly empty")

    base_columns = list(candidates.columns)
    additional_columns = [
        "plan04_s1_matching_decision_ids",
        "candidate_raw_city_id",
        "candidate_canonical_city_id",
        "paired_raw_city_id",
        "paired_canonical_city_id",
        "matching_contract",
        "matching_status",
        "match_basis",
        "matched_decision_count",
        "matched_decision_ids",
        "matched_decision_types",
        "matched_s1_actions",
        "matched_year_min",
        "matched_year_max",
        "match_coverage",
        "resolution_class",
    ]
    ordered = [
        column for column in base_columns + additional_columns if column in frame.columns
    ]
    ordered += [column for column in frame.columns if column not in ordered]
    return frame[ordered], label_map_frame


def decision_validation_v2(validation: pd.DataFrame) -> pd.DataFrame:
    frame = validation.copy()

    def level(row: pd.Series) -> str:
        external = clean(row.external_validation_status)
        klass = clean(row.validation_class)
        if external.startswith("EXTERNALLY_VERIFIED"):
            return (
                "EXTERNALLY_VERIFIED_CORRECT"
                if klass.startswith("CORRECT")
                else "EXTERNALLY_VERIFIED_WRONG"
            )
        if klass.startswith("CORRECT"):
            return "INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED"
        return "CAUTIOUS_UNRESOLVED"

    frame["interpretation_level"] = frame.apply(level, axis=1)
    frame["interpretation_note"] = frame["interpretation_level"].map(
        {
            "EXTERNALLY_VERIFIED_CORRECT": "Definition-matched official exact-cell comparison supports the recorded S1 action.",
            "EXTERNALLY_VERIFIED_WRONG": "Definition-matched official exact-cell comparison conflicts with the recorded S1 action.",
            "INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED": "S1 historical action is internally defensible; external exact-cell validation is unavailable.",
            "CAUTIOUS_UNRESOLVED": "The available evidence does not support an external correctness claim.",
        }
    )
    return frame


def scorecard_v2(
    inventory: pd.DataFrame,
    validation: pd.DataFrame,
    candidates: pd.DataFrame,
    plan04_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for old in plan04_scorecard.to_dict(orient="records"):
        dataset = clean(old.get("dataset"))
        inv = inventory[inventory.dataset == dataset]
        val = validation[validation.dataset == dataset]
        cand = candidates[candidates.dataset == dataset]
        externally_verified_decisions = int(
            val.interpretation_level.str.startswith("EXTERNALLY_VERIFIED").sum()
        ) if not val.empty else 0
        externally_verified_correct = int(
            (val.interpretation_level == "EXTERNALLY_VERIFIED_CORRECT").sum()
        ) if not val.empty else 0
        externally_verified_wrong = int(
            (val.interpretation_level == "EXTERNALLY_VERIFIED_WRONG").sum()
        ) if not val.empty else 0
        internally_supported = int(
            (
                val.interpretation_level
                == "INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED"
            ).sum()
        ) if not val.empty else 0
        cautious_unresolved = int(
            (val.interpretation_level == "CAUTIOUS_UNRESOLVED").sum()
        ) if not val.empty else 0
        verified_anomalies = int(
            (cand.external_anomaly == "YES").sum()
        ) if not cand.empty else 0
        caught = int(
            (
                (cand.external_anomaly == "YES")
                & cand.matching_status.isin(CAUGHT_STATUSES)
            ).sum()
        ) if not cand.empty else 0
        missed = int(
            (
                (cand.external_anomaly == "YES")
                & (cand.matching_status == "S1_MISSED")
            ).sum()
        ) if not cand.empty else 0
        critical = int(
            (
                (cand.resolution_class == "VERIFIED_FALSE_NEGATIVE")
                & (cand.miss_severity == "CRITICAL")
            ).sum()
        ) if not cand.empty else 0
        precision = (
            externally_verified_correct / externally_verified_decisions
            if externally_verified_decisions
            else np.nan
        )
        recall = caught / verified_anomalies if verified_anomalies else np.nan
        externally_verified_cells = as_int(old.get("externally_verified_cells")) or 0
        if externally_verified_cells == 0:
            assessment = "INSUFFICIENT_EXTERNAL_EVIDENCE"
        elif critical:
            assessment = "MIXED"
        elif precision == precision and precision >= 0.9 and (
            recall != recall or recall >= 0.8
        ):
            assessment = "STRONG"
        else:
            assessment = "ADEQUATE_WITH_GAPS"
        caught_candidate_count = int(
            cand.matching_status.isin(CAUGHT_STATUSES).sum()
        ) if not cand.empty else 0
        unresolved_candidate_count = int(
            cand.matching_status.isin(
                {"UNRESOLVED_MATCH", "NO_COMPARABLE_S1_RULE"}
            ).sum()
        ) if not cand.empty else 0
        rows.append(
            {
                "dataset": dataset,
                "externally_verified_cells": externally_verified_cells,
                "s1_material_decisions": len(inv),
                "externally_verified_decisions": externally_verified_decisions,
                "externally_verified_correct_decisions": externally_verified_correct,
                "externally_verified_wrong_decisions": externally_verified_wrong,
                "internally_supported_not_externally_verified": internally_supported,
                "cautious_unresolved_decisions": cautious_unresolved,
                "historically_defensible_s1_decisions": externally_verified_correct
                + internally_supported,
                "verified_correct_decisions": externally_verified_correct,
                "correct_decisions": externally_verified_correct + internally_supported,
                "wrong_decisions": externally_verified_wrong,
                "false_positive_count": externally_verified_wrong,
                "not_externally_verifiable_decisions": len(inv)
                - externally_verified_decisions,
                "verified_anomaly_count": verified_anomalies,
                "caught_anomaly_count": caught,
                "missed_anomaly_count": missed,
                "verified_universe_precision": precision,
                "verified_universe_recall": recall,
                "critical_misses": critical,
                "s1_caught_candidate_count": caught_candidate_count,
                "unresolved_candidate_count": unresolved_candidate_count,
                "overall_assessment": assessment,
                "interpretation_note": "correct_decisions is historical/internal defensibility, not universal external proof",
            }
        )
    return pd.DataFrame(rows)


def structural_summary(
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    year_shift: pd.DataFrame,
    semantics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_summary(
        candidate_type: str,
        dataset: str,
        frame: pd.DataFrame,
        material: str,
        notes: str,
        decision_count_override: int | None = None,
    ) -> None:
        caught = int(frame.matching_status.isin(CAUGHT_STATUSES).sum()) if not frame.empty else 0
        missed = int((frame.matching_status == "S1_MISSED").sum()) if not frame.empty else 0
        external = int((frame.external_status == "EXTERNALLY_VERIFIED").sum()) if not frame.empty else 0
        unresolved = int(
            frame.matching_status.isin(
                {"UNRESOLVED_MATCH", "NO_COMPARABLE_S1_RULE"}
            ).sum()
        ) if not frame.empty else 0
        rows.append(
            {
                "candidate_type": candidate_type,
                "dataset": dataset,
                "candidate_count": decision_count_override
                if decision_count_override is not None
                else len(frame),
                "s1_caught_count": caught,
                "s1_missed_count": missed,
                "externally_verified_count": external,
                "unresolved_count": unresolved,
                "material_new_findings": material,
                "evidence": notes,
            }
        )

    grouped = candidates.groupby(
        ["candidate_type", "dataset"], sort=True, dropna=False
    )
    for (candidate_type, dataset), frame in grouped:
        candidate_type_text = clean(candidate_type)
        if candidate_type_text == "SHIFTED_DUPLICATE_TRAJECTORY":
            material = (
                "S1_CAUGHT_BLOCK_RELATION"
                if dataset == "employees"
                else "NO_NEW_CONFIRMED_FINDING"
            )
        elif candidate_type_text == "INVALID_CITY_LABEL":
            material = "S1_CAUGHT_NORMALIZED_LABEL" if (
                frame.matching_status.isin(CAUGHT_STATUSES).all()
            ) else "UNRESOLVED_RAW_LABEL_EVIDENCE"
        elif (frame.resolution_class == "VERIFIED_FALSE_NEGATIVE").any():
            material = "CITY40_CRITICAL_FALSE_NEGATIVE_RETAINED"
        elif frame.matching_status.isin(CAUGHT_STATUSES).any():
            material = "S1_CAUGHT_INTERNAL_RULE_MATCH"
        else:
            material = "NO_NEW_CONFIRMED_FINDING"
        append_summary(
            candidate_type_text,
            clean(dataset),
            frame,
            material,
            "Plan04 candidate rows resolved under the Plan05 matching contract.",
        )

    permutation = candidates[
        candidates.candidate_type == "DUPLICATED_CITY_TRAJECTORY"
    ]
    append_summary(
        "CITY_PERMUTATION",
        "ALL",
        permutation,
        "NO_CONFIRMED_GLOBAL_PERMUTATION",
        "No candidate survived as a confirmed canonical multi-year city permutation.",
    )

    true_shift = year_shift[year_shift.status == "TRUE_YEAR_SHIFT"]
    true_shift_candidates = candidates[
        (candidates.candidate_type == "SHIFTED_DUPLICATE_TRAJECTORY")
        & (candidates.status == "TRUE_YEAR_SHIFT")
    ]
    append_summary(
        "YEAR_SHIFT",
        "employees",
        true_shift_candidates,
        "S1_CAUGHT_BLOCK_RELATION",
        "Both TRUE_YEAR_SHIFT blocks match historical S1 quarantine rows.",
        decision_count_override=len(true_shift),
    )

    block_count = int(
        (
            (inventory.dataset == "overall_income")
            & (inventory.decision_type == "QUARANTINE")
        ).sum()
    )
    block_frame = pd.DataFrame(
        {
            "matching_status": ["S1_CAUGHT_EXACT"] * block_count,
            "external_status": ["NOT_EXTERNALLY_VERIFIABLE"] * block_count,
        }
    )
    append_summary(
        "BLOCK_CONTAMINATION",
        "overall_income",
        block_frame,
        "S1_QUARANTINE_RETAINED",
        "The existing contaminated income block remains an internally supported S1 quarantine boundary.",
        decision_count_override=block_count,
    )

    for row in semantics.sort_values(
        ["dataset", "check_id"], kind="mergesort"
    ).itertuples(index=False):
        count = as_int(row.candidate_count) or 0
        covered = 1 if clean(row.s1_covered) in {"YES", "PARTIAL"} else 0
        unresolved = 1 if re.search(
            r"CONFLICT|POSSIBLE|UNRESOLVED", clean(row.status)
        ) else 0
        rows.append(
            {
                "candidate_type": "INDICATOR_SEMANTICS",
                "dataset": clean(row.dataset),
                "candidate_count": count,
                "s1_caught_count": covered,
                "s1_missed_count": 0,
                "externally_verified_count": 0,
                "unresolved_count": unresolved,
                "material_new_findings": clean(row.status),
                "evidence": clean(row.notes),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["candidate_type", "dataset"], kind="mergesort"
    ).reset_index(drop=True)


def metric_delta(
    old_candidates: pd.DataFrame,
    new_candidates: pd.DataFrame,
    old_validation: pd.DataFrame,
    new_validation: pd.DataFrame,
    old_scorecard: pd.DataFrame,
    new_scorecard: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(metric: str, old: Any, new: Any, reason: str) -> None:
        old_float = as_float(old)
        new_float = as_float(new)
        if old_float is not None and new_float is not None:
            difference = new_float - old_float
            delta: Any = int(difference) if difference.is_integer() else difference
        else:
            delta = "status corrected"
        rows.append(
            {
                "metric": metric,
                "plan04_value": old,
                "plan05_value": new,
                "delta": delta,
                "reason": reason,
            }
        )

    old_misses = int(
        (old_candidates.classification == "FALSE_NEGATIVE_MISSED_ANOMALY").sum()
    )
    new_misses = int(
        (new_candidates.resolution_class == "VERIFIED_FALSE_NEGATIVE").sum()
    )
    old_critical = int((old_candidates.miss_severity == "CRITICAL").sum())
    new_critical = int(
        (
            (new_candidates.resolution_class == "VERIFIED_FALSE_NEGATIVE")
            & (new_candidates.miss_severity == "CRITICAL")
        ).sum()
    )
    old_shift = old_candidates[
        (old_candidates.candidate_type == "SHIFTED_DUPLICATE_TRAJECTORY")
        & (old_candidates.status == "TRUE_YEAR_SHIFT")
    ]
    new_shift = new_candidates[
        (new_candidates.candidate_type == "SHIFTED_DUPLICATE_TRAJECTORY")
        & (new_candidates.status == "TRUE_YEAR_SHIFT")
    ]
    old_ctiy = old_candidates[old_candidates.city_id.isin(["ctiy22", "ctiy31"])]
    new_ctiy = new_candidates[
        new_candidates.candidate_raw_city_id.isin(["ctiy22", "ctiy31"])
    ]
    old_trailing = old_candidates[old_candidates.city_id == "city29 "]
    new_trailing = new_candidates[
        new_candidates.candidate_raw_city_id == "city29 "
    ]
    old_ext_correct = int(
        (
            old_validation.external_validation_status.str.startswith(
                "EXTERNALLY_VERIFIED"
            )
            & old_validation.validation_class.str.startswith("CORRECT")
        ).sum()
    )
    new_ext_correct = int(
        (new_validation.interpretation_level == "EXTERNALLY_VERIFIED_CORRECT").sum()
    )
    old_ext = int(
        old_validation.external_validation_status.str.startswith(
            "EXTERNALLY_VERIFIED"
        ).sum()
    )
    new_ext = int(
        new_validation.interpretation_level.str.startswith("EXTERNALLY_VERIFIED").sum()
    )

    add(
        "verified_false_negative_count",
        old_misses,
        new_misses,
        "unchanged: matching cleanup does not change the definition-matched official universe or city40 comparison.",
    )
    add(
        "critical_miss_count",
        old_critical,
        new_critical,
        "unchanged: city40 registered=65 remains the same verified critical miss.",
    )
    add(
        "true_year_shift_s1_caught_count",
        int((old_shift.was_flagged_in_S1 == "YES").sum()),
        int(new_shift.matching_status.isin(CAUGHT_STATUSES).sum()),
        "The two employment shifted trajectories now match city25/city29 quarantine blocks by pairwise relation.",
    )
    add(
        "true_year_shift_uncaught_count",
        int((old_shift.was_flagged_in_S1 != "YES").sum()),
        int((~new_shift.matching_status.isin(CAUGHT_STATUSES)).sum()),
        "The former false non-match is removed by block-level relation matching.",
    )
    add(
        "malformed_ctiy_label_s1_caught_count",
        int((old_ctiy.was_flagged_in_S1 == "YES").sum()),
        int(
            (
                new_ctiy.matching_status == "S1_CAUGHT_NORMALIZED_LABEL"
            ).sum()
        ),
        "ctiy22 and ctiy31 now resolve through the S1-derived raw-label map.",
    )
    add(
        "trailing_space_city29_label_caught_count",
        int((old_trailing.was_flagged_in_S1 == "YES").sum()),
        int(
            (
                new_trailing.matching_status == "S1_CAUGHT_NORMALIZED_LABEL"
            ).sum()
        ),
        "city29 trailing-space rows retain their caught status with explicit canonical identity.",
    )
    add(
        "externally_verified_correct_decisions",
        old_ext_correct,
        new_ext_correct,
        "unchanged: no new external research and the Plan04 external universe is reused as frozen evidence.",
    )
    add(
        "externally_verified_decisions",
        old_ext,
        new_ext,
        "unchanged: matching cleanup changes candidate-to-decision attribution, not exact-cell external evidence.",
    )
    old_not_ext = len(old_validation) - old_ext
    new_not_ext = len(new_validation) - new_ext
    add(
        "not_externally_verifiable_decisions",
        old_not_ext,
        new_not_ext,
        "unchanged: non-external decisions remain explicitly internal or cautious, never externally proven.",
    )
    add(
        "candidate_rows",
        len(old_candidates),
        len(new_candidates),
        "unchanged: Plan05 reuses the Plan04 candidate universe and only repairs matching.",
    )
    add(
        "s1_caught_candidate_rows",
        int((old_candidates.posthoc_status == "S1_CAUGHT").sum()),
        int(new_candidates.matching_status.isin(CAUGHT_STATUSES).sum()),
        "The count changes only where canonical label and pairwise block relations were previously missed.",
    )
    add(
        "verified_missed_anomalies_in_scorecard",
        sum(as_int(value) or 0 for value in old_scorecard.missed_anomaly_count),
        sum(as_int(value) or 0 for value in new_scorecard.missed_anomaly_count),
        "unchanged: the only externally verified anomaly remains city40 registered=65.",
    )
    return pd.DataFrame(rows)


def validate_known_cases(
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    normalization_map: pd.DataFrame,
    scorecard: pd.DataFrame,
    structural: pd.DataFrame,
) -> dict[str, Any]:
    statuses = set(candidates.matching_status)
    if not statuses.issubset(REQUIRED_MATCHING_STATUSES):
        raise AssertionError(
            f"unknown matching statuses: {statuses - REQUIRED_MATCHING_STATUSES}"
        )
    required_map = {
        ("urban_income", "ctiy22", "city22"),
        ("urban_income", "ctiy31", "city31"),
        ("employees", "city29 ", "city29"),
    }
    map_keys = {
        (clean(row.dataset), clean(row.raw_label), clean(row.canonical_label))
        for row in normalization_map.itertuples(index=False)
        if clean(row.mapping_scope) == "CITY_LABEL"
    }
    if not required_map.issubset(map_keys):
        raise AssertionError(f"required label map entries missing: {required_map - map_keys}")

    city40 = candidates[
        (candidates.dataset == "population")
        & (candidates.candidate_raw_city_id == "city40")
        & (candidates.candidate_canonical_city_id == "city40")
        & (candidates.year == "2021")
        & (candidates.field == "户籍人口（万人）")
    ]
    if len(city40) != 1:
        raise AssertionError(f"expected one city40 registered candidate, got {len(city40)}")
    city40_row = city40.iloc[0]
    for column, expected in {
        "current_value": "65",
        "was_flagged_in_S1": "NO",
        "matching_status": "S1_MISSED",
        "classification": "FALSE_NEGATIVE_MISSED_ANOMALY",
        "resolution_class": "VERIFIED_FALSE_NEGATIVE",
        "miss_severity": "CRITICAL",
    }.items():
        actual = (
            str(as_int(city40_row[column]))
            if column == "current_value" and as_int(city40_row[column]) is not None
            else clean(city40_row[column])
        )
        if actual != expected:
            raise AssertionError(
                f"city40 assertion failed for {column}: {city40_row[column]}"
            )

    shift_cases = {
        ("city2", "city25", "1"): "city25",
        ("city29", "city8", "-1"): "city29",
    }
    shift = candidates[
        (candidates.candidate_type == "SHIFTED_DUPLICATE_TRAJECTORY")
        & (candidates.dataset == "employees")
        & (candidates.status == "TRUE_YEAR_SHIFT")
    ]
    if len(shift) != 2:
        raise AssertionError(f"expected two TRUE_YEAR_SHIFT candidates, got {len(shift)}")
    for key, expected_city in shift_cases.items():
        row = shift[
            (shift.candidate_canonical_city_id == key[0])
            & (shift.paired_canonical_city_id == key[1])
            & shift.candidate_offset.map(as_float).eq(float(key[2]))
        ]
        if len(row) != 1:
            raise AssertionError(f"missing TRUE_YEAR_SHIFT case {key}")
        record = row.iloc[0]
        if clean(record.matching_status) != "S1_CAUGHT_BLOCK_RELATION":
            raise AssertionError(f"shift not caught by block relation: {key}")
        if clean(record.was_flagged_in_S1) != "YES":
            raise AssertionError(f"shift was not flagged: {key}")
        if as_int(record.matched_decision_count) != 19:
            raise AssertionError(f"shift match count is not 19: {key}")
        if as_int(record.matched_year_min) != 2003 or as_int(record.matched_year_max) != 2021:
            raise AssertionError(f"shift block year range is wrong: {key}")
        if abs((as_float(record.match_coverage) or 0.0) - 1.0) > 1e-9:
            raise AssertionError(f"shift coverage is not 1.0: {key}")
        ids = set(clean(record.matched_decision_ids).split(";"))
        matching_inventory = inventory[
            (inventory.dataset == "employees")
            & (inventory.city_id == expected_city)
            & (inventory.decision_type == "QUARANTINE")
        ]
        if not set(matching_inventory.decision_id).issubset(ids):
            raise AssertionError(f"shift did not match expected quarantine block: {key}")

    for raw, canonical in [("ctiy22", "city22"), ("ctiy31", "city31")]:
        row = candidates[
            (candidates.candidate_raw_city_id == raw)
            & (candidates.candidate_canonical_city_id == canonical)
            & (candidates.field == "city_label")
        ]
        if len(row) != 1:
            raise AssertionError(f"expected one malformed label candidate for {raw}")
        record = row.iloc[0]
        if clean(record.was_flagged_in_S1) != "YES":
            raise AssertionError(f"{raw} was not caught")
        if clean(record.matching_status) != "S1_CAUGHT_NORMALIZED_LABEL":
            raise AssertionError(f"{raw} did not use normalized-label matching")
        if "LABEL_REMAP" not in clean(record.matched_decision_types):
            raise AssertionError(f"{raw} did not match LABEL_REMAP evidence")

    trailing = candidates[
        (candidates.candidate_raw_city_id == "city29 ")
        & (candidates.dataset == "employees")
        & (candidates.field == "city_label")
    ]
    if len(trailing) != 36:
        raise AssertionError(
            f"expected 36 trailing-space label candidates, got {len(trailing)}"
        )
    if not (
        (trailing.candidate_canonical_city_id == "city29").all()
        and (trailing.matching_status == "S1_CAUGHT_NORMALIZED_LABEL").all()
        and (trailing.was_flagged_in_S1 == "YES").all()
    ):
        raise AssertionError("trailing-space city29 labels were not canonically caught")

    if int(scorecard.s1_material_decisions.sum()) != len(inventory):
        raise AssertionError("scorecard decision counts do not reconcile to inventory")
    if int(scorecard.missed_anomaly_count.sum()) != 1:
        raise AssertionError("scorecard must retain exactly one verified missed anomaly")
    if int(scorecard.critical_misses.sum()) != 1:
        raise AssertionError("scorecard must retain exactly one critical miss")
    permutation = structural[structural.candidate_type == "CITY_PERMUTATION"]
    if len(permutation) != 1 or as_int(permutation.iloc[0].candidate_count) != 0:
        raise AssertionError("confirmed global permutation was not resolved to zero")

    return {
        "city40_registered": True,
        "true_year_shift_cases": 2,
        "malformed_label_cases": 2,
        "trailing_space_label_rows": len(trailing),
        "confirmed_global_permutation": False,
        "verified_false_negatives": int(scorecard.missed_anomaly_count.sum()),
        "critical_misses": int(scorecard.critical_misses.sum()),
    }


def score_value(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return clean(value)


def build_report(
    candidates: pd.DataFrame,
    inventory: pd.DataFrame,
    validation: pd.DataFrame,
    scorecard: pd.DataFrame,
    structural: pd.DataFrame,
    delta: pd.DataFrame,
    plan04_scorecard_json: dict[str, Any],
    assertions: dict[str, Any],
) -> str:
    externally_correct = int(
        (validation.interpretation_level == "EXTERNALLY_VERIFIED_CORRECT").sum()
    )
    externally_wrong = int(
        (validation.interpretation_level == "EXTERNALLY_VERIFIED_WRONG").sum()
    )
    internal = int(
        (
            validation.interpretation_level
            == "INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED"
        ).sum()
    )
    not_external = int(
        (~validation.interpretation_level.str.startswith("EXTERNALLY_VERIFIED")).sum()
    )
    score_lines: list[str] = []
    for row in scorecard.itertuples(index=False):
        score_lines.append(
            f"| {row.dataset} | {row.externally_verified_cells} | "
            f"{row.s1_material_decisions} | {row.externally_verified_decisions} | "
            f"{row.externally_verified_correct_decisions} | "
            f"{row.externally_verified_wrong_decisions} | "
            f"{row.internally_supported_not_externally_verified} | "
            f"{row.verified_anomaly_count} | {row.caught_anomaly_count} | "
            f"{row.missed_anomaly_count} | {score_value(row.verified_universe_precision)} | "
            f"{score_value(row.verified_universe_recall)} | {row.critical_misses} | "
            f"{row.overall_assessment} |"
        )
    selected_structural = structural[
        structural.candidate_type.isin(
            [
                "CITY_PERMUTATION",
                "YEAR_SHIFT",
                "BLOCK_CONTAMINATION",
                "INDICATOR_SEMANTICS",
            ]
        )
    ]
    structural_lines = "\n".join(
        f"| {row.candidate_type} | {row.dataset} | {row.candidate_count} | "
        f"{row.s1_caught_count} | {row.s1_missed_count} | "
        f"{row.unresolved_count} | {row.material_new_findings} |"
        for row in selected_structural.itertuples(index=False)
    )
    delta_lines = "\n".join(
        f"| {row.metric} | {row.plan04_value} | {row.plan05_value} | "
        f"{row.delta} | {row.reason} |"
        for row in delta.itertuples(index=False)
    )
    return f"""# S8-Plan05｜Final S1 Audit Validation Freeze

## Result

- Status: S8_S1_AUDIT_VALIDATION_FROZEN
- Branch: {BRANCH}
- Base: Plan04 commit {BASE_COMMIT}
- Frozen Plan04 evidence package: read-only
- Plan04 source artifacts: not overwritten
- No new large-scale external research was performed

Plan05 is an audit-of-the-audit artifact cleanup. It changes only the
candidate-to-decision matching and the derived validation summaries. It does
not rewrite S1 canonical data, quarantine, provenance, S8 frozen truth, or
any S9 artifact.

## Isolation

- S9 branch read: NO
- S9 predictions read: NO
- S9 metrics read: NO
- main modified: NO
- S1 canonical/quarantine/provenance modified: NO
- S8 frozen truth modified: NO
- training, prediction scoring, and prediction error calculation: NO

## Matching fixes

The Plan04 candidate universe contains {len(candidates)} rows. Plan05 applies
four matching paths: exact canonical cell, S1-derived raw-label normalization,
pairwise shifted-trajectory relation, and block-level same-city-year coverage.
Each candidate receives one of the required matching statuses:
{", ".join(sorted(REQUIRED_MATCHING_STATUSES))}.

- Raw-label mappings are derived from S1 LABEL_REMAP inventory rows.
- ctiy22 -> city22 and ctiy31 -> city31 resolve to S1 label repairs.
- city29 trailing-space rows resolve to canonical city29 label evidence.
- The two employment TRUE_YEAR_SHIFT candidates resolve to S1 quarantine blocks:
  city2 ↔ city25 and city29 ↔ city8.
- Shift-block coverage is recorded as matched count, year range, and coverage
  instead of requiring the candidate's primary city to equal the quarantine side.

## Known assertions

- city40 / population / registered / 2021 = 65: post-hoc candidate found,
  S1 flag NO, matching status S1_MISSED, classification
  FALSE_NEGATIVE_MISSED_ANOMALY, severity CRITICAL.
- city2 ↔ city25: S1_CAUGHT_BLOCK_RELATION, 19 matched quarantine rows,
  2003–2021, coverage 1.0.
- city29 ↔ city8: S1_CAUGHT_BLOCK_RELATION, 19 matched quarantine rows,
  2003–2021, coverage 1.0.
- ctiy22 -> city22: S1_CAUGHT_NORMALIZED_LABEL.
- ctiy31 -> city31: S1_CAUGHT_NORMALIZED_LABEL.

## Evidence semantics

The report separates four non-equivalent counts:

- Historically defensible S1 decisions: {int(validation.interpretation_level.str.startswith(("EXTERNALLY_VERIFIED", "INTERNALLY_SUPPORTED")).sum())}
- Externally verified correct decisions: {externally_correct}
- Externally verified wrong decisions: {externally_wrong}
- Not externally verifiable decisions: {not_external}

The historical/internal count is not presented as universal external proof.
For the {internal} internally supported but non-external decisions, the
precise interpretation is: S1 historical action is internally defensible;
external exact-cell validation unavailable. Cautious unresolved evidence is
kept separate in the v2 decision table.

## Plan04 to Plan05 metric delta

| metric | plan04_value | plan05_value | delta | reason |
|---|---:|---:|---:|---|
{delta_lines}

## Structural candidate resolution

| candidate_type | dataset | candidate_count | s1_caught_count | s1_missed_count | unresolved_count | material_new_findings |
|---|---|---:|---:|---:|---:|---|
{structural_lines}

The city40 finding remains the only externally verified false negative in the
available definition-matched universe. The two TRUE_YEAR_SHIFT results and the
two malformed urban-income labels are not new S1 misses after matching cleanup.

## Final scorecard

| dataset | external cells | S1 decisions | external decisions | external correct | external wrong | internal only | verified anomalies | caught | missed | precision | recall | critical | assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(score_lines)}

## Required conclusions

1. True S1 false negatives: {assertions["verified_false_negatives"]} in the
   available definition-matched external universe, with
   {assertions["critical_misses"]} critical miss.
2. TRUE_YEAR_SHIFT: No. Both employment shifted duplicate blocks were already
   caught and quarantined by S1.
3. ctiy22 / ctiy31: No, not new Plan04 findings. They were already represented
   as S1 LABEL_REMAP decisions.
4. city40 registered=65: Yes, still a material miss. Plan05 did not reopen
   the official comparison and the matching fix does not weaken it.
5. Confirmed global city permutation: No.

## Reproducibility and limits

The script reads only the Plan04 evidence package and writes the new Plan05
package. Its input and output ordering is deterministic. Re-running with
identical inputs must produce identical output hashes. This package cannot
prove the true values of historical quarantine cells without definition-matched
external truth, and it does not authorize S1 remediation.

Plan04 frozen truth metadata was carried forward from its scorecard JSON:
{clean(plan04_scorecard_json.get("frozen_truth_version"))}.

## Files

- HANDOFF.md
- S1_POSTHOC_AUDIT_VALIDATION_FINAL_REPORT.md
- candidate_decision_matching_audit.csv
- raw_label_normalization_map.csv
- s1_false_negative_candidates_v2.csv
- s1_structural_candidate_resolution.csv
- s1_audit_scorecard_v2.csv
- s1_audit_scorecard_v2.json
- s1_decision_validation_results_v2.csv
- plan04_to_plan05_metric_delta.csv
- run_s8_plan05.py in scripts/
"""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if pd.isna(value):
        return None
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        name for name in INPUT_FILES.values() if not (PLAN04_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing immutable Plan04 inputs: {missing}")

    inventory = read_csv(INPUT_FILES["inventory"])
    old_candidates = read_csv(INPUT_FILES["candidates"])
    old_validation = read_csv(INPUT_FILES["decision_validation"])
    external = read_csv(INPUT_FILES["external"])
    year_shift = read_csv(INPUT_FILES["year_shift"])
    semantics = read_csv(INPUT_FILES["semantics"])
    old_scorecard = read_csv(INPUT_FILES["plan04_scorecard"])
    plan04_scorecard_json = json.loads(
        (PLAN04_DIR / INPUT_FILES["plan04_scorecard_json"]).read_text(
            encoding="utf-8"
        )
    )
    if external.empty:
        raise AssertionError("Plan04 external universe unexpectedly empty")

    candidates, normalization_map = matching_audit(old_candidates, inventory)
    validation = decision_validation_v2(old_validation)
    scorecard = scorecard_v2(inventory, validation, candidates, old_scorecard)
    structural = structural_summary(candidates, inventory, year_shift, semantics)
    delta = metric_delta(
        old_candidates,
        candidates,
        old_validation,
        validation,
        old_scorecard,
        scorecard,
    )
    assertions = validate_known_cases(
        candidates, inventory, normalization_map, scorecard, structural
    )

    write_csv(normalization_map, "raw_label_normalization_map.csv")
    write_csv(candidates, "candidate_decision_matching_audit.csv")
    write_csv(candidates, "s1_false_negative_candidates_v2.csv")
    write_csv(validation, "s1_decision_validation_results_v2.csv")
    write_csv(structural, "s1_structural_candidate_resolution.csv")
    write_csv(scorecard, "s1_audit_scorecard_v2.csv")
    write_csv(delta, "plan04_to_plan05_metric_delta.csv")

    score_json = {
        "status": "S8_S1_AUDIT_VALIDATION_FROZEN",
        "plan_id": PLAN_ID,
        "generated_at": GENERATED_AT,
        "branch": BRANCH,
        "base_commit": BASE_COMMIT,
        "matching_contract": MATCHING_CONTRACT,
        "source_plan": "S8-Plan04",
        "source_plan04_artifacts_unchanged": True,
        "s9_isolation": {
            "s9_branch_read": False,
            "s9_predictions_read": False,
            "s9_metrics_read": False,
            "main_modified": False,
            "s1_canonical_modified": False,
            "s8_frozen_truth_modified": False,
            "test_xls_accessed": False,
        },
        "inventory_rows": len(inventory),
        "candidate_rows": len(candidates),
        "matching_status_counts": candidates.matching_status.value_counts().sort_index().to_dict(),
        "resolution_class_counts": candidates.resolution_class.value_counts().sort_index().to_dict(),
        "externally_verified_correct_decisions": int(
            (validation.interpretation_level == "EXTERNALLY_VERIFIED_CORRECT").sum()
        ),
        "externally_verified_wrong_decisions": int(
            (validation.interpretation_level == "EXTERNALLY_VERIFIED_WRONG").sum()
        ),
        "internally_supported_not_externally_verified": int(
            (
                validation.interpretation_level
                == "INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED"
            ).sum()
        ),
        "not_externally_verifiable_decisions": int(
            (~validation.interpretation_level.str.startswith("EXTERNALLY_VERIFIED")).sum()
        ),
        "verified_false_negatives": assertions["verified_false_negatives"],
        "critical_misses": assertions["critical_misses"],
        "known_assertions": assertions,
        "scorecard": json_safe(scorecard.to_dict(orient="records")),
        "plan04_scorecard_source_sha256": sha256(
            PLAN04_DIR / INPUT_FILES["plan04_scorecard_json"]
        ),
        "source_hashes": {
            filename: sha256(PLAN04_DIR / filename)
            for filename in sorted(INPUT_FILES.values())
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT_DIR / "s1_audit_scorecard_v2.json").write_text(
        json.dumps(json_safe(score_json), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    report = build_report(
        candidates,
        inventory,
        validation,
        scorecard,
        structural,
        delta,
        plan04_scorecard_json,
        assertions,
    )
    (OUT_DIR / "S1_POSTHOC_AUDIT_VALIDATION_FINAL_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    handoff = f"""# S8-Plan05 Handoff

Status: S8_S1_AUDIT_VALIDATION_FROZEN
Branch: {BRANCH}
Base: {BASE_COMMIT}
Generated: {GENERATED_AT}
Matching contract: {MATCHING_CONTRACT}

Plan04 source artifacts are read-only and remain unchanged. No S1 canonical,
quarantine, provenance, S8 truth, main, or S9 artifact was modified.

Isolation:
- S9 branch read: NO
- S9 predictions read: NO
- S9 metrics read: NO
- main modified: NO
- S1 canonical/quarantine/provenance modified: NO
- S8 frozen truth modified: NO

Matching:
- Candidate rows: {len(candidates)}
- S1-derived raw-label map rows: {len(normalization_map)}
- city40 / registered / 2021 = 65: S1_MISSED, FALSE_NEGATIVE_MISSED_ANOMALY, CRITICAL
- city2 ↔ city25: S1_CAUGHT_BLOCK_RELATION, 19 rows, 2003–2021, coverage 1.0
- city29 ↔ city8: S1_CAUGHT_BLOCK_RELATION, 19 rows, 2003–2021, coverage 1.0
- ctiy22 -> city22: S1_CAUGHT_NORMALIZED_LABEL
- ctiy31 -> city31: S1_CAUGHT_NORMALIZED_LABEL
- city29 trailing-space rows: explicit canonical city29 mapping

Evidence interpretation:
- Externally verified correct decisions: {int((validation.interpretation_level == 'EXTERNALLY_VERIFIED_CORRECT').sum())}
- Externally verified wrong decisions: {int((validation.interpretation_level == 'EXTERNALLY_VERIFIED_WRONG').sum())}
- Internally supported, not externally verified: {int((validation.interpretation_level == 'INTERNALLY_SUPPORTED_NOT_EXTERNALLY_VERIFIED').sum())}
- Verified false negatives: {assertions['verified_false_negatives']}
- Critical misses: {assertions['critical_misses']}
- Confirmed global city permutation: NO

Files:
- HANDOFF.md
- S1_POSTHOC_AUDIT_VALIDATION_FINAL_REPORT.md
- candidate_decision_matching_audit.csv
- raw_label_normalization_map.csv
- s1_false_negative_candidates_v2.csv
- s1_structural_candidate_resolution.csv
- s1_audit_scorecard_v2.csv
- s1_audit_scorecard_v2.json
- s1_decision_validation_results_v2.csv
- plan04_to_plan05_metric_delta.csv

Return: S8 S1 audit validation artifacts finalized and remain isolated from S9.
"""
    (OUT_DIR / "HANDOFF.md").write_text(handoff, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "S8_S1_AUDIT_VALIDATION_FROZEN",
                "inventory_rows": len(inventory),
                "candidate_rows": len(candidates),
                "matching_status_counts": candidates.matching_status.value_counts().sort_index().to_dict(),
                "verified_false_negatives": assertions["verified_false_negatives"],
                "critical_misses": assertions["critical_misses"],
                "true_year_shift_caught": int(
                    (
                        candidates.candidate_type.eq(
                            "SHIFTED_DUPLICATE_TRAJECTORY"
                        )
                        & candidates.matching_status.eq(
                            "S1_CAUGHT_BLOCK_RELATION"
                        )
                    ).sum()
                ),
                "malformed_label_caught": int(
                    (
                        candidates.candidate_raw_city_id.isin(["ctiy22", "ctiy31"])
                        & candidates.matching_status.eq(
                            "S1_CAUGHT_NORMALIZED_LABEL"
                        )
                    ).sum()
                ),
                "output_dir": str(OUT_DIR),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
