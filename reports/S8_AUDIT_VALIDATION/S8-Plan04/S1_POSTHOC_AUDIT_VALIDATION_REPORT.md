# S8-Plan04｜Post-hoc S1 Data Audit Validation

## Result

- Status: `COMPLETE_WITH_MATERIAL_S1_AUDIT_FINDING`
- Branch: `codex/s8-plan04-s1-audit-validation`
- Parent branch: `codex/s8-truth-reconstruction-v1` @ `cbfc39c2471f3390c80e4b6e4642308b7fbc5a7c`
- Frozen truth read: `candidate_truth_2023_v2`
- Frozen evaluator SHA256: `de8ae0feb8b4061610b82a235a673fc2a841d2969f04924d7f69185c5e4db02c`
- Frozen truth modified: `NO`
- `PROJECT_DOC_SYNC_NEEDED_BY_MAIN`

This is a post-hoc audit. It does not rewrite `data/raw`, canonical tables,
quarantine, provenance, S1 reports, S6 artifacts, or frozen S8 truth.

## Scope and isolation

- S9 branch read: `NO`
- S9 predictions read: `NO`
- S9 metrics read: `NO`
- main modified: `NO`
- `test.xls` read: `NO`
- Candidate generation was completed before the S1 inventory was loaded.

## Verification universe

The quantitative universe contains definition-matched observed S1 cells from
the official Jiangsu 2021 yearbook table 19-2, frozen S8 target anchors that
are explicitly marked outside the S1 observed range, and the explicit city40
2020 census/yearbook anchors. Cells without a definition-matched official
comparison remain `NOT_EXTERNALLY_VERIFIABLE` and are excluded from precision
and recall denominators.

- external universe rows: `146`
- externally verified observed cells: `93`
- S1 material decision rows: `382`
- externally verifiable decision rows: `4`

## S1 decisions

- correct / caution decisions: `382`
- verified wrong decisions: `0`
- false-positive decisions: `0`
- unresolved external evidence: `378`

The inventory includes zero-sentinel repair, duplicate-key repair, label
remaps, quarantine, invalidation, derived reconstruction, preserved warnings,
and source-domain boundaries. The correct/caution labels distinguish a sound
historical action from an externally proven exact value.

## False-negative search

- independent anomaly/structure candidates: `1031`
- externally verified missed anomalies: `1`
- critical missed anomalies: `1`
- population temporal/cross-field candidates: `18`
- population S1-caught candidates: `9`
- population S1-missed candidates: `1`

Highest-severity verified misses:

- `population/city40/2021/户籍人口（万人）`: EXTREME, SOURCE_ERROR_LIKELY; large year-over-year change

### city40 registered=65

The generic population temporal rule detected `city40 / registered / 2021`
because the 2020→2021 change is `-75.999%`, classified `EXTREME`, and the
next official comparison is 268.10 万人 in the official Jiangsu 2021 yearbook.
The raw competition value is 65 and the preceding canonical value is 270.
S1 historical status was `NO_FLAG`; final classification is
`FALSE_NEGATIVE_MISSED_ANOMALY` with `miss_severity=CRITICAL`.

This is a design blind spot, not an isolated luck event: `scripts/audit_population.py`
implements schema, key, key-format, coverage, missing-value, value-sanity, and
zero-sentinel checks, but no first-difference, relative-YoY, rolling-MAD, or
resident/registered cross-field plausibility rule.

## Permutation, shift, and semantics results

- city label permutation: no canonical multi-year permutation was promoted to a confirmed remap; raw structural candidates remain evidence rows.
- local label swap: raw malformed labels and duplicate-key candidates were detected; the two S1 urban-income label repairs are retained as `CORRECT_LABEL_REMAP`.
- year shift: `180` raw shifted-trajectory candidates were found; employment shifted duplicates are classified `TRUE_YEAR_SHIFT`, while other low-cardinality matches remain unresolved.
- indicator label mismatch: the raw label scan found `38` malformed label candidates; population resident/registered has `1` semantic check row with a city40 cross-field conflict.
- block contamination: the independent semantic scan identifies the city3 lagged-copy warning and the S1 overall-income quarantine boundary; it does not rewrite either.
- magnitude / decimal anomaly: the city21 overall-consumption `446007 → 44607` repair remains internally supported; no new unverified correction is applied.

## Dataset scorecard

| dataset | externally_verified_cells | s1_material_decisions | externally_verified_decisions | correct_decisions | wrong_decisions | verified_anomaly_count | caught_anomaly_count | missed_anomaly_count | precision | verified_universe_recall | critical_misses | overall_assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| population | 28 | 10 | 0 | 10 | 0 | 1 | 0 | 1 | NA | 0.000 | 1 | MIXED |
| density | 0 | 27 | 0 | 27 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urbanization | 13 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | NA | NA | 0 | ADEQUATE_WITH_GAPS |
| unemployment | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| employees | 13 | 176 | 4 | 176 | 0 | 0 | 0 | 0 | 1.000 | NA | 0 | STRONG |
| sector | 39 | 121 | 0 | 121 | 0 | 0 | 0 | 0 | NA | NA | 0 | ADEQUATE_WITH_GAPS |
| wage | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| age | 0 | 7 | 0 | 7 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_income | 0 | 32 | 0 | 32 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_consumption | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urban_income | 0 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| rural_income | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urban_consumption | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| rural_consumption | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_income+urban_income | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| source | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |

## Required answers

### Q1 — confirmed strengths

S1's zero-sentinel handling, duplicate-key handling, employment remap and
sector-sum reconstruction, shifted-duplicate quarantine, malformed urban-income
label repairs, and explicit quarantine boundaries are supported by the recorded
internal invariants. The official Jiangsu 2021 cross-check also provides a
definition-matched external reference for population, density, urbanization,
employees, and sector fields in the Jiangsu subset.

### Q2 — false positives

No verified false-positive repair, invalidation, or quarantine was found in the
available exact-cell external universe. This is not a claim that every S1
decision has been externally proven; most historical quarantine cells remain
`NOT_EXTERNALLY_VERIFIABLE`.

### Q3 — false negatives

The material verified miss is city40 registered 2021 = 65. Other independent
candidates without matching official cells remain `UNRESOLVED`, not false
positives or false negatives.

### Q4 — structural findings

No canonical global city permutation was proven. Raw employment contains
multi-year shifted duplicate candidates, and malformed raw labels are visible.
The city3 income lag remains a semantic warning rather than a proven column
swap. The known city21 magnitude repair and S1 block contamination boundaries
are preserved as historical decisions.

### Q5 — why city40 was missed

S1 had no temporal plausibility or cross-field relationship detector in the
population audit. The omission is therefore a reusable audit-contract gap.

### Q6 — depth by table

Employment, sector employment, density, and living-standard tables have the
deepest local repair/quarantine/provenance logic. Population has strong
schema/coverage and sentinel checks but weak anomaly detection. Wage and age
have isolated invalidation rules but limited external verification. Income has
important contamination and lag warnings, while external historical coverage
is thin.

### Q7 — minimum-cost methodology changes

Add per-city first-difference and robust-MAD scans, resident/registered
cross-field checks, duplicate-trajectory and lagged-copy detection, raw label
format/duplicate-key scans, and an external verification sample with explicit
definition/year/revision fields. These recommendations follow the observed
city40 miss and the unresolved raw shifted candidates; they are audit rules,
not S9 features.

### Q8 — overall credibility

S1 structural integrity is strong. Repair/quarantine logic is generally
adequate where explicit invariants exist. Semantic alignment and external
historical truth coverage are mixed. False-negative control is weak for
single-point temporal anomalies, with city40 as a critical verified miss.

## Files

- `HANDOFF.md`
- `s1_audit_decision_inventory.csv`
- `external_verification_universe.csv`
- `population_posthoc_anomaly_audit.csv`
- `cross_table_city_identity_validation.csv`
- `label_permutation_candidates.csv`
- `cross_table_year_alignment_validation.csv`
- `year_shift_candidates.csv`
- `indicator_semantics_validation.csv`
- `s1_decision_validation_results.csv`
- `s1_false_negative_candidates.csv`
- `s1_audit_scorecard.csv`
- `s1_audit_scorecard.json`
- `source_ledger_extension.csv`
- `run_s8_plan04.py` (reproducibility script in `scripts/`)

## Remaining uncertainty

The verification universe is deliberately partial. It cannot validate every
historical cell or prove the true values of quarantined rows. Revision,
census-rebasing, and definition changes remain explicit ambiguity classes.
Main must decide separately whether the material city40 miss warrants a future
S1 remediation plan after the S9 freeze/reveal boundary.
