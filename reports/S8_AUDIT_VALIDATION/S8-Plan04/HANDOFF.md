# S8-Plan04 Handoff

Status: `COMPLETE_WITH_MATERIAL_S1_AUDIT_FINDING`
Branch: `codex/s8-plan04-s1-audit-validation`
Parent branch: `codex/s8-truth-reconstruction-v1`
Parent HEAD: `cbfc39c2471f3390c80e4b6e4642308b7fbc5a7c`
Generated: `2026-09-22`

Frozen truth: `candidate_truth_2023_v2`
Frozen evaluator SHA256: `de8ae0feb8b4061610b82a235a673fc2a841d2969f04924d7f69185c5e4db02c`
Frozen truth modified: `NO`

Isolation:
- S9 branch read: `NO`
- S9 predictions read: `NO`
- S9 metrics read: `NO`
- main modified: `NO`
- test.xls accessed: `NO`

Validation:
- Decision inventory: `382` rows
- Independent candidate rows: `1031`
- External universe rows: `146`
- Definition-matched observed cells: `93`
- Verified critical false negatives: `1`
- v1/v2 frozen truth remains unchanged

Material finding:

`city40 / population / registered / 2021 = 65` was detected by the generic
post-hoc temporal rule as an EXTREME, CRITICAL false negative. S1 population
audit had no temporal plausibility or resident/registered cross-field rule.

The result remains isolated. No canonical, quarantine, provenance, S6, S8, or
S9 artifact was corrected or regenerated.

Project documentation synchronization is intentionally delegated to Main:
`PROJECT_DOC_SYNC_NEEDED_BY_MAIN`.

Files:

- `S1_POSTHOC_AUDIT_VALIDATION_REPORT.md`
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
