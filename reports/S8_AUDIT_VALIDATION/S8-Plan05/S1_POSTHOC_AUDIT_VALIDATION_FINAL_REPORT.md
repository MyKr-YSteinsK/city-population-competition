# S8-Plan05｜Final S1 Audit Validation Freeze

## Result

- Status: S8_S1_AUDIT_VALIDATION_FROZEN
- Branch: codex/s8-plan05-audit-validation-freeze
- Base: Plan04 commit 6aa4c88c98ad5e526466291541c21b71b4057fc8
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

The Plan04 candidate universe contains 1031 rows. Plan05 applies
four matching paths: exact canonical cell, S1-derived raw-label normalization,
pairwise shifted-trajectory relation, and block-level same-city-year coverage.
Each candidate receives one of the required matching statuses:
NO_COMPARABLE_S1_RULE, S1_CAUGHT_BLOCK_RELATION, S1_CAUGHT_EXACT, S1_CAUGHT_NORMALIZED_LABEL, S1_CAUGHT_WARNING, S1_MISSED, UNRESOLVED_MATCH.

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

- Historically defensible S1 decisions: 382
- Externally verified correct decisions: 4
- Externally verified wrong decisions: 0
- Not externally verifiable decisions: 378

The historical/internal count is not presented as universal external proof.
For the 378 internally supported but non-external decisions, the
precise interpretation is: S1 historical action is internally defensible;
external exact-cell validation unavailable. Cautious unresolved evidence is
kept separate in the v2 decision table.

## Plan04 to Plan05 metric delta

| metric | plan04_value | plan05_value | delta | reason |
|---|---:|---:|---:|---|
| verified_false_negative_count | 1 | 1 | 0 | unchanged: matching cleanup does not change the definition-matched official universe or city40 comparison. |
| critical_miss_count | 1 | 1 | 0 | unchanged: city40 registered=65 remains the same verified critical miss. |
| true_year_shift_s1_caught_count | 0 | 2 | 2 | The two employment shifted trajectories now match city25/city29 quarantine blocks by pairwise relation. |
| true_year_shift_uncaught_count | 2 | 0 | -2 | The former false non-match is removed by block-level relation matching. |
| malformed_ctiy_label_s1_caught_count | 0 | 2 | 2 | ctiy22 and ctiy31 now resolve through the S1-derived raw-label map. |
| trailing_space_city29_label_caught_count | 36 | 36 | 0 | city29 trailing-space rows retain their caught status with explicit canonical identity. |
| externally_verified_correct_decisions | 4 | 4 | 0 | unchanged: no new external research and the Plan04 external universe is reused as frozen evidence. |
| externally_verified_decisions | 4 | 4 | 0 | unchanged: matching cleanup changes candidate-to-decision attribution, not exact-cell external evidence. |
| not_externally_verifiable_decisions | 378 | 378 | 0 | unchanged: non-external decisions remain explicitly internal or cautious, never externally proven. |
| candidate_rows | 1031 | 1031 | 0 | unchanged: Plan05 reuses the Plan04 candidate universe and only repairs matching. |
| s1_caught_candidate_rows | 57 | 118 | 61 | The count changes only where canonical label and pairwise block relations were previously missed. |
| verified_missed_anomalies_in_scorecard | 1 | 1 | 0 | unchanged: the only externally verified anomaly remains city40 registered=65. |

## Structural candidate resolution

| candidate_type | dataset | candidate_count | s1_caught_count | s1_missed_count | unresolved_count | material_new_findings |
|---|---|---:|---:|---:|---:|---|
| BLOCK_CONTAMINATION | overall_income | 32 | 32 | 0 | 0 | S1_QUARANTINE_RETAINED |
| CITY_PERMUTATION | ALL | 0 | 0 | 0 | 0 | NO_CONFIRMED_GLOBAL_PERMUTATION |
| INDICATOR_SEMANTICS | age | 19 | 1 | 0 | 1 | POSSIBLE_DEFINITION_OR_SOURCE_CONFLICT |
| INDICATOR_SEMANTICS | employees+sector | 18 | 1 | 0 | 1 | CONFLICT_WITH_EXTERNAL_OR_INTERNAL |
| INDICATOR_SEMANTICS | overall_income+urban_income | 7 | 1 | 0 | 1 | POSSIBLE_SHIFTED_COPY |
| INDICATOR_SEMANTICS | population | 1 | 0 | 0 | 1 | CROSS_FIELD_CONFLICT |
| INDICATOR_SEMANTICS | raw_workbooks | 38 | 1 | 0 | 0 | LABEL_REMAP_CANDIDATE |
| YEAR_SHIFT | employees | 2 | 2 | 0 | 0 | S1_CAUGHT_BLOCK_RELATION |

The city40 finding remains the only externally verified false negative in the
available definition-matched universe. The two TRUE_YEAR_SHIFT results and the
two malformed urban-income labels are not new S1 misses after matching cleanup.

## Final scorecard

| dataset | external cells | S1 decisions | external decisions | external correct | external wrong | internal only | verified anomalies | caught | missed | precision | recall | critical | assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| population | 28 | 10 | 0 | 0 | 0 | 10 | 1 | 0 | 1 | NA | 0.000 | 1 | MIXED |
| density | 0 | 27 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urbanization | 13 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | NA | NA | 0 | ADEQUATE_WITH_GAPS |
| unemployment | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| employees | 13 | 176 | 4 | 4 | 0 | 172 | 0 | 0 | 0 | 1.000 | NA | 0 | STRONG |
| sector | 39 | 121 | 0 | 0 | 0 | 121 | 0 | 0 | 0 | NA | NA | 0 | ADEQUATE_WITH_GAPS |
| wage | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| age | 0 | 7 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_income | 0 | 32 | 0 | 0 | 0 | 32 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_consumption | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urban_income | 0 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| rural_income | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| urban_consumption | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| rural_consumption | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| overall_income+urban_income | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |
| source | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | NA | NA | 0 | INSUFFICIENT_EXTERNAL_EVIDENCE |

## Required conclusions

1. True S1 false negatives: 1 in the
   available definition-matched external universe, with
   1 critical miss.
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
candidate_truth_2023_v2.

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
