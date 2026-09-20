# S6-Plan02 Handoff｜Clean Raw Rebuild 与双目标 2023 特征基座复现

## Result

- Status：`READY_FOR_S6_HISTORICAL_REPRODUCTION_CONTROLS`
- Execution baseline：`124bf189c50ba136fa6c7bd74283dd9a8f90a026`
- Contract：`S6_FINAL_F4_RESIDENT_PRIMARY_2023_V1` / `f5deff350d96d2b688abf19ea63ad2edd0588750d20d6f9ecf52fd4d70561e2b`
- This is a disposable-worktree clean-room rebuild; no model fit, final prediction, or submission was run.

## Raw and S1

- Official PDF + 7 legal XLSX：`PASS`; unexpected legal raw workbooks：`0`.
- `test.xls` content was not opened, parsed, copied, inspected, or hashed; only existence and size metadata were observed.
- S1 raw-to-canonical rebuild：`PASS`; equivalence mismatches：`0`.

## Plan04 and Plan04B

- Plan04 rebuilt formal matrix：`1760` rows, `144` features; equivalence：`PASS`.
- Plan04B rebuilt replay：`2320` rows; counts：`{"registered|R1_REFERENCE_SNAPSHOT_LAG": 587, "registered|R2_CONSERVATIVE_PUBLICATION_STRESS": 587, "resident|R1_REFERENCE_SNAPSHOT_LAG": 573, "resident|R2_CONSERVATIVE_PUBLICATION_STRESS": 573}`.
- Outer geometry：`44`; inner geometry：`484`; boundary violations：`{'source_cap': 0, 'quarantine': 0, 'benchmark': 0, 'population_history': 0, 'b3_history': 0}`.
- B3 compatibility：`PASS`.
- Legacy Plan04 branch/SHA identity gate was bypassed only in memory by the S6 wrapper because the disposable worktree is detached; feature definitions, source caps, T-2 rule, B3 implementation, and replay semantics were not changed.

## Current 2023 foundation

- Rows：`160`; group counts：`{"registered|R1_REFERENCE_SNAPSHOT_LAG": 40, "registered|R2_CONSERVATIVE_PUBLICATION_STRESS": 40, "resident|R1_REFERENCE_SNAPSHOT_LAG": 40, "resident|R2_CONSERVATIVE_PUBLICATION_STRESS": 40}`.
- Boundary audit rows：`2240`; source-cap/quarantine/benchmark violations：`0/0/0`.
- resident role: `primary`; registered role: `alternative_required`; no target label after 2021 is present.

## Independence and scope

- Historical Plan04/Plan04B matrices were unavailable during generation and were read from Git objects only after independent generation.
- The dependency audit concludes raw + frozen contracts/code are sufficient to regenerate the S6 foundation.
- No CatBoost/LightGBM/XGBoost/ExtraTrees/Ridge/HGB/neural fit, config selection, tuning, prediction, submission, checkpoint edit, or S7 entry.

## Evidence paths

- `HANDOFF.md`
- `clean_rebuild_summary.json`
- `raw_integrity_check.csv`
- `input_dependency_audit.csv`
- `s1_rebuild_equivalence.csv`
- `plan04_rebuild_equivalence.csv`
- `plan04b_rebuild_equivalence.csv`
- `rebuilt_training_replay_matrix.csv`
- `current_2023_feature_matrix.csv`
- `current_2023_feature_boundary_audit.csv`
- `dual_target_role_mapping.csv`
- `rebuild_artifact_equivalence.csv`
- `environment_snapshot.json`

## Recommendation

Accept this package as `READY_FOR_S6_HISTORICAL_REPRODUCTION_CONTROLS`. The next boundary is S6-Plan03: reproduce frozen historical B3/F4 controls from this newly rebuilt foundation before any final 2023 fit.
