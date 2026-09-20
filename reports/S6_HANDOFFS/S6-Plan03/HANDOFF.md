# TASK_RESULT — S6-Plan03 Fast First Submission

## Result

- Status: `BLOCKED_SUBMISSION_SCHEMA`
- Strategy: `FAST_FIRST_SUBMISSION`
- Generation source: `codex/s6-plan03-fast-first` at `2c64f5ec6eb8ed1ad6c571ecbe01f1dd2947d288`
- Platform upload: not performed; manual upload remains with Main/user
- S6 completion: not declared; S7 not entered

## Main recommendation

The resident-primary R1 candidate remains the provisional first-upload choice because it is the stronger frozen historical information scenario and passed its own schema checks. The local unit remains `万人`; official resident/registered mapping and unit confirmation remain unresolved and must not be silently treated as proved. Main/user should review the packet status before any manual upload.

## Final-fit evidence

| Candidate | Selected config | Eligible inner folds | Training rows | Max label year | Inner scores |
|---|---|---:|---:|---:|---|
| resident F4 R1_REFERENCE_SNAPSHOT_LAG | `FRM_C3_FLEXIBLE_REGULARIZED` | 13 | 573 | 2021 | `{"FRM_C1_CONSERVATIVE": 688.5431302040054, "FRM_C2_MEDIUM": 681.7136425379174, "FRM_C3_FLEXIBLE_REGULARIZED": 673.646095001205}` |
| resident F4 R2_CONSERVATIVE_PUBLICATION_STRESS | `FRM_C1_CONSERVATIVE` | 13 | 573 | 2021 | `{"FRM_C1_CONSERVATIVE": 725.8420888837446, "FRM_C2_MEDIUM": 738.1610465092817, "FRM_C3_FLEXIBLE_REGULARIZED": 729.7669337824736}` |

## Candidate files

- `reports/S6_HANDOFFS/S6-Plan03/submission_resident_f4_r1.csv`
- `reports/S6_HANDOFFS/S6-Plan03/submission_resident_f4_r2.csv`
- `reports/S6_HANDOFFS/S6-Plan03/submission_registered_rlt3.csv`
- `reports/S6_HANDOFFS/S6-Plan03/submission_registered_b3.csv`
- `reports/S6_HANDOFFS/S6-Plan03/submission_resident_b3.csv`

The five files were generated, but the packet is blocked because one or more candidates fail the positive-prediction schema. No clipping or manual city repair was applied.

## Schema blocker evidence

- `registered_b3` — `city40=-107.0` fails the positive-prediction check; this is the unmodified OLS extrapolation from registered `(2019,271),(2020,270),(2021,65)`.
- `registered_rlt3` — `city40=-141.0` fails the positive-prediction check; this is the unmodified median-slope extrapolation from the same three legal points.

## Controls and boundaries

- Plan02 clean replay/current foundation: PASS; 2,320 replay rows, 160 current rows, 144 numeric features, 2,240 boundary rows.
- F4 uses only the frozen Plan01 semantics and C1/C2/C3; no new model family, tuning sweep, ensemble, routing, clipping, or manual city edit.
- Registered RLT3 and B3 are target-history-only controls; RLT3 R1/R2 equivalence: PASS.
- No `test.xls` content/hash, B-data, external data, or target truth after 2021 was used.
- `CURRENT_LEARNING_CHECKPOINT.md` remained unchanged.

## Deferred reproduction debt

Full historical F4 OOF replay, exact all-origin selection reproduction, full RLT3 historical regeneration, and deeper fresh end-to-end release proof remain explicitly deferred until platform feedback. This is a fast-first handoff, not a claim that S6 is complete.

## Required next action

If Main accepts the resident-primary path after reviewing the registered sensitivity anomaly, manually upload `submission_resident_f4_r1.csv` and return the platform score. Do not automatically upload from this repository.
