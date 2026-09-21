# S8 Truth Reconstruction Report

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE

This is the post-hoc target-forensics track requested by S8-Plan01. It does
not train, select, score, or compare models. The plan is task input; it is
kept separate from the user's short request to continue this line.

## City identity

All 40 competition IDs are mapped with resident trajectory anchors plus
registered population, urbanization, density, and province evidence.

Mapping counts: CONFIRMED=19,
HIGH_CONFIDENCE=20,
PROVISIONAL=1,
UNRESOLVED=0.

city3 has a retained Shenzhen registered-field mismatch. city40 maps to
Zhenjiang, but its 2021 registered value is 65 versus the 2020 anchor 269.25.
No competition raw value is corrected.

## Year alignment

The selected hypothesis is competition year = real statistical year + 1.
Therefore the candidate target year 2023 is reconstructed from real statistical
year 2022. Resident, registered, urbanization, and density evidence supports
the 2021 to 2020 comparison. Offsets -2, -1, 0, and +1 are all recorded.

## Target semantics

Selected semantics: RESIDENT_POPULATION.
Confidence: HIGH_WITH_EXPLICIT_EXCEPTIONS.
Resident matches: 31/40.
Registered matches: 32/40.
Urbanization matches: 40/40.
Density matches: 23/40.

Registered is retained as an independent comparator. No model performance,
MSE, ranking, or S9 artifact was used.

## Candidate truth and uncertainty

candidate_truth_2023_raw.csv preserves 2022 raw values in 万人. The v1
candidate uses ROUND_HALF_UP_TO_INTEGER_WAN, and the evaluator file is
derived from the full provenance table. Primary raw rows: 19;
reproduced-source raw rows: 21.

The freeze is provisional because some 2022 values are Tier-3 reproductions
and because the city40 identity/registered conflict remains explicit.

## Isolation audit

- S9 branch read: NO
- S9 predictions or scoreboard read: NO
- model evaluation performed: NO
- test.xls accessed: NO
- Main worktree modified: NO

## Reproduction

Run python scripts/build_s8_truth_v1.py --input-root <data-root> --repo-root <worktree-root>.
Outputs are under reports/S8_TRUTH/truth_v1.

Artifact commit at generation time: not yet committed.
