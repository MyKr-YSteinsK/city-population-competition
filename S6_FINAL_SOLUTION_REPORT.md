# S6 Final Solution Report

**Status:** `FINAL_SOLUTION_FROZEN_FOR_MAIN_CLOSEOUT`

## Final frozen solution

The local S6 solution is frozen for Main closeout as a 2023 resident-population
submission for 40 cities:

- target: `resident`
- framing: `F4_LOG_GROWTH_FROM_LATEST`
- scenario: `R1_REFERENCE_SNAPSHOT_LAG`
- model configuration: `FRM_C3_FLEXIBLE_REGULARIZED`
- seed: `20260919`
- training rows: `573`
- maximum label year: `2021`
- prediction year: `2023`

No new fitting or candidate search was performed in Plan04. The final file is
the exact two-decimal, standard-rounded rendering of the previously selected
resident F4-R1 predictions. It has 40 rows, columns `city_id,year,pred`,
`city1` through `city40` in order, and year `2023` throughout.

## Data and contract boundary

Plan02 established the clean-raw foundation from the legal official PDF and
seven legal XLSX sources, with canonicalization and quarantine checks. The
final foundation uses the project T-2 information boundary, 144 features, and
the clean replay/training contract. `test.xls`, B-data, external sources, and
2022-2023 target truth were not read, copied, hashed, or used.

## Why F4-R1 was frozen

B3 remains the resident reference baseline (pooled historical MSE
`872.452651515143`). S4/S5 attempts with Ridge corrections, urbanization and
other external signals, low-capacity boosted trees, stronger model families,
hierarchical fitting, and city-local fitting did not establish a stable,
reproducible resident replacement. F4 was the strongest resident framing in
the completed evidence, with resident pooled MSE approximately `739.600665`
for R1 versus approximately `802.445747` for R2. R1 therefore has the better
historical scenario behavior.

Within the current S6 resident R1 configuration selection, C3 had inner
validation MSE `673.646095001205`, versus `688.5431302040054` for C1 and
`681.7136425379174` for C2. The selected C3 replay uses 573 training rows and
does not require any post-hoc clipping or manual prediction adjustment.

## Routes not adopted

The final solution does not use the registered target, registered F4
alternatives, Ridge/urbanization residual corrections, low-capacity GBDT,
stronger Plan06A-D model families, Plan06G partial pooling, Plan06H city-local
models, blending, routing, or gating. The registered B3/RLT3 exploratory
submissions also exposed an abnormal historical city40 drop and negative
extrapolated predictions; this was not repaired by clipping and is not part of
the final resident submission.

## Remaining boundaries and risks

- No leaderboard score was available before freeze.
- The official “总人口” semantic mapping was not externally verified; resident
  was intentionally frozen as the project decision.
- Full historical F4 replay was deferred after the fast first submission path.
- The final artifact is a local candidate; completion status remains with Main.
  The checkpoint remains unchanged and the next stage is not declared.

## Output artifacts

- Final submission: `reports/S6_HANDOFFS/S6-Plan04/submission_final.csv`
- High-precision provenance source: `reports/S6_HANDOFFS/S6-Plan04/submission_resident_f4_r1_high_precision.csv`
- Machine-readable manifest: `reports/S6_HANDOFFS/S6-Plan04/final_solution_manifest.json`
- Main handoff: `reports/S6_HANDOFFS/S6-Plan04/HANDOFF.md`

The high-precision provenance file is byte-recovered from Git commit
`e7040aff56eac45d3cbaadc8045ba446d9b9f02d`. This is necessary because the separately authorized Plan03
formatting step changed the original working copy to two decimals; it does not
alter the Plan03 output or introduce a new model run. The maximum absolute
rounding delta in the final rendering is `0.004750777693`.
