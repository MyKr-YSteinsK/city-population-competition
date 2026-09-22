# TASK_RESULT — S10-Plan02

## Status

- Status: `COMPLETE`
- Branch: `codex/s10-plan02-same-contract-selector-audit`
- Base SHA: `8cb00034791482d34cc3e5d3bc8df7fdcdf9d174`
- Final commit: recorded in the final delivery response after commit/push.

## Frozen contract

- Primary target: `resident`; sensitivity: `registered`.
- Common strict origins: `2011–2021` (11 origins), 40 cities.
- Horizon: 2 years; target-history cutoff: `T-2`.
- Primary metric: pooled city-year MSE; macro-origin and per-origin metrics
  persisted.
- R1/R2 source-specific caps were applied before as-of lookup. Target-only
  candidates are numerically identical across R1/R2.

## Key fixed-method results

- Best fixed resident external expert: `local_slope3`.
- 11-origin pooled MSE: `764.101263` in both R1 and R2, 440/440 rows.
- Matched 2013–2021 MSE: `819.683333` in both R1 and R2, 360/360 rows.
- B3: `872.452652` full; `964.292361` matched.
- F4: `739.600665` R1 / `802.445747` R2 full; `789.911392` R1 /
  `878.042231` R2 matched.

## Selector result

- Non-nested apparent full-pool resident MSE: `557.174683` R1 and
  `721.948486` R2; label: `INVALID_FOR_UNBIASED_HISTORICAL_COMPARISON`.
- Valid nested target-only selector A: `860.349469` on 2013–2021 in both
  scenarios, 360/360 rows.
- Valid nested full pool D: `1202.715484` R1 and `969.094445` R2, 360/360.
- Nested A resident selected-method distribution per scenario: `local_last`
  80, `local_slope3` 120, `local_slope5` 38, `local_mean_slope5` 43,
  `local_linear3` 43, `local_linear5` 36.
- Nested A resident origin wins/losses: 3/6 vs B3 and 4/5 vs F4 in both
  scenarios; switching/stability by city-scenario is persisted in
  `metrics/selector_method_switching.csv`.
- Nested selector rows contain prior-origin histories, fallback reasons,
  current prediction/error, and temporal leakage status `PASS`.
- The oracle diagnostic is explicitly `UNDEPLOYABLE_ORACLE_UPPER_BOUND`.

## Ablation

Resident nested matched MSE:

- A target-only: R1 `860.349469`; R2 `860.349469`.
- B plus density: R1 `1143.492824`; R2 `860.349469`.
- C plus pooled ML: R1 `918.425507`; R2 `969.094445`.
- D full legal pool: R1 `1202.715484`; R2 `969.094445`.

All four pools were frozen before scoring. No hyperparameter sweep was run.

## Leakage / legality

- Source-cap audit: PASS for all 23,520 audit rows; no used source year exceeds
  its scenario cap.
- Manual target repair sensitivity: 4,400 resident model rows; all prediction
  deltas are zero under the historical T-2 window.
- Selector temporal leakage audit: PASS for nested selectors; current origin
  targets are not used for selection or fallback.
- Canonical vs original source semantics: both are persisted separately;
  original employment duplicate averaging and empty income-parser behavior are
  diagnostic-only.
- Density failure: investigated and retained; R1 407/440 coverage with
  approximately 116k MSE, R2 no legal T-1 density rows.
- Parser repaired ablation: `NOT_RUN_LOW_PRIORITY`; primary result does not
  include income features.

## Mechanism verdict

- `SAME_SAMPLE_SELECTOR_OPTIMISM_EXPLAINS_APPARENT_GAIN`
- `COVARIATE_AVAILABILITY_DEPENDENT_NOT_SAME_CONTRACT`

## S10 classification update

`METHOD_INTERESTING_BUT_RESULT_NOT_COMPARABLE`

## S9 eligibility

`S9_ELIGIBLE_METHOD_CANDIDATE = NO` — the fixed external expert does not beat
F4 on the matched primary slice, and the prior-only selector does not retain
the non-nested apparent gain.

## Remaining uncertainty

- The exact provenance of the historical external `675` remains unresolved;
  this Plan explains the selector optimism mechanism but does not claim exact
  reproduction of that number.
- The parser-repaired income diagnostic remains deferred and is not part of the
  primary conclusion.

## Recommendation

Return the evidence to S10/Main for review. Do not execute Plan03
automatically.

## Verification actually passed

- `python -m py_compile scripts/s10_plan02_same_contract_selector_audit.py`
- Plan02 runner completed with fixed OOF, source-cap audit, selector OOF,
  selector history, sensitivity, and comparison artifacts.
- Artifact-level checks passed: every fixed method group has 440 rows;
  every nested selector has 360 rows; canonical R1/R2 target-only identity
  mismatches `0`; training cutoff violations `0`; source-cap violations `0`;
  nested temporal-leakage rows not PASS `0`; forbidden origin-label rows
  `0`.
