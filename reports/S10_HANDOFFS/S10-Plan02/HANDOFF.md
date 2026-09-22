# HANDOFF — S10-Plan02 11-origin same-contract selector audit

## Status

`COMPLETE` for the bounded S10-Plan02 scope. The primary result is a
canonical, no-manual-repair historical replay. The living-standard parser
repair diagnostic was intentionally deferred as `NOT_RUN_LOW_PRIORITY`; it is
not needed for the selector mechanism judgment.

## Repository identity

- Repository: `MyKr-YSteinsK/city-population-competition`
- Base SHA: `8cb00034791482d34cc3e5d3bc8df7fdcdf9d174`
- Base branch: `codex/s10-plan01-external-script-forensics`
- Working branch: `codex/s10-plan02-same-contract-selector-audit`
- Final commit / push: recorded in the delivery response to avoid a
  self-referential handoff hash.
- `main`, S6 frozen artifacts, S8 truth, S9, and `CURRENT_LEARNING_CHECKPOINT.md`
  were not modified.

## Contract and evidence

- Primary target: resident; registered sensitivity.
- Origins: 2011–2021, 11 expanding origins, 40 cities.
- Horizon: 2 years; target history cutoff `T-2`.
- Primary metric: pooled city-year MSE; macro-origin MSE and per-origin rows
  retained.
- Primary source semantics: canonical/provenance-safe; raw/original semantics
  is diagnostic only.
- Primary repair variant: `NO_MANUAL_TARGET_REPAIR`.
- Nested selector scoring domain: 2013–2021; 2011–2012 are warm-up only.

## Main finding

The best fixed external target-only expert is `local_slope3` at resident
11-origin MSE `764.101263`, still above F4 (`739.600665` R1;
`802.445747` R2). The invalid non-nested apparent full-pool selector scores
`557.174683` R1 and `721.948486` R2 on all 11 origins. On the matched
2013–2021 slice the valid nested selectors score `860.349469` for target-only
pool A in both scenarios; adding density or pooled ML does not establish a
gain.

Mechanism judgment: `SAME_SAMPLE_SELECTOR_OPTIMISM_EXPLAINS_APPARENT_GAIN`.
S10 classification: `METHOD_INTERESTING_BUT_RESULT_NOT_COMPARABLE`.
`S9_ELIGIBLE_METHOD_CANDIDATE = NO`.

See `reports/selector_mechanism_report.md` for the full comparison and
interpretation.
