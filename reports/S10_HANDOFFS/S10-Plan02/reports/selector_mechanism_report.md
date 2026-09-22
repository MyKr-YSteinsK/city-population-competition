# S10-Plan02 selector mechanism report

## Scope and contract

This report covers only the common strict historical replay: resident as the
primary target, registered as sensitivity, origins 2011–2021, 40 cities,
horizon two years, target-history cutoff `T-2`, and pooled city-year MSE as
the primary aggregate. The valid nested selector is scored only on 2013–2021;
2011–2012 are selector warm-up evidence and are not scored as if they had
prior history.

The primary replay uses `CANONICAL_PROVENANCE_SAFE` source semantics and
`NO_MANUAL_TARGET_REPAIR`. `ORIGINAL_SOURCE_SEMANTICS` is retained as an
explicit diagnostic variant. No S8 truth, 2022/2023 target label, or 2023
row-level error was read.

## Fixed candidates

Resident primary, canonical semantics, no repair:

| slice | control / candidate | R1 MSE | R2 MSE | coverage |
|---|---|---:|---:|---:|
| 2011–2021 | B3 | 872.452652 | 872.452652 | 440/440 |
| 2011–2021 | F4 | 739.600665 | 802.445747 | 440/440 |
| 2011–2021 | best fixed external expert: `local_slope3` | 764.101263 | 764.101263 | 440/440 |
| 2013–2021 matched | B3 | 964.292361 | 964.292361 | 360/360 |
| 2013–2021 matched | F4 | 789.911392 | 878.042231 | 360/360 |
| 2013–2021 matched | `local_slope3` | 819.683333 | 819.683333 | 360/360 |

`local_slope3` is target-only, so its R1/R2 predictions and metrics are
numerically identical. It improves on B3 on the full 11-origin slice but does
not beat F4 on either source scenario. The pooled sklearn family is not
competitive in this replay; its best resident canonical 11-origin MSE is
`model_ridge` R2 at `2363.776656`.

## Selector audit

The apparent selector uses all 2011–2021 target errors to select a city expert
and then scores those same rows. It is therefore explicitly invalid for an
unbiased historical estimate. On the canonical full legal pool it gives:

| target | scenario | apparent 11-origin MSE | matched-domain apparent MSE |
|---|---|---:|---:|
| resident | R1 | 557.174683 | 590.786785 |
| resident | R2 | 721.948486 | 760.226863 |
| registered | R1 | 217.196453 | 261.557774 |
| registered | R2 | 216.632373 | 260.744849 |

The matched-domain values are the only valid basis for a direct apparent vs
nested comparison. The valid expanding prior-only selector is materially
worse:

| target | scenario | A target-only | B + density | C + pooled ML | D full legal pool |
|---|---|---:|---:|---:|---:|
| resident R1 | 2013–2021 MSE | 860.349469 | 1143.492824 | 918.425507 | 1202.715484 |
| resident R2 | 2013–2021 MSE | 860.349469 | 860.349469 | 969.094445 | 969.094445 |
| registered R1 | 2013–2021 MSE | 271.444460 | 271.444460 | 301.431826 | 301.431826 |
| registered R2 | 2013–2021 MSE | 271.444460 | 271.444460 | 283.067305 | 283.067305 |

All nested rows have 360/360 coverage. For resident, the best fixed matched
expert is `local_slope3` at `819.683333`, so even the best nested A selector
is worse by `40.666136` MSE points. The full legal pool does not help: density
and pooled ML increase nested error in the primary target.

The oracle upper bound, which chooses the expert from the current true error,
is undeployable. Its resident matched MSE is `239.849131` (R1) and
`398.748088` (R2). The large oracle-to-nested gaps show that the pool has
diversity, but the prior-only selector does not reliably identify the current
winner.

## Density and legality

All 23,520 source-cap audit rows pass the `used_year <= source cap` check.
The R1 density candidates score 407/440 resident rows (92.5%) and have pooled
MSE `116653.747519` (`density_hold`) and `116462.459656`
(`density_plus_slope5`). R2 has zero legal T-1 density rows because its cap is
T-2; those candidates are unavailable, not carried forward or imputed.

The 2017/2018 density blow-up is real candidate behavior: the largest error is
city8 in 2017 (`4020.982206` absolute error), followed by city26 in 2018
(`3414.034388`). The exact source lookup and cap evidence is retained in the
audit CSV. No unit, area-ratio, regime, or density formula repair was made.

## Repair and source-semantics sensitivity

The registered city40 2021 repair was isolated to a diagnostic variant. All
4,400 resident model comparison rows (`5 models × 2 scenarios × 11 origins ×
40 cities`) have `changed=0`: that 2021 value is outside every historical
model feature window because each origin uses `T-2`. This is a historical
replay result only and is not evidence about a 2023 prediction.

Target-only methods are identical between canonical and original source
semantics. The diagnostic raw-source semantics changes pooled model values,
but does not produce a resident fixed expert that beats F4; it is not promoted
to the primary contract. The original living-standard parser produced zero
income frames in Plan01. The bounded parser-repair ablation is recorded as
`NOT_RUN_LOW_PRIORITY`; it was not allowed to contaminate the primary pool.

## Mechanism judgment

Primary classification:

`SAME_SAMPLE_SELECTOR_OPTIMISM_EXPLAINS_APPARENT_GAIN`

Secondary qualifiers:

- `COVARIATE_AVAILABILITY_DEPENDENT_NOT_SAME_CONTRACT`
- `SOURCE_OR_TARGET_MUTATION_DEPENDENT` is not supported by the historical
  repair sensitivity; the tested repair has no effect in the T-2 replay.

The result does not prove the external 675 claim used exactly this selector,
but it demonstrates that same-sample city routing can create a much lower
apparent score than the temporally valid selector. No S9 eligibility is
recommended.
