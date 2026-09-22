# S10-Plan01 Script Contract vs Project Contract

## Direct conclusion

`DIRECTLY_COMPARABLE_675_TO_F4 = NO`.

The external script and the frozen project contract share horizon 2 and the T-2 target-history direction, but they do not share the historical domain, primary aggregate, stability guardrails, source-cap contract, source canonicalization, target mutation rule, or final selector semantics. The external script also does not print MSE. A number called `675` cannot be treated as the project F4 historical pooled MSE.

The project evidence used here is the current S2/S3 frozen validation contract plus the S5 comparison panel. The resident F4 reference uses 11 common strict origins × 40 cities = 440 rows per target/scenario and reports pooled city-year MSE; its S5 resident R1 pooled MSE is 739.6006650728775. That value is cited only as contract evidence, not as a new evaluation or promotion.

## Dimension comparison

| Dimension | External Script | Project Frozen Contract | Finding |
|---|---|---|---|
| Primary target | Resident and registered are run/output separately | Resident primary; registered mandatory sensitivity | Different role semantics |
| Horizon | 2 | 2 | Shared |
| Target cutoff | T-2; final history through 2021 | T-2 | Shared direction |
| Historical origins | 2017–2021, 5 origins | 2011–2021, 11 common strict origins | Not comparable |
| Primary aggregate | Prints MAE/MedAE/RMSE; no MSE | Pooled city-year MSE | Not comparable |
| Stability guardrail | None equivalent | Macro-origin MSE and per-origin evidence | Missing from script |
| Covariate timing | Exogenous as-of T-1, stale carry-forward possible | Explicit R1/R2 source-specific caps | Not established equivalent |
| Preprocessing | Train+test city dummy union; train-only imputer/scaler | Frozen fold/replay feature contract | Different contract |
| Final selector | Same 2017–2021 rows select per-city method by MAE and are re-scored in the apparent diagnostic | Candidate-specific frozen rules | Same-sample selector is not an unbiased score |
| Source handling | Local reshape/groupby mean and positional income-sheet parser | S1 canonical/provenance/quarantine | Different semantics |
| Target mutation | city40/2021 registered 65.0 → 270.5 when condition triggers | No equivalent project manual repair | Different target contract |
| `test.xls` | No static reference; absent from disposable runtime | Quarantined and not read | Boundary compatible only |

The machine-readable version is `script_vs_project_contract.csv`. The exact source path/hash, runtime glob matches, and raw hash comparison are in `../runtime/raw_input_manifest.csv`.

## Selector diagnostic boundary

The harness computed `NON_NESTED_APPARENT_SELECTOR` only as a bounded diagnostic. It is explicitly labeled `INVALID_FOR_UNBIASED_HISTORICAL_MODEL_COMPARISON` in `selector_optimism_diagnostic.csv`; it is not a project score and was not used to promote any model or alter S6/S9 artifacts.
