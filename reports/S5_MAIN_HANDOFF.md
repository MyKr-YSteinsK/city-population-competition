# S5-Plan07 Main Handoff — Final Wave Consolidation

## Review status

- `ready_for_main_stage_review: true`
- `S5_COMPLETE: false`
- `S6_ENTERED: false`
- Scope: evidence consolidation only; no new model, tuning, routing, ensemble, 2023 prediction, or submission.
- Validation and feature contracts remain frozen.

## Strongest valid resident candidate

`F4_LOG_GROWTH_FROM_LATEST` is the strongest resident candidate in the unified panel under review:

- R1 pooled MSE `739.600665`, relative improvement `15.227%`, origin wins `5/11`, latest-2021 MSE `2956.289001`.
- R2 pooled MSE `802.445747`, relative improvement `8.024%`, origin wins `7/11`, latest-2021 MSE `3595.001285`.
- F3 remains a valid framing signal but is weaker than F4 on resident pooled MSE in both scenarios: R1 `788.841312`, R2 `818.447737`.
- CatBoost F2 is the strongest valid Plan06A-D residual control on the resident primary, but remains above B3: R1 `973.309135`.

This is a Main-review recommendation, not a final winner or deployment promotion. B3 remains the official frozen competition reference until Main completes its stage review.

## Registered sensitivity conclusion

The small legacy RLT3 signal is real but narrow: registered R1 pooled MSE `223.690909` versus frozen B3 `225.421212`, with 8/11 origin wins in the Plan02 artifact. Plan06E does not reproduce that registered gain: F3/F4 are above B3 in both scenarios. Therefore the resident-primary contract supports reviewing F4 as a resident candidate, while registered remains a sensitivity constraint and does not support a global replacement claim.

## Unsupported or incomplete families

- Plan06A LightGBM and Plan06B CatBoost: complete but competitive/unstable; residual controls do not replace B3 on resident primary.
- Plan06C XGBoost and Plan06D ExtraTrees/Ridge: unsupported negative controls.
- Plan06F EBM: recovery-only; execution/selection facts preserved, core performance evidence unavailable; not compared.
- Plan06G hierarchical/partial pooling: `coverage_incomplete` (`2760/3520` combined OOF; F2 `1000/1760`; 19 outer failures); low registered/R1 F2 MSE is inadmissible.
- Plan06H city-local: all candidates `city_local_unsupported`; all-144 local Ridge shows n << p collapse; no routing inference.

## Unresolved risks

- F4's resident improvement is a framing signal, not yet a deployment decision; Main must review tail errors, city concentration, bias, and R1/R2 behavior together.
- RLT3 auxiliary quantiles and city-tail fields are unavailable because Plan02 did not persist row-level RLT3 OOF; no rerun was used to fill them.
- 06F and 06G cannot support performance claims beyond their explicit recovery/coverage facts.

## Evidence package

- `reports/S5_FINAL_CANDIDATE_COMPARISON.csv`
- `reports/S5_FINAL_CANDIDATE_COMPARISON.json`
- `reports/S5_EXPERIMENT_REGISTRY.csv`
- `reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md`
- `S5_EVIDENCE_DRIVEN_ITERATION_FINAL_REPORT.md`
