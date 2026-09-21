# S7-Supplemental-Plan01 Handoff — 人口跳变事件与可预测前兆调查

## Result

- Status: `COMPLETE_RETURN_TO_MAIN`
- Decision: `H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE`
- Decision note: `PRECURSOR_GATE_B_OR_C_NOT_PASSED`
- Worker branch: `codex/s7-supplemental-jump-plan01`
- Base SHA: `bfe265d3d57f43938e967890567a4162d7b3c896`
- Worker commit: the exact delivery tip is returned in the TASK_RESULT after commit; this handoff was generated from `bfe265d3d57f43938e967890567a4162d7b3c896`.
- Main was not modified; the frozen S6 final artifacts and checkpoint were not modified.

This was a read-only re-analysis. No classifier, magnitude model, hurdle,
mixture, blend, routing, 2023 prediction, submission, threshold tuning,
feature sweep, or model fitting was performed.

## Direct answers to the required questions

1. **Was q90 strictly defined by the outer-training universe?** Yes. For each
   scenario and origin, the threshold is the linear-interpolation Q90 of
   `abs(log(actual / B3))` from replay rows with `pseudo_target_year <= origin-2`.
   The current outer rows are never included in the threshold. q=0.85 and
   q=0.95 are descriptive sensitivity only.
2. **Are jump positives and per-origin coverage sufficient?** Primary q90 has
   `68` unique resident jump rows and `6/11`
   origins with at least two rows; sample gate pass is `True`.
3. **Does F4 remaining error concentrate on jump rows?** Gate A passes for both
   R1/R2: `23.293x` / `25.656x`
   jump-vs-normal F4 MSE ratios, jump SSE/row-share concentration ratios
   `5.240` /
   `5.333`, and F4 is worse
   on `8` / `9`
   of 11 origins.
4. **Which precursors are stable across origins?** R1 Gate-B descriptive
   candidates: `none`. R1/R2 Gate-C survivors: `none`.
5. **Does the signal survive R2?** Gate-C pass is `False`;
   only features with matching R1/R2 effect direction and the preregistered
   R2 threshold are counted.
6. **Can scale/city identity explain the association?** Scale-control rows are
   retained in the panel and reported with Spearman association to
   `pop_log1p_latest`. City is used only as a join/descriptive recurrence key;
   no city route is created. This is not causal identification.
7. **Does old F2 show jump-helpful / normal-harmful utility?** F2 OOF was
   recovered from immutable tag `archive/pre-s6/s5-plan06b-catboost-screening` and reanalyzed without retraining.
   Gate-D pass is `False`; per-scenario jump/normal MSE, delta SSE and
   origin wins are in `f2_conditional_utility.csv`.
8. **Which conclusion applies?** `H_JUMP_RISK_IDENTIFIABLE_BUT_NOT_DEPLOYABLE`.
9. **Should Main upgrade to a conditional-model Track?** `NO — keep this as a limitation/closed supplemental track unless Main makes a new decision.`
10. **Exact branch / commit / base / commands / deviations:** branch is
    `codex/s7-supplemental-jump-plan01`, base is `bfe265d3d57f43938e967890567a4162d7b3c896`, run head before artifact generation is
    `bfe265d3d57f43938e967890567a4162d7b3c896`; commands are listed in `investigation_summary.json`;
    deviations: none. F4/F3 were recovered from `archive/pre-s6/s5-plan06e-delta-growth-framing`, and F2 from
    `archive/pre-s6/s5-plan06b-catboost-screening` because those worker artifacts are not present in current main
    but are preserved by immutable tags. No worker file was re-run.

## Evidence outputs

- `jump_event_labels.csv` — q90 row labels and training-derived thresholds.
- `jump_event_geometry.csv` — detailed B3/F4 event rows and errors.
- `f4_jump_tail_overlap.csv` — origin/pooled jump concentration and tail diagnostics.
- `precursor_event_study.csv` — fixed primary panel plus scale/availability controls.
- `precursor_r1_r2_comparison.csv` — R1/R2 effect and gate comparison.
- `information_boundary_audit.csv` — leakage and replay-boundary checks.
- `f2_conditional_utility.csv` — recovered Plan06B F2 conditional utility.
- `investigation_summary.json` — machine-readable decision and provenance.

## Return to Main

The worker has reached its exit condition and returns to `Main｜主教学 / Mainline Authority`.
The frozen S6 resident F4-R1 solution remains unchanged. Main decides whether
to close the supplemental track, continue S7 review, or authorize a new
low-capacity conditional investigation; this worker does not enter that track.
