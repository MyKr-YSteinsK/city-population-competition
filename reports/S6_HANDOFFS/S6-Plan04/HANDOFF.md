# S6-Plan04 Handoff — Final Freeze & Closeout

## Result

- Status: `FINAL_SOLUTION_FROZEN_FOR_MAIN_CLOSEOUT`
- Final local solution: resident / F4 / R1 / CatBoost C3
- Final submission: `reports/S6_HANDOFFS/S6-Plan04/submission_final.csv`
- High-precision provenance: `reports/S6_HANDOFFS/S6-Plan04/submission_resident_f4_r1_high_precision.csv`
- Manifest: `reports/S6_HANDOFFS/S6-Plan04/final_solution_manifest.json`
- Model fits in Plan04: `0`

The final local solution is frozen as resident F4-R1. There was no leaderboard
score before freeze. Per the closeout decision, no further optimization or
full reproduction was started; the deferred reproduction boundary is recorded
in the final report and manifest.

The earlier Plan03 two-decimal formatting commit changed the working copy at
the historical source path. For traceability, the exact high-precision R1
artifact was recovered from immutable commit `e7040aff56eac45d3cbaadc8045ba446d9b9f02d` and used only
as provenance for the final rendering. The current Plan03 formatted candidate
and `submission_final.csv` contain the same values at two-decimal precision.

Registered alternatives are no longer selected. The incomplete or unsupported
families and the registered city40 anomaly remain explicitly documented; no
clipping, routing, ensemble, or manual correction was introduced.

## Main review request

Lab recommends that Main close the S6 implementation phase and decide the
project/learning-state handling. This handoff does not declare `S6_COMPLETE`,
does not rewrite `CURRENT_LEARNING_CHECKPOINT.md`, and does not declare S7.
