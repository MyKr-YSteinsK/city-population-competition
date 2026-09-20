# S5-Plan06F Recovery Note

Status: `execution_completed_but_core_performance_evidence_unrecoverable`

This note is a recovery archive only. Plan06F was not rerun during S5-Plan07 consolidation, and no missing performance metric is inferred or fabricated.

## Verified local session evidence

- Local Codex session record: `C:\Users\Yukis\.codex\sessions\2026\09\19\rollout-2026-09-19T22-28-10-01a0ba11-4a43-7662-89d1-0aacd29e45a1.jsonl`.
- The session record contains the completed first-run stdout and the final recovery statement (record line 5841); the first run reported a unique continuous `outer_fit=1..88` sequence.
- First run duration was approximately `23,398` seconds (about 6 hours 30 minutes).
- Recovered final selections: `C1=64/88`, `C2=24/88`; `F1=44/44 C1`; `F2=20 C1 / 24 C2`.
- Independently recoverable geometry: `44` outer groups, `264` eligible folds, `8` fallback groups. Correct inner evidence row count is `1560`; the runner assertion expected `1632` and failed before final artifact writing.
- The worker branch has no committed Plan06F performance artifacts beyond environment evidence; the session record states that no OOF/aggregate/per-origin/inner-MSE/term-importance/city-error files were written before the assertion.

## Evidence boundary

The following are **unavailable**, not zero and not reconstructed:

- EBM OOF predictions;
- outer/per-origin/aggregate metrics;
- inner per-configuration MSE and mean validation MSE;
- term importance, interaction summaries, and city errors.

Selection counts are execution diagnostics only and are not model-performance evidence. Plan06F is excluded from the final candidate comparison and no rerun is authorized by this consolidation.

## Reproducibility pointers

- Historical worker branch name (removed after recovery preservation during Plan08 cleanup): `s5/plan06f-ebm-additive-screening`.
- Frozen Wave base: `1f97f67488cd7aec120ef8b9fa04407dc13d54ba`.
- Environment evidence, if needed for audit, is retained in final main at `reports/S5_HANDOFFS/S5-Plan06F/worker_environment.json`.
