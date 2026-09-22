# TASK_RESULT — S10-Plan01

## Result

- Status: `COMPLETE`
- Branch: `codex/s10-plan01-external-script-forensics`
- Base SHA: `0f8479f740ca938acc5ff2d7026d22ca1cdd1755`
- Final commit / remote push: reported after delivery; the report itself avoids a self-referential commit hash.

## What Changed

- Accepted the user-supplied external script immutably after exact size/SHA256 verification.
- Added bounded static forensics, raw-input manifests, disposable exact-reproduction evidence, original-semantics validation predictions, independent metrics, `675` search, selector diagnostic, contract comparison, leakage matrix, and source-handling audit.
- Added only non-model forensic runners; no production/S6/S8/S9 artifact was changed.

## Verification Actually Passed

- Original hash: 20,738 bytes, SHA256 `f719fa577d2c5b8fd51a024af0dd084390f849e7c3564a34ac816a4b35fb8342`, expected match `YES`.
- Seven legal raw files copied with exact hashes; runtime has no `test.xls`.
- Syntax check passed.
- Exact original command succeeded in Run B with exit code 0; Run A/B/C stdout and both submission files are byte-identical.
- Forensic harness exit code 0; independent RMSE identity and script metric comparison passed.
- No S8 truth, real-2022/real-2023 target, B-data, S6 mutation, or S8/S9 merge was performed.

## Plan Deviations

- None material. The disposable run used a workspace-local temporary directory rather than committing duplicate raw XLSX copies; all paths and hashes are recorded.
- Run A and extra Run C exit codes were not retained by the outer 30-second host wait, so they are reported as output-successful with exit code unobserved; Run B provides the captured exit-code evidence.

## Material Project-State Changes

- PROJECT_BRIEF: none.
- DECISIONS: none.
- CURRENT_STATE: none; `CURRENT_LEARNING_CHECKPOINT.md` was intentionally not modified.

## Remaining Risks / Known Limits

- `675 provenance unresolved`; this Plan cannot identify an external evaluator or hidden target without new authorized evidence.
- External T-1 covariate timing is not proven equivalent to project R1/R2 caps.
- The registered hard-coded repair and same-sample selector make the script's historical numbers unsuitable as an unbiased project comparison.
- The source parser's six living-standard frames are empty because of its header assumption; Plan01 records this behavior and does not repair it.

## Checkpoint Signal

- None.

## Suggested Next Boundary

- Main/S10 review should decide whether to authorize Plan02 same-contract 11-origin re-evaluation, nested/prior-only selector audit, and controlled ablation.
