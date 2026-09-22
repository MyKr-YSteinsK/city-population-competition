# S8 Plan02 Truth Provenance Hardening Handoff

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE
Branch: codex/s8-truth-reconstruction-v1
Base SHA: 0a74372ea4e3555b3d659e6a2c77647b92eabea4
Artifact Freeze Commit: PENDING_ARTIFACT_FREEZE_COMMIT
Final Branch HEAD: PENDING_FINAL_BRANCH_HEAD
Working tree at generation: metadata may be pending after artifact commit

Truth version: candidate_truth_2023_v1 (successor v2 required)
Evaluator SHA256: 026c2c4723eadad8ae8519586e0a146d47b1bd0525fde315aa4ec4994e86927d
Full truth SHA256: ca7b86cc5cbc98f01562a2a7cabf14e01f90eae44986f841b3030451da940ad7

Source upgrade:
- Tier1 truth rows before: 19
- Tier1 truth rows after: 22
- Tier2 truth rows after: 13
- fallback truth rows after: 5
- unresolved source rows: 4

City40:
- mapping before: PROVISIONAL
- mapping after: HIGH_CONFIDENCE
- registered=65 explanation: ANOMALY_CAUSE_UNRESOLVED; literal raw value propagated unchanged; no correction
- remaining uncertainty: density conflict and anomaly cause

City23:
- official 2022 value: 247.72
- rendered truth: v1=245, v2=248
- changed vs v1: YES; v2 required
- old chat 248: ignored as evidence

Truth delta:
- raw value rows changed in v2: 1
- evaluator rows changed: 1
- v2 required: YES

Isolation:
- S9 branch read: NO
- S9 predictions read: NO
- model scoring performed: NO
- test.xls accessed: NO

Return:
candidate_truth_2023_v1 (successor v2 required) remains provisional and waits for Main review. Do not reveal truth
to S9 before the independent prediction freeze.
