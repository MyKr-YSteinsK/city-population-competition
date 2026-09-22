# S8 Plan02 Truth Provenance Hardening Handoff

Status: S8_TRUTH_PROVISIONAL_FREEZE_RETAINED
Branch: codex/s8-truth-reconstruction-v1
Base SHA: 0a74372ea4e3555b3d659e6a2c77647b92eabea4
Artifact Freeze Commit: PENDING_ARTIFACT_FREEZE_COMMIT
Final Branch HEAD: PENDING_FINAL_BRANCH_HEAD
Working tree at generation: metadata may be pending after artifact commit

Truth version: candidate_truth_2023_v2
Evaluator SHA256: de8ae0feb8b4061610b82a235a673fc2a841d2969f04924d7f69185c5e4db02c
Full truth SHA256: 2c4d3eafe6eb1eb0b0297944cbdf421a60fae809cb86dc81949b955337e0b692

Source upgrade:
- Tier1 truth rows before: 19
- Tier1 truth rows after: 22
- Tier2 truth rows after: 14
- fallback truth rows after: 4
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
candidate_truth_2023_v2 remains provisional and waits for Main review. Do not reveal truth
to S9 before the independent prediction freeze.
