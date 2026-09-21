# S8 Truth Track Handoff

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE
Branch: codex/s8-truth-reconstruction-v1
Base SHA: 0a74372ea4e3555b3d659e6a2c77647b92eabea4
Final Commit SHA: 2560c6e4b78c4d24eae2119b9eaab5a025af54e7
Working tree at generation: M reports/S8_TRUTH/truth_v1/S8_TRUTH_RECONSTRUCTION_REPORT.md

Truth version: candidate_truth_2023_v1
Truth SHA256 evaluator: 026c2c4723eadad8ae8519586e0a146d47b1bd0525fde315aa4ec4994e86927d

City mapping:
- confirmed: 19
- high confidence: 20
- provisional: 1
- unresolved: 0

Year alignment:
- selected hypothesis: competition year = real year + 1
- evidence scope: 40 cities; resident, registered, urbanization, density; four province groups
- exceptions: city3 and city40 registered conflicts; census/yearbook presentation differences

Target semantics:
- result: RESIDENT_POPULATION
- confidence: HIGH_WITH_EXPLICIT_EXCEPTIONS
- conflicts: retained in truth_conflicts.csv

Sources:
- total source ledger rows: 29
- Tier 1 sources: 15
- fallback source rows: 9

Truth reconstruction:
- resolved rows: 40
- provisional rows: 1
- unresolved rows: 0
- unit: raw 万人; evaluator integer 万人
- rendering rule: ROUND_HALF_UP_TO_INTEGER_WAN

Isolation:
- S9 branch read: NO
- S9 predictions read: NO
- model evaluation performed: NO
- test.xls accessed: NO

Project documentation drift:
- CURRENT_LEARNING_CHECKPOINT.md and learning documents were not modified.
- Main was not modified or merged.

Return:
S8 truth track remains provisional and waits for Main review. Do not feed the
truth values to S9 from this worker.
