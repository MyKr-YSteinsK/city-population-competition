# S6-Plan01 Handoff｜F4 最终方案合同冻结与复现边界审计

## Result

- Status: `PARTIALLY_FROZEN_SUBMISSION_BLOCKED`
- `ready_for_main_stage_review`: `true`
- `ready_for_s6_clean_rebuild`: `true`
- `formal_submission_generated`: `false`
- Execution source main SHA: `f9b97f32adc0e60e6d270db16392e4d46e7ddbe4`
- `origin/main` at audit: `f9b97f32adc0e60e6d270db16392e4d46e7ddbe4`
- Branch at audit: `main`
- This handoff records the audited source state; the exact delivery HEAD is reported after commit/push verification.

## What was frozen

- Resident-primary proxy target and F4 log-growth framing.
- Two future final-pipeline target paths: resident `primary` and registered `alternative`, each to be trained and submitted separately only after authorization; no averaging or mixing.
- Plan04/04B exact 144-feature contract, raw NaN, native `city_id`, and S1 guards.
- Existing Plan02 R1/R2 current-information mappings: `R1_REFERENCE_SNAPSHOT_LAG=EXACT_R1; R2_CONSERVATIVE_PUBLICATION_STRESS=EXACT_R2`; no new policy was introduced.
- Conceptual 2023 final-training geometry with target labels capped at 2021.
- CatBoost 1.2.10, three pre-registered configs, seed, deterministic single-thread expectations, and historical inner selection/fallback algorithm.
- B3 authoritative control and whole-solution fallback.

## Evidence paths

- `FINAL_SOLUTION_CONTRACT.md`
- `final_solution_contract.json`
- `dependency_map.md`
- `rebuild_manifest.csv`
- `current_2023_information_set_audit.csv`
- `final_training_geometry.csv`
- `historical_lineage_hashes.csv`
- `submission_schema_evidence.md`
- `environment_snapshot.json`

## Verification actually passed

- `main` was clean before the audit and was synchronized to `origin/main` at the audited source SHA.
- 7 legal XLSX files and the official PDF were read; `test.xls` was not opened or hashed.
- 14 logical source mappings × R1/R2 were matched to existing Plan02 policy rows; all rows are `EXACT_R1` or `EXACT_R2`.
- 144 feature names match Plan04 definitions, Plan04 schema, and Plan04B worker contract in order.
- Replay geometry: 2,320 unique rows; final label maximum 2021; 40-city coverage; zero duplicate keys; zero source-cap, benchmark-domain, population-cutoff, B3-cutoff, and quarantine-reintroduction violations in the audited boundary rows.
- Historical Plan06E runner/config/lineage was read from immutable Git objects; no runner cherry-pick and no model fit was performed.
- Official PDF schema evidence was extracted and cross-checked with the current Main checkpoint.

## Submission and risks

- The official target says `总人口`, but resident versus registered and official unit are not proven. This is the reason for `PARTIALLY_FROZEN_SUBMISSION_BLOCKED`.
- R1/R2 are observed-snapshot freshness proxies, not proven publication dates.
- F4 is a resident-primary local candidate; it is not an official-target or universal-target claim.
- The current Main checkpoint requires retaining a separately trainable registered alternative until a real platform score can inform target-mapping review; this Plan does not execute that later step.
- No 2023 prediction, submission, model selection experiment, tuning, feature change, ensemble, routing, S7 entry, or checkpoint edit was performed.

## Recommendation to Main

Accept this packet for Main stage review and, if the target/unit boundary is accepted, proceed to the separately authorized clean raw rebuild while preserving both target paths. Keep formal submission generation blocked until Main resolves or explicitly accepts the resident-proxy/unit limitation. Do not declare `S6_COMPLETE`.
