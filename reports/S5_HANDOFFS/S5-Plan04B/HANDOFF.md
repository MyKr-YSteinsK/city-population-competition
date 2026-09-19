# S5-Plan04B Handoff — 多表训练回放矩阵与时间切分基座

## Result

- Plan ID：`S5-Plan04B`
- Stage：`S5_EVIDENCE_DRIVEN_ITERATION`
- Plan04 integrated base：`fac7ed9bf11459bc5d7c6772857683417bf70eac`（pre-integration gate 为 ff-only）
- Worker branch：`s5/plan04b-training-replay-foundation`
- Build commit：`fac7ed9bf11459bc5d7c6772857683417bf70eac`
- Result：`training_replay_foundation_ready`
- 本 worker 只建立历史训练回放和时间切分契约；未训练强模型，S5 未声明完成，S6 未进入。

## Replay matrix

- Rows：`2320`；unique keys：`2320`。
- Pseudo-target year：`2004`–`2021`，共 `18` 个年份。
- Row counts by target/scenario：`{"registered|R1_REFERENCE_SNAPSHOT_LAG":587,"registered|R2_CONSERVATIVE_PUBLICATION_STRESS":587,"resident|R1_REFERENCE_SNAPSHOT_LAG":573,"resident|R2_CONSERVATIVE_PUBLICATION_STRESS":573}`。
- Excluded target-label rows：`480`；reasons：`{"b3_prediction_unavailable":480,"no_target_label":560}`。
- Replay eligibility requires actual target label, complete generated feature schema, and computable frozen B3 reference; individual feature cells remain missing when unavailable.
- Direct and residual framings share the same replay-row universe.

## Temporal split geometry

- Outer rows：`44`；all training pools non-empty：`True`。
- Outer validation groups retain 40 rows/cities：`True` / `True`。
- Outer boundary violations：`0`。
- Inner split rows：`484`；earliest available inner validation years：`{"registered|R1_REFERENCE_SNAPSHOT_LAG":2004,"registered|R2_CONSERVATIVE_PUBLICATION_STRESS":2004,"resident|R1_REFERENCE_SNAPSHOT_LAG":2004,"resident|R2_CONSERVATIVE_PUBLICATION_STRESS":2004}`。
- Inner boundary violations：`0`；zero-training folds reported but not removed：`88`。
- Frozen rules: outer training `pseudo_target_year <= outer_origin - 2`; inner training `pseudo_target_year <= inner_validation_year - 2`.

## Contract and leakage verification

- Plan04 formal compatibility：`PASS`; 1760 rows; model-feature mismatch cells：`0`; label/reference mismatch cells：`0`。
- B3 compatibility：`True`; resident expected pooled MSE `872.452651515143`; registered expected pooled MSE `225.421212121213`.
- Replay boundary rows：`32480`; source-cap violations：`0`; quarantine reintroductions：`0`; benchmark violations：`0`。
- Population-history cutoff violations：`0`; B3-history cutoff violations：`0`。
- Plan04's exact 144-feature contract, source-specific R1/R2 caps, benchmark-only age domain, S1 quarantine, and explicit missingness are retained.

## Required artifacts

- `HANDOFF.md`
- `full_multisource_training_replay_matrix.csv`
- `training_replay_summary.json`
- `training_replay_year_coverage.csv`
- `outer_training_geometry.csv`
- `inner_split_geometry.csv`
- `plan04_feature_compatibility.csv`
- `replay_boundary_audit.csv`
- `parallel_worker_input_contract.json`

## Forbidden actions not run

- No LightGBM/CatBoost/XGBoost/ExtraTrees/Ridge/HGB/neural training, tuning, feature selection, routing, gating, blending, 2023 prediction, or submission.
- `CURRENT_LEARNING_CHECKPOINT.md` was not modified.

## Next boundary

- Main/Lab should review and integrate this worker tip before creating Plan05A/B/C/D from one exact frozen base.
- Future workers must consume this replay matrix and split geometry; feature changes require `feature_contract_change_required` / Main review.
