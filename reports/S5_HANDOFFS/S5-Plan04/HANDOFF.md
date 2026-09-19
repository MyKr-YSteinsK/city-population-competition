# S5-Plan04 Handoff — 全量多表特征基座与城市异质性接口

## Result

- Plan ID：`S5-Plan04`
- Stage：`S5_EVIDENCE_DRIVEN_ITERATION`
- Frozen base：`d75ec4b9b1b392003c18201fa4b27899720f3c81`
- Worker branch：`s5/plan04-full-multisource-feature-foundation`
- Exact worker commit at handoff generation：`d75ec4b9b1b392003c18201fa4b27899720f3c81`
- Foundation result：`full_multisource_foundation_ready`
- S5 未声明完成，S6 未进入；Stage authority 仍为 Main。

## Scope and actual files

本 worker 只建立 model-agnostic full multi-source matrix 与接口，不训练模型、不调参、不做 feature selection、不生成 2023 人口预测或 submission。

实际生成文件：

- `scripts/build_s5_plan04_full_multisource_feature_foundation.py`
- `reports/S5_HANDOFFS/S5-Plan04/full_multisource_feature_matrix.csv`
- `reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv`
- `reports/S5_HANDOFFS/S5-Plan04/feature_group_manifest.csv`
- `reports/S5_HANDOFFS/S5-Plan04/feature_availability_summary.csv`
- `reports/S5_HANDOFFS/S5-Plan04/source_boundary_audit.csv`
- `reports/S5_HANDOFFS/S5-Plan04/feature_matrix_schema.json`
- `reports/S5_HANDOFFS/S5-Plan04/foundation_summary.json`
- `reports/S5_HANDOFFS/S5-Plan04/oof_prediction_contract.csv`

## Commands executed

- `python -m py_compile scripts/build_s5_plan04_full_multisource_feature_foundation.py`
- `python scripts/build_s5_plan04_full_multisource_feature_foundation.py`
- `git diff --check`

## Frozen matrix checks

- Rows：`1760` / expected `1760`; unique keys：`1760`.
- Keys：`(target_id, scenario_id, origin_year, city_id)`; targets resident/registered; scenarios R1/R2; origins 2011–2021; 40 cities.
- Model features：`144` across `14` logical source groups and `9` semantic feature groups.
- `city_id` remains raw categorical metadata; no one-hot encoding is performed.
- Numeric model missingness is written as `NaN`; no global fill or Plan04 imputer is fitted.

## Source and leakage evidence

- S2 `replay_scenario_definitions.csv` and `reference_forecast_geometry.csv` are reused; no publication date is invented.
- Population target-history features and B1/B2/B3 references use `<= T-2` only; origin labels are label/reference-only.
- Annual source caps are checked per source and scenario; age remains benchmark-only `{2000, 2010, 2020}` with no interpolation.
- Density regime crossings suppress only affected dynamics; levels and the retained regime audit remain visible.
- Employment derived rows remain provenance-flagged; quarantined values are not reintroduced.
- Source boundary audit rows：`24640`; cap violations：`0`; quarantine reintroductions：`0`.

## B3 compatibility

- resident R1 pooled MSE：`872.452651515142861`; R2：`872.452651515142861`.
- registered R1 pooled MSE：`225.421212121213244`; R2：`225.421212121213244`.
- Frozen expected values passed within floating tolerance: resident `872.452651515143`, registered `225.421212121213`.

## R1/R2 feature differences

- Compared `880` target×origin×city pairs; pairs with any model-feature difference：`880`.
- Label/reference columns numerically equivalent across R1/R2：`True`.
- Differences are expected from source-specific R2 stress caps; target history and benchmark-only age retain separate geometry.

## 2023 readiness and forbidden actions

- Optional 2023 feature-readiness audit：not executed; no 2023 population prediction was created.
- No LightGBM/CatBoost/XGBoost/ExtraTrees/Ridge/HGB/neural training, tuning, target-based feature ranking, routing, blend, gating, or submission was run.

## Recommendation to Main

- The common feature contract is safe for review and, after Main integrates this worker tip, can be consumed by parallel Plan05A/B/C/D workers.
- Future workers must consume this exact contract and return `feature_contract_change_required` rather than privately extending the matrix.
- Main must independently review/integrate Plan04; this handoff does not declare S5 complete or authorize S6.

## Unresolved risks

- Target proxy and source publication timing remain conditional S2 assumptions.
- Downstream model-specific missing-value handling and city-aware variants remain untested by design.
- Feature foundation completeness does not establish that any future model will beat B3.
