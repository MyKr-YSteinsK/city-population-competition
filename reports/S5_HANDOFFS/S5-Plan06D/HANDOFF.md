# S5-Plan06D Handoff — ExtraTrees 与正则线性全量多表对照 screening

## Result

- Plan ID：S5-Plan06D
- Stage：S5_EVIDENCE_DRIVEN_ITERATION
- Worker branch：s5/plan06d-extratrees-linear-controls
- Exact base SHA：1f97f67488cd7aec120ef8b9fa04407dc13d54ba
- Runner commit used for training：006f1446b702975d68d1fed93309218c13411d98
- Final evidence commit：006f1446b702975d68d1fed93309218c13411d98
- Seed：20260919；CPU single-thread estimator：n_jobs=1

本 worker 只比较预注册的 ExtraTrees 与 Ridge 控制族；没有改动 144 个特征、replay universe、outer/inner geometry 或 Plan04/04B/05 证据。

## Coverage and selection

- Formal OOF：每个 family（两个 framing）3520 行；两 family 合计 7040 行。每个 framing-specific promoted candidate 均覆盖 1760 行。
- Framings：F1_DIRECT 与 F2_B3_RESIDUAL；targets：resident / registered；scenarios：R1 / R2。
- Nested selection groups：每个 family 88；合计 176。
- Eligible inner folds：264 / 484。
- Fallback groups：ExtraTrees 16；Ridge 16。
- Selection objective：eligible inner folds 的 mean validation MSE；tie tolerance 1e-12，按注册配置顺序优先。
- Fallback：eligible folds 少于 3 时使用预注册 conservative default，未扩大边界。

## Resident primary metrics

| family | framing | scenario | pooled MSE | B3 MSE | improvement % | origin wins | top1 city SSE share |
|---|---|---|---:|---:|---:|---:|---:|
| EXTRATREES | F1_DIRECT | R1 | 2497.691574 | 872.452652 | -186.284 | 1 | 0.493 |
| EXTRATREES | F1_DIRECT | R2 | 2527.092322 | 872.452652 | -189.654 | 0 | 0.497 |
| EXTRATREES | F2_B3_RESIDUAL | R1 | 1013.779368 | 872.452652 | -16.199 | 3 | 0.276 |
| EXTRATREES | F2_B3_RESIDUAL | R2 | 1037.456011 | 872.452652 | -18.913 | 3 | 0.275 |
| RIDGE | F1_DIRECT | R1 | 5462.366043 | 872.452652 | -526.093 | 2 | 0.089 |
| RIDGE | F1_DIRECT | R2 | 4271.202153 | 872.452652 | -389.563 | 2 | 0.124 |
| RIDGE | F2_B3_RESIDUAL | R1 | 1020.553558 | 872.452652 | -16.975 | 3 | 0.276 |
| RIDGE | F2_B3_RESIDUAL | R2 | 1039.411507 | 872.452652 | -19.137 | 3 | 0.275 |

Registered-target metrics、macro-origin MSE、latest-2021 MSE、RMSE、MAE、bias、per-origin 与 per-city SSE/MSE 均在 CSV evidence 中提供。

## Classification

- EXTRATREES / F1_DIRECT：family_unsupported； R1/R2 improvement={"R1_REFERENCE_SNAPSHOT_LAG": -186.28391122757787, "R2_CONSERVATIVE_PUBLICATION_STRESS": -189.6538073177256}
- EXTRATREES / F2_B3_RESIDUAL：family_unsupported； R1/R2 improvement={"R1_REFERENCE_SNAPSHOT_LAG": -16.198783548537808, "R2_CONSERVATIVE_PUBLICATION_STRESS": -18.9125861103307}
- RIDGE / F1_DIRECT：family_unsupported； R1/R2 improvement={"R1_REFERENCE_SNAPSHOT_LAG": -526.0931219525892, "R2_CONSERVATIVE_PUBLICATION_STRESS": -389.5626307511948}
- RIDGE / F2_B3_RESIDUAL：family_unsupported； R1/R2 improvement={"R1_REFERENCE_SNAPSHOT_LAG": -16.975237082895188, "R2_CONSERVATIVE_PUBLICATION_STRESS": -19.136723980084422}
- Family-level screening label — ExtraTrees：family_unsupported；Ridge：family_unsupported。
- Overall worker classification：family_unsupported。
- Overall lower resident pooled-MSE family across the registered screening cells：EXTRATREES。

## Diagnostics

- ExtraTrees：outer-selected feature importance、feature group、city one-hot、tree capacity 均写入 feature_importance.csv。
- Ridge：outer-selected coefficient magnitude、feature group、city one-hot、missing indicators、alpha 与稀疏度均写入同一诊断表。
- feature_importance.csv 同时包含按 selected outer fits 聚合的 feature/group importance，以及 Ridge coefficient sign stability。
- Fold-local median imputation + missing indicators；Ridge 仅在同一 fold 内追加 StandardScaler；ExtraTrees 不缩放。
- Library/runtime limitations：使用项目已冻结的 Python / numpy / pandas / scikit-learn 环境；ExtraTrees 与 Ridge 均为 CPU、single-thread；没有 GPU 或并行随机性。

## Boundary

- No Optuna、Bayesian optimization、random search、large grid、city routing、gating、blend、2023 forecast、submission、S5 complete 或 S6。
- No checkpoint update；本 handoff 不是全局 winner 宣告。下一步由主线在所有 Plan06A/B/C/D worker evidence 汇总后统一判断。

## Artifacts

- oof_predictions.csv
- aggregate_metrics.csv
- per_origin_metrics.csv
- configuration_selection.csv
- inner_selection_evidence.csv
- city_model_error.csv
- missingness_diagnostics.csv
- feature_importance.csv
- model_summary.json
