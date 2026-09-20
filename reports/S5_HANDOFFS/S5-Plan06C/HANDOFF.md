# S5-Plan06C Handoff — XGBoost 全量多表强模型 screening

## TASK_RESULT

- 分支：`s5/plan06c-xgboost-screening`。
- exact Wave base：`1f97f67488cd7aec120ef8b9fa04407dc13d54ba`。
- 本次 runner 读取并生成 artifacts 时的代码 tip：`af56a4bd3e1bbfc47f4af18470b60bbd88921479`。最终 worker tip 以该分支提交后的 git tip 与主结果回复为准。
- worker classification：`family_unsupported`。这只是 XGBoost family-level screening 结论，不是全局 winner。

## Frozen contract and execution

- Feature contract：`S5-Plan04-FULL-MULTISOURCE-V1`；144 numeric features；city_id 使用固定 `city1..city40` one-hot。
- Replay contract：`S5-Plan04B-TRAINING-REPLAY-V1`；replay 2320 rows；formal validation 1760 rows；每个 framing 的 OOF coverage 为 1760，合并 F1/F2 后文件为 3520 rows。
- Outer training：`pseudo_target_year <= outer_origin_year - 2`；inner eligible fold：training >= 40、validation >= 20、`boundary_pass=PASS`。
- 264 eligible inner folds / 484 geometry rows；8 个 outer fit group 使用 C1 conservative fallback。
- XGBoost：`objective=reg:squarederror`、`tree_method=hist`、`random_state=20260919`、`n_jobs=1`、无 early stopping；原始 NaN 交给 native missing handling。

## Resident / registered metrics

| target | framing | scenario | pooled MSE | B3 MSE | origin wins | latest-2021 MSE | RMSE | MAE | bias |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| resident | F1_DIRECT | R1_REFERENCE_SNAPSHOT_LAG | 2409.651079 | 872.452652 | 1/10/0 | 2874.812591 | 49.088197 | 24.457750 | -5.320709 |
| resident | F2_B3_RESIDUAL | R1_REFERENCE_SNAPSHOT_LAG | 1047.165414 | 872.452652 | 3/8/0 | 3268.798444 | 32.359935 | 16.121199 | -0.095024 |
| resident | F1_DIRECT | R2_CONSERVATIVE_PUBLICATION_STRESS | 2347.954808 | 872.452652 | 1/10/0 | 3608.386196 | 48.455699 | 23.880002 | -6.020159 |
| resident | F2_B3_RESIDUAL | R2_CONSERVATIVE_PUBLICATION_STRESS | 1102.814816 | 872.452652 | 3/8/0 | 3677.318467 | 33.208656 | 16.504312 | -0.839685 |
| registered | F1_DIRECT | R1_REFERENCE_SNAPSHOT_LAG | 576.446134 | 225.421212 | 0/11/0 | 1697.333899 | 24.009293 | 10.288767 | -4.398187 |
| registered | F2_B3_RESIDUAL | R1_REFERENCE_SNAPSHOT_LAG | 232.893638 | 225.421212 | 5/6/0 | 1543.911755 | 15.260853 | 6.320385 | 0.353544 |
| registered | F1_DIRECT | R2_CONSERVATIVE_PUBLICATION_STRESS | 536.738605 | 225.421212 | 0/11/0 | 1653.143348 | 23.167620 | 9.995191 | -3.816758 |
| registered | F2_B3_RESIDUAL | R2_CONSERVATIVE_PUBLICATION_STRESS | 228.691523 | 225.421212 | 5/6/0 | 1502.702772 | 15.122550 | 6.306928 | 0.600817 |

胜负顺序为 `wins/losses/ties`；pooled MSE 与 latest-2021 MSE 越低越好。

## Direct / residual and R1 / R2 comparisons

- `resident` / `R1_REFERENCE_SNAPSHOT_LAG`：F2 residual - F1 direct pooled MSE `-1362.485665`；latest-2021 delta `393.985853`；origin wins F1/F2 `1/3`。
- `resident` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：F2 residual - F1 direct pooled MSE `-1245.139992`；latest-2021 delta `68.932271`；origin wins F1/F2 `1/3`。
- `registered` / `R1_REFERENCE_SNAPSHOT_LAG`：F2 residual - F1 direct pooled MSE `-343.552496`；latest-2021 delta `-153.422144`；origin wins F1/F2 `0/5`。
- `registered` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：F2 residual - F1 direct pooled MSE `-308.047083`；latest-2021 delta `-150.440576`；origin wins F1/F2 `0/5`。
- `resident` / `F1_DIRECT`：R2 - R1 pooled MSE `-61.696271`；latest-2021 delta `733.573605`；origin wins R1/R2 `1/1`。
- `resident` / `F2_B3_RESIDUAL`：R2 - R1 pooled MSE `55.649402`；latest-2021 delta `408.520023`；origin wins R1/R2 `3/3`。
- `registered` / `F1_DIRECT`：R2 - R1 pooled MSE `-39.707529`；latest-2021 delta `-44.190552`；origin wins R1/R2 `0/0`。
- `registered` / `F2_B3_RESIDUAL`：R2 - R1 pooled MSE `-4.202116`；latest-2021 delta `-41.208984`；origin wins R1/R2 `5/5`。

## Configuration and fallback distribution

- Selected outer fits：`88`；fallback fit count：`16`；selection fit count：`72`。
- Config counts：`XGB_C1_CONSERVATIVE`=37, `XGB_C2_MEDIUM`=11, `XGB_C3_FLEXIBLE_REGULARIZED`=40

## City concentration

- `resident` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG`：top-1 `city17` concentration `0.5232`；top-3 `city17;city5;city3` concentration `0.7220`。
- `resident` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG`：top-1 `city5` concentration `0.2769`；top-3 `city5;city33;city21` concentration `0.5140`。
- `resident` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：top-1 `city17` concentration `0.5127`；top-3 `city17;city5;city3` concentration `0.7090`。
- `resident` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：top-1 `city5` concentration `0.2864`；top-3 `city5;city33;city21` concentration `0.5108`。
- `registered` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG`：top-1 `city15` concentration `0.4222`；top-3 `city15;city40;city24` concentration `0.7406`。
- `registered` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG`：top-1 `city40` concentration `0.4063`；top-3 `city40;city24;city3` concentration `0.7684`。
- `registered` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：top-1 `city15` concentration `0.3875`；top-3 `city15;city40;city24` concentration `0.7276`。
- `registered` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS`：top-1 `city40` concentration `0.4135`；top-3 `city40;city24;city3` concentration `0.7620`。

## Top feature groups by normalized gain

- `FG-P` rank `1`，mean normalized gain `0.584532` （raw gain sum `30960062.527978`）。
- `FG-AGE` rank `2`，mean normalized gain `0.080086` （raw gain sum `577654.277600`）。
- `FG-LIFE` rank `3`，mean normalized gain `0.078818` （raw gain sum `271973.503332`）。
- `FG-EMP` rank `4`，mean normalized gain `0.072927` （raw gain sum `2108255.212475`）。
- `FG-SECTOR` rank `5`，mean normalized gain `0.063457` （raw gain sum `621477.676399`）。
- `FG-D` rank `6`，mean normalized gain `0.037835` （raw gain sum `169736.073403`）。
- `FG-UNEMP` rank `7`，mean normalized gain `0.030648` （raw gain sum `128974.797788`）。
- `FG-U` rank `8`，mean normalized gain `0.025383` （raw gain sum `108255.279546`）。

## B3 error complementarity diagnostic

- `resident` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG` / `origin`：signed-error corr `0.9399751191086174`；model/B3 better units `1/10/0`。
- `resident` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG` / `city`：signed-error corr `-0.1208153225527612`；model/B3 better units `8/32/0`。
- `resident` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG` / `origin`：signed-error corr `0.9166436013730126`；model/B3 better units `3/8/0`。
- `resident` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG` / `city`：signed-error corr `0.791887634961493`；model/B3 better units `14/26/0`。
- `resident` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `origin`：signed-error corr `0.9501852611124812`；model/B3 better units `1/10/0`。
- `resident` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `city`：signed-error corr `-0.08022795666237234`；model/B3 better units `7/33/0`。
- `resident` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `origin`：signed-error corr `0.9125589501666208`；model/B3 better units `3/8/0`。
- `resident` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `city`：signed-error corr `0.8332979739010795`；model/B3 better units `7/33/0`。
- `registered` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG` / `origin`：signed-error corr `0.3737689834580257`；model/B3 better units `0/11/0`。
- `registered` / `F1_DIRECT` / `R1_REFERENCE_SNAPSHOT_LAG` / `city`：signed-error corr `0.418940103684327`；model/B3 better units `5/35/0`。
- `registered` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG` / `origin`：signed-error corr `0.8292992983294296`；model/B3 better units `5/6/0`。
- `registered` / `F2_B3_RESIDUAL` / `R1_REFERENCE_SNAPSHOT_LAG` / `city`：signed-error corr `0.9403852112357481`；model/B3 better units `9/31/0`。
- `registered` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `origin`：signed-error corr `0.37118342894933587`；model/B3 better units `0/11/0`。
- `registered` / `F1_DIRECT` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `city`：signed-error corr `0.4697047461668352`；model/B3 better units `4/36/0`。
- `registered` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `origin`：signed-error corr `0.8676873478025733`；model/B3 better units `5/6/0`。
- `registered` / `F2_B3_RESIDUAL` / `R2_CONSERVATIVE_PUBLICATION_STRESS` / `city`：signed-error corr `0.9713973250048722`；model/B3 better units `6/34/0`。

## Deviations and explicit boundaries

- 未修改 requirements.txt、Plan04/04B/05 artifacts、CURRENT_LEARNING_CHECKPOINT.md 或其他 worker 文件。
- 未运行 Optuna、Bayesian optimization、random search 或 large grid search；未根据 outer labels 选 config。
- 未创建 city routing/gating/ensemble/blend；未生成 2023 prediction/submission；未声明 S5 完成；未进入 S6。
- XGBoost 使用固定 CPU 单线程参数；本环境未强制额外 GPU/外部 deterministic plugin。

## Worker artifacts

- `scripts/build_s5_plan06c_xgboost_screening.py`
- `reports/S5_HANDOFFS/S5-Plan06C/oof_predictions.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/aggregate_metrics.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/per_origin_metrics.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/configuration_selection.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/inner_selection_evidence.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/city_model_error.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/missingness_diagnostics.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/feature_importance.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/model_summary.json`
- `reports/S5_HANDOFFS/S5-Plan06C/scenario_comparison.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/framing_comparison.csv`
- `reports/S5_HANDOFFS/S5-Plan06C/error_complementarity.csv`
