# S5-Plan05 Handoff — 强模型并行 Wave 启动与统一实验合同

## Result

- Plan ID：`S5-Plan05`
- Stage：`S5_EVIDENCE_DRIVEN_ITERATION`
- Plan04B ff-only integration：PASS
- Integrated Plan04B SHA：`58cb1434b7c543b8e5ea5e0ae67b2f93008e50c5`
- Wave base SHA：`1f97f67488cd7aec120ef8b9fa04407dc13d54ba`
- No competing model was trained by this Plan.
- S5 未声明完成，S6 未进入。

## Plan04B integration evidence

- `origin/main` 在集成前为 `fac7ed9bf11459bc5d7c6772857683417bf70eac`。
- Plan04B remote worker 为 `58cb1434b7c543b8e5ea5e0ae67b2f93008e50c5`。
- merge-base 与原 main 完全一致；ahead/behind 为 `0/4`。
- Plan04B diff 仅包含其 runner 与 `reports/S5_HANDOFFS/S5-Plan04B/*` 的 10 个文件。
- main 已 fast-forward 并正常推送；当前 local main 与 `origin/main` 一致。

## Environment freeze

实际项目 Python 环境已安装并通过 import smoke test：

- Python `3.14.6`
- numpy `2.5.3`
- pandas `3.0.6`
- scikit-learn `1.9.1`
- LightGBM `4.7.0`
- CatBoost `1.2.10`
- XGBoost `3.4.1`

环境记录见 `environment_versions.json`。未添加 GPU-only package，未训练 estimator。

## Frozen common contract

- Feature contract：`S5-Plan04-FULL-MULTISOURCE-V1`
- Replay contract：`S5-Plan04B-TRAINING-REPLAY-V1`
- Formal matrix：1760 rows，144 features，outer origins 2011–2021，40 cities。
- Replay matrix：2320 rows，pseudo-target years 2004–2021。
- Outer training：`pseudo_target_year <= outer_origin_year - 2`。
- Inner eligibility：training rows >= 40、validation rows >= 20、`boundary_pass=PASS`。
- Inner eligible folds：264 / 484；44 个 outer fit 中 36 个至少有 3 个 eligible folds，8 个使用 worker 预注册 conservative default fallback。
- Framings：`F1_DIRECT` 与 `F2_B3_RESIDUAL` 均必须评估，共享完全相同的 replay rows、时间切分和特征契约。
- Primary metric：pooled city-year MSE；registered target、R2、macro-origin、per-origin、latest 2021、RMSE、MAE、bias 与 city concentration 均为 mandatory evidence。
- B3 frozen pooled MSE：resident `872.452651515143`；registered `225.421212121213`。
- OOF schema：`reports/S5_HANDOFFS/S5-Plan04/oof_prediction_contract.csv`。

完整机器合同见 `strong_model_wave_contract.json`。

## Shared-file governance

后续 Plan06A/B/C/D worker 只能从该 Wave base 开分支，并只新增自己的 runner 与 handoff 目录：

- 不修改 `requirements.txt`；
- 不修改 Plan04/Plan04B/Plan05 artifacts；
- 不修改 checkpoint；
- 不修改其他 worker 文件；
- 不在 worker 内改变 144 特征、replay rows、validation geometry 或 OOF 必需列。

## Forbidden actions not run

- 未训练 LightGBM、CatBoost、XGBoost、ExtraTrees 或任何控制模型。
- 未运行 Optuna、Bayesian optimization、random search 或大规模 grid search。
- 未生成 2023 预测或 submission。
- 未进行 city routing、ensemble、blend 或 gating。
- 未修改 `CURRENT_LEARNING_CHECKPOINT.md`。

## Recommendation to Main / Lab

Wave contract 已冻结。下一边界是从返回的精确 Wave base SHA 创建四个独立 Plan：

- `S5-Plan06A` — LightGBM
- `S5-Plan06B` — CatBoost
- `S5-Plan06C` — XGBoost
- `S5-Plan06D` — ExtraTrees + regularized linear control

各 worker 必须共享本合同，并独立返回 `family_signal_found`、`family_competitive_but_unstable` 或 `family_unsupported`，不得自行宣布全局 winner、S5 完成或进入 S6。
