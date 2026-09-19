# Project Context

## Purpose

项目目标是为 40 个城市的人口相关数据建立可追溯的 source-level canonical layer，并在其上保留可由 Main 审阅的 S2 Validation Architecture、S3 Baseline Ladder、已授权的 S4 Plan01/02/03 feature/ML evidence 与 Wave A 单 source screening evidence。当前交付对象仍不是最终预测或提交。

## Ownership and layers

- `data/raw/`：7 个合法 XLSX 原始工作簿和任务参考 PDF。原始内容只读。
- `data/normalized/`：7 个审计脚本重建的 14 张逻辑 canonical CSV，以及员工派生行 provenance、repair 和 density regime 报告。
- `data/quarantine/`：确认无效或冲突的值，保留来源上下文；`source/test.xls` 的来源关系仍未解析，因此不进入 canonical。
- `scripts/`：7 个领域审计脚本、统一 S1 runner，以及 S2 Plan01–05、S3 Plan01–03、S4 Plan01–03 和 S4 Wave A Plan04A/B/C 的 focused evidence utilities。
- `docs/`：当前上下文、执行指令、数据契约、质量登记、文件地图和 Main handoff。
- `reports/S1_FINAL_REVIEW_PACK.zip`：唯一保留的当前 S1 审阅包；历史 Plan01-04 报告不再作为当前事实源。
- `reports/S2_HANDOFFS/S2-Plan01/`：S2 前一轮历史 information-set feasibility matrix 与 handoff。
- `reports/S2_HANDOFFS/S2-Plan02/`：当前 S2 source-specific forecast geometry、replay freshness matrix 与 handoff；不包含正式模型验证或提交结果。
- `reports/S2_HANDOFFS/S2-Plan03/`：当前 S2 target/metric contract investigation、年龄结构一致性证据与 evaluator sanity artifacts；常住人口作为主代理、户籍人口作为敏感性目标，尚未执行正式模型评分。
- `reports/S2_HANDOFFS/S2-Plan04/`：官方人口 source reconciliation、S1 contract refresh 与 Plan01–03 population-derived before/after delta；历史 Plan01–03 handoff 保持不变。
- `reports/S2_HANDOFFS/S2-Plan05/`：single latest holdout、multi-origin expanding backtest、leakage、pooled/macro aggregation、稳定性与 architecture recommendation；不包含正式 ML 模型、最终 2023 预测或 submission。
- `reports/S3_HANDOFFS/S3-Plan01/`：B0 persistence、B1 recent two-point trend、B2 expanding per-city linear trend 的正式 baseline evidence、逐 origin metrics、pairwise comparison、稳定性、城市改善集中度与 leakage audit。
- `reports/S3_HANDOFFS/S3-Plan02/`：B0–B4 fixed ladder、global/city prequential selector、oracle upper-bound diagnostics、城市偏好稳定性、late-origin、slope mechanism、集中度与 selector leakage evidence；不包含 2023 预测或 submission。
- `reports/S3_HANDOFFS/S3-Plan03/`：B0–B7 fixed ladder、pooled city-fixed-effect additive trend、recent log-growth、local/pooled slope、growth、time-regime、leave-one-origin/city、concentration、leakage、compatibility 与 future-ML hurdle evidence；不包含 formal ML、2023 预测或 submission。
- `reports/S4_HANDOFFS/S4-Plan01/`：target-history-only `TH_CORE` / `TH_SHAPE` feature manifest、outer-origin Ridge delta evidence、B3 compatibility、aggregate/per-origin/latest、ablation、concentration、leave-one-origin/city、leakage 与 R1/R2 equivalence artifacts；不包含 covariates、tuning、ensemble、2023 预测或 submission。
- `reports/S4_HANDOFFS/S4-Plan02/`：城镇化率 `URB_LEVEL_V1` / `URB_DYNAMICS_V1` 的 B3 residual correction evidence，包含 D2 global residual bias control、M2A/M2B Ridge、R1/R2 freshness、training geometry、missingness、aggregate/per-origin/latest、ablation、concentration、leave-one-origin/city、coefficient/correlation、leakage 与 B3 compatibility artifacts；结论为 urbanization unsupported，不包含 2023 预测或 submission。
- `reports/S4_HANDOFFS/S4-Plan03/`：低容量 `HistGradientBoostingRegressor` 的 B3 residual correction evidence，严格复用 Plan01 `TH_CORE/TH_SHAPE` 与 Plan02 `URB` semantics，包含 M3A/M3B/M3C、固定配置、三层 feature ablation、R1/R2 equivalence/sensitivity、training geometry、missingness、aggregate/per-origin/latest、concentration、leave-one-origin/city、leakage 与 B3 compatibility artifacts；结论为 Ridge 线性容量不足不是 Plan01/02 失败的主要解释，当前 feature 组合仍不支持 material ML 增益。
- `reports/S4_HANDOFFS/S4-Plan04A/`：人口密度 `DENSITY_RECENT_V1` 的 B3 residual screening evidence；R1/R2 source-cap、regime-crossing suppression、quarantine guard、aggregate/per-origin 与 boundary audit 均已归档，结论为 density unsupported。
- `reports/S4_HANDOFFS/S4-Plan04B/`：工资 `WAGE_RECENT_V1` 的 B3 residual screening evidence；R1/R2 wage cap 等价性、invalidated-key guard、aggregate/per-origin 与 boundary audit 均已归档，结论为 wage unsupported。
- `reports/S4_HANDOFFS/S4-Plan04C/`：从业人员数 `EMPLOYMENT_RECENT_V1` 的 B3 residual screening evidence；R1/R2 boundary、derived/quarantine provenance、aggregate/per-origin 与 source audit 均已归档，结论为 employment unsupported。
- `reports/S4_HANDOFFS/S4-WaveA/`：三个单 source screening branch 的整合摘要；Wave A 已整合，S4 仍进行中，尚未进入 S5。

## S4 Wave B and final evidence

- reports/S4_HANDOFFS/S4-Plan05A/：城市级 B3 error-memory Ridge residual screening；T-2 residual-history boundary、B3 compatibility 与 R1/R2 equivalence 均已归档，结论为 error_memory_unsupported。
- reports/S4_HANDOFFS/S4-Plan05B/：cross-city relative percentile Ridge residual screening；同一 pseudo-origin 横截面、T-2、R1/R2 equality 与 preprocessing boundary 均已归档，结论为 unsupported。
- reports/S4_HANDOFFS/S4-Plan05C/：density、urbanization、wage、employment 四 source 的 compact RidgeCV residual screening；source-specific caps、quarantine/provenance、alpha 与 coefficient diagnostics 均已归档，结论为 multisource_joint_unsupported。
- reports/S4_HANDOFFS/S4-WaveB/：Wave B 三分支整合摘要和 S4 机器可读 final summary。
- S4_FEATURE_ML_MODELING_FINAL_REPORT.md：S4 Feature/ML 阶段最终 evidence report。B3 仍是 strongest fixed model；evidence package 已准备 Main review，但 s4_complete=false、未进入 S5，也未生成 2023 forecast/submission。Stage completion 仍由 Main 判定。

## Source facts

当前 canonical 层包含人口规模、人口密度、城镇化率、失业率、从业人员数、三产业就业人数、平均工资、年龄结构和生活水平六类表。所有 14 张逻辑表保留城市/年份键的唯一性。

当前人口 source 已吸收官方 `人口规模.xlsx`：SHA256 为 `9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0`；本次只改变人口 raw workbook，并由 S1 pipeline 生成 `city40-2001` 常住人口 284、户籍人口缺失的 canonical 记录。

Plan05 使用上述刷新后的 canonical population：共同严格 origin 为 `2011–2021`，resident 与 registered 的最新可评分 origin 均为 `2021`。本轮只使用 persistence 与两点线性趋势两个固定 target-history-only naive predictor；每个 origin 的 target 输入最晚为 `T-2`。R1/R2 的 source-specific information-set contract 被逐 origin 保留，但因本轮不使用 covariates，两个 scenario 的数值 residual 完全相同。

Plan05 的内部验证架构建议是 hybrid：multi-origin expanding 作为稳定性主证据，latest holdout 作为最终任务接近性诊断；pooled city-year MSE 作为任务对齐的 primary aggregate，macro-origin MSE 作为稳定性敏感性。该建议不改变 target contract，不把 R1/R2 变成 publication facts，也不授权 S3。

S3 Plan01 在不修改 S2 contract 的前提下重新运行 B0/B1，并新增唯一的 B2 expanding per-city linear trend。B1 在 resident 与 registered 的 pooled MSE 上均优于 B0 和 B2；B2 仍优于 B0，但 resident 的 2021 latest 与 origin stability 有局部优势，registered 则在 2021 和跨 origin 波动上较差。该结果是 baseline evidence，不是 S3 champion 或最终预测。

S3 Plan02 在不修改 S2 contract 的前提下只增加预注册 B3 recent-3-point OLS、B4 recent-5-point OLS，并实现严格 prequential 的 global/city selector。B3 在 resident 与 registered 的 pooled/macro MSE 上均成为当前 strongest fixed baseline；resident 的 2021 latest 与 B1 有冲突，registered 则略优于 B1。city selector 对 registered 有小幅 aggregate 改善，但 resident aggregate 变差；oracle 仅作为 NON-DEPLOYABLE upper-bound diagnostic。该结果不是 S3 COMPLETE 或最终预测授权。

S3 Plan03 在不修改 S2 contract 的前提下完整重跑 B0–B4，并新增 B5 pooled expanding city-fixed-effect additive trend、B6 pooled recent-3 city-fixed-effect additive trend、B7 recent-3 per-city log-linear growth。B3 仍是 strongest deployable fixed baseline：resident pooled MSE `872.4527`、registered `225.4212`；B7 接近但略差，B5/B6 共享绝对斜率明显劣化。Plan03 同时落盘 local/pooled slope、growth、time-regime、leave-one-origin/city、concentration 与 ML hurdle evidence；所有 leakage、R1/R2、S2、Plan01、Plan02 compatibility checks PASS，candidate sufficiency 为 yes。随后 S3 已在当前 checkpoint 中完成并授权进入 S4。

S4 Plan01 在不修改 S2 target/metric/origin contract 的前提下，新增 target-history-only 的 `M1A_RIDGE_DELTA_TH_CORE` 与 `M1B_RIDGE_DELTA_TH_SHAPE`。两者均使用每个 outer origin 内 fit 的 median imputation、missing indicators、standardization 与 `Ridge(alpha=1.0)`，不使用 covariates、city identity、tuning 或 ensemble。B3 row/metric compatibility、full `40 × 11` coverage、leakage 与 R1/R2 equivalence 均 PASS；但 resident pooled MSE 分别为 `910.3499`、`938.6950`，均未超过 B3 的 `872.4527`，registered sensitivity 也发生同方向 reversal。因此当前结论为 `S4-Plan01 evidence complete`，不是整个 S4 完成，也未进入 S5。

S4 Plan02 在不修改 S2 target/metric/origin contract 的前提下，新增只使用 canonical `urbanization` 的 D2 global residual bias control、M2A level Ridge 与 M2B level+dynamics Ridge。R1 复用 S2 lag 1 cap，R2 复用共享规则得到的 lag 2 cap；B3 row/metric compatibility、full `40 × 11` validation coverage、population `T-2`、urbanization boundary、leakage 与 source-year audits 均 PASS。B3 仍为 strongest fixed baseline：resident pooled MSE `872.4527`、registered `225.4212`；D2 与 M2A/M2B 均未改善 pooled aggregate，registered latest 也全部恶化。Plan02 结论为 `urbanization unsupported`，仍不是整个 S4 完成，也未进入 S5。

S4 Plan03 在不修改 S2 target/metric/origin contract、且不引入新 source 的前提下，新增唯一的低容量 `HistGradientBoostingRegressor` 家族，分别运行 M3A `TH_CORE`、M3B `TH_CORE + TH_SHAPE`、M3C `TH_CORE + TH_SHAPE + URB` 的 B3 residual correction。固定配置为 `loss=squared_error`、`learning_rate=0.05`、`max_iter=100`、`max_depth=2`、`min_samples_leaf=20`、`l2_regularization=1.0`、`early_stopping=False`、`random_state=0`；每个 target/scenario/origin/model 独立 fit imputer 与 GBDT。B3、TH semantics、R1/R2、population T-2、urbanization cap、preprocessing boundary、leakage 与 full coverage 均 PASS，但 M3A/M3B/M3C 都未超过 B3，且 M3C 在 R1→R2 下明显 reversal。Plan03 结论为 `low-capacity nonlinear target-history and conditional urbanization signal unsupported`，仍不是整个 S4 完成，也未进入 S5。

两个需要特别防止回退的事实：

1. `overall_income` 是 `source_plus_invalidation`，repair count 为 0，32 个值被隔离；`urban_income` 的 repair count 为 2，来自 `ctiy22→city22` 和 `ctiy31→city31` 标签修复。
2. 年龄结构的域是 40 城市 × 3 个 benchmark 年份，共 120 个预期键；实际 118 个，唯一缺失键是 `city25-2000` 和 `city26-2000`。2001–2009、2011–2019 不属于该表域。
