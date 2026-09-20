# Codex 工作边界

## 当前任务

本仓库保留 S1 最终冻结的数据基础，并维护经 Main/Lab 授权的 S2 Validation Architecture、S3 Baseline Ladder、S4 evidence、S5 final evidence 与 S6-Plan01 Final Solution Contract。当前允许的工作是：维护原始数据目录、运行 S1 审计、验证历史证据、执行 S6-Plan01 合同审计，并为后续 clean raw rebuild 保持可复现边界。

## 强制边界

- 可以执行经 Main/Lab 授权的 S2 Validation Architecture 计划，包括历史 information-set / pseudo-test 可行性审查；不得因此修改 S1 canonical/data-quality contract。
- 不把缺失、隔离值、未知真值或源派生行静默填补为 0 或插值结果。
- 不把 `overall_income` 与 `urban_income` 自动当作两个独立可信信号。
- 年龄结构只接受 benchmark years `{2000, 2010, 2020}`；非 benchmark 年份不是该表的缺失键。
- `test.xls` 保留在 `data/quarantine/source/`，不得进入 canonical pipeline。
- `test.xls` 内容不得被打开、解析、哈希或用于推导 label、feature、city universe、submission schema 或 final prediction；只能保留文件本体元数据边界。
- 不修改 7 个合法 XLSX 原始文件；变更前必须记录哈希并在重建后复核。
- S3 Plan01/Plan02/Plan03 与 S4 Plan01/Plan02/Plan03 只能在冻结 S2 contract 内执行少量、可解释、可复核的 evidence。S4 Plan01 限定为 target-history-only `TH_CORE`/`TH_SHAPE`、outer-origin `Ridge(alpha=1.0)`、训练 fold 内 preprocessing、R1/R2 equivalence 与完整稳定性审计；S4 Plan02 限定为 canonical `urbanization` 的 D2 residual-bias control、M2A level、M2B level+dynamics，复用 S2 source-specific cap；S4 Plan03 限定为固定低容量 `HistGradientBoostingRegressor` 的 M3A/M3B/M3C B3 residual correction，严格复用 Plan01/02 feature semantics。三者均不得加入其他 covariates、city identity、tuning、model sweep、ensemble、预测或 submission。S4 Plan01/02/03 evidence complete 均不等于整个 S4 完成，也不授权 S5。

## S4 final boundary

S4 Plan01–05C 的 evidence 已归档；S5 Plan01–06H 的真实证据、recovery/coverage 边界和 final candidate 已整合，S5 已由 Main 判定完成。当前 S6-Plan01 已冻结 resident-primary `F4_LOG_GROWTH_FROM_LATEST`、B3 reference/fallback、Plan04/04B 144-feature contract、现有 R1/R2 mapping 和 T-2 final-training geometry。官方 target 仍只有“总人口”表述，resident/registered mapping 与单位未完全证明，因此 submission 继续 blocked。

## S6-Plan01 final boundary

- 只能使用 `reports/S6_HANDOFFS/S6-Plan01/` 中的 contract、audit、lineage、environment 和 rebuild manifest。
- F4 label 为 `log(actual_target_value / pop_latest_level)`，reconstruction 为 `pop_latest_level * exp(raw_prediction)`；不 clipping、不新增 feature、不调参、不重新选择模型。
- current 2023 information set 只能映射到已经冻结的 Plan02 R1/R2；不能新造 source policy 或使用 2022/2023 target truth。
- CatBoost 只允许 historical Plan06E 的三组 pre-registered configs、seed `20260919`、native `city_id`、raw NaN 和 inner selection/fallback rule；不允许 per-city routing/gating/ensemble。
- 不生成 2023 prediction、submission，不修改 `CURRENT_LEARNING_CHECKPOINT.md`，不进入 S7；正式提交必须等待 Main 处理 target/unit boundary。后续 final pipeline 必须保留 resident primary 与 registered alternative 两条独立路径，分别训练、分别生成候选，不得平均或混合。

## 运行与交付

从仓库根目录运行 `python scripts/run_s1.py` 可重建 S1；S2-S5 计划使用各自 handoff 中记录的 focused utility；S6-Plan01 使用 `python scripts/build_s6_plan01_final_solution_contract_audit.py`。所有产物必须区分 observed fact、assumed availability、unavailable、irregular domain、quality/provenance boundary 和 pseudo-origin information boundary。

不要提交 `.venv`、缓存、临时文件、认证状态、绝对机器路径或秘密。GitHub 发布需要另行确认远端身份和权限；本地 Git 初始化不等于已推送或已发布。
