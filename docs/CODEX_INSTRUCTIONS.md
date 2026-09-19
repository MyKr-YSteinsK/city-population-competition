# Codex 工作边界

## 当前任务

本仓库保留 S1 最终冻结的数据基础，并维护经 Main/Lab 授权的 S2 Validation Architecture 与 S3 Baseline Ladder 证据。当前允许的工作是：维护原始数据目录、运行 7 个领域审计脚本、验证规范化/隔离/溯源输出、维护数据契约、质量登记，以及在冻结 S2 contract 内执行当前 S3 baseline 计划。

## 强制边界

- 可以执行经 Main/Lab 授权的 S2 Validation Architecture 计划，包括历史 information-set / pseudo-test 可行性审查；不得因此修改 S1 canonical/data-quality contract。
- 不把缺失、隔离值、未知真值或源派生行静默填补为 0 或插值结果。
- 不把 `overall_income` 与 `urban_income` 自动当作两个独立可信信号。
- 年龄结构只接受 benchmark years `{2000, 2010, 2020}`；非 benchmark 年份不是该表的缺失键。
- `test.xls` 保留在 `data/quarantine/source/`，不得进入 canonical pipeline。
- 不修改 7 个合法 XLSX 原始文件；变更前必须记录哈希并在重建后复核。
- 当前 S3 Plan01/Plan02 只能在冻结 S2 contract 内执行少量、可解释、target-history-only baseline、预注册 recent-window 与 prequential selector 证据；不得进入 S4，不得擅自扩大为 covariate feature engineering、复杂 ML、tuning、ensemble、预测、排行榜或 submission。

## 运行与交付

从仓库根目录运行 `python scripts/run_s1.py` 可重建 S1；S2/S3 计划使用各自 handoff 中记录的 focused utility。S2/S3 产物必须区分 observed fact、assumed availability、unavailable、irregular domain、quality/provenance boundary 和 pseudo-origin information boundary。

不要提交 `.venv`、缓存、临时文件、认证状态、绝对机器路径或秘密。GitHub 发布需要另行确认远端身份和权限；本地 Git 初始化不等于已推送或已发布。
