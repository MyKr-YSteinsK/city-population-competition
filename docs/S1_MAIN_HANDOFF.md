# S1 Main Handoff

## Result

S1 minimal repository freeze candidate ready for Main review. Main 接收的是可重建的数据契约和清理后的仓库，不是模型结果。

## Start here

1. 阅读 `docs/PROJECT_CONTEXT.md`、`docs/S1_DATA_CONTRACT.md`、`docs/s1_quality_issue_registry.csv`。
2. 运行 `python scripts/run_s1.py`，确认 `S1_REBUILD_PASS`、raw/output hash equality 和 关键断言。
3. 检查 `reports/S1_FINAL_REVIEW_PACK.zip` 的 `MANIFEST.md` 和 机器可读 CSV。

## Accepted S1 facts

- 14 个 canonical logical tables；原始资料为 7 个 XLSX 和 1 个参考 PDF。
- `test.xls` 位于 `data/quarantine/source/test.xls`，来源未解析，不进入 canonical。
- 员工 provenance 91 行；独立员工—产业关系 484/484 within absolute difference 1。
- 密度/员工/产业/工资/年龄/整体收入的隔离计数为 13/80/18/1/6/32。
- 年龄表只按 `{2000, 2010, 2020}` 解释，唯一缺失键为 `city25-2000`、`city26-2000`。
- 整体收入 repair=0、invalidated=32；城镇收入标签 repair=2。

## Main decision requested

请确认以上 S1 数据契约、覆盖边界、隔离边界和派生 provenance 可作为后续工作的唯一 source-level 输入，并明确授权后续是否另开 S2 任务。当前仓库不进入 S2。

## Boundary declaration

No S2 / modeling / validation was entered.
