# Project Context

## Purpose

项目目标是为 40 个城市的人口相关数据建立可追溯的 source-level canonical layer。当前交付对象不是预测模型，而是能被 Main 继续审阅的 S1 数据基础。

## Ownership and layers

- `data/raw/`：7 个合法 XLSX 原始工作簿和任务参考 PDF。原始内容只读。
- `data/normalized/`：7 个审计脚本重建的 14 张逻辑 canonical CSV，以及员工派生行 provenance、repair 和 density regime 报告。
- `data/quarantine/`：确认无效或冲突的值，保留来源上下文；`source/test.xls` 的来源关系仍未解析，因此不进入 canonical。
- `scripts/`：7 个领域审计脚本、统一 S1 runner，以及 S2 Plan01–04 的 focused evidence utilities。
- `docs/`：当前上下文、执行指令、数据契约、质量登记、文件地图和 Main handoff。
- `reports/S1_FINAL_REVIEW_PACK.zip`：唯一保留的当前 S1 审阅包；历史 Plan01-04 报告不再作为当前事实源。
- `reports/S2_HANDOFFS/S2-Plan01/`：S2 前一轮历史 information-set feasibility matrix 与 handoff。
- `reports/S2_HANDOFFS/S2-Plan02/`：当前 S2 source-specific forecast geometry、replay freshness matrix 与 handoff；不包含正式模型验证或提交结果。
- `reports/S2_HANDOFFS/S2-Plan03/`：当前 S2 target/metric contract investigation、年龄结构一致性证据与 evaluator sanity artifacts；常住人口作为主代理、户籍人口作为敏感性目标，尚未执行正式模型评分。
- `reports/S2_HANDOFFS/S2-Plan04/`：官方人口 source reconciliation、S1 contract refresh 与 Plan01–03 population-derived before/after delta；历史 Plan01–03 handoff 保持不变。

## Source facts

当前 canonical 层包含人口规模、人口密度、城镇化率、失业率、从业人员数、三产业就业人数、平均工资、年龄结构和生活水平六类表。所有 14 张逻辑表保留城市/年份键的唯一性。

当前人口 source 已吸收官方 `人口规模.xlsx`：SHA256 为 `9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0`；本次只改变人口 raw workbook，并由 S1 pipeline 生成 `city40-2001` 常住人口 284、户籍人口缺失的 canonical 记录。

两个需要特别防止回退的事实：

1. `overall_income` 是 `source_plus_invalidation`，repair count 为 0，32 个值被隔离；`urban_income` 的 repair count 为 2，来自 `ctiy22→city22` 和 `ctiy31→city31` 标签修复。
2. 年龄结构的域是 40 城市 × 3 个 benchmark 年份，共 120 个预期键；实际 118 个，唯一缺失键是 `city25-2000` 和 `city26-2000`。2001–2009、2011–2019 不属于该表域。
