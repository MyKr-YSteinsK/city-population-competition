# S1 Final State

## Freeze candidate

当前仓库是 S1 minimal repository freeze candidate，等待 Main review。规范化数据、隔离值和 provenance 均由 `scripts/run_s1.py` 调用的 7 个领域脚本从原始工作簿重建；runner 只做当前断言与审阅包交付，不承担历史探索报告。

## Reproduced evidence

- 14 张逻辑 canonical 表，城市/年份键无重复。
- 员工派生 provenance 91 行；独立员工—产业就业关系为 484/484，绝对差不超过 1。
- 密度隔离 13 行、密度 regime 报告 13 行、工资隔离 1 行、年龄隔离 6 行、整体收入隔离 32 行。
- 人口 canonical 层零哨兵为 0；`city21` 的 2022 年整体消费支出为 44607。
- 原始 XLSX/PDF 哈希在重建前后保持一致；canonical、quarantine、provenance、regime CSV 在重建前后保持一致。

## Boundaries

年龄结构按 `{2000, 2010, 2020}` 解释，预期 120 键、实际 118 键，缺失只报 `city25-2000` 与 `city26-2000`。整体收入的 repair count 为 0；城镇收入的标签 repair count 为 2。

## Remaining uncertainty

隔离值的真实值、city3 收入的时间错位语义、源表发布时间/可用时间和部分不对称覆盖仍未知。它们被显式登记，不在 S1 静默修复。
