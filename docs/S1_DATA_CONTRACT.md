# S1 Data Contract

机器可读的完整登记在 `docs/s1_data_contract.csv`。每行对应一张逻辑 canonical 表，记录原始工作簿/工作表、键和值列、源覆盖、年份域、repair、invalidated、derived、quarantine、provenance 和下游边界。

## State semantics

- `source_preserved`：按源表保留，不代表已证明绝对正确。
- `source_plus_repair`：有明确、可追溯的源内修复；修复报告必须保留。
- `source_plus_invalidation`：值仍在隔离区，canonical 中保持缺失，真实值未知。
- `source_plus_label_repair`：只修复源标签，不改变数值事实。
- `source_plus_remap_quarantine_and_derived`：保留重映射、隔离和显式派生 provenance，后续使用必须区分来源边界。

## Non-negotiable corrections

- `overall_income`：`source_plus_invalidation`，`repair_count=0`，`invalidated_count=32`。
- `urban_income`：`source_plus_label_repair`，`repair_count=2`，即两个城市标签修复。
- `age`：`year_domain=2000;2010;2020`，`expected_key_count=120`，`actual_key_count=118`，缺失键仅 `city25-2000`、`city26-2000`。非 benchmark 年份不进入缺失统计。

## Use rule

canonical 非空值不自动等于绝对真实；quarantine 真值不应回灌；派生行不应被当作独立验证证据；缺失 coverage 不应被当作零。
