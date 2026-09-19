# S2-Plan01 Handoff｜Historical Information-Set Feasibility Matrix

## Result

本轮完成 S2-Plan01 的 source-level temporal inventory、availability policy 对比和 historical pseudo-target feasibility matrix。没有进入正式 baseline、模型训练、回测评分、预测或 submission。

## Starting repository

- Repository: `MyKr-YSteinsK/city-population-competition`
- Branch: `main`
- Starting commit: `f14b9f2a597bcc3a52d0067367ab69ad435e6217`
- Starting working tree: clean
- 起始前已执行 `git fetch origin main` 和 `git pull --ff-only origin main`，使本地事实与远端 `main` 对齐。

## Actual changes

- 新增 [`scripts/build_s2_plan01_matrix.py`](../../../scripts/build_s2_plan01_matrix.py)，只读取 S1 contract 和 canonical CSV，使用 Python 标准库生成可复核矩阵。
- 新增本目录下的 14-source temporal inventory、40-city coverage、policy definitions、pseudo-target matrix、candidate summary 和 JSON summary。
- 最小修正 [`docs/CODEX_INSTRUCTIONS.md`](../../../docs/CODEX_INSTRUCTIONS.md)，解除已获授权的 S2 阻塞，同时保留 S1 contract 不变和 S3+ 禁止边界。
- README 仅更新当前 S2 授权入口；`docs/PROJECT_CONTEXT.md` 和 `docs/PROJECT_FILE_MAP.md` 补入当前 checkpoint 与 S2 handoff 路径，不重写 S1 数据契约。

## Commands actually executed

```text
git fetch origin main
git pull --ff-only origin main
python scripts/build_s2_plan01_matrix.py
python -c "...AST parse of scripts/build_s2_plan01_matrix.py..."
```

实际 runner 输出：

```text
S2_PLAN01_MATRIX_PASS
source_inventory_rows=14
source_city_coverage_rows=560
matrix_rows=644
candidate_summary_rows=46
candidate_years_by_policy=P1/P2: 2009-2021
target_label_unavailable_years=[2022, 2023]
test_xls_used=False
```

## Target interpretation used in this matrix

S1 population source 同时包含常住人口和户籍人口。本轮以 `常住人口（万人）` 作为 primary target-label proxy，仅用于结构性可行性筛选；这不是最终 validation contract，也没有宣称它已经解决赛题 target 口径冲突。`户籍人口（万人）` 仍在 source inventory 中保留为同源 secondary target indicator。

结构性筛选规则是：target label 城市覆盖至少 90%，至少 3 个历史 target labels 的城市覆盖至少 90%，并且 6 个 core annual sources 各自至少有 90% 城市在 policy cutoff 前存在非空 canonical value。这些阈值只用于形成 candidate set，不是最终模型验证门槛。

## Availability policies

### P1 — source-specific as-of T-1

对 pseudo-target `T`，每个 source 只允许使用年份 `<= T-1` 的 canonical non-empty values；不同 source 的 observed coverage、最晚年份、城市缺口和不规则域都保留。它是最接近简单历史 replay、同时保留 freshness asymmetry 的候选假设，但 observed max year 仍不是 publication date 证据。

### P2 — conservative as-of T-2

对 pseudo-target `T`，所有 source 只允许使用年份 `<= T-2` 的 canonical non-empty values。它比 P1 多一年的统一 freshness stress，用来检查候选 origin 对 availability 假设是否敏感；它同样不声称真实发布时间是 `T-2`。

两种 policy 的机器定义见 `policy_definitions.csv`；每条 matrix 行同时记录 policy cutoff、实际 source cap、城市/值计数和缺口类别。

## Candidate pseudo-target years

### Candidate set

在 P1 和 P2 下，结构性 candidate set 均为：

```text
2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
2017, 2018, 2019, 2020, 2021
```

2009 有 40/40 target labels、37/40 城市至少有 3 个 prior target labels，且两种 policy 下 6 个 core annual sources 均达到 90% 城市覆盖。2010–2021 为 40/40 target labels、40/40 prior-history coverage，并满足同一 core screen。

### Excluded or conditional years

- 2001–2006：target label coverage 分别为 21/40、11/40、11/40、11/40、11/40、27/40，低于 90%；同时 prior target history 也不足。
- 2007：target label 为 40/40，但仅 11/40 城市有至少 3 个 prior labels；P1 的 urbanization/sector、P2 的 density/urbanization/sector 还未达到 core screen。
- 2008：target label 为 40/40，但仅 11/40 城市有至少 3 个 prior labels；P1 因 history 排除，P2 另有 urbanization/sector coverage gap。
- 2022、2023：矩阵保留它们用于观察 information-set 边界，但 population `常住人口（万人）` 没有 target label，不能形成 retrospective scoreable pseudo-test。

## Evidence interpretation

Working hypothesis 得到部分支持：当前材料不足以证明一个唯一可信的 historical information set；P1/P2 是非等价且均需声明的 assumptions。两种 policy 对 core candidate years 的集合恰好相同，但 cutoff、age benchmark 可用性、收入/消费覆盖和其他 optional source 的可用城市数不同，因此不能把“candidate set 稳定”解释成 availability 已被证明。

矩阵还显示：

- age 只按 `{2000, 2010, 2020}` benchmark 解释；早期 T 只能看到 2000 benchmark，不能伪造成逐年序列。
- income/consumption source 在较早 T 不可用或仅部分城市可用；source absence 没有被转换为 zero。
- employees 的 derived rows 在每条矩阵行中单独计数并保留 provenance；不能把它们当作独立 raw evidence。
- invalidated/quarantined keys 没有被重新计入 non-empty observation；`test.xls` 没有被读取。
- overall/urban income 的 shift/dependence warning 仍逐 source 保留，没有因矩阵生成而被宣布解决。

## S1 boundary and verification

- 实际读取了 `docs/S1_DATA_CONTRACT.md`、`docs/s1_data_contract.csv`、`docs/s1_quality_issue_registry.csv`、`docs/S1_FINAL_STATE.md` 和 `CURRENT_LEARNING_CHECKPOINT.md`。
- runner 读取并验证 14 行 S1 contract、14 张 canonical 表的 `(city, year)` key uniqueness，以及 age benchmark domain；没有改写任何 S1 data CSV。
- `data/quarantine/source/test.xls` 仅做存在性边界检查；raw `test.xls` 不存在于 canonical 输入路径，且 `test_xls_used=False`。
- local `.venv` 已在前一 S1 交付中移出仓库，本轮没有重新运行依赖 pandas 的 S1 runner；本轮只执行了与矩阵直接相关的标准库验证，未把未执行的 S1 full rebuild 宣称为通过。

## Failures / deviations

- 本地起始 ref 曾落后远端 `main` 两个文档提交；已用 `pull --ff-only` 快进到计划要求的实际起始 commit，没有覆盖或重写历史。
- 没有发现影响本轮矩阵的 S1 contract 与 canonical CSV 不一致。
- 没有使用 `test.xls` 的任何 target-like 内容，也没有运行正式 validation 或 model comparison。

## Remaining uncertainty

- 各 source 的真实 publication timing 和 2023 forecast-origin availability 仍为 unknown。
- 赛题 target 的常住/户籍精确口径和 PDF 中 MSE/RMSE 表述冲突仍未在本轮解决。
- 是否将 employee derived rows 纳入未来 information set、以及 income shift 下的独立信号边界，仍需下一轮验证架构判断。
- P1/P2 只回答结构可行性，不证明任何 policy 下的模型分数无偏或可比较。

## Artifact locations

- `source_temporal_inventory.csv`：14 个 logical source 的角色、域、覆盖、状态和 warning。
- `source_city_coverage.csv`：40 城市逐 source coverage 摘要。
- `policy_definitions.csv`：P1/P2 的明确假设。
- `pseudo_target_source_policy_matrix.csv`：23 个 pseudo-target years × 14 sources × 2 policies 的机器矩阵。
- `candidate_origin_summary.csv`：candidate/exclusion 分类及原因。
- `matrix_summary.json`：行数、阈值、候选年份和 test.xls 边界的机器摘要。

## Resulting commit

本轮 artifact/implementation commit SHA：`2d5620fd021efa880462a8002fcd208af5f1f6f5`。随后仅为记录最终交付状态追加 metadata commit；最终仓库 HEAD 也在 TASK_RESULT 中明确报告。

## Interpretation Requirement

基于本轮 evidence，下一轮最值得验证的问题是：**在不把 observed max year 当作 publication fact 的前提下，P1 的 source-specific freshness 与 P2 的额外一年度 lag 是否会对同一组 2009–2021 candidate origins 产生实质不同的可用特征集合和后续验证结论？**

本轮不执行该问题对应的下一轮实验。
