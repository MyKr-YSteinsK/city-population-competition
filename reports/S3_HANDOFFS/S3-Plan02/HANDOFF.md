# S3-Plan02 Handoff｜Recent-Window Trend + City Heterogeneity

## Result / status

本轮完成 S3 Baseline Ladder 的 S3-Plan02 evidence。状态为
`S3_PLAN02_RECENT_WINDOW_SELECTOR_PASS`。本 Plan 不宣布 S3 COMPLETE，不进入 S4，
也不产生 2023 forecast 或 submission。

核心结论：

- B3 `B3_RECENT_3POINT_OLS_TREND` 是当前两个 target 上 pooled/macro MSE 都最强的 fixed baseline。
- B3 相比 B1 的 resident pooled MSE 改善为 `9.8972%`，registered 改善为 `5.7639%`；registered 的 2021 也略优于 B1，但 resident 的 2021 反而劣于 B1。
- B4 没有稳定超过 B3：resident 仅略优于 B1、但 latest 明显变差；registered 在 pooled 和 latest 都劣于 B1。
- global prequential selector 在 resident 上劣于 B1，在 registered 上改善 `2.5083%`；city-level selector 在 resident 上劣于 B1，在 registered 上改善 `4.7548%`，但收益不能推广到 resident。
- city heterogeneity 的描述性证据存在，但偏好跨 origin 不稳定、跨 target 仅 `12/40` 城市一致；当前不能把后验城市分组当作部署规则。
- oracle 有明显理论 headroom，但两个 oracle 都是 `ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND`，不能进入真实模型排名。

## Starting repository state

- Repository: `city-population-competition`
- Branch: `main`
- Starting local/remote main SHA: `ed006d91174889280451878a2c1a820fb05a95bf`
- Starting working tree: clean
- 当前执行前已 `git fetch origin main`，本地 `main` 与 `origin/main` 一致。
- Canonical population profile: 718 rows, 718 unique city-year keys, resident non-empty 693, registered non-empty 707
- Canonical population SHA256: `7f12e54c5c8d72a20182a117ad26564e6f1892d6f4b9add0dc80135aabb18797`

S2 frozen contract 未修改：common origins 为 `2011–2021`，pseudo-target `T` 的 target
history cutoff 为 `T-2`，resident 为 primary、registered 为 sensitivity，pooled
city-year MSE 为 primary、macro-origin MSE 为 sensitivity，2021 为独立 latest diagnostic。

## Actual changes

- 新增 `scripts/build_s3_plan02_recent_window_and_selector.py`。
- 新增本目录下的 fixed baseline、prequential selector、oracle、city preference、slope、late-origin、concentration 与 leakage artifacts。
- 完整重跑 B0/B1/B2，并新增且仅新增预注册的 B3 3-point OLS、B4 5-point OLS。
- B3 全部记录使用 3 个点；B4 有 136 条记录不足 5 个点，按固定规则回退 B1，无静默扩窗；所有 B3/B4 记录仍可评分。
- selector candidate 固定为 B1/B2/B3/B4；少于 3 个 prior scored origins 时固定回退 B1。selector 只使用严格早于当前 origin 的 pseudo-origin squared errors。
- 更新 `CURRENT_LEARNING_CHECKPOINT.md`、`docs/CODEX_INSTRUCTIONS.md`、`docs/PROJECT_CONTEXT.md`、`docs/PROJECT_FILE_MAP.md`。
- 没有修改 S1 canonical truth、raw/quarantine、S2 contract、S2 handoff 或 S3 Plan01 artifact。
- 没有使用 covariates、复杂 ML、tuning、ensemble、clustering、learned gate、2023 forecast 或 submission。

## Frozen validation contract confirmation

- common strict origins: `2011–2021`，11 origins；每个 target/scenario/predictor 都覆盖 40 个城市。
- fixed predictor 与 selected fixed output 的 history max year 均不超过 `T-2`。
- selector history max origin 严格小于当前 origin；当前 origin residual 与未来 origin residual 均未用于 selector decision。
- R1/R2 保留；由于所有 predictors/selectors 只使用 target history，预测、residual、squared error 数值等价。
- S2 B0/B1 compatibility 与 S3 Plan01 B0/B1/B2 compatibility 均逐行 machine-check。

## Commands actually executed

从 repository root 执行：

    git fetch origin main
    git status --short --branch
    git rev-parse HEAD
    git rev-parse origin/main
    python -m py_compile scripts/build_s3_plan02_recent_window_and_selector.py
    python scripts/build_s3_plan02_recent_window_and_selector.py
    git diff --check

最终 runner output：

    S3_PLAN02_RECENT_WINDOW_SELECTOR_PASS
    starting_commit=ed006d91174889280451878a2c1a820fb05a95bf
    common_origin_domain=2011-2021
    fixed_predictor_count=5
    selector_count=2
    prediction_rows=12320
    per_origin_metric_rows=308
    aggregate_metric_rows=28
    pairwise_summary_rows=52
    oracle_rows=3520
    leakage_audit_all_passed=True
    selector_history_audit_all_passed=True
    r1_r2_equivalence_all_passed=True
    s2_b0_b1_compatibility_all_passed=True
    plan01_b0_b1_b2_compatibility_all_passed=True
    formal_ml_model_used=False
    covariates_used=False
    tuning_run=False
    ensemble_run=False
    clustering_run=False
    learned_gate_run=False
    final_2023_forecast_run=False
    submission_run=False
    test_xls_used=False

执行中曾发现并修复三类 artifact serialization 问题：oracle aggregate-only fields、混合 fixed/selector rows 的选择字段，以及 cold-start JSON 中的 `Infinity`。修复后重新执行完整 runner，最终 artifact 中 aggregate MSE、selector decision 与 JSON `null` 均已写入。

## Phase A｜Fixed baseline ladder

以下为 R1；R2 数值完全一致。MSE 是 primary，score 只是 `1/(1+MSE)` 的同 scope 单调变换。

| Target | Predictor | Pooled MSE | Pooled MAE | Bias | Macro MSE | 2021 MSE | 2021 rank | MSE std across origins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| resident | B0 | 1926.1682 | 21.2091 | -19.0045 | 1926.1682 | 5133.0500 | 7 | 1209.3295 |
| resident | B1 | 968.2864 | 14.1182 | -3.4500 | 968.2864 | 3457.5500 | 3 | 1266.1776 |
| resident | B2 | 1827.6882 | 23.9642 | -6.7271 | 1827.6882 | 3405.7259 | 2 | 532.8905 |
| resident | B3 | 872.4527 | 14.1970 | -3.4447 | 872.4527 | 3705.4278 | 4 | 1109.2996 |
| resident | B4 | 959.4945 | 15.7732 | -3.8645 | 959.4945 | 4209.0620 | 6 | 1192.3303 |
| registered | B0 | 420.9636 | 10.4773 | -8.9682 | 420.9636 | 1965.0250 | 6 | 540.4028 |
| registered | B1 | 239.2091 | 6.0682 | -0.4364 | 239.2091 | 1501.5250 | 4 | 418.4150 |
| registered | B2 | 356.4385 | 7.8562 | -2.1414 | 356.4385 | 2173.1839 | 7 | 636.7271 |
| registered | B3 | 225.4212 | 5.6061 | -0.6273 | 225.4212 | 1485.6049 | 1 | 419.0499 |
| registered | B4 | 247.2048 | 6.2914 | -1.0314 | 247.2048 | 1578.3430 | 5 | 451.8469 |

Phase A pairwise evidence：

- B3 vs B1：resident `+9.8972%` pooled improvement、origin wins `6/11`；registered `+5.7639%`、origin wins `8/11`。
- B4 vs B1：resident `+0.9080%` pooled improvement但 latest `-21.7354%`；registered `-3.3426%` pooled、latest `-5.1160%`。
- B3 vs B0：resident `+54.7053%`、registered `+46.4511%`；B3 是两个 target 的 strongest fixed candidate。
- B3/B4 均没有改变 common coverage；B4 的不足窗口只按固定 B1 fallback 处理。

## Phase B｜City heterogeneity and oracle

### City preference profile

基于完整 2011–2021 validation period 的 city profile 仅用于描述性诊断，不是部署 selector：

- resident 的 historical best predictor 分布：B0 `10`、B1 `9`、B2 `10`、B3 `8`、B4 `3` 城市。
- registered 的分布：B0 `5`、B1 `12`、B2 `9`、B3 `9`、B4 `5` 城市。
- resident descriptor：stable `4`、outlier-dominated `12`、weak/unstable `24`。
- registered descriptor：stable `5`、outlier-dominated `12`、weak/unstable `23`。
- resident 与 registered 的 historical best predictor 只有 `12/40` 城市一致；`28/40` 不一致。

因此 Q2 的答案是：城市偏好现象存在，但多数城市没有足够稳定、跨 target 的单一偏好；后验 static grouping 不能直接部署。

### Oracle upper bound

两类 oracle 都明确标记为 `ORACLE_NON_DEPLOYABLE_POST_HOC_UPPER_BOUND`：

| Target | Best fixed | Best fixed MSE | Row/origin oracle MSE | Row oracle headroom | Static-city oracle MSE | Static-city headroom |
|---|---|---:|---:|---:|---:|---:|
| resident | B3 | 872.4527 | 413.3766 | 52.62% | 711.7988 | 18.42% |
| registered | B3 | 225.4212 | 188.1700 | 16.52% | 206.9321 | 8.20% |

Q3 的答案是：理论 headroom 对 resident 较大、对 registered 较小；这支持继续观察 selector，但不授权把 oracle 当实际性能。

## Phase C｜Prequential selectors

selector candidate 固定为 B1/B2/B3/B4，cold start 固定为少于 3 个 prior scored origins 回退 B1；所有 selector rows 的 current/future leakage 为 0。

| Target | Selector | Pooled MSE | 相对 B1 | 2021 MSE | 相对 B1 latest | City improved share | Concentration interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| resident | global | 1109.4027 | -14.5738% | 3705.4278 | -7.1692% | 20.0% | no net pooled improvement |
| resident | city | 1121.6613 | -15.8398% | 3331.8930 | +3.6343% | 20.0% | concentrated / mixed |
| registered | global | 233.2091 | +2.5083% | 1485.6049 | +1.0603% | 47.5% | concentrated / mixed |
| registered | city | 227.8352 | +4.7548% | 1491.9337 | +0.6388% | 50.0% | concentrated / mixed |

selector concentration detail：

- resident city selector 相对 B1 只有 `8/40` 城市累计改善，top-1 positive delta SSE 占 `73.88%`，去掉最大改善城市后净改善消失。
- registered city selector 相对 B1 有 `20/40` 城市改善，去掉最大改善城市后仍保持正向 delta，但 top-3 positive delta SSE 占 `74.73%`，仍不是无条件 broad gain。
- registered city selector 相对 global selector 的 pooled improvement 为 `2.3043%`，但 2021 略差 `0.4260%`；resident city selector 相对 global selector 仍劣化 `1.1050%`。

late-origin selector choice distribution 已写入 `late_origin_comparison.csv`。R1 中 global selector 在 resident 的 2019–2021 均选择 B3；registered 在 2019 选择 B4、2020–2021 选择 B3。city selector 在 late origins 仍在 B1/B2/B3/B4 间切换，没有形成稳定单一 city routing 规则。

Q4–Q7 的答案是：global/city prequential selector 能在 registered 上兑现一部分 headroom，city selector 还略优于 global；但 resident 上两者均失败，且 registered 收益仍有明显 concentration 与 latest/aggregate tension，因此尚不足以形成跨 target 的 actionable selector evidence。

## Slope / mechanism diagnostics

`slope_mechanism_diagnostics.csv` 和 `slope_disagreement_summary.csv` 记录每个 city × origin 的 history point count、拟合 slope、extrapolation distance、B1/B2 slope disagreement 与 error reversal。

- B1/B2 slope sign disagreement 的全期平均 share：resident `14%`、registered `7%`；late 2019–2021 平均约为 resident `22%`、registered `9%`。
- late period B2 better share 平均约为 resident `31%`、registered `23%`。
- B2 的胜负并不主要由 slope sign disagreement 单独解释：多数 late origin 中，B2 在 slope sign agreement 行上的改善 share 高于 disagreement 行。

Q10 的答案是：近期 slope 与长期 slope 的分歧是可观测机制，但不足以支持本 Plan 内新增手工 rule；B3 的收益更像有限 recent smoothing 的 aggregate effect。

## Verification actually passed

- `python -m py_compile scripts/build_s3_plan02_recent_window_and_selector.py`
- runner pass：12,320 fixed/selector prediction rows；308 per-origin metric rows；28 aggregate metric rows；52 pairwise summary rows；3,520 oracle rows。
- `leakage_audit.csv`：12,320 rows，全部 PASS。
- `selector_history_audit.csv`：3,520 selector audit rows，全部 PASS；current/future selector-history leakage 为 0。
- `r1_r2_equivalence_check.csv`：154 rows，全部 PASS。
- `s2_b0_b1_compatibility_check.csv`：88 rows，全部 PASS。
- `plan01_b0_b1_b2_compatibility_check.csv`：132 rows，全部 PASS。
- B3/B4、global selector、city selector 均保持 40-city common coverage。
- `git diff --check` 通过。

## Plan deviations

- 无实验问题、scope 或 validation-contract deviation。
- 实现过程中修复了三个 artifact schema/serialization 问题，并在最终 runner 通过后才进入交付；没有改变 predictor、selector、fallback 或 scoring 定义。
- 未生成图表：当前 CSV/JSON 已直接回答 fixed ladder、selector、oracle、concentration、late-origin 与 slope 机制问题，图表不会增加新的 machine evidence。

## Material project-state changes

- PROJECT_BRIEF：无变化。
- DECISIONS：当前 strongest fixed baseline 从 B1 更新为 B3；B4 不作为统一 leader；selector 只保留为 registered 条件性证据，oracle 永远不进入 deployable ranking。
- CURRENT_STATE：S3 Plan02 evidence 已交付，等待 Main/Lab 复审；S3 未 COMPLETE，S4 未进入。

## Remaining uncertainty / known limits

- resident 的 B3 latest 2021 劣于 B1，说明 fixed leader 仍有 late-regime conflict。
- registered selector 的 aggregate 小幅收益可能不足以抵消 routing 复杂度，且 top-3 city concentration 较高。
- city preference profile 使用完整 validation period，只能作 descriptive diagnostic；不能直接转成部署时可复现的 static group。
- oracle headroom 说明理论空间存在，但当前 selector 的样本识别能力不足以统一兑现。
- 官方 target mapping、真实 publication timing 与最终 2023 任务仍未被本 Plan 解决。

## Required question answers

1. **B3/B4 是否解决 trade-off？** B3 在 aggregate 上超过 B1/B2，但 resident 2021 仍冲突；B4 没有进一步改善，故是部分解决，不是完全消除。
2. **Strongest fixed baseline？** B3；两个 target 的 pooled/macro MSE 均最优，且 B3 vs B1 在 registered 的 latest 也略优。
3. **城市偏好是否稳定？** 只有少数城市稳定；大多数弱/不稳定或 outlier-dominated，跨 target 仅 12/40 一致。
4. **Oracle headroom？** resident static `18.42%`、row `52.62%`；registered static `8.20%`、row `16.52%`；足以继续观察，但不是部署证据。
5. **Global selector？** resident 失败；registered pooled 改善 `2.51%`。
6. **City selector 是否超过 global？** registered pooled 进一步改善 `2.30%` 相对 global；resident 变差 `1.10%`。
7. **selector 收益 broad-based？** resident 否；registered 有一定 broadness，但 concentration 仍明显。
8. **resident/registered 是否一致？** 不一致；B3 fixed 方向一致，但 selector 只有 registered 正向。
9. **2021 与 aggregate 是否冲突？** resident B3 aggregate 第一但 2021 劣于 B1；registered B3 两者一致。
10. **slope disagreement 是否解释胜负？** 只能部分解释，不能据此加入新 rule。
11. **S3 ladder 如何更新？** fixed ladder 记录为 B0 → B1 → B3/B4 → B2 的 recency/stability evidence，当前 fixed leader 为 B3；adaptive selector 作为 target-conditional secondary evidence。
12. **下一最高信息价值问题？** 由 Main/Lab 决定是否设计一个更严格预注册的 late-regime robustness check；本 Plan 不自行 sweep 窗口、不进入 S4。

## Delivery

- Artifact/implementation commit SHA: to be recorded after the focused commit.
- Metadata follow-up commit: if needed, record the full artifact SHA without rewriting history or force pushing.
- Delivery target: `main`，focused commit + push，最终确认 local `main == origin/main` 且 worktree clean。
