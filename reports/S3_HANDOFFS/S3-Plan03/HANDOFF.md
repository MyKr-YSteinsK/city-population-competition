# S3-Plan03 Handoff｜Per-City vs Pooled Simple Regression + Growth + ML Hurdle

## Result / status

本轮完成 S3 Baseline Ladder 的 S3-Plan03 evidence，runner 状态为
`S3_PLAN03_POOLED_GROWTH_ML_HURDLE_PASS`。本 Plan 不宣布 S3 COMPLETE，不进入 S4，
不运行 formal ML、2023 forecast 或 submission。

核心结论：

- 当前 strongest deployable fixed baseline 仍为 B3
  `B3_RECENT_3POINT_OLS_TREND`。
- B6 与 B3 使用相同 recent-3 information geometry，但 pooled shared absolute slope
  明显更差；B5 的 expanding pooled shared slope 也明显更差。
- B7 recent-3 per-city log-linear growth 接近 B3，但两个 target 上都没有超过 B3；它只
  在 middle regime `2016–2018` 取得局部优势。
- resident 与 registered 的机制结论方向一致：local recent trend 优于 pooled shared
  slope，multiplicative growth 是有信息的近邻但不足以改写 fixed leader；registered
  的数值差异较小。
- `s3_evidence_sufficient_candidate=yes` 仅是 evidence recommendation；
  `s3_complete=false`，是否生成 S3 final report 由 Main/Lab 决定。

## Starting repository state

- Repository: `city-population-competition`
- Branch: `main`
- Starting local/remote main SHA: `7e741ab91cdee3d9df396212da3b7e454629bf5d`
- Starting working tree: clean；执行前 local `main == origin/main`
- Canonical population profile: 718 rows, 718 unique city-year keys, resident non-empty 693,
  registered non-empty 707
- S2 frozen contract 未修改：common origins `2011–2021`，target history cutoff `T-2`，
  resident primary、registered sensitivity，pooled city-year MSE primary、macro-origin MSE
  sensitivity，2021 latest diagnostic，R1/R2 target-only numerical equivalence。

## Scope and actual changes

- 新增 `scripts/build_s3_plan03_pooled_growth_and_ml_hurdle.py`。
- 完整重跑 B0–B4，并新增且仅新增：
  - B5 `B5_POOLED_EXPANDING_CITY_FE_LINEAR_TREND`
  - B6 `B6_POOLED_RECENT3_CITY_FE_LINEAR_TREND`
  - B7 `B7_RECENT3_PER_CITY_LOG_LINEAR_GROWTH`
- B5 使用全部 `year <= T-2` target rows、每城独立 intercept、全城共享 absolute slope；
  B6 使用每城恰好最近 3 个 allowed rows、每城独立 intercept、全城共享 absolute slope；
  B7 使用每城最近 3 个 positive allowed rows 的 log-linear fit，不 clipping。
- B5/B6 pooled fit diagnostics 显示每次都使用 40 个城市；B5 resident/registered 的
  training row count 范围分别为 `213–613` / `227–627`，B6 固定为 `120`。
- B7 共生成每个 target `880` 条 city-origin diagnostics；positive history count 为
  `3–19`，non-positive eligible history count 为 `0`，fallback count 为 `0`。
- 新增完整的 per-origin、aggregate、pairwise、latest、early/middle/late、leave-one-origin-out、
  leave-one-city、concentration、local/shared slope、growth、leakage、compatibility 与
  ML hurdle artifacts。
- Plan03 的 leakage audit 对 B0–B7 统一重建，包含 `T-1` presence audit；共 `14,080` 行，
  所有 target history / pooled training rows 均满足 `<= T-2`。
- 未修改 raw、canonical、quarantine、S2 contract、S2 handoff、S3 Plan01 artifact 或
  S3 Plan02 artifact。

Optional D1 `D1_POOLED_RECENT3_CITY_FE_LOG_GROWTH` 未实现。B7 已覆盖 multiplicative
growth 的主要机制问题，而新增共享 log-slope diagnostic 不会改变 decision boundary；因此
省略 D1 以保持本 Plan 的预注册范围。B5/B6/B7 均实际运行。

## Frozen validation contract confirmation

- common strict origins 保持 `2011–2021`，11 origins；每个 target/scenario/predictor
  保持 40-city scoreable coverage。
- 所有 local、pooled、growth fit 的 target history 都不超过 `T-2`；origin `T` label
  只用于 scoring，不能进入本城或其他城市训练。
- B5/B6 的 shared slope 与 city intercept 只使用 allowed pooled rows；不存在 pooled
  cross-city current-origin label leakage。
- resident 仍为 primary，registered 仍为 mandatory sensitivity；没有 city reweighting，
  没有排除困难 origin/city。
- pooled city-year MSE 是 primary ranking；macro-origin MSE、RMSE、MAE、bias、score、
  latest 2021 与 stability 均同时落盘。
- R1/R2 仍是 target-only replay；predictions、residuals、squared errors、metrics 与
  rankings 均 machine-check numerical equivalent。

## Commands actually executed

从 repository root 执行：

    git fetch origin main
    git status --short --branch
    git rev-parse HEAD
    git rev-parse origin/main
    python -m py_compile scripts/build_s3_plan03_pooled_growth_and_ml_hurdle.py
    python scripts/build_s3_plan03_pooled_growth_and_ml_hurdle.py
    git diff --check

执行中的一次失败：首次把旧 B0–B4 audit 行统一到 Plan03 schema 时，B0 的
`regression_fit_max_year=None` 被错误地直接转为整数；已修复为显式空值安全转换，并重新
通过完整 runner。另修正了新模型 `T-1 target presence` audit 初版使用 eligible history
字符串的语义问题，改为基于原始 series 的实际 presence audit；没有改变 predictor 或 score。

最终 runner output：

    S3_PLAN03_POOLED_GROWTH_ML_HURDLE_PASS
    starting_commit=7e741ab91cdee3d9df396212da3b7e454629bf5d
    common_origin_domain=2011-2021
    predictor_count=8
    prediction_rows=14080
    per_origin_metric_rows=352
    aggregate_metric_rows=32
    pairwise_summary_rows=68
    pooled_fit_diagnostics_rows=88
    leave_one_origin_out_rows=352
    leakage_audit_all_passed=True
    r1_r2_equivalence_all_passed=True
    s2_b0_b1_compatibility_all_passed=True
    plan01_b0_b1_b2_compatibility_all_passed=True
    plan02_b0_b1_b2_b3_b4_compatibility_all_passed=True
    strongest_deployable_fixed_baseline_candidate=B3_RECENT_3POINT_OLS_TREND
    formal_ml_model_used=False
    covariates_used=False
    tuning_run=False
    ensemble_run=False
    selector_expansion_run=False
    final_2023_forecast_run=False
    submission_run=False
    s3_evidence_sufficient_candidate=yes
    s3_complete=False
    s4_entered=False

## Aggregate evidence

R1 与 R2 数值完全一致；下表因此代表两种 target-only replay scenario。MSE 是 primary，
score 只是同 scope 的 `1/(1+MSE)` 单调变换。

| Target | Predictor | Pooled MSE | Pooled RMSE | Pooled MAE | Bias | 2021 MSE | 2021 rank | Origin MSE std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| resident | B1 | 968.2864 | 31.1173 | 14.1182 | -3.4500 | 3457.5500 | 2 | 1266.1776 |
| resident | B3 | **872.4527** | 29.5373 | 14.1970 | -3.4447 | 3705.4278 | 3 | 1109.2996 |
| resident | B5 | 12154.0898 | 110.2456 | 65.8944 | 0.4158 | 23066.2252 | 8 | 6023.8063 |
| resident | B6 | 2783.7578 | 52.7613 | 31.3192 | -3.4447 | 5161.2028 | 7 | 892.1819 |
| resident | B7 | 938.7316 | 30.6387 | 14.6156 | -2.0688 | 3777.1004 | 4 | 1151.9579 |
| registered | B1 | 239.2091 | 15.4664 | 6.0682 | -0.4364 | 1501.5250 | 3 | 418.4150 |
| registered | B3 | **225.4212** | 15.0140 | 5.6061 | -0.6273 | 1485.6049 | 1 | 419.0499 |
| registered | B5 | 1294.2794 | 35.9761 | 23.1725 | 1.5584 | 4581.3485 | 8 | 1249.2976 |
| registered | B6 | 480.5422 | 21.9213 | 12.3430 | -0.6273 | 2338.9400 | 7 | 648.6866 |
| registered | B7 | 227.0369 | 15.0677 | 5.6614 | -0.2400 | 1499.9508 | 2 | 424.1722 |

Pairwise primary evidence：

- B3 vs B1：resident `+9.8972%` pooled MSE improvement，registered `+5.7639%`。
- B6 vs B3：resident `-219.0726%`，registered `-113.1753%`；11/11 origins 都输给 B3。
- B5 vs B3：resident `-1293.0945%`，registered `-474.1604%`；11/11 origins 都输给 B3。
- B7 vs B3：resident `-7.5969%`，registered `-0.7167%`；resident 4/11、registered 3/11
  origins 胜出，仍不足以改变 full-domain leader。
- B7 vs B1：resident `+3.0523%`，registered `+5.0885%`，但这不等于超过 B3。

## Mechanism and robustness evidence

### Per-city versus pooled slope

- B2 expanding local slope 的 resident origin-level population std 范围为 `8.01–11.90`，
  registered 为 `3.24–3.69`；B3 recent local slope 的 resident 范围为 `13.19–20.02`，
  registered 为 `3.31–8.36`。城市 slope heterogeneity 是实际存在的，不是 pooled model
  可以忽略的纯噪声。
- B5 shared slope 范围为 resident `7.0135–9.1433`、registered `4.4047–4.9015`；
  B6 shared slope 范围为 resident `5.2250–12.0500`、registered `2.9750–5.7000`。
- B6 在 resident 的 origin-MSE std 数值上低于 B3，但其 pooled MSE 约为 B3 的 3.19 倍；
  在 registered 上 B6 的 origin-MSE std 反而高于 B3。B5 的 pooled error 与 origin
  dispersion 都显著恶化。因此 slope sharing 没有形成可部署的 variance-reduction trade-off。
- `|local slope - shared slope|` 与 candidate-minus-baseline squared-error delta 的
  correlation 已写入 `slope_error_relationship.csv`；resident B5-vs-B2 为约 `0.65–0.95`，
  resident B6-vs-B3 为约 `0.41–0.98`（registered 也有正相关范围）。这是 diagnostic only，
  没有据此新增 gating rule。

### Additive versus multiplicative growth

- B7 没有超过 B3：resident pooled MSE 高 `7.5969%`，registered 高 `0.7167%`。
- B7 在 `2016–2018` middle regime 胜出；但在完整 domain 和 2021 latest 上仍不能改写
  B3 的 deployable fixed ranking。
- 所有 B7 eligible history 都为 positive；本数据没有触发非正值排除、clipping 或 fallback，
  这只是当前数据事实，不是未来数据的普遍保证。

### Stability and concentration

- leave-one-origin-out：resident full-domain leader B3 在 `10/11` 次移除后保持，移除
  2013 时切换为 B1；registered 在 `11/11` 次保持 B3。R1/R2 的 leader stability
  结论一致。
- B5/B6 相对 B3 的 city-origin improved shares 只有 resident `13.41%` / `17.27%`、
  registered `11.14%` / `20.45%`，且 pooled MSE 全面恶化。
- B7 相对 B3 的 improved shares 为 resident `51.59%`、registered `43.18%`，但 pooled
  MSE 仍分别恶化；因此“超过一半 city-origin 行改善”不能替代 primary MSE 与 origin
  stability。完整 dominant-city、top-3 SSE、top delta-SSE contributor 与 remove-city
  sensitivity 见 `improvement_concentration_summary.csv` 与 `leave_one_city_sensitivity.csv`。
- resident aggregate leader 与 latest 存在冲突：B3 是 full-domain 第一，但 2021 为第 3，
  B1 为第 2、B2 为第 1。registered 的 B3 aggregate 与 2021 latest 均为第一。

## Required question answers

1. **当前 strongest deployable fixed baseline 是否仍是 B3？** 是。B3 在 resident 与
   registered 的 pooled/macro MSE 都是 full-domain 最优，B5/B6/B7 均未改写它。
2. **B3 与 B6 谁更好？** B3 明显更好。B6 的 shared recent slope 在两个 target、11/11
   origins 都没有超过 B3；B6 只在 resident 的 origin-MSE std 上有表面下降，代价是更大
   的 pooled error。
3. **B2 与 B5 谁更稳定？** B2 更稳定且更准确。resident B2/B5 pooled MSE 为
   `1827.6882/12154.0898`、origin std 为 `532.8905/6023.8063`；registered 为
   `356.4385/1294.2794`、origin std 为 `636.7271/1249.2976`。
4. **共享 slope 是否降低 variance，代价是什么？** 不能作一般性结论。B6 resident 的
   origin dispersion 下降但误差大幅上升；registered dispersion 上升；B5 两者都恶化。
   代价是牺牲 city-specific slope flexibility，导致 systematic city-level residual。
5. **B7 是否优于 B3？** 否。B7 接近 B3，但 resident/registered pooled MSE 分别差
   `7.5969%/0.7167%`；只在 middle regime 局部胜出。
6. **resident / registered 机制是否一致？** 方向一致但强度不同：都支持 B3 的 local
   recent additive trend，反对 pooled shared absolute slope；registered 上 B7 与 B3 更接近。
7. **新 improvement 是否 broad-based？** B5/B6 明确不是。B7 虽有较高 city-origin
   行改善比例，但 full pooled MSE、origin wins 与 concentration 仍不支持 broad deployable
   improvement。
8. **2021 latest 是否与 historical aggregate 冲突？** resident 是：B3 full-domain 第一，
   但 2021 rank 3；registered 不冲突，B3 两者均第一。
9. **leave-one-origin-out 后 leader 是否稳定？** resident B3 保持 `10/11`，仅移除 2013
   时转为 B1；registered 保持 `11/11`。
10. **不同城市是否应使用不同模型？** city heterogeneity 确实存在，但 Plan02 的 routing
    evidence 不稳定，Plan03 的 pooled regression 也不优于 B3；当前不能把“城市不同”升级
    为 static city routing 可部署结论。后续若研究 routing，必须是新的 prequential、预注册实验。
11. **后续 ML 至少必须超过哪个 baseline？** 必须在 resident primary 上真实低于 B3 的 pooled
    city-year MSE `872.452651515143`，而不是只超过 B1 或只提高 score。
12. **后续 ML 必须看哪些 guardrails？** 必须同时报告 macro-origin MSE、multi-origin
    stability、latest 2021、registered sensitivity、dominant/top-3 city concentration、
    leave-one-origin/city sensitivity，以及逐 origin `T-2` 与 R1/R2 information-set audit。
13. **S3 evidence 是否足够生成 final report？** recommendation 为 `yes`：S3 已覆盖
    B0–B7、growth、pooled/local mechanism、stability、latest、concentration、leakage 与
    ML hurdle。该 recommendation 不等于 S3 COMPLETE，也不替代 Main/Lab 的 final report decision。

## Verification actually passed

- `python -m py_compile scripts/build_s3_plan03_pooled_growth_and_ml_hurdle.py`
- runner：`14,080` prediction/residual rows；`352` per-origin metric rows；`32` aggregate
  metric rows；`68` pairwise summary rows；`88` pooled fit diagnostic rows；`352` leave-one-
  origin-out ranking rows。
- `leakage_audit.csv`：`14,080` rows，全部 `PASS`；`T-1` presence audit 全覆盖。
- `r1_r2_equivalence_check.csv`：全部 `PASS`。
- `s2_b0_b1_compatibility_check.csv`：全部 `PASS`。
- `plan01_b0_b1_b2_compatibility_check.csv`：全部 `PASS`。
- `plan02_b0_b1_b2_b3_b4_compatibility_check.csv`：全部 `PASS`。
- `validation_contract_evidence.csv`：全部 `PASS`。
- `s3_sufficiency_candidate.json` 与 `ml_hurdle_candidate.csv` 已生成且可解析。
- `git diff --check` 通过。

## Plan deviations

- Optional D1 pooled recent-3 city-FE log-growth 未实现；理由见上文，不影响 B5/B6/B7 MUST
  evidence 或 decision boundary。
- 无 frozen validation contract、predictor definition、coverage 或 forbidden-scope deviation。
- 实现中修复了两处审计/serialization edge case，并在最终完整 runner 通过后才交付；修复不
  改变模型定义、fallback、ranking 或目标数据。

## Material project-state changes

- `PROJECT_BRIEF`：无变化。
- `DECISIONS`：无独立文件变化；当前 strongest fixed baseline 继续为 B3，B5/B6 不适合
  部署，B7 作为接近但失败的 growth evidence；ML hurdle 固定为 B3 resident pooled MSE
  与配套 guardrails。
- `CURRENT_STATE`：通过 `CURRENT_LEARNING_CHECKPOINT.md`、`docs/PROJECT_CONTEXT.md`、
  `docs/CODEX_INSTRUCTIONS.md` 与 `docs/PROJECT_FILE_MAP.md` 记录 Plan03；S3 未正式
  COMPLETE，S4 未进入。

## Remaining risks / known limits

- resident 的 B3 aggregate 与 2021 latest 冲突，后续 final report 不能只保留一个数字。
- target 的官方精确 mapping、真实 source publication timing 与最终 2023 task 仍未解决。
- 当前 B7 无非正值只是本 canonical data 的 measured fact，不能替代未来输入的 policy。
- `s3_evidence_sufficient_candidate=yes` 仍需 Main/Lab 审阅；本 Plan 不自行生成 S3 final
  report。

## Delivery

- Artifact/implementation commit SHA：待 focused commit 完成后回填。
- Metadata follow-up commit：待 artifact SHA 回填后生成；不改写历史、不 force push。
- Delivery target：`main`；focused commit + push，最终确认 local `main == origin/main` 且
  worktree clean。
