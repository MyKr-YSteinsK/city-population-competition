# S2-Plan05 Handoff｜Holdout vs Multi-Origin Expanding Backtest

## Result / status

本轮完成 S2 Validation Architecture 范围内的实际 backtest 与跨 origin aggregation 证据。结果状态为 S2_PLAN05_VALIDATION_ARCHITECTURE_PASS，是交给 Main/Lab 复审的 architecture recommendation，不是 S2 COMPLETE，也没有进入 S3。

主要 recommendation：

- 采用 hybrid validation architecture：multi-origin expanding backtest 作为稳定性主证据，latest single holdout 作为接近最终 2023 任务的诊断窗口。
- resident / registered 的主 architecture comparison 使用共同严格 origin 2011–2021，共 11 个 origin；两者最新可评分 origin 都是 2021，其 target-history cutoff 为 2019。
- 以 pooled city-year MSE 作为任务对齐的 primary aggregate；macro-origin MSE 作为必须保留的 stability sensitivity。若未来 coverage 不等导致两者排名不同，应触发 review，而不能静默选择较好的一个。
- resident 保持 primary proxy，registered 保持 mandatory sensitivity；R1/R2 仍是 information-set replay assumptions，不是 publication facts。

## Starting repository

- Repository: city-population-competition
- Branch: main
- Starting local/remote main SHA: f97d363e4b2fdcff6b475c6e3fd6a9bf2a20ca66
- Starting working tree: clean
- Refreshed canonical population: data/normalized/人口规模.csv
- Plan04 official source SHA256: 9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0
- Current population canonical profile: 718 rows, 718 unique city-year keys, resident non-empty 693, registered non-empty 707, latest observed year 2021 for both targets

Artifact/implementation commit SHA: 6beb826. A metadata-only follow-up commit will record this SHA; no history rewrite or force push is allowed.

## Actual changes

- Added scripts/build_s2_plan05_validation_architecture.py.
- Added this handoff folder with machine-readable validation architecture, predictor, boundary, leakage, residual, metric, aggregation, comparison, stability and recommendation artifacts.
- Updated CURRENT_LEARNING_CHECKPOINT.md, docs/PROJECT_CONTEXT.md and docs/PROJECT_FILE_MAP.md with the Plan05 state.
- No S1 canonical truth, S1 quality registry, raw workbook, target contract or historical Plan01–04 artifact was rewritten.
- test.xls was checked only as a quarantine boundary; it was not used.

## Commands actually executed

From the repository root:

    git fetch origin main
    git status --short --branch
    git rev-parse HEAD
    git rev-parse origin/main
    git pull --ff-only origin main

    <bundled-python> -m py_compile scripts/build_s2_plan05_validation_architecture.py
    <bundled-python> scripts/build_s2_plan05_validation_architecture.py
    git diff --check

The bundled Python runtime used for the actual run was the workspace dependency runtime. The generator output was:

    S2_PLAN05_VALIDATION_ARCHITECTURE_PASS
    starting_commit=f97d363e4b2fdcff6b475c6e3fd6a9bf2a20ca66
    common_strict_origin_domain=2011-2021
    latest_valid_origin_by_target={"registered":2021,"resident":2021}
    predictor_count=2
    prediction_rows=3520
    per_origin_metric_rows=88
    leakage_audit_all_passed=True
    aggregation_sanity_all_passed=True
    r1_r2_numeric_residuals_exactly_equal=True
    formal_ml_model_used=False
    tuning_run=False
    final_2023_forecast_run=False
    submission_run=False
    test_xls_used=False

## Validation contract implemented

### Target and horizon

The local target contract remains PRIMARY_PROXY_WITH_SENSITIVITY_TARGET:

- primary proxy: resident / 常住人口（万人）;
- mandatory sensitivity: registered / 户籍人口（万人）;
- official competition mapping remains unresolved;
- reference forecast geometry is history through 2021 → predict 2023, so every retrospective origin T uses a target-history cutoff of T-2.

The main architecture comparison uses common strict origins 2011–2021. The resident-only strict 2010 origin was not mixed into the main comparison, because doing so would break the resident/registered apples-to-apples domain.

### Latest single out-of-time holdout

The latest valid origin is selected from actual current target coverage and predictor scoreability, not mechanically from a plan year. The implemented rule requires:

- Plan02 candidate status;
- at least 90% target labels at origin T;
- at least 90% scoreable cities for every fixed predictor;
- target input history ending at T-2;
- origin label T used only for retrospective scoring.

Under the refreshed population source, both resident and registered satisfy this rule at T=2021, with cutoff 2019 and 40 scoreable cities.

### Multi-origin expanding backtest

For each T in 2011–2021, the history is all non-missing target observations with year <= T-2. The available history expands with later origins; no target label from T-1, T, or later is available to the predictor. The score unit is one city-year residual within one origin.

### Fixed naive predictors

Only two deterministic rules were used, both fixed before scoring:

1. PERSISTENCE_LAST_OBSERVED: use the latest non-missing target value at or before T-2.
2. TWO_POINT_LINEAR_TREND: use the two latest non-missing target observations at or before T-2; extrapolate their observed-year slope to T.

There was no formal ML model, feature engineering, hyperparameter search, feature selection, ensemble, leaderboard, final forecast or submission. A city with insufficient history is left unscoreable and is recorded as coverage; it is never imputed.

## Leakage controls and replay geometry

leakage_audit.csv contains 3,520 city-level audit rows. All rows are PASS. The audit records the allowed cutoff, latest and previous target years used, scoring target year, future target observations ignored, and the T-2 rule.

origin_information_boundary.csv contains 44 target × scenario × origin rows. Each row records the Plan02 feasibility classification, target cutoff, scoring year, source-specific cap snapshot and hash, publication-timing status, and covariate-use role.

Both Plan02 scenarios were retained:

- R1_REFERENCE_SNAPSHOT_LAG;
- R2_CONSERVATIVE_PUBLICATION_STRESS.

The two predictors use target history only. Therefore R1/R2 do not change their numerical predictions or scores in this Plan. The generator compared all matching residuals and recorded exact equality. This is not a claim that R1/R2 are equivalent for a later covariate model.

## Actual backtest evidence

All 11 common origins have 40 scoreable cities for both targets and both predictors. Consequently each origin has pooled weight 40 / 440 = 1/11, which equals its macro-origin weight. In this current domain, pooled and macro MSE are numerically equal; this is a coverage fact, not a general identity.

### Latest holdout and multi-origin pooled result

MSE is lower-is-better. The same ranking appears in latest, pooled and macro scopes for both targets and both scenarios:

| Target | Predictor | Latest 2021 MSE | Latest RMSE | Multi-origin pooled MSE | Multi-origin macro MSE | Rank |
|---|---|---:|---:|---:|---:|---:|
| resident | persistence | 5133.050 | 71.6453 | 1926.1682 | 1926.1682 | 2 |
| resident | two-point trend | 3457.550 | 58.8009 | 968.2864 | 968.2864 | 1 |
| registered | persistence | 1965.025 | 44.3286 | 420.9636 | 420.9636 | 2 |
| registered | two-point trend | 1501.525 | 38.7495 | 239.2091 | 239.2091 | 1 |

single_vs_multi_origin_comparison.csv directly records the rank and best-predictor comparisons. For every target × scenario combination, the latest best predictor, pooled best predictor and macro best predictor are all TWO_POINT_LINEAR_TREND; no judgment change is observed in this fixed-rule experiment.

### Stability and latest-origin location

The latest origin is the worst MSE origin for both predictors under both target hypotheses. Compared with the historical median MSE, the 2021 latest-origin ratios are:

- resident: persistence 3.592x, trend 14.353x;
- registered: persistence 11.306x, trend 19.189x.

Early versus late mean MSE also shifts upward:

- resident persistence: 1491.005 → 2540.045; trend: 886.165 → 1228.305;
- registered persistence: 129.690 → 765.720; trend: 73.350 → 446.580.

The resident ranking flips once at 2013: persistence is better there, while two-point trend is better at the other 10 origins. Registered keeps the two-point trend ranked first at all 11 origins. This supports multi-origin evidence as the stability layer even though the latest judgment happens to agree.

The pooled SSE concentration diagnostic flags a material single-city share at the 25% threshold:

- resident persistence: city3 26.12%; resident trend: city5 30.86%;
- registered persistence: city24 25.70%; registered trend: city40 40.35%.

The top-three shares are also recorded in validation_stability_summary.csv; city concentration is a diagnostic and is not corrected by silently reweighting cities.

### Aggregation comparison and sanity checks

cross_origin_aggregation_comparison.csv records:

- pooled city-year MSE and RMSE;
- macro-origin MSE and sqrt(mean(origin MSE));
- mean origin RMSE separately;
- competition-score transform separately for each explicit MSE scope;
- origin counts, weights, ranks and weighting interpretation.

aggregation_sanity_checks.csv machine-verifies both RMSE/MSE identity and an unequal-origin fixture where macro selects A but pooled selects B. Therefore pooled, macro, mean-origin-RMSE and sqrt-macro-MSE are not interchangeable quantities.

## Failures / deviations

- No execution failure occurred.
- The main comparison deliberately did not add resident-only conditional origins or the resident-only strict 2010 origin; those would weaken the common target comparison and were outside the main architecture contract.
- R1/R2 did not produce score differences because no covariate was used. The handoff records this as an explicit zero-effect result rather than fabricating scenario sensitivity.
- Historical Plan01–04 handoffs were read and preserved; they were not regenerated or overwritten.

## Remaining uncertainty

- The official competition mapping between the task's “total population” wording and resident versus registered population remains unresolved.
- Actual source publication timing remains unknown; R1/R2 are explicit replay assumptions.
- The PDF's formal MSE wording and FAQ RMSE wording remain a separate evidence conflict; this Plan keeps MSE primary and RMSE auxiliary.
- Fixed naive rules are architecture probes, not evidence that a future formal model will have the same ranking or error scale.

## Validation Architecture Recommendation to Main

1. Freeze the main S2 comparison domain at common strict origins 2011–2021.
2. Use multi-origin expanding as the primary architecture for stability evidence.
3. Keep the latest 2021 single holdout as a final-task proximity and stress-window diagnostic, not as the only validation result.
4. Use pooled city-year MSE as the task-aligned primary aggregate, and always report macro-origin MSE as a guardrail. If later scoreable counts differ and ranks diverge, stop for a contract review.
5. Keep resident primary and registered sensitivity separate; do not average their errors.
6. Preserve R1/R2 source-specific information-set boundaries. If future predictors use covariates, recompute under both scenarios and report any numerical sensitivity.
7. Any later model comparison must freeze origins, target cutoff, aggregation scope and predictor/model-selection rules before using validation results.

## Interpretation requirement

基于本轮 actual backtest evidence，S2 的 validation architecture 已足够稳定，可以建议 Main 形成 S2 final report：latest holdout、multi-origin、pooled/macro、target sensitivity、R1/R2 role、leakage and stability 都有可追溯 artifacts。剩下信息价值最高的问题不是再扩充 naive predictor，而是取得官方 target mapping 与真实 source publication timing 的直接证据。

这只是 S2 内部 recommendation。本 handoff 不宣布 S2 COMPLETE，不进入 S2-Plan06 或 S3。

## Artifact index

- validation_architecture_definitions.csv
- naive_predictor_definitions.csv
- origin_information_boundary.csv
- leakage_audit.csv
- per_origin_predictions_or_residuals.csv
- per_origin_metrics.csv
- cross_origin_aggregation_comparison.csv
- cross_origin_aggregation_weight_evidence.csv
- aggregation_sanity_checks.csv
- single_vs_multi_origin_comparison.csv
- validation_stability_summary.csv
- architecture_recommendation.csv
- plan05_summary.json

Final delivery consists of the artifact/implementation commit above plus the metadata-only follow-up commit that records it.
