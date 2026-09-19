# S3-Plan01 Handoff｜Long-History Trend vs Recent Two-Point Trend

## Result / status

本轮完成 S3 Baseline Ladder 的第一组正式 baseline evidence。状态为 S3_PLAN01_BASELINE_LADDER_PASS；本 Plan 不宣布 S3 COMPLETE，也没有进入 S4。

核心结论：

- B1 recent two-point trend 是 resident 与 registered 两个 target 上 pooled/macro MSE 都最强的 aggregate baseline。
- B2 expanding per-city linear trend 相比 B0 persistence 在两个 target 上都有 pooled MSE 改善，但相比 B1 在两个 target 上都变差，因此不能宣布 B2 为统一 champion。
- resident 出现明确 recency–stability trade-off：B2 的 2021 MSE 略低于 B1，且跨 origin MSE dispersion 更低；registered 中 B2 的 2021 MSE 与跨 origin dispersion 都比 B1 差。
- B2 对 B0 的改善在 resident 主要受少数城市贡献，去掉最大改善城市后净改善消失；registered 的改善更 broad-based，去掉最大改善城市后仍保持。

## Starting repository state

- Repository: city-population-competition
- Branch: main
- Starting local/remote main SHA: 1e896466c45e79b557ec0a0dc386f301b65f1954
- Starting working tree: clean
- Main authorization commit: 1e89646 docs: complete S2 and enter S3
- Canonical population: data/normalized/人口规模.csv
- Canonical population profile: 718 rows, 718 unique city-year keys, resident non-empty 693, registered non-empty 707
- Official population source SHA256 inherited from Plan04: 9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0

The remote fast-forward from the previous local HEAD was pulled before any S3 edit. The current S2 contract was read from the updated checkpoint and was not changed.

## Actual changes

- Added scripts/build_s3_plan01_baseline_ladder.py.
- Added this S3 handoff folder with definitions, contract evidence, predictions/residuals, metrics, pairwise comparisons, stability, concentration, leakage and scenario-equivalence artifacts.
- Updated CURRENT_LEARNING_CHECKPOINT.md, docs/CODEX_INSTRUCTIONS.md, docs/PROJECT_CONTEXT.md and docs/PROJECT_FILE_MAP.md with the authorized S3 state and Plan01 evidence.
- No S1 canonical truth, raw XLSX, quarantine status, target contract, S2 origin domain, S2 aggregation contract or historical S2 handoff was rewritten.
- No final 2023 forecast, submission, leaderboard operation, covariate, formal complex ML model, tuning or ensemble was run.

## Frozen validation contract confirmation

The implementation inherits and machine-checks the S2 contract:

- common main origins: 2011–2021, 11 origins;
- target history for pseudo-target T: all allowed target rows are year <= T-2;
- resident / 常住人口（万人）: primary proxy;
- registered / 户籍人口（万人）: mandatory separate sensitivity;
- primary ranking: pooled city-year MSE;
- mandatory sensitivity: macro-origin MSE;
- latest diagnostic: 2021 holdout, not a replacement for the multi-origin evidence;
- R1_REFERENCE_SNAPSHOT_LAG and R2_CONSERVATIVE_PUBLICATION_STRESS retained;
- no covariates, so R1/R2 numerical equivalence is expected and machine-checked.

For the current canonical source, all three predictors have 40 scoreable cities at every common origin for both targets. Pooled and macro MSE therefore coincide numerically in this run because every origin contributes the same 40 city rows; the artifacts keep the scopes separate.

## Predictors under test

### B0_PERSISTENCE_LAST_OBSERVED

Use the latest non-missing target observation at or before T-2. This is the reference baseline.

### B1_TWO_POINT_LINEAR_TREND

Use the two latest non-missing target observations at or before T-2, compute their observed-year slope, and extrapolate to T. This formula was not changed from S2. The S3 rerun is row-level compatible with the S2 Plan05 B1 probe.

### B2_EXPANDING_PER_CITY_LINEAR_TREND

For each target, origin and city independently, fit ordinary least squares:

    target = intercept + slope * year

using every non-missing target row for that city with year <= T-2, then predict T. The minimum effective history is two distinct years. No other city, covariate, window, polynomial term, robust fit, regularization, or tuning is used. Insufficient history would remain unscoreable; no imputation is performed.

## Commands actually executed

From the repository root:

    git fetch origin main
    git status --short --branch
    git rev-parse HEAD
    git rev-parse origin/main
    git pull --ff-only origin main

    bundled-python -m py_compile scripts/build_s3_plan01_baseline_ladder.py
    bundled-python scripts/build_s3_plan01_baseline_ladder.py
    git diff --check

Generator output:

    S3_PLAN01_BASELINE_LADDER_PASS
    starting_commit=1e896466c45e79b557ec0a0dc386f301b65f1954
    common_origin_domain=2011-2021
    predictor_count=3
    prediction_rows=5280
    per_origin_metric_rows=132
    aggregate_metric_rows=12
    pairwise_summary_rows=12
    leakage_audit_all_passed=True
    r1_r2_equivalence_all_passed=True
    s2_b0_b1_compatibility_all_passed=True
    formal_ml_model_used=False
    covariates_used=False
    tuning_run=False
    ensemble_run=False
    final_2023_forecast_run=False
    submission_run=False
    test_xls_used=False

## Metric evidence

The table below is for R1; R2 is numerically identical for all target-history-only predictors. Scores are shown per explicit MSE scope and are not independent votes.

| Target | Predictor | Pooled MSE | Pooled RMSE | Pooled MAE | Pooled bias | Median abs error | Pooled score | Latest 2021 MSE | Latest score | Latest predictor rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resident | B0 persistence | 1926.1682 | 43.8881 | 21.2091 | -19.0045 | 5.5000 | 0.000518896 | 5133.0500 | 0.000194778 | 3 |
| resident | B1 two-point | 968.2864 | 31.1173 | 14.1182 | -3.4500 | 4.0000 | 0.001031687 | 3457.5500 | 0.000289139 | 2 |
| resident | B2 expanding | 1827.6882 | 42.7515 | 23.9642 | -6.7271 | 11.5681 | 0.000546840 | 3405.7259 | 0.000293537 | 1 |
| registered | B0 persistence | 420.9636 | 20.5174 | 10.4773 | -8.9682 | 5.0000 | 0.002369872 | 1965.0250 | 0.000508641 | 2 |
| registered | B1 two-point | 239.2091 | 15.4664 | 6.0682 | -0.4364 | 3.0000 | 0.004163040 | 1501.5250 | 0.000665546 | 1 |
| registered | B2 expanding | 356.4385 | 18.8796 | 7.8562 | -2.1414 | 3.2751 | 0.002797684 | 2173.1839 | 0.000459943 | 3 |

Macro-origin MSE, sqrt(macro-origin MSE), mean origin RMSE, macro score, per-origin MSE/RMSE/MAE/bias/score and all score scopes are in aggregate_metrics.csv and per_origin_metrics.csv.

## Pairwise comparison

Positive MSE delta means the candidate has lower MSE than its baseline. Negative values mean the candidate is worse.

| Target | Pair | Pooled MSE relative improvement | Candidate origin wins | Candidate origin losses | City-origin rows improved | Latest 2021 outcome |
|---|---|---:|---:|---:|---:|---|
| resident | B1 vs B0 | +49.73% | 10 | 1 | 57.50% | win |
| resident | B2 vs B0 | +5.11% | 4 | 7 | 41.14% | win |
| resident | B2 vs B1 | -88.75% | 3 | 8 | 30.45% | win |
| registered | B1 vs B0 | +43.18% | 11 | 0 | 56.36% | win |
| registered | B2 vs B0 | +15.33% | 9 | 2 | 67.50% | loss |
| registered | B2 vs B1 | -49.01% | 3 | 8 | 42.95% | loss |

B1 is therefore the strongest aggregate baseline in both target hypotheses. B2 is an improvement over the reference B0, but not over B1.

## Stability evidence

### Resident

- B1 wins 8 of 11 origins; B0 wins 1; B2 wins 2.
- B1 pooled MSE is 968.2864; B2 is 1827.6882.
- B1 origin MSE standard deviation is 1266.18; B2 is 532.89. B2 is substantially less dispersed across origins.
- B2 latest 2021 MSE is 3405.7259 versus B1 3457.5500, so B2 is slightly better in the latest window; both are still high-error late-regime results and 2021 is B2's worst origin.
- B2 latest-to-historical-median MSE ratio is 1.907x versus B1's 14.353x. This is the clearest evidence for a resident recency–stability trade-off rather than a simple aggregate win.

### Registered

- B1 wins 8 of 11 origins; B2 wins 3; B0 wins 0.
- B1 pooled MSE is 239.2091; B2 is 356.4385.
- B1 origin MSE standard deviation is 418.41; B2 is 636.73, so long-history trend is less stable here.
- B2 latest 2021 MSE is 2173.1839, worse than B0 1965.0250 and B1 1501.5250; B2 is rank 3 at the latest origin and 2021 is its worst origin.
- B2 latest-to-historical-median MSE ratio is 50.411x versus B1's 19.189x.

The exact per-origin MSE/RMSE/MAE/bias/score sequence, rank sequence and winner sequence are in validation_stability_summary.csv. 2021 remains a separate final-task proximity diagnostic and does not replace the full expanding evidence.

## Error-concentration and broad-vs-concentrated improvement

The city-level artifacts decompose baseline-minus-candidate squared error by city and origin. The broad-based rule is explicit: positive pooled improvement, at least 50% of cities improved, and the top three positive delta-SSE share no more than 60%.

### B2 versus B0

- resident: 21 of 40 cities improve, 19 worsen; top city is city3; top-three positive delta-SSE share is 63.80%; removing city3 changes the MSE delta to -19.91. Assessment: concentrated_or_mixed_improvement, not robust broad gain.
- registered: 29 of 40 cities improve, 10 worsen; top city is city3; top-three positive delta-SSE share is 53.78%; removing city3 leaves a +45.88 MSE improvement. Assessment: broad_based_improvement.

### B2 versus B1

There is no pooled net improvement for either target. B2 improves only 15 resident cities and 17 registered cities; the top-three positive contributions are highly concentrated. This reinforces that B1 remains the stronger aggregate baseline even where B2 has a late resident advantage.

Dominant-city pooled SSE shares for each predictor, all city-level deltas, leave-one-city sensitivity and top-contributor shares are in city_improvement_concentration.csv and improvement_concentration_summary.csv.

## Leakage and R1/R2 evidence

leakage_audit.csv contains 5,280 rows, one per target × scenario × predictor × origin × city. Every row is PASS and records:

- scoring target year T;
- allowed target-history cutoff T-2;
- history rows and history years actually used;
- latest target year used;
- regression fit maximum year for B2;
- T-1/T/future label exclusion status.

r1_r2_equivalence_check.csv contains 66 target × predictor × origin checks. Predictions, residuals and squared errors are exactly equal between R1 and R2 for all rows. This is expected because all three predictors use target history only.

s2_b0_b1_compatibility_check.csv contains 88 comparisons against the S2 Plan05 B0/B1 probe rows. All are PASS, so the formal S3 rerun did not silently change the frozen S2 definitions.

## Interpretation Q1–Q6

### Q1 — Is trend stably better than persistence?

Yes for the aggregate trend signal, but not uniformly for every mechanism. B1 beats B0 in pooled/macro MSE for resident and registered, wins 10/11 and 11/11 origins respectively, and wins the latest 2021 comparison for both. B2 also beats B0 in pooled MSE for both targets, but wins only 4/11 resident origins and 9/11 registered origins; its latest outcome is resident win and registered loss.

### Q2 — Is long-history trend better than two-point trend?

No on aggregate: B2 is worse than B1 by 88.75% relative MSE in resident and 49.01% in registered. The resident case is a real trade-off rather than a simple failure: B2 has lower origin dispersion and slightly lower 2021 MSE, but loses the pooled comparison because B1 is much better in most historical origins. In registered, B2 loses both aggregate and latest stability comparisons.

### Q3 — Is B2 improvement broad-based?

Target-dependent. Registered B2 versus B0 is broad-based under the predeclared rule and survives removal of city3. Resident B2 versus B0 is concentrated or mixed; removal of city3 removes the net improvement. B2 versus B1 has no net pooled improvement in either target.

### Q4 — Do resident and registered agree?

They agree that B1 is the strongest aggregate baseline and B2 is better than B0 on pooled MSE. They disagree on the latest/stability interpretation of B2: resident shows a late-regime advantage and lower dispersion, while registered shows latest degradation and higher dispersion. The target sensitivity must remain explicit.

### Q5 — Does 2021 conflict with historical evidence?

For resident, yes: B2 is the latest winner but not the historical aggregate winner. For registered, no such B2 recency benefit appears; B1 wins both historical aggregate and latest. Therefore latest proximity and historical stability answer different questions and must both remain in the baseline ladder.

### Q6 — Highest-value next S3 question

The next S3 question should be a limited, pre-registered recency-versus-stability experiment: can one explicitly fixed recent-history trend mechanism preserve the resident late-regime benefit without reproducing the registered latest-origin failure? It should compare against B0/B1/B2 under the same frozen contract, avoid an unbounded window sweep, and be designed in the next Plan only. This Plan does not execute it.

## Failures / deviations

- No execution failure occurred.
- No optional chart was generated because the machine-readable origin curves and city decomposition answer the required questions without adding presentation-only artifacts.
- No resident-only 2010 origin was added; the frozen common main domain remains 2011–2021.
- R1/R2 produced no score difference, as expected for target-history-only predictors; this was recorded as an equivalence result, not treated as evidence that the scenarios are generally interchangeable.

## Remaining uncertainty

- Official target mapping between total population wording and resident/registered population remains unresolved.
- Source publication timing remains unknown; R1/R2 remain replay assumptions.
- The PDF MSE/RMSE wording conflict remains; this Plan follows the frozen MSE-primary contract.
- B1 is the strongest current formal aggregate baseline, but it remains a simple probe and does not establish final model or competition performance.
- City concentration remains material; future baseline/model improvements must continue to report broad versus concentrated gains.

## Recommendation for the next S3 question only

Do not enter S4 and do not declare a champion beyond the current evidence. Prepare one constrained S3 Plan around the recency–stability trade-off exposed by resident versus registered B2 results. Keep B1 as the aggregate reference baseline, keep B2 as the long-history comparator, and pre-register the next mechanism before scoring.

## Artifact index

- HANDOFF.md
- baseline_predictor_definitions.csv
- validation_contract_evidence.csv
- origin_information_boundary.csv
- per_origin_predictions_or_residuals.csv
- per_origin_metrics.csv
- aggregate_metrics.csv
- pairwise_baseline_comparison.csv
- pairwise_origin_comparison.csv
- validation_stability_summary.csv
- city_improvement_concentration.csv
- improvement_concentration_summary.csv
- leakage_audit.csv
- r1_r2_equivalence_check.csv
- s2_b0_b1_compatibility_check.csv
- plan01_summary.json

Artifact/implementation commit SHA: pending artifact commit. A metadata-only follow-up will record the full SHA.
