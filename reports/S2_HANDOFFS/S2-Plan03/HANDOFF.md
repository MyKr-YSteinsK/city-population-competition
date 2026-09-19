# S2-Plan03 Handoff | Target and Metric Contract Investigation

## Result

本轮完成 S2-Plan03 的 target semantics、target sensitivity、MSE/RMSE conflict 和 local evaluator contract investigation。本轮没有运行正式 baseline、out-of-time model scoring、feature engineering、模型比较、prediction 或 submission。

结论不是 RESOLVED，而是：

    target_contract_status = PRIMARY_PROXY_WITH_SENSITIVITY_TARGET
    primary_target_proxy = resident / 常住人口（万人）
    secondary_sensitivity_target = registered / 户籍人口（万人）
    primary_metric = MSE
    auxiliary_metric = RMSE
    competition_score = 1/(1+MSE)
    cross_origin_aggregation = UNRESOLVED_DEFERRED

常住人口是当前更有内部一致性证据的主 proxy，但原始 PDF 没有把“总人口数”唯一映射到常住或户籍字段；后续结果必须保留户籍人口 sensitivity，不得把本地分数宣称为已证明的官方比赛分数。

## Starting repository

- Repository: MyKr-YSteinsK/city-population-competition
- Branch: main
- Starting remote/main and local HEAD: e46cea437cd3e05ef4d125ee791adc3fb5754972
- Starting working tree: clean
- Execution date: 2026-09-19
- The required git fetch origin main and git pull --ff-only origin main both completed; pull reported Already up to date.

## Actual changes

- 新增 scripts/build_s2_plan03_target_metric_contract.py。
- 新增 reports/S2_HANDOFFS/S2-Plan03/ 下的 target evidence、cross-source consistency、target geometry comparison、metric conflict、sanity test、local contract 和 summary artifacts。
- 脚本只读取 S1 contract、canonical CSV、Plan02 matrix 和 provenance/quarantine artifacts；不读取 raw workbook 或 test.xls。
- 更新项目上下文/文件地图与当前 checkpoint，使条件性 target/metric contract 和未决 cross-origin aggregation 可被下一轮直接发现。
- 没有修改 S1 canonical truth、quality registry、quarantine 或原始 XLSX/PDF。

## Commands actually executed and results

    git status --short
    git branch --show-current
    git fetch origin main
    git pull --ff-only origin main
    git rev-parse HEAD
    git ls-remote origin refs/heads/main
    python -m py_compile scripts/build_s2_plan03_target_metric_contract.py
    python scripts/build_s2_plan03_target_metric_contract.py
    git diff --check

Actual generator result:

    S2_PLAN03_TARGET_METRIC_PASS
    target_evidence_rows=2
    age_complete_rows_used=91
    target_geometry_rows=92
    target_comparison_rows=46
    metric_sanity_tests=6
    target_contract_status=PRIMARY_PROXY_WITH_SENSITIVITY_TARGET
    primary_target=resident
    secondary_target=registered
    primary_metric=MSE
    auxiliary_metric=RMSE
    cross_origin_aggregation=UNRESOLVED_DEFERRED
    formal_model_scoring_run=False
    test_xls_used=False

The generator also rechecked all 14 canonical key sets, the age benchmark key boundary, quarantine row counts, overall-income invalidation, urban-income label repair and 91 employee provenance rows.

## PDF evidence actually reviewed

The bundled pypdf reader extracted all four PDF pages. The data-description pages contain embedded JPEG images; pypdf extracted the page 2 and page 3 embedded images, and page 2 was visually inspected.

Direct evidence:

- PDF page 1 task description says the A and B preliminary tasks predict each city's 2023 total population.
- Page 2 data table labels pr_Population as 常住人口规模 and r_Population as 户籍人口规模.
- The task wording does not state that total population means either one of those fields.
- PDF page 4 formal 评测标准 says MSE, lower is better, and gives score = 1/(1+MSE).
- PDF page 4 FAQ item 5 says RMSE is used for model evaluation.

The MSE/RMSE conflict is therefore a direct source conflict between a formal evaluation section and FAQ guidance, not a missing calculation.

## Target evidence matrix

target_evidence_matrix.csv contains one row for each candidate target and separates:

- direct official wording;
- source schema semantics;
- cross-source consistency;
- inference-only interpretation;
- ambiguity/counterevidence.

Both candidates have the same observed domain and horizon:

| Candidate | Canonical non-empty rows | Missing canonical cells | Latest year | Latest cities | Implied 2023 horizon |
| --- | ---: | ---: | ---: | ---: | ---: |
| resident / 常住人口（万人） | 692 | 25 | 2021 | 40/40 | 2 years |
| registered / 户籍人口（万人） | 707 | 10 | 2021 | 40/40 | 2 years |

The row counts and missing cells are source observations only; no missing value was filled.

## Cross-source target consistency

The selected internal consistency check was the age-structure sum:

    0-14 + 15-64 + 65+

The PDF describes these age fields as population-scale age groups. The comparison uses only 91 canonical age rows with all three age-band cells present. Partial rows and the six age quarantine rows were excluded, not filled or treated as zero.

Same-year results are in age_population_consistency_summary.csv:

| Candidate | Overlap | Mean absolute difference | Median absolute difference | Mean relative absolute difference | Paired closer count |
| --- | ---: | ---: | ---: | ---: | ---: |
| resident | 75 | 23.493333 | 10.000000 | 0.039913 | 69/75 |
| registered | 75 | 150.613333 | 70.000000 | 0.313752 | 6/75 |

On the 75 city-year pairs where both population columns were available, the age-band sum was closer to resident in 69 cases and closer to registered in 6 cases, with no tie. This materially strengthens resident as an internal consistency proxy.

The offset artifact also records a temporal-alignment diagnostic. Resident has a lower mean absolute difference at population year = age year + 1, while registered's best tested offset has a smaller overlap. These offsets are diagnostic only; no publication lag or official semantic shift is declared.

This is consistency evidence, not a proof of the competition target definition. The result remains conditional because the PDF explicitly names both population fields and only says total population.

## Target contract classification

The evidence supports:

    PRIMARY_PROXY_WITH_SENSITIVITY_TARGET

Operational meaning for the next S2 round:

- Use resident / 常住人口（万人） as the primary local target proxy.
- Re-run the key validation conclusion under registered / 户籍人口（万人） as a required sensitivity target.
- If resident and registered produce materially different validation conclusions, do not interpret either result as a proven official competition score.
- Do not relabel the target as RESOLVED.
- Keep the two-year target history horizon for both hypotheses because both latest observed target columns end in 2021 for the current snapshot.

## Target hypothesis impact on Plan02 geometry

target_hypothesis_origin_geometry.csv reuses Plan02 R1/R2 covariate freshness screens and changes only the target hypothesis.

Both scenarios give the same candidate set within each target:

- resident: strict candidates 2010-2021; conditional 2007-2009; excluded 2001-2006, 2022-2023.
- registered: strict candidates 2011-2021; conditional 2007-2010; excluded 2001-2006, 2022-2023.

The target choice therefore changes the origin feasibility set even though it does not change the reference horizon:

- origin 2010 is a resident candidate but only a registered conditional origin;
- origins 2011-2021 are candidates under both target hypotheses;
- 2009 is conditional under both;
- 2022-2023 cannot be scored because neither candidate has a target label there.

target_hypothesis_origin_comparison.csv provides the row-level proof. R1 and R2 covariate policies were not redesigned in this Plan.

## Metric evidence and conflict

metric_evidence_conflict_table.csv records:

1. Formal evaluation section:
   - metric: MSE;
   - direction: lower is better;
   - transform: score = 1/(1+MSE);
   - authority: formal evaluation section.
2. FAQ model-evaluation section:
   - metric: RMSE;
   - no replacement score formula;
   - authority: FAQ guidance.
3. Derived mathematical relation:
   - RMSE = sqrt(MSE);
   - inference only, not an authority decision.

## Evaluator sanity evidence

metric_sanity_results.csv has six passing tests:

- hand-check MSE;
- hand-check RMSE;
- hand-check competition score;
- RMSE = sqrt(MSE);
- same residual set and aggregation preserve ranking;
- cross-origin aggregation is not interchangeable.

The aggregation test uses two origins with unequal sample counts:

- macro-origin MSE ranks model A better: A=0.5, B=2.0;
- pooled city-year MSE ranks model B better: A=0.9, B=0.4;
- macro-origin RMSE and pooled RMSE reverse in the same way.

Therefore MSE/RMSE are monotonic equivalents only when residual set and aggregation are identical. Average RMSE, macro-origin MSE and pooled city-year MSE are different contracts and cannot be silently mixed.

## Local metric contract

local_target_metric_contract.csv records the following local contract:

- Primary metric: MSE, lower is better, evaluated per pseudo-origin over city-year residual rows.
- Auxiliary metric: RMSE, lower is better, computed on the same residual set for interpretability in population units.
- Competition score: 1/(1+MSE), higher is better, reportable only after the relevant MSE aggregation is explicitly named.
- Sample unit: one city-year target row within one pseudo-origin.
- Cross-origin aggregation: UNRESOLVED_DEFERRED. The next backtest architecture must choose between, at minimum, pooled city-year weighting and macro-origin weighting; it must not default to averaging incompatible metrics.
- Formal model scoring: not run in this Plan.

## S1/S2 boundary preservation

- test.xls remains only under data/quarantine/source/ and was not used as target, truth, availability or metric evidence.
- S1 canonical truth was not modified.
- Age remains benchmark-only; no annual rows or interpolation were created.
- Age quarantine rows were excluded from the consistency check and not reintroduced.
- Employee derived provenance remains present and counted at 91 rows.
- Income shift/invalidation warnings remain in the S1 contract.
- Missingness is not treated as zero.
- Publication timing remains unknown; current snapshot max years are not declared publication schedules.
- No formal baseline or model score was produced.

## Failures and deviations

- PyMuPDF was unavailable in the bundled Python runtime. This did not block the PDF investigation: pypdf extracted the embedded data-table JPEG, which was visually inspected with the local image viewer.
- A full S1 rebuild was not rerun; the Plan03 generator rechecked canonical key uniqueness and the relevant S1 contract/quarantine/provenance boundaries. No S1 files changed.
- No formal model scoring was executed, as required by the Plan non-goals.

## Remaining uncertainty

- The official target mapping from “总人口数” to resident versus registered remains unresolved.
- The relative age/population temporal alignment is a diagnostic, not an official lag.
- Source publication timing remains unknown.
- Cross-origin weighting/aggregation is deliberately not frozen.
- The MSE/RMSE wording conflict cannot be resolved further from the current PDF alone.
- Any future local score remains conditional and must not be presented as the official leaderboard score.

## Artifact locations

- target_evidence_matrix.csv - resident/registered evidence classes and target contract conclusion.
- age_population_consistency_detail.csv - city-year age-sum comparisons.
- age_population_consistency_summary.csv - same-year summary and paired closeness.
- age_population_consistency_offset.csv - temporal-offset diagnostics.
- target_hypothesis_origin_geometry.csv - target × scenario × origin feasibility.
- target_hypothesis_origin_comparison.csv - resident versus registered origin differences.
- target_hypothesis_summary.csv - candidate/conditional/excluded sets.
- metric_evidence_conflict_table.csv - PDF MSE/RMSE evidence and authority.
- metric_sanity_results.csv - evaluator correctness and aggregation tests.
- local_target_metric_contract.csv - machine-readable local contract.
- plan03_summary.json - summary flags and counts.
- scripts/build_s2_plan03_target_metric_contract.py - reproducible standard-library generator.

## Resulting commit

Artifact/implementation commit SHA: c8bd4f812fa2db65654fdb844f511e0f0852eb10. This metadata-only follow-up commit records the SHA; no history rewrite or force push is allowed.

## Interpretation requirement: next S2 question

With target and metric contracts explicit to the extent current evidence allows, the next highest-value Validation Architecture question is:

> Given resident-primary and registered-sensitivity targets, should S2 use a single out-of-time holdout or a multi-origin expanding backtest, and how should origins be weighted without changing the local MSE contract?

The next Plan should compare pooled city-year weighting versus macro-origin weighting under R1/R2, while reporting resident/registered sensitivity. This Plan does not execute that next round and does not enter S2-Plan04.
