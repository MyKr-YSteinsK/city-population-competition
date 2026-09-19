# S2-Plan02 Handoff | Real-Like Forecast Geometry and Source-Specific Replay

## Result

本轮完成了 source-specific historical replay geometry 审查，没有进入正式 baseline、模型训练、模型比较、评分、预测或 submission。

主要结论：当前 2023 snapshot 支持一个条件性的两年 target-history horizon，但不支持把所有 source 统一描述为 T-1 或 T-2。在保留 S1 provenance / quarantine 边界的前提下，Plan02 建立了两个显式 replay scenario，并重新评估了 historical origins。

## Starting repository

- Repository: MyKr-YSteinsK/city-population-competition
- Branch: main
- Starting remote/main and local HEAD: 9f12658629869e7e470fe36a0d360fe65f77ba45
- Starting working tree: clean
- Execution date: 2026-09-19

## Actual changes

- 新增 scripts/build_s2_plan02_geometry.py。
- 新增 reports/S2_HANDOFFS/S2-Plan02/ 下的 reference geometry、scenario definitions、aggregate/city-level freshness matrix、origin feasibility、Plan01 comparison、heuristic review 和 summary JSON。
- 脚本只读取 S1 contract、canonical CSV、provenance/quarantine CSV 和 Plan01 artifacts；不读取 raw workbook 或 test.xls。
- 没有修改 S1 canonical CSV、quality registry、quarantine 内容或任何原始 XLSX/PDF。

## Commands actually executed

    git status --short
    git branch --show-current
    git ls-remote origin refs/heads/main
    python -m py_compile scripts/build_s2_plan02_geometry.py
    python scripts/build_s2_plan02_geometry.py
    git diff --check

PDF evidence was read from all four pages with the bundled pypdf runtime. The system pdftotext command was unavailable, so no claim relies on that missing command.

Actual generator result:

    S2_PLAN02_GEOMETRY_PASS
    reference_geometry_rows=14
    scenario_rows=2
    aggregate_matrix_rows=644
    city_matrix_rows=25760
    feasibility_rows=46
    candidate_years_by_scenario=R1/R2: 2010-2021
    conditional_years_by_scenario=R1/R2: 2007-2009
    target_history_horizon_years=2
    test_xls_used=False
    model_scoring_run=False

The script also passed AST compilation and its S1 boundary assertions: 14 canonical source rows, unique canonical keys, age benchmark domain, employee provenance count, income invalidation/label-repair boundaries, quarantine row counts, and the test.xls boundary.

## Target geometry and evidence

### Direct source evidence

- The original PDF page 1 says both preliminary A and B tasks predict the total population of each city for 2023.
- The PDF does not directly choose between 常住人口（万人） and 户籍人口（万人） as the local label column. The source workbook contains both columns.
- The PDF page 4 states MSE as the competition reference and gives score = 1/(1+MSE), while its FAQ also mentions RMSE. This conflict remains unresolved and was not changed by this Plan.

### Conditional target proxy

The Plan01 proxy population / 常住人口（万人） is retained only as a conditional temporary target contract. It has useful source coverage, but the source materials do not uniquely prove that it is the competition's “total population” label. 户籍人口（万人） remains present in the canonical source and was not discarded.

Under this proxy, the latest non-empty target-label year is 2021. Therefore the current 2023 snapshot has:

    latest observed target history = 2021
    reference target year = 2023
    target-history horizon proxy = 2 years

The horizon is an observed snapshot-derived boundary under the conditional proxy, not a publication-timing fact. Historical replay therefore exposes target labels only through T-2; the label at T is reserved for retrospective scoring and is not used as an input.

## 2023 reference forecast geometry

reference_forecast_geometry.csv records, for all 14 logical sources:

- canonical value-year domain and latest usable year;
- observed snapshot freshness lag relative to 2023;
- latest-year city coverage and missing-city list;
- annual/source-defined versus benchmark-only domain;
- derived/provenance-sensitive latest information;
- invalidation/quarantine counts and S1 quality warnings;
- publication timing status and the fact/heuristic boundary.

The observed latest-year pattern is:

| Source group | Latest usable year | Snapshot lag | Notes |
| --- | ---: | ---: | --- |
| population target proxy | 2021 | 2 | 40/40 cities for the resident label |
| density, urbanization, unemployment, employees | 2022 | 1 | latest-year city coverage is source-specific; employees has 91 derived rows elsewhere |
| sector, wage | 2021 | 2 | quality/provenance boundaries remain active |
| age | 2020 | 3 | benchmark-only {2000, 2010, 2020}; no annual rows inferred |
| income/consumption sources | 2022 | 1 | asymmetric coverage and income-shift/invalidation warnings remain |

These latest years are observed canonical snapshot facts. They are not proven source publication years.

## Replay scenarios

Definitions are machine-readable in replay_scenario_definitions.csv.

### R1 - R1_REFERENCE_SNAPSHOT_LAG

This is the primary real-like replay proxy. For each source, the required historical cap is T minus observed_snapshot_lag_for_that_source. The population target history is separately capped at T-2. Age remains benchmark-only and is filtered by the cap without interpolation or annualization.

This scenario simulates the current snapshot's relative freshness pattern. It does not claim that the historical source publication schedule followed the same pattern.

### R2 - R2_CONSERVATIVE_PUBLICATION_STRESS

This is a sensitivity scenario for unknown publication timing. It adds one extra year of lag only to annual sources whose observed 2023 snapshot lag is one year, i.e. sources whose latest usable canonical value is in 2022. Sources already at a two-year lag keep that boundary; the target history stays at T-2; age keeps its three-year benchmark-aware boundary.

This is intentionally different from blindly applying a universal global T-2 rule. In particular, target history and benchmark-only age retain their own geometry. It remains an availability assumption, not a publication fact.

## Freshness-aware origin feasibility

origin_source_scenario_freshness_matrix.csv contains 644 source-level rows. origin_city_source_scenario_freshness.csv contains 25,760 city-level rows with required cap, actual latest year, staleness, missingness, exact-cap state, derived-row flag and quarantine boundary count.

The source-level freshness check reports exact-cap coverage separately from any-value-as-of-cap coverage. A prior value is classified as stale; it is never promoted to “fresh” merely because it exists. Benchmark age is classified as irregular/domain-aware rather than forced into an annual exact-year rule.

For the 2023 reference origin, examples show the scenarios are not equivalent:

- Density: R1 requires 2022 and has 20 cities exactly at 2022 plus 20 stale prior values; R2 requires 2021 and has 27 exact plus 13 stale.
- Employees: R1 requires 2022 and has only 2 cities exactly at 2022 plus 38 stale; R2 requires 2021 and has 34 exact plus 6 stale. The 91 derived employee rows remain provenance-sensitive.
- Age: both scenarios require the 2020 benchmark boundary and retain domain-aware handling rather than manufacturing annual freshness.

The structural feasibility screen uses the Plan01 thresholds only as diagnostics:

- target label coverage at least 90%;
- at least 3 target labels per city through T-2;
- all six provisional core sources at least 90% city coverage as of their scenario-specific cap.

These are not final validation or scoring contracts.

## Origin changes versus Plan01

plan01_vs_plan02_candidate_origin_comparison.csv compares both Plan01 policies with both Plan02 scenarios.

- Plan01 P1/P2 candidate set: 2009-2021.
- Plan02 R1/R2 strict real-like candidate set: 2010-2021.
- 2009 becomes conditional because target history through T-2 reaches only 11/40 cities with at least three labels; Plan01 had effectively allowed the one-step T-1 history shortcut.
- 2007-2008 remain conditional in Plan02 because target labels exist but target-history and/or core coverage is incomplete under the structural screen. They are not promoted to candidates.
- 2001-2006 and 2022-2023 are excluded; 2022/2023 lack the resident target label required for retrospective scoring.
- R1 and R2 happen to have the same strict candidate years under these structural screens, but their source-level exact/stale/missing patterns differ materially. Stable candidate years therefore do not prove that the availability scenarios are equivalent.

## Plan01 conclusions retained and corrected

Retained:

- source-level observed coverage, missingness and irregular domains;
- P1/P2 as explicit assumptions rather than facts, now kept only for comparison;
- the resident population proxy as a temporary, clearly conditional label;
- S1 canonical/provenance/quarantine boundaries and the exclusion of test.xls;
- no claim that current observed max years prove publication timing.

Corrected:

- Plan01's P1 is no longer described as genuinely source-specific freshness geometry; its global T-1 rule is only a legacy comparison.
- Target history is now explicitly two years under the resident proxy, so pseudo-tests do not silently become one-step target forecasts.
- Historical origin screening now considers per-source required caps, exact-cap freshness, stale prior values and city-level missingness instead of only any old value before a global cutoff.

## Plan01 heuristic review

screening_heuristics_review.csv records the treatment of the 90% coverage threshold, the 3-history-label threshold and the six-source core set.

- 90% coverage remains a structural diagnostic only.
- Three prior labels are now counted only at or before T-2; this corrects the target horizon but does not create a final contract.
- The six core sources remain a provisional comparability set inherited from Plan01. No model evidence, importance evidence or independence evidence was used to freeze them as a final source contract.

## S1 boundary preservation

- Age remains restricted to benchmark years {2000, 2010, 2020}.
- Employee derived rows remain separate from raw/source observations through the provenance artifact.
- Quarantined or invalidated values remain unavailable; they are never backfilled or recoded as zero.
- Overall/urban income shift and independence warnings remain visible.
- Absence remains distinct from numeric zero.
- data/quarantine/source/test.xls was checked only as a quarantine boundary; it did not enter truth, availability inference or policy construction.
- No S1 canonical or quality-contract file was changed.

## Failures and deviations

- The system pdftotext binary was unavailable. The bundled pypdf reader successfully extracted all four PDF pages, including the relevant target and evaluator passages.
- A full S1 rebuild was not rerun in this Plan. The Plan02 generator re-read the S1 contract and canonical artifacts and asserted the key S1 boundaries; the previously committed S1 raw/output evidence remains the separate rebuild evidence.
- No model or scoring run was performed, by design.

## Remaining uncertainty

- The source materials still do not uniquely resolve resident versus registered population as the competition target.
- Source publication timing and actual 2023 forecast-origin availability remain unknown.
- The PDF's MSE/RMSE wording conflict remains a separate metric-contract question.
- Whether employee derived rows should be included in a later model information set remains open; this Plan only preserves the distinction.
- Income shift semantics, invalidated truth and asymmetric source coverage remain unresolved S1 boundaries.
- R1/R2 are replay assumptions. Their stable candidate-year set is not evidence of unbiased model scores or equivalence of information sets.

## Artifact locations

- reference_forecast_geometry.csv - 2023 source-level geometry and fact/assumption boundaries.
- replay_scenario_definitions.csv - R1/R2 machine definitions.
- origin_source_scenario_freshness_matrix.csv - source-level aggregate freshness matrix.
- origin_city_source_scenario_freshness.csv - city-level latest-year/staleness evidence.
- origin_feasibility_summary.csv - scenario-level candidate/conditional/exclude decisions.
- plan01_vs_plan02_candidate_origin_comparison.csv - direct origin-set comparison.
- screening_heuristics_review.csv - Plan01 threshold/source-set review.
- geometry_summary.json - machine summary and boundary flags.
- scripts/build_s2_plan02_geometry.py - reproducible standard-library generator.

## Resulting commit

Artifact/implementation commit SHA: to be recorded after the first delivery commit. Any later metadata-only commit will be reported separately; no history rewrite or force push is allowed.

## Interpretation requirement: next S2 question

The next most valuable question is whether an expanding/out-of-time backtest structure produces stable conclusions when the exact same origins are replayed under R1 versus R2, while preserving target horizon and source provenance boundaries.

- The repository now has enough geometry to design that expanding/out-of-time structure.
- Actual availability-sensitivity scoring is not yet ready to be treated as a final result, because the target proxy and MSE/RMSE metric contract remain unresolved.
- The next Plan should first make the local target/metric contract explicit, then run the smallest backtest comparison under R1/R2. This Plan does not execute that next round.
