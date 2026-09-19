# S2-Plan04 Handoff | Official Population Source Reconciliation

## Result

本轮完成官方人口 source reconciliation。Plan04 只吸收用户提供且 SHA256 已核验的官方 `人口规模.xlsx`，没有替换其他 6 个 raw XLSX，没有重写 Plan01–03 历史 handoff，也没有进入模型评分或 S3。

结论是：人口 source 事实发生了局部、可追溯的变化，但现有 S2 validation contract 与核心 candidate-origin 结论没有发生实质改变。当前 source-consistent 状态已经具备进入下一轮 out-of-time / expanding backtest architecture comparison 的前置条件；下一轮仍须保留 resident/registered sensitivity，并显式决定跨 origin 聚合方式。

## Repository and source identity

- Repository: `MyKr-YSteinsK/city-population-competition`
- Branch: `main`
- Starting remote/local HEAD after `git fetch origin main` and `git pull --ff-only origin main`: `a394ea204ddcc26a5767a34bf47ede37fa4c9e64`
- Required official source: `D:\Downloads\official_population.xlsx`
- Required SHA256: `9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0`
- Actual download SHA256: `9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0`
- Historical handoffs under `reports/S2_HANDOFFS/S2-Plan01/` through `S2-Plan03/` were clean before focused recomputation and were not written by this Plan.

Machine-readable hash evidence is in `raw_source_hash_evidence.csv`. It proves that only `data/raw/城市人口分析与预测/数据集/人口规模.xlsx` changed; the other six raw XLSX hashes are identical before and after.

## Actual execution

Key commands actually run:

1. `git fetch origin main`
2. `git pull --ff-only origin main`
3. SHA256 and workbook structure check for `D:\Downloads\official_population.xlsx`.
4. Binary replacement of only `data/raw/城市人口分析与预测/数据集/人口规模.xlsx`.
5. Full `scripts/run_s1.py` rebuild.
6. Targeted generator correction in `scripts/run_s1.py`: QI-004 population missing counts now derive from the regenerated canonical population instead of retaining the old hardcoded counts.
7. Full `scripts/run_s1.py` rebuild again; it passed with `S1_REBUILD_PASS`, `raw_hash_equal=True`, and `output_hash_equal=True`.
8. `scripts/build_s2_plan04_official_population_reconciliation.py`; it passed with `S2_PLAN04_OFFICIAL_POPULATION_RECONCILIATION_PASS`.

The first post-source-refresh invocation of `run_s1.py` reached all domain audits but stopped at its old rebuild invariant because the intended population canonical output had changed. The second full invocation from the refreshed state passed and verified reproducibility. This deviation is recorded rather than treated as a clean first-attempt pass.

## Canonical population before/after

| Fact | Before | After | Interpretation |
|---|---:|---:|---|
| Population raw/canonical rows | 717 | 718 | One official city-year row added |
| Resident non-empty observations | 692 | 693 | `city40-2001` resident label added |
| Registered non-empty observations | 707 | 707 | No registered value added |
| Resident canonical missing cells | 25 | 25 | Existing resident missingness unchanged |
| Registered canonical missing cells | 10 | 11 | New row keeps registered population missing |
| Total population canonical missing cells | 35 | 36 | Contract/quality registry refreshed |
| Resident 2001 label count | 21 | 22 | Early target coverage improves by one city |
| Registered 2001 label count | 16 | 16 | Registered target coverage unchanged |
| Resident 2010 T-2 history-ready cities | 37 | 38 | One additional city passes the 3-label structural screen |
| Population unique city-year keys | 717 | 718 | Unique-key invariant remains true |

The new canonical row is verified as:

```text
city40, 2001, 常住人口（万人）=284, 户籍人口（万人）=missing
```

No zero-sentinel rule changed, and no value was filled. The refreshed S1 contract reports `row_count=718`, `raw_row_count=718`, `actual_key_count=718`, `missing_cell_count=36`, and `raw_missing_cell_count=27`. QI-004 now reports resident missing `25` and registered missing `11` while preserving the same source-coverage decision.

## Plan01–Plan03 impact

The focused recomputation used the refreshed canonical population in memory and reused the unchanged non-population Plan02 freshness evidence. The full machine-readable before/after record is `s2_population_impact_delta.csv`; current recomputed facts are in `s2_recomputed_facts.csv`.

### Plan01

- Population inventory changes from 717 to 718 canonical/value rows and missing value cells from 35 to 36.
- Resident 2001 target-label count changes from 21 to 22.
- Resident 2010 prior-history-ready count changes from 37 to 38.
- P1 and P2 candidate-origin sets do not change.
- No Plan01 candidate classification row changes.

### Plan02

- Resident 2023 target horizon remains 2 years because the latest usable resident year remains 2021.
- Resident R1 candidate origins remain `2010–2021`.
- Resident R2 candidate origins remain `2010–2021`.
- Conditional and excluded origin sets remain unchanged.

### Plan03

- Resident target evidence count changes from 692 to 693.
- Registered target evidence count remains 707.
- Resident and registered candidate-origin sets remain unchanged:
  - resident R1/R2: `2010–2021`;
  - registered R1/R2: `2011–2021`.
- Same-year age-sum evidence remains unchanged. Age benchmark domain and six-row quarantine boundary are preserved.
- The new row does not overturn the target conclusion:
  - `PRIMARY_PROXY_WITH_SENSITIVITY_TARGET`;
  - resident as primary proxy;
  - registered as sensitivity target.
- MSE remains primary, RMSE remains auxiliary, and `UNRESOLVED_DEFERRED` remains the cross-origin aggregation status.
- Employee provenance remains 91 rows; income warnings and `test.xls` quarantine remain unchanged.

The official population correction therefore changes source coverage and one early-history diagnostic, but does not change the existing S2 validation contract or core validation conclusions.

## Required boundary checks

- Only the population raw workbook changed: passed.
- Other six raw XLSX content unchanged: passed.
- Canonical population generated through `audit_population.py` via the S1 runner: passed.
- `city40-2001` resident 284 and registered missing: passed.
- Population `(city, year)` key uniqueness: passed.
- Updated S1 population contract and QI-004 evidence: passed.
- Full S1 rebuild after source refresh: passed.
- Plan01–03 historical handoffs not rewritten: passed.
- `test.xls` not used as a raw input: passed.
- Formal model scoring, backtest, prediction, submission, tuning and S3 work: not executed.

## Artifacts

- `population_source_reconciliation.csv` — official source identity and `city40-2001` check.
- `raw_source_hash_evidence.csv` — before/after hashes for all seven raw XLSX files.
- `population_before_after_delta.csv` — population source/canonical delta.
- `s1_population_contract_evidence.csv` — old/new population S1 contract rows.
- `s1_population_quality_evidence.csv` — old/new QI-004 rows.
- `s2_population_impact_delta.csv` — Plan01–03 affected-fact delta.
- `s2_recomputed_facts.csv` — current candidate origins and target/metric facts.
- `reproducibility_evidence.json` — scope, boundary and reproducibility assertions.
- `plan04_summary.json` — machine-readable Plan04 conclusion.
- `scripts/build_s2_plan04_official_population_reconciliation.py` — focused recomputation utility.

## Remaining uncertainty

- The official population source improves the repository’s source truth but still does not prove whether the competition’s “total population” target is resident or registered population.
- Publication timing remains unknown; R1/R2 remain replay assumptions rather than official calendars.
- Cross-origin aggregation remains unresolved and must be selected explicitly before interpreting local validation scores.
- No local score or leaderboard claim is made by this Plan.

## Resulting commit

Artifact/implementation commit SHA: to be recorded after the first delivery commit. Any later metadata-only commit will be reported separately; no history rewrite or force push is allowed.

## Interpretation requirement: next S2 question

Official population source reconciliation changes the authoritative population source and the early resident-history count, but it leaves the resident-primary/registered-sensitivity target contract, the two-year horizon, the R1/R2 candidate-origin sets, and the MSE/RMSE contract unchanged. The core S2 validation conclusions therefore remain valid as conditional conclusions against the refreshed source.

The project now has a source-consistent starting point for the next S2 architecture comparison. The next Plan may compare single holdout versus multi-origin expanding backtest and pooled versus macro-origin aggregation, while retaining resident/registered sensitivity. This Plan does not execute that next round and does not enter S2-Plan05 or S3.
