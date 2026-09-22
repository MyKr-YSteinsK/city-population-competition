# S10-Plan01 Source-Handling Audit

This is runtime evidence for `ORIGINAL_SCRIPT_SEMANTICS`; it does not repair or replace any source behavior.

## Runtime file selection

- Runtime raw file names: `人口密度.xlsx; 人口规模.xlsx; 城镇化率.xlsx; 就业信息.xlsx; 工资水平.xlsx; 年龄结构.xlsx; 生活水平.xlsx`.
- `test.xls` present in disposable runtime: `False`.
- The seven `_pick()` patterns each had their exact match list and hash recorded in `../runtime/raw_input_manifest.csv`.

## Employment duplicate handling

The external script reads three employment sheets and applies `groupby(['city', 'year'])[value_cols].mean(numeric_only=True)` before merging. The observed pre-aggregation duplicate evidence is:

| Sheet | Rows | Unique keys | Duplicate key groups | Extra duplicate rows | Max group size |
|---|---:|---:|---:|---:|---:|
| unemployment | 0 | 0 | 0 | 0 | 1 |
| employees | 762 | 690 | 72 | 72 | 2 |
| sectors | 698 | 698 | 0 | 0 | 1 |

This mean aggregation is the external script's behavior; it is not the S1 canonical/provenance pipeline and no quarantine or provenance flag is applied by the script.

## Living-standard sheet order and known S1 block

The external script assigns fixed variable names by sheet index rather than checking sheet names:

| Index | Actual sheet name | Script variable |
|---:|---|---|
| 0 | 人均可支配收入 | income_disposable |
| 1 | 人均消费支出 | income_consumption |
| 2 | 城镇居民消费支出 | income_urban_consumption |
| 3 | 农村居民消费支出 | income_rural_consumption |
| 4 | 城镇居民人均收入 | income_urban_disposable |
| 5 | 农村居民人均收入 | income_rural_disposable |

The S1 quarantine file `data/quarantine/生活水平_人均可支配收入_quarantine.csv` has 32 rows. A direct diagnostic of the raw first sheet finds 32 exact `(city, year, raw_disposableIncome)` values and 32 keys. However, the original `_long_income_tables()` parser produced 0 income frames with row counts `[]` and exog income columns `[]` because it treats raw row 0 as the year row. Therefore the known S1 block is not ingested by original semantics; it is absent because of the parser/header mismatch, not because the script implements the S1 quarantine boundary.

## Age sparse-year carry-forward

Observed age years by city are recorded in the machine-readable audit. Across the feature rows used by the original script, requested age values=2640, resolved=2580, stale as-of carry-forward=2226, missing=60. This follows `_asof()` and does not infer annual age observations from the benchmark-only source.

## Target mutation evidence

The exact city40 registered repair evidence is stored in `../forensics/runtime_repair_evidence.json` and repeated in the leakage matrix. This is recorded as a script behavior, not endorsed as a project correction.

## Contract difference

The source behavior above is local reshape/groupby/as-of behavior. The project contract keeps canonical values, provenance, quarantine, source-specific caps, and benchmark-domain boundaries explicit. No correction is made in Plan01.
