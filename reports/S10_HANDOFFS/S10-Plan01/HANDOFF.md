# HANDOFF — S10-Plan01 外部脚本不可变接收、原样复现与 675 口径取证

## Status

`COMPLETE` for the bounded S10-Plan01 scope. No model optimization, parameter search, S8 truth access, real-2022/real-2023 target access, S8/S9 merge, S6 artifact mutation, or checkpoint mutation was performed.

## Repository identity

- Repository: `MyKr-YSteinsK/city-population-competition`
- Base SHA: `0f8479f740ca938acc5ff2d7026d22ca1cdd1755`
- `origin/main` at preflight: `0f8479f740ca938acc5ff2d7026d22ca1cdd1755`
- Branch: `codex/s10-plan01-external-script-forensics`
- Final commit and push status: recorded in the final TASK_RESULT after delivery; this handoff avoids a self-referential commit hash.
- Main/S8/S9 branches were not modified or merged.

## Original identity and immutable intake

- Source: `D:\CS\temp\实验-城市人口\forecast_population.py`
- Intake: `reports/S10_HANDOFFS/S10-Plan01/intake/original/forecast_population.py`
- Byte size: `20738`
- LF-split line count: `441`
- SHA256: `f719fa577d2c5b8fd51a024af0dd084390f849e7c3564a34ac816a4b35fb8342`
- Expected hash match: `YES`
- Original source modified: `NO`
- Syntax check: `PASS`

## Legal raw input boundary

The disposable runtime copied exactly these seven current-project official raw XLSX files: 人口规模、人口密度、城镇化率、就业信息、工资水平、年龄结构、生活水平. Every source/runtime pair is byte-identical to the S2-Plan04 after-hash evidence and every runtime glob had exactly one match. `test.xls` was not opened, hashed, copied, or present in the runtime. Full path and hash evidence is in `runtime/raw_input_manifest.csv`.

## Exact reproduction

- Exact command: `python forecast_population.py`
- Environment: Windows 11, Python 3.14.6, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, openpyxl 3.1.5, no virtualenv.
- Run A: outputs generated, stdout/stderr saved; outer host timeout left exit code unobserved.
- Run B: exit code `0`, 65.847 seconds, stdout 7027 bytes, stderr 0.
- Extra Run C: outputs generated, stdout/stderr saved; outer host timeout left exit code unobserved.
- All three stdout files are byte-identical.
- All three resident submissions have SHA256 `023b711c07d9efb72fdb1cf697195f50b8e4e68dd5cb94644c630b2c5e341676`.
- All three registered submissions have SHA256 `6ea774709b3712faad0e9fab587f8dde5d46b52f3112221b860a3bbfd0df70ea`.
- Reproducibility verdict: `deterministic_exact`.

## Independent validation and `675`

The forensic harness imported the immutable runtime copy and called the original functions without modifying them. It saved 4,760 validation rows across resident/registered and 2017–2021; 24 target×method metric rows have five origins and 40 cities where finite. The independent pooled MSE, RMSE, MAE, MedAE and bias are in `forensics/independently_recomputed_metrics.csv`; per-origin values are in `forensics/per_origin_metrics.csv`.

- Original script printed MSE: `NO`.
- Original script printed RMSE: `YES`; it printed `n`, `mae`, `medae`, `rmse`.
- Independent identity: `sqrt(pooled MSE) == script RMSE` within `5.7e-14` maximum absolute difference; MAE and MedAE match exactly.
- Pre-registered `675` search: no defensible exact/near provenance from the original method-level MSE, macro-origin MSE, latest-origin MSE, RMSE-vs-MSE, rounded-prediction, MAE-ranking, or apparent-selector calculations.
- Nearest diagnostic is resident `NON_NESTED_APPARENT_SELECTOR = 687.5360436617348`; it is explicitly invalid for unbiased historical comparison.
- Nearest legitimate method-level pooled/macro value is registered `model_random_forest = 653.6645536511874`.
- Finding: `675 provenance unresolved`; it may come from an external evaluator, another script, hidden target, or post-hoc computation, but this Plan does not guess which.

## Script contract vs project contract

`forensics/script_vs_project_contract.md` and `.csv` record the full comparison. Shared facts are horizon 2 and target history through T-2. The contracts differ on origins (5 vs 11), primary aggregate (printed RMSE/MAE vs pooled MSE), stability guardrails, source-cap/provenance semantics, target mutation, and same-sample city selector. Therefore `DIRECTLY_COMPARABLE_675_TO_F4 = NO`.

## Initial leakage/source-handling findings

The machine matrix is `forensics/initial_leakage_audit.csv`.

- `FAIL`: hard-coded city40/2021 registered repair; runtime value 65.0 became 270.5 from the last-two prior median, while resident was unchanged.
- `FAIL`: same 2017–2021 city records select a method by MAE and are reused for the apparent selector diagnostic; this is not an unbiased historical score.
- `FAIL`: validation-target-informed post-processing affects the registered 2021 scoring label.
- `UNKNOWN`: T-1 exogenous as-of values cannot be mapped to project R1/R2 source-specific publication caps from script code alone.
- `UNKNOWN`: source-local duplicate/groupby semantics differ from canonical provenance semantics; employees had 72 duplicate city-year groups, each with two rows.
- `PASS`: target-history direction, train-only imputer/scaler statistics, no random split, no `test.xls`, no B-data, no S8 truth, no external future population, and no best-seed search.

The source audit found that the external employment parser averages duplicate employee rows by city/year. It also found the six living-standard sheets are not actually ingested: `_long_income_tables()` treats the title row as the year row and returns zero frames, while the S1 quarantine block has 32 exact values in the raw workbook. Age values are carried forward from sparse benchmark years by `_asof()`; exact counts are in `forensics/source_handling_runtime.json`.

## Artifacts

- `intake/intake_manifest.json`
- `intake/original/forecast_population.py`
- `runtime/raw_input_manifest.csv`
- `runtime/environment.json`
- `runtime/pip_freeze.txt`
- `runtime/reproduction_manifest.json`
- `runtime/run_a_stdout.txt`, `run_a_stderr.txt`, `run_b_stdout.txt`, `run_b_stderr.txt`, `run_c_stdout.txt`, `run_c_stderr.txt`
- `runtime/submission_resident_run_{a,b,c}.csv`
- `runtime/submission_registered_run_{a,b,c}.csv`
- `forensics/static_code_forensics.md`
- `forensics/dataflow_inventory.csv`
- `forensics/validation_predictions_original_semantics.csv`
- `forensics/script_printed_metrics.csv`
- `forensics/independently_recomputed_metrics.csv`
- `forensics/per_origin_metrics.csv`
- `forensics/675_pre_registered_search.csv`
- `forensics/selector_optimism_diagnostic.csv`
- `forensics/script_vs_project_contract.md/.csv`
- `forensics/initial_leakage_audit.csv`
- `forensics/source_handling_audit.md`
- `forensics/source_handling_runtime.json`
- `forensics/runtime_repair_evidence.json`

## Recommendation for S10-Plan02

Only after Main/S10 review should a follow-up consider same-contract 11-origin re-evaluation, nested/prior-only selector audit, and controlled ablation. No such work was started here.
