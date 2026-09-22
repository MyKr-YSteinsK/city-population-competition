# S8 Truth Provenance Hardening Report (candidate_truth_2023_v1)

Status: S8_TRUTH_V1_PROVISIONAL_FREEZE

This report covers S8-Plan02 truth provenance hardening only. No model was
trained, scored, ranked, or inspected, and no S9 artifact was read.

## Source hardening

- Reproduced truth rows before: 21
- Tier1 truth rows after hardening: 22
- Tier2 truth rows after hardening: 13
- Tier3 fallback truth rows after hardening: 5
- Tier4 fallback truth rows after hardening: 0
- Direct source upgrade queue rows: 16
- Rows retained without direct first-party value: 4

The remaining fallback rows are explicit in `truth_source_upgrade_queue.csv`.
The old reproduced source rows remain in `source_ledger.csv` as historical
fallback/cross-check records.

## City40

`city40 -> 镇江` moves from `PROVISIONAL` to `HIGH_CONFIDENCE`. The raw
registered value 65 is present in the legal XLSX and normalized CSV. It is not
corrected. The non-anomalous registered trajectory, resident trajectory,
urbanization and province fingerprint support Zhenjiang; the density conflict
and cause of 65 remain unresolved.

## City23 and versioning

The current v1 value is 245. A 2022 official-statistics-attributed Zhuhai
source gives 247.72, which renders to 248. The old chat value 248 is recorded
only as ignored trigger context, not evidence. This material rendered change
requires `candidate_truth_2023_v2`; v1 remains preserved.

- v1 evaluator SHA256: 026c2c4723eadad8ae8519586e0a146d47b1bd0525fde315aa4ec4994e86927d
- candidate_truth_2023_v1 evaluator SHA256: 026c2c4723eadad8ae8519586e0a146d47b1bd0525fde315aa4ec4994e86927d
- Rendered rows changed from v1: 0

## Freeze gate

The gate is not promoted to full freeze because four truth rows still lack a
directly retrieved Tier1 source in this pass, and the city40 source anomalies
remain explicit. No value was changed to improve any model result.

Isolation: S9 branch read NO; S9 predictions read NO; model scoring NO;
test.xls accessed NO.
