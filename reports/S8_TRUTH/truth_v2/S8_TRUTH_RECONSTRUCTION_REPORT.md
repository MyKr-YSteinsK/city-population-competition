# S8 Truth Provenance Hardening Report (candidate_truth_2023_v2)

Status: S8_TRUTH_PROVISIONAL_FREEZE_RETAINED

This report covers S8-Plan02 truth provenance hardening only. No model was
trained, scored, ranked, or inspected, and no S9 artifact was read.

## Source hardening

- Reproduced truth rows before: 21
- Tier1 truth rows after hardening: 22
- Tier2 truth rows after hardening: 14
- Tier3 fallback truth rows after hardening: 4
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
- candidate_truth_2023_v2 evaluator SHA256: de8ae0feb8b4061610b82a235a673fc2a841d2969f04924d7f69185c5e4db02c
- Rendered rows changed from v1: 1

## Freeze gate

The gate is not promoted to full freeze because four truth rows still lack a
directly retrieved Tier1 source in this pass, and the city40 source anomalies
remain explicit. No value was changed to improve any model result.

Isolation: S9 branch read NO; S9 predictions read NO; model scoring NO;
test.xls accessed NO.
