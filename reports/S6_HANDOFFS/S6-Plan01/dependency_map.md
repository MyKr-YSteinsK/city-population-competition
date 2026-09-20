# S6-Plan01 Dependency Map

```text
official PDF + 7 legal raw XLSX
  -> S1 contract / canonical / quarantine / provenance / regime guards
  -> S2 source-specific R1/R2 geometry and current-information policy
  -> S3 B3 implementation and compatibility evidence
  -> S5 Plan04 144-feature contract
  -> S5 Plan04B replay matrix and T-2 split geometry
  -> historical Plan06E archive tag archive/pre-s6/s5-plan06e-delta-growth-framing
  -> S6 F4 contract audit (this packet)
  -> later clean rebuild (not executed here)
```

## Current source-of-truth files

- S1 source contract: `docs/s1_data_contract.csv`
- S2 geometry: `scripts/build_s2_plan02_geometry.py` and `reports/S2_HANDOFFS/S2-Plan02/`
- B3: `scripts/build_s3_plan02_recent_window_and_selector.py`
- Plan04 feature names: `reports/S5_HANDOFFS/S5-Plan04/feature_definitions.csv`
- Plan04B model contract: `reports/S5_HANDOFFS/S5-Plan04B/parallel_worker_input_contract.json`
- Plan04B replay: `reports/S5_HANDOFFS/S5-Plan04B/full_multisource_training_replay_matrix.csv`
- F4 historical implementation: immutable Plan06E artifact tip `343970c608fa1a0f6d5e7ab5d355c29757d6ce2e`; runner blob is recorded in `historical_lineage_hashes.csv`.

## Boundary

The current Plan04/04B CSVs are reproducible derived artifacts, not new raw truth. The historical Plan06E branch artifact is preserved for implementation lineage only; it is not silently cherry-picked into current main. No model fit or 2023 output is part of this audit.
