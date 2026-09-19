# Project File Map

```text
.
├── README.md
├── CURRENT_LEARNING_CHECKPOINT.md
├── requirements.txt
├── data/
│   ├── raw/城市人口分析与预测/数据集/       # 7 XLSX；同级 raw 目录保留参考 PDF
│   ├── normalized/                         # 14 canonical CSV + provenance/repair/regime
│   └── quarantine/                         # 6 quarantine CSV + source/test.xls
├── docs/
│   ├── CODEX_INSTRUCTIONS.md
│   ├── PROJECT_CONTEXT.md
│   ├── PROJECT_FILE_MAP.md
│   ├── S1_DATA_CONTRACT.md
│   ├── S1_FINAL_STATE.md
│   ├── S1_MAIN_HANDOFF.md
│   ├── s1_data_contract.csv
│   └── s1_quality_issue_registry.csv
├── scripts/
│   ├── audit_age_structure.py
│   ├── audit_employment.py
│   ├── audit_living_standard.py
│   ├── audit_population.py
│   ├── audit_population_density.py
│   ├── audit_urbanization.py
│   ├── audit_wage.py
│   ├── build_s2_plan01_matrix.py
│   ├── build_s2_plan02_geometry.py
│   ├── build_s2_plan03_target_metric_contract.py
│   ├── build_s2_plan04_official_population_reconciliation.py
│   ├── build_s2_plan05_validation_architecture.py
│   ├── build_s3_plan01_baseline_ladder.py
│   ├── build_s3_plan02_recent_window_and_selector.py
│   ├── build_s3_plan03_pooled_growth_and_ml_hurdle.py
│   └── run_s1.py
└── reports/
    ├── S1_FINAL_REVIEW_PACK.zip
    └── S2_HANDOFFS/S2-Plan01/
        ├── HANDOFF.md
        ├── policy_definitions.csv
        ├── source_temporal_inventory.csv
        ├── source_city_coverage.csv
        ├── pseudo_target_source_policy_matrix.csv
        ├── candidate_origin_summary.csv
        └── matrix_summary.json
    └── S2_HANDOFFS/S2-Plan02/
        ├── HANDOFF.md
        ├── reference_forecast_geometry.csv
        ├── replay_scenario_definitions.csv
        ├── origin_source_scenario_freshness_matrix.csv
        ├── origin_city_source_scenario_freshness.csv
        ├── origin_feasibility_summary.csv
        ├── plan01_vs_plan02_candidate_origin_comparison.csv
        ├── screening_heuristics_review.csv
        └── geometry_summary.json
    └── S2_HANDOFFS/S2-Plan03/
        ├── HANDOFF.md
        ├── target_evidence_matrix.csv
        ├── age_population_consistency_detail.csv
        ├── age_population_consistency_summary.csv
        ├── age_population_consistency_offset.csv
        ├── target_hypothesis_origin_geometry.csv
        ├── target_hypothesis_origin_comparison.csv
        ├── target_hypothesis_summary.csv
        ├── metric_evidence_conflict_table.csv
        ├── metric_sanity_results.csv
        ├── local_target_metric_contract.csv
        └── plan03_summary.json
    └── S2_HANDOFFS/S2-Plan04/
        ├── HANDOFF.md
        ├── raw_source_hash_evidence.csv
        ├── population_source_reconciliation.csv
        ├── population_before_after_delta.csv
        ├── s1_population_contract_evidence.csv
        ├── s1_population_quality_evidence.csv
        ├── s2_population_impact_delta.csv
        ├── s2_recomputed_facts.csv
        ├── reproducibility_evidence.json
        └── plan04_summary.json
    └── S2_HANDOFFS/S2-Plan05/
        ├── HANDOFF.md
        ├── validation_architecture_definitions.csv
        ├── naive_predictor_definitions.csv
        ├── origin_information_boundary.csv
        ├── leakage_audit.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── per_origin_metrics.csv
        ├── cross_origin_aggregation_comparison.csv
        ├── cross_origin_aggregation_weight_evidence.csv
        ├── aggregation_sanity_checks.csv
        ├── single_vs_multi_origin_comparison.csv
        ├── validation_stability_summary.csv
        ├── architecture_recommendation.csv
        └── plan05_summary.json
    └── S3_HANDOFFS/S3-Plan01/
        ├── HANDOFF.md
        ├── baseline_predictor_definitions.csv
        ├── validation_contract_evidence.csv
        ├── origin_information_boundary.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── per_origin_metrics.csv
        ├── aggregate_metrics.csv
        ├── pairwise_baseline_comparison.csv
        ├── pairwise_origin_comparison.csv
        ├── validation_stability_summary.csv
        ├── city_improvement_concentration.csv
        ├── improvement_concentration_summary.csv
        ├── leakage_audit.csv
        ├── r1_r2_equivalence_check.csv
        ├── s2_b0_b1_compatibility_check.csv
        └── plan01_summary.json
    └── S3_HANDOFFS/S3-Plan02/
        ├── HANDOFF.md
        ├── baseline_predictor_definitions.csv
        ├── selector_definitions.csv
        ├── validation_contract_evidence.csv
        ├── origin_information_boundary.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── per_origin_metrics.csv
        ├── aggregate_metrics.csv
        ├── pairwise_baseline_comparison.csv
        ├── pairwise_origin_comparison.csv
        ├── validation_stability_summary.csv
        ├── city_improvement_concentration.csv
        ├── improvement_concentration_summary.csv
        ├── selector_improvement_concentration_summary.csv
        ├── leakage_audit.csv
        ├── selector_history_audit.csv
        ├── r1_r2_equivalence_check.csv
        ├── s2_b0_b1_compatibility_check.csv
        ├── plan01_b0_b1_b2_compatibility_check.csv
        ├── slope_mechanism_diagnostics.csv
        ├── slope_disagreement_summary.csv
        ├── city_predictor_performance.csv
        ├── city_preference_stability.csv
        ├── target_preference_consistency.csv
        ├── oracle_predictions_or_residuals.csv
        ├── oracle_metrics.csv
        ├── oracle_assignment_distribution.csv
        ├── oracle_static_city_assignments.csv
        ├── late_origin_comparison.csv
        └── plan02_summary.json
    └── S3_HANDOFFS/S3-Plan03/
        ├── HANDOFF.md
        ├── baseline_predictor_definitions.csv
        ├── validation_contract_evidence.csv
        ├── origin_information_boundary.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── per_origin_metrics.csv
        ├── aggregate_metrics.csv
        ├── pairwise_baseline_comparison.csv
        ├── pairwise_origin_comparison.csv
        ├── validation_stability_summary.csv
        ├── latest_origin_comparison.csv
        ├── time_regime_comparison.csv
        ├── leave_one_origin_out_ranking.csv
        ├── leave_one_origin_out_summary.csv
        ├── city_improvement_concentration.csv
        ├── improvement_concentration_summary.csv
        ├── leave_one_city_sensitivity.csv
        ├── leakage_audit.csv
        ├── r1_r2_equivalence_check.csv
        ├── s2_b0_b1_compatibility_check.csv
        ├── plan01_b0_b1_b2_compatibility_check.csv
        ├── plan02_b0_b1_b2_b3_b4_compatibility_check.csv
        ├── pooled_fit_diagnostics.csv
        ├── local_vs_pooled_slope_comparison.csv
        ├── local_slope_dispersion.csv
        ├── slope_error_relationship.csv
        ├── growth_diagnostics.csv
        ├── additive_vs_growth_comparison.csv
        ├── ml_hurdle_candidate.csv
        ├── s3_sufficiency_candidate.json
        └── plan03_summary.json
```

## Generated and excluded

`.venv/`、Python cache、临时压缩包、历史 Plan01-04 报告和历史探索 runner 不属于当前交付树。S2 Plan handoff 只保留本轮审查所需的当前 artifacts，不进入 S3+ 模型或提交工作。
