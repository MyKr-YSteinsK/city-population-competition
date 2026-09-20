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
│   ├── build_s4_plan01_ridge_delta_dynamics.py
│   ├── build_s4_plan02_urbanization_residual.py
│   ├── build_s4_plan03_hgb_interaction.py
│   ├── build_s4_plan04a_density_residual.py
│   ├── build_s4_plan04b_wage_screen.py
│   ├── build_s4_plan04c_employment_signal.py
│   ├── build_s4_plan05a_error_memory.py
│   ├── build_s4_plan05b_cross_city_relative.py
│   ├── build_s4_plan05c_multisource_joint.py
│   ├── build_s5_plan04_full_multisource_feature_foundation.py
│   ├── build_s5_plan04b_training_replay_foundation.py
│   ├── build_s5_plan07_wave_consolidation.py
│   ├── build_s6_plan01_final_solution_contract_audit.py
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
    └── S4_HANDOFFS/S4-Plan01/
        ├── HANDOFF.md
        ├── feature_manifest.csv
        ├── feature_missingness_imputation.csv
        ├── modeling_dataset_diagnostics.csv
        ├── training_geometry.csv
        ├── aggregate_metrics.csv
        ├── per_origin_metrics.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── pairwise_b3_comparison.csv
        ├── ablation_comparison.csv
        ├── latest_origin_comparison.csv
        ├── city_improvement_concentration.csv
        ├── leave_one_origin_out.csv
        ├── leave_one_city_sensitivity.csv
        ├── leakage_audit.csv
        ├── r1_r2_equivalence_check.csv
        ├── b3_compatibility_check.csv
        ├── b3_metric_compatibility_check.csv
        └── plan01_summary.json
    └── S4_HANDOFFS/S4-Plan02/
        ├── HANDOFF.md
        ├── feature_manifest.csv
        ├── urbanization_information_boundary.csv
        ├── r1_r2_feature_freshness_comparison.csv
        ├── modeling_dataset_diagnostics.csv
        ├── training_geometry.csv
        ├── feature_missingness_imputation.csv
        ├── aggregate_metrics.csv
        ├── per_origin_metrics.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── pairwise_b3_comparison.csv
        ├── bias_control_comparison.csv
        ├── level_vs_dynamics_ablation.csv
        ├── latest_origin_comparison.csv
        ├── city_improvement_concentration.csv
        ├── leave_one_origin_out.csv
        ├── leave_one_city_sensitivity.csv
        ├── ridge_coefficient_diagnostics.csv
        ├── residual_feature_relationship.csv
        ├── leakage_audit.csv
        ├── b3_compatibility_check.csv
        ├── b3_metric_compatibility_check.csv
        └── plan02_summary.json
    └── S4_HANDOFFS/S4-Plan03/
        ├── HANDOFF.md
        ├── feature_manifest.csv
        ├── model_definitions.csv
        ├── modeling_dataset_diagnostics.csv
        ├── training_geometry.csv
        ├── feature_missingness_imputation.csv
        ├── aggregate_metrics.csv
        ├── per_origin_metrics.csv
        ├── per_origin_predictions_or_residuals.csv
        ├── pairwise_b3_comparison.csv
        ├── feature_ablation_comparison.csv
        ├── latest_origin_comparison.csv
        ├── r1_r2_equivalence_check.csv
        ├── r1_r2_urbanization_sensitivity.csv
        ├── city_improvement_concentration.csv
        ├── leave_one_origin_out.csv
        ├── leave_one_city_sensitivity.csv
        ├── leakage_audit.csv
        ├── b3_compatibility_check.csv
        ├── b3_metric_compatibility_check.csv
        ├── th_feature_compatibility_check.csv
        ├── urbanization_information_boundary.csv
        └── plan03_summary.json
    └── S4_HANDOFFS/S4-Plan04A/
        ├── HANDOFF.md
        ├── density_boundary_audit.csv
        ├── plan04a_summary.json
        ├── screening_metrics.csv
        └── build_s4_plan04a_density_residual.py (source in scripts/)
    └── S4_HANDOFFS/S4-Plan04B/
        ├── HANDOFF.md
        ├── plan04b_summary.json
        ├── screening_metrics.csv
        ├── wage_boundary_audit.csv
        └── build_s4_plan04b_wage_screen.py (source in scripts/)
    └── S4_HANDOFFS/S4-Plan04C/
        ├── HANDOFF.md
        ├── employment_provenance_boundary_audit.csv
        ├── plan04c_summary.json
        ├── screening_metrics.csv
        └── build_s4_plan04c_employment_signal.py (source in scripts/)
    └── S4_HANDOFFS/S4-WaveA/
        └── WAVE_A_SUMMARY.md
    └── S4_HANDOFFS/S4-WaveB/
        ├── WAVE_B_SUMMARY.md
        └── s4_final_summary.json
    └── S4_FEATURE_ML_MODELING_FINAL_REPORT.md
    └── S5_HANDOFFS/
        ├── S5_MAIN_HANDOFF.md
        ├── S5_EXPERIMENT_REGISTRY.csv
        ├── S5_FINAL_CANDIDATE_COMPARISON.csv
        ├── S5_EVIDENCE_DRIVEN_ITERATION_FINAL_REPORT.md
        └── S5-Plan04/ and S5-Plan04B/ derived feature/replay foundations
    └── S6_HANDOFFS/S6-Plan01/
        ├── HANDOFF.md
        ├── FINAL_SOLUTION_CONTRACT.md
        ├── final_solution_contract.json
        ├── dependency_map.md
        ├── rebuild_manifest.csv
        ├── current_2023_information_set_audit.csv
        ├── final_training_geometry.csv
        ├── historical_lineage_hashes.csv
        ├── submission_schema_evidence.md
        └── environment_snapshot.json
```

## Generated and excluded

`.venv/`、Python cache、临时压缩包、未授权模型/预测缓存和外部数据不属于当前交付树。`data/normalized/`、quarantine/provenance/regime CSV、Plan04/04B feature/replay matrix、future model/prediction artifacts 均属于可重建派生物；7 个合法 XLSX、官方 PDF、Git history/tags、S5 historical identity 和 S6 contract evidence 必须保留。`data/quarantine/source/test.xls` 可以保留文件本体，但其内容永远是 forbidden input。
