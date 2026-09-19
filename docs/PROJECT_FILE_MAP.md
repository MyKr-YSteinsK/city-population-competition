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
```

## Generated and excluded

`.venv/`、Python cache、临时压缩包、历史 Plan01-04 报告和历史探索 runner 不属于当前交付树。S2 Plan handoff 只保留本轮审查所需的当前 artifacts，不进入 S3+ 模型或提交工作。
