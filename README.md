# City Population Competition — S6 Final Solution Contract

本仓库保留城市人口预测项目从 S1 数据合同到 S5 evidence wave 的可追溯产物，并在 S6-Plan01 冻结 resident-primary F4 的最终方案合同与复现边界。当前合同状态为 `PARTIALLY_FROZEN_SUBMISSION_BLOCKED`：可以供 Main review 和后续 clean raw rebuild 使用，但尚未生成 2023 prediction 或正式 submission。后续 final pipeline 需保留 resident primary 与 registered alternative 两条独立候选路径，不得平均或混合。

```powershell
python -m pip install -r requirements.txt
python scripts/run_s1.py
```

`scripts/run_s1.py` 会依次运行 7 个领域审计脚本，重建 `data/normalized/` 与 `data/quarantine/`，验证关键不变量，刷新 `docs/s1_data_contract.csv` 和 `docs/s1_quality_issue_registry.csv`，并生成唯一审阅包 `reports/S1_FINAL_REVIEW_PACK.zip`。

S1 canonical/data-quality contract、7 个合法 XLSX 和官方 PDF 不可随意改写；`data/quarantine/source/test.xls` 始终是禁止读取的输入。S2/S3/S4/S5 的历史验证与模型证据位于对应 handoff 目录，S6-Plan01 合同包位于 `reports/S6_HANDOFFS/S6-Plan01/`。可重跑本轮审计：

```powershell
python scripts/build_s6_plan01_final_solution_contract_audit.py
```

后续工作必须先阅读 `CURRENT_LEARNING_CHECKPOINT.md`、`reports/S6_HANDOFFS/S6-Plan01/HANDOFF.md` 和 `final_solution_contract.json`；除非 Main 明确授权，不得生成正式 2023 prediction/submission 或进入 S6 后续边界。
