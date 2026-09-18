# City Population Competition — S1 Data Contract

本仓库冻结的是城市人口预测项目的 S1 源数据层：原始资料、可重建的规范化 CSV、隔离值、派生行溯源和主任务交接。

```powershell
python -m pip install -r requirements.txt
python scripts/run_s1.py
```

`scripts/run_s1.py` 会依次运行 7 个领域审计脚本，重建 `data/normalized/` 与 `data/quarantine/`，验证关键不变量，刷新 `docs/s1_data_contract.csv` 和 `docs/s1_quality_issue_registry.csv`，并生成唯一审阅包 `reports/S1_FINAL_REVIEW_PACK.zip`。

本仓库明确不包含 S2 验证设计、回测、特征工程、建模、预测或提交结果。后续工作必须先阅读 `docs/S1_MAIN_HANDOFF.md` 和 `docs/S1_DATA_CONTRACT.md`。
