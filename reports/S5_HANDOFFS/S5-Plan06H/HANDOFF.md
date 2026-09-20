# S5-Plan06H handoff

## 结论

本 worker 完成严格 City-Local 独立训练与 B3 残差校准筛选。每个外层验证单元只使用同一 `target_id + scenario_id + city_id` 的合法历史 replay 行；所有统计量、imputer、scaler、Ridge 和模型标签均未跨城市共享。结果仅用于证据筛选，不产生全局 winner、路由、融合或提交预测。

- 分支：`s5/plan06h-city-local-screening`；精确基线：`1f97f67488cd7aec120ef8b9fa04407dc13d54ba`；runner commit：`d04a17f1096b3d7e0944fedfca1b2d237f1b6ed1`。
- 覆盖：6 candidates × 2 targets × 2 scenarios × 11 origins × 40 cities = 10560 OOF rows；每 candidate 为 1760 行。
- 诊断最优（resident R1/R2 pooled MSE 均值，仅诊断）：`LOCAL_FGP_RIDGE_RESIDUAL`，均值 `956.111910`。
- all-144 Ridge 在 n 远小于 p 的本地几何下：`collapses`；详细 delta 见 `aggregate_metrics.csv` 与 `model_summary.json`。

## 固定候选与规则

| candidate | framing | 固定规则 | fallback 行数 |
|---|---|---|---:|
| `LOCAL_RESIDUAL_MEAN` | `F2_B3_RESIDUAL` | same-city legal B3 residual mean; minimum 3 residual rows; otherwise B3 unchanged | 242 |
| `LOCAL_RESIDUAL_EWMA` | `F2_B3_RESIDUAL` | same-city legal B3 residual EWMA with fixed decay 0.7; minimum 3 residual rows; otherwise B3 unchanged | 242 |
| `LOCAL_RESIDUAL_TIME_TREND` | `F2_B3_RESIDUAL` | same-city legal residual OLS trend; minimum 4 rows; otherwise Candidate A mean fallback | 348 |
| `LOCAL_FGP_RIDGE_RESIDUAL` | `F2_B3_RESIDUAL` | same-city 17-feature FG-P Ridge residual with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows | 454 |
| `LOCAL_ALL144_RIDGE_RESIDUAL` | `F2_B3_RESIDUAL` | same-city all-144-feature Ridge residual with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows | 454 |
| `LOCAL_FGP_RIDGE_DIRECT` | `F1_DIRECT` | diagnostic same-city 17-feature direct-actual Ridge with local median imputer, indicators, scaler, alpha=1000; minimum 5 rows | 454 |

Ridge 候选严格使用 `SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)` + `StandardScaler()` + `Ridge(alpha=1000.0)`；FG-P 为冻结的 17 个特征，all-144 为冻结的 144 个数值特征。没有 inner tuning、per-city tuning 或其他城市样本。

## 全量 pooled 结果

| candidate | target | scenario | MSE | B3 MSE | delta vs B3 | origin wins/11 | latest 2021 MSE | top-3 city SSE share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `LOCAL_ALL144_RIDGE_RESIDUAL` | registered | R1_REFERENCE_SNAPSHOT_LAG | 52540.915213 | 225.421212 | 52315.494000 | 1 | 1487.085855 | 0.998632 |
| `LOCAL_ALL144_RIDGE_RESIDUAL` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 70614.462929 | 225.421212 | 70389.041717 | 1 | 775611.251476 | 0.999010 |
| `LOCAL_ALL144_RIDGE_RESIDUAL` | resident | R1_REFERENCE_SNAPSHOT_LAG | 1309.048144 | 872.452652 | 436.595492 | 3 | 3705.161178 | 0.542610 |
| `LOCAL_ALL144_RIDGE_RESIDUAL` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 1428.400573 | 872.452652 | 555.947922 | 2 | 8531.083757 | 0.575832 |
| `LOCAL_FGP_RIDGE_DIRECT` | registered | R1_REFERENCE_SNAPSHOT_LAG | 1479.907256 | 225.421212 | 1254.486044 | 0 | 4605.916432 | 0.555063 |
| `LOCAL_FGP_RIDGE_DIRECT` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 1479.907256 | 225.421212 | 1254.486044 | 0 | 4605.916432 | 0.555063 |
| `LOCAL_FGP_RIDGE_DIRECT` | resident | R1_REFERENCE_SNAPSHOT_LAG | 10219.099125 | 872.452652 | 9346.646474 | 0 | 21501.146740 | 0.839615 |
| `LOCAL_FGP_RIDGE_DIRECT` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 10219.099125 | 872.452652 | 9346.646474 | 0 | 21501.146740 | 0.839615 |
| `LOCAL_FGP_RIDGE_RESIDUAL` | registered | R1_REFERENCE_SNAPSHOT_LAG | 231.713684 | 225.421212 | 6.292472 | 0 | 1490.013229 | 0.787701 |
| `LOCAL_FGP_RIDGE_RESIDUAL` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 231.713684 | 225.421212 | 6.292472 | 0 | 1490.013229 | 0.787701 |
| `LOCAL_FGP_RIDGE_RESIDUAL` | resident | R1_REFERENCE_SNAPSHOT_LAG | 956.111910 | 872.452652 | 83.659259 | 3 | 3947.844303 | 0.477044 |
| `LOCAL_FGP_RIDGE_RESIDUAL` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 956.111910 | 872.452652 | 83.659259 | 3 | 3947.844303 | 0.477044 |
| `LOCAL_RESIDUAL_EWMA` | registered | R1_REFERENCE_SNAPSHOT_LAG | 235.467375 | 225.421212 | 10.046163 | 2 | 1492.104379 | 0.768937 |
| `LOCAL_RESIDUAL_EWMA` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 235.467375 | 225.421212 | 10.046163 | 2 | 1492.104379 | 0.768937 |
| `LOCAL_RESIDUAL_EWMA` | resident | R1_REFERENCE_SNAPSHOT_LAG | 1127.910516 | 872.452652 | 255.457865 | 3 | 3679.280502 | 0.520069 |
| `LOCAL_RESIDUAL_EWMA` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 1127.910516 | 872.452652 | 255.457865 | 3 | 3679.280502 | 0.520069 |
| `LOCAL_RESIDUAL_MEAN` | registered | R1_REFERENCE_SNAPSHOT_LAG | 231.871425 | 225.421212 | 6.450212 | 1 | 1487.868419 | 0.788678 |
| `LOCAL_RESIDUAL_MEAN` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 231.871425 | 225.421212 | 6.450212 | 1 | 1487.868419 | 0.788678 |
| `LOCAL_RESIDUAL_MEAN` | resident | R1_REFERENCE_SNAPSHOT_LAG | 1073.355931 | 872.452652 | 200.903279 | 3 | 3947.297661 | 0.517010 |
| `LOCAL_RESIDUAL_MEAN` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 1073.355931 | 872.452652 | 200.903279 | 3 | 3947.297661 | 0.517010 |
| `LOCAL_RESIDUAL_TIME_TREND` | registered | R1_REFERENCE_SNAPSHOT_LAG | 251.878711 | 225.421212 | 26.457499 | 2 | 1500.038933 | 0.724663 |
| `LOCAL_RESIDUAL_TIME_TREND` | registered | R2_CONSERVATIVE_PUBLICATION_STRESS | 251.878711 | 225.421212 | 26.457499 | 2 | 1500.038933 | 0.724663 |
| `LOCAL_RESIDUAL_TIME_TREND` | resident | R1_REFERENCE_SNAPSHOT_LAG | 1819.495534 | 872.452652 | 947.042883 | 0 | 4001.085760 | 0.586055 |
| `LOCAL_RESIDUAL_TIME_TREND` | resident | R2_CONSERVATIVE_PUBLICATION_STRESS | 1819.495534 | 872.452652 | 947.042883 | 0 | 4001.085760 | 0.586055 |

## 信号分类与残差持久性

分类规则：classify_candidates 中冻结的 Plan05 family rule；`city_local_signal_found` 要求同一 candidate 在 resident 的 R1/R2 均优于 B3 且各自至少 6/11 个 origin 胜出；仅在任一场景距 B3 不超过 1% 但不满足稳定条件时记为 `city_local_competitive_but_unstable`。

| candidate | classification |
|---|---|
| `LOCAL_RESIDUAL_MEAN` | `city_local_unsupported` |
| `LOCAL_RESIDUAL_EWMA` | `city_local_unsupported` |
| `LOCAL_RESIDUAL_TIME_TREND` | `city_local_unsupported` |
| `LOCAL_FGP_RIDGE_RESIDUAL` | `city_local_unsupported` |
| `LOCAL_ALL144_RIDGE_RESIDUAL` | `city_local_unsupported` |
| `LOCAL_FGP_RIDGE_DIRECT` | `city_local_unsupported` |

B3 residual persistence diagnostics 覆盖 160 个 target/scenario/city 单元；可估计 lag-1 相关的单元平均相关系数为 `0.419640`，可估计非零符号持久性的单元平均值为 `0.684482`。这些值仅作诊断，不用于 routing 或 candidate selection。

## 证据边界

- `city_local_training_geometry.csv` 逐 outer city-year 记录 cutoff、训练行数、同城检查和阈值；`correction_diagnostics.csv` 记录校准量、fallback 与 B3 对照。
- 明确验证：任何 city-specific fit/preprocessing 都只接收同城且不晚于 `origin_year-2` 的 replay 行；没有外层验证标签进入训练。
- 本计划未执行 routing、ensemble/blend/gating、2023、submission、checkpoint、S6，也未修改 requirements。

## 产物

- `oof_predictions.csv`
- `aggregate_metrics.csv`
- `per_origin_metrics.csv`
- `city_model_error.csv`
- `city_local_training_geometry.csv`
- `residual_persistence_diagnostics.csv`
- `correction_diagnostics.csv`
- `model_summary.json`
- `HANDOFF.md`
