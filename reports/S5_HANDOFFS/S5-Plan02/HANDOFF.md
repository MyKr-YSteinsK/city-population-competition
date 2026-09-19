# S5-Plan02 Handoff — 高波动机制下的稳健局部趋势 screening

## 1. 结果与边界

- Plan ID：`S5-Plan02`
- Stage：`S5｜Evidence-driven Iteration`
- Plan 类型：serial mechanism-driven screening
- Plan01 decision：`mechanism_signal_found:robust_local_trend`
- Plan01 integration base：`04e5cd469c45468a60b1f9b65a96e76311d5326b`
- Plan01 reviewed tip：`1a6770f763966b58dd7ae60fef34e332febcea09`
- Plan01 evidence commit：`db7bf05fa2447c6cdc316a054d08e1d248641a45`
- Integrated `main`：`1a6770f763966b58dd7ae60fef34e332febcea09`
- Worker branch：`s5/plan02-robust-local-trend-screening`
- Worker evidence commit：`e8602f51148807cbc0aa5566bea09ed989b7e6d8`
- Final worker tip：docs-only 最终化提交 SHA 将在本次 `TASK_RESULT` 返回
- 最终筛查决定：`robust_local_trend_unsupported`

本 Plan 只评估冻结 B3/B4 与预注册的 RLT3/RLT5，未拟合 blend、未创建 routing、未调参、未添加外部数据。Plan02 完成不等于 S5 完成；S5 未被声明完成，S6 未进入。

## 2. Plan01 集成门

集成前已执行 fetch 并验证：

1. `origin/main` 为冻结 base `04e5cd469c45468a60b1f9b65a96e76311d5326b`。
2. 远端 Plan01 branch tip 为 `1a6770f763966b58dd7ae60fef34e332febcea09`。
3. Plan01 tip 相对冻结 base：merge-base 精确为冻结 base，ahead `2`，behind `0`。
4. `git diff base..Plan01-tip` 仅包含 `scripts/build_s5_plan01_residual_regime_screen.py` 与 `reports/S5_HANDOFFS/S5-Plan01/*`。
5. 本地 `main` 无未提交修改；按计划 fast-forward 到 Plan01 tip，并正常 push；push 后 local `main == origin/main`。
6. Plan02 worker 从整合后的 `1a6770f763966b58dd7ae60fef34e332febcea09` 创建。

`docs/CODEX_INSTRUCTIONS.md` 仍有 S4 时代的 S5 授权旧文案；按最新 checkpoint 和已通过的 Plan01 integration gate 执行，未把该漂移作为本 Plan 的拒绝理由，也未在本 worker 中顺手清理共享文档。

## 3. 实际文件与运行

新增且仅限于 Plan02 的文件：

- `scripts/build_s5_plan02_robust_local_trend_screen.py`
- `reports/S5_HANDOFFS/S5-Plan02/HANDOFF.md`
- `reports/S5_HANDOFFS/S5-Plan02/screening_metrics.csv`
- `reports/S5_HANDOFFS/S5-Plan02/per_origin_comparison.csv`
- `reports/S5_HANDOFFS/S5-Plan02/volatility_stratum_comparison.csv`
- `reports/S5_HANDOFFS/S5-Plan02/scale_confounding_check.csv`
- `reports/S5_HANDOFFS/S5-Plan02/candidate_definitions.csv`
- `reports/S5_HANDOFFS/S5-Plan02/screening_summary.json`
- `reports/S5_HANDOFFS/S5-Plan02/hypothesis_activation_matrix.csv`

实际运行：

```powershell
python -m py_compile scripts/build_s5_plan02_robust_local_trend_screen.py
python scripts/build_s5_plan02_robust_local_trend_screen.py
```

第一次 runner 执行暴露了实现层 tuple-key 汇总错误（latest 2021 MSE 被取成空值），未产生有效结论；修复后重新执行，最终输出通过。没有改变任何 Plan 规则、阈值、来源或验证合同。

未进行完整 S1–S4 replay，因为冻结 B3 兼容性检查通过。

## 4. 输入、候选与信息边界

复用：

- `reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv`
- `reports/S3_HANDOFFS/S3-Plan03/per_origin_metrics.csv`
- `reports/S3_HANDOFFS/S3-Plan03/baseline_predictor_definitions.csv`
- `reports/S5_HANDOFFS/S5-Plan01/screening_summary.json`
- `data/normalized/人口规模.csv`

冻结覆盖：targets `resident` / `registered`、R1/R2、origins `2011–2021`、40 城；每个 target × scenario 有 440 个 scoreable rows，总诊断记录 1,760 行。所有历史观察严格 `<= T-2`；origin target 只用于 outcome 评分。

候选：

- `B3_RECENT_3POINT_OLS_TREND`：冻结 OLS 控制。
- `B4_RECENT_5POINT_OLS_TREND`：冻结 5-point OLS 窗口控制。
- `RLT3_RECENT_3POINT_MEDIAN_SLOPE_TREND`：最近 3 个非缺失点的全 pairwise slope 中位数，加 `y_i - slope*year_i` 中位数截距。
- `RLT5_RECENT_5POINT_MEDIAN_SLOPE_TREND`：同一确定性 median-slope 估计器，窗口为最近 5 点。

RLT5 在 1,760 个 target × scenario × origin × city 记录中有 1,624 行实际 median-slope fit，136 行按冻结 B4 规则使用 B1 fallback，0 行不可计算；没有静默回退到 B3。raw volatility、relative volatility 和 target scale 均为诊断分层，不参与拟合或路由。

relative volatility 固定为：

`population_std(first_differences) / max(abs(latest_allowed_target), 1e-12)`

高风险组严格为每个 origin 内按诊断值降序的 top `ceil(25%)`，同值按 `city_id` 打破；scale-confounding 的“非平凡”操作化门槛固定为高 raw volatility 且非 high-scale 子集的 pooled 相对改善至少 `1%`，没有事后调阈值。

## 5. 冻结兼容性与总体结果

四个 target × scenario 的 B3 兼容性全部 PASS；每组 440 行：

| target | pooled B3 MSE | macro-origin MSE | latest 2021 MSE |
|---|---:|---:|---:|
| resident | 872.4526515151432 | 872.4526515151432 | 3705.427777777728 |
| registered | 225.42121212121316 | 225.4212121212132 | 1485.6048611111178 |

R1/R2 target-history-only predictions 数值等价，最大 candidate prediction delta 为 0。

R1 总体指标：

| target | candidate | pooled MSE | macro MSE | latest 2021 MSE | origin wins vs B3 |
|---|---|---:|---:|---:|---:|
| resident | B3 | 872.4526515151432 | 872.4526515151432 | 3705.427777777728 | — |
| resident | B4 | 959.4945000000087 | 959.4945000000087 | 4209.061999999994 | 3/11 |
| resident | RLT3 | 861.4909090909091 | 861.4909090909091 | 3666.8 | 6/11 |
| resident | RLT5 | 933.8669013713675 | 933.8669013713675 | 4256.498090277777 | 3/11 |
| registered | B3 | 225.42121212121316 | 225.4212121212132 | 1485.6048611111178 | — |
| registered | B4 | 247.20481818181926 | 247.20481818181926 | 1578.3430000000096 | 4/11 |
| registered | RLT3 | 223.6909090909091 | 223.6909090909091 | 1483.15 | 8/11 |
| registered | RLT5 | 256.7165601325758 | 256.71656013257575 | 1576.6209635416667 | 4/11 |

RLT3 的 resident pooled 改善约 `1.256%`，低于 strong gate 的 `2%`；RLT5 虽然 resident pooled MSE 优于 B4，但仍比 B3 差约 `7.04%`，且 registered 比 B3 差约 `13.88%`。

## 6. Volatility mechanism evidence

resident R1 的 pooled strata：

| candidate | raw high-volatility delta vs B3 | raw remaining delta vs B3 | raw high wins | relative high wins |
|---|---:|---:|---:|---:|
| RLT3 | -35.9626 | -2.6281 | 6/11 | 7/11 |
| RLT5 | +178.2637 | +22.4644 | 3/11 | 3/11 |

RLT3 的 raw high-volatility 层确实改善更多，但未达到 complementary signal 要求的 `8/11` raw high wins；RLT5 方向相反。RLT3 在高 raw volatility 且非 high target scale 子集上 pooled 相对改善仅 `0.1595%`（28 个 pooled rows），低于固定的 `1%` 非平凡门槛，故不能排除改善主要由 scale/误差量级带来。

城市集中度 sanity：RLT3 resident pooled MSE 虽下降，但改善城市为 `37/40`，top city 为 `city17`，top-3 positive SSE improvement share 为 `51.75%`；没有单一城市路由或重加权。RLT5 未改善 resident pooled MSE。

## 7. 预注册规则判定

### Strong screening signal

- RLT3：失败。resident pooled gain `1.256% < 2%`，resident origin wins `6/11 < 7/11`，scale-outside gain `0.1595% < 1%`；registered pooled worsening gate 虽通过，整体仍不能激活。
- RLT5：失败。resident pooled gain为负、仅 `3/11` wins、registered pooled worsening 超过 `2%`、raw high volatility 方向不对；虽优于 B4，不足以优于冻结 B3。

### Complementary mechanism signal

- RLT3：失败。relative high volatility wins 达 `7/11`，remaining-stratum 没有恶化，但 raw high volatility wins 只有 `6/11`，scale-outside 改善也低于 `1%`。
- RLT5：失败。raw/relative high volatility wins 均为 `3/11`，且 high-volatility pooled error 增加。

因此最终决定为：`robust_local_trend_unsupported`。没有激活 fixed low-capacity blend、partial pooling、stronger boosting、remaining source groups 或 tuning。

## 8. 偏差、失败与剩余不确定性

没有 Plan-level deviation。第一次运行的 tuple-key bug 已在最终运行前修复并通过验证，属于 runner 实现纠正。

剩余不确定性：

- RLT3 的方向性改善存在，但幅度和跨 origin 稳定性不足；不能把 1.26% aggregate gain升级为可部署胜出。
- raw volatility 与 target scale 仍有重叠；非 high-scale volatile 子集只出现很小的改善。
- RLT5 的结果说明增加窗口和 robust estimator 并不自动修复 B3 风险；在 registered 上出现明显负面敏感性。
- 本 Plan 没有尝试更多稳健估计器、窗口、常数、blend 或 tuning；按 failure 规则停止扩展。

## 9. 状态与推荐下一边界

- S5 未声明完成。
- S6 未进入。
- Plan02 worker 只需提交并推送供 Lab review，不自动 merge 回 `main`。
- 推荐下一步：停止 robust local trend 机制，不在该方向追加 estimator 或参数搜索；是否重新审查其他 S5 假设由 Main/Lab 单独决定。
