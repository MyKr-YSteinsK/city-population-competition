# S5-Plan01 Handoff — B3 残差结构与可预测趋势失稳机制筛查

## 1. 交付身份与边界

- Plan ID：`S5-Plan01`
- Stage：`S5｜Evidence-driven Iteration`
- 类型：低成本、只读诊断筛查；不训练新模型
- 冻结基线：`main @ 04e5cd469c45468a60b1f9b65a96e76311d5326b`
- Worker branch：`s5/plan01-residual-regime-screening`
- Worker evidence commit：`db7bf05fa2447c6cdc316a054d08e1d248641a45`
- Final worker tip：本文件的 docs-only 最终化提交 SHA 将在 `TASK_RESULT` 中返回
- 最终建议：`mechanism_signal_found:robust_local_trend`

本分支没有修改 `main`、`CURRENT_LEARNING_CHECKPOINT.md`、共享状态文件或 S1–S4 基础设施；没有声明 S5 完成，也没有进入 S6。

仓库事实中仍存在文档漂移：`CURRENT_LEARNING_CHECKPOINT.md` 与最新 `main` 已授权 S5，但 `docs/CODEX_INSTRUCTIONS.md` 保留了 S4 时代的“S5 未授权”表述。本任务按最新 checkpoint 与冻结 base 执行，未在筛查分支顺手修改该共享文档。

## 2. 实际变更与可复现运行

新增且仅限于 Plan01 的文件：

- `scripts/build_s5_plan01_residual_regime_screen.py`
- `reports/S5_HANDOFFS/S5-Plan01/screening_summary.json`
- `reports/S5_HANDOFFS/S5-Plan01/preorigin_risk_signal_screen.csv`
- `reports/S5_HANDOFFS/S5-Plan01/model_tradeoff_by_risk.csv`
- `reports/S5_HANDOFFS/S5-Plan01/city_concentration_repeatability.csv`
- `reports/S5_HANDOFFS/S5-Plan01/hypothesis_activation_matrix.csv`
- 本文件 `HANDOFF.md`

实际运行：

```powershell
python -m py_compile scripts/build_s5_plan01_residual_regime_screen.py
python scripts/build_s5_plan01_residual_regime_screen.py
```

第一次运行后的自检发现 `ALL` 风险分层如果在跨 origin 的合并行上直接排名，会把预注册的“每个 origin 内 top ceil(25%)”错误地变成全局排名。runner 已修正为先在每个 origin 内分层、再汇总 pooled trade-off，并重新运行通过。该修正没有改变 Plan 的假设、阈值、模型或验证合同。

## 3. 输入复用与信息边界

复用的冻结证据：

- `reports/S3_HANDOFFS/S3-Plan03/per_origin_predictions_or_residuals.csv`
- `reports/S3_HANDOFFS/S3-Plan03/per_origin_metrics.csv`
- `reports/S3_HANDOFFS/S3-Plan03/local_slope_dispersion.csv`
- `reports/S3_HANDOFFS/S3-Plan03/slope_error_relationship.csv`
- `reports/S3_HANDOFFS/S3-Plan03/time_regime_comparison.csv`
- `reports/S4_HANDOFFS/S4-Plan05A/plan05a_summary.json`（负对照，仅作已知边界记录）

缺失的 pre-origin 诊断从 `data/normalized/人口规模.csv` 在 runner 内确定性重构。所有目标历史特征均截断为 `<= T-2`；origin 的实际 target、residual 与 squared error 只作为 outcome 使用。没有新外部源、`test.xls`、2023 预测、submission、随机切分或新模型拟合。

固定覆盖为 targets `resident` / `registered`、scenarios `R1_REFERENCE_SNAPSHOT_LAG` / `R2_CONSERVATIVE_PUBLICATION_STRESS`、origins `2011–2021`、40 城。每个 target × scenario 有 440 个 scoreable city-origin rows；诊断明细共 1,760 行。

局部实现决定：

- 使用五个紧凑诊断：`slope_disagreement_abs`、`baseline_prediction_spread`、`history_difference_volatility`、`recent_acceleration_abs`、`preorigin_target_scale`。
- 风险组严格使用每个 origin 内 top `ceil(25%)`，同值以 `city_id` 稳定打破；没有事后阈值搜索。
- Spearman 用排序相关汇总，保留每 origin 方向和风险组 MSE；`ALL` trade-off 也保持 within-origin 分层后再 pooled。
- 历史波动使用可用一阶差分的 population standard deviation；加速度使用最新一阶差分与前一阶差分的绝对差；scale 使用最新允许的 pre-origin target level。
- `R1`/`R2` 目标历史诊断相同是预期结果，因为本 Plan 的诊断只依赖目标历史和已冻结 B1/B2/B3 预测。

## 4. 冻结 B3 兼容性

四个 target × scenario 组合均通过兼容性检查，均为 440 行、无历史边界违规：

| target | 冻结 pooled MSE | 观察 pooled / macro-origin MSE | latest 2021 MSE | 状态 |
|---|---:|---:|---:|---|
| resident | 872.452651515143 | 872.4526515151432 / 872.4526515151432 | 3705.427777777728 | PASS |
| registered | 225.421212121213 | 225.42121212121316 / 225.4212121212132 | 1485.6048611111178 | PASS |

R1/R2 的上述数值相同；最大 pooled 误差约 `2.3e-13`，在浮点容差内。`screening_summary.json` 记录了四个组合的完整检查值。

## 5. 风险筛查与决策相关性

resident primary 的五个诊断均通过预注册 risk association gate；R1/R2 结果相同。下表为 resident 的核心风险统计：

| 诊断 | median abs Spearman | 方向一致 origins | pooled high/remaining B3 MSE | high-risk worse origins | risk gate |
|---|---:|---:|---:|---:|---|
| slope disagreement | 0.3748 | 11/11 | 4.4271 | 9/11 | PASS |
| B1/B2/B3 prediction spread | 0.3510 | 11/11 | 5.4239 | 9/11 | PASS |
| historical difference volatility | 0.4281 | 11/11 | 6.0006 | 10/11 | PASS |
| recent acceleration | 0.3129 | 11/11 | 1.7219 | 8/11 | PASS |
| pre-origin target scale | 0.5220 | 11/11 | 12.1582 | 11/11 | PASS |

registered sensitivity 没有出现预注册的强方向反转；五个诊断均保留正向风险方向。具体 per-origin 相关、MSE、比例和 high-risk worse 标记均在 `preorigin_risk_signal_screen.csv` 和 JSON 中。

风险识别本身没有自动激活 shrinkage 或 fixed blend。决策相关性为：

- partial pooling / shrinkage：未激活。高风险 comparator gain 仅 `6/11` origins，低风险 B3 advantage 为 `5/11`，未达到 `7/11`。
- fixed low-capacity blend：未激活。同样只有 `6/11` 高风险 comparator gain 与 `5/11` 低风险 B3 advantage。
- robust local trend：激活为下一步候选方向。`history_difference_volatility` 通过 resident risk gate，且高不稳定 stratum 集中出现 B3 excess error；registered 未形成强反转。

这里的“激活”只表示允许 Lab 设计下一步 robust local trend screening；本 Plan 没有拟合 robust estimator、blend weight、shrinkage parameter 或 routing rule，也没有证明下一模型一定改善。

## 6. H1–H4 判断

- **H1：部分支持但不具备 shrinkage/blend 决策相关性。** 趋势 disagreement 与 B3 error 有稳定风险关联，但现有 B1/B2 comparator 没有在 `>=7/11` origins 中提供足够一致的高风险相对收益。
- **H2：支持，且给出 robust local trend 方向。** 历史一阶差分波动通过全部 resident risk gate；该结果来自跨 origin 的预注册分层，不是单一 2021 现象。
- **H3：部分支持、尚未唯一归因。** resident late B3 MSE 为 `1289.3975`，early 为 `616.7165`；late volatility 均值从 `8.4209` 上升到 `8.9515`，prediction spread 从 `24.4541` 上升到 `25.8605`。但 slope disagreement 与 acceleration 并未同向上升，因此只能说 late difficulty 与可观测 instability 状态相容，不能把年份本身升级成规则。
- **H4：错误集中持续，但不是充分的 identity-only 可部署信号。** resident top city `city5` 占累计 B3 SSE `28.61%`，top-3 占 `50.22%`，最大 high-error-quartile recurrence 为 `10/11`；registered top city `city40` 占 `42.32%`，top-3 占 `80.90%`，最大 recurrence 为 `9/11`。但 dominant city 与 pre-origin high-risk overlap 不一致（例如 resident `city24` 为 0），所以没有创建城市路由。

## 7. 偏差、失败与剩余不确定性

没有 Plan-level deviation。唯一的实现纠正是第 2 节所述的 pooled `ALL` 风险分层 bug，已在最终运行前修复并复核。

主要不确定性：

- volatility 可能同时携带 target scale 或城市误差集中度信息，当前筛查不能证明因果机制；scale 作为控制诊断正是为此保留。
- robust local trend 仅是可证据支持的实验方向，尚未进行 estimator、窗口、权重或参数选择。
- late/middle 的描述性区间沿用 S3 的既有 framing；它们不是部署路由规则。
- registered 在总体方向上未反转，但个别 late origin 的风险组差异较弱，后续实验仍需保留 registered sensitivity。

## 8. 推荐下一边界

推荐交付决定：`mechanism_signal_found:robust_local_trend`。

Main/Lab 可以只围绕“robust local trend”设计后续低容量筛查；在新证据出现前，不应因为本 Plan 的风险关联而自动开启 shrinkage、fixed blend、stronger boosting、remaining source groups 或 tuning。该 worker 不声明 S5 完成，S6 未进入。
