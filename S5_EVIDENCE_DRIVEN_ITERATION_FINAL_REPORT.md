# S5｜Evidence-driven Iteration Final Evidence Report

> 本报告整合已审核并已合入 `main` 的 S5-Plan01 与 S5-Plan02 证据。它提供 Main stage review 输入，不声明 S5 完成，也不授权 S6。

## 1. S5 started from what problem?

S4 已确认：在冻结 S2 validation contract 下，B3 仍是最强的可部署 fixed baseline；target-history、外生 source、低容量 HGB、error memory、cross-city relative position 与 multisource RidgeCV 均未形成稳定的 resident aggregate gain。S4 的主要产出是可复核的机制否证链，而不是更高分模型。

因此 S5 没有继续无假设地增加模型，而是先问：

> B3 的失败风险能否在 target 观察到之前被识别？如果能，风险结构是否足以支持一个具体的低容量修正机制？

Plan01 发现历史一阶差分波动具有可重复的风险关联，于是唯一激活的下一步是 Plan02 的稳健局部趋势 screening。Plan02 未达到预注册 promotion gate；本阶段停止扩展该机制。

## 2. Frozen evaluation boundary

- common origins：`2011–2021`，11 个 expanding origins
- target history：严格 `<= T-2`
- resident：primary target proxy
- registered：mandatory sensitivity
- pooled city-year MSE：primary aggregate
- macro-origin MSE：稳定性 sensitivity
- latest 2021：单独诊断，不替代 multi-origin aggregate
- per-origin wins/losses、R1/R2 equivalence 与 city concentration：保留
- 无随机切分、target leakage、外部新 source、`test.xls`、2023 final prediction 或 submission

S5-Plan01 与 Plan02 均沿用该合同。Plan02 的 target-history-only candidates 在 R1/R2 数值等价。

本次报告生成前，Plan02 已从 `main @ 1a6770f763966b58dd7ae60fef34e332febcea09` 以 fast-forward 方式整合至 `abe92fc60a132a35cfcf2d78be0bc5422b05bc6f`；报告提交会在该整合状态之上继续前进。

## 3. Experiment registry

| ID | Hypothesis / Question | Change | Key Result | Interpretation | Decision |
|---|---|---|---|---|---|
| S5-Plan01 | pre-origin diagnostics 是否能识别 B3 failure risk，并指向可修复机制？ | 复用 B1/B2/B3 predictions；从 `<=T-2` target history 构造 5 个诊断；不训练模型 | 五个 resident diagnostics 均通过 risk gate；volatility 通过 decision relevance；shrinkage/blend 仅有 6/11 高风险 comparator gains | risk predictability 存在，但风险识别不等于 repairability | `mechanism_signal_found:robust_local_trend` |
| S5-Plan02 | deterministic median-slope robust local trend 是否能修复高波动风险？ | 固定评估 B3、B4、RLT3、RLT5；raw/relative volatility 与 scale 分层；不 blend、不调参 | RLT3 resident MSE `861.4909`，仅改善 `1.256%`、赢 `6/11`；RLT5 resident/registered 明显不稳定 | 没有达到 strong 或 complementary gate；不应继续添加稳健 estimator | `robust_local_trend_unsupported` |

轻量机器 registry 见 `reports/S5_EXPERIMENT_REGISTRY.csv`，每个 material Plan 仅一行。

## 4. Plan01 — what structure actually exists?

resident primary、R1/R2 数值相同的风险统计如下；高风险组是每个 origin 内按诊断值排序的 top `ceil(25%)`，同值按 `city_id` 打破：

| Diagnostic | Median abs Spearman | Positive origins | High / remaining B3 MSE | High-risk worse origins |
|---|---:|---:|---:|---:|
| slope disagreement | 0.3748 | 11/11 | 4.4271× | 9/11 |
| B1/B2/B3 prediction spread | 0.3510 | 11/11 | 5.4239× | 9/11 |
| historical difference volatility | 0.4281 | 11/11 | 6.0006× | 10/11 |
| recent acceleration | 0.3129 | 11/11 | 1.7219× | 8/11 |
| pre-origin target scale | 0.5220 | 11/11 | 12.1582× | 11/11 |

这些数值证明的是：部分 B3 failure risk 可以被 pre-origin state 分层。它们没有证明任何修正模型必然有效。

Plan01 的决策相关性进一步限制了解释：

- partial pooling / shrinkage 的高风险 comparator gain 只有 `6/11` origins；
- 低风险 B3 advantage 只有 `5/11` origins；
- fixed low-capacity blend 未达到 `7/11` 的预注册稳定性要求；
- registered sensitivity 没有形成强方向反转；
- 因此只有 volatility → robust local trend 获得下一实验资格。

Plan01 还发现错误集中具有重复性，但 dominant city 与 pre-origin state 的重叠并不充分，不能把 post-hoc city identity 变成 routing rule。

## 5. Plan02 — did robust estimation repair the risk?

下表为 R1；R2 对 target-history-only candidates 数值等价。`Origin wins` 是 candidate 严格优于 B3 的 origin 数：

| Target | Candidate | Pooled MSE | Latest 2021 MSE | Origin wins vs B3 |
|---|---|---:|---:|---:|
| resident | B3 | 872.4527 | 3705.4278 | — |
| resident | B4 | 959.4945 | 4209.0620 | 3/11 |
| resident | RLT3 | 861.4909 | 3666.8000 | 6/11 |
| resident | RLT5 | 933.8669 | 4256.4981 | 3/11 |
| registered | B3 | 225.4212 | 1485.6049 | — |
| registered | B4 | 247.2048 | 1578.3430 | 4/11 |
| registered | RLT3 | 223.6909 | 1483.1500 | 8/11 |
| registered | RLT5 | 256.7166 | 1576.6210 | 4/11 |

RLT3 的 resident aggregate 表面上优于 B3，但不能晋级：

- resident pooled gain 约 `1.256%`，strong gate 要求至少 `2%`；
- resident origin wins 为 `6/11`，要求至少 `7/11`；
- raw high-volatility 层 wins 仅 `6/11`，complementary gate 要求至少 `8/11`；
- normalized-volatility 层 wins `7/11`，但不是 raw gate 的替代；
- high raw volatility 且非 high target scale 子集的 pooled 改善约 `0.1595%`，低于固定 `1%` scale-confounding gate；
- RLT3 的小幅 aggregate gain 不能被解释为稳定、scale-independent 的机制修复。

RLT5 虽比 B4 的 5-point OLS 控制更好，但比 B3 的 resident pooled MSE 高约 `7.04%`，registered pooled MSE 高约 `13.88%`，因此不能把“更长窗口”或“median slope”升级为可部署替代。

Plan02 的 concentration sanity 也没有发现隐藏的单城修复：RLT3 resident 改善城市为 `37/40`，top-3 positive SSE improvement share 为 `51.75%`；没有创建 routing 或 reweighting。

## 6. Hypotheses now closed / inactive

| Hypothesis | Status | Evidence interpretation |
|---|---|---|
| robust local trend | tested and unsupported | RLT3 有小幅 aggregate gain但不满足 strong/complementary gate；RLT5 在两个 target 上不稳定 |
| partial pooling / shrinkage | not activated | Plan01 comparator decision relevance 未达到 7/11；Plan02 没有新增证据 |
| fixed low-capacity blend | not activated | Plan01 未分离出稳定高风险 comparator gain 与低风险 B3 advantage；Plan02 禁止 fit blend |
| stronger boosting | not activated | 没有新 nonlinear interaction 证据；S4 low-capacity HGB 已为负证据 |
| remaining source groups | not activated | S5 没有新的 residual/source hypothesis |
| tuning | not activated | 没有通过 promotion gate 的 candidate 值得 tuning |

“tested and unsupported”只适用于本次 robust local trend screening；“not activated”不表示对应方向理论上永远无效，而是当前 S5 evidence budget 没有为其开实验。

## 7. What S5 actually taught us

1. B3 failure 不是完全随机的；多个 pre-origin diagnostics 可以识别 hard rows，尤其是 historical first-difference volatility。
2. risk predictability 不等于 repairability。能识别 hard cases，并不代表已有 comparator 或 robust estimator 能在这些 cases 中稳定获益。
3. 绝对误差集中与 target scale 有明显纠缠；scale-normalized check 是必要的，否则很容易把大城市 MSE 误读为纯 volatility mechanism。
4. RLT3 的 `861.49` 说明单一 aggregate 改善不足以晋级；multi-origin stability、risk-stratum alignment 和 scale-confounding gate 阻止了 latest/pooled 过度解读。
5. 固定 validation contract 与预注册门槛成功阻止了追加 estimator、blend 或 tuning 来挽救负结果。
6. 在当前证据预算和冻结合同下，B3 仍是最强的 deployable fixed reference。

## 8. Current best solution status

当前最强 fixed model 仍为 `B3_RECENT_3POINT_OLS_TREND`：

- resident pooled MSE：`872.452651515143`
- registered pooled MSE：`225.421212121213`
- resident latest 2021 MSE：`3705.42777777773`
- registered latest 2021 MSE：`1485.60486111112`

S5 没有产生足够稳定的 replacement。尚未生成 2023 final prediction，也没有生成 submission。不能据此声称 B3 理论最优、robust estimator 永远无效、或未来数据上的最终名次已知。

## 9. Lab recommendation to Main

最终建议：`ready_for_main_stage_review`

理由：S5 已完成高信息量的 residual/regime decomposition，并对唯一通过 Plan01 decision-relevance 的机制做了冻结合同下的独立 screening；该机制未通过 promotion gate，当前没有其他 active hypothesis。后续若继续实验，必须由 Main/Lab 提出新的具体 unresolved hypothesis，而不是继续增加同类模型。

Main 需要独立决定：

- 是否认可 S5 completion signal；
- 是否更新 `CURRENT_LEARNING_CHECKPOINT.md`；
- 是否授权进入 S6。

本报告不替 Main 作出上述阶段授权决定。

## 10. Evidence paths

- Plan01：`reports/S5_HANDOFFS/S5-Plan01/`
- Plan02：`reports/S5_HANDOFFS/S5-Plan02/`
- upstream S4 context：`S4_FEATURE_ML_MODELING_FINAL_REPORT.md`
- registry：`reports/S5_EXPERIMENT_REGISTRY.csv`
