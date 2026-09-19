# S4｜Feature & ML Modeling Final Report

## 1. S4 question

本阶段回答的问题是：

> 在冻结的 S2 validation contract 下，leakage-safe feature engineering 与 formal ML 是否能够稳定超过 B3？

结论是：截至 Plan01–05C，尚无 material candidate 在 resident primary 的 pooled multi-origin evidence 上稳定超过 B3。S4 的主要产出是完整、可复核的机制证据链，而不是一个更高分的模型。

## 2. Frozen baseline and validation boundary

当前 strongest fixed baseline 为 B3_RECENT_3POINT_OLS_TREND：对每个城市和 pseudo-origin，使用不晚于 T-2 的最近三个 target observations 做 local OLS，预测 target year T。

- common origins：2011–2021，共 11 个 expanding origins
- target history：严格遵守 T-2
- resident：primary target proxy
- registered：mandatory sensitivity target
- R1：reference snapshot lag；R2：conservative publication stress
- pooled city-year MSE：primary aggregate
- macro-origin MSE：稳定性敏感性
- latest 2021 MSE：单独的任务接近性诊断，不替代 multi-origin aggregate

B3 pooled/macro MSE：

- resident：872.452652
- registered：225.421212

B3 latest 2021 MSE：

- resident：3705.427778
- registered：1485.604861

本报告不重写 S2 contract；完整 validation、information-set 与 aggregation 定义仍以 S2 handoff 为准。

## 3. Experiment map

下表列出每一组 S4 material experiment 的 candidate pooled MSE。斜杠两侧依次为 R1 / R2；target-history-only candidate 的两侧相同是因为 R1/R2 information set 数值等价。

| Experiment | Mechanism / model | Resident pooled MSE | Registered pooled MSE | Main result | Decision |
|---|---|---:|---:|---|---|
| Plan01 | M1A TH_CORE Ridge；M1B TH_SHAPE Ridge | M1A 910.349895；M1B 938.694952 | M1A 231.360016；M1B 238.433125 | target-history additive/shape correction 均高于 B3 | unsupported |
| Plan02 | D2 global residual bias；M2A level Ridge；M2B level+dynamics Ridge | best D2 883.812539 / 883.812539 | best D2 225.592430 / 225.592430 | urbanization level/dynamics 与 global bias control 均未形成 gain | unsupported |
| Plan03 | M3A/M3B/M3C low-capacity HGB | best R1 911.168287；best R2 938.209015 | best R1/R2 229.697127 | nonlinear TH 或 TH×urbanization 未解决失败 | unsupported |
| Plan04A | M4A density Ridge | 906.652625 / 917.775198 | 229.902996 / 230.299976 | density recent signal 未超过 B3 | unsupported |
| Plan04B | M4B wage Ridge | 943.919472 / 943.919472 | 225.850578 / 225.850578 | wage development signal 未超过 B3 | unsupported |
| Plan04C | M4C employment Ridge | 908.508412 / 912.418465 | 224.904724 / 225.389926 | registered R1 有微小局部 gain，但 resident primary 恶化且 R2 不稳定 | unsupported |
| Plan05A | M5A city-level B3 error-memory Ridge | 1330.678943 / 1330.678943 | 230.502499 / 230.502499 | historical residual memory 明显恶化 resident aggregate | error_memory_unsupported |
| Plan05B | M5B cross-city relative percentile Ridge | 917.617207 / 917.617207 | 230.549370 / 230.549370 | cross-city relative position 未超过 B3 | unsupported |
| Plan05C | M5C four-source RidgeCV residual | 903.388981 / 959.539982 | 226.838540 / 227.216432 | compact multi-source joint model 两个 scenario 均未超过 B3 | multisource_joint_unsupported |

所有表中 candidate 均保留对应 Plan 的 per-origin、latest、R1/R2、leakage/boundary 与 compatibility artifacts；表格只提供 stage-level overview。

## 4. What actually failed

### A. Target-history extension

TH_CORE、TH_SHAPE、low-capacity HGB target-history variants、cross-city relative position 和 city-level error memory 都没有把人口历史中的可预测 trend 转化为超过 B3 的 residual correction。B3 已经利用了 recent-3 local dynamics；新增特征更多是在重解释同一信号，或引入不稳定的 residual variation。M5A 的 resident pooled MSE 达到 1330.678943，是最清楚的失败案例。

### B. External covariates

Urbanization、density、wage 和 employment 均按 source-specific as-of geometry 构造。外部 source 的 freshness、derived/provenance、quarantine 与 regime 约束没有被静默放宽，但这些信号仍没有形成 resident aggregate gain。Employment 在 registered R1 出现 224.904724 的微小局部改善，但 registered R2 只剩 225.389926，且 resident 两个 scenario 均高于 B3。

### C. Model capacity and interaction

Plan03 的低容量 HGB 没有证明“Ridge 太线性”是 Plan01/02 失败的主要原因。Plan05C 的 RidgeCV 提供了正规化选择和多源联合检验，但不能把 weak signal 变成稳定增益。不同 origin/scenario 的系数翻号、以及多数 outer fits 选择 alpha=100，反而显示增量信号弱且不稳定。

重复的 B3 row/metric compatibility、T-2、source as-of、preprocessing、leakage、R1/R2 和 full-coverage checks 均通过。因此负结果不能归因于一个明显的工程边界错误。

## 5. Strongest observations

1. 没有 S4 ML candidate 稳定超过 resident B3 aggregate hurdle 872.452652。
2. 多个 candidate 改善 resident latest 2021，却恶化历史 aggregate。例如 Plan01 M1A 的 latest 从 3705.427778 降到 3413.978047，但 pooled MSE 从 872.452652 升到 910.349895；这说明存在 latest-regime tension，而不是稳定总体增益。
3. External sources 的 R1 freshness 有时带来局部改善，但在 R2 下会退化或发生 reversal；这在 density、employment 和 multisource joint evidence 中尤其明显。
4. Multisource RidgeCV 大多数 outer fits 选择上界 alpha=100；多个 feature coefficient 在 origin/scenario 间翻号，说明正则化是在压制弱/不稳定增量信号，而不是发现稳定结构。
5. Registered sensitivity 偶尔出现很小的 local improvement，尤其是 employment R1，但没有与 resident primary evidence 对齐。
6. B3 仍是 S4 close 时最强的可部署 fixed model。

## 6. What S4 DID achieve

即使没有分数提升，S4 已建立并验证了：

- leakage-safe pseudo-origin feature generation；
- target-history feature pipeline，包括 trend、shape、residual memory 与 cross-city relative representation；
- source-specific as-of covariate pipeline；
- provenance、quarantine、derived-row 与 regime-aware feature handling；
- formal Ridge residual-correction pipeline；
- formal low-capacity HGB pipeline；
- RidgeCV inner-selection geometry；
- feature ablation 与 mechanism comparison practice；
- R1/R2 information sensitivity evidence；
- 跨 target-history、外生 source、非线性交互、error memory、横截面相对位置和多源联合等机制的 falsification evidence。

这使得后续是否重新开放搜索可以基于证据，而不是基于“还没试过某个模型”的猜测。

## 7. S4 final technical recommendation

建议：

**retain B3_RECENT_3POINT_OLS_TREND as the strongest model entering Main review.**

可在未来 S5 或 conditional reopening 中考虑的方向包括 ensemble/blending、stronger boosting、selective city/global partial pooling、更多 source groups 与 tuning。但当前 S4 evidence 没有要求立即开启这些方向；本 Plan 不启动 S5，也不生成 2023 forecast 或 submission。

## 8. S4 completion evidence against Plan

| S4 completion signal | Evidence |
|---|---|
| reproducible formal ML pipeline | yes；Ridge、HGB、RidgeCV runners 与 machine artifacts |
| feature groups constructed leakage-safe | yes；target T-2、source as-of、provenance/quarantine/regime audits |
| multi-source R1/R2 pipeline | yes；Plan02、Plan03、Plan04A–C、Plan05C |
| model mechanism comparison | yes；linear、nonlinear、error-memory、cross-city、multi-source |
| ablation / failure interpretation | yes；feature ablation、latest-vs-aggregate、origin and source sensitivity |
| stable ML improvement over B3 | no |
| explanation of why candidate families failed | yes |

S4 evidence package is complete for Main review; Stage completion remains Main authority. 本报告不宣告 S4 COMPLETE。

## 9. Main review handoff

1. **What exact branch SHAs were integrated?**
   Plan05A: cbabf585357b0cae069efbf4aa4eed543ecae202；Plan05B: e14bbfb30122a8d397542871c4af017bdcb96193；Plan05C: 0b39868aa6027a26ebcbb13f7ebe1bfcd4394948。三者共同 merge-base 为 decfd7e0c68d14da31eabe5720ab706122c1040d。
2. **Were there conflicts/deviations?**
   No。三条分支是独立文件集，无冲突；未修改 checkpoint、S2 contract、raw/canonical data 或其他 worker artifacts。
3. **Did all S4 evidence survive integration?**
   Yes。Plan01–03、Wave A Plan04A–C、Wave B Plan05A–C 的 machine summaries、metrics、audits、handoffs 与 runners 均在整合树中。
4. **What is the final S4 experiment matrix?**
   见第 3 节；覆盖 target-history、urbanization、density、wage、employment、HGB interaction、error memory、cross-city relative 与 multisource RidgeCV。
5. **Did any material S4 candidate beat B3?**
   No。没有 candidate 同时满足 resident primary aggregate hurdle、稳定性与 R1/R2 interpretation。
6. **What is the strongest model entering Main review?**
   B3_RECENT_3POINT_OLS_TREND。
7. **What did S4 establish despite no score improvement?**
   见第 6 节：完整 leakage-safe feature/ML、source provenance、R1/R2 sensitivity 和 mechanism falsification pipeline。
8. **Which ideas are deferred rather than justified now?**
   Ensemble/blending、stronger boosting、partial pooling、更多弱 source 与 tuning；它们只作为 S5/conditional ideas，不由本 Plan 授权。
9. **Is this final report complete?**
   Yes。它与当前 S4 machine artifacts 和 Wave A/B summaries 对齐。
10. **What is final main SHA / push state?**
    本报告与 Wave B evidence 提交后，integration branch 将 fast-forward 到 main 并正常 push；最终 main SHA 与 push/clean 状态以本次任务交付记录为准。原始 Wave A/B source branches 保留。
