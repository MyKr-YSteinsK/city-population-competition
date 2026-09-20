# S5｜Evidence-driven Iteration Final Evidence Report — Plan07 Wave Consolidation

> 状态：`ready_for_main_stage_review`。本报告整合 S5-Plan01～06H 的真实 evidence；不声明 `S5 COMPLETE`，不进入 S6。Plan07 只做 evidence consolidation，没有新增模型、调参、validation/feature/replay contract 变更、2023 prediction、submission、ensemble、stacking、routing 或 gating。

## 1. Main review decision

完整 Wave 的结论不再是“B3 仍是最强而忽略 Plan06E”。在统一 panel 下：

- resident primary：`F4_LOG_GROWTH_FROM_LATEST` 是当前最强候选，R1 pooled MSE `739.6007`（相对 B3 `+15.227%`），R2 `802.4457`（`+8.024%`）。它应进入 Main Stage Completion Review，但尚不是本报告宣布的 final winner 或部署替代。
- resident comparison：F3 也形成真实 framing signal；CatBoost F2 是 Plan06A-D 中最好的有效 residual control，但 resident 仍比 B3 差 `11.560%–13.724%`。
- registered sensitivity：RLT3 有小幅 legacy gain（`223.6909` vs B3 `225.4212`，`8/11` origin wins），但 Plan06E 的 F3/F4 没有复制该 registered gain；因此不能把 resident 的 F4 信号解释成两个 target 都稳定改善。
- Plan06F 只做 recovery 归档；Plan06G 因 coverage 不完整隔离；Plan06H 全部 city-local candidates unsupported。三者都不进入最终候选胜出比较。

最终推荐：`ready_for_main_stage_review = true`；等待 Main 做 Stage Completion Review。此状态不等价于 S5 完成，也不授权 S6。

## 2. Frozen evaluation boundary and reference

| 项目 | 冻结事实 |
|---|---|
| Origins | `2011–2021`, 11 个 expanding origins |
| Cities | 40 |
| History boundary | target history `<= T-2` |
| Feature contract | `S5-Plan04-FULL-MULTISOURCE-V1`, 144 numeric features |
| Replay contract | `S5-Plan04B-TRAINING-REPLAY-V1`, 2,320 replay rows |
| Primary competition metric | pooled city-year MSE |
| Primary target | resident |
| Mandatory sensitivity | registered、R1/R2、macro-origin、per-origin、latest-2021 |
| Frozen B3 resident | pooled `872.452651515143`; latest-2021 `3705.427777777728` |
| Frozen B3 registered | pooled `225.421212121213`; latest-2021 `1485.604861111118` |

B3 在本次 panel 中从已提交的 S4 frozen OOF 重新计算辅助指标；没有重新训练 B3。机器可读完整结果见 [S5_FINAL_CANDIDATE_COMPARISON.csv](reports/S5_FINAL_CANDIDATE_COMPARISON.csv) 和 [S5_FINAL_CANDIDATE_COMPARISON.json](reports/S5_FINAL_CANDIDATE_COMPARISON.json)。

## 3. Unified evidence panel semantics

MSE 仍是官方 primary competition metric。panel 同时保留：relative improvement vs B3、RMSE、MAE、median absolute error、P90/P95 absolute error、signed bias、macro-origin MSE、origin wins/losses、latest-2021 MSE、worst cities、top-1/top-3 city SSE concentration、已有的 scale-bin diagnostics 和 R1/R2 sensitivity。

本次新增的辅助指标均来自已提交 OOF predictions 的重算，P90/P95 使用绝对误差排序后 `(n-1) * q` 的线性插值。Plan02 的 RLT3 没有持久化 row-level RLT3 OOF，因此其 pooled/RMSE/MAE/median/bias/origin/latest/scale 只复用已提交 aggregate artifacts；无法重算的 quantile、worst-city 和 city-SSE 字段在 panel 中明确写为 `UNAVAILABLE_NOT_PERSISTED`，没有为了补齐而重跑或伪造 OOF。

panel 中的角色分为：

- `frozen_primary_reference`：B3；
- `legacy_valid_candidate`：RLT3；
- `best_valid_residual_family_control`：CatBoost F2；
- `framing_candidate`：F3/F4；
- `valid_residual_family_control`：其余 A-D residual controls，仅作 family 对照，不进入收缩后的 final candidate set。

## 4. Plan01～05 evidence lineage

| Plan | 真实产出 | 收口解释 | 本次处理 |
|---|---|---|---|
| S5-Plan01 | B3 failure-risk diagnostics；历史一阶差分 volatility 的 median abs Spearman `0.4281`，风险方向 `11/11` | 可识别 risk，不等于可修复；唯一激活 robust local trend | 保留为机制起点 |
| S5-Plan02 | B3/B4/RLT3/RLT5；RLT3 resident pooled `861.4909`，registered `223.6909` | resident gain 仅 `1.256%`、`6/11`；scale-outside gain `0.1595%`，未过 strong/complementary gate | RLT3 作为 legacy comparison，不晋级 |
| S5-Plan03 | 旧的 Plan01/02-only final evidence report | 该 report 没有覆盖后续 Plan04～06 Wave；不存在独立新模型证据 | 标记为 superseded，由本报告替换 |
| S5-Plan04 | 1,760-row、144-feature full multi-source matrix；B3 compatibility pass；source-cap violations `0` | feature foundation ready，不是模型胜出证据 | 作为所有 worker 的冻结输入 |
| S5-Plan04B | 2,320 replay rows；44 outer groups；484 inner split rows；264 eligible folds；boundary violations `0` | replay/time-split foundation ready | 作为统一 worker geometry |
| S5-Plan05 | frozen feature/replay/OOF/metric contract；未训练模型 | 只建立强模型并行 Wave 的治理边界 | 作为 06A-H 的共同合同 |

## 5. Plan06A～06H worker ledger

| Plan | Branch tip | classification | 核心事实 | final panel treatment |
|---|---|---|---|---|
| 06A LightGBM | `4ecbd41` | `family_competitive_but_unstable` | F2 residual closest；resident still above B3；F1 direct failed | residual control |
| 06B CatBoost | `9ffdb15` | `family_competitive_but_unstable` | F2 residual is best valid A-D control；still no resident replacement | retained as best A-D control |
| 06C XGBoost | `2570210` | `family_unsupported` | direct failed；residual above B3 | negative control |
| 06D ExtraTrees + Ridge | `01ab61b` | `family_unsupported` | complete OOF/diagnostics；both residual controls above resident B3 | negative controls |
| 06E Delta/Growth | `343970c` | `framing_signal_found` | F3/F4 resident signal；F4 strongest；registered no stable replacement | final framing candidates |
| 06F EBM | `1f97f67` | recovery-only | `88/88` execution and selection counts recovered；core performance files unavailable | excluded; see recovery note |
| 06G Hierarchical | `d468215` | `coverage_incomplete` | combined `2760/3520`; F2 `1000/1760`; 19 outer failures | excluded from candidate ranking |
| 06H City-Local | `baf1a90` | `city_local_unsupported` | all candidates unsupported；all-144 local Ridge `n << p` collapse | excluded; no routing |

## 6. Resident primary comparison

### 6.1 Competition score, stability and latest origin

The values below are the compact view of the full panel. Relative improvement is positive when MSE is lower than B3.

| Candidate | R1 pooled MSE / gain | R1 wins | R1 latest-2021 | R2 pooled MSE / gain | R2 wins | R2 latest-2021 | R2-R1 pooled MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3 | `872.4527 / 0.000%` | — | `3705.4278` | `872.4527 / 0.000%` | — | `3705.4278` | `0.0000` |
| RLT3 | `861.4909 / +1.256%` | `6/11` | `3666.8000` | `861.4909 / +1.256%` | `6/11` | `3666.8000` | `0.0000` |
| best A-D residual: CatBoost F2 | `973.3091 / -11.560%` | `4/11` | `3401.4606` | `992.1917 / -13.724%` | `3/11` | `3378.6405` | `18.8826` |
| F3 delta | `788.8413 / +9.583%` | `4/11` | `2988.5422` | `818.4477 / +6.190%` | `5/11` | `3469.7645` | `29.6064` |
| F4 log growth | **`739.6007 / +15.227%`** | `5/11` | `2956.2890` | **`802.4457 / +8.024%`** | `7/11` | `3595.0013` | `62.8451` |

F4 is the strongest resident candidate in pooled competition score across both scenarios. Its latest-2021 error improves versus B3 in both R1 and R2, but R2 remains materially higher than R1. That temporal sensitivity is a review risk, not a reason to hide the aggregate gain.

### 6.2 Typical error, tail risk, bias and city concentration

| Candidate / scenario | RMSE | MAE | Median AE | P90 / P95 AE | bias | top-1 / top-3 city SSE |
|---|---:|---:|---:|---:|---:|---:|
| B3 R1 | `29.5373` | `14.1970` | `4.8333` | `36.9167 / 55.5083` | `-3.4447` | `28.61% / 50.22%` |
| B3 R2 | `29.5373` | `14.1970` | `4.8333` | `36.9167 / 55.5083` | `-3.4447` | `28.61% / 50.22%` |
| CatBoost F2 R1 | `31.1979` | `15.5281` | `5.4297` | `38.8364 / 67.9965` | `-1.5114` | `27.65% / 50.86%` |
| CatBoost F2 R2 | `31.4991` | `15.7819` | `6.0535` | `38.0452 / 68.9007` | `-1.4461` | `28.37% / 50.71%` |
| F3 R1 | `28.0863` | `14.2429` | `5.1460` | `37.6124 / 65.5255` | `-1.1922` | `25.51% / 47.96%` |
| F3 R2 | `28.6085` | `14.2271` | `4.9167` | `37.5963 / 61.3884` | `-1.9910` | `26.70% / 49.52%` |
| F4 R1 | **`27.1956`** | **`13.6773`** | **`4.6633`** | `37.5663 / 61.7663` | `-1.2072` | **`24.69% / 48.55%`** |
| F4 R2 | `28.3275` | `14.0268` | `4.9517` | `38.3911 / 59.4285` | `-1.1823` | `26.85% / 50.09%` |

Interpretation:

- F4 improves typical error, bias magnitude, and city SSE concentration relative to B3 in the resident primary.
- F4's P90 is roughly B3-like and its R1 P95 is higher than B3; the gain is not a blanket tail-risk win. The panel keeps tail metrics visible so Main does not promote on MSE alone.
- The shared worst-SSE cities for the resident comparison remain concentrated around `city5`, `city24`, and `city21` for the worker candidates; no post-hoc city routing is inferred.
- F4 improves all resident low/medium/high scale bins in both R1 and R2. F3 also improves all three resident bins, while A-D controls do not carry a comparable persisted scale-bin panel.

### 6.3 Resident decision

`F4_LOG_GROWTH_FROM_LATEST` is the strongest resident candidate for Main review. It is not declared a global winner because the registered sensitivity is negative, R2 is weaker than R1, and tail risk is mixed. `F3` remains a credible framing comparator. `RLT3` and the A-D residual controls are not resident replacements.

## 7. Registered sensitivity comparison

### 7.1 Competition score and temporal behavior

| Candidate | R1 pooled MSE / gain | R1 wins | R1 latest-2021 | R2 pooled MSE / gain | R2 wins | R2 latest-2021 | R2-R1 pooled MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3 | `225.4212 / 0.000%` | — | `1485.6049` | `225.4212 / 0.000%` | — | `1485.6049` | `0.0000` |
| RLT3 | **`223.6909 / +0.768%`** | **`8/11`** | `1483.1500` | **`223.6909 / +0.768%`** | **`8/11`** | `1483.1500` | `0.0000` |
| best A-D residual: CatBoost F2 | `227.3468 / -0.854%` | `7/11` | `1535.7843` | `227.1253 / -0.756%` | `7/11` | `1508.4479` | `-0.2215` |
| F3 delta | `228.3363 / -1.293%` | `5/11` | `1542.8513` | `227.5234 / -0.933%` | `5/11` | `1516.4408` | `-0.8129` |
| F4 log growth | `243.2445 / -7.907%` | `4/11` | `1563.9760` | `246.8899 / -9.524%` | `4/11` | `1558.6918` | `3.6454` |

RLT3's `223.6909` is a real small signal in the Plan02 artifact, with `8/11` origin wins and a `0.768%` pooled gain. It does not become the final candidate because its resident gain is below the pre-registered `2%` strong gate, resident wins are only `6/11`, and scale-outside improvement is only `0.1595%`. Plan06E's F3/F4 are not equivalent to RLT3 on registered: F3 is only locally helpful in the high-scale bin and is worse in low/medium bins; F4 worsens all registered scale bins in the persisted diagnostics.

### 7.2 Registered error profile

| Candidate / scenario | RMSE | MAE | Median AE | P90 / P95 AE | bias | top-1 / top-3 city SSE |
|---|---:|---:|---:|---:|---:|---:|
| B3 R1/R2 | `15.0140` | `5.6061` | `2.7500` | `11.3500 / 16.6917` | `-0.6273` | `42.32% / 80.90%` |
| RLT3 R1/R2 | `14.9563` | `5.4909` | `3.0000` | unavailable | `-0.6000` | unavailable |
| CatBoost F2 R1 | `15.0780` | `6.1450` | `3.1339` | `11.6265 / 17.7061` | `+0.3187` | `41.67% / 77.67%` |
| CatBoost F2 R2 | `15.0707` | `6.2656` | `3.4586` | `12.5144 / 17.5676` | `+0.3205` | `41.70% / 76.96%` |
| F3 R1 | `15.1108` | `6.1630` | `3.2029` | `11.8582 / 16.8503` | `+0.1670` | `42.29% / 80.62%` |
| F3 R2 | `15.0839` | `6.2135` | `3.4199` | `12.0445 / 17.4985` | `+0.1918` | `42.52% / 80.29%` |
| F4 R1 | `15.5963` | `6.5554` | `3.3530` | `12.6941 / 20.5268` | `+0.3480` | `40.04% / 76.31%` |
| F4 R2 | `15.7127` | `6.4636` | `3.2917` | `13.0446 / 20.2158` | `+0.2305` | `39.36% / 77.21%` |

Registered interpretation is therefore sensitivity-only: RLT3 is a small legacy signal, CatBoost is close but still worse in MSE, F3 is near B3 but not better, and F4 is clearly not a registered replacement. The resident-primary/registered-sensitivity contract means F4 can remain under Main review as a resident model framing while the registered result blocks a claim of universal target improvement.

## 8. Unsupported, incomplete and recovery evidence

### Plan06F recovery

The persistent recovery note is [reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md](reports/S5_HANDOFFS/S5-Plan06F/RECOVERY.md). Local Codex session evidence verifies the first run's `88/88` outer fits, `C1=64/88`, `C2=24/88`, `F1=44/44 C1`, `F2=20 C1 / 24 C2`, geometry `44/264/8`, and the `1560` correct inner-row count versus the erroneous `1632` assertion. OOF, aggregate/per-origin metrics, inner MSE, term importance and city errors were not persisted. Selection counts are not model-performance evidence.

### Plan06G coverage

Plan06G is isolated as `coverage_incomplete`: combined OOF `2760/3520`, F2 `1000/1760`, and 19 outer fit failures. Its abnormal registered/R1 F2 MSE is not admissible for candidate selection. No fallback OOF rows are invented.

### Plan06H city-local

Plan06H's six city-local candidates are all `city_local_unsupported`. The all-144 local Ridge's `n << p` collapse is a diagnostic failure. No city identity, local error, or persistence statistic is converted into routing or gating.

## 9. Final candidate contraction

The final candidate set for Main review is intentionally small:

1. B3 frozen reference;
2. RLT3 legacy registered-sensitivity comparator;
3. CatBoost F2 as the best valid Plan06A-D residual control;
4. F3 delta framing;
5. F4 log-growth framing, strongest resident candidate.

LightGBM, XGBoost, ExtraTrees and Ridge remain in the panel as reproducible controls, but are not promoted into the final candidate set. Plan06F/G/H are not ranked as if their evidence were complete or admissible.

## 10. Main handoff and boundary

The concise handoff is [reports/S5_MAIN_HANDOFF.md](reports/S5_MAIN_HANDOFF.md). The machine-readable registry is [reports/S5_EXPERIMENT_REGISTRY.csv](reports/S5_EXPERIMENT_REGISTRY.csv).

`ready_for_main_stage_review = true` means the evidence package is ready for Main's review. It does not mean `S5 COMPLETE`; this branch does not enter S6, produce a 2023 forecast, or create a submission.
