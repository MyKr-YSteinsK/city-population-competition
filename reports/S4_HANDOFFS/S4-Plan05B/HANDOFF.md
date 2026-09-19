# S4-Plan05B Handoff｜跨城市相对位置特征快速筛选

## 状态与结论

- Plan 状态：`evidence complete`；本 worker 只完成 screening，不自动执行 deep dive。
- screening decision：`unsupported`；`deep_dive_recommended=false`。
- `s4_complete=false`、`s5_entered=false`、`final_2023_prediction_run=false`、`submission_run=false`。
- cross-city percentile representation 只由同一 pseudo-origin、同一 target 的 population history <= T-2 构造；R1/R2 完全等价，M5B 只实际 fit 一次。

## Actual changes

1. 新增 `scripts/build_s4_plan05b_cross_city_relative.py`。
2. 新增本目录下的 `HANDOFF.md`、`plan05b_summary.json`、`screening_metrics.csv`、`cross_city_feature_audit.csv`。
3. 未修改 main、Wave A evidence、S2/S3 contract、CURRENT_LEARNING_CHECKPOINT.md 或其他 Wave B worker 文件。

## Execution contract

- integration 后 frozen base / execution starting SHA：`decfd7e0c68d14da31eabe5720ab706122c1040d`。
- current evidence-generation HEAD：`decfd7e0c68d14da31eabe5720ab706122c1040d`。
- branch：`s4/wave-b-cross-city-relative`；独立 worktree 执行；完成后只 push 本 branch。
- common origins：`2011–2021`；resident primary、registered sensitivity；population boundary=`T-2`。

## Actual feature geometry

- candidate：`M5B_B3_PLUS_RIDGE_CROSS_CITY_RELATIVE`；model=`Ridge(alpha=1.0)` residual correction。
- raw B3 state：latest value、recent-3 OLS slope、B3 projected change、recent relative slope；model 只使用对应 cross-city percentile，不重新加入 raw TH_CORE。
- percentile rule：同一 pseudo-origin 的 available cities 使用 average-tie-rank，按 `(rank-1)/(n-1)` 归一化到 `[0,1]`，单个 available city 取 `0.5`。
- preprocessing：outer-origin training rows 内 fit `SimpleImputer(median, add_indicator=True) + StandardScaler + Ridge(alpha=1.0)`。
- 未使用 city ID、外部 source、target year label、tuning、GBDT、2023 forecast 或 submission。

## Core screening metrics

数值为 R1 实际 model run 的 pooled MSE / macro-origin MSE / latest 2021 MSE；R2 因 exact feature equality 不重复 fit。括号为 candidate 相对 B3 的 improvement percentage，正值表示更好。

| target | B3 pooled / macro / latest | M5B pooled / macro / latest | origin wins/losses/ties |
|---|---:|---:|---:|
| resident | 872.452652 / 872.452652 / 3705.427778 | 917.617207 / 917.617207 / 3671.890459 (-5.176734%) | 4/7/0 |
| registered | 225.421212 / 225.421212 / 1485.604861 | 230.549370 / 230.549370 / 1517.981668 (-2.274923%) | 4/7/0 |

## Handoff questions

1. **cross-city relative representation 是否超过 B3？** 否。 resident pooled MSE 为 `917.617207` vs B3 `872.452652`，relative improvement=`-5.176734%`。
2. **resident / registered 是否一致？** 不一致或共同不支持。 resident improvement=`-5.176734%`，registered improvement=`-2.274923%`。
3. **origin wins 是否多数？** resident=`4/7/0`，registered=`4/7/0`，顺序为 wins/losses/ties。
4. **latest 是否支持？** resident latest improvement=`0.905086%`，registered=`-2.179369%`；latest 仅为 proximity diagnostic，不能覆盖 pooled 结论。
5. **R1/R2 是否严格等价？** 是。cross-city feature mismatch count=`0`，candidate 只实际运行 R1，R2 未复制 model workload。
6. **gain/loss 是否明显由极端城市驱动？** 轻量 audit 的 resident top-3 absolute prediction-change cities=`city34,city26,city15`，share=`0.144720`；registered=`city4,city26,city37`，share=`0.134999`。这只是 prediction-change concentration，不是 leave-one-city 结论。
7. **是否值得 deep dive？** `deep_dive_recommended=false`；当前 resident pooled/origin/latest evidence 未满足 Wave B deep-dive rule。

## Verification and evidence index

- B3 row compatibility：`PASS`；B3 metric compatibility：`PASS`。
- cross-city information boundary、R1/R2 equality、population T-2、preprocessing fit rows：`PASS`。
- `screening_metrics.csv`：aggregate、per-origin、pairwise、R1/R2 equality、sanity 与 top-3 prediction-change records。
- `cross_city_feature_audit.csv`：pseudo-origin state、percentile feature、same-cross-section diagnostics、missingness 与 T-2 boundary。
- `plan05b_summary.json`：机器可读 screening decision、metrics、dispersion/missingness 和 scope flags。

## Failure / deviation / remaining uncertainty

- R1/R2 完全等价来自 target-only population information set；因此按 screening efficiency 规则只 fit 一次 M5B，不复制相同 workload。
- cross-city percentile 会丢失绝对尺度，且横截面排名可能受早期 missingness 和极端城市影响；本轮只记录 top-3 prediction-change sanity，不把它解释为稳健性证明。
- 本轮没有 leave-one-city、完整 concentration、调参或其他 feature family；若结果有希望，由 Lab 决定下一轮 deep dive。
