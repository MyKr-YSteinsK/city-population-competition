# S4-Plan04C Handoff｜就业规模信号与来源可靠性快速筛选

## 状态与结论

- Plan 状态：evidence complete；本轮 employment-scale residual candidate 不支持超过 B3，deep_dive_recommended=false。
- resident 是 primary decision target：M4C 在 R1/R2 的 pooled MSE 都高于 B3，分别高 4.1327%、4.5809%。
- registered sensitivity 的 pooled MSE 在 R1/R2 略低于 B3，但幅度只有 0.2291%、0.0139%，且 per-origin 都不是多数 wins，不足以支持稳定 employment signal。
- derived canonical rows 已纳入 feature，但 provenance 始终可见；正向 SSE gain 主要不由 derived window 驱动，当前不标记为 provenance-sensitive employment signal。
- s4_complete=false、s5_entered=false、final_2023_prediction_run=false、submission_run=false。

## Execution contract

- Frozen base：b8c302abe7ebf1710869aaa55a00d4b926d2bcac。
- 独立 worktree：D:\CS\MLDM\city-population-competition-s4-employment。
- Branch：s4/wave-a-employment。
- Evidence commit：1b4f85a（本地提交，完整 SHA 由 Git delivery 记录）。
- 没有在 main 上实验或提交；没有修改 CURRENT_LEARNING_CHECKPOINT.md、docs/PROJECT_CONTEXT.md、docs/PROJECT_FILE_MAP.md、docs/CODEX_INSTRUCTIONS.md、S2 contract、前序 handoff 或其他 Wave A worker 文件。
- 实际执行：

      python -m py_compile scripts/build_s4_plan04c_employment_signal.py
      python scripts/build_s4_plan04c_employment_signal.py
      git diff --check

- runner 最终输出 S4_PLAN04C_EMPLOYMENT_SIGNAL_PASS；中途发现并修复了三个实现级问题：scenario loader 模块引用、最早 training year 的 S2 policy-row 缺失边界、summary 字段别名。修复后用同一 frozen base 完整重跑通过。

## Actual changes

1. 新增 scripts/build_s4_plan04c_employment_signal.py。
2. 新增本目录下的 screening_metrics.csv、employment_provenance_boundary_audit.csv 和 plan04c_summary.json。
3. 新增本 Handoff；没有修改 raw XLSX、canonical data、quarantine data、requirements 或项目共享状态文件。

## Actual model and feature geometry

- 只运行 B3_RECENT_3POINT_OLS_TREND 与 M4C_B3_PLUS_RIDGE_EMPLOYMENT。
- M4C 形式为 final = B3 + predicted_B3_residual。
- 固定 Ridge(alpha=1.0)；每个 target/scenario/outer-origin 独立 fit。
- preprocessing 是 outer-origin training rows 内的 SimpleImputer(strategy=median, add_indicator=True) + StandardScaler；没有 tuning、model sweep、GBDT 或 ensemble。
- employment features：
  - employees_log_latest = log1p(latest allowed positive employeesNumber)；
  - employees_recent_log_growth = 最近两个 allowed positive observations 的真实年份 annualized log growth；
  - employees_recent3_log_slope = 最近三个 allowed positive observations 对真实年份的 log(employeesNumber) OLS slope。
- R1 每个 pseudo-target 只允许 employment year <= T-1；R2 只允许 <= T-2。训练 row 也按其自身 pseudo-target year 独立构造 cap；quarantine 不回流。
- 没有加入 sector shares、unemployment、wage、density、city ID 或其他 covariates；没有生成 2023 prediction / submission。

## Source and provenance boundary

- canonical employment：773 rows、40 cities、2000–2022；non-derived canonical rows 682，provenance 记录的 sector-sum derived rows 91。
- quarantined source rows：80 rows，59 unique normalized keys；source status 保持 source_plus_remap_quarantine_and_derived。
- observed latest employment year 2022，relative to reference target year 2023 的 observed snapshot lag 为 1；这不是 publication timing 证明。
- validation provenance audit 共 1760 rows（2 targets × 2 scenarios × 11 origins × 40 cities）：
  - latest feature 来自 derived row：220/1760 rows；
  - recent-2 window 的 derived feature cells：440；
  - recent-3 window 的 derived feature cells：660；
  - recent-3 中包含 derived row：220/1760 rows；这些 rows 同时满足 derived-heavy（至少 2 个 derived points）。
- 2021 latest validation subset 共 160 rows，其中 latest derived 20 rows，recent-3 derived cells 60。
- 全部 20240 条 training/validation provenance boundary audit 为 PASS；exact (city, year, value) quarantine candidate hit 为 0，quarantine_value_reintroduced=no。normalized city-year key 的重叠本身不被当作回流，因为 canonical 中可能存在不同来源记录；本轮 gate 使用 exact quarantined value triple。

## Core aggregate evidence

数值为 pooled city-year MSE / latest 2021 MSE；百分比是 M4C 相对 B3 的 pooled MSE improvement，正数代表更好。

| target | scenario | B3 | M4C | relative improvement | origin wins / losses |
|---|---|---:|---:|---:|---:|
| resident | R1 | 872.452652 / 3705.427778 | 908.508412 / 3623.586025 | -4.1327% | 6 / 5 |
| resident | R2 | 872.452652 / 3705.427778 | 912.418465 / 3671.838952 | -4.5809% | 6 / 5 |
| registered | R1 | 225.421212 / 1485.604861 | 224.904724 / 1482.314932 | +0.2291% | 5 / 6 |
| registered | R2 | 225.421212 / 1485.604861 | 225.389926 / 1487.627151 | +0.0139% | 4 / 7 |

## Handoff questions

1. **M4C R1/R2 是否超过 B3？**
   resident：否。R1 pooled MSE 908.508412，比 B3 高 36.055761；R2 为 912.418465，高 39.965813。两种 scenario 的 latest 2021 MSE 虽分别改善 2.2087%、0.9065%，但不能抵消全域 pooled loss。registered：pooled MSE 只在数值上略胜，不能升级为稳定 signal。

2. **registered 是否支持？**
   仅作 sensitivity，方向弱支持 pooled 指标：R1 224.904724 < 225.421212，R2 225.389926 < 225.421212；但 R1/R2 origin wins/losses 为 5/6、4/7，latest 在 R2 反而恶化 0.1361%，因此不支持 material、跨 origin 稳定增益。

3. **per-origin wins/losses？**
   resident R1 wins 2011;2012;2016;2017;2020;2021、losses 2013;2014;2015;2018;2019；R2 相同。registered R1 wins 2016;2018;2019;2020;2021、losses 2011;2012;2013;2014;2015;2017；R2 wins 2013;2018;2019;2020、losses 2011;2012;2014;2015;2016;2017;2021。

4. **R1→R2 是否明显衰减？**
   是。M4C resident pooled MSE 从 908.508412 增至 912.418465（+3.910053），latest 增 48.252928；registered pooled 增 0.485202，latest 增 5.312219。对每个 target，M4C validation employment feature 在 433/440 个 origin×city scope 中发生变化；两个 target 合计为 866/880。B3 的 R1/R2 prediction/residual/metric 数值保持完全一致。

5. **validation/latest features 中有多少来自 derived rows？**
   全部 validation audit 的 latest feature 为 220/1760 rows；recent-2 derived cells 440；recent-3 derived cells 660，包含 derived 的 recent-3 rows 220/1760。2021 latest subset 为 160 rows，其中 latest derived 20、recent-3 derived cells 60。这些是 feature-use counts，不把同一 canonical derived key 在不同 target/scenario scope 的重复使用误报成更多独立 source rows。

6. **gain/loss 是否明显与 derived provenance 关联？**
   否。按 row-level positive SSE gain 计算，resident derived-window share 为 R1 4.53%、R2 4.93%；candidate wins 落在 derived recent-3 window 的比例分别为 21/183、18/181。registered 对应 positive-gain share 为 2.74%、5.16%。因此没有出现“主要改善依赖 derived-latest / derived-heavy windows”的证据，provenance-sensitive employment signal 不成立。

7. **quarantine 是否零回流？**
   是。employment_provenance_boundary_audit.csv 的 20240 rows 全部 boundary_status=PASS，exact quarantine candidate hit 为 0，且所有 rows 的 quarantine_value_reintroduced=no。runner 只从 canonical employment map 构造 features，quarantine 文件只用于审计边界。

8. **是否值得 deep dive 或拆产业结构？**
   当前不值得。deep_dive_recommended=false：resident 未过 B3，也未达到“距 B3 ≤1% 且 origin/latest evidence 明显支持”的条件；derived provenance 也没有造成明确 gain concentration。因此本 worker 不继续拆 sector shares、不进入 robustness 扩展、不宣布 S4 complete 或进入 S5。

## Verification actually passed

- python -m py_compile scripts/build_s4_plan04c_employment_signal.py。
- 完整 runner：B3 row-level 与 aggregate sanity PASS；M4C candidate rows 1760；R1/R2 policy boundary PASS；provenance tracking PASS；quarantine zero reintroduction PASS。
- 独立只读核对：canonical/provenance/quarantine counts、validation audit counts、screening metric scope counts、所有 boundary statuses 均符合上述数字；git diff --check PASS。
- 当前变更只位于本 worker branch 的新增 runner、Plan04C evidence 与 Handoff；未触碰计划禁止文件。

## Remaining risks / known limits

- R1/R2 是 S2 frozen contract 下的 observed-cap replay assumptions，不是 source publication timing 的事实证明。
- 本轮按 screening-first 规则未做 leave-one-origin、leave-one-city、完整 SSE concentration 或其他 robustness 扩展；这些不是本 Plan 的 acceptance。
- latest 2021 improvement 与 historical pooled loss 并存，不能只用 latest diagnostic 宣布 candidate 成功。

## Delivery

- 交付目标：只 push s4/wave-a-employment，不 merge main。
- plan04c_summary.json 的机器标志：deep_dive_recommended=false、s4_complete=false、s5_entered=false。
