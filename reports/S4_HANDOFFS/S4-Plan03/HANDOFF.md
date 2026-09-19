# S4-Plan03 Handoff｜低容量 GBDT 的目标历史与城镇化非线性交互

## 状态与结论

- Plan 状态：`evidence complete`；本 Plan 不支持把低容量 GBDT 或当前两组 feature 升级为可部署的 material ML 增益。
- B3 仍是 strongest fixed baseline。M3A/M3B/M3C 均未超过 resident pooled MSE hurdle `872.452651515143`。
- Plan01/02 失败不能主要归因于 Ridge 的线性容量不足：M3A 只使用 target-history state 的非线性 GBDT，resident aggregate 仍明显劣于 B3。
- M3B 相对 M3A 在 resident 上有小幅 R1/R2 形状增益，但 registered 变差，且两者都没有接近 B3 hurdle。
- M3C 在 R1 下相对 M3B 有 resident 局部增益，在 R2 下明显 reversal；registered 虽相对 M3B 改善，但仍劣于 B3。因此 `conditional urbanization interaction` 不支持。
- `s4_complete=false`、`s5_entered=false`、`final_2023_forecast_run=false`、`submission_run=false`。

## Actual changes

1. 新增 `scripts/build_s4_plan03_hgb_interaction.py`。
2. 新增本目录下的 Plan03 evidence：feature/model definitions、dataset/training geometry、missingness、aggregate/per-origin/latest、三层 ablation、R1/R2 等价性与 urbanization sensitivity、concentration、leave-one-origin/city、leakage、B3 compatibility 与 machine-readable summary。
3. 更新 `docs/PROJECT_CONTEXT.md`、`docs/PROJECT_FILE_MAP.md`、`docs/CODEX_INSTRUCTIONS.md`，记录 Plan03 的实际结论和工作边界。
4. 未修改 raw XLSX、canonical data、S2 contract 或 `CURRENT_LEARNING_CHECKPOINT.md`。

## Execution contract

- 实际执行起始 SHA：`23c770e76271b1db5ac4316850bbaff798b60e81`。
- 执行前 `git fetch origin main`；branch 为 `main`，upstream 为 `origin/main`，起始 `HEAD == origin/main`，工作树无用户本地修改。
- 计划内实际运行：

  ```text
  git fetch origin main
  python -m py_compile scripts/build_s4_plan03_hgb_interaction.py
  python scripts/build_s4_plan03_hgb_interaction.py
  ```

- 首次 runner 暴露了一个本地审计实现错误：`no contamination / no quarantine reintroduction` 的事实布尔值被直接当作失败值汇总；修正为“check passed”语义后重新运行，最终 runner 通过。
- 最终产物规模：`7040` prediction rows、`5280` candidate rows、`176` per-origin metric rows、`132` training geometry rows、`39690` leakage audit rows。

## Actual model and feature geometry

- 唯一新增模型家族：`HistGradientBoostingRegressor`。
- 实际固定配置：

  ```text
  loss=squared_error
  learning_rate=0.05
  max_iter=100
  max_depth=2
  min_samples_leaf=20
  l2_regularization=1.0
  early_stopping=False
  random_state=0
  ```

- preprocessing：每个 target/scenario/outer-origin/model 独立使用 `SimpleImputer(strategy="median", add_indicator=True)`；只在训练 rows fit；不使用 StandardScaler。
- raw feature counts：M3A=`4`、M3B=`6`、M3C=`9`。M3C 早期训练 fold 产生 3 个 urbanization missing indicators；所有 validation scope 的 feature missing count 为 0。
- M3A=`TH_CORE`；M3B=`TH_CORE + TH_SHAPE`；M3C=`TH_CORE + TH_SHAPE + URB_LEVEL_V1 + URB_DYNAMICS_V1`。未加入 audit-only `urbanization_latest_year` / staleness。
- B3 prediction 与 residual label 继续使用 `population history <= y-2` / `<= T-2`；M3C 的 urbanization features 继续复用 S2 `required_lag_years`，R1/R2 独立重跑。
- 没有 tuning、model sweep、其他 model family、第三个 source、city identity、manual routing、ensemble、2023 forecast 或 submission。

## Core aggregate metrics

数值为 `pooled city-year MSE / latest 2021 MSE`；R1/R2 对 B3、M3A、M3B 完全等价，M3C 单独列出。

| target | scenario | B3 | M3A | M3B | M3C |
|---|---|---:|---:|---:|---:|
| resident | R1 | 872.452652 / 3705.427778 | 938.209015 / 3346.481908 | 934.874827 / 3199.877531 | 911.168287 / 3003.686941 |
| resident | R2 | 872.452652 / 3705.427778 | 938.209015 / 3346.481908 | 934.874827 / 3199.877531 | 974.098701 / 3623.402624 |
| registered | R1 | 225.421212 / 1485.604861 | 229.697127 / 1500.288498 | 238.290378 / 1566.428228 | 230.575716 / 1535.235895 |
| registered | R2 | 225.421212 / 1485.604861 | 229.697127 / 1500.288498 | 238.290378 / 1566.428228 | 234.115366 / 1577.682231 |

R1 resident 相对 B3：M3A `+7.5370%`、M3B `+7.1548%`、M3C `+4.4376%`；R2 M3C 变为 `+11.6506%`。registered R1/R2 的所有 candidate 也都高于 B3。

## Handoff questions

1. **B3 是否再次精确复现？** 是。B3 row-level prediction/residual/squared-error compatibility 与 S3 frozen rows 通过；metric-level pooled/macro/latest compatibility 也通过。resident pooled/macro 为 `872.452651515143`，registered 为 `225.421212121213`。

2. **实际 GBDT configuration 是什么，是否严格无 tuning？** 使用上面列出的单一固定 HGB 配置，`early_stopping=False`、`random_state=0`；没有 parameter search、model sweep、validation-driven adjustment 或 ensemble。训练 geometry 与 model definitions 已落盘。

3. **M3A 是否超过 B3？** 否。resident pooled MSE 为 `938.209014586509`，比 B3 高 `65.756363071366`；R1/R2 完全相同。latest resident 降到 `3346.481908`，但不能抵消 historical aggregate 变差。

4. **如果 M3A 失败，能否排除“Ridge 太线性”作为主要解释？** 可以显著削弱该解释。M3A 已把同一组 TH_CORE 交给低容量非线性 GBDT，仍在 resident aggregate、registered sensitivity 和 origin stability 上未过 B3；因此当前证据更支持“TH_CORE 没有稳定增量 residual signal”，而不是单纯 Ridge 表达能力不足。

5. **M3B 是否相对 M3A 有稳定 shape increment？** 否。resident R1/R2 均从 `938.209015` 降到 `934.874827`，相对改善约 `0.3554%`，per-origin 为 `7 wins / 4 losses`；但 registered 从 `229.697127` 升到 `238.290378`，为 `4 wins / 7 losses`。且 M3B 仍远差于 B3，不能称为稳定 material shape increment。

6. **M3C 是否相对 M3B 有 conditional urbanization increment？** 不支持。resident R1 相对 M3B 改善 `23.706540` MSE、`7/4` origins，但 R2 变差 `39.223874`、虽为 `8/3` origins；registered R1/R2 相对 M3B 分别改善 `7.714662`/`4.175012`，但仍高于 B3。没有跨 target、scenario、aggregate 的稳定增量。

7. **M3C 的 R1 → R2 是否明显衰减 / reversal？** 是。resident pooled MSE 增加 `62.930414`，latest 增加 `619.715683`；registered pooled 增加 `3.539650`，latest 增加 `42.446336`。M3A/M3B 的 R1/R2 row 和 metric equivalence 均为 PASS。

8. **resident aggregate 最强 candidate 是谁？** R1 是 M3C（`911.168287`），但仍比 B3 差 `4.4376%`；R2 是 M3B（`934.874827`），M3C 因 freshness stress 变为最差。没有 candidate 通过 resident hurdle。

9. **registered sensitivity 是否支持同一结论？** 支持“没有 material ML 增益”的结论。M3A 是两个 scenario 下最接近 B3 的 candidate，但仍分别高 `1.8969%`；M3B/M3C 更差，且 latest registered 全部恶化。

10. **resident 2021 latest 是否与 aggregate 冲突？** 是。所有 candidate 的 resident latest 都低于 B3，M3C R1 最低为 `3003.686941`；但所有 candidate 的 historical pooled MSE 都高于 B3，M3C R2 还明显 reversal。这只能称 latest-regime local improvement，不能称整体模型更强。

11. **per-origin gain 是否 broad-based？** 否。相对 B3，M3A/M3B 在 resident 两个 scenario 都是 `4/7` wins/losses；M3C 是 R1 `5/6`、R2 `6/5`。registered 三个 candidate 均没有胜过半数 origins。M3C 相对 M3B 的局部 wins 不能弥补其大误差。

12. **improvement / degradation 是否被少数城市支配？** 存在明显集中。相对 B3，resident M3C 的 top-3 positive-delta city 贡献约 `71.4%`/`71.8%`，registered 约 `60.0%`/`58.8%`；candidate top-3 total SSE share 仍约 `49%`（resident）和 `80%`（registered）。移除任意一个城市后，M3A/M3B/M3C 的 candidate improvement 仍全部为负，因此没有被某一城市移除后恢复成 aggregate gain。

13. **leave-one-origin / leave-one-city 后 leader 是否稳定？** B3 在 resident leave-one-origin 中存活 `10/11`，registered `11/11`；resident 仅省略 2014 时 R1 由 M3C、R2 由 M3A 暂时领先。leave-one-city 后所有 candidate 仍劣于 B3，结论稳定。

14. **missingness / information boundary / leakage 是否全部 PASS？** 是。TH feature manifest 与 Plan01 row-level semantics compatibility PASS；urbanization manifest 与 Plan02 semantics、R1/R2 cap PASS；validation 为完整 `40 × 11`，M3C validation missing 为 0；population T-2、urbanization cap、preprocessing/model-fit boundary、leakage audit 全部 PASS。

15. **Plan01、Plan02 的 negative conclusions 经非线性检验后应如何更新？** Plan01 的主要失败不能主要归因于 Ridge 太线性：M3A 仍失败。TH_SHAPE 在 M3B 中只产生小的 resident 相对 M3A 改善，registered reversal，仍应判为 unsupported。Plan02 的 urbanization 在 M3C 中仅 R1 相对 M3B 局部改善，R2 明显 reversal，不能升级为 conditional interaction；应继续判为 unsupported。

16. **下一轮最高信息价值问题是什么？** 停止围绕同一 TH/urbanization feature 反复换模型，优先由 Main/Lab 评审一个具有独立 source mechanism 的新 feature source（例如人口密度），并保持同一 S2 boundary、B3 residual geometry 与 robustness gate。不要自动执行下一 Plan，也不要进入 S5。

## Evidence index

- `feature_manifest.csv`、`model_definitions.csv`：冻结 TH/URB semantics、model scope 和实际 HGB configuration。
- `modeling_dataset_diagnostics.csv`、`training_geometry.csv`、`feature_missingness_imputation.csv`：每个 outer-origin 的 rows、raw/expanded feature count、imputation 与 model-fit geometry。
- `aggregate_metrics.csv`、`per_origin_metrics.csv`、`per_origin_predictions_or_residuals.csv`：primary metrics 与可复核预测/残差。
- `pairwise_b3_comparison.csv`、`feature_ablation_comparison.csv`、`latest_origin_comparison.csv`：B3 hurdle 与三层 ablation。
- `r1_r2_equivalence_check.csv`、`r1_r2_urbanization_sensitivity.csv`：target-history equivalence、urbanization feature freshness 与 M3C metric sensitivity。
- `city_improvement_concentration.csv`、`leave_one_origin_out.csv`、`leave_one_city_sensitivity.csv`：稳定性、城市集中度和鲁棒性。
- `th_feature_compatibility_check.csv`、`urbanization_information_boundary.csv`、`leakage_audit.csv`：前序 semantics、source boundary 和 leakage evidence。
- `b3_compatibility_check.csv`、`b3_metric_compatibility_check.csv`、`plan03_summary.json`：baseline compatibility、acceptance flags 和机器可读汇总。

## Failure / deviation / remaining uncertainty

- 最初 runner 的 leakage aggregation 将“无污染”事实值误当作失败值，已修复并以同一 starting SHA 重新执行；这是实现级审计 bug，不是计划边界或数据 contract deviation。
- 本轮没有使用真实 publication timing；R1/R2 仍是 observed-cap replay assumptions。M3C 的 R1/R2 reversal 是该 frozen information boundary 下的实测敏感性，不应叙述为真实发布时间因果。
- 训练 fold 的 urbanization missingness 是历史 coverage 边界，使用 fold-local median + missing indicator；没有填 0、插值或 future fill。后续若改变 imputation、lag、feature semantics 或模型配置，必须重新生成全套 evidence。
