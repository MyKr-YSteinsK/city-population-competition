# S8 Closeout

## 1. S8 最终解决了什么

S8 为 2023 城市人口预测确认城市映射、目标口径和年份，处理来源冲突，冻结
可审计 hidden truth，并复核 S1 历史审计的捕获、漏检和证据边界。

已完成 Target Forensics、Candidate Truth Reconstruction、Truth Freeze 和
Post-hoc S1 Audit Validation；未训练模型或向 S9 暴露 truth。

## 2. 最终 hidden truth

truth_version 为 candidate_truth_2023_v2，freeze_status 为
S8_TRUTH_V2_FROZEN_WAITING_FOR_S9。2023 competition target 对应真实统计
年份 2022，target semantics 为 resident population，单位为万人，evaluator
按 ROUND_HALF_UP_TO_INTEGER_WAN 输出整数万人。

40/40 城市已解析，provisional/unresolved 均为 0；TIER1 22 行、TIER2 18 行、
TIER3/TIER4 均为 0。evaluator SHA256：
de8ae0feb8b4061610b82a235a673fc2a841d2969f04924d7f69185c5e4db02c

该 truth 只用于 hidden evaluator，不用于模型选择，不共享给 S9。city40
registered population 2021 = 65 是 manifest 保留的来源冲突，不是冻结预测目标。

## 3. S1 post-hoc audit 结论

S1 的 schema、key、coverage、quarantine、provenance 和 source boundary 处理
扎实；Employment 的 shifted blocks、sector-sum invariant、quarantine 和 derived
reconstruction 是最成功的部分。无法证明的 cell 没有被凭插值制造真值。

最终 inventory 有 382 条 material decisions：4 条 externally verified correct、
0 条 externally verified wrong、378 条 internally supported but not externally
exact-cell verified。378 不是外部证明正确，382 也不能整体称为 external proof。

## 4. 唯一 confirmed critical miss

city40 / registered population / 2021 = 65 是唯一外部定义匹配证据确认的 S1
漏检。其前一年约 270，post-hoc temporal rule 识别 270 → 65 为 EXTREME，
官方对照值为 268.10 万人，因此仍为 FALSE_NEGATIVE_MISSED_ANOMALY 和
CRITICAL。S1 population audit 缺少 temporal jump、relative YoY、robust
deviation 和 resident/registered cross-field 检查。

两条 employment TRUE_YEAR_SHIFT block 已被 S1 捕获；ctiy22 -> city22、
ctiy31 -> city31 已有 LABEL_REMAP evidence；没有 confirmed global city
permutation。

## 5. S1 判断的总体评价

后续 audit 支持 zero sentinel、employment quarantine、sector reconstruction、
label repair、污染 block 隔离和 test.xls source boundary，但这不等于所有历史
cell 的真实值已恢复。

Structural integrity：STRONG；repair/quarantine：STRONG TO ADEQUATE；
employment audit：STRONG；population anomaly detection：MIXED / NEEDS
IMPROVEMENT；semantic/external coverage：LIMITED。整体基础良好，但有 material
false-negative blind spot。

## 6. 下一次审计规则

每个 entity × year × numeric field 检查 first difference、relative YoY、
isolated spike/drop、return-to-baseline、rolling median/MAD 和 cross-field
consistency；同时扫描 malformed label、duplicate trajectory、lagged copy、
city permutation、year shift、column swap、数量级错误和 block contamination。

流程固定为 Detect → Verify source → Gather evidence → Classify → Repair /
Quarantine / Preserve。检测不自动等于修改；报告区分 EXTERNALLY_VERIFIED、
INTERNALLY_SUPPORTED 和 CAUTIOUS_UNRESOLVED，审计深度按影响排序。

## 7. Evidence limits

当前 external universe 是有限且定义匹配的抽样 universe，不是全部历史 cell 的
ground truth。Plan04/05 记录 146 条 external rows，其中 93 个 observed cells
可用于 S1 对照；4/0/378 和唯一 verified miss 不能外推。本 closeout 不新增
外部取证，不修改 frozen truth，不把 S1 结论当作 S9 特征。

## 8. Canonical artifacts

正式阅读清单只有以下 5 项：

1. S8_CLOSEOUT.md
2. reports/S8_TRUTH/truth_v2/candidate_truth_2023_v2_evaluator.csv
3. reports/S8_TRUTH/truth_v2/truth_manifest_v2.json
4. reports/S8_AUDIT_VALIDATION/S8-Plan05/S1_POSTHOC_AUDIT_VALIDATION_FINAL_REPORT.md
5. reports/S8_AUDIT_VALIDATION/S8-Plan05/s1_audit_scorecard_v2.csv

Plan01–05 的其他 evidence 留在 Git/worktree history 供 forensic 回溯，不列入
canonical 清单；本 Plan 不删除或复制 artifacts。

## 9. Isolation / Return Point

本文件位于独立 codex/s8-closeout 分支，不 merge main。S9 branch、predictions、
metrics 未读取，truth 和 S1 data 未写回 main。原 S8-C～F 继续 paused。下一边界
只能是 S9 prediction freeze 后 Main reveal/evaluate，或 reveal 后明确授权 S1
remediation / 恢复 S8-C～F。
