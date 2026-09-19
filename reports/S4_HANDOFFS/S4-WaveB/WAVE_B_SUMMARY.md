# S4 Wave B Summary｜Plan05A–05C 整合与负结果收口

## Integration record

| source branch | exact source commit | merge-base with Wave A base | scope |
|---|---|---|---|
| s4/wave-b-error-memory | cbabf585357b0cae069efbf4aa4eed543ecae202 | decfd7e0c68d14da31eabe5720ab706122c1040d | Plan05A runner + evidence |
| s4/wave-b-cross-city-relative | e14bbfb30122a8d397542871c4af017bdcb96193 | decfd7e0c68d14da31eabe5720ab706122c1040d | Plan05B runner + evidence |
| s4/wave-b-multisource-joint | 0b39868aa6027a26ebcbb13f7ebe1bfcd4394948 | decfd7e0c68d14da31eabe5720ab706122c1040d | Plan05C runner + evidence |

Integration branch: s4/wave-b-integration.

三条分支均从同一 Wave A 整合基点派生，且只新增各自 Plan05A/B/C 的 runner、handoff 和 machine artifacts。未发现冲突或 deviation；没有修改 CURRENT_LEARNING_CHECKPOINT.md、S2 validation contract、raw/canonical data 或其他 worker 文件。原始 Wave B source branches 保留。

## Wave B screening matrix

共同验证边界为 2011–2021 origins、population target history T-2、resident primary、registered sensitivity、pooled city-year MSE primary、latest 2021 secondary diagnostic。数值为 candidate 的 R1 / R2 pooled MSE；B3 resident / registered pooled MSE 分别为 872.452652 / 225.421212。

| Plan | hypothesis / candidate | resident pooled MSE R1 / R2 | registered pooled MSE R1 / R2 | origin wins/losses/ties | screening | deep dive |
|---|---|---:|---:|---|---|---|
| 05A | B3 error-memory + Ridge residual | 1330.678943 / 1330.678943 | 230.502499 / 230.502499 | resident 3/8/0; registered 3/8/0 | error_memory_unsupported | no |
| 05B | cross-city relative percentile + Ridge residual | 917.617207 / 917.617207 | 230.549370 / 230.549370 | resident 4/7/0; registered 4/7/0 | unsupported | no |
| 05C | four-source compact RidgeCV residual | 903.388981 / 959.539982 | 226.838540 / 227.216432 | resident R1 4/7/0, R2 3/8/0; registered 4/7/0 in both | multisource_joint_unsupported | no |

Latest 2021 只是辅助诊断，不替代 pooled aggregate。05A 的 resident latest 为 3636.481458，05B 的 resident R1 latest 为 3671.890459，05C 的 resident latest 为 3341.066612 / 3711.371037；这些局部变化都没有形成稳定的 aggregate gain。

## Mechanism interpretation

- error memory：历史 B3 残差的城市内持续性没有转化为可迁移的稳定修正，resident aggregate 明显恶化。
- cross-city relative：横截面 percentile 丢失绝对尺度，虽然特征边界与 R1/R2 等价性通过，仍未超过 B3。
- multisource joint：urbanization、density、wage、employment 的紧凑组合在 R1/R2 下均未超过 B3；RidgeCV 多数 outer fit 选择 alpha=100，且多个系数跨 origin/scenario 翻号，支持“增量信号弱且不稳定”的解释。

## Unified Wave B conclusion

Wave B 三条机制分别得到：

- error_memory_unsupported
- cross_city_relative_unsupported
- multisource_joint_unsupported

因此 Wave B 没有产生值得 deep dive 的候选。它补齐了城市级误差记忆、跨城市相对位置和多源联合正则化三类机制的负结果证据；这些证据与 Wave A 的 density、wage、employment 负结果共同支持在当前 S4 边界停止继续扩展实验。

## Evidence and scope

- Plan05A/B/C handoff、summary、screening metrics、boundary audit 与 runner 均已进入当前整合树。
- 本次只做 branch integration、artifact integrity review 和 summary/report construction；没有重跑历史实验。
- 没有生成 2023 forecast 或 submission，没有进入 S5。
- B3 仍是进入 Main review 的 strongest fixed model；S4 stage completion 仍由 Main authority 判定。
