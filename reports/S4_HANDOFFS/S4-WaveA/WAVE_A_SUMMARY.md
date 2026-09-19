# S4 Wave A Summary｜单 source screening 整合

## Integration scope

本目录归档三个从同一 frozen base 独立完成的 Wave A screening branch。本轮只整合已完成 evidence，不新增实验、不改变单 Plan 结论、不进入 S5。

- Frozen base / integration starting SHA：`b8c302abe7ebf1710869aaa55a00d4b926d2bcac`
- Integration branch：`s4/wave-a-integration`
- `origin/main` 在整合前仍为 frozen base；三个原始远端 branch 保留为审计引用。

| Plan | branch | source HEAD | candidate | screening | deep dive |
|---|---|---|---|---|---|
| S4-Plan04A | `s4/wave-a-density` | `97b182dc4867229af0723110f4f4fbb79553781d` | `M4A_B3_PLUS_RIDGE_DENSITY` | unsupported | no |
| S4-Plan04B | `s4/wave-a-wage` | `558ca4b1b1b9fb92be08fc002fe4fb401a4121e7` | `M4B_B3_PLUS_RIDGE_WAGE` | unsupported | no |
| S4-Plan04C | `s4/wave-a-employment` | `dc65063ad76d86c4216aabd735d656218f7e1b19` | `M4C_B3_PLUS_RIDGE_EMPLOYMENT` | unsupported | no |

## Frozen comparison

Primary metric is pooled city-year MSE over resident, with registered as mandatory sensitivity. The fixed B3 reference is resident `872.452652` and registered `225.421212`; all plans use common origins `2011–2021` and population history `T-2`.

| source / candidate | resident B3 | resident candidate R1 | resident candidate R2 | registered B3 | registered candidate R1 | registered candidate R2 |
|---|---:|---:|---:|---:|---:|---:|
| density / M4A | 872.452652 | 906.652625 (-3.920%) | 917.775198 (-5.195%) | 225.421212 | 229.902996 (-1.988%) | 230.299976 (-2.164%) |
| wage / M4B | 872.452652 | 943.919472 (-8.191%) | equivalent cap; no rerun | 225.421212 | 225.850578 (-0.190%) | equivalent cap; no rerun |
| employment / M4C | 872.452652 | 908.508412 (-4.133%) | 912.418465 (-4.581%) | 225.421212 | 224.904724 (+0.229%) | 225.389926 (+0.014%) |

括号为 candidate 相对 B3 的 pooled MSE improvement percentage；正值才代表 candidate 更好。employment 在 registered 上有很小的局部增益，但 resident primary 在两个 scenario 均变差，不能改变 Wave A 结论。

## R1/R2 and source-specific risks

- **Density**：R1/R2 使用 S2 source-specific lag `1/2`；13 个 regime-report 城市保留显式 regime boundary，跨 regime 的 dynamics 被抑制并作为 missing 进入 training-fold imputation。风险是 observed-cap replay 与 regime/denominator uncertainty，不是可证明的 publication timing。
- **Wage**：工资 observed lag 为 `2`，R1/R2 required cap 完全等价，M4B 按计划只 fit 一次；`city31/2011` invalidated value 保持 missing，未回流。风险是 publication timing 未知及 source coverage/missingness，不能把等价 replay 当作真实发布时间事实。
- **Employment**：R1/R2 使用 `T-1/T-2`，source status 为 `source_plus_remap_quarantine_and_derived`，包含 derived rows 与 quarantine/provenance 边界；zero-reintroduction 与 provenance tracking 通过，但 source provenance 是主要解释风险。

三个 Plan 均复用 B3 与 S2 boundary；未使用其他 source、city ID、tuning、GBDT、2023 forecast 或 submission。

## Unified decision

三个单 source screening 均未提供 resident material gain：density、wage、employment 的 resident pooled candidate 均高于 B3，且至少一个 scenario 的 aggregate/origin evidence 不支持稳定增益。因此：

- `screening decision = unsupported` for all three Wave A plans;
- `deep_dive_recommended = false` for all three;
- `s4_complete = false`；`s5_entered = false`；
- 本整合只形成统一 evidence base，后续是否测试新的 mechanism 由 Main/Lab 决定，不在本 Plan 自动扩展。

## Evidence index

- `S4-Plan04A/`：density handoff、summary、screening metrics、boundary audit、runner。
- `S4-Plan04B/`：wage handoff、summary、screening metrics、boundary audit、runner。
- `S4-Plan04C/`：employment handoff、summary、screening metrics、provenance boundary audit、runner。
- 本文件：跨 source 的最小整合结论；各 Plan 原始 evidence 仍是逐 source 事实源。
