# S4-Plan05A Handoff｜B3 城市级残差记忆与误差持续性快速筛选

## Result
- Status: Complete（screening 已完成；不代表 S4 complete）
- Branch: `s4/wave-b-error-memory`
- Integration starting commit: `decfd7e0c68d14da31eabe5720ab706122c1040d`
- Decision: `error_memory_unsupported`
- `deep_dive_recommended`: `false`

## Scope
- Candidate: `M5A_B3_PLUS_RIDGE_ERROR_MEMORY` = frozen B3 + Ridge(alpha=1.0) residual correction.
- Features: last residual, recent-3 mean, all-history mean, and recent-3 residual slope.
- Resident primary; registered sensitivity; common origins 2011–2021.
- No covariates, city ID, tuning, GBDT, 2023 prediction, submission, or full robustness suite.

## Screening answers
1. **是否存在可利用的 city-level residual persistence？**
   - `error_memory_unsupported`；resident R1 M5A pooled `1330.678942566045` vs B3 `872.452651515143`，R2 `1330.678942566045` vs B3 `872.452651515143`。
2. **M5A 是否超过 B3？**
   - resident R1/R2 relative improvement: `-52.521623%` / `-52.521623%`。
   - registered R1/R2 relative improvement: `-2.254130%` / `-2.254130%`；两 scenario pooled 支持：`false`。
3. **per-origin 是否 broad-based？**
   - resident R1 `3/8/0`，R2 `3/8/0`（wins/losses/ties）。
   - registered R1 `3/8/0`，R2 `3/8/0`。
4. **latest 2021 是否支持？**
   - resident B3 latest `3705.427777777730`；M5A R1 `3636.481457573134`、R2 `3636.481457573134`。
   - registered B3 latest `1485.604861111120`；M5A R1 `1507.733668829354`、R2 `1507.733668829354`。
5. **boundary 是否无 self/future residual leakage？**
   - audit rows `14960`；all boundary PASS=`true`；self residual violations=`0`；`y-1/T-1` residual violations=`0`。
   - residual history max pseudo-target always `<= y-2`；each historical residual B3 fit history `<= z-2`。
6. **coefficient direction / deep dive？**
   - resident 四个 memory feature 的系数方向均为 `mixed`（跨 11 个 origins）；registered 的 all-history mean 与 recent-3 mean 为 `negative`，last residual 与 recent-3 slope 为 `mixed`。
   - deep dive=`false`。resident M5A does not provide a stable pooled, multi-origin, and latest improvement over B3

## Evidence
- `plan05a_summary.json`
- `screening_metrics.csv`
- `error_memory_boundary_audit.csv`
- `scripts/build_s4_plan05a_error_memory.py`

## Project state
- `s4_complete=false`; `s5_entered=false`。Wave A integration 与前序 contract 未修改。
