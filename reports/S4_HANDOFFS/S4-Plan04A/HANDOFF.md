# S4-Plan04A Handoff — 人口密度近期信号与 B3 残差修正快速筛选

## Result
- Status: Complete（screening 已完成；不代表 S4 complete）
- Branch: `s4/wave-a-density`
- Frozen base: `b8c302abe7ebf1710869aaa55a00d4b926d2bcac`
- Decision: `unsupported`
- `deep_dive_recommended`: `false`

## Frozen scope
- Candidate: `M4A_B3_PLUS_RIDGE_DENSITY` = frozen B3 + Ridge(alpha=1.0) residual correction.
- Features: `density_latest`, annualized `density_recent_relative_change`, and normalized `density_recent3_relative_slope`.
- Resident / registered are evaluated separately; only common origins 2011–2021 are scored.
- No tuning, GBDT, city ID, other source, 2023 prediction, submission, or full robustness suite was run.

## Screening answers
1. **M4A vs B3**
   - resident R1 pooled MSE: `906.652624719085` vs B3 `872.452651515143`; latest 2021 `3183.888728214740`.
   - resident R2 pooled MSE: `917.775198142649` vs B3 `872.452651515143`; latest 2021 `3791.571296366275`.
2. **Registered sensitivity**
   - R1 candidate pooled MSE `229.902996464799` vs B3 `225.421212121213`.
   - R2 candidate pooled MSE `230.299975818413` vs B3 `225.421212121213`.
   - Supports candidate in both scenarios: `false`.
3. **Per-origin wins / losses vs B3**
   - resident R1: `4` wins / `7` losses / `0` ties.
   - resident R2: `3` wins / `8` losses / `0` ties.
   - registered R1: `4` wins / `7` losses / `0` ties.
   - registered R2: `3` wins / `8` losses / `0` ties.
4. **R1 → R2**
   - resident candidate pooled delta R2−R1: `11.122573423564`; latest delta: `607.682568151535`.
   - resident relative improvement vs B3: R1 `-3.919980%`, R2 `-5.194843%`.
5. **Regime suppression**
   - audit feature rows: `14960`; rows with at least one suppressed dynamic: `330`; suppressed dynamic feature cells: `594`.
6. **Quarantine guard**
   - candidate-hit count: `0`; reintroduction count: `0`; guard: `PASS`.
7. **Deep dive**
   - `false` — resident pooled MSE does not show a sufficiently stable gain over B3 under the frozen R1/R2 screen.

## Evidence
- `plan04a_summary.json`
- `screening_metrics.csv`
- `density_boundary_audit.csv`
- `scripts/build_s4_plan04a_density_residual.py`

## Project state
- `s4_complete=false`; `s5_entered=false`.
- Shared checkpoint, project context, file map, S2 contract, and prior S3/S4 handoffs were not modified.
