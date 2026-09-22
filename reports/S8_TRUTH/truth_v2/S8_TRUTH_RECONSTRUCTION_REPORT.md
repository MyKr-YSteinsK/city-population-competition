# S8 Truth Final Four Resolution Report (candidate_truth_2023_v2)

Status: S8_TRUTH_V2_FROZEN_WAITING_FOR_S9

This report covers S8-Plan03 final-four source resolution only. No model was
trained, scored, ranked, or inspected, and no S9 artifact was read.

## Final-four resolution

- city16 韶关: TIER2_OFFICIAL_REPOST, raw 286.18, rendered 286
- city36 枣庄: TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK, raw 382.97, rendered 383
- city38 肇庆: TIER2_OFFICIAL_REPOST, raw 412.84, rendered 413
- city39 东营: TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK, raw 220.9, rendered 221

Resolution status counts: TIER1_DIRECT_OFFICIAL=0, TIER2_OFFICIAL_REPOST=2, TRACEABLE_OFFICIAL_WITH_INDEPENDENT_CROSSCHECK=2, TIER3_FALLBACK_ONLY=0, CONFLICTING_VALUES=0, UNRESOLVED=0.
All four raw values and rendered evaluator values are unchanged. The four
TIER3 truth rows are replaced by TIER2 provenance records; historical TIER3
records remain in the source ledger for traceability.

## Value delta

- Raw rows changed: 0
- Evaluator rows changed: 0
- New truth version required: NO
- Material truth-value conflicts: 0

## Full-freeze decision

All 40 cities remain resolved, all final-four rows meet the Plan03 eligible
source statuses, rendered truth is unchanged, hashes and validator checks are
complete, and isolation flags remain false. The decision is therefore
`S8_TRUTH_V2_FROZEN_WAITING_FOR_S9`.

city40's registered=65 anomaly and density conflict remain explicit context
from Plan02; they do not change city identity or the evaluator value here.

Isolation: S9 branch read NO; S9 predictions read NO; model scoring NO;
test.xls accessed NO.
