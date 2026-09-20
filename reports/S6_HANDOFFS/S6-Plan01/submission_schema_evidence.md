# Submission Schema Evidence

Status is intentionally split between schema evidence and target semantics.

| Field | Value | Status | Evidence |
|---|---|---|---|
| official prediction year | 2023 | CONFIRMED | official PDF page 1; current Main checkpoint |
| official target wording | 各个城市 2023 年的总人口数 | CONFIRMED | official PDF page 1 |
| target mapping | resident versus registered not proven | UNKNOWN | PDF says 总人口; source contains both columns |
| city universe | city1..city40 | CONFIRMED | Main checkpoint's direct official-sample check; canonical source coverage |
| row count | 40 | CONFIRMED | Main checkpoint's direct official-sample check |
| ordering | sample is city1..city40; formal ordering rule unstated | INFERRED_NOT_SAFE_TO_FREEZE | Main checkpoint and sample evidence |
| file format | CSV | CONFIRMED | official PDF page 3 |
| headers | city_id, year, pred | CONFIRMED | official PDF page 3 |
| target unit | local source contract is 万人; official PDF does not state unit | INFERRED_NOT_SAFE_TO_FREEZE | PDF plus S1 contract |
| official submission sample | confirmed in Main evidence, not present in current raw tree | CONFIRMED | Main checkpoint and recorded MD5 |

The official target wording does not resolve the local resident/registered mapping. Therefore this Plan freezes the resident-primary F4 reproduction contract only and does not generate a formal submission.
