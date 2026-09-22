# city40 registered-field anomaly audit

Status: `ANOMALY_CAUSE_UNRESOLVED`

## Raw-to-normalized propagation

- Legal raw workbook: `data/raw/城市人口分析与预测/数据集/人口规模.xlsx`
- Sheet / row: `Sheet1` / `704`
- Raw cells: `{"A704": "city40", "B704": 2021, "C704": 321, "D704": 65}`
- Raw workbook SHA256: `9775cf2b5cfb5e930a4c721816e72dbec614ed221d7c6778998d28459e373cb0`
- Normalized file: `data/normalized/人口规模.csv`
- Normalized file SHA256: `7f12e54c5c8d72a20182a117ad26564e6f1892d6f4b9add0dc80135aabb18797`
- Normalized row: `line 704: city40,2021,321.0,65.0`

The literal `65` is present in the legal source workbook and is propagated
unchanged into the normalized CSV. This audit found no transformation-layer
unit conversion, column shift, interpolation, or replacement that created the
value. `test.xls` was not opened, parsed, hashed, or used.

## Independent identity evidence

- Resident trajectory: city40 is approximately 318–321万人 in 2017–2021,
  matching Zhenjiang's official 2020 census anchor of 321.0418万人.
- Registered trajectory excluding the anomalous row: 2017–2020 is 272, 272,
  271, 270万人, matching the official Zhenjiang anchor of 269.25万人.
- Urbanization: competition 2021 is 80%, versus the Zhenjiang 2020 anchor of
  79.5%.
- Density: competition source values around 1600 are inconsistent with the
  Zhenjiang anchor around 835 and are retained as a separate source conflict.
- Alternate-city falsification found no other Jiangsu city jointly explaining
  resident, non-anomalous registered, and urbanization trajectories.

## Resolution

`city40 -> 镇江` is upgraded from `PROVISIONAL` to `HIGH_CONFIDENCE` using
independent evidence. The registered value 65 and density conflict remain
raw, explicit, and unresolved. No competition source value is corrected.
