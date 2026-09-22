# Density candidate failure audit

This is a diagnostic of the frozen density formulas; no density formula was repaired.

## Per-origin evidence

| method | origin | rows | MSE | MAE | max absolute error |
|---|---:|---:|---:|---:|---:|
| density_hold | 2011 | 40 | 1184.542963 | 19.865570 | 127.880697 |
| density_hold | 2012 | 40 | 1518.889194 | 20.681571 | 153.189655 |
| density_hold | 2013 | 40 | 1334.284876 | 13.164732 | 174.090672 |
| density_hold | 2014 | 33 | 1032.818676 | 12.565820 | 125.283037 |
| density_hold | 2015 | 34 | 716.440346 | 12.544688 | 105.148600 |
| density_hold | 2016 | 40 | 3604.044054 | 22.434598 | 240.910152 |
| density_hold | 2017 | 40 | 656220.341808 | 284.023760 | 4020.982206 |
| density_hold | 2018 | 33 | 626465.201525 | 271.260078 | 3414.034388 |
| density_hold | 2019 | 34 | 1885.127698 | 23.569881 | 142.898975 |
| density_hold | 2020 | 40 | 1891.817847 | 26.609520 | 145.096089 |
| density_hold | 2021 | 33 | 1576.679709 | 21.016035 | 157.184483 |
| density_plus_slope5 | 2011 | 40 | 627.930779 | 12.916165 | 96.880697 |
| density_plus_slope5 | 2012 | 40 | 701.948056 | 15.130541 | 98.189655 |
| density_plus_slope5 | 2013 | 40 | 660.425302 | 10.850735 | 132.090672 |
| density_plus_slope5 | 2014 | 33 | 343.075288 | 9.391622 | 83.283037 |
| density_plus_slope5 | 2015 | 34 | 211.887610 | 8.461420 | 63.148600 |
| density_plus_slope5 | 2016 | 40 | 5541.821753 | 26.488237 | 301.910152 |
| density_plus_slope5 | 2017 | 40 | 656587.548012 | 280.461268 | 4020.982206 |
| density_plus_slope5 | 2018 | 33 | 626119.702938 | 266.425790 | 3414.034388 |
| density_plus_slope5 | 2019 | 34 | 1484.463982 | 17.648199 | 145.898975 |
| density_plus_slope5 | 2020 | 40 | 1550.865800 | 22.445416 | 138.096089 |
| density_plus_slope5 | 2021 | 33 | 1286.419884 | 17.855810 | 140.184483 |

## Largest observed rows

| method | origin | city | prediction | actual | absolute error |
|---|---:|---|---:|---:|---:|
| density_hold | 2017 | city8 | 4509.982206 | 489.000000 | 4020.982206 |
| density_plus_slope5 | 2017 | city8 | 4509.982206 | 489.000000 | 4020.982206 |
| density_plus_slope5 | 2018 | city26 | 4145.034388 | 731.000000 | 3414.034388 |
| density_hold | 2018 | city26 | 4145.034388 | 731.000000 | 3414.034388 |
| density_hold | 2018 | city34 | 3297.116564 | 724.000000 | 2573.116564 |
| density_plus_slope5 | 2018 | city34 | 3297.116564 | 724.000000 | 2573.116564 |
| density_hold | 2017 | city15 | 2985.287356 | 870.000000 | 2115.287356 |
| density_plus_slope5 | 2017 | city15 | 2985.287356 | 870.000000 | 2115.287356 |
| density_plus_slope5 | 2017 | city5 | 2823.701299 | 1065.000000 | 1758.701299 |
| density_hold | 2017 | city5 | 2820.701299 | 1065.000000 | 1755.701299 |
| density_plus_slope5 | 2018 | city37 | 1799.690987 | 451.000000 | 1348.690987 |
| density_hold | 2018 | city37 | 1799.690987 | 451.000000 | 1348.690987 |

## Mechanism interpretation

The candidate uses the latest legal resident/density area ratio through T-2 and an exact density observation at T-1 when the source cap permits it. The audit records the exact source cap and lookup availability per row. The 2017/2018 blow-up is therefore retained as observed candidate behavior; this Plan does not infer a unit conversion or regime correction without a separate source-level proof.

Selector interaction is recorded in `selector_choice_history.csv`; a density method being selected after a large error is evidence about prior-only learning, not permission to repair the formula.
