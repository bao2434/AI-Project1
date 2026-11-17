# Search V3 (TT) vs V2 evaluation

The same opening is played twice with swapped agent sides. Scores are not independent.

## Search probes

| Opening | Agent | Depth | Nodes | BFS calls | TT hits | TT entries |
|---:|---|---:|---:|---:|---:|---:|
| 1 | v2 | 3 | 711 | 9186 | 0 | 0 |
| 1 | v3_tt | 2 | 662 | 8688 | 21 | 66 |
| 2 | v2 | 3 | 738 | 10661 | 0 | 0 |
| 2 | v3_tt | 3 | 737 | 10649 | 23 | 88 |
| 3 | v2 | 3 | 799 | 10928 | 0 | 0 |
| 3 | v3_tt | 3 | 795 | 10710 | 24 | 84 |

## Arena results

TT W/D/L: **29/0/31**
TT score: **48.3%**
TT mean latency: 0.720s

A shorter search or larger TT hit count does not establish stronger playing strength.
