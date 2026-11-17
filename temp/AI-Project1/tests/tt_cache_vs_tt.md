# Search comparison: Agents/minimax_tt_cached.py vs Agents/minimax_tt.py

The same opening is played twice with swapped agent sides. Scores are not independent.

## Search probes

| Opening | Agent | Depth | Nodes | BFS calls | TT hits | TT entries |
|---:|---|---:|---:|---:|---:|---:|
| 1 | v2 | 3 | 778 | 11040 | 26 | 88 |
| 1 | v3_tt | 3 | 1274 | 9144 | 46 | 133 |
| 2 | v2 | 3 | 1419 | 18456 | 48 | 140 |
| 2 | v3_tt | 3 | 1604 | 12883 | 58 | 209 |
| 3 | v2 | 3 | 751 | 10226 | 22 | 84 |
| 3 | v3_tt | 3 | 969 | 8842 | 28 | 130 |

## Arena results

TT W/D/L: **22/0/18**
TT score: **55.0%**
TT mean latency: 0.655s

A shorter search or larger TT hit count does not establish stronger playing strength.
