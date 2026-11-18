# Gedik QNN1: a one-second Quoridor agent

**Result:** The [standalone Python agent](submissions/gedik_qnn1.py) won **40 of 40 games** against five local opponents, including the previous tournament champion. It won 20 games moving first and 20 moving second. It lost one turn to a timeout. The strongest public result for the model itself comes from a different experiment: the original Rust author reports **473–104 (+263 Elo)** for neural evaluation against linear evaluation within Gedik.

## Source and version

| Item | Source |
|---|---|
| Original repository | [msametkose-ctrl/gedik on GitHub](https://github.com/msametkose-ctrl/gedik) — Rust engine by Samet Köse |
| Version used for the Python port | [Commit `bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab`](https://github.com/msametkose-ctrl/gedik/tree/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab). This revision contains the compact QNN1 model; newer revisions use different weights. |
| Pretrained weights | [`ag.bin` at the pinned revision](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/ag.bin), 92,948 bytes. SHA-256: `ec9085f955fb5216205e4a8ac2571e3e7d3fc69e82fb93134838dd077b28b0e7` |
| Original implementation | [Neural evaluator](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/src/nn.rs) · [Features and move priorities](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/src/heuristics.rs) · [Tree search](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/src/mcts.rs) |
| License | [MIT license](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/LICENSE); the Python file includes the copyright and permission notice. |

[GitHub repository metadata](https://api.github.com/repos/msametkose-ctrl/gedik), checked 18 November 2026: **0 stars, 0 forks, 1 open issue or pull request**. These are repository activity counts, not measures of playing strength. The Python port is a local project file; the GitHub links above point to its upstream source.

## Model and strategy

QNN1 is a **position evaluator**: it estimates the probability that the player to move will win. A single-threaded Monte Carlo tree search uses those estimates to choose a pawn move or wall placement. The weights were trained upstream using self-play data and were kept fixed for the local benchmark.

| Property | Python agent |
|---|---|
| Network | 329 inputs → 64 hidden units → 32 hidden units → 1 output; ReLU hidden layers and a sigmoid output |
| Size | 23,233 learned weights and biases |
| Board inputs | 313 sparse features: pawn squares, horizontal and vertical walls, each player's remaining walls, and the side to move |
| Strategic inputs | 16 numerical features describing the race, usable walls, game phase, and how much a single legal wall can lengthen either player's path |
| Runtime | One `.py` file, Python standard library only. The agent decodes the embedded weights at startup and exposes `choose_action(observation)`. |

A pawn four steps from its goal is not necessarily ahead of a pawn five steps away. If one wall can turn the four-step route into an eight-step detour while the five-step route is safe, raw distance favors the wrong pawn. Gedik gives the evaluator both the current distance and the largest detour one legal wall can cause.

The agent first takes an immediate winning move if one exists and saves a legal fallback. Its search then balances moves that have scored well with promising moves it has explored less often (PUCT). Hand-written priorities guide exploration; the network scores the positions reached. A nominal **0.80-second search budget** leaves 0.20 seconds of the arena's one-second limit for communication and timing variation.

## Published results for the original Rust engine

The [upstream benchmark](https://github.com/msametkose-ctrl/gedik/blob/bda3a6ad0ff12364714b5a2e95c4e91b15c5ccab/README.md#how-strong-and-how-do-we-know) compares versions of Gedik using randomized openings and both seats. The numbers below are the **author's reported results**. Each Elo figure is a difference against the comparator in its own row; the rows do not establish an absolute rating for the Python agent.

| Comparison | Reported result | Reported Elo difference |
|---|---:|---:|
| Evaluate leaf positions instead of playing random rollouts | 93.8% | +450 |
| PUCT with move priorities vs. plain UCT | 80.0% | +241 |
| Fitted linear weights vs. hand-written weights | 149–32 | +267 |
| Neural evaluator vs. linear evaluator | **473–104** | **+263** |
| 200,000 vs. 20,000 search iterations | 132–62 (68.0%) | +131 |
| 2,000,000 vs. 200,000 search iterations | 19–17 (52.8%) | +19; inconclusive |

## Local results under the one-second rule

The frozen Python agent played standard two-player Quoridor on a **9×9 board with ten walls per player**. The arena allowed one second per action including communication, five seconds for startup, and 500 individual turns before a draw. Each opponent played four seed-matched pairs from the official starting position: the agents swapped seats, and player 0 always moved first. Up to four games ran concurrently. The last two opponents were held out until the first 24 games were complete.

| Opponent | Split | Wins as first | Wins as second | Game points | Slowest Gedik turn | Distinct games¹ |
|---|---|---:|---:|---:|---:|---:|
| `greedy` | Development | 4/4 | 4/4 | 24/24 | 0.917644 s | 3/8 |
| `hard_waller` | Development | 4/4 | 4/4 | 24/24 | 0.829956 s | 8/8 |
| `quoridor_search_v2` | Development | 4/4 | 4/4 | 24/24 | 0.880891 s | 5/8 |
| `wrongway_bot` (previous champion) | Held out | 4/4 | 4/4 | 24/24 | 0.859678 s | 6/8 |
| `quoridor_hybrid_bot` | Held out | 4/4 | 4/4 | 24/24 | 1.007686 s | 7/8 |
| **Total** | | **20/20** | **20/20** | **120/120** | **1.007686 s** | **29/40** |

¹ Distinct complete action and status sequences within each opponent/seat pairing, ignoring elapsed time. Class scoring awards three points for a win and one for a draw. Gedik finished with **40 wins, 0 draws, 0 losses** and beat **5/5 opponents**. No opponent recorded a timeout, invalid action, or runtime error.

**Reliability:** Across **1,267 Gedik turns**, there was **1 timeout (0.079%)**, **0 invalid actions**, **0 runtime errors**, and **0 voluntary passes**. The timeout occurred against `quoridor_hybrid_bot`, with Gedik moving first (seed `24889`). The arena skipped that turn; Gedik still won the game. The zero-timeout target was therefore missed despite the perfect game score.

**Correctness:** All **six local automated tests passed** in the implementation run. They checked the model checksum and neural arithmetic against a separate dense calculation, move and wall legality against the arena, wall-threat features against exhaustive wall placement, straight and diagonal jumps, deadline fallback and immediate wins, and cold startup through the actual timed worker.

## Interpretation and reproduction

The most useful evidence for competition is the **16–0 record against the two held-out agents**, eight wins from each seat. It shows that the frozen model beat both strong local opponents without matchup-specific changes. All games started from the same board and 11 repeated a full sequence, so the result does not establish a 100% win rate against unseen entrants. The next useful test is a new opponent pool with varied openings; a shorter search budget can be compared against this baseline to see whether it removes the occasional timeout without costing wins.

Local evidence: [40 per-game CSV records](benchmarks/results/gedik_40.csv) · [Generated benchmark summary](benchmarks/results/gedik_40.md) · [Benchmark runner and seeds](benchmarks/gedik_40.py) · [Correctness tests](tests/test_gedik_qnn1.py).

From the project root on Windows:

```powershell
py -3 -B -m unittest discover -s tests -p test_gedik_qnn1.py
py -3 -B benchmarks/gedik_40.py
```

The second command runs a fresh tournament and replaces the generated CSV and benchmark summary. Wall-clock search can produce different moves across runs and machines.
