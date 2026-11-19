
  I have written the full documentation report in Gedix/GEDIK_DEVELOPMENT_AND_EVALUATION_REPORT.md and updated Gedix/gedik_no_nn.py. I am now
  waiting for a final end-to-end Arena.cli verification match against wrongway_bot to finish.

  I have implemented the full development roadmap (G0 → G5, covering directions D1–D5) proposed in Gedix/Gedik_NoNN_De_Xuat_Phat_Trien_Project1.md
  using only the Python standard library, evaluated all variants in Arena/, and documented the complete methodology and empirical results in
  Gedix/GEDIK_DEVELOPMENT_AND_EVALUATION_REPORT.md.
  ──────
  Files Created & Updated in Gedix/

  • Production Single-File Agent: Gedix/gedik_no_nn.py — Standalone standard-library submission agent combining D1–D5 (defaults to EVAL_MODE =
  "G5_MLP", switchable to "G4_HEURISTIC").
  • Roadmap & Ablation Versions:
      • G0 (Frozen Baseline): Gedix/g0_baseline.py
      • G1 (D1 Threat-Aware Evaluation): Gedix/g1_threat_eval.py (variants G1a, G1b, G1c)
      • G2 (D2 Tactical & Defensive Wall Priors): Gedix/g2_tactical_priors.py (variants G2a, G2b)
      • G3 (D3 Phase-Adaptive PUCT & Progressive Widening): Gedix/g3_adaptive_puct.py (variants G3a, G3b)
      • G4 (D4 Zero-Flood Engineering + Combined D1–D4): Gedix/g4_combined.py
      • G5 (D5 Self-Trained 221-Parameter Residual Tactical MLP): Gedix/g5_custom_mlp.py
  • Training & Evaluation Pipelines:
      • Gedix/train_g5_mlp.py & Gedix/g5_mlp_metrics.json
      • Gedix/run_evaluation_pipeline.py & Gedix/evaluation_results.json
  • Full Documentation Report: Gedix/GEDIK_DEVELOPMENT_AND_EVALUATION_REPORT.md
  ──────
  Summary of the Three Required Documentation Sections
  (1) Key Idea Behind the Agent

  1. Bitboard Shortest-Path & Cut-Mask Threat Profiling (D1):
      • Uses 81-bit board flood-fill (waves) and shortest-path edge-cut masks (route returning (d, cuts)).
      • Because any single legal wall that increases a player's shortest path in one turn must intersect at least one edge of their current shortest
      path (cuts), route_threat_profile() computes exact single-wall route vulnerability (R_{self}) and offensive detour potential (R_{opp}) by
      scanning only indices(cuts & ~forbidden) (∼ 6--14 walls instead of ∼ 90).
  2. Preventive Defensive Wall Priors & Quota (D2):
      • Identifies the opponent's highest-impact threat wall(s) k^* (R_{self} ≥ 2) and builds a preventive conflict bitmask CONFLICT[k*] ^ BIT[k*].
      • Boosts non-self-blocking walls (Δd_{self} = 0) that physically prevent the opponent from placing k^*, rewards wall-chaining cohesion (near &
      BIT[k]), and reserves a top-3 defensive quota at the front of the expansion ordering.
  3. Phase-Adaptive PUCT & Progressive Widening (D3):
      • Scales exploration c(s) = (1.25 + 0.45 × phase)√N by remaining wall density phase = (w₀ + w₁)/20.
      • Widens active child selection W(N) = min(|kids|, 12 + ⌊3√N⌋) while guaranteeing all pawn moves, defensive quota walls, and high-prior
      tactical walls (prior >= 0.04) are never excluded.
  4. Two-Stage Zero-Flood Candidate Filtering (D4):
      • In priors(), any candidate wall k that misses both players' current shortest paths (not ((cuts_me | cuts_op) & BIT[k])) has Δd_{self} = 0,
      Δd_{opp} = 0, and cannot disconnect either player. Skipping waves() and distance() on those walls increases search throughput by +49.7% to +
      57.5% (2,291.2–2,410.6 simulations/turn vs. 1,530.3 for G0 at a 0.25s budget).
  5. Self-Trained Residual Tactical MLP Leaf Evaluator (D5):
      • A 221-parameter Residual Tactical MLP (12 → 16 ReLU → 1 Sigmoid with a 12-dim linear skip connection) trained on 12-dimensional strategic
      features (path lead, raw & effective wall differences, game phase, R_{opp}, R_{self}, total distance, optimistic/pessimistic margins, and 
      pure-race flag), without using any weights from upstream ag.bin.


  (2) Development Process & AI Variant Selection Pipeline
  • Variants Generated: 9 distinct variants across D1–D5 (G1a, G1b, G1c, G2a, G2b, G3a, G3b, G4, G5) compared against the frozen baseline G0 (10
  configurations total).
  • Evaluation & Selection Funnel:
      1. Tier 1 (Legality & Throughput Profiling): Tested on 10 tactical positions at a 0.25s budget. Confirmed 100% action legality and exposed
      that unfiltered setup priors (G2b) reduced simulations/turn by -18% (1,255.1 sims) whereas two-stage filtered priors (G2a/G4/G5) boosted
      simulations/turn to 2,131–2,410 sims.
      2. Tier 2 (Dev-Opening Screening vs. G0, 84 games): 12-game paired series (6 randomized 6-ply openings × swapped P0/P1 sides). Selected G1b
      (58.3% vs G0, balanced race/threat weights), G2a (+39.3% throughput + defensive quota), and G3a (58.3% vs G0, +13.9% throughput), while
      rejecting G1a (weak threat weight), G2b (slow unfiltered setup), and G3b (overly narrow window 5 + √N).
      3. Tier 3 (Holdout MLP Validation & Disjoint Eval-Opening Ablation, 160 games):
          • On 126 internal games (8,107 tactical states split strictly by game_id: 88 train / 18 val / 20 holdout test games), G5 achieved 82.09%
          holdout accuracy and 0.1184 Brier score (21.48 µs/eval in pure Python), outperforming G0 (70.56% accuracy, 0.2202 Brier) and G1b (70.94%
          accuracy, 0.2154 Brier).
          • In 16-game paired ablation series on 8 disjoint evaluation openings, both G4_Combined and G5_CustomMLP defeated G0_Baseline (10–6, 62.5%,
          5/8 as P0 and 5/8 as P1), and G5_CustomMLP defeated G1b_ThreatEval (10–6, 62.5%).
  (3) Evaluation Method & Unknowns

  • Total Games Played: 576 games across Stage 1 screening (84 games), Stage 2 roadmap ablations (160 games), Stage 3 external opponent benchmarks
  (312 games: 104 per agent across G5, G4, G0), and Stage 4 official 1.0s subprocess verification (20 games via Arena.agent_runner.StudentAgent).
  • Opponents Tested Against (7 Agents + G0):
      • G5_CustomMLP scored 88–0–16 (84.6%) overall across the 104-game external suite and 80–0–8 (90.9%) against the 6 non-Gedik local opponents:
          • vs. mcts_agent_v7 (Development): 16–0–0 (100.0%)
          • vs. hard_waller (Development): 15–0–1 (93.8%)
          • vs. minimax_agent_summary (Development): 14–0–2 (87.5%)
          • vs. wrongway_bot (Held-out): 5–0–3 (62.5%) (improving over G0's 4–0–4, 50.0%)
          • vs. quoridor_mcts_agent (Held-out): 16–0–0 (100.0%)
          • vs. minimax_tt_cached (Held-out): 14–0–2 (87.5%)
          • vs. gedik_qnn1 (External 23,233-param NN Reference): 8–0–8 (50.0%)
      • In Stage 4 official subprocess verification (StudentAgent, timeout=1.0s, TIME_BUDGET=0.78s), G5_CustomMLP went 10–0 (100.0%) and G4_Combined
      went 9–1 (90.0%) across 548 turns with 0 timeouts, 0 invalid actions, and 0 runtime errors (max turn time 0.8176s).
  • Explicitly Stated Unknowns:
      1. We do not know the upstream author's training dataset size, self-play schedule, or hyperparameters used to produce the original Rust ag.bin
      (QNN1) weights beyond public repository commit metadata.
      2. We do not know the identities or strategies of unseen student submissions in the upcoming class tournament.
      3. We do not know the exact CPU model/clock speed of the grading server, though our wall-clock deadline (TIME_BUDGET = 0.78s) dynamically
      adapts simulation counts to prevent timeouts under the 1.0s limit.