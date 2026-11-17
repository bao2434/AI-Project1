# Quoridor Agents Architecture and Algorithm Documentation

This document provides a comprehensive analysis and algorithmic breakdown of all agents residing in `AI-Project1/Agents/`.

---

## Table of Contents
1. [Overview & Quoridor Rules Summary](#1-overview--quoridor-rules-summary)
2. [Comparison Matrix](#2-comparison-matrix)
3. [Agent 1: `mcts_agent-v7.py` (Greedy Shortest-Path / Heuristic Wall Agent)](#3-agent-1-mcts_agent-v7py)
4. [Agent 2: `minimax_agent_summary.py` (Iterative Deepening Minimax with Alpha-Beta Pruning)](#4-agent-2-minimax_agent_summarypy)
5. [Agent 3: `quoridor_mcts_agent.py` (Monte Carlo Tree Search with Domain Heuristics)](#5-agent-3-quoridor_mcts_agentpy)
6. [Agent 4: `hard_waller.py` (Minimax with Robustness & Anti-Oscillation Port)](#6-agent-4-hard_wallerpy)
7. [Agent 5: `wrongway_bot.py` (Iterative Deepening Negamax with 6-Feature Evaluation)](#7-agent-5-wrongway_botpy)
8. [Agent 6: `value_agent.py` (ValueCNN Minimax Agent using models/value_model.pt)](#8-agent-6-value_agentpy)
9. [Common Utilities & Best Practices](#9-common-utilities--best-practices)

---

## 1. Overview & Quoridor Rules Summary

All agents in this repository adhere to the **Wrongway Arena** standard protocol:
- **Board**: $9 \times 9$ grid ($0 \le r, c \le 8$).
- **Goal**: Player reaches their designated goal row ($0$ or $8$).
- **Walls**: Horizontal (`'H'`) or Vertical (`'V'`), each spanning 2 grid cells. A placed wall permanently blocks passage between affected cell pairs and must never completely cut off either player from their goal.
- **Pawn Movement**: Orthogonal steps by 1 cell. Straight jumps over adjacent opponents; diagonal jumps when the straight jump is blocked by a wall or board boundary.
- **Entry Point**: Each agent exposes `choose_action(observation) -> dict` returning either:
  - `{"type": "move", "to": [row, col]}`
  - `{"type": "wall", "orientation": "H"|"V", "row": r, "col": c}`
  - `{"type": "pass"}`

---

## 2. Comparison Matrix

| Agent File | Core Algorithm | Search Depth / Budget | Evaluation / Rollout Strategy | Wall Generation / Pruning | Anti-Oscillation / Path Robustness |
|---|---|---|---|---|---|
| **`mcts_agent-v7.py`** | Greedy Heuristic (1-ply lookahead) | Instantaneous ($0$ plies lookahead) | Difference of BFS path increases $(\Delta d_{\text{enemy}} - \Delta d_{\text{own}})$ | Evaluates all legal walls only when enemy is within $d_{\text{own}} + 1$ | None (picks lowest distance from precomputed distances) |
| **`minimax_agent_summary.py`** | Iterative Deepening Minimax + $\alpha$-$\beta$ | Iterative up to depth 50 (time limit bounded) | $10 \cdot (d_{\text{opp}} - d_{\text{me}}) + 2 \cdot (w_{\text{me}} - w_{\text{opp}}) + 0.5 \cdot \text{prog}$ | Move ordering via child heuristic; limits wall branching to top $18$ | `REPEAT_PENALTY * count` on visited positions in last 10 turns |
| **`quoridor_mcts_agent.py`** | MCTS (UCT Selection, Backpropagation) | Time limit (0.90s, up to 100k simulations) | 70% forward BFS rollout, 30% backward/random moves; win/loss backpropagation | Intersects shortest-path edges + nearby cells; ranked by net distance gain | Implicitly handled by MCTS tree visit statistics |
| **`hard_waller.py`** | Alpha-Beta Minimax with Root Scoring | Fixed depth 3 (0.70s limit) | Relative distance ($25\times$), own distance, wall difference, path robustness penalty | Evaluates candidate walls along opponent's BFS path (depth 6) | `worst_wall_delay` path robustness penalty + exponential recent position penalty |
| **`wrongway_bot.py`** | Iterative Deepening Negamax + $\alpha$-$\beta$ | Depth 1 to 3+ (0.85s limit) | 6-feature linear model (weights from Wrongway App duel9) | Downhill BFS along opponent's shortest path (`botKandidaten`), top 4–8 walls | Dynamic backtracking penalty (`BACKTRACK_PENALTY`) + forward drive bonus |
| **`value_agent.py`** | Minimax + $\alpha$-$\beta$ Pruning | Iterative up to depth 3+ (0.75s limit) | Deep Convolutional Value Network (`models/value_model.pt`) evaluating 8-plane 9x9 board state | Shortest-path obstruction filter + net gain heuristic ranking (top 2-4 walls) | Anti-oscillation history penalty (`REPEAT_PENALTY = 8.0`) |

---

## 3. Agent 1: `mcts_agent-v7.py`

### 3.1 Overview
Despite the file name, `mcts_agent-v7.py` is a **lightweight, 1-ply greedy heuristic agent**. It uses fast Breadth-First Search (BFS) distance calculations rather than Monte Carlo Tree Search.

### 3.2 Key Mechanisms
1. **Immediate Win Check**: Checks if any legal pawn move reaches `you["goal_row"]`.
2. **Shortest Path Calculation (`distance_to_goal`)**: Standard BFS returning the shortest path length ignoring the opponent's pawn as a solid wall.
3. **Pawn Move Choice**: Selects the move that minimizes distance using `observation["distances"]`.
4. **Conditional Wall Evaluation**:
   - Walls are only considered if the opponent is dangerously close:
     $$\text{enemy\_distance} \le \text{own\_distance} + 1$$
   - Each legal wall is scored by:
     $$\text{gain} = (d_{\text{enemy}}^{\text{after}} - d_{\text{enemy}}^{\text{before}}) - (d_{\text{own}}^{\text{after}} - d_{\text{own}}^{\text{before}})$$
   - Must strictly ensure both paths remain open ($d < \text{size}^2$).
5. **Action Decision**: If $\max(\text{gain}) > 0$, places the best wall; otherwise, advances pawn.

### 3.3 Strengths & Weaknesses
- **Strengths**: Extremely fast ($< 5\text{ms}$ per turn); no risk of timing out; excellent baseline.
- **Weaknesses**: No multi-turn lookahead; vulnerable to opponent traps and wall counter-play.

---

## 4. Agent 2: `minimax_agent_summary.py`

### 4.1 Overview
A complete, structured implementation of **Iterative Deepening Minimax with Alpha-Beta Pruning** and comprehensive search logging/statistics.

### 4.2 State Representation
Uses an immutable/compact Python class `State` with `__slots__`:
- `pos`: `((my_r, my_c), (opp_r, opp_c))`
- `goals`: `(my_goal_row, opp_goal_row)`
- `walls_rem`: `(my_walls, opp_walls)`
- `h_walls`, `v_walls`: Frozensets of `(r, c)`
- `blocked`: Frozenset of blocked directed edges $((u, v), (v, u))$.

### 4.3 Evaluation Function
$$\text{Eval}(s) = 10 \cdot (d_{\text{opp}} - d_{\text{me}}) + 2 \cdot (w_{\text{me}} - w_{\text{opp}}) + 0.5 \cdot (\text{prog}_{\text{me}} - \text{prog}_{\text{opp}})$$
- **Terminal States**: Immediate win receives $+10000 + 10 \cdot \text{depth}$ (incentivizing fast wins); terminal losses receive $-10000 - 10 \cdot \text{depth}$ (incentivizing delayed losses).
- **Anti-Oscillation**: At the root, subtracts $\text{REPEAT\_PENALTY} \times \text{recent\_visit\_count}$ to break 2-cycle or 3-cycle pawn shuffling.

### 4.4 Move Ordering & Pruning
- Legal pawn moves (including jumps) and legal wall moves are scored by 1-ply static evaluation.
- Walls are filtered to the top `MAX_WALL_BRANCHES = 18` candidates to keep the branching factor manageable for Alpha-Beta cutoffs.
- Alpha-Beta cutoffs are monitored and written out to `minimax_search_summary.txt`.

---

## 5. Agent 3: `quoridor_mcts_agent.py`

### 5.1 Overview
An adaptation of Monte Carlo Tree Search (MCTS) with Upper Confidence Bounds applied to Trees (UCT), specifically tailored for Quoridor.

### 5.2 Four MCTS Phases
1. **Selection**:
   - Traverses existing tree nodes using the UCT formula:
     $$\text{UCT}(v) = \frac{W(v)}{N(v)} + C \cdot \sqrt{\frac{\ln N(\text{parent})}{N(v)}}, \quad C = \sqrt{2}$$
2. **Expansion**:
   - Selects an unexpanded action from `generate_expansion_actions`.
   - Generates all pawn moves plus **probable wall actions** (walls intersecting opponent's BFS shortest paths or adjacent to the opponent pawn).
3. **Simulation (Rollout)**:
   - Guided rollout with $70\%$ probability of taking a shortest-path-forward move (`ROLLOUT_FORWARD_PROB = 0.70`).
   - If not moving forward, considers placing probable walls or moving away/laterally.
   - Bounded by `MAX_ROLLOUT_PLIES = 250` and the turn deadline.
4. **Backpropagation**:
   - Propagates visit count $+1$ and win count $+1.0$ if the node's acting player won the rollout.

### 5.3 Safety & Time Management
- Uses a deadline check `TIME_LIMIT = 0.90s` inside rollout and simulation loops.
- Fallback mechanism to BFS distance minimization if search terminates prematurely.

---

## 6. Agent 4: `hard_waller.py`

### 6.1 Overview
Ported from `aiHard` and `aiDuelHard` (from `ai.js`), designed for high-level competitive play with a focus on **path robustness** and **anti-cycling**.

### 6.2 Key Algorithmic Features
1. **Endgame Sprint**: If out of walls ($w_{\text{me}} \le 0$) or significantly ahead ($d_{\text{me}} \le 3$ and $d_{\text{opp}} > d_{\text{me}} + 1$), the agent bypasses minimax search entirely and rushes straight toward the goal along the BFS path.
2. **Wall Generation via BFS Lookahead**: Inspects the opponent's BFS path up to `WALL_CANDIDATE_DEPTH = 6` steps ahead and evaluates adjacent wall slots.
3. **Alpha-Beta Search**: Explores up to depth 3 with adaptive branching:
   - Root max walls: 10
   - Inner depth $\ge 2$: 6 walls
   - Inner depth $< 2$: 4 walls
4. **Path-Robustness Penalty (`worst_wall_delay`)**:
   - Before committing to a move or wall at root, computes how severely a single opponent wall could delay the agent's path:
     $$\Delta \text{delay}_{\text{worst}} = \max_{w \in \text{opp\_walls}} (d_{\text{me}}^{\text{after } w} - d_{\text{me}})$$
   - Subtracts $\Delta \text{delay}_{\text{worst}} \times 5.0$ from the action's score to avoid getting bottlenecked.
5. **Anti-Cycling**: Penalizes recently occupied cells using a linearly decaying penalty (up to $-60$).

---

## 7. Agent 5: `wrongway_bot.py`

### 7.1 Overview
Direct Python port of the engine powering [wrongway.app](https://wrongway.app/) (`bot.js` & `ai.js`). It utilizes a **Negamax** search formulation with a specialized **6-feature evaluation model** derived from competitive Quoridor duel statistics.

### 7.2 6-Feature Evaluation Model (`botWert`)
```python
botWert(d_me, d_op, w_me, w_op) = (
    WEIGHTS[0]                                   # Base tempo bias (42.8)
    + WEIGHTS[1] * (d_op - d_me)                 # Path lead (100.0)
    + WEIGHTS[2] * (w_me - w_op)                 # Wall count advantage (141.2)
    + WEIGHTS[3] * (min(w_me, d_op) - min(w_op, d_me)) # Effective walls (30.4)
    + WEIGHTS[4] * race_endgame_indicator        # Pure sprint race bonus (646.2)
    + WEIGHTS[5] * wall_exhaustion_indicator     # Penalty if out of walls (-21.9)
)
```

### 7.3 Downhill BFS Wall Candidate Generation (`botKandidaten`)
Instead of evaluating all $\approx 128$ wall positions:
1. Starts from opponent's current position and moves **downhill** along their distance map ($d - 1$).
2. Identifies edges traversed by the opponent's shortest path.
3. Proposes horizontal and vertical walls directly bisecting those edges.
4. Ranks candidate walls by:
   $$\text{Gain} = (d_{\text{opp}}^{\text{after}} - d_{\text{opp}}^{\text{before}}) - (d_{\text{me}}^{\text{after}} - d_{\text{me}}^{\text{before}})$$
5. Keeps only top $4$ to $8$ walls with strictly positive net gain.

### 7.4 Search Strategy
- **Negamax with $\alpha$-$\beta$ Pruning**: Evaluates pawn steps first, then candidate walls.
- **Iterative Deepening**: Searches depths $1 \to 2 \to 3$ with a hard cutoff deadline of `0.85s`.
- **Root Bonuses**: Incorporates `FORWARD_DRIVE = 40.0` and `BACKTRACK_PENALTY = 60.0`.

---

## 8. Agent 6: `value_agent.py`

### 8.1 Overview
`value_agent.py` combines **Iterative Deepening Minimax with Alpha-Beta Pruning** and a **Deep Value Convolutional Neural Network (`ValueCNN`)** loaded directly from `models/value_model.pt` (or `models/value_models.pt`).

### 8.2 Dual-Backend Model Loader
- **PyTorch Acceleration**: Attempts PyTorch CPU/GPU inference when available.
- **Pure-Python Fallback**: If PyTorch is absent or encounters environment/NumPy issues, it extracts the 10 weight tensors (181,073 float32 parameters) directly from the `.pt` ZIP container using standard library `zipfile` and `struct.unpack`, evaluating the CNN with precomputed 3x3 convolution neighborhoods without any external dependencies.

### 8.3 State Representation (8-Plane Encoding)
Matches `temp/AI-Project1/training/state_encoder.py`:
- Plane 0: One-hot acting player position.
- Plane 1: One-hot opponent pawn position.
- Plane 2: Placed horizontal wall anchors.
- Plane 3: Placed vertical wall anchors.
- Plane 4: Acting player's remaining wall ratio ($w_{\text{me}} / 10$).
- Plane 5: Opponent's remaining wall ratio ($w_{\text{opp}} / 10$).
- Plane 6: Acting player's goal row (one-hot row).
- Plane 7: Elapsed game turn fraction ($\min(1.0, \text{turn}/\text{max\_turns})$).

### 8.4 Search & Pruning
- **Immediate Win Detection**: Instantly returns pawn moves reaching the goal row.
- **Heuristic Move Ordering & Wall Pruning**: Prunes wall candidates to top $K$ walls by net BFS path gain to maintain search speed.
- **Leaf Evaluation**: Deep positions are evaluated by $100.0 \times \text{ValueCNN}(s) \in [-100, +100]$.
- **Anti-Oscillation**: Subtracts `REPEAT_PENALTY * visit_count` to break repetition loops.

---

## 9. Common Utilities & Best Practices

Across the 5 agents, several shared patterns are critical for Quoridor AI performance:

1. **Edge-Based Wall Representation**:
   - Rather than checking wall coordinates during pathfinding, all agents convert placed walls into a set/frozenset of blocked adjacent coordinate pairs:
     $$((r_1, c_1), (r_2, c_2)) \in \text{Blocked}$$
   - This enables $\mathcal{O}(1)$ neighbor traversal checks during BFS.
2. **Move Ordering Is Crucial**:
   - In both Minimax and Negamax, sorting pawn moves closest to the goal first and filtering walls by net gain allows Alpha-Beta to achieve high pruning cutoff rates ($>70\%$).
3. **Strict Path Preservation**:
   - Any wall placement that closes off the last path to the goal for *either* player ($d = \infty$) is immediately discarded as illegal.
4. **Endgame Shortcut**:
   - When wall counts hit zero, Quoridor collapses to a deterministic shortest-path race. Advanced agents (`hard_waller`, `wrongway_bot`) skip recursive search entirely and take greedy shortest path steps.
