# Arena Guide

A complete reference for the **Wrongway Arena** package — the game engine you import to run matches, build agents, and inspect every move programmatically.

---

## Table of Contents

1. [Project layout](#1-project-layout)
2. [Installation & import](#2-installation--import)
3. [The game in one minute](#3-the-game-in-one-minute)
4. [The `observation` dict](#4-the-observation-dict)
5. [Actions](#5-actions)
6. [Mode 1 — `run_simulation`](#6-mode-1--run_simulation)
7. [Mode 2 — `Session`](#7-mode-2--session)
8. [Mode 3 — Instant simulation (rollout)](#8-mode-3--instant-simulation-rollout)
9. [Return types — `MatchResult` & `TurnRecord`](#9-return-types--matchresult--turnrecord)
10. [CLI — `python -m Arena.cli`](#10-cli--python--m-arenacli)
11. [Visualising the board](#11-visualising-the-board)
12. [Low-level access — `Environment`](#12-low-level-access--environment)
13. [Running isolated file-based agents](#13-running-isolated-file-based-agents)
14. [Recipes](#14-recipes)

---

## 1. Project layout

```
wrongway/
├── Arena/                    ← importable package (this guide)
│   ├── __init__.py           ← public API surface
│   ├── environment.py        ← pure game rules & state
│   ├── match.py              ← run_simulation + Session + simulate_action
│   ├── agent_runner.py       ← subprocess-isolated .py file runner
│   ├── visualizer.py         ← terminal board renderer
│   └── cli.py                ← command-line entry point
├── submissions/              ← student agent files (.py)
│   ├── TEMPLATE.py
│   ├── greedy.py
│   └── wall_builder.py
└── logs/                     ← JSON replay files
```

Add `wrongway/` to `sys.path` (or install as a package) and you can `import Arena` from anywhere.

---

## 2. Installation & import

### Option A — `sys.path` (no install needed)

```python
import sys
sys.path.insert(0, r"D:\project\wrongway")   # adjust to your path

from Arena import run_simulation, Session
```

### Option B — editable install

Create `wrongway/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "wrongway-arena"
version = "1.0.0"
requires-python = ">=3.9"

[tool.setuptools.packages.find]
where = ["."]
```

Then run once from `wrongway/`:

```bash
pip install -e .
```

After that, `from Arena import ...` works in any virtual environment without touching `sys.path`.

---

## 3. The game in one minute

| Item | Detail |
|---|---|
| Board | 9 × 9 grid, `[row, col]` where row 0 is top, row 8 is bottom |
| Player 0 | Starts at `[8, 4]`, must reach **row 0** |
| Player 1 | Starts at `[0, 4]`, must reach **row 8** |
| Turn order | Player 0 moves first, then alternate |
| Win | First to reach their goal row wins immediately |
| Draw | 500 half-turns elapsed with no winner (configurable) |
| Actions | **Move** one square, **place a wall**, or **pass** |
| Walls | 10 per player; block two adjacent edges; cannot strand either player |

Coordinates are always **absolute** (never flipped). Read `obs["you"]["goal_row"]` to know which direction is yours.

---

## 4. The `observation` dict

Both modes deliver the same observation structure to agents on every turn.

```python
obs = {
    "board_size":     9,              # always 9
    "player_id":      0,              # the player this observation is for
    "current_player": 0,              # whose turn it is right now
    "turn":           4,              # half-turns elapsed so far
    "max_turns":      500,            # draw limit

    "you": {
        "position":        [7, 4],    # your current [row, col]
        "goal_row":        0,         # the row you need to reach
        "walls_remaining": 10,        # walls left in your hand
    },

    "opponent": {
        "position":        [1, 4],
        "goal_row":        8,
        "walls_remaining": 10,
    },

    "walls": {
        "horizontal": [[3, 4]],       # list of anchor [row, col] already placed
        "vertical":   [],
    },

    "legal_moves":  [[6, 4], [7, 3], [7, 5]],   # cells you can move to this turn
    "legal_walls":  [{"orientation": "H", "row": 2, "col": 3}, ...],  # walls you can legally place

    "distances": [                    # 9×9 grid; distances[r][c] = shortest path
        [8, 7, 6, 5, 4, 5, 6, 7, 8], # from cell [r,c] to YOUR goal row
        ...                           # -1 means unreachable (after illegal wall, never in practice)
    ],

    "done":   False,
    "winner": None,                   # 0, 1, or None
}
```

> [!IMPORTANT]
> `distances` is precomputed **for your side only** — it encodes distances to `you["goal_row"]`, not the opponent's. Use `obs["distances"][row][col]` directly without any coordinate flipping.

> [!NOTE]
> `observation` is a snapshot copy. Mutating it has no effect on the real board.

---

## 5. Actions

Return exactly one of these dicts from your agent (or pass to `Session.step`):

```python
# Move to a cell in legal_moves
{"type": "move", "to": [row, col]}

# Place a wall listed in legal_walls
{"type": "wall", "orientation": "H", "row": row, "col": col}
# orientation: "H" (horizontal) or "V" (vertical)
# anchor row and col: 0–7

# Yield this turn
{"type": "pass"}
```

**What happens on an illegal / malformed action:** the turn is forfeited (the board is unchanged) and play passes to the opponent. This matches the original arena behaviour. The `TurnRecord` for that turn will have `status="invalid"` or `status="error"`.

### Wall geometry quick reference

| Orientation | Anchor `(r, c)` | Edges blocked |
|---|---|---|
| `H` (horizontal) | `(r, c)` | `[r,c]↔[r+1,c]` and `[r,c+1]↔[r+1,c+1]` |
| `V` (vertical)   | `(r, c)` | `[r,c]↔[r,c+1]` and `[r+1,c]↔[r+1,c+1]` |

Anchor row and col must be in `0–7`. Always pick from `obs["legal_walls"]` to avoid illegal placements.

---

## 6. Mode 1 — `run_simulation`

Both agents are **plain Python callables** in the same process. Call the function, get back a complete `MatchResult`.

### Signature

```python
from Arena import run_simulation

result = run_simulation(
    agent0,                   # callable(obs: dict) -> dict
    agent1,                   # callable(obs: dict) -> dict
    max_turns      = 500,     # draw limit
    starting_player= 0,       # 0 or 1
    seed           = None,    # int → seeds random before each call; None = unseeded
    names          = None,    # ["name0", "name1"]  default: ["player0", "player1"]
    on_turn        = None,    # optional callback(TurnRecord) called after every turn
)
```

### Minimal example

```python
from Arena import run_simulation

def greedy(obs):
    moves = obs["legal_moves"]
    if not moves:
        return {"type": "pass"}
    dist = obs["distances"]
    return {"type": "move", "to": min(
        moves, key=lambda m: dist[m[0]][m[1]] if dist[m[0]][m[1]] >= 0 else 9999
    )}

result = run_simulation(greedy, greedy, seed=42, names=["Alice", "Bob"])
print(result.winner, result.winner_name)   # 1  'Bob'
print(result.total_turns)                  # 14
```

### Live turn callback

```python
def show(t):
    print(f"Turn {t.turn} | P{t.player} | {t.status} | {t.action}")

result = run_simulation(greedy, greedy, on_turn=show)
```

### Error handling

If an agent raises an exception, the turn is **forfeited automatically** — the simulation never crashes. The turn record will have `status="error"` and `error="ExceptionType: message"`.

```python
def broken(obs):
    raise ValueError("I don't know what to do")

result = run_simulation(greedy, broken)
errored = [t for t in result.turns if t.status == "error"]
print(errored[0].error)   # "ValueError: I don't know what to do"
```

---

## 7. Mode 2 — `Session`

You own the game loop. `Session` is a stateful object: call `.observe()` to read the board, make a decision externally, then call `.step(action)` to advance one half-turn.

### Create a session

```python
from Arena import Session

session = Session(
    max_turns       = 500,     # draw limit
    starting_player = 0,       # 0 or 1
    names           = None,    # ["name0", "name1"]
)
```

### Game loop

```python
while not session.done:
    obs    = session.observe()           # observation dict for the active player
    pid    = obs["current_player"]       # 0 or 1 — whose turn it is
    action = your_agents[pid](obs)       # your logic, external model, human input, …
    record = session.step(action)        # TurnRecord for this half-turn
```

### Properties

| Property | Type | Description |
|---|---|---|
| `session.done` | `bool` | `True` once the game has ended |
| `session.current_player` | `int \| None` | Active player, `None` when done |
| `session.winner` | `int \| None` | `0`, `1`, or `None` |
| `session.history` | `list[TurnRecord]` | Live list of all records so far |
| `session.result` | `MatchResult` | Snapshot valid at any point |

### Methods

```python
obs    = session.observe()           # observation for active player (raises if done)
obs    = session.observe(player=1)   # observation from player 1's POV regardless of turn
record = session.step(action)        # submit action, returns TurnRecord
record = session.skip("my_reason")   # forfeit current turn, returns TurnRecord
snap   = session.snapshot()          # raw board dict (Environment.snapshot format)
```

### Reading the result mid-game

`session.result` is always safe to call — even before the game ends:

```python
mid = session.result
print(mid.done)         # False while still playing
print(mid.final_state)  # None while still playing
print(mid.total_turns)  # turns played so far
```

### Two-player example

```python
session = Session(names=["Human", "Bot"])

while not session.done:
    obs = session.observe()
    pid = obs["current_player"]

    if pid == 0:
        # Player 0 picks from legal_moves interactively
        for i, m in enumerate(obs["legal_moves"]):
            print(f"  {i}: move to {m}")
        idx = int(input("Pick move index: "))
        action = {"type": "move", "to": obs["legal_moves"][idx]}
    else:
        # Player 1 is a bot
        action = greedy(obs)

    session.step(action)

result = session.result
print(f"Winner: {result.winner_name} after {result.total_turns} turns")
```

> [!TIP]
> `session.result` returns a **copy** of the turn list each time. Cache it in a variable if you need to iterate over it after the game.

---

## 8. Mode 3 — Instant simulation (rollout)

The **instant simulation** (rollout) feature lets a training agent evaluate any candidate action by applying it on a *copy* of the current board state and auto-playing both sides to the end using supplied policies. The live session is **never modified**.

This is the standard building block for tree-search algorithms (MCTS, minimax) and RL value estimation.

### How it works

```
current state  ──┬──► copy of env ──► apply candidate action ──► auto-play to end ──► MatchResult
  (unchanged)    │                         (copy only)
                 │
                 └──► session keeps running normally
```

### Standalone helper — `simulate_action`

```python
from Arena import simulate_action

rollout = simulate_action(
    action,          # candidate action dict to evaluate
    state_snapshot,  # dict from session.snapshot() or env.snapshot()
    policy0,         # callable(obs) -> action for player 0's remaining turns
    policy1,         # callable(obs) -> action for player 1's remaining turns
    names  = None,   # optional ["name0", "name1"]
    on_turn= None,   # optional callback(TurnRecord) per rollout turn
)
```

`state_snapshot` is any snapshot dict — typically `session.snapshot()` from a live session. The snapshot is **not mutated**.

### Session convenience method — `Session.simulate`

```python
rollout = session.simulate(
    action,     # candidate action dict
    policy0,    # policy for player 0's remaining turns
    policy1,    # policy for player 1's remaining turns
    names  = None,
    on_turn= None,
)
```

Identical to calling `simulate_action(action, session.snapshot(), policy0, policy1, ...)` but reads the snapshot for you automatically.

> [!IMPORTANT]
> Both `simulate_action` and `session.simulate` **do not modify the live session or environment**. After a rollout you can keep calling `session.step()` as normal.

### Return value

Both return a fully-populated `MatchResult` covering the entire rollout starting from the candidate action. Key fields for training:

| Field | Use |
|---|---|
| `result.winner` | Projected winner (`0`, `1`, or `None` for draw) |
| `result.total_turns` | Projected game length |
| `result.turns` | Every turn played during the rollout |
| `result.initial_state` | Board state *before* the rollout action |
| `result.final_state` | Board state at the end of the rollout |

### Minimal example — pick the best move by rollout

```python
from Arena import Session, simulate_action

def my_policy(obs):
    moves = obs["legal_moves"]
    if not moves:
        return {"type": "pass"}
    dist = obs["distances"]
    return {"type": "move", "to": min(moves, key=lambda m: dist[m[0]][m[1]] if dist[m[0]][m[1]] >= 0 else 9999)}

session = Session(names=["Trainer", "Opponent"])

while not session.done:
    obs = session.observe()
    pid = obs["current_player"]

    if pid == 0:
        # Evaluate every legal move via instant simulation
        best_action, best_winner = None, -1
        for m in obs["legal_moves"]:
            candidate = {"type": "move", "to": m}
            rollout = session.simulate(candidate, my_policy, my_policy)
            if rollout.winner == 0:
                best_action = candidate
                break   # found a winning move — no need to search further
        action = best_action or {"type": "move", "to": obs["legal_moves"][0]}
    else:
        action = my_policy(obs)

    # Commit the REAL turn — session.simulate never touched it
    session.step(action)

print(session.result.winner, session.result.total_turns)
```

### Training loop — collect rollout results for RL

```python
from Arena import Session, simulate_action

training_data = []   # (state_snapshot, action, outcome) triples

session = Session()
while not session.done:
    obs    = session.observe()
    pid    = obs["current_player"]
    state  = session.snapshot()          # save state before acting

    # Your RL agent picks a candidate action
    candidate = my_rl_agent(obs)

    # Instant simulation: let both sides play the rest with the current policy
    rollout = session.simulate(candidate, current_policy, current_policy)

    # Outcome from the perspective of the acting player
    if rollout.winner == pid:
        outcome = 1.0    # win
    elif rollout.winner is None:
        outcome = 0.5    # draw
    else:
        outcome = 0.0    # loss

    training_data.append((state, candidate, outcome))

    # Commit the real turn and continue
    session.step(candidate)

# Use training_data to update your model
for state, action, outcome in training_data:
    my_rl_agent.update(state, action, outcome)
```

> [!TIP]
> For speed, pass a **cheap heuristic** (like `greedy`) as `policy0` / `policy1` instead of your full model. The rollout only needs to be good enough to give a useful value signal.

> [!NOTE]
> `simulate_action` accepts any snapshot dict, not just live session snapshots. You can store snapshots from past games and replay rollouts from arbitrary past states.

---

## 9. Return types — `MatchResult` & `TurnRecord`

Both `run_simulation`, `simulate_action`, and `Session.result` return a `MatchResult` containing a list of `TurnRecord` objects.

### `MatchResult`

| Attribute | Type | Description |
|---|---|---|
| `.winner` | `int \| None` | `0`, `1`, or `None` for draw / in-progress |
| `.winner_name` | `str \| None` | Friendly name from `names=` |
| `.termination_reason` | `str \| None` | `'goal'`, `'max_turns'`, or `None` |
| `.total_turns` | `int` | Half-turns played |
| `.turns` | `list[TurnRecord]` | Every turn, in order |
| `.history` | `list[TurnRecord]` | Alias for `.turns` |
| `.initial_state` | `dict` | Board snapshot before turn 0 |
| `.final_state` | `dict \| None` | Board snapshot after last turn (`None` if live) |
| `.config` | `dict` | `max_turns`, `seed`, `starting_player`, `player_names` |
| `.done` | `bool` | `True` once `termination_reason` is set |

### `TurnRecord`

| Attribute | Type | Description |
|---|---|---|
| `.turn` | `int` | Turn counter after the action (starts at 1) |
| `.player` | `int` | Which player acted: `0` or `1` |
| `.action` | `dict \| None` | The applied action dict, or `None` for a forfeit |
| `.status` | `str` | `'ok'`, `'pass'`, `'invalid'`, or `'error'` |
| `.reason` | `str \| None` | Engine reason: `'move'`, `'wall'`, `'illegal_move'`, … |
| `.elapsed` | `float` | Seconds the agent took (`0.0` in Session mode) |
| `.error` | `str \| None` | Agent exception message, if any |
| `.state` | `dict` | Full board snapshot **after** this action |

### Common queries

```python
# All moves player 0 made
p0_moves = [t for t in result.turns if t.player == 0 and t.status == "ok"]

# All walls placed
walls_placed = [t for t in result.turns if t.action and t.action["type"] == "wall"]

# Any error turns
errors = [t for t in result.turns if t.status == "error"]

# Reconstruct positions over time
positions = [t.state["positions"] for t in result.turns]

# Save to JSON
import json
data = {
    "winner": result.winner,
    "turns": [
        {"turn": t.turn, "player": t.player, "action": t.action, "status": t.status}
        for t in result.turns
    ]
}
json.dump(data, open("replay.json", "w"), indent=2)
```

---

## 10. CLI — `python -m Arena.cli`

Run from the `wrongway/` directory (the folder that contains `Arena/`):

```bash
python -m Arena.cli --player1 submissions/greedy.py --player2 submissions/wall_builder.py
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--player1 / -p1` | *(required)* | Path to a `.py` agent file or student ID |
| `--player2 / -p2` | *(required)* | Path to a `.py` agent file or student ID |
| `--submissions-dir` | `submissions/` | Folder to look up agents by name |
| `--timeout` | `1.0` | Seconds per action |
| `--startup-timeout` | `5.0` | Seconds to import the agent module |
| `--max-turns` | `500` | Draw limit |
| `--seed` | `0` | RNG seed |
| `--starting-player` | `0` | Which player goes first (`0` or `1`) |
| `--render` | off | Print the board after every turn |
| `--delay` | `0.0` | Seconds to pause between renders |
| `--log` | *(none)* | Save full JSON replay to this path |

### Examples

```bash
# Render a match in the terminal
python -m Arena.cli -p1 submissions/greedy.py -p2 submissions/wall_builder.py --render

# Short name lookup (looks in submissions/ automatically)
python -m Arena.cli -p1 greedy -p2 wall_builder

# Save a replay for later analysis
python -m Arena.cli -p1 greedy -p2 wall_builder --log logs/my_match.json

# Slow down rendering for visibility
python -m Arena.cli -p1 greedy -p2 wall_builder --render --delay 0.3

# Player 1 goes first
python -m Arena.cli -p1 greedy -p2 wall_builder --starting-player 1
```

> [!NOTE]
> The original `arena.py` at the top level still works unchanged. `Arena.cli` is an equivalent copy adapted for package-relative imports.

---

## 11. Visualising the board

```python
from Arena import render_board

# Works on any snapshot dict (MatchResult.initial_state, TurnRecord.state, etc.)
print(render_board(result.final_state))
print(render_board(result.final_state, names=["Alice", "Bob"]))
```

**Example output:**

```
     0   1   2   3   4   5   6   7   8
 0   .   .   .   .   1   .   .   .   .
 1   .   .   .   .   .   .   .   .   .
 2   .   .   .   .   .   .   .   .   .
 3   .   .   .   .   .   .   .   .   .
 4   .   .   .   .   .   .   .   .   .
 5   .   .   .   .   .   .   .   .   .
 6   .   .   .   .   .   .   .   .   .
 7   .   .   .   .   .   .   .   .   .
 8   .   .   .   .   0   .   .   .   .
P0 Alice: 10 walls, goal row 0 | P1 Bob: 10 walls, goal row 8
```

- `0` / `1` — pawn positions
- `=` — horizontal wall
- `#` — vertical wall
- `.` — empty cell

Print the board on every turn during a simulation:

```python
from Arena import render_board, run_simulation

def on_turn(t):
    print(f"\nTurn {t.turn} | P{t.player} | {t.status}")
    print(render_board(t.state, names=["Alice", "Bob"]))

run_simulation(greedy, greedy, on_turn=on_turn)
```

---

## 12. Low-level access — `Environment`

`Environment` is the pure game engine. You can drive it directly if you need fine-grained control.

```python
from Arena import Environment

env = Environment(max_turns=500, starting_player=0)

# Read state
obs = env.get_observation(player=0)
obs = env.get_observation()          # defaults to env.current_player

# Query legal actions without committing
moves = env.legal_moves()            # list of [row, col]
walls = env.legal_wall_actions()     # list of {"orientation", "row", "col"}

# BFS distances to goal (walls respected, pawns ignored)
dist = env.distance_map(player=0)    # 9×9 list of lists

# Apply an action
result = env.step({"type": "move", "to": [7, 4]})
# result dict: {"status": "ok"/"invalid", "reason": ..., "action": ..., ...}

# Forfeit a turn
result = env.skip_turn("timeout")

# Full board snapshot (JSON-safe)
snap = env.snapshot()

# Useful attributes
env.current_player    # 0 or 1
env.turn_count        # half-turns elapsed
env.done              # True when game over
env.winner            # 0, 1, or None
env.positions         # [(row0, col0), (row1, col1)]
env.walls_remaining   # [walls_p0, walls_p1]
env.horizontal_walls  # set of (row, col) anchors
env.vertical_walls    # set of (row, col) anchors
env.termination_reason  # 'goal', 'max_turns', or None
```

> [!WARNING]
> Calling `env.step()` or `env.skip_turn()` after `env.done is True` raises `RuntimeError`. Check `env.done` first.

---

## 13. Running isolated file-based agents

`StudentAgent` runs a `.py` submission in a **separate subprocess**, with time limits and fault isolation. This is what the CLI uses under the hood.

```python
from Arena.agent_runner import StudentAgent
from Arena import Environment
import json

env = Environment()

with StudentAgent("submissions/greedy.py", timeout=1.0, seed=0) as agent:
    while not env.done:
        obs      = env.get_observation()
        response = agent.act(obs)       # ActionResult

        print(response.status)         # 'ok', 'timeout', or 'error'
        print(response.elapsed)        # seconds taken
        print(response.error)          # None or error string

        if response.status == "ok":
            env.step(response.action)
        else:
            env.skip_turn(response.status)
```

| `ActionResult` field | Type | Meaning |
|---|---|---|
| `.action` | `dict \| None` | The returned action dict, or `None` on failure |
| `.status` | `str` | `'ok'`, `'timeout'`, or `'error'` |
| `.elapsed` | `float` | Wall-clock seconds for the call |
| `.error` | `str \| None` | Error message if applicable |

> [!CAUTION]
> `StudentAgent` does **not** sandbox file-system or network access. Only run trusted submissions, or use an OS-level container for untrusted code.

---

## 14. Recipes

### Tournament — round-robin

```python
from itertools import combinations
from Arena import run_simulation

agents = {"greedy": greedy_fn, "wall_builder": wall_fn, "my_bot": my_fn}
scores = {name: 0 for name in agents}

for (n0, a0), (n1, a1) in combinations(agents.items(), 2):
    # Play both sides
    for p0, p1, name0, name1 in [(a0, a1, n0, n1), (a1, a0, n1, n0)]:
        result = run_simulation(p0, p1, names=[name0, name1])
        if result.winner == 0:
            scores[name0] += 1
        elif result.winner == 1:
            scores[name1] += 1

for name, s in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"{name}: {s} wins")
```

### Save & reload a replay as JSON

```python
import json
from Arena import run_simulation
from dataclasses import asdict

result = run_simulation(greedy, greedy, names=["A", "B"])

replay = {
    "winner": result.winner,
    "winner_name": result.winner_name,
    "total_turns": result.total_turns,
    "termination_reason": result.termination_reason,
    "config": result.config,
    "initial_state": result.initial_state,
    "final_state": result.final_state,
    "turns": [asdict(t) for t in result.turns],
}

with open("my_replay.json", "w") as f:
    json.dump(replay, f, indent=2)
```

### Render every turn to the terminal during simulation

```python
from Arena import run_simulation, render_board

def printer(t):
    print(f"\n── Turn {t.turn} | Player {t.player} | {t.status} ──")
    print(render_board(t.state))

run_simulation(greedy, greedy, on_turn=printer)
```

### Collect all states for ML training

```python
result = run_simulation(greedy, greedy)

# Each element: (observation_snapshot_dict, action_taken)
training_pairs = [
    (t.state, t.action)
    for t in result.turns
    if t.status == "ok"
]
```

### Interrupt a session on a condition

```python
session = Session()

while not session.done:
    obs = session.observe()
    action = greedy(obs)
    record = session.step(action)

    # Stop after 10 turns for debugging
    if record.turn >= 10:
        print("Stopping early for inspection")
        break

# session.result is still valid mid-game
mid = session.result
print(mid.done, mid.total_turns)
```

### Use `skip()` to simulate a timeout

```python
session = Session()

while not session.done:
    obs = session.observe()
    pid = obs["current_player"]

    if pid == 0:
        action = slow_agent(obs)
        record = session.step(action)
    else:
        # Simulate player 1 timing out every other turn
        record = session.skip("timeout")
```

### Rollout-based value estimation for RL

Use `session.simulate` to get outcome signals for any candidate action without committing it to the game — ideal for policy-gradient or value-network training:

```python
from Arena import Session, simulate_action

training_data = []   # (state_snapshot, action, outcome) triples

session = Session()
while not session.done:
    obs    = session.observe()
    pid    = obs["current_player"]
    state  = session.snapshot()          # save the board before acting

    # Your agent picks a candidate (from your model, epsilon-greedy, etc.)
    candidate = my_rl_agent.choose(obs)

    # Roll out the game from this action with the current policy
    rollout = session.simulate(candidate, current_policy, current_policy)

    # Map outcome to a scalar reward from the acting player's perspective
    if rollout.winner == pid:
        outcome = 1.0    # win
    elif rollout.winner is None:
        outcome = 0.5    # draw
    else:
        outcome = 0.0    # loss

    training_data.append((state, candidate, outcome))

    # Commit the REAL turn and continue the live game
    session.step(candidate)

# Update your model with the collected (state, action, outcome) triples
for state, action, outcome in training_data:
    my_rl_agent.update(state, action, outcome)
```

### MCTS-style look-ahead: score every legal move

```python
from Arena import Session

session = Session()
obs = session.observe()

scores = {}
for m in obs["legal_moves"]:
    candidate = {"type": "move", "to": m}
    rollout   = session.simulate(candidate, rollout_policy, rollout_policy)
    my_pid    = obs["player_id"]
    scores[tuple(m)] = 1.0 if rollout.winner == my_pid else (0.5 if rollout.winner is None else 0.0)

# Pick the highest-scoring move
best_move = max(scores, key=scores.get)
session.step({"type": "move", "to": list(best_move)})
```
