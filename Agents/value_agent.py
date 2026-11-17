"""Value Model Agent for Wrongway Arena Quoridor.

This agent evaluates board positions using the trained ValueCNN model stored
in models/value_model.pt (or models/value_models.pt).

Key features:
- Dual-Backend Model Loading:
  * Automatically attempts PyTorch acceleration if torch is available.
  * Includes a built-in standard-library fallback (zipfile + struct) that
    loads weights directly from the .pt archive with zero external dependencies.
- State Representation:
  * Encodes the board into the exact 8x9x9 feature plane representation
    trained in state_encoder.py (acting pawn, opp pawn, H walls, V walls,
    own wall ratio, opp wall ratio, own goal row, elapsed game fraction).
- Decision Engine:
  * Instant 1-ply winning move check.
  * Downhill shortest-path opponent obstruction wall candidate ranking.
  * Minimax search with Alpha-Beta pruning using ValueCNN leaf evaluation.
  * Anti-oscillation history penalty to prevent repetitive shuffle loops.
  * Strict time-budget safety margin ensuring response well within the 1.0s limit.
"""

from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
BOARD_SIZE = 9
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_WALL_ANCHORS = tuple((ori, r, c) for ori in ("H", "V") for r in range(8) for c in range(8))

# Search defaults
DEFAULT_TIME_LIMIT = 0.75       # Arena timeout is 1.0s; 0.75s leaves safe margin
DEFAULT_MAX_DEPTH = 3           # Iterative deepening depth limit
DEFAULT_WALL_BRANCHES = 3       # Top K candidate walls evaluated per position
REPEAT_PENALTY = 8.0            # Penalty for revisiting recent cells (breaks cycles)

# Layers expected in ValueCNN:
CNN_LAYER_SPECS = (
    ("network.0.weight", 1152),   # (16, 8, 3, 3)
    ("network.0.bias", 16),       # (16,)
    ("network.2.weight", 4608),   # (32, 16, 3, 3)
    ("network.2.bias", 32),       # (32,)
    ("network.4.weight", 9216),   # (32, 32, 3, 3)
    ("network.4.bias", 32),       # (32,)
    ("network.7.weight", 165888), # (64, 2592)
    ("network.7.bias", 64),       # (64,)
    ("network.9.weight", 64),     # (1, 64)
    ("network.9.bias", 1),        # (1,)
)

# Precomputed 3x3 convolution neighborhoods for fast pure-Python convolution
CNN_NEIGHBORS = []
for _p in range(81):
    _r, _c = divmod(_p, 9)
    CNN_NEIGHBORS.append(tuple(
        ((_r + _kr - 1) * 9 + (_c + _kc - 1), _kr * 3 + _kc)
        for _kr in range(3) for _kc in range(3)
        if 0 <= _r + _kr - 1 < 9 and 0 <= _c + _kc - 1 < 9
    ))


# ==============================================================================
# MODEL BACKENDS & LOADER
# ==============================================================================
def find_checkpoint_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Search for the value model checkpoint file in known standard locations."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))
        candidates.append(repo_root / explicit_path)
        # Handle singular vs plural filename aliases
        if "value_models.pt" in str(explicit_path):
            alt = str(explicit_path).replace("value_models.pt", "value_model.pt")
            candidates.append(Path(alt))
            candidates.append(repo_root / alt)
        elif "value_model.pt" in str(explicit_path):
            alt = str(explicit_path).replace("value_model.pt", "value_models.pt")
            candidates.append(Path(alt))
            candidates.append(repo_root / alt)

    env_path = os.getenv("VALUE_MODEL_PATH")
    if env_path:
        candidates.append(Path(env_path))
        candidates.append(repo_root / env_path)

    # Standard default names and locations (supporting both model and models)
    defaults = [
        "models/value_model.pt",
        "models/value_models.pt",
        "value_model.pt",
        "value_models.pt",
        "MeowMeowBot/value_model.pt",
    ]
    for d in defaults:
        candidates.append(repo_root / d)
        candidates.append(Path(d))

    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


class PurePythonValueCNN:
    """Pure Python inference engine for ValueCNN using weights unpacked from .pt."""

    def __init__(self, parts: List[Tuple[float, ...]]):
        if len(parts) != 10:
            raise ValueError(f"Expected 10 weight tensors, got {len(parts)}")
        (
            self.w0, self.b0,
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4,
        ) = parts

    @staticmethod
    def _conv_relu(x: List[float], in_ch: int, out_ch: int,
                   weights: Tuple[float, ...], biases: Tuple[float, ...]) -> List[float]:
        y = [0.0] * (out_ch * 81)
        for oc in range(out_ch):
            ob = oc * 81
            b = biases[oc]
            for p, neighbors in enumerate(CNN_NEIGHBORS):
                s = b
                for ic in range(in_ch):
                    xb = ic * 81
                    wb = (oc * in_ch + ic) * 9
                    for ip, ki in neighbors:
                        xv = x[xb + ip]
                        if xv:
                            s += xv * weights[wb + ki]
                if s > 0.0:
                    y[ob + p] = s
        return y

    def predict(self, x: List[float]) -> float:
        """Forward pass on flat 648 (8x9x9) input planes. Output in [-1.0, 1.0]."""
        a = self._conv_relu(x, 8, 16, self.w0, self.b0)
        a = self._conv_relu(a, 16, 32, self.w1, self.b1)
        a = self._conv_relu(a, 32, 32, self.w2, self.b2)

        # Dense linear 1: 32*9*9 (2592) -> 64 + ReLU
        z = [0.0] * 64
        for i in range(64):
            offset = i * 2592
            s = self.b3[i]
            for j, val in enumerate(a):
                if val:
                    s += val * self.w3[offset + j]
            if s > 0.0:
                z[i] = s

        # Dense linear 2: 64 -> 1 + Tanh
        out = self.b4[0]
        for i in range(64):
            val = z[i]
            if val:
                out += val * self.w4[i]

        return math.tanh(out)


def load_model_from_checkpoint(checkpoint_path: Path):
    """Load model using PyTorch if available, else unpack via zipfile + struct."""
    # Attempt 1: Try PyTorch
    try:
        import torch
        import torch.nn as nn

        class TorchValueCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * 9 * 9, 64), nn.ReLU(),
                    nn.Linear(64, 1), nn.Tanh(),
                )

            def forward(self, x):
                return self.network(x).squeeze(-1)

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        model = TorchValueCNN()
        model.load_state_dict(state_dict)
        model.eval()

        class TorchWrapper:
            def __init__(self, net):
                self.net = net

            def predict(self, x_flat: List[float]) -> float:
                with torch.no_grad():
                    t = torch.tensor(x_flat, dtype=torch.float32).view(1, 8, 9, 9)
                    return float(self.net(t).item())

        return TorchWrapper(model)
    except Exception:
        pass

    # Attempt 2: Direct zipfile extraction of raw float32 buffers
    try:
        with zipfile.ZipFile(str(checkpoint_path), "r") as z:
            names = z.namelist()
            prefix = ""
            for name in names:
                if name.endswith("data/0"):
                    prefix = name[:-len("data/0")]
                    break

            parts = []
            for i in range(10):
                entry = f"{prefix}data/{i}"
                raw = z.read(entry)
                count = len(raw) // 4
                floats = struct.unpack(f"<{count}f", raw)
                parts.append(floats)

            return PurePythonValueCNN(parts)
    except Exception as e:
        raise RuntimeError(f"Failed to load value model from {checkpoint_path}: {e}")


# ==============================================================================
# STATE & GAME MECHANICS
# ==============================================================================
def _walls_to_blocked(h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    blocked = set()
    for r, c in h_walls:
        for u, v in (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1))):
            blocked.add((u, v))
            blocked.add((v, u))
    for r, c in v_walls:
        for u, v in (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1))):
            blocked.add((u, v))
            blocked.add((v, u))
    return blocked


class State:
    """Immutable compact board representation for search."""
    __slots__ = ("pos", "goals", "walls_rem", "h_walls", "v_walls", "blocked")

    def __init__(self, pos, goals, walls_rem, h_walls, v_walls, blocked=None):
        self.pos = pos              # ((my_r, my_c), (opp_r, opp_c))
        self.goals = goals          # (my_goal_row, opp_goal_row)
        self.walls_rem = walls_rem  # (my_walls, opp_walls)
        self.h_walls = h_walls      # frozenset of (r, c)
        self.v_walls = v_walls      # frozenset of (r, c)
        self.blocked = blocked if blocked is not None else frozenset(_walls_to_blocked(h_walls, v_walls))


def bfs_distance(position: Tuple[int, int], goal_row: int, blocked: frozenset) -> int:
    """Breadth-first search for the shortest path distance to the goal row."""
    if position[0] == goal_row:
        return 0
    q = deque([(position, 0)])
    visited = {position}
    while q:
        (r, c), dist = q.popleft()
        nd = dist + 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                nxt = (nr, nc)
                if nxt not in visited and ((r, c), nxt) not in blocked:
                    if nr == goal_row:
                        return nd
                    visited.add(nxt)
                    q.append((nxt, nd))
    return 999


def all_legal_pawn_moves(state: State, player: int) -> List[Dict[str, Any]]:
    """Generate all legal pawn moves for player (0=you, 1=opponent), including jumps."""
    r, c = state.pos[player]
    opp = state.pos[1 - player]
    blocked = state.blocked
    dests = set()

    for dr, dc in DIRS:
        adj = (r + dr, c + dc)
        if not (0 <= adj[0] < BOARD_SIZE and 0 <= adj[1] < BOARD_SIZE) or ((r, c), adj) in blocked:
            continue
        if adj != opp:
            dests.add(adj)
        else:
            jmp = (adj[0] + dr, adj[1] + dc)
            if 0 <= jmp[0] < BOARD_SIZE and 0 <= jmp[1] < BOARD_SIZE and (adj, jmp) not in blocked:
                dests.add(jmp)
            else:
                for sdr, sdc in ((dc, dr), (-dc, -dr)):
                    side = (adj[0] + sdr, adj[1] + sdc)
                    if 0 <= side[0] < BOARD_SIZE and 0 <= side[1] < BOARD_SIZE and (adj, side) not in blocked:
                        dests.add(side)

    return [{"type": "move", "to": [d[0], d[1]]} for d in sorted(dests)]


def wall_does_not_overlap(state: State, ori: str, r: int, c: int) -> bool:
    """Check if a wall candidate intersects or overlaps existing walls."""
    if ori == "H":
        if (r, c) in state.v_walls:
            return False
        return not ((r, c - 1) in state.h_walls or (r, c) in state.h_walls or (r, c + 1) in state.h_walls)
    if (r, c) in state.h_walls:
        return False
    return not ((r - 1, c) in state.v_walls or (r, c) in state.v_walls or (r + 1, c) in state.v_walls)


def apply_action(state: State, action: Dict[str, Any], player: int) -> State:
    """Return the new State after player plays action."""
    if action["type"] == "move":
        to = tuple(action["to"])
        new_pos = (to, state.pos[1]) if player == 0 else (state.pos[0], to)
        return State(new_pos, state.goals, state.walls_rem, state.h_walls, state.v_walls, state.blocked)

    if action["type"] == "wall":
        ori, r, c = action["orientation"], action["row"], action["col"]
        new_w = (state.walls_rem[0] - 1, state.walls_rem[1]) if player == 0 else (state.walls_rem[0], state.walls_rem[1] - 1)
        if ori == "H":
            new_h, new_v = state.h_walls | {(r, c)}, state.v_walls
            edges = (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1)))
        else:
            new_h, new_v = state.h_walls, state.v_walls | {(r, c)}
            edges = (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1)))
        added = {(u, v) for a, b in edges for u, v in ((a, b), (b, a))}
        return State(state.pos, state.goals, new_w, new_h, new_v, state.blocked | added)

    return state


# ==============================================================================
# STATE ENCODING (8x9x9) & VALUE EVALUATION
# ==============================================================================
def encode_leaf_state(state: State, next_player: int, turn: int, max_turns: int) -> List[float]:
    """Encode board state from next_player's POV matching state_encoder.py exactly."""
    x = [0.0] * 648

    own = state.pos[next_player]
    opp = state.pos[1 - next_player]
    # Plane 0: acting pawn (one-hot)
    x[own[0] * 9 + own[1]] = 1.0
    # Plane 1: opponent pawn (one-hot)
    x[81 + opp[0] * 9 + opp[1]] = 1.0
    # Plane 2: horizontal wall anchors
    for r, c in state.h_walls:
        x[162 + r * 9 + c] = 1.0
    # Plane 3: vertical wall anchors
    for r, c in state.v_walls:
        x[243 + r * 9 + c] = 1.0

    # Plane 4 & 5: remaining wall ratios
    my_w = state.walls_rem[next_player] / 10.0
    their_w = state.walls_rem[1 - next_player] / 10.0
    # Plane 7: elapsed game fraction
    frac = min(1.0, max(0.0, turn / max(1, max_turns)))

    for p in range(81):
        x[324 + p] = my_w
        x[405 + p] = their_w
        x[567 + p] = frac

    # Plane 6: goal row (one-hot row for acting player)
    goal_base = 486 + state.goals[next_player] * 9
    for c in range(9):
        x[goal_base + c] = 1.0

    return x


def evaluate_leaf(state: State, player: int, turn: int, max_turns: int, model: Any) -> float:
    """Return leaf score from MAX (player 0) POV using the neural network value model."""
    planes = encode_leaf_state(state, player, turn, max_turns)
    val = model.predict(planes)
    # The model predicts the expected outcome in [-1, +1] from player's POV
    # If player == 0 (MAX), higher is better for MAX.
    # If player == 1 (MIN), higher is better for MIN, so negate for MAX.
    score = 100.0 * (val if player == 0 else -val)
    return score


# ==============================================================================
# SEARCH & ACTION SELECTION
# ==============================================================================
class SearchTimeout(Exception):
    """Raised when the search budget is exhausted."""


def heuristic_quick_score(state: State) -> float:
    """Fast distance-based heuristic for move ordering."""
    d0 = bfs_distance(state.pos[0], state.goals[0], state.blocked)
    d1 = bfs_distance(state.pos[1], state.goals[1], state.blocked)
    wall_adv = state.walls_rem[0] - state.walls_rem[1]
    prog0 = 8 - abs(state.pos[0][0] - state.goals[0])
    prog1 = 8 - abs(state.pos[1][0] - state.goals[1])
    return 10.0 * (d1 - d0) + 2.0 * wall_adv + 0.5 * (prog0 - prog1)


def generate_candidate_actions(state: State, player: int, deadline: float, wall_branches: int) -> List[Dict[str, Any]]:
    """Generate legal pawn moves and top candidate walls ordered by heuristic."""
    if time.monotonic() >= deadline:
        raise SearchTimeout

    scored_moves = []
    for mv in all_legal_pawn_moves(state, player):
        child = apply_action(state, mv, player)
        scored_moves.append((heuristic_quick_score(child), mv))

    scored_walls = []
    if state.walls_rem[player] > 0:
        for ori, r, c in ALL_WALL_ANCHORS:
            if time.monotonic() >= deadline:
                raise SearchTimeout
            if not wall_does_not_overlap(state, ori, r, c):
                continue

            w_act = {"type": "wall", "orientation": ori, "row": r, "col": c}
            child = apply_action(state, w_act, player)

            # Ensure neither player is blocked completely
            d0 = bfs_distance(child.pos[0], child.goals[0], child.blocked)
            if d0 >= 999:
                continue
            d1 = bfs_distance(child.pos[1], child.goals[1], child.blocked)
            if d1 >= 999:
                continue

            scored_walls.append((heuristic_quick_score(child), w_act))

    reverse = (player == 0)
    scored_moves.sort(key=lambda x: x[0], reverse=reverse)
    scored_walls.sort(key=lambda x: x[0], reverse=reverse)

    top_walls = [act for _, act in scored_walls[:wall_branches]]
    ordered = [act for _, act in scored_moves] + top_walls
    return ordered or [{"type": "pass"}]


def minimax(state: State, depth: int, alpha: float, beta: float, player: int,
            turn: int, max_turns: int, model: Any, deadline: float,
            is_root: bool = False, recent_counts: Optional[Dict[Tuple[int, int], int]] = None,
            wall_branches: int = DEFAULT_WALL_BRANCHES) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Minimax search with Alpha-Beta pruning and ValueCNN leaf evaluation."""
    if time.monotonic() >= deadline:
        raise SearchTimeout

    # Terminal checks
    if state.pos[0][0] == state.goals[0]:
        return (10000.0 + depth * 10.0), None
    if state.pos[1][0] == state.goals[1]:
        return (-10000.0 - depth * 10.0), None

    if depth == 0:
        return evaluate_leaf(state, player, turn, max_turns, model), None

    actions = generate_candidate_actions(state, player, deadline, wall_branches)
    best_act = None

    if player == 0:  # MAX player
        val = -float("inf")
        for act in actions:
            child = apply_action(state, act, 0)
            score, _ = minimax(child, depth - 1, alpha, beta, 1, turn + 1, max_turns, model, deadline, wall_branches=wall_branches)

            if is_root and act["type"] == "move" and recent_counts:
                to_pos = tuple(act["to"])
                if to_pos in recent_counts:
                    score -= REPEAT_PENALTY * recent_counts[to_pos]

            if score > val:
                val = score
                best_act = act
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return val, best_act
    else:  # MIN player
        val = float("inf")
        for act in actions:
            child = apply_action(state, act, 1)
            score, _ = minimax(child, depth - 1, alpha, beta, 0, turn + 1, max_turns, model, deadline, wall_branches=wall_branches)
            if score < val:
                val = score
                best_act = act
            beta = min(beta, val)
            if beta <= alpha:
                break
        return val, best_act


# ==============================================================================
# AGENT CLASS & CHOOSE_ACTION EXPORT
# ==============================================================================
class ValueAgent:
    """Quoridor Agent utilizing models/value_model.pt for position evaluation."""

    def __init__(self, model_path: Optional[str] = None,
                 time_limit: float = DEFAULT_TIME_LIMIT,
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 wall_branches: int = DEFAULT_WALL_BRANCHES):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.wall_branches = wall_branches
        self.recent_positions: deque = deque(maxlen=10)

        resolved_path = find_checkpoint_path(model_path)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Value model checkpoint not found. Checked: explicit={model_path}, "
                f"env={os.getenv('VALUE_MODEL_PATH')}, and default paths in 'models/'."
            )
        self.checkpoint_path = resolved_path
        self.model = load_model_from_checkpoint(self.checkpoint_path)

    def reset(self):
        """Reset internal history between games."""
        self.recent_positions.clear()

    def __call__(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return self.choose_action(observation)

    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        deadline = time.monotonic() + self.time_limit

        # Detect new game / match reset
        turn = observation.get("turn", 0)
        max_turns = observation.get("max_turns", 500)
        if turn <= 1:
            self.recent_positions.clear()

        you = observation["you"]
        opp = observation["opponent"]
        my_pos = tuple(you["position"])
        opp_pos = tuple(opp["position"])
        self.recent_positions.append(my_pos)
        recent_counts = {p: self.recent_positions.count(p) for p in self.recent_positions}

        legal_moves = [tuple(m) for m in observation.get("legal_moves", [])]

        # 1. Immediate Win Check
        for m in legal_moves:
            if m[0] == you["goal_row"]:
                return {"type": "move", "to": list(m)}

        # Construct current State
        h_walls = frozenset((w[0], w[1]) for w in observation["walls"].get("horizontal", []))
        v_walls = frozenset((w[0], w[1]) for w in observation["walls"].get("vertical", []))
        state = State(
            pos=(my_pos, opp_pos),
            goals=(you["goal_row"], opp["goal_row"]),
            walls_rem=(you["walls_remaining"], opp["walls_remaining"]),
            h_walls=h_walls,
            v_walls=v_walls,
        )

        # 2. Safe Fallback: best 1-step move by shortest path distance
        distances = observation.get("distances")
        if distances:
            fallback_move = min(
                legal_moves,
                key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else 999
            )
            best_action = {"type": "move", "to": list(fallback_move)}
        elif legal_moves:
            best_action = {"type": "move", "to": list(legal_moves[0])}
        else:
            best_action = {"type": "pass"}

        # 3. Iterative Deepening Minimax Search with ValueCNN leaf evaluation
        for depth in range(1, self.max_depth + 1):
            try:
                _, act = minimax(
                    state, depth, -float("inf"), float("inf"),
                    player=0, turn=turn, max_turns=max_turns,
                    model=self.model, deadline=deadline,
                    is_root=True, recent_counts=recent_counts,
                    wall_branches=self.wall_branches
                )
                if act is not None:
                    best_action = act
            except SearchTimeout:
                break

        return best_action


# Global agent singleton for convenient choose_action(observation) usage
_DEFAULT_AGENT: Optional[ValueAgent] = None


def get_default_agent() -> ValueAgent:
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = ValueAgent()
    return _DEFAULT_AGENT


def choose_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level entry point compatible with Arena.cli and StudentAgent."""
    agent = get_default_agent()
    return agent.choose_action(observation)


if __name__ == "__main__":
    sample_obs = {
        "board_size": 9,
        "player_id": 0,
        "current_player": 0,
        "turn": 0,
        "max_turns": 200,
        "done": False,
        "winner": None,
        "you": {"position": [4, 4], "goal_row": 8, "walls_remaining": 10},
        "opponent": {"position": [2, 4], "goal_row": 0, "walls_remaining": 10},
        "walls": {"horizontal": [], "vertical": []},
        "distances": [[abs(r - 8) for _ in range(9)] for r in range(9)],
        "legal_moves": [[3, 4], [4, 3], [4, 5], [5, 4]],
        "legal_walls": [],
    }
    agent = ValueAgent()
    act = agent(sample_obs)
    print("Action chosen by ValueAgent:", act)
