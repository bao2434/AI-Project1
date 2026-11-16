"""
Quoridor MCTS agent adapted to the Wrongway Arena observation interface.

Entry point:
    choose_action(observation) -> {"type": "move", "to": [r, c]}
                               or {"type": "wall", ...}

The implementation translates the main ideas from the supplied JavaScript
MCTS agent:
- UCT selection
- expansion of pawn moves and probable wall placements
- BFS shortest-path-guided rollout (70% forward-path policy)
- wall placements focused on interrupting shortest paths / nearby pawns
- backpropagation of wins
- time-bounded search with a legal-action fallback

This is an adaptation, not a line-by-line port. The arena observation is the
source of truth for legal root actions. Internal search generates actions
using the Quoridor rules represented below.
"""

from collections import deque
import math
import random
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIME_LIMIT = 0.90          # Keep below the arena's 1-second limit.
MAX_SIMULATIONS = 100000   # Time limit normally stops the search first.
UCT_C = math.sqrt(2.0)
ROLLOUT_FORWARD_PROB = 0.70
MAX_ROLLOUT_PLIES = 250    # Safety limit against pathological cycles.
MAX_WALL_CANDIDATES = 36   # Limit wall branching for Python runtime.
RANDOM_SEED = None

SIZE = 9
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

class SearchTimeout(Exception):
    """Raised when iterative deepening reaches the turn deadline."""
def _walls_to_blocked(h_walls, v_walls):
    b = set()
    for r, c in h_walls:
        for u, v in (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1))):
            b.add((u, v))
            b.add((v, u))
    for r, c in v_walls:
        for u, v in (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1))):
            b.add((u, v))
            b.add((v, u))
    return frozenset(b)


# ---------------------------------------------------------------------------
# State and wall representation
# ---------------------------------------------------------------------------

def add_wall_edges(blocked, orientation, row, col):
    if orientation == "H":
        pairs = (
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        )
    else:
        pairs = (
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        )

    for a, b in pairs:
        blocked.add((a, b))
        blocked.add((b, a))


def walls_to_blocked(h_walls, v_walls):
    blocked = set()
    for r, c in h_walls:
        add_wall_edges(blocked, "H", r, c)
    for r, c in v_walls:
        add_wall_edges(blocked, "V", r, c)
    return frozenset(blocked)


class State:
    __slots__ = (
        "pos", "goals", "walls_rem", "h_walls", "v_walls",
        "blocked", "to_move"
    )

    def __init__(
        self, pos, goals, walls_rem, h_walls, v_walls,
        blocked=None, to_move=0
    ):
        self.pos = (tuple(pos[0]), tuple(pos[1]))
        self.goals = tuple(goals)
        self.walls_rem = tuple(walls_rem)
        self.h_walls = frozenset(h_walls)
        self.v_walls = frozenset(v_walls)
        self.blocked = (
            frozenset(blocked)
            if blocked is not None
            else walls_to_blocked(self.h_walls, self.v_walls)
        )
        self.to_move = to_move

    def clone_with(self, **kwargs):
        return State(
            kwargs.get("pos", self.pos),
            kwargs.get("goals", self.goals),
            kwargs.get("walls_rem", self.walls_rem),
            kwargs.get("h_walls", self.h_walls),
            kwargs.get("v_walls", self.v_walls),
            kwargs.get("blocked", self.blocked),
            kwargs.get("to_move", self.to_move),
        )


# ---------------------------------------------------------------------------
# BFS utilities
# ---------------------------------------------------------------------------

def bfs_distances(start, goal_row, blocked):
    """Return distance grid and predecessor lists for all shortest paths."""
    dist = [[math.inf] * SIZE for _ in range(SIZE)]
    prev = [[[] for _ in range(SIZE)] for _ in range(SIZE)]
    q = deque([start])
    dist[start[0]][start[1]] = 0

    while q:
        r, c = q.popleft()
        nd = dist[r][c] + 1

        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                continue
            if ((r, c), (nr, nc)) in blocked:
                continue

            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                prev[nr][nc] = [(r, c)]
                q.append((nr, nc))
            elif nd == dist[nr][nc]:
                prev[nr][nc].append((r, c))

    goal_distance = min(dist[goal_row])
    return dist, prev, goal_distance


def shortest_distance(position, goal_row, blocked):
    if position[0] == goal_row:
        return 0
    _, _, d = bfs_distances(position, goal_row, blocked)
    return d


def shortest_path_next_positions(state, player):
    """All legal adjacent cells that reduce this pawn's BFS distance by 1."""
    position = state.pos[player]
    goal = state.goals[player]
    dist, _, _ = bfs_distances(position, goal, state.blocked)
    r, c = position
    result = []

    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            continue
        if ((r, c), (nr, nc)) in state.blocked:
            continue
        if dist[nr][nc] == dist[r][c] - 1:
            result.append((nr, nc))

    return result


def all_shortest_path_edges(state, player):
    """Edges belonging to at least one shortest path to the goal."""
    start = state.pos[player]
    goal = state.goals[player]
    dist, prev, best = bfs_distances(start, goal, state.blocked)
    if not math.isfinite(best):
        return set()

    goals = [(goal, c) for c in range(SIZE) if dist[goal][c] == best]
    stack = list(goals)
    visited = set()
    edges = set()

    while stack:
        cell = stack.pop()
        if cell in visited:
            continue
        visited.add(cell)
        r, c = cell
        for pr, pc in prev[r][c]:
            edges.add(((pr, pc), (r, c)))
            stack.append((pr, pc))

    return edges


# ---------------------------------------------------------------------------
# Quoridor legality and state transitions
# ---------------------------------------------------------------------------

def pawn_moves(state, player):
    r, c = state.pos[player]
    opponent = state.pos[1 - player]
    moves = set()

    for dr, dc in DIRS:
        adj = (r + dr, c + dc)
        if not (0 <= adj[0] < SIZE and 0 <= adj[1] < SIZE):
            continue
        if ((r, c), adj) in state.blocked:
            continue

        if adj != opponent:
            moves.add(adj)
            continue

        # Opponent is adjacent: try a straight jump first.
        jump = (adj[0] + dr, adj[1] + dc)
        if (
            0 <= jump[0] < SIZE and 0 <= jump[1] < SIZE
            and (adj, jump) not in state.blocked
        ):
            moves.add(jump)
        else:
            # If straight jump is blocked / off-board, allow diagonal jumps.
            for sdr, sdc in ((dc, dr), (-dc, -dr)):
                side = (adj[0] + sdr, adj[1] + sdc)
                if (
                    0 <= side[0] < SIZE and 0 <= side[1] < SIZE
                    and (adj, side) not in state.blocked
                ):
                    moves.add(side)

    return [{"type": "move", "to": [r, c]} for r, c in sorted(moves)]


def wall_does_not_overlap(state, ori, r, c):
    if not (0 <= r < 8 and 0 <= c < 8):
        return False

    if ori == "H":
        if (r, c) in state.v_walls:
            return False
        return not (
            (r, c - 1) in state.h_walls
            or (r, c) in state.h_walls
            or (r, c + 1) in state.h_walls
        )

    if (r, c) in state.h_walls:
        return False
    return not (
        (r - 1, c) in state.v_walls
        or (r, c) in state.v_walls
        or (r + 1, c) in state.v_walls
    )


def apply_action(state, action, player):
    next_player = 1 - player

    if action["type"] == "move":
        destination = tuple(action["to"])
        positions = list(state.pos)
        positions[player] = destination
        return state.clone_with(pos=tuple(positions), to_move=next_player)

    if action["type"] == "wall":
        ori = action["orientation"]
        r, c = action["row"], action["col"]
        h_walls = set(state.h_walls)
        v_walls = set(state.v_walls)
        walls_rem = list(state.walls_rem)
        walls_rem[player] -= 1

        if ori == "H":
            h_walls.add((r, c))
        else:
            v_walls.add((r, c))

        blocked = set(state.blocked)
        add_wall_edges(blocked, ori, r, c)

        return State(
            state.pos, state.goals, walls_rem,
            h_walls, v_walls, blocked, next_player
        )

    return state.clone_with(to_move=next_player)


def wall_preserves_paths(child):
    return (
        math.isfinite(shortest_distance(child.pos[0], child.goals[0], child.blocked))
        and math.isfinite(shortest_distance(child.pos[1], child.goals[1], child.blocked))
    )


def legal_wall_actions(state, player):
    if state.walls_rem[player] <= 0:
        return []

    actions = []
    for ori in ("H", "V"):
        for r in range(8):
            for c in range(8):
                if not wall_does_not_overlap(state, ori, r, c):
                    continue

                action = {
                    "type": "wall",
                    "orientation": ori,
                    "row": r,
                    "col": c,
                }
                child = apply_action(state, action, player)
                if wall_preserves_paths(child):
                    actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Probable wall selection: shortest-path interruptions + walls near pawns
# ---------------------------------------------------------------------------

def wall_interruption_candidates(state, target_player):
    """
    Approximate the JS engine's getValidNextWallsDisturbPathOf():
    mark walls that block an edge on at least one shortest path, plus walls
    adjacent to the target pawn. Legality is checked afterwards.
    """
    edges = all_shortest_path_edges(state, target_player)
    candidates = set()

    for (r1, c1), (r2, c2) in edges:
        dr, dc = r2 - r1, c2 - c1

        if dr == 1 and dc == 0:       # movement down
            for wc in (c1 - 1, c1):
                if 0 <= wc < 8 and 0 <= r1 < 8:
                    candidates.add(("H", r1, wc))
        elif dr == -1 and dc == 0:    # movement up
            for wc in (c1 - 1, c1):
                if 0 <= wc < 8 and 0 <= r2 < 8:
                    candidates.add(("H", r2, wc))
        elif dr == 0 and dc == 1:     # movement right
            for wr in (r1 - 1, r1):
                if 0 <= wr < 8 and 0 <= c1 < 8:
                    candidates.add(("V", wr, c1))
        elif dr == 0 and dc == -1:    # movement left
            for wr in (r1 - 1, r1):
                if 0 <= wr < 8 and 0 <= c2 < 8:
                    candidates.add(("V", wr, c2))

    # Add nearby wall positions around the target pawn.
    pr, pc = state.pos[target_player]
    for ori in ("H", "V"):
        for r in range(max(0, pr - 2), min(8, pr + 2)):
            for c in range(max(0, pc - 2), min(8, pc + 2)):
                candidates.add((ori, r, c))

    result = []
    for ori, r, c in candidates:
        if wall_does_not_overlap(state, ori, r, c):
            action = {
                "type": "wall",
                "orientation": ori,
                "row": r,
                "col": c,
            }
            child = apply_action(state, action, state.to_move)
            if wall_preserves_paths(child):
                result.append(action)

    return result


def probable_wall_actions(state, player):
    if state.walls_rem[player] <= 0:
        return []

    # The JS source focuses walls on the opponent's shortest paths.
    candidates = wall_interruption_candidates(state, 1 - player)

    # Rank by the increase in opponent distance minus own distance.
    own_before = shortest_distance(state.pos[player], state.goals[player], state.blocked)
    opp_before = shortest_distance(state.pos[1 - player], state.goals[1 - player], state.blocked)

    scored = []
    for action in candidates:
        child = apply_action(state, action, player)
        own_after = shortest_distance(child.pos[player], child.goals[player], child.blocked)
        opp_after = shortest_distance(child.pos[1 - player], child.goals[1 - player], child.blocked)
        gain = (opp_after - opp_before) - (own_after - own_before)
        scored.append((gain, action))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [action for _, action in scored[:MAX_WALL_CANDIDATES]]


def generate_expansion_actions(state, player):
    actions = pawn_moves(state, player)

    # Match the source's broad policy: walls are considered while the
    # opponent still has walls; when they have none, only path-interrupting
    # walls are generated.
    if state.walls_rem[player] > 0:
        actions.extend(probable_wall_actions(state, player))

    return actions


# ---------------------------------------------------------------------------
# Rollout policy
# ---------------------------------------------------------------------------

def rollout_choose_action(state):
    player = state.to_move
    forward_moves = shortest_path_next_positions(state, player)

    if random.random() < ROLLOUT_FORWARD_PROB and forward_moves:
        destination = random.choice(forward_moves)
        return {"type": "move", "to": [destination[0], destination[1]]}

    if state.walls_rem[player] > 0:
        walls = probable_wall_actions(state, player)
        if walls:
            return random.choice(walls)

    # Fallback: move away from the goal, as in the supplied JS rollout.
    moves = pawn_moves(state, player)
    if not moves:
        return {"type": "pass"}

    current_dist = shortest_distance(
        state.pos[player], state.goals[player], state.blocked
    )
    backward = []
    for action in moves:
        child = apply_action(state, action, player)
        d = shortest_distance(
            child.pos[player], child.goals[player], child.blocked
        )
        if d > current_dist:
            backward.append(action)

    if backward:
        return random.choice(backward)

    return random.choice(moves)


def terminal_winner(state):
    if state.pos[0][0] == state.goals[0]:
        return 0
    if state.pos[1][0] == state.goals[1]:
        return 1
    return None


def rollout(state, deadline):
    """Play a simulated game to completion; return winner index or None."""
    current = state
    for ply in range(MAX_ROLLOUT_PLIES):
        if terminal_winner(current) is not None:
            return terminal_winner(current)

        if time.monotonic() >= deadline:
            raise SearchTimeout

        action = rollout_choose_action(current)
        if action["type"] == "pass":
            break
        current = apply_action(current, action, current.to_move)

    # Safety evaluation if a rollout reaches the ply cap.
    d0 = shortest_distance(current.pos[0], current.goals[0], current.blocked)
    d1 = shortest_distance(current.pos[1], current.goals[1], current.blocked)
    if d0 < d1:
        return 0
    if d1 < d0:
        return 1
    return None


# ---------------------------------------------------------------------------
# MCTS tree
# ---------------------------------------------------------------------------

class Node:
    __slots__ = (
        "state", "parent", "action", "children", "visits",
        "wins", "player_just_moved", "unexpanded"
    )

    def __init__(self, state, parent=None, action=None, player_just_moved=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.player_just_moved = player_just_moved
        self.unexpanded = None

    def uct(self, exploration=UCT_C):
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return float("inf")
        return (
            self.wins / self.visits
            + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        )


class MCTS:
    def __init__(self, root_state, deadline):
        self.root = Node(root_state)
        self.deadline = deadline
        self.simulations = 0

    def select_child(self, node):
        return max(node.children, key=lambda child: child.uct())

    def run_simulation(self):
        node = self.root

        # 1. Selection
        while node.children and node.unexpanded == []:
            node = self.select_child(node)

        # 2. Expansion
        if node.unexpanded is None:
            node.unexpanded = generate_expansion_actions(
                node.state, node.state.to_move
            )

        if node.unexpanded:
            action = node.unexpanded.pop(random.randrange(len(node.unexpanded)))
            player = node.state.to_move
            child_state = apply_action(node.state, action, player)
            child = Node(
                child_state,
                parent=node,
                action=action,
                player_just_moved=player,
            )
            node.children.append(child)
            node = child

        # 3. Simulation
        winner = rollout(node.state, self.deadline)
        self.simulations += 1

        # 4. Backpropagation
        current = node
        while current is not None:
            current.visits += 1
            if current.player_just_moved is not None and winner is not None:
                if winner == current.player_just_moved:
                    current.wins += 1.0
            current = current.parent

    def search(self, max_simulations):
        while self.simulations < max_simulations:
            if time.monotonic() >= self.deadline:
                break
            self.run_simulation()

    def best_action(self):
        if not self.root.children:
            return None
        # Match the JavaScript implementation: select the most-visited root child.
        best = max(self.root.children, key=lambda child: child.visits)
        return best.action


# ---------------------------------------------------------------------------
# Arena adapter
# ---------------------------------------------------------------------------

def _observation_to_state(observation):
    you = observation["you"]
    opponent = observation["opponent"]

    h_walls = frozenset(tuple(w) for w in observation["walls"]["horizontal"])
    v_walls = frozenset(tuple(w) for w in observation["walls"]["vertical"])

    return State(
        pos=(tuple(you["position"]), tuple(opponent["position"])),
        goals=(you["goal_row"], opponent["goal_row"]),
        walls_rem=(you["walls_remaining"], opponent["walls_remaining"]),
        h_walls=h_walls,
        v_walls=v_walls,
        to_move=0,
    )


def _normalize_root_actions(observation):
    """Use arena-provided legal actions at the root."""
    actions = []

    for move in observation.get("legal_moves", []):
        actions.append({"type": "move", "to": [int(move[0]), int(move[1])]})

    for wall in observation.get("legal_walls", []):
        actions.append({
            "type": "wall",
            "orientation": wall["orientation"],
            "row": int(wall["row"]),
            "col": int(wall["col"]),
        })

    return actions


def _fallback_action(observation, root_actions):
    if not root_actions:
        return {"type": "pass"}

    moves = observation.get("legal_moves", [])
    distances = observation.get("distances")

    if moves and distances is not None:
        best_move = min(
            moves,
            key=lambda m: (
                distances[m[0]][m[1]]
                if distances[m[0]][m[1]] >= 0 else 10**6
            )
        )
        return {"type": "move", "to": [int(best_move[0]), int(best_move[1])]}

    return root_actions[0]


def choose_action(observation):
    """
    Wrongway Arena entry point.

    Expected observation fields:
      you.position, you.goal_row, you.walls_remaining
      opponent.position, opponent.goal_row, opponent.walls_remaining
      walls.horizontal, walls.vertical
      legal_moves, legal_walls
      distances (used only for fallback, if available)
    """
    start = time.monotonic()
    deadline = start + TIME_LIMIT

    root_actions = _normalize_root_actions(observation)
    if not root_actions:
        return {"type": "pass"}

    # Immediate win.
    goal_row = observation["you"]["goal_row"]
    for move in observation.get("legal_moves", []):
        if move[0] == goal_row:
            return {"type": "move", "to": [int(move[0]), int(move[1])]}

    fallback = _fallback_action(observation, root_actions)

    try:
        state = _observation_to_state(observation)
        mcts = MCTS(state, deadline)
        # At the root, trust the arena's legal-action list exactly.
        # Deeper nodes use the translated internal move/wall generator.
        mcts.root.unexpanded = list(root_actions)

        # The supplied JS code runs a fixed number of simulations.
        # In this Python arena adaptation, the wall-clock deadline is primary.
        mcts.search(MAX_SIMULATIONS)

        action = mcts.best_action()
        if action is not None:
            # Guard against returning a root action not accepted by the arena.
            if action in root_actions:
                return action

            # If internal candidate generation differs from arena legality,
            # choose the most visited legal root action instead.
            legal_children = [
                child for child in mcts.root.children
                if child.action in root_actions
            ]
            if legal_children:
                return max(legal_children, key=lambda child: child.visits).action

    except (SearchTimeout, RecursionError):
        pass

    return fallback


# Common aliases used by some arena runners.
def agent(observation):
    return choose_action(observation)


if __name__ == "__main__":
    # Minimal smoke test with a representative initial observation.
    sample = {
        "you": {
            "position": [8, 4],
            "goal_row": 0,
            "walls_remaining": 10,
        },
        "opponent": {
            "position": [0, 4],
            "goal_row": 8,
            "walls_remaining": 10,
        },
        "walls": {"horizontal": [], "vertical": []},
        "legal_moves": [[7, 4], [8, 3], [8, 5]],
        "legal_walls": [],
        "distances": [[0] * 9 for _ in range(9)],
        "turn": 0,
    }
    print(choose_action(sample))

from typing import Any, Dict
class QuoridorMCTSAgent:

    def __init__(self):
       pass

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return choose_action(obs)