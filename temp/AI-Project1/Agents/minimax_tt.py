"""Wrongway Minimax V3: path-aware wall filtering, unified ordering, and transposition table."""
from collections import deque
import time
# ==============================================================================
# EXPLICIT TUNABLE PARAMETERS
# ==============================================================================
TIME_LIMIT = 0.75            # Max search time per turn in seconds (Arena limit is 1.0s)
MAX_DEPTH = 50               # Iterative deepening maximum depth
W_DIST = 10.0                # Weight for (opponent_distance - my_distance)
W_WALL = 2.0                 # Weight for (my_walls_remaining - opponent_walls_remaining)
W_PROG = 0.5                 # Weight for (my_progress - opponent_progress)
MAX_WALL_BRANCHES = 18       # Keep top K walls after move ordering so Alpha-Beta searches deeper
REPEAT_PENALTY = 8.0         # Penalty for revisiting recent cells (prevents oscillation)
TT_ENABLED = True             # Toggle for ablation: True vs False
TT_MAX_ENTRIES = 50000        # Per-decision bounded transposition table
# ==============================================================================
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_WALLS = tuple((ori, r, c) for ori in ("H", "V") for r in range(8) for c in range(8))
RECENT_POSITIONS = deque(maxlen=10)


# ==============================================================================
# MINIMAX SEARCH STATISTICS (added)
# ==============================================================================
SEARCH_STATS = {}
SEARCH_DEPTH_STATS = {}
SEARCH_ROOT_DEPTH = 0
SEARCH_SUMMARY_FILE = "minimax_search_summary.txt"
# Keys contain positions, BOTH goals, wall inventories, exact walls, and side to play.
# The table is reset before each decision so independent matches do not leak state.
TRANSPOSITION_TABLE = {}
TT_EXACT, TT_LOWER, TT_UPPER = "EXACT", "LOWER", "UPPER"


def reset_search_stats():
    global SEARCH_STATS, SEARCH_DEPTH_STATS
    TRANSPOSITION_TABLE.clear()
    SEARCH_STATS = {
        "nodes": 0, "internal_nodes": 0, "leaf_nodes": 0,
        "terminal_nodes": 0, "heuristic_evaluations": 0, "bfs_calls": 0,
        "generated_pawn_moves": 0, "wall_candidates_examined": 0, "wall_candidates_preselected": 0,
        "legal_wall_candidates": 0, "wall_actions_kept": 0,
        "actions_searched": 0, "alpha_beta_cutoffs": 0,
        "max_ply_reached": 0, "completed_depth": 0, "timed_out": False,
        "tt_hits": 0, "tt_exact_hits": 0, "tt_bound_cutoffs": 0,
        "tt_stores": 0, "tt_move_first": 0,
    }
    SEARCH_DEPTH_STATS = {}


def count_stat(name, amount=1):
    SEARCH_STATS[name] = SEARCH_STATS.get(name, 0) + amount


def record_depth_stat(depth, name, amount=1):
    row = SEARCH_DEPTH_STATS.setdefault(depth, {})
    row[name] = row.get(name, 0) + amount


def write_search_summary(turn, elapsed):
    lines = [
        f"\n=== TURN {turn} MINIMAX SEARCH SUMMARY ===",
        f"Elapsed search time: {elapsed:.6f} seconds",
        f"Total nodes visited: {SEARCH_STATS['nodes']}",
        f"Internal nodes: {SEARCH_STATS['internal_nodes']}",
        f"Leaf nodes (depth limit): {SEARCH_STATS['leaf_nodes']}",
        f"Terminal nodes: {SEARCH_STATS['terminal_nodes']}",
        f"Heuristic evaluations: {SEARCH_STATS['heuristic_evaluations']}",
        f"BFS calls: {SEARCH_STATS['bfs_calls']}",
        f"Pawn moves generated: {SEARCH_STATS['generated_pawn_moves']}",
        f"Wall candidates examined: {SEARCH_STATS['wall_candidates_examined']}",
        f"Wall candidates preselected: {SEARCH_STATS['wall_candidates_preselected']}",
        f"Legal wall candidates: {SEARCH_STATS['legal_wall_candidates']}",
        f"Wall actions kept after ordering: {SEARCH_STATS['wall_actions_kept']}",
        f"Actions searched: {SEARCH_STATS['actions_searched']}",
        f"Average actions searched per internal node: {SEARCH_STATS['actions_searched'] / max(1, SEARCH_STATS['internal_nodes']):.2f}",
        f"Alpha-Beta cutoff rate: {SEARCH_STATS['alpha_beta_cutoffs'] / max(1, SEARCH_STATS['internal_nodes']):.2%}",
        f"Alpha-Beta cutoffs: {SEARCH_STATS['alpha_beta_cutoffs']}",
        f"Maximum ply reached: {SEARCH_STATS['max_ply_reached']}",
        f"Deepest fully completed iteration: {SEARCH_STATS['completed_depth']}",
        f"Timed out during search: {SEARCH_STATS['timed_out']}",
        f"TT hits / exact hits: {SEARCH_STATS['tt_hits']} / {SEARCH_STATS['tt_exact_hits']}",
        f"TT bound cutoffs / stored entries: {SEARCH_STATS['tt_bound_cutoffs']} / {SEARCH_STATS['tt_stores']}",
        f"TT entries currently held: {len(TRANSPOSITION_TABLE)}",
        "", "Per-depth statistics (depth: nodes / leaves / cutoffs):",
    ]
    for depth in sorted(SEARCH_DEPTH_STATS, reverse=True):
        row = SEARCH_DEPTH_STATS[depth]
        lines.append(
            f"  {depth}: {row.get('nodes', 0)} / "
            f"{row.get('leaves', 0)} / {row.get('cutoffs', 0)}"
        )
    try:
        with open(SEARCH_SUMMARY_FILE, "a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
    except OSError:
        pass

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
class State:
    """Compact board state for Minimax tree search."""
    __slots__ = ("pos", "goals", "walls_rem", "h_walls", "v_walls", "blocked")
    def __init__(self, pos, goals, walls_rem, h_walls, v_walls, blocked=None):
        self.pos = pos              # ((my_r, my_c), (opp_r, opp_c))
        self.goals = goals          # (my_goal_row, opp_goal_row)
        self.walls_rem = walls_rem  # (my_walls, opp_walls)
        self.h_walls = h_walls      # frozenset of (r, c)
        self.v_walls = v_walls      # frozenset of (r, c)
        self.blocked = blocked if blocked is not None else _walls_to_blocked(h_walls, v_walls)
def bfs(position, goal_row, blocked):
    count_stat("bfs_calls")
    """BFS shortest path from position to goal_row (ignores pawns)."""
    if position[0] == goal_row:
        return 0
    q = deque([(position, 0)])
    visited = {position}
    while q:
        (r, c), dist = q.popleft()
        nd = dist + 1
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                nxt = (nr, nc)
                if nxt not in visited and ((r, c), nxt) not in blocked:
                    if nr == goal_row:
                        return nd
                    visited.add(nxt)
                    q.append((nxt, nd))
    return float("inf")
def terminal(state):
    return state.pos[0][0] == state.goals[0] or state.pos[1][0] == state.goals[1]
def terminal_score(state, depth):
    # Prefer faster wins and slower losses via depth bonus
    if state.pos[0][0] == state.goals[0]:
        return 10000.0 + depth * 10.0
    if state.pos[1][0] == state.goals[1]:
        return -10000.0 - depth * 10.0
    return 0.0
def heuristic(state):
    count_stat("heuristic_evaluations")
    """HEURISTIC(state): 10*(opp_dist - my_dist) + 2*wall_adv + 0.5*progress_adv."""
    my_distance = bfs(state.pos[0], state.goals[0], state.blocked)
    opponent_distance = bfs(state.pos[1], state.goals[1], state.blocked)
    wall_advantage = state.walls_rem[0] - state.walls_rem[1]
    my_progress = 8 - abs(state.pos[0][0] - state.goals[0])
    opponent_progress = 8 - abs(state.pos[1][0] - state.goals[1])
    progress_advantage = my_progress - opponent_progress
    return (
        W_DIST * (opponent_distance - my_distance)
        + W_WALL * wall_advantage
        + W_PROG * progress_advantage
    )
def all_legal_pawn_moves(state, player):
    """Generate legal pawn moves (player=0 for MAX, player=1 for MIN), including jumps."""
    r, c = state.pos[player]
    opp = state.pos[1 - player]
    blocked = state.blocked
    dests = set()
    for dr, dc in DIRS:
        adj = (r + dr, c + dc)
        if not (0 <= adj[0] < 9 and 0 <= adj[1] < 9) or ((r, c), adj) in blocked:
            continue
        if adj != opp:
            dests.add(adj)
        else:
            jmp = (adj[0] + dr, adj[1] + dc)
            if 0 <= jmp[0] < 9 and 0 <= jmp[1] < 9 and (adj, jmp) not in blocked:
                dests.add(jmp)
            else:
                for sdr, sdc in ((dc, dr), (-dc, -dr)):
                    side = (adj[0] + sdr, adj[1] + sdc)
                    if 0 <= side[0] < 9 and 0 <= side[1] < 9 and (adj, side) not in blocked:
                        dests.add(side)
    return [{"type": "move", "to": [d[0], d[1]]} for d in sorted(dests)]
def _wall_does_not_overlap(state, ori, r, c):
    if ori == "H":
        if (r, c) in state.v_walls:
            return False
        return not ((r, c - 1) in state.h_walls or (r, c) in state.h_walls or (r, c + 1) in state.h_walls)
    if (r, c) in state.h_walls:
        return False
    return not ((r - 1, c) in state.v_walls or (r, c) in state.v_walls or (r + 1, c) in state.v_walls)
def apply_action(state, action, player):
    """Return new child State after player applies action."""
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
# PATH-AWARE WALL PRESELECTION (pawn-agnostic BFS, as required by Arena)
# ==============================================================================
MAX_WALL_PRESELECT = 36  # Upper bound on walls requiring two expensive legality BFS calls


def _distance_map(starts, blocked):
    """Return distance to each board cell from one/multiple sources."""
    count_stat("bfs_calls")
    distance = {}
    queue = deque()
    for start in starts:
        start = tuple(start)
        if start not in distance:
            distance[start] = 0
            queue.append(start)
    while queue:
        u = queue.popleft()
        r, c = u
        next_distance = distance[u] + 1
        for dr, dc in DIRS:
            v = (r + dr, c + dc)
            if (0 <= v[0] < 9 and 0 <= v[1] < 9
                    and v not in distance and (u, v) not in blocked):
                distance[v] = next_distance
                queue.append(v)
    return distance


def _shortest_path_edges(position, goal_row, blocked):
    """Undirected edges on ANY shortest route to the goal (not one arbitrary route).

    This is a selection heuristic, not the wall legality test. It ignores pawns
    just like Arena's path-preservation BFS.
    """
    from_start = _distance_map((position,), blocked)
    from_goal = _distance_map(((goal_row, col) for col in range(9)), blocked)
    optimum = from_goal.get(tuple(position))
    if optimum is None:
        return set()
    edges = set()
    for u, distance in from_start.items():
        if distance >= optimum:
            continue
        r, c = u
        for dr, dc in DIRS:
            v = (r + dr, c + dc)
            if (u, v) not in blocked and (distance + 1 + from_goal.get(v, 9999) == optimum):
                # Pair is direction independent, since a wall blocks both directions.
                edges.add(frozenset((u, v)))
    return edges


def _wall_edge_pairs(ori, row, col):
    if ori == "H":
        return (((row, col), (row + 1, col)),
                ((row, col + 1), (row + 1, col + 1)))
    return (((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)))


def _wall_pawn_distance(row, col, pawn):
    """Distance from pawn to nearest square of the wall's 2x2 footprint."""
    pr, pc = pawn
    nearest_r = min(max(pr, row), row + 1)
    nearest_c = min(max(pc, col), col + 1)
    return abs(pr - nearest_r) + abs(pc - nearest_c)


def _preselect_wall_candidates(state, player, deadline):
    """Cheap geometry/path screen BEFORE hypothetical-wall BFS calls.

    Retain walls on any opponent shortest route, near either pawn, or on the
    current player's shortest route (for optional defensive tactics). Do not
    claim this is a complete game-theoretic pruning rule.
    """
    opponent_edges = _shortest_path_edges(
        state.pos[1 - player], state.goals[1 - player], state.blocked
    )
    self_edges = _shortest_path_edges(
        state.pos[player], state.goals[player], state.blocked
    )
    ranked = []
    for ori, row, col in ALL_WALLS:
        if time.monotonic() >= deadline:
            raise SearchTimeout
        count_stat("wall_candidates_examined")
        if not _wall_does_not_overlap(state, ori, row, col):
            continue
        edges = _wall_edge_pairs(ori, row, col)
        opponent_hit = sum(frozenset(pair) in opponent_edges for pair in edges)
        self_hit = sum(frozenset(pair) in self_edges for pair in edges)
        near_opp = _wall_pawn_distance(row, col, state.pos[1 - player])
        near_self = _wall_pawn_distance(row, col, state.pos[player])
        # Include promising immediate path blockers and locally tactical walls.
        if not (opponent_hit or self_hit or near_opp <= 2 or near_self <= 2):
            continue
        # Cheap static ordering. Full path impact is computed AFTER preselection.
        priority = (
            100 * opponent_hit - 55 * self_hit
            + 7 * max(0, 4 - near_opp)
            + 2 * max(0, 3 - near_self)
        )
        ranked.append((priority, ori, row, col))
    ranked.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    selected = ranked[:MAX_WALL_PRESELECT]
    count_stat("wall_candidates_preselected", len(selected))
    return [(ori, row, col) for _, ori, row, col in selected]


def generate_and_order_actions(state, player, deadline):
    """V2: path-aware wall preselection + ONE unified move/wall ordering.

    Candidate walls are first bounded to MAX_WALL_BRANCHES by their precise
    post-wall heuristic; then those walls and ALL pawn moves are sorted together.
    Immediate winning pawn moves always outrank non-terminal actions.
    """
    if time.monotonic() >= deadline:
        raise SearchTimeout
    scored_moves = []
    for mv in all_legal_pawn_moves(state, player):
        count_stat("generated_pawn_moves")
        child = apply_action(state, mv, player)
        value = terminal_score(child, 0) if terminal(child) else heuristic(child)
        scored_moves.append((value, mv))
    scored_walls = []
    if state.walls_rem[player] > 0:
        for ori, r, c in _preselect_wall_candidates(state, player, deadline):
            if time.monotonic() >= deadline:
                raise SearchTimeout
            w_act = {"type": "wall", "orientation": ori, "row": r, "col": c}
            child = apply_action(state, w_act, player)
            # The real legality check: neither player may lose every goal path.
            d0 = bfs(child.pos[0], child.goals[0], child.blocked)
            if d0 == float("inf"):
                continue
            d1 = bfs(child.pos[1], child.goals[1], child.blocked)
            if d1 == float("inf"):
                continue
            wall_adv = child.walls_rem[0] - child.walls_rem[1]
            prog_adv = (8 - abs(child.pos[0][0] - child.goals[0])) - (8 - abs(child.pos[1][0] - child.goals[1]))
            value = W_DIST * (d1 - d0) + W_WALL * wall_adv + W_PROG * prog_adv
            count_stat("legal_wall_candidates")
            scored_walls.append((value, w_act))
    reverse = (player == 0)
    scored_walls.sort(key=lambda x: x[0], reverse=reverse)
    best_walls = scored_walls[:MAX_WALL_BRANCHES]
    count_stat("wall_actions_kept", len(best_walls))
    # Fix legacy bias: no longer force every pawn move before every wall.
    combined = scored_moves + best_walls
    combined.sort(key=lambda x: x[0], reverse=reverse)
    return [action for _, action in combined] or [{"type": "pass"}]


def _tt_key(state, player):
    """Full immutable state identity. The player is the LOCAL MAX/MIN turn.

    A state's value is defined from the current choose_action root's POV.
    Never retain this table across decisions/agent-side changes.
    """
    return (state.pos, state.goals, state.walls_rem,
            state.h_walls, state.v_walls, player)


def _tt_save(key, depth, value, flag, best_action):
    if not TT_ENABLED or key is None:
        return
    prev = TRANSPOSITION_TABLE.get(key)
    # Preserve deeper entries as ordering hints, but do not reuse their scores
    # at other depths: terminal_score uses remaining depth as a bonus.
    if prev is not None and prev[0] > depth:
        return
    if prev is None and len(TRANSPOSITION_TABLE) >= TT_MAX_ENTRIES:
        return
    TRANSPOSITION_TABLE[key] = (depth, value, flag, best_action)
    count_stat("tt_stores")


def minimax(state, depth, alpha, beta, player, deadline, is_root=False, recent_counts=None):
    """Alpha-Beta with a bounded per-decision TT and safe depth-aware bounds.

    Root value is NOT cached: its optional repeat penalty is history dependent.
    A cached value is only reusable at exactly the same remaining depth:
    the original terminal_score has a remaining-depth win/loss bonus.
    Other-depth entries may still supply the best action as an ordering hint.
    """
    if time.monotonic() >= deadline:
        raise SearchTimeout
    count_stat("nodes")
    record_depth_stat(depth, "nodes")
    ply = SEARCH_ROOT_DEPTH - depth
    SEARCH_STATS["max_ply_reached"] = max(SEARCH_STATS["max_ply_reached"], ply)
    if terminal(state):
        count_stat("terminal_nodes")
        record_depth_stat(depth, "leaves")
        return terminal_score(state, depth), None
    if depth == 0:
        count_stat("leaf_nodes")
        record_depth_stat(depth, "leaves")
        return heuristic(state), None

    original_alpha, original_beta = alpha, beta
    key = _tt_key(state, player) if TT_ENABLED and not is_root else None
    entry = TRANSPOSITION_TABLE.get(key) if key is not None else None
    pv_action = None
    if entry is not None:
        count_stat("tt_hits")
        cached_depth, cached_value, cached_flag, pv_action = entry
        if cached_depth == depth:
            if cached_flag == TT_EXACT:
                count_stat("tt_exact_hits")
                return cached_value, pv_action
            if cached_flag == TT_LOWER:
                alpha = max(alpha, cached_value)
            elif cached_flag == TT_UPPER:
                beta = min(beta, cached_value)
            if alpha >= beta:
                count_stat("tt_bound_cutoffs")
                return cached_value, pv_action

    count_stat("internal_nodes")
    actions = generate_and_order_actions(state, player, deadline)
    # Only promote a hint if it survives the normal legal candidate filter.
    if pv_action is not None and pv_action in actions:
        actions.remove(pv_action)
        actions.insert(0, pv_action)
        count_stat("tt_move_first")
    best_action = None
    if player == 0:  # MAX
        best_score = -float("inf")
        for action in actions:
            count_stat("actions_searched")
            child = apply_action(state, action, 0)
            score, _ = minimax(child, depth - 1, alpha, beta, 1, deadline)
            if is_root and recent_counts and action["type"] == "move":
                score -= REPEAT_PENALTY * recent_counts.get(tuple(action["to"]), 0)
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, best_score)
            if alpha >= beta:
                count_stat("alpha_beta_cutoffs")
                record_depth_stat(depth, "cutoffs")
                break
    else:  # MIN
        best_score = float("inf")
        for action in actions:
            count_stat("actions_searched")
            child = apply_action(state, action, 1)
            score, _ = minimax(child, depth - 1, alpha, beta, 0, deadline)
            if score < best_score:
                best_score, best_action = score, action
            beta = min(beta, best_score)
            if alpha >= beta:
                count_stat("alpha_beta_cutoffs")
                record_depth_stat(depth, "cutoffs")
                break

    if key is not None:
        flag = (TT_UPPER if best_score <= original_alpha else
                TT_LOWER if best_score >= original_beta else TT_EXACT)
        _tt_save(key, depth, best_score, flag, best_action)
    return best_score, best_action


def choose_action(observation):
    """Arena entry point: Iterative Deepening Alpha-Beta Minimax."""
    start_time = time.monotonic()
    deadline = start_time + TIME_LIMIT
    reset_search_stats()  # Also reset TT on immediate-win decisions.
    legal_moves = observation.get("legal_moves", [])
    if not legal_moves:
        return {"type": "pass"}
    if observation.get("turn", 0) <= 1:
        RECENT_POSITIONS.clear()
    you = observation["you"]
    opp = observation["opponent"]
    my_pos = tuple(you["position"])
    opp_pos = tuple(opp["position"])
    RECENT_POSITIONS.append(my_pos)
    recent_counts = {pos: RECENT_POSITIONS.count(pos) for pos in RECENT_POSITIONS}
    # Immediate 1-step win check
    for m in legal_moves:
        if m[0] == you["goal_row"]:
            return {"type": "move", "to": list(m)}
    h_walls = frozenset((w[0], w[1]) for w in observation["walls"]["horizontal"])
    v_walls = frozenset((w[0], w[1]) for w in observation["walls"]["vertical"])
    state = State(
        pos=(my_pos, opp_pos),
        goals=(you["goal_row"], opp["goal_row"]),
        walls_rem=(you["walls_remaining"], opp["walls_remaining"]),
        h_walls=h_walls,
        v_walls=v_walls,
    )
    # Safe fallback: best 1-step pawn move by shortest path
    distances = observation["distances"]
    best_action = {
        "type": "move",
        "to": list(min(legal_moves, key=lambda m: distances[m[0]][m[1]] if distances[m[0]][m[1]] >= 0 else 999)),
    }
    # Iterative Deepening loop
    global SEARCH_ROOT_DEPTH
    search_start = time.monotonic()
    # if observation.get("turn", 0) <= 1:
    #     try:
    #         with open(SEARCH_SUMMARY_FILE, "w", encoding="utf-8") as file:
    #             file.write("MINIMAX SEARCH STATISTICS\n")
    #     except OSError:
    #         pass
    for depth in range(1, MAX_DEPTH + 1):
        SEARCH_ROOT_DEPTH = depth
        try:
            _, action = minimax(
                state, depth, -float("inf"), float("inf"), 0, deadline,
                is_root=True, recent_counts=recent_counts,
            )
            if action is not None:
                best_action = action
            SEARCH_STATS["completed_depth"] = depth
        except SearchTimeout:
            SEARCH_STATS["timed_out"] = True
            break
    # search_elapsed = time.monotonic() - search_start
    # write_search_summary(observation.get("turn", 0), search_elapsed)
    return best_action
