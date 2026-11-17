"""Wrongway baseline: shortest-path BFS + tactical wall heuristic.

Independent submission-friendly Python file (standard library only).
- Move using the Arena-provided legal moves and shortest-path map.
- Compute the opponent's shortest path with our own BFS.
- Spend a wall only when it materially increases the opponent's route.
- Handle immediate wins and immediate opponent threats.

Coordinates remain ABSOLUTE for both player 0 and player 1.
This is a non-trained baseline for later Minimax/MCTS + neural nets.
"""

from collections import deque

_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _add_wall(blocked, orientation, row, col):
    """Insert wall's two impassable edges, in both directions."""
    if orientation == "H":
        edges = (
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        )
    else:  # V
        edges = (
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        )
    for a, b in edges:
        blocked.add((a, b))
        blocked.add((b, a))


def _blocked_edges(walls):
    blocked = set()
    for row, col in walls["horizontal"]:
        _add_wall(blocked, "H", row, col)
    for row, col in walls["vertical"]:
        _add_wall(blocked, "V", row, col)
    return blocked


def _distance_map(goal_row, size, blocked):
    """Multi-source BFS from entire goal row, ignoring pawn occupancy."""
    dist = [[-1] * size for _ in range(size)]
    q = deque()
    for col in range(size):
        dist[goal_row][col] = 0
        q.append((goal_row, col))

    while q:
        row, col = q.popleft()
        next_distance = dist[row][col] + 1
        for dr, dc in _DIRECTIONS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if dist[nr][nc] >= 0 or ((row, col), (nr, nc)) in blocked:
                continue
            dist[nr][nc] = next_distance
            q.append((nr, nc))
    return dist


def _legal_moves_for(position, opponent_position, size, blocked):
    """Pawn move generator for emergency defense; mirrors Arena jump rules."""
    row, col = position
    other = tuple(opponent_position)
    here = (row, col)
    destinations = set()

    def edge_open(a, b):
        return (0 <= b[0] < size and 0 <= b[1] < size
                and (a, b) not in blocked)

    for dr, dc in _DIRECTIONS:
        adjacent = (row + dr, col + dc)
        if not edge_open(here, adjacent):
            continue
        if adjacent != other:
            destinations.add(adjacent)
            continue
        beyond = (adjacent[0] + dr, adjacent[1] + dc)
        if edge_open(adjacent, beyond):
            destinations.add(beyond)
        else:
            for sr, sc in ((dc, dr), (-dc, -dr)):
                beside = (adjacent[0] + sr, adjacent[1] + sc)
                if edge_open(adjacent, beside):
                    destinations.add(beside)
    return destinations


def choose_action(observation):
    """Return exactly one Arena-compatible action; no game-state mutation."""
    moves = observation["legal_moves"]
    if not moves:
        return {"type": "pass"}

    you = observation["you"]
    opp = observation["opponent"]
    own_position = you["position"]
    opp_position = opp["position"]
    own_goal, opp_goal = you["goal_row"], opp["goal_row"]
    size = observation["board_size"]

    # Rule 1: finish the game immediately if possible.
    for move in moves:
        if move[0] == own_goal:
            return {"type": "move", "to": list(move)}

    # Rule 2: select the best legal pawn move using Arena's BFS distances.
    own_map = observation["distances"]
    own_distance = own_map[own_position[0]][own_position[1]]
    best_move = min(
        moves,
        key=lambda m: (
            own_map[m[0]][m[1]] if own_map[m[0]][m[1]] >= 0 else 999,
            abs(m[1] - (size // 2)),  # deterministic tie-break, prefer center
            m[1],
        ),
    )
    moving_action = {"type": "move", "to": list(best_move)}

    legal_walls = observation["legal_walls"]
    if not legal_walls:
        return moving_action

    blocked = _blocked_edges(observation["walls"])
    opponent_map = _distance_map(opp_goal, size, blocked)
    opp_distance = opponent_map[opp_position[0]][opp_position[1]]

    # Search all legal candidate walls; they're pre-validated by the Arena.
    # A wall costs one move, so distance gained must outweigh lost tempo.
    current_threat = any(
        cell[0] == opp_goal
        for cell in _legal_moves_for(opp_position, own_position, size, blocked)
    )

    # No wall is worth the tempo if safely far ahead (unless opponent wins now).
    if not current_threat and opp_distance > own_distance + 2:
        return moving_action

    best_wall = None
    best_gain = -999
    best_enemy_extension = -999
    blocks_immediate_win = False
    best_priority = None

    for wall in legal_walls:
        candidate = blocked.copy()
        _add_wall(candidate, wall["orientation"], wall["row"], wall["col"])

        # Defense is urgent: test whether wall stops all immediate goal moves.
        blocks_win = False
        if current_threat:
            can_still_win = any(
                m[0] == opp_goal
                for m in _legal_moves_for(opp_position, own_position, size, candidate)
            )
            blocks_win = not can_still_win

        new_opp_map = _distance_map(opp_goal, size, candidate)
        new_opp_distance = new_opp_map[opp_position[0]][opp_position[1]]
        enemy_extension = new_opp_distance - opp_distance

        # A wall not delaying the opponent cannot improve our path.
        if enemy_extension <= 0 and not blocks_win:
            continue

        new_own_map = _distance_map(own_goal, size, candidate)
        new_own_distance = new_own_map[own_position[0]][own_position[1]]
        own_extension = new_own_distance - own_distance
        gain = enemy_extension - own_extension

        priority = (1 if blocks_win else 0, gain, enemy_extension, -own_extension)
        if best_priority is None or priority > best_priority:
            best_priority = priority
            best_wall = wall
            best_gain = gain
            best_enemy_extension = enemy_extension
            blocks_immediate_win = blocks_win

    # Emergency: do not let the opponent reach the goal in their next turn.
    if current_threat and blocks_immediate_win:
        return {"type": "wall", **best_wall}

    # Ordinarily, the wall must give at least 2 net steps of path advantage.
    # Spending the wall consumes one full move, so +1 is not enough.
    if best_wall is not None and best_gain >= 2:
        return {"type": "wall", **best_wall}

    return moving_action
