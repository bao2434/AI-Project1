
from collections import deque


def add_wall_edges(blocked, orientation, row, col):
    """Add blocked edges in both directions."""
    if orientation == "H":
        pairs = [
            ((row, col), (row + 1, col)),
            ((row, col + 1), (row + 1, col + 1)),
        ]
    else:
        pairs = [
            ((row, col), (row, col + 1)),
            ((row + 1, col), (row + 1, col + 1)),
        ]

    for a, b in pairs:
        blocked.add((a, b))
        blocked.add((b, a))


def blocked_edges(walls):
    blocked = set()

    for row, col in walls["horizontal"]:
        add_wall_edges(blocked, "H", row, col)

    for row, col in walls["vertical"]:
        add_wall_edges(blocked, "V", row, col)

    return blocked


def distance_to_goal(position, goal_row, size, blocked):
    """Return shortest path length using BFS."""
    start = tuple(position)

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, distance = queue.popleft()
        row, col = current

        if row == goal_row:
            return distance

        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor

            if not (0 <= nr < size and 0 <= nc < size):
                continue

            if neighbor in visited:
                continue

            if (current, neighbor) in blocked:
                continue

            visited.add(neighbor)
            queue.append((neighbor, distance + 1))

    return size * size


def evaluate_wall(
    wall,
    blocked,
    you,
    opponent,
    size,
    own_distance,
    enemy_distance,
):
    """Evaluate relative path-length increase after placing a wall."""
    candidate = blocked.copy()

    add_wall_edges(
        candidate,
        wall["orientation"],
        wall["row"],
        wall["col"],
    )

    own_after = distance_to_goal(
        you["position"],
        you["goal_row"],
        size,
        candidate,
    )

    enemy_after = distance_to_goal(
        opponent["position"],
        opponent["goal_row"],
        size,
        candidate,
    )

    # A wall must preserve a path for both players.
    if own_after >= size * size or enemy_after >= size * size:
        return float("-inf")

    own_increase = own_after - own_distance
    enemy_increase = enemy_after - enemy_distance

    return enemy_increase - own_increase


def choose_action(observation):
    moves = observation["legal_moves"]
    walls = observation["legal_walls"]

    you = observation["you"]
    opponent = observation["opponent"]

    size = observation["board_size"]

    # 1. Immediate winning move.
    for move in moves:
        if move[0] == you["goal_row"]:
            return {
                "type": "move",
                "to": list(move),
            }

    # 2. Calculate current shortest paths.
    blocked = blocked_edges(observation["walls"])

    own_distance = distance_to_goal(
        you["position"],
        you["goal_row"],
        size,
        blocked,
    )

    enemy_distance = distance_to_goal(
        opponent["position"],
        opponent["goal_row"],
        size,
        blocked,
    )

    # 3. Find the best pawn move.
    distances = observation["distances"]

    best_move = min(
        moves,
        key=lambda move: (
            distances[move[0]][move[1]]
            if distances[move[0]][move[1]] >= 0
            else float("inf")
        ),
        default=None,
    )

    # 4. Evaluate walls only when the opponent is close.
    best_wall = None
    best_gain = 0

    if enemy_distance <= own_distance + 1:
        for wall in walls:
            gain = evaluate_wall(
                wall,
                blocked,
                you,
                opponent,
                size,
                own_distance,
                enemy_distance,
            )

            if gain > best_gain:
                best_gain = gain
                best_wall = wall

    # 5. Choose wall if it provides a positive advantage.
    if best_wall is not None:
        return {
            "type": "wall",
            "orientation": best_wall["orientation"],
            "row": best_wall["row"],
            "col": best_wall["col"],
        }

    # 6. Otherwise move.
    if best_move is not None:
        return {
            "type": "move",
            "to": list(best_move),
        }

    return {"type": "pass"}
