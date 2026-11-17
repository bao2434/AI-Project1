from collections import deque
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

BOARD_SIZE = 9
INFINITY = 100_000
BOT_WIN = 100_000
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

WEIGHTS = [42.8, 100.0, 141.2, 30.4, 646.2, -21.9]
WALL_BIAS = 20.0
FORWARD_DRIVE = 40.0
BACKTRACK_PENALTY = 60.0

def _inside(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _edge_open(r1: int, c1: int, r2: int, c2: int,
               h_walls: Set[Tuple[int, int]], v_walls: Set[Tuple[int, int]]) -> bool:
    if not _inside(r1, c1) or not _inside(r2, c2):
        return False
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return False
    if r1 != r2:
        edge_row = min(r1, r2)
        return (edge_row, c1) not in h_walls and (edge_row, c1 - 1) not in h_walls
    else:
        edge_col = min(c1, c2)
        return (r1, edge_col) not in v_walls and (r1 - 1, edge_col) not in v_walls


def _compute_distance_map(goal_row: int,
                          h_walls: Set[Tuple[int, int]],
                          v_walls: Set[Tuple[int, int]]) -> List[List[int]]:
    dist = [[999] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    q = deque()
    for c in range(BOARD_SIZE):
        dist[goal_row][c] = 0
        q.append((goal_row, c))

    while q:
        r, c = q.popleft()
        d = dist[r][c] + 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if _inside(nr, nc) and dist[nr][nc] == 999:
                if _edge_open(r, c, nr, nc, h_walls, v_walls):
                    dist[nr][nc] = d
                    q.append((nr, nc))
    return dist

