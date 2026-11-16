
"""Hard Waller Agent (ported from ai.js `aiHard` + `aiDuelHard` for Wrongway Arena)."""
from collections import deque
import time

# ==============================================================================
# EXPLICIT TUNABLE PARAMETERS (Ported from aiHard / aiDuelHard in ai.js)
# ==============================================================================
TIME_LIMIT = 0.70            # Time budget in seconds (aiDuelHard uses 700ms)
SEARCH_DEPTH = 3             # Root lookahead depth (aiDuelHard depth 3 / aiHard depth 2-3)
ROOT_MAX_WALLS = 10          # Max candidate walls evaluated at root (aiHard uses 10)
INNER_MAX_WALLS_D2 = 6       # Max candidate walls at depth >= 2 (aiHard uses 6)
INNER_MAX_WALLS_D1 = 4       # Max candidate walls at depth < 2 (aiHard uses 4)
WALL_CANDIDATE_DEPTH = 6     # Look ahead N steps along opponent's BFS path for wall candidates

# Heuristic & Root Tie-Breaker Weights (from aiHard / aiDuelHard)
W_REL_DIST = 25.0            # (humD - aiD) * 25
W_AI_DIST = 3.5              # - aiD * 3.5
W_HUM_DIST = 0.8             # + humD * 0.8
W_BARR_DIFF = 3.0            # + (ab - hb) * 3
W_PATH_FOLLOW = 20.0         # Bonus for following BFS shortest path step 1
W_FWD_STEP = 6.0             # Bonus for moving forward toward goal_row
W_BACK_STEP = -4.0           # Penalty for moving backward away from goal_row
W_ROBUST_PENALTY = 5.0       # Penalty for leaving own path vulnerable to single wall
# ==============================================================================

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
RECENT_POSITIONS = deque(maxlen=6)


def _walls_to_blocked(h_walls, v_walls):
    """Build frozen set of blocked directed edges from horizontal and vertical wall sets."""
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


def bfs_path(start, blocked, goal_row):
    """Port of `bfsPath(pos, walls, goalRow)`: returns list of (r, c) from start to goal_row, or None."""
    if start[0] == goal_row:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 9 and 0 <= nc < 9:
                nxt = (nr, nc)
                if nxt not in parent and (cur, nxt) not in blocked:
                    parent[nxt] = cur
                    if nr == goal_row:
                        path = []
                        node = nxt
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    q.append(nxt)
    return None


def get_moves_from(pos, opp, blocked):
    """Port of `getMovesFrom(pos, opp, walls)` including straight and diagonal Quoridor jumps."""
    r, c = pos
    moves = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 9 and 0 <= nc < 9) or ((r, c), (nr, nc)) in blocked:
            continue
        if nr == opp[0] and nc == opp[1]:
            jr, jc = nr + dr, nc + dc
            if 0 <= jr < 9 and 0 <= jc < 9 and ((nr, nc), (jr, jc)) not in blocked:
                moves.append((jr, jc))
                continue
            for sr, sc in ((dc, dr), (-dc, -dr)):
                kr, kc = nr + sr, nc + sc
                if 0 <= kr < 9 and 0 <= kc < 9 and ((nr, nc), (kr, kc)) not in blocked:
                    moves.append((kr, kc))
        else:
            moves.append((nr, nc))
    return moves


def wall_candidates(target_pos, blocked, depth, goal_row):
    """Port of `wallCandidates(targetPos, walls, depth, goalRow)`: walls adjacent to target's BFS path."""
    path = bfs_path(target_pos, blocked, goal_row)
    if not path:
        return []
    result = set()
    limit = min(depth, len(path))
    for i in range(limit):
        pr, pc = path[i]
        for dr in (-1, 0):
            for dc in (-1, 0):
                r, c = pr + dr, pc + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    result.add(("H", r, c))
                    result.add(("V", r, c))
    return list(result)


def try_wall(ori, r, c, h_walls, v_walls, blocked, p_a, p_b, goal_a, goal_b):
    """Port of `tryWall(key, walls, pA, pB, goalA, goalB)`: validates geometry + both paths remain open."""
    if ori == "H":
        if (r, c) in v_walls:
            return None
        if (r, c - 1) in h_walls or (r, c) in h_walls or (r, c + 1) in h_walls:
            return None
        new_h = h_walls | {(r, c)}
        new_v = v_walls
        edges = (((r, c), (r + 1, c)), ((r, c + 1), (r + 1, c + 1)))
    else:
        if (r, c) in h_walls:
            return None
        if (r - 1, c) in v_walls or (r, c) in v_walls or (r + 1, c) in v_walls:
            return None
        new_h = h_walls
        new_v = v_walls | {(r, c)}
        edges = (((r, c), (r, c + 1)), ((r + 1, c), (r + 1, c + 1)))

    added = {(u, v) for a, b in edges for u, v in ((a, b), (b, a))}
    new_blocked = blocked | added
    if bfs_path(p_a, new_blocked, goal_a) is None:
        return None
    if bfs_path(p_b, new_blocked, goal_b) is None:
        return None
    return new_h, new_v, new_blocked


def move_toward_goal(ai_pos, human_pos, blocked, ai_path):
    """Port of `moveTowardGoal(aiPos, humanPos, walls, aiPath)`."""
    if not ai_path or len(ai_path) < 2:
        return None
    ns = ai_path[1]
    if ns != human_pos:
        return ns
    if (ai_pos, ns) in blocked:
        return None
    dr, dc = ns[0] - ai_pos[0], ns[1] - ai_pos[1]
    for d2r, d2c in ((dr, dc), (-dc, dr), (dc, -dr), (-dr, -dc)):
        jr, jc = ns[0] + d2r, ns[1] + d2c
        if (jr, jc) == ai_pos:
            continue
        if 0 <= jr < 9 and 0 <= jc < 9 and (ns, (jr, jc)) not in blocked:
            return (jr, jc)
    return None


def worst_wall_delay(target, h_walls, v_walls, blocked, goal, p_a, p_b, goal_a, goal_b):
    """Port of `worstWallDelay`: measures how badly 1 enemy wall can delay target's path."""
    base_p = bfs_path(target, blocked, goal)
    if not base_p:
        return 99
    base = len(base_p) - 1
    worst = 0
    for ori, r, c in wall_candidates(target, blocked, 4, goal):
        res = try_wall(ori, r, c, h_walls, v_walls, blocked, p_a, p_b, goal_a, goal_b)
        if res is None:
            continue
        np = bfs_path(target, res[2], goal)
        if np:
            delay = (len(np) - 1) - base
            if delay > worst:
                worst = delay
    return worst


def choose_action(observation):
    """Arena entry point implementing Hard Waller (`aiHard` + `aiDuelHard` path robustness)."""
    legal_moves = observation.get("legal_moves", [])
    if not legal_moves:
        return {"type": "pass"}

    if observation.get("turn", 0) <= 1:
        RECENT_POSITIONS.clear()

    you = observation["you"]
    opp = observation["opponent"]
    ai_pos = tuple(you["position"])
    hum_pos = tuple(opp["position"])
    ai_goal = you["goal_row"]
    hum_goal = opp["goal_row"]
    ai_barr = you["walls_remaining"]
    hum_barr = opp["walls_remaining"]

    RECENT_POSITIONS.append(ai_pos)
    recent = list(reversed(RECENT_POSITIONS))  # newest first, matching recentAi in ai.js

    h_walls = frozenset((w[0], w[1]) for w in observation["walls"]["horizontal"])
    v_walls = frozenset((w[0], w[1]) for w in observation["walls"]["vertical"])
    blocked = _walls_to_blocked(h_walls, v_walls)

    init_moves = get_moves_from(ai_pos, hum_pos, blocked)

    # 1. Immediate win has absolute priority
    for m in init_moves:
        if m[0] == ai_goal:
            return {"type": "move", "to": [m[0], m[1]]}

    ai_fwd_down = ai_goal > ai_pos[0]
    ai_path_init = bfs_path(ai_pos, blocked, ai_goal)
    hum_path_init = bfs_path(hum_pos, blocked, hum_goal)

    # 2. Endgame sprint (from aiHard): deterministic sprint when out of walls or clearly ahead
    if ai_path_init and len(ai_path_init) > 1:
        ai_d_init = len(ai_path_init) - 1
        hum_d_init = len(hum_path_init) - 1 if hum_path_init else 999
        if ai_barr <= 0 or (ai_d_init <= 3 and hum_d_init > ai_d_init + 1) or hum_d_init > ai_d_init + 2:
            m = move_toward_goal(ai_pos, hum_pos, blocked, ai_path_init)
            if m and [m[0], m[1]] in legal_moves:
                return {"type": "move", "to": [m[0], m[1]]}

    t0 = time.monotonic()

    # Static evaluation (from aiHard `evalState`)
    def eval_state(ap, hp, blk, ab, hb):
        apath = bfs_path(ap, blk, ai_goal)
        hpath = bfs_path(hp, blk, hum_goal)
        if not apath:
            return -99999.0
        if not hpath:
            return 99999.0
        if ap[0] == ai_goal:
            return 50000.0
        if hp[0] == hum_goal:
            return -50000.0
        ai_d = len(apath) - 1
        hum_d = len(hpath) - 1
        return (hum_d - ai_d) * W_REL_DIST - ai_d * W_AI_DIST + hum_d * W_HUM_DIST + (ab - hb) * W_BARR_DIFF

    # Action generator + move ordering (from aiHard `genActions`)
    def gen_actions(ap, hp, hw, vw, blk, ab, hb, for_ai, max_walls):
        me = ap if for_ai else hp
        opp_p = hp if for_ai else ap
        my_barr = ab if for_ai else hb
        me_goal = ai_goal if for_ai else hum_goal
        opp_g = hum_goal if for_ai else ai_goal

        acts = []
        my_path = bfs_path(me, blk, me_goal)
        moves = get_moves_from(me, opp_p, blk)
        for m in moves:
            pri = 0.0
            if my_path and len(my_path) > 1 and my_path[1] == m:
                pri = 30.0
            if (m[0] > me[0]) if (me_goal > me[0]) else (m[0] < me[0]):
                pri += 8.0
            acts.append((pri, "move", m, None))

        if my_barr > 0:
            cands = wall_candidates(opp_p, blk, WALL_CANDIDATE_DEPTH, opp_g)
            old_opp_path = bfs_path(opp_p, blk, opp_g)
            old_opp_d = len(old_opp_path) - 1 if old_opp_path else 0
            my_old_d = len(my_path) - 1 if my_path else 0
            scored = []
            for ori, r, c in cands:
                res = try_wall(ori, r, c, hw, vw, blk, ap, hp, ai_goal, hum_goal)
                if res is None:
                    continue
                nhw, nvw, nblk = res
                np = bfs_path(opp_p, nblk, opp_g)
                my_new = bfs_path(me, nblk, me_goal)
                if not np or not my_new:
                    continue
                gain = ((len(np) - 1) - old_opp_d) - ((len(my_new) - 1) - my_old_d)
                if gain <= 0:
                    continue
                scored.append((gain, (ori, r, c, nhw, nvw, nblk)))
            scored.sort(key=lambda x: x[0], reverse=True)
            for gain, wall_pack in scored[:max_walls]:
                acts.append((gain * 8.0, "wall", None, wall_pack))

        acts.sort(key=lambda x: x[0], reverse=True)
        return acts

    # Alpha-Beta Minimax search (from aiHard `search`)
    def search(ap, hp, hw, vw, blk, ab, hb, depth, maximizing, alpha, beta):
        if ap[0] == ai_goal:
            return 50000.0 - depth
        if hp[0] == hum_goal:
            return -50000.0 + depth
        if depth == 0 or (time.monotonic() - t0) > TIME_LIMIT:
            return eval_state(ap, hp, blk, ab, hb)

        max_w = INNER_MAX_WALLS_D2 if depth >= 2 else INNER_MAX_WALLS_D1
        acts = gen_actions(ap, hp, hw, vw, blk, ab, hb, maximizing, max_w)
        if not acts:
            return eval_state(ap, hp, blk, ab, hb)

        if maximizing:
            best = -float("inf")
            for _, kind, m_pos, w_pack in acts:
                if kind == "move":
                    s = search(m_pos, hp, hw, vw, blk, ab, hb, depth - 1, False, alpha, beta)
                else:
                    _, _, _, nhw, nvw, nblk = w_pack
                    s = search(ap, hp, nhw, nvw, nblk, ab - 1, hb, depth - 1, False, alpha, beta)
                if s > best:
                    best = s
                if best > alpha:
                    alpha = best
                if beta <= alpha:
                    break
            return best
        else:
            best = float("inf")
            for _, kind, m_pos, w_pack in acts:
                if kind == "move":
                    s = search(ap, m_pos, hw, vw, blk, ab, hb, depth - 1, True, alpha, beta)
                else:
                    _, _, _, nhw, nvw, nblk = w_pack
                    s = search(ap, hp, nhw, nvw, nblk, ab, hb - 1, depth - 1, True, alpha, beta)
                if s < best:
                    best = s
                if best < beta:
                    beta = best
                if beta <= alpha:
                    break
            return best

    # 3. Root evaluation combining Minimax + path robustness + anti-cycling penalties
    best_score = -float("inf")
    best_action = None
    my_acts = gen_actions(ai_pos, hum_pos, h_walls, v_walls, blocked, ai_barr, hum_barr, True, ROOT_MAX_WALLS)

    for _, kind, m_pos, w_pack in my_acts:
        if kind == "move":
            s = search(m_pos, hum_pos, h_walls, v_walls, blocked, ai_barr, hum_barr, SEARCH_DEPTH - 1, False, -float("inf"), float("inf"))
            r_pos, r_hw, r_vw, r_blk = m_pos, h_walls, v_walls, blocked
        else:
            ori, r, c, nhw, nvw, nblk = w_pack
            s = search(ai_pos, hum_pos, nhw, nvw, nblk, ai_barr - 1, hum_barr, SEARCH_DEPTH - 1, False, -float("inf"), float("inf"))
            r_pos, r_hw, r_vw, r_blk = ai_pos, nhw, nvw, nblk

        # Path-robustness penalty from aiDuelHard (avoid leaving own path easily blocked)
        worst = worst_wall_delay(r_pos, r_hw, r_vw, r_blk, ai_goal, r_pos, hum_pos, ai_goal, hum_goal)
        s -= worst * W_ROBUST_PENALTY

        if kind == "move":
            # Bias toward BFS main path to prevent zigzagging across equal-length paths
            if ai_path_init and len(ai_path_init) > 1 and ai_path_init[1] == m_pos:
                s += W_PATH_FOLLOW
            # Forward progress bonus / backward step penalty
            if (m_pos[0] > ai_pos[0]) if ai_fwd_down else (m_pos[0] < ai_pos[0]):
                s += W_FWD_STEP
            elif (m_pos[0] < ai_pos[0]) if ai_fwd_down else (m_pos[0] > ai_pos[0]):
                s += W_BACK_STEP
            # Strong cycling penalty (from aiHard: -60 / -48 / -36 / -24 for i=1..4)
            if len(recent) >= 2:
                for i in range(1, min(len(recent), 5)):
                    if recent[i] == m_pos:
                        s -= (6 - i) * 12.0
                        break

        if s > best_score:
            best_score = s
            best_action = (
                {"type": "move", "to": [m_pos[0], m_pos[1]]}
                if kind == "move"
                else {"type": "wall", "orientation": w_pack[0], "row": w_pack[1], "col": w_pack[2]}
            )

    if best_action is not None:
        return best_action

    # Fallback: step along BFS path or first legal move
    if ai_path_init and len(ai_path_init) > 1:
        m = move_toward_goal(ai_pos, hum_pos, blocked, ai_path_init)
        if m and [m[0], m[1]] in legal_moves:
            return {"type": "move", "to": [m[0], m[1]]}
    return {"type": "move", "to": list(legal_moves[0])}

from typing import Any, Dict
class HardWaller:

    def __init__(self):
       pass

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return choose_action(obs)