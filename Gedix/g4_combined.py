"""G4 (Direction D4 + Combined Ablation Winner: D1 + D2 + D3 + D4):
Standalone single-file Quoridor / Wrongway agent using only the Python standard library.

Key Components:
  - D1 (Threat-Aware Value): Evaluates shortest-path lead (d_opp - d_self), effective
    usable walls (min(w_self, d_opp) - min(w_opp, d_self)), single-wall route vulnerability
    (R_self) & offensive threat potential (R_opp), and wall-exhaustion pure-race margin.
  - D2 (Tactical & Defensive Wall Priors): Boosts walls that physically conflict with the
    opponent's highest-detour threat wall (CONFLICT[k_threat]), rewards wall-chaining
    cohesion (near & BIT[k]), and enforces top-K candidate selection with a dedicated
    defensive wall quota.
  - D3 (Phase-Adaptive PUCT & Progressive Widening): Scales exploration c(s) by game phase
    and widens active wall branches gradually with node visit count sqrt(N) while always
    preserving pawn moves and high-prior tactical/defensive moves.
  - D4 (Search Efficiency & Engineering):
    * Two-stage bitmask candidate filtering in priors(): any wall k not intersecting
      (cuts_me | cuts_op) has delta_me == 0, delta_op == 0, and is guaranteed legal
      (cannot disconnect either player since both shortest paths remain intact), skipping
      flood-fill/distance calls completely.
    * Unified LRU-cached route_threat_profile() computing both R_victim and the preventive
      conflict bitmask in one pass.
    * Calibrated 0.78s time budget for zero timeouts under the 1.0s Arena subprocess limit.
"""
from functools import lru_cache
from math import exp, sqrt
from time import perf_counter
from typing import Any, Dict


TIME_BUDGET = 0.78
FULL = (1 << 81) - 1
GOAL = (511, 511 << 72)
DELTA = (-9, 1, 9, -1)  # upstream direction / tie-breaking order N E S W
BIT = tuple(1 << i for i in range(128))
LOW64 = BIT[64] - 1
EAST = FULL ^ sum(BIT[9 * r + 8] for r in range(9))
EDGE = {}  # pawn edge -> bit mask of walls that block it
CONFLICT = []
WALL_NEAR = []
PAWN_NEAR = []
BORDER = 0
for k in range(128):
    r, c = divmod(k % 64, 8)
    s = 9 * r + c
    edges = ((s, s+9), (s+1, s+10)) if k < 64 else ((s, s+1), (s+9, s+10))
    for a, b in edges:
        EDGE[a, b] = EDGE[b, a] = EDGE.get((a, b), 0) | BIT[k]
    conflict = BIT[k % 64] | BIT[64 + k % 64]
    if k < 64:
        if c: conflict |= BIT[k-1]
        if c < 7: conflict |= BIT[k+1]
    else:
        if r: conflict |= BIT[k-8]
        if r < 7: conflict |= BIT[k+8]
    CONFLICT.append(conflict)
    if (c if k < 64 else r) in (0, 7): BORDER |= BIT[k]
for s in range(64):
    r, c = divmod(s, 8)
    mask = sum(BIT[i] for i in range(64) if abs(i//8-r) <= 1 and abs(i%8-c) <= 2)
    WALL_NEAR.append(mask | mask << 64)
for s in range(81):
    r, c = divmod(s, 9)
    mask = sum(BIT[i] for i in range(64) if abs(i//8-r) <= 2 and abs(i%8-c) <= 2)
    PAWN_NEAR.append(mask | mask << 64)

LAST_STATS = {
    "simulations": 0,
    "expanded_nodes": 0,
    "root_kids": 0,
    "elapsed": 0.0,
    "skipped_floods": 0,
}


def indices(bits):
    while bits:
        b = bits & -bits
        yield b.bit_length() - 1
        bits ^= b


@lru_cache(16384)
def wall_context(walls):
    south, east, near, forbidden = FULL ^ GOAL[1], EAST, BORDER, 0
    for k in indices(walls):
        r, c = divmod(k % 64, 8)
        b = BIT[9*r+c]
        if k < 64: south &= ~(b | b << 1)
        else: east &= ~(b | b << 9)
        near |= WALL_NEAR[k % 64]
        forbidden |= CONFLICT[k]
    return (south << 9, east, south, east << 1), near, forbidden


@lru_cache(32768)
def waves(walls, side):
    """Cumulative bitboard flood from a goal, one entry per distance."""
    n, e, s, w = wall_context(walls)[0]
    seen = GOAL[side]
    out = [seen]
    while seen != FULL:
        more = seen | (seen & n) >> 9 | (seen & e) << 1 | (seen & s) << 9 | (seen & w) >> 1
        if more == seen: break
        seen = more
        out.append(seen)
    return out


def distance(walls, side, sq):
    layers = waves(walls, side)
    b = BIT[sq]
    if not layers[-1] & b: return 127
    lo, hi = 0, len(layers)-1
    while lo < hi:
        mid = (lo+hi)//2
        if layers[mid] & b: hi = mid
        else: lo = mid+1
    return lo


@lru_cache(32768)
def route(walls, side, sq):
    """Distance and walls intersecting one shortest path (upstream tie order)."""
    d = distance(walls, side, sq)
    cuts = 0
    dirs = wall_context(walls)[0]
    layers = waves(walls, side)
    for i in range(d-1, -1, -1):
        for mask, step in zip(dirs, DELTA):
            if mask & BIT[sq] and layers[i] & BIT[sq+step]:
                cuts |= EDGE[sq, sq+step]
                sq += step
                break
    return d, cuts


@lru_cache(32768)
def route_threat_profile(walls, victim_side, victim_sq, other_sq):
    """Unified D1 + D2 threat profile for `victim_side`.

    Returns `(max_detour, preventive_conflict_mask)`:
      - `max_detour`: largest shortest-path increase a single legal wall can cause `victim_side`
      - `preventive_conflict_mask`: bitmask of legal walls that physically conflict with
        the opponent's strongest threat wall(s) (detour >= 2).
    """
    d_v, cuts_v = route(walls, victim_side, victim_sq)
    if d_v >= 127 or not cuts_v:
        return 0, 0
    forbidden = wall_context(walls)[2]
    candidates = cuts_v & ~forbidden
    if not candidates:
        return 0, 0
    other_side = 1 - victim_side
    _, cuts_o = route(walls, other_side, other_sq)
    v_bit = BIT[victim_sq]
    o_bit = BIT[other_sq]
    max_detour = 0
    threat_list = []
    for k in indices(candidates):
        bit = BIT[k]
        w = walls | bit
        v_layers = waves(w, victim_side)
        if not (v_layers[-1] & v_bit):
            continue
        if (cuts_o & bit) and not (waves(w, other_side)[-1] & o_bit):
            continue
        lo, hi = d_v, len(v_layers) - 1
        if hi <= d_v:
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if v_layers[mid] & v_bit:
                hi = mid
            else:
                lo = mid + 1
        detour = lo - d_v
        if detour > 0:
            if detour > max_detour:
                max_detour = detour
            if detour >= 2:
                threat_list.append((detour, k))
    if max_detour < 2:
        return max_detour, 0
    block_mask = 0
    thresh = max(2, max_detour - 1)
    for detour, k in threat_list:
        if detour >= thresh:
            block_mask |= (CONFLICT[k] ^ BIT[k])
    return max_detour, block_mask & ~forbidden


def from_obs(obs):
    side = obs['player_id']
    players = (obs['you'], obs['opponent']) if side == 0 else (obs['opponent'], obs['you'])
    a, b = players
    walls = sum(BIT[8*r+c+offset] for key, offset in [('horizontal', 0), ('vertical', 64)]
                for r, c in obs['walls'][key])
    return (9*a['position'][0]+a['position'][1], 9*b['position'][0]+b['position'][1],
            a['walls_remaining'], b['walls_remaining'], walls, side)


def pawns(st):
    side = st[5]
    me, op = st[side], st[1-side]
    dirs = wall_context(st[4])[0]
    dest = []
    for d, step in enumerate(DELTA):
        if not dirs[d] & BIT[me]: continue
        if me+step != op: dest.append(me+step)
        elif dirs[d] & BIT[op]: dest.append(op+step)
        else:
            for sd in ((1, 3) if d in (0, 2) else (0, 2)):
                if dirs[sd] & BIT[op]: dest.append(op+DELTA[sd])
    return dest


def child(st, k):
    a, b, x, y, walls, side = st
    if k < 128:
        return a, b, x-(side == 0), y-(side == 1), walls | BIT[k], 1-side
    dest = k-128
    return (a if side else dest, dest if side else b, x, y, walls, 1-side)


@lru_cache(32768)
def value(st):
    """D1 Threat-Aware & Phase-Aware leaf evaluation in [0, 1]."""
    walls, me = st[4], st[5]
    op = 1 - me
    sq_me, sq_op = st[me], st[op]
    dm = distance(walls, me, sq_me)
    do = distance(walls, op, sq_op)
    wm = st[2 + me]
    wo = st[2 + op]

    um = min(wm, do)
    uo = min(wo, dm)

    tm = route_threat_profile(walls, me, sq_me, sq_op)[0] if wo > 0 else 0
    to = route_threat_profile(walls, op, sq_op, sq_me)[0] if wm > 0 else 0
    tm_cap = min(tm, 10)
    to_cap = min(to, 10)

    t = do - dm
    race_bonus = 0.35 if (wo == 0 and t >= 0) else (-0.35 if (wm == 0 and t < 0) else 0.0)
    score = (
        0.82 * t
        + 0.10 * (wm - wo)
        + 0.14 * (um - uo)
        + 0.28 * (0.95 * to_cap - 1.05 * tm_cap)
        + race_bonus
    )
    return 1.0 / (1.0 + exp(-score)) if score >= 0 else exp(score) / (1.0 + exp(score))


class Expired(Exception):
    pass


def priors(st, deadline, root_legal=None):
    """D2 + D4 Tactical wall priors with two-stage bitmask filtering & defensive quota."""
    a, b, x, y, walls, me = st
    op = 1 - me
    pm = pawns(st)
    if root_legal is not None:
        pm = [p for p in pm if p + 128 in root_legal]
    ds = {p: distance(walls, me, p) for p in pm}
    best = min(ds.values(), default=127)
    if st[2 + op]:
        pm.sort(key=ds.get)

    pawn_logits = []
    for p in pm:
        if not st[2 + op] and ds[p] != best:
            continue
        if ds[p] == best:
            col_center_bonus = 0.04 * (4 - abs((p % 9) - 4))
            pawn_logits.append((128 + p, 2.0 + col_center_bonus))
        else:
            pawn_logits.append((128 + p, -1.0))

    wall_logits = []
    defensive_candidates = []

    if st[2 + me]:
        d0, cuts0 = route(walls, 0, a)
        d1, cuts1 = route(walls, 1, b)
        d_me, cuts_me = (d0, cuts0) if me == 0 else (d1, cuts1)
        d_op, cuts_op = (d1, cuts1) if me == 0 else (d0, cuts0)
        both_cuts = cuts_me | cuts_op
        sq_me, sq_op = st[me], st[op]

        _, near, forbidden = wall_context(walls)
        max_threat, def_mask = (
            route_threat_profile(walls, me, sq_me, sq_op) if st[2 + op] > 0 else (0, 0)
        )

        candidate_mask = (near | PAWN_NEAR[a] | PAWN_NEAR[b]) & ~forbidden
        for j, k in enumerate(indices(candidate_mask)):
            if j % 8 == 0 and perf_counter() >= deadline:
                raise Expired
            if root_legal is not None and k not in root_legal:
                continue

            bit = BIT[k]
            # D4 Optimization: Fast path when wall k misses both shortest paths!
            if not (both_cuts & bit):
                LAST_STATS["skipped_floods"] += 1
                if not st[2 + op]:
                    continue
                is_defensive = bool(def_mask & bit)
                if is_defensive:
                    defensive_candidates.append((k, -0.45 + 0.35 * min(max_threat, 6)))
                elif (near & bit) and (WALL_NEAR[k % 64] & cuts_op):
                    wall_logits.append((k, -1.35))
                else:
                    wall_logits.append((k, -1.70))
                continue

            # Wall intersects at least one player's shortest path -> compute exact detour
            w = walls | bit
            if cuts_me & bit:
                new_d_me = distance(w, me, sq_me)
                if new_d_me == 127:
                    continue
                mine = new_d_me - d_me
            else:
                mine = 0

            if cuts_op & bit:
                new_d_op = distance(w, op, sq_op)
                if new_d_op == 127:
                    continue
                theirs = new_d_op - d_op
            else:
                theirs = 0

            if not st[2 + op] and theirs == 0:
                continue

            is_defensive = bool(def_mask & bit) and mine == 0
            is_connected = bool(near & bit)

            if theirs > 0:
                cohesion = 0.20 if is_connected else 0.0
                def_bonus = 0.35 if is_defensive else 0.0
                logit = 0.62 * (theirs - 1.4 * mine) - 0.38 + cohesion + def_bonus
                wall_logits.append((k, logit))
            elif is_defensive:
                defensive_candidates.append((k, -0.45 + 0.35 * min(max_threat, 6)))
            elif mine == 0 and is_connected and (WALL_NEAR[k % 64] & cuts_op):
                wall_logits.append((k, -1.35))
            else:
                wall_logits.append((k, -1.70 - 0.8 * mine))

    if wall_logits or defensive_candidates:
        defensive_candidates.sort(key=lambda item: item[1], reverse=True)
        def_quota_set = {k for k, _ in defensive_candidates[:3]}
        logits = pawn_logits + defensive_candidates + wall_logits
    else:
        def_quota_set = set()
        logits = pawn_logits

    if not logits:
        return []
    mx = max(v for _, v in logits)
    ps = [(k, exp(v - mx)) for k, v in logits]
    total = sum(v for _, v in ps)
    out = [(k, v / total) for k, v in ps]
    # Sort descending for D3 progressive widening (pawn moves, top-3 defensive quota, and high-prior walls first)
    out.sort(key=lambda item: (item[0] >= 128 or item[0] in def_quota_set or item[1] >= 0.04, item[1]), reverse=True)
    return out


class Node:
    __slots__ = ('move','prior','visits','total','proof','kids')
    def __init__(self, move, prior):
        self.move, self.prior = move, prior
        self.visits, self.total, self.proof, self.kids = 0, 0., 0, None

    def select(self, phase):
        v_count = self.visits
        sv = sqrt(max(1, v_count))
        fpu = max(0., (1 - self.total / v_count if v_count else .5) - .18)
        c = (1.25 + 0.45 * phase) * sv
        best, mx = None, -1.
        kids = self.kids
        limit = min(len(kids), 12 + int(3.0 * sv))
        for idx in range(limit):
            n = kids[idx]
            if n.proof == -1:
                return n
            q = n.total / n.visits if n.visits else fpu
            score = 0. if n.proof == 1 else q + c * n.prior / (1 + n.visits)
            if score > mx:
                best, mx = n, score
        if best is None or mx <= 0.:
            for idx in range(limit, len(kids)):
                n = kids[idx]
                if n.proof == -1:
                    return n
                q = n.total / n.visits if n.visits else fpu
                score = 0. if n.proof == 1 else q + c * n.prior / (1 + n.visits)
                if score > mx:
                    best, mx = n, score
        return best


def search(st, legal, fallback, deadline):
    t0 = perf_counter()
    LAST_STATS["skipped_floods"] = 0
    root = Node(-1, 1.)
    expanded = 0
    while perf_counter() < deadline:
        pos, n, path = st, root, [root]
        try:
            while True:
                if (GOAL[0] & BIT[pos[0]]) or (GOAL[1] & BIT[pos[1]]):
                    n.proof = -1
                    v = 0.
                    break
                if n is not root and n.visits < 2:
                    v = value(pos)
                    break
                if n.kids is None:
                    n.kids = [Node(k, p) for k, p in priors(pos, deadline, legal if n is root else None)]
                    expanded += 1
                if not n.kids:
                    n.proof = -1
                    v = value(pos)
                    break
                if perf_counter() >= deadline:
                    raise Expired
                phase = (pos[2] + pos[3]) / 20.0
                n = n.select(phase)
                pos = child(pos, n.move)
                path.append(n)
        except Expired:
            break
        for cur in reversed(path):
            v = 1 - v
            cur.visits += 1
            cur.total += v
        for parent, desc in zip(reversed(path[:-1]), reversed(path[1:])):
            if parent.proof:
                break
            if desc.proof == -1:
                parent.proof = 1
            elif all(c.proof == 1 for c in parent.kids):
                parent.proof = -1
            else:
                break
        if root.proof or (root.kids and len(root.kids) == 1):
            break
    LAST_STATS["simulations"] = root.visits
    LAST_STATS["expanded_nodes"] = expanded
    LAST_STATS["root_kids"] = len(root.kids) if root.kids else 0
    LAST_STATS["elapsed"] = perf_counter() - t0
    if not root.kids or not root.visits:
        return fallback
    for n in root.kids:
        if n.proof == -1:
            return n.move
    best = max(root.kids, key=lambda n: (n.visits, n.total / n.visits if n.visits else 0.))
    if best.total / max(1, best.visits) < .03:
        best = max(reversed([n for n in root.kids if n.proof != 1] or root.kids), key=lambda n: n.prior)
    return best.move


def choose_action(observation):
    deadline = perf_counter() + TIME_BUDGET
    legal = {}
    ds = observation['distances']
    for r, c in observation['legal_moves']:
        act = {'type': 'move', 'to': [r, c]}
        if r == observation['you']['goal_row']:
            return act
        legal[128 + 9 * r + c] = act
    fallback = min(legal, key=lambda k: ds[(k - 128) // 9][(k - 128) % 9]) if legal else None
    for w in observation['legal_walls']:
        k = (64 if w['orientation'] == 'V' else 0) + 8 * w['row'] + w['col']
        legal[k] = dict(w, type='wall')
    if not legal:
        return {'type': 'pass'}
    if fallback is None:
        fallback = next(iter(legal))
    return legal[search(from_obs(observation), legal, fallback, deadline)]


class GedikG4:
    def __init__(self, time_budget: float = TIME_BUDGET):
        self.time_budget = time_budget

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        global TIME_BUDGET
        old = TIME_BUDGET
        TIME_BUDGET = self.time_budget
        try:
            return choose_action(obs)
        finally:
            TIME_BUDGET = old
