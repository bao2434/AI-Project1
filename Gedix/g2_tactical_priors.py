"""G2 (Direction D2 — Tactical Wall Priors): Isolated ablation on top of G0.

Keeps G0's exact 2-feature value(st) and Node.select() unchanged.
Upgrades priors(st) with:
  1. Defensive wall detection: identifies opponent single-wall threats (delta >= 2) along
     our shortest path (routes[me][1] & ~forbidden) and boosts non-self-blocking walls in
     CONFLICT[k_threat] that physically prevent the opponent's threat placement.
  2. Structural cohesion / setup signal: rewards offensive walls that chain with existing
     walls/borders (near & BIT[k]) and setup walls adjacent to opponent's shortest-path cuts.
  3. Top-K wall filtering with mandatory defensive quota (reserving up to 3 defensive slots
     so critical defensive walls are never drowned out by noise).

Supports variants G2a (Defense Quota + Top-K Tactical Priors, default) and
G2b (Unfiltered Heavy Setup Priors) for the AI variant selection pipeline.
"""
from functools import lru_cache
from math import exp, sqrt
from time import perf_counter
from typing import Any, Dict


TIME_BUDGET = 0.80
VARIANT = "G2a"  # "G2a" or "G2b"

FULL = (1 << 81) - 1
GOAL = (511, 511 << 72)
DELTA = (-9, 1, 9, -1)
BIT = tuple(1 << i for i in range(128))
LOW64 = BIT[64] - 1
EAST = FULL ^ sum(BIT[9 * r + 8] for r in range(9))
EDGE = {}
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

LAST_STATS = {"simulations": 0, "expanded_nodes": 0, "root_kids": 0, "elapsed": 0.0, "defensive_priors": 0}


def indices(bits):
    while bits:
        b = bits & -bits
        yield b.bit_length() - 1
        bits ^= b


@lru_cache(8192)
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


@lru_cache(16000)
def waves(walls, side):
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


@lru_cache(16000)
def route(walls, side, sq):
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


@lru_cache(16000)
def defensive_block_mask(walls, me, sq_me, sq_op):
    """Find the opponent's strongest single-wall threats (detour >= 2) against `me`
    and return (max_threat_detour, bitmask_of_conflicting_preventive_walls).
    """
    d_me, cuts_me = route(walls, me, sq_me)
    if d_me >= 127 or not cuts_me:
        return 0, 0
    forbidden = wall_context(walls)[2]
    candidates = cuts_me & ~forbidden
    if not candidates:
        return 0, 0
    op = 1 - me
    _, cuts_op = route(walls, op, sq_op)
    me_bit = BIT[sq_me]
    op_bit = BIT[sq_op]
    max_detour = 0
    threat_walls = []
    for k in indices(candidates):
        bit = BIT[k]
        w = walls | bit
        me_layers = waves(w, me)
        if not (me_layers[-1] & me_bit):
            continue
        if (cuts_op & bit) and not (waves(w, op)[-1] & op_bit):
            continue
        lo, hi = d_me, len(me_layers) - 1
        if hi < d_me + 2:
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if me_layers[mid] & me_bit:
                hi = mid
            else:
                lo = mid + 1
        detour = lo - d_me
        if detour >= 2:
            threat_walls.append((detour, k))
            if detour > max_detour:
                max_detour = detour
    if max_detour < 2:
        return 0, 0
    # Include conflicts for threats within 1 step of max_detour
    block_mask = 0
    threshold = max(2, max_detour - 1)
    for detour, k in threat_walls:
        if detour >= threshold:
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


def after_wall(st, k, routes):
    bit = BIT[k]
    w = st[4] | bit
    return tuple(distance(w, i, st[i]) if cuts & bit else dist
                 for i, (dist, cuts) in enumerate(routes))


HEURISTIC_DISTANCE = 0.80
HEURISTIC_WALLS = 0.15


@lru_cache(20000)
def value(st):
    """Exact G0 2-feature value function (isolated for G2 ablation)."""
    walls, me = st[4], st[5]
    op = 1 - me
    dm = distance(walls, me, st[me])
    do = distance(walls, op, st[op])
    wm = st[2 + me]
    wo = st[2 + op]
    score = HEURISTIC_DISTANCE * (do - dm) + HEURISTIC_WALLS * (wm - wo)
    return 1.0 / (1.0 + exp(-score)) if score >= 0 else exp(score) / (1.0 + exp(score))


class Expired(Exception):
    pass


def priors(st, deadline, root_legal=None):
    """Tactical wall priors with defensive blocking quota and wall-chaining setup signal."""
    a, b, x, y, walls, me = st
    op = 1 - me
    pm = pawns(st)
    if root_legal is not None:
        pm = [p for p in pm if p + 128 in root_legal]
    ds = {p: distance(walls, me, p) for p in pm}
    best = min(ds.values(), default=127)
    if st[2 + op]:
        pm.sort(key=ds.get)

    # Pawn move logits with slight central column tie-breaker for best moves
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
        routes = (route(walls, 0, a), route(walls, 1, b))
        _, near, forbidden = wall_context(walls)
        sq_me, sq_op = st[me], st[op]
        cuts_me = routes[me][1]
        cuts_op = routes[op][1]

        # Defensive threat analysis when opponent has walls remaining
        max_threat, def_mask = (
            defensive_block_mask(walls, me, sq_me, sq_op) if st[2 + op] > 0 else (0, 0)
        )

        candidate_mask = (near | PAWN_NEAR[a] | PAWN_NEAR[b]) & ~forbidden
        for j, k in enumerate(indices(candidate_mask)):
            if j % 8 == 0 and perf_counter() >= deadline:
                raise Expired
            if root_legal is not None and k not in root_legal:
                continue
            lengths = after_wall(st, k, routes)
            if 127 in lengths:
                continue
            mine = lengths[me] - routes[me][0]
            theirs = lengths[op] - routes[op][0]
            if not st[2 + op] and theirs == 0:
                continue

            bit = BIT[k]
            is_defensive = bool(def_mask & bit) and mine == 0 and not (cuts_me & bit)
            is_connected = bool(near & bit)
            near_op_route = bool(WALL_NEAR[k % 64] & cuts_op)

            if VARIANT == "G2b":
                # Variant G2b: Unfiltered heavy setup + defense priors
                if theirs > 0:
                    logit = 0.65 * (theirs - 1.3 * mine) - 0.3 + (0.30 if is_connected else 0.0)
                elif is_defensive:
                    logit = 0.20 + 0.35 * min(max_threat, 6)
                elif mine == 0 and near_op_route:
                    logit = -0.35  # Heavy setup weight without top-K pruning
                else:
                    logit = -1.6 - 0.8 * mine
                wall_logits.append((k, logit))
            else:
                # Variant G2a (default): Calibrated Offensive + Defensive Quota + Setup Cohesion
                if theirs > 0:
                    cohesion = 0.22 if is_connected else 0.0
                    def_bonus = 0.35 if is_defensive else 0.0
                    logit = 0.62 * (theirs - 1.4 * mine) - 0.38 + cohesion + def_bonus
                    wall_logits.append((k, logit))
                elif is_defensive:
                    # Pure defensive wall blocking opponent's high-impact threat
                    def_logit = -0.55 + 0.32 * min(max_threat, 6)
                    defensive_candidates.append((k, def_logit))
                    LAST_STATS["defensive_priors"] += 1
                elif mine == 0 and is_connected and near_op_route:
                    # Low-cost setup wall connected to existing structure near opponent route
                    wall_logits.append((k, -1.25))
                else:
                    wall_logits.append((k, -1.75 - 0.8 * mine))

    if VARIANT == "G2a" and (wall_logits or defensive_candidates):
        # Step 3: Top-K wall selection + mandatory defensive quota (up to 3 defensive walls)
        wall_logits.sort(key=lambda item: item[1], reverse=True)
        top_walls = wall_logits[:16]
        if defensive_candidates:
            defensive_candidates.sort(key=lambda item: item[1], reverse=True)
            top_walls.extend(defensive_candidates[:3])
        logits = pawn_logits + top_walls
    else:
        logits = pawn_logits + wall_logits

    if not logits:
        return []
    mx = max(v for _, v in logits)
    ps = [(k, exp(v - mx)) for k, v in logits]
    total = sum(v for _, v in ps)
    return [(k, v / total) for k, v in ps]


class Node:
    __slots__ = ('move','prior','visits','total','proof','kids')
    def __init__(self, move, prior):
        self.move, self.prior = move, prior
        self.visits, self.total, self.proof, self.kids = 0, 0., 0, None

    def select(self):
        fpu = max(0., (1-self.total/self.visits if self.visits else .5)-.2)
        c = 1.6*sqrt(max(1,self.visits))
        best, mx = None, -1.
        for n in self.kids:
            if n.proof == -1: return n
            q = n.total/n.visits if n.visits else fpu
            score = 0. if n.proof == 1 else q+c*n.prior/(1+n.visits)
            if score > mx: best,mx = n,score
        return best


def search(st, legal, fallback, deadline):
    t0 = perf_counter()
    LAST_STATS["defensive_priors"] = 0
    root = Node(-1,1.)
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
                    n.kids = [Node(k,p) for k,p in priors(pos,deadline,legal if n is root else None)]
                    expanded += 1
                if not n.kids:
                    n.proof = -1
                    v = value(pos)
                    break
                if perf_counter() >= deadline: raise Expired
                n = n.select()
                pos = child(pos,n.move)
                path.append(n)
        except Expired:
            break
        for cur in reversed(path):
            v = 1-v
            cur.visits += 1
            cur.total += v
        for parent,desc in zip(reversed(path[:-1]),reversed(path[1:])):
            if parent.proof: break
            if desc.proof == -1: parent.proof = 1
            elif all(c.proof == 1 for c in parent.kids): parent.proof = -1
            else: break
        if root.proof or (root.kids and len(root.kids) == 1): break
    LAST_STATS["simulations"] = root.visits
    LAST_STATS["expanded_nodes"] = expanded
    LAST_STATS["root_kids"] = len(root.kids) if root.kids else 0
    LAST_STATS["elapsed"] = perf_counter() - t0
    if not root.kids or not root.visits: return fallback
    for n in root.kids:
        if n.proof == -1: return n.move
    best = max(root.kids,key=lambda n:(n.visits,n.total/n.visits if n.visits else 0.))
    if best.total / max(1,best.visits) < .03:
        best = max(reversed([n for n in root.kids if n.proof != 1] or root.kids),key=lambda n:n.prior)
    return best.move


def choose_action(observation):
    deadline = perf_counter()+TIME_BUDGET
    legal = {}
    ds = observation['distances']
    for r,c in observation['legal_moves']:
        act = {'type':'move','to':[r,c]}
        if r == observation['you']['goal_row']: return act
        legal[128+9*r+c] = act
    fallback = min(legal, key=lambda k: ds[(k-128)//9][(k-128)%9]) if legal else None
    for w in observation['legal_walls']:
        k = (64 if w['orientation'] == 'V' else 0)+8*w['row']+w['col']
        legal[k] = dict(w,type='wall')
    if not legal: return {'type':'pass'}
    if fallback is None: fallback = next(iter(legal))
    return legal[search(from_obs(observation),legal,fallback,deadline)]


class GedikG2:
    def __init__(self, variant: str = "G2a", time_budget: float = TIME_BUDGET):
        self.variant = variant
        self.time_budget = time_budget

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        global VARIANT, TIME_BUDGET
        old_v, old_t = VARIANT, TIME_BUDGET
        VARIANT, TIME_BUDGET = self.variant, self.time_budget
        try:
            return choose_action(obs)
        finally:
            VARIANT, TIME_BUDGET = old_v, old_t
