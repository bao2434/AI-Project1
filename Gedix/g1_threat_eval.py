"""G1 (Direction D1 — Threat-Aware Evaluation): Isolated ablation on top of G0.

Keeps G0's exact priors(), Node.select(), and search() unchanged.
Replaces G0's 2-feature value(st) with a threat-aware, phase-aware evaluation that measures:
  - Shortest-path lead: (d_opp - d_self)
  - Single-wall route vulnerability (R_self) & offensive wall threat potential (R_opp)
    computed strictly over shortest-path cut masks (cuts & ~forbidden) and cached via LRU
  - Effective usable walls: min(w_self, d_opp) - min(w_opp, d_self)
  - Wall-exhaustion pure-race bonus when opponent has no walls left.

Supports variants G1a (conservative), G1b (balanced, default), and G1c (over-aggressive)
for the AI variant selection pipeline.
"""
from functools import lru_cache
from math import exp, sqrt
from time import perf_counter
from typing import Any, Dict


TIME_BUDGET = 0.80
VARIANT = "G1b"  # "G1a", "G1b", or "G1c"

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

LAST_STATS = {"simulations": 0, "expanded_nodes": 0, "root_kids": 0, "elapsed": 0.0, "threat_evals": 0}


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


@lru_cache(24000)
def max_single_wall_threat(walls, victim_side, victim_sq, other_sq):
    """Maximum shortest-path detour a single legal wall can inflict on victim_side.

    Only walls intersecting victim's current shortest path (cuts & ~forbidden) can
    increase victim's shortest-path distance in one move.
    """
    LAST_STATS["threat_evals"] += 1
    d, cuts = route(walls, victim_side, victim_sq)
    if d >= 127 or not cuts:
        return 0
    forbidden = wall_context(walls)[2]
    candidates = cuts & ~forbidden
    if not candidates:
        return 0
    other_side = 1 - victim_side
    _, other_cuts = route(walls, other_side, other_sq)
    other_bit = BIT[other_sq]
    victim_bit = BIT[victim_sq]
    best_detour = 0
    for k in indices(candidates):
        bit = BIT[k]
        w = walls | bit
        # Check victim reachability and new distance
        v_layers = waves(w, victim_side)
        if not (v_layers[-1] & victim_bit):
            continue
        # Only check other player's reachability if wall also cuts their shortest path
        if (other_cuts & bit) and not (waves(w, other_side)[-1] & other_bit):
            continue
        # Binary search new distance in v_layers
        lo, hi = d, len(v_layers) - 1
        if hi <= d + best_detour:
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if v_layers[mid] & victim_bit:
                hi = mid
            else:
                lo = mid + 1
        detour = lo - d
        if detour > best_detour:
            best_detour = detour
    return best_detour


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


@lru_cache(24000)
def value(st, variant=VARIANT):
    """Threat-aware and phase-aware estimated win probability for player to move, in [0, 1]."""
    walls, me = st[4], st[5]
    op = 1 - me
    sq_me, sq_op = st[me], st[op]
    dm = distance(walls, me, sq_me)
    do = distance(walls, op, sq_op)
    wm = st[2 + me]
    wo = st[2 + op]

    # Effective usable walls
    um = min(wm, do)
    uo = min(wo, dm)

    # Single-wall route threats (0 if attacking player has no walls left)
    tm = max_single_wall_threat(walls, me, sq_me, sq_op) if wo > 0 else 0
    to = max_single_wall_threat(walls, op, sq_op, sq_me) if wm > 0 else 0
    tm_cap = min(tm, 10)
    to_cap = min(to, 10)

    t = do - dm
    if variant == "G1a":
        score = (
            0.80 * t
            + 0.12 * (wm - wo)
            + 0.08 * (um - uo)
            + 0.18 * (to_cap - tm_cap)
        )
    elif variant == "G1c":
        score = (
            0.65 * t
            + 0.10 * (wm - wo)
            + 0.55 * (to_cap - 1.4 * tm_cap)
        )
    else:  # G1b (default balanced threat + usable walls + pure-race bonus)
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
    a,b,x,y,walls,me = st
    op = 1-me
    pm = pawns(st)
    if root_legal is not None: pm = [p for p in pm if p+128 in root_legal]
    ds = {p:distance(walls,me,p) for p in pm}
    best = min(ds.values(), default=127)
    if st[2+op]: pm.sort(key=ds.get)
    logits = [(128+p, 2. if ds[p] == best else -1.) for p in pm if st[2+op] or ds[p] == best]
    if st[2+me]:
        routes = (route(walls,0,a), route(walls,1,b))
        _,near,forbidden = wall_context(walls)
        for j,k in enumerate(indices((near | PAWN_NEAR[a] | PAWN_NEAR[b]) & ~forbidden)):
            if j % 8 == 0 and perf_counter() >= deadline: raise Expired
            if root_legal is not None and k not in root_legal: continue
            lengths = after_wall(st,k,routes)
            if 127 in lengths: continue
            mine = lengths[me]-routes[me][0]
            theirs = lengths[op]-routes[op][0]
            if not st[2+op] and theirs == 0: continue
            logits.append((k, .6*(theirs-1.4*mine)-.4 if theirs else -1.6-.8*mine))
    if not logits: return []
    mx = max(v for _,v in logits)
    ps = [(k,exp(v-mx)) for k,v in logits]
    total = sum(v for _,v in ps)
    return [(k,v/total) for k,v in ps]


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
    LAST_STATS["threat_evals"] = 0
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
                    v = value(pos, VARIANT)
                    break
                if n.kids is None:
                    n.kids = [Node(k,p) for k,p in priors(pos,deadline,legal if n is root else None)]
                    expanded += 1
                if not n.kids:
                    n.proof = -1
                    v = value(pos, VARIANT)
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


class GedikG1:
    def __init__(self, variant: str = "G1b", time_budget: float = TIME_BUDGET):
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
