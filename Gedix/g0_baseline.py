"""G0 Baseline: Frozen reference port of Gedik No-NN (single-threaded PUCT, 2-feature handcrafted value heuristic).

Source: Adapted from Samet Kose's Gedik (https://github.com/msametkose-ctrl/gedik, MIT License),
with neural network removed and replaced by a 2-feature distance + wall stock sigmoid heuristic.
"""
from functools import lru_cache
import hashlib
from math import exp, sqrt
from time import perf_counter
from typing import Any, Dict


TIME_BUDGET = 0.80
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

LAST_STATS = {"simulations": 0, "expanded_nodes": 0, "root_kids": 0, "elapsed": 0.0}


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


@lru_cache(16000)
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
    """Estimated win chance for player to move, in [0, 1]. No NN."""
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


class GedikNoNN:
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
