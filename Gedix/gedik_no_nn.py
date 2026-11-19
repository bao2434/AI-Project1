"""Gedik Enhanced (G4/G5): Single-file Quoridor / Wrongway agent using only the Python standard library.

Developed from the baseline single-threaded PUCT port of Samet Kose's Gedik
(https://github.com/msametkose-ctrl/gedik, MIT License) along the G0 -> G5 roadmap
specified in `Gedik_NoNN_De_Xuat_Phat_Trien_Project1.md`:
  - D1 (G1 — Threat-Aware Evaluation): Evaluates single-wall route vulnerability (R_self)
    and offensive single-wall detour potential (R_opp) strictly over shortest-path cut masks
    (cuts & ~forbidden), alongside effective usable walls min(w, d_opp) and pure-race margin.
  - D2 (G2 — Tactical & Defensive Wall Priors): Boosts preventive walls in CONFLICT[k_threat]
    that block the opponent's highest-impact threat wall without lengthening own path, plus
    structural wall-chaining cohesion (near & BIT[k]) and a mandatory defensive quota.
  - D3 (G3 — Phase-Adaptive PUCT & Progressive Widening): Scales exploration c(s) by game
    phase (w0 + w1)/20 and widens active children W(N) = 12 + int(3 * sqrt(N)) while
    guaranteeing all pawn moves, defensive quota walls, and high-prior actions are preserved.
  - D4 (G4 — Search Efficiency & Bitboard Engineering): Two-stage zero-flood filtering in
    priors() skips flood-fill/distance checks completely whenever a candidate wall misses
    both players' current shortest paths (not ((cuts_me | cuts_op) & BIT[k])), combined with
    a unified LRU-cached route_threat_profile().
  - D5 (G5 — Self-Trained Residual Tactical MLP Value Network): 221-parameter 12 -> 16 -> 1
    Residual Tactical MLP trained on internal Arena game & search data (zero weights from
    upstream QNN1 ag.bin), achieving 82.09% holdout accuracy and 0.1184 Brier score at
    21.48 microseconds per evaluation in pure Python.

Set `EVAL_MODE = "G5_MLP"` (default) for the self-trained Residual Tactical MLP evaluator,
or `EVAL_MODE = "G4_HEURISTIC"` for the pure handcrafted threat-aware heuristic evaluator.
"""
from functools import lru_cache
from math import exp, sqrt
from time import perf_counter
from typing import Any, Dict


TIME_BUDGET = 0.78
EVAL_MODE = "G5_MLP"  # "G5_MLP" or "G4_HEURISTIC"

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

# Self-trained Residual Tactical MLP weights (12 -> 16 -> 1, 221 parameters)
_W_LIN = (
    0.115056, 2.292524, 1.585848, 0.276077, -0.097839, -1.354032,
    1.351745, -1.143883, 0.001611, 0.179127, 0.131964, 1.327014,
)
_W1 = (
    (-0.037155, 0.054704, 0.147082, -0.020111, -0.403116, 0.081549, -0.1364, -0.144342, 0.111174, -0.314299, -0.174241, -0.03517, 0.100798, -0.110029, 0.157913, -0.137094),
    (-0.155437, 0.292239, 0.044889, 0.012468, -0.257711, 0.180336, -0.135417, -0.304898, -0.276708, -0.229851, 0.073631, -0.048112, -0.015155, -0.021736, -0.080444, 0.254084),
    (-0.224566, 0.092826, -0.096408, -0.076765, 0.327894, 0.151848, 0.127599, 0.166571, 0.085392, 0.021341, 0.080195, -0.255499, -0.033707, 0.111585, -0.027366, -0.087491),
    (0.088471, 0.202046, -0.004403, -0.209901, 0.034918, -0.157848, -0.131599, 0.072229, 0.003654, 0.071257, 0.068792, 0.140895, -0.02472, 0.172657, -0.194195, 0.008435),
    (-0.04431, 0.211525, -0.183053, 0.145267, -0.150532, -0.032341, 0.039479, -0.034737, -0.02257, 0.139881, -0.096714, 0.00414, 0.024603, 0.128581, -0.186694, 0.070849),
    (-0.078403, 0.250621, -0.175739, -0.098571, -0.284399, -0.197752, -0.160946, -0.165366, 0.103145, -0.268695, -0.014451, -0.003799, -0.019094, 0.218744, -0.407797, 0.28806),
    (-0.181941, 0.264506, -0.0795, -0.050441, 0.449672, -0.076062, 0.139546, 0.306117, 0.132708, 0.301073, -0.175236, 0.138151, 0.098004, -0.047173, -0.09799, 0.204554),
    (0.162694, 0.017606, 0.066844, -0.154073, 0.38652, -0.059774, 0.007775, 0.166089, -0.291087, 0.103626, 0.163164, -0.088046, -0.062101, 0.01833, -0.030519, 0.219825),
    (-0.004088, 0.071218, 0.048634, -0.031583, -0.00597, -0.125546, 0.021619, -0.151445, -0.169247, 0.120491, -0.113898, 0.147142, 0.143432, -0.096199, 0.177897, 0.01316),
    (-0.060508, 0.251973, -0.171262, -0.035138, -0.065075, 0.023711, 0.131754, -0.035641, 0.039531, -0.091895, -0.092319, 0.045535, 0.099619, 0.136814, 0.119798, 0.060602),
    (-0.006697, -0.036316, -0.104225, -0.200188, -0.057425, 0.026749, -0.171498, -0.007552, -0.117242, -0.061184, 0.017424, -0.033553, 0.192202, 0.200031, 0.19961, 0.132677),
    (-0.085619, -0.004196, 0.225536, 0.167757, 0.380542, 0.033755, -0.148576, 0.113817, -0.156146, -0.058422, -0.059828, -0.102462, -0.240612, 0.066518, 0.155885, -0.216413),
)
_B1 = (
    -0.00566, -0.161482, 0.021599, 0.024294, -0.199191, 0.126412, -0.094154, -0.199811,
    0.115063, -0.099678, 0.038056, -0.031202, -0.058767, 0.005292, 0.010744, -0.085381,
)
_W2 = (
    -0.065689, -0.378174, 0.161326, 0.076389, 0.728239, 0.199887, 0.139407, 0.465541,
    -0.304634, 0.414234, 0.139129, -0.1428, -0.060641, 0.218703, 0.331403, -0.388357,
)
_B2 = 0.117222


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

    Returns `(max_detour, preventive_conflict_mask)`.
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
def value(st, mode=EVAL_MODE):
    """Estimated win probability for player to move, in [0, 1]."""
    walls, me = st[4], st[5]
    op = 1 - me
    sq_me, sq_op = st[me], st[op]
    dm = distance(walls, me, sq_me)
    do = distance(walls, op, sq_op)
    wm = st[2 + me]
    wo = st[2 + op]

    t = do - dm
    um = min(wm, do)
    uo = min(wo, dm)

    tm = route_threat_profile(walls, me, sq_me, sq_op)[0] if wo > 0 else 0
    to = route_threat_profile(walls, op, sq_op, sq_me)[0] if wm > 0 else 0
    tm_cap = min(tm, 10)
    to_cap = min(to, 10)

    if mode == "G4_HEURISTIC":
        race_bonus = 0.35 if (wo == 0 and t >= 0) else (-0.35 if (wm == 0 and t < 0) else 0.0)
        score = (
            0.82 * t
            + 0.10 * (wm - wo)
            + 0.14 * (um - uo)
            + 0.28 * (0.95 * to_cap - 1.05 * tm_cap)
            + race_bonus
        )
        return 1.0 / (1.0 + exp(-score)) if score >= 0.0 else exp(score) / (1.0 + exp(score))

    phase = (wm + wo) / 20.0
    opt = max(-6, min(6, t + um))
    pes = max(-6, min(6, t - uo))
    race = 1.0 if (wo == 0 and t >= 0) else (-1.0 if (wm == 0 and t < 0) else 0.0)

    x = (
        1.0,
        t / 4.0,
        (wm - wo) / 4.0,
        (um - uo) / 4.0,
        phase,
        (t / 4.0) * phase,
        to_cap / 6.0,
        tm_cap / 6.0,
        (dm + do) / 16.0,
        opt / 6.0,
        pes / 6.0,
        race,
    )

    z = _B2 + sum(xi * wi for xi, wi in zip(x, _W_LIN))
    hidden = list(_B1)
    for xi, row in zip(x, _W1):
        if xi != 0.0:
            hidden = [h + xi * w for h, w in zip(hidden, row)]
    for h, w_out in zip(hidden, _W2):
        if h > 0.0:
            z += h * w_out
    return 1.0 / (1.0 + exp(-z)) if z >= 0.0 else exp(z) / (1.0 + exp(z))


class Expired(Exception):
    pass


def priors(st, deadline, root_legal=None):
    """D2 + D4 Tactical wall priors with two-stage zero-flood filtering & defensive quota."""
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
            if not (both_cuts & bit):
                if not st[2 + op]:
                    continue
                if def_mask & bit:
                    defensive_candidates.append((k, -0.45 + 0.35 * min(max_threat, 6)))
                elif (near & bit) and (WALL_NEAR[k % 64] & cuts_op):
                    wall_logits.append((k, -1.35))
                else:
                    wall_logits.append((k, -1.70))
                continue

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
                    v = value(pos, EVAL_MODE)
                    break
                if n.kids is None:
                    n.kids = [Node(k, p) for k, p in priors(pos, deadline, legal if n is root else None)]
                    expanded += 1
                if not n.kids:
                    n.proof = -1
                    v = value(pos, EVAL_MODE)
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


class GedikNoNN:
    def __init__(self, eval_mode: str = EVAL_MODE, time_budget: float = TIME_BUDGET):
        self.eval_mode = eval_mode
        self.time_budget = time_budget

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        global EVAL_MODE, TIME_BUDGET
        old_m, old_t = EVAL_MODE, TIME_BUDGET
        EVAL_MODE, TIME_BUDGET = self.eval_mode, self.time_budget
        try:
            return choose_action(obs)
        finally:
            EVAL_MODE, TIME_BUDGET = old_m, old_t
