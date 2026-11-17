"""Paired randomized-opening benchmark for Wrongway hybrid agents.

Place: AI-Project1/tests/benchmark_hybrid_openings.py
Run from the repo root, after training/build_hybrid_agents.py.

Each opening is generated ONCE with a fixed seed, then played twice with
hybrid and baseline swapping P0/P1. The board, walls and current player are
identical within each pair. The Arena's real StudentAgent subprocess runner is
used with per-action timeouts. The original benchmark_agents.py seeds agent
workers, but does not randomize the initial board; this script DOES.

NOTE: A paired experiment with 4 openings has 8 matches per lambda. This is a
smoke/diagnostic test, not a statistically reliable win-rate estimate.
"""

import argparse
from decimal import Decimal, InvalidOperation
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Arena.environment import Environment  # noqa: E402
from Arena.agent_runner import StudentAgent  # noqa: E402


def resolve(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def opening_action(env, rng, wall_prob):
    """Create legal, varied but reasonably directed opening actions.

    Bias pawn actions toward reaching the goal, while allowing occasional
    lateral/backward moves. Prefer walls near either pawn to avoid unrealistic
    random corners. This is a test distribution, NOT the game's normal start.
    """
    obs = env.get_observation()
    moves = obs["legal_moves"]
    if rng.random() < wall_prob:
        walls = obs["legal_walls"]
        if walls:
            a, b = env.positions
            nearby = [w for w in walls if min(
                abs(w["row"] - a[0]) + abs(w["col"] - a[1]),
                abs(w["row"] - b[0]) + abs(w["col"] - b[1]),
            ) <= 4]
            if nearby:
                w = rng.choice(nearby)
                return {"type": "wall", **w}
    if moves:
        # Use shortest-path distance to own goal to bias forward moves.
        dm = obs["distances"]
        best = min(dm[r][c] for r, c in moves)
        progress = [m for m in moves if dm[m[0]][m[1]] == best]
        chosen = rng.choice(progress if rng.random() < 0.75 else moves)
        return {"type": "move", "to": chosen}
    return {"type": "pass"}


def make_opening(seed, plies, wall_prob, max_turns):
    rng = random.Random(seed)
    env = Environment(max_turns=max_turns, starting_player=0)
    actions = []
    for _ in range(plies):
        if env.done:
            break
        a = opening_action(env, rng, wall_prob)
        result = env.step(a)
        if result["status"] != "ok":
            raise RuntimeError("Opening action was not legal: " + repr(a))
        actions.append(a)
    if env.done:
        raise RuntimeError("Opening already ended the game; reduce opening plies")
    return env.snapshot(), actions


def one_match(opening, paths, seed, timeout, startup_timeout):
    """Play from saved opening; paths[0] is P0 agent and paths[1] is P1."""
    env = Environment.from_snapshot(opening)
    counts = [Counter({s: 0 for s in ("ok", "pass", "invalid", "timeout", "error")})
              for _ in range(2)]
    timings = [[], []]
    action_types = [Counter(), Counter()]
    with ExitStack() as stack:
        workers = [stack.enter_context(StudentAgent(
            paths[side], timeout=timeout, startup_timeout=startup_timeout,
            seed=seed * 2 + side,
        )) for side in (0, 1)]
        while not env.done:
            side = env.current_player
            obs = env.get_observation(side)
            result = workers[side].act(obs)
            timings[side].append(result.elapsed)
            if result.status == "ok":
                action_result = env.step(result.action)
                status = action_result["status"]
                act = action_result.get("action")
                if isinstance(act, dict):
                    action_types[side][act.get("type", "unknown")] += 1
            else:
                status = result.status
                env.skip_turn(status)
            counts[side][status] += 1
    return {
        "winner": env.winner,
        "termination_reason": env.termination_reason,
        "turns_total": env.turn_count,
        "turns_after_opening": env.turn_count - opening["turn"],
        "statuses": [dict(c) for c in counts],
        "action_types": [dict(c) for c in action_types],
        "time_mean": [statistics.mean(t) if t else 0.0 for t in timings],
        "time_max": [max(t) if t else 0.0 for t in timings],
        "actions_timed": [len(t) for t in timings],
    }


def parse_lambdas(text):
    """Same lambda-to-filename mapping as training/build_hybrid_agents.py."""
    parsed = []
    used = set()
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            raise ValueError("Empty lambda; use --lambdas 0.25,0.75")
        try:
            lam = Decimal(raw)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid lambda: {raw!r}") from exc
        if not lam.is_finite() or not 0 <= lam <= 10:
            raise ValueError("Lambda must be a finite number between 0 and 10")
        if lam * 100 != (lam * 100).to_integral_value():
            raise ValueError("Use at most 2 decimal places for lambda")
        if lam in used:
            raise ValueError(f"Duplicate lambda: {raw}")
        used.add(lam)
        if lam * 10 == (lam * 10).to_integral_value():
            tag = f"{int(lam * 10):02d}"
        else:
            tag = f"{int(lam * 100):03d}"
        parsed.append((lam, tag))
    if not parsed:
        raise ValueError("Provide at least one lambda")
    return parsed


def run(args):
    baseline = resolve(args.baseline)
    selected = parse_lambdas(args.lambdas)
    if any(lam == 0 for lam, _ in selected):
        raise ValueError("--lambdas must contain nonzero challengers (0 is the baseline)")
    variant_paths = [
        (float(lam), resolve(args.agent_pattern.format(lambda_tag=tag)))
        for lam, tag in selected
    ]
    if args.openings < 1 or args.opening_plies < 0 or args.opening_plies % 2:
        raise ValueError("--openings >= 1 and --opening-plies must be even >=0")
    if not 0.0 <= args.wall_prob <= 1.0:
        raise ValueError("--wall-prob must be in [0,1]")
    if not 0 < args.timeout <= 1.0:
        raise ValueError("0 < --timeout <= 1.0 is required")
    if args.opening_plies >= args.max_turns:
        raise ValueError("--max-turns must exceed --opening-plies")

    openings = []
    seen_snapshots = set()
    attempts = 0
    while len(openings) < args.openings and attempts < max(100, 100 * args.openings):
        seed = args.seed + attempts * 7919
        attempts += 1
        snapshot, actions = make_opening(seed, args.opening_plies, args.wall_prob, args.max_turns)
        signature = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        if signature in seen_snapshots:
            continue
        seen_snapshots.add(signature)
        openings.append((seed, snapshot, actions))
    if len(openings) != args.openings:
        raise RuntimeError(
            f"Could only generate {len(openings)} unique openings of {args.openings} requested. "
            "Try --opening-plies 8 or increase --wall-prob."
        )

    results = {
        "experiment": "paired_seeded_openings_hybrid_vs_lambda0",
        "baseline": str(baseline),
        "config": {
            "openings": args.openings, "opening_plies": args.opening_plies,
            "wall_prob": args.wall_prob, "timeout": args.timeout,
            "max_turns": args.max_turns, "seed": args.seed,
        },
        "notes": (
            "Each opening is played twice with exact same board and swapped agent sides. "
            "The seeded opening policy is a deliberately randomized test distribution. "
            "Do not interpret repeated games as independent if they share openings."
        ),
        "comparisons": [],
    }

    for lam, challenger in variant_paths:
        stats = Counter()
        game_logs = []
        timings = []
        print(f"\n=== lambda={lam:g} vs lambda=0 | {2*len(openings)} paired games ===", flush=True)
        for opening_id, (opening_seed, opening, actions) in enumerate(openings, start=1):
            for challenger_side in (0, 1):
                sides = [str(challenger), str(baseline)] if challenger_side == 0 else [str(baseline), str(challenger)]
                game = one_match(opening, sides,
                                 seed=args.seed + opening_id * 11,
                                 timeout=args.timeout,
                                 startup_timeout=args.startup_timeout)
                winner = game["winner"]
                outcome = "draw" if winner is None else ("win" if winner == challenger_side else "loss")
                stats[outcome] += 1
                ch_statuses = game["statuses"][challenger_side]
                for status, count in ch_statuses.items():
                    stats["status_" + status] += count
                timings.append((game["time_mean"][challenger_side],
                                game["actions_timed"][challenger_side],
                                game["time_max"][challenger_side]))
                game_logs.append({
                    "opening_id": opening_id,
                    "opening_seed": opening_seed,
                    "opening_actions": actions,
                    "opening_snapshot": opening,
                    "challenger_side": challenger_side,
                    "outcome": outcome,
                    **game,
                })
                print(f"Opening {opening_id}/{len(openings)} | P{challenger_side} "
                      f"{outcome.upper()} | t={game['turns_after_opening']} "
                      f"| timeout={ch_statuses['timeout']} "
                      f"| invalid={ch_statuses['invalid']} "
                      f"| error={ch_statuses['error']}", flush=True)
        n = len(game_logs)
        total_turns = sum(t[1] for t in timings)
        mean = sum(t[0] * t[1] for t in timings) / total_turns if total_turns else None
        latency_max = max((t[2] for t in timings), default=None)
        entry = {
            "lambda": lam,
            "challenger": str(challenger),
            "games": n,
            "wins": stats["win"], "draws": stats["draw"], "losses": stats["loss"],
            "win_rate": stats["win"] / n,
            "nonloss_rate": (stats["win"] + stats["draw"]) / n,
            "statuses": {key.replace("status_", ""): stats[key]
                         for key in ("status_ok", "status_pass", "status_invalid", "status_timeout", "status_error")},
            "latency_mean": mean,
            "latency_max": latency_max,
            "moves_timed": total_turns,
            "games_detail": game_logs,
        }
        results["comparisons"].append(entry)
        print(f"SUMMARY lambda={lam:g}: W/D/L={stats['win']}/{stats['draw']}/{stats['loss']} "
              f"win_rate={stats['win']/n:.1%} avg={mean:.3f}s "
              f"max={latency_max:.3f}s", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="Agents/minimax_hybrid_l00.py")
    ap.add_argument("--agent-pattern", default="Agents/minimax_hybrid_l{lambda_tag}.py")
    ap.add_argument("--lambdas", default="0.25,0.75",
                    help="Comma-separated lambda values with at most 2 decimals, e.g. 0.25,0.75")
    ap.add_argument("--openings", type=int, default=4,
                    help="Number of distinct randomized initial positions; 2 games each per lambda")
    ap.add_argument("--opening-plies", type=int, default=6,
                    help="Even number of seed-controlled opening plies")
    ap.add_argument("--wall-prob", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument("--startup-timeout", type=float, default=6.0)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="tests/hybrid_openings_results.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
