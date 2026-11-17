"""Generate outcome-labeled Wrongway states from matches among existing agents.

Place at: AI-Project1/training/generate_data.py
Run from the repository root. Python 3.9+ and standard library only.

NOTE: Session() executes agent functions IN-PROCESS and does not impose the
Arena's hard 1-second timeout. Use tests/benchmark_agents.py for actual timing.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Arena import Session  # noqa: E402

DEFAULT_AGENTS = (
    "minimax_agent_summary", "hard_waller", "wrongway_bot",
    "bfs_heuristic", "mcts_agent-v7", "quoridor_mcts_agent",
)


def load_agent(name):
    """Import Agents/<name>.py, including names containing a hyphen."""
    path = ROOT / "Agents" / (name + ".py")
    if not path.is_file():
        raise FileNotFoundError(f"Agent file does not exist: {path}")
    module_name = "wrongway_training_" + name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import agent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "choose_action", None)
    if not callable(fn):
        raise ValueError(f"Agent {name} has no choose_action(observation)")
    return fn


def compact_state(obs):
    """Capture PRE-ACTION, POV-specific state. Drop large derived legal_walls."""
    return {
        "player_id": obs["player_id"],
        "turn": obs["turn"],
        "max_turns": obs["max_turns"],
        "you": dict(obs["you"]),
        "opponent": dict(obs["opponent"]),
        "walls": {
            "horizontal": [list(w) for w in obs["walls"]["horizontal"]],
            "vertical": [list(w) for w in obs["walls"]["vertical"]],
        },
    }


def sample_opening_action(obs, rng, wall_prob):
    """Seeded exploratory opening to avoid repeated identical match trajectories."""
    walls = obs.get("legal_walls", [])
    moves = obs.get("legal_moves", [])
    if walls and rng.random() < wall_prob:
        wall = rng.choice(walls)
        return {"type": "wall", "orientation": wall["orientation"],
                "row": wall["row"], "col": wall["col"]}
    if moves:
        return {"type": "move", "to": list(rng.choice(moves))}
    return {"type": "pass"}


def play_game(p0, p1, agents, game_id, game_seed, args):
    random.seed(game_seed)  # agents that call Python's random also get a fresh seed
    rng = random.Random(game_seed + 1000003)
    session = Session(max_turns=args.max_turns, names=[p0, p1])
    rows, action_sequence, latencies = [], [], []
    problem = None
    while not session.done:
        obs = session.observe()  # IMPORTANT: state BEFORE the action!
        pid = obs["player_id"]
        name = (p0, p1)[pid]
        policy = "agent"
        start = time.perf_counter()
        try:
            if obs["turn"] < args.opening_random_plies:
                action = sample_opening_action(obs, rng, args.opening_wall_prob)
                policy = "opening_random"
            else:
                action = agents[name](obs)
        except Exception as exc:
            problem = f"{name} crashed: {type(exc).__name__}: {exc}"
            break
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        # Invalid actions would forfeit a turn; do not train on corrupt games.
        if not isinstance(action, dict):
            problem = f"{name} returned non-dict: {action!r}"
            break
        record = session.step(action)
        if record.status not in ("ok", "pass"):
            problem = f"{name} returned {record.status}: {record.reason}"
            break
        action_sequence.append({"player": pid, "action": action})
        rows.append({
            "game_id": game_id,
            "state": compact_state(obs),
            "player_id": pid,
            "agent": name,
            "policy": policy,
            "action": action,
        })

    if problem:
        return [], {
            "game_id": game_id, "p0": p0, "p1": p1,
            "seed": game_seed, "valid": False, "error": problem,
            "turns": len(rows),
        }

    result = session.result
    winner = result.winner  # None when drawn at the full turn limit
    for row in rows:
        row["outcome"] = (0 if winner is None else
                          (1 if row["player_id"] == winner else -1))
    trace_data = json.dumps(
        {"p0": p0, "p1": p1, "moves": action_sequence},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    trace_hash = hashlib.sha256(trace_data).hexdigest()
    return rows, {
        "game_id": game_id, "p0": p0, "p1": p1,
        "seed": game_seed, "valid": True,
        "winner": winner, "termination_reason": result.termination_reason,
        "turns": result.total_turns, "trace_hash": trace_hash,
        "max_decision_seconds": max(latencies, default=0.0),
        "over_one_second_calls": sum(t > 1.0 for t in latencies),
    }


def schedule(names, games_per_pair, self_play, seed):
    """Round-robin matchups. Each repeated pair alternates player sides."""
    pairs = list(itertools.combinations_with_replacement(names, 2) if self_play
                 else itertools.combinations(names, 2))
    if not pairs:
        raise ValueError("Need >=2 agents when --no-self-play is given")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    for pair_index, pair in enumerate(pairs):
        for k in range(games_per_pair):
            a, b = pair if k % 2 == 0 else (pair[1], pair[0])
            yield a, b


def main():
    parser = argparse.ArgumentParser(description="Generate Wrongway outcome-labeled data")
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS),
                        help="Comma-separated agent file stems, without .py")
    parser.add_argument("--games-per-pair", type=int, default=2,
                        help="Even number so each matchup switches sides")
    parser.add_argument("--no-self-play", action="store_true",
                        help="Only distinct-agent matchups")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Optional cap for a quick trial")
    parser.add_argument("--opening-random-plies", type=int, default=6)
    parser.add_argument("--opening-wall-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--out", default="training/datasets/v1")
    args = parser.parse_args()
    names = [s.strip().removesuffix(".py") for s in args.agents.split(",") if s.strip()]
    if len(names) != len(set(names)) or not names:
        parser.error("--agents must have unique nonempty names")
    if args.games_per_pair <= 0 or args.games_per_pair % 2:
        parser.error("--games-per-pair must be a positive even integer")
    if args.max_games is not None and args.max_games <= 0:
        parser.error("--max-games must be positive")
    if args.max_turns <= 0 or args.opening_random_plies < 0:
        parser.error("max-turns must be positive; opening-random-plies nonnegative")
    if not 0 <= args.opening_wall_prob <= 1:
        parser.error("--opening-wall-prob must be in [0,1]")

    agents = {name: load_agent(name) for name in names}
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    file_data = out / "states.jsonl"
    file_games = out / "games.jsonl"
    seen_traces = set()
    summary = {
        "config": {"agents": names, "games_per_pair": args.games_per_pair,
                   "opening_random_plies": args.opening_random_plies,
                   "opening_wall_prob": args.opening_wall_prob,
                   "max_turns": args.max_turns, "seed": args.seed},
        "games_played": 0, "games_saved": 0, "duplicate_games_skipped": 0,
        "invalid_games_skipped": 0, "samples_saved": 0,
        "label_counts": {"-1": 0, "0": 0, "1": 0},
        "games_over_one_second": 0,
    }
    if args.max_turns < 500:
        print("WARNING: max-turns < 500 may label prematurely truncated games as draws.", flush=True)

    # Output files are newly written for this run; do not use the same --out
    # for separate datasets that must be merged later.
    with file_data.open("w", encoding="utf-8") as states_out, \
         file_games.open("w", encoding="utf-8") as games_out:
        for game_id, (p0, p1) in enumerate(
                schedule(names, args.games_per_pair, not args.no_self_play, args.seed), 1):
            if args.max_games is not None and game_id > args.max_games:
                break
            samples, info = play_game(p0, p1, agents, game_id,
                                      args.seed + 1009 * game_id, args)
            summary["games_played"] += 1
            if not info["valid"]:
                summary["invalid_games_skipped"] += 1
            elif info["trace_hash"] in seen_traces:
                info["duplicate"] = True
                summary["duplicate_games_skipped"] += 1
            else:
                seen_traces.add(info["trace_hash"])
                info["duplicate"] = False
                summary["games_saved"] += 1
                if info["over_one_second_calls"]:
                    summary["games_over_one_second"] += 1
                for sample in samples:
                    states_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    summary["samples_saved"] += 1
                    summary["label_counts"][str(sample["outcome"])] += 1
            games_out.write(json.dumps(info, ensure_ascii=False) + "\n")
            print(f'Game {game_id}: {p0} vs {p1}, turns={info["turns"]}, '
                  f'valid={info["valid"]}, saved={info.get("duplicate") is False}, '
                  f'samples_total={summary["samples_saved"]}', flush=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone:", out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("NOTE: split train/validation/test by game_id, NEVER by individual rows.")


if __name__ == "__main__":
    main()
