"""Comprehensive Evaluation, Variant Selection, and Ablation Pipeline for Gedik Quoridor Agents.

Executes:
  1. Correctness & Search Efficiency Profiling across 10 tactical board positions (G0..G5).
  2. AI Variant Selection Screening (comparing sub-variants G1a/G1b/G1c, G2a/G2b, G3a/G3b against G0).
  3. Roadmap Ablation Study (G1b, G2a, G3a, G4, G5 vs G0 and G4/G5 head-to-head) with paired openings
     and swapped P0/P1 seats.
  4. External Opponent Suite Benchmark against 7 distinct local opponents:
     - mcts_agent_v7, hard_waller, wrongway_bot, minimax_agent_summary,
       quoridor_mcts_agent, minimax_tt_cached, and gedik_qnn1 (external reference).
  5. Subprocess Isolation & 1.0-Second Timeout Verification via Arena.agent_runner.StudentAgent.

Saves all raw data and summary tables to `Gedix/evaluation_results.json`.
"""
from collections import Counter
import json
from math import sqrt
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Arena import Environment, Session, run_simulation
from Arena.agent_runner import StudentAgent

import Gedix.g0_baseline as mod_g0
import Gedix.g1_threat_eval as mod_g1
import Gedix.g2_tactical_priors as mod_g2
import Gedix.g3_adaptive_puct as mod_g3
import Gedix.g4_combined as mod_g4
import Gedix.g5_custom_mlp as mod_g5

from Agents.mcts_agent_v7 import choose_action as opp_mcts_v7
import Agents.hard_waller as mod_hw
from Agents.hard_waller import choose_action as opp_hard_waller
from Agents.wrongway_bot import choose_action as opp_wrongway_bot
import Agents.minimax_agent_summary as mod_ms
from Agents.minimax_agent_summary import choose_action as opp_minimax_summary
import Agents.quoridor_mcts_agent as mod_qm
from Agents.quoridor_mcts_agent import choose_action as opp_quoridor_mcts
from Agents.gedik_qnn1 import choose_action as opp_gedik_qnn1
import Agents.gedik_qnn1 as mod_qnn1
import importlib.util


def load_minimax_tt_cached():
    path = ROOT / "temp/AI-Project1/Agents/minimax_tt_cached.py"
    spec = importlib.util.spec_from_file_location("minimax_tt_cached", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



def wilson_ci(wins: int, draws: int, total: int, z: float = 1.96):
    """95% Wilson score interval for score rate (W + 0.5*D) / N."""
    if total <= 0:
        return (0.0, 0.0)
    p = (wins + 0.5 * draws) / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def generate_paired_openings(count: int, plies: int = 6, wall_prob: float = 0.20, base_seed: int = 1000):
    """Generate `count` unique, legal, non-terminal board snapshots after `plies` half-turns."""
    openings = []
    seen = set()
    attempt = 0
    while len(openings) < count and attempt < count * 50:
        seed = base_seed + attempt * 7919
        attempt += 1
        rng = random.Random(seed)
        env = Environment(max_turns=300, starting_player=0)
        ok = True
        for _ in range(plies):
            if env.done:
                ok = False
                break
            obs = env.get_observation()
            moves = obs["legal_moves"]
            action = None
            if rng.random() < wall_prob and obs["legal_walls"]:
                a, b = env.positions
                nearby = [
                    w for w in obs["legal_walls"]
                    if min(abs(w["row"] - a[0]) + abs(w["col"] - a[1]),
                           abs(w["row"] - b[0]) + abs(w["col"] - b[1])) <= 3
                ]
                if nearby:
                    action = {"type": "wall", **rng.choice(nearby)}
            if action is None and moves:
                dm = obs["distances"]
                best = min(dm[r][c] for r, c in moves)
                forward = [m for m in moves if dm[m[0]][m[1]] == best]
                chosen = rng.choice(forward if rng.random() < 0.75 else moves)
                action = {"type": "move", "to": chosen}
            if action is None:
                action = {"type": "pass"}
            step_res = env.step(action)
            if step_res["status"] != "ok":
                ok = False
                break
        if not ok or env.done:
            continue
        snap = env.snapshot()
        key = json.dumps(snap["positions"]) + "|" + json.dumps(snap["walls"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            openings.append((seed, snap))
    return openings


def play_from_snapshot(snap, agent_p0, agent_p1, max_turns=220):
    """Play a game starting from an Environment snapshot with two in-process callables."""
    env = Environment.from_snapshot(snap)
    env.max_turns = max_turns
    agents = [agent_p0, agent_p1]
    timings = [[], []]
    statuses = [Counter(), Counter()]
    actions_seq = []
    while not env.done:
        pid = env.current_player
        obs = env.get_observation(pid)
        t0 = perf_counter()
        try:
            act = agents[pid](obs)
            dt = perf_counter() - t0
            timings[pid].append(dt)
            res = env.step(act)
            statuses[pid][res["status"]] += 1
            actions_seq.append((pid, res["status"], act))
        except Exception as e:
            dt = perf_counter() - t0
            timings[pid].append(dt)
            env.skip_turn("error")
            statuses[pid]["error"] += 1
            actions_seq.append((pid, "error", str(e)))
    return {
        "winner": env.winner,
        "total_turns": env.turn_count,
        "termination_reason": env.termination_reason,
        "timings": timings,
        "statuses": statuses,
        "signature": json.dumps(actions_seq, sort_keys=True),
    }


def run_paired_series(agent_a, agent_b, name_a, name_b, openings, max_turns=220):
    """Play 2 * len(openings) games (swapping P0 and P1 on every opening)."""
    w_a = w_b = draws = 0
    w_a_p0 = w_a_p1 = 0
    p0_games = p1_games = 0
    turns_list = []
    times_a = []
    invalid_a = error_a = 0
    distinct_sigs = set()

    for seed, snap in openings:
        # Game 1: A is P0, B is P1
        g1 = play_from_snapshot(snap, agent_a, agent_b, max_turns=max_turns)
        p0_games += 1
        turns_list.append(g1["total_turns"])
        times_a.extend(g1["timings"][0])
        invalid_a += g1["statuses"][0]["invalid"]
        error_a += g1["statuses"][0]["error"]
        distinct_sigs.add(g1["signature"])
        if g1["winner"] == 0:
            w_a += 1
            w_a_p0 += 1
        elif g1["winner"] == 1:
            w_b += 1
        else:
            draws += 1

        # Game 2: B is P0, A is P1
        g2 = play_from_snapshot(snap, agent_b, agent_a, max_turns=max_turns)
        p1_games += 1
        turns_list.append(g2["total_turns"])
        times_a.extend(g2["timings"][1])
        invalid_a += g2["statuses"][1]["invalid"]
        error_a += g2["statuses"][1]["error"]
        distinct_sigs.add(g2["signature"])
        if g2["winner"] == 1:
            w_a += 1
            w_a_p1 += 1
        elif g2["winner"] == 0:
            w_b += 1
        else:
            draws += 1

    total = w_a + w_b + draws
    score_rate = (w_a + 0.5 * draws) / max(1, total)
    class_points = 3 * w_a + draws
    max_points = 3 * total
    ci_lo, ci_hi = wilson_ci(w_a, draws, total)

    return {
        "matchup": f"{name_a} vs {name_b}",
        "agent_a": name_a,
        "agent_b": name_b,
        "games": total,
        "wins": w_a,
        "draws": draws,
        "losses": w_b,
        "wins_as_p0": f"{w_a_p0}/{p0_games}",
        "wins_as_p1": f"{w_a_p1}/{p1_games}",
        "win_rate": round(w_a / max(1, total), 4),
        "score_rate": round(score_rate, 4),
        "wilson_95_ci": [ci_lo, ci_hi],
        "tournament_points": f"{class_points}/{max_points}",
        "avg_turns": round(statistics.mean(turns_list), 1) if turns_list else 0.0,
        "mean_move_time_s": round(statistics.mean(times_a), 4) if times_a else 0.0,
        "max_move_time_s": round(max(times_a), 4) if times_a else 0.0,
        "invalid_actions": invalid_a,
        "errors": error_a,
        "distinct_games": f"{len(distinct_sigs)}/{total}",
    }


def clear_all_caches():
    for mod in (mod_g0, mod_g1, mod_g2, mod_g3, mod_g4, mod_g5):
        for attr in ("wall_context", "waves", "route", "value", "max_single_wall_threat",
                     "defensive_block_mask", "route_threat_profile"):
            fn = getattr(mod, attr, None)
            if fn is not None and hasattr(fn, "cache_clear"):
                fn.cache_clear()


def profile_variants(openings, budget: float = 0.25):
    """Profile search throughput (simulations/turn, expanded nodes, root kids, move time) across all 9 variants."""
    variants = [
        ("G0_baseline", mod_g0.GedikNoNN(time_budget=budget), mod_g0),
        ("G1a_threat_conservative", mod_g1.GedikG1("G1a", time_budget=budget), mod_g1),
        ("G1b_threat_balanced", mod_g1.GedikG1("G1b", time_budget=budget), mod_g1),
        ("G1c_threat_aggressive", mod_g1.GedikG1("G1c", time_budget=budget), mod_g1),
        ("G2a_priors_def_quota", mod_g2.GedikG2("G2a", time_budget=budget), mod_g2),
        ("G2b_priors_heavy_setup", mod_g2.GedikG2("G2b", time_budget=budget), mod_g2),
        ("G3a_adaptive_puct_mod", mod_g3.GedikG3("G3a", time_budget=budget), mod_g3),
        ("G3b_adaptive_puct_narrow", mod_g3.GedikG3("G3b", time_budget=budget), mod_g3),
        ("G4_combined_d1_d2_d3_d4", mod_g4.GedikG4(time_budget=budget), mod_g4),
        ("G5_custom_mlp", mod_g5.GedikG5(time_budget=budget), mod_g5),
    ]
    results = {}
    for v_name, agent, mod in variants:
        clear_all_caches()
        sims, exps, kids, times = [], [], [], []
        legal_ok = True
        for _, snap in openings:
            env = Environment.from_snapshot(snap)
            obs = env.get_observation()
            t0 = perf_counter()
            act = agent(obs)
            dt = perf_counter() - t0
            step_res = env.step(act)
            if step_res["status"] != "ok":
                legal_ok = False
            stats = mod.LAST_STATS
            sims.append(stats.get("simulations", 0))
            exps.append(stats.get("expanded_nodes", 0))
            kids.append(stats.get("root_kids", 0))
            times.append(dt)
        results[v_name] = {
            "budget_s": budget,
            "positions_tested": len(openings),
            "all_legal": legal_ok,
            "mean_simulations": round(statistics.mean(sims), 1),
            "median_simulations": round(statistics.median(sims), 1),
            "mean_expanded_nodes": round(statistics.mean(exps), 1),
            "mean_root_kids": round(statistics.mean(kids), 1),
            "mean_elapsed_s": round(statistics.mean(times), 4),
            "max_elapsed_s": round(max(times), 4),
        }
        print(f"[PROFILE] {v_name:25s}: sims={results[v_name]['mean_simulations']:6.1f} "
              f"expanded={results[v_name]['mean_expanded_nodes']:5.1f} "
              f"root_kids={results[v_name]['mean_root_kids']:4.1f} "
              f"max_t={results[v_name]['max_elapsed_s']:.4f}s legal={legal_ok}", flush=True)
    return results


def run_subprocess_verification(agent_path: Path, opponent_paths, seed_base: int = 9000):
    """Run real Arena StudentAgent subprocess matches with strict 1.0s timeout to verify 0 timeouts/errors."""
    records = []
    for opp_name, opp_path in opponent_paths:
        w = d = l = 0
        timeouts = invalids = errors = 0
        max_elapsed = 0.0
        total_agent_turns = 0
        for side in (0, 1):
            env = Environment(max_turns=250, starting_player=0)
            p0_path = str(agent_path) if side == 0 else str(opp_path)
            p1_path = str(opp_path) if side == 0 else str(agent_path)
            with StudentAgent(p0_path, timeout=1.0, startup_timeout=5.0, seed=seed_base) as a0, \
                 StudentAgent(p1_path, timeout=1.0, startup_timeout=5.0, seed=seed_base + 1) as a1:
                workers = [a0, a1]
                while not env.done:
                    pid = env.current_player
                    obs = env.get_observation(pid)
                    res = workers[pid].act(obs)
                    if pid == side:
                        total_agent_turns += 1
                        if res.elapsed > max_elapsed:
                            max_elapsed = res.elapsed
                    if res.status == "ok":
                        step = env.step(res.action)
                        if pid == side and step["status"] != "ok":
                            invalids += 1
                    else:
                        if pid == side:
                            if res.status == "timeout":
                                timeouts += 1
                            else:
                                errors += 1
                        env.skip_turn(res.status)
            if env.winner == side:
                w += 1
            elif env.winner is None:
                d += 1
            else:
                l += 1
        records.append({
            "opponent": opp_name,
            "games": 2,
            "wins": w,
            "draws": d,
            "losses": l,
            "agent_turns": total_agent_turns,
            "max_turn_elapsed_s": round(max_elapsed, 4),
            "timeouts": timeouts,
            "invalids": invalids,
            "errors": errors,
        })
        print(f"[SUBPROCESS 1.0s] vs {opp_name:20s}: W={w} D={d} L={l} "
              f"max_t={max_elapsed:.4f}s timeouts={timeouts} invalids={invalids}", flush=True)
    return records


def main():
    # Generate two disjoint sets of paired openings:
    # 1. Development/Screening openings (6 openings -> 12 games per matchup, seed=1000)
    # 2. Final Evaluation openings (8 openings -> 16 games per matchup, plus standard start -> 18 games, seed=5000)
    dev_openings = generate_paired_openings(6, plies=6, wall_prob=0.20, base_seed=1000)
    eval_openings = generate_paired_openings(7, plies=6, wall_prob=0.20, base_seed=5000)
    # Prepend standard official starting board (0 plies) to eval_openings -> 8 openings (16 games per pair)
    official_start = (0, Environment(max_turns=250, starting_player=0).snapshot())
    eval_openings = [official_start] + eval_openings

    print(f"Generated {len(dev_openings)} dev openings and {len(eval_openings)} eval openings.", flush=True)

    # 1. Profile all 10 variants (G0 + 9 developed variants) on 10 positions at 0.25s budget
    probe_positions = eval_openings + dev_openings[:2]
    profile_data = profile_variants(probe_positions, budget=0.25)

    # Use 0.12s per turn for internal PUCT-vs-PUCT screening & ablation series so both sides
    # get identical search budgets while completing ~200 games within a few minutes.
    BUDGET = 0.12
    g0 = mod_g0.GedikNoNN(time_budget=BUDGET)
    g1a = mod_g1.GedikG1("G1a", time_budget=BUDGET)
    g1b = mod_g1.GedikG1("G1b", time_budget=BUDGET)
    g1c = mod_g1.GedikG1("G1c", time_budget=BUDGET)
    g2a = mod_g2.GedikG2("G2a", time_budget=BUDGET)
    g2b = mod_g2.GedikG2("G2b", time_budget=BUDGET)
    g3a = mod_g3.GedikG3("G3a", time_budget=BUDGET)
    g3b = mod_g3.GedikG3("G3b", time_budget=BUDGET)
    g4 = mod_g4.GedikG4(time_budget=BUDGET)
    g5 = mod_g5.GedikG5(time_budget=BUDGET)

    # 2. Stage 1: AI Variant Screening on Dev Openings (12 games per pair)
    print("\n--- STAGE 1: AI VARIANT SCREENING (DEV OPENINGS) ---", flush=True)
    screening_results = []
    for name, agent in [
        ("G1a_Conservative", g1a),
        ("G1b_Balanced", g1b),
        ("G1c_Aggressive", g1c),
        ("G2a_DefQuota_TopK", g2a),
        ("G2b_Unfiltered_Setup", g2b),
        ("G3a_ModWidening_AdaptC", g3a),
        ("G3b_NarrowWidening_LowC", g3b),
    ]:
        clear_all_caches()
        res = run_paired_series(agent, g0, name, "G0_Baseline", dev_openings, max_turns=200)
        screening_results.append(res)
        print(f"[SCREEN] {res['matchup']:38s}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
              f"Score={res['score_rate']*100:5.1f}% (P0:{res['wins_as_p0']}, P1:{res['wins_as_p1']})", flush=True)

    # 3. Stage 2: Roadmap Ablation Tournament on Eval Openings (16 games per pair)
    print("\n--- STAGE 2: ROADMAP ABLATION TOURNAMENT (EVAL OPENINGS) ---", flush=True)
    ablation_results = []
    ablation_pairs = [
        (g1b, g0, "G1b_ThreatEval", "G0_Baseline"),
        (g2a, g0, "G2a_TacticalPriors", "G0_Baseline"),
        (g3a, g0, "G3a_AdaptivePUCT", "G0_Baseline"),
        (g4, g0, "G4_Combined", "G0_Baseline"),
        (g5, g0, "G5_CustomMLP", "G0_Baseline"),
        (g4, g1b, "G4_Combined", "G1b_ThreatEval"),
        (g4, g2a, "G4_Combined", "G2a_TacticalPriors"),
        (g4, g3a, "G4_Combined", "G3a_AdaptivePUCT"),
        (g5, g1b, "G5_CustomMLP", "G1b_ThreatEval"),
        (g4, g5, "G4_Combined", "G5_CustomMLP"),
    ]
    for ag_a, ag_b, na, nb in ablation_pairs:
        clear_all_caches()
        res = run_paired_series(ag_a, ag_b, na, nb, eval_openings, max_turns=200)
        ablation_results.append(res)
        print(f"[ABLATION] {res['matchup']:38s}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
              f"Score={res['score_rate']*100:5.1f}% CI={res['wilson_95_ci']}", flush=True)

    # 4. Stage 3: External Opponent Suite Benchmark
    print("\n--- STAGE 3: EXTERNAL OPPONENT BENCHMARK ---", flush=True)
    mod_tt = load_minimax_tt_cached()
    opp_minimax_tt = mod_tt.choose_action
    # Set equal search budget for time-bounded external bots when running paired multi-game series
    old_qnn_budget = mod_qnn1.TIME_BUDGET
    old_hw_limit = mod_hw.TIME_LIMIT
    old_ms_limit = mod_ms.TIME_LIMIT
    old_qm_limit = mod_qm.TIME_LIMIT
    old_tt_limit = getattr(mod_tt, "TIME_LIMIT", 0.5)
    mod_qnn1.TIME_BUDGET = BUDGET
    mod_hw.TIME_LIMIT = BUDGET
    mod_ms.TIME_LIMIT = BUDGET
    mod_qm.TIME_LIMIT = BUDGET
    if hasattr(mod_tt, "TIME_LIMIT"):
        mod_tt.TIME_LIMIT = BUDGET

    external_opponents = [
        ("mcts_agent_v7", opp_mcts_v7, "Development"),
        ("hard_waller", opp_hard_waller, "Development"),
        ("minimax_agent_summary", opp_minimax_summary, "Development"),
        ("wrongway_bot", opp_wrongway_bot, "Held-out"),
        ("quoridor_mcts_agent", opp_quoridor_mcts, "Held-out"),
        ("minimax_tt_cached", opp_minimax_tt, "Held-out"),
        ("gedik_qnn1", opp_gedik_qnn1, "External-Ref"),
    ]

    external_results = {"G4_Combined": [], "G5_CustomMLP": [], "G0_Baseline": []}
    try:
        for opp_name, opp_fn, split in external_opponents:
            subset_openings = eval_openings[:8] if opp_name not in ("wrongway_bot",) else eval_openings[:4]
            for our_name, our_agent in [("G4_Combined", g4), ("G5_CustomMLP", g5), ("G0_Baseline", g0)]:
                clear_all_caches()
                res = run_paired_series(our_agent, opp_fn, our_name, opp_name, subset_openings, max_turns=200)
                res["split"] = split
                external_results[our_name].append(res)
                print(f"[EXT-{split:12s}] {res['matchup']:38s}: W={res['wins']:2d} D={res['draws']:2d} L={res['losses']:2d} "
                      f"Score={res['score_rate']*100:5.1f}% (P0:{res['wins_as_p0']}, P1:{res['wins_as_p1']})", flush=True)
    finally:
        mod_qnn1.TIME_BUDGET = old_qnn_budget
        mod_hw.TIME_LIMIT = old_hw_limit
        mod_ms.TIME_LIMIT = old_ms_limit
        mod_qm.TIME_LIMIT = old_qm_limit
        if hasattr(mod_tt, "TIME_LIMIT"):
            mod_tt.TIME_LIMIT = old_tt_limit

    # 5. Stage 4: Official 1.0-Second Subprocess Verification (StudentAgent) at full 0.78s budget
    print("\n--- STAGE 4: OFFICIAL 1.0s SUBPROCESS VERIFICATION (FULL BUDGET) ---", flush=True)
    subproc_opponents = [
        ("mcts_agent_v7", ROOT / "Agents/mcts_agent_v7.py"),
        ("hard_waller", ROOT / "Agents/hard_waller.py"),
        ("wrongway_bot", ROOT / "Agents/wrongway_bot.py"),
        ("minimax_tt_cached", ROOT / "temp/AI-Project1/Agents/minimax_tt_cached.py"),
        ("g0_baseline", ROOT / "Gedix/g0_baseline.py"),
    ]
    subproc_g4 = run_subprocess_verification(ROOT / "Gedix/g4_combined.py", subproc_opponents)
    subproc_g5 = run_subprocess_verification(ROOT / "Gedix/g5_custom_mlp.py", subproc_opponents)

    out_file = ROOT / "Gedix/evaluation_results.json"
    output_payload = {
        "profile_025s": profile_data,
        "stage1_variant_screening": screening_results,
        "stage2_ablation_tournament": ablation_results,
        "stage3_external_opponents": external_results,
        "stage4_subprocess_1s_verification": {
            "G4_Combined": subproc_g4,
            "G5_CustomMLP": subproc_g5,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\nSaved all Stage 1-4 results to Gedix/evaluation_results.json", flush=True)


if __name__ == "__main__":
    main()
