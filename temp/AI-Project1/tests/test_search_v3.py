"""Smoke/correctness checks for V2 vs V3 and teacher labels.
Place: AI-Project1/tests/test_search_v3.py
Run: python tests/test_search_v3.py
"""
from pathlib import Path
import importlib.util
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Arena.environment import Environment


def import_agent(name):
    path = ROOT / 'Agents' / (name + '.py')
    spec = importlib.util.spec_from_file_location('test_v3_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_state(mod, obs):
    return mod.State(
        pos=(tuple(obs['you']['position']), tuple(obs['opponent']['position'])),
        goals=(obs['you']['goal_row'], obs['opponent']['goal_row']),
        walls_rem=(obs['you']['walls_remaining'], obs['opponent']['walls_remaining']),
        h_walls=frozenset(map(tuple, obs['walls']['horizontal'])),
        v_walls=frozenset(map(tuple, obs['walls']['vertical'])),
    )


def legal(obs, act):
    if act['type'] == 'move':
        return act['to'] in obs['legal_moves']
    if act['type'] == 'wall':
        return any(all(act[k] == wall[k] for k in ('orientation', 'row', 'col'))
                   for wall in obs['legal_walls'])
    return False


def run():
    base = import_agent('minimax_wallfilter_ordered')
    tt = import_agent('minimax_tt')
    rng = random.Random(4404)
    observations = []
    for game in range(3):
        env = Environment(max_turns=90)
        for step in range(8):
            moves = env.legal_moves()
            # Mix some walls and pawn moves, but always use the Arena oracle.
            obs = env.get_observation()
            if obs['legal_walls'] and rng.random() < 0.35:
                wall = rng.choice(obs['legal_walls'])
                action = {'type': 'wall', **wall}
            else:
                action = {'type': 'move', 'to': list(rng.choice(moves))}
            env.step(action)
        observations.append(env.get_observation())
        # Check P1 perspective as well.
        obs = env.get_observation()
        env.step({'type': 'move', 'to': obs['legal_moves'][0]})
        observations.append(env.get_observation())

    for idx, obs in enumerate(observations):
        for depth in (1, 2):
            scores = []
            for module in (base, tt):
                module.reset_search_stats()
                module.SEARCH_ROOT_DEPTH = depth
                state = make_state(module, obs)
                score, action = module.minimax(
                    state, depth, -math.inf, math.inf, 0,
                    time.monotonic() + 10.0,
                    is_root=True, recent_counts=None,
                )
                assert legal(obs, action), (idx, depth, module.__name__, action)
                scores.append(score)
            assert math.isclose(*scores, rel_tol=0, abs_tol=1e-8), (idx, depth, scores)
        # Timed actual agent action (cannot rely on equal choice with time-limited trees).
        tt.reset_search_stats()
        tt.RECENT_POSITIONS.clear()
        action = tt.choose_action(obs)
        assert legal(obs, action), (idx, action)
    print(f'PASS: {len(observations)} P0/P1 POV positions; depth 1-2 scores match V2; V3 returns legal moves')


if __name__ == '__main__':
    run()
