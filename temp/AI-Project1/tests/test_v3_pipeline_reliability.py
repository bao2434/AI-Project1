"""Non-destructive v3 pipeline regression checks (no Arena tournament)."""
from pathlib import Path
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Arena.environment import Environment


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'Agents' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def compact(obs):
    return {'player_id': obs['player_id'], 'turn': obs['turn'],
            'max_turns': obs['max_turns'],
            'you': dict(obs['you']), 'opponent': dict(obs['opponent']),
            'walls': {'horizontal': [list(w) for w in obs['walls']['horizontal']],
                      'vertical': [list(w) for w in obs['walls']['vertical']]}}


def make_data(path, games, states_per_game=4):
    rng = random.Random(3011)
    rows = []
    for i in range(games):
        env = Environment(max_turns=90)
        for step in range(states_per_game):
            obs = env.get_observation()
            rows.append({'game_id': f'game_{i:04d}', 'outcome': (-1 if i % 2 else 1),
                         'state': compact(obs)})
            if obs['legal_walls'] and rng.random() < .24:
                action = {'type': 'wall', **rng.choice(obs['legal_walls'])}
            else:
                action = {'type': 'move', 'to': rng.choice(obs['legal_moves'])}
            env.step(action)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    return rows


def call(script, args, expect_ok):
    command = [sys.executable, str(ROOT / 'training' / script), *args]
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    if (result.returncode == 0) != expect_ok:
        print('STDOUT:', result.stdout[-2500:])
        print('STDERR:', result.stderr[-2500:])
        raise AssertionError(f'Unexpected return code {result.returncode}: {command}')
    return result


def main():
    v3 = module('minimax_tt')
    cached = module('minimax_tt_cached')
    rng = random.Random(16)
    for i in range(4):
        env = Environment(max_turns=100)
        for _ in range(8 + i):
            obs = env.get_observation()
            if obs['legal_walls'] and rng.random() < .3:
                a = {'type':'wall', **rng.choice(obs['legal_walls'])}
            else:
                a = {'type':'move', 'to': rng.choice(obs['legal_moves'])}
            env.step(a)
        obs = env.get_observation()
        if env.done:
            continue
        scores = []
        for agent in (v3, cached):
            s = agent.State(
                pos=(tuple(obs['you']['position']), tuple(obs['opponent']['position'])),
                goals=(obs['you']['goal_row'], obs['opponent']['goal_row']),
                walls_rem=(obs['you']['walls_remaining'], obs['opponent']['walls_remaining']),
                h_walls=frozenset(map(tuple,obs['walls']['horizontal'])),
                v_walls=frozenset(map(tuple,obs['walls']['vertical'])),
            )
            agent.reset_search_stats(); agent.SEARCH_ROOT_DEPTH = 2
            score, action = agent.minimax(s,2,-math.inf,math.inf,0,time.monotonic()+10,is_root=True)
            scores.append(score)
            assert action is not None and (action['type'] != 'move' or action['to'] in obs['legal_moves'])
        assert scores[0] == scores[1], (i, scores)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        small, large = tmp/'input'/'old'/'states.jsonl', tmp/'input'/'full'/'states.jsonl'
        make_data(small,4)
        make_data(large,8)
        output = tmp/'output'/'states.jsonl'
        call('recover_training_data_v3.py', ['--scan-dir',str(tmp/'input'), '--out',str(output),
             '--min-games','7', '--min-states','28'], True)
        assert len({json.loads(x)['game_id'] for x in output.read_text().splitlines()}) == 8
        labels = tmp/'labels'
        # Refuse tiny source BEFORE making any changes to existing labels.
        labels.mkdir(parents=True)
        original = labels/'states_search.jsonl'
        original.write_text('KEEP_EXISTING_DATA\n',encoding='utf-8')
        call('generate_search_targets_v3.py', ['--data',str(small),'--out',str(labels)], False)
        assert original.read_text() == 'KEEP_EXISTING_DATA\n'
        call('generate_search_targets_v3.py', ['--data',str(output),'--out',str(labels),
             '--per-game','3','--max-states','24', '--max-depth','2','--min-depth','1',
             '--time-limit','0.15','--min-games','7','--min-selected','20',
             '--min-labels','16','--min-labeled-games','6'], True)
        labelled = [json.loads(x) for x in original.read_text().splitlines()]
        assert len(labelled) >= 16 and len({str(x['game_id']) for x in labelled}) >= 6
        assert all(-1 <= x['teacher_value'] <= 1 for x in labelled)
        train_out = tmp/'model'
        call('train_value_search_v3.py', ['--data',str(original),'--out',str(train_out),
             '--epochs','2','--batch-size','16'], False)
        assert not (train_out/'best_model.pt').exists(), 'A tiny dataset created a checkpoint!'
        # Explicitly allow smoke-only training to verify checkpoint compatibility.
        call('train_value_search_v3.py', ['--data',str(original),'--out',str(train_out),
             '--epochs','2','--batch-size','16','--allow-small-data'], True)
        assert (train_out/'best_model.pt').is_file()
        metrics=json.loads((train_out/'metrics.json').read_text())
        assert metrics['smoke_only'] is True
    print('PASS: 4 POV searches, source recovery, tiny-data preflight, '
          'atomic labels, teacher generation, strict training and smoke checkpoint')


if __name__ == '__main__':
    main()
