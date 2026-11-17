"""Distill a stronger Wrongway Minimax teacher into CNN-ready state-value labels.

Place: AI-Project1/training/generate_search_targets_v3.py
Run from the repository root; no PyTorch required.

Input is the existing training/generate_data.py states.jsonl (PRE-ACTION states
from the acting player's POV). Unlike game-outcome labels, the new target is
an evaluation of THAT state by a depth-limited Alpha-Beta teacher.

Crucial: teacher values are a proxy for strategic quality, NOT ground truth.
The default --scale 40 converts the Minimax heuristic score to [-1,+1] using
 tanh(score/scale). Terminal scores are saturated to exactly -1/+1.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import time

ROOT = Path(__file__).resolve().parent.parent


def load_teacher(path):
    spec = importlib.util.spec_from_file_location('wrongway_search_teacher_v3', str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import teacher: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('State', 'minimax', 'reset_search_stats', 'SearchTimeout'):
        if not hasattr(module, name):
            raise TypeError(f'Teacher must provide {name}: {path}')
    return module


def to_search_state(teacher, compact):
    """Convert POV-specific recorded state to the teacher's local MAX/other MIN."""
    you, opp = compact['you'], compact['opponent']
    walls = compact['walls']
    return teacher.State(
        pos=(tuple(you['position']), tuple(opp['position'])),
        goals=(you['goal_row'], opp['goal_row']),
        walls_rem=(you['walls_remaining'], opp['walls_remaining']),
        h_walls=frozenset(map(tuple, walls['horizontal'])),
        v_walls=frozenset(map(tuple, walls['vertical'])),
    )


def normalize_score(score, scale):
    if score >= 1000:
        return 1.0
    if score <= -1000:
        return -1.0
    return math.tanh(score / scale)


def select_rows(rows, per_game, seed, max_states, max_games):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row['game_id'])].append(row)
    game_ids = sorted(grouped)
    random.Random(seed).shuffle(game_ids)
    if max_games:
        game_ids = game_ids[:max_games]
    chosen = []
    for gid in game_ids:
        candidates = sorted(grouped[gid], key=lambda x: int(x['state']['turn']))
        if len(candidates) > per_game:
            # Evenly sample from the full game, not only its opening.
            indices = [((2 * i + 1) * len(candidates)) // (2 * per_game)
                       for i in range(per_game)]
            candidates = [candidates[i] for i in indices]
        chosen.extend(candidates)
    random.Random(seed + 13).shuffle(chosen)
    return chosen[:max_states] if max_states else chosen


def label_one(teacher, row, args):
    search_state = to_search_state(teacher, row['state'])
    teacher.reset_search_stats()  # Fresh table/search statistics per position
    deadline = time.monotonic() + args.time_limit
    best = None
    timed_out = False
    for depth in range(1, args.max_depth + 1):
        teacher.SEARCH_ROOT_DEPTH = depth
        try:
            score, action = teacher.minimax(
                search_state, depth, -math.inf, math.inf,
                player=0, deadline=deadline, is_root=True, recent_counts=None,
            )
            best = (score, action, depth)
        except teacher.SearchTimeout:
            timed_out = True
            break
    if best is None or best[2] < args.min_depth:
        return None, timed_out
    score, action, completed_depth = best
    output = dict(row)  # preserve outcome + game_id for later game-level split
    output.update({
        'teacher_score': round(float(score), 7),
        'teacher_value': round(normalize_score(score, args.scale), 7),
        'teacher_action': action,
        'teacher_depth': completed_depth,
        'teacher_timed_out': timed_out,
        'teacher_nodes': teacher.SEARCH_STATS.get('nodes', 0),
        'teacher_bfs_calls': teacher.SEARCH_STATS.get('bfs_calls', 0),
        'teacher_tt_hits': teacher.SEARCH_STATS.get('tt_hits', 0),
    })
    return output, timed_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='training/datasets/v1/states.jsonl')
    p.add_argument('--teacher', default='Agents/minimax_tt.py')
    p.add_argument('--out', default='training/datasets/search_v3')
    p.add_argument('--per-game', type=int, default=12,
                   help='Sample at most N states, spread over each game')
    p.add_argument('--max-states', type=int, default=800,
                   help='Overall label budget; use 0 for all sampled states')
    p.add_argument('--max-games', type=int, default=0, help='0 means every game')
    p.add_argument('--max-depth', type=int, default=3)
    p.add_argument('--min-depth', type=int, default=1,
                   help='Skip labels when the last fully completed depth is too shallow')
    p.add_argument('--time-limit', type=float, default=0.50,
                   help='Per-sample teacher budget in seconds, independent of Arena limit')
    p.add_argument('--scale', type=float, default=40.0,
                   help='tanh(teacher_score/scale) for nonterminal scores')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-games', type=int, default=30,
                   help='Required number of independent games BEFORE labeling')
    p.add_argument('--min-selected', type=int, default=300,
                   help='Required sampled states BEFORE labeling')
    p.add_argument('--min-labels', type=int, default=250,
                   help='Minimum saved labels to accept the dataset')
    p.add_argument('--min-labeled-games', type=int, default=25,
                   help='Minimum independent games represented in saved labels')
    args = p.parse_args()
    if (args.per_game < 1 or args.max_states < 0 or args.max_games < 0
            or args.max_depth < 1 or not 1 <= args.min_depth <= args.max_depth
            or args.time_limit <= 0 or args.scale <= 0
            or args.min_games < 3 or args.min_selected < 1
            or args.min_labels < 1 or args.min_labeled_games < 3):
        p.error('Invalid sampling/depth/budget/scale parameters')
    data_path, teacher_path = Path(args.data), Path(args.teacher)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not teacher_path.is_absolute():
        teacher_path = ROOT / teacher_path
    if not data_path.is_file() or not teacher_path.is_file():
        p.error(f'Missing data or teacher: {data_path} / {teacher_path}')
    teacher = load_teacher(teacher_path)
    original_rows = []
    with data_path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not all(k in row for k in ('game_id', 'state', 'outcome')):
                    raise ValueError(f'Missing game_id/state/outcome at line {line_number}')
                original_rows.append(row)
    unique_source_games = len({str(row['game_id']) for row in original_rows})
    print(f'SOURCE: {len(original_rows)} states from {unique_source_games} independent game IDs', flush=True)
    if unique_source_games < args.min_games:
        raise ValueError(f'Source has only {unique_source_games} games; need >= {args.min_games}. '
                         'Do not train on this file. Restore the larger v1 dataset from Kaggle Input.')
    selected = select_rows(original_rows, args.per_game, args.seed,
                           args.max_states, args.max_games)
    selected_games = len({str(row['game_id']) for row in selected})
    print(f'SAMPLED: {len(selected)} states from {selected_games} games', flush=True)
    if len(selected) < args.min_selected or selected_games < args.min_labeled_games:
        raise ValueError(f'Sampled too few states/games: {len(selected)} / {selected_games}. '
                         'Increase --max-states, --per-game or restore more games.')
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'states_search.jsonl'
    outcomes, depths = Counter(), Counter()
    timed_out_total = skipped = 0
    values, saved_game_ids = [], set()
    tmp_path = out_dir / 'states_search.jsonl.partial'
    print('Writing labels to temporary file; original output is preserved until validation passes.', flush=True)
    with tmp_path.open('w', encoding='utf-8') as f:
        for index, row in enumerate(selected, 1):
            output, timeout = label_one(teacher, row, args)
            timed_out_total += int(timeout)
            if output is None:
                skipped += 1
                continue
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            outcomes[str(output['outcome'])] += 1
            depths[str(output['teacher_depth'])] += 1
            values.append(output['teacher_value'])
            saved_game_ids.add(str(output['game_id']))
            if index % 25 == 0 or index == len(selected):
                print(f'Labeled {index}/{len(selected)} | saved={len(values)} '
                      f'| last_depth={output["teacher_depth"]}', flush=True)
    summary = {
        'description': 'Search-distilled labels; same POV as acting player, NOT win probabilities',
        'original_rows': len(original_rows), 'original_games': unique_source_games,
        'selected': len(selected), 'selected_games': selected_games,
        'saved': len(values), 'skipped_min_depth': skipped,
        'teacher_timeouts': timed_out_total,
        'unique_games_saved': len(saved_game_ids),
        'depth_counts': dict(depths), 'outcome_counts': dict(outcomes),
        'teacher_value_mean': statistics.mean(values) if values else None,
        'teacher_value_std': statistics.pstdev(values) if len(values) > 1 else None,
        'settings': vars(args), 'teacher_file': str(teacher_path),
        'input_file': str(data_path), 'output_file': str(out_path),
    }
    if len(values) < args.min_labels or len(saved_game_ids) < args.min_labeled_games:
        diagnostic = out_dir / 'failed_summary.json'
        diagnostic.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        raise RuntimeError(
            f'Labeling rejected: {len(values)} labels from {len(saved_game_ids)} games; '
            f'need >= {args.min_labels} labels / {args.min_labeled_games} games. '
            f'Diagnostic: {diagnostic}. Existing states_search.jsonl was NOT overwritten. '
            'Increase --time-limit, reduce --min-depth, or restore a larger dataset.'
        )
    tmp_path.replace(out_path)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print('ACCEPTED:', out_path, flush=True)


if __name__ == '__main__':
    main()
