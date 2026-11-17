"""Train a Wrongway v3 ValueCNN on search-based teacher labels.

Place: AI-Project1/training/train_value_search_v3.py
Requires PyTorch. Reuses unchanged training/value_network.py CNN architecture;
thus checkpoints can be fed to the existing training/export_minimax_cnn.py.

Main loss: MSE(prediction, teacher_value). Optionally regularize with a
small game-outcome loss (different objective, not an exact win probability).
Split by GAME, never by position. Mirror augmentation is train-only.
"""
import argparse
from collections import Counter
import statistics
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from state_encoder import encode_cnn
from value_network import build_model


def split_games(rows, seed):
    game_ids = sorted({str(row['game_id']) for row in rows})
    if len(game_ids) < 3:
        raise ValueError('Need >=3 games for train / validation / test')
    random.Random(seed).shuffle(game_ids)
    n = len(game_ids)
    n_train = min(n - 2, max(1, round(0.70 * n)))
    n_val = min(n - n_train - 1, max(1, round(0.15 * n)))
    return {
        'train': set(game_ids[:n_train]),
        'validation': set(game_ids[n_train:n_train + n_val]),
        'test': set(game_ids[n_train + n_val:]),
    }


def mirror_state(state):
    """Reflect columns: pawn col 8-c; wall-anchor col 7-c.

    Horizontal/vertical wall orientation remains unchanged by reflection.
    The transformation preserves game rules and the state-value target.
    """
    you = dict(state['you'])
    opponent = dict(state['opponent'])
    you['position'] = [you['position'][0], 8 - you['position'][1]]
    opponent['position'] = [opponent['position'][0], 8 - opponent['position'][1]]
    walls = {
        'horizontal': [[r, 7-c] for r, c in state['walls']['horizontal']],
        'vertical': [[r, 7-c] for r, c in state['walls']['vertical']],
    }
    return {**state, 'you': you, 'opponent': opponent, 'walls': walls}


def tensor_data(rows, game_ids, augment_mirror=False):
    x, teacher_targets, outcomes = [], [], []
    for row in rows:
        if str(row['game_id']) not in game_ids:
            continue
        x.append(encode_cnn(row['state']))
        teacher_targets.append(float(row['teacher_value']))
        outcomes.append(float(row['outcome']))
        if augment_mirror:
            x.append(encode_cnn(mirror_state(row['state'])))
            teacher_targets.append(float(row['teacher_value']))
            outcomes.append(float(row['outcome']))
    if not x:
        raise ValueError('A split has no labeled states; generate more data')
    return TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(teacher_targets, dtype=torch.float32),
        torch.tensor(outcomes, dtype=torch.float32),
    )


@torch.no_grad()
def evaluate(model, loader, device, sign_deadzone):
    model.eval()
    n, mse_total, mae_total = 0, 0.0, 0.0
    sign_matches = sign_total = 0
    outcome_matches = outcome_total = 0
    for x, teacher_y, outcome_y in loader:
        pred = model(x.to(device)).cpu()
        mse_total += (pred - teacher_y).square().sum().item()
        mae_total += (pred - teacher_y).abs().sum().item()
        n += len(teacher_y)
        mask = teacher_y.abs() >= sign_deadzone
        sign_matches += ((pred[mask] >= 0) == (teacher_y[mask] >= 0)).sum().item()
        sign_total += int(mask.sum())
        mask_outcome = outcome_y != 0
        outcome_matches += ((pred[mask_outcome] >= 0) ==
                            (outcome_y[mask_outcome] > 0)).sum().item()
        outcome_total += int(mask_outcome.sum())
    return {
        'teacher_mse': mse_total / n,
        'teacher_mae': mae_total / n,
        'teacher_sign_accuracy': sign_matches / sign_total if sign_total else None,
        'outcome_sign_accuracy': outcome_matches / outcome_total if outcome_total else None,
        'states': n,
        'teacher_sign_evaluated': sign_total,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='training/datasets/search_v3/states_search.jsonl')
    p.add_argument('--out', default='training/models/cnn_search_v3_fixed')
    p.add_argument('--architecture', choices=('cnn', 'mlp'), default='cnn')
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--outcome-weight', type=float, default=0.0,
                   help='0=teacher-only, 0.1=10 percent loss on final outcomes')
    p.add_argument('--augment-mirror', action='store_true',
                   help='Mirror train states horizontally (not validation/test)')
    p.add_argument('--sign-deadzone', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--min-games', type=int, default=30)
    p.add_argument('--min-states', type=int, default=250)
    p.add_argument('--min-eval-games', type=int, default=4)
    p.add_argument('--min-eval-states', type=int, default=20)
    p.add_argument('--allow-small-data', action='store_true',
                   help='SMOKE TEST ONLY: disables minimum size checks; do not report metrics as valid')
    p.add_argument('--overwrite', action='store_true',
                   help='Allow replacing a checkpoint already present at --out')
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1
            or not 0 <= args.outcome_weight <= 1 or args.sign_deadzone < 0
            or args.min_games < 3 or args.min_states < 1
            or args.min_eval_games < 1 or args.min_eval_states < 1):
        p.error('Invalid training hyperparameters')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA not available')
    data_path = Path(args.data)
    if not data_path.is_file():
        p.error(f'Missing data: {data_path}')
    rows = []
    with data_path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                for key in ('state', 'game_id', 'teacher_value', 'outcome'):
                    if key not in row:
                        raise ValueError(f'Line {line_num} missing {key}')
                if not -1 <= row['teacher_value'] <= 1:
                    raise ValueError(f'Line {line_num} teacher value is outside [-1,1]')
                rows.append(row)
    unique_games = len({str(row['game_id']) for row in rows})
    if not args.allow_small_data and (unique_games < args.min_games or len(rows) < args.min_states):
        raise ValueError(
            f'REFUSING training on only {unique_games} games / {len(rows)} states. '
            f'Need >= {args.min_games} games and >= {args.min_states} labeled states. '
            'Restore a larger source dataset and regenerate search labels. '
            'Use --allow-small-data ONLY for a smoke test.'
        )
    splits = split_games(rows, args.seed)
    if not args.allow_small_data:
        for part in ('validation', 'test'):
            games_in_part = len(splits[part])
            states_in_part = sum(str(row['game_id']) in splits[part] for row in rows)
            if games_in_part < args.min_eval_games or states_in_part < args.min_eval_states:
                raise ValueError(f'{part} too small: {games_in_part} games / {states_in_part} states; '
                                 f'need >= {args.min_eval_games} games / {args.min_eval_states} states.')
    else:
        print('WARNING: --allow-small-data is SMOKE TEST ONLY; no reliability conclusions.', flush=True)
    teacher_values = [float(row['teacher_value']) for row in rows]
    sign_counts = Counter('positive' if value > args.sign_deadzone else
                          ('negative' if value < -args.sign_deadzone else 'neutral')
                          for value in teacher_values)
    depth_counts = Counter(str(row.get('teacher_depth', 'unknown')) for row in rows)
    label_diagnostics = {
        'games': unique_games, 'states': len(rows),
        'mean': statistics.mean(teacher_values),
        'std': statistics.pstdev(teacher_values) if len(teacher_values) > 1 else 0,
        'sign_counts': dict(sign_counts), 'teacher_depth_counts': dict(depth_counts),
        'saturated_abs_ge_0.98': sum(abs(v) >= 0.98 for v in teacher_values),
        'majority_sign_accuracy_baseline': (max(sign_counts['positive'], sign_counts['negative']) /
                                            max(1, sign_counts['positive']+sign_counts['negative']))
    }
    print('LABEL DIAGNOSTICS:', json.dumps(label_diagnostics), flush=True)
    datasets = {part: tensor_data(rows, ids,
                        augment_mirror=(args.augment_mirror and part == 'train'))
                for part, ids in splits.items()}
    loaders = {part: DataLoader(ds, batch_size=args.batch_size,
                  shuffle=(part == 'train'), num_workers=0)
               for part, ds in datasets.items()}
    print('Device:', device, '| CNN-compatible architecture:', args.architecture, flush=True)
    for name, ds in datasets.items():
        print(f'{name}: {len(splits[name])} games / {len(ds)} positions', flush=True)
    out = Path(args.out)
    checkpoint = out / 'best_model.pt'
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f'{checkpoint} already exists. Use a NEW --out, or explicitly --overwrite.')
    model = build_model(args.architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val, best_epoch, stale = float('inf'), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, t, outcome in loaders['train']:
            x, t, outcome = x.to(device), t.to(device), outcome.to(device)
            pred = model(x)
            loss = ((1-args.outcome_weight) * criterion(pred, t)
                    + args.outcome_weight * criterion(pred, outcome))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_metric = evaluate(model, loaders['train'], device, args.sign_deadzone)
        val_metric = evaluate(model, loaders['validation'], device, args.sign_deadzone)
        history.append({'epoch': epoch, 'train': train_metric, 'validation': val_metric})
        print(f'Epoch {epoch:02d}: teacher MSE train={train_metric["teacher_mse"]:.4f}, '
              f'val={val_metric["teacher_mse"]:.4f}', flush=True)
        if val_metric['teacher_mse'] < best_val - 1e-6:
            best_val, best_epoch, stale = val_metric['teacher_mse'], epoch, 0
            # Same format/architecture as v2: existing CNN exporter can read it.
            torch.save({
                'architecture': args.architecture,
                'input_shape': [8, 9, 9],
                'model_state_dict': model.state_dict(),
                'epoch': epoch, 'seed': args.seed,
                'target': 'teacher_search_tanh',
                'outcome_weight': args.outcome_weight,
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping', flush=True)
                break
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metric = evaluate(model, loaders['test'], device, args.sign_deadzone)
    results = {
        'architecture': args.architecture, 'seed': args.seed,
        'best_epoch': best_epoch,
        'best_validation_teacher_mse': best_val,
        'test': test_metric, 'history': history,
        'label_diagnostics': label_diagnostics,
        'smoke_only': bool(args.allow_small_data),
        'games': {k: len(v) for k, v in splits.items()},
        'positions': {k: len(v) for k, v in datasets.items()},
        'args': vars(args),
        'note': 'Teacher-value agreement is NOT game win rate; validate against Arena opponents.',
    }
    (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (out / 'split_games.json').write_text(
        json.dumps({k: sorted(v) for k, v in splits.items()}, indent=2),
        encoding='utf-8',
    )
    print('Saved:', checkpoint, 'and', out / 'metrics.json', flush=True)
    print('TEST:', test_metric, flush=True)


if __name__ == '__main__':
    main()
