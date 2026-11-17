"""Locate an intact outcome-labeled dataset from Kaggle inputs, without overwriting v1.

Destination: AI-Project1/training/recover_training_data_v3.py
Run from any cwd:
  python training/recover_training_data_v3.py --scan-dir /kaggle/input
  python training/recover_training_data_v3.py --source /kaggle/input/.../states.jsonl

In auto mode choose the single source with the MOST distinct game_ids, not a naive
concatenation of notebook outputs (which can duplicate entire games). This cannot
recover a file that is absent from both the working directory and Kaggle input.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def scan_source(path):
    games, outcomes = set(), Counter()
    rows = 0
    with path.open('r', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                gid = record['game_id']
                state = record['state']
                outcome = record['outcome']
                if outcome not in (-1, 0, 1) or not isinstance(state, dict):
                    raise ValueError('invalid outcome or state')
                if not all(key in state for key in ('you', 'opponent', 'walls', 'turn')):
                    raise ValueError('incomplete state')
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc
            games.add(str(gid))
            outcomes[str(outcome)] += 1
            rows += 1
    return {'path': str(path), 'games': len(games), 'states': rows, 'outcome_counts': dict(outcomes)}


def locate(args):
    if args.source:
        paths = [Path(p).expanduser() for p in args.source]
    else:
        roots = [ROOT / 'training/datasets'] + [Path(d).expanduser() for d in args.scan_dir]
        paths = []
        for scan in roots:
            if scan.is_dir():
                paths.extend(scan.rglob('states.jsonl'))
        # Never select the destination as an input for itself.
    target = args.out if args.out.is_absolute() else ROOT / args.out
    unique = sorted({p.resolve() for p in paths if p.is_file() and p.resolve() != target.resolve()})
    if not unique:
        raise RuntimeError('No states.jsonl found. Attach the original Kaggle notebook output as Input, '
                           'or supply --source /absolute/path/to/states.jsonl')
    reports = []
    for p in unique:
        try:
            report = scan_source(p)
            reports.append(report)
        except (OSError, ValueError) as exc:
            print(f'SKIP invalid dataset: {p} ({exc})', file=sys.stderr)
    if not reports:
        raise RuntimeError('No valid outcome-labeled datasets found.')
    reports.sort(key=lambda item: (item['games'], item['states']), reverse=True)
    print('Candidate outcome datasets (unique game IDs, not positions):', flush=True)
    for index, report in enumerate(reports, 1):
        print(f'{index:2d}. {report["games"]:5d} games, {report["states"]:7d} states: {report["path"]}', flush=True)
    return reports, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan-dir', action='append', default=[], help='Scan this dir recursively (repeatable); Kaggle: /kaggle/input')
    p.add_argument('--source', action='append', default=[], help='Explicit states.jsonl; repeat to compare, not merge')
    p.add_argument('--out', type=Path, default=Path('training/datasets/v3_source/states.jsonl'))
    p.add_argument('--min-games', type=int, default=30)
    p.add_argument('--min-states', type=int, default=300)
    p.add_argument('--list-only', action='store_true')
    args = p.parse_args()
    if args.min_games < 3 or args.min_states < 1:
        p.error('min-games must be >=3; min-states >=1')
    if not args.source and not args.scan_dir:
        args.scan_dir = ['/kaggle/input']
    reports, target = locate(args)
    if args.list_only:
        return
    best = reports[0]
    if best['games'] < args.min_games or best['states'] < args.min_states:
        raise RuntimeError(f'Largest available dataset is only {best["games"]} games / '
                           f'{best["states"]} states; need >= {args.min_games} games and '
                           f'{args.min_states} states. Attach the old dataset with 126 games, '
                           'or generate additional games into a NEW output folder.')
    source = Path(best['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.jsonl.tmp')
    try:
        shutil.copyfile(source, temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    digest = sha256(target.read_bytes()).hexdigest()
    manifest = {'chosen_source': str(source), 'destination': str(target),
                'sha256': digest, 'chosen': best, 'candidates': reports,
                'note': 'Single source selected to avoid duplicated games or overlapping game IDs.'}
    (target.parent / 'source_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'RESTORED: {target} | {best["games"]} games / {best["states"]} states', flush=True)


if __name__ == '__main__':
    main()
