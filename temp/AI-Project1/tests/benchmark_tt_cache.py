"""Ablation: compare V3 TT with V3.1 cached-BFS TT (or user-supplied agents).

Place: AI-Project1/tests/benchmark_tt_cache.py
Requires existing tests/benchmark_hybrid_openings.py for seeded Arena matches.
Outputs JSON and Markdown with TT probes, matched openings, swapped sides.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Arena.environment import Environment
from benchmark_hybrid_openings import make_opening, one_match


def load_agent(path, name):
    spec = importlib.util.spec_from_file_location('search_v3_probe_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legal(obs, action):
    if action.get('type') == 'move':
        return action.get('to') in obs.get('legal_moves', ())
    if action.get('type') == 'wall':
        return any(all(action.get(k) == wall.get(k) for k in ('orientation', 'row', 'col'))
                   for wall in obs.get('legal_walls', ()))
    return action.get('type') == 'pass' and not obs.get('legal_moves')


def probe(paths, openings, count):
    agents = {name: load_agent(path, name) for name, path in paths.items()}
    reports = []
    for i, (seed, snapshot, _) in enumerate(openings[:count], 1):
        obs = Environment.from_snapshot(snapshot).get_observation()
        for name, agent in agents.items():
            agent.RECENT_POSITIONS.clear()
            before = time.monotonic()
            action = agent.choose_action(obs)
            elapsed = time.monotonic() - before
            row = {'opening_id': i, 'opening_seed': seed, 'variant': name,
                   'legal_action': legal(obs, action), 'elapsed': round(elapsed, 5),
                   'action': action,
                   'stats': dict(agent.SEARCH_STATS),
                   'tt_entries': len(getattr(agent, 'TRANSPOSITION_TABLE', {}))}
            if not row['legal_action']:
                raise ValueError(f'{name} produced illegal action: {action}')
            reports.append(row)
            stats = row['stats']
            print('PROBE', name, 'opening', i,
                  'depth', stats['completed_depth'], 'nodes', stats['nodes'],
                  'BFS', stats['bfs_calls'], 'TT hits', stats.get('tt_hits', 0),
                  'TT entries', row['tt_entries'], flush=True)
    return reports


def run(args):
    if (args.openings < 1 or args.probe_openings < 0 or
            args.opening_plies < 0 or args.opening_plies % 2 or
            args.max_turns <= args.opening_plies or
            not 0 <= args.wall_prob <= 1):
        raise ValueError('Invalid openings/probes/turn limits/wall probability')
    paths = {
        'v2': (ROOT / args.agent_baseline).resolve(),
        'v3_tt': (ROOT / args.agent_challenger).resolve(),
    }
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    openings, keys, attempt = [], set(), 0
    while len(openings) < args.openings and attempt < max(100, args.openings * 100):
        seed = args.seed + attempt * 7919
        attempt += 1
        snap, actions = make_opening(seed, args.opening_plies, args.wall_prob, args.max_turns)
        key = json.dumps(snap, sort_keys=True, separators=(',', ':'))
        if key not in keys:
            keys.add(key)
            openings.append((seed, snap, actions))
    if len(openings) != args.openings:
        raise RuntimeError('Could not generate unique openings')
    probes = probe(paths, openings, args.probe_openings)
    games = []
    if not args.probe_only:
        for oid, (seed, snap, _) in enumerate(openings, 1):
            for side in (0, 1):
                players = ([str(paths['v3_tt']), str(paths['v2'])] if side == 0
                           else [str(paths['v2']), str(paths['v3_tt'])])
                match = one_match(snap, players, seed=args.seed+oid*11,
                                  timeout=args.timeout, startup_timeout=args.startup_timeout)
                outcome = ('draw' if match['winner'] is None else
                           'win' if match['winner'] == side else 'loss')
                games.append({'opening_id': oid, 'opening_seed': seed,
                              'tt_side': side, 'outcome': outcome, **match})
                print('MATCH', oid, 'TT=P'+str(side), outcome.upper(),
                      'timeout=', match['statuses'][side].get('timeout',0), flush=True)
    counts = Counter(g['outcome'] for g in games)
    tt_action_count = sum(g['actions_timed'][g['tt_side']] for g in games)
    mean_tt_latency = (sum(g['time_mean'][g['tt_side']]*g['actions_timed'][g['tt_side']]
                           for g in games)/tt_action_count if tt_action_count else None)
    summary = {
        'games': len(games), 'wins': counts['win'], 'draws': counts['draw'],
        'losses': counts['loss'],
        'score_rate': ((counts['win'] + 0.5*counts['draw'])/len(games)) if games else None,
        'tt_p0_wins': sum(g['outcome']=='win' for g in games if g['tt_side']==0),
        'tt_p1_wins': sum(g['outcome']=='win' for g in games if g['tt_side']==1),
        'tt_mean_latency': mean_tt_latency,
        'tt_timeouts': sum(g['statuses'][g['tt_side']].get('timeout',0) for g in games),
        'tt_invalid': sum(g['statuses'][g['tt_side']].get('invalid',0) for g in games),
        'opening_pair_outcomes': dict(Counter(
            f"{sum(g['outcome']=='win' for g in games if g['opening_id']==i)} wins / "
            f"{sum(g['outcome']=='draw' for g in games if g['opening_id']==i)} draws"
            for i in range(1, len(openings)+1))) if games else {},
    }
    output = {'experiment': 'minimax_paired_agents_ablation', 'config': vars(args),
              'agents': {name: str(p) for name, p in paths.items()},
              'openings': [{'id': i, 'seed': o[0], 'snapshot': o[1]} for i,o in enumerate(openings,1)],
              'probes': probes, 'summary': summary, 'games': games}
    path = Path(args.out)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    md = [f'# Search comparison: {args.agent_challenger} vs {args.agent_baseline}', '',
          'The same opening is played twice with swapped agent sides. Scores are not independent.', '',
          '## Search probes', '',
          '| Opening | Agent | Depth | Nodes | BFS calls | TT hits | TT entries |',
          '|---:|---|---:|---:|---:|---:|---:|']
    for row in probes:
        stats = row['stats']
        md.append(f"| {row['opening_id']} | {row['variant']} | {stats['completed_depth']} | "
                  f"{stats['nodes']} | {stats['bfs_calls']} | {stats.get('tt_hits',0)} | {row['tt_entries']} |")
    md += ['', '## Arena results', '',
           f"TT W/D/L: **{summary['wins']}/{summary['draws']}/{summary['losses']}**",
           f"TT score: **{summary['score_rate']:.1%}**" if games else 'Probe-only; no matches run.',
           f"TT mean latency: {mean_tt_latency:.3f}s" if mean_tt_latency is not None else '',
           '', 'A shorter search or larger TT hit count does not establish stronger playing strength.']
    path.with_suffix('.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('Saved:', path, 'and', path.with_suffix('.md'), flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agent-baseline', default='Agents/minimax_tt.py')
    ap.add_argument('--agent-challenger', default='Agents/minimax_tt_cached.py')
    ap.add_argument('--openings', type=int, default=20)
    ap.add_argument('--opening-plies', type=int, default=8)
    ap.add_argument('--wall-prob', type=float, default=0.35)
    ap.add_argument('--max-turns', type=int, default=500)
    ap.add_argument('--seed', type=int, default=9016)
    ap.add_argument('--timeout', type=float, default=1.0)
    ap.add_argument('--startup-timeout', type=float, default=6.0)
    ap.add_argument('--probe-openings', type=int, default=3)
    ap.add_argument('--probe-only', action='store_true')
    ap.add_argument('--out', default='tests/search_v3_cached_vs_tt.json')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
