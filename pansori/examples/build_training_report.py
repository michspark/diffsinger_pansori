#!/usr/bin/env python3
"""Render two arms' training curves and their paired A/B table as one HTML page.

The wandb project already holds the curves, but reading an A/B off it means
knowing which two of nine runs to select and which step each arm's optimum sits
at. This page states the comparison instead: the same three validation metrics
for both arms, the checkpoint each arm was evaluated at marked on the curve, and
the paired table from ab_ladder underneath, so the training picture and the
synthesis result are read together.

The step-0 point is dropped from the plots. It is the sanity check Lightning runs
before training, so it carries the untrained loss (variance dur 34 and 125 in the
two arms) and would flatten every curve after it into a line.

    python pansori/examples/build_training_report.py \
        --runs pansori_ac_sofa pansori_ac_hyb pansori_var_sofa pansori_var_hyb \
        --labels Whisper "음절보존 하이브리드"

Needs a wandb login (reads history through the public API).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AB_OUT = REPO / 'pansori' / 'examples' / 'ab_out'
OUT_HTML = Path(__file__).resolve().parent / 'training_report.html'

# The two categorical hues the listening test already uses, so the pair of arms
# keeps one identity across both pages. Checked with the dataviz validator:
# adjacent ΔE 24.5 normal / 17.9 protan, both above the floor.
COLOR_A, COLOR_B = '#3b5ba5', '#b5533c'

PANELS = [
    ('acoustic', 'validation/total_loss', 'acoustic · validation total loss', 4),
    ('variance', 'validation/dur_loss', 'variance · validation dur loss', 4),
    ('variance', 'validation/pitch_loss', 'variance · validation pitch loss', 5),
]


def fetch(entity_project, runs):
    import wandb
    api = wandb.Api()
    keys = sorted({k for _, k, _, _ in PANELS})
    out = {}
    for name in runs:
        run = api.run(f'{entity_project}/{name}')
        series = {}
        # One key at a time: scan_history with a key list returns only the rows that
        # carry *every* key, so asking an acoustic run for dur_loss empties it.
        for key in keys:
            points = [[row['global_step'], row[key]]
                      for row in run.scan_history(keys=['global_step', key], page_size=2000)
                      if row.get(key) is not None]
            if points:
                series[key] = sorted(points)
        out[name] = series
        print(f'  {name}: ' + ', '.join(f'{k.split("/")[1]} x{len(v)}'
                                        for k, v in sorted(series.items())), file=sys.stderr)
    return out


def ab_table(scores, arm_a, arm_b):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_ladder import sign_test
    changed, sc = scores['changed'], scores['scores']
    groups = []
    for label, want in (('가사가 바뀐 발화', True), ('대조군 — 음소열 동일', False)):
        subset = [i for i in sorted(sc[arm_a]) if changed.get(i) is want and i in sc[arm_b]]
        rows = []
        for key, name, lower_better, digits in (
                ('mel_l1', 'mel L1', True, 3),
                ('f0_rmse_cent', 'f0 RMSE ¢', True, 0),
                ('voiced_agree', 'voiced agreement', False, 3)):
            pairs = [(sc[arm_a][i][key], sc[arm_b][i][key]) for i in subset
                     if key in sc[arm_a][i] and key in sc[arm_b][i]]
            if not pairs:
                continue
            wins, losses, p = sign_test([b - a for a, b in pairs], lower_better)
            rows.append({'metric': name, 'digits': digits,
                         'a': round(statistics.median(a for a, _ in pairs), digits),
                         'b': round(statistics.median(b for _, b in pairs), digits),
                         'bWins': wins, 'pairs': wins + losses, 'p': round(p, 3),
                         'lowerBetter': lower_better})
        groups.append({'label': label, 'n': len(subset), 'rows': rows})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', default='michpark-georgia-institute-of-technology/pansori')
    parser.add_argument('--runs', nargs=4, required=True,
                        metavar=('AC_A', 'AC_B', 'VAR_A', 'VAR_B'))
    parser.add_argument('--labels', nargs=2, default=['Whisper', 'hybrid'])
    parser.add_argument('--best', nargs='*', default=[], metavar='RUN=STEP',
                        help='mark the evaluated checkpoint on that run\'s curve')
    parser.add_argument('--arms', nargs=2, default=['sofa', 'hybrid'],
                        help='ab_ladder arm names inside scores.json')
    parser.add_argument('--out', default=str(OUT_HTML))
    args = parser.parse_args()

    ac_a, ac_b, var_a, var_b = args.runs
    print('fetching wandb history', file=sys.stderr)
    hist = fetch(args.project, args.runs)
    best = dict(kv.split('=') for kv in args.best)

    panels = []
    for kind, key, title, digits in PANELS:
        run_a, run_b = (ac_a, ac_b) if kind == 'acoustic' else (var_a, var_b)
        panels.append({
            'title': title, 'digits': digits, 'metric': key.split('/')[1],
            'a': {'run': run_a, 'points': [p for p in hist[run_a].get(key, []) if p[0] > 0],
                  'best': int(best.get(run_a, 0)) or None},
            'b': {'run': run_b, 'points': [p for p in hist[run_b].get(key, []) if p[0] > 0],
                  'best': int(best.get(run_b, 0)) or None},
        })

    scores_path = AB_OUT / 'scores.json'
    scores = json.loads(scores_path.read_text(encoding='utf-8')) if scores_path.exists() else None
    payload = {
        'labels': {'a': args.labels[0], 'b': args.labels[1]},
        'colors': {'a': COLOR_A, 'b': COLOR_B},
        'project': f'https://wandb.ai/{args.project}',
        'runs': {name: f'https://wandb.ai/{args.project}/runs/{name}' for name in args.runs},
        'panels': panels,
        'ab': ab_table(scores, *args.arms) if scores else [],
        'abSeeds': 3,
    }
    template = Path(__file__).resolve().parent / 'training_report.template.html'
    html = template.read_text(encoding='utf-8').replace(
        '"__PAYLOAD__"', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    out = Path(args.out)
    out.write_text(html, encoding='utf-8')
    print(f'\n{out}  {out.stat().st_size / 1e3:.0f} KB')


if __name__ == '__main__':
    main()
