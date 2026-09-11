#!/usr/bin/env python3
"""Paired A/B over many held-out utterances — does the treatment change the model?

The two arms are chosen with --arms. It was built for the lyrics A/B (whisper vs
snapped), but it is the same ruler for the aligner A/B (whisper vs sofa) -- whether
what changed is the .lab or the aligner, the remaining question is the same one:
"does the model trained on those labels sound closer to the original?"

`run_examples.sh` renders three utterances, and three is not enough here. Measured
seed spread on a single clip is ±3-12% on mel L1 and ±13-28% on f0 RMSE
(`seed_spread.py`), so a per-clip difference smaller than that says nothing. The
way out is more clips, paired: the same held-out utterance rendered by both models
and scored against the same recording, then a sign test over the pairs.

Only the **2_resynth** rung is rendered — ground-truth `ph_dur` and ground-truth f0,
both from the arm's own labels, so the acoustic model and the phone sequence it was
trained on are what differ. The variance heads stay out of it; their sampling noise
is larger than the effect being looked for, and duration/pitch prediction is a
separate question from whether the lyrics were right.

Three processes, not one, because `utils.hparams.set_hparams` writes a module-level
global that every model constructor reads — the two arms have different configs and
cannot coexist in one interpreter.

    PYTHONPATH=. python pansori/examples/ab_ladder.py --n 30 --seeds 3
    PYTHONPATH=. python pansori/examples/ab_ladder.py --arms whisper sofa \
        --ckpt sofa=32000 --n 24 --seeds 3

Outputs `pansori/examples/ab_out/{arm}/` wavs and prints the paired table.
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'pansori' / 'examples'))

OUT_ROOT = REPO / 'pansori' / 'examples' / 'ab_out'

ARMS = {
    'whisper': dict(config='configs/exp/ac_v1.yaml', data='data/pansori',
                    ac_exp='pansori_ac_v1', ckpt=28000),
    'snapped': dict(config='configs/exp/ac_snap.yaml', data='data/pansori_snap',
                    ac_exp='pansori_ac_snap', ckpt=36000),
    'sofa': dict(config='configs/exp/ac_sofa.yaml', data='data/pansori_sofa',
                 ac_exp='pansori_ac_sofa', ckpt=None),   # supplied via --ckpt sofa=N
}

# Which two arms are compared. --arms changes it; it is passed to child processes as-is.
PAIR = ['whisper', 'snapped']

# What separates the control group. For the lyrics A/B the treatment group is
# "utterances whose text changed", but for the aligner A/B every utterance's phoneme
# sequence changes, so that criterion separates nothing. Instead the treatment group is
# the utterances where MFA actually collapsed (the top half by floor rate) -- because
# those utterances are precisely why the aligner was changed.
GROUP_LABELS = {
    ('whisper', 'snapped'): ('lyrics changed', 'lyrics identical (control)'),
}
DEFAULT_LABELS = ('high MFA floor rate', 'low MFA floor rate (control)')


# ---------------------------------------------------------------- items

def corpus_rows(data_root: Path):
    rows = {}
    for speaker in sorted(p for p in data_root.iterdir() if p.is_dir()):
        path = speaker / 'transcriptions.csv'
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows[row['name']] = (speaker.name, row)
    return rows


def holdout_prefixes(config_path: Path):
    """test_prefixes out of the config chain -- the songs no model ever trained on.

    Descends through base_config. Commit 3cb4989 moved the datasets block out of the
    experiment file into configs/pansori/data_*.yaml, so reading only the experiment
    file finds nothing (after that commit this function was quietly returning an empty
    list).
    """
    import yaml

    seen, prefixes = set(), []

    def walk(path: Path):
        path = path.resolve()
        if path in seen or not path.exists():
            return
        seen.add(path)
        text = path.read_text(encoding='utf-8')
        # make_configs.py writes the count as a trailing comment on the same line
        # ("- 'c235c995'      # 3,792 utterances"), so take what is between the quotes.
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- '") and line.count("'") >= 2:
                prefixes.append(line.split("'")[1])
        base = (yaml.safe_load(text) or {}).get('base_config') or []
        for child in ([base] if isinstance(base, str) else base):
            walk(REPO / child)

    walk(config_path)
    if not prefixes:
        sys.exit(f'no test_prefixes found in the {config_path} chain — '
                 f'without a hold-out there is no A/B.')
    return prefixes


def _floor_rate(row):
    """The fraction of this utterance's ph_dur pinned to MFA's floor. Same definition as label_report."""
    sys.path.insert(0, str(REPO / 'pansori' / 'eval'))
    from label_report import MFA_FLOOR, EPS
    durs = [float(x) for x in row['ph_dur'].split()]
    phones = [d for d, p in zip(durs, row['ph_seq'].split()) if p not in ('SP', 'AP')]
    return sum(1 for d in phones if d <= MFA_FLOOR + EPS) / len(phones) if phones else 1.0


def pick_items(n, seed):
    a = corpus_rows(REPO / ARMS[PAIR[0]]['data'])
    b = corpus_rows(REPO / ARMS[PAIR[1]]['data'])
    prefixes = tuple(holdout_prefixes(REPO / ARMS[PAIR[0]]['config']))
    shared = sorted(set(a) & set(b))
    held = [i for i in shared if i.startswith(prefixes)]
    if tuple(PAIR) in GROUP_LABELS:
        # Only utterances whose text actually differs carry the treatment; the rest are
        # a control group, and both are worth having in the table.
        changed = [i for i in held if a[i][1]['ph_seq'] != b[i][1]['ph_seq']]
        same = [i for i in held if a[i][1]['ph_seq'] == b[i][1]['ph_seq']]
    else:
        rates = {i: _floor_rate(a[i][1]) for i in held}
        cut = statistics.median(rates.values()) if rates else 0.0
        changed = [i for i in held if rates[i] > cut]
        same = [i for i in held if rates[i] <= cut]
    rng = random.Random(seed)
    rng.shuffle(changed)
    rng.shuffle(same)
    half = max(1, n // 2)
    picked = changed[:n - min(half, len(same))] + same[:half]
    return sorted(picked), {i: (i in set(changed)) for i in picked}


# ---------------------------------------------------------------- prep / render

def mode_prep(arm, items):
    """Write {item}_gt.ds for one arm: its own ph_dur plus f0 read off the recording."""
    spec = ARMS[arm]
    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / spec['config']), print_hparams=False)
    import librosa
    from modules.pe import initialize_pe
    sys.path.insert(0, str(REPO / 'pansori'))
    from make_ds import F0_TIMESTEP, set_speaker, write_ds

    pe = initialize_pe()          # loaded once; make_ds reloads RMVPE per invocation
    rows = corpus_rows(REPO / spec['data'])
    ds_dir = OUT_ROOT / arm / 'ds'
    ds_dir.mkdir(parents=True, exist_ok=True)
    sample_rate, hop = hparams['audio_sample_rate'], hparams['hop_size']

    for item in items:
        out = ds_dir / f'{item}_gt.ds'
        if out.exists():
            continue
        speaker, row = rows[item]
        wav_path = REPO / spec['data'] / speaker / 'wavs' / f'{item}.wav'
        waveform, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
        f0, _ = pe.get_pitch(waveform, samplerate=sample_rate,
                             length=round(len(waveform) / hop), hop_size=hop,
                             f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
                             interp_uv=True)
        # make_ds resamples the hop grid to a 5 ms grid; do the same so the .ds
        # matches what run_examples.sh produces.
        import numpy as np
        step = hop / sample_rate
        times = np.arange(len(f0)) * step
        grid = np.arange(0, times[-1] + F0_TIMESTEP, F0_TIMESTEP)
        f0 = np.interp(grid, times, f0)
        segment = {'offset': 0.0, 'ph_seq': row['ph_seq'], 'ph_num': row['ph_num'],
                   'note_seq': row['note_seq'], 'note_dur': row['note_dur'],
                   'ph_dur': row['ph_dur'],
                   'f0_seq': ' '.join(f'{v:.1f}' for v in f0),
                   'f0_timestep': F0_TIMESTEP}
        set_speaker(segment, speaker)
        write_ds(segment, out)
    print(f'| prep {arm}: {len(items)} items')


def mode_acoustic(arm, items, seeds, ckpt):
    spec = ARMS[arm]
    sys.argv = [sys.argv[0], '--exp_name', spec['ac_exp'], '--infer']
    from utils.hparams import set_hparams
    set_hparams(print_hparams=False)
    from inference.ds_acoustic import DiffSingerAcousticInfer

    infer = DiffSingerAcousticInfer(ckpt_steps=ckpt)
    ds_dir = OUT_ROOT / arm / 'ds'
    wav_dir = OUT_ROOT / arm / 'wav'
    wav_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        src = ds_dir / f'{item}_gt.ds'
        for seed in seeds:
            title = f'{item}__s{seed}'
            if (wav_dir / f'{title}.wav').exists() or not src.exists():
                continue
            params = json.loads(src.read_text(encoding='utf-8'))
            params = params if isinstance(params, list) else [params]
            for p in params:
                p['seed'] = seed
            infer.run_inference(params, out_dir=wav_dir, title=title,
                                num_runs=1, seed=seed)
    print(f'| acoustic {arm}: {len(items)} items x {len(seeds)} seeds')


# ---------------------------------------------------------------- scoring

def mode_score(items, seeds, changed):
    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / ARMS[PAIR[0]]['config']), print_hparams=False)
    import measure as M
    from modules.pe import initialize_pe
    pe = initialize_pe()
    sample_rate = hparams['audio_sample_rate']
    rows = corpus_rows(REPO / ARMS[PAIR[0]]['data'])

    scores = {arm: {} for arm in PAIR}
    per_seed_scores = {arm: {} for arm in PAIR}
    for item in items:
        speaker, _ = rows[item]
        ref = M.load_wav(REPO / ARMS[PAIR[0]]['data'] / speaker / 'wavs' / f'{item}.wav',
                         sample_rate)
        for arm in PAIR:
            per_seed = []
            for seed in seeds:
                path = OUT_ROOT / arm / 'wav' / f'{item}__s{seed}.wav'
                if not path.exists():
                    continue
                per_seed.append(M.compare(ref, M.load_wav(path, sample_rate), hparams, pe))
            if per_seed:
                # median over seeds: one render is a draw, not the model's answer
                scores[arm][item] = {
                    key: statistics.median(r[key] for r in per_seed if r.get(key) is not None)
                    for key in ('f0_rmse_cent', 'mel_l1', 'voiced_agree')
                    if any(r.get(key) is not None for r in per_seed)
                }
                # keep the draws too: anything that plays one of these renders back
                # (a listening test) has to play the draw whose number is on screen
                per_seed_scores[arm][item] = {str(s): r for s, r in zip(seeds, per_seed)}
    report(items, changed, scores)
    (OUT_ROOT / 'scores.json').write_text(
        json.dumps({'changed': changed, 'scores': scores,
                    'per_seed': per_seed_scores}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    print(f'\n→ {OUT_ROOT / "scores.json"}')


def sign_test(deltas, lower_is_better=True):
    """Wins for PAIR[1], wins for PAIR[0], two-sided binomial p."""
    wins = sum(1 for d in deltas if (d < 0) == lower_is_better and d != 0)
    losses = sum(1 for d in deltas if d != 0) - wins
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    from math import comb
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return wins, losses, min(1.0, 2 * tail)


def report(items, changed, scores):
    metrics = [('mel_l1', 'mel L1', True, 3),
               ('f0_rmse_cent', 'f0 RMSE ¢', True, 0),
               ('voiced_agree', 'voiced agreement', False, 3)]
    hi, lo = GROUP_LABELS.get(tuple(PAIR), DEFAULT_LABELS)
    groups = [(hi, [i for i in items if changed.get(i)]),
              (lo, [i for i in items if not changed.get(i)]),
              ('all', list(items))]

    for title, subset in groups:
        subset = [i for i in subset if i in scores[PAIR[0]] and i in scores[PAIR[1]]]
        if not subset:
            continue
        print(f'\n{title} — {len(subset)} pairs')
        print(f"  {'metric':<18}{PAIR[0]:>10}{PAIR[1]:>10}{'median Δ':>11}"
              f"{PAIR[1] + ' wins':>13}{'p':>9}")
        for key, label, lower_better, nd in metrics:
            pairs = [(scores[PAIR[0]][i].get(key), scores[PAIR[1]][i].get(key))
                     for i in subset]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                continue
            deltas = [b - a for a, b in pairs]
            wins, losses, p = sign_test(deltas, lower_better)
            fmt = f'{{:.{nd}f}}'
            print(f'  {label:<12}'
                  f'{fmt.format(statistics.median(a for a, _ in pairs)):>10}'
                  f'{fmt.format(statistics.median(b for _, b in pairs)):>10}'
                  f'{fmt.format(statistics.median(deltas)):>11}'
                  f'{f"{wins}/{wins + losses}":>11}{p:>9.3f}')


# ---------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=30, help='number of items (half are the control)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--sample-seed', type=int, default=20260829)
    parser.add_argument('--arms', nargs=2, default=['whisper', 'snapped'],
                        metavar=('A', 'B'), choices=list(ARMS),
                        help='the two arms to compare (e.g. --arms whisper sofa)')
    parser.add_argument('--ckpt', action='append', default=[], metavar='ARM=STEP',
                        help='checkpoint step per arm (e.g. --ckpt sofa=32000). Repeatable')
    parser.add_argument('--mode', default='driver',
                        choices=['driver', 'prep', 'acoustic', 'score'])
    parser.add_argument('--arm', choices=list(ARMS))
    parser.add_argument('--items-file')
    args = parser.parse_args()

    PAIR[:] = args.arms
    for spec in args.ckpt:
        name, _, step = spec.partition('=')
        if name not in ARMS:
            sys.exit(f'unknown arm name in --ckpt: {name} (available: {sorted(ARMS)})')
        ARMS[name]['ckpt'] = int(step)
    missing = [a for a in PAIR if ARMS[a].get('ckpt') is None]
    if missing:
        sys.exit(f'no checkpoint step for: {missing} — pass --ckpt {missing[0]}=STEP')

    if args.items_file:
        payload = json.loads(Path(args.items_file).read_text(encoding='utf-8'))
        items, changed = payload['items'], payload['changed']
    else:
        items, changed = pick_items(args.n, args.sample_seed)
    seeds = list(range(1, args.seeds + 1))

    if args.mode == 'prep':
        return mode_prep(args.arm, items)
    if args.mode == 'acoustic':
        return mode_acoustic(args.arm, items, seeds, ARMS[args.arm]['ckpt'])
    if args.mode == 'score':
        return mode_score(items, seeds, changed)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    items_file = OUT_ROOT / 'items.json'
    items_file.write_text(json.dumps({'items': items, 'changed': changed},
                                     ensure_ascii=False), encoding='utf-8')
    n_changed = sum(1 for i in items if changed[i])
    hi, lo = GROUP_LABELS.get(tuple(PAIR), DEFAULT_LABELS)
    print(f'{PAIR[0]} vs {PAIR[1]} — {len(items)} items '
          f'({hi} {n_changed} / {lo} {len(items) - n_changed}), {len(seeds)} seeds')

    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    ckpt_args = [x for a in PAIR for x in ('--ckpt', f'{a}={ARMS[a]["ckpt"]}')]
    common = ['--arms', *PAIR, '--items-file', str(items_file),
              '--seeds', str(args.seeds), *ckpt_args]
    for arm in PAIR:
        for mode in ('prep', 'acoustic'):
            subprocess.run([sys.executable, __file__, '--mode', mode, '--arm', arm,
                            *common], cwd=REPO, env=env, check=True)
    subprocess.run([sys.executable, __file__, '--mode', 'score', *common],
                   cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
