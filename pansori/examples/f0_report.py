#!/usr/bin/env python3
"""Compare the f0 the model invents against the f0 that was actually sung.

pansori/docs/SIGIMSAE_PLAN.md diagnosed the old cascade model as flattening
pitch: a deterministic regressor under an L1 loss converges on the conditional
median, so sigimsae -- the ornament that carries most of the expression in this
music -- averages away. openvpi's pitch predictor is a reflow generative model
instead, so that argument does not transfer, and the question has to be re-asked
of the new model rather than assumed.

This asks it. For each utterance it takes the f0 of the original recording and
of each generated version and reports:

  range       p5--p95 spread in cents -- how much of the register is used
  |Δf0|       median absolute frame-to-frame motion in cents/frame -- how much
              the contour actually moves, which is what an ornament is
  amp ratio   the generated |Δf0| as a fraction of the original's

A model that flattens keeps the range roughly right (it lands on the notes) but
loses the motion between them, so an amp ratio well under 1 is the signature to watch.

Both models are reflow samplers, and the ratio moves a lot between draws -- one
utterance measured 0.46 on one seed and 1.01 on another. A single sample
therefore cannot answer the question, so --seeds N regenerates the predicted
tracks under N seeds and reports the median and range. The default single-file
mode is only for producing the contour that gets plotted.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/examples/f0_report.py --json pansori/examples/out/f0.json
  PYTHONPATH=. python pansori/examples/f0_report.py --seeds 5
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRACKS = [('1_orig', 'original'), ('2_resynth', 'real f0'),
          ('3_pitch', 'f0 predicted'), ('4_full', 'dur+f0 predicted')]


def contour(path, hparams, pe):
    import librosa
    wav, _ = librosa.load(path, sr=hparams['audio_sample_rate'], mono=True)
    length = round(len(wav) / hparams['hop_size'])
    f0, uv = pe.get_pitch(
        wav, samplerate=hparams['audio_sample_rate'], length=length,
        hop_size=hparams['hop_size'], f0_min=hparams['f0_min'],
        f0_max=hparams['f0_max'], interp_uv=False
    )
    return f0, ~uv


def motion(f0, voiced):
    """Median |Δf0| in cents per frame, over runs of consecutive voiced frames.

    Restricted to runs so a jump across an unvoiced gap -- which is a phrase
    boundary, not an ornament -- does not count as movement.
    """
    deltas = []
    run = []
    for value, is_voiced in zip(f0, voiced):
        if is_voiced:
            run.append(value)
        else:
            if len(run) > 2:
                deltas.append(np.abs(np.diff(1200 * np.log2(np.maximum(run, 1e-5)))))
            run = []
    if len(run) > 2:
        deltas.append(np.abs(np.diff(1200 * np.log2(np.maximum(run, 1e-5)))))
    return float(np.median(np.concatenate(deltas))) if deltas else float('nan')


def spread(f0, voiced):
    v = 1200 * np.log2(np.maximum(f0[voiced], 1e-5))
    return float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size > 10 else float('nan')


def _infer(kind, proj, ckpt, seed, out, title, predict=None):
    cmd = [sys.executable, 'scripts/infer.py', kind, str(proj),
           '--exp', f'pansori_{"var" if kind == "variance" else "ac"}_v1',
           '--ckpt', str(ckpt), '--seed', str(seed),
           '--out', str(out), '--title', title]
    for p in (predict or []):
        cmd += ['--predict', p]
    subprocess.run(cmd, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def seeded(items, out_dir, n_seeds, hparams, pe):
    """Regenerate the predicted tracks under n seeds and report median motion ratio."""
    ac, dur, pitch = (int(x) for x in (out_dir / 'CKPT_STEPS').read_text().split())
    print(f'{n_seeds} seeds · acoustic {ac:,} · dur {dur:,} · pitch {pitch:,}\n')
    print(f'{"item":22s} {"track":16s} {"median amp ratio":>17s} {"range":>16s}')
    print('-' * 72)
    results = {}
    for item in items:
        base = motion(*contour(out_dir / f'{item}_1_orig.wav', hparams, pe))
        results[item] = {}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for track, label, ds, steps in (
                ('2_resynth', 'real f0', f'{item}_gt.ds', None),
                ('3_pitch', 'f0 predicted', f'{item}_dur.ds', 'pitch'),
                ('4_full', 'dur+f0 predicted', f'{item}_notes.ds', 'both'),
            ):
                ratios = []
                for seed in range(1, n_seeds + 1):
                    src = REPO / 'pansori/examples/ds' / ds
                    if steps == 'pitch':
                        _infer('variance', src, pitch, seed, tmp, 'v', ['pitch'])
                        src = tmp / 'v.ds'
                    elif steps == 'both':
                        _infer('variance', src, dur, seed, tmp, 'd', ['dur'])
                        _infer('variance', tmp / 'd.ds', pitch, seed, tmp, 'v', ['pitch'])
                        src = tmp / 'v.ds'
                    _infer('acoustic', src, ac, seed, tmp, 'a')
                    ratios.append(motion(*contour(tmp / 'a.wav', hparams, pe)) / base)
                r = np.array(ratios)
                results[item][track] = {'label': label, 'median': float(np.median(r)),
                                        'min': float(r.min()), 'max': float(r.max()),
                                        'seeds': n_seeds}
                print(f'{item:22s} {label:14s} {np.median(r):14.2f} '
                      f'{f"{r.min():.2f}–{r.max():.2f}":>16s}')
        print()
    return results


def main():
    parser = argparse.ArgumentParser(description='measure f0 flattening')
    parser.add_argument('--out', default='pansori/examples/out')
    parser.add_argument('--json')
    parser.add_argument('--seeds', type=int, default=0,
                        help='regenerate N times and judge by median and range (0 = existing files only)')
    args = parser.parse_args()

    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs' / 'exp' / 'ac_v1.yaml'), print_hparams=False)
    from modules.pe import initialize_pe
    pe = initialize_pe()

    out_dir = REPO / args.out
    items = sorted({p.name.rsplit('_1_orig', 1)[0]
                    for p in out_dir.glob('*_1_orig.wav')})

    payload = {'timestep': hparams['hop_size'] / hparams['audio_sample_rate'], 'items': {}}
    print(f'{"item":22s} {"track":16s} {"range ¢":>9s} {"|Δf0| ¢/frame":>15s} {"amp ratio":>10s}')
    print('-' * 74)
    for item in items:
        base_motion = None
        payload['items'][item] = {}
        for track, label in TRACKS:
            path = out_dir / f'{item}_{track}.wav'
            if not path.exists():
                continue
            f0, voiced = contour(path, hparams, pe)
            m, s = motion(f0, voiced), spread(f0, voiced)
            if track == '1_orig':
                base_motion = m
            ratio = m / base_motion if base_motion else float('nan')
            payload['items'][item][track] = {
                'label': label, 'motion_cent': m, 'spread_cent': s, 'ratio': ratio,
                # thinned for plotting; unvoiced frames become null
                'f0': [None if not v else round(float(x), 1)
                       for x, v in zip(f0[::2], voiced[::2])],
            }
            print(f'{item:22s} {label:14s} {s:8.0f} {m:15.1f} {ratio:8.2f}')
        print()

    if args.seeds:
        print()
        payload['seeded'] = seeded(items, out_dir, args.seeds, hparams, pe)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        print(f'→ {args.json}')


if __name__ == '__main__':
    main()
