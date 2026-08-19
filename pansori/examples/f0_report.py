#!/usr/bin/env python3
"""Compare the f0 the model invents against the f0 that was actually sung.

pansori/docs/SIGIMSAE_PLAN.md diagnosed the old cascade model as flattening
pitch: a deterministic regressor under an L1 loss converges on the conditional
median, so 시김새 -- the ornament that carries most of the expression in this
music -- averages away. openvpi's pitch predictor is a reflow generative model
instead, so that argument does not transfer, and the question has to be re-asked
of the new model rather than assumed.

This asks it. For each utterance it takes the f0 of the original recording and
of each generated version and reports, over commonly-voiced frames:

  범위        p5--p95 spread in cents -- how much of the register is used
  |Δf0|       median absolute frame-to-frame motion in cents/frame -- how much
              the contour actually moves, which is what an ornament is
  진폭비      the generated |Δf0| as a fraction of the original's

A model that flattens keeps the range roughly right (it lands on the notes) but
loses the motion between them, so 진폭비 well under 1 is the signature to watch.

It also writes the contours themselves so they can be plotted.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/examples/f0_report.py --json pansori/examples/out/f0.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRACKS = [('1_orig', '원본'), ('2_resynth', '실제 f0'),
          ('3_pitch', 'f0 예측'), ('4_full', 'dur·f0 예측')]


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


def main():
    parser = argparse.ArgumentParser(description='f0 평탄화 측정')
    parser.add_argument('--out', default='pansori/examples/out')
    parser.add_argument('--json')
    args = parser.parse_args()

    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs' / 'pansori_acoustic.yaml'), print_hparams=False)
    from modules.pe import initialize_pe
    pe = initialize_pe()

    out_dir = REPO / args.out
    items = sorted({p.name.rsplit('_1_orig', 1)[0]
                    for p in out_dir.glob('*_1_orig.wav')})

    payload = {'timestep': hparams['hop_size'] / hparams['audio_sample_rate'], 'items': {}}
    print(f'{"항목":22s} {"트랙":14s} {"범위 ¢":>8s} {"|Δf0| ¢/frame":>15s} {"진폭비":>8s}')
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

    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        print(f'→ {args.json}')


if __name__ == '__main__':
    main()
