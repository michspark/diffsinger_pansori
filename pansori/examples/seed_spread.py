#!/usr/bin/env python3
"""How much of a single example's score is the model, and how much is the draw.

Both models are reflow samplers, so one generated file is one sample, not the
model's answer. Before reading anything into a metric on a single file, this
measures how far that metric moves when only the random seed changes.

It regenerates the full-inference track for one utterance under N seeds and
reports the median and range of each score against the original recording.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/examples/seed_spread.py 8b4d0824_0078 --seeds 5
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'pansori' / 'examples'))


def infer(kind, proj, exp, ckpt, seed, out, title, predict=None):
    cmd = [sys.executable, 'scripts/infer.py', kind, str(proj), '--exp', exp,
           '--ckpt', str(ckpt), '--seed', str(seed), '--out', str(out), '--title', title]
    for p in (predict or []):
        cmd += ['--predict', p]
    subprocess.run(cmd, cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description='measure how much the scores vary with the seed')
    parser.add_argument('item')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--ac-ckpt', type=int)
    parser.add_argument('--dur-ckpt', type=int)
    parser.add_argument('--pitch-ckpt', type=int)
    # The lyrics A/B measures the same utterance with two models, so there are two
    # experiment names and two output paths.
    parser.add_argument('--ac-exp', default='pansori_ac_v1')
    parser.add_argument('--var-exp', default='pansori_var_v1')
    parser.add_argument('--out-dir', default='pansori/examples/out')
    parser.add_argument('--ds-dir', default='pansori/examples/ds')
    args = parser.parse_args()

    steps_file = REPO / args.out_dir / 'CKPT_STEPS'
    steps = steps_file.read_text().split() if steps_file.exists() else []
    ac = args.ac_ckpt or int(steps[0])
    dur = args.dur_ckpt or int(steps[1])
    pitch = args.pitch_ckpt or int(steps[2] if len(steps) > 2 else steps[1])

    notes_ds = REPO / args.ds_dir / f'{args.item}_notes.ds'
    orig = REPO / args.out_dir / f'{args.item}_1_orig.wav'
    for p in (notes_ds, orig):
        if not p.exists():
            sys.exit(f'{p} missing — run run_examples.sh first')

    import measure as M
    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs/exp/ac_v1.yaml'), print_hparams=False)
    from modules.pe import initialize_pe
    pe = initialize_pe()
    sr = hparams['audio_sample_rate']
    ref = M.load_wav(orig, sr)

    print(f'{args.item} · {args.ac_exp} {ac:,} · {args.var_exp} dur {dur:,} / pitch {pitch:,}')
    print(f'{"seed":>6s} {"f0 RMSE ¢":>10s} {"mel L1":>8s} {"voiced":>8s}')
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for seed in range(1, args.seeds + 1):
            infer('variance', notes_ds, args.var_exp, dur, seed, tmp, 'd', ['dur'])
            infer('variance', tmp / 'd.ds', args.var_exp, pitch, seed, tmp, 'p', ['pitch'])
            infer('acoustic', tmp / 'p.ds', args.ac_exp, ac, seed, tmp, 'a')
            r = M.compare(ref, M.load_wav(tmp / 'a.wav', sr), hparams, pe)
            rows.append(r)
            print(f'{seed:6d} {r["f0_rmse_cent"]:10.0f} {r["mel_l1"]:8.3f} '
                  f'{r["voiced_agree"]:7.1%}')

    for key, label, spec in (('f0_rmse_cent', 'f0 RMSE ¢', '.0f'),
                             ('mel_l1', 'mel L1', '.3f'),
                             ('voiced_agree', 'voiced agreement', '.1%')):
        v = np.array([r[key] for r in rows])
        print(f'{label:18s} median {format(np.median(v), spec)}  '
              f'range {format(v.min(), spec)}–{format(v.max(), spec)}  '
              f'relative spread {(v.max() - v.min()) / np.median(v):.0%}')


if __name__ == '__main__':
    main()
