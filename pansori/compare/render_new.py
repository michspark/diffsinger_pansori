#!/usr/bin/env python3
"""Render the new model's rungs, loading each model once instead of once per file.

pansori/examples/run_examples.sh spawns a fresh `scripts/infer.py` for every
render, and each spawn spends ~25 s loading a model and a vocoder before it does
0.3 s of work. At 19 items x 4 rungs x 5 seeds that overhead alone is ~2.5 hours.
Here the loop is inside the process and the whole job is a coffee break.

It cannot be ONE process, because utils.hparams.set_hparams writes a module-level
global that is read by every model constructor -- the variance and acoustic
experiments have different configs, so they cannot coexist. So this is three
processes, not 380:

  prep       build the seed-independent .ds inputs from the labels
  variance   load the dur head (16k) and the pitch head (32k), predict
  acoustic   load the acoustic model (28k) + vocoder, render every .ds to wav,
             and produce the 0_vocoder ceiling rung

Rungs produced:
  0_vocoder   GT mel + GT f0 -> vocoder. No diffusion, deterministic, ~0.2 s.
              This is the counterpart of the old ladder's 2_gt_crepe, and it is
              what makes any part of the difference attributable: whatever gap
              shows up HERE is vocoder and extractor, not the model.
  2_resynth   GT ph_dur + GT f0 -> acoustic
  3_pitch     GT ph_dur, f0 predicted -> acoustic     <- pairs with old 5_pred_own
  4_full      dur and f0 both predicted -> acoustic

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/compare/render_new.py --items fair --seeds 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import items as I  # noqa: E402

OUT_WAV = C.OUT / 'new'
OUT_DS = C.OUT / 'ds'


def resolve_items(spec):
    fair, excluded = I.paired_items()
    if spec == 'fair':
        return fair
    if spec == 'excluded':
        return excluded
    if spec == 'paired':
        return fair + excluded
    if spec.startswith('sample:'):
        return I.holdout_sample(int(spec.split(':', 1)[1]))
    return [s for s in spec.split(',') if s]


# ---------------------------------------------------------------- checkpoints

def restore_best():
    """Put back checkpoints that num_ckpt_keep deleted.

    This actually fires: the acoustic optimum at 28k was already gone from the
    experiment directory when training finished, and checkpoints/best/ is the
    only copy (pansori/docs/NEXT.md).
    """
    best = C.REPO / 'checkpoints' / 'best'
    if not best.is_dir():
        return
    for saved in sorted(best.glob('*_step*.ckpt')):
        exp, step = saved.stem.split('_step')
        dest = C.REPO / 'checkpoints' / exp / f'model_ckpt_steps_{step}.ckpt'
        if dest.parent.is_dir() and not dest.exists():
            print(f'restored: {dest.relative_to(C.REPO)} <- best/')
            dest.write_bytes(saved.read_bytes())


def assert_ckpts():
    for exp, step in (('pansori_ac_v1', C.CKPT_ACOUSTIC),
                      ('pansori_var_v1', C.CKPT_DUR),
                      ('pansori_var_v1', C.CKPT_PITCH)):
        p = C.REPO / 'checkpoints' / exp / f'model_ckpt_steps_{step}.ckpt'
        if not p.exists():
            raise FileNotFoundError(f'required checkpoint missing: {p}')


# ---------------------------------------------------------------- prep

def mode_prep(items):
    """Seed-independent .ds inputs, built straight from transcriptions.csv."""
    OUT_DS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(C.REPO / 'pansori'))
    import make_ds

    class A:
        pass

    made, skipped = 0, []
    for item in items:
        for kind, with_dur, with_f0 in (('gt', True, True),
                                        ('dur', True, False),
                                        ('notes', False, False)):
            out = OUT_DS / f'{item}_{kind}.ds'
            if out.exists():
                continue
            a = A()
            a.item, a.data, a.spk, a.out = item, 'data/pansori', None, str(out)
            a.with_dur, a.with_f0 = with_dur, with_f0
            try:
                make_ds.cmd_from_label(a)
            except SystemExit as e:
                # note_slur is reconstructed by DP from the real timing and cannot
                # always be recovered -- when an utterance has more words than
                # notes there is no assignment. That only blocks 4_full (the
                # notes-only input); the paired rungs are unaffected, so the item
                # keeps its place in the comparison and loses one rung.
                if kind != 'notes':
                    raise
                skipped.append(f'{item}: {e}')
                continue
            made += 1
    print(f'| prep: {made} .ds built, {len(items)} items')
    if skipped:
        print(f'| 4_full impossible for {len(skipped)} items (note_slur recovery failed): '
              f'{", ".join(s.split(":")[0] for s in skipped[:8])}'
              f'{" …" if len(skipped) > 8 else ""}')


# ---------------------------------------------------------------- variance

def _set_hparams(exp):
    sys.argv = [sys.argv[0], '--exp_name', exp, '--infer']
    from utils.hparams import set_hparams
    set_hparams(print_hparams=False)


def mode_variance(items, seeds):
    _set_hparams('pansori_var_v1')
    from inference.ds_variance import DiffSingerVarianceInfer

    dur = DiffSingerVarianceInfer(ckpt_steps=C.CKPT_DUR, predictions={'dur'})
    pitch = DiffSingerVarianceInfer(ckpt_steps=C.CKPT_PITCH, predictions={'pitch'})

    def run(infer, src, title, seed):
        out = OUT_DS / f'{title}.ds'
        if out.exists():
            return out
        if not Path(src).exists():
            return None
        params = json.loads(Path(src).read_text(encoding='utf-8'))
        if not isinstance(params, list):
            params = [params]
        for p in params:
            p['seed'] = seed
        infer.run_inference(params, out_dir=OUT_DS, title=title, num_runs=1, seed=seed)
        return out

    t0 = time.time()
    for item in items:
        for seed in seeds:
            # 3_pitch: ground-truth durations, pitch predicted
            run(pitch, OUT_DS / f'{item}_dur.ds', f'{item}_dur_var_s{seed}', seed)
            # 4_full: durations then pitch, both predicted
            d = run(dur, OUT_DS / f'{item}_notes.ds', f'{item}_notes_dur_s{seed}', seed)
            if d is not None:
                run(pitch, d, f'{item}_notes_var_s{seed}', seed)
    print(f'| variance: {len(items)} items x {len(seeds)} seeds, {time.time() - t0:.1f}s')


# ---------------------------------------------------------------- acoustic

def mode_acoustic(items, seeds):
    _set_hparams('pansori_ac_v1')
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from utils.hparams import hparams
    from utils.binarizer_utils import get_mel_torch
    from inference.ds_acoustic import DiffSingerAcousticInfer

    OUT_WAV.mkdir(parents=True, exist_ok=True)
    infer = DiffSingerAcousticInfer(ckpt_steps=C.CKPT_ACOUSTIC)
    sr = hparams['audio_sample_rate']

    def render(src, title, seed):
        out = OUT_WAV / f'{title}.wav'
        if out.exists() or not Path(src).exists():
            return
        params = json.loads(Path(src).read_text(encoding='utf-8'))
        if not isinstance(params, list):
            params = [params]
        for p in params:
            p['seed'] = seed
        infer.run_inference(params, out_dir=OUT_WAV, title=title, num_runs=1, seed=seed)

    def vocoder_ceiling(item):
        """GT mel + GT f0 -> vocoder. The ceiling every model number is read against."""
        out = OUT_WAV / f'{item}__0_vocoder__s0.wav'
        if out.exists():
            return
        wav, _ = librosa.load(str(I.wav_of(item)), sr=sr, mono=True)
        mel = get_mel_torch(
            wav, sr, num_mel_bins=hparams['audio_num_mel_bins'],
            hop_size=hparams['hop_size'], win_size=hparams['win_size'],
            fft_size=hparams['fft_size'], fmin=hparams['fmin'], fmax=hparams['fmax'],
            device=infer.device,
        )
        from modules.pe import initialize_pe
        f0, _ = initialize_pe().get_pitch(
            wav, samplerate=sr, length=mel.shape[0], hop_size=hparams['hop_size'],
            f0_min=hparams['f0_min'], f0_max=hparams['f0_max'], interp_uv=True,
        )
        spec = torch.from_numpy(mel).to(infer.device)[None]
        f0_t = torch.from_numpy(np.asarray(f0, dtype=np.float32)).to(infer.device)[None]
        y = infer.run_vocoder(spec, f0=f0_t)[0].cpu().numpy()
        sf.write(out, y, sr)

    t0 = time.time()
    for item in items:
        vocoder_ceiling(item)
        for seed in seeds:
            render(OUT_DS / f'{item}_gt.ds', f'{item}__2_resynth__s{seed}', seed)
            render(OUT_DS / f'{item}_dur_var_s{seed}.ds', f'{item}__3_pitch__s{seed}', seed)
            render(OUT_DS / f'{item}_notes_var_s{seed}.ds', f'{item}__4_full__s{seed}', seed)
    n = len(items) * (1 + 3 * len(seeds))
    dt = time.time() - t0
    print(f'| acoustic: {n} renders / {dt:.1f}s = {dt / max(n, 1):.2f}s per render')


# ---------------------------------------------------------------- driver

def main():
    p = argparse.ArgumentParser(description='render the new model\'s ladder')
    p.add_argument('--items', default='fair',
                   help='fair | excluded | paired | sample:N | a comma-separated list')
    p.add_argument('--seeds', type=int, default=len(C.SEEDS))
    p.add_argument('--mode', default='driver',
                   choices=['driver', 'prep', 'variance', 'acoustic'])
    args = p.parse_args()

    items = resolve_items(args.items)
    seeds = list(C.SEEDS[:args.seeds])

    if args.mode == 'driver':
        restore_best()
        assert_ckpts()
        print(f'{len(items)} items x {len(seeds)} seeds '
              f'(acoustic {C.CKPT_ACOUSTIC} · dur {C.CKPT_DUR} · pitch {C.CKPT_PITCH})')
        env = {**os.environ, 'PYTHONPATH': str(C.REPO), 'PYTHONNOUSERSITE': '1'}
        for mode in ('prep', 'variance', 'acoustic'):
            print(f'\n== {mode} ==')
            r = subprocess.run(
                [sys.executable, __file__, '--mode', mode,
                 '--items', ','.join(items), '--seeds', str(len(seeds))],
                cwd=C.REPO, env=env,
            )
            if r.returncode:
                sys.exit(f'{mode} failed (exit {r.returncode})')
        print(f'\ndone — {OUT_WAV}')
        return

    {'prep': lambda: mode_prep(items),
     'variance': lambda: mode_variance(items, seeds),
     'acoustic': lambda: mode_acoustic(items, seeds)}[args.mode]()


if __name__ == '__main__':
    main()
