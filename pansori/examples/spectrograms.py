#!/usr/bin/env python3
"""Render the ladder's mel spectrograms side by side, so the difference is visible.

The scores say a rung is worse; they do not say how. Stacking the mels shows it:
where a formant smears, where a consonant burst is missing, where the model put
energy the recording does not have. Same 128-bin log-mel the model is trained
on, so what is on screen is what the loss is computed over.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/examples/spectrograms.py 8b4d0824_0078
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRACKS = [('1_orig', '원본 녹음'), ('2_resynth', '재합성 — dur·f0 실측'),
          ('3_pitch', '피치 예측'), ('4_full', '전체 추론')]   # figure labels


def main():
    parser = argparse.ArgumentParser(description='a figure comparing the ladder\'s mels')
    parser.add_argument('item')
    parser.add_argument('--out', default=None)
    parser.add_argument('--wav-dir', default='pansori/examples/out',
                        help='where the ladder wavs are (the lyrics A/B has one per variant)')
    parser.add_argument('--dpi', type=int, default=110)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import librosa

    # matplotlib's default font (DejaVu Sans) has no Hangul glyphs, so the labels become
    # tofu. A .ttc collection registers only under its first face name, so it is found as
    # 'JP' rather than 'KR' -- but Noto Sans CJK carries all four scripts in one file, so
    # Hangul comes out under that name too.
    for path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            registered = {f.name for f in font_manager.fontManager.ttflist}
            for name in ('Noto Sans CJK KR', 'Noto Sans CJK JP'):
                if name in registered:
                    plt.rcParams['font.family'] = name
                    break
            break

    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs/exp/ac_v1.yaml'), print_hparams=False)
    from utils.binarizer_utils import get_mel_torch

    out_dir = REPO / args.wav_dir
    sr = hparams['audio_sample_rate']
    mels, labels = [], []
    for track, label in TRACKS:
        path = out_dir / f'{args.item}_{track}.wav'
        if not path.exists():
            continue
        wav, _ = librosa.load(path, sr=sr, mono=True)
        mels.append(get_mel_torch(
            wav, sr, num_mel_bins=hparams['audio_num_mel_bins'],
            hop_size=hparams['hop_size'], win_size=hparams['win_size'],
            fft_size=hparams['fft_size'], fmin=hparams['fmin'],
            fmax=hparams['fmax'], device='cuda'
        ).T)
        labels.append(label)
    if not mels:
        sys.exit(f'{args.item}: no wav')

    frames = max(m.shape[1] for m in mels)
    seconds = frames * hparams['hop_size'] / sr
    # A shared scale, or the panels cannot be compared. The floor sits where the
    # STFT clamps (log 1e-5) rather than at each panel's own minimum.
    vmin, vmax = -11.5, float(max(m.max() for m in mels))

    fig, axes = plt.subplots(
        len(mels), 1, figsize=(11, 1.55 * len(mels) + 0.5),
        sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, mel, label in zip(axes, mels, labels):
        ax.imshow(mel, origin='lower', aspect='auto', cmap='magma',
                  vmin=vmin, vmax=vmax,
                  extent=[0, mel.shape[1] * hparams['hop_size'] / sr, 0, mel.shape[0]])
        ax.set_xlim(0, seconds)
        ax.set_yticks([])
        ax.set_ylabel('')
        ax.text(0.006, 0.9, label, transform=ax.transAxes, ha='left', va='top',
                fontsize=9, color='white',
                bbox=dict(facecolor='black', alpha=.45, edgecolor='none', pad=2.2))
        for spine in ax.spines.values():
            spine.set_visible(False)
    # The background is transparent so it sits on either a light or a dark page. Black
    # axis text would disappear on dark, so a mid grey readable on both is used.
    AXIS = '#8A929E'
    axes[-1].set_xlabel('초', fontsize=9, color=AXIS)   # 'seconds'
    axes[-1].tick_params(labelsize=8, colors=AXIS)

    out = Path(args.out) if args.out else out_dir / f'{args.item}_mel.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight',
                facecolor='none', transparent=True)
    # This figure is embedded into the page as base64, so its size is the page's weight.
    # A spectrogram has few colours, so palettising shrinks it a lot without loss.
    try:
        from PIL import Image
        im = Image.open(out)
        # FASTOCTREE takes RGBA as-is -- converting to RGB fills the transparent
        # background with black and the figure becomes a black slab on a light theme.
        im.quantize(colors=192, method=Image.FASTOCTREE).save(out, optimize=True)
    except Exception as exc:
        print(f'  (palette optimisation skipped: {exc})')
    print(f'{out}  ({out.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
