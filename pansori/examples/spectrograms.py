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
          ('3_pitch', '피치 예측'), ('4_full', '전체 추론')]


def main():
    parser = argparse.ArgumentParser(description='사다리 mel 비교 그림')
    parser.add_argument('item')
    parser.add_argument('--out', default=None)
    parser.add_argument('--wav-dir', default='pansori/examples/out',
                        help='사다리 wav 이 있는 곳 (가사 A/B 는 변종마다 다르다)')
    parser.add_argument('--dpi', type=int, default=110)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import librosa

    # matplotlib 기본 폰트(DejaVu Sans)에는 한글 글리프가 없어 라벨이 두부가 된다.
    # .ttc 컬렉션은 첫 face 이름으로만 등록되므로 'KR' 이 아니라 'JP' 로 잡히는데,
    # Noto Sans CJK 는 한 파일에 네 문자권을 다 담고 있어 한글도 이 이름으로 나온다.
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
    set_hparams(config=str(REPO / 'configs/pansori_acoustic.yaml'), print_hparams=False)
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
        sys.exit(f'{args.item}: wav 없음')

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
    # 배경이 투명해 라이트/다크 어느 쪽에도 얹힌다. 축 글자를 검정으로 두면 다크에서
    # 사라지므로 두 바탕 모두에서 읽히는 중간 회색으로 그린다.
    AXIS = '#8A929E'
    axes[-1].set_xlabel('초', fontsize=9, color=AXIS)
    axes[-1].tick_params(labelsize=8, colors=AXIS)

    out = Path(args.out) if args.out else out_dir / f'{args.item}_mel.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight',
                facecolor='none', transparent=True)
    # 이 그림은 base64 로 페이지에 박히므로 크기가 곧 페이지 무게다.
    # 스펙트로그램은 색 수가 적어 팔레트화하면 손실 없이 크게 준다.
    try:
        from PIL import Image
        im = Image.open(out)
        # FASTOCTREE 는 RGBA 를 그대로 받는다 — RGB 로 바꾸면 투명 배경이 검게 채워져
        # 라이트 테마에서 그림만 검은 판이 된다.
        im.quantize(colors=192, method=Image.FASTOCTREE).save(out, optimize=True)
    except Exception as exc:
        print(f'  (팔레트 최적화 건너뜀: {exc})')
    print(f'{out}  ({out.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
