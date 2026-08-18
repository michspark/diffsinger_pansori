#!/usr/bin/env python3
"""Measure this corpus's per-bin mel envelope and write it as a config fragment.

GaussianDiffusion normalises its target with spec_min/spec_max
(usr/diff/shallow_diffusion_tts.py:278):

    norm = (mel - spec_min) / (spec_max - spec_min) * 2 - 1

so those two vectors decide whether the diffusion target actually lands in
[-1, 1]. The pansori config inherited Opencpop's numbers, and pansori mel is
louder in the low bins: measured on the current training set, 72 of 80 bins go
past +1, peaking at +1.174. The model then has to spend capacity reproducing
out-of-range values instead of detail.

Sampling does not settle this -- the envelope is an extreme-value statistic and
kept growing between a 600- and a 3000-item sample (min moved 0.52). So this
reads every item by default.

Usage (diffsinger env, from the repo root, PYTHONPATH=.):
  python pansori/compute_spec_stats.py --binary_data_dir data/binary/pansori_v2 \
      --out usr/configs/midi/cascade/pansori/pansori_statis.yaml
"""

import argparse

import numpy as np

from utils.indexed_datasets import IndexedDataset

# Headroom on each side so a frame slightly outside the observed envelope still
# normalises inside [-1, 1] instead of just past it.
MARGIN = 0.05

# Mel floor: wav2spec clamps there, so there is nothing below to represent.
MEL_FLOOR = -6.0


def envelope(binary_data_dir, prefix, limit):
    ds = IndexedDataset(f'{binary_data_dir}/{prefix}')
    idx = (np.linspace(0, len(ds) - 1, limit).astype(int) if limit
           else np.arange(len(ds)))
    n_bins = np.asarray(ds[int(idx[0])]['mel']).shape[1]
    lo, hi = np.full(n_bins, np.inf), np.full(n_bins, -np.inf)
    for n, i in enumerate(idx):
        mel = np.asarray(ds[int(i)]['mel'])
        lo, hi = np.minimum(lo, mel.min(0)), np.maximum(hi, mel.max(0))
        if (n + 1) % 5000 == 0:
            print(f'  {n + 1}/{len(idx)}', flush=True)
    return lo, hi, len(idx)


def as_yaml_block(name, values, per_line=8):
    rows = [', '.join(f'{v:8.4f}' for v in values[i:i + per_line])
            for i in range(0, len(values), per_line)]
    body = ',\n           '.join(rows)
    return f'{name}: [{body}]\n'


def main():
    parser = argparse.ArgumentParser(description='판소리 mel 통계 측정')
    parser.add_argument('--binary_data_dir', default='data/binary/pansori_v2')
    parser.add_argument('--prefix', default='train')
    parser.add_argument('--limit', type=int, default=0,
                        help='표본 수 (0 = 전체, 권장)')
    parser.add_argument('--out',
                        default='usr/configs/midi/cascade/pansori/pansori_statis.yaml')
    args = parser.parse_args()

    lo, hi, n = envelope(args.binary_data_dir, args.prefix, args.limit)
    spec_min = np.maximum(lo - MARGIN, MEL_FLOOR)
    spec_max = hi + MARGIN

    header = (
        '# 판소리 mel 통계 — pansori/compute_spec_stats.py 로 생성 (직접 수정하지 말 것)\n'
        f'# 출처: {args.binary_data_dir}/{args.prefix}, {n:,} 발화 전수 측정, 여유 {MARGIN}\n'
        '#\n'
        '# Opencpop 값(usr/configs/midi/cascade/opencs/opencpop_statis.yaml)을 쓰면\n'
        '# 판소리 mel 저역이 상한을 넘어 정규화 후 +1.174까지 나간다 (80 bin 중 72개 초과).\n'
        '\n')
    footer = (
        '\n# 아래 세 값은 Opencpop 것을 그대로 쓴다 — 플롯 범위와 wav2spec 상수라\n'
        '# 코퍼스와 무관하다.\n'
        'mel_vmin: -6.\n'
        'mel_vmax: 1.5\n'
        'wav2spec_eps: 1e-6\n')

    with open(args.out, 'w') as f:
        f.write(header)
        f.write(as_yaml_block('spec_min', spec_min))
        f.write('\n')
        f.write(as_yaml_block('spec_max', spec_max))
        f.write(footer)

    print(f'\n{n:,} 발화 측정 → {args.out}')
    print(f'spec_min  {spec_min.min():.3f} ~ {spec_min.max():.3f}')
    print(f'spec_max  {spec_max.min():.3f} ~ {spec_max.max():.3f}')


if __name__ == '__main__':
    main()
