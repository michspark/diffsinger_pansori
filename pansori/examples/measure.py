#!/usr/bin/env python3
"""Score the generated examples against their originals.

The ladder in run_examples.sh only means something if each rung is measured, so
this reports, per generated file:

  peak / RMS      sanity -- a silent or clipped file is a bug, not a result
  f0 RMSE (cent)  pitch accuracy on frames both sides call voiced
  mel L1          spectral distance in the model's own 128-bin log-mel space
  voiced agree    fraction of frames where both agree on voiced/unvoiced

f0 and mel both come from the same extractors the training pipeline uses (RMVPE
with the config's 55--750 Hz range, and nvSTFT), so the numbers are in the
units the model was actually optimised in rather than a third-party proxy.

Lengths differ by a frame or two between an original and its resynthesis; both
series are truncated to the shorter one rather than resampled, which is exact
for the aligned tracks and the honest floor for the predicted-duration track.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/examples/measure.py
  PYTHONPATH=. python pansori/examples/measure.py --json pansori/examples/out/metrics.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRACKS = {
    '1_orig': 'original recording',
    '2_resynth': 'real dur + real f0',
    '3_pitch': 'real dur, f0 predicted',
    '4_full': 'dur and f0 both predicted',
}


def load_wav(path, sr):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def get_f0(wav, hparams, pe):
    length = round(len(wav) / hparams['hop_size'])
    f0, uv = pe.get_pitch(
        wav, samplerate=hparams['audio_sample_rate'], length=length,
        hop_size=hparams['hop_size'], f0_min=hparams['f0_min'],
        f0_max=hparams['f0_max'], interp_uv=False
    )
    return f0, uv


def get_mel(wav, hparams):
    from utils.binarizer_utils import get_mel_torch
    return get_mel_torch(
        wav, hparams['audio_sample_rate'], num_mel_bins=hparams['audio_num_mel_bins'],
        hop_size=hparams['hop_size'], win_size=hparams['win_size'],
        fft_size=hparams['fft_size'], fmin=hparams['fmin'], fmax=hparams['fmax'],
        device='cuda'
    )


def cents(a, b):
    return 1200.0 * np.log2(np.maximum(a, 1e-5) / np.maximum(b, 1e-5))


def compare(ref_wav, gen_wav, hparams, pe):
    out = {
        'peak': float(np.abs(gen_wav).max()),
        'rms_db': float(20 * np.log10(max(np.sqrt((gen_wav ** 2).mean()), 1e-9))),
        'seconds': len(gen_wav) / hparams['audio_sample_rate'],
    }
    if ref_wav is None:
        return out

    f0_r, uv_r = get_f0(ref_wav, hparams, pe)
    f0_g, uv_g = get_f0(gen_wav, hparams, pe)
    n = min(len(f0_r), len(f0_g))
    f0_r, uv_r, f0_g, uv_g = f0_r[:n], uv_r[:n], f0_g[:n], uv_g[:n]

    both = ~uv_r & ~uv_g
    out['voiced_agree'] = float((uv_r == uv_g).mean())
    out['f0_rmse_cent'] = (float(np.sqrt((cents(f0_g[both], f0_r[both]) ** 2).mean()))
                           if both.sum() > 10 else None)

    mel_r, mel_g = get_mel(ref_wav, hparams), get_mel(gen_wav, hparams)
    m = min(len(mel_r), len(mel_g))
    out['mel_l1'] = float(np.abs(mel_g[:m] - mel_r[:m]).mean())
    out['frames'] = int(n)
    return out


def main():
    parser = argparse.ArgumentParser(description='score the generated examples')
    parser.add_argument('--out', default='pansori/examples/out')
    parser.add_argument('--json', help='also save the results as JSON')
    args = parser.parse_args()

    from utils.hparams import set_hparams, hparams
    set_hparams(config=str(REPO / 'configs' / 'exp' / 'ac_v1.yaml'), print_hparams=False)
    from modules.pe import initialize_pe
    pe = initialize_pe()
    sr = hparams['audio_sample_rate']

    out_dir = REPO / args.out
    wavs = sorted(out_dir.glob('*.wav'))
    items = sorted({m.group(1) for w in wavs
                    if (m := re.match(r'(.+?)_[1-4]_', w.name))})

    results = {}
    print(f'{"item":26s} {"track":24s} {"len":>6s} {"peak":>6s} {"RMS dB":>7s} '
          f'{"f0 RMSE¢":>9s} {"mel L1":>7s} {"voiced":>8s}')
    print('-' * 100)
    for item in items:
        ref_path = out_dir / f'{item}_1_orig.wav'
        ref = load_wav(ref_path, sr) if ref_path.exists() else None
        for track, label in TRACKS.items():
            path = out_dir / f'{item}_{track}.wav'
            if not path.exists():
                continue
            gen = load_wav(path, sr)
            r = compare(None if track == '1_orig' else ref, gen, hparams, pe)
            results[path.name] = r
            f0s = f'{r["f0_rmse_cent"]:.0f}' if r.get('f0_rmse_cent') else '—'
            mel = f'{r["mel_l1"]:.3f}' if 'mel_l1' in r else '—'
            uvs = f'{r["voiced_agree"]:.1%}' if 'voiced_agree' in r else '—'
            print(f'{item:26s} {label:22s} {r["seconds"]:5.2f}s {r["peak"]:6.3f} '
                  f'{r["rms_db"]:7.1f} {f0s:>9s} {mel:>7s} {uvs:>7s}')
        print()

    # Scores have no reference recording -- sanity numbers only.
    score_wavs = [w for w in wavs if w.name.startswith('5_score')]
    if score_wavs:
        print(f'{"score synthesis (no original)":30s} {"":20s} {"len":>6s} {"peak":>6s} {"RMS dB":>7s}')
        print('-' * 70)
        for w in score_wavs:
            gen = load_wav(w, sr)
            r = compare(None, gen, hparams, pe)
            results[w.name] = r
            print(f'{w.stem:49s} {r["seconds"]:5.2f}s {r["peak"]:6.3f} {r["rms_db"]:7.1f}')

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
        print(f'\n→ {args.json}')


if __name__ == '__main__':
    main()
