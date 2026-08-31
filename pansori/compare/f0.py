#!/usr/bin/env python3
"""One f0 extractor, one grid, both systems.

RMVPE resamples whatever it is given to 16 kHz and runs at 100 Hz internally
(modules/pe/rmvpe/inference.py:68-78), then maps onto the requested hop. So it is
sample-rate agnostic -- which is the whole reason a 24 kHz system and a 44.1 kHz
system can be compared at all -- and asking for hop = round(0.01*sr) makes that
final resample the identity on both sides.

Note that get_pitch's f0_min/f0_max arguments are accepted and never used by this
backend; the range is fixed in modules/pe/rmvpe/constants.py. That is fine here
(RMVPE's range comfortably covers 55-750 Hz) but it means the pansori config's
f0_min/f0_max are not doing what one would assume, so nothing in this file
pretends they are.

The old pipeline's CREPE contours are also loadable, because the old model was
trained on them and reporting its pitch metrics only against RMVPE would penalise
it for reproducing its own training target.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402

_PE = None
_HP = False


def _init():
    global _PE, _HP
    if _PE is None:
        from utils.hparams import set_hparams
        if not _HP:
            set_hparams(config=str(C.REPO / 'configs' / 'exp' / 'ac_v1.yaml'),
                        print_hparams=False)
            _HP = True
        from modules.pe import initialize_pe
        _PE = initialize_pe()
    return _PE


def _cache_key(path):
    st = Path(path).stat()
    raw = f'{Path(path).resolve()}|{st.st_size}|{int(st.st_mtime)}|{C.TIMESTEP}'
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def contour(path, use_cache=True):
    """(f0[K], voiced[K]) on the 10 ms grid. Unvoiced frames are 0.

    K is derived from the file's own duration, so two files of the same content
    at different sample rates land on the same number of frames.
    """
    path = Path(path)
    cache = C.CACHE / 'f0' / f'{path.stem}_{_cache_key(path)}.npz'
    if use_cache and cache.exists():
        d = np.load(cache)
        return d['f0'], d['voiced']

    import librosa
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    hop = int(round(C.TIMESTEP * sr))
    length = int(round(len(wav) / sr / C.TIMESTEP))
    pe = _init()
    f0, uv = pe.get_pitch(wav, samplerate=sr, length=length, hop_size=hop,
                          f0_min=55, f0_max=750, interp_uv=False)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = ~np.asarray(uv, dtype=bool)
    f0[~voiced] = 0.0

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, f0=f0, voiced=voiced)
    return f0, voiced


def crepe_contour(item, n_frames):
    """The OLD pipeline's CREPE contour, resampled 187.5 Hz -> 100 Hz.

    Returns None when the file is absent. The resampling smooths slightly, which
    biases within-note std downward for this reference -- so CREPE-referenced and
    RMVPE-referenced numbers cannot be compared naively either, and the report
    says so rather than quietly putting them in one column.
    """
    path = C.OLD_REPO / 'pansori' / 'data' / 'full' / 'f0' / f'{item}.npy'
    if not path.exists():
        return None
    from utils.infer_utils import resample_align_curve
    raw = np.load(path).astype(np.float64)
    voiced_raw = raw > 0
    # interpolate through unvoiced before resampling, or the zeros bleed into
    # neighbouring voiced frames and manufacture huge fake pitch excursions
    if voiced_raw.any():
        idx = np.flatnonzero(voiced_raw)
        filled = np.interp(np.arange(len(raw)), idx, raw[idx])
    else:
        filled = raw
    f0 = resample_align_curve(filled, 1.0 / 187.5, C.TIMESTEP, n_frames)
    v = resample_align_curve(voiced_raw.astype(np.float64), 1.0 / 187.5,
                             C.TIMESTEP, n_frames) > 0.5
    f0 = np.asarray(f0, dtype=np.float64)
    f0[~v] = 0.0
    return f0, np.asarray(v, dtype=bool)


def cents(a, b):
    return 1200.0 * np.log2(np.maximum(a, 1e-5) / np.maximum(b, 1e-5))


def truncate(*series):
    """Truncate a set of frame series to the shortest. Callers assert the gap first."""
    n = min(len(s) for s in series)
    return [s[:n] for s in series]


def main():
    import argparse
    p = argparse.ArgumentParser(description='f0 추출 확인')
    p.add_argument('wav')
    args = p.parse_args()
    f0, v = contour(args.wav)
    print(f'{len(f0)} 프레임 ({len(f0) * C.TIMESTEP:.3f}s), 유성 {v.mean():.1%}, '
          f'{f0[v].min():.1f}-{f0[v].max():.1f} Hz')


if __name__ == '__main__':
    main()
