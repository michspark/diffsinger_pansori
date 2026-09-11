#!/usr/bin/env python3
"""Timbre distance, on a basis neither system was trained on.

The two systems have different mel bases -- 80 bins to 12 kHz against 128 bins to
16 kHz, at different hops, in different log bases -- so comparing one system's
mel L1 to the other's compares two different definitions and means nothing. Both
outputs are therefore resampled to a third, common analysis: 24 kHz, 80 mels,
0-11 kHz, at the same 10 ms hop the f0 metrics use.

Down to 24 kHz, never up. Upsampling the old system's output would hand it a
band its vocoder never produced, and every high-frequency error metric would
improve for free.

What this can claim: under a common 24 kHz / 0-11 kHz analysis, one system's
timbre sits closer to the recording than the other's, on frame-aligned rungs.

What it cannot: anything about 11-22 kHz. The old system physically cannot
produce that band, so it is excluded rather than scored -- which makes every
number here a LOWER bound on the new system's advantage. band_energy() reports
what was set aside, as a fact rather than as a score.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402

_MEL_BASIS = None


def _basis():
    global _MEL_BASIS
    if _MEL_BASIS is None:
        import librosa
        _MEL_BASIS = librosa.filters.mel(
            sr=C.SPEC['sr'], n_fft=C.SPEC['n_fft'], n_mels=C.SPEC['n_mels'],
            fmin=C.SPEC['fmin'], fmax=C.SPEC['fmax'],
        )
    return _MEL_BASIS


def load_common(path):
    """Mono waveform at the common 24 kHz analysis rate."""
    import librosa
    wav, _ = librosa.load(str(path), sr=C.SPEC['sr'], mono=True)
    return wav


def logmel(wav):
    """log10 mel power on the common basis, [T, n_mels]."""
    import librosa
    S = np.abs(librosa.stft(wav, n_fft=C.SPEC['n_fft'], hop_length=C.SPEC['hop'],
                            win_length=C.SPEC['n_fft'])) ** 2
    mel = _basis() @ S
    return np.log10(np.maximum(mel, 1e-10)).T


def mfcc(log_mel, n=C.MFCC_N):
    """DCT-II of the log-mel, c0 dropped.

    c0 is overall gain, which differs between systems for reasons that have
    nothing to do with timbre (vocoder output level, per-file normalisation), so
    keeping it would let a loudness offset masquerade as spectral distortion.
    """
    from scipy.fftpack import dct
    return dct(log_mel, type=2, axis=1, norm='ortho')[:, 1:n]


SILENCE_DB = 40.0   # frames this far below a file's own peak are not scored


def _active(log_mel, floor_db=SILENCE_DB):
    """Frames whose broadband level is within `floor_db` of this file's peak.

    Without this the metric is dominated by silence: in a near-silent frame one
    signal sits on the 1e-10 floor and the other does not, and the cepstral
    difference that produces is enormous and says nothing about timbre. Every
    published MCD restricts to non-silent frames; this is that restriction.
    """
    level = 10.0 * log_mel.mean(axis=1)          # dB-ish, log10 power
    return level > (level.max() - floor_db)


def evaluate(gen_path, ref_path):
    """MCD (dB), log-spectral distance, and mel L1, all on the common basis."""
    g, r = load_common(gen_path), load_common(ref_path)
    mg, mr = logmel(g), logmel(r)
    n = min(len(mg), len(mr))
    mg, mr = mg[:n], mr[:n]

    keep = _active(mg) & _active(mr)
    if keep.sum() < 10:
        keep = np.ones(n, dtype=bool)
    mg_a, mr_a = mg[keep], mr[keep]

    # Cepstra from the natural-log AMPLITUDE spectrum, which is what the standard
    # MCD constant assumes: log_amp = 0.5 * ln(power) = 0.5*ln(10) * log10(power).
    cg, cr = mfcc(0.5 * np.log(10.0) * mg_a), mfcc(0.5 * np.log(10.0) * mr_a)
    d = np.sqrt(2.0 * ((cg - cr) ** 2).sum(axis=1))
    return {
        'mcd_db': float((10.0 / np.log(10.0)) * d.mean()),
        'lsd': float(np.sqrt(((mg_a - mr_a) ** 2).mean())),
        'mel_l1_common': float(np.abs(mg_a - mr_a).mean()),
        'spec_frames': int(n),
        'spec_active_frac': float(keep.mean()),
    }


def band_energy(path, cutoff=11000.0):
    """Fraction of energy above `cutoff`, at the file's own sample rate.

    Not a score. It is the statement of what the band-limited comparison had to
    set aside, and the only place the new system's extra octave is visible.
    """
    import librosa
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    if sr / 2 <= cutoff:
        return {'above_cutoff_frac': 0.0, 'nyquist': sr / 2}
    S = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = S.sum()
    return {
        'above_cutoff_frac': float(S[freqs > cutoff].sum() / total) if total > 0 else 0.0,
        'nyquist': sr / 2,
    }
