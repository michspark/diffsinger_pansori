#!/usr/bin/env python3
"""Pitch accuracy, with the denominator that cannot be gamed.

pansori/eval/f0_metrics.py:gross_error_rate divides by the *commonly voiced*
frames, so a system that voices only the easy frames scores better by dropping
the hard ones -- and RMSE, computed on the same frames, hides it too. RPA here
divides by the reference's voiced frames instead, which does not have that
property. Both are reported: when they order the systems differently, the V/UV
column is the explanation and it gets its own paragraph in the report.

Everything is plain numpy so it runs in `dsp` without mir_eval; mir_eval 0.8.2 is
present there and cross_check() uses it to confirm RPA/RCA independently.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare.f0 import cents  # noqa: E402

GROSS_CENTS = 100.0
OCTAVE_CENTS = 600.0
TOL = 50.0


def evaluate(f0_g, voiced_g, f0_r, voiced_r):
    n = min(len(f0_g), len(f0_r))
    f0_g, voiced_g, f0_r, voiced_r = (a[:n] for a in (f0_g, voiced_g, f0_r, voiced_r))
    ref = voiced_r & (f0_r > 0)
    both = ref & voiced_g & (f0_g > 0)

    out = {
        'voiced_agree': float((voiced_r == voiced_g).mean()),
        'vuv_error': float((voiced_r != voiced_g).mean()),
        'ref_voiced_frac': float(ref.mean()),
        'n_frames': int(n),
    }
    if both.sum() < 10:
        out.update({k: float('nan') for k in
                    ('rpa50', 'rca50', 'f0_rmse_cent', 'gross_error', 'rpa50_v_denom')})
        out['octave_jump'] = octave_jump_rate(f0_g, voiced_g)
        return out

    err = cents(f0_g[both], f0_r[both])
    # chroma: wrap into (-600, 600] so an octave slip stops counting as detuning
    chroma = (err + 600.0) % 1200.0 - 600.0

    out['rpa50'] = float((np.abs(err) <= TOL).sum() / ref.sum())
    out['rca50'] = float((np.abs(chroma) <= TOL).sum() / ref.sum())
    out['rpa50_v_denom'] = float((np.abs(err) <= TOL).mean())      # the legacy denominator
    out['f0_rmse_cent'] = float(np.sqrt((err ** 2).mean()))
    out['f0_rmse_cent_chroma'] = float(np.sqrt((chroma ** 2).mean()))
    out['gross_error'] = float((np.abs(err) > GROSS_CENTS).mean())
    out['octave_jump'] = octave_jump_rate(f0_g, voiced_g)
    return out


def octave_jump_rate(f0, voiced):
    """Fraction of voiced->voiced frame transitions that jump more than 600 cents."""
    sel = voiced & (f0 > 0)
    pair = sel[:-1] & sel[1:]
    if pair.sum() < 5:
        return float('nan')
    d = np.abs(cents(f0[1:][pair], f0[:-1][pair]))
    return float((d > OCTAVE_CENTS).mean())


def cross_check(f0_g, voiced_g, f0_r, voiced_r):
    """Independent RPA/RCA from mir_eval, to confirm the hand-rolled versions."""
    try:
        import mir_eval.melody as mel
    except ImportError:
        return {}
    n = min(len(f0_g), len(f0_r))
    t = np.arange(n) * 0.01
    ref_v, ref_c = mel.freq_to_voicing(f0_r[:n].copy())
    est_v, est_c = mel.freq_to_voicing(f0_g[:n].copy())
    ref_c = mel.hz2cents(np.maximum(f0_r[:n], 1e-5))
    est_c = mel.hz2cents(np.maximum(f0_g[:n], 1e-5))
    return {
        'mir_rpa': float(mel.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=TOL)),
        'mir_rca': float(mel.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=TOL)),
        'mir_oa': float(mel.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=TOL)),
    }
