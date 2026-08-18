#!/usr/bin/env python3
"""Pitch-accuracy metrics, all measured against a reference f0 contour.

Everything here works in cents rather than Hz, because a 20 Hz error at 80 Hz
is a different musical event from a 20 Hz error at 600 Hz, and this corpus
spans both. 100 cents = one semitone.

The reference is the CREPE contour from Stage 8, which is the closest thing to
ground truth we have for this corpus (see pansori/preprocess/08_f0.py).
"""

import numpy as np

# One semitone. A pitch error smaller than this is inaudible as a wrong note;
# anything larger means the model sang a different note than it should have.
GROSS_ERROR_CENTS = 50.0

# Adjacent 5.333ms frames cannot really move this far. See 08_f0.py.
OCTAVE_JUMP_CENTS = 600.0


def _align(pred, ref):
    n = min(len(pred), len(ref))
    return np.asarray(pred[:n], dtype=np.float64), np.asarray(ref[:n], dtype=np.float64)


def cents_error(pred, ref):
    """Signed cents difference on frames both contours call voiced."""
    pred, ref = _align(pred, ref)
    both = (pred > 0) & (ref > 0)
    if not both.any():
        return np.zeros(0)
    return 1200.0 * np.log2(pred[both] / ref[both])


def cents_rmse(pred, ref):
    err = cents_error(pred, ref)
    return float(np.sqrt((err ** 2).mean())) if len(err) else float('nan')


def gross_error_rate(pred, ref, tol=GROSS_ERROR_CENTS):
    """Fraction of commonly-voiced frames off by more than a semitone."""
    err = cents_error(pred, ref)
    return float((np.abs(err) > tol).mean()) if len(err) else float('nan')


def vuv_error_rate(pred, ref):
    """Fraction of frames where the two contours disagree about voicing.

    Kept separate from the cents metrics on purpose: a contour can be perfectly
    in tune on the frames it does voice while dropping half the utterance, and
    that failure is inaudible to an RMSE computed over the surviving frames.
    """
    pred, ref = _align(pred, ref)
    return float(((pred > 0) != (ref > 0)).mean()) if len(pred) else float('nan')


def octave_jump_rate(f0):
    """Fraction of voiced-to-voiced transitions that jump implausibly far."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    both = voiced[:-1] & voiced[1:]
    if not both.any():
        return 0.0
    jump = 1200.0 * np.log2(f0[1:][both] / f0[:-1][both])
    return float((np.abs(jump) > OCTAVE_JUMP_CENTS).mean())


def summarize(pred, ref):
    """All four metrics for one utterance, as a dict."""
    return {
        'cents_rmse': cents_rmse(pred, ref),
        'gross_error': gross_error_rate(pred, ref),
        'vuv_error': vuv_error_rate(pred, ref),
        'octave_jump': octave_jump_rate(pred),
    }


def aggregate(rows):
    """Mean of each metric across utterances, ignoring NaN."""
    if not rows:
        return {}
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def mel_l1(pred_mel, gt_mel):
    """Mean absolute mel difference over the frames the two have in common."""
    pred_mel, gt_mel = np.asarray(pred_mel), np.asarray(gt_mel)
    n = min(len(pred_mel), len(gt_mel))
    return float(np.abs(pred_mel[:n] - gt_mel[:n]).mean())
