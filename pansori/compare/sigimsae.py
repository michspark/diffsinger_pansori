#!/usr/bin/env python3
"""시김새 measured from the f0 curve, with no manual labelling.

The decomposition is the one pansori/docs/SIGIMSAE_PLAN.md §1 already applied to
the old model, kept field-for-field so the historical table extends rather than
restarts:

    residual(t) = f0(t) - note_staircase(t)      in cents
                = bias(n)                        per-note constant -- tuning
                + within(t)                      the part that moves -- 시김새

Three numbers decide whether a model sings 시김새, and only the third is hard to
fake: how much ground truth moves inside a note, how much the model moves, and
whether the two move *together*. A model with the right amount of motion and zero
correlation is generating decorative noise, which is worse than a flat line
because it sounds drunk rather than plain.

One correction is applied before any of it: **octave snapping to the note**.
RMVPE accepts f0_min/f0_max and ignores them (modules/pe/rmvpe/inference.py:63-78),
and on male 통성 it drops octaves -- on 8b4d0824_0000 it reads 92 Hz where CREPE
reads 192 Hz, inside a single sustained B3. Across the 19 paired items a median
4.4% of voiced frames sit more than 600 cents from their own note, and two items
reach 24-28%. Left alone that inflates the *reference's* own within-note std to
650 cents against a published 161.4.

So the residual is wrapped into (-600, 600] before the decomposition, identically
for the reference and for every system. 시김새 excursions in this corpus are
135-160 cents (docs/VIBRATO_VERDICT.md), nowhere near 600, and the note step
itself is already removed by the staircase -- so a >600-cent excursion inside a
note is extractor or label error, not ornament. After snapping the reference
lands at 130-200 cents, which is where the historical measurement put it.

The correction is confined to Tier 2. Pitch accuracy (pitch_metrics.py) is
measured on the raw contour, so a model that genuinely sings the wrong octave is
still penalised there -- RPA counts it, RCA does not, and the gap between them is
the report's octave-error column. The snap rate is carried alongside every row.

Two deliberate omissions:

  * No sinusoid vibrato fitting. pansori/docs/VIBRATO_VERDICT.md refuted it in
    this very corpus -- the same sine fit *unrelated* notes' residuals as well or
    better (control R^2 0.829 vs actual 0.782), and only 3.9% of notes have a
    clean periodic component. A rate/depth metric built on that fit measures the
    fitter's flexibility over short windows, and would flatter whichever system
    produces smoother contours for reasons unrelated to 요성.

  * No |Δf0| per *frame*. The old pipeline framed at 5.33 ms and this one at
    11.61 ms, so the same physical contour scores ~2.2x higher on the new grid.
    Motion here is cents per *second* on the shared 10 ms grid; the per-frame
    ratios in docs/NEXT.md §C5 are not comparable across systems and are not
    reused.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import notes as N  # noqa: E402


# ---------------------------------------------------------------- decomposition

def octave_snap(residual, ok):
    """Wrap the note residual into (-600, 600]. Returns (snapped, rate corrected)."""
    out = residual.copy()
    out[ok] = (residual[ok] + 600.0) % 1200.0 - 600.0
    rate = float((np.abs(residual[ok] - out[ok]) > 1e-6).mean()) if ok.any() else float('nan')
    return out, rate


def snap_f0(f0, voiced, note_id, stair):
    """f0 with each voiced frame moved to the octave nearest its own note.

    Used for the motion and wobble statistics, which read the contour directly
    rather than the residual and would otherwise count a 1200-cent extractor slip
    as a very fast ornament.
    """
    ok = voiced & (stair > 0) & (note_id >= 0) & (f0 > 0)
    out = f0.copy()
    resid = np.zeros(len(f0))
    resid[ok] = 1200.0 * np.log2(f0[ok] / stair[ok])
    snapped, _ = octave_snap(resid, ok)
    out[ok] = stair[ok] * 2.0 ** (snapped[ok] / 1200.0)
    return out


def decompose(f0, voiced, note_id, stair, snap=True):
    """residual, bias-per-frame, within, and the mask they are valid on."""
    ok = voiced & (stair > 0) & (note_id >= 0) & (f0 > 0)
    residual = np.full(len(f0), np.nan)
    residual[ok] = 1200.0 * np.log2(f0[ok] / stair[ok])
    snap_rate = float('nan')
    if snap:
        residual, snap_rate = octave_snap(residual, ok)

    bias = np.full(len(f0), np.nan)
    for n in np.unique(note_id[ok]):
        sel = ok & (note_id == n)
        if sel.sum() >= 1:
            bias[sel] = residual[sel].mean()
    within = residual - bias
    return residual, bias, within, ok, snap_rate


def _rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt((x ** 2).mean())) if x.size else float('nan')


def _std(x):
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size > 1 else float('nan')


def note_stats(f0, voiced, item, n_frames=None):
    """Per-utterance 시김새 statistics for one contour."""
    n_frames = n_frames or len(f0)
    note_id, stair = N.frame_notes(item, n_frames)
    f0, voiced, note_id, stair = (a[:n_frames] for a in (f0, voiced, note_id, stair))
    residual, bias, within, ok, snap_rate = decompose(f0, voiced, note_id, stair)
    core = N.note_core_mask(note_id, ok)
    f0_snapped = snap_f0(f0, voiced, note_id, stair)

    # note-weighted variants: frame weighting lets one long held note dominate an
    # utterance, which in this corpus (notes up to 5.3 s) is not a corner case
    per_note = []
    for n in np.unique(note_id[ok]):
        w = within[ok & (note_id == n)]
        if w.size > 1:
            per_note.append(float(w.std()))

    return {
        'bias_rms': _rms(bias[ok]),
        'within_std': _std(within[ok]),
        'within_std_core': _std(within[core]),
        'residual_rms': _rms(residual[ok]),
        'within_std_note_weighted': float(np.mean(per_note)) if per_note else float('nan'),
        'n_notes': int(len(per_note)),
        'n_voiced': int(ok.sum()),
        'octave_snap_rate': snap_rate,
        '_within': within,
        '_ok': ok,
        '_per_note_std': np.array(per_note),
        '_f0_snapped': f0_snapped,
        '_voiced': voiced,
    }


def within_correlation(stats_g, stats_r):
    """Pearson r between model and reference within-note motion, and its best lag.

    The lag matters: a low correlation with a large best lag means the ornament
    exists but is mistimed, which is a different failure from not producing one.
    """
    n = min(len(stats_g['_within']), len(stats_r['_within']))
    both = stats_g['_ok'][:n] & stats_r['_ok'][:n]
    a = stats_g['_within'][:n][both]
    b = stats_r['_within'][:n][both]
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if a.size < 8 or a.std() < 1e-6 or b.std() < 1e-6:
        return float('nan'), 0
    r = float(np.corrcoef(a, b)[0, 1])

    best_r, best_lag = r, 0
    for lag in range(-10, 11):
        if lag == 0:
            continue
        x = a[max(0, lag):len(a) + min(0, lag)]
        y = b[max(0, -lag):len(b) + min(0, -lag)]
        if x.size < 8 or x.std() < 1e-6 or y.std() < 1e-6:
            continue
        rr = float(np.corrcoef(x, y)[0, 1])
        if rr > best_r:
            best_r, best_lag = rr, lag
    return r, best_lag


def within_std_distance(stats_g, stats_r):
    """Wasserstein-1 between the per-note within-std distributions.

    Catches the case where the pooled std matches but the shape does not -- e.g.
    uniform mild wobble everywhere instead of a few strongly ornamented notes.
    """
    from scipy.stats import wasserstein_distance
    a, b = stats_g['_per_note_std'], stats_r['_per_note_std']
    if a.size < 3 or b.size < 3:
        return float('nan')
    return float(wasserstein_distance(a, b))


# ---------------------------------------------------------------- motion

def motion(f0, voiced):
    """Median |Δf0| in CENTS PER SECOND, over runs of consecutive voiced frames.

    Restricted to runs so a jump across an unvoiced gap -- a phrase boundary, not
    an ornament -- does not count as movement. That rule is from
    pansori/examples/f0_report.py:motion() and is correct; only the unit changes.
    """
    deltas, run = [], []
    for value, v in zip(f0, voiced):
        if v and value > 0:
            run.append(value)
        else:
            if len(run) > 2:
                deltas.append(np.abs(np.diff(1200 * np.log2(np.maximum(run, 1e-5)))))
            run = []
    if len(run) > 2:
        deltas.append(np.abs(np.diff(1200 * np.log2(np.maximum(run, 1e-5)))))
    if not deltas:
        return float('nan')
    return float(np.median(np.concatenate(deltas))) / C.TIMESTEP


def spread(f0, voiced):
    v = 1200 * np.log2(np.maximum(f0[voiced & (f0 > 0)], 1e-5))
    return float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size > 10 else float('nan')


def wobble(f0, voiced):
    """IQR of the contour after removing a 150 ms median-filtered baseline.

    Separates ornamental wobble from note-to-note movement with no periodicity
    assumption at all. When motion and within-note std disagree, this is the one
    to trust -- motion is sensitive to frame-level extractor jitter and
    within-note std inherits ROSVOT's note boundaries, while this inherits
    neither.
    """
    from scipy.signal import medfilt
    sel = voiced & (f0 > 0)
    if sel.sum() < 20:
        return float('nan')
    c = np.full(len(f0), np.nan)
    c[sel] = 1200 * np.log2(f0[sel])
    idx = np.flatnonzero(sel)
    filled = np.interp(np.arange(len(f0)), idx, c[idx])
    k = int(round(C.WOBBLE_MEDIAN_MS / 1000.0 / C.TIMESTEP)) | 1
    resid = (filled - medfilt(filled, k))[sel]
    return float(np.percentile(resid, 75) - np.percentile(resid, 25))


# ---------------------------------------------------------------- transitions

def _slope(y, x_step=C.TIMESTEP):
    """OLS slope in cents/second. NaN if fewer than 3 finite points."""
    good = np.isfinite(y)
    if good.sum() < 3:
        return float('nan')
    x = np.arange(len(y))[good] * x_step
    y = y[good]
    return float(np.polyfit(x, y, 1)[0])


def transitions(f0, voiced, item, n_frames=None):
    """Approach / release behaviour at each interior note boundary.

    Measured on the *residual*, so the note step itself is already removed and
    what is left is the singer's approach to the note -- 추성 (pushing up into it),
    퇴성 (falling away), 꺾기 (the pre-strike against the direction of the step).

    The window is 120 ms rather than 40 ms because ROSVOT boundaries carry
    +-1-2 frames of error; direction agreement is the robust statistic here and
    absolute slope is not.
    """
    n_frames = n_frames or len(f0)
    note_id, stair = N.frame_notes(item, n_frames)
    f0, voiced, note_id, stair = (a[:n_frames] for a in (f0, voiced, note_id, stair))
    residual, bias, within, ok, _ = decompose(f0, voiced, note_id, stair)

    w = int(round(C.TRANSITION_MS / 1000.0 / C.TIMESTEP))
    ret = int(round(C.KKEOKGI_RETURN_MS / 1000.0 / C.TIMESTEP))
    rows = []
    for k, midi_a, midi_b, step in N.boundaries(item, n_frames):
        after = np.where(ok[k:k + w], residual[k:k + w], np.nan)
        before = np.where(ok[max(0, k - w):k], residual[max(0, k - w):k], np.nan)
        if np.isfinite(after).sum() < 3 or np.isfinite(before).sum() < 3:
            continue

        bias_after = bias[k] if np.isfinite(bias[k]) else np.nan
        over = np.nan
        if np.isfinite(after).any() and np.isfinite(bias_after):
            dev = after - bias_after
            over = float(dev[np.nanargmax(np.abs(dev))])

        # 꺾기: an excursion against the direction of the note step that comes
        # back through zero quickly. Defined by shape, not by fitting anything.
        kkeokgi = False
        if step != 0 and np.isfinite(bias_after):
            dev = after - bias_after
            against = -np.sign(step)
            hit = np.flatnonzero(np.isfinite(dev) & (dev * against > C.KKEOKGI_CENTS))
            if hit.size:
                tail = dev[hit[0]:hit[0] + ret]
                if np.nanmin(tail * against) <= 0:
                    kkeokgi = True

        rows.append({
            'frame': k,
            'step_cents': step,
            'approach_slope': _slope(after),
            'release_slope': _slope(before),
            'overshoot': over,
            'kkeokgi': kkeokgi,
        })
    return rows


def transition_agreement(rows_g, rows_r, floor=C.SLOPE_FLOOR):
    """Direction agreement and slope magnitude ratio against the reference.

    Only boundaries where the reference slope exceeds `floor` count: a near-flat
    boundary has no direction, and including it would inflate agreement toward
    50% by coin flip regardless of what the model did.
    """
    by_frame = {r['frame']: r for r in rows_r}
    agree_a, agree_b, mag_a, mag_b = [], [], [], []
    for g in rows_g:
        r = by_frame.get(g['frame'])
        if r is None:
            continue
        for key, agree, mag in (('approach_slope', agree_a, mag_a),
                                ('release_slope', agree_b, mag_b)):
            gv, rv = g[key], r[key]
            if not (np.isfinite(gv) and np.isfinite(rv)):
                continue
            if abs(rv) > floor:
                agree.append(float(np.sign(gv) == np.sign(rv)))
            mag.append((abs(gv), abs(rv)))

    def ratio(pairs):
        if not pairs:
            return float('nan')
        a = np.median([p[0] for p in pairs])
        b = np.median([p[1] for p in pairs])
        return float(a / b) if b > 1e-9 else float('nan')

    return {
        'transition_dir_agree': float(np.mean(agree_a)) if agree_a else float('nan'),
        'transition_dir_agree_release': float(np.mean(agree_b)) if agree_b else float('nan'),
        'transition_n': len(agree_a),
        'slope_mag_ratio': ratio(mag_a),
        'kkeokgi_rate': (float(np.mean([r['kkeokgi'] for r in rows_g]))
                         if rows_g else float('nan')),
        'kkeokgi_rate_ref': (float(np.mean([r['kkeokgi'] for r in rows_r]))
                             if rows_r else float('nan')),
    }


# ---------------------------------------------------------------- top level

def evaluate(f0_g, voiced_g, f0_r, voiced_r, item):
    """Every 시김새 metric for one (system, reference) pair on one item."""
    n = min(len(f0_g), len(f0_r))
    g = note_stats(f0_g[:n], voiced_g[:n], item, n)
    r = note_stats(f0_r[:n], voiced_r[:n], item, n)
    corr, lag = within_correlation(g, r)

    out = {
        'bias_rms': g['bias_rms'],
        'within_std': g['within_std'],
        'within_std_core': g['within_std_core'],
        'within_std_note_weighted': g['within_std_note_weighted'],
        'residual_rms': g['residual_rms'],
        'within_corr': corr,
        'within_corr_lag': lag,
        'within_std_wasserstein': within_std_distance(g, r),
        # motion and wobble read the contour directly, so they use the
        # octave-snapped series for the same reason the residual is snapped
        'motion_cps': motion(g['_f0_snapped'], g['_voiced']),
        'spread_cents': spread(g['_f0_snapped'], g['_voiced']),
        'wobble_iqr': wobble(g['_f0_snapped'], g['_voiced']),
        'octave_snap_rate': g['octave_snap_rate'],
        'ref_bias_rms': r['bias_rms'],
        'ref_within_std': r['within_std'],
        'ref_within_std_core': r['within_std_core'],
        'ref_motion_cps': motion(r['_f0_snapped'], r['_voiced']),
        'ref_wobble_iqr': wobble(r['_f0_snapped'], r['_voiced']),
        'ref_octave_snap_rate': r['octave_snap_rate'],
        'n_notes': g['n_notes'],
    }
    # Ratios are the headline rather than raw levels because they cancel the
    # staircase choice, most of the reference-extractor choice, and per-singer
    # register -- the three things that would otherwise make the old and new
    # tables incommensurable.
    for key, num, den in (('within_std_ratio', 'within_std', 'ref_within_std'),
                          ('within_std_core_ratio', 'within_std_core', 'ref_within_std_core'),
                          ('motion_ratio', 'motion_cps', 'ref_motion_cps'),
                          ('wobble_ratio', 'wobble_iqr', 'ref_wobble_iqr')):
        d = out[den]
        out[key] = float(out[num] / d) if d and np.isfinite(d) and d > 1e-9 else float('nan')

    out.update(transition_agreement(
        transitions(f0_g[:n], voiced_g[:n], item, n),
        transitions(f0_r[:n], voiced_r[:n], item, n),
    ))
    return out
