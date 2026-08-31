#!/usr/bin/env python3
"""Inline SVG figures for the comparison report.

Hand-written SVG rather than matplotlib PNGs, because the report is published as
an artifact that renders in whichever theme the reader is using: colours come from
CSS custom properties so light and dark are one document, and text stays selectable.

Three series only -- 원본 / 구 / 신 -- which is the largest set that clears the
all-pairs CVD and normal-vision floors in both modes. The note staircase is drawn
as recessive structure, not as a fourth series, because that is what it is.
Every series is direct-labelled as well as legended, which is also the relief the
light-mode aqua needs for its sub-3:1 contrast.
"""

import html
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import f0 as F  # noqa: E402
from pansori.compare import items as I  # noqa: E402
from pansori.compare import notes as N  # noqa: E402
from pansori.compare import sigimsae as S  # noqa: E402

SERIES = {'gt': 'var(--series-1)', 'old': 'var(--series-2)', 'new': 'var(--series-3)'}
LABEL = {'gt': '원본', 'old': '구 모델', 'new': '신 모델'}


def _esc(s):
    return html.escape(str(s), quote=True)


def _poly(xs, ys, colour, width=2, dash=None, opacity=1.0):
    """One polyline per unbroken run; NaN breaks the line instead of bridging it."""
    out = []
    run = []
    for x, y in zip(xs, ys):
        if y is None or not np.isfinite(y):
            if len(run) > 1:
                out.append(run)
            run = []
        else:
            run.append((x, y))
    if len(run) > 1:
        out.append(run)
    d = ' '.join(
        'M ' + ' L '.join(f'{x:.1f},{y:.1f}' for x, y in seg) for seg in out)
    if not d:
        return ''
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{extra}/>')


# ---------------------------------------------------------------- f0 overlay

def f0_overlay(item, seed=1, width=880, height=300, window=None):
    """Ground truth, note staircase, old model and new model on one pitch axis.

    This is the figure the whole report rests on: the flattening the old model was
    diagnosed with, and whatever the new one does instead, are visible here before
    any statistic is quoted.
    """
    ref_f0, ref_v = F.contour(I.wav_of(item))
    n = len(ref_f0)
    note_id, stair = N.frame_notes(item, n)

    tracks = {'gt': (ref_f0, ref_v)}
    p_old = C.OLD_LADDER[C.OLD_PRIMARY] / f'{item}__5_pred_own.wav'
    if p_old.exists():
        f, v = F.contour(p_old)
        tracks['old'] = (f[:n], v[:n])
    p_new = C.OUT / 'new' / f'{item}__3_pitch__s{seed}.wav'
    if p_new.exists():
        f, v = F.contour(p_new)
        tracks['new'] = (f[:n], v[:n])

    # octave-snapped, same as the metrics -- otherwise an extractor slip dominates
    # the vertical scale and the ornament is invisible
    snapped = {}
    for k, (f, v) in tracks.items():
        m = min(len(f), n)
        snapped[k] = (S.snap_f0(f[:m], v[:m], note_id[:m], stair[:m]), v[:m])

    a, b = window or (0, n)
    cents = {k: np.where(v[a:b] & (f[a:b] > 0),
                         1200 * np.log2(np.maximum(f[a:b], 1e-5)), np.nan)
             for k, (f, v) in snapped.items()}
    stair_c = np.where(stair[a:b] > 0,
                       1200 * np.log2(np.maximum(stair[a:b], 1e-5)), np.nan)

    allv = np.concatenate([v[np.isfinite(v)] for v in cents.values()] +
                          [stair_c[np.isfinite(stair_c)]])
    if allv.size == 0:
        return '<p>표시할 f0 가 없다.</p>'
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    pad = max(60.0, (hi - lo) * 0.12)
    lo, hi = lo - pad, hi + pad

    L, R, T, B = 54, 96, 16, 34
    pw, ph = width - L - R, height - T - B

    def X(i):
        return L + pw * i / max(1, (b - a) - 1)

    def Y(c):
        return T + ph * (1 - (c - lo) / (hi - lo))

    xs = [X(i) for i in range(b - a)]

    def ys_of(arr):
        # Break the line wherever the octave snap flips sides. Snapping keeps every
        # frame within +-600 cents of its note, so a flip is a discontinuity in the
        # representation, not a leap the singer made -- drawing it would put a
        # 1200-cent spike through the middle of the ornament we came to look at.
        out = [Y(c) if np.isfinite(c) else np.nan for c in arr]
        for i in range(1, len(arr)):
            if (np.isfinite(arr[i]) and np.isfinite(arr[i - 1])
                    and abs(arr[i] - arr[i - 1]) > 600):
                out[i] = np.nan
        return out
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="{_esc(item)} f0 비교" '
             f'style="max-width:{width}px;height:auto">']

    # note boundaries + staircase: recessive structure
    for k, _, _, _ in N.boundaries(item, n):
        if a <= k < b:
            parts.append(f'<line x1="{X(k - a):.1f}" y1="{T}" x2="{X(k - a):.1f}" '
                         f'y2="{T + ph}" stroke="var(--grid)" stroke-width="1"/>')
    parts.append(_poly(xs, ys_of(stair_c), 'var(--grid-strong)', width=2, dash='5 4'))

    # y axis: semitone gridlines relative to the lowest labelled note
    step = 100.0 if (hi - lo) < 900 else 200.0
    c0 = np.ceil(lo / step) * step
    while c0 < hi:
        y = Y(c0)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'class="tick">{_midi_name(c0)}</text>')
        c0 += step

    for k in ('gt', 'old', 'new'):
        if k in cents:
            parts.append(_poly(xs, ys_of(cents[k]), SERIES[k], width=2 if k == 'gt' else 1.8))

    # direct labels at each series' last visible point
    used = []
    for k in ('gt', 'old', 'new'):
        if k not in cents:
            continue
        fin = np.flatnonzero(np.isfinite(cents[k]))
        if not fin.size:
            continue
        y = Y(cents[k][fin[-1]])
        while any(abs(y - u) < 15 for u in used):
            y += 15
        used.append(y)
        parts.append(f'<text x="{L + pw + 8}" y="{y + 4:.1f}" class="dlabel" '
                     f'fill="{SERIES[k]}">{LABEL[k]}</text>')

    secs = (b - a) * C.TIMESTEP
    parts.append(f'<text x="{L}" y="{height - 8}" class="tick">0 s</text>')
    parts.append(f'<text x="{L + pw}" y="{height - 8}" text-anchor="end" '
                 f'class="tick">{secs:.1f} s</text>')
    parts.append(f'<text x="{L + pw / 2}" y="{height - 8}" text-anchor="middle" '
                 f'class="tick">점선 = ROSVOT 노트 계단</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _midi_name(cents_val):
    """Absolute cents (1200*log2 Hz) -> note name. A4 = MIDI 69 = 440 Hz."""
    midi = int(round(69 + (cents_val - 1200 * np.log2(440.0)) / 100.0))
    return f'{_NAMES[midi % 12]}{midi // 12 - 1}'


# ---------------------------------------------------------------- dot plot

def paired_dots(summary, metric, width=880, row_h=22, target=None):
    """One row per utterance: where the old model sat, where the new one sits.

    A paired design deserves a paired figure. The median-and-CI in the table says
    how big the shift is; this says whether it happened everywhere or in three
    lucky items.
    """
    rec = next((r for r in summary['primary'] + summary['exploratory']
                if r['metric'] == metric), None)
    if rec is None:
        return ''
    import json
    rows = [json.loads(l) for l in
            (C.OUT / 'results.jsonl').read_text(encoding='utf-8').splitlines() if l]
    old_key = (summary['old_run'], '5_pred_own')
    vals = {'old': {}, 'new': {}}
    for r in rows:
        if r['metric'] != metric:
            continue
        if (r['system'], r['rung']) == old_key:
            vals['old'].setdefault(r['item'], []).append(r['value'])
        elif (r['system'], r['rung']) == ('NEW', '3_pitch'):
            vals['new'].setdefault(r['item'], []).append(r['value'])
    items = [i for i in rec['items'] if i in vals['old'] and i in vals['new']]
    if not items:
        return ''
    items.sort(key=lambda i: np.median(vals['new'][i]))

    lo = min(min(np.min(vals[k][i]) for k in ('old', 'new')) for i in items)
    hi = max(max(np.max(vals[k][i]) for k in ('old', 'new')) for i in items)
    if target is not None:
        lo, hi = min(lo, target), max(hi, target)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - span * 0.08, hi + span * 0.08

    L, R, T = 132, 60, 26
    height = T + row_h * len(items) + 30
    pw = width - L - R

    def X(v):
        return L + pw * (v - lo) / (hi - lo)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="{_esc(rec["label"])} 항목별 비교" '
             f'style="max-width:{width}px;height:auto">']
    if target is not None:
        parts.append(f'<line x1="{X(target):.1f}" y1="{T - 8}" x2="{X(target):.1f}" '
                     f'y2="{T + row_h * len(items)}" stroke="var(--grid-strong)" '
                     f'stroke-width="2" stroke-dasharray="4 4"/>')
        parts.append(f'<text x="{X(target):.1f}" y="{T - 12}" text-anchor="middle" '
                     f'class="tick">원본 수준 {target:g}</text>')

    for n_i, item in enumerate(items):
        y = T + row_h * n_i + row_h / 2
        o, nw = float(np.median(vals['old'][item])), float(np.median(vals['new'][item]))
        parts.append(f'<line x1="{X(o):.1f}" y1="{y:.1f}" x2="{X(nw):.1f}" y2="{y:.1f}" '
                     f'stroke="var(--grid-strong)" stroke-width="1.5"/>')
        # seed spread on the new side -- the noise floor, drawn where it applies
        vs = vals['new'][item]
        if len(vs) > 1:
            q1, q3 = np.percentile(vs, 25), np.percentile(vs, 75)
            parts.append(f'<line x1="{X(q1):.1f}" y1="{y:.1f}" x2="{X(q3):.1f}" '
                         f'y2="{y:.1f}" stroke="{SERIES["new"]}" stroke-width="6" '
                         f'opacity="0.28" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{X(o):.1f}" cy="{y:.1f}" r="5" fill="{SERIES["old"]}" '
                     f'stroke="var(--surface-1)" stroke-width="2"><title>'
                     f'{_esc(item)} 구 {o:.3f}</title></circle>')
        parts.append(f'<circle cx="{X(nw):.1f}" cy="{y:.1f}" r="5" fill="{SERIES["new"]}" '
                     f'stroke="var(--surface-1)" stroke-width="2"><title>'
                     f'{_esc(item)} 신 {nw:.3f}</title></circle>')
        parts.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">'
                     f'{_esc(I.speaker_of(item))} {_esc(item.split("_")[1])}</text>')

    for v in (lo + span * 0.08, hi - span * 0.08):
        parts.append(f'<text x="{X(v):.1f}" y="{height - 10}" text-anchor="middle" '
                     f'class="tick">{v:.2f}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------- scatter

def amplitude_vs_timing(summary, width=560, height=420):
    """Within-note motion against correlation with the recording.

    The one figure that separates "moves like pansori" from "moves WITH the
    singer". Moving right is amplitude; moving up is content. A system can do the
    first without the second, and this shows whether it did.
    """
    import json
    rows = [json.loads(l) for l in
            (C.OUT / 'results.jsonl').read_text(encoding='utf-8').splitlines() if l]
    keys = {(summary['old_run'], '5_pred_own'): 'old', ('NEW', '3_pitch'): 'new'}
    pts = {'old': {}, 'new': {}}
    for r in rows:
        k = keys.get((r['system'], r['rung']))
        if k and r['metric'] in ('within_std_ratio', 'within_corr'):
            pts[k].setdefault(r['item'], {}).setdefault(r['metric'], []).append(r['value'])

    L, R, T, B = 58, 18, 30, 46
    pw, ph = width - L - R, height - T - B
    xlo, xhi, ylo, yhi = 0.0, 1.6, -0.35, 1.0

    def X(v):
        return L + pw * (min(max(v, xlo), xhi) - xlo) / (xhi - xlo)

    def Y(v):
        return T + ph * (1 - (min(max(v, ylo), yhi) - ylo) / (yhi - ylo))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="진폭 대 타이밍" style="max-width:{width}px;height:auto">']
    for gx in (0.0, 0.5, 1.0, 1.5):
        parts.append(f'<line x1="{X(gx):.1f}" y1="{T}" x2="{X(gx):.1f}" y2="{T + ph}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{X(gx):.1f}" y="{T + ph + 18}" text-anchor="middle" '
                     f'class="tick">{gx:g}</text>')
    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{L}" y1="{Y(gy):.1f}" x2="{L + pw}" y2="{Y(gy):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{L - 8}" y="{Y(gy) + 4:.1f}" text-anchor="end" '
                     f'class="tick">{gy:g}</text>')
    parts.append(f'<line x1="{X(1.0):.1f}" y1="{T}" x2="{X(1.0):.1f}" y2="{T + ph}" '
                 f'stroke="var(--grid-strong)" stroke-width="2" stroke-dasharray="4 4"/>')

    for k in ('old', 'new'):
        for item, d in pts[k].items():
            if 'within_std_ratio' not in d or 'within_corr' not in d:
                continue
            x = float(np.median(d['within_std_ratio']))
            y = float(np.median(d['within_corr']))
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5.5" '
                         f'fill="{SERIES[k]}" stroke="var(--surface-1)" stroke-width="2" '
                         f'opacity="0.85"><title>{_esc(item)} {LABEL[k]} '
                         f'진폭 {x:.2f} / 상관 {y:.2f}</title></circle>')
    for k, ly in (('old', T + 14), ('new', T + 32)):
        parts.append(f'<circle cx="{L + pw - 96}" cy="{ly - 4}" r="5.5" fill="{SERIES[k]}"/>')
        parts.append(f'<text x="{L + pw - 84}" y="{ly}" class="dlabel" '
                     f'fill="{SERIES[k]}">{LABEL[k]}</text>')

    parts.append(f'<text x="{L + pw / 2}" y="{height - 8}" text-anchor="middle" '
                 f'class="axis">노트내 변동비 (진폭 — 1.0 이 원본)</text>')
    parts.append(f'<text x="14" y="{T + ph / 2}" text-anchor="middle" class="axis" '
                 f'transform="rotate(-90 14 {T + ph / 2})">원본과의 상관 (타이밍)</text>')
    parts.append('</svg>')
    return '\n'.join(parts)
