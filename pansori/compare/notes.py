#!/usr/bin/env python3
"""The note staircase both systems were conditioned on.

The old repo built this from `mel2ph` and per-phone `(pitch_midi, midi_dur)`
(pansori/note_seg.py, which does not exist in this repo). It does not need to be
ported: in data/pansori/*/transcriptions.csv the `note_dur` column is an absolute
timeline -- sum(note_dur) == sum(ph_dur) == the wav duration to the millisecond --
so note boundaries are just cumsum(note_dur), and note identity is the position
in note_seq. Verified against the old repo's notes.jsonl in smoke.py (gate G2).

This matters more than it looks. Both models were conditioned on the same ROSVOT
notes, so a single staircase serves both sides, which removes what would
otherwise have been the largest confound in the 시김새 decomposition.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import items as I  # noqa: E402

_STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_to_midi(token):
    """'F#4' -> 66, 'B3' -> 59, 'rest' -> 0. A4 = MIDI 69 = 440 Hz.

    0 is the rest marker, matching the binarizer, so it must never be treated as
    a pitch -- librosa would map it to 8.18 Hz.
    """
    token = token.strip()
    if token in ('rest', 'Rest', 'REST', ''):
        return 0
    name = token[0].upper()
    i = 1
    acc = 0
    while i < len(token) and token[i] in '#b':
        acc += 1 if token[i] == '#' else -1
        i += 1
    return (int(token[i:]) + 1) * 12 + _STEP[name] + acc


def note_grid(item):
    """(onsets[N+1] seconds, midi[N]) for an item, from its transcriptions.csv row."""
    row = I.row_of(item)
    midi = np.array([note_to_midi(t) for t in row['note_seq'].split()], dtype=np.int64)
    dur = np.array([float(x) for x in row['note_dur'].split()], dtype=np.float64)
    if len(midi) != len(dur):
        raise ValueError(f'{item}: {len(midi)} notes but {len(dur)} durations')
    return np.concatenate([[0.0], np.cumsum(dur)]), midi


def frame_notes(item, n_frames, timestep=C.TIMESTEP):
    """(note_id[K], staircase_hz[K]) on the analysis grid.

    note_id is -1 outside the labelled span; staircase is 0 on rests, which the
    residual code reads as 'no note here' rather than as a pitch.
    """
    onsets, midi = note_grid(item)
    t = (np.arange(n_frames) + 0.5) * timestep
    idx = np.searchsorted(onsets, t, side='right') - 1
    inside = (idx >= 0) & (idx < len(midi))
    note_id = np.where(inside, idx, -1)
    m = np.where(inside, midi[np.clip(idx, 0, len(midi) - 1)], 0)
    stair = np.where(m > 0, 440.0 * 2.0 ** ((m.astype(np.float64) - 69.0) / 12.0), 0.0)
    return note_id.astype(np.int64), stair


def note_core_mask(note_id, voiced, trim=C.NOTE_TRIM, min_voiced=C.NOTE_MIN_VOICED):
    """Drop note edges, where ROSVOT boundary error masquerades as ornament.

    Same rule as the old midi_baseline.note_core_mask -- skip notes with fewer
    than `min_voiced` voiced frames, then trim max(1, round(trim*L)) frames from
    each end -- because the two tables are only comparable if the mask is.
    """
    mask = np.zeros(len(note_id), dtype=bool)
    for n in np.unique(note_id[note_id >= 0]):
        idx = np.flatnonzero((note_id == n) & voiced)
        if len(idx) < min_voiced:
            continue
        cut = max(1, int(round(trim * len(idx))))
        if len(idx) - 2 * cut < 1:
            continue
        mask[idx[cut:len(idx) - cut]] = True
    return mask


def boundaries(item, n_frames, timestep=C.TIMESTEP):
    """Interior note boundaries as (frame, midi_before, midi_after, step_cents).

    Only boundaries between two sounding notes -- a rest edge is a phrase
    boundary, not a 꺾기, and treating it as one would inflate every transition
    statistic with silence.
    """
    onsets, midi = note_grid(item)
    out = []
    for n in range(1, len(midi)):
        if midi[n - 1] <= 0 or midi[n] <= 0:
            continue
        k = int(round(onsets[n] / timestep))
        if 0 < k < n_frames:
            out.append((k, int(midi[n - 1]), int(midi[n]),
                        100.0 * (midi[n] - midi[n - 1])))
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description='노트 격자 확인')
    p.add_argument('item', nargs='?', default='8b4d0824_0000')
    args = p.parse_args()

    onsets, midi = note_grid(args.item)
    row = I.row_of(args.item)
    ph_total = sum(float(x) for x in row['ph_dur'].split())
    print(f'{args.item}  화자 {I.speaker_of(args.item)}')
    print(f'노트 {len(midi)}개, 총 {onsets[-1]:.4f}s / ph_dur 합 {ph_total:.4f}s '
          f'(차이 {abs(onsets[-1] - ph_total) * 1000:.2f} ms)')
    print(f'note_seq: {row["note_seq"]}')
    print('온셋:', ' '.join(f'{o:.3f}' for o in onsets))
    print('MIDI :', ' '.join(str(m) for m in midi))
    n = int(round(onsets[-1] / C.TIMESTEP))
    nid, stair = frame_notes(args.item, n)
    print(f'프레임 {n}개, 유음 프레임 {int((stair > 0).sum())}개, '
          f'내부 경계 {len(boundaries(args.item, n))}개')


if __name__ == '__main__':
    main()
