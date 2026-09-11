#!/usr/bin/env python3
"""Score every rendered rung against one reference, into one long-format file.

Output is tidy rows -- (system, item, rung, seed, metric, value) -- rather than a
nested dict, so stats.py never needs to know which metrics exist and adding a
metric never touches it.

The reference is RMVPE on the 44.1 kHz original, for both sides. The old
pipeline's CREPE contours are scored too, but only as the reference-swap
diagnostic: the old model was trained on CREPE targets, so reporting its pitch
only against RMVPE would penalise it for reproducing its own training target.

Frame alignment is asserted, not assumed. Rungs driven by ground-truth durations
must land within 3 frames of the original; 4_full predicts its own durations, so
it is exempt and is excluded from every frame-aligned metric instead of being
silently stretched to fit.

Usage (dsp env, from the repo root):
  PYTHONPATH=. python pansori/compare/collect.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402
from pansori.compare import f0 as F  # noqa: E402
from pansori.compare import items as I  # noqa: E402
from pansori.compare import pitch_metrics as P  # noqa: E402
from pansori.compare import sigimsae as S  # noqa: E402
from pansori.compare import spectral as SP  # noqa: E402

RESULTS = C.OUT / 'results.jsonl'
MAX_FRAME_GAP = 3


class Row(dict):
    pass


def rows_for(system, item, rung, seed, gen_path, ref_f0, ref_v, ref_path,
             frame_aligned=True, spectral=True):
    """Every metric for one rendered file, as long-format rows."""
    out = []
    f0_g, v_g = F.contour(gen_path)

    def emit(metric, value):
        if value is None:
            return
        v = float(value)
        if np.isfinite(v):
            out.append({'system': system, 'item': item, 'speaker': I.speaker_of(item),
                        'song': item.split('_', 1)[0], 'rung': rung, 'seed': seed,
                        'metric': metric, 'value': v})

    if frame_aligned:
        gap = abs(len(f0_g) - len(ref_f0))
        if gap > MAX_FRAME_GAP:
            raise AssertionError(
                f'{gen_path.name}: {len(f0_g)} frames vs reference {len(ref_f0)} '
                f'(gap {gap} > {MAX_FRAME_GAP}). A ground-truth-duration rung must '
                f'match the original; investigate before trusting any number.')
        for k, v in P.evaluate(f0_g, v_g, ref_f0, ref_v).items():
            emit(k, v)
        for k, v in S.evaluate(f0_g, v_g, ref_f0, ref_v, item).items():
            if not k.startswith('_'):
                emit(k, v)
        if spectral:
            for k, v in SP.evaluate(gen_path, ref_path).items():
                emit(k, v)
    else:
        # 4_full predicts its own durations, so nothing frame-aligned applies.
        # Distribution-level statistics still do.
        note_id, stair = __import__('pansori.compare.notes', fromlist=['x']).frame_notes(
            item, len(f0_g))
        st = S.note_stats(f0_g, v_g, item, len(f0_g))
        emit('within_std', st['within_std'])
        emit('within_std_core', st['within_std_core'])
        emit('bias_rms', st['bias_rms'])
        emit('octave_snap_rate', st['octave_snap_rate'])
        emit('motion_cps', S.motion(st['_f0_snapped'], st['_voiced']))
        emit('wobble_iqr', S.wobble(st['_f0_snapped'], st['_voiced']))
        emit('length_ratio', len(f0_g) / len(ref_f0))

    emit('above_11k_frac', SP.band_energy(gen_path)['above_cutoff_frac'])
    return out


def collect(items, seeds, old_runs=('aux', 'ds3'), only_new=False, spectral=True):
    ref_rows, out = [], []
    new_dir = C.OUT / 'new'

    for item in items:
        I.assert_held_out(item)
        ref_path = I.wav_of(item)
        ref_f0, ref_v = F.contour(ref_path)

        # the reference's own sigimsae level -- the row every ratio is divided by
        st = S.note_stats(ref_f0, ref_v, item)
        for k in ('within_std', 'within_std_core', 'bias_rms', 'residual_rms',
                  'octave_snap_rate', 'n_notes'):
            ref_rows.append({'system': 'GT', 'item': item, 'speaker': I.speaker_of(item),
                             'song': item.split('_', 1)[0], 'rung': '1_orig', 'seed': 0,
                             'metric': k, 'value': float(st[k])})
        for k, v in (('motion_cps', S.motion(st['_f0_snapped'], st['_voiced'])),
                     ('wobble_iqr', S.wobble(st['_f0_snapped'], st['_voiced'])),
                     ('above_11k_frac', SP.band_energy(ref_path)['above_cutoff_frac'])):
            ref_rows.append({'system': 'GT', 'item': item, 'speaker': I.speaker_of(item),
                             'song': item.split('_', 1)[0], 'rung': '1_orig', 'seed': 0,
                             'metric': k, 'value': float(v)})

        # reference-swap diagnostic: what changes if the old CREPE contour is
        # used as the reference instead
        crepe = None if only_new else F.crepe_contour(item, len(ref_f0))
        if crepe is not None:
            f0_c, v_c = crepe
            for k, v in P.evaluate(f0_c, v_c, ref_f0, ref_v).items():
                ref_rows.append({'system': 'CREPE_vs_RMVPE', 'item': item,
                                 'speaker': I.speaker_of(item),
                                 'song': item.split('_', 1)[0], 'rung': '1_orig',
                                 'seed': 0, 'metric': k, 'value': float(v)})
            sc = S.note_stats(f0_c, v_c, item, len(ref_f0))
            for k in ('within_std', 'within_std_core', 'bias_rms', 'octave_snap_rate'):
                ref_rows.append({'system': 'CREPE_vs_RMVPE', 'item': item,
                                 'speaker': I.speaker_of(item),
                                 'song': item.split('_', 1)[0], 'rung': '1_orig',
                                 'seed': 0, 'metric': k, 'value': float(sc[k])})

        # ---- old system ------------------------------------------------------
        for run in (() if only_new else old_runs):
            d = C.OLD_LADDER[run]
            for rung in ('2_gt_crepe', '5_pred_own'):
                p = d / f'{item}__{rung}.wav'
                if not p.exists():
                    continue
                out += rows_for(f'OLD_{run}', item, rung, 0, p, ref_f0, ref_v, ref_path)

        # ---- new system ------------------------------------------------------
        rungs = ('3_pitch',) if only_new else ('2_resynth', '3_pitch')
        if not only_new:
            p = new_dir / f'{item}__0_vocoder__s0.wav'
            if p.exists():
                out += rows_for('NEW', item, '0_vocoder', 0, p, ref_f0, ref_v, ref_path)
        for seed in seeds:
            for rung in rungs:
                p = new_dir / f'{item}__{rung}__s{seed}.wav'
                if p.exists():
                    out += rows_for('NEW', item, rung, seed, p, ref_f0, ref_v,
                                    ref_path, spectral=spectral)
            p = new_dir / f'{item}__4_full__s{seed}.wav'
            if p.exists() and not only_new:
                out += rows_for('NEW', item, '4_full', seed, p, ref_f0, ref_v, ref_path,
                                frame_aligned=False, spectral=False)

        print(f'  {item} ({I.speaker_of(item)}) done', flush=True)

    return ref_rows + out


def main():
    p = argparse.ArgumentParser(description='collect the metrics')
    p.add_argument('--items', default='fair')
    p.add_argument('--seeds', type=int, default=len(C.SEEDS))
    p.add_argument('--out', default=str(RESULTS))
    p.add_argument('--only-new', action='store_true',
                   help='the new model\'s 3_pitch only (for the hold-out extension)')
    p.add_argument('--no-spectral', action='store_true')
    args = p.parse_args()

    from pansori.compare.render_new import resolve_items
    items = resolve_items(args.items)
    seeds = list(C.SEEDS[:args.seeds])
    print(f'{len(items)} items x {len(seeds)} seeds')

    rows = collect(items, seeds, only_new=args.only_new,
                   spectral=not args.no_spectral)
    # sorted so a re-run is byte-identical -- determinism is a check, not a nicety
    rows.sort(key=lambda r: (r['system'], r['item'], r['rung'], r['seed'], r['metric']))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    print(f'\n{len(rows)} rows -> {args.out}')


if __name__ == '__main__':
    main()
