"""The old-vs-new comparison can be pointed at a second frozen system (hyb2).

The old model's ladder directory is no longer on disk, so a re-run for another
system reuses the old side's already-measured rows. These tests pin the three
pieces that make that legitimate: which checkpoints each system name means,
where each system's renders go (so v1's artifacts are never overwritten), and
that reused rows are only accepted when the recomputed reference agrees.

Run (dsp env): python -m unittest pansori.tests.test_compare_systems
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pansori.compare import config as C  # noqa: E402


def _row(system, item, rung, metric, value, seed=0):
    return {'system': system, 'item': item, 'speaker': 'x', 'song': item.split('_')[0],
            'rung': rung, 'seed': seed, 'metric': metric, 'value': value}


class SystemsTest(unittest.TestCase):
    def test_v1_entry_is_the_frozen_constants(self):
        v1 = C.SYSTEMS['v1']
        self.assertEqual((v1['ac'], v1['dur'], v1['pitch']),
                         (C.CKPT_ACOUSTIC, C.CKPT_DUR, C.CKPT_PITCH))
        self.assertEqual(v1['labels'], 'data/pansori')
        self.assertEqual(v1['tag'], '')

    def test_hyb2_is_what_the_demo_serves(self):
        h = C.SYSTEMS['hyb2']
        self.assertEqual((h['ac_exp'], h['var_exp']), ('pansori_ac_hyb2', 'pansori_var_hyb2'))
        self.assertEqual((h['ac'], h['dur'], h['pitch']), (32000, 28000, 36000))
        self.assertEqual(h['labels'], 'data/pansori_hyb2')

    def test_every_system_checkpoint_exists(self):
        for name, s in C.SYSTEMS.items():
            for exp, step in ((s['ac_exp'], s['ac']), (s['var_exp'], s['dur']),
                              (s['var_exp'], s['pitch'])):
                ckpt = C.REPO / 'checkpoints' / exp / f'model_ckpt_steps_{step}.ckpt'
                best = C.REPO / 'checkpoints' / 'best' / f'{exp}_step{step}.ckpt'
                self.assertTrue(ckpt.exists() or best.exists(), f'{name}: {ckpt}')


class OutputDirsTest(unittest.TestCase):
    def test_v1_keeps_its_original_directories(self):
        from pansori.compare import render_new as R
        self.assertEqual(R.out_dirs('v1'), (C.OUT / 'new', C.OUT / 'ds'))

    def test_hyb2_never_writes_into_v1s_directories(self):
        from pansori.compare import render_new as R
        self.assertEqual(R.out_dirs('hyb2'), (C.OUT / 'new_hyb2', C.OUT / 'ds_hyb2'))


class RowsItemsTest(unittest.TestCase):
    def test_rows_spec_returns_the_items_the_old_model_was_measured_on(self):
        from pansori.compare import render_new as R
        rows = [_row(f'OLD_{C.OLD_PRIMARY}', 'b_0002', '5_pred_own', 'rpa50', 0.1),
                _row(f'OLD_{C.OLD_PRIMARY}', 'a_0001', '5_pred_own', 'rpa50', 0.2),
                _row('NEW', 'c_0003', '3_pitch', 'rpa50', 0.3)]
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
            f.write('\n'.join(json.dumps(r) for r in rows))
        self.assertEqual(R.resolve_items(f'rows:{f.name}'), ['a_0001', 'b_0002'])


class ReuseRowsTest(unittest.TestCase):
    STORED = [
        _row('GT', 'a_0001', '1_orig', 'within_std', 150.0),
        _row('CREPE_vs_RMVPE', 'a_0001', '1_orig', 'f0_rmse_cent', 162.0),
        _row('OLD_aux', 'a_0001', '5_pred_own', 'within_std_ratio', 0.29),
        _row('NEW', 'a_0001', '3_pitch', 'within_std_ratio', 0.92, seed=1),
    ]

    def test_reused_rows_are_the_old_and_crepe_rows_only(self):
        from pansori.compare import collect as K
        got = K.reusable_rows(self.STORED)
        self.assertEqual(sorted({r['system'] for r in got}), ['CREPE_vs_RMVPE', 'OLD_aux'])

    def test_matching_reference_is_accepted(self):
        from pansori.compare import collect as K
        K.assert_same_reference([self.STORED[0]], self.STORED)

    def test_moved_reference_is_refused(self):
        from pansori.compare import collect as K
        moved = dict(self.STORED[0], value=151.0)
        with self.assertRaises(AssertionError):
            K.assert_same_reference([moved], self.STORED)

    def test_missing_reference_row_is_refused(self):
        from pansori.compare import collect as K
        extra = _row('GT', 'a_0001', '1_orig', 'bias_rms', 70.0)
        with self.assertRaises(AssertionError):
            K.assert_same_reference([self.STORED[0], extra], self.STORED)


if __name__ == '__main__':
    unittest.main()


class NoteGridIdentityTest(unittest.TestCase):
    """smoke.py G5: a system can share data/pansori's staircase only if its labels
    carry the same notes at the same times."""

    A = {'note_seq': 'A4 rest C#4', 'note_dur': '0.2 0.1 0.3'}

    def test_identical_grids_pass(self):
        from pansori.compare import notes as N
        self.assertIsNone(N.grid_mismatch(self.A, dict(self.A)))

    def test_a_different_note_is_named(self):
        from pansori.compare import notes as N
        b = dict(self.A, note_seq='A4 rest D4')
        self.assertIn('MIDI', N.grid_mismatch(self.A, b))

    def test_an_onset_moved_by_two_ms_is_named(self):
        from pansori.compare import notes as N
        b = dict(self.A, note_dur='0.202 0.098 0.3')
        self.assertIn('ms', N.grid_mismatch(self.A, b))

    def test_a_different_note_count_is_named(self):
        from pansori.compare import notes as N
        b = {'note_seq': 'A4 C#4', 'note_dur': '0.3 0.3'}
        self.assertIn('notes', N.grid_mismatch(self.A, b))

    def test_other_corpus_rows_are_readable(self):
        from pansori.compare import items as I
        rows = I.corpus_rows('data/pansori_hyb2')
        self.assertIn('8b4d0824_0000', rows)
        self.assertIn('note_dur', rows['8b4d0824_0000'])
