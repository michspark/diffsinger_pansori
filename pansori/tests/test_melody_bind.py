"""bind(): cut a note sequence into one contiguous run per syllable.

Run: python -m unittest pansori.tests.test_melody_bind
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "melody"))

from bind_notes import bind, rests_of  # noqa: E402


def groups(words, notes):
    """bind() as the notes themselves, which is what the assertions read on."""
    starts = bind(words, notes)
    if starts is None:
        return None
    bounds = starts[1:] + [len(notes)]
    return [[notes[k][0] for k in range(a, b)] for a, b in zip(starts, bounds)]


class Binding(unittest.TestCase):
    # 02b8fd02_0000, "아이고 장군님" -- the utterance that rules greedy out. 이 is
    # 78 ms and shares every note it touches with a longer neighbour, so handing
    # each note to whichever syllable overlaps it most leaves 이 and 님 empty.
    WORDS = [["아", 0.000, 0.215], ["이", 0.215, 0.293], ["고", 0.293, 0.752],
             ["장", 0.752, 1.396], ["군", 1.396, 2.290], ["님", 2.290, 2.688]]
    NOTES = [[69, 0.000, 0.214], [66, 0.214, 0.528], [59, 0.528, 0.837],
             [59, 0.837, 1.130], [61, 1.130, 1.280], [61, 1.280, 1.419],
             [60, 1.419, 2.688]]

    def test_every_syllable_receives_a_note(self):
        self.assertEqual(groups(self.WORDS, self.NOTES),
                         [[69], [66], [59], [59, 61], [61], [60]])

    def test_a_syllable_may_take_several_notes(self):
        # sigimsae: 장 rolls across two. The representation has to allow it --
        # 33.4% of syllables in the corpus carry more than one note.
        self.assertEqual(len(groups(self.WORDS, self.NOTES)[3]), 2)

    def test_runs_are_contiguous_and_ordered(self):
        starts = bind(self.WORDS, self.NOTES)
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0)
        self.assertEqual(len(set(starts)), len(starts))

    def test_the_overlap_is_what_is_maximised(self):
        # Two syllables, four notes, the split plainly at the second boundary.
        words = [["가", 0.0, 1.0], ["나", 1.0, 2.0]]
        notes = [[60, 0.0, 0.5], [61, 0.5, 1.0], [62, 1.0, 1.5], [63, 1.5, 2.0]]
        self.assertEqual(groups(words, notes), [[60, 61], [62, 63]])

    def test_fewer_notes_than_syllables_is_unbindable(self):
        words = [["가", 0.0, 1.0], ["나", 1.0, 2.0], ["다", 2.0, 3.0]]
        self.assertIsNone(bind(words, [[60, 0.0, 3.0], [61, 1.0, 2.0]]))

    def test_no_notes_at_all_is_unbindable(self):
        self.assertIsNone(bind([["가", 0.0, 1.0]], []))

    def test_as_many_notes_as_syllables_gives_one_each(self):
        words = [["가", 0.0, 1.0], ["나", 1.0, 2.0]]
        self.assertEqual(groups(words, [[60, 0.0, 1.0], [61, 1.0, 2.0]]), [[60], [61]])


class Rests(unittest.TestCase):
    def test_a_gap_between_notes_is_a_rest(self):
        self.assertEqual(rests_of([[60, 0.0, 1.0], [61, 1.5, 2.0]], 2.0),
                         [[1.0, 1.5]])

    def test_uncovered_tail_and_head_are_rests(self):
        self.assertEqual(rests_of([[60, 0.5, 1.0]], 2.0), [[0.0, 0.5], [1.0, 2.0]])

    def test_rounding_between_two_notes_is_not_a_rest(self):
        self.assertEqual(rests_of([[60, 0.0, 1.0], [61, 1.02, 2.0]], 2.0), [])

    def test_overlapping_notes_do_not_invent_a_rest(self):
        self.assertEqual(rests_of([[60, 0.0, 1.2], [61, 0.9, 2.0]], 2.0), [])


if __name__ == "__main__":
    unittest.main()
