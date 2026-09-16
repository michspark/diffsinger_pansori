"""estimate() and resolve(): the tonic that degrees are measured from.

Run: python -m unittest pansori.tests.test_melody_tonic
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "melody"))

from tonic import PITCH_CLASSES, build_table, estimate, resolve  # noqa: E402


def histogram(**seconds):
    """A pitch-class histogram written by note name: histogram(D=40, A=30)."""
    h = np.zeros(12)
    for name, value in seconds.items():
        h[PITCH_CLASSES.index(name.replace("s", "#"))] = value
    return h


class Estimate(unittest.TestCase):
    def test_the_pair_wins_over_the_single_loudest_pitch(self):
        # B is the loudest class but its fifth (F#) is silent, so it is a dwelt-on
        # note, not a tonic. D and A are quieter each yet hold 50 s as a fifth. This
        # is the whole reason the estimate is a pair: the argmax would say B.
        h = histogram(B=30, D=25, A=25)
        self.assertEqual(estimate(h, interval=7), PITCH_CLASSES.index("D"))
        self.assertEqual(estimate(h, interval=0), PITCH_CLASSES.index("B"))

    def test_a_fourth_names_the_other_end_of_the_same_pair(self):
        # {D, A} is a fifth up from D and a fourth up from A, so +5 picks A where
        # +7 picks D. Same pair, opposite end.
        h = histogram(D=40, A=35, G=10)
        self.assertEqual(estimate(h, interval=7), PITCH_CLASSES.index("D"))
        self.assertEqual(estimate(h, interval=5), PITCH_CLASSES.index("A"))

    def test_a_clean_tonic_and_fifth_are_found(self):
        h = histogram(C=50, G=40, E=10)
        self.assertEqual(estimate(h, interval=7), PITCH_CLASSES.index("C"))

    def test_the_estimate_wraps_around_the_octave(self):
        # A + E is a fifth that crosses the top of the pitch-class circle.
        h = histogram(A=45, E=40, C=5)
        self.assertEqual(estimate(h, interval=7), PITCH_CLASSES.index("A"))


class Resolve(unittest.TestCase):
    def table(self):
        return build_table({
            ("long", "우조"): histogram(C=60, G=50),
            ("long", "계면조"): histogram(D=70, A=60),
            ("short", "우조"): histogram(E=100, B=90),
            ("short", "계면조"): histogram(F=5, C=4),        # under min_seconds
        }, interval=7, min_seconds=60.0)

    def test_a_long_span_uses_its_own_mode(self):
        pc, level = resolve(self.table(), "long", "계면조")
        self.assertEqual((PITCH_CLASSES[pc], level), ("D", "span"))

    def test_a_short_span_falls_back_to_its_piece(self):
        # gyemyeonjo holds 9s here, so the piece's modes are pooled instead. E and
        # B dominate the pool, so the piece's tonic is E.
        pc, level = resolve(self.table(), "short", "계면조")
        self.assertEqual((PITCH_CLASSES[pc], level), ("E", "piece"))

    def test_an_unknown_piece_falls_back_to_the_corpus(self):
        _, level = resolve(self.table(), "never-seen", "우조")
        self.assertEqual(level, "corpus")

    def test_a_piece_too_short_even_pooled_falls_back_to_the_corpus(self):
        table = build_table({("tiny", "우조"): histogram(C=3, G=2)},
                            interval=7, min_seconds=60.0)
        self.assertIsNone(table["pieces"]["tiny"]["pooled"])
        _, level = resolve(table, "tiny", "우조")
        self.assertEqual(level, "corpus")

    def test_every_level_reports_the_share_its_pair_holds(self):
        entry = self.table()["pieces"]["long"]["modes"]["우조"]
        self.assertEqual(entry["pair_share"], 1.0)      # C and G are all of it
        self.assertEqual(entry["seconds"], 110.0)


if __name__ == "__main__":
    unittest.main()
