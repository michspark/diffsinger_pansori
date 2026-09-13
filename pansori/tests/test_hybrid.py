"""hybridize(): borrow the changbon's syllables, keep Whisper's syllable count.

Run: python -m unittest pansori.tests.test_hybrid
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))

from hybrid import hybridize  # noqa: E402


def book(text):
    """A changbon excerpt as (syllables, space_before), the way Book stores it."""
    syllables, space_before = [], []
    for word_index, word in enumerate(text.split()):
        for char_index, char in enumerate(word):
            syllables.append(char)
            space_before.append(word_index > 0 and char_index == 0)
    return syllables, space_before


class EqualLengthRun(unittest.TestCase):
    def test_edge_run_after_a_strong_anchor_is_replaced(self):
        # The case that motivated the stage: 4 anchors, then a 4<->4 run, then
        # book syllables Whisper never heard. Those trailing ones must NOT be added.
        result = hybridize("오 시기 가비니지오!", *book("오시기가 웬일이요 이치제"))
        self.assertEqual(result.text, "오시기가 웬일이요")
        self.assertEqual(result.n_replaced, 4)

    def test_interior_runs_between_anchors_are_replaced(self):
        result = hybridize("가득 잡며 넘치느라", *book("가득차면 넘치니라"))
        self.assertEqual(result.text, "가득차면 넘치니라")
        self.assertEqual(result.n_replaced, 3)          # 잡→차 며→면 느→니; 라 already matched

    def test_spacing_follows_the_book_where_syllables_come_from_it(self):
        result = hybridize("오 시기 가비니지오!", *book("오시기가 웬일이요 이치제"))
        self.assertNotIn(" 시기 ", result.text)


class UnequalLengthRun(unittest.TestCase):
    def test_a_stretch_whisper_heard_but_the_book_lacks_stays_whisper(self):
        # 03b currently ADOPTS this one as "여안고" (9 syllables -> 3).
        result = hybridize("여기를 무글무글 안고", *book("여안고"))
        self.assertEqual(result.text, "여기를 무글무글 안고")
        self.assertEqual(result.n_replaced, 0)

    def test_a_run_one_syllable_off_is_refused_even_when_pinned(self):
        # Strict: 잡며 (2) vs 차면서 (3) is left alone; 느→니 next door still goes through.
        result = hybridize("가득 잡며 넘치느라", *book("가득 차면서 넘치니라"))
        self.assertEqual(result.text, "가득 잡며 넘치니라")
        self.assertEqual(result.n_replaced, 1)

    def test_book_syllables_whisper_never_heard_are_not_added(self):
        result = hybridize("웬일이요", *book("오시기가 웬일이요 이치제"))
        self.assertEqual(result.text, "웬일이요")
        self.assertEqual(result.span, (4, 8))


class Pinning(unittest.TestCase):
    def test_edge_run_behind_a_single_common_syllable_is_refused(self):
        result = hybridize("이 비니지오", *book("이 웬일이요"))
        self.assertEqual(result.n_replaced, 0)

    def test_run_longer_than_max_run_is_refused(self):
        result = hybridize("오시기가 나나나나나나", *book("오시기가 웬일이요이치"))
        self.assertEqual(result.n_replaced, 0)
        result = hybridize("오시기가 나나나나나나", *book("오시기가 웬일이요이치"), max_run=6)
        self.assertEqual(result.n_replaced, 6)

    def test_whole_segment_differing_is_never_rewritten(self):
        result = hybridize("비니지오", *book("웬일이요"))
        self.assertEqual(result.n_replaced, 0)


class Invariants(unittest.TestCase):
    CASES = [
        ("오 시기 가비니지오!", "오시기가 웬일이요 이치제"),
        ("여기를 무글무글 안고", "여안고"),
        ("가득 잡며 넘치느라", "가득 차면서 넘치니라"),
        ("1라이 꽝", "름이로구나 광"),
        ("분양사 매운 온말이", "분향사배 통곡을 한다"),
    ]

    def test_syllable_count_never_changes(self):
        for whisper, ref in self.CASES:
            with self.subTest(whisper=whisper):
                result = hybridize(whisper, *book(ref))
                hangul = [c for c in result.text if "\uac00" <= c <= "\ud7a3"]
                self.assertEqual(len(hangul), result.n_syllables)
                self.assertEqual(result.n_syllables,
                                 sum("\uac00" <= c <= "\ud7a3" for c in whisper))

    def test_empty_whisper_passes_through(self):
        result = hybridize("...", *book("오시기가"))
        self.assertEqual((result.text, result.n_syllables), ("...", 0))


if __name__ == "__main__":
    unittest.main()
