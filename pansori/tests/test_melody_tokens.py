"""Tokenizer: notes and seconds <-> DEG_/DUR_/CONT/NEXT, and back to a score.

Run: python -m unittest pansori.tests.test_melody_tokens
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "melody"))

from tokens import (  # noqa: E402
    Tokenizer, anchor_octave, build_bins, note_name, score_text, units,
)

# 02b8fd02_0000, "아이고 장군님", gyemyeonjo, tonic B3 -- the worked example in
# pansori/docs/MELODY.md. 장 carries two notes, which is the melisma case.
ROW = {
    "item_name": "02b8fd02_0000", "uuid": "02b8fd02", "mode": "계면조",
    "duration": 2.688, "text": "아이고 장군님",
    "syllables": [
        {"syl": "아", "start": 0.0, "end": 0.2147, "notes": [[69, 0.0, 0.2145]]},
        {"syl": "이", "start": 0.2147, "end": 0.2931, "notes": [[66, 0.2145, 0.5281]]},
        {"syl": "고", "start": 0.2931, "end": 0.7517, "notes": [[59, 0.5281, 0.8372]]},
        {"syl": "장", "start": 0.7517, "end": 1.3961,
         "notes": [[59, 0.8372, 1.1304], [61, 1.1304, 1.2804]]},
        {"syl": "군", "start": 1.3961, "end": 2.2901, "notes": [[61, 1.2804, 1.419]]},
        {"syl": "님", "start": 2.2901, "end": 2.688, "notes": [[60, 1.419, 2.688]]},
    ],
    "rests": [],
    "qc": {"pass": True},
}

TONIC_B3 = 59


def tokenizer(step=0.05, count=40):
    """Even bins, so a test can predict which bin a duration lands in."""
    edges = [round(step * (k + 1), 4) for k in range(count - 1)]
    t = Tokenizer(edges)
    # Representatives at the bin midpoints; the top bin stands in for anything above.
    t._representative = [round(step * (k + 0.5), 4) for k in range(count)]
    return t


class Vocabulary(unittest.TestCase):
    def test_every_token_has_one_id(self):
        t = tokenizer()
        self.assertEqual(len(t.vocab), len(set(t.vocab)))
        self.assertEqual(len(t.ids), len(t.vocab))

    def test_note_names_round_trip_through_midi(self):
        for midi, name in ((59, "B3"), (60, "C4"), (69, "A4"), (66, "F#4")):
            self.assertEqual(note_name(midi), name)

    def test_degrees_outside_the_range_are_clamped_not_dropped(self):
        t = tokenizer()
        self.assertEqual(t.degree_token(TONIC_B3 - 40, TONIC_B3), f"DEG_{t.deg_min:+d}")
        self.assertEqual(t.degree_token(TONIC_B3 + 40, TONIC_B3), f"DEG_{t.deg_max:+d}")

    def test_a_rest_has_no_degree(self):
        self.assertEqual(tokenizer().degree_token(None, TONIC_B3), "REST")


class Encoding(unittest.TestCase):
    def test_one_syllable_yields_three_tokens_per_note(self):
        syllables, tokens = tokenizer().encode(ROW, TONIC_B3)
        self.assertEqual(syllables, ["아", "이", "고", "장", "군", "님"])
        self.assertEqual(len(tokens), 3 * 7)          # 6 syllables, 7 notes

    def test_a_held_syllable_is_marked_CONT_and_only_the_last_note_NEXT(self):
        _, tokens = tokenizer().encode(ROW, TONIC_B3)
        boundaries = tokens[2::3]
        self.assertEqual(boundaries, ["NEXT", "NEXT", "NEXT", "CONT", "NEXT", "NEXT", "NEXT"])
        self.assertEqual(boundaries.count("NEXT"), 6)  # exactly one per syllable

    def test_degrees_are_measured_from_the_tonic(self):
        _, tokens = tokenizer().encode(ROW, TONIC_B3)
        self.assertEqual(tokens[0::3],
                         ["DEG_+10", "DEG_+7", "DEG_+0", "DEG_+0", "DEG_+2", "DEG_+2", "DEG_+1"])

    def test_the_same_music_under_a_different_tonic_shifts_every_degree(self):
        _, a = tokenizer().encode(ROW, TONIC_B3)
        _, b = tokenizer().encode(ROW, TONIC_B3 + 2)
        for x, y in zip(a[0::3], b[0::3]):
            self.assertEqual(int(x.split("_")[1]) - int(y.split("_")[1]), 2)


class Rests(unittest.TestCase):
    def row_with_rest(self):
        row = dict(ROW, rests=[[1.419, 1.9]],
                   syllables=ROW["syllables"][:1] + [
                       dict(ROW["syllables"][5], start=1.9, end=2.688)])
        return row

    def test_a_rest_becomes_its_own_syllable_in_time_order(self):
        syllables, tokens = tokenizer().encode(self.row_with_rest(), TONIC_B3)
        self.assertEqual(syllables, ["아", "SP", "님"])
        self.assertEqual(tokens[3], "REST")

    def test_a_rest_decodes_to_the_score_underscore(self):
        t = tokenizer()
        syllables, tokens = t.encode(self.row_with_rest(), TONIC_B3)
        lines = t.decode(syllables, tokens, TONIC_B3)
        self.assertEqual([label for label, _, _ in lines], ["아", "_", "님"])
        self.assertEqual(lines[1][1], "rest")


class RoundTrip(unittest.TestCase):
    def test_pitches_come_back_exactly(self):
        t = tokenizer()
        syllables, tokens = t.encode(ROW, TONIC_B3)
        names = [name for _, name, _ in t.decode(syllables, tokens, TONIC_B3)]
        self.assertEqual(names, ["A4", "F#4", "B3", "B3", "C#4", "C#4", "C4"])

    def test_syllables_come_back_with_a_hyphen_for_the_held_note(self):
        t = tokenizer()
        syllables, tokens = t.encode(ROW, TONIC_B3)
        labels = [label for label, _, _ in t.decode(syllables, tokens, TONIC_B3)]
        self.assertEqual(labels, ["아", "이", "고", "장", "-", "군", "님"])

    def test_durations_come_back_within_one_bin(self):
        t = tokenizer(step=0.05)
        syllables, tokens = t.encode(ROW, TONIC_B3)
        decoded = [seconds for _, _, seconds in t.decode(syllables, tokens, TONIC_B3)]
        original = [seconds for _, notes in units(ROW) for _, seconds in notes]
        for got, want in zip(decoded, original):
            self.assertLessEqual(abs(got - want), 0.05)

    def test_the_decoded_lines_are_a_score_make_ds_can_parse(self):
        t = tokenizer()
        syllables, tokens = t.encode(ROW, TONIC_B3)
        text = score_text(t.decode(syllables, tokens, TONIC_B3))
        for line in text.splitlines():
            self.assertEqual(len(line.split()), 3)      # make_ds.parse_score's check
        self.assertIn("-\tC#4", text)


class Bins(unittest.TestCase):
    def test_quantile_bins_spread_the_bulk_evenly(self):
        durations = [0.1 + 0.001 * k for k in range(1000)]
        edges = build_bins(durations, n_bins=8, n_tail=0)
        self.assertEqual(len(edges), 7)
        self.assertEqual(edges, sorted(edges))

    def test_the_tail_gets_its_own_edges_above_the_last_quantile(self):
        # Without tail bins everything over the top quantile shares one bin. The
        # tail edges must land above it and reach towards the longest note.
        durations = [0.2] * 990 + [1.0 + 0.05 * k for k in range(10)]
        plain = build_bins(durations, n_bins=8, n_tail=0)
        with_tail = build_bins(durations, n_bins=8, n_tail=4)
        self.assertGreater(len(with_tail), len(plain))
        self.assertGreater(max(with_tail), max(plain))

    def test_the_open_top_bin_covers_far_less_of_the_tail(self):
        # The top bin stays open whatever happens -- the point is to shrink it. Here
        # it goes from covering 1.25 s of range down to 0.57 s.
        durations = [0.2] * 990 + [1.0 + 0.05 * k for k in range(10)]
        plain = build_bins(durations, n_bins=8, n_tail=0)
        with_tail = build_bins(durations, n_bins=8, n_tail=4)
        self.assertLess(max(durations) - max(with_tail), max(durations) - max(plain))

    def test_enough_tail_bins_separate_two_long_notes(self):
        durations = [0.2] * 990 + [1.0 + 0.05 * k for k in range(10)]
        t = Tokenizer(build_bins(durations, n_bins=8, n_tail=8))
        self.assertNotEqual(t.bin_of(1.0), t.bin_of(1.45))


class Anchor(unittest.TestCase):
    def test_the_tonic_lands_in_the_octave_nearest_the_register(self):
        self.assertEqual(anchor_octave(11, 60.0), 59)      # B, near middle C -> B3
        self.assertEqual(anchor_octave(11, 70.0), 71)      # the same B, an octave up

    def test_the_anchor_is_always_the_named_pitch_class(self):
        for reference in (40.0, 55.0, 62.0, 78.0):
            self.assertEqual(anchor_octave(2, reference) % 12, 2)


if __name__ == "__main__":
    unittest.main()
