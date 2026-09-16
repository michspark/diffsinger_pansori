#!/usr/bin/env python3
"""Bind each syllable to the notes it was sung on -> melody_data.jsonl.

The corpus keeps lyrics and melody on two timelines that never met. The aligner
puts syllables on the audio clock; ROSVOT puts notes on the same clock without
ever seeing the lyrics. A model that generates a melody from lyrics needs them
joined: syllable -> the notes carrying it, several of them where the singer
rolls one syllable (sigimsae).

This joins them, and nothing else. Degrees, duration bins and tokens are left to
the tokenizer, because bin counts and the tonic estimate will move while the DP
above them will not -- baking tokens in here would mean re-running the DP over
the whole corpus to change a bin count.

Read from `phones_sofa2.jsonl`, NOT `transcriptions.csv`:

  * SOFA is handed one syllable per dictionary headword (align.yaml `lab_unit:
    syllable`), so its words tier IS the syllable sequence -- measured 100.0% of
    27,720 utterances, against 7.5% for MFA, which groups by eojeol.
  * 06_merge then folds sub-20 ms phones into a neighbour (`MIN_PH_DUR`), and
    SOFA emits 24.4% of its phones under 20 ms, so by the time it reaches
    transcriptions.csv only 28.0% of utterances still carry every syllable.
    The melody model has no use for phonemes, so it branches off above that.

Input : data/{set}/segments.jsonl + {phones} + notes.jsonl + lyrics.jsonl
Output: data/{set}/melody_data.jsonl

Runs in any env -- it reads jsonl and writes jsonl.

  python pansori/melody/bind_notes.py --set full
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))

from pansori_lib import read_jsonl, set_dir, write_jsonl  # noqa: E402

# A note must lie under a syllable for this long to count as reaching it. Below it
# the note is touching the syllable's edge, not carrying it.
MIN_NOTE_OVERLAP = 0.02

# Syllables shorter than this are the aligner giving up, not singing: SOFA crams
# the tail of an utterance it has lost into a few milliseconds. Measured over the
# whole corpus, 7.2% of SOFA2's syllables land here (MFA: 1.4%).
MIN_SYLLABLE_DUR = 0.03

# Uncovered time shorter than this is rounding between two notes, not a rest. The
# same threshold 06_merge uses to decide a note covers a silence.
MIN_REST = 0.05


def overlap_matrix(words: Sequence[Sequence], notes: Sequence[Sequence]) -> np.ndarray:
    """[n_syllables, n_notes] of seconds shared between each syllable and note."""
    w_start = np.array([w[1] for w in words], dtype=np.float64)[:, None]
    w_end = np.array([w[2] for w in words], dtype=np.float64)[:, None]
    n_start = np.array([n[1] for n in notes], dtype=np.float64)[None, :]
    n_end = np.array([n[2] for n in notes], dtype=np.float64)[None, :]
    return np.maximum(0.0, np.minimum(w_end, n_end) - np.maximum(w_start, n_start))


def bind(words: Sequence[Sequence], notes: Sequence[Sequence]) -> Optional[List[int]]:
    """Cut the note sequence into one contiguous run per syllable.

    Returns the index of each syllable's first note, or None when there are
    fewer notes than syllables.

    Maximises the total syllable/note overlap, which is the obvious objective,
    under two constraints that rule out just handing each note to whichever
    syllable it overlaps most: the assignment has to be monotonic (a later
    syllable cannot take an earlier note), and every syllable has to receive at
    least one note or the melody has a hole where a word was. Greedy satisfies
    neither -- on `02b8fd02_0000` ("아이고 장군님") it starves two of six
    syllables, because a 78 ms syllable loses every note it shares to its longer
    neighbours. Same reason `make_ds.derive_note_slur` is a DP.

    O(n_syllables x n_notes): the inner maximisation is a running one, since
    every candidate cut only has to beat the best cut to its left.
    """
    n_w, n_n = len(words), len(notes)
    if n_w == 0 or n_n < n_w:
        return None

    ov = overlap_matrix(words, notes)
    # cum[i, j] = overlap of syllable i with notes 0..j, with a zero column
    # prepended so a run starting at note 0 needs no special case.
    cum = np.concatenate([np.zeros((n_w, 1)), np.cumsum(ov, axis=1)], axis=1)

    NEG = -np.inf
    dp = np.full((n_w, n_n), NEG)           # syllable i's run ends at note j
    back = np.zeros((n_w, n_n), dtype=int)  # ... and started at note back[i, j]
    dp[0] = cum[0, 1:]

    for i in range(1, n_w):
        # A run for syllable i starting at note a is worth
        #   dp[i-1, a-1] - cum[i, a] + cum[i, j+1],
        # so only the first two terms depend on a. Carry their best forward.
        best, best_a = NEG, -1
        for j in range(i, n_n):
            a = j                      # the shortest run for i that reaches j
            head = dp[i - 1, a - 1] - cum[i, a]
            if head > best:
                best, best_a = head, a
            dp[i, j] = best + cum[i, j + 1]
            back[i, j] = best_a

    end = n_n - 1
    starts = [0] * n_w
    for i in range(n_w - 1, 0, -1):
        starts[i] = int(back[i, end])
        end = starts[i] - 1
    return starts


def rests_of(notes: Sequence[Sequence], duration: float) -> List[List[float]]:
    """Stretches of [0, duration] no note covers, longer than MIN_REST.

    74.3% of utterances have one. They belong to no syllable, so they have
    nowhere to live in the syllable list, and dropping them would leave a score
    with nowhere to breathe.
    """
    rests, cursor = [], 0.0
    for _, start, end in sorted(notes, key=lambda n: n[1]):
        if start - cursor > MIN_REST:
            rests.append([round(cursor, 4), round(start, 4)])
        cursor = max(cursor, end)
    if duration - cursor > MIN_REST:
        rests.append([round(cursor, 4), round(duration, 4)])
    return rests


def build_record(item: str, segment: dict, words: Sequence[Sequence],
                 notes: Sequence[Sequence], text: str) -> dict:
    """One utterance -> one melody_data.jsonl row. `pass` false rows are kept.

    A failed row still carries why it failed, so the thresholds can be moved
    without rebuilding the file.
    """
    duration = float(segment["duration"])
    starts = bind(words, notes)

    syllables: List[dict] = []
    if starts is not None:
        bounds = starts[1:] + [len(notes)]
        for (syllable, start, end), first, stop in zip(words, starts, bounds):
            syllables.append({
                "syl": syllable,
                "start": round(float(start), 4),
                "end": round(float(end), 4),
                # ROSVOT's own spans, uncut: the note is the musical unit, so one
                # reaching past its syllable stays whole.
                "notes": [[int(p), round(float(s), 4), round(float(e), 4)]
                          for p, s, e in notes[first:stop]],
            })

    if starts is None:
        qc = {"reason": "fewer notes than syllables" if notes else "no notes",
              "n_syllables": len(words), "n_notes": len(notes), "pass": False}
    else:
        min_dur = min(s["end"] - s["start"] for s in syllables)
        no_note = sum(
            1 for s in syllables
            if not any(min(s["end"], e) - max(s["start"], st) > MIN_NOTE_OVERLAP
                       for _, st, e in s["notes"])
        )
        qc = {
            "min_syl_dur": round(min_dur, 4),
            "n_syl_no_note": no_note,
            "n_syllables": len(words),
            "n_notes": len(notes),
            "pass": min_dur >= MIN_SYLLABLE_DUR and no_note == 0,
        }

    return {
        "item_name": item,
        "uuid": segment["uuid"],          # hold-out splits go by song, never by utterance
        "mode": segment["mode"],
        "duration": round(duration, 4),
        "text": text,
        "syllables": syllables,
        "rests": rests_of(notes, duration),
        "qc": qc,
    }


def report(rows: Sequence[dict]) -> None:
    kept = [r for r in rows if r["qc"]["pass"]]
    print(f"\n{len(rows):,} utterances -> {len(kept):,} pass "
          f"({len(kept) / max(len(rows), 1):.1%})")

    reasons = Counter()
    for row in rows:
        qc = row["qc"]
        if qc["pass"]:
            continue
        if "reason" in qc:
            reasons[qc["reason"]] += 1
        else:
            if qc["min_syl_dur"] < MIN_SYLLABLE_DUR:
                reasons[f"syllable under {MIN_SYLLABLE_DUR}s"] += 1
            if qc["n_syl_no_note"]:
                reasons["syllable with no note"] += 1
    for reason, count in reasons.most_common():
        print(f"  dropped: {reason:32s} {count:6,d}")

    print()
    by_mode = Counter(r["mode"] for r in kept)
    hours = Counter()
    for row in kept:
        hours[row["mode"]] += row["duration"]
    for mode, count in by_mode.most_common():
        print(f"  {mode:6s} {count:6,d} utterances  {hours[mode] / 3600:5.2f} h")

    notes = sum(len(s["notes"]) for r in kept for s in r["syllables"])
    syllables = sum(len(r["syllables"]) for r in kept)
    melisma = sum(1 for r in kept for s in r["syllables"] if len(s["notes"]) > 1)
    print(f"\n  {syllables:,} syllables · {notes:,} notes "
          f"({notes / max(syllables, 1):.2f} per syllable, "
          f"{melisma / max(syllables, 1):.1%} carrying more than one)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind syllables to notes for melody training")
    parser.add_argument("--set", default="full")
    parser.add_argument("--phones", default="phones_sofa2.jsonl",
                        help="alignment to read the syllable tier from. Must be a SOFA "
                             "run: MFA's words tier is eojeol, not syllables")
    parser.add_argument("--lyrics", default="lyrics.jsonl")
    parser.add_argument("--out", default="melody_data.jsonl")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = {r["item_name"]: r for r in read_jsonl(out_dir / "segments.jsonl")}
    notes = {r["item_name"]: r["notes"] for r in read_jsonl(out_dir / "notes.jsonl")}
    lyrics = {r["item_name"]: (r.get("text") or "") for r in read_jsonl(out_dir / args.lyrics)}

    rows, skipped = [], 0
    for record in read_jsonl(out_dir / args.phones):
        item = record["item_name"]
        if item not in notes or item not in segments:
            skipped += 1        # aligned but never note-sliced
            continue
        rows.append(build_record(item, segments[item], record["words"],
                                 notes[item], lyrics.get(item, "")))

    write_jsonl(out_dir / args.out, rows)
    if skipped:
        print(f"{skipped:,} aligned utterances had no note slice")
    report(rows)
    print(f"\n-> {out_dir / args.out}")


if __name__ == "__main__":
    main()
