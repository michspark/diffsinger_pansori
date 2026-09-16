#!/usr/bin/env python3
"""Turn bound syllables and notes into tokens, and back into a score.

(Named tokens.py, not tokenize.py: the stdlib has a tokenize, inspect imports it,
and shadowing it breaks numpy at import time.)

`melody_data.jsonl` holds what was measured -- MIDI numbers and seconds. A model
needs neither. This converts, per note, three things at once:

    pitch       MIDI 59, tonic D4  ->  DEG_+2     degrees from the piece's tonic
    duration    0.29 s             ->  DUR_20     40 bins, quantile + tail
    boundary                       ->  CONT/NEXT  does the syllable continue

    장 -> [(59, 0.29s), (61, 0.15s)]
       -> DEG_+2 DUR_20 CONT  DEG_+4 DUR_4 NEXT

`CONT` is the text score's `-`: one note at a time, then a decision to carry on or
move to the next syllable, so melisma has no length limit. 33.4% of this corpus's
syllables carry more than one note, and 6.3% carry six or more, so a fixed
notes-per-syllable vocabulary would truncate the very thing the model exists for.

Durations are binned rather than regressed because a syllable's length is not a
function of its lyrics -- the singer decides -- and L1 against an unpredictable
target is minimised by its mean. The 40 bins requantise at a median 4.3 ms, under
one frame at hop 512 / 44.1 kHz.

`decode` is the inverse and writes the text score `make_ds.py from-score` reads, so
the round trip is testable end to end and a generated token sequence needs no
further machinery to become audio.

Nothing here is written into melody_data.jsonl. The bin edges and the tonic will
move while the binding above them will not, so the conversion is a lookup applied
when the data is loaded; only its settings are stored.

Input : data/{set}/melody_data.jsonl + tonic.json
Output: data/{set}/tokenizer.json

  python pansori/melody/tokens.py --set full          # build and report
  python pansori/melody/tokens.py --set full --show 02b8fd02_0000
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))

from pansori_lib import read_jsonl, set_dir  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tonic import PITCH_CLASSES, resolve  # noqa: E402

# Degrees outside this are clamped. Measured over the 161,796 bound notes the range
# is -31..+23 and this holds 99.30% of them; the tail is ROSVOT octave slips and the
# very bottom of male tongseong, neither worth 20 tokens of vocabulary.
DEG_MIN, DEG_MAX = -12, 24

N_DUR_BINS = 32
N_TAIL_BINS = 8

SPECIAL = ["PAD", "BOS", "EOS"]
REST = "REST"
CONT, NEXT = "CONT", "NEXT"

# The syllable a rest occupies in the encoder's input. Same symbol the phoneme side
# uses, and `make_ds.from-score` turns the `_` it becomes back into one.
REST_SYLLABLE = "SP"


def note_name(midi: int) -> str:
    """MIDI number -> the name make_ds.from-score expects, e.g. 59 -> B3."""
    return f"{PITCH_CLASSES[midi % 12]}{midi // 12 - 1}"


def build_bins(durations: Sequence[float], n_bins: int = N_DUR_BINS,
               n_tail: int = N_TAIL_BINS) -> List[float]:
    """Quantile edges over the bulk, log-spaced edges over the tail.

    Quantiles for the bulk, because the corpus's durations sit almost entirely
    between 0.11 s and 0.80 s and a log grid over the whole range spends half its
    bins on an empty low end.

    But quantiles alone leave the top bin open, and it swallows everything from the
    last quantile to 4.92 s: 3.1% of notes, requantised at a median error of 209 ms.
    Raising the bin count makes that worse, not better -- the bin holds fewer notes
    but still spans the whole tail (64 bins: 270 ms, 96 bins: 293 ms). Long held
    tones are what pansori is made of, so the tail gets its own log-spaced edges.
    Measured, 32 + 8: p99 error 270 ms -> 99 ms, worst 3.8 s -> 0.6 s, median
    unchanged at 4.3 ms.
    """
    d = np.asarray([x for x in durations if x > 0], dtype=np.float64)
    q_hi = 100.0 * (n_bins - 1) / n_bins
    head = np.percentile(d, np.linspace(0, q_hi, n_bins + 1)[1:-1])
    if n_tail <= 0:
        return [float(x) for x in np.unique(head)]
    tail = np.exp(np.linspace(np.log(np.percentile(d, q_hi)),
                              np.log(d.max()), n_tail + 1))[:-1]
    return [float(x) for x in np.unique(np.concatenate([head, tail]))]


def anchor_octave(tonic_pc: int, reference_midi: float) -> int:
    """The tonic pitch class placed in the octave nearest the piece's own register.

    Degrees need an origin with an octave, and it has to be a property of the piece
    rather than of the utterance -- anchoring per utterance would give the same
    phrase different degrees depending on where it happened to start.
    """
    candidates = [tonic_pc + 12 * k for k in range(11)]
    return min(candidates, key=lambda c: abs(c - reference_midi))


def piece_registers(rows: Sequence[dict]) -> Dict[str, float]:
    """piece -> duration-weighted median note pitch, the register to anchor against."""
    weighted: Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    for row in rows:
        for syllable in row["syllables"]:
            for pitch, start, end in syllable["notes"]:
                weighted[row["uuid"]].append((int(pitch), end - start))
    registers = {}
    for piece, pairs in weighted.items():
        pitches = np.array([p for p, _ in pairs])
        weights = np.array([w for _, w in pairs])
        order = np.argsort(pitches)
        pitches, weights = pitches[order], weights[order]
        half = np.searchsorted(np.cumsum(weights), weights.sum() / 2)
        registers[piece] = float(pitches[min(half, len(pitches) - 1)])
    return registers


def units(row: dict) -> List[Tuple[str, List[Tuple[int, float]]]]:
    """One utterance -> [(syllable, [(midi, seconds), ...]), ...], rests interleaved.

    A rest belongs to no syllable, so it becomes one of its own. 74.3% of utterances
    have at least one, and a score with none has nowhere to breathe.
    """
    out: List[Tuple[float, str, List[Tuple[int, float]]]] = []
    for syllable in row["syllables"]:
        notes = [(int(p), e - s) for p, s, e in syllable["notes"]]
        out.append((syllable["start"], syllable["syl"], notes))
    for start, end in row["rests"]:
        out.append((start, REST_SYLLABLE, [(None, end - start)]))
    out.sort(key=lambda u: u[0])
    return [(syllable, notes) for _, syllable, notes in out]


class Tokenizer:
    def __init__(self, bins: Sequence[float], deg_min: int = DEG_MIN, deg_max: int = DEG_MAX):
        self.bins = list(bins)
        self.deg_min, self.deg_max = deg_min, deg_max
        self.vocab = (SPECIAL
                      + [f"DEG_{d:+d}" for d in range(deg_min, deg_max + 1)]
                      + [REST]
                      + [f"DUR_{k}" for k in range(len(self.bins) + 1)]
                      + [CONT, NEXT])
        self.ids = {token: index for index, token in enumerate(self.vocab)}
        # A bin's representative is the median of the durations that fall in it, which
        # is what decode emits. Quantile edges make the midpoint a poor stand-in for
        # the widest bins, which are the long held notes.
        self._representative = None

    # -- durations ------------------------------------------------------------
    def bin_of(self, seconds: float) -> int:
        return int(np.digitize(seconds, self.bins))

    def fit_representatives(self, durations: Sequence[float]) -> None:
        d = np.asarray([x for x in durations if x > 0], dtype=np.float64)
        index = np.digitize(d, self.bins)
        self._representative = [
            float(np.median(d[index == k])) if np.any(index == k) else 0.0
            for k in range(len(self.bins) + 1)
        ]

    def seconds_of(self, bin_index: int) -> float:
        if self._representative is None:
            raise RuntimeError("fit_representatives() must run before decoding")
        return self._representative[bin_index]

    # -- notes ----------------------------------------------------------------
    def degree_token(self, midi, tonic_midi: int) -> str:
        if midi is None:
            return REST
        degree = int(np.clip(midi - tonic_midi, self.deg_min, self.deg_max))
        return f"DEG_{degree:+d}"

    def encode(self, row: dict, tonic_midi: int) -> Tuple[List[str], List[str]]:
        """-> (syllables the encoder reads, the target token sequence)."""
        syllables, tokens = [], []
        for syllable, notes in units(row):
            syllables.append(syllable)
            for index, (midi, seconds) in enumerate(notes):
                tokens.append(self.degree_token(midi, tonic_midi))
                tokens.append(f"DUR_{self.bin_of(seconds)}")
                tokens.append(CONT if index < len(notes) - 1 else NEXT)
        return syllables, tokens

    def decode(self, syllables: Sequence[str], tokens: Sequence[str],
               tonic_midi: int) -> List[Tuple[str, str, float]]:
        """Inverse of encode -> [(syllable or '-' or '_', note name or 'rest', seconds)].

        Exactly the three columns of a text score, in the order make_ds reads them.
        """
        lines: List[Tuple[str, str, float]] = []
        at, first = 0, True
        for index in range(0, len(tokens) - 2, 3):
            pitch, duration, boundary = tokens[index:index + 3]
            if at >= len(syllables):
                break
            rest = pitch == REST
            if first:
                label = REST_SYLLABLE if rest else syllables[at]
            else:
                label = "-"
            lines.append(("_" if rest and first else label,
                          "rest" if rest else note_name(tonic_midi + int(pitch.split("_")[1])),
                          self.seconds_of(int(duration.split("_")[1]))))
            first = boundary == NEXT
            if first:
                at += 1
        return lines

    def to_dict(self) -> dict:
        return {"bins": self.bins, "representative": self._representative,
                "deg_min": self.deg_min, "deg_max": self.deg_max, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, blob: dict) -> "Tokenizer":
        tokenizer = cls(blob["bins"], blob["deg_min"], blob["deg_max"])
        tokenizer._representative = blob["representative"]
        return tokenizer


def score_text(lines: Sequence[Tuple[str, str, float]]) -> str:
    """The decoded lines as the file make_ds.py from-score reads."""
    return "\n".join(f"{label}\t{name}\t{seconds:.4f}" for label, name, seconds in lines)


def tonic_midis(rows: Sequence[dict], table: dict) -> Dict[str, Tuple[int, str]]:
    """piece -> (tonic as a MIDI number, which fallback level it came from)."""
    registers = piece_registers(rows)
    out = {}
    for row in rows:
        if row["uuid"] in out:
            continue
        pitch_class, level = resolve(table, row["uuid"], row["mode"])
        out[row["uuid"]] = (anchor_octave(pitch_class, registers[row["uuid"]]), level)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the melody tokenizer")
    parser.add_argument("--set", default="full")
    parser.add_argument("--bins", type=int, default=N_DUR_BINS)
    parser.add_argument("--tail-bins", type=int, default=N_TAIL_BINS)
    parser.add_argument("--show", help="print one utterance's tokens and decoded score")
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    table = json.loads((out_dir / "tonic.json").read_text(encoding="utf-8"))
    rows = [r for r in read_jsonl(out_dir / "melody_data.jsonl") if r["qc"]["pass"]]
    print(f"{len(rows):,} bindable utterances")

    durations = [seconds for row in rows for _, notes in units(row) for _, seconds in notes]
    tokenizer = Tokenizer(build_bins(durations, args.bins, args.tail_bins))
    tokenizer.fit_representatives(durations)
    anchors = tonic_midis(rows, table)

    clamped = total = 0
    lengths = []
    for row in rows:
        tonic_midi = anchors[row["uuid"]][0]
        for _, notes in units(row):
            for midi, _ in notes:
                if midi is None:
                    continue
                total += 1
                clamped += not (DEG_MIN <= midi - tonic_midi <= DEG_MAX)
        lengths.append(len(tokenizer.encode(row, tonic_midi)[1]))

    error = np.abs([tokenizer.seconds_of(tokenizer.bin_of(d)) - d for d in durations])
    print(f"vocabulary {len(tokenizer.vocab)}  "
          f"({DEG_MAX - DEG_MIN + 1} degrees + rest, {len(tokenizer.bins) + 1} durations, "
          f"{CONT}/{NEXT}, {len(SPECIAL)} special)")
    print(f"notes {total:,}  degrees clamped {clamped / total:.3%}")
    print(f"requantised duration error  median {np.median(error) * 1000:.1f} ms  "
          f"p90 {np.percentile(error, 90) * 1000:.1f} ms")
    print(f"token sequence length  median {np.median(lengths):.0f}  "
          f"p95 {np.percentile(lengths, 95):.0f}  max {max(lengths)}")
    levels = collections.Counter(level for _, level in anchors.values())
    print("tonic level, by piece: " + "  ".join(f"{k} {v}" for k, v in levels.most_common()))

    (out_dir / args.out).write_text(
        json.dumps({**tokenizer.to_dict(),
                    "tonic_midi": {p: m for p, (m, _) in anchors.items()}},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {out_dir / args.out}")

    if args.show:
        row = next(r for r in rows if r["item_name"] == args.show)
        tonic_midi = anchors[row["uuid"]][0]
        syllables, tokens = tokenizer.encode(row, tonic_midi)
        print(f"\n{row['item_name']}  {row['mode']}  "
              f"tonic {note_name(tonic_midi)}  \"{row['text']}\"")
        print("  encoder input: " + " ".join(syllables))
        print("  target       : " + " ".join(tokens))
        print("  decoded score:")
        for label, name, seconds in tokenizer.decode(syllables, tokens, tonic_midi):
            print(f"    {label}\t{name}\t{seconds:.4f}")


if __name__ == "__main__":
    main()
