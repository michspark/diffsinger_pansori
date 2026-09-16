#!/usr/bin/env python3
"""Estimate each piece's tonic, then ask whether ujo and gyemyeonjo differ at all.

A mode is defined against a reference pitch, not against absolute pitch names. Our
notes are absolute (`A#4`), and every singer and piece sits somewhere different, so
one mode's skeletal tones land on another's note names and the two modes look alike
(`pansori/docs/MELODY.md` §0). Putting every piece on a common tonic is what should
make them comparable.

The tonic is taken as the pitch class maximising `h[p] + h[(p + interval) % 12]`
over a duration-weighted pitch-class histogram -- the tonic and its structural
partner together, rather than the single loudest note, which picks whichever of the
pair happened to be held longer. `--interval` selects the partner: 7 is a perfect
fifth, 5 a perfect fourth, and the two are the same pair seen from its two ends, so
they produce the same grouping rotated by 7. 0 falls back to the plain argmax.

This is the gate for everything downstream in MELODY.md. It trains nothing; it is an
aggregation over notes we already have, and the question it answers is whether the
mode is in the note skeleton at all. So it reports the separation it finds, honestly,
including when there is none:

  * the per-mode degree profiles, aggregated over pieces
  * Jensen-Shannon divergence between ujo's and gyemyeonjo's profile
  * a logistic probe, split by piece, predicting the mode from the profile alone

Input : data/{set}/segments.jsonl + notes.jsonl
Output: data/{set}/tonic.json + docs/img/mode_profiles.png

Runs in any env with numpy, sklearn and matplotlib.

  python pansori/melody/tonic.py --set full
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

REPO = Path(__file__).resolve().parents[2]

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# A span shorter than this cannot support a 12-bin histogram: the corpus's spans run
# to a median 87 s, but the 10th percentile is 12 s, and at that length one held note
# decides the estimate.
MIN_SECONDS = 60.0

# The two modes with enough material to compare. pyeongjo has 1.5 h, seolleongje and
# gyeongdeureum together under 20 minutes.
COMPARED = ("우조", "계면조")

# Matplotlib ships no Hangul glyphs, and adding a font dependency to read two labels
# is not worth it. The figure is part of the English documentation anyway.
ROMAN = {"우조": "ujo", "계면조": "gyemyeonjo", "평조": "pyeongjo"}


def histograms(segments: Dict[str, dict], notes: Dict[str, list],
               min_seconds: float) -> Dict[Tuple[str, str], np.ndarray]:
    """(piece, mode) -> duration-weighted pitch-class histogram, normalised.

    Weighted by how long each pitch class sounds rather than by how often it is
    struck, because a tonic is held and a passing tone is not.

    Built from notes.jsonl rather than melody_data.jsonl: the tonic is a property of
    the piece, and the binding QC that drops 46.5% of utterances is about syllables,
    which have no bearing on which pitches the piece dwells on.
    """
    raw = collections.defaultdict(lambda: np.zeros(12))
    for item, note_list in notes.items():
        segment = segments.get(item)
        if segment is None:
            continue
        key = (segment["uuid"], segment["mode"])
        for pitch, start, end in note_list:
            raw[key][int(pitch) % 12] += end - start
    return {key: h / h.sum() for key, h in raw.items() if h.sum() >= min_seconds}


def estimate(histogram: np.ndarray, interval: int) -> int:
    """The pitch class of the tonic: the pair (p, p + interval) carrying the most time."""
    if interval == 0:
        return int(histogram.argmax())
    score = np.array([histogram[p] + histogram[(p + interval) % 12] for p in range(12)])
    return int(score.argmax())


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """In bits. 0 when the two profiles are identical, 1 when disjoint."""
    m = (p + q) / 2

    def kl(a, b):
        return float(np.sum(np.where(a > 0, a * np.log2(a / np.maximum(b, 1e-12)), 0.0)))

    return (kl(p, m) + kl(q, m)) / 2


def probe(features: np.ndarray, labels: Sequence[str], groups: Sequence[str]) -> float:
    """Balanced accuracy of a logistic probe, split by piece. 0.5 is chance.

    Split by piece and not by span: the two modes of one recording share a singer, a
    key and a room, and letting them straddle the split would score those instead of
    the mode.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    y = np.array([1 if m == COMPARED[0] else 0 for m in labels])
    predicted = np.zeros_like(y)
    for train, test in GroupKFold(n_splits=5).split(features, y, groups):
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(features[train], y[train])
        predicted[test] = model.predict(features[test])
    return float(balanced_accuracy_score(y, predicted))


def profile_table(profiles: Dict[str, np.ndarray]) -> str:
    lines = ["degree  " + " ".join(f"{d:5d}" for d in range(12))]
    for mode, profile in profiles.items():
        lines.append(f"{mode:6s}  " + " ".join(f"{x * 100:5.1f}" for x in profile))
    if all(m in profiles for m in COMPARED):
        delta = (profiles[COMPARED[0]] - profiles[COMPARED[1]]) * 100
        lines.append("diff    " + " ".join(f"{x:+5.1f}" for x in delta))
    return "\n".join(lines)


def plot(profiles: Dict[str, np.ndarray], counts: Dict[str, int], interval: int,
         divergence: float, accuracy: float, by_singer: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = np.arange(12)
    width = 0.38
    colours = {"우조": "#2b6cb0", "계면조": "#c05621"}

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.2),
                                gridspec_kw={"height_ratios": [3, 1.15]})

    top = axes[0]
    for offset, mode in zip((-width / 2, width / 2), COMPARED):
        top.bar(degrees + offset, profiles[mode] * 100, width,
                label=f"{ROMAN[mode]}  (n={counts[mode]} spans)", color=colours[mode])
    top.set_xticks(degrees)
    top.set_xticklabels([f"{d}" for d in degrees])
    top.set_xlabel("semitones above the estimated tonic")
    top.set_ylabel("share of sung time (%)")
    top.set_title(f"Mode profiles on a common tonic  "
                  f"(partner interval +{interval}, spans ≥ {MIN_SECONDS:.0f}s)")
    top.legend(frameon=False)
    top.grid(axis="y", alpha=0.25)
    # The tonic and its partner, the pair the estimate is built on.
    for degree in (0, interval % 12):
        top.axvspan(degree - 0.5, degree + 0.5, color="#000000", alpha=0.045, zorder=0)

    bottom = axes[1]
    delta = (profiles[COMPARED[0]] - profiles[COMPARED[1]]) * 100
    bottom.bar(degrees, delta, 0.6,
               color=[colours[COMPARED[0]] if d > 0 else colours[COMPARED[1]] for d in delta])
    bottom.axhline(0, color="#333333", linewidth=0.8)
    bottom.set_xticks(degrees)
    bottom.set_xlabel("semitones above the estimated tonic")
    bottom.set_ylabel(f"{ROMAN[COMPARED[0]]} − {ROMAN[COMPARED[1]]}\n(percentage points)")
    bottom.grid(axis="y", alpha=0.25)
    bottom.set_title(f"difference:  JSD {divergence:.4f} bits  ·  logistic probe "
                     f"{accuracy:.1%} by piece / {by_singer:.1%} by singer  (chance 50%)",
                     fontsize=10)

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=144)
    print(f"\n-> {out_path}")


def piece_speakers(segments: Dict[str, dict], spk_map: Path) -> Dict[str, str]:
    """piece -> the speaker most of its utterances are attributed to."""
    if not spk_map.exists():
        return {}
    item2spk = json.loads(spk_map.read_text(encoding="utf-8"))["item2spk"]
    tally: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item, speaker in item2spk.items():
        segment = segments.get(item)
        if segment is not None:
            tally[segment["uuid"]][speaker] += 1
    return {piece: counter.most_common(1)[0][0] for piece, counter in tally.items()}


def rotate(spans: Dict[Tuple[str, str], np.ndarray], interval: int):
    """Put every span on its own tonic. -> (tonics, rows) with rows (profile, mode, piece)."""
    tonics: Dict[str, dict] = {}
    rows: List[Tuple[np.ndarray, str, str]] = []
    for (piece, mode), histogram in sorted(spans.items()):
        tonic = estimate(histogram, interval)
        tonics.setdefault(piece, {})[mode] = {
            "tonic_pc": tonic,
            "tonic": PITCH_CLASSES[tonic],
            "partner_share": round(float(histogram[tonic]
                                         + histogram[(tonic + interval) % 12]), 4),
        }
        rows.append((np.roll(histogram, -tonic), mode, piece))
    return tonics, rows


def evaluate(rows: Sequence[Tuple[np.ndarray, str, str]]):
    """-> (profiles, counts, divergence, probe accuracy) over the two compared modes."""
    rotated = collections.defaultdict(list)
    for profile, mode, _ in rows:
        rotated[mode].append(profile)
    profiles = {mode: np.mean(rotated[mode], axis=0) for mode in COMPARED}
    counts = {mode: len(rotated[mode]) for mode in COMPARED}
    divergence = jensen_shannon(profiles[COMPARED[0]], profiles[COMPARED[1]])
    compared = [r for r in rows if r[1] in COMPARED]
    accuracy = probe(np.array([f for f, _, _ in compared]),
                     [m for _, m, _ in compared],
                     [p for _, _, p in compared])
    return profiles, counts, divergence, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the tonic and compare mode profiles")
    parser.add_argument("--set", default="full")
    parser.add_argument("--interval", type=int, default=7,
                        help="the tonic's structural partner, in semitones. "
                             "7 perfect fifth, 5 perfect fourth, 0 plain argmax")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    parser.add_argument("--sweep", action="store_true",
                        help="score every partner interval and exit, writing nothing")
    parser.add_argument("--out", default="tonic.json")
    parser.add_argument("--plot", default="pansori/docs/img/mode_profiles.png")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = {r["item_name"]: r for r in read_jsonl(out_dir / "segments.jsonl")}
    notes = {r["item_name"]: r["notes"] for r in read_jsonl(out_dir / "notes.jsonl")}

    spans = histograms(segments, notes, args.min_seconds)
    pieces = {piece for piece, _ in spans}
    print(f"{len(spans)} spans over {args.min_seconds:.0f}s, across {len(pieces)} pieces")

    if args.sweep:
        # Intervals k and 12-k select the same pair of pitch classes, differing only in
        # which end of it is called the tonic, so they rotate every profile by the same
        # amount and score identically. Both are printed rather than folded away,
        # because which end is the tonic is exactly what the sweep cannot decide.
        print(f"\n{'partner':>7s} {'JSD (bits)':>11s} {'probe':>8s}")
        for interval in range(12):
            _, rows = rotate(spans, interval)
            _, _, divergence, accuracy = evaluate(rows)
            print(f"{'+%d' % interval:>7s} {divergence:11.4f} {accuracy:8.1%}")
        return

    tonics, rows = rotate(spans, args.interval)
    (out_dir / args.out).write_text(
        json.dumps({"interval": args.interval,
                    "min_seconds": args.min_seconds,
                    "pieces": tonics}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    profiles, counts, divergence, accuracy = evaluate(rows)
    print("\nper-mode profile, aggregated over pieces (% of sung time):")
    print(profile_table(profiles))

    compared = [r for r in rows if r[1] in COMPARED]
    features = np.array([f for f, _, _ in compared])
    labels = [m for _, m, _ in compared]
    pieces_of = [p for _, _, p in compared]

    # What the rotation bought: the same probe on absolute pitch classes.
    absolute = probe(np.array([spans[(p, m)] for _, m, p in compared]), labels, pieces_of)
    # Singers are not evenly spread over the modes (정정렬 45% ujo, 이일주 14%), and a
    # singer appears in many pieces, so a piece-level split still lets the probe learn
    # a singer's habits. Holding out whole singers is the control that matters.
    speakers = piece_speakers(segments, out_dir / "spk_map.json")
    by_singer = (probe(features, labels, [speakers.get(p, "?") for p in pieces_of])
                 if speakers else float("nan"))
    # And a probe that scores above chance on shuffled labels is measuring the
    # protocol, not the corpus.
    shuffled = list(labels)
    np.random.default_rng(0).shuffle(shuffled)
    chance = probe(features, shuffled, pieces_of)

    print(f"\n{COMPARED[0]} vs {COMPARED[1]}  ({len(compared)} spans)")
    print(f"  Jensen-Shannon divergence  {divergence:.4f} bits  (0 identical, 1 disjoint)")
    print(f"  logistic probe, by piece   {accuracy:.1%} balanced accuracy  (chance 50%)")
    print(f"  ... held out by singer     {by_singer:.1%}")
    print(f"  ... without the rotation   {absolute:.1%}")
    print(f"  ... on shuffled labels     {chance:.1%}")

    plot(profiles, counts, args.interval, divergence, accuracy, by_singer,
         REPO / args.plot)
    print(f"-> {out_dir / args.out}")


if __name__ == "__main__":
    main()
