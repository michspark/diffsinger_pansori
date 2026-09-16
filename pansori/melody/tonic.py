#!/usr/bin/env python3
"""Find each piece's tonic, so notes can be written as degrees rather than pitch names.

Tokenisation needs this. `A#4` means nothing on its own -- every singer and piece
sits somewhere different, so the same scale degree appears under a different name in
every recording and a model trained on names learns the recordings. Degrees measured
from the tonic are comparable across the corpus; pitch names are not.

**The estimate.** The tonic is the pitch class `p` maximising

    h[p] + h[(p + interval) % 12]

over a duration-weighted pitch-class histogram: the tonic and its structural partner
together, a perfect fifth apart by default. Taking the single loudest pitch class
instead picks whichever end of that pair the singer happened to dwell on, which is
not stable across pieces -- and the pair is visible in the data, both modes resting
on degree 0 and degree 7 once rotated (`pansori/docs/MELODY.md` §2).

`--interval` selects the partner: 7 a perfect fifth, 5 a perfect fourth (the same
pair seen from its other end), 0 the plain argmax.

**The scope.** A tonic is estimated per (piece, mode) span, falling back to the piece
with its modes pooled, then to the corpus. Per span rather than per piece because the
mode-specific estimate carries real modal information -- pooling a piece's modes
costs the mode probe 4.5 points (72.3% -> 67.8%) -- and the fallback because 7.9% of
bindable utterances sit in a span too short to estimate from.

Input : data/{set}/segments.jsonl + notes.jsonl  (+ melody_data.jsonl for coverage)
Output: data/{set}/tonic.json + docs/img/mode_profiles.png

Runs in any env with numpy, sklearn and matplotlib.

  python pansori/melody/tonic.py --set full
  python pansori/melody/tonic.py --set full --sweep        # score every partner interval
  python pansori/melody/tonic.py --set full --stability    # split-half reliability
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

# Below this a histogram is one held note, not a distribution: spans run to a median
# 91 s but a 10th percentile of 12 s. Spans under it fall back to their piece.
MIN_SECONDS = 60.0

# The two modes with enough material to compare. pyeongjo has 1.5 h, seolleongje and
# gyeongdeureum together under 20 minutes.
COMPARED = ("우조", "계면조")

# Matplotlib ships no Hangul glyphs, and adding a font dependency to read two labels
# is not worth it. The figure is part of the English documentation anyway.
ROMAN = {"우조": "ujo", "계면조": "gyemyeonjo", "평조": "pyeongjo"}


# ---------------------------------------------------------------- the estimate

def histograms(segments: Dict[str, dict],
               notes: Dict[str, list]) -> Dict[Tuple[str, str], np.ndarray]:
    """(piece, mode) -> pitch-class histogram in seconds, unnormalised.

    Weighted by how long each pitch class sounds rather than how often it is struck,
    because a tonic is held and a passing tone is not.

    Built from notes.jsonl, not melody_data.jsonl: the tonic is a property of the
    piece, and the binding QC that drops 46.5% of utterances is about syllables,
    which have no bearing on which pitches a piece dwells on.
    """
    raw: Dict[Tuple[str, str], np.ndarray] = collections.defaultdict(lambda: np.zeros(12))
    for item, note_list in notes.items():
        segment = segments.get(item)
        if segment is None:
            continue
        key = (segment["uuid"], segment["mode"])
        for pitch, start, end in note_list:
            raw[key][int(pitch) % 12] += end - start
    return dict(raw)


def estimate(histogram: np.ndarray, interval: int = 7) -> int:
    """The tonic's pitch class: the pair (p, p + interval) carrying the most time."""
    if interval == 0:
        return int(histogram.argmax())
    score = np.array([histogram[p] + histogram[(p + interval) % 12] for p in range(12)])
    return int(score.argmax())


def build_table(hists: Dict[Tuple[str, str], np.ndarray], interval: int,
                min_seconds: float) -> dict:
    """The tonic of every span, piece and of the corpus, as written to tonic.json."""
    pooled: Dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(12))
    for (piece, _), histogram in hists.items():
        pooled[piece] += histogram
    whole = sum(pooled.values(), np.zeros(12))

    def entry(histogram: np.ndarray) -> dict:
        tonic = estimate(histogram, interval)
        share = histogram / histogram.sum()
        return {
            "tonic_pc": tonic,
            "tonic": PITCH_CLASSES[tonic],
            # How much of the piece the estimate rests on. A pair holding 20% of the
            # time is a skeleton; one holding 10% is the histogram being flat.
            "pair_share": round(float(share[tonic] + share[(tonic + interval) % 12]), 4),
            "seconds": round(float(histogram.sum()), 1),
        }

    pieces: Dict[str, dict] = {}
    for piece, histogram in sorted(pooled.items()):
        pieces[piece] = {"pooled": entry(histogram) if histogram.sum() >= min_seconds else None,
                         "modes": {}}
    for (piece, mode), histogram in sorted(hists.items()):
        if histogram.sum() >= min_seconds:
            pieces[piece]["modes"][mode] = entry(histogram)

    return {"interval": interval, "min_seconds": min_seconds,
            "corpus": entry(whole), "pieces": pieces}


def resolve(table: dict, piece: str, mode: str) -> Tuple[int, str]:
    """(tonic pitch class, which level it came from) for one utterance.

    The fallback chain is span -> piece -> corpus. The level is returned because a
    corpus-level tonic is a guess and whatever consumes this should be able to say so.
    """
    entry = table["pieces"].get(piece)
    if entry is not None:
        if mode in entry["modes"]:
            return entry["modes"][mode]["tonic_pc"], "span"
        if entry["pooled"] is not None:
            return entry["pooled"]["tonic_pc"], "piece"
    return table["corpus"]["tonic_pc"], "corpus"


# ---------------------------------------------------------------- the checks

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """In bits. 0 when the two profiles are identical, 1 when disjoint."""
    m = (p + q) / 2

    def kl(a, b):
        return float(np.sum(np.where(a > 0, a * np.log2(a / np.maximum(b, 1e-12)), 0.0)))

    return (kl(p, m) + kl(q, m)) / 2


def probe(features: np.ndarray, labels: Sequence[str], groups: Sequence[str]) -> float:
    """Balanced accuracy of a logistic probe over grouped folds. 0.5 is chance.

    Deliberately linear: a strong model would leave it unclear whether the mode was
    in the features or in the classifier. Balanced because the classes are 306 to
    124, where plain accuracy rewards answering gyemyeonjo every time.
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


def split_half(segments: Dict[str, dict], notes: Dict[str, list],
               interval: int, by_piece: bool) -> Tuple[float, int, float]:
    """Does the estimate survive being made twice, from alternate utterances?

    The honest reliability check for an estimator nothing else can validate. Each
    half sees half the material, so it understates the full estimate's reliability,
    but the comparison between scopes is fair.
    """
    groups: Dict[object, List[Tuple[str, list]]] = collections.defaultdict(list)
    for item, note_list in notes.items():
        segment = segments.get(item)
        if segment is None:
            continue
        key = segment["uuid"] if by_piece else (segment["uuid"], segment["mode"])
        groups[key].append((item, note_list))

    agree = total = 0
    lengths = []
    for _, items in groups.items():
        halves = [np.zeros(12), np.zeros(12)]
        for index, (_, note_list) in enumerate(sorted(items)):
            for pitch, start, end in note_list:
                halves[index % 2][int(pitch) % 12] += end - start
        if halves[0].sum() <= 0 or halves[1].sum() <= 0:
            continue
        total += 1
        lengths.append(halves[0].sum() + halves[1].sum())
        agree += estimate(halves[0], interval) == estimate(halves[1], interval)
    return agree / max(total, 1), total, float(np.median(lengths)) if lengths else 0.0


# ---------------------------------------------------------------- reporting

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


def rotated_rows(hists: Dict[Tuple[str, str], np.ndarray], interval: int,
                 min_seconds: float) -> List[Tuple[np.ndarray, str, str]]:
    """(profile on its own tonic, mode, piece) for every span long enough to stand alone."""
    rows = []
    for (piece, mode), histogram in sorted(hists.items()):
        if histogram.sum() < min_seconds:
            continue
        share = histogram / histogram.sum()
        rows.append((np.roll(share, -estimate(share, interval)), mode, piece))
    return rows


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


def profile_table(profiles: Dict[str, np.ndarray]) -> str:
    lines = ["degree  " + " ".join(f"{d:5d}" for d in range(12))]
    for mode, profile in profiles.items():
        lines.append(f"{mode:6s}  " + " ".join(f"{x * 100:5.1f}" for x in profile))
    if all(m in profiles for m in COMPARED):
        delta = (profiles[COMPARED[0]] - profiles[COMPARED[1]]) * 100
        lines.append("diff    " + " ".join(f"{x:+5.1f}" for x in delta))
    return "\n".join(lines)


def coverage(table: dict, melody_data: Path) -> None:
    """Which fallback level the utterances that will actually be tokenised land on."""
    if not melody_data.exists():
        return
    levels = collections.Counter()
    for row in read_jsonl(melody_data):
        if row["qc"]["pass"]:
            levels[resolve(table, row["uuid"], row["mode"])[1]] += 1
    total = sum(levels.values())
    print(f"\ntonic coverage over {total:,} bindable utterances:")
    for level in ("span", "piece", "corpus"):
        print(f"  {level:7s} {levels[level]:6,d}  {levels[level] / max(total, 1):6.1%}")


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


# ---------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the tonic of every piece")
    parser.add_argument("--set", default="full")
    parser.add_argument("--interval", type=int, default=7,
                        help="the tonic's structural partner, in semitones. "
                             "7 perfect fifth, 5 perfect fourth, 0 plain argmax")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    parser.add_argument("--sweep", action="store_true",
                        help="score every partner interval and exit, writing nothing")
    parser.add_argument("--stability", action="store_true",
                        help="split-half reliability per scope and exit, writing nothing")
    parser.add_argument("--out", default="tonic.json")
    parser.add_argument("--plot", default="pansori/docs/img/mode_profiles.png")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = {r["item_name"]: r for r in read_jsonl(out_dir / "segments.jsonl")}
    notes = {r["item_name"]: r["notes"] for r in read_jsonl(out_dir / "notes.jsonl")}
    hists = histograms(segments, notes)
    long_enough = sum(1 for h in hists.values() if h.sum() >= args.min_seconds)
    print(f"{len(hists)} (piece, mode) spans, {long_enough} over {args.min_seconds:.0f}s, "
          f"across {len({p for p, _ in hists})} pieces")

    if args.stability:
        print("\nthe same estimate made twice, from alternate utterances:")
        for by_piece, label in ((False, "per (piece, mode)"), (True, "per piece")):
            rate, count, median = split_half(segments, notes, args.interval, by_piece)
            print(f"  {label:18s} {rate:6.1%} agree   ({count} units, median {median:.0f}s)")
        return

    if args.sweep:
        # Intervals k and 12-k select the same pair of pitch classes, differing only in
        # which end of it is called the tonic, so they rotate every profile by the same
        # amount and score identically. Both are printed rather than folded away,
        # because which end is the tonic is exactly what the sweep cannot decide.
        print(f"\n{'partner':>7s} {'JSD (bits)':>11s} {'probe':>8s}")
        for interval in range(12):
            _, _, divergence, accuracy = evaluate(
                rotated_rows(hists, interval, args.min_seconds))
            print(f"{'+%d' % interval:>7s} {divergence:11.4f} {accuracy:8.1%}")
        return

    table = build_table(hists, args.interval, args.min_seconds)
    (out_dir / args.out).write_text(
        json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    coverage(table, out_dir / "melody_data.jsonl")

    rows = rotated_rows(hists, args.interval, args.min_seconds)
    profiles, counts, divergence, accuracy = evaluate(rows)
    print("\nper-mode profile, aggregated over pieces (% of sung time):")
    print(profile_table(profiles))

    compared = [r for r in rows if r[1] in COMPARED]
    features = np.array([f for f, _, _ in compared])
    labels = [m for _, m, _ in compared]
    pieces_of = [p for _, _, p in compared]

    # What the rotation bought: the same probe on absolute pitch classes.
    absolute = probe(np.array([hists[(p, m)] / hists[(p, m)].sum() for _, m, p in compared]),
                     labels, pieces_of)
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
