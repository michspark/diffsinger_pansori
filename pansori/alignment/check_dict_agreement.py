#!/usr/bin/env python3
"""Measure how often convert_phone reproduces the dictionary, syllable by syllable.

ipa_to_coda.csv is a hand-built table and every edit to it is a guess until it
is scored. This is the scorer. It reads the MFA TextGrids straight out of
data/{set}/aligned/ (raw IPA, before conversion), keeps only the `words`
intervals that are exactly one Hangul syllable -- those have no segmentation
ambiguity -- converts their phones with the same context rules Stage 4 uses,
and compares the result against dictionaries/korean-coda.txt.

--syllables widens that to N-syllable words, scored against the concatenated
dictionary entries. Longer words score lower (every liaison in them counts
against the whole word), so use it to reach symbols that single syllables never
expose -- ʎ only ever appears inside 2-syllable words like 살려/빨리.

The dictionary is the reference here, not the other way round: it is generated
from the jamo table in pansori/build_dictionary.py and is independent of the
csv, so a disagreement points at one side or the other rather than at nothing.

Not every disagreement is a defect. Liaison is regular Korean and a
context-free syllable dictionary cannot express it (은 -> `eu n`, 좋 -> `j o`),
so the score will never reach 100%. Use it as a differential: run before an
edit, run after, and keep the edit if the number goes up.

TextGrids are parsed here rather than with praatio so this runs in any env
(praatio is only installed in `aligner`).

Usage (any env, from the repo root):
  python pansori/alignment/check_dict_agreement.py
  python pansori/alignment/check_dict_agreement.py --top 40 --focus ç ɸ ʎ
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pansori" / "preprocess"))

from pansori_lib import (SILENCE_LABELS, convert_phone,  # noqa: E402
                         load_ipa_to_coda)

_INTERVAL = re.compile(
    r'intervals \[\d+\]:\s*'
    r'xmin = ([\d.]+)\s*'
    r'xmax = ([\d.]+)\s*'
    r'text = "((?:[^"]|"")*)"',
    re.S,
)
_TIER = re.compile(r'name = "(\w+)"')


def parse_textgrid(path):
    """{tier_name: [(label, xmin, xmax), ...]} from a long-format TextGrid.

    praatio would do this, but it only exists in the `aligner` env and the
    long format is regular enough that a regex over each tier's slice is
    equivalent for our two interval tiers.
    """
    text = path.read_text(encoding="utf-8")
    marks = [(m.start(), m.group(1)) for m in _TIER.finditer(text)]
    tiers = {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        tiers[name] = [(m.group(3).replace('""', '"').strip(),
                        float(m.group(1)), float(m.group(2)))
                       for m in _INTERVAL.finditer(text[pos:end])]
    return tiers


def load_dictionary(path):
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            syllable, phones = line.rstrip("\n").split("\t", 1)
            entries[syllable] = tuple(phones.split())
    return entries


def convert_span(intervals, lo, hi, mapping):
    """Coda tokens for the phones inside [lo, hi], with global next-phone context.

    The context phone can sit outside the span -- a syllable-final nasal is
    decided by whatever starts the *next* syllable -- so next_ipa is taken from
    the whole utterance, exactly as 04_align.py does.
    """
    tokens = []
    for idx, (label, start, end) in enumerate(intervals):
        if start < lo - 1e-6 or end > hi + 1e-6:
            continue
        if label.lower() in SILENCE_LABELS:
            continue
        next_ipa = None
        for nlab, _, _ in intervals[idx + 1:]:
            if nlab.lower() not in SILENCE_LABELS:
                next_ipa = nlab
                break
        tokens.extend(convert_phone(label, next_ipa, mapping))
    return tokens


def main():
    parser = argparse.ArgumentParser(description="score the IPA mapping against the dictionary")
    parser.add_argument("--set", default="full")
    parser.add_argument("--dictionary", default="dictionaries/korean-coda.txt")
    parser.add_argument("--top", type=int, default=25,
                        help="how many mismatch patterns to print")
    parser.add_argument("--focus", nargs="*", default=[],
                        help="tally separately the syllables containing this IPA symbol")
    parser.add_argument("--syllables", type=int, default=1,
                        help="syllable count of the eojeol to compare (0 = any length)")
    parser.add_argument("--limit", type=int, default=0,
                        help="number of TextGrids to sample (0 = all)")
    args = parser.parse_args()

    aligned = REPO / "pansori" / "data" / args.set / "aligned"
    grids = sorted(aligned.rglob("*.TextGrid"))
    if args.limit:
        grids = grids[:args.limit]
    if not grids:
        parser.error(f"no TextGrid at: {aligned}")

    mapping = load_ipa_to_coda()
    dictionary = load_dictionary(REPO / args.dictionary)

    total = match = 0
    mismatches = Counter()      # (syllable, observed, dictionary) -> count
    focus_stats = {sym: [0, 0] for sym in args.focus}   # sym -> [matched, total]
    oov_syllables = Counter()

    for n, grid in enumerate(grids):
        tiers = parse_textgrid(grid)
        words = tiers.get("words", [])
        phones = tiers.get("phones", [])
        if not words or not phones:
            continue
        for label, lo, hi in words:
            if not label or not all("가" <= ch <= "힣" for ch in label):
                continue
            if args.syllables and len(label) != args.syllables:
                continue
            entries = [dictionary.get(ch) for ch in label]
            if any(e is None for e in entries):
                oov_syllables[label] += 1
                continue
            expected = tuple(p for e in entries for p in e)
            raw = [p for p, s, e in phones
                   if s >= lo - 1e-6 and e <= hi + 1e-6
                   and p.lower() not in SILENCE_LABELS]
            if not raw:
                continue
            got = tuple(convert_span(phones, lo, hi, mapping))
            total += 1
            ok = got == expected
            match += ok
            if not ok:
                mismatches[(label, " ".join(got), " ".join(expected))] += 1
            for sym in args.focus:
                if any(sym in p for p in raw):
                    focus_stats[sym][1] += 1
                    focus_stats[sym][0] += ok
        if (n + 1) % 5000 == 0:
            print(f"  {n + 1}/{len(grids)} grids", file=sys.stderr, flush=True)

    if not total:
        print("no comparable eojeol")
        return

    unit = f"{args.syllables}-syllable" if args.syllables else "all"
    print(f"\n{unit} eojeol: {total:,} · matched {match:,} · "
          f"agreement {100 * match / total:.2f}%")

    print(f"\ntop {args.top} mismatches:")
    print(f"  {'syl':<4} {'observed':<16} {'dictionary':<16} {'count':>7}")
    for (syl, got, exp), count in mismatches.most_common(args.top):
        print(f"  {syl:<4} {got:<16} {exp:<16} {count:>7,}")

    if args.focus:
        print("\nby symbol:")
        for sym, (ok, n) in focus_stats.items():
            rate = f"{100 * ok / n:.2f}%" if n else "—"
            print(f"  {sym:<5} {ok:>7,}/{n:>7,}  {rate}")

    if oov_syllables:
        print(f"\n{len(oov_syllables)} distinct syllables not in the dictionary "
              f"({sum(oov_syllables.values()):,} occurrences): "
              + " ".join(s for s, _ in oov_syllables.most_common(10)))


if __name__ == "__main__":
    main()
