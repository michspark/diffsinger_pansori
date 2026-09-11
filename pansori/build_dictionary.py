#!/usr/bin/env python3
"""Generate dictionaries/korean-coda.txt — every Hangul syllable -> Coda phones.

openvpi/DiffSinger builds its phoneme set from the dictionary files listed under
`dictionaries:` (utils/phoneme_utils.py), and uses the same file at inference to
turn lyrics into a phoneme sequence. The old MoonInTheRiver pipeline had no such
file: its phone inventory was the hardcoded CODA_36 set and there was no
lyrics -> phoneme path at all.

The mapping here is the syllable-level inverse of alignment/ipa_to_coda.csv, which
maps korean_mfa IPA to the same 34 Coda tokens. A syllable is written as

    [onset] [glide] vowel [coda]

so 갸 -> `g y a`, 의 -> `eu i`, 왕 -> `w a NG`.

CAVEAT: this is a syllable-local mapping with no sandhi. Training labels come from
MFA, which does apply liaison and consonant assimilation (책이 -> ch e g i, not
ch e K i), so inference
phonemes will differ from training phonemes at some syllable boundaries. That is
inherent to a syllable dictionary and is the same trade-off upstream makes; it is
documented here rather than silently absorbed.

Usage:  python pansori/build_dictionary.py [-o dictionaries/korean-coda.txt]
"""

import argparse
from pathlib import Path

BASE = 0xAC00  # '가'
N_JUNG, N_JONG = 21, 28

# 19 onsets -- ㅇ is silent, hence the empty string.
ONSET = [
    'g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's',
    'ss', '', 'j', 'jj', 'ch', 'k', 't', 'p', 'h',
]

# 21 nuclei -- the glide precedes the vowel; ㅐ/ㅔ and the ㅙ/ㅚ/ㅞ group are merged,
# matching ipa_to_coda.csv (ɛ -> e, ɥ -> w).
NUCLEUS = [
    'a', 'e', 'y a', 'y e', 'eo', 'e', 'y eo', 'y e', 'o', 'w a',
    'w e', 'w e', 'y o', 'u', 'w eo', 'w e', 'w i', 'y u', 'eu', 'eu i',
    'i',
]

# 28 codas -- index 0 is "no coda"; clusters collapse to their representative sound.
CODA = [
    '', 'K', 'K', 'K', 'N', 'N', 'N', 'T', 'L', 'K',
    'M', 'L', 'L', 'L', 'P', 'L', 'M', 'P', 'P', 'T',
    'T', 'NG', 'T', 'T', 'K', 'T', 'P', 'T',
]

# The 34 tokens the trained corpus actually uses (SP excluded — it is a reserved
# global phoneme in openvpi and must not appear in a dictionary).
EXPECTED = {
    'g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', 'j', 'jj',
    'ch', 'k', 't', 'p', 'h', 'y', 'w', 'a', 'eo', 'o', 'u', 'eu', 'i', 'e',
    'K', 'N', 'T', 'L', 'M', 'P', 'NG',
}


def decompose(code: int) -> str:
    """Coda phoneme string for one Hangul syllable code point."""
    index = code - BASE
    onset = ONSET[index // (N_JUNG * N_JONG)]
    nucleus = NUCLEUS[(index // N_JONG) % N_JUNG]
    coda = CODA[index % N_JONG]
    return ' '.join(part for part in (onset, nucleus, coda) if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--out', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'dictionaries' / 'korean-coda.txt')
    args = parser.parse_args()

    lines = []
    produced = set()
    for code in range(BASE, BASE + 11172):
        phones = decompose(code)
        produced.update(phones.split())
        lines.append(f'{chr(code)}\t{phones}')

    missing = EXPECTED - produced
    extra = produced - EXPECTED
    if missing or extra:
        raise SystemExit(
            f'Phoneme set mismatch against the trained corpus.\n'
            f'  missing: {sorted(missing)}\n'
            f'  extra:   {sorted(extra)}'
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{args.out}: {len(lines)} syllables, {len(produced)} phonemes')


if __name__ == '__main__':
    main()
