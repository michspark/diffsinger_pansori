"""Borrow the changbon's syllables without changing how many Whisper heard.

Stage 03b replaces a Whisper segment with a changbon span whose edges come from
one diff over the whole song. When that diff slips, the span is the wrong
length -- measured over the 11,174 adopted rows, the span is a median 1.00x
Whisper's syllable count but p90 2.00x below match_score 0.55 -- and the label
then asks the aligner to place syllables the singer never sang.

This keeps the two sources to what each is good at. Whisper listened to this
recording, so its syllable *count* is the acoustic fact; the changbon knows
which syllables they were. Whisper's syllables are aligned one-to-one against
the changbon neighbourhood, and a stretch that differs is rewritten only when
it is the same length on both sides and pinned by exact matches around it.
Anything else stays as Whisper wrote it. The syllable count never changes.
"""

import difflib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

from pansori_lib import to_jamo

GAP_WHISPER = 1.1      # a Whisper syllable left on nothing -- dearer than any substitution
GAP_BOOK = 0.6         # an interior book syllable skipped over
MAX_RUN = 5            # longest differing stretch that will still be rewritten
MIN_EDGE_ANCHOR = 2    # a run touching the segment edge needs this many matches on its other side


@dataclass
class Hybrid:
    text: str
    n_syllables: int
    n_replaced: int
    coverage: float                     # share of Whisper syllables sitting on a book syllable
    edits: List[dict] = field(default_factory=list)   # {pos, whisper, book}
    span: Optional[Tuple[int, int]] = None            # [first, last) book syllables touched


@lru_cache(maxsize=None)
def syllable_cost(a: str, b: str) -> float:
    """0 for the same syllable, 1 for nothing in common, jamo overlap between."""
    if a == b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, to_jamo(a), to_jamo(b), autojunk=False).ratio()


def align(whisper: List[str], ref: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Semi-global alignment: all of `whisper`, any contiguous part of `ref`.

    Leading and trailing book syllables are free -- Whisper simply did not hear
    them, and this segment must not acquire them. A Whisper syllable, on the
    other hand, was sung, so leaving it unpaired (GAP_WHISPER) costs more than
    pairing it with any book syllable at all (at most 1.0). Skipping an
    interior book syllable costs GAP_BOOK. The two together mean equal-length
    stretches pair off one-to-one, and gaps appear only where the lengths
    genuinely differ -- which is exactly what hybridize() refuses to rewrite.
    """
    n, m = len(whisper), len(ref)
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        cost[0][j] = 0.0
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i * GAP_WHISPER, "up"
        for j in range(1, m + 1):
            best = cost[i - 1][j - 1] + syllable_cost(whisper[i - 1], ref[j - 1])
            move = "diag"
            if cost[i - 1][j] + GAP_WHISPER < best:
                best, move = cost[i - 1][j] + GAP_WHISPER, "up"
            if cost[i][j - 1] + GAP_BOOK < best:
                best, move = cost[i][j - 1] + GAP_BOOK, "left"
            cost[i][j], back[i][j] = best, move

    columns, i, j = [], n, min(range(m + 1), key=lambda j: cost[n][j])
    while i > 0:
        move = back[i][j]
        if move == "diag":
            columns.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif move == "up":
            columns.append((i - 1, None))
            i -= 1
        else:
            columns.append((None, j - 1))
            j -= 1
    columns.reverse()
    return columns


def whisper_syllables(text: str) -> Tuple[List[str], List[bool]]:
    """Hangul syllables of a Whisper line, with where its spaces fell."""
    syllables, space_before = [], []
    for word in text.split():
        first = True
        for char in word:
            if "가" <= char <= "힣":
                syllables.append(char)
                space_before.append(first and bool(syllables) and len(syllables) > 1)
                first = False
    return syllables, space_before


def hybridize(whisper_text: str, ref: List[str], ref_space_before: List[bool],
              max_run: int = MAX_RUN, min_edge_anchor: int = MIN_EDGE_ANCHOR) -> Hybrid:
    """Rewrite Whisper's syllables from the changbon where -- and only where -- it is safe.

    A *run* is a maximal stretch of columns that are not exact matches. It is
    rewritten when every column pairs one Whisper syllable with one book
    syllable (same length by construction), it is no longer than `max_run`,
    and it is pinned: exact matches on both sides, or, if it touches the
    segment edge, at least `min_edge_anchor` matches on the other side.
    """
    whisper, whisper_space = whisper_syllables(whisper_text)
    if not whisper:
        return Hybrid(whisper_text, 0, 0, 0.0)

    columns = align(whisper, ref)
    kind = ["eq" if i is not None and j is not None and whisper[i] == ref[j]
            else "sub" if i is not None and j is not None
            else "gap"
            for i, j in columns]

    accepted = [False] * len(columns)
    start = 0
    while start < len(columns):
        if kind[start] == "eq":
            start += 1
            continue
        end = start
        while end < len(columns) and kind[end] != "eq":
            end += 1
        left = _anchor_len(kind, start - 1, -1)
        right = _anchor_len(kind, end, +1)
        clean = all(k == "sub" for k in kind[start:end])
        pinned = (left > 0 and right > 0) or (left >= min_edge_anchor) or (right >= min_edge_anchor)
        if clean and pinned and end - start <= max_run:
            for index in range(start, end):
                accepted[index] = True
        start = end

    out, edits, touched, on_book = [], [], [], 0
    for index, (i, j) in enumerate(columns):
        if i is None:
            continue                                  # book syllable Whisper never heard
        if kind[index] == "eq" or accepted[index]:
            out.append((ref[j], ref_space_before[j]))
            touched.append(j)
            on_book += 1
            if accepted[index]:
                edits.append({"pos": i, "whisper": whisper[i], "book": ref[j]})
        else:
            out.append((whisper[i], whisper_space[i]))

    if not edits:
        text = whisper_text
    else:
        text = "".join((" " if space and k else "") + syllable
                       for k, (syllable, space) in enumerate(out))
    span = (min(touched), max(touched) + 1) if touched else None
    return Hybrid(text, len(whisper), len(edits), on_book / len(whisper), edits, span)


def _anchor_len(kind: List[str], index: int, step: int) -> int:
    count = 0
    while 0 <= index < len(kind) and kind[index] == "eq":
        count += 1
        index += step
    return count
