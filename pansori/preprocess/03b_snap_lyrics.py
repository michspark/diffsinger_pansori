#!/usr/bin/env python3
"""Stage 3b: snap Whisper's transcription onto the 창본 text -> lyrics_snapped.jsonl.

Whisper cannot hold pansori. On 01ed19bb (김소희 · 춘향가 군로사령) it produced
'굴러 살인길 나간다 살인길 나간다' where the 창본 reads '군로사령이 나간다 사령군로가
나간다'. That matters because the lyrics are not the deliverable -- they are MFA's
input, and a wrong word makes MFA drop the span to silence and pin its phones to
the 0.03s floor (pansori/docs/SEGMENTATION_VERDICT.md §2).

The fix does not need a better transcriber, only the book. segments.jsonl carries
song_stem plus song_start, so a song's segments are already in time order: paste
one song's Whisper text into a single string, lay it over the book, and slide.
Anchor once and the rest follows, because the order cannot cross.

Two passes:
  B. anchor  -- jamo 5-gram density picks the 창본 and the region, per song
  C. snap    -- difflib's monotone opcodes carry each segment to its own span

Nothing is overwritten: lyrics.jsonl stays as it is and every row records both
texts plus a `source` of "book" or "whisper".

Input : lyricbook/corpus.jsonl + data/{set}/segments.jsonl + data/{set}/lyrics.jsonl
Output: data/{set}/lyrics_snapped.jsonl

Run in any env -- stdlib only.
"""

import argparse
import difflib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pansori_lib import PANSORI_DIR, hangul_only, read_jsonl, set_dir, to_jamo, write_jsonl

CORPUS_PATH = PANSORI_DIR / "lyricbook" / "corpus.jsonl"

# ---- Stage B: anchoring -------------------------------------------------

NGRAM = 5              # jamo n-gram; ~1.8 syllables, short enough to survive errors
WINDOW_FACTOR = 2.0    # search window = this many times the query's jamo length
MIN_QUERY_JAMO = 60    # a song this short cannot be anchored with confidence
MIN_WORK_MARGIN = 1.5  # best 바탕 must beat the other 바탕's mean by this factor
# Density cannot be judged against a fixed number, because chance agreement
# grows with the query: the window is a multiple of the query's length, so a
# long song sweeps up common n-grams no matter which book it is held against.
# Measured floor across the other four 바탕 -- books that cannot be the answer:
#
#   query jamo    background     the right 바탕
#          757   0.037-0.052   0.224   (군로사령)
#        1,388   0.067-0.087   0.359   (청이 밥 빌러)
#        7,495   0.160-0.186   0.342   (어사 변복)
#
# The floor triples while the signal does not, so an absolute cut of 0.15 waved
# long songs through on noise and failed short ones at 0.148. Every book sees
# the same query, so the other 바탕 measure that song's own noise floor; the
# test is how far the winner stands above it.
#
# 1.5 is where that ratio starts telling the truth. Running all 387 songs with
# the gate open, and scoring the chosen 창본 against the 바탕 named in the
# filename -- which never feeds the matching, so it is a held-out answer key:
#
#   background ratio    songs   바탕 correct   segments snapped
#         0.0 - 1.2        31           36%              3.1%
#         1.2 - 1.5        61           75%              7.8%
#         1.5 - 1.8        39           95%             16.3%
#         1.8 - 2.0        26          100%             25.3%
#         2.0 +           230          100%             50.4%
#
# Note the last column at the bottom of the table: even with a wrong anchor
# only 3.1% of segments clear Stage C, which is the fallback working as
# designed. A wrong anchor is cheap; refusing a song outright is not.

# ---- Stage C: per-segment acceptance ------------------------------------

MARGIN_FACTOR = 0.3    # widen the anchored window by this much before aligning
MIN_RATIO = 0.45       # jamo similarity between Whisper's text and the book span
MIN_SYLLABLES = 2      # 04_align.py's MIN_HANGUL -- shorter cannot align anyway
MAX_SYLLABLE_RATE = 8.0
# Measured on 2,873 candidate matches over 38 anchored songs, binned by ratio:
#
#   ratio        n   rate p50   p90   p95   p99    max
#   0.00-0.30  411       2.5   77.0  126.0 286.6  392.8   <- window slipped
#   0.45-0.60  570       2.0    4.3    5.1   7.2   16.7
#   0.75-1.00  853       1.9    3.3    4.0   5.1    7.5
#
# Similarity does nearly all the filtering: once a span clears MIN_RATIO its
# rate is a sane 2/sec with a p99 of 7.2, while a slipped window drags in whole
# paragraphs and shows up as 100+/sec. So the cap only has to trim the tail
# above the honest distribution, and 8.0 is where that sits. An earlier 5.0
# threw away correct 자진모리 spans (ratio 0.86 at 6.2 syllables/sec) -- 판소리
# really is sung that fast when the 장단 is fast.


class Book:
    """One 창본, flattened to its sung syllables with an index back to blocks."""

    def __init__(self, name, work, blocks):
        self.name = name
        self.work = work
        self.syllables = []      # sung Hangul, one char per entry
        self.space_before = []   # rebuild the book's own spacing on output
        self.block_of = []
        for block in blocks:
            for word_index, word in enumerate(block["text"].split()):
                for char_index, char in enumerate(word):
                    self.syllables.append(char)
                    self.space_before.append(word_index > 0 and char_index == 0)
                    self.block_of.append(block["block"])

        self.jamo = []           # flat jamo string
        self.jamo_start = []     # syllable index -> first jamo offset
        self.syllable_of = []    # jamo offset -> syllable index
        for index, char in enumerate(self.syllables):
            self.jamo_start.append(len(self.jamo))
            for jamo_char in unicodedata.normalize("NFD", char):
                self.jamo.append(jamo_char)
                self.syllable_of.append(index)
        self.jamo = "".join(self.jamo)

        # gram -> sorted positions. Built once; every song reuses it.
        self.index = defaultdict(list)
        for position in range(len(self.jamo) - NGRAM + 1):
            self.index[self.jamo[position:position + NGRAM]].append(position)

    def text_of(self, start, end):
        """Syllables [start, end) as readable text, with the book's own spacing."""
        out = []
        for index in range(start, end):
            if self.space_before[index] and out:
                out.append(" ")
            out.append(self.syllables[index])
        return "".join(out)


def load_books():
    # corpus.jsonl is not tracked -- it is the text of six published 창본. Say so
    # here rather than letting read_jsonl raise a bare FileNotFoundError.
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"창본 코퍼스가 없다: {CORPUS_PATH}\n"
            f"  Pansori_Data/Lyrics 를 두고 먼저 실행할 것: "
            f"python pansori/lyricbook/build_corpus.py")

    blocks_by_book = defaultdict(list)
    work_of = {}
    for block in read_jsonl(CORPUS_PATH):
        if block["is_sung"]:
            blocks_by_book[block["book"]].append(block)
            work_of[block["book"]] = block["work"]
    return [Book(name, work_of[name], blocks)
            for name, blocks in sorted(blocks_by_book.items())]


def anchor(book, query_jamo):
    """Best (density, jamo position) for this query inside this book.

    Density counts *distinct* query n-grams landing in the window, not raw hits:
    pansori repeats phrases, and counting hits lets one much-repeated gram alone
    win a window it has no business winning.
    """
    grams = {query_jamo[i:i + NGRAM] for i in range(len(query_jamo) - NGRAM + 1)}
    if not grams:
        return 0.0, 0

    hits = sorted((position, gram)
                  for gram in grams for position in book.index.get(gram, ()))
    if not hits:
        return 0.0, 0

    span = int(len(query_jamo) * WINDOW_FACTOR)
    window = Counter()
    best_count, best_position, left = 0, hits[0][0], 0
    for right, (position, gram) in enumerate(hits):
        window[gram] += 1
        while position - hits[left][0] > span:
            left_gram = hits[left][1]
            window[left_gram] -= 1
            if not window[left_gram]:
                del window[left_gram]
            left += 1
        if len(window) > best_count:
            best_count, best_position = len(window), hits[left][0]
    return best_count / len(grams), best_position


def choose_book(books, query_jamo):
    """Score every 창본; return (best, density, position, margin, reason).

    The gate can afford to be generous. A wrong anchor does not put wrong words
    in the output -- it puts the alignment somewhere the per-segment similarity
    check will not clear, and every segment falls back to Whisper. Refusing a
    song here, by contrast, is final. So the anchor only has to rule out songs
    with no signal at all and leave the rest to Stage C.
    """
    scored = [(book, *anchor(book, query_jamo)) for book in books]
    scored.sort(key=lambda row: -row[1])
    best, density, position = scored[0]

    others = [row[1] for row in scored[1:] if row[0].work != best.work]
    background = sum(others) / len(others) if others else 0.0
    margin = density / background if background > 0 else float("inf")

    if len(query_jamo) < MIN_QUERY_JAMO:
        return best, density, position, margin, "너무_짧음"
    if margin < MIN_WORK_MARGIN:
        return best, density, position, margin, "신호부족"
    return best, density, position, margin, None


def build_query(rows):
    """Whisper text of one song -> (jamo string, jamo offset per row boundary).

    Returns bounds[i] = the jamo offset where row i starts, plus a final entry
    for the end, so a row owns query jamo [bounds[i], bounds[i + 1]).
    """
    jamo, bounds = [], [0]
    for row in rows:
        jamo.append(to_jamo(row["text"]))
        bounds.append(bounds[-1] + len(jamo[-1]))
    return "".join(jamo), bounds


def map_query_to_reference(query_jamo, reference_jamo):
    """Monotone query-jamo -> reference-jamo map, one entry per boundary.

    get_opcodes() never crosses itself, so walking it in order gives a map that
    only ever moves forward -- the property that lets one anchor carry a whole
    song. Where the book has syllables Whisper missed (an 'insert'), the map
    jumps across them at that boundary, which attaches them to the preceding
    segment; if that makes the segment absurdly dense the rate cap rejects it.
    """
    mapping = [0] * (len(query_jamo) + 1)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, query_jamo, reference_jamo, autojunk=False).get_opcodes():
        if i2 == i1:
            continue
        for offset in range(i2 - i1):
            if tag == "equal":
                mapping[i1 + offset] = j1 + offset
            elif tag == "delete":          # Whisper heard what the book lacks
                mapping[i1 + offset] = j1
            else:                          # replace -- spread evenly
                mapping[i1 + offset] = j1 + (j2 - j1) * offset // (i2 - i1)
    mapping[len(query_jamo)] = len(reference_jamo)
    return mapping


def snap_song(rows, books):
    """One song's lyric rows -> (snapped rows, info). Rejected rows keep Whisper."""
    query_jamo, bounds = build_query(rows)
    book, density, position, margin, reason = choose_book(books, query_jamo)

    info = dict(book=book.name, density=round(density, 3),
                margin=round(margin, 2) if margin != float("inf") else None,
                reason=reason, rejections=[])
    if reason:
        return [passthrough(row) for row in rows], info

    margin = int(len(query_jamo) * MARGIN_FACTOR)
    start = max(0, position - margin)
    end = min(len(book.jamo),
              position + int(len(query_jamo) * WINDOW_FACTOR) + margin)
    mapping = map_query_to_reference(query_jamo, book.jamo[start:end])

    out, cursor = [], 0
    for index, row in enumerate(rows):
        snapped, cursor = snap_row(row, book, mapping, start, end, cursor,
                                   bounds[index], bounds[index + 1],
                                   density, info["rejections"])
        out.append(snapped)
    return out, info


def snap_row(row, book, mapping, window_start, window_end, cursor,
             query_start, query_end, density, rejections):
    """Carry one segment to its book span; return (row, new cursor).

    `cursor` is the first syllable this segment may claim. A jamo boundary
    usually lands inside a syllable, and without the cursor both neighbours
    round to it and the syllable is emitted twice ('… 아부' then '부 몸을 …').
    Whoever gets there first keeps it.
    """
    if query_end == query_start:
        return passthrough(row), cursor

    ref_start = window_start + mapping[query_start]
    ref_end = window_start + mapping[query_end]
    if ref_end <= ref_start:
        return passthrough(row), cursor

    first = max(book.syllable_of[ref_start], cursor)
    last = book.syllable_of[min(ref_end, window_end) - 1] + 1
    if last <= first:
        return passthrough(row), cursor

    text = book.text_of(first, last)
    syllables = last - first
    ratio = difflib.SequenceMatcher(
        None, to_jamo(row["text"]), to_jamo(text), autojunk=False).ratio()
    rate = syllables / row["duration"] if row["duration"] > 0 else 99.0

    if syllables < MIN_SYLLABLES or ratio < MIN_RATIO or rate > MAX_SYLLABLE_RATE:
        rejections.append(f"    기각 {row['item_name']} 유사도={ratio:.2f} "
                          f"음절={syllables} 속도={rate:.1f}  후보: {text[:40]}")
        return passthrough(row), last

    snapped = passthrough(row)
    snapped.update(
        text=text,
        source="book",
        book_ref={"book": book.name, "block": book.block_of[first],
                  "span": [first, last]},
        match_score=round(ratio, 3),
        song_anchor_density=round(density, 3),
    )
    return snapped, last


def passthrough(row):
    """Whisper's row, restated in the snapped schema."""
    return {
        "item_name": row["item_name"],
        "duration": row["duration"],
        "text_raw": row.get("text_raw", ""),
        "text_whisper": row["text"],
        "text": row["text"],
        "source": "whisper",
        "book_ref": None,
        "match_score": None,
        "song_anchor_density": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b: snap lyrics to the 창본")
    parser.add_argument("--set", default="full", help="data/{set} working dir")
    parser.add_argument("--limit-songs", type=int, default=0,
                        help="only process this many songs (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print before/after instead of writing")
    parser.add_argument("--seed", type=int, default=0, help="song sampling seed")
    parser.add_argument("--verbose", action="store_true", help="show rejections")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = read_jsonl(out_dir / "segments.jsonl")
    lyrics = read_jsonl(out_dir / "lyrics.jsonl")
    stem_of = {r["item_name"]: r["song_stem"] for r in segments}

    songs = defaultdict(list)
    for row in lyrics:
        songs[stem_of.get(row["item_name"], "?")].append(row)
    for rows in songs.values():
        rows.sort(key=lambda r: r["item_name"])

    order = sorted(songs)
    if args.limit_songs:
        random.Random(args.seed).shuffle(order)
        order = order[:args.limit_songs]

    print(f"창본 로드 중... ({CORPUS_PATH})")
    books = load_books()
    print(f"  창본 {len(books)}개 / 창 음절 "
          f"{sum(len(b.syllables) for b in books):,}자\n")
    print(f"곡 {len(order)}개, 세그먼트 {sum(len(songs[s]) for s in order):,}개 처리\n")

    results, song_report = [], []
    for count, stem in enumerate(order, start=1):
        rows, info = snap_song(songs[stem], books)
        results.extend(rows)
        taken = sum(1 for r in rows if r["source"] == "book")
        song_report.append(dict(stem=stem, taken=taken, total=len(rows), **info))
        if args.dry_run or args.verbose:
            report_song(stem, rows, info, taken, args.verbose)
        elif count % 40 == 0 or count == len(order):
            print(f"  [{count}/{len(order)}]")

    summarize(song_report, results)
    if args.dry_run:
        print("\n(--dry-run: 파일을 쓰지 않았습니다)")
        return

    results.sort(key=lambda r: r["item_name"])
    write_jsonl(out_dir / "lyrics_snapped.jsonl", results)
    print(f"\n저장: {out_dir / 'lyrics_snapped.jsonl'}")


def report_song(stem, rows, info, taken, show_rejections):
    print(f"\n{'=' * 78}\n{stem}")
    verdict = info["reason"] or f"채택 {taken}/{len(rows)}"
    print(f"  창본={info['book']}  밀도={info['density']}  "
          f"배경대비={info['margin']}  {verdict}")
    if info["reason"]:
        return
    for row in rows[:14]:
        mark = "O" if row["source"] == "book" else "."
        print(f"  {mark} {row['text_whisper'][:34]:<34} -> {row['text'][:38]}")
    if show_rejections:
        for line in info["rejections"][:20]:
            print(line)


def summarize(song_report, rows):
    taken = sum(1 for r in rows if r["source"] == "book")
    print(f"\n{'=' * 78}")
    print(f"세그먼트 채택 {taken:,}/{len(rows):,} ({taken / max(1, len(rows)):.1%})")

    reasons = Counter(s["reason"] or "처리됨" for s in song_report)
    print("곡 판정:", "  ".join(f"{k} {v}" for k, v in reasons.most_common()))

    by_book = Counter(s["book"] for s in song_report if not s["reason"])
    print("선택된 창본:", "  ".join(f"{k[:18]} {v}" for k, v in by_book.most_common()))

    # A 바탕 in the filename that disagrees with the 창본 we picked is the
    # cheapest bug detector there is: the filename never fed the matching.
    mismatched = [s for s in song_report if not s["reason"]
                  and (work := re.search(r"춘향가|심청가|수궁가|적벽가|흥보가", s["stem"]))
                  and work.group() not in s["book"]]
    print(f"파일명 바탕과 불일치: {len(mismatched)}곡")
    for song in mismatched[:10]:
        print(f"    {song['stem'][:58]:<58} -> {song['book']}")


if __name__ == "__main__":
    main()
