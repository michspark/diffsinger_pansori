#!/usr/bin/env python3
"""Stage 3b: snap Whisper's transcription onto the changbon text -> lyrics_snapped.jsonl.

Whisper cannot hold pansori. On 01ed19bb (Kim So-hee, Chunhyangga, the constables)
it produced '굴러 살인길 나간다 살인길 나간다' where the changbon reads '군로사령이
나간다 사령군로가 나간다'. That matters because the lyrics are not the deliverable -- they are MFA's
input, and a wrong word makes MFA drop the span to silence and pin its phones to
the 0.03s floor (pansori/docs/SEGMENTATION_VERDICT.md §2).

The fix does not need a better transcriber, only the book. segments.jsonl carries
song_stem plus song_start, so a song's segments are already in time order: paste
one song's Whisper text into a single string, lay it over the book, and slide.
Anchor once and the rest follows, because the order cannot cross.

Two passes:
  B. anchor  -- jamo 5-gram density picks the changbon and the region, per song
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
from collections import Counter, defaultdict
from pathlib import Path

from lyricbook_lib import CORPUS_PATH, NGRAM, load_books
from pansori_lib import read_jsonl, set_dir, to_jamo, write_jsonl

# ---- Stage B: anchoring -------------------------------------------------

# NGRAM lives in lyricbook_lib.py: the Book index and the query grams must agree.
WINDOW_FACTOR = 2.0    # search window = this many times the query's jamo length
MIN_QUERY_JAMO = 60    # a song this short cannot be anchored with confidence
MIN_WORK_MARGIN = 1.5  # the best bata must beat the other bata's mean by this factor
# Density cannot be judged against a fixed number, because chance agreement
# grows with the query: the window is a multiple of the query's length, so a
# long song sweeps up common n-grams no matter which book it is held against.
# Measured floor across the other four bata -- books that cannot be the answer:
#
#   query jamo    background     the right bata
#          757   0.037-0.052   0.224   (the constables)
#        1,388   0.067-0.087   0.359   (Cheong goes begging for rice)
#        7,495   0.160-0.186   0.342   (the inspector in disguise)
#
# The floor triples while the signal does not, so an absolute cut of 0.15 waved
# long songs through on noise and failed short ones at 0.148. Every book sees
# the same query, so the other bata measure that song's own noise floor; the
# test is how far the winner stands above it.
#
# 1.5 is where that ratio starts telling the truth. Running all 387 songs with
# the gate open, and scoring the chosen changbon against the bata named in the
# filename -- which never feeds the matching, so it is a held-out answer key:
#
#   background ratio    songs   bata correct   segments snapped
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
# threw away correct jajinmori spans (ratio 0.86 at 6.2 syllables/sec) -- pansori
# really is sung that fast when the jangdan is fast.




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
    """Score every changbon; return (best, density, position, margin, reason).

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
        return best, density, position, margin, "too_short"
    if margin < MIN_WORK_MARGIN:
        return best, density, position, margin, "weak_signal"
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
        rejections.append(f"    rejected {row['item_name']} similarity={ratio:.2f} "
                          f"syllables={syllables} rate={rate:.1f}  candidate: {text[:40]}")
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
    parser = argparse.ArgumentParser(description="Stage 3b: snap lyrics to the changbon")
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

    print(f"loading changbon... ({CORPUS_PATH})")
    books = load_books()
    print(f"  {len(books)} changbon / "
          f"{sum(len(b.syllables) for b in books):,} sung syllables\n")
    print(f"processing {len(order)} songs, {sum(len(songs[s]) for s in order):,} segments\n")

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
        print("\n(--dry-run: nothing was written)")
        return

    results.sort(key=lambda r: r["item_name"])
    write_jsonl(out_dir / "lyrics_snapped.jsonl", results)
    print(f"\nsaved: {out_dir / 'lyrics_snapped.jsonl'}")


def report_song(stem, rows, info, taken, show_rejections):
    print(f"\n{'=' * 78}\n{stem}")
    verdict = info["reason"] or f"adopted {taken}/{len(rows)}"
    print(f"  changbon={info['book']}  density={info['density']}  "
          f"vs-background={info['margin']}  {verdict}")
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
    print(f"segments adopted {taken:,}/{len(rows):,} ({taken / max(1, len(rows)):.1%})")

    reasons = Counter(s["reason"] or "processed" for s in song_report)
    print("per-song verdicts:", "  ".join(f"{k} {v}" for k, v in reasons.most_common()))

    by_book = Counter(s["book"] for s in song_report if not s["reason"])
    print("changbon chosen:", "  ".join(f"{k[:18]} {v}" for k, v in by_book.most_common()))

    # A bata in the filename that disagrees with the changbon we picked is the
    # cheapest bug detector there is: the filename never fed the matching.
    mismatched = [s for s in song_report if not s["reason"]
                  and (work := re.search(r"춘향가|심청가|수궁가|적벽가|흥보가", s["stem"]))
                  and work.group() not in s["book"]]
    print(f"disagreeing with the filename's bata: {len(mismatched)} songs")
    for song in mismatched[:10]:
        print(f"    {song['stem'][:58]:<58} -> {song['book']}")


if __name__ == "__main__":
    main()
