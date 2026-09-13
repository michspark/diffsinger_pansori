#!/usr/bin/env python3
"""Stage 03c -- rewrite Whisper's syllables from the changbon, keeping their count.

Reads what 03b decided (`lyrics_snapped.jsonl`: which changbon, roughly where)
and redoes the per-segment step under a stricter rule, in `hybrid.py`: Whisper's
syllables are kept one for one, and a differing stretch is rewritten only when
the changbon has the same number of syllables there and exact matches pin it in
place. The syllable count of every segment is therefore identical to Whisper's,
which 03b could not promise -- see the numbers in hybrid.py's docstring.

Rows 03b left as Whisper (no `book_ref`) pass through unchanged; this stage can
only refine a placement, not invent one.

Output schema is 03b's plus `n_replaced`, `coverage`, `edits`. `match_score` is
carried over from 03b so downstream tools can still stratify by how far the two
sources were apart; `source` is "hybrid" when at least one syllable changed.

Run inside the `pansori` env.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

from hybrid import hybridize
from lyricbook_lib import CORPUS_PATH, load_books
from pansori_lib import PANSORI_DIR, read_jsonl, write_jsonl

MIN_MARGIN = 3   # widen 03b's span by at least this many syllables each side


def hybrid_row(row, book, margin_floor=MIN_MARGIN):
    """One 03b row -> one 03c row. Never changes the syllable count."""
    out = dict(row, source="whisper", text=row["text_whisper"],
               n_replaced=0, coverage=None, edits=[])
    reference = row.get("book_ref")
    if not reference:
        return out

    first, last = reference["span"]
    # 03b's edges are the unreliable part, so look well past them: a whole
    # segment's worth of syllables each way lets the alignment find where
    # Whisper's text really sits, wherever the song-level diff put the cut.
    n_whisper = sum("가" <= c <= "힣" for c in row["text_whisper"])
    margin = max(margin_floor, n_whisper)
    start = max(0, first - margin)
    end = min(len(book.syllables), last + margin)

    result = hybridize(row["text_whisper"], book.syllables[start:end],
                       book.space_before[start:end])
    out.update(n_replaced=result.n_replaced, coverage=round(result.coverage, 3),
               edits=result.edits)
    if result.span:
        span = [start + result.span[0], start + result.span[1]]
        out["book_ref"] = {"book": book.name, "block": book.block_of[span[0]], "span": span}
    if result.n_replaced:
        out.update(text=result.text, source="hybrid")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="full")
    parser.add_argument("--lyrics", default="lyrics_snapped.jsonl", help="03b's output")
    parser.add_argument("--out", default="lyrics_hybrid.jsonl")
    parser.add_argument("--data-root", default="", help="override pansori/data (e.g. from a worktree)")
    parser.add_argument("--corpus", default="", help="override lyricbook/corpus.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PANSORI_DIR / "data"
    set_dir = data_root / args.set
    books = {b.name: b for b in load_books(Path(args.corpus) if args.corpus else CORPUS_PATH)}
    rows = read_jsonl(set_dir / args.lyrics)

    out = []
    for row in rows:
        reference = row.get("book_ref")
        book = books[reference["book"]] if reference else None
        out.append(hybrid_row(row, book))

    report(rows, out)
    if args.dry_run:
        return
    write_jsonl(set_dir / args.out, out)
    print(f"\nwrote {set_dir / args.out}")


def report(before, after):
    placed = [r for r in before if r.get("book_ref")]
    hybrid = [r for r in after if r["source"] == "hybrid"]
    replaced = sum(r["n_replaced"] for r in hybrid)
    whisper_syllables = sum(sum("가" <= c <= "힣" for c in r["text_whisper"])
                            for r in placed)
    print(f"{len(before):,} rows | 03b placed {len(placed):,} | 03c rewrote {len(hybrid):,} "
          f"({100 * len(hybrid) / max(1, len(placed)):.1f}% of placed)")
    print(f"syllables rewritten: {replaced:,} of {whisper_syllables:,} in placed rows "
          f"({100 * replaced / max(1, whisper_syllables):.1f}%)")

    # The invariant 03b could not keep. Checked on every row, not sampled.
    for row in after:
        assert (sum("가" <= c <= "힣" for c in row["text"])
                == sum("가" <= c <= "힣" for c in row["text_whisper"])), row["item_name"]
    print("syllable count == Whisper's on every row: yes")

    sizes = Counter(min(r["n_replaced"], 5) for r in hybrid)
    print("rewritten syllables per row: " + "  ".join(
        f"{k if k < 5 else '5+'}:{sizes[k]:,}" for k in sorted(sizes)))
    cov = [r["coverage"] for r in after if r["coverage"] is not None]
    if cov:
        print(f"coverage (whisper syllables sitting on a book syllable): "
              f"median {statistics.median(cov):.2f}")

    changed_before = sum(1 for r in placed if r["source"] == "book")
    print(f"for comparison, 03b had changed the text of {changed_before:,} rows")


if __name__ == "__main__":
    sys.exit(main())
