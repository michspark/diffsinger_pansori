"""The changbon as an indexable syllable sequence -- shared by 03b and 03c.

`corpus.jsonl` is the output of pansori/lyricbook/build_corpus.py: one block per
jangdan section of each published libretto. `Book` flattens a changbon's sung
blocks into parallel per-syllable arrays so a segment can be located in it
(03b_snap_lyrics.py, by jamo n-gram) and its syllables read back with the
book's own spacing (both stages).
"""

import unicodedata
from collections import defaultdict

from pansori_lib import PANSORI_DIR, read_jsonl

CORPUS_PATH = PANSORI_DIR / "lyricbook" / "corpus.jsonl"
NGRAM = 5              # jamo n-gram; ~1.8 syllables, short enough to survive errors

class Book:
    """One changbon, flattened to its sung syllables with an index back to blocks."""

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


def load_books(corpus_path=CORPUS_PATH):
    # corpus.jsonl is not tracked -- it is the text of six published changbon. Say so
    # here rather than letting read_jsonl raise a bare FileNotFoundError.
    if not corpus_path.exists():
        raise SystemExit(
            f"no changbon corpus at: {corpus_path}\n"
            f"  put Pansori_Data/Lyrics in place and run this first: "
            f"python pansori/lyricbook/build_corpus.py")

    blocks_by_book = defaultdict(list)
    work_of = {}
    for block in read_jsonl(corpus_path):
        if block["is_sung"]:
            blocks_by_book[block["book"]].append(block)
            work_of[block["book"]] = block["work"]
    return [Book(name, work_of[name], blocks)
            for name, blocks in sorted(blocks_by_book.items())]
