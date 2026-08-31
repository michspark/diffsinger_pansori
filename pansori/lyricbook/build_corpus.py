#!/usr/bin/env python3
"""Turn the 15 hand-collected 창본 into one block-per-line corpus.jsonl.

The books come from three sources and mark 장단 three different ways:

    <중중모리>      angle brackets on their own line
    중중모리        bare, on its own line          (춘향가_김소희)
    【중중몰이】     lenticular brackets            (춘향가_성우향)

Nine of them were scraped off the 한국판소리보존회 site and carry ~120 lines of
site navigation before the text starts; everything before the first 장단 line
is dropped.

Blocks tagged 아니리 / 창조 / 무장단 (spoken, not sung) are kept but marked
is_sung=False. Stage 1 already drops those spans from the audio
(01_segment.py EXCLUDED_LABELS), so 03b matches against sung blocks only.

The 15 files are only 6 distinct 창본: within each 바탕 the copies agree to
0.998-0.9997 (심청가 and 심청가_성창순 are byte-identical), the sole exception
being 춘향가_성우향_김세종제, which shares 0.37 with the other 춘향가. Near
duplicates are folded to one representative -- it cuts 03b's matching cost 2.5x
and makes "pick the best-scoring 창본" mean something instead of choosing at
random between three copies of one text.

Output: lyricbook/corpus.jsonl   {book, work, block, jangdan, is_sung, text,
                                  clean, aliases}
"""

import json
import re
import sys
from pathlib import Path

LYRICS_DIR = Path("/home/sangheon/Desktop/Pansori_Data/Lyrics")
OUT_PATH = Path(__file__).resolve().parent / "corpus.jsonl"

# The five 바탕. Every filename contains exactly one of them.
WORKS = ("춘향가", "심청가", "수궁가", "적벽가", "흥보가")

# Spoken (not sung) 장단 labels. Same set 01_segment.py excludes from the audio,
# plus the 무장단 ("no rhythm") variants the books use for the same thing.
SPOKEN = {"아니리", "창조", "무장단", "무장단 창조"}

# A 장단 line names a rhythm: it ends in 모리/몰이/머리, or is one of these.
BARE_JANGDAN = SPOKEN | {"진양", "진양조", "세마치"}


def parse_jangdan(line: str):
    """Return the normalized 장단 name if this line is a section header, else None.

    Strips the three bracket styles and any '=조' suffix ('중머리=평계면').
    Length and punctuation guards keep prose out -- one book has a 24-char
    narration line ending in '늦은 중중 머리로 나가것다.' that would otherwise pass.
    """
    s = line.strip()
    if not s or len(s) > 20 or re.search(r'[.,"?!]', s):
        return None
    s = re.sub(r'^[<【\[(]\s*|\s*[>】\])]$', "", s).strip()
    s = s.split("=")[0].strip()
    if not s or len(s) > 14:
        return None
    if s in BARE_JANGDAN or re.search(r"(모리|몰이|머리)$", s):
        return s
    return None


def is_sung(jangdan: str) -> bool:
    """아니리/창조/무장단 are spoken; everything with a rhythm is sung."""
    return jangdan not in SPOKEN


def clean_text(raw: str) -> str:
    """Book prose -> the Hangul-and-spaces form MFA's .lab files take.

    Parenthesised glosses go first: they are either hanja readings
    ('곽씨부인(郭氏婦人)') or plain-Korean glosses ('낭기(나무)'), and neither is
    sung. They are deleted rather than spaced, because the gloss sits between a
    word and its particle -- '자로(子路)는' has to come back as '자로는', not
    '자로 는'. Everything that is not Hangul then becomes a space.
    """
    text = re.sub(r"\([^)]*\)", "", raw)
    text = re.sub(r"[^가-힣]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def work_of(stem: str) -> str:
    for work in WORKS:
        if work in stem:
            return work
    raise ValueError(f"바탕을 알 수 없는 파일: {stem}")


def parse_book(path: Path):
    """One .txt -> list of block dicts. Header noise before the first 장단 is dropped."""
    stem = path.stem
    work = work_of(stem)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    blocks = []
    current, buffer = None, []

    def flush():
        if current is None:
            return
        text = clean_text(" ".join(buffer))
        if not text:
            return
        blocks.append({
            "book": stem,
            "work": work,
            "block": len(blocks),
            "jangdan": current,
            "is_sung": is_sung(current),
            "text": text,
            "clean": text.replace(" ", ""),
        })

    for line in lines:
        jangdan = parse_jangdan(line)
        if jangdan is not None:
            flush()
            current, buffer = jangdan, []
        elif current is not None:      # before the first 장단 = site nav, skip
            buffer.append(line)
    flush()
    return blocks


DUPLICATE_RATIO = 0.99   # above this two 창본 are the same document
NGRAM = 5


def _grams(text: str) -> set:
    return {text[i:i + NGRAM] for i in range(len(text) - NGRAM + 1)}


def dedupe(books: dict) -> list:
    """Group near-identical 창본; return [(keeper, [alias, ...]), ...].

    The cheap n-gram Jaccard is only a prefilter -- SequenceMatcher.ratio()
    over 25k characters is slow enough that running it on all 105 pairs is
    wasteful, and it is order-sensitive where the Jaccard is not, so the two
    together are both fast and correct. The longest text wins, on the theory
    that a copy missing lines is the damaged one.
    """
    import difflib

    order = sorted(books, key=lambda b: (-len(books[b]), b))
    grams = {b: _grams(books[b]) for b in order}
    keepers, alias_of = [], {}

    for book in order:
        for keeper in keepers:
            longer = max(len(books[book]), len(books[keeper]))
            if abs(len(books[book]) - len(books[keeper])) / longer > 0.05:
                continue
            overlap = grams[book] & grams[keeper]
            union = grams[book] | grams[keeper]
            if not union or len(overlap) / len(union) < 0.8:
                continue
            ratio = difflib.SequenceMatcher(
                None, books[keeper], books[book], autojunk=False).ratio()
            if ratio >= DUPLICATE_RATIO:
                alias_of.setdefault(keeper, []).append(book)
                break
        else:
            keepers.append(book)

    return [(k, alias_of.get(k, [])) for k in keepers]


def main() -> None:
    paths = sorted(LYRICS_DIR.glob("*.txt"))
    if not paths:
        sys.exit(f"가사집을 찾을 수 없음: {LYRICS_DIR}")

    parsed, sung_text = {}, {}
    for path in paths:
        blocks = parse_book(path)
        parsed[path.stem] = blocks
        sung_text[path.stem] = "".join(b["clean"] for b in blocks if b["is_sung"])

    kept = dedupe(sung_text)
    print(f"가사집 {len(paths)}개 -> 창본 {len(kept)}개\n")
    print(f"{'창본':<30} {'바탕':<6} {'블록':>5} {'창블록':>6} {'창음절':>7}  중복")

    all_blocks = []
    for book, aliases in kept:
        blocks = parsed[book]
        for alias in aliases:
            for b in blocks:
                b["aliases"] = aliases
        sung = [b for b in blocks if b["is_sung"]]
        print(f"{book[:29]:<30} {blocks[0]['work']:<6} {len(blocks):>5} "
              f"{len(sung):>6} {len(sung_text[book]):>7}  "
              f"{', '.join(a[:16] for a in aliases)}")
        all_blocks.extend(blocks)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for block in all_blocks:
            block.setdefault("aliases", [])
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

    sung = [b for b in all_blocks if b["is_sung"]]
    print(f"\n저장: {OUT_PATH}")
    print(f"  블록 {len(all_blocks)}개 중 창 {len(sung)}개 "
          f"({len(sung) / len(all_blocks):.0%}), 창 음절 "
          f"{sum(len(b['clean']) for b in sung):,}자")


if __name__ == "__main__":
    main()
