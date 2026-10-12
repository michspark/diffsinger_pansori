#!/usr/bin/env python3
"""Appendix A figure: one utterance as Whisper heard it, as the libretto span 03b
snapped it to, and as 03c's hybrid text rewrote it.

Reads the two lyric files rather than transcribing them, so the figure follows the data:

    conda run -n pansori python pansori/paper/figs/lyrics_example.py [--item ID]
    → pansori/paper/figs/lyrics_example.pdf
"""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "full"

BLUE = "#2f4f7f"      # Fig. 1's pipeline blue
ORANGE = "#c8622a"    # Fig. 1's singing-model orange
GREY = "#9a9a9a"

# Neither is on matplotlib's default search path. A .ttc registers under its first face
# (JP), whose Hangul glyphs are the KR face's; Liberation Serif stands in for Times.
FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
         "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf"]


def read(path, item):
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["item_name"] == item:
                return row
    raise SystemExit(f"{item} not in {path}")


def syllables(text):
    return [c for c in text if "가" <= c <= "힣"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", default="380ef561_0048")
    ap.add_argument("--out", default=str(HERE / "lyrics_example.pdf"))
    args = ap.parse_args()

    snap = read(DATA / "lyrics_snapped.jsonl", args.item)
    hyb = read(DATA / "lyrics_hybrid.jsonl", args.item)
    whisper = syllables(hyb["text_whisper"])
    book = syllables(snap["text"])
    hybrid = syllables(hyb["text"])
    assert len(hybrid) == len(whisper), "the hybrid text never changes a syllable count"
    rewritten = {e["pos"] for e in hyb["edits"]}

    # Place the sung syllables under the stretch of the libretto span they pair with.
    block = max(SequenceMatcher(None, book, hybrid, autojunk=False).get_matching_blocks(),
                key=lambda m: m.size)
    offset = block.a - block.b
    sung = range(offset, offset + len(hybrid))

    for path in FONTS:
        font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.serif": ["Liberation Serif"],
                         "pdf.fonttype": 3})
    n = max(len(book), offset + len(hybrid))
    fig, ax = plt.subplots(figsize=(6.0, 1.45))
    w, h, gap = 0.86, 0.72, 1.0
    yh = -0.25    # the hybrid row sits lower, leaving room to bracket the unsung span

    def box(x, y, s, edge, face, ls="-", color="black", weight="normal"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    lw=0.8, ls=ls, ec=edge, fc=face))
        ax.text(x, y - 0.02, s, ha="center", va="center", fontsize=8.5, color=color,
                weight=weight)

    rows = [("Whisper", 2), (f"Libretto span ({len(book)})", 1), ("Hybrid", yh)]
    for label, y in rows:
        ax.text(-0.9, y if y == yh else y * gap, label, ha="right", va="center", fontsize=8,
                family="serif", style="italic")

    for i, s in enumerate(whisper):
        x = offset + i
        box(x, 2 * gap, s, ORANGE if i in rewritten else BLUE, "white",
            color=ORANGE if i in rewritten else "black")
    for i, s in enumerate(book):
        if i in sung:
            box(i, gap, s, BLUE, "#e8edf5")
        else:
            box(i, gap, s, GREY, "#f2f2f2", ls=(0, (2, 1.5)), color=GREY)
    for i, s in enumerate(hybrid):
        x = offset + i
        if i in rewritten:
            box(x, yh, s, ORANGE, "#f8e3d6", color=ORANGE, weight="bold")
        else:
            box(x, yh, s, BLUE, "white")

    unsung = len(book) - len(hybrid)
    yb = gap - h / 2 - 0.12
    ax.plot([-w / 2, -w / 2, offset - 1 + w / 2, offset - 1 + w / 2],
            [yb + 0.08, yb, yb, yb + 0.08], color=GREY, lw=0.6)
    ax.text((offset - 1) / 2, yb - 0.08, f"{unsung} syllables never sung", ha="center",
            va="top", fontsize=7.5, family="serif", style="italic", color=GREY)

    ax.set_xlim(-0.6 - 3.6, n - 0.4)
    ax.set_ylim(yh - h / 2 - 0.05, 2 * gap + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"{args.item}: whisper {len(whisper)}, libretto span {len(book)}, "
          f"hybrid {len(hybrid)} ({len(rewritten)} rewritten) → {args.out}")


if __name__ == "__main__":
    main()
