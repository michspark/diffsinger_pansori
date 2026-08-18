#!/usr/bin/env python3
"""Stage 5: slice ROSVOT notes for each segment -> notes.jsonl.

Reuses the whole-song ROSVOT MIDI (no re-run). For every aligned segment we
cut the notes inside its absolute song-time window and shift them to local
time. This is what lets Stage 6 give a held vowel its true length and mark
시김새 with is_slur where several notes ride one vowel (alignment/VERDICT.md).

Input : data/{set}/segments.jsonl + phones.jsonl (which items survived align)
Output: data/{set}/notes.jsonl   {item_name, notes:[[pitch,start,end],...]}

Run in the `aligner` or `base` env (pretty_midi).
"""

import argparse

from pansori_lib import read_jsonl, slice_notes, write_jsonl, set_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: ROSVOT note slicing")
    parser.add_argument("--set", default="pilot_6s")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = {r["item_name"]: r for r in read_jsonl(out_dir / "segments.jsonl")}
    # only items that survived alignment need notes
    aligned_items = [r["item_name"] for r in read_jsonl(out_dir / "phones.jsonl")]

    rows, empty = [], 0
    for item in aligned_items:
        seg = segments[item]
        notes = slice_notes(seg["song_stem"], seg["song_start"], seg["song_end"])
        if not notes:
            empty += 1
        rows.append({"item_name": item,
                     "notes": [[p, round(s, 4), round(e, 4)] for p, s, e in notes]})

    write_jsonl(out_dir / "notes.jsonl", rows)
    print(f"음표 슬라이스 {len(rows)}개 → notes.jsonl (음표 없음 {empty}개)")


if __name__ == "__main__":
    main()
