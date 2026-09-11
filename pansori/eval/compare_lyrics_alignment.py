#!/usr/bin/env python3
"""Does snapping the lyrics to the changbon actually improve MFA's alignment?

Prettier text proves nothing. The lyrics exist to be MFA's input, so the test
is MFA's own failure signature, the one pansori/docs/SEGMENTATION_VERDICT.md §2 established:
when the aligner gives up it pins phones to its 0.03s minimum and writes the
rest of the clip off as silence. Both numbers should fall if the words are now
the words that were sung.

The same clips are aligned twice -- once with Whisper's text, once with the
snapped text -- so the comparison is paired and nothing but the .lab differs.

data/{set}/wavs is gone once the pipeline has run, so the sampled clips are
re-cut from the source audio the way 01_segment.py cut them (44.1kHz mono).

The aligner is whatever preprocess/align.yaml says (--aligner overrides it).
This is a ruler for the lyrics axis, so the aligner has to be held fixed -- both
arms pass through the same one.

Run in the `aligner` env (needs praatio, librosa, soundfile, and the mfa CLI).

    python pansori/eval/compare_lyrics_alignment.py --set full --n 200
    python pansori/eval/compare_lyrics_alignment.py --set full --aligner sofa
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocess"))

from aligners import get_aligner
from pansori_lib import (PANSORI_DATA, TARGET_SAMPLE_RATE, load_align_config,
                         read_jsonl, set_dir)

# The per-phone minimum an aligner pins a phone to when it gives up. A phone
# sitting on it is one the aligner declined to place; pansori/docs/SEGMENTATION_VERDICT.md measured 25.8%
# of phones pinned there on clips under 6s (MFA's floor is 0.03s).
FLOOR_SEC = 0.031
LONG_SILENCE_SEC = 1.0
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def cut_clips(rows, segments, wav_dir):
    """Re-cut the sampled clips from Pansori_Data/Audio, one song load at a time."""
    import librosa
    import soundfile as sf

    wav_dir.mkdir(parents=True, exist_ok=True)
    meta = {r["item_name"]: r for r in segments}
    by_song = {}
    for row in rows:
        by_song.setdefault(meta[row["item_name"]]["song_stem"], []).append(row)

    for index, (stem, song_rows) in enumerate(sorted(by_song.items()), start=1):
        audio, _ = librosa.load(PANSORI_DATA / "Audio" / f"{stem}.wav",
                                sr=TARGET_SAMPLE_RATE, mono=True)
        for row in song_rows:
            record = meta[row["item_name"]]
            start = int(record["song_start"] * TARGET_SAMPLE_RATE)
            end = int(record["song_end"] * TARGET_SAMPLE_RATE)
            sf.write(wav_dir / f"{row['item_name']}.wav",
                     audio[start:end], TARGET_SAMPLE_RATE)
        print(f"  [{index}/{len(by_song)}] {stem[:52]}")


def build_corpus(rows, wav_dir, corpus_dir, field, aligner):
    """wav symlink + .lab holding `field`'s text, for every sampled item.

    The .lab formatting is the aligner's, not ours -- MFA reads eojeol, SOFA reads
    whatever `sofa.lab_unit` says. Same text, the shape each one expects.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        item = row["item_name"]
        link = corpus_dir / f"{item}.wav"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink((wav_dir / f"{item}.wav").resolve(), link)
        (corpus_dir / f"{item}.lab").write_text(aligner.lab_text(row[field]),
                                                encoding="utf-8")


def measure(aligned_dir, durations):
    """item -> the two VERDICT metrics, plus what they are made of."""
    from praatio import textgrid as tgio

    out = {}
    for path in sorted(aligned_dir.glob("*.TextGrid")):
        grid = tgio.openTextgrid(str(path), includeEmptyIntervals=True)
        phones, silences = [], []
        for interval in grid.getTier("phones").entries:
            span = interval.end - interval.start
            if interval.label.strip().lower() in SILENCE_LABELS:
                silences.append(span)
            else:
                phones.append(span)
        if not phones:
            continue
        out[path.stem] = {
            "n_phones": len(phones),
            "floor_pct": 100.0 * sum(s <= FLOOR_SEC for s in phones) / len(phones),
            "long_silence": sum(s for s in silences if s >= LONG_SILENCE_SEC),
            "silence_pct": 100.0 * sum(silences) / durations[path.stem],
        }
    return out


def report(before, after, rows):
    """Paired comparison over the items both runs aligned."""
    shared = sorted(set(before) & set(after))
    if not shared:
        sys.exit("no item succeeded in both alignments.")

    print(f"\n{'=' * 68}")
    print(f"aligned: whisper {len(before)} / snapped {len(after)} "
          f"/ in common {len(shared)} (sample {len(rows)})\n")
    print(f"{'metric':<26}{'whisper':>11}{'snapped':>11}{'change':>12}")

    def mean(source, key):
        return sum(source[i][key] for i in shared) / len(shared)

    for key, label, unit in [
            ("floor_pct", "fraction at the 0.03 s floor", "%"),
            ("long_silence", "1s+ silence (mean)", "s"),
            ("silence_pct", "overall silence fraction", "%"),
            ("n_phones", "phonemes (mean)", "")]:
        a, b = mean(before, key), mean(after, key)
        delta = f"{b - a:+.2f}{unit}"
        if a:
            delta += f" ({(b - a) / a:+.0%})"
        print(f"{label:<26}{a:>11.2f}{b:>11.2f}{delta:>12}")

    better = sum(after[i]["floor_pct"] < before[i]["floor_pct"] for i in shared)
    worse = sum(after[i]["floor_pct"] > before[i]["floor_pct"] for i in shared)
    print(f"\nclips whose floor rate improved {better} / worsened {worse} "
          f"/ unchanged {len(shared) - better - worse}")

    floor_before, floor_after = mean(before, "floor_pct"), mean(after, "floor_pct")
    silence_ok = mean(after, "long_silence") <= mean(before, "long_silence") + 0.05
    passed = floor_before and (floor_before - floor_after) / floor_before >= 0.10
    print(f"\npass line: floor rate down 10% or more AND no increase in 1s+ silence  ->  "
          f"{'PASS' if passed and silence_ok else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="test whether snapped lyrics improve alignment")
    parser.add_argument("--set", default="full")
    parser.add_argument("--lyrics", default="lyrics_snapped.jsonl")
    parser.add_argument("--n", type=int, default=200, help="number of segments to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default="", help="default: data/{set}/lyric_eval")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse already-cut wavs and alignment output; analyse only")
    parser.add_argument("--aligner", default=None, choices=["mfa", "sofa"],
                        help="defaults to the aligner in preprocess/align.yaml")
    args = parser.parse_args()

    config = load_align_config()
    aligner = get_aligner(args.aligner or config["aligner"], config)

    out_dir = set_dir(args.set)
    segments = read_jsonl(out_dir / "segments.jsonl")
    lyrics = read_jsonl(out_dir / args.lyrics)
    durations = {r["item_name"]: r["duration"] for r in segments}

    snapped = [r for r in lyrics if r["source"] == "book"]
    if not snapped:
        sys.exit(f"{args.lyrics} has no segments with source=book.")
    rows = random.Random(args.seed).sample(snapped, min(args.n, len(snapped)))
    print(f"sampling {len(rows)} of {len(snapped):,} corrected segments "
          f"(seed={args.seed}, aligner={aligner.name})")

    work = Path(args.work_dir) if args.work_dir else out_dir / "lyric_eval"
    wav_dir = work / "wavs"

    if not args.reuse:
        print("\nre-cutting clips...")
        cut_clips(rows, segments, wav_dir)
        for field, tag in [("text_whisper", "whisper"), ("text", "snapped")]:
            print(f"\n{aligner.name} aligning ({tag})...")
            build_corpus(rows, wav_dir, work / f"corpus_{tag}", field, aligner)
            aligner.run(work / f"corpus_{tag}", work / f"aligned_{tag}")

    before = measure(work / "aligned_whisper", durations)
    after = measure(work / "aligned_snapped", durations)
    report(before, after, rows)
    json.dump({"before": before, "after": after},
              open(work / "metrics.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nraw metrics: {work / 'metrics.json'}")


if __name__ == "__main__":
    main()
