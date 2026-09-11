#!/usr/bin/env python3
"""Stage 7: validate transcriptions.csv before binarizing.

openvpi's binarizer asserts on malformed labels inside a worker process, which
kills the whole run mid-way rather than skipping the item
(`preprocessing/variance_binarizer.py:173-183`). This stage front-runs every one
of those asserts plus the checks the binarizer does *not* make, so a failure is a
readable report instead of a crashed 40-minute job.

Checked, in the order the binarizer would hit them:
  * len(ph_seq) == len(ph_dur)                    acoustic + variance
  * sum(ph_num) == len(ph_seq)                    variance, predict_dur
  * len(note_seq) == len(note_dur)                variance, predict_pitch
  * at least one non-rest note                    variance, predict_pitch
  * ph_dur >= 0 and note_dur >= 0                 both
  * every phone resolves in the dictionary        PhonemeDictionary.encode KeyError
  * sum(ph_dur) ~= sum(note_dur) ~= wav length    silent frame drift, not asserted
  * no phone rounds to zero frames                wasted token, not asserted

Input : data/{set}/transcriptions.csv (+ segments.jsonl)
Output: PASS/FAIL report on stdout; exit code 1 on FAIL.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from pansori_lib import read_jsonl, set_dir

DICT_PATH = Path(__file__).resolve().parents[2] / "dictionaries" / "korean-coda.txt"
GLOBAL_PHONEMES = {"SP", "AP"}

# openvpi acoustic default: 44.1 kHz / hop 512.
FRAME_SEC = 512 / 44100
DUR_TOLERANCE = 0.05          # seconds of drift allowed against the segment length


def dictionary_phonemes():
    phonemes = set(GLOBAL_PHONEMES)
    with open(DICT_PATH, encoding="utf-8") as f:
        for line in f:
            phonemes.update(line.rstrip("\n").split("\t")[1].split())
    return phonemes


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: validate transcriptions.csv")
    parser.add_argument("--set", default="full")
    parser.add_argument("--csv", default=None,
                        help="validate this file instead of data/{set}/transcriptions.csv")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    csv_path = Path(args.csv) if args.csv else out_dir / "transcriptions.csv"
    legal = dictionary_phonemes()

    seg_path = out_dir / "segments.jsonl"
    durations = ({r["item_name"]: r["duration"] for r in read_jsonl(seg_path)}
                 if seg_path.exists() else {})

    failures = Counter()
    examples = {}
    unknown_phonemes = Counter()
    zero_frame = 0
    total = 0

    def fail(kind, item, detail=""):
        failures[kind] += 1
        examples.setdefault(kind, f"{item}: {detail}")

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            item = row["name"]
            ph_seq = row["ph_seq"].split()
            ph_dur = [float(x) for x in row["ph_dur"].split()]
            ph_num = [int(x) for x in row["ph_num"].split()]
            note_seq = row["note_seq"].split()
            note_dur = [float(x) for x in row["note_dur"].split()]

            if len(ph_seq) != len(ph_dur):
                fail("ph_seq/ph_dur length mismatch", item, f"{len(ph_seq)} vs {len(ph_dur)}")
            if sum(ph_num) != len(ph_seq):
                fail("sum(ph_num) != len(ph_seq)", item, f"{sum(ph_num)} vs {len(ph_seq)}")
            if len(note_seq) != len(note_dur):
                fail("note_seq/note_dur length mismatch", item, f"{len(note_seq)} vs {len(note_dur)}")
            if not any(n != "rest" for n in note_seq):
                fail("all rest", item)
            if any(d < 0 for d in ph_dur):
                fail("negative ph_dur", item)
            if any(d < 0 for d in note_dur):
                fail("negative note_dur", item)

            for phoneme in ph_seq:
                if phoneme not in legal:
                    unknown_phonemes[phoneme] += 1
                    fail("phoneme not in the dictionary", item, phoneme)

            zero_frame += sum(1 for d in ph_dur if round(d / FRAME_SEC) == 0)

            ph_total, note_total = sum(ph_dur), sum(note_dur)
            if abs(ph_total - note_total) > DUR_TOLERANCE:
                fail("sum(ph_dur) != sum(note_dur)", item, f"{ph_total:.3f} vs {note_total:.3f}")
            if item in durations and abs(ph_total - durations[item]) > DUR_TOLERANCE:
                fail("sum(ph_dur) != segment length", item,
                     f"{ph_total:.3f} vs {durations[item]:.3f}")

    print(f"checking: {csv_path}  ({total:,} items)")
    if durations:
        print(f"cross-checked against segment lengths: {seg_path.name}")
    else:
        print("segment-length cross-check: skipped (no segments.jsonl)")

    if zero_frame:
        print(f"⚠ {zero_frame:,} phonemes round to zero frames "
              f"(hop {FRAME_SEC * 1000:.1f} ms) — a wasted token, not fatal")

    if failures:
        print("\nFAIL")
        for kind, count in failures.most_common():
            print(f"  {kind}: {count:,}   e.g. {examples[kind]}")
        if unknown_phonemes:
            print(f"  unregistered phoneme kinds: {dict(unknown_phonemes)}")
        sys.exit(1)

    print("\nPASS — safe to binarize")


if __name__ == "__main__":
    main()
