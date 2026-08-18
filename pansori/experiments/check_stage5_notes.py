#!/usr/bin/env python3
"""Stage 5 sanity check: slice ROSVOT notes for a segment and see whether the
note timeline lines up with the MFA phoneme alignment.

For each sample segment we print, in local (segment) time:
  - the sliced ROSVOT notes (pitch name, start-end)
  - the MFA vowel onsets
and check whether note onsets roughly coincide with vowel onsets -- if the
melody is honest, a new note should start about where a new sung vowel starts.
"""

import json
from pathlib import Path

import librosa
import pretty_midi
from praatio import textgrid as tgio

PANSORI = Path(__file__).resolve().parent.parent
DATA = PANSORI / "data" / "pilot_6s"
MIDI_DIR = Path("/home/sangheon/Desktop/Pansori_Data/rosvot_midi")
ALIGNED = PANSORI / "alignment" / "aligned_6s"

SILENCE = {"", "sil", "sp", "spn", "<eps>"}
# Coda vowels, to spot which aligned phones are vowels (note-bearing).
VOWELS_IPA = {"i", "iː", "e", "eː", "ɛ", "ɛː", "a", "ɐ", "ʌ", "ʌː",
              "o", "oː", "u", "uː", "ɨ", "ɨː"}


def slice_notes(stem: str, start_sec: float, end_sec: float):
    """Notes overlapping [start,end], clipped and shifted to local time."""
    midi = pretty_midi.PrettyMIDI(str(MIDI_DIR / f"{stem}_vocal.mid"))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            # Keep any note that overlaps the window at all.
            if note.end <= start_sec or note.start >= end_sec:
                continue
            local_start = max(note.start, start_sec) - start_sec
            local_end = min(note.end, end_sec) - start_sec
            notes.append((note.pitch, local_start, local_end))
    notes.sort(key=lambda x: x[1])
    return notes


def vowel_onsets(item_name: str):
    """Local-time onsets of vowel phones from the MFA alignment."""
    tg_path = ALIGNED / f"{item_name}.TextGrid"
    if not tg_path.exists():
        return None
    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    onsets = []
    for entry in grid.getTier("phones").entries:
        label = entry.label.strip()
        if label.lower() in SILENCE:
            continue
        # strip length/palatalization diacritics for the vowel test
        base = label.replace("ː", "").replace("ʲ", "").replace("ʷ", "")
        if base in {v.replace("ː", "") for v in VOWELS_IPA}:
            onsets.append((label, entry.start))
    return onsets


def main() -> None:
    segments = {
        json.loads(line)["item_name"]: json.loads(line)
        for line in open(DATA / "segments.jsonl")
    }
    lyrics = {
        r["item_name"]: r["text"]
        for r in (json.loads(l) for l in open(DATA / "lyrics.jsonl"))
    }

    samples = ["01ed19bb_0007", "039d6b6e_0000", "02b8fd02_0011"]

    for item_name in samples:
        seg = segments[item_name]
        notes = slice_notes(seg["song_stem"], seg["song_start"], seg["song_end"])
        onsets = vowel_onsets(item_name)

        print("=" * 64)
        print(f"{item_name}  ({seg['duration']:.2f}s, {seg['mode']})  "
              f"\"{lyrics.get(item_name, '')[:30]}\"")
        print(f"  ROSVOT 음표 {len(notes)}개:")
        for pitch, start, end in notes:
            name = librosa.midi_to_note(pitch)
            bar = "#" * max(1, int((end - start) * 30))
            print(f"    {start:5.2f}-{end:5.2f}s  {name:>4s}({pitch})  {bar}")

        if onsets is None:
            print("  MFA 정렬 없음 (지속음 세그먼트 — 정규화본 참조)")
        else:
            print(f"  MFA 모음 시작점 {len(onsets)}개:")
            print("    " + "  ".join(f"{lab}@{t:.2f}" for lab, t in onsets))

        # Pitch range sanity: pansori spans roughly A2..A5.
        if notes:
            pitches = [p for p, _, _ in notes]
            print(f"  음역: {librosa.midi_to_note(min(pitches))}~"
                  f"{librosa.midi_to_note(max(pitches))}")


if __name__ == "__main__":
    main()
