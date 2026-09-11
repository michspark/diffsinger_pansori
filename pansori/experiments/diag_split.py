"""Diagnose why 01_segment.py pins utterances at the 15s cap.

Compares two candidate split signals on the same sung spans:
  (a) energy-based silence  -> librosa.effects.split(top_db=...)
  (b) MIDI note gaps        -> silence between consecutive ROSVOT notes
"""

import json
import re
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

DATA_DIR = Path("/home/sangheon/Desktop/Pansori_Data")
SAMPLE_RATE = 24000
EXCLUDED_LABELS = {"아니리", "창조"}
PILOT_UUIDS = ["01ed19bb", "02b8fd02", "037e72a6", "039d6b6e", "03abdd70"]


def load_sung_spans():
    """{uuid: [(start, end)]} for the pilot songs only."""
    tasks = json.load(open(DATA_DIR / "labels_2.json", encoding="utf-8"))
    spans = {}
    for task in tasks:
        match = re.match(r"^([0-9a-f]{8})", task.get("file_upload") or "")
        if not match or match.group(1) not in PILOT_UUIDS:
            continue
        for annotation in task.get("annotations") or []:
            for result in annotation.get("result") or []:
                value = result.get("value") or {}
                labels = value.get("labels") or []
                if not labels or labels[0] in EXCLUDED_LABELS:
                    continue
                spans.setdefault(match.group(1), []).append(
                    (float(value["start"]), float(value["end"]))
                )
    return spans


def main() -> None:
    spans_by_uuid = load_sung_spans()

    audio_by_uuid = {}
    for wav_path in sorted((DATA_DIR / "Audio").glob("*.wav")):
        match = re.match(r"^([0-9a-f]{8})", wav_path.stem)
        if match and match.group(1) in PILOT_UUIDS:
            audio_by_uuid[match.group(1)] = wav_path

    # Silence thresholds to compare.
    top_db_candidates = [25, 30, 35, 40, 45]

    # Accumulators: run-length stats per threshold.
    run_lengths = {db: [] for db in top_db_candidates}
    gap_lengths = {db: [] for db in top_db_candidates}
    midi_gap_lengths = []
    midi_run_lengths = []

    for song_uuid, wav_path in audio_by_uuid.items():
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

        midi_path = DATA_DIR / "rosvot_midi" / f"{wav_path.stem}_vocal.mid"
        notes = []
        if midi_path.exists():
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            notes = sorted(
                (note.start, note.end)
                for instrument in midi.instruments
                for note in instrument.notes
            )
        notes = np.asarray(notes, dtype=float).reshape(-1, 2)

        for span_start, span_end in spans_by_uuid[song_uuid]:
            span_audio = audio[
                int(span_start * SAMPLE_RATE) : int(span_end * SAMPLE_RATE)
            ]
            if len(span_audio) == 0:
                continue

            # (a) energy-based silence at each threshold
            for top_db in top_db_candidates:
                nonsilent = librosa.effects.split(span_audio, top_db=top_db)
                if len(nonsilent) == 0:
                    continue
                lengths = (nonsilent[:, 1] - nonsilent[:, 0]) / SAMPLE_RATE
                run_lengths[top_db].extend(lengths.tolist())
                if len(nonsilent) > 1:
                    gaps = (nonsilent[1:, 0] - nonsilent[:-1, 1]) / SAMPLE_RATE
                    gap_lengths[top_db].extend(gaps.tolist())

            # (b) MIDI note gaps inside this span
            inside = notes[(notes[:, 0] >= span_start) & (notes[:, 1] <= span_end)]
            if len(inside) > 1:
                gaps = inside[1:, 0] - inside[:-1, 1]
                midi_gap_lengths.extend(gaps[gaps > 0].tolist())

                # Contiguous note runs, cut wherever a gap exceeds 0.15s
                breaks = np.where(gaps > 0.15)[0]
                boundaries = [0, *(breaks + 1).tolist(), len(inside)]
                for i in range(len(boundaries) - 1):
                    chunk = inside[boundaries[i] : boundaries[i + 1]]
                    if len(chunk):
                        midi_run_lengths.append(chunk[-1, 1] - chunk[0, 0])

    def describe(name, values):
        if not values:
            print(f"{name:28s} (none)")
            return
        arr = np.asarray(values)
        print(
            f"{name:28s} n={len(arr):5d}  median={np.median(arr):6.2f}  "
            f"p75={np.percentile(arr,75):6.2f}  p90={np.percentile(arr,90):6.2f}  "
            f"max={arr.max():7.2f}   >15s={int((arr>15).sum()):4d}"
        )

    print("=== (a) energy-based silence detection: length of continuous phonation ===")
    for top_db in top_db_candidates:
        describe(f"top_db={top_db} run length", run_lengths[top_db])

    print("\n=== (a) silence gap lengths ===")
    for top_db in top_db_candidates:
        describe(f"top_db={top_db} gap length", gap_lengths[top_db])

    print("\n=== (b) MIDI note gaps / note runs ===")
    describe("MIDI gap length", midi_gap_lengths)
    describe("MIDI run (gap>0.15s cut)", midi_run_lengths)

    if midi_gap_lengths:
        arr = np.asarray(midi_gap_lengths)
        print("\nMIDI gap quantiles:")
        for q in [50, 75, 90, 95, 99]:
            print(f"  p{q}: {np.percentile(arr, q):.3f}s")


if __name__ == "__main__":
    main()
