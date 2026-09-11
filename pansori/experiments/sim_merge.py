"""Compare the current greedy 'pack to cap' merge against a gap-priority merge.

Current : merge runs until the next one would exceed MAX -> pins at MAX.
Proposed: cut at every gap >= GAP_THRESHOLD, then merge only the pieces that
          are shorter than TARGET, so cuts land on real breath pauses.
"""

import json
import re
from pathlib import Path

import librosa
import numpy as np

DATA_DIR = Path("/home/sangheon/Desktop/Pansori_Data")
SAMPLE_RATE = 24000
EXCLUDED_LABELS = {"아니리", "창조"}
PILOT_UUIDS = ["01ed19bb", "02b8fd02", "037e72a6", "039d6b6e", "03abdd70"]

MIN_SEC = 2.0
MAX_SEC = 15.0
SILENCE_TOP_DB = 35


def load_sung_spans():
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


def merge_current(runs):
    """Existing behaviour: pack runs together until the MAX cap."""
    utterances = []
    current_start, current_end = runs[0]
    for next_start, next_end in runs[1:]:
        if next_end - current_start <= MAX_SEC:
            current_end = next_end
        else:
            utterances.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    utterances.append((current_start, current_end))
    return utterances


def merge_gap_priority(runs, gap_threshold, target_sec):
    """Cut at every wide gap first; only merge pieces still under target."""
    # Step 1: hard cut wherever the silence between runs is a real pause.
    groups = [[runs[0]]]
    for previous, current in zip(runs, runs[1:]):
        gap = current[0] - previous[1]
        if gap >= gap_threshold:
            groups.append([current])
        else:
            groups[-1].append(current)
    pieces = [(group[0][0], group[-1][1]) for group in groups]

    # Step 2: merge a piece into the previous one only while the result stays
    # under target -- short fragments get absorbed, long ones stay separate.
    merged = [pieces[0]]
    for piece_start, piece_end in pieces[1:]:
        last_start, last_end = merged[-1]
        last_duration = last_end - last_start
        piece_duration = piece_end - piece_start
        combined = piece_end - last_start
        if (last_duration < MIN_SEC or piece_duration < MIN_SEC) and combined <= target_sec:
            merged[-1] = (last_start, piece_end)
        else:
            merged.append((piece_start, piece_end))
    return merged


def describe(name, values):
    arr = np.asarray(values)
    if arr.size == 0:
        print(f"{name:34s} (none)")
        return
    under_min = int((arr < MIN_SEC).sum())
    over_max = int((arr > MAX_SEC).sum())
    print(
        f"{name:34s} n={arr.size:4d}  median={np.median(arr):5.2f}  "
        f"p90={np.percentile(arr,90):5.2f}  max={arr.max():5.2f}  "
        f"total{arr.sum()/3600:5.2f}h  <2s={under_min:3d}  >15s={over_max:3d}"
    )


def main() -> None:
    spans_by_uuid = load_sung_spans()
    audio_by_uuid = {}
    for wav_path in sorted((DATA_DIR / "Audio").glob("*.wav")):
        match = re.match(r"^([0-9a-f]{8})", wav_path.stem)
        if match and match.group(1) in PILOT_UUIDS:
            audio_by_uuid[match.group(1)] = wav_path

    configs = [
        ("current (pack to 15s)", None, None),
        ("gap>=0.3s, target 10s", 0.3, 10.0),
        ("gap>=0.4s, target 10s", 0.4, 10.0),
        ("gap>=0.5s, target 10s", 0.5, 10.0),
        ("gap>=0.4s, target 8s", 0.4, 8.0),
        ("gap>=0.6s, target 12s", 0.6, 12.0),
    ]
    results = {name: [] for name, _, _ in configs}

    for song_uuid, wav_path in audio_by_uuid.items():
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        for span_start, span_end in spans_by_uuid[song_uuid]:
            span_audio = audio[
                int(span_start * SAMPLE_RATE) : int(span_end * SAMPLE_RATE)
            ]
            if len(span_audio) == 0:
                continue
            nonsilent = librosa.effects.split(span_audio, top_db=SILENCE_TOP_DB)
            if len(nonsilent) == 0:
                continue
            runs = [
                (
                    span_start + run_start / SAMPLE_RATE,
                    span_start + run_end / SAMPLE_RATE,
                )
                for run_start, run_end in nonsilent
            ]
            for name, gap_threshold, target_sec in configs:
                if gap_threshold is None:
                    utterances = merge_current(runs)
                else:
                    utterances = merge_gap_priority(runs, gap_threshold, target_sec)
                results[name].extend(
                    end - start
                    for start, end in utterances
                    if end - start >= MIN_SEC
                )

    print("=== merge strategies compared (5 pilot songs, 0.42h of labelled sung spans) ===\n")
    for name, _, _ in configs:
        describe(name, results[name])

    print("\n=== length histogram ===")
    bins = [2, 4, 6, 8, 10, 12, 15]
    header = "".join(f"{bins[i]}-{bins[i+1]}s".rjust(9) for i in range(len(bins) - 1))
    print(f"{'':34s}{header}")
    for name, _, _ in configs:
        counts, _ = np.histogram(results[name], bins=bins)
        row = "".join(str(c).rjust(9) for c in counts)
        print(f"{name:34s}{row}")


if __name__ == "__main__":
    main()
