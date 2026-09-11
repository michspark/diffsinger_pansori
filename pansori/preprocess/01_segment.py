#!/usr/bin/env python3
"""
Stage 1: Extract sung (chang) segments from pansori songs and split them into
DiffSinger-sized utterances.

Input:
  - Pansori_Data/labels_2.json   (human-annotated modal / aniri spans, seconds)
  - Pansori_Data/Audio/*.wav     (demucs-separated vocals, 44.1kHz stereo)
  - Pansori_Data/rosvot_midi/*_vocal.mid  (note timeline; only consulted when a
                                           run has no internal silence to cut on)

Output:
  - {out_dir}/wavs/{item_name}.wav   (24kHz mono)
  - {out_dir}/segments.jsonl         (one JSON object per utterance)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from tqdm import tqdm

from pansori_lib import TARGET_SAMPLE_RATE, extract_uuid

# Vocal-type labels that are NOT sung; these are dropped entirely.
EXCLUDED_LABELS = {"아니리", "창조"}

# Utterance length bounds in seconds.
# The 6s cap comes from the MFA alignment check (pansori/docs/SEGMENTATION_VERDICT.md):
# korean_mfa's speech-trained self-loop penalty accumulates on long sustained
# notes, and above ~6s the aligner collapses (phones pinned to the 0.03s floor,
# the rest swallowed as silence). Utterances at/under 6s aligned cleanly.
MIN_UTTERANCE_SEC: float = 2.0
MAX_UTTERANCE_SEC: float = 6.0

# Preferred utterance length. Fragments are merged only while the result stays
# under this; it is a soft target, unlike MAX_UTTERANCE_SEC.
TARGET_UTTERANCE_SEC: float = 4.5

# A pause this long between two non-silent runs is treated as a real breath
# boundary and always becomes a cut. Measured gap distribution on the pilot
# songs: median 0.13s, p90 0.66s -- so 0.4s separates breaths from the short
# pauses inside sigimsae ornamentation.
SILENCE_GAP_THRESHOLD_SEC: float = 0.4

# Silence detection threshold (dB below peak). Higher = more aggressive
# silence detection. 35 dB is a conservative starting point for separated vocals.
SILENCE_TOP_DB: int = 35


def load_sung_spans(labels_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Parse labels_2.json into {uuid: [(start_sec, end_sec, mode_label), ...]}.

    Only sung spans are kept; aniri and changjo are filtered out here.
    """
    tasks = json.load(open(labels_path, encoding="utf-8"))
    spans_by_uuid: Dict[str, List[Tuple[float, float, str]]] = {}

    for task in tasks:
        # 'file_upload' holds a clean filename; 'data.audio' is URL-encoded with
        # mixed NFC/NFD Hangul, so we never match on that field.
        song_uuid = extract_uuid(task.get("file_upload") or "")
        if song_uuid is None:
            continue

        for annotation in task.get("annotations") or []:
            for result in annotation.get("result") or []:
                value = result.get("value") or {}
                labels = value.get("labels") or []
                if not labels:
                    continue

                mode_label = labels[0]
                if mode_label in EXCLUDED_LABELS:
                    continue

                start_sec = float(value["start"])
                end_sec = float(value["end"])
                if end_sec - start_sec <= 0:
                    continue

                spans_by_uuid.setdefault(song_uuid, []).append(
                    (start_sec, end_sec, mode_label)
                )

    return spans_by_uuid


def load_note_intervals(midi_path: Path) -> np.ndarray:
    """Load ROSVOT note (start, end) times in seconds as an (N, 2) array."""
    if not midi_path.exists():
        return np.zeros((0, 2), dtype=float)

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    intervals = [
        (note.start, note.end)
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    intervals.sort()
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def find_largest_note_gap(
    note_intervals: np.ndarray, start_sec: float, end_sec: float
) -> Optional[float]:
    """Find the midpoint of the widest silence between notes inside [start, end].

    This is the fallback split point when a stretch of audio has no detectable
    silence but is still too long. Splitting between notes avoids cutting
    through a sustained note (sigimsae).
    """
    # Keep only notes that lie within the window.
    inside = note_intervals[
        (note_intervals[:, 0] >= start_sec) & (note_intervals[:, 1] <= end_sec)
    ]
    if len(inside) < 2:
        return None

    # Gap between note i's end and note i+1's start.
    gap_starts = inside[:-1, 1]
    gap_ends = inside[1:, 0]
    gap_widths = gap_ends - gap_starts

    largest_index = int(np.argmax(gap_widths))
    if gap_widths[largest_index] <= 0:
        return None

    # Split at the middle of the widest gap.
    return float((gap_starts[largest_index] + gap_ends[largest_index]) / 2.0)


def find_best_internal_gap(group: List[Tuple[float, float]]) -> Optional[int]:
    """Index i meaning 'cut between run i and run i+1', or None if no cut fits.

    Each candidate gap is scored by its width, discounted by how lopsided the
    resulting two halves would be. Without the balance term the widest pause
    often sits right next to an edge, which would shave off a 2s sliver and
    leave the rest still over the cap.
    """
    group_start = group[0][0]
    total_duration = group[-1][1] - group_start

    best_index: Optional[int] = None
    best_score = -1.0

    for index in range(len(group) - 1):
        gap_width = group[index + 1][0] - group[index][1]

        left_duration = group[index][1] - group_start
        right_duration = total_duration - left_duration

        # Never create a piece that is doomed to be dropped as too short.
        if left_duration < MIN_UTTERANCE_SEC or right_duration < MIN_UTTERANCE_SEC:
            continue

        # 1.0 for a perfectly even split, 0.0 at either extreme.
        balance = 1.0 - abs(left_duration / total_duration - 0.5) * 2.0
        score = gap_width * (0.5 + 0.5 * balance)

        if score > best_score:
            best_index = index
            best_score = score

    return best_index


def split_long_group(
    group: List[Tuple[float, float]],
    note_intervals: np.ndarray,
) -> List[Tuple[float, float]]:
    """Recursively split a group of runs that is still longer than the cap."""
    group_start = group[0][0]
    group_end = group[-1][1]

    if group_end - group_start <= MAX_UTTERANCE_SEC:
        return [(group_start, group_end)]

    # Case 1: the group holds several runs, so there is an internal silence to
    # cut on. This is the common case and keeps the cut on a real pause.
    if len(group) >= 2:
        best_index = find_best_internal_gap(group)
        if best_index is not None:
            left = split_long_group(group[: best_index + 1], note_intervals)
            right = split_long_group(group[best_index + 1 :], note_intervals)
            return left + right

    # Case 2: one uninterrupted run -- a long melisma with no silence in it.
    # Fall back to the widest ROSVOT note gap so we cut between notes rather
    # than through a sustained one.
    split_point = find_largest_note_gap(note_intervals, group_start, group_end)

    # No usable note gap, or one that would produce a degenerate piece: cut at
    # the midpoint so we do not silently drop long melismatic passages.
    if split_point is None or not (
        group_start + MIN_UTTERANCE_SEC < split_point < group_end - MIN_UTTERANCE_SEC
    ):
        split_point = (group_start + group_end) / 2.0

    left = split_long_group([(group_start, split_point)], note_intervals)
    right = split_long_group([(split_point, group_end)], note_intervals)
    return left + right


def segment_span(
    audio: np.ndarray,
    span_start_sec: float,
    span_end_sec: float,
    note_intervals: np.ndarray,
) -> List[Tuple[float, float]]:
    """Split one sung span into utterances, in absolute song-time seconds.

    Strategy: never cut inside a non-silent run. Cut at every real breath pause
    first, then absorb leftover fragments, then break anything still over the
    cap. Cutting first (rather than packing runs up to the cap) is what keeps
    boundaries on breaths instead of at an arbitrary 15s mark.
    """
    start_sample = int(span_start_sec * TARGET_SAMPLE_RATE)
    end_sample = int(span_end_sec * TARGET_SAMPLE_RATE)
    span_audio = audio[start_sample:end_sample]

    if len(span_audio) == 0:
        return []

    # Non-silent regions as sample offsets relative to the span start.
    nonsilent = librosa.effects.split(span_audio, top_db=SILENCE_TOP_DB)
    if len(nonsilent) == 0:
        return []

    # Convert to absolute song time in seconds.
    runs = [
        (
            span_start_sec + run_start / TARGET_SAMPLE_RATE,
            span_start_sec + run_end / TARGET_SAMPLE_RATE,
        )
        for run_start, run_end in nonsilent
    ]

    # Step 1: cut wherever the pause between two runs is a real breath.
    # Runs separated by less than the threshold stay in the same group, so
    # short pauses inside ornamented phrases never become boundaries.
    groups: List[List[Tuple[float, float]]] = [[runs[0]]]
    for previous_run, current_run in zip(runs, runs[1:]):
        if current_run[0] - previous_run[1] >= SILENCE_GAP_THRESHOLD_SEC:
            groups.append([current_run])
        else:
            groups[-1].append(current_run)

    # Step 2: absorb fragments. A group is merged into the previous one only if
    # one of the two is below the minimum length AND the result stays under the
    # soft target -- so short bits get rescued without inflating good groups.
    merged_groups: List[List[Tuple[float, float]]] = [list(groups[0])]
    for group in groups[1:]:
        last_group = merged_groups[-1]
        last_duration = last_group[-1][1] - last_group[0][0]
        group_duration = group[-1][1] - group[0][0]
        combined_duration = group[-1][1] - last_group[0][0]

        too_short = last_duration < MIN_UTTERANCE_SEC or group_duration < MIN_UTTERANCE_SEC
        if too_short and combined_duration <= TARGET_UTTERANCE_SEC:
            last_group.extend(group)
        else:
            merged_groups.append(list(group))

    # Step 3: break any group that is still over the cap.
    utterances: List[Tuple[float, float]] = []
    for group in merged_groups:
        utterances.extend(split_long_group(group, note_intervals))

    # Drop utterances that are too short to be useful.
    return [
        (utt_start, utt_end)
        for utt_start, utt_end in utterances
        if utt_end - utt_start >= MIN_UTTERANCE_SEC
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: cut sung segments into DiffSinger-sized utterances"
    )
    parser.add_argument("--data_dir", default="/home/sangheon/Desktop/Pansori_Data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--limit_songs",
        type=int,
        default=0,
        help="Process only the first N songs (0 = all). Use for the pilot.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    wav_out_dir = out_dir / "wavs"
    wav_out_dir.mkdir(parents=True, exist_ok=True)

    # Build uuid -> audio file map.
    audio_by_uuid: Dict[str, Path] = {}
    for wav_path in sorted((data_dir / "Audio").glob("*.wav")):
        song_uuid = extract_uuid(wav_path.stem)
        if song_uuid is not None:
            audio_by_uuid[song_uuid] = wav_path

    spans_by_uuid = load_sung_spans(data_dir / "labels_2.json")

    # Only songs present in both sources can be processed.
    usable_uuids = sorted(set(spans_by_uuid) & set(audio_by_uuid))
    if args.limit_songs > 0:
        usable_uuids = usable_uuids[: args.limit_songs]

    print(f"songs with audio: {len(audio_by_uuid)}")
    print(f"songs with sung labels: {len(spans_by_uuid)}")
    print(f"songs to process: {len(usable_uuids)}")

    manifest_path = out_dir / "segments.jsonl"
    total_utterances = 0
    total_seconds = 0.0

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for song_uuid in tqdm(usable_uuids, desc="songs"):
            song_path = audio_by_uuid[song_uuid]

            # Load the whole song once at the target rate, mono.
            audio, _ = librosa.load(song_path, sr=TARGET_SAMPLE_RATE, mono=True)

            # ROSVOT MIDI carries a '_vocal' suffix that Audio/ does not.
            midi_path = data_dir / "rosvot_midi" / f"{song_path.stem}_vocal.mid"
            note_intervals = load_note_intervals(midi_path)

            utterance_index = 0
            for span_start, span_end, mode_label in sorted(spans_by_uuid[song_uuid]):
                for utt_start, utt_end in segment_span(
                    audio, span_start, span_end, note_intervals
                ):
                    item_name = f"{song_uuid}_{utterance_index:04d}"
                    utterance_index += 1

                    start_sample = int(utt_start * TARGET_SAMPLE_RATE)
                    end_sample = int(utt_end * TARGET_SAMPLE_RATE)
                    sf.write(
                        wav_out_dir / f"{item_name}.wav",
                        audio[start_sample:end_sample],
                        TARGET_SAMPLE_RATE,
                    )

                    record = {
                        "item_name": item_name,
                        "uuid": song_uuid,
                        "song_stem": song_path.stem,
                        "song_start": round(utt_start, 4),
                        "song_end": round(utt_end, 4),
                        "duration": round(utt_end - utt_start, 4),
                        "mode": mode_label,
                    }
                    manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_utterances += 1
                    total_seconds += utt_end - utt_start

    print(f"\nutterances written: {total_utterances}")
    print(f"total audio: {total_seconds / 3600:.2f} h")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
