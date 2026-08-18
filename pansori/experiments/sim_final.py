"""Gap-priority merge + recursive split of the leftover long pieces.

This is the full candidate pipeline: cut at breath pauses, absorb fragments,
then break anything still over the cap at its widest internal pause.
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


def group_runs(runs, gap_threshold):
    """Cut the run list wherever the silence between runs is a real pause."""
    groups = [[runs[0]]]
    for previous, current in zip(runs, runs[1:]):
        if current[0] - previous[1] >= gap_threshold:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def absorb_fragments(groups, target_sec):
    """Merge a group into the previous one while the result stays under target."""
    merged = [list(groups[0])]
    for group in groups[1:]:
        last = merged[-1]
        last_duration = last[-1][1] - last[0][0]
        group_duration = group[-1][1] - group[0][0]
        combined = group[-1][1] - last[0][0]
        if (last_duration < MIN_SEC or group_duration < MIN_SEC) and combined <= target_sec:
            last.extend(group)
        else:
            merged.append(list(group))
    return merged


def split_long_group(group):
    """Recursively halve a too-long group at its widest internal pause."""
    duration = group[-1][1] - group[0][0]
    if duration <= MAX_SEC or len(group) < 2:
        return [(group[0][0], group[-1][1])]

    # Widest gap between consecutive runs inside this group.
    gaps = [group[i + 1][0] - group[i][1] for i in range(len(group) - 1)]

    # Prefer a wide gap near the middle: score = gap width, tie-broken by how
    # close the resulting split is to an even halving.
    best_index, best_score = None, -1.0
    for index, gap in enumerate(gaps):
        left_duration = group[index][1] - group[0][0]
        if left_duration < MIN_SEC or duration - left_duration < MIN_SEC:
            continue
        balance = 1.0 - abs(left_duration / duration - 0.5) * 2.0
        score = gap * (0.5 + 0.5 * balance)
        if score > best_score:
            best_index, best_score = index, score

    if best_index is None:
        return [(group[0][0], group[-1][1])]

    left = split_long_group(group[: best_index + 1])
    right = split_long_group(group[best_index + 1 :])
    return left + right


def describe(name, values):
    arr = np.asarray(values)
    print(
        f"{name:30s} n={arr.size:4d}  median={np.median(arr):5.2f}  "
        f"p90={np.percentile(arr,90):5.2f}  max={arr.max():5.2f}  "
        f"총{arr.sum()/3600:5.2f}h  >15s={int((arr>MAX_SEC).sum()):3d}"
    )


def main() -> None:
    spans_by_uuid = load_sung_spans()
    audio_by_uuid = {}
    for wav_path in sorted((DATA_DIR / "Audio").glob("*.wav")):
        match = re.match(r"^([0-9a-f]{8})", wav_path.stem)
        if match and match.group(1) in PILOT_UUIDS:
            audio_by_uuid[match.group(1)] = wav_path

    configs = [(0.3, 10.0), (0.4, 10.0), (0.4, 8.0), (0.5, 10.0)]
    results = {c: [] for c in configs}

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
            for gap_threshold, target_sec in configs:
                groups = absorb_fragments(
                    group_runs(runs, gap_threshold), target_sec
                )
                for group in groups:
                    for start, end in split_long_group(group):
                        if end - start >= MIN_SEC:
                            results[(gap_threshold, target_sec)].append(end - start)

    print("=== gap 우선 분할 + 긴 구간 재귀 분할 (파일럿 5곡) ===\n")
    for config in configs:
        describe(f"gap>={config[0]}s, target {config[1]}s", results[config])

    print("\n=== 길이 히스토그램 ===")
    bins = [2, 4, 6, 8, 10, 12, 15]
    header = "".join(f"{bins[i]}-{bins[i+1]}s".rjust(9) for i in range(len(bins) - 1))
    print(f"{'':30s}{header}")
    for config in configs:
        counts, _ = np.histogram(results[config], bins=bins)
        print(
            f"{f'gap>={config[0]}s, target {config[1]}s':30s}"
            + "".join(str(c).rjust(9) for c in counts)
        )


if __name__ == "__main__":
    main()
