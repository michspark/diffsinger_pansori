#!/usr/bin/env python3
"""Stage 8: cut 44.1 kHz waveforms and fan the corpus out per speaker.

This stage replaces the old 08_f0.py. openvpi/DiffSinger extracts f0 itself during
binarization via the configured `pe` (see `preprocessing/acoustic_binarizer.py`),
so injecting pre-computed CREPE .npy files is no longer needed — and would be
wrong anyway, since the old grid was hop 128 @ 24 kHz (5.33 ms) and the new one is
hop 512 @ 44.1 kHz (11.6 ms).

What this stage does instead is produce the on-disk layout openvpi expects. It
assigns speakers through `datasets:` entries, one raw_data_dir per speaker
(`basics/base_binarizer.py:75-94`), rather than through a binarizer subclass:

    data/pansori/{speaker}/wavs/{item}.wav
    data/pansori/{speaker}/transcriptions.csv

Audio is cut straight from Pansori_Data/Audio, which is already 44.1 kHz stereo,
so there is no resampling — only stereo-to-mono and a slice on the sample grid.
Cutting is done one source song at a time because every song holds many segments.

Also writes datasets.yaml, the `datasets:` block to paste into the model config,
with speakers ordered by utterance count so spk_id 0 is the best-represented one.

Input : data/{set}/segments.jsonl + spk_map.json + transcriptions.csv
Output: data/pansori/{speaker}/... + data/pansori/datasets.yaml
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pansori_lib import PANSORI_DATA, TARGET_SAMPLE_RATE, read_jsonl, set_dir

AUDIO_DIR = PANSORI_DATA / "Audio"


def load_speakers(out_dir: Path, drop_unknown: bool):
    with open(out_dir / "spk_map.json", encoding="utf-8") as f:
        spk_map = json.load(f)
    item2spk = spk_map["item2spk"]
    if drop_unknown:
        item2spk = {k: v for k, v in item2spk.items() if v != "unknown"}
    return item2spk


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8: 44.1 kHz export, per speaker")
    parser.add_argument("--set", default="full")
    parser.add_argument("--out", default="data/pansori",
                        help="repo-relative output root")
    parser.add_argument("--drop_unknown", action="store_true",
                        help="exclude the pooled 'unknown' speaker")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N songs (smoke test)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / args.out
    src_dir = set_dir(args.set)

    item2spk = load_speakers(src_dir, args.drop_unknown)
    segments = {r["item_name"]: r for r in read_jsonl(src_dir / "segments.jsonl")}
    with open(src_dir / "transcriptions.csv", encoding="utf-8") as f:
        labels = {row["name"]: row for row in csv.DictReader(f)}

    # Only items that survived every stage AND have a speaker.
    items = [name for name in labels if name in item2spk and name in segments]
    by_song = collections.defaultdict(list)
    for name in items:
        by_song[segments[name]["song_stem"]].append(name)

    songs = sorted(by_song)
    if args.limit:
        songs = songs[:args.limit]

    written = collections.Counter()
    missing_audio = []
    for stem in tqdm(songs, desc="songs"):
        wav_path = AUDIO_DIR / f"{stem}.wav"
        if not wav_path.exists():
            missing_audio.append(stem)
            continue
        audio, samplerate = sf.read(wav_path, dtype="float32", always_2d=True)
        assert samplerate == TARGET_SAMPLE_RATE, \
            f"{stem}: {samplerate} Hz, expected {TARGET_SAMPLE_RATE}"
        mono = audio.mean(axis=1)

        for name in by_song[stem]:
            speaker = item2spk[name]
            record = segments[name]
            begin = int(round(record["song_start"] * samplerate))
            end = begin + int(round(record["duration"] * samplerate))
            chunk = mono[begin:end]
            if chunk.size == 0:
                continue
            dest = out_root / speaker / "wavs"
            dest.mkdir(parents=True, exist_ok=True)
            sf.write(dest / f"{name}.wav", chunk, samplerate, subtype="PCM_16")
            written[speaker] += 1

    fields = ["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur"]
    for speaker in sorted(written):
        rows = [labels[n] for n in sorted(items)
                if item2spk[n] == speaker and (out_root / speaker / "wavs" / f"{n}.wav").exists()]
        with open(out_root / speaker / "transcriptions.csv", "w",
                  encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # datasets: block, most-represented speaker first so spk_id 0 is the anchor.
    ordered = [s for s, _ in written.most_common()]
    lines = ["# 생성물: pansori/preprocess/08_export.py — 모델 config 의 datasets: 에 붙여넣기",
             "datasets:"]
    for spk_id, speaker in enumerate(ordered):
        lines += [f"  - raw_data_dir: {args.out}/{speaker}",
                  f"    speaker: {speaker}",
                  f"    spk_id: {spk_id}",
                  "    language: ko",
                  "    test_prefixes: []            # 홀드아웃 곡 uuid8 을 여기에",
                  f"    # 발화 {written[speaker]:,}개"]
    (out_root / "datasets.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n화자 {len(written)}명 / 발화 {sum(written.values()):,}개 → {out_root}")
    for speaker, count in written.most_common():
        print(f"  {speaker:12s} {count:6,d}")
    if missing_audio:
        print(f"\n⚠ 원본 오디오 없음: {len(missing_audio)}곡  예) {missing_audio[:3]}")
    print(f"\ndatasets 블록: {out_root / 'datasets.yaml'}")


if __name__ == "__main__":
    main()
