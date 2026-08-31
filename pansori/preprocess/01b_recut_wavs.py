#!/usr/bin/env python3
"""Stage 1b: rebuild data/{set}/wavs from segments.jsonl.

Stage 4 needs the cut waveforms, but the pipeline throws `wavs/` away once the
corpus has been exported, so any later re-alignment (a lyrics change, a dictionary
fix) has to get them back first.

**Re-running 01_segment.py is not the way to get them back.** It is no longer
reproducible: the `base` env it was documented to run under now fails to import
numba (numpy moved to 2.5), and under any other env `librosa.effects.split` lands
a few milliseconds off -- a re-run measured 31,378 utterances against the recorded
31,529, with boundaries shifted by 3-11 ms. Cutting on those boundaries would put
every phone in `phones.jsonl` a few milliseconds away from the audio it describes.

`segments.jsonl` is the record of where the cuts actually are, so this rebuilds
from that record instead of re-deriving it:

  - items already exported to data/pansori are **symlinked**. Those files are the
    audio the model trains on, so aligning against them is exact, not merely close,
    and it costs no disk.
  - the rest -- items that failed stage 4 and so never reached the corpus -- are cut
    from the source with 08_export.py's sample arithmetic.

Input : data/{set}/segments.jsonl + spk_map.json (+ data/pansori, + Pansori_Data/Audio)
Output: data/{set}/wavs/{item_name}.wav

    python pansori/preprocess/01b_recut_wavs.py --set full
"""

import argparse
import collections
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from pansori_lib import PANSORI_DATA, TARGET_SAMPLE_RATE, read_jsonl, set_dir

AUDIO_DIR = PANSORI_DATA / "Audio"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1b: segments.jsonl -> wavs/")
    parser.add_argument("--set", default="full")
    parser.add_argument("--reuse", default="data/pansori",
                        help="exported corpus to symlink from ('' to cut everything)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = set_dir(args.set)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    segments = read_jsonl(out_dir / "segments.jsonl")
    with open(out_dir / "spk_map.json", encoding="utf-8") as f:
        item2spk = json.load(f)["item2spk"]
    reuse_root = repo_root / args.reuse if args.reuse else None

    linked, to_cut = 0, collections.defaultdict(list)
    for record in segments:
        name = record["item_name"]
        dest = wav_dir / f"{name}.wav"
        if dest.exists() or dest.is_symlink():
            linked += 1
            continue
        speaker = item2spk.get(name)
        source = (reuse_root / speaker / "wavs" / f"{name}.wav"
                  if reuse_root and speaker else None)
        if source is not None and source.exists():
            dest.symlink_to(source.resolve())
            linked += 1
        else:
            to_cut[record["song_stem"]].append(record)

    print(f"링크 {linked:,}개 / 잘라야 할 것 {sum(len(v) for v in to_cut.values()):,}개 "
          f"({len(to_cut)}곡)")

    missing_audio, cut = [], 0
    for stem in tqdm(sorted(to_cut), desc="songs"):
        wav_path = AUDIO_DIR / f"{stem}.wav"
        if not wav_path.exists():
            missing_audio.append(stem)
            continue
        audio, samplerate = sf.read(wav_path, dtype="float32", always_2d=True)
        assert samplerate == TARGET_SAMPLE_RATE, \
            f"{stem}: {samplerate} Hz, expected {TARGET_SAMPLE_RATE}"
        mono = audio.mean(axis=1)

        for record in to_cut[stem]:
            begin = int(round(record["song_start"] * samplerate))
            end = begin + int(round(record["duration"] * samplerate))
            chunk = mono[begin:end]
            if chunk.size == 0:
                continue
            sf.write(wav_dir / f"{record['item_name']}.wav", chunk, samplerate,
                     subtype="PCM_16")
            cut += 1

    total = len(list(wav_dir.iterdir()))
    print(f"\n{wav_dir}: {total:,}개 (링크 {linked:,} + 신규 {cut:,}) / "
          f"segments {len(segments):,}개")
    if missing_audio:
        print(f"⚠ 원본 오디오 없음: {len(missing_audio)}곡  예) {missing_audio[:3]}")


if __name__ == "__main__":
    main()
