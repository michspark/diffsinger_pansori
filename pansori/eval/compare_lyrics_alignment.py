#!/usr/bin/env python3
"""Does snapping the lyrics to the 창본 actually improve MFA's alignment?

Prettier text proves nothing. The lyrics exist to be MFA's input, so the test
is MFA's own failure signature, the one pansori/docs/SEGMENTATION_VERDICT.md §2 established:
when the aligner gives up it pins phones to its 0.03s minimum and writes the
rest of the clip off as silence. Both numbers should fall if the words are now
the words that were sung.

The same clips are aligned twice -- once with Whisper's text, once with the
snapped text -- so the comparison is paired and nothing but the .lab differs.

data/{set}/wavs is gone once the pipeline has run, so the sampled clips are
re-cut from the source audio the way 01_segment.py cut them (44.1kHz mono).

정렬기는 preprocess/align.yaml 이 정한 것을 쓴다 (--aligner 로 덮어쓸 수 있다).
가사 축을 재는 자이므로 정렬기는 고정해야 한다 — 두 팔이 같은 정렬기를 통과한다.

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

    The .lab formatting is the aligner's, not ours -- MFA reads 어절, SOFA reads
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
        sys.exit("두 정렬에 공통으로 성공한 항목이 없습니다.")

    print(f"\n{'=' * 68}")
    print(f"정렬 성공: whisper {len(before)} / snapped {len(after)} "
          f"/ 공통 {len(shared)} (표본 {len(rows)})\n")
    print(f"{'지표':<26}{'whisper':>11}{'snapped':>11}{'변화':>12}")

    def mean(source, key):
        return sum(source[i][key] for i in shared) / len(shared)

    for key, label, unit in [
            ("floor_pct", "0.03초 하한선 비율", "%"),
            ("long_silence", "1초+ 침묵 (평균)", "s"),
            ("silence_pct", "전체 침묵 비율", "%"),
            ("n_phones", "음소 수 (평균)", "")]:
        a, b = mean(before, key), mean(after, key)
        delta = f"{b - a:+.2f}{unit}"
        if a:
            delta += f" ({(b - a) / a:+.0%})"
        print(f"{label:<26}{a:>11.2f}{b:>11.2f}{delta:>12}")

    better = sum(after[i]["floor_pct"] < before[i]["floor_pct"] for i in shared)
    worse = sum(after[i]["floor_pct"] > before[i]["floor_pct"] for i in shared)
    print(f"\n하한선 비율이 개선된 클립 {better} / 악화 {worse} "
          f"/ 동일 {len(shared) - better - worse}")

    floor_before, floor_after = mean(before, "floor_pct"), mean(after, "floor_pct")
    silence_ok = mean(after, "long_silence") <= mean(before, "long_silence") + 0.05
    passed = floor_before and (floor_before - floor_after) / floor_before >= 0.10
    print(f"\n합격선: 하한선 -10% 이상 그리고 1초+ 침묵 증가 없음  ->  "
          f"{'통과' if passed and silence_ok else '미달'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="스냅 가사의 정렬 개선 검증")
    parser.add_argument("--set", default="full")
    parser.add_argument("--lyrics", default="lyrics_snapped.jsonl")
    parser.add_argument("--n", type=int, default=200, help="표본 세그먼트 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", default="", help="기본: data/{set}/lyric_eval")
    parser.add_argument("--reuse", action="store_true",
                        help="이미 잘라둔 wav와 정렬 결과를 재사용하고 분석만")
    parser.add_argument("--aligner", default=None, choices=["mfa", "sofa"],
                        help="기본값은 preprocess/align.yaml 의 aligner")
    args = parser.parse_args()

    config = load_align_config()
    aligner = get_aligner(args.aligner or config["aligner"], config)

    out_dir = set_dir(args.set)
    segments = read_jsonl(out_dir / "segments.jsonl")
    lyrics = read_jsonl(out_dir / args.lyrics)
    durations = {r["item_name"]: r["duration"] for r in segments}

    snapped = [r for r in lyrics if r["source"] == "book"]
    if not snapped:
        sys.exit(f"{args.lyrics} 에 source=book 인 세그먼트가 없습니다.")
    rows = random.Random(args.seed).sample(snapped, min(args.n, len(snapped)))
    print(f"교정된 {len(snapped):,}개 중 {len(rows)}개 표본 "
          f"(seed={args.seed}, 정렬기={aligner.name})")

    work = Path(args.work_dir) if args.work_dir else out_dir / "lyric_eval"
    wav_dir = work / "wavs"

    if not args.reuse:
        print("\n클립 재절단...")
        cut_clips(rows, segments, wav_dir)
        for field, tag in [("text_whisper", "whisper"), ("text", "snapped")]:
            print(f"\n{aligner.name} 정렬 ({tag})...")
            build_corpus(rows, wav_dir, work / f"corpus_{tag}", field, aligner)
            aligner.run(work / f"corpus_{tag}", work / f"aligned_{tag}")

    before = measure(work / "aligned_whisper", durations)
    after = measure(work / "aligned_snapped", durations)
    report(before, after, rows)
    json.dump({"before": before, "after": after},
              open(work / "metrics.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n원본 지표: {work / 'metrics.json'}")


if __name__ == "__main__":
    main()
