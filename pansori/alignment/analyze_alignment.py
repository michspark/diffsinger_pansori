#!/usr/bin/env python3
"""Measure whether MFA's alignment of sung pansori is trustworthy.

A speech-trained aligner failing on singing has a characteristic signature:
the phones get crammed into the first part of the clip and the long sustained
vowel that follows is written off as silence. These metrics expose that.

Run inside the `aligner` conda env (needs praatio).
"""

import json
from pathlib import Path
from typing import Dict, List

from praatio import textgrid as tgio

CHECK_DIR = Path(__file__).resolve().parent
ALIGNED_DIR = CHECK_DIR / "aligned"
CORPUS_DIR = CHECK_DIR / "corpus"
SEGMENTS = CHECK_DIR.parent / "data" / "pilot" / "segments.jsonl"

# MFA marks non-speech intervals with these labels (or an empty string).
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def load_durations() -> Dict[str, float]:
    """item_name -> segment duration in seconds, from the Stage 1 manifest."""
    durations = {}
    for line in open(SEGMENTS, encoding="utf-8"):
        record = json.loads(line)
        durations[record["item_name"]] = record["duration"]
    return durations


def analyze_one(tg_path: Path, clip_duration: float) -> Dict:
    """Pull phone-level statistics out of a single TextGrid."""
    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)

    phone_tier = grid.getTier("phones")
    word_tier = grid.getTier("words")

    speech_intervals = []
    silence_total = 0.0
    for interval in phone_tier.entries:
        span = interval.end - interval.start
        if interval.label.strip().lower() in SILENCE_LABELS:
            silence_total += span
        else:
            speech_intervals.append((interval.label, span, interval.start, interval.end))

    if not speech_intervals:
        return {"item": tg_path.stem, "failed": True}

    spans = [span for _, span, _, _ in speech_intervals]
    speech_total = sum(spans)

    # Where does the aligned speech actually sit inside the clip? If MFA gave up
    # on the tail, last_phone_end will be far short of clip_duration.
    first_phone_start = speech_intervals[0][2]
    last_phone_end = speech_intervals[-1][3]

    longest = max(speech_intervals, key=lambda x: x[1])

    return {
        "item": tg_path.stem,
        "failed": False,
        "clip_sec": clip_duration,
        "n_phones": len(speech_intervals),
        "n_words": len([i for i in word_tier.entries if i.label.strip()]),
        "speech_sec": speech_total,
        "silence_sec": silence_total,
        "silence_pct": 100.0 * silence_total / clip_duration,
        "covered_pct": 100.0 * (last_phone_end - first_phone_start) / clip_duration,
        "tail_gap_sec": clip_duration - last_phone_end,
        "head_gap_sec": first_phone_start,
        "mean_phone_ms": 1000.0 * speech_total / len(speech_intervals),
        "max_phone_label": longest[0],
        "max_phone_sec": longest[1],
        "grid_total": phone_tier.maxTimestamp,
    }


def main() -> None:
    durations = load_durations()
    rows: List[Dict] = []
    for tg_path in sorted(ALIGNED_DIR.glob("*.TextGrid")):
        rows.append(analyze_one(tg_path, durations[tg_path.stem]))

    header = (
        f"{'item':16s} {'길이':>6s} {'음소':>4s} {'발화s':>6s} {'침묵%':>6s} "
        f"{'커버%':>6s} {'끝여백':>6s} {'평균ms':>7s} {'최장음소':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("failed"):
            print(f"{row['item']:16s}  정렬 실패")
            continue
        print(
            f"{row['item']:16s} {row['clip_sec']:6.2f} {row['n_phones']:4d} "
            f"{row['speech_sec']:6.2f} {row['silence_pct']:6.1f} "
            f"{row['covered_pct']:6.1f} {row['tail_gap_sec']:6.2f} "
            f"{row['mean_phone_ms']:7.1f} "
            f"{row['max_phone_label']:>5s}{row['max_phone_sec']:7.2f}s"
        )

    json.dump(rows, open(CHECK_DIR / "alignment_stats.json", "w"),
              ensure_ascii=False, indent=1)

    ok = [r for r in rows if not r.get("failed")]
    print()
    print(f"정렬 성공: {len(ok)}/{len(rows)}")
    print(f"평균 침묵 비율: {sum(r['silence_pct'] for r in ok)/len(ok):.1f}%")
    print(f"평균 커버율:    {sum(r['covered_pct'] for r in ok)/len(ok):.1f}%")
    print(f"평균 음소 길이: {sum(r['mean_phone_ms'] for r in ok)/len(ok):.0f} ms")


if __name__ == "__main__":
    main()
