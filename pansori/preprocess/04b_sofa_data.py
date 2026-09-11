#!/usr/bin/env python3
"""Stage 4b: 기존 MFA 정렬을 SOFA 학습 데이터로 내보낸다.

한국어 SOFA 공개 체크포인트는 없다 (공개된 것은 만다린 opencpop-extension). 그래서
정렬기를 이 코퍼스에 맞추려면 직접 학습해야 하고, SOFA 는 라벨 강도가 다른 세 종류를
동시에 받는다. 여기서 갈리는 지점이 이 스크립트의 요점이다:

    MFA 결과를 통째로 full_label 로 주면 SOFA 는 MFA 의 0.03초 바닥을 그대로 배운다.

그것이야말로 정렬기를 바꾸려는 이유이므로, 발화 단위 floor_rate 로 나눈다:

    floor_rate <= --max_floor_rate  ->  full_label   타이밍까지 믿는다
    초과                             ->  weak_label   음소열만 주고 타이밍은 버린다
    정렬 자체가 없음                  ->  no_label     오디오만 쓴다

버리는 게 아니라 강등이다 — 나쁜 타이밍을 지우고도 오디오와 음소열은 계속 쓴다.
pansori/docs/SEGMENTATION_VERDICT.md 가 권고했지만 구현되지 않은 "발화별 floor-rate 기준"이 여기다.

Input : data/{set}/{phones,segments,lyrics}.jsonl + spk_map.json + wavs/*.wav
Output: {out}/full_label/{화자}/wavs/*.wav + transcriptions.csv   (name,ph_seq,ph_dur)
        {out}/weak_label/{화자}/wavs/*.wav + transcriptions.csv   (name,ph_seq)
        {out}/no_label/*.wav

    python pansori/preprocess/04b_sofa_data.py --set full --out ~/src/SOFA/data

wav 는 심링크다 (SOFA 의 binarize 가 읽기만 한다). 어느 env 에서든 돈다 — jsonl 과
심링크만 만진다.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from pansori_lib import (SILENCE_LABELS, hangul_only, load_align_config,
                         read_jsonl, set_dir)

# label_report.py 와 같은 정의를 쓴다 — 여기서 새로 정하면 두 숫자가 갈라진다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from label_report import MFA_FLOOR, EPS  # noqa: E402


def phone_rows(record):
    """phones.jsonl 한 줄 -> [(ph, start, end), ...]. sil 은 SOFA 규약대로 SP.

    OOV 가 하나라도 있으면 음소열이 사전과 맞지 않으므로 None 을 돌려준다 —
    SOFA 의 binarize 는 사전에 없는 음소에서 죽는다.
    """
    out = []
    for ph, start, end in record["phones"]:
        if ph.startswith("<OOV"):
            return None
        out.append(("SP" if ph.lower() in SILENCE_LABELS else ph, start, end))
    return out or None


def floor_rate(phones) -> float:
    """MFA 가 최소 길이에 밀어 넣은 음소의 비율. SP 는 세지 않는다."""
    spoken = [(ph, s, e) for ph, s, e in phones if ph != "SP"]
    if not spoken:
        return 1.0
    return sum(1 for _, s, e in spoken if e - s <= MFA_FLOOR + EPS) / len(spoken)


def link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def write_csv(path: Path, rows, with_dur: bool) -> None:
    """SOFA transcriptions.csv. weak_label 은 ph_dur 열이 없다 (SOFA README)."""
    fields = ["name", "ph_seq"] + (["ph_dur"] if with_dur else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])


def main() -> None:
    config = load_align_config()
    parser = argparse.ArgumentParser(description="Stage 4b: SOFA 학습 데이터 내보내기")
    parser.add_argument("--set", default="full")
    parser.add_argument("--out", required=True,
                        help="SOFA 저장소의 data/ 디렉터리 (예: ~/src/SOFA/data)")
    parser.add_argument("--phones", default="phones.jsonl")
    parser.add_argument("--lyrics", default="lyrics.jsonl")
    parser.add_argument("--max_floor_rate", type=float, default=0.15,
                        help="이 비율을 넘으면 full_label 대신 weak_label 로 강등")
    parser.add_argument("--no_label_limit", type=int, default=0,
                        help="no_label 로 내보낼 최대 발화 수 (0 = 전부)")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser()
    out_dir = set_dir(args.set)
    wav_dir = out_dir / "wavs"
    if not wav_dir.is_dir():
        sys.exit(f"{wav_dir} 가 없습니다. 먼저 01b_recut_wavs.py --set {args.set} 를 "
                 f"돌려 08_export 가 지운 wav 를 복원하십시오.")

    with open(out_dir / "spk_map.json", encoding="utf-8") as f:
        item2spk = json.load(f)["item2spk"]
    lyrics = {r["item_name"]: r["text"] for r in read_jsonl(out_dir / args.lyrics)}
    aligned = {r["item_name"]: r for r in read_jsonl(out_dir / args.phones)}
    min_hangul = config.get("min_hangul", 2)

    buckets = {"full_label": {}, "weak_label": {}}
    no_label, skipped, rates = [], Counter(), []

    for record in read_jsonl(out_dir / "segments.jsonl"):
        item = record["item_name"]
        wav = wav_dir / f"{item}.wav"
        if not wav.exists():
            skipped["wav 없음"] += 1
            continue
        speaker = item2spk.get(item)
        if speaker is None:
            skipped["화자 미상"] += 1
            continue

        phones = phone_rows(aligned[item]) if item in aligned else None
        if phones is None:
            # 정렬이 없거나 OOV 를 물었다 -> 오디오만 쓴다.
            if len(hangul_only(lyrics.get(item, ""))) < min_hangul:
                skipped["가사 부족"] += 1        # 그래도 no_label 로는 쓸 수 있다
            no_label.append((item, wav))
            continue

        rate = floor_rate(phones)
        rates.append(rate)
        bucket = "full_label" if rate <= args.max_floor_rate else "weak_label"
        row = {"name": item,
               "ph_seq": " ".join(ph for ph, _, _ in phones),
               "ph_dur": " ".join(f"{e - s:.6f}" for _, s, e in phones)}
        buckets[bucket].setdefault(speaker, []).append((row, wav))

    # 문턱값 하나를 고르게 하려면 그 하나만 보여 줘서는 안 된다. 어디서 갈라도
    # 얼마가 남는지 같이 찍는다 -- 0.15 는 기본값일 뿐 측정된 최적값이 아니다.
    if rates:
        rates.sort()
        print("floor_rate 분포 (정렬된 " + f"{len(rates):,}발화):")
        print("  " + "  ".join(f"p{q}={rates[min(len(rates) - 1, q * len(rates) // 100)]:.2f}"
                               for q in (10, 25, 50, 75, 90)))
        print("  문턱값별 full_label: " + "  ".join(
            f"{t:.2f}→{sum(r <= t for r in rates):,}" for t in (0.05, 0.10, 0.15, 0.20, 0.30)))
        print(f"  선택: {args.max_floor_rate:.2f}\n")

    for bucket, by_speaker in buckets.items():
        total = 0
        for speaker, entries in sorted(by_speaker.items()):
            base = out_root / bucket / speaker
            for row, wav in entries:
                link(wav, base / "wavs" / f"{row['name']}.wav")
            write_csv(base / "transcriptions.csv", [r for r, _ in entries],
                      with_dur=(bucket == "full_label"))
            total += len(entries)
        print(f"{bucket:<11} {total:>7,}발화  화자 {len(by_speaker)}명")

    if args.no_label_limit:
        no_label = no_label[:args.no_label_limit]
    for item, wav in no_label:
        link(wav, out_root / "no_label" / f"{item}.wav")
    print(f"{'no_label':<11} {len(no_label):>7,}발화")

    if skipped:
        print("제외: " + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common()))
    print(f"\n→ {out_root}")
    print(f"  다음: conda activate {config['sofa']['conda_env']} && cd {config['sofa']['root']}")
    print(f"        configs/binarize_config.yaml 의 data_folder 를 {out_root} 로 두고")
    print(f"        python binarize.py && python train.py -p <사전학습.ckpt>")


if __name__ == "__main__":
    main()
