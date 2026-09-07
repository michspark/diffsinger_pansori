#!/usr/bin/env python3
"""Stage 4: force-align, then emit Coda-36 phones with local-time spans.

정렬기는 두 개 중에 고른다 (preprocess/align.yaml 의 `aligner`, 또는 --aligner):

    mfa   korean_mfa. 사전이 IPA 를 내므로 pansori_lib.convert_phone 의 문맥
          규칙(on-glide 분리, 비음/유음 위치)을 거쳐 Coda 가 된다.
    sofa  노래 전용 정렬기. dictionaries/korean-coda.txt 를 그대로 사전으로
          쓰므로 Coda 가 바로 나오고, breath_ap 를 켜면 들숨(AP)까지 라벨된다.

무엇을 골랐든 aligned{suffix}/*.TextGrid 를 남기고 phones{suffix}.jsonl 을 쓴다.
무음은 여기서 'sil' 자리표시자로 두고, Stage 6 이 ROSVOT 노트를 보고 held vowel
인지 진짜 SP 인지 정한다. 백엔드의 속은 aligners.py 에 있다.

Input : data/{set}/segments.jsonl + lyrics.jsonl + wavs/*.wav
        (--lyrics picks another lyrics file, e.g. 03b's lyrics_snapped.jsonl;
         --suffix keeps a second run's corpus/aligned/phones side by side so
         two lyric sources -- or two aligners -- can be compared)
Output: data/{set}/phones.jsonl   {item_name, phones:[[coda,start,end],...],
                                   words:[[word,start,end],...]}
        (+ data/{set}/{mfa,sofa}_corpus/, data/{set}/aligned/ as intermediates)

Run in the `aligner` env (praatio; mfa CLI for the MFA arm). SOFA 는 자기
conda env 로 셸아웃하므로 이 env 에 torch 가 없어도 된다.
"""

import argparse
from pathlib import Path

from aligners import get_aligner
from pansori_lib import load_align_config, read_jsonl, set_dir, write_jsonl


# ---------------------------------------------------------------- 호환 진입점
# pansori/eval/intelligibility.py 가 이 파일을 importlib 로 열어 아래 두 이름을
# 직접 부른다 ("04_align.py 는 숫자로 시작해 import 문으로 못 부른다"). 거기서
# MFA 는 *측정 도구* 이므로 — 과거 수치와의 비교선을 지키려면 정렬기가 바뀌면
# 안 된다 — 설정과 무관하게 MFA 로 고정한 얇은 위임으로 남긴다.

def _mfa():
    return get_aligner("mfa", load_align_config())


def run_mfa(corpus: Path, aligned: Path) -> None:
    """Invoke the MFA CLI. Assumes the `aligner` env is active."""
    _mfa().run(corpus, aligned)


def textgrid_to_coda(tg_path: Path, mapping=None) -> dict:
    """One MFA TextGrid -> {"phones": [...], "words": [...]}.

    `mapping` 은 예전 시그니처를 지키려고 남겨 둔 자리다. MfaAligner 가 필요할 때
    스스로 load_ipa_to_coda() 를 부르므로 넘기지 않아도 된다.
    """
    return _mfa().parse(Path(tg_path))


def main() -> None:
    config = load_align_config()
    parser = argparse.ArgumentParser(description="Stage 4: force-align + Coda phones")
    parser.add_argument("--set", default="pilot_6s")
    parser.add_argument("--aligner", default=None, choices=["mfa", "sofa"],
                        help=f"기본값은 align.yaml 의 aligner ({config['aligner']})")
    parser.add_argument("--skip_align", "--skip_mfa", dest="skip_align",
                        action="store_true",
                        help="reuse an existing aligned/ dir, only re-parse")
    parser.add_argument("--lyrics", default="lyrics.jsonl",
                        help="which lyrics file under data/{set} to align")
    parser.add_argument("--suffix", default="",
                        help="tag corpus/aligned/phones so runs can coexist")
    parser.add_argument("--items", default="",
                        help="file of item_names to align, one per line")
    args = parser.parse_args()

    aligner = get_aligner(args.aligner or config["aligner"], config)
    out_dir = set_dir(args.set)
    segments = read_jsonl(out_dir / "segments.jsonl")
    lyrics = read_jsonl(out_dir / args.lyrics)
    aligned = out_dir / f"aligned{args.suffix}"

    if args.items:
        wanted = set(Path(args.items).read_text(encoding="utf-8").split())
        segments = [r for r in segments if r["item_name"] in wanted]
        lyrics = [r for r in lyrics if r["item_name"] in wanted]

    print(f"정렬기: {aligner.name}")
    if not args.skip_align:
        corpus = aligner.build_corpus(out_dir, segments, lyrics, args.suffix)
        aligner.run(corpus, aligned)

    rows = []
    for tg_path in sorted(aligned.glob("*.TextGrid")):
        rows.append({"item_name": tg_path.stem, **aligner.parse(tg_path)})

    write_jsonl(out_dir / f"phones{args.suffix}.jsonl", rows)
    aligned_n = len(rows)
    oov = sorted({p[0] for r in rows for p in r["phones"] if p[0].startswith("<OOV")})
    breath = sum(1 for r in rows for p in r["phones"] if p[0] == "AP")
    print(f"정렬 완료 {aligned_n}/{len(segments)} → phones{args.suffix}.jsonl")
    if breath:
        print(f"들숨(AP) {breath:,}개")
    if oov:
        print(f"⚠ OOV 발생: {oov}")


if __name__ == "__main__":
    main()
