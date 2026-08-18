#!/usr/bin/env python3
"""Stage 4: force-align with korean_mfa, then convert IPA -> Coda phones.

Builds an MFA corpus (wav + normalized .lab) from segments + lyrics, runs
`mfa align`, and parses each TextGrid into Coda-36 phones with local-time
spans. The context rules (on-glide split, nasal/liquid position) live in
pansori_lib.convert_phone. Silence is kept as a 'sil' placeholder here;
Stage 6 decides (using ROSVOT notes) whether it is a held vowel or a real SP.

Input : data/{set}/segments.jsonl + lyrics.jsonl + wavs/*.wav
Output: data/{set}/phones.jsonl   {item_name, phones:[[coda,start,end],...],
                                   words:[[syllable,start,end],...]}
        (+ data/{set}/mfa_corpus/, data/{set}/aligned/ as intermediates)

Run in the `aligner` env (mfa CLI + praatio).
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

from praatio import textgrid as tgio

from pansori_lib import (SILENCE_LABELS, convert_phone, load_ipa_to_coda,
                         read_jsonl, write_jsonl, set_dir)

ACOUSTIC = "korean_mfa"
DICTIONARY = "korean_mfa"
MIN_HANGUL = 2  # skip utterances with too little text to align


def build_corpus(out_dir: Path, segments, lyrics) -> Path:
    """Create mfa_corpus/{item}.wav (symlink) + {item}.lab (normalized text)."""
    corpus = out_dir / "mfa_corpus"
    corpus.mkdir(exist_ok=True)
    text_by_item = {r["item_name"]: r["text"] for r in lyrics}

    n = 0
    for record in segments:
        item = record["item_name"]
        text = text_by_item.get(item, "").strip()
        if sum(1 for ch in text if "가" <= ch <= "힣") < MIN_HANGUL:
            continue                      # nothing alignable
        wav_src = (out_dir / "wavs" / f"{item}.wav").resolve()
        wav_link = corpus / f"{item}.wav"
        if wav_link.exists() or wav_link.is_symlink():
            wav_link.unlink()
        os.symlink(wav_src, wav_link)
        (corpus / f"{item}.lab").write_text(text, encoding="utf-8")
        n += 1
    print(f"MFA 코퍼스: {n}개 (텍스트 부족분 제외 {len(segments) - n}개)")
    return corpus


def run_mfa(corpus: Path, aligned: Path) -> None:
    """Invoke the MFA CLI. Assumes the `aligner` env is active."""
    subprocess.run(
        ["mfa", "align", str(corpus), DICTIONARY, ACOUSTIC, str(aligned),
         "--clean", "--output_format", "long_textgrid"],
        check=True,
    )


def textgrid_to_coda(tg_path: Path, mapping) -> dict:
    """One TextGrid -> {"phones": [[coda,start,end],...], "words": [[syl,start,end],...]}.

    A single IPA interval can split into two Coda tokens (e.g. mʲ -> m y);
    the interval duration is split evenly (harmless for length parity/is_slur).
    sil/spn is kept as a single 'sil' placeholder for Stage 6 to resolve.

    The `words` tier carries one Hangul syllable per interval. openvpi/DiffSinger's
    variance model needs `ph_num` (phones per word), which is derivable only from
    this tier, so it is emitted alongside the phones rather than discarded.
    """
    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    intervals = [(iv.label.strip(), iv.start, iv.end)
                 for iv in grid.getTier("phones").entries]

    phones = []
    for idx, (label, start, end) in enumerate(intervals):
        if label.lower() in SILENCE_LABELS:
            phones.append(["sil", start, end])
            continue
        # next non-silence IPA label drives the context rules
        next_ipa = None
        for nlab, _, _ in intervals[idx + 1:]:
            if nlab.lower() not in SILENCE_LABELS:
                next_ipa = nlab
                break
        tokens = convert_phone(label, next_ipa, mapping)
        step = (end - start) / len(tokens)
        for k, tok in enumerate(tokens):
            phones.append([tok, start + k * step, start + (k + 1) * step])

    words = [[iv.label.strip(), iv.start, iv.end]
             for iv in grid.getTier("words").entries
             if iv.label.strip() not in SILENCE_LABELS]
    return {"phones": phones, "words": words}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: MFA align + IPA->Coda")
    parser.add_argument("--set", default="pilot_6s")
    parser.add_argument("--skip_mfa", action="store_true",
                        help="reuse an existing aligned/ dir, only re-parse")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    segments = read_jsonl(out_dir / "segments.jsonl")
    lyrics = read_jsonl(out_dir / "lyrics.jsonl")
    aligned = out_dir / "aligned"

    if not args.skip_mfa:
        corpus = build_corpus(out_dir, segments, lyrics)
        run_mfa(corpus, aligned)

    mapping = load_ipa_to_coda()
    rows = []
    for tg_path in sorted(aligned.glob("*.TextGrid")):
        parsed = textgrid_to_coda(tg_path, mapping)
        rows.append({"item_name": tg_path.stem, **parsed})

    write_jsonl(out_dir / "phones.jsonl", rows)
    aligned_n = len(rows)
    oov = sorted({p[0] for r in rows for p in r["phones"] if p[0].startswith("<OOV")})
    print(f"정렬 완료 {aligned_n}/{len(segments)} → phones.jsonl")
    if oov:
        print(f"⚠ OOV 발생: {oov}")


if __name__ == "__main__":
    main()
