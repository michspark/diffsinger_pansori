#!/usr/bin/env python3
"""Stage 4: force-align, then emit Coda-36 phones with local-time spans.

The aligner is picked between two (`aligner` in preprocess/align.yaml, or --aligner):

    mfa   korean_mfa. Its dictionary emits IPA, so it becomes Coda by passing
          through pansori_lib.convert_phone's contextual rules (on-glide
          separation, nasal and liquid position).
    sofa  An aligner built for singing. It uses dictionaries/korean-coda.txt
          directly as its dictionary, so Coda comes out natively, and with
          breath_ap on it labels breaths (AP) as well.

Whichever is chosen, it leaves aligned{suffix}/*.TextGrid and writes
phones{suffix}.jsonl. Silence is left here as a 'sil' placeholder, and Stage 6
decides from the ROSVOT notes whether it is a held vowel or a real SP. The
backends themselves live in aligners.py.

Input : data/{set}/segments.jsonl + lyrics.jsonl + wavs/*.wav
        (--lyrics picks another lyrics file, e.g. 03b's lyrics_snapped.jsonl;
         --suffix keeps a second run's corpus/aligned/phones side by side so
         two lyric sources -- or two aligners -- can be compared)
Output: data/{set}/phones.jsonl   {item_name, phones:[[coda,start,end],...],
                                   words:[[word,start,end],...]}
        (+ data/{set}/{mfa,sofa}_corpus/, data/{set}/aligned/ as intermediates)

Run in the `aligner` env (praatio; the mfa CLI for the MFA arm). SOFA shells
out to its own conda env, so this env does not need torch.
"""

import argparse
from pathlib import Path

from aligners import get_aligner
from pansori_lib import load_align_config, read_jsonl, set_dir, write_jsonl


# ------------------------------------------------------- compatibility entry points
# pansori/eval/intelligibility.py opens this file with importlib and calls the two
# names below directly ("04_align.py starts with a digit and cannot be reached by an
# import statement"). There, MFA is a *measuring instrument* -- the aligner must not
# change if the comparison line with earlier numbers is to hold -- so these stay as
# thin delegations pinned to MFA regardless of the config.

def _mfa():
    return get_aligner("mfa", load_align_config())


def run_mfa(corpus: Path, aligned: Path) -> None:
    """Invoke the MFA CLI. Assumes the `aligner` env is active."""
    _mfa().run(corpus, aligned)


def textgrid_to_coda(tg_path: Path, mapping=None) -> dict:
    """One MFA TextGrid -> {"phones": [...], "words": [...]}.

    `mapping` is kept only to preserve the old signature. MfaAligner calls
    load_ipa_to_coda() itself when it needs it, so there is no need to pass one.
    """
    return _mfa().parse(Path(tg_path))


def main() -> None:
    config = load_align_config()
    parser = argparse.ArgumentParser(description="Stage 4: force-align + Coda phones")
    parser.add_argument("--set", default="pilot_6s")
    parser.add_argument("--aligner", default=None, choices=["mfa", "sofa"],
                        help=f"defaults to align.yaml's aligner ({config['aligner']})")
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

    print(f"aligner: {aligner.name}")
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
    print(f"aligned {aligned_n}/{len(segments)} -> phones{args.suffix}.jsonl")
    if breath:
        print(f"breaths (AP): {breath:,}")
    if oov:
        print(f"⚠ OOV encountered: {oov}")


if __name__ == "__main__":
    main()
