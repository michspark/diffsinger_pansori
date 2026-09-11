#!/usr/bin/env python3
"""Stage 4b: export the existing MFA alignment as SOFA training data.

There is no public Korean SOFA checkpoint (the published one is the Mandarin
opencpop-extension). So fitting the aligner to this corpus means training it,
and SOFA accepts three label strengths at once. Where those are separated is
the point of this script:

    Hand MFA's output over wholesale as full_label and SOFA learns MFA's
    0.03 s floor exactly as it is.

Which is the very reason for switching aligners, so the split is by
per-utterance floor_rate:

    floor_rate <= --max_floor_rate  ->  full_label   the timings are trusted too
    above it                        ->  weak_label   phoneme sequence only, timings dropped
    no alignment at all             ->  no_label     audio only

This is demotion, not discarding -- the audio and the phoneme sequence stay in
use even where the bad timings are erased. The "per-utterance floor-rate
criterion" that pansori/docs/SEGMENTATION_VERDICT.md recommended but was never
implemented is this.

Input : data/{set}/{phones,segments,lyrics}.jsonl + spk_map.json + wavs/*.wav
Output: {out}/full_label/{speaker}/wavs/*.wav + transcriptions.csv   (name,ph_seq,ph_dur)
        {out}/weak_label/{speaker}/wavs/*.wav + transcriptions.csv   (name,ph_seq)
        {out}/no_label/*.wav

    python pansori/preprocess/04b_sofa_data.py --set full --out ~/src/SOFA/data

The wavs are symlinks (SOFA's binarize only reads them). This runs under any env --
it touches nothing but jsonl and symlinks.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from pansori_lib import (SILENCE_LABELS, hangul_only, load_align_config,
                         read_jsonl, set_dir)

# Uses the same definition as label_report.py -- define it afresh here and the two
# numbers drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from label_report import MFA_FLOOR, EPS  # noqa: E402


def phone_rows(record):
    """One phones.jsonl row -> [(ph, start, end), ...]. sil becomes SP, per SOFA's convention.

    Returns None if there is even one OOV, because then the phoneme sequence does
    not match the dictionary -- SOFA's binarize dies on a phoneme it has no entry
    for.
    """
    out = []
    for ph, start, end in record["phones"]:
        if ph.startswith("<OOV"):
            return None
        out.append(("SP" if ph.lower() in SILENCE_LABELS else ph, start, end))
    return out or None


def floor_rate(phones) -> float:
    """The fraction of phonemes MFA pushed to its minimum duration. SP is not counted."""
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
    """SOFA transcriptions.csv. weak_label has no ph_dur column (SOFA README)."""
    fields = ["name", "ph_seq"] + (["ph_dur"] if with_dur else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])


def main() -> None:
    config = load_align_config()
    parser = argparse.ArgumentParser(description="Stage 4b: export SOFA training data")
    parser.add_argument("--set", default="full")
    parser.add_argument("--out", required=True,
                        help="the SOFA repository's data/ directory (e.g. ~/src/SOFA/data)")
    parser.add_argument("--phones", default="phones.jsonl")
    parser.add_argument("--lyrics", default="lyrics.jsonl")
    parser.add_argument("--max_floor_rate", type=float, default=0.15,
                        help="above this rate, demote from full_label to weak_label")
    parser.add_argument("--no_label_limit", type=int, default=0,
                        help="maximum utterances to export as no_label (0 = all)")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser()
    out_dir = set_dir(args.set)
    wav_dir = out_dir / "wavs"
    if not wav_dir.is_dir():
        sys.exit(f"{wav_dir} does not exist. Run 01b_recut_wavs.py --set {args.set} first "
                 f"to restore the wavs 08_export removed.")

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
            skipped["no wav"] += 1
            continue
        speaker = item2spk.get(item)
        if speaker is None:
            skipped["speaker unknown"] += 1
            continue

        phones = phone_rows(aligned[item]) if item in aligned else None
        if phones is None:
            # No alignment, or it hit an OOV -> use the audio only.
            if len(hangul_only(lyrics.get(item, ""))) < min_hangul:
                skipped["too little text"] += 1  # still usable as no_label
            no_label.append((item, wav))
            continue

        rate = floor_rate(phones)
        rates.append(rate)
        bucket = "full_label" if rate <= args.max_floor_rate else "weak_label"
        row = {"name": item,
               "ph_seq": " ".join(ph for ph, _, _ in phones),
               "ph_dur": " ".join(f"{e - s:.6f}" for _, s, e in phones)}
        buckets[bucket].setdefault(speaker, []).append((row, wav))

    # Asking someone to choose a threshold means not showing them only one. Print how
    # much survives at each cut point -- 0.15 is a default, not a measured optimum.
    if rates:
        rates.sort()
        print("floor_rate distribution (over " + f"{len(rates):,} aligned utterances):")
        print("  " + "  ".join(f"p{q}={rates[min(len(rates) - 1, q * len(rates) // 100)]:.2f}"
                               for q in (10, 25, 50, 75, 90)))
        print("  full_label by threshold: " + "  ".join(
            f"{t:.2f}→{sum(r <= t for r in rates):,}" for t in (0.05, 0.10, 0.15, 0.20, 0.30)))
        print(f"  chosen: {args.max_floor_rate:.2f}\n")

    for bucket, by_speaker in buckets.items():
        total = 0
        for speaker, entries in sorted(by_speaker.items()):
            base = out_root / bucket / speaker
            for row, wav in entries:
                link(wav, base / "wavs" / f"{row['name']}.wav")
            write_csv(base / "transcriptions.csv", [r for r, _ in entries],
                      with_dur=(bucket == "full_label"))
            total += len(entries)
        print(f"{bucket:<11} {total:>7,} utterances  {len(by_speaker)} speakers")

    if args.no_label_limit:
        no_label = no_label[:args.no_label_limit]
    for item, wav in no_label:
        link(wav, out_root / "no_label" / f"{item}.wav")
    print(f"{'no_label':<11} {len(no_label):>7,} utterances")

    if skipped:
        print("excluded: " + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common()))
    print(f"\n→ {out_root}")
    print(f"  next: conda activate {config['sofa']['conda_env']} && cd {config['sofa']['root']}")
    print(f"        point data_folder in configs/binarize_config.yaml at {out_root}, then")
    print(f"        python binarize.py && python train.py -p <pretrained.ckpt>")


if __name__ == "__main__":
    main()
