#!/usr/bin/env python3
"""Stage 6: phones x words x notes -> transcriptions.csv (openvpi/DiffSinger).

This replaces the MoonInTheRiver-era assembly. That format packed the melody onto
the phoneme axis — `note_seq`, `midi_dur` and `is_slur` all had to be the same
length as `ph_seq`, so a vowel spanning N notes was duplicated N times and every
copy after the first was flagged `is_slur=1`.

openvpi keeps the two axes separate (`preprocessing/variance_binarizer.py`): notes
carry their own durations and are aligned to frames independently of the phonemes.
So the duplication and the slur flag are gone. 시김새 — one syllable held across
several notes — is now expressed directly as several notes over one word.

What is kept from the old assembly, because it is about *this data*, not the format:

  1. Silence covered by a note is a held sung vowel, not a pause, so it is folded
     into the preceding phone. Silence with no note becomes SP. korean_mfa is
     trained on speech and labels long held notes as silence; without this the
     model would be told to go quiet where the singer is still singing.
  2. Phones shorter than MIN_PH_DUR are folded into a neighbour. At hop 512 /
     44.1 kHz one frame is 11.6 ms, so a sub-20 ms phone can round to zero frames.

New in this version: `ph_num`, the phones-per-word grouping the variance duration
predictor needs. It comes from the MFA `words` tier (stage 4 now emits it). Words
are whitespace tokens, so a 2-syllable 어절 arrives as one unit; where the phones
match what the dictionary predicts for its syllables, the unit is split so that
one word = one syllable, matching how notes are grouped at inference time. Where
sandhi changed the phone count (굴러 -> `g u L eo`, not `g u L r eo`) the 어절 is
left whole rather than split on a guess.

Fields: name, ph_seq, ph_dur, ph_num, note_seq, note_dur
Input : data/{set}/phones.jsonl + notes.jsonl + segments.jsonl
Output: data/{set}/transcriptions.csv

Run in the `aligner` or `base` env (librosa).
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import librosa

from pansori_lib import CODA_VOWELS, overlap, read_jsonl, set_dir

MIN_PH_DUR = 0.02   # < 2 frames at hop 512 / 44.1 kHz
NOTE_COVER = 0.05   # a note must cover this much of a silence to claim it
EPS = 1e-4

DICT_PATH = Path(__file__).resolve().parents[2] / "dictionaries" / "korean-coda.txt"


def load_syllable_dict() -> Dict[str, List[str]]:
    """Hangul syllable -> Coda phones, from the same file the model is trained with."""
    table = {}
    with open(DICT_PATH, encoding="utf-8") as f:
        for line in f:
            syllable, phones = line.rstrip("\n").split("\t")
            table[syllable] = phones.split()
    return table


class Phone:
    """A Coda phone with a local-time span and the MFA word it belongs to."""

    def __init__(self, coda: str, start: float, end: float, word: Optional[int] = None):
        self.coda = coda
        self.start = start
        self.end = end
        self.word = word
        self.is_vowel = coda in CODA_VOWELS

    @property
    def dur(self) -> float:
        return self.end - self.start


def assign_words(phones: List[Phone], words) -> None:
    """Tag each phone with the index of the MFA word interval containing it.

    Containment is tested at the phone's midpoint: an IPA interval that split into
    two Coda tokens gets midpoints strictly inside the original span, so both land
    in the same word even after the even-split in stage 4.
    """
    for phone in phones:
        if phone.coda == "sil":
            continue
        mid = (phone.start + phone.end) / 2
        for index, (_, start, end) in enumerate(words):
            if start - EPS <= mid < end + EPS:
                phone.word = index
                break


def absorb_silence(phones: List[Phone], notes) -> List[Phone]:
    """Silence under a note = held note -> extend previous phone. Else -> SP."""
    out: List[Phone] = []
    for phone in phones:
        if phone.coda != "sil":
            out.append(phone)
            continue
        covered = any(overlap(phone.start, phone.end, ns, ne) > NOTE_COVER
                      for _, ns, ne in notes)
        if covered and out:
            out[-1].end = phone.end
        else:
            out.append(Phone("SP", phone.start, phone.end))
    return out


def merge_tiny(phones: List[Phone], min_dur: float = MIN_PH_DUR) -> List[Phone]:
    """Fold sub-min-duration phones into a neighbour, transferring their span.

    The survivor keeps its own word tag, so a phone absorbed across a word
    boundary moves into the neighbour's word rather than leaving a dangling group.
    """
    out = list(phones)
    i = 0
    while i < len(out) and len(out) > 1:
        if out[i].dur >= min_dur:
            i += 1
            continue
        if i > 0:
            out[i - 1].end = out[i].end
            del out[i]
            i = max(0, i - 1)
        else:
            out[i + 1].start = out[i].start
            del out[i]
    return out


def group_words(phones: List[Phone], words, table) -> List[List[Phone]]:
    """Split the phone sequence into words, one group per SP and per syllable.

    A run of phones sharing an MFA word index is one group. If that word's text is
    multi-syllable and the dictionary's phones for those syllables concatenate to
    exactly this run, the run is re-split per syllable; otherwise it stays whole.
    """
    groups: List[List[Phone]] = []
    index = 0
    while index < len(phones):
        phone = phones[index]
        if phone.coda == "SP" or phone.word is None:
            groups.append([phone])
            index += 1
            continue
        run = [phone]
        while index + len(run) < len(phones) and phones[index + len(run)].word == phone.word:
            run.append(phones[index + len(run)])
        index += len(run)

        syllables = [c for c in words[phone.word][0] if "가" <= c <= "힣"]
        if len(syllables) > 1:
            expected = [table.get(s, []) for s in syllables]
            if [p.coda for p in run] == [ph for group in expected for ph in group]:
                offset = 0
                for group in expected:
                    groups.append(run[offset:offset + len(group)])
                    offset += len(group)
                continue
        groups.append(run)
    return groups


def note_timeline(notes, duration: float):
    """Notes -> (note_seq, note_dur) tiling [0, duration] with 'rest' in the gaps."""
    seq, boundaries = [], [0.0]
    cursor = 0.0
    for pitch, start, end in notes:
        start, end = max(start, cursor), min(end, duration)
        if end - start <= EPS:
            continue
        if start - cursor > EPS:
            seq.append("rest")
            boundaries.append(start)
        seq.append(librosa.midi_to_note(int(pitch), unicode=False))
        boundaries.append(end)
        cursor = end
    if duration - cursor > EPS:
        seq.append("rest")
        boundaries.append(duration)
    if not seq:
        return [], []
    boundaries[-1] = duration
    durs = [boundaries[i + 1] - boundaries[i] for i in range(len(seq))]
    return seq, durs


def assemble(record, notes_raw, duration: float, table):
    """One phones.jsonl record + its notes -> the six CSV fields, or None."""
    notes = [(int(p), s, e) for p, s, e in notes_raw]
    phones = [Phone(c, s, e) for c, s, e in record["phones"]]
    words = record["words"]

    assign_words(phones, words)
    phones = absorb_silence(phones, notes)
    phones = merge_tiny(phones)
    if not phones:
        return None

    # Phone spans tile the utterance; re-derive durations from the boundaries so
    # rounding cannot drift the total away from the waveform length.
    phones[0].start = 0.0
    phones[-1].end = duration
    ph_dur = [round(p.end, 6) - round(p.start, 6) for p in phones]

    groups = group_words(phones, words, table)
    ph_num = [len(g) for g in groups]
    assert sum(ph_num) == len(phones), "ph_num does not cover ph_seq"

    note_seq, note_dur = note_timeline(notes, duration)
    if not note_seq or all(n == "rest" for n in note_seq):
        return None

    return {
        "ph_seq": " ".join(p.coda for p in phones),
        "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur),
        "ph_num": " ".join(str(n) for n in ph_num),
        "note_seq": " ".join(note_seq),
        "note_dur": " ".join(f"{d:.6f}" for d in note_dur),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: phones x notes -> transcriptions.csv")
    parser.add_argument("--set", default="full")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    table = load_syllable_dict()
    notes = {r["item_name"]: r["notes"] for r in read_jsonl(out_dir / "notes.jsonl")}
    duration = {r["item_name"]: r["duration"]
                for r in read_jsonl(out_dir / "segments.jsonl")}

    rows, skipped = [], 0
    for record in read_jsonl(out_dir / "phones.jsonl"):
        item = record["item_name"]
        if not notes.get(item):
            skipped += 1                  # no melody to condition on
            continue
        fields = assemble(record, notes[item], duration[item], table)
        if fields is None:
            skipped += 1
            continue
        rows.append({"name": item, **fields})

    out_path = out_dir / "transcriptions.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur"])
        writer.writeheader()
        writer.writerows(rows)

    split = sum(1 for r in rows if len(r["ph_num"].split()) > len(r["note_seq"].split()))
    print(f"조립 {len(rows)}개 → {out_path}  (음표없음 등 제외 {skipped}개)")
    print(f"  평균 단어수 {sum(len(r['ph_num'].split()) for r in rows) / max(len(rows), 1):.1f}, "
          f"평균 노트수 {sum(len(r['note_seq'].split()) for r in rows) / max(len(rows), 1):.1f}")


if __name__ == "__main__":
    main()
