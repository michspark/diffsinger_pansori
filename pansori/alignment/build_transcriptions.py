#!/usr/bin/env python3
"""Assemble transcriptions.txt 7-field lines for a handful of segments and
self-check them. This is the last safety pin before the 394-song expansion:
prove the verified pieces (IPA->Coda, silence->vowel absorption, ROSVOT notes)
compose into length-consistent fields that express 시김새 via is_slur.

7 fields: item_name | text | ph_seq | note_seq | midi_dur | ph_durs | is_slur
Read-only on existing artifacts (aligned_*, corpus_*, ROSVOT midi). Non-destructive.

Run in `aligner` env (praatio + pretty_midi + librosa + soundfile).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import pretty_midi
import soundfile as sf
from praatio import textgrid as tgio

CHECK_DIR = Path(__file__).resolve().parent
PANSORI = CHECK_DIR.parent
DATA = PANSORI / "data" / "pilot_6s"
MIDI_DIR = Path("/home/sangheon/Desktop/Pansori_Data/rosvot_midi")

SILENCE = {"", "sil", "sp", "spn", "<eps>"}

# Coda phoneme sets.
CODA_VOWELS = {"a", "eo", "o", "u", "eu", "i", "e"}
CODA_36 = {
    "g", "n", "d", "r", "m", "b", "s", "j", "ch", "k", "t", "p", "h",
    "kk", "tt", "pp", "ss", "jj", "K", "N", "T", "L", "M", "P", "NG",
    "a", "eo", "o", "u", "eu", "i", "e", "y", "w",
}
ALLOWED_PH = CODA_36 | {"SP", "AP"}

# IPA vowel bases (diacritics stripped), for the "next phone is a vowel" tests.
IPA_VOWEL_BASE = {"i", "e", "ɛ", "a", "ɐ", "ʌ", "o", "u", "ɨ"}

EPS = 1e-4

# Which segments to assemble, and which alignment dir each lives in.
SAMPLE = [
    ("039d6b6e_0000", "aligned_6s"),   # (a) wordy
    ("039d6b6e_0001", "aligned_6s"),   # (a) wordy
    ("01ed19bb_0019", "aligned_6s"),   # (a) wordy
    ("03abdd70_0069", "aligned_norm"),  # (b) melisma, silence absorbed
    ("02b8fd02_0018", "aligned_norm"),  # (b) melisma
    ("01ed19bb_0007", "aligned_6s"),   # (c) slur: 14 notes vs 10 vowels
    ("02b8fd02_0011", "aligned_norm"),  # (b)+(c): one held vowel over A4->G#4->F4
]


# ---------------------------------------------------------------- IPA -> Coda

def load_mapping() -> Dict[str, str]:
    """mfa_ipa -> coda string (may contain '|' for position or ' ' for glide split)."""
    mapping = {}
    with open(CHECK_DIR / "ipa_to_coda.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["mfa_ipa"]] = row["coda"]
    return mapping


def vowel_base(ipa_label: str) -> Optional[str]:
    """Return the vowel base of an IPA label if it is a vowel, else None."""
    base = ipa_label.replace("ː", "").replace("ʲ", "").replace("ʷ", "")
    return base if base in IPA_VOWEL_BASE else None


def convert_phone(ipa_label: str, next_ipa: Optional[str], mapping: Dict[str, str]) -> List[str]:
    """Apply the IPA->Coda conversion with the two context rules.

    - pipe 'A|B' (nasal/liquid position): next phone is a vowel -> A (onset), else B (coda).
    - space 'X y' / 'X w' (on-glide split): drop the glide when the next phone
      is the vowel 'i' (that ʲ is plain palatalization, not an on-glide).
    """
    coda_str = mapping.get(ipa_label)
    if coda_str is None:
        return [f"<OOV:{ipa_label}>"]

    next_vowel = vowel_base(next_ipa) if next_ipa else None

    if "|" in coda_str:
        onset, coda = coda_str.split("|")
        return [onset if next_vowel is not None else coda]

    tokens = coda_str.split()
    if len(tokens) == 2 and tokens[1] in ("y", "w"):
        if next_vowel == "i":
            return [tokens[0]]          # drop the glide before ㅣ
    return tokens


# ---------------------------------------------------------------- ROSVOT notes

def slice_notes(stem: str, start_sec: float, end_sec: float) -> List[Tuple[int, float, float]]:
    """ROSVOT notes overlapping [start,end], clipped and shifted to local time."""
    midi = pretty_midi.PrettyMIDI(str(MIDI_DIR / f"{stem}_vocal.mid"))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end <= start_sec or note.start >= end_sec:
                continue
            notes.append((
                note.pitch,
                max(note.start, start_sec) - start_sec,
                min(note.end, end_sec) - start_sec,
            ))
    notes.sort(key=lambda x: x[1])
    return notes


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# ------------------------------------------------------------------- assembly

class Phone:
    """A Coda phone with a local-time span, before note merging."""
    def __init__(self, coda: str, start: float, end: float):
        self.coda = coda
        self.start = start
        self.end = end
        self.is_vowel = coda in CODA_VOWELS

    @property
    def dur(self) -> float:
        return self.end - self.start


def build_coda_phones(ipa_intervals, mapping) -> List[Phone]:
    """IPA intervals -> Coda phones. sil kept as a placeholder 'sil'.

    A single IPA interval may split into two Coda tokens (e.g. mʲ -> m y);
    the interval duration is split evenly between them (an approximation that
    does not affect length parity or is_slur, the things we verify).
    """
    phones: List[Phone] = []
    non_sil = [(lab, s, e) for lab, s, e in ipa_intervals if lab.lower() not in SILENCE]

    for idx, (label, start, end) in enumerate(ipa_intervals):
        if label.lower() in SILENCE:
            phones.append(Phone("sil", start, end))
            continue
        # next non-silence IPA label, for the context rules
        next_ipa = None
        for nlab, ns, ne in ipa_intervals[idx + 1:]:
            if nlab.lower() not in SILENCE:
                next_ipa = nlab
                break
        tokens = convert_phone(label, next_ipa, mapping)
        step = (end - start) / len(tokens)
        for k, tok in enumerate(tokens):
            phones.append(Phone(tok, start + k * step, start + (k + 1) * step))
    return phones


def absorb_silence(phones: List[Phone], notes) -> List[Phone]:
    """Silence covered by a ROSVOT note = a held sung note -> fold into the
    preceding phone. Silence with no note = a real pause -> keep as SP.
    """
    out: List[Phone] = []
    for phone in phones:
        if phone.coda != "sil":
            out.append(phone)
            continue
        covered = any(overlap(phone.start, phone.end, ns, ne) > 0.05 for _, ns, ne in notes)
        if covered and out:
            out[-1].end = phone.end        # extend previous phone over the held note
        else:
            out.append(Phone("SP", phone.start, phone.end))
    return out


def note_name(pitch: int) -> str:
    return librosa.midi_to_note(pitch, unicode=False)


def merge_with_notes(phones: List[Phone], notes):
    """Produce parallel field lists by overlapping phones with ROSVOT notes.

    Vowel spanning N notes -> duplicated N times, all sharing one group id so
    the 2nd..Nth are slur continuations. Consonant/glide/SP -> its own group.
    is_slur is derived from group adjacency at the end, which keeps it correct
    even after tiny-phone merging deletes tokens.
    """
    ph_seq, note_seq, midi_dur, ph_durs, groups = [], [], [], [], []
    counter = [0]

    def emit(ph, note, mdur, pdur, group):
        ph_seq.append(ph)
        note_seq.append(note)
        midi_dur.append(round(mdur, 4))
        ph_durs.append(round(pdur, 4))
        groups.append(group)

    for phone in phones:
        if phone.coda == "SP":
            counter[0] += 1
            emit("SP", "rest", phone.dur, phone.dur, counter[0])
            continue

        covering = [(p, s, e) for (p, s, e) in notes
                    if overlap(phone.start, phone.end, s, e) > EPS]

        if not covering:
            counter[0] += 1
            emit(phone.coda, "rest", phone.dur, phone.dur, counter[0])
            continue

        if not phone.is_vowel:
            counter[0] += 1
            pitch, s, e = max(covering, key=lambda n: overlap(phone.start, phone.end, n[1], n[2]))
            emit(phone.coda, note_name(pitch), e - s, phone.dur, counter[0])
            continue

        # vowel: partition its span at note boundaries so durations tile exactly.
        # All pieces share one group id -> pieces after the first are slurs.
        counter[0] += 1
        group = counter[0]
        cuts = {phone.start, phone.end}
        for _, s, e in covering:
            if phone.start < s < phone.end:
                cuts.add(s)
            if phone.start < e < phone.end:
                cuts.add(e)
        edges = sorted(cuts)
        for j in range(len(edges) - 1):
            a, b = edges[j], edges[j + 1]
            pitch, s, e = max(covering, key=lambda n: overlap(a, b, n[1], n[2]))
            emit(phone.coda, note_name(pitch), e - s, b - a, group)

    return ph_seq, note_seq, midi_dur, ph_durs, groups


# ------------------------------------------------------------------- main

MIN_PH_DUR = 0.02  # phones shorter than this become 0 mel frames -> merge away


def merge_tiny(ph_seq, note_seq, midi_dur, ph_durs, groups, min_dur: float = MIN_PH_DUR):
    """Fold any sub-min-duration phone into a neighbour, transferring its time.

    A note boundary landing near a vowel edge (or an MFA floor artifact) can
    produce a ~0s token that would occupy zero mel frames in binarize. We add
    its duration to the previous token (or the next, if it is the first) and
    drop it. Groups ride along so is_slur can be recomputed consistently.
    """
    ph_seq, note_seq, midi_dur, ph_durs, groups = (
        list(ph_seq), list(note_seq), list(midi_dur), list(ph_durs), list(groups))
    i = 0
    while i < len(ph_durs):
        if ph_durs[i] >= min_dur or len(ph_durs) == 1:
            i += 1
            continue
        donor = i - 1 if i > 0 else i + 1
        ph_durs[donor] += ph_durs[i]
        for lst in (ph_seq, note_seq, midi_dur, ph_durs, groups):
            del lst[i]
        i = max(0, donor if donor < i else i)
    return ph_seq, note_seq, midi_dur, ph_durs, groups


def slur_from_groups(ph_seq, groups) -> List[int]:
    """is_slur = 1 on a token that continues the same vowel group as its
    predecessor. Derived after merging, so deletions can't desync it."""
    out = []
    for i, group in enumerate(groups):
        out.append(1 if i > 0 and groups[i - 1] == group else 0)
    return out


def assemble(item_name: str, aligned_subdir: str, seg: Dict, mapping):
    grid = tgio.openTextgrid(str(CHECK_DIR / aligned_subdir / f"{item_name}.TextGrid"),
                             includeEmptyIntervals=True)
    ipa_intervals = [(iv.label.strip(), iv.start, iv.end)
                     for iv in grid.getTier("phones").entries]

    notes = slice_notes(seg["song_stem"], seg["song_start"], seg["song_end"])

    phones = build_coda_phones(ipa_intervals, mapping)
    phones = absorb_silence(phones, notes)
    ph_seq, note_seq, midi_dur, ph_durs, groups = merge_with_notes(phones, notes)
    ph_seq, note_seq, midi_dur, ph_durs, groups = merge_tiny(
        ph_seq, note_seq, midi_dur, ph_durs, groups)
    is_slur = slur_from_groups(ph_seq, groups)
    return (ph_seq, note_seq, midi_dur, ph_durs, is_slur), notes, ipa_intervals


def main() -> None:
    mapping = load_mapping()
    segments = {json.loads(l)["item_name"]: json.loads(l)
                for l in open(DATA / "segments.jsonl")}
    lyrics = {r["item_name"]: r["text"]
              for r in (json.loads(l) for l in open(DATA / "lyrics.jsonl"))}
    # normalized lyrics live in the corpus_norm .lab files
    def text_for(item, subdir):
        corpus = "corpus_norm" if subdir == "aligned_norm" else "corpus_6s"
        p = CHECK_DIR / corpus / f"{item}.lab"
        return p.read_text(encoding="utf-8").strip() if p.exists() else lyrics.get(item, "")

    lines = []
    reports = []
    for item_name, subdir in SAMPLE:
        seg = segments[item_name]
        text = text_for(item_name, subdir)
        (ph_seq, note_seq, midi_dur, ph_durs, is_slur), notes, ipa = assemble(
            item_name, subdir, seg, mapping)

        line = "|".join([
            item_name, text,
            " ".join(ph_seq),
            " ".join(note_seq),
            " ".join(f"{x:.4f}" for x in midi_dur),
            " ".join(f"{x:.4f}" for x in ph_durs),
            " ".join(str(x) for x in is_slur),
        ])
        lines.append(line)

        # per-line validation
        wav_path = (CHECK_DIR / ("corpus_norm" if subdir == "aligned_norm" else "corpus_6s")
                    / f"{item_name}.wav")
        audio_len = sf.info(str(wav_path)).duration
        lens = {"ph_seq": len(ph_seq), "note_seq": len(note_seq),
                "midi_dur": len(midi_dur), "ph_durs": len(ph_durs), "is_slur": len(is_slur)}
        parity = len(set(lens.values())) == 1
        dur_sum = sum(ph_durs)
        dur_ok = abs(dur_sum - audio_len) < 0.02
        oov = sorted({p for p in ph_seq if p not in ALLOWED_PH})
        # note parse check
        note_ok = True
        for n in note_seq:
            if n == "rest":
                continue
            try:
                librosa.note_to_midi(n)
            except Exception:
                note_ok = False
        # is_slur sanity: 1 only on a vowel token whose previous token is the same vowel
        slur_ok = True
        for i, s in enumerate(is_slur):
            if s == 1:
                if ph_seq[i] not in CODA_VOWELS or i == 0 or ph_seq[i - 1] != ph_seq[i]:
                    slur_ok = False

        reports.append({
            "item": item_name, "type": subdir, "text": text,
            "lens": lens, "parity": parity,
            "dur_sum": dur_sum, "audio_len": audio_len, "dur_ok": dur_ok,
            "oov": oov, "note_ok": note_ok, "slur_ok": slur_ok,
            "n_notes": len(notes), "n_slur": sum(is_slur),
            "fields": (ph_seq, note_seq, midi_dur, ph_durs, is_slur),
        })

    (CHECK_DIR / "transcriptions_sample.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_report(reports)

    # console summary
    npass = sum(1 for r in reports if r["parity"] and r["dur_ok"] and not r["oov"]
                and r["note_ok"] and r["slur_ok"])
    print(f"조립 {len(reports)}개, 전체검증 PASS {npass}/{len(reports)}")
    for r in reports:
        flags = []
        if not r["parity"]: flags.append("길이불일치")
        if not r["dur_ok"]: flags.append(f"합{r['dur_sum']:.2f}≠{r['audio_len']:.2f}")
        if r["oov"]: flags.append(f"OOV{r['oov']}")
        if not r["note_ok"]: flags.append("note파싱")
        if not r["slur_ok"]: flags.append("slur오류")
        status = "PASS" if not flags else "FAIL:" + ",".join(flags)
        print(f"  {r['item']:16s} 음표{r['n_notes']:2d} slur{r['n_slur']:2d}  {status}")


def write_report(reports) -> None:
    lines = ["# transcriptions.txt 7필드 조립 검증 리포트", ""]
    npass = sum(1 for r in reports if r["parity"] and r["dur_ok"] and not r["oov"]
                and r["note_ok"] and r["slur_ok"])
    lines += [f"**{len(reports)}개 중 전체검증 PASS {npass}개**", ""]
    lines += ["| item | 유형 | 길이정합 | ph_durs합 | is_slur | OOV | note |",
              "|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(
            f"| {r['item']} | {r['type']} | {'✅' if r['parity'] else '❌ '+str(r['lens'])} | "
            f"{'✅' if r['dur_ok'] else '❌'} {r['dur_sum']:.2f}/{r['audio_len']:.2f} | "
            f"{'✅' if r['slur_ok'] else '❌'}({r['n_slur']}) | "
            f"{'✅' if not r['oov'] else '❌'+str(r['oov'])} | {'✅' if r['note_ok'] else '❌'} |")

    # human-readable slur breakdown for the (c) cases
    lines += ["", "## 슬러(시김새) 케이스 상세", ""]
    for r in reports:
        if r["n_slur"] == 0:
            continue
        ph, note, mdur, pdur, slur = r["fields"]
        lines += [f"### {r['item']}  \"{r['text'][:24]}\"  (음 {r['n_notes']}개, slur {r['n_slur']}개)", "",
                  "| # | ph | note | midi_dur | ph_dur | slur |", "|--|--|--|--|--|--|"]
        for i in range(len(ph)):
            mark = " ⬅ 시김새" if slur[i] == 1 else ""
            lines.append(f"| {i} | {ph[i]} | {note[i]} | {mdur[i]:.2f} | {pdur[i]:.2f} | {slur[i]}{mark} |")
        lines.append("")

    # verdict
    all_pass = npass == len(reports)
    lines += ["## 최종 판정", ""]
    if all_pass:
        lines += ["**PASS — 394 확장 가능.** 모든 세그먼트가 길이 정합·ph_durs 합·is_slur·OOV·note 파싱을 통과했다."]
    else:
        lines += ["**FAIL — 아래 항목을 고친 뒤 재검증.** 394로 넘어가지 말 것.", ""]
        for r in reports:
            probs = []
            if not r["parity"]: probs.append(f"길이불일치 {r['lens']}")
            if not r["dur_ok"]: probs.append(f"ph_durs합 {r['dur_sum']:.3f}≠오디오 {r['audio_len']:.3f}")
            if r["oov"]: probs.append(f"OOV {r['oov']}")
            if not r["note_ok"]: probs.append("note 파싱 실패")
            if not r["slur_ok"]: probs.append("is_slur 위치 오류")
            if probs:
                lines.append(f"- **{r['item']}**: {'; '.join(probs)}")
    (CHECK_DIR / "FIELD_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
