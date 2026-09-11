---
name: pansori-preprocess
description: Use when working on the data pipeline — segmentation, transcription, forced alignment (MFA or SOFA), note slicing, merging, validation or export — or when fixing the IPA-to-Coda phoneme mapping or regenerating a corpus under data/pansori*.
---

# Preprocessing

Eight stages under `pansori/preprocess/`, joined by jsonl so each can be re-run alone. Re-run
cost differs by ~100× between them, which is why the seams are where they are. All stages work
in `pansori/data/{set}/` and take `--set` (in practice `--set full`).

| # | script | in → out | env |
|---|---|---|---|
| 1 | `01_segment.py` | labels + Audio + midi → `segments.jsonl` + `wavs/` | `pansori` |
| 1b | `01b_recut_wavs.py` | segments → `wavs/` restored | `pansori` |
| 2 | `02_speaker_map.py` | segments → `spk_map.json` | `pansori` |
| 3 | `03_transcribe.py` | wavs → `lyrics.jsonl` (whisper-large-v3) | `pansori` |
| 3b | `03b_snap_lyrics.py` | + 창본 → `lyrics_snapped.jsonl` (optional) | `pansori` |
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` | `aligner` |
| 4b | `04b_sofa_data.py` | phones → SOFA training data (optional) | `pansori` |
| 5 | `05_notes.py` | rosvot_midi → `notes.jsonl` | `aligner` |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | `pansori` |
| 7 | `07_validate.py` | → PASS/FAIL, exit 1 on FAIL | `pansori` |
| 8 | `08_export.py` | 44.1 kHz re-cut + per-speaker fan-out | `pansori` |

## Two rules that are not negotiable

**Never re-run `01_segment.py`.** It does not reproduce. In a different env
`librosa.effects.split` lands boundaries 3–11 ms differently, which silently invalidates every
`phones.jsonl` already aligned against the old cuts — the run completes, nothing errors, and
the labels are now wrong. To get the wavs back:

```sh
python pansori/preprocess/01b_recut_wavs.py --set full    # 31,529 items, 4,574 actually re-cut
```

**Never binarize without a `07_validate.py` PASS.** Upstream's binarizer asserts inside worker
processes, so a bad item kills a 40-minute job midway rather than being skipped. Stage 07
checks every assert the binarizer will hit, in the order it hits them, plus two it does not:
that `sum(ph_dur) ≈ sum(note_dur) ≈ wav length` (0.05 s tolerance) and that no phone rounds to
zero frames.

## The usual loop — fixing labels without re-aligning

Because stage 04 records the MFA `words` tier, a phoneme-mapping fix costs **2 minutes**, not
hours. Alignment does not need to run again.

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_align   # re-parse only, ~2 min
conda activate pansori
python 06_merge.py --set full && python 07_validate.py --set full      # PASS is mandatory
python 08_export.py --set full --out data/pansori                      # re-cut, ~20 min
```

**Before and after any change to `ipa_to_coda.csv` or `convert_phone`, score it:**

```sh
python pansori/alignment/check_dict_agreement.py                   # 1-syllable, currently 89.59%
python pansori/alignment/check_dict_agreement.py --syllables 2 --focus ʎ
```

It works because `dictionaries/korean-coda.txt` is generated mechanically from
`build_dictionary.py` and is completely independent of `ipa_to_coda.csv`, so it can serve as an
answer key. Use it as a **before/after diff, not an absolute score** — most remaining
mismatches are ordinary liaison (은 → `eu n`) that a context-free syllable dictionary cannot
reproduce. 100% is not the goal.

## Stage 4 seam — MFA or SOFA

One line in `pansori/preprocess/align.yaml` (`aligner: mfa | sofa`), overridable with
`04_align.py --aligner {mfa,sofa}`. Backends sit behind `build_corpus` / `run` / `parse` in
`pansori/preprocess/aligners.py`. Both leave `aligned{suffix}/*.TextGrid`, so **nothing from
stage 06 down knows which aligner ran.**

The difference is the phone set: MFA emits IPA, so its `parse` still walks `ipa_to_coda.csv`;
SOFA's `DictionaryG2P` uses dictionary values as its phone set, so handing it
`dictionaries/korean-coda.txt` makes the conversion step disappear entirely.

**SOFA must be trained before it can align.** There is no public Korean checkpoint, the vocab
is baked into the checkpoint (`forced_alignment.py:309` does `self.vocab[ph]`), and running
`--aligner sofa` against the Mandarin checkpoint dies with `KeyError`. Bootstrap its training
data from the MFA alignment with `04b_sofa_data.py` — **not wholesale**, which would teach SOFA
the very 0.03 s floor being escaped; it demotes by per-utterance `floor_rate` into
full/weak/no_label. Train with `-p` on the Mandarin checkpoint anyway: `load_pretrained` keeps
the backbone and re-initializes only the head when vocab size differs.

`check_dict_agreement.py` does not apply to the SOFA arm (no conversion step to score). The
aligner A/B verdict is this table instead:

```sh
python pansori/eval/label_report.py --corpus data/pansori data/pansori_sofa \
       --html pansori/eval/out/label_report.html
```

## Judgment that lives in stage 06

`absorb_silence` is the single most consequential rule in the pipeline. **52.2% of corpus time
is "MFA says silence, ROSVOT says singing"** — held tones the speech aligner gave up on, with
median note coverage 1.00. Silence covered ≥50 ms by a note is absorbed into the previous
phone; only note-free silence stays `SP`. Left alone, you would teach the model to be quiet
while singing. A real `AP` (in-breath, when `breath_ap` is on) is passed through untouched.

## Constraints carried from the verdicts

- **Utterances must be ≤6 s.** Past that MFA collapses: measured, ≤6 s had zero silences over
  1 s while >6 s averaged 4.06 s of silence (`pansori/alignment/VERDICT.md`).
- **Never string-match phone symbols across systems.** MFA `k` is ㄱ, Coda `k` is ㅋ; MFA `p` is
  ㅂ, `t` is ㄷ. Always convert through the explicit table.
- Paths in `labels_2.json` mix NFC and NFD — join on the 8-char uuid prefix, never the filename.
- **Snapping lyrics to the 창본 is a closed question** — it makes alignment measurably worse
  (`pansori/lyricbook/VERDICT.md`). `lyrics.jsonl` stays the default input.
