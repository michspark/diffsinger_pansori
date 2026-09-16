# Melody — generating the note skeleton from lyrics

Everything downstream of `make_ds.py from-score` already works: hand it a text score and it
becomes a `.ds`, then f0, then a mel, then a wav ([`PIPELINE.md`](PIPELINE.md) §D). What does
not exist is the thing that *writes* the score. Today a human does, or the notes are copied
out of an existing recording's labels.

This is the plan for a model that writes it, and the record of the one stage that is built.

The target is not "a melody" but **a melody in a named mode** — ujo, gyemyeonjo — because that
is the axis a pansori score is actually chosen along, and because the corpus already carries
the label: `segments.jsonl` has a human-annotated `mode` per utterance.

---

## 0. What is settled, and what is still a question

**Settled — the output format.** The generator emits the text score `make_ds.py from-score`
already reads (`syllable note duration_seconds`, `-` for a held syllable, `_` for a rest). So
the seam between the new work and the working pipeline is a file format that exists, and the
melody model can be replaced wholesale without touching anything below it.

**Settled — the two decisions inside stage [2].**

- **Note duration is a classification, not a regression.** A syllable's length is not a
  function of its lyrics — the singer decides — so L1 or MSE is minimised by the mean and the
  model returns 0.3 s every time. The same failure the f0 predictor already has
  ([`SIGIMSAE_PLAN.md`](SIGIMSAE_PLAN.md): the model moves 12–20% as much as ground truth).
  Durations go into **32 quantile bins**; measured requantisation error is 4.5 ms at the median
  (16 bins: 6.8 ms, 64 bins: 2.0 ms). 4.5 ms is under one frame at hop 512 / 44.1 kHz, so the
  binning gives up nothing the binariser would have kept.
- **Notes per syllable is emitted, not predicted as a count.** One note at a time, then
  `CONT` or `NEXT`. 66.6% of syllables take one note but 6.3% take six or more, so a fixed
  count vocabulary truncates the very thing — sigimsae — the model exists to produce. `CONT`
  is the text score's `-` exactly.

Because the destination is a `.ds` measured in **seconds**, none of this needs a beat grid,
which is fortunate: the corpus has no jangdan annotation. The bar/position tokens every
lyric-to-melody paper uses are a detour we do not have to take.

**The open question, and it is the one that decides whether any of this is worth building:**
does the mode live in the *notes* at all? Measured on the raw note sequence, before any
tonic-relative conversion, ujo and gyemyeonjo are nearly indistinguishable:

| mode | repeat (0) | +1 | −1 | −2 | +2 | leaps ≥5 |
|---|---|---|---|---|---|---|
| gyemyeonjo | 27.6% | 12.7% | **17.1%** | 10.3% | 5.5% | 11.3% |
| ujo | 27.1% | **14.9%** | 14.7% | 8.2% | 6.8% | 13.6% |
| pyeongjo | 27.9% | 13.4% | 16.4% | 8.6% | 6.9% | 12.6% |

Only the direction separates them — gyemyeonjo descends more than it rises, ujo does not — and
by 3–4 points. Two reasons it may be hiding rather than absent:

1. **There is no tonic.** A mode is defined against a reference pitch, and our notes are
   absolute names (`A#4`). Every singer and piece sits somewhere different, so one mode's
   skeletal tones land on another's note names. Stage S3 exists to fix this.
2. **Part of the mode is not in the notes.** The bent and shaken tones that identify
   gyemyeonjo are f0 movement inside a note, which in this pipeline belongs to the variance
   model's `base_pitch + deviation`, not to the note skeleton.

**S3 is therefore the gate.** It needs no training — it is an aggregation over notes we
already have. If the fitted per-mode profiles separate, build [2] and [3]. If they do not, the
mode lives in the f0 deviation and the work belongs in variance, not in a new model.

---

## 1. Data preparation

```
 segments.jsonl     mode, uuid, duration
 phones_sofa2.jsonl words tier = the syllable sequence, on the audio clock
 notes.jsonl        ROSVOT notes, on the same clock, having never seen the lyrics
 lyrics.jsonl       the text
         │
         ▼
   S2  bind_notes.py        syllable -> its notes, and the quality verdict      ✅ built
         ▼
   melody_data.jsonl
         │
   S3  tonic estimation     per-piece tonic + per-mode profile                  ⬜ next
   S4  tokenisation         degrees, duration bins, CONT/NEXT                   ⬜
```

### Why it reads `phones_sofa2.jsonl` and not `transcriptions.csv`

The unit this needs is the **syllable**, and only the SOFA arm has it.

| | words tier == the syllable list | phones under 20 ms |
|---|---|---|
| MFA (`phones.jsonl`) | 7.5% | 4.2% |
| **SOFA2 (`phones_sofa2.jsonl`)** | **100.0%** | 24.4% |

SOFA is handed one syllable per dictionary headword (`align.yaml` `lab_unit: syllable`), so its
words tier *is* the syllable sequence. MFA's is in eojeol, and 06_merge only splits an eojeol
back into syllables when the dictionary's predicted phonemes match the actual ones exactly
(`PIPELINE.md` §06 ⑤) — which, measured against the lyrics, leaves 23.2% of utterances at
syllable granularity.

But reading the *merged* output loses it again: `merge_tiny` folds sub-20 ms phones into a
neighbour, and where every phone of a syllable is folded the syllable disappears. SOFA has
24.4% of its phones under that threshold, so the SOFA corpus's `transcriptions.csv` is down to
28.0%. **The 100% is only available above 06_merge**, and since the melody model uses no
phonemes at all, that is where this branches off.

### S2 — binding, `melody/bind_notes.py`

Two timelines that have never met: syllables from the aligner, notes from ROSVOT. Binding them
means cutting the note sequence into one contiguous run per syllable, maximising the total
syllable/note overlap.

**Greedy does not work**, which is the only reason this is a DP. Handing each note to whichever
syllable overlaps it most starves short syllables of everything they share with a longer
neighbour. On `02b8fd02_0000` ("아이고 장군님") it leaves two syllables of six with no note:

```
greedy   아←A4   이←(none)   고←F#4,B3   장←B3,C#4,C#4   군←C4   님←(none)
DP       아←A4   이←F#4      고←B3       장←B3,C#4       군←C#4  님←C4
```

The assignment has to be monotonic and every syllable has to receive at least one note, so it
is the same shape of problem as `make_ds.derive_note_slur` — and for the same reason that
function is a DP too. Running-maximum inner loop, so O(syllables × notes).

**Notes keep ROSVOT's own spans, uncut.** A note reaching past its syllable's end stays whole:
the note is the musical unit, not the syllable.

**Rests are stored separately.** They belong to no syllable and so have no place in the
syllable list, but 74.3% of utterances have an uncovered stretch over 50 ms (8.0% of utterance
time on average). Dropping them would produce a score with nowhere to breathe.

**Tokens are deliberately not stored.** Bin counts and the tonic estimate will move while the
DP above them will not; baking tokens into the file would mean re-running the DP over the
corpus to change a bin count. The file holds MIDI numbers and seconds; S4 is a lookup applied
at load time.

```sh
python pansori/melody/bind_notes.py --set full        # any env
```

One row per utterance, failures included so a threshold can be revisited without a rebuild:

```json
{"item_name": "02b8fd02_0000", "uuid": "02b8fd02", "mode": "계면조",
 "duration": 2.688, "text": "아이고 장군님",
 "syllables": [{"syl": "아", "start": 0.0, "end": 0.2147, "notes": [[69, 0.0, 0.2145]]},
               {"syl": "장", "start": 0.7517, "end": 1.3961,
                "notes": [[59, 0.8372, 1.1304], [61, 1.1304, 1.2804]]}],
 "rests": [],
 "qc": {"min_syl_dur": 0.0784, "n_syl_no_note": 0,
        "n_syllables": 6, "n_notes": 7, "pass": true}}
```

`uuid` is carried because **hold-out splits go by piece, never by utterance** — the same
passage recurs across utterances of one recording, and splitting below the piece puts it on
both sides.

### What survives

| | utterances | hours |
|---|---|---|
| gyemyeonjo | 10,309 | 10.87 |
| ujo | 3,081 | 3.24 |
| pyeongjo | 854 | 0.90 |
| seolleongje | 112 | 0.11 |
| gyeongdeureum | 57 | 0.06 |
| **total** | **14,413 of 26,955 (53.5%)** | |

92,716 syllables · 161,796 notes — **1.75 notes per syllable, 42.2% of syllables carrying more
than one.** That last figure is the whole argument for `CONT`: the one-syllable-one-note
assumption every lyric-to-melody paper makes would discard two fifths of this corpus's melodic
structure.

Why the other 46.5% goes:

| reason | utterances |
|---|---|
| a syllable whose assigned notes do not overlap it | 10,526 |
| a syllable under 0.03 s | 8,103 |
| fewer notes than syllables | 1,707 |

Only the second is SOFA's doing — cramming the tail of an utterance it has lost into a few
milliseconds, the price of the syllable tier it gives in exchange. Running the same binding
over the MFA alignment separates the two:

| | pass | no overlapping note | under 0.03 s | too few notes |
|---|---|---|---|---|
| MFA (eojeol units) | 54.8% | 11,770 | 318 | 226 |
| SOFA2 (syllable units) | 53.5% | 10,526 | 8,103 | 1,707 |

**The dominant cause is not the aligner.** It costs MFA slightly more than SOFA2 despite MFA's
units being coarser, so it is a property of the note track, not of the syllable track. Nor is
it simply syllables sung where ROSVOT found nothing: only 13.1% of them fall inside a rest. It
is undiagnosed, and it is the first thing to look at if 14,413 utterances turn out to be too
few.

The test is deliberately the strict one: if the DP could only satisfy "every syllable gets a
note" by handing one a note it does not overlap, that syllable's melody is a product of the
constraint rather than an observation. Relaxing it to one such syllable per utterance would
recover 1,591 utterances (53.5% → 59.4%); the `qc` fields are stored so that can be
reconsidered without rebuilding.

The unit difference is visible in the totals too: MFA's 14,774 rows carry 68,193 eojeol at 2.43
notes each, SOFA2's 14,413 carry 92,716 syllables at 1.75. Same music, cut at a different
granularity — and only the finer one is a unit the model can be asked to sing.

Separately, 765 aligned utterances have no note slice at all and never enter the file.

---

## 2. The models, when S3 says to build them

Two stages, because rhythm is decided by the lyrics (how many syllables, where the phrase
breaks) and pitch by the mode. Different conditioning, so they are learned apart — the split
TeleMelody and the ISMIR 2022 lyric-to-rhythm model both use.

```
        syllable encoder  ⊕ mode ⊕ tonic
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
 [2] rhythm                [3] pitch
 DUR_0..31, CONT, NEXT     DEG_-12..+24
 NEXT count == syllables   length fixed by [2], so no stop token
```

Both are plain cross-entropy. Decoding for [3] can carry mode constraints as a reward without
any extra training — scale adherence against the S3 profile, a cadential degree at phrase end,
the descending/ascending tendency measured above — which is ReLyMe's method applied to mode
instead of Mandarin tone.

**No speaker conditioning.** In this corpus a speaker is a stand-in for a small set of pieces
(조상현 and 박동진 have 13 each, 김창룡 one), so a speaker embedding would learn the passages
that singer recorded rather than how they sing, and 21% of utterances have no speaker at all.
Mode is already skewed per singer (정정렬 45% ujo, 김창룡 100%), so a speaker token would also
compete with the mode token for the very signal being measured. The voice is chosen later
anyway — `make_ds.py from-score --spk` writes it into the `.ds`, and the personal character
lives in variance and acoustic. If bodi (동편제 / 서편제) were ever labelled per utterance,
that — not the individual — is the axis worth conditioning on.

---

## 3. Order of work

1. **S3, tonic estimation.** Duration-weighted pitch-class histogram per piece and mode span;
   fit the rotation and the per-mode profile together across pieces. No training.
2. **Read the profiles.** Do ujo and gyemyeonjo separate once every piece is on a common
   tonic? Does the fitted profile resemble the skeletal tones the musicology describes?
3. Only then S4 and the two models. If the profiles do not separate, the finding is that mode
   lives in the f0 deviation, and the work moves to variance.
