# Lyricbook snapping verdict — does correcting the lyrics improve MFA alignment?

**Verdict: ❌ it does not.** Text corrected against the *changbon* (창본, the published
libretto) makes MFA alignment **worse**. The tooling works and the corrections are mostly
right, but they do not help this pipeline's purpose. `04_align.py`'s default input stays
`lyrics.jsonl`.

Verified 2026-08-26 / MFA 3.4.1 / korean_mfa v3.0 / sample of 200 segments

---

## 1. What was built

| File | Role |
|---|---|
| `lyricbook/build_corpus.py` | 15 lyricbooks → `corpus.jsonl` (713 blocks, 366 sung, 95,845 sung syllables) |
| `preprocess/03b_snap_lyrics.py` | Per-song anchoring + per-segment snapping → `lyrics_snapped.jsonl` |
| `eval/compare_lyrics_alignment.py` | Aligns the same clip with both lyric sources and compares |
| `preprocess/04_align.py` | Added `--lyrics` / `--suffix` / `--items` (default behaviour unchanged) |

Full run: **11,174 of 31,529 segments (35.4%)** across 295 songs were corrected.

### Incidental finding — the 15 lyricbooks are really 6

Within a given *bata* (바탕, the work/repertoire piece) similarity runs 0.998–0.9997 — 심청가
and 심청가_성창순 are byte-identical. The one exception is 춘향가_성우향_김세종제, which
shares 0.37 with the other 춘향가. So of the five bata, only 춘향가 has two changbon; the
rest have one each.

### Automatic changbon selection was accurate

The bata in the filename was **never** fed into the matching, so the filename acts as an
answer key. Opening the gate and running all 387 songs:

| Contrast vs background | Songs | Bata hit rate | Segment adoption rate |
|---:|---:|---:|---:|
| 0.0-1.2 | 31 | 36% | 3.1% |
| 1.2-1.5 | 61 | 75% | 7.8% |
| 1.5-1.8 | 39 | **95%** | 16.3% |
| 1.8-2.0 | 26 | **100%** | 25.3% |
| 2.0+ | 230 | **100%** | 50.4% |

With the gate at 1.5, **the changbon choice for all 295 songs matches the filename 100% of
the time**. The last column is worth reading too — even where the anchor is wrong, only
3.1% of segments get through. That means the second-stage similarity check really does undo
bad anchors.

---

## 2. The corrections themselves are right

`fe9992ba` (Kim Su-yeon · Simcheongga, the passage where Cheong goes begging for rice),
72 of 123 adopted:

```
자료난 현인으로        →  자로는 현인으로
백리의 품이 오고        →  백리에 부미하고
순이웃딸 태영이는 나영   →  순우의 딸 제영이는 낙양
빈 불만 못 어릴까       →  미물만 못 하리까
한 물 며오ㅈㅈㅈㅈㅈ…   →  하물며
```

Even the clips that got the *worst* are visibly correct as corrections —
`극공단 맞이` → `흑공단 바지`, `길어 길은` → `기러기는`.

---

## 3. And yet the alignment gets worse

The same 182 clips were aligned with each lyric source. Everything but the `.lab` is
identical.

| Metric | whisper | snapped | Change |
|---|---:|---:|---:|
| Fraction at the 0.03 s floor | 34.03% | 38.54% | **+13%** |
| 1s+ silence (mean) | 1.15s | 1.40s | **+21%** |
| Overall silence fraction | 51.91% | 57.67% | +11% |
| Phonemes aligned | 13.66 | 12.59 | −8% |

Per clip: 65 improved / 99 worsened / 18 unchanged. **Below the pass line (floor −10%).**

### Two reasons to believe the experiment

1. **The control matches.** Since `wavs/` had already been cleaned up, the sample was re-cut
   from the source audio. The whisper alignment of that 200-item corpus reproduces the
   original 31,529-item corpus's result — floor 34.03 vs 34.72, 1s+ silence 1.15 vs 1.21,
   phoneme count 13.66 vs 13.66.
2. **No non-determinism.** For the 7 clips whose text did not change by a single character,
   every metric changed by exactly 0.00.

---

## 4. Cause — three of four hypotheses rejected

| Hypothesis | Test | Result |
|---|---|---|
| More word boundaries let MFA hide time in silence | Δword count vs Δfloor correlation | **Rejected** (+0.046; word count 3.10→3.08) |
| Archaic and dialectal lyricbook words are OOV and smear into `spn` | Checked against the korean_mfa dictionary | **Rejected** (OOV 70.3% vs 71.4%, effectively identical) |
| The text is shifted along time and neighbouring clips' content bleeds in | The 0.85–0.95 similarity band | **Rejected** (+6.1% in a band where shifting is impossible) |
| The content substitution itself fits the acoustics less well | Identical vs differing syllables | **Holds** (identical 0.0% / differing +4.4%) |

`spn` (a word MFA acoustically gave up on) rose 23%, from 164 to 202. OOV stayed put while
`spn` alone rose, which means this is not about whether the word is in the dictionary —
MFA judged that **that word does not fit this sound**.

### Why

MFA is an acoustic aligner. **Whisper's transcription came out of the sound of this very
recording; the lyricbook came from a different person's different performance.** Wherever
the two diverge, the Whisper side is by definition closer to the sound actually present in
this audio. In pansori that gap is especially wide — Jeolla dialect, archaic diction, vocal
transformation and sigimsae are all realised differently from the spelling on the page.

So **MFA metrics cannot decide "which lyrics are correct".** They structurally favour text
derived from the audio. This experiment's pass line was built on top of that bias.

### The question that remains

Whether the DiffSinger acoustic model wants "semantically correct lyrics" or "a phoneme
sequence matching the sound that was actually made" is not something this experiment
answers. If agreement between the training labels and the audio is what matters, the Whisper
side may in fact be better. Deciding that means training on both label sets and comparing
the synthesis, which is outside this verification's scope.

---

## 5. There is no salvageable subset

A signal was sought that would pick out the 65 improved clips. There is none.

| Feature | Correlation with the floor-rate change |
|---|---:|
| Similarity (`match_score`) | +0.020 |
| Song anchor density | +0.069 |
| Clip length | −0.076 |
| Whisper syllable count | +0.057 |
| Original floor rate | −0.420 ← a regression-to-the-mean artefact |

Only the last one correlates, and since a low starting value has nowhere to go but up, it is
not a usable signal. **The route of raising the adoption threshold and applying the
correction partially is closed.**

---

## 6. Limits of this verdict

- **The sample is 200.** That is 0.6% of 31,529, and 1.8% of the 11,174 corrected.
- The 38 words where `spn` increased were not listened to individually. MFA's judgement of
  "does not fit the acoustics" was taken at face value.
- **No listening verification was done.** Nobody has checked by ear which of the before/after
  texts matches the actual singing. §2's "the corrections are right" means only "they agree
  with the lyricbook".
- The MFA floor rate is 34% even on the whisper side — higher than the 25.8% that
  SEGMENTATION_VERDICT.md measured under 6 seconds. In other words **neither one is a good
  alignment.** This compared two bad alignments.

---

## 7. Recommendations

1. **Keep `04_align.py`'s default input as `lyrics.jsonl`.** It already is.
2. **Keep `lyrics_snapped.jsonl` and the tooling.** Every segment carries `source`,
   `match_score` and `book_ref`, so the judgement can be revisited at any time, and the
   original was never touched.
3. **Do not use MFA metrics to judge lyrics again**, because of the bias in §4. The next
   judgement has to be listening verification, or a synthesis comparison after training.
4. **Look at the bigger problem first.** Even on the whisper side the floor rate is 34% and
   silence is 51.9%. There is far more headroom in fixing the alignment itself than in
   changing the lyrics.

## 8. Outputs

| File | Contents |
|---|---|
| `lyricbook/corpus.jsonl` | The 6 cleaned changbon, 713 blocks |
| `data/full/lyrics_snapped.jsonl` | 31,529 rows, of which 11,174 are `source: "book"` |
| `data/full/lyric_eval/metrics.json` | Raw per-clip metrics (whisper / snapped) |
| `data/full/lyric_eval/aligned_*/` | Both sides' TextGrids — the starting point for listening verification |

---

## 9. Follow-up — decided by training (2026-08-29)

This answers the question §4 left open: **"does the DiffSinger acoustic model want
semantically correct lyrics, or a phoneme sequence matching the sound that was actually
made?"** Both label sets were trained and their synthesis compared.

**Verdict: the correction does not help.** Voiced/unvoiced agreement gets significantly
worse, and only on utterances whose lyrics actually changed. That is the same direction seen
in the MFA metrics, and this time it comes from a place where the bias §4 identified — MFA
favouring text derived from the audio — cannot intervene.

### 9-1. Design — everything but the labels is the same

| | whisper | snapped |
|---|---|---|
| Corpus | `data/pansori`, 26,955 utterances | `data/pansori_snap`, 26,779 utterances (a subset of whisper's) |
| Audio | The same files (`08_export.py --wavs_from` symlinks them) | ← |
| Spec stats, hold-out, config | Identical (`pansori/spec_stats.yaml` reused) | ← |
| acoustic | `pansori_ac_v1` 28k (lowest validation) | `pansori_ac_snap` 36k (lowest validation) |

Of the 26,779 shared utterances, **40.5% (10,857)** actually use changbon text.

Evaluation is **60 pairs × 3 seeds** drawn from the hold-out songs, rendering only the
`2_resynth` rung — since real `ph_dur` and real f0 are supplied, only the acoustic model and
its phoneme sequence differ, and the variance head's sampling noise does not mix in. The
median over the 3 seeds is one utterance's value, and a sign test is applied.
`pansori/examples/ab_ladder.py`.

**Half of the 60 pairs are a control group** — utterances whose phoneme sequences do not
differ by a single character across the two corpora. Since both are measured with the same
two models, incidental differences between the training runs appear equally in both groups.
Only a difference that appears in the treatment group alone can be attributed to the lyrics.

### 9-2. Results

| Group | Metric | whisper | snapped | snapped wins | p |
|---|---|---:|---:|---:|---:|
| **Lyrics changed (30 pairs)** | mel L1 | 1.097 | 1.101 | 11/30 | 0.200 |
| | f0 RMSE ¢ | 42 | 39 | 14/30 | 0.856 |
| | **Voiced/unvoiced agreement** | **0.952** | **0.950** | **6/27** | **0.006** |
| Control (30 pairs) | mel L1 | 1.070 | 1.112 | 12/30 | 0.362 |
| | f0 RMSE ¢ | 39 | 43 | 11/30 | 0.200 |
| | Voiced/unvoiced agreement | 0.975 | 0.969 | 12/22 | 0.832 |

**Only one metric separates, and it is present only in the treatment group.** On utterances
whose lyrics changed, whisper wins 21–6 with 3 ties (median −0.53 pp); in the control it is
10–12 with 8 ties (median 0.00). Under Holm correction, treating the treatment group's 3
tests as the primary family gives p = 0.018; treating all 9 as one family gives p = 0.054,
which is borderline. **The effect is small** — 0.5 pp of voiced/unvoiced agreement.

### 9-3. Why voiced/unvoiced

Voiced/unvoiced agreement measures whether the model's judgement of "should this frame make
sound" matches the original. Changing the phoneme sequence changes that judgement directly —
when the changbon corrects `극공단 맞이` to `흑공단 바지`, the model places its plosives and
fricatives elsewhere. As §2 showed, that correction **is right on paper.** But the sound
actually made at that point is the sound the singer sang, not the changbon's spelling.

### 9-4. What this verdict does not answer

- **There is only one training run per arm.** The control group filters out global
  differences between runs, but the possibility that the interaction itself — "worse only in
  the treatment group" — is chance can only be ruled out by running more.
- Only the `2_resynth` rung was measured. Whether full inference, including duration and
  pitch prediction, goes the same way is not something this table says (on a 3-utterance
  ladder it was buried in seed noise).
- **There is still no listening verification.** §6's limit remains exactly as it was.

### 9-5. Recommendations (these reinforce §7 rather than replacing it)

1. Keep `04_align.py`'s default input as `lyrics.jsonl` — there are now two grounds for it.
2. Keep `lyrics_snapped.jsonl` and `data/pansori_snap`. Changbon alignment itself is still
   right for non-label purposes such as **lyric search, subtitling and bata identification**
   (§2; automatic changbon selection was 100% accurate).
3. To dig further from here, either run 2–3 training runs per arm to confirm the
   interaction, or move to listening verification. Squeezing more out of the metrics is
   exhausted.
