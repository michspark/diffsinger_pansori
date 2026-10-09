# Verdict on plan A — can korean_mfa forced alignment give us pansori `ph_durs`?

> **Superseded in part by §6 (2026-10-09).** On the corpus, MFA does not collapse past
> 6 s: aligning the same audio as one 6–20 s clip instead of its 2–6 s parts adds +1.9
> points of floor phones and +0.4 points of voiced silence. The 6 s cap stands, but not
> for the reason §2 gives.

**Verdict: 🔶 plan A holds, conditionally** — the phoneme table is fine. Alignment quality
**depends strongly on utterance length**, and **it holds when cut to 6 seconds or less and
collapses above that**.

Verified 2026-07-20 / MFA 3.4.1 / korean_mfa acoustic · dictionary · g2p v3.0 (trained
2024-02-17)

---

## 1. Axis A — phoneme-table correspondence: ✅ holds

The `korean_mfa` acoustic model has **108** actual phonemes (15 vowels + 93 consonants).
Checked against all of Coda 36:

| Grade | Count | Meaning |
|---|---:|---|
| ✅ clean | 62 | Unconditional 1:1 |
| 🔶 needs_rule | 46 | Needs a contextual rule |
| ❌ unmapped | **0** | None |

**No Coda 36 symbol is unreachable.** The conversion table is `ipa_to_coda.csv`.

### Three things that must be respected

**① Symbol collisions — never do string matching**

| MFA | Actual grapheme | Coda |
|---|---|---|
| `k` | **ㄱ** | `g` |
| `kʰ` | ㅋ | `k` |
| `p` | **ㅂ** | `b` |
| `pʰ` | ㅍ | `p` |
| `t` | **ㄷ** | `d` |

MFA's `k` is ㄱ; Coda's `k` is ㅋ. The same letter means opposite things, so conversion must
go through an explicit correspondence table and nothing else.

**② On-glides are fused into the consonant — a decomposition rule is required**

With **no** onset the semivowel comes out as its own phoneme; **with** one it fuses into the
preceding consonant.

```
야    = j ɐ        → y a          (independent)
향    = ç ɐ ŋ      → h y a NG     (ç = ㅎ + palatalisation; y must be inserted)
과일  = kʷ ɐ i ɭ   → g w a i L    (ʷ → split off as w)
교육  = c oː j u   → g y o  y u   (c = ㄱ + palatalisation)
```

⚠️ **Trap**: palatalisation happens not only before an on-glide (ㅑㅕㅛㅠ) but also **simply
when ㅣ follows**. Same symbol, different output.

```
소리   = sʰ o ɾʲ i   → s o r i     (ɾʲ followed by i → do not insert y)
가려야 = k ɐ ɾʲ ʌ j ɐ → g a r y eo y a  (ɾʲ followed by ʌ → insert y)
```

→ **Rule: if the vowel after `Cʲ` is `i`, drop the `y`; otherwise insert it.**

**③ Nasals use the same symbol for onset and coda**

`m`, `n` and `ɲ` do not distinguish position. The G2P output proves it.

```
아니리  = ɐ ɲ i ɾ i          ← ɲ is an onset (a vowel follows)
계면조  = c eː mʲ ʌ ɲ dʑ o   ← ɲ is a coda (a consonant follows)
```

→ **Rule: a following vowel makes it an onset (`n`/`m`), otherwise a coda (`N`/`M`).**
`ŋ` only ever occurs as a coda in Korean, so it is unconditionally `NG` ✅.

### Incidental confirmations

- Coda plosives are distinguished by their own symbols: `k̚ t̚ p̚` → `K T P` ✅
  (`박` = `p ɐ k̚`)
- Liaison is reflected: `국악` = `k u ɡ ɐ k̚` ✅
- Long vowels and long consonants (`iː`, `mː`) can be dropped from the symbol — length is
  carried separately by `ph_durs`
- At 21,009 entries the dictionary's coverage of pansori vocabulary is low, but **the G2P
  model handles OOV well** (`시김새` → `ɕʰ iː ɟ i m sʰ ɛː`, `춘향` → `tɕʰ u n ʝ ɐ ŋ`)

---

## 2. Axis B — singing alignment quality: ⚠️ length-dependent

**Alignment succeeded 10/10, with 0 failures.** But succeeding and being accurate are not
the same thing.

### Two indicators of collapse

1. **Fraction at the 0.03 s floor** — when MFA gives up on aligning, it pushes the phoneme
   into the minimum duration (0.03 s)
2. **Silence of 1 second or more** — if there is more than a second of silence in the middle
   of a sung section that was itself cut on silence, the aligner failed to align a sustained
   sound and discarded it

### Measurements

| Length | item | Phonemes | Floor % | 1s+ silence |
|---:|---|---:|---:|---:|
| 2.45 | 02b8fd02_0001 | 21 | 28.6% | 0.00s |
| 3.07 | 039d6b6e_0000 | 21 | 9.5% | 0.00s |
| 4.10 | 02b8fd02_0012 | 20 | 40.0% | 0.00s |
| 4.59 | 03abdd70_0037 | 23 | 26.1% | 0.00s |
| 5.32 | 039d6b6e_0004 | 40 | 25.0% | 0.00s |
| 6.04 | 03abdd70_0043 | 13 | 38.5% | 2.88s |
| 6.72 | 01ed19bb_0004 | 30 | 26.7% | 1.42s |
| 10.45 | 039d6b6e_0005 | 52 | 26.9% | 1.12s |
| 10.50 | 037e72a6_0079 | 34 | 50.0% | **4.42s** |
| 14.46 | 01ed19bb_0006 | 40 | 57.5% | **10.46s** |

| Bucket | Mean floor % | Mean 1s+ silence |
|---|---:|---:|
| **6 s or under (5 items)** | 25.8% | **0.00s** |
| **Over 6 s (5 items)** | 39.9% | **4.06s** |

**The boundary falls clearly at 6 seconds.** Under 6 s there is not a single instance of
silence over one second.

### What collapse actually looks like (`01ed19bb_0006`, 14.46 s)

```
  0.31-  1.40   the opening phonemes (some of them fine)
  1.40-  5.51   4.11 s of silence      ← mid-song. A discarded span
  5.51-  5.68   k ʌː ʎː ʌ t̚ all at 0.03 s ← "걸렸다" compressed into 0.17 s
  8.69- 10.96   2.27 s of silence
 11.40- 12.81   1.41 s of silence
 13.04- 14.46   1.42 s of silence
```

23 of the 40 phonemes (57.5%) sit on the floor, and 10.46 of the 14.46 seconds were treated
as silence.

### There are successes too (`02b8fd02_0012`, 4.10 s)

```
0.42-1.84   ŋ  1.42 s   ← the coda ㅇ of "대왕" captured accurately as a 1.42 s sustain
```

**Sustained singing does get captured correctly in some cases.** So the acoustic model is
not fundamentally incapable of singing; it **collapses under accumulated error as length
grows**.

### What a mean phoneme length of 107 ms means

The overall mean across all 10 is 107 ms, which is **a typical figure for speech data**. In
singing, vowels should stretch to seconds. It is evidence that the acoustic model is trying
to interpret pansori as "speech, spoken fast".

---

## 3. Limits of this verdict (stated honestly)

- **A sample of 10** is small. The 6-second boundary is a strong hypothesis, not a
  determined threshold.
- **The lyrics are Whisper transcriptions.** If Whisper missed lyrics, those spans register
  as silence, so some of the silence may not be MFA's fault. That said, the 10.46 s of
  silence in `01ed19bb_0006` is not explained by this alone.
- **Listening verification has not been done.** Final confirmation means opening
  `alignment_review.html` and listening.
- Even under 6 s, a **floor rate of 25.8%** is not low. It means "did not collapse", not
  "is accurate".

---

## 4. Recommended actions

### Immediately (keeping plan A)

1. **Lower the length cap in `01_segment.py`.** Currently `MAX_UTTERANCE_SEC = 15.0` /
   `TARGET_UTTERANCE_SEC = 10.0` → re-split at **`MAX = 6.0` / `TARGET = 4.5`**.
   - The current median of 7.06 s and p90 of 12.67 s land squarely in the collapse range
   - Shorter segments also raise Whisper's accuracy (less context, but less spread of
     mis-transcription)
2. **Re-verify on 50–100 samples after re-splitting** to settle the 6-second hypothesis.
3. **Implement the 46 `needs_rule` entries of `ipa_to_coda.csv` as an actual conversion
   function** (Stage 4 work). Rules ①②③ above are all of it.
4. **Add the floor rate to the Stage 7 validation checks.** Discard any utterance whose floor
   rate exceeds a threshold (say 40%). There are 37 hours, so discarding is affordable.

### If it still collapses after re-verification

5. **`mfa adapt`** — before full retraining (plan B), there is an intermediate step of
   *adapting* the pre-trained acoustic model to pansori. Far cheaper than plan B.
6. Failing that, the plan's **fallback B** — align only down to the syllable and distribute
   within the syllable by jamo proportion.

### Plan C is unnecessary

Since `unmapped` is 0 and all of Coda 36 is reachable, there is no reason to take plan C and
use MFA phonemes directly as `ph_seq`. Coda notation is easier for a human to read and is
compatible with the OpenVPI family.

---

## 5. Outputs

| File | Contents |
|---|---|
| `mfa_phone_inventory.txt` | The 108 korean_mfa phonemes |
| `mfa_acoustic_inspect.txt` | Raw `mfa model inspect` output |
| `dict_samples.txt` | Hangul→IPA dictionary examples + OOV G2P results |
| `ipa_to_coda.csv` | The conversion table, 108 rows (clean 62 / needs_rule 46 / unmapped 0) |
| `build_mapping.py` | The script that generates the table (the rules documented as code) |
| `alignment_stats.json` | Raw alignment-quality metrics |
| `analyze_alignment.py` | The quality measurement script |
| `alignment_review.html` | **For listening verification** — audio + phoneme-boundary visualisation |
| `aligned/*.TextGrid` | The 10 MFA alignment results |
| `corpus/` | The input (wav + lab) |

**Next action: open `alignment_review.html` in a browser and check that the five items under
6 s actually sound right.** If they do, proceed with recommendation 1 (re-splitting).


---

## 6. Measured on the corpus (2026-10-09): no cliff at 6 s

§3 asked for the 6 s boundary to be re-checked on more than ten clips. The pilot's clips
and alignments no longer exist, and 01_segment.py must not be re-run, so the long clips are
rebuilt from the current corpus: 2–4 consecutive utterances of one song, same mode, joined
wherever the segmenter had cut them only for length (every gap ≥ 0 s and < 0.4 s, the
pause threshold). Each merged span is re-cut from the source audio and aligned once by MFA
with korean_mfa and the parts' own Whisper text joined, so the comparison is paired: same
audio, same words, same aligner, only the length differs.

Sample: 400 spans per 2 s bucket from 6 to 20 s (2,800 spans, 379 recordings, no utterance
in two spans, songs drawn round-robin), from 28,427 candidates. Script:
`pansori/experiments/long_utterance_mfa.py`; numbers: `pansori/paper/numbers/long_utterance_mfa.json`.

**Metrics.** *Floor* = non-silence phones ≤ 0.031 s, % of phones (the §2 metric, from MFA's
own TextGrid). *Voiced silence* = % of ROSVOT-note-covered time MFA labels silence — the
quantity 06_merge absorbs. §2's "silence ≥ 1 s" is not used: it grows with clip length by
construction. On the corpus's own 2–6 s utterances it is already present in 42–75% of
clips, which contradicts §2's "not a single instance" under 6 s:

| Length | n | Floor % | Voiced silence % | Has silence ≥ 1 s % |
|---|---:|---:|---:|---:|
| 2-3 s | 5,466 | 35.2 | 50.2 | 42.5 |
| 3-4 s | 10,224 | 36.6 | 51.9 | 58.1 |
| 4-5 s | 6,594 | 37.0 | 53.1 | 68.5 |
| 5-6 s | 4,671 | 36.5 | 53.3 | 75.0 |

**Paired result** (merged minus its own parts, mean difference with 95% bootstrap CI over
spans; MFA failures = spans with no TextGrid, every part of which did align):

| Length | n | Floor % parts → merged | Δ floor | Voiced sil. % parts → merged | Δ voiced sil. | MFA failed |
|---|---:|---:|---:|---:|---:|---:|
| 6-8 s | 391 | 35.1 → 36.4 | +1.3 [+0.3, +2.3] | 50.7 → 50.3 | -0.4 [-1.3, +0.5] | 9/400 |
| 8-10 s | 381 | 37.0 → 38.7 | +1.8 [+0.8, +2.7] | 52.8 → 53.3 | +0.5 [-0.5, +1.4] | 19/400 |
| 10-12 s | 379 | 34.3 → 36.1 | +1.8 [+0.8, +2.7] | 49.3 → 49.3 | -0.0 [-0.9, +0.9] | 21/400 |
| 12-14 s | 382 | 34.8 → 36.3 | +1.5 [+0.7, +2.4] | 51.1 → 51.1 | +0.0 [-0.9, +1.0] | 18/400 |
| 14-16 s | 374 | 33.3 → 35.1 | +1.9 [+1.1, +2.6] | 49.7 → 50.3 | +0.7 [-0.2, +1.5] | 26/400 |
| 16-18 s | 377 | 34.1 → 36.7 | +2.6 [+1.8, +3.3] | 49.7 → 50.4 | +0.7 [-0.1, +1.6] | 23/400 |
| 18-20 s | 383 | 31.0 → 33.6 | +2.6 [+1.9, +3.3] | 47.8 → 49.1 | +1.3 [+0.6, +2.0] | 17/400 |
| **all** | 2667 | 34.2 → 36.1 | **+1.9** [+1.6, +2.2] | 50.2 → 50.6 | **+0.4** [+0.1, +0.7] | 133/2800 |

For reference, MFA failed on 2.8% of the corpus's alignable 2–6 s utterances (765 of 27,720).

**Reading.**
- Floor phones rise by 1–3 points with length and keep rising (+1.3 at 6–8 s, +2.6 at
  16–20 s). Real, small, and gradual: no step at 6 s.
- Voiced silence does not move: +0.4 points over all spans, inside ±1 point in every bucket
  up to 18 s. MFA mislabels about half of sung time as silence at every length; that is
  why 06_merge absorbs it and why SOFA replaced MFA, not a long-clip effect.
- MFA's outright failure rate rises from 2.8% (2–6 s) to 2.3–6.5% (6–20 s).
- "Swallowed spans" (§2's `01ed19bb_0006`) are not a length effect. A merged clip's longest
  silence exceeds the parts' (on one timeline, gaps counted) by ≥ 1 s in 7–19% of spans;
  the parts' exceeds the merged clip's by ≥ 1 s in 12–20%. It is noise in both directions.
- Zero-gap joins (a held tone cut mid-note, 979 spans) and pause joins (1,688) agree:
  Δ floor +2.1 and +1.8.

**Why the pilot differed** (not measured): the pilot's long clips came from a 15 s-cap
segmentation and were transcribed by Whisper as long clips, so missing lyrics would have
been aligned as silence, the confound §3 already named. Here the long clips carry the
short clips' transcriptions.

**Consequence.** The 6 s cap is not justified by MFA. It stays because (a) it is what every
downstream label was built on, (b) MFA's failures and floor phones do rise slowly with
length, and (c) Whisper and ROSVOT are not tested here. Text citing "MFA collapses past
6 s" (PIPELINE.md, the paper's Segmentation paragraph) should say instead what this section
measured.
