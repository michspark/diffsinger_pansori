# Port record — MoonInTheRiver → openvpi

A record of moving the pansori pipeline built in `~/Desktop/DiffSinger_moonInRiver` (a
MoonInTheRiver/DiffSinger fork) over to openvpi/DiffSinger. This was a **port, not a
merge** — the two repositories share no common ancestor, and their layout, sample rate and
model structure all differ. `git merge` and `git remote add` simply do not apply.

The pansori work in the original repository was committed as `341e927` before porting.
Before that, most of the pipeline was untracked and a single `git stash` could have erased
it.

---

## 1. Correspondence

| MoonInTheRiver | openvpi | Action |
|---|---|---|
| 24 kHz / 80 mel / hop 128 | 44.1 kHz / 128 mel / hop 512 | Source audio is 44.1 kHz, so re-cut losslessly |
| 7-field `transcriptions.txt` (`is_slur`, `midi_dur`) | `transcriptions.csv` (`ph_num`, `note_seq`, `note_dur`) | Rewrote 06 |
| Hardcoded `CODA_36` | `dictionaries/*.txt` + a language tag | New `dictionaries/korean-coda.txt` |
| Speaker override in `usr/pansori_binarizer.py` | `speaker`/`spk_id` under `datasets:` | Deleted; per-speaker directories instead |
| `data_gen_utils.py` f0_min patch | `f0_min`/`f0_max` are already config keys | No patch needed |
| CREPE `.npy` injection (`08_f0.py`) | Extracted at binarize time via `pe: rmvpe` | Deleted → replaced by `08_export.py` |
| `pansori/warm_start.py` | `finetune_ckpt_path` / `finetune_ignored_params` | Deleted |
| `pansori/train_wandb.py` | `scripts/train.py` + TensorBoard | Deleted |
| Two-stage cascade aux/ds | Already split into acoustic / variance | Two configs |
| No inference path | `scripts/infer.py` + `.ds` files | The dictionary now exists, so the lyrics→phoneme path opened too |

---

## 2. Label schema

The old format laid the melody **onto the phoneme axis**: `note_seq`/`midi_dur`/`is_slur`
all had to be the same length as `ph_seq`, so when one vowel spanned N notes, that vowel was
duplicated N times with `is_slur=1` from the second copy onward.

openvpi separates the two axes (`preprocessing/variance_binarizer.py`). Notes carry their
own duration and are aligned to frames independently. So **both the duplication and
`is_slur` are gone.** Sigimsae is expressed directly as "several notes over one word" — not
a loss of information, just a more natural representation.

The one new requirement is `ph_num` (phonemes per word), derived from MFA's `words` tier.
`04_align.py` now records that tier into `phones.jsonl` alongside everything else, so
**re-parsing is enough; MFA does not have to be re-run.**

The `words` tier is in *eojeol* units (whitespace tokens), so a two-syllable eojeol arrives
as one chunk. When the dictionary's predicted phoneme sequence matches the actual one
exactly, it is split into syllables (나간다 → 2/3/2); when liaison has changed the count
(굴러 → `g u L eo`, dictionary says `g u L r eo`), it is left whole rather than guessed at.

What was kept — because these are facts about **this data**, not about a format:

1. **Silence covered by a note is a held vowel, not a rest.** korean_mfa was trained on
   speech and labels long sustained tones as silence. Measured: **52.2% (15.11 h)** of corpus
   time is "MFA says silence, ROSVOT says singing", and the median note coverage of those
   spans is **1.00**. Without absorbing them, the model is quietly taught to be silent where
   the singer is singing. Only silence with no note (0.74 h) becomes SP.
2. **Phonemes under 20 ms are folded into a neighbour.** At hop 512 / 44.1 kHz, one frame is
   11.6 ms.

Result: all 26,955 items PASS `07_validate.py`, zero zero-frame phonemes, averaging 6.6
words / 13.0 notes.

---

## 3. Real defects fixed during the port

`pansori_lib.convert_phone` had a three-way bug. It was confirmed by checking 89,303
single-syllable eojeol (which are unambiguous) against the dictionary; after the fix,
dictionary agreement rose **77.3% → 87.9%**.

1. **The glide vanished on entries that had both `|` and a glide.** Splitting
   `mʲ → "m|M y"` on `|` first produces `coda = "M y"`, a single token that does not exist,
   while the onset side keeps only `["m"]` — so `y` evaporated. The 며/면/벼 family all lost
   their glide (며 → `m eo`, 794 cases).
2. **Onset/coda classification fell through to coda when the next phone was a glide.**
   `vowel_base("w")` is None, so 뭐 came out as `M w eo` — a coda ㅁ at the start of a word
   (168 cases).
3. After the fix, `y` tokens increased substantially (the glides found their places again).
   OOV 0, and the 34-phoneme inventory unchanged.

---

## 4. Label quality — fixed and outstanding

`alignment/check_dict_agreement.py` does the scoring. It converts the raw IPA in MFA's
TextGrid using the same contextual rules as Stage 4 and compares against
`dictionaries/korean-coda.txt`. The dictionary comes from `build_dictionary.py`'s jamo
tables and is independent of `ipa_to_coda.csv`, so it serves as the answer key.

100% agreement is not the goal. **Liaison is normal Korean** (은 → `eu n`, 좋 → `j o`) and a
context-free syllable dictionary cannot reproduce it. Use the score as a difference: measure
before the change, measure after, adopt if the number goes up.

### The three-way fix (1 syllable 87.92% → **89.59%**, 2 syllables 75.01% → **76.43%**)

| Defect | Action | Agreement by symbol |
|---|---|---|
| Palatalisation glide missing on velar/glottal consonants (냐 → `n a`) | Added ` y` to `c cʰ cʰː c͈ ç ɲ ɟ ʝ ɕʰ ɕ͈` | ç 22.8→**74.1**, ɲ 61.0→**84.5**, c 79.0→**96.2** |
| `ɸ → p` (후 → `p w u`). ɸ is not ㅍ but **ㅎ's allophone before rounded vowels** | `ɸ→h`, `ɸʷ→h w`, plus dropping the glide when `ʷ` precedes o/u | ɸ 0→**96.9**, ʷ overall **93.9** |
| `ʎ` was wrong in every position (`L\|r` in the wrong order, and a doubled ㄹㄹ cannot be expressed as one token) | `L r\|L y` — between vowels it becomes two tokens, the previous syllable's coda plus the next one's onset; before a consonant, one coda | ʎ 0→**84.5** |

The third one is why `convert_phone` can now hold multiple tokens in a positional
alternative (빨리 = `pp a` + `L r` + `i`). The glide-dropping rule was generalised at the
same time: `ʲ` drops before `i` and `ʷ` drops before `o`/`u`, because if the following vowel
already carries that articulation, it is co-articulation rather than a glide.

The phoneme inventory stayed at 34 and OOV at 0. Applying this changed the `ph_seq` of
**3,109 of 26,955 items (11.5%)**, and `07_validate.py` passes.

### Outstanding (measured only)

| Issue | Evidence | Scale |
|---|---|---|
| ㅈ/ㅉ/ㅊ + glide in the dictionary | 죠 is [tɕo], so MFA says `j o` while the dictionary says `j y o`. 죠 and 조 are homophones, so this is a **dictionary-side** problem — adding `y` to the mapping would break the far more common 조 | 96+ |
| 요 → `y eo` vs dictionary `y o` | A vowel-side issue, not yet investigated | 156 |
| ㄴ coming out as `d` (내/나/니/너) | MFA misrecognition. The mapping cannot fix it | 700+ |
| Silence-absorption threshold | A 50 ms overlap absorbs the entire silence. 6.2% (3,772) of absorbed spans have under 50% coverage | Small |

---

## 4b. Things that only surfaced on the first openvpi run

Facts that could not have been known at porting time and only came out in actual execution.

- **`AP` is hardcoded into the phoneme set** (`utils/phoneme_utils.py:18`) and binarize only
  passes if every member of the set appears in the corpus. Our labels have no breath
  distinction, so they were merged onto one index via
  `merged_phoneme_groups: [[SP, AP]]`. Leaving the labels to say what they say is more
  correct than inventing a distinction that is not there.
- **The `spec_max` default of `[0]` does not suit this corpus.** The measured ceiling is
  **+2.838**, so after normalisation the diffusion target leaves [-1, 1].
  `configs/pansori/spec_stats.yaml` is layered onto `base_config` for this. The floor is
  safe the other way — `nvSTFT` compresses with `log(clamp(x, 1e-5))`, so nothing below
  -11.513 can exist, and the measured floor is -11.563 (including 0.05 of headroom).
  **Spec values are read only at training time, so no re-binarize is needed.**
- **Acoustic binarize also demands the HN-SEP checkpoint.** With breathiness, voicing and
  tension all switched off, `DecomposedWaveform` is still built unconditionally and the
  default `hnsep: vr` looks for `checkpoints/vr/`.
- **`requirements.txt` has no `torchaudio`** (an upstream omission). RMVPE uses `Resample`.
- **`~/.local` cuts in front of the conda env.** An old tensorflow there kills the
  `tensorboard` import at protobuf, and a numpy there makes pip misjudge requirements as
  "already satisfied" and never install them into the env. `PYTHONNOUSERSITE=1` is required.
- **`pansori/server/run.sh` had never been committed.** It was caught by `*.sh` in the root
  `.gitignore`, so `3e8d541` carried only the README. Reverted in `pansori/.gitignore`.
- **The binarize output is smaller than expected.** acoustic 4.5 GB / variance 381 MB. It
  stores no waveforms, and 44.1 kHz at hop 512 has fewer frames than 24 kHz at hop 128 —
  a third of the old 16 GB.

---

## 5. Constraints that do not change

The conclusions in `SEGMENTATION_VERDICT.md` and `alignment/FIELD_CHECK.md` are about
pansori and about this data, so they remain valid no matter what the architecture changes to.

- **Utterances must be 6 seconds or less.** korean_mfa was trained on speech and alignment
  collapses past 6 s.
- **Re-segmentation is mandatory.** Of the 37.5 h of labelled sung sections, only 3.3 h are
  under 21 s.
- **Join on the 8-digit uuid prefix.** The path strings in `labels_2.json` mix NFC and NFD.
- **f0 range 55–750 Hz.** Male *tongseong* goes down to MIDI 35 (61.7 Hz). The default of
  65 Hz cuts it off.
- **Korean codas break Chinese word_boundary logic.** openvpi takes `ph_num` directly from
  the labels, so this problem disappeared structurally.

---

## 6. What was discarded

- **The `pansori_ds2` / `pansori_aux` / `pansori_ds3` checkpoints** — 24 kHz, 80 mel,
  FastSpeech2, so not a single tensor matches the 44.1 kHz / 128 mel / reflow model.
  Convergence knowledge like "overfits at 56k steps" does not carry over either.
- **`0102_xiaoma_pe`, `0109_hifigan_bigpopcs_hop128`** — 24 kHz assets.
- **The sigimsae / contour-tokenizer family** (`usr/contour_tokenizer.py`,
  `pansori_pitch.py`, `note_seg.py`) — out of scope for this port. Only the diagnostic
  documents (`SIGIMSAE_PLAN.md`, `VIBRATO_VERDICT.md`) came across.
  **The diagnosis rests on a premise that has changed**: the hypothesised cause of
  flattening was "a deterministic regression under L1 converges to the conditional median",
  but openvpi's pitch predictor is a reflow-based generative model. Before attaching a
  tokenizer, the right move is to re-measure whether flattening still happens at all with
  the new variance model. (`eval/midi_baseline.py`, which would have done that, was itself
  not ported — it needs `note_seg.py`. See the README's legacy section.)
