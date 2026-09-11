# Pansori DiffSinger — the full pipeline, stage by stage

A record of getting from one pile of source audio to a model that sings, written in the order
it actually ran. Each stage records **what it takes in and puts out · why it is done that
way · the measured numbers · the cost of re-running it**.

- The command summary is in [`../README.md`](../README.md); the port history is in
  [`PORTING.md`](PORTING.md); **what happens inside a single training step** is in
  [`TRAINING.md`](TRAINING.md); the basis for the 6-second constraint is in
  [`SEGMENTATION_VERDICT.md`](SEGMENTATION_VERDICT.md); what is left to do is in
  [`NEXT.md`](NEXT.md).
- Branch `feat/pansori`. **Not one line of an upstream file has been changed.**

```
394 source recordings (28 GB)
  └ 01 segment ─ 02 speaker map ─ 03 transcribe ─ 04 align ─ 05 notes ─ 06 merge ─ 07 validate ─ 08 export
      └ build dictionary ─ generate configs ─ binarize ─ bootstrap spec stats
          └ train acoustic ┐
            train variance ┴─ select checkpoints (per head)
                └ write .ds ─ variance(dur) ─ variance(pitch) ─ acoustic ─ NSF-HiFiGAN ─ wav
                    └ 4-rung ladder · score synthesis ─ measure ─ report
```

---

## 0. Prerequisites

### Hardware / source material

| Item | Value |
|---|---|
| GPU | RTX 4090 (24 GB) — 6.1 GB actually used |
| Source audio | `~/Desktop/Pansori_Data/Audio/`, 394 recordings, 44.1 kHz stereo, **28 GB** (vocals separated with demucs) |
| Sung-section labels | `labels_2.json` — the modal sections within each piece (gyemyeonjo / ujo / pyeongjo / seolleongje / gyeongdeureum) plus aniri and changjo |
| Note MIDI | `rosvot_midi/*_vocal.mid` — the note timeline ROSVOT extracted per piece |
| Lyrics text | **None** → stage 03 has to transcribe it first |

The source path is hardcoded in exactly one place: `PANSORI_DATA` in
`preprocess/pansori_lib.py`.

### Three conda environments

| env | Python | Used by | Why |
|---|---|---|---|
| `pansori` | 3.10 | 01–03, 06–08 | Under `base`, the numba ↔ numpy 2.5 conflict kills 06 |
| `aligner` | 3.14 | 04–05 | MFA 3.4.1 + `korean_mfa` + mecab-ko |
| `dsp` | 3.10 | binarize / train / infer | openvpi itself. **`PYTHONNOUSERSITE=1` is mandatory** |

Without `PYTHONNOUSERSITE=1`, an old tensorflow in
`~/.local/lib/python3.10/site-packages` kills the `tensorboard` import at protobuf, and the
numpy there makes pip misjudge requirements as "already satisfied" and never install them
into the env. It is set in the `dsp` env's `activate.d`.
`requirements.txt` is **missing `torchaudio`** (an upstream omission) — RMVPE uses
`Resample`.

`bash pansori/server/run.sh setup` does all of the above.

---

## A. Preprocessing, eight stages

The stages are **separated by jsonl files on disk**, because their re-run costs differ by two
orders of magnitude — MFA alignment takes hours, but re-parsing its output takes two minutes.
When the phoneme mapping was fixed, stage 04 re-ran only the parse, not the alignment.

Every stage runs under `pansori/data/{set}/` and is selected with `--set` (in practice
`--set full`).

### Where the data shrinks

| Point | Remaining | Why it shrank |
|---|---|---|
| Source audio | 394 pieces | — |
| Intersected with the section labels | **387 pieces** | No label, or no audio |
| 01 segmentation | **31,529 utterances / 33.48 h** | aniri and changjo excluded; fragments under 2 s discarded |
| 04 MFA alignment succeeded | **26,955 utterances** | Under 2 Hangul characters so not submitted, plus alignment failures (−4,574) |
| 06 merge | 26,955 | Nothing dropped for missing or all-rest notes |
| 08 final corpus | **26,955 utterances / 28.93 h / 23 speakers** | — |

Final label volume: 370,831 phonemes · 349,774 notes · 179,776 words
(per utterance, a mean of 13.8 phonemes · 13.0 notes · 6.7 words).

---

### 01 — segmentation `preprocess/01_segment.py`

| | |
|---|---|
| In | `labels_2.json` + `Audio/*.wav` + `rosvot_midi/*_vocal.mid` |
| Out | `data/full/segments.jsonl` + `data/full/wavs/{item}.wav` |
| env | `pansori` |

Cuts each piece into **2–6 second utterances**. `item_name` has the form
`{uuid8}_{index:04d}`, and that 8-digit uuid prefix later becomes the hold-out unit (the path
strings in `labels_2.json` mix NFC and NFD, so the join is on the uuid rather than the
filename).

**Why 6 seconds.** korean_mfa was trained on speech, so its self-loop penalty accumulates
over long sustained tones. Past 6 seconds the alignment collapses — phonemes pile onto the
0.03 s floor value and the remainder is swallowed whole as silence. Measured on a sample of
10: under 6 seconds there is **not a single** silence of 1 second or more; over 6 seconds an
average of **4.06 s** was registered as silence. The sample is small, so this is a strong
hypothesis rather than a settled threshold — but re-segmentation itself is not optional: of
the 37.5 hours of labelled sung sections, only 3.3 hours are under 21 seconds.

**Where it cuts** (`segment_span`):

1. Take the non-silent spans with `librosa.effects.split(top_db=35)` and cut wherever the
   gap between spans is **0.4 s or more**, treating that as a real breath. The measured gap
   distribution has median 0.13 s and p90 0.66 s, so that value separates a breath from
   **a short pause inside sigimsae**.
2. Fragments under 2 s are absorbed into the preceding group — but only if the combination
   stays under 4.5 s (the soft target).
3. Groups still over 6 s are split. If there is silence inside, cut at the maximum of
   **width × balance score** (without the balance term, the widest pause sits at an edge and
   only a 2-second sliver breaks off); a long melisma with no silence at all is cut at
   **the widest gap between ROSVOT notes** — so as not to cut through the middle of a
   sustained tone.

Result: 31,529 utterances / 33.48 h / mean 3.82 s (min 2.00 / max 6.00).
Modal distribution: gyemyeonjo 22,321 · ujo 7,057 · pyeongjo 1,746 · seolleongje 310 ·
gyeongdeureum 95.

> The algorithm simulations behind this stage are `experiments/sim_merge.py`,
> `sim_final.py` and `diag_split.py`.

---

### 02 — speaker mapping `preprocess/02_speaker_map.py`

| | |
|---|---|
| In | `segments.jsonl` |
| Out | `data/full/spk_map.json` |
| env | `pansori` |

In the filename `{uuid8}-{track}-{myeongchang}-{bata}_{passage}`, the third field is taken as
the myeongchang if it is 2–4 Hangul characters.

**Pieces whose myeongchang cannot be identified are not discarded but pooled into a single
`unknown`.** Giving each piece its own id would create ~122 singleton speakers with a handful
of utterances each, whose embeddings never train. Pooled, it becomes the **single largest
speaker** at 6.16 hours (21.3% of the corpus).

---

### 03 — lyric transcription `preprocess/03_transcribe.py`

| | |
|---|---|
| In | `segments.jsonl` + `wavs/*.wav` |
| Out | `data/full/lyrics.jsonl` (`text_raw`, `text`) |
| env | `pansori` (GPU) |

`openai/whisper-large-v3`, fp16, 16 kHz, `language="korean"`. transformers' ASR *pipeline* is
not used — it unconditionally imports a broken torchcodec. Every segment is under 30 seconds
so long-form chunking is unnecessary, and the model is called directly.

**Repeat collapsing is necessary.** A single sustained tone sounds to Whisper like the same
syllable repeated, producing output such as `지이이이…이(×200)`.
`pansori_lib.normalize_repeats` collapses runs of 3 or more into one, and alignment uses the
collapsed side (`text`). The original (`text_raw`) is kept for review.

Whisper is not perfect on pansori. The goal is not perfect lyrics but **phonemes that line
up in time**.

---

### 04 — forced alignment `preprocess/04_align.py`

| | |
|---|---|
| In | `segments.jsonl` + `lyrics.jsonl` + `wavs/*.wav` |
| Out | `data/full/phones.jsonl` (`phones`, **`words`**) + `mfa_corpus/`, `aligned/` |
| env | `aligner` |

1. Lay out wav symlinks and `.lab` files (normalised text) in `mfa_corpus/`. Anything under
   2 Hangul characters is excluded.
2. `mfa align {corpus} korean_mfa korean_mfa {aligned} --clean --output_format long_textgrid`
3. Parse the TextGrid and convert **108 IPA symbols → the 34 jamo-based Coda phonemes**.

`pansori_lib.convert_phone` does the conversion, with two kinds of contextual rule attached —
whether a consonant is an onset or a coda, and whether to keep or drop the glide (`y`/`w`).
When one IPA span splits into two Coda tokens (`mʲ → m y`), the span's duration is divided
evenly.

**Also recording the `words` tier is this stage's key change.** `ph_num` (phonemes per word),
which openvpi requires, can only be derived here. Leaving it recorded means every mapping fix
can re-run **parsing only, without aligning**, via `--skip_align` (the old name `--skip_mfa`
is still accepted) — hours down to two minutes.

The aligner is chosen by `aligner` in `preprocess/align.yaml` — `mfa` or `sofa`. The
backends live in `preprocess/aligners.py`, and since both leave
`aligned{suffix}/*.TextGrid`, nothing from here downward changes. For SOFA the dictionary
*is* the phoneme set, so handing it `dictionaries/korean-coda.txt` removes the IPA→Coda
conversion step entirely. The procedure is in README §8.

Silence is left as a `sil` placeholder — whether it is a rest or a sustained tone can only be
known by looking at the notes, and 06 makes that call.

Result: 26,955 of 31,529 aligned successfully, OOV 0.

---

### 05 — note slicing `preprocess/05_notes.py`

| | |
|---|---|
| In | `segments.jsonl` + `phones.jsonl` (items that survived alignment) |
| Out | `data/full/notes.jsonl` |
| env | `aligner` or `pansori` |

ROSVOT is not re-run. The per-piece MIDI is simply sliced to each utterance's absolute time
window and shifted to local time. The note range is **MIDI 35 (B1, 61.7 Hz) to 78 (F♯5,
740 Hz)** — that range is what later justifies `f0_min: 55` / `f0_max: 750`.

---

### 06 — merge `preprocess/06_merge.py`

| | |
|---|---|
| In | `phones.jsonl` + `notes.jsonl` + `segments.jsonl` + `dictionaries/korean-coda.txt` |
| Out | `data/full/transcriptions.csv` (`name, ph_seq, ph_dur, ph_num, note_seq, note_dur`) |
| env | `pansori` |

The stage carrying the most judgement in the pipeline. In order:

**① Assign phonemes to words** (`assign_words`) — decided by which `words` span the phoneme's
**midpoint** falls into. Even the two tokens that 04 split evenly have midpoints inside the
original span, so they go to the same word.

**② Absorb silence covered by a note** (`absorb_silence`) — the single most important rule in
this pipeline. korean_mfa was trained on speech and **labels long sustained tones as
silence.** Measured: **52.2% (15.11 hours)** of corpus time is "MFA says silence, ROSVOT says
singing", and the median note coverage of those spans is **1.00**. Left alone, it teaches the
model to be quiet exactly where the singer is singing. So silence covered by a note for 50 ms
or more is **absorbed by extending the preceding phoneme's end**, and only silence with no
note (0.74 hours) remains as `SP`.

**③ Fold phonemes under 20 ms into a neighbour** (`merge_tiny`) — at 44.1 kHz with hop 512,
one frame is 11.6 ms, so anything below that rounds to zero frames. The folded phoneme's span
is inherited by the surviving neighbour.

**④ Re-derive boundaries** — the first phoneme is pinned to 0.0 and the last to the utterance
length so that the phoneme spans tile the utterance without gaps, then `ph_dur` is re-derived
from the boundaries. This stops rounding from pushing the total away from the waveform
length.

**⑤ Produce `ph_num`** (`group_words`) — MFA's `words` tier is in **eojeol units**
(whitespace tokens), so a two-syllable eojeol arrives as one chunk. When the dictionary's
predicted phoneme sequence **matches the actual one exactly**, it is split into syllables
(나간다 → 2/3/2); an eojeol whose count changed through liaison
(굴러 → `g u L eo`, dictionary says `g u L r eo`) is **left whole rather than guessed at**.

**⑥ Note timeline** (`note_timeline`) — gaps between notes are filled with `rest` so that
`[0, duration]` is tiled without holes. Note names come from `librosa.midi_to_note`.

> **Why `is_slur` disappeared.** The old (MoonInTheRiver) format laid the melody onto the
> phoneme axis — `note_seq`/`midi_dur`/`is_slur` all had to be the same length as `ph_seq`,
> so when one vowel spanned N notes the vowel was duplicated N times with `is_slur=1` from
> the second copy onward. In music that rolls one syllable at length, as pansori does, that
> duplication accounts for a substantial fraction of the labels. openvpi separates the two
> axes (`preprocessing/variance_binarizer.py`), so notes carry their own duration and align
> to frames independently. **Neither the duplication nor the slur flag is needed, and
> sigimsae is expressed directly as "several notes over one word"** — not a loss of
> information, just a more natural representation. For the real utterance `01ed19bb_0003`,
> 16 phonemes became 24 slots in the old format, 8 of them duplicates.

---

### 07 — validation `preprocess/07_validate.py`

| | |
|---|---|
| In | `transcriptions.csv` (+ `segments.jsonl`) |
| Out | PASS/FAIL on stdout; exit 1 on FAIL |
| env | `pansori` |

openvpi's binarizer **dies on an assert inside a worker process** — it does not skip the
item, it halts a 40-minute job partway. So every one of those asserts is checked up front.

In the order the binarizer hits them:

- `len(ph_seq) == len(ph_dur)` — acoustic + variance
- `sum(ph_num) == len(ph_seq)` — variance, `predict_dur`
- `len(note_seq) == len(note_dur)` — variance, `predict_pitch`
- At least one non-rest note — variance, `predict_pitch`
- `ph_dur ≥ 0`, `note_dur ≥ 0`
- Every phoneme resolves in the dictionary — `PhonemeDictionary.encode` KeyError

It also runs checks the binarizer **does not**:

- `sum(ph_dur) ≈ sum(note_dur) ≈ wav length` (0.05 s tolerance) — quiet frame drift
- Phonemes that round to zero frames (a wasted token; not fatal)

**Never binarize without a PASS.** Current state: all 26,955 items PASS, zero zero-frame
phonemes.

---

### 08 — export `preprocess/08_export.py`

| | |
|---|---|
| In | `segments.jsonl` + `spk_map.json` + `transcriptions.csv` + the source `Audio/` |
| Out | `data/pansori/{speaker}/wavs/*.wav` + `{speaker}/transcriptions.csv` + `datasets.yaml` |
| env | `pansori` · about 20 minutes |

It does not use the wavs 01 made; it **re-cuts from the source.** The source is already
44.1 kHz stereo, so there is no resampling — just a stereo→mono average and a sample-grid
slice (PCM_16). It opens one piece and extracts all of that piece's segments, so I/O happens
only as many times as there are pieces.

**How speakers are specified has changed.** The old pipeline overrode the speaker in a
binarizer subclass; openvpi specifies it per `datasets:` entry in the config
(`basics/base_binarizer.py:75-94`). So the export fans out into per-speaker directories and
writes a `datasets.yaml` sorted by utterance count descending — so that `spk_id 0` is the
speaker with the most data.

> The old `08_f0.py` (injecting CREPE `.npy`) was retired. openvpi extracts f0 itself at
> binarize time via `pe: rmvpe`. Injection is not merely unnecessary but **wrong** — the old
> grid was hop 128 @ 24 kHz (5.33 ms), and it is now hop 512 @ 44.1 kHz (11.6 ms).

The final corpus (`data/pansori/`, 8.7 GB):

| Speaker | Utterances | Hours | Share | Hold-out |
|---|---:|---:|---:|---|
| unknown | 5,759 | 6.16 h | 21.3% | |
| 이일주 | 3,792 | 4.06 h | 14.0% | `c235c995` |
| 김소희 | 3,275 | 3.58 h | 12.4% | `d207244a` |
| 조상현 | 2,690 | 2.85 h | 9.9% | `8b4d0824` |
| 안숙선 | 1,738 | 1.86 h | 6.4% | |
| 김일구 | 1,624 | 1.77 h | 6.1% | |
| 박동진 | 1,512 | 1.61 h | 5.6% | `e719dced` |
| 박초월 | 1,137 | 1.25 h | 4.3% | `d7ec123c` |
| 박록주 | 1,005 | 1.08 h | 3.7% | |
| 성우향 | 717 | 0.77 h | 2.6% | |
| 오정숙 | 691 | 0.73 h | 2.5% | |
| 정정렬 | 598 | 0.62 h | 2.1% | |
| 김연수 | 405 | 0.43 h | 1.5% | |
| 신금홍 | 319 | 0.35 h | 1.2% | |
| 남해성 | 259 | 0.27 h | 0.9% | |
| 성창순 | 254 | 0.27 h | 0.9% | |
| 박양덕 | 245 | 0.27 h | 0.9% | |
| 김수연 | 240 | 0.26 h | 0.9% | |
| 안향련 | 228 | 0.24 h | 0.8% | |
| 정권진 | 226 | 0.24 h | 0.8% | **released automatically** |
| 김정문 | 148 | 0.15 h | 0.5% | |
| 이동백 | 47 | 0.05 h | 0.2% | |
| 김창룡 | 46 | 0.05 h | 0.2% | |

(Speaker directories keep their Hangul names, which is why they appear untranslated here.)

The speaker skew is extreme — the top 3 excluding `unknown` (이일주, 김소희, 조상현) are 36%,
while 이동백 and 김창룡 are barely three minutes each.

---

### A checkpoint in the middle — the phoneme-mapping scorer

`alignment/check_dict_agreement.py`. The IPA→Coda mapping table was **made by hand**, so
every edit to it was guesswork. So the scorer was built first.

The principle: `dictionaries/korean-coda.txt` is generated mechanically from
`build_dictionary.py`'s jamo tables and is **completely independent of `ipa_to_coda.csv`**,
so it can serve as the answer key. Converting the raw IPA in MFA's TextGrid with the same
contextual rules 04 uses and comparing against the dictionary turns the rightness of a change
into a number. It is measured over the **89,303 single-syllable eojeol**, which are
unambiguous.

Three defects this harness caught:

| Defect | Symptom | Action | Agreement by symbol |
|---|---|---|---|
| Palatalisation glide not applied | 냐 → `n a` | Added ` y` to `c cʰ cʰː c͈ ç ɲ ɟ ʝ ɕʰ ɕ͈` | ç 22.8→**74.1** · ɲ 61.0→**84.5** · c 79.0→**96.2** |
| `ɸ → p` mis-mapping (ɸ is not ㅍ but **ㅎ's allophone before rounded vowels**) | 후 → `p w u` | `ɸ→h`, `ɸʷ→h w`, drop the glide when `ʷ` precedes o/u | ɸ 0→**96.9** · ʷ overall **93.9** |
| `ʎ` wrong in every position | 빨리 → `pp a L i` | `L r\|L y` — between vowels, **two tokens**: the previous syllable's coda plus the next one's onset; before a consonant, one coda | ʎ 0→**84.5** |

Overall agreement **1 syllable 87.92 → 89.59%**, 2 syllables 75.01 → 76.43%.
It is these fixes that gave `convert_phone` the ability to hold **multiple tokens** in a
positional alternative (빨리 = `pp a` + `L r` + `i`). The glide-dropping rule was generalised
at the same time — `ʲ` drops before `i` and `ʷ` drops before `o`/`u`, because when the
following vowel already carries that articulation it is co-articulation, not a glide.

Applying the fixes changed the `ph_seq` of **3,109 of 26,955 items (11.5%)**, and 07 passes.
The phoneme inventory stays at 34 and OOV at 0.

> **100% agreement is not the goal.** Most of the remaining disagreement is normal
> **liaison** (은 → `eu n`, 좋 → `j o`). A context-free syllable dictionary genuinely cannot
> reproduce that. Use this number not as an absolute score but as **the difference before and
> after a change.**

```sh
python pansori/alignment/check_dict_agreement.py                       # 1 syllable
python pansori/alignment/check_dict_agreement.py --syllables 2 --focus ʎ
```

---

## B. Training preparation

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
```

### B1. Build the dictionary `build_dictionary.py`

**All 11,172 Hangul syllables** → the 34 Coda phonemes. `dictionaries/korean-coda.txt`.

The old pipeline had no such file — the phoneme inventory was hardcoded as `CODA_36` and
**there was no lyrics→phoneme path at all.** openvpi builds the phoneme set from the file
named in `dictionaries:` (`utils/phoneme_utils.py`) and converts lyrics to phonemes with that
same file at inference. In other words, this file existing is what **opened the path to
singing a new song with no recording of it.**

A syllable is written `[onset] [glide] vowel [coda]` — 갸 → `g y a`, 의 → `eu i`,
왕 → `w a NG`. After generation it is checked against the 34 the corpus actually uses, and
**errors out and stops** on a mismatch.

The limitation is stated explicitly: being a per-syllable mapping, it has **no liaison.** The
training labels come from MFA and therefore do have liaison applied (책이 → `ch e g i`, not
`ch e K i`), so the inference phonemes and the training phonemes differ at some syllable
boundaries. That is an intrinsic limitation of a syllable dictionary, and upstream makes the
same trade-off.

### B2. Generate the configs `make_configs.py`

**Generates** `configs/pansori/{acoustic,variance}.yaml` (do not edit them directly — fix the
generator). The reason is that the `datasets:` block has to match what 08 actually wrote to
disk.

The key settings:

| Key | Value | Why |
|---|---|---|
| `f0_min` / `f0_max` | **55 / 750** | This corpus's lowest note is MIDI 35 (61.7 Hz). At the default 65 Hz the low register of male *tongseong* is treated as unvoiced wholesale |
| `merged_phoneme_groups` | `[[SP, AP]]` | `AP` (breath) is hardcoded into the phoneme set (`utils/phoneme_utils.py:18`) and binarize only passes if every member of the set appears in the corpus. Our labels carry no breath distinction |
| `use_spk_id` / `num_spk` | true / 24 | 23 speakers + 1 of headroom |
| `pe` / `pe_ckpt` | `rmvpe` | f0 extraction at binarize time |
| `max_batch_frames` | 50,000 (ac) / 80,000 (var) | |
| `pl_trainer_precision` | `16-mixed` | |
| Augmentation | All off | Pansori's range is already wide, and attaching key_shift/speed embeddings means those values have to be passed at inference too — one more stage in the pipeline |

**Hold-out is per piece** — put one 8-digit uuid in `test_prefixes` and
`split_train_valid_set` matches it by `startswith`, pulling one whole piece out. One was
assigned for each of six myeongchang, but **when the hold-out piece accounts for a speaker's
entire output, the generator releases it automatically.** 정권진 (226 utterances) was that
case — with zero training data, the speaker embedding never receives a gradient, yet
validation runs through that speaker. Better to train the speaker than to protect a hold-out
that cannot be learned from.

### B3. binarize

```sh
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
```

Result: **acoustic 4.5 GB / variance 381 MB**, train **27.8 h** · valid **1.08 h**, and all
23 speakers have training data. Smaller than expected — it stores no waveforms, and 44.1 kHz
at hop 512 has fewer frames than 24 kHz at hop 128 (about a third of the old pipeline's
16 GB).

> **Acoustic binarize also demands the HN-SEP checkpoint.** With breathiness, voicing and
> tension all switched off, `DecomposedWaveform` is still built unconditionally and the
> default `hnsep: vr` looks for `checkpoints/vr/`. The three prerequisite downloads are
> `checkpoints/rmvpe/`, `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/` and
> `checkpoints/vr/`.

### B4. Spec-statistics bootstrap — the order matters

```sh
python pansori/compute_spec_stats.py     # → configs/pansori/spec_stats.yaml
python pansori/make_configs.py           # regenerate with that file layered into base_config
```

openvpi's default `spec_max: [0]` does not suit this corpus. **The measured ceiling is
+2.838**, so left alone the diffusion target leaves `[-1, 1]` after normalisation. The floor
is safe the other way — `nvSTFT` compresses with `log(clamp(x, 1e-5))`, so nothing below
−11.513 can exist, and the measured floor is −11.563 (including 0.05 of headroom).

So the order is **binarize once with the defaults → measure → regenerate the configs →
train.** `spec_min`/`spec_max` are **used only at training time, so no re-binarize is
needed.**

---

## C. Training

```sh
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset
```

> What happens inside a single step — from batch assembly through the losses, the optimizer
> and the checkpoint rules — is written up separately in [`TRAINING.md`](TRAINING.md). What
> follows is a summary of the results.

Both are **trained from scratch** (no pre-trained checkpoint — the published models use a
Chinese phoneme set, so the phoneme embedding is wholly mismatched. The `finetune_*` slots
exist in the config but are `false`).

### What the two models do

| Model | In | Out |
|---|---|---|
| **variance** | Lyrics (phonemes) + notes | Per-phoneme duration `ph_dur` + the f0 curve |
| **acoustic** | Phonemes + `ph_dur` + f0 + speaker | Mel spectrogram (128 bins) |
| **NSF-HiFiGAN** (not trained) | mel | Waveform at 44.1 kHz |

Both are **reflow-based generative models** sampled with 20 euler steps. That fact governs
every evaluation that follows — the same input gives a different answer every time.

### Training results

| | Steps run | Lowest validation | Last | Status |
|---|---|---|---|---|
| acoustic | **100,000** (ran to the configured end, 123 epochs) | **0.17800 @ 28,000** | 0.18834 | After 28k, only oscillation between 0.178 and 0.203 for 72k steps |
| variance | **32,796** (stopped, 40 epochs) | **0.08761 @ 20,000** | 0.10423 | Stopped after `dur_loss` rose three times in a row |

VRAM used was **6.1 GB** of 24 GB.

**acoustic ran 3.5× longer than it needed to.** 28,000 steps was the best, and the remaining
72,000 merely burned three GPU-hours. The last checkpoint is **6% worse** than the best — and
watching only the training loss, that is the one you would have picked.

**The more dangerous part is `num_ckpt_keep: 8`.** Saving every 4,000 steps, only the most
recent 32,000 steps' worth survives. When it stopped at 100,000, **the 28,000 checkpoint had
already been deleted.** Without the separate copy in `checkpoints/best/` it would have
quietly settled for the lowest survivor, 76,000 (0.18335).

### Checkpoints are chosen per head

variance trains duration and pitch together, and the two heads' validation losses **move in
opposite directions.**

| Metric | Lowest step | Value |
|---|---|---|
| acoustic `validation/total_loss` | **28,000** | 0.17800 |
| variance `validation/dur_loss` | **16,000** | 0.07797 — rises continuously after |
| variance `validation/pitch_loss` | **32,000** | 0.00861 — still falling when stopped |
| variance `validation/total_loss` | 20,000 | A compromise dominated by dur |

Pulling pitch from the 20k that `total_loss` picks gives **25% more f0 error** than 32k. So
`run_examples.sh` picks per head and **runs variance twice** — first `--predict dur`, then
`--predict pitch` over that output (with `ph_dur` read as manual). If no single checkpoint can
satisfy both heads, there is no reason to pick only one.

The tooling enforces this:

- `examples/best_ckpt.py {exp} [--metric ...]` — picks the lowest-validation-loss step
- `checkpoints/best/{exp}_step{N}.ckpt` — the best checkpoints kept separately (1.6 GB,
  gitignored)
- `run_examples.sh` **restores** them into the experiment directory before running (it
  actually fired this time)

The combination currently in use: **acoustic 28,000 · variance dur 16,000 · variance pitch
32,000** (`examples/out/CKPT_STEPS`).

---

## D. Inference

### D1. Building the `.ds` `make_ds.py`

openvpi inference reads a `.ds` file, which normally comes out of an editor such as OpenUtau.
There is no CLI that makes one, and pansori has no editor project to export from, so this
script fills in both ends.

**`from-label`** — builds a `.ds` from an utterance that exists in the labels. Choosing what
to supply and what to withhold **isolates one stage at a time**.

- `--with-dur` : inject the real `ph_dur`
- `--with-f0` : extract f0 from the source wav and inject it (5 ms grid). To use **the same
  extractor and the same range** as the binarizer, it reads the config and extracts via
  `initialize_pe()` — at the default 65 Hz, male *tongseong* gets cut off

To infer without `--with-dur`, **`note_slur` is required.** It is the information binding
notes to words (`inference/ds_variance.py:193` — 0 starts a new word, 1 continues, exactly
one 0 per word), and since our labels keep notes and words on independent axes, that binding
was never recorded. So it is **recovered by DP** from the measured timings
(`derive_note_slur`). DP rather than a greedy method because the assignment has to be
**monotonically increasing** and every word must receive at least one note — a greedy method
starves short words and quietly produces a broken `.ds`.

**`from-score`** — lyrics plus melody only, no recording. The dictionary handles
lyrics→phonemes.

```
_   rest  0.30      # '_' is a rest -> SP
춘  A4    0.45
향  G4    0.40
-   F4    0.55      # '-' continues the previous syllable (sigimsae / melisma)
```

`ph_seq` and `ph_num` come from the dictionary; `note_slur` comes straight from the `-`
notation. Drop a `.txt` into `examples/scores/` and `run_examples.sh` synthesises it for
three speakers automatically.

> **`--spk` cannot take a Hangul speaker name.** `utils/infer_utils.py:56` parses it as
> `[0-9A-Za-z_-]+`. To avoid touching upstream, `make_ds.py` writes `spk_mix` / `ph_spk_mix`
> into the `.ds` directly — the same dict `--spk` would have filled.

### D2. The 4-rung ladder `examples/run_examples.sh`

When the synthesis is bad, you cannot fix it without knowing **which stage is at fault**. So
for each hold-out utterance a 4-rung ladder is built, replacing one measured value with a
predicted one at each rung.

| | Given | Model produced |
|---|---|---|
| `1_orig` | — | — (the original recording, the ceiling) |
| `2_resynth` | `ph_dur` + `f0` | mel — **acoustic alone** |
| `3_pitch` | `ph_dur` | `f0` + mel — pitch prediction added |
| `4_full` | Notes + lyrics | `ph_dur` + `f0` + mel — full inference |
| `5_score` | Score + lyrics (no recording) | Everything |

Three example utterances, mixing male and female voices (male *tongseong* has spans that go
entirely unvoiced without `f0_min: 55`):

- `d207244a_0085` Kim So-hee · Chunhyangga, Yi Doryeong's farewell · female voice
- `8b4d0824_0078` Cho Sang-hyeon · Chunhyangga, "and then the royal inspector" · male voice
- `e719dced_0197` Park Dong-jin · Sugungga · male voice

What the script does: restore from `checkpoints/best/` → select the best checkpoint per head →
**fix the seed** (`SEED=20260818`) → 4 rungs × 3 utterances → 2 scores × 3 speakers →
spectrograms.

Without a fixed seed, the same input gives a different result every run and the numbers in
the table wobble (measured: f0 RMSE moves 40–90 cents between runs).

---

## E. Evaluation

### E1. Ladder scores `examples/measure.py`

f0 RMSE is the error in **cents** (100 cents = a semitone) over frames both sides judged
voiced; mel L1 is a distance in the 128-bin log-mel space the model actually optimises. Both
are against the original; lower is closer.

| Utterance | Track | f0 RMSE (cents) | mel L1 | Voiced agreement |
|---|---|---:|---:|---:|
| Kim So-hee `d207244a_0085` | 2_resynth | **52.2** | 1.011 | 86.4% |
| | 3_pitch | 297.9 | 1.130 | 82.3% |
| | 4_full | 268.4 | 1.428 | 79.4% |
| Cho Sang-hyeon `8b4d0824_0078` | 2_resynth | **86.8** | 0.959 | 90.2% |
| | 3_pitch | 405.1 | 1.037 | 89.6% |
| | 4_full | 347.5 | 1.224 | 79.2% |
| Park Dong-jin `e719dced_0197` | 2_resynth | **84.5** | 1.599 | 93.4% |
| | 3_pitch | 384.7 | 1.721 | 85.6% |
| | 4_full | 394.1 | 1.643 | 85.4% |

How to read it: **resynthesis (with the measured f0 injected) is 52–87 cents**, inside a
semitone — the acoustic + vocoder path is where it should be. The moment the model starts
producing the pitch it jumps to 268–405 cents — **the current bottleneck is variance's pitch
prediction.** There are cells where `4_full` looks better than `3_pitch`; given the spread
below, that difference is meaningless.

### E2. Seed spread `examples/seed_spread.py` — before trusting a number

**The numbers above are the result of one draw.** Both models are reflow samplers, so
changing nothing but the seed moves **f0 RMSE by ±17% and mel L1 by ±8%.** To compare tracks,
either fix the seed or draw several times and use the median. A single value from the table
is far too wide a measurement to rank two tracks by.

### E3. Sigimsae flattening `examples/f0_report.py --seeds 5`

The old pipeline's diagnosis was **pitch flattening** — put an L1 loss on a deterministic
regression and it converges to the conditional median, which smears sigimsae (the
ornamentation that carries most of the expression in this music) into an average.

**That argument does not carry over to the new model.** openvpi's pitch predictor is not a
deterministic regression but a reflow generative model. So rather than assume, it had to be
re-measured.

What is measured: the **median of the inter-frame pitch movement `|Δf0|`, as a ratio against
the original**. The signature of flattening is hitting the notes while losing the movement
between them, so if that ratio is **consistently** below 1, it is flattening. "Consistently"
matters — the same utterance swung between 0.46 and 1.01 across seeds. So each track is
regenerated 5 times with different seeds and read as a **median and a range**.

| Utterance | Track | Median amplitude ratio | Range over 5 | Includes 1.0 |
|---|---|---:|---|:--:|
| Cho Sang-hyeon | Real f0 | 0.859 | 0.810 – 0.875 | |
| | f0 predicted | 0.723 | 0.644 – 0.943 | |
| | dur + f0 predicted | 0.825 | 0.652 – 0.993 | |
| Kim So-hee | Real f0 | 0.871 | 0.810 – 0.951 | |
| | f0 predicted | 0.847 | 0.637 – 1.006 | ✓ |
| | dur + f0 predicted | 1.071 | 0.635 – 1.105 | ✓ |
| Park Dong-jin | Real f0 | 0.977 | 0.914 – 1.040 | ✓ |
| | f0 predicted | 0.852 | 0.709 – 1.306 | ✓ |
| | dur + f0 predicted | 0.882 | 0.642 – 1.207 | ✓ |

**Conclusion: systematic flattening does not reproduce.** Four of the six predicted tracks
have ranges straddling 1.0. And even the track with the measured f0 injected directly sits at
0.86–0.98 — so **2–14% of the attenuation already happens in the acoustic model and the
vocoder.** It is not the pitch predictor's fault. At least by this measurement, there is no
basis for attaching a contour tokenizer.

### E4. Spectrograms `examples/spectrograms.py`

Draws the 4 rungs' mels on a single sheet. *Why* a score is bad is only visible in the
picture.

### E5. Report `examples/build_report.py`

Renders the explanation and the audio as one page. Every number the page claims is not typed
in by hand but **read back out of the artefacts** — the corpus table from `data/pansori`, the
scores from the `measure.py` / `f0_report.py` outputs, the audio from `examples/out`. The
audio is **embedded** as base64 mp3 (the Artifact CSP blocks every external host, so linked
audio will not play).

Published: <https://claude.ai/code/artifact/fe492821-71fa-4566-8b8d-459e6d619fc6>

---

## F. Full reproduction

```sh
# --- Preprocessing (no MFA re-run needed when only the labels change) ------
cd pansori/preprocess
conda activate pansori && python 01_segment.py --out_dir ../data/full
python 02_speaker_map.py --set full
python 03_transcribe.py --set full
conda activate aligner
python 04_align.py --set full            # if only the mapping changed, --skip_align (~2 min)
                                         # to run under SOFA, --aligner sofa (README §8)
python 05_notes.py  --set full
conda activate pansori
python 06_merge.py    --set full
python 07_validate.py --set full         # confirming PASS is mandatory
python 08_export.py   --set full --out data/pansori   # re-cut, ~20 min

# --- Training preparation --------------------------------------------------
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
python pansori/build_dictionary.py
python pansori/make_configs.py
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
python pansori/compute_spec_stats.py && python pansori/make_configs.py

# --- Training --------------------------------------------------------------
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset

# --- Inference · evaluation · report ---------------------------------------
bash pansori/examples/run_examples.sh
python pansori/examples/measure.py
python pansori/examples/f0_report.py --seeds 5
python pansori/examples/build_report.py
```

---

## G. Things that hurt when you step on them

**Environment**
- Using `dsp` without `PYTHONNOUSERSITE=1` kills the tensorboard import at protobuf
- Running preprocessing under the `base` env kills 06 on the numba ↔ numpy 2.5 conflict
- `requirements.txt` has no `torchaudio` (RMVPE uses it)

**Data**
- Utterances must be **6 seconds or less** — past that, MFA alignment collapses
- The path strings in `labels_2.json` mix NFC and NFD → **join on the 8-digit uuid prefix**
- Silence covered by a note is not a rest but a **sustained tone** (52.2% of corpus time)

**Config**
- Omitting `f0_min: 55` / `f0_max: 750` makes the low register of male *tongseong* come out
  unvoiced wholesale
- Leaving `spec_max` at its default `[0]` lets the diffusion target escape `[-1, 1]`
  (measured ceiling +2.838)
- Acoustic binarize also demands HN-SEP (`checkpoints/vr/`)

**Training and inference**
- **The most recent checkpoint is not the best.** And `num_ckpt_keep: 8` deletes the optimum
- variance's best step differs per head → run it twice
- **Do not judge a model from a single draw** (f0 RMSE ±17%, mel L1 ±8%)
- `--spk` cannot take a Hangul speaker name → work around it via the `.ds`'s `spk_mix`
- No `git merge` / `git remote add` — the two repositories share no common ancestor
- The three old checkpoints cannot be reused (24 kHz / 80 mel / FastSpeech2)

---

## H. Still open

Measured but not acted on — details in [`NEXT.md`](NEXT.md).

| Issue | Evidence | Scale |
|---|---|---|
| ㅈ/ㅉ/ㅊ + glide in the dictionary | 죠 is [tɕo], so MFA says `j o` while the dictionary says `j y o`. 죠 and 조 are homophones, so this is a **dictionary-side** problem — adding `y` to the mapping breaks the far more common 조 | 96+ |
| 요 → `y eo` vs dictionary `y o` | A vowel-side issue, not yet investigated | 156 |
| ㄴ coming out as `d` (내/나/니/너) | MFA misrecognition. The mapping cannot fix it | 700+ |
| Silence-absorption threshold | A 50 ms overlap absorbs the entire silence; absorbed spans with under 50% coverage | 6.2% (3,772) |
| Breath (AP) labelling | Pansori's breaths are loudly audible. There is now a path for this — SOFA + FoxBreatheLabeler behind `breath_ap.enabled` in `preprocess/align.yaml` (README §8); what remains is to run it and measure | — |
| The next training run | Cut `max_updates` or add early stopping, raise `num_ckpt_keep`, consider augmentation | — |
