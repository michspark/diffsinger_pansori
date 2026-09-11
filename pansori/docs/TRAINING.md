# How training actually proceeds — one step at a time

Once preprocessing is done (`PIPELINE.md` stages A and B) and you type
`scripts/train.py`, this document records what happens inside, in order —
**from a batch being assembled to the parameters being updated once.**

- The full pipeline record is in [`PIPELINE.md`](PIPELINE.md); the command summary is in
  [`../README.md`](../README.md).
- Every number here was pulled from the actual logs and TensorBoard of the
  `pansori_ac_v1` / `pansori_var_v1` runs. None of it is an estimate.

---

## 0. One-page summary

**Two models are trained separately.** They exchange nothing, are entirely independent, and
the order does not matter (this time variance was started while acoustic was mid-run).

| | Parameters | What it takes in | What it learns |
|---|---|---|---|
| **variance** `pansori_var_v1` | 25.1 M | Phonemes + notes (MIDI, duration) | ① per-phoneme duration `ph_dur` ② the f0 curve |
| **acoustic** `pansori_ac_v1` | 69.1 M | Phonemes + `ph_dur` + f0 + speaker | The mel spectrogram (128 bins) |
| NSF-HiFiGAN | — | mel + f0 | The waveform — **not trained.** Taken and used as-is |

The skeleton of one step is the same for both:

```
  pull one batch from the binary data
        ↓
  [encoder] phonemes → vectors → stretched onto the frame axis = condition
        ↓
  [head]    hit the target, conditioned on condition
        ↓
  loss → backprop → one Muon+AdamW update
        ↓
  every 4,000 steps: validation + figures/audio + checkpoint
```

Five terms are all you need.

| Name | Shape | Meaning |
|---|---|---|
| `tokens` | `[T_ph]` | The phoneme index sequence. This corpus has 36 (Coda 34 + merged `SP`/`AP` + PAD) |
| `ph_dur` | `[T_ph]` | Per-phoneme duration (**in frames**, not seconds) |
| `mel2ph` | `[T_s]` | Per frame, "which phoneme is this". `ph_dur` unrolled |
| `condition` | `[T_s, 384]` | The condition vector stretched to frame resolution. Every head's input |
| `base_pitch` | `[T_s]` | The reference pitch, notes unrolled to frames and smoothed (semitones) |

A **frame** is 44,100 / 512 = **86.13 per second**. The mean utterance of 3.82 s ≈ **329
frames**. Everything below keeps using that mean utterance (14 phonemes · 329 frames) as the
example.

---

## 1. When you type `train.py` — setup before anything starts

```sh
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
```

### 1-1. Configs are layered

Merged recursively by following `base_config:`. The actual chain is printed on the log's
first line:

```
| Hparams chains: ['configs/base.yaml', 'configs/acoustic.yaml',
                   'configs/pansori/spec_stats.yaml', 'configs/pansori/acoustic.yaml',
                   'configs/pansori/data_pansori.yaml', 'configs/exp/ac_v1.yaml']
```

**Later wins.** `f0_min` is 65 in base and gets overwritten to 55 by pansori, for instance.
The merged result is saved whole to `checkpoints/{exp}/config.yaml`, and **from the next run
onward that file is the authority.** `--reset` means "ignore the saved copy and rebuild from
the yaml" (`utils/hparams.py:87-115`). Edit a config and re-run without `--reset` and your
edit does nothing.

### 1-2. The task class is found by string and started

`task_cls: training.acoustic_task.AcousticTask` is imported via `importlib` and
`AcousticTask.start()` is called (`scripts/train.py:22-27`). The structure lets a different
model run purely by changing the config.

### 1-3. Trainer assembly (`basics/base_task.py:398-440`)

| Setting | Value | Meaning |
|---|---|---|
| `precision` | `16-mixed` | forward/backward in fp16; weights and optimizer in fp32 |
| `max_steps` | 100,000 (ac) / 80,000 (var) | Stop on reaching this |
| `val_check_interval` | 4,000 | **In steps, not epochs** |
| `check_val_every_n_epoch` | `None` | Epoch boundaries are ignored entirely |
| `accumulate_grad_batches` | 1 | One batch = one step |
| `gradient_clip_val` | 1.0 | |
| `num_sanity_val_steps` | 1 | Runs exactly **one** validation batch before training |
| callbacks | `DsModelCheckpoint`, `DsTQDMProgressBar` | |
| logger | `DsTensorBoardLogger` → `checkpoints/{exp}/lightning_logs/latest` | |

### 1-4. What inference will need is copied into work_dir

`spk_map.json`, `lang_map.json` and `dictionary-ko.txt` are copied from the binary directory
into `checkpoints/{exp}/` (`base_task.py:444-462`).
**Edit the dictionary later and this checkpoint still infers with the dictionary it was
trained on.** It is a guard against speaker indices and phoneme indices drifting apart.

### 1-5. Resuming is the default

```python
trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))
```

Among `checkpoints/{exp}/model_ckpt_steps_*.ckpt` it automatically resumes from **the one
with the largest step** — largest step, not most recent; it picks by name, not by file time.
Optimizer state, scheduler and epoch are all restored. To start over, the directory has to
be emptied.

> On resume, the optimizer hyperparameters are **taken from the config, overriding the
> checkpoint** (`base_task.py:471-514`). Change the lr and resume and that lr applies, with
> the scheduler re-rolled up to the current step by `simulate_lr_scheduler`.

---

## 2. How a batch gets assembled

### 2-1. From disk

`data/pansori/binary_acoustic/` holds `train.data` (one contiguous binary), `train.idx`
(an offset table) and `train.meta` (metadata such as lengths, names and speaker names).
`IndexedDataset` reads straight from the offset — nothing is loaded whole into memory.

For this corpus:

| | Utterances | Duration |
|---|---|---|
| train | 25,946 | 100,101.68 s = **27.8 h** |
| valid | 1,009 | 3,881.78 s = **1.08 h** |

valid comes from **5 hold-out songs** — only Lee Il-ju, Kim So-hee, Cho Sang-hyeon,
Park Dong-jin and Park Cho-wol, and **the other 18 speakers have not a single utterance in
valid** (see `valid respective duration` in the binarize log). That means the validation loss
is a value about those 5. It was originally 6; Jeong Gwon-jin, whose hold-out song was that
speaker's entire output, was released automatically by `make_configs.py` (with zero training
data, a speaker embedding never receives a gradient at all).

### 2-2. Forming batches — `DsBatchSampler` (`utils/training_utils.py:76-197`)

Each epoch goes through this order:

1. **Shuffle** — the full index list is randomly permuted.
2. **Sort by length** — lengths are rounded to a 6-frame grid
   (`sampler_frame_count_grid`) and sorted descending.
   → **Similar lengths end up in the same batch. The point is to reduce padding.**
   (Step 1's shuffle is what varies the order among equal-length items each epoch.)
3. **Cut into batches** — cut at whichever of `max_batch_frames` 50,000 or
   `max_batch_size` 32 hits first.
4. **Shuffle the batch order** — the batches themselves are shuffled again, or there would be
   stretches containing nothing but long utterances.

**Measured: 32 always hits first.** The mean of 329 frames × 32 ≈ 10,500 frames, and even at
the 6-second maximum (517 frames), 32 of them is 16,500 frames — nowhere near the 50,000
cap. So the log's `batch_size` is always 32, and **the number of batches per epoch is fixed
at 811** (= ⌈25,946 / 32⌉).

```
100,000 steps ÷ 811 batches/epoch = 123.3 epochs   ← the log's last line says "Epoch 123"
```

### 2-3. Padding — `collater`

Padded to the longest item in the batch. **The fill value differs per field**:

| Field | Pad value | Why |
|---|---|---|
| `tokens`, `mel2ph`, `ph_dur` | `0` | 0 is the PAD phoneme. `mel2ph > 0` *is* the valid-frame mask |
| `mel` | `log(1e-5)` = −11.513 | Real silence. Filling with 0 would be maximum amplitude |
| `note_midi` | `-1` | Negative marks padding (`padding_mask = note_midi < 0`) |
| `note_rest` | `True` | Padding is treated as a rest |

---

## 3. One acoustic step — 69.1 M

The single function `training/acoustic_task.py:run_model` is all of 3-1 to 3-4 below.
Batch `B=32`, example utterance `T_ph=14`, `T_s=329`, mel bins `M=128`.

### 3-1. Building the condition (`FastSpeech2Acoustic`)

```
tokens [32,14]  ──txt_embed(36→384)──────┐
mel2ph [32,329] ──count → recover dur─→ log(1+dur) ──dur_embed──┤
                                          ├─→ encoder (4 layers, self-attn 2 heads, RoPE)
                                          │        ↓  [32,14,384]
                                          │   gather by mel2ph = stretch onto the frame axis
                                          │        ↓  [32,329,384]
                                     + stretch_embed   ← how much this phoneme was stretched
                                     + spk_embed       ← 24 speakers
                                     + pitch_embed(f0) ← log(1 + f0/700)
                                          ↓
                                    condition [32,329,384]
```

The key move is **length regulation**. The 14-phoneme vector sequence is copied and expanded
to 329 frames, using `mel2ph` as the index. A 6-frame phoneme has its vector repeated 6
times.

**During training the ground-truth `ph_dur` is used** (via the `mel2ph` built from it). So
acoustic does not decide durations itself — that is variance's job. At inference too,
somebody (variance, or a human) has to supply the durations.

`stretch_embed` supplies "by what factor was the phoneme this frame belongs to stretched".
Copying alone leaves a 6-frame `a` and a 60-frame `a` indistinguishable frame by frame. It
is an especially important input for singing that holds one note long, as pansori does.

### 3-2. aux decoder — draws the mel in one shot

Because `use_shallow_diffusion: true`, a 6-layer ConvNeXt auxiliary decoder pulls a mel
**directly out of `condition`, with no generative process**. Its loss is **L1** against the
normalised mel.

```python
aux_cond = condition * 0.1 + condition.detach() * 0.9    # aux_decoder_grad
```

The auxiliary decoder's gradient reaches the encoder at **only 1/10** strength. It is a
guard against the encoder being pulled towards "the rough-sketch side"
(`modules/toplevel.py:109`).

### 3-3. reflow — this is where the real mel is learned

`modules/core/reflow.py:36-56`. Four lines.

```python
spec  = norm_spec(gt_mel)                       # normalised to [-1, 1] by spec_min/max
t     = 0.4 + 0.6 * rand(B)                     # one scalar per utterance in the batch
x_t   = noise + t * (spec - noise)              # noise ~ N(0, I)
v_pred = backbone(x_t, t*1000, cond)            # LynxNet2, 6 layers × 1024ch, kernel 31
loss   = L2(v_pred, spec - noise)               # true velocity = destination − origin
```

**A straight line is drawn from noise to the true mel, and the model is asked, at a random
point on it, "which way should I go".** The true direction is always the constant
`spec - noise`. At inference the model walks that velocity field in 20 steps (euler).

**`T_start = 0.4` determines this model's character.** Because `t` is drawn from `[0.4, 1]`
rather than `[0, 1]`, **acoustic never learns to start from pure noise.** The starting line
at 0.4 is filled at inference by the mel the aux decoder drew:

```
inference:  x = 0.4 * aux_decoder(condition) + 0.6 * noise    ← starts here
            euler 20 steps, dt = (1 − 0.4)/20 = 0.03
```

That is shallow diffusion. Since it is not drawing from the ground up, 20 steps suffice.

With `main_loss_log_norm: false` (acoustic), there is no t-dependent loss weighting. All t
are equal.

> **Padding mixes into the loss value.** The loss zeroes the padded frames and then takes
> `.mean()`, but the denominator includes the padding
> (`modules/losses/reflow_loss.py:42-50`). A batch with a lot of padding reports a lower
> loss. That is another reason the length sort (2-2) helps training stability, and
> **validation has no padding at all, being one utterance per batch** — so the validation
> loss is free of this effect.

### 3-4. Total loss

```
total_loss = mel_loss + 0.2 × aux_mel_loss
```

The last values actually seen in the log: `mel_loss=0.15771`, `aux_mel_loss=0.01899`
(already multiplied by 0.2 — the `lambda` is applied before the dict is populated).

---

## 4. One variance step — 25.1 M, two heads

`training/variance_task.py:run_model`. **Duration and pitch are learned in the same step.**

### 4-1. The duration head

```
tokens ──txt_embed──┐
onset  ─────────────┤   onset = is this the word's first phoneme (the diff of ph2word)
word_dur ───────────┤   word duration (ground-truth ph_dur summed per word)
                    ├─→ encoder, 4 layers → encoder_out [B, T_ph, 384]
                    │        + midi_embed (mean MIDI per phoneme, 128-way embedding)
                    │        + spk_embed
                    │              ↓
                    │      DurationPredictor (resnet 5 layers × 256ch, Conv1d k=3)
                    │              ↓  output in the log domain
                    └─────→ ph_dur_pred [B, T_ph]
```

**During training the word durations are built from ground truth**
(`variance_encoder.py:76-79`). So what the duration head learns is less "how many seconds is
this sentence" and more **"given a word's duration, how do I divide it among the
phonemes"**. At inference the note durations in the `.ds` take that place.

The loss has three layers, all **MSE in the log domain** (`modules/losses/dur_loss.py`):

| Term | Weight | What it targets |
|---|---|---|
| `pdur` | 0.3 | Each individual phoneme's duration |
| `wdur` | 1.0 | The per-word sum |
| `sdur` | **3.0** | The whole sentence's duration |

**Sentence length weighs 10× more than phoneme length.** Individual phonemes each being
slightly off is far less bad than the total length drifting — if the total drifts, the f0 and
mel that follow leave the notes entirely. Measuring the log is for the same reason: a 0.02 s
error on a 0.05 s phoneme is not treated the same as a 0.02 s error on a 2 s one.

### 4-2. The pitch head

```
condition (phonemes stretched to frames, the same way acoustic does it)
   + MelodyEncoder(note_midi, note_dur)  ← encodes the note axis separately, gathered to frames
   + pitch_retake_embed                  ← see 4-3
   + delta_pitch_embed
        ↓
   PitchRectifiedFlow      target = pitch − base_pitch  (deviation in semitones)
```

**The target is not f0 itself but "how far from the note".** The notes already carry absolute
pitch, so the model only has to learn sigimsae, yoseong and the joins between them. Clipped
to ±12 semitones, normalised over ±8.

`repeat_bins: 64` — the one-value-per-frame curve is replicated into 64 bins to make it 2D
like a mel. It is a trick for reusing the same backbone (LynxNet2); at inference the 64 bins
are averaged.

Two differences from acoustic:

- **`t ~ U(0, 1)`.** variance is not shallow diffusion (there is no `use_shallow_diffusion`
  in its config, so `RectifiedFlow` forces `t_start` to 0). It starts from pure noise.
- **`main_loss_log_norm: true`.** A log-normal-shaped weighting over t makes it **weight
  t≈0.5 heavily** (`reflow_loss.py:27-35`). Both ends (nearly noise / nearly ground truth)
  count for less.

### 4-3. The retake mask — why part of the ground truth is shown during training

```python
pitch_retake = random_retake_masks(B, T)   # on average half the frames are True
```

Spans that are `True` mean "redraw this"; spans that are `False` **get the ground-truth pitch
fed in as input.** A random contiguous span is chosen fresh every step
(`variance_task.py:74-81`).

For an editor's **"re-roll just this part"** to work, the model has to know how to join onto
given pitch before and after. Train it only to draw whole spans from scratch and the seams
will jump on partial regeneration. At inference, redrawing everything makes the mask all
`True`.

### 4-4. Total loss

```
total_loss = 1.0 × dur_loss + 1.0 × pitch_loss
```

**This becomes a problem later.** The two heads learn different things but the losses are
summed into one, and since `dur_loss` is an order of magnitude larger than `pitch_loss`,
`total_loss` is effectively `dur_loss` (see the table in §7). Choose a checkpoint by
`total_loss` and you throw away the pitch.

---

## 5. Backprop and update

Lightning handles this part, but the settings are worth knowing.

| | Value | Note |
|---|---|---|
| Precision | `16-mixed` | fp16 forward + `GradScaler`. Used **6.1 GB** of 24 GB of VRAM |
| Gradient clip | 1.0 (L2 norm) | |
| Optimizer | `Muon_AdamW` | **Splits the parameters in two** ↓ |
| lr | 6e-4 | betas (0.9, 0.98) |
| Scheduler | `StepLR(5000, 0.8)` | **In steps.** ×0.8 every 5,000 |

**Muon and AdamW are mixed** (`modules/optimizer/muon.py:161-200`):

- **Muon** — ordinary weights of 2 dimensions or more (attention and conv kernels, etc.). The
  gradient is orthogonalised by Newton–Schulz iteration before being applied. weight decay
  0.1.
- **AdamW** — anything declared as `nn.Embedding`, `AdamWLinear` or `AdamWConv1d`, plus every
  1-dimensional parameter (biases, LayerNorm). weight decay 0.

**Every projection lifting a single scalar such as `f0` or `ph_dur` into 384 dimensions is
declared `AdamWLinear`**, because orthogonalisation is meaningless for a layer with
1-dimensional input.

lr measured: **8.647e-06** at step 100,000 — 1/69 of its initial value. The late stage of
training is effectively a fine-tuning regime, and that is half the reason the loss stops
moving in §7.

**Speed**: about **7.0 it/s** (RTX 4090, batch 32). 811 batches = 1 epoch ≈ 115 s.
100,000 steps is **about 4 hours**, validation excluded.

---

## 6. What happens every 4,000 steps

`val_check_interval: 4000`. Cut by step, independent of epochs.

### 6-1. Validation — runs twice

```python
losses = self.run_model(sample, infer=False)     # ① loss: computed exactly as in training
if this batch contains an utterance with index < 10:
    out = self.run_model(sample, infer=True)     # ② real generation: 20 euler steps
```

- **① runs over all 1,009 valid utterances.** With `max_val_batch_size: 1` it is **one
  utterance at a time** — no padding whatsoever.
- **② only the first 10 utterances.** These are really sampled over 20 steps, and for
  acoustic the waveform is built with NSF-HiFiGAN and **audio and mel figures** are posted to
  TensorBoard (`num_valid_plots: 10`). variance draws duration bar charts and pitch curves.
- Validation runs in fp32 with `autocast` disabled (`base_task.py:266`).

> **There is randomness in the validation loss too.** ① is the same code as training, so `t`
> and the noise are drawn fresh each time. A one-notch difference in the curve (±0.005 or so)
> is not a signal. **acoustic's 28,000 (0.17800) and 64,000 (0.17988) are effectively
> indistinguishable.** Read the trend only.

### 6-2. Checkpoints — **not chosen by loss**

```python
DsModelCheckpoint(monitor='step', mode='max', save_top_k=num_ckpt_keep)
```

`monitor='step'`, `mode='max'` — it keeps **the N with the largest step**, that is, **the N
most recent**. The validation loss has no influence on retention at all. acoustic
`num_ckpt_keep: 8`, variance `num_ckpt_keep: 5`.

There is a separate permanent-retention rule (`training_utils.py:269-286`): a checkpoint is
exempt from deletion if `step ≥ permanent_ckpt_start` and
`(step − start) % permanent_ckpt_interval == 0`. For acoustic, start 60,000 / interval
10,000.

**But that rule only half fires.** Saving itself only happens at multiples of 4,000, so
70,000 and 90,000 **never existed in the first place.** What actually survived:

```
Permanent:  60,000 · 80,000 · 100,000          ← only common multiples of 4,000 and 10,000
Recent 8:   72,000 76,000 80,000 84,000 88,000 92,000 96,000 100,000
```

The file listing in `checkpoints/pansori_ac_v1/` is exactly that (plus 12,000 and 28,000,
restored from `best/`). **The lowest one, 28,000, had already been deleted by the time
training ended.** Without the separate copy in `checkpoints/best/` it would have quietly
settled for the lowest survivor, 76,000 (0.18335).

> 4,000 and 10,000 not lining up is an upstream default's problem. Set
> `permanent_ckpt_interval` to a multiple of `val_check_interval` (12,000, say) and the rule
> works as intended.

---

## 7. How it actually went

### acoustic (`validation/*`)

| step | total | mel | aux | |
|---:|---:|---:|---:|---|
| 0 | 1.25401 | 1.08727 | 0.16675 | sanity check, one batch — exclude from the curve |
| 4,000 | 0.19593 | 0.16167 | 0.03426 | |
| 12,000 | 0.18394 | 0.15173 | 0.03220 | |
| 16,000 | 0.20279 | 0.17124 | 0.03155 | Highest after 4,000 |
| **28,000** | **0.17800** | 0.14499 | 0.03301 | ← **lowest** |
| 52,000 | 0.17981 | 0.14820 | 0.03161 | |
| 64,000 | 0.17988 | 0.14747 | 0.03241 | |
| 76,000 | 0.18335 | 0.15095 | 0.03240 | Lowest among the surviving files |
| 100,000 | 0.18834 | 0.15625 | 0.03209 | Last — **5.8% worse** than the lowest |

**It reached 0.196 within 4,000 steps and then merely oscillated between 0.178 and 0.203 for
the next 96,000.** `aux_mel_loss` is effectively frozen at 0.032 after 8,000 steps. It ran
the configured 100,000 to completion, but **under 30,000 was what it needed.**

### variance (`validation/*` + `metrics/*`)

| step | total | dur | pitch | rhythm_corr | pitch_r2 |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 0.20016 | 0.18772 | 0.01244 | 0.325 | 0.433 |
| 12,000 | 0.10261 | 0.09188 | 0.01073 | 0.500 | 0.465 |
| **16,000** | 0.08764 | **0.07797** | 0.00967 | 0.675 | 0.372 |
| 20,000 | **0.08761** | 0.07807 | 0.00955 | **0.750** | 0.434 |
| 24,000 | 0.09049 | 0.08169 | 0.00881 | 0.650 | 0.454 |
| 28,000 | 0.09877 | 0.09008 | 0.00868 | 0.575 | **0.501** |
| **32,000** | 0.10423 | 0.09561 | **0.00861** | 0.650 | 0.416 |

**The two heads move in opposite directions.** `dur_loss` bottoms out at 16,000 and rises
steadily after; `pitch_loss` keeps falling all the way to 32,000. `total_loss` is dominated
by `dur` by an order of magnitude, so it is effectively the `dur` curve — and its minimum at
20,000 is `dur`'s doing.

Training was stopped at step 32,796, after `dur_loss` rose three times in a row.

### Which is why checkpoints are chosen per head

| Used for | Chosen by | step |
|---|---|---|
| acoustic | `validation/total_loss` | **28,000** |
| variance — duration | `validation/dur_loss` | **16,000** |
| variance — pitch | `validation/pitch_loss` | **32,000** |

Pulling pitch from the 20,000 that `total_loss` picks gives **25% more f0 error** than
32,000. That is why `examples/run_examples.sh` **runs variance twice** — first
`--predict dur`, then `--predict pitch` over that output.

The tool: `python pansori/examples/best_ckpt.py {exp} [--metric validation/pitch_loss]`
(step 0 is the sanity check and is excluded; only steps whose files still exist are
candidates).

---

## 8. Rules for next time

1. **Do not trust `max_updates`.** The configured 100,000 was a default with no basis behind
   it. **Stop when the validation loss fails to set a new minimum three times in a row
   (= 12,000 steps)** — that rule was applied to variance, cutting it at 32,796, and was not
   applied to acoustic, which burned three GPU-hours.
2. **Copy to `checkpoints/best/` before stopping.** `num_ckpt_keep` does not look at the
   loss. The lowest checkpoint gets quietly deleted while training carries on.
3. **With multiple heads, do not use `total_loss`.** Pick the minimum step per head.
4. **Ignore a one-notch difference in the curve.** There is randomness in the validation loss
   itself (6-1).
5. **There is room to run variance longer.** `pitch_loss` was still falling when it stopped.
   Freezing the dur head (`frozen_params`) and running pitch further is a candidate for the
   next experiment.

### Things frequently stepped on when re-running

- **Editing a config and re-running without `--reset`** → the saved
  `checkpoints/{exp}/config.yaml` wins and the edit does nothing.
- **Adding `--reset` but not emptying the directory** → the config is rebuilt, but
  `trainer.fit` still **resumes** from the last checkpoint. `--reset` resets only the config.
- **Changing `spec_min`/`spec_max`** → these are used only at training time, so
  **no re-binarize is needed.** Just regenerate the configs with `make_configs.py`.
- **The two trainings are independent.** Each is around 6 GB, so both run concurrently on one
  4090.

---

## 9. How training inverts at inference

| | Training | Inference |
|---|---|---|
| `mel2ph` | From ground-truth `ph_dur` | From the `ph_dur` variance predicted |
| f0 (acoustic input) | Measured, extracted with RMVPE | The curve variance generated |
| reflow | One random `t`, one velocity prediction | `t_start`→1 split into 20, 20 predictions |
| retake mask | A random contiguous span | All `True` (or only the edited span) |
| aux decoder | For computing the loss | **For generating the starting line at 0.4** |

This correspondence is what makes the **4-rung ladder** in `examples/run_examples.sh` a
diagnostic — descending `2_resynth` (`ph_dur`+f0 given) → `3_pitch` (`ph_dur` only) →
`4_full` (notes and lyrics only), the model has to invent one more thing at each rung. Where
it collapses tells you which training fell short.

Inference and evaluation are in [`PIPELINE.md`](PIPELINE.md) sections D and E.
