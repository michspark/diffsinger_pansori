# A plan for bringing sigimsae back — using a pitch-contour tokenizer

Measurement confirmed why `pansori_aux`'s (cascade stage 1) f0 does not sound like pansori.
This document is the plan for how to attach
`~/Desktop/pitch-contour-tokenizer` (ISMIR 2026, a VQ-VAE pansori contour tokenizer).

The evidence is `pansori/eval/midi_baseline.py`; the reproduction command is in §6.

> **Note on the state of this document.** All of it was written and run in the
> MoonInTheRiver repository, before the port to openvpi. The files it produced —
> `usr/contour_tokenizer.py`, `pansori/eval/contour_tokens.py`,
> `pansori/eval/tokenizer_parity.py` — were **not** carried across to this branch, and
> `pansori/eval/midi_baseline.py`, which came across, cannot run here either (it needs
> `note_seg.py`). Keep it as a record of two negative results and the reasoning behind
> them. `docs/PORTING.md` §6 also notes that the premise has changed: openvpi's pitch
> predictor is a reflow-based generative model, not a deterministic regression.

---

## 1. Diagnosis — we already know what the problem is

`midi_baseline.py`, 24 test utterances, against CREPE:

| | Note-bias RMS | Within-note variation | (note interiors only) | Correlation with CREPE |
|---|---|---|---|---|
| CREPE (ground truth) | 131.2 | **161.4** | **138.0** | 1.000 |
| ROSVOT note staircase | 0.0 | 0.0 | 0.0 | — |
| `pansori_aux` | 70.7 | **31.8** | **17.4** | **0.145** |

The model moves only **12–20%** as much as ground truth within a note, and even that
movement is nearly unrelated to ground truth. Its pitch accuracy (194.3 vs the staircase's
203.2 on identical frames) beats the staircase by a mere 8.9 cents.
**The model sings notes flat.**

Hypothesised cause: `add_f0_loss` is an **L1 regression** in log₂ space. Sigimsae's timing
cannot be predicted exactly, so the optimum of a deterministic output under L1 is the
**conditional median — a flat line**. Change the parameterisation and the same flattening
recurs as long as it is a deterministic regression under L1.

---

## 2. What exactly the tokenizer is

Per `src/models/model_zoo.py` and `src/configs/model/foffset_conv1d_vqvae.yaml`.

- Trained on the CREPE contours of **1,914** pansori and danga pieces
  (`splits/inhouse_crepe_split.csv`)
- **1 token = 1.28 s** (128 frames @ 100 Hz) of contour shape, **with the median removed**
- A codebook of **256** = a dictionary of pansori local pitch patterns; 32-dimensional latent
- `OffsetMSELoss`: among candidates of tempo (50/54/60) × zoom (0.85/1.0/1.15) × y-offset
  (±0.1), **only the minimum error is backpropagated** → tokens are invariant to timing and
  register changes
- The paper classifies six sigimsae types with this codebook (`southern_flick`,
  `southern_vibrato`, `western_vibrato`, `soft_vibrato`, `upward_accent`,
  `descending_slide`) → **there is already evidence the codebook distinguishes sigimsae**

---

## 3. Two reasons it should attach well (confirmed)

**The units already match.** The tokenizer's input is
`norm_midi = midi/12 = log₂(f0) − 3.0314` (`dataset_zoo.py:45-64`; `norm_midi=False`, so
tonic=0). DiffSinger's f0 target is already `log₂(f0)` (`pitch_norm: log`), and on top of
that the tokenizer subtracts the median per window — **the constant term cancels entirely**.
The only conversion needed is a resample.

**The frame ratio is an integer.** DiffSinger runs at hop 128 @ 24 kHz = **187.5 Hz**, the
tokenizer at **100 Hz**. The ratio is 15:8, so

> **240 DiffSinger frames = 128 tokenizer frames = 1.28 s**

One line of `F.interpolate(size=128, mode='linear')` converts it exactly and differentiably.

---

## 4. Three reasons it is hard to attach (stated honestly)

**(a) The tokens are too coarse.** One token = 1.28 s = 240 frames. A 6-second utterance is
4.7 tokens. So **the naive plan of "replace f0 regression with token classification" does
not work** — it loses note-level control. The tokenizer was designed as an analysis tool,
not a synthesis representation.

**(b) The Python environments differ.** The tokenizer needs 3.10–3.13; the `diffsinger` env
is **3.8**. However, checked:

| | diffsinger env |
|---|---|
| torch | **2.0.1** (recent enough) |
| `nn.Dropout1d` | Present (and with `cnn_dropout: 0` it is never even constructed) |
| `torch.nanmedian` | Present |
| torchaudio | **Absent** — but only used in `process_offset` (the decoder) |

→ **The encoder path is nothing but `Conv1d / ReLU / MaxPool1d / median`, so it runs as-is
under 3.8.** Only `ConvEncoder` + `calc_conv_params` + the median logic need porting
(~80 lines).

**(c) The learned representation is not "residual against the note".** The median of a
1.28 s window is roughly the mean pitch of that span, so a token from a window containing
several notes also encodes **the note transitions**. Adding it directly on top of the note
staircase double-counts. Stage C addresses this head on.

---

## 5. Plan by stage

### Stage A — use it as a judge (half a day, zero risk)

Two purposes: ① quantify the flattening in the vocabulary of sigimsae, and ② **gate whether
Stage B can work at all**.

Setup:
```sh
cd ~/Desktop/pitch-contour-tokenizer
curl -LO https://github.com/SeongUkJu/pitch-contour-tokenizer/releases/download/v1.0.0/checkpoints.tar.gz
tar xzf checkpoints.tar.gz        # → weights/all_transformation_vqvae/ and 3 others, 0.5 MB
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
(The 2 GB data asset is **not needed** — we use our own CREPE contours.)

Work: new file `pansori/eval/contour_tokens.py`. Over the same 24 utterances
`midi_baseline.py` uses, tokenize the CREPE f0, the model f0 and the note staircase after
resampling 187.5→100 Hz into 128-frame windows, and compare.

| Metric | Meaning |
|---|---|
| Token agreement | GT token == model token |
| Token distribution KLD | `D(model ‖ GT)` — large if the model piles onto a few tokens |
| Tokens used / perplexity | How many distinct tokens GT uses vs the model |
| Latent distance | `‖z_pred − z_gt‖` |

**Gate B0 — check this first, without exception:**
> When the **CREPE contour** and a **staircase (perfectly flat) contour** of the same span
> are tokenized, **do different tokens come out?**

- Different → the codebook is sensitive to "flat vs sigimsae" → Stage B's loss gives a real
  gradient
- Same → Stage B is pointless → **go straight to Stage C**

---

### ✅ Stage A result (complete)

`pansori/eval/contour_tokens.py`, 150 test utterances → 318 windows of 1.28 s
(1,134 discarded for missing data — see the limits below). Checkpoint
`all_transformation_vqvae`.

**Gate B0 = 84.9% pass.** The codebook clearly distinguishes sigimsae from flatness.
**Stage B can proceed.**

| Token agreement | |
|---|---|
| `token(CREPE) ≠ token(staircase)` — **gate B0** | **84.9 %** |
| `token(model) == token(staircase)` | **54.4 %** |
| `token(model) == token(CREPE)` | **17.3 %** |

The model emits the staircase's token **3.1×** more often than ground truth's. The
flattening that `within_std` showed is confirmed again at the level of the sigimsae
vocabulary.

| Vocabulary | Tokens used | Perplexity | KLD vs CREPE |
|---|---|---|---|
| CREPE (ground truth) | 112 | 78.1 | — |
| ROSVOT note staircase | 73 | 21.3 | 1.991 |
| Model f0 | 71 | 29.7 | 1.743 |

| Latent distance (the direct proxy for Stage B's loss) | |
|---|---|
| Random CREPE pairs (baseline) | 0.454 |
| `‖z(staircase) − z(CREPE)‖` | 0.248 (55% of baseline) |
| `‖z(model) − z(CREPE)‖` | **0.247 (54% of baseline)** |

**This is the heart of the measurement.** In contour-shape space, the model's f0 is **not
one bit closer to ground truth than a flat MIDI staircase is** (0.247 vs 0.248). The
9.2-cent advantage it won on pitch bought nothing in shape. And since this value is exactly
what Stage B would drive down, **the headroom is maximal** (0.247 → 0, against a random
baseline of 0.454).

Both are nevertheless closer than random (0.454) — meaning the staircase and the model both
share the **note skeleton** with CREPE and share only the **sigimsae** with nothing. That is
a consistent reading.

Outputs: `pansori/review/contour_tokens.html` (including the shapes of the top codebook
tokens), `infer_out/contours_aux150.npz`

**Two limits**
- **1,134 of 1,452 windows (78%) were discarded.** The repo's filter drops any window with
  3 or more consecutive unvoiced frames (30 ms), and a 1.28 s pansori phrase almost always
  contains a consonant or a breath. The surviving 22% skew towards long sustained-vowel
  spans — that is where sigimsae lives, so the direction of the bias is not bad, but it
  should be stated that these are **statistics of "spans where phonation does not break"**.
- Windows were taken overlapping at hop 32 (`--hop`), so the samples are not independent and
  the confidence intervals on the proportions look narrower than they really are. The
  conclusion's magnitude (84.9% vs 17.3%) is large enough that the direction does not flip.

### Stage B — contour perceptual loss (the main prescription, 1–2 days)

Add exactly one loss term to `pansori_aux`'s training. The model structure is unchanged.

```
log₂f0 ──resample 240→128── window ──frozen encoder── z (32-dim)
loss_contour = ‖ z(pred) − z(gt) ‖²
total_loss += λ_contour · loss_contour
```

**Why this should break the flattening** — this is the plan's central argument:

1. The encoder subtracts the median per window, so it **never sees absolute pitch.** It sees
   shape alone.
2. Being trained with `OffsetMSELoss`, it produces **the same latent even when timing is
   slightly off.** The reason L1 flattens is "since the timing cannot be matched, averaging
   is better" — but **this latent is lenient about timing, so the signal to "move" survives.**
3. The latents of a flat contour and a sigimsae contour are far apart (verified by gate B0).

Implementation points:

- `usr/contour_tokenizer.py` — port `modules.ConvEncoder`,
  `model_utils.calc_conv_params` and `Conv1DAE.subtract_median_from_x`. Load only
  `encoder.*` from the state_dict, with `requires_grad=False` and `eval()`. The checkpoint
  was saved by torch 2.8 but uses zipfile serialisation, so torch 2.0.1 can read it (**do the
  load test first**).
- Resample: slice into 240-frame units and
  `F.interpolate(size=128, mode='linear', align_corners=False)`
- **Unvoiced handling**: the original dataset fills `confidence < 0.5` by interpolation
  (`dataset_zoo.py:183-204`) and discards any window whose consecutive gaps exceed
  `num_interpolate` (2) (`filter_receptive_field`). Use the same rules — uv frames are
  already interpolated by `norm_interp_f0`, and windows with long gaps are excluded from the
  loss.
- λ scale: z is 32-dimensional and the input is in octaves. Measure its magnitude against
  `loss_f0` over a few warmup steps and match. Starting candidates: **λ = 0.1 – 1.0**.
- The success metrics **already exist**: `within_std` and `within_corr` in
  `midi_baseline.py`.
  - Target: `within_std` 17.4 → at least half of GT's 138; `within_corr` 0.145 → 0.4+
  - Watch: if `val/f0_cents_rmse` gets much worse, λ is too high

### 🔄 Stage B progress

**Done**

- The checkpoint **loads** under the `diffsinger` env (py3.8 / torch 2.0.1). 51 tensors; the
  encoder is 6 layers of Conv1d(k=3) with maxpool `[4,2,2,2,2,2]` (product 128), and
  `quantizer.embedding` is 256×32.
- `usr/contour_tokenizer.py` — encoder + codebook ported. It loads under the upstream key
  names unchanged.
- `pansori/eval/tokenizer_parity.py` — **bit-exact verification passed.** Over 366 real
  contour windows, maximum latent error `0.000e+00`, token agreement 100%, and the
  multi-window path agrees too. Re-run this test whenever the port is touched.
- **Gradient check**: gradients flow to all 23 pitch_predictor parameters, and into the
  encoder too via `predictor_grad=0.1`. The tokenizer is frozen. Only
  `pitch_predictor.linear.bias` has a gradient near zero (9.3e-10), and that is **by
  design** — the bias translates the whole contour and the median subtraction erases it.
  Evidence of the absolute-pitch invariance.
- **λ decided**: on `pansori_aux`, f0 L1 = 0.1464 (log2, = 176 cents) and contour MSE =
  0.0023 → at λ=20 the contour term is about 0.3× the f0 term (0.43× measured on the
  validation set). For reference, a contour MSE of 0.0023 matches the 0.0019 predicted from
  Stage A's `‖z‖=0.247` — two independent implementations cross-checking each other.

**Experiment**: `usr/configs/midi/cascade/pansori/aux_contour.yaml`, exp `pansori_aux_c1`.
Continued from `pansori_aux`'s best (13k) via `load_ckpt` for 20k steps at λ=20, hop=32.
It is not trained from scratch because the question is "**can sigimsae be put into a
predictor that already sings the right notes?**", and that answer comes within a few
thousand steps.

**Caution — window survival rate**: with batch padding also counted as unvoiced, only 10.3%
of windows survived in the probe (hop=64). Lowering to hop=32 compensated, but real training
is better than this because `sort_by_len` keeps padding low. If the `contour` loss sticks
near zero, the windows have been wiped out — suspect that first.

**Not yet present — a control.** There is no run continued for the same number of steps at
λ=0. That said, λ=0's behaviour is already known (`pansori_aux` has flat `val/f0` after 10k,
worsening `uv`/`pdur`, within 17.4 cents). If the result is ambiguous, the control has to be
run.

### ❌ Stage B result — the premise was wrong. Aborted.

`pansori_aux_c1`, λ=20, hop=32, continued from `pansori_aux`'s best for 4,000 steps.

**The validation loss does not move.**

| step | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| contour | 0.0354 | 0.0365 | 0.0351 | 0.0350 |
| f0 | 0.1536 | 0.1627 | 0.1549 | 0.1568 |

Sigimsae is nearly unchanged too (150 utterances):

| | `pansori_aux` | `pansori_aux_c1` |
|---|---|---|
| Within-note variation | 33.6 | 36.7 |
| (note interiors only) | 18.9 | 20.6 |
| Correlation with CREPE | 0.129 | 0.126 |

**It is not a shortage of λ.** At the pitch_predictor, the contour loss's gradient norm is
0.3025 against f0 L1's 0.2575 — **the contour side is 1.175× stronger.** Windows are
plentiful too, at 48.2 per batch. A gradient stronger than f0's that still does not lower
the loss means **there is nowhere lower to go**.

**Why — measure the latent's phase sensitivity and it comes out.** (The table below uses
loose preprocessing, without the gap filter or within-window interpolation, so it cannot be
compared in absolute terms against the training loss — only within the table.)

| contour | latent MSE |
|---|---|
| Staircase (perfectly flat) | 0.01727 |
| Model c1 | 0.01428 |
| **CREPE shifted by 30 ms** | **0.00123** |
| **CREPE shifted by 100 ms** | **0.00686** |
| **CREPE shifted by 250 ms** | **0.01499** |
| Entirely zero contour | 0.00991 |

**Producing exactly the right sigimsae 250 ms out of place is no better than a flat
staircase** (0.01499 vs 0.01727). Being off by only 100 ms already loses 40% of the distance
to the staircase.

So the premise Stage B rested on — "thanks to `OffsetMSELoss` the latent is lenient about
timing" — **does not hold at the scale that matters.** Those candidates are tempo ±10% and
small shifts within the decoder window, not arbitrary phase. Which means the logic that
flattened L1 — the timing cannot be hit, so the mean is optimal — applies to latent MSE
just as well. **MSE averages, whatever space it is measured in.**

Incidentally: a completely flat zero contour (0.00991) is closer to ground truth than the
staircase (0.01727). What this latent punishes hardest is not flatness but **the staircase's
note leaps**.

**Conclusion: Stage B is structurally blocked. Changing the loss space is not enough; the
output has to be made impossible to average.** → §5.5 or Stage C.

The artefacts are kept: `usr/contour_tokenizer.py` (bit-exact verified) and
`pansori/eval/contour_tokens.py` are used as-is in Stage C.
`checkpoints/pansori_aux_c1` can be deleted.

Risks and mitigations (in hindsight, the third one did happen — but in reverse: not "any
sigimsae passes", rather that emitting one with the wrong phase is worse than emitting none,
so emitting none became optimal):

| Risk | Mitigation |
|---|---|
| The median's gradient flows only to the selected element | `detach()` the median, treat it as a constant, and only subtract |
| 6 maxpool layers dilute the gradient | Raise λ, or also match intermediate-layer features (standard perceptual-loss practice) |
| **The latent is invariant to tempo/zoom, so "any sigimsae" passes** | That is an expected outcome. If `within_std` rises but `within_corr` does not, read it as a **partial success** — if the goal is sounding like pansori, the timing need not be exact. But admit this in advance and interpret the metrics that way |

### Stage C — retrain a residual tokenizer (the structural fix, 1 week+)

For when Stage B is not enough. Solves §4(c) head on.

1. **Build a residual corpus** — take `CREPE − note staircase` (cents) as a
   pseudo-frequency: `f = 440 · 2^(resid_cents/1200)`. CREPE is in
   `pansori/data/full/f0/*.npy` and the staircase comes from
   `midi_baseline.render_staircase`. Write it as a two-column `frequency,confidence` CSV and
   the tokenizer pipeline takes it unchanged.
2. **Retrain with a shorter window** — `dataset.params.window` 128 → **32–48** (0.32–0.48 s
   = one sigimsae unit). Everything else stays as `foffset_vqvae`.
   ```sh
   python src/train.py --config-name foffset_vqvae \
       data.data_pth=/path/to/residual_csvs dataset.params.window=48
   ```
3. **Replace the pitch predictor with a token classifier** — 256-way cross-entropy per
   window, and add the decoded residual onto the note staircase.
   **Cross-entropy does not average. That is the structural fix for L1's flattening.**

Get that far and the staircase carries note control while the tokens carry sigimsae.

---

## 5.5. A parallel recommendation — a cheap experiment that needs no tokenizer (half a day)

The root cause of the flattening is **deterministic regression + L1**, not the
parameterisation. That can be verified without the tokenizer, and it **pre-confirms Stage
C's premise (classification-based f0)**.

`utils/pitch_utils.py` already has a **256-bin discrete f0 representation** — `f0_to_coarse`
produces 1..255 per frame for `pitch_embed` (resolution ≈ 21 cents/bin).

- Change `PitchPredictor`'s `odim` from 2 to 257 (256 bins + uv)
- Change `add_f0_loss`'s L1 to cross-entropy against an `f0_to_coarse(f0)` target
- Inference by argmax or sampling

Cheap, orthogonal to the tokenizer work, and it answers "is L1 the culprit?" within a day.

### 🔄 §5.5 implementation (in progress)

`usr/pansori_fs2.py` + `usr/configs/midi/cascade/pansori/aux_cls.yaml`, exp
`pansori_aux_cls1`. **Not one upstream file was touched** — it is all subclasses and
overrides inside `usr/` (`build_tts_model`, `add_pitch_loss`, `plot_pitch`, `load_ckpt`,
`add_pitch`).

**A new bin grid was built.** The original plan was to reuse the 256 bins of
`utils.pitch_utils.f0_to_coarse`, but on measuring, that is a mel scale and so has different
resolution at different registers:

| f0 | `f0_to_coarse` resolution |
|---|---|
| 61.7 Hz (male tongseong, MIDI 35) | **72.7 cents/bin** |
| 220 Hz | 24.2 |
| 740 Hz | 10.4 |

A 135-cent sigimsae collapses into 2 bins down low, and 55–750 Hz only uses bins 3–192 out
of 256. So a **cents-uniform** grid is used instead: 55–750 Hz = 3.77 octaves = 4,527 cents
over 256 bins = **17.7 cents/bin, uniform across the whole register**. The measured
quantisation round-trip error is 4.4 cents mean / 8.8 cents max, matching the theoretical
bound and negligible against the 135 cents we are trying to recover. `f0_to_coarse` is left
alone for `pitch_embed`.

**One thing the smoke test exposed** — decode standard deviation from a randomly initialised
state:

| Decode | std |
|---|---|
| `sample` | 1.09 |
| `argmax` | 0.60 |
| `expect` (probability-weighted mean) | **0.004** |

`expect` collapses to almost a constant. Averaging *is* flattening, reproduced right in front
of us, and it is the evidence that **sampling is the essential component of this
prescription**. All three modes are kept because running with `f0_decode=argmax` gives the
control group directly.

Warm start: the conv stack is inherited from `pansori_aux`'s best, and only the output layer
(2 channels → 257) is dropped on a shape mismatch and learned fresh. Confirmed by f0 CE
starting at 5.565 ≈ ln(256).

**Caution**: `losses['f0']` is now CE, so it **cannot be compared** against a regression
run's `val/f0` (L1, around 0.15). `f0cent` (mean absolute cents error of the argmax decode)
is logged alongside it, and since that is a diagnostic rather than a loss it is kept out of
`val_loss` — otherwise a value of order 200 would dominate `save_best`.

### ❌ §5.5 result — half right. Aborted.

`pansori_aux_cls1`, 4,000 steps. f0 CE went 5.54 → 3.75 and `f0cent` 928 → 186 cents,
**reaching the regression baseline's pitch accuracy (184 cents)**. Training itself is fine.

**Sigimsae (150 utterances; ground truth is 134.9 over note interiors)**

| | Within-note variation | vs ground truth | Correlation |
|---|---|---|---|
| Regression baseline | 18.9 | 14% | 0.129 |
| Stage B | 20.6 | 15% | 0.126 |
| cls + argmax | 20.9 | 16% | 0.103 |
| **cls + sample** | **216.9** | **161%** | **0.015** |

**Temperature sweep** (60 utterances; ground truth 125.6)

| Decode | Within-note (interior) | vs GT | Correlation | cents RMSE | Octave jumps |
|---|---|---|---|---|---|
| argmax | 22.4 | 18% | 0.049 | 254.1 | 0.02% |
| sample T=0.2 | 47.5 | 38% | 0.048 | 260.3 | 0.49% |
| sample T=0.4 | 80.2 | 64% | 0.016 | 269.6 | 1.60% |
| sample T=0.6 | 117.0 | 93% | 0.027 | 286.0 | 3.40% |
| sample T=0.8 | 163.9 | 130% | 0.017 | 315.6 | 6.69% |
| sample T=1.0 | 217.7 | 173% | 0.018 | 350.7 | 10.35% |

**Temperature is only an amplitude knob; the correlation is 0.02–0.05 everywhere.** At
T=0.6 the amplitude matches ground truth at 93%, but the content is unrelated. There is no
sweet spot.

**Why — measure the width of the predicted distribution and it comes out.**

| | |
|---|---|
| Predicted distribution std | **228 cents** |
| Effective width exp(H) | 36 bins = **638 cents** |
| Sigimsae width | **126 cents** |

**The distribution is twice as wide as sigimsae, and its effective width exceeds five
semitones.** What that uncertainty is about is not "which sigimsae" but **"which note"**. So
sampling draws note-level errors, and those show up as octave jumps.

**What §5.5 answered and what it did not**

- ✅ "Is L1's averaging the culprit?" → **partly yes.** Removing the averaging let the model
  express variance (the difference between argmax and sample is the evidence).
- ❌ But it is **not sufficient.** Drawing independently from per-frame marginals gives white
  noise no matter how well you draw. What is needed is a joint distribution over time.

**The two missing pieces are now clear:**

1. **Residual parameterisation** — condition on the note and model only the deviation above
   it, so that the distribution being sampled is at sigimsae scale (±126 cents) rather than
   note scale (±228 cents).
2. **Temporal coherence** — a contour is not a sequence of independent draws. It has to be
   drawn as one piece.

**Stage C gives exactly these two at once.** One token is one residual shape and one
temporally coherent piece. §5.5, taken as a detour, ended up as the evidence supporting
Stage C's design.

Note: the codebase's `pitch_ar` path (the AR pitch predictor) is **broken** —
`modules/fastspeech/fs2.py:216` calls `self.pitch_predictor(decoder_inp, f0)` but
`PitchPredictor.forward` takes only one argument. There is no cheap AR option.

---

## 6. Order and gates

```
Stage A (half a day) ──gate B0──┬─ pass ─→ Stage B (1-2 days) ──┬─ within_std ≥ 50% of GT ─→ done, retrain ds3
                                │                                └─ short ─→ Stage C
                                └─ fail ─→ Stage C (1 week+)

        └─ in parallel: §5.5 cross-entropy f0 (half a day)
```

Reproduction:
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python pansori/eval/midi_baseline.py \
    --config usr/configs/midi/cascade/pansori/aux.yaml \
    --exp_name pansori_aux --target aux --n 24 --html
```

## 7. Not yet confirmed

- Whether the checkpoint loads under torch 2.0.1 / Python 3.8 (Stage B's first task)
- Gate B0 — whether the codebook really distinguishes flatness from sigimsae
- Whether the tokenizer's training corpus overlaps our `Pansori_Data`. If it does, Stage A's
  held-out character is weakened (check the song titles against
  `splits/inhouse_crepe_split.csv`)
- 24 utterances is a small sample. If conclusions are marginal, re-confirm at `--n 100` or
  more
