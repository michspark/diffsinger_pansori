# Things to know — config and data traps

When the procedure in `README.md` stops working, start here. Every one of these was hit for
real at least once.

---

- **Speakers are defined per `datasets:` entry.** All 23 (22 myeongchang + the pooled
  `unknown`) get their own directory. `make_configs.py` assigns `spk_id` in descending order
  of utterance count.
- **A hold-out song that accounts for a speaker's entire output is released automatically.**
  Jeong Gwon-jin (226 utterances) was that case and was dropped from the hold-out — a
  speaker with zero training data never learns an embedding at all.
- **Do not omit `f0_min: 55` / `f0_max: 750`.** The default of 65 Hz cuts off male
  *tongseong*.
- **Do not leave `spec_max` at its default `[0]`.** The measured ceiling for this corpus is
  **+2.838**, so after normalisation the diffusion target leaves [-1, 1]. The floor is safe
  the other way round — `nvSTFT` compresses with `log(clamp(x, 1e-5))`, so nothing below
  -11.513 can exist.
- **Whether `AP`/`SP` are merged is decided by the corpus, not by a person.** openvpi always
  includes `AP` (breath) in the phoneme set, and binarize only passes if every member of the
  set actually appears in the corpus. Labels from the MFA arm carry no breath distinction,
  so `merged_phoneme_groups: [[SP, AP]]` gets attached. If breath is genuinely labelled via
  SOFA + FoxBreatheLabeler (`breath_ap.enabled` in `preprocess/align.yaml`), then
  `make_configs.py` sees `AP` in `transcriptions.csv` and **removes that block
  automatically** — do not delete it by hand. When the labels and the phoneme set disagree,
  binarize dies, and which one is wrong is knowable only from the CSV.
- **The most recent checkpoint is not the best one.** In the old pipeline, overfitting
  started around step 56k. The architecture has changed since, so the number does not carry
  over, but the habit should.

---

## Training and inference

- **There are three prerequisite downloads** (`server/run.sh setup` fetches them):
  `checkpoints/rmvpe/`, `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`, and
  `checkpoints/vr/`. The last is HN-SEP, which is needed even if breathiness and everything
  like it is switched off, because acoustic binarize builds a `DecomposedWaveform`
  unconditionally and stops there without it.
- **wandb will not reuse a run id that has been deleted once.** If you uploaded to the wrong
  place and deleted it, pass a fresh id with `--run-id` and keep the display name via
  `--exp`.
- **Checkpoints differ per head.** The variance model's `dur_loss` and `pitch_loss` bottom
  out at different steps. `run_examples.sh` picks per head and runs variance twice.
- **`--spk` cannot take a Hangul speaker name** (`utils/infer_utils.py:56` matches
  `[0-9A-Za-z_-]+`). `make_ds.py` works around it by writing `spk_mix` into the `.ds`
  directly.
