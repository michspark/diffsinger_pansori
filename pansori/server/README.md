# `run.sh` — the stage runner

A script bundling four stages: `setup / binarize / train-ac / train-var`. It was originally
written for a server, but runs unchanged on a local GPU — the first training run was
actually started on a local RTX 4090 (24 GB).

```sh
tmux new -s pansori                   # these stages run long; being able to detach helps
bash pansori/server/run.sh setup      # env + dependencies + checkpoints (once)
bash pansori/server/run.sh binarize   # re-validate 07 → acoustic + variance binaries
bash pansori/server/run.sh train-ac   # train acoustic
bash pansori/server/run.sh train-var  # train variance
# Ctrl-b d to detach, tmux a -t pansori to come back
```

Logs go to `logs/`. The env name comes from `ENV_NAME` (default `dsp`), the GPU from
`CUDA_VISIBLE_DEVICES`.

## Moving to another machine

The repository moves by git; the data is gitignored, so copy it separately.

```sh
git clone <this repository> DiffSinger && cd DiffSinger && git checkout feat/pansori
rsync -av --info=progress2 ~/Desktop/DiffSinger/data/pansori/ server:~/DiffSinger/data/pansori/
```

`data/pansori/` (8.7 GB) is all you need to get as far as training.
`pansori/data/full/` (160 MB of intermediates + 119 MB of `aligned/`) is needed **only when
rebuilding the labels** — bring it along and you can start from
`04_align.py --skip_align` without re-running MFA.

`checkpoints/` does not need moving; `setup` downloads it again.

## What `setup` does

`conda create -n dsp python=3.10` → torch/torchaudio (cu124) → `requirements.txt` → the
three checkpoints. The following are just things you have to know:

- It plants **`PYTHONNOUSERSITE=1`** in the env's `activate.d`.
  `~/.local/lib/python3.10/site-packages` cuts in front of the conda env: an old tensorflow
  there kills the `tensorboard` import at protobuf, and a numpy there makes pip misjudge
  requirements as "already satisfied" and never install them into the env.
- **`torchaudio`** is not in `requirements.txt` (an upstream omission). RMVPE uses
  `Resample`.
- **`setuptools<81`** — lightning 2.3 imports `pkg_resources`.
- **`checkpoints/vr/` (HN-SEP)** is required even with breathiness / voicing / tension all
  switched off: acoustic binarize builds a `DecomposedWaveform` unconditionally, and the
  default `hnsep` is `vr`.

## Where a human is still needed

- **Spec statistics are a bootstrap.** Run `binarize` once with the defaults, then
  `python pansori/compute_spec_stats.py` → `python pansori/make_configs.py` to regenerate
  the configs, so that `configs/pansori/spec_stats.yaml` gets layered into `base_config`.
  The measured ceiling for this corpus is **+2.838**, so with openvpi's default
  `spec_max: [0]` the diffusion target leaves [-1, 1] after normalisation.
  **No re-binarize is needed** — the value is read at training time.
- **Pre-trained model (optional).** Right now it trains from scratch. To fine-tune, set
  `finetune_enabled: true` and fill in `finetune_ckpt_path` in `make_configs.py`. Because
  the phoneme set is Chinese, `finetune_ignored_params` already lists the phoneme and
  speaker embeddings — but **the actual state_dict keys differ per checkpoint, so check
  them against the file after downloading it**:
  ```sh
  python -c "import torch;print('\n'.join(torch.load(CKPT, map_location='cpu')['state_dict'].keys()))" | grep -i embed
  ```
