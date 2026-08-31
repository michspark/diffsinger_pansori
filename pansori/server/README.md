# `run.sh` — 단계별 실행기

`setup / binarize / train-ac / train-var` 네 단계를 묶어 둔 스크립트다. 원래는 서버용으로
썼지만 로컬 GPU 에서도 그대로 돈다 — 실제로 첫 학습은 로컬 RTX 4090(24 GB)에서 기동했다.

```sh
tmux new -s pansori                   # 오래 도는 단계라 detach 가능한 쪽이 낫다
bash pansori/server/run.sh setup      # env + 의존성 + 체크포인트 (한 번만)
bash pansori/server/run.sh binarize   # 07 재검증 → acoustic + variance 바이너리
bash pansori/server/run.sh train-ac   # acoustic 학습
bash pansori/server/run.sh train-var  # variance 학습
# Ctrl-b d 로 detach, tmux a -t pansori 로 복귀
```

로그는 `logs/` 에 남는다. env 이름은 `ENV_NAME`(기본 `dsp`), GPU 는 `CUDA_VISIBLE_DEVICES` 로.

## 다른 기계로 옮길 때

저장소는 git 으로, 데이터는 gitignore 대상이라 따로 복사한다.

```sh
git clone <이 저장소> DiffSinger && cd DiffSinger && git checkout feat/pansori
rsync -av --info=progress2 ~/Desktop/DiffSinger/data/pansori/ 서버:~/DiffSinger/data/pansori/
```

`data/pansori/`(8.7 GB) 만 있으면 학습까지 간다. `pansori/data/full/`(중간 산출물 160 MB +
`aligned/` 119 MB)은 **라벨을 다시 만들 때만** 필요하다 — 같이 옮겨두면 MFA 재실행 없이
`04_align.py --skip_mfa` 부터 돌릴 수 있다.

`checkpoints/` 는 `setup` 이 다시 받으므로 옮기지 않아도 된다.

## `setup` 이 하는 일

`conda create -n dsp python=3.10` → torch/torchaudio(cu124) → `requirements.txt` →
체크포인트 3종. 아래는 그냥 알고 있어야 하는 것들이다:

- **`PYTHONNOUSERSITE=1`** 을 env 의 `activate.d` 에 심는다. `~/.local/lib/python3.10/
  site-packages` 가 conda env 앞에 끼어드는데, 거기 낡은 tensorflow 가 있으면 tensorboard
  import 가 protobuf 에서 죽고, 거기 numpy 가 있으면 pip 이 요구사항을 "이미 설치됨"으로
  오판해 env 안에 안 넣는다.
- **`torchaudio`** 는 `requirements.txt` 에 없다 (upstream 누락). RMVPE 가 `Resample` 을 쓴다.
- **`setuptools<81`** — lightning 2.3 이 `pkg_resources` 를 import 한다.
- **`checkpoints/vr/`(HN-SEP)** 는 breathiness/voicing/tension 을 전부 꺼도 필요하다.
  acoustic binarize 가 `DecomposedWaveform` 을 무조건 만들고 기본 `hnsep` 이 `vr` 이다.

## 손이 필요한 지점

- **spec 통계는 부트스트랩이다.** `binarize` 를 기본값으로 한 번 돌린 뒤
  `python pansori/compute_spec_stats.py` → `python pansori/make_configs.py` 로 config 를
  재생성해야 `configs/pansori/spec_stats.yaml` 이 `base_config` 에 얹힌다.
  이 코퍼스 실측 상한이 **+2.838** 이라 openvpi 기본값 `spec_max: [0]` 으로는 diffusion
  타깃이 정규화 후 [-1, 1] 을 벗어난다. **재-binarize 는 필요 없다** — 학습 시점 값이다.
- **사전학습 모델 (선택).** 지금은 scratch 로 돈다. fine-tune 하려면 `make_configs.py` 에서
  `finetune_enabled: true` + `finetune_ckpt_path` 를 채운다. 중국어 음소셋이라
  `finetune_ignored_params` 에 phoneme/speaker embedding 이 들어가 있는데,
  **실제 state_dict 키는 체크포인트마다 다르므로 받은 뒤 대조**할 것:
  ```sh
  python -c "import torch;print('\n'.join(torch.load(CKPT, map_location='cpu')['state_dict'].keys()))" | grep -i embed
  ```
