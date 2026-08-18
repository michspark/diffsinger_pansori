# 서버에서 이어서 돌리기 (tmux)

로컬에서 끝난 것: 전처리 전 단계, 44.1 kHz 재커팅(`data/pansori/`, 8.7 GB), 라벨 검증 PASS,
config 생성. 여기서부터는 GPU 와 사전 다운로드가 필요해 서버에서 돌린다.

## 0. 옮길 것

저장소는 git 으로 가져오고, 데이터는 gitignore 대상이라 따로 복사해야 한다.

```sh
# 서버에서
git clone <이 저장소> DiffSinger && cd DiffSinger && git checkout feat/pansori

# 로컬에서 — 8.7 GB. 이것만 있으면 학습까지 간다 (pansori/data/full 은 필요 없음)
rsync -av --info=progress2 ~/Desktop/DiffSinger/data/pansori/ 서버:~/DiffSinger/data/pansori/
```

`pansori/data/full/`(중간 산출물 160 MB + aligned 119 MB)은 라벨을 다시 만들 때만 필요하다.
같이 옮겨두면 `04_align.py --skip_mfa` 부터 재실행할 수 있어 편하다.

## 1. 실행

```sh
tmux new -s pansori
bash pansori/server/run.sh setup      # env + 의존성 + 체크포인트 (한 번만)
bash pansori/server/run.sh binarize   # acoustic + variance 바이너리
bash pansori/server/run.sh train-ac   # acoustic 학습
bash pansori/server/run.sh train-var  # variance 학습
# Ctrl-b d 로 detach
```

`run.sh` 는 각 단계 로그를 `logs/` 에 남긴다. `tmux a -t pansori` 로 다시 붙는다.

## 2. 손이 필요한 지점

- **체크포인트 URL.** `run.sh` 의 `RMVPE_URL` / `VOCODER_URL` 이 비어 있다. openvpi 릴리스에서
  받아 채울 것. 이게 없으면 binarize(f0 추출)와 학습 검증(vocoder)이 돌지 않는다.
- **사전학습 모델.** fine-tune 하려면 `configs/pansori_*.yaml` 의 `finetune_enabled: true` +
  `finetune_ckpt_path` 를 채운다. 중국어 음소셋이라 `finetune_ignored_params` 에 phoneme/speaker
  embedding 이 들어가 있는데, **실제 state_dict 키는 체크포인트마다 다르므로 받은 뒤 대조**할 것:
  ```sh
  python -c "import torch;print('\n'.join(torch.load(CKPT, map_location='cpu')['state_dict'].keys()))" | grep -i embed
  ```
  비워두면 scratch 학습으로 그냥 돈다.
- **spec_min/spec_max.** binarize 후 `compute_spec_stats.py` 로 실측해 config 에 반영한다.
  구 파이프라인에서 Opencpop 기본값을 쓰면 정규화 후 mel 이 +1 을 넘겼다(실측 1.257).
  ※ 이 스크립트는 아직 24 kHz/80 mel 기준이라 128 mel 로 손봐야 한다.
