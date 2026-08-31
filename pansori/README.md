# 판소리 fine-tune — openvpi/DiffSinger

판소리(한국 전통 서사창)로 DiffSinger 를 fine-tune 하기 위한 전처리 파이프라인.
원본 오디오에서 합성음까지 **단계별 전체 기록**은 [`docs/PIPELINE.md`](docs/PIPELINE.md) 에 있다 —
아래는 실행법 요약이다. 이식 경위와 설계 판단은 [`docs/PORTING.md`](docs/PORTING.md),
6초 제약의 근거는 [`alignment/VERDICT.md`](alignment/VERDICT.md).

우리 작업은 `pansori/` 와 `dictionaries/korean-coda.txt`, `configs/pansori_*.yaml` 에만 있다.
**upstream 파일은 한 줄도 고치지 않았다** — rebase 를 계속 감당할 수 있게 그 상태를 유지할 것.

---

## 1. 레이아웃

```
pansori/
├── preprocess/            01~08 단계. 단계 사이는 jsonl 로 이어져 개별 재실행 가능
│   └── pansori_lib.py     공유 상수, IPA→Coda 변환 규칙, 노트 슬라이싱, jsonl IO
├── alignment/             MFA 타당성 검토 + IPA→Coda 매핑표
│   ├── VERDICT.md         ★ 왜 6초인가 — 세그멘테이션을 바꾸기 전에 읽을 것
│   ├── ipa_to_coda.csv    korean_mfa IPA (108) → Coda (34)
│   └── check_dict_agreement.py  ★ 매핑을 고치기 전후로 반드시 돌릴 채점기
├── examples/              ★ 추론·평가. 학습 후 여기서부터 돈다 (§5)
│   ├── run_examples.sh    사다리 4단 + 악보 합성 일괄 생성
│   ├── scores/*.txt       악보 (음절 노트 길이초). 파일을 넣으면 자동으로 합성된다
│   ├── measure.py         f0 RMSE · mel L1 · 유무성 일치
│   ├── f0_report.py       시김새 평탄화 측정 (--seeds N 으로 중앙값 판정)
│   ├── seed_spread.py     같은 입력의 시드별 편차 — 수치를 믿기 전에 볼 것
│   ├── ab_ladder.py       두 모델을 홀드아웃 N쌍으로 비교 + 부호검정 (§7)
│   ├── best_ckpt.py       검증 손실 최저 체크포인트 선택
│   └── build_report.py    설명 + 오디오를 한 페이지로 렌더링
├── lyricbook/             창본 15개 → corpus.jsonl · VERDICT.md (가사 A/B 의 재료, §7)
├── compare/               구 모델(MoonInTheRiver) 대비 정량 비교 패키지
├── experiments/           세그멘테이션 알고리즘 시뮬레이션 (01 의 근거)
├── eval/                  ★ 음소·발음 계측 (§8)
│   ├── intelligibility.py 사다리 4단 × {whisper CER, MFA 재정렬} — 발음이 어디서 무너지나
│   ├── label_report.py    학습 라벨 진단 (ph_dur 하한선 스파이크) → 콘솔·json·HTML
│   └── (구) resynth_ladder.py / f0_metrics.py / midi_baseline.py
│                          ※ 24 kHz 기준 구 스크립트. examples/ 가 대체하므로 폐기 검토
├── metrics/               부류별 음소 길이 지표 (torchmetrics) — variance_task.py 가 등록
├── variance_task.py       VarianceTask 서브클래스. upstream 무수정으로 지표를 붙인다 (§8-2)
├── docs/                  ★ PIPELINE.md — 단계별 전체 기록 · TRAINING.md — 학습 한 스텝씩
│                          PORTING.md, NEXT.md 외
├── train.py               scripts/train.py + wandb 래퍼 (upstream 무수정, §4)
├── tb_to_wandb.py         끝난 학습의 텐서보드 로그 → wandb (§4)
├── build_dictionary.py    한글 11,172음절 → Coda 34음소 사전 생성
├── make_ds.py             라벨 또는 악보 → .ds (추론 입력)
├── make_configs.py        화자 목록을 읽어 configs/pansori_*.yaml 렌더링
└── compute_spec_stats.py  이 코퍼스의 mel 포락선 실측 → spec_min/spec_max
```

데이터는 전부 gitignore 대상이며 저장소에 없다.

| 무엇 | 어디 |
|---|---|
| 원본 오디오/라벨/노트 | `~/Desktop/Pansori_Data/` (`Audio/`, `labels_2.json`, `rosvot_midi/`) |
| 전처리 중간 산출물 | `pansori/data/full/` |
| 학습용 코퍼스 (44.1 kHz) | `data/pansori/{화자}/` (8.7 GB) |
| binarize 결과 | `data/pansori/binary_{acoustic,variance}/` |

원본 경로는 `pansori/preprocess/pansori_lib.py` 의 `PANSORI_DATA` 하나에만 하드코딩돼 있다.

---

## 2. 환경

| env | Python | 쓰는 곳 |
|---|---|---|
| `pansori` | 3.10 | 전처리 01–03, 06–08 |
| `aligner` | 3.14 | MFA 3.4.1 + `korean_mfa` + mecab-ko → 04–05 |
| `dsp` | 3.10 | openvpi 본체 (binarize / train / infer) |

- 옛 `diffsinger`(3.8) env 는 못 쓴다 — openvpi 는 Python ≥ 3.10 / PyTorch ≥ 2.4 를 요구한다.
- **`base` 도 못 쓴다** — numba 가 numpy 2.5 와 충돌해 06 이 죽는다. 전처리는 `pansori` 로.
- **`dsp` 는 `PYTHONNOUSERSITE=1` 이 필수다.** `~/.local/lib/python3.10/site-packages` 의
  낡은 tensorflow 가 tensorboard import 를 protobuf 에서 죽이고, 거기 numpy 때문에 pip 이
  요구사항을 "이미 설치됨"으로 오판해 env 안에 안 넣는다. env 의 `activate.d` 에 넣어 뒀다.
- `requirements.txt` 에 **`torchaudio` 가 없다** (upstream 누락). RMVPE 가 `Resample` 을 쓴다.

`bash pansori/server/run.sh setup` 이 위를 전부 한다.

---

## 3. 파이프라인

각 단계는 `pansori/data/{set}/` 아래에서 동작하고 `--set` 으로 선택한다.

| # | 스크립트 | in → out | env |
|---|---|---|---|
| 1 | `01_segment.py` | labels + Audio + midi → `segments.jsonl` + `wavs/` | base |
| 1b | `01b_recut_wavs.py` | segments → `wavs/` **복원** (01 을 다시 돌리지 말 것 — 아래) | pansori |
| 2 | `02_speaker_map.py` | segments → `spk_map.json` | base |
| 3 | `03_transcribe.py` | wavs → `lyrics.jsonl` (whisper-large-v3) | base |
| 3b | `03b_snap_lyrics.py` | lyrics + 창본 → `lyrics_snapped.jsonl` (선택, §7) | pansori |
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` (음소 + **words tier**) | aligner |
| 5 | `05_notes.py` | rosvot_midi → `notes.jsonl` | aligner |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | pansori |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | pansori |
| 8 | `08_export.py` | 44.1 kHz 재커팅 + 화자별 분할 | pansori |

**`01_segment.py` 를 다시 돌리면 안 된다.** 재현되지 않는다 — 문서상 실행 env 인 `base` 는
이제 numba 가 numpy 2.5 를 거부해 임포트부터 죽고, 다른 env 에서 돌리면
`librosa.effects.split` 이 경계를 3~11 ms 옮겨 31,529 대신 31,378 발화가 나온다. 그 경계로
자르면 이미 만들어 둔 `phones.jsonl` 의 모든 음소가 오디오에서 몇 ms 씩 어긋난다.
`wavs/` 가 필요하면 **매니페스트에서 복원**한다 — 이미 내보낸 발화는 `data/pansori` 로
심볼릭 링크하고 나머지만 원본에서 잘라낸다:

```sh
python pansori/preprocess/01b_recut_wavs.py --set full     # 31,529개, 새로 자르는 건 4,574개
```

정렬 결과(`aligned/`)가 남아 있으면 **MFA 재실행 없이** 4단계를 다시 파싱만 할 수 있다:

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_mfa   # ~2분
conda activate pansori
python 06_merge.py --set full && python 07_validate.py --set full   # PASS 확인 필수
python 08_export.py --set full --out data/pansori                   # 오디오 재커팅 ~20분
```

`ipa_to_coda.csv` 나 `convert_phone` 을 고쳤다면 **그 전후로 채점기를 돌릴 것.**
추측이 아니라 숫자로 판정한다:

```sh
python pansori/alignment/check_dict_agreement.py                     # 1음절, 현재 89.59%
python pansori/alignment/check_dict_agreement.py --syllables 2 --focus ʎ
```

`07_validate.py` 는 openvpi binarizer 가 워커 안에서 assert 로 죽는 지점을 전부 앞당겨 검사한다.
**PASS 없이 binarize 하지 말 것** — 실패는 스킵이 아니라 실행 중단이다.

---

## 4. 학습

한 스텝 안에서 벌어지는 일(배치 구성 → condition → reflow → 손실 → 체크포인트 규칙)은
[`docs/TRAINING.md`](docs/TRAINING.md) 에 단계별로 적었다.

```sh
conda activate dsp && export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0

python pansori/build_dictionary.py          # dictionaries/korean-coda.txt
python pansori/make_configs.py              # configs/pansori_{acoustic,variance}.yaml

python scripts/binarize.py --config configs/pansori_acoustic.yaml
python scripts/binarize.py --config configs/pansori_variance.yaml
python pansori/compute_spec_stats.py        # → configs/pansori_spec_stats.yaml
python pansori/make_configs.py              # 위 파일을 base_config 에 얹어 재생성

python scripts/train.py --config configs/pansori_acoustic.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/pansori_variance.yaml --exp_name pansori_var_v1 --reset
```

### wandb 로 보기

`pansori/train.py` 가 `scripts/train.py` 를 감싼다. **upstream 은 그대로 두고** wandb 를
붙이는 유일한 방법이다 — 트레이너의 로거는 `DsTensorBoardLogger` 로 고정돼 있고
(`basics/base_task.py:427`), 검증 오디오와 mel 그림은 로거를 아예 거치지 않고
SummaryWriter 로 직접 나간다 (`training/acoustic_task.py:215-242`). 그래서 로거를 갈아 끼우면
스칼라만 얻고 정작 볼 것을 놓친다. `sync_tensorboard=True` 는 SummaryWriter 클래스 자체를
패치하므로 셋 다 따라온다.

```sh
pip install wandb        # dsp env 에 없다. CLI 는 python -m wandb 로 부를 것
                         # (~/.local/bin/wandb 는 PYTHONNOUSERSITE=1 에서 죽는다)

python pansori/train.py --config configs/pansori_acoustic.yaml \
  --exp_name pansori_ac_v1 --reset --lyrics-source whisper --wandb-group lyrics-ab
```

`--no-wandb` 를 주면 `scripts/train.py` 와 동일하게 동작한다. 이미 끝난 학습의 텐서보드
로그는 `tb_to_wandb.py` 로 올린다:

```sh
python pansori/tb_to_wandb.py --exp pansori_ac_v1 --run-id pansori_ac_v1_tb \
  --group lyrics-ab --lyrics-source whisper
```

**`wandb sync --sync-tensorboard` 를 쓰지 말 것.** 동작은 하지만 tfevents 가 들어 있는
디렉토리 이름으로 모든 지표에 네임스페이스를 붙인다 — `lightning_logs/latest` 는
`latest/validation/total_loss` 가 되어 살아 있는 런의 `validation/total_loss` 와 **같은
차트에 겹치지 않는다.** 평평한 디렉토리로 복사해도 wandb 가 부모로 거슬러 올라가 그
이름을 쓴다. `tb_to_wandb.py` 는 이벤트를 직접 읽어 원래 이름 그대로 올리고 검증
그림·오디오도 같이 가져온다.

> wandb 는 **한 번 삭제한 run id 를 다시 쓰지 못한다.** 잘못 올려서 지웠다면
> `--run-id` 로 새 id 를 주고 표시 이름만 `--exp` 로 남길 것.

variance 런은 `metrics/` 아래에 **부류별 음소 길이 지표**(`dur_acc_onset` · `floor_rate_pred`
· `phone_dur_acc` 그림 등)를 같이 낸다 — 무엇이 나오고 어떻게 읽는지는 [§8-2](#8-2-a--학습중-wandb-지표).
config 의 `task_cls` 가 `pansori.variance_task.PansoriVarianceTask` 를 가리키기 때문이고,
upstream 파일은 여전히 한 줄도 안 고쳤다.

**spec 통계는 부트스트랩이 필요하다**: 기본값으로 한 번 binarize → 실측 → config 재생성.
`spec_min`/`spec_max` 는 학습 시점에만 쓰이므로 **재-binarize 는 필요 없다.**

선행 다운로드 (`run.sh setup` 이 받는다): `checkpoints/rmvpe/`,
`checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`, `checkpoints/vr/`.
마지막 것은 HN-SEP 인데, breathiness 등을 전부 꺼도 acoustic binarize 가 `DecomposedWaveform`
을 무조건 만들기 때문에 없으면 거기서 멈춘다.

---

## 5. 추론과 평가

```sh
bash pansori/examples/run_examples.sh          # 사다리 + 악보 합성 → examples/out/
python pansori/examples/measure.py             # 원본 대비 점수
python pansori/examples/f0_report.py --seeds 5 # 평탄화 판정
python pansori/examples/build_report.py        # 설명 + 오디오 한 페이지
```

`run_examples.sh` 는 홀드아웃 발화마다 **사다리 4단**을 만든다. 위에서 아래로 갈수록 모델이
스스로 만들어야 하는 것이 늘어나므로, 결과가 나쁠 때 어느 단계 탓인지 가를 수 있다.

| | 준 것 | 모델이 만든 것 |
|---|---|---|
| `1_orig` | — | — (원본 녹음) |
| `2_resynth` | `ph_dur` + `f0` | mel |
| `3_pitch` | `ph_dur` | `f0` + mel |
| `4_full` | 노트 + 가사 | `ph_dur` + `f0` + mel |

새 노래를 부르게 하려면 `examples/scores/` 에 파일 하나를 넣으면 된다:

```
_   rest  0.30      # '_' 는 쉼표 → SP
춘  A4    0.45
향  G4    0.40
-   F4    0.55      # '-' 는 앞 음절이 이어짐 (시김새·melisma)
```

### 수치를 믿기 전에

- **두 모델 다 reflow 샘플러다.** 시드만 바꿔도 `f0 RMSE` 가 ±17%, `mel L1` 이 ±8% 움직인다
  (`seed_spread.py`). 트랙끼리 비교하려면 시드를 고정하거나 여러 번 뽑아 중앙값을 쓸 것.
- **체크포인트는 헤드마다 다르다.** variance 의 `dur_loss` 와 `pitch_loss` 는 서로 다른
  step 에서 최저다. `run_examples.sh` 가 헤드별로 골라 variance 를 두 번 돌린다.
- **`--spk` 는 한글 화자명을 못 받는다** (`utils/infer_utils.py:55`). `make_ds.py` 가
  `.ds` 의 `spk_mix` 에 직접 써서 우회한다.

---

## 6. 알아둘 것

- **화자는 `datasets:` 항목 단위로 지정된다.** 23명(명창 22 + pooled `unknown`)이 각각 자기
  디렉토리를 갖는다. `make_configs.py` 가 발화 수 내림차순으로 `spk_id` 를 매긴다.
- **홀드아웃 곡이 한 화자의 전부를 차지하면 자동 해제된다.** 정권진(226발화)이 그 경우라
  홀드아웃에서 빠졌다 — 학습 데이터 0인 화자는 임베딩이 아예 학습되지 않는다.
- **`f0_min: 55` / `f0_max: 750` 을 빠뜨리지 말 것.** 기본값 65 Hz 는 남성 통성을 잘라낸다.
- **`spec_max` 기본값 `[0]` 을 그대로 두지 말 것.** 이 코퍼스 실측 상한은 **+2.838** 이라
  정규화 후 diffusion 타깃이 [-1, 1] 을 벗어난다. 하한은 반대로 안전하다 — `nvSTFT` 가
  `log(clamp(x, 1e-5))` 로 압축해 -11.513 아래가 존재할 수 없다.
- **`AP` 는 `SP` 와 합쳐 뒀다.** openvpi 는 음소셋에 `AP`(들숨) 를 항상 넣고 binarize 는 셋의
  모든 항목이 코퍼스에 나타나야 통과시키는데, 우리 라벨에는 들숨 구분이 없다. 들숨을 따로
  라벨하게 되면 `make_configs.py` 의 `merged_phoneme_groups` 를 지우고 재-binarize 할 것.
- **가장 최근 체크포인트가 최선이 아니다.** 구 파이프라인에서는 56k step 부근에서 과적합이
  시작됐다. 아키텍처가 달라졌으니 수치는 이월되지 않지만 습관은 유지할 것.

---

## 7. 가사 A/B — 창본 보정을 하고 / 안 하고

`03b_snap_lyrics.py` 는 Whisper 전사를 창본(`lyricbook/corpus.jsonl`)에 맞춰 교정한다.
[`lyricbook/VERDICT.md`](lyricbook/VERDICT.md) 는 **MFA 정렬 지표로는** 이 교정이 오히려
나쁘다고 판정했지만, 같은 문서 §4 가 그 판정의 한계를 명시한다 — MFA 는 오디오에서 파생된
텍스트를 구조적으로 편애하므로 "어느 가사가 **학습에** 좋은가"는 답할 수 없다. 그 질문은
두 라벨로 각각 학습시켜 합성음을 비교해야 답이 나온다. 이 절이 그 절차다.

**바꾸는 것은 라벨 하나뿐이다.** 오디오·노트·홀드아웃·spec 통계·config·학습 설정이 전부
동일해야 차이를 가사 탓으로 돌릴 수 있다.

### 7-1. snapped 코퍼스 만들기

```sh
conda activate pansori
python pansori/preprocess/01b_recut_wavs.py --set full        # 04 의 입력 복원 (§3)

conda activate aligner && cd pansori/preprocess
python 04_align.py --set full --lyrics lyrics_snapped.jsonl --suffix _snap   # 수 시간

# 06·07 은 data/{set}/ 의 고정된 파일명을 읽는다. 새 --set 을 링크로 조립하면
# 스크립트를 고치지 않고 그대로 돌릴 수 있다.
cd ../data && mkdir -p full_snap && cd full_snap
ln -s ../full/{segments.jsonl,notes.jsonl,spk_map.json,wavs} .
ln -s ../full/phones_snap.jsonl phones.jsonl                  # 다른 것은 이것뿐이다

conda activate pansori && cd ../../preprocess
python 06_merge.py    --set full_snap
python 07_validate.py --set full_snap                         # PASS 없이 진행 금지
python 08_export.py   --set full_snap --out data/pansori_snap --wavs_from data/pansori
```

`--wavs_from` 은 화자별 `wavs/` 를 기존 코퍼스로 심볼릭 링크한다. 두 변종의 오디오는
바이트 단위로 같으므로 14 GB 를 다시 자를 이유가 없고, 같은 파일을 가리키는 편이
"라벨 말고는 전부 같다"를 보장한다.

### 7-2. config 와 binarize

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_snap --name pansori_snap
python scripts/binarize.py --config configs/pansori_snap_acoustic.yaml
python scripts/binarize.py --config configs/pansori_snap_variance.yaml
```

**`compute_spec_stats.py` 를 다시 돌리지 않는다.** mel 통계는 오디오에서만 나오고 오디오는
동일하다. `configs/pansori_spec_stats.yaml` 을 그대로 재사용하는 것이 오히려 조건을
묶어 준다.

### 7-3. 학습 — 두 런을 한 group 으로

```sh
python pansori/train.py --config configs/pansori_snap_acoustic.yaml \
  --exp_name pansori_ac_snap --reset \
  --hparams max_updates=50000,num_ckpt_keep=12 \
  --lyrics-source snapped --wandb-group lyrics-ab

python pansori/train.py --config configs/pansori_snap_variance.yaml \
  --exp_name pansori_var_snap --reset \
  --hparams max_updates=32000,num_ckpt_keep=8 \
  --lyrics-source snapped --wandb-group lyrics-ab
```

- **스텝 수는 baseline 에 맞춘 것이다.** `pansori_var_v1` 은 config 의 80k 가 아니라 실제로
  **32k 에서 멈춰 있다**(체크포인트·tfevents 둘 다). 더 오래 돌리면 snapped 쪽 피치 헤드만
  더 학습돼 비교가 기운다. acoustic 은 baseline 이 100k 까지 갔지만 최적이 28k 였으므로
  50k 로 충분하다.
- **`num_ckpt_keep` 을 올린다.** 기본값(acoustic 8 / variance 5)이면 초반 체크포인트가
  지워지는데, 실측 최적은 acoustic 28k · dur 16k · pitch 32k 로 전부 초반이다.

### 7-4. 평가 — 같은 자로 두 번

`run_examples.sh` 는 환경변수로 모델과 코퍼스를 받는다 (§5).

```sh
bash pansori/examples/run_examples.sh                          # whisper
python pansori/examples/measure.py --json pansori/examples/out/metrics.json

AC_EXP=pansori_ac_snap VAR_EXP=pansori_var_snap DATA=data/pansori_snap \
DS_DIR=pansori/examples/ds_snap OUT_DIR=pansori/examples/out_snap \
  bash pansori/examples/run_examples.sh
python pansori/examples/measure.py --out pansori/examples/out_snap \
  --json pansori/examples/out_snap/metrics.json
```

`SEED` 를 두 실행에서 같게 둘 것. reflow 는 확률적 샘플러라 시드만 바꿔도 f0 RMSE 가
±17% 움직인다 — **두 모델의 차이가 `seed_spread.py` 의 편차보다 작으면 차이가 없는 것이다.**

### 7-5. 결과 (2026-08-29 실행분)

**보정은 도움이 되지 않았다.** 홀드아웃 60쌍 × 3시드를 `2_resynth` 단에서 재면, 분리되는
지표는 하나뿐이고 그것이 **가사가 실제로 바뀐 발화에서만** 나온다 — 유무성 일치가
whisper 21승 6패 3무(p = 0.006), 대조군(음소열이 동일한 30쌍)에서는 10승 12패 8무
(p = 0.83). 전체 수치와 해석은 [`lyricbook/VERDICT.md`](lyricbook/VERDICT.md) §9.

```sh
python pansori/examples/ab_ladder.py --n 60 --seeds 3      # 두 팔을 한 번에, ~5분
```

`ab_ladder.py` 는 팔마다 모델을 **한 번만** 올려 60항목을 렌더한다 (`run_examples.sh` 는
렌더마다 프로세스를 새로 띄워 25초씩 쓴다). 절반은 두 코퍼스에서 음소열이 글자 하나
다르지 않은 **대조군**이다 — 같은 모델 두 개로 재므로 학습 런 사이의 우연한 차이는 두
군에 똑같이 나타나고, 처치군에서만 나오는 차이라야 가사 탓이다.

### 7-6. 읽을 때 주의

- **두 런의 검증 손실을 높낮이로 비교하면 안 된다.** 각자 자기 라벨에 대해 계산된 값이다.
  wandb 에 나란히 보이는 것은 수렴 양상과 과적합 시점을 보기 위한 것이고, 우열 판정은
  7-4 의 합성음 지표로만 한다.
- 두 변종에서 07 을 통과한 아이템 집합이 완전히 같지는 않다. 차이를 세어 명시할 것.
- 청취 검증은 여전히 없다. 이 실험은 "어느 라벨이 더 나은 합성음을 만드는가"만 답한다.

---

## 8. 음소·발음 계측

합성음의 발음이 부정확할 때, **그것을 잴 자가 이 저장소에 없었다.** `compare/` 의 지표
22개는 전부 피치 아니면 스펙트럼이고 (`compare/stats.py:40-60`), 가장 가까운 `voiced_agree`
조차 "이 프레임에서 소리를 내야 하는가"만 재므로 자음을 통째로 빼먹어도 떨어지지 않는다
(홀드아웃에서 0.95~0.97). §7 의 가사 A/B 를 포함해 지금까지의 모든 판정이 발음을 보지
못한 채 내려졌다. 이 절의 셋이 그 구멍을 메운다.

| 자 | 답하는 질문 | 무엇을 보나 | 비용 |
|---|---|---|---|
| **A** `variance_task.py` | 모델이 음소 길이를 정확히 예측하는가 | wandb, 매 검증 | 공짜 (자동) |
| **B** `eval/intelligibility.py` | 렌더된 소리가 실제로 알아들리는가 | 콘솔·wandb | 표본 40개 ~15분 |
| **C** `eval/label_report.py` | 학습 라벨 자체가 얼마나 망가졌나 | 콘솔·HTML | 몇 초 |

**셋 중 C 부터 볼 것.** 모델을 의심하기 전에 무엇을 보고 배웠는지 아는 편이 빠르다.

### 8-1. C — 라벨 진단

```sh
python pansori/eval/label_report.py                                    # data/pansori
python pansori/eval/label_report.py --corpus data/pansori data/pansori_snap \
       --html pansori/eval/out/label_report.html
```

모델도 GPU 도 필요 없다. `transcriptions.csv` 의 `ph_dur` 만 읽는다.

**현재 알려진 수치 (다음에 돌렸을 때 이것과 비교할 것):**

| | `data/pansori` | `data/pansori_snap` |
|---|---:|---:|
| 정확히 `0.030000` 초인 음소 | **22.3%** | 23.7% |
| ≤30 ms | 26.4% | 27.9% |
| ≤3프레임 (학습 지표와 같은 자) | 30.7% | 32.2% |
| 초성 ≤30 ms | 34.8% | 36.9% |
| 종성 ≤30 ms | 12.0% | 13.3% |
| 무음 시간 비율 | 21.7% | 24.0% |

`0.030000` 에 22.3% 가 쌓인 것은 분포가 아니라 **스파이크**다 — 다음으로 흔한 값(0.04)의
3.2배다. 0.03 은 MFA 의 최소 음소 길이이므로, 거기 있는 음소는 길이가 그렇게 측정된 것이
아니라 **정렬기가 음향적 근거를 못 찾고 최소값에 박아 둔 것**이다. 그리고 초성 파열음에
몰려 있다 (`pp` 48% · `tt` 46% · `b` 42% · `g` 42%). 30 ms 는 hop 512 에서 2.6프레임이라,
모델은 그 자음에 프레임 두세 개만 배정받고 그마저 자리가 틀렸을 수 있다.

`lyricbook/VERDICT.md` §7-4 가 이미 같은 것을 지목했다 — *"가사를 바꾸는 것보다 정렬
자체를 고치는 쪽의 여지가 훨씬 크다."*

### 8-2. A — 학습중 wandb 지표

`pansori/train.py` 로 variance 를 돌리면 자동으로 붙는다 (§4). config 의
`task_cls: pansori.variance_task.PansoriVarianceTask` 가 그 일을 한다 —
**upstream 은 그대로 두고 서브클래스만 갈아 끼운다.**

wandb 차트 (`metrics/` 네임스페이스):

| 키 | 읽는 법 |
|---|---|
| `dur_acc_onset` / `dur_acc_coda` / `dur_acc_vowel` / `dur_acc_glide` / `dur_acc_sp` | 부류별 길이 정확도 (±20%). **초성만 낮으면 자음이 뭉개지고 있는 것이다** |
| `dur_mae_onset_ms` / `dur_mae_vowel_ms` | 부류별 절대오차. 초성은 대부분 2~4프레임이라 정확도가 바닥에 붙어 안 움직일 때가 있는데, 이건 그 아래에서도 움직인다 |
| `floor_rate_pred` | 예측 길이 중 ≤3프레임 비율 |
| `floor_rate_gt` | **라벨** 중 ≤3프레임 비율. 모델과 무관한 상수 — 기준선이자 검산용 |
| `floor_rate_pred_onset` | 초성만 본 예측 하한선율 |
| `dur_acc_macro` | 음소 35종 정확도의 매크로 평균. upstream `ph_dur_acc`(마이크로)와 나란히 보면 "흔한 음소만 맞히고 있는가"가 보인다 |
| `phone_dur_acc` (그림) | 음소 하나하나의 정확도와 하한선율 막대. 부류별로 묶여 있다 |

**`floor_rate_pred` 와 `floor_rate_gt` 를 겹쳐 볼 것.** 예측이 라벨을 따라가면 모델이
라벨의 병을 그대로 배운 것이고, 크게 낮으면 라벨과 다르게 굴고 있다는 뜻이다. 어느 쪽이
옳은지는 이 지표가 답하지 않는다 — 그건 8-3 의 일이다.

**함정 둘.**

- **범위가 다르다.** upstream 의 `ph_dur_acc` / `rhythm_corr` 은 `num_valid_plots`(10)
  게이트 안에 있어 **앞 10발화**만 채점한다 (`variance_task.py:253,260`). 새 `dur_acc_*` 는
  `num_valid_metric_items: 0` 으로 **검증셋 전체**를 본다. 두 수치를 직접 빼지 말 것.
  기존 둘은 이전 런과의 비교선을 위해 일부러 그대로 뒀다.
- **단위가 프레임이다.** binarize 가 `ph_dur` 을 mel 프레임으로 양자화한 뒤에야 모델이
  보므로 여기서도 프레임으로 잰다. 하한선은 3프레임(34.8 ms)이라 8-1 의 **`≤30ms` 열이
  아니라 `≤3프레임` 열**과 맞춰 봐야 한다. 실측 검산: 검증셋 라벨의 ≤3프레임 비율은
  30.6899%, 학습이 기록한 `floor_rate_gt` 는 30.6974% — 13,379개 중 1개 차이(반올림 타이)다.

### 8-3. B — 발음 명료도 eval

사다리 각 단을 재서 **어디서 무너지는지** 가른다. env 가 셋이라 단계를 나눈다.

```sh
conda activate dsp     && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/eval/intelligibility.py --stage render --n 40 --seeds 3

conda activate pansori && export PYTHONPATH=.
python pansori/eval/intelligibility.py --stage asr    --n 40 --seeds 3

conda activate aligner && export PYTHONPATH=.          # mfa CLI 가 PATH 에 있어야 한다
python pansori/eval/intelligibility.py --stage mfa    --n 40 --seeds 3

conda activate dsp     && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/eval/intelligibility.py --stage score  --n 40 --seeds 3 \
       --wandb --wandb-group intelligibility
```

각 단계는 이미 만든 행을 건너뛰므로 중간에 끊고 다시 시작해도 된다. 렌더는
`compare/render_new.py` 에 그대로 넘긴다(렌더 로직을 두 벌 두지 않는다). 표본은
`compare/items.py` 의 층화표본이고 `assert_held_out()` 이 학습 데이터 누출을 막는다.

**함정 넷 — 이걸 모르면 표를 거꾸로 읽는다.**

1. **`1_orig` 은 whisper 쪽에서 천장이 아니라 널 대조다.** CER 의 참조 텍스트가
   `data/full/lyrics.jsonl` 인데 그것 자체가 Whisper 가 이 녹음에서 뽑은 전사다
   (`03_transcribe.py`). 원본을 다시 전사하면 자기 자신과 비교하는 셈이라 CER 이 0 에
   가깝게 나온다. **판소리를 잘 알아듣는다는 뜻이 아니다.** 0 근처가 아니면 Whisper 의
   디코딩이 재현되지 않는다는 뜻이고, 그러면 그 뒤 숫자는 전부 무효다.
   합성음의 실질적 천장은 **`0_vocoder`** (GT mel → 보코더) 다. MFA 쪽에서는 반대로
   `1_orig` 이 제대로 된 천장이다 — 정렬기는 오디오와 텍스트를 독립적으로 받는다.
2. **`spn` 은 단 비교에 쓰지 않는다.** 처음엔 "소리에 자음이 없으면 MFA 가 spn 을 낸다"고
   보고 넣었는데, 실측하니 **같은 발화의 다섯 단에서 값이 완전히 같았다.** spn 은 사전에
   없는 단어(OOV)에 대해 나오는 것이라 오디오가 아니라 **가사의 성질**이다. 표본의 OOV
   정도를 알려 주는 참고값으로만 한 번 인쇄한다. `phone_match` 도 같은 이유로 거의 상수다
   — 음소열은 우리가 준 가사가 정한다.
3. **MFA 쪽 실질 지표는 `floor_rate` 와 `boundary_mae_ms` 둘이다.** `floor_rate` 는 8-1 이
   라벨에 대는 것과 **같은 자를 합성음에 대는 것**이라, 두 표를 나란히 읽을 수 있다.
4. **Whisper 는 없는 소리를 채워 넣는다.** 그래서 두 backend 를 같이 본다. Whisper 가
   그럴듯한 문장을 뱉는데 MFA `floor_rate` 가 높으면, 그 문장은 상상이다.

**판독표 — 어느 단에서 떨어지면 무엇이 범인인가.**

| 증상 | 범인 | 다음에 볼 것 |
|---|---|---|
| `1_orig` 의 CER 이 0 근처가 아니다 | 계측 자체가 불안정 | 여기서 멈춘다 |
| `0_vocoder` 가 이미 나쁘다 | 보코더 · 추출기 | 그 아래 단은 읽을 필요 없다 |
| `2_resynth` 에서 이미 무너진다 | **라벨(MFA 정렬) 또는 음향모델** | 8-1 의 초성 하한선율 |
| `2_resynth` 는 멀쩡한데 `4_full` 에서 무너진다 | **duration 예측** | 8-2 의 `dur_acc_onset` |
| 전 단이 고르게 나쁘다 | 음소셋 · 사전 | `dictionaries/korean-coda.txt` |

**첫 관측 (n = 3, 시드 1 — 방향만 보는 예비 수치다).** 널 대조 0.000 통과. CER 은
`0_vocoder` 0.33 → `2_resynth` 1.00 → `4_full` 1.69, MFA 하한선율은 `1_orig` 30% →
`2_resynth` 40% → `4_full` 65%. 즉 **`2_resynth` 단계에서 이미 무너진다** — GT `ph_dur` 과
GT f0 를 주었는데도 그렇다. 판독표대로면 duration 예측이 아니라 라벨·음향모델 쪽이다.
표본이 3개뿐이므로 `--n 40 --seeds 3` 으로 다시 재고 나서 판정할 것.
