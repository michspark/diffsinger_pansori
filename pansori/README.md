# 판소리 fine-tune — openvpi/DiffSinger

판소리(한국 전통 서사창)로 DiffSinger 를 fine-tune 하는 전처리·학습·평가 파이프라인.
이 문서는 **실행법**만 담는다 — 왜 그렇게 하는지는 아래 문서들에 있다.

**upstream 파일은 한 줄도 고치지 않았다.** 우리 코드는 `pansori/`, `dictionaries/korean-coda.txt`,
`configs/pansori/`, `configs/exp/` 에만 있다. rebase 를 계속 감당하려면 그 상태를 유지할 것.

| 문서 | 내용 |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | 원본 오디오 → 합성음, 단계별 전체 기록 |
| [`docs/TRAINING.md`](docs/TRAINING.md) | 학습 한 스텝 안에서 벌어지는 일 |
| [`docs/EVAL.md`](docs/EVAL.md) | 음소·발음 계측 — 세 개의 자 |
| [`docs/GOTCHAS.md`](docs/GOTCHAS.md) | 설정·데이터의 함정 (막히면 여기부터) |
| [`docs/LYRICS_AB.md`](docs/LYRICS_AB.md) | 가사 A/B 실험 절차와 결과 |
| [`docs/PORTING.md`](docs/PORTING.md) | MoonInTheRiver → openvpi 이식 경위 |
| [`docs/NEXT.md`](docs/NEXT.md) | 남은 일 |
| [`docs/SEGMENTATION_VERDICT.md`](docs/SEGMENTATION_VERDICT.md) | 왜 6초인가 — 세그멘테이션을 바꾸기 전에 |
| [`docs/LYRICBOOK_VERDICT.md`](docs/LYRICBOOK_VERDICT.md) | 창본 보정 판정 |
| [`compare/README.md`](compare/README.md) | 구 모델 대비 정량 비교 |

---

## 1. 디렉토리 구조

```
pansori/
├── preprocess/          01~08 전처리. 단계 사이는 jsonl 로 이어져 개별 재실행 가능
│   └── pansori_lib.py   공유 상수 · IPA→Coda 규칙 · 음소 부류 · jsonl IO
├── alignment/           MFA 타당성 검토 + IPA→Coda 매핑표
│   ├── (VERDICT.md → docs/SEGMENTATION_VERDICT.md)
│   ├── ipa_to_coda.csv          korean_mfa IPA(108) → Coda(34)
│   └── check_dict_agreement.py  ★ 매핑을 고치기 전후로 반드시 돌릴 채점기
├── examples/            ★ 추론. 학습 후 여기서부터 돈다 (§5)
│   ├── run_examples.sh          사다리 4단 + 악보 합성 일괄 생성
│   ├── scores/*.txt             악보. 파일을 넣으면 자동으로 합성된다
│   ├── measure.py               f0 RMSE · mel L1 · 유무성 일치
│   ├── seed_spread.py           시드별 편차 — 수치를 믿기 전에 볼 것
│   ├── f0_report.py             시김새 평탄화 측정
│   ├── ab_ladder.py             두 모델을 홀드아웃 N쌍으로 비교 + 부호검정
│   ├── build_listening_test.py  블라인드 청취 페이지
│   ├── best_ckpt.py             검증 손실 최저 체크포인트 선택
│   └── build_report.py          설명 + 오디오를 한 페이지로 렌더링
├── eval/                ★ 음소·발음 계측 (§6)
│   ├── label_report.py          학습 라벨 진단 (ph_dur 하한선 스파이크)
│   ├── intelligibility.py       사다리 5단 × {whisper CER, MFA 재정렬}
│   └── compare_lyrics_alignment.py  두 가사의 MFA 정렬 페어드 비교
├── metrics/             부류별 음소 길이 지표 (torchmetrics)
├── compare/             구 모델(MoonInTheRiver) 대비 정량 비교 패키지
├── lyricbook/           창본 → corpus.jsonl (build_corpus.py 로 재생성)
├── experiments/         세그멘테이션 알고리즘 시뮬레이션 (01 의 근거)
├── server/run.sh        서버쪽 setup / 학습 tmux 러너
├── docs/                위 표 참조
├── train.py             scripts/train.py + wandb (upstream 무수정)
├── variance_task.py     VarianceTask 서브클래스 — 음소 길이 지표를 등록한다
├── tb_to_wandb.py       끝난 학습의 텐서보드 로그 → wandb
├── build_dictionary.py  한글 11,172음절 → Coda 34음소 사전
├── make_configs.py      화자 목록을 읽어 configs/pansori/ 렌더링
├── make_ds.py           라벨 또는 악보 → .ds (추론 입력)
└── compute_spec_stats.py  이 코퍼스의 mel 포락선 실측 → spec_min/spec_max
```

우리 config 는 저장소 루트의 `configs/` 아래 두 디렉토리에 있다:

```
configs/
├── pansori/          make_configs.py 가 생성 — 직접 고치지 않는다
│   ├── acoustic.yaml · variance.yaml    kind 공통층 (코퍼스를 모른다)
│   ├── data_pansori.yaml · data_snap.yaml   코퍼스층 (datasets + num_spk)
│   └── spec_stats.yaml                  compute_spec_stats.py 가 생성
└── exp/              ★ 실험 하나 = 파일 하나, 손으로 쓴다 (§4)
    └── ac_v1.yaml · var_v1.yaml · ac_snap.yaml · var_snap.yaml
```

데이터는 전부 gitignore 대상이라 저장소에 없다.

| 무엇 | 어디 |
|---|---|
| 원본 오디오/라벨/노트 | `~/Desktop/Pansori_Data/` (`Audio/`, `labels_2.json`, `rosvot_midi/`) |
| 전처리 중간 산출물 | `pansori/data/full/` |
| 학습용 코퍼스 (44.1 kHz) | `data/pansori/{화자}/` (8.7 GB) |
| binarize 결과 | `data/pansori/binary_{acoustic,variance}/` |

원본 경로는 `preprocess/pansori_lib.py` 의 `PANSORI_DATA` 하나에만 하드코딩돼 있다.

---

## 2. 환경

| env | Python | 쓰는 곳 |
|---|---|---|
| `pansori` | 3.10 | 전처리 01–03, 06–08 |
| `aligner` | 3.14 | MFA 3.4.1 + `korean_mfa` + mecab-ko → 04–05 |
| `dsp` | 3.10 | openvpi 본체 (binarize / train / infer / eval) |
| `sofa` | 3.10 | SOFA 정렬기 (선택 — §3.1) |
| `fbl` | 3.10 | FoxBreatheLabeler 들숨 라벨 (선택 — §3.1) |

`bash pansori/server/run.sh setup` 이 앞의 셋을 만들고 체크포인트도 받는다.
`sofa`/`fbl` 은 정렬기를 바꿀 때만 필요하다 (§3.1 에 설치 명령).

- **`dsp` 는 `PYTHONNOUSERSITE=1` 이 필수다** (env 의 `activate.d` 에 넣어 뒀다).
  `~/.local` 의 낡은 tensorflow 가 tensorboard import 를 죽인다.
- `base` 와 옛 `diffsinger`(3.8) env 는 못 쓴다 — numpy 2.5 충돌, Python 버전 미달.
- `requirements.txt` 에 `torchaudio` 가 빠져 있다 (upstream 누락). RMVPE 가 쓴다.

---

## 3. 전처리

각 단계는 `pansori/data/{set}/` 에서 동작하고 `--set` 으로 고른다.

| # | 스크립트 | in → out | env |
|---|---|---|---|
| 1 | `01_segment.py` | labels + Audio + midi → `segments.jsonl` + `wavs/` | base |
| 1b | `01b_recut_wavs.py` | segments → `wavs/` 복원 | pansori |
| 2 | `02_speaker_map.py` | segments → `spk_map.json` | base |
| 3 | `03_transcribe.py` | wavs → `lyrics.jsonl` (whisper-large-v3) | base |
| 3b | `03b_snap_lyrics.py` | lyrics + 창본 → `lyrics_snapped.jsonl` (선택) | pansori |
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` (MFA 또는 SOFA — §3.1) | aligner |
| 4b | `04b_sofa_data.py` | phones → SOFA 학습 데이터 (선택) | pansori |
| 5 | `05_notes.py` | rosvot_midi → `notes.jsonl` | aligner |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | pansori |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | pansori |
| 8 | `08_export.py` | 44.1 kHz 재커팅 + 화자별 분할 | pansori |

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_align  # 정렬 결과 재파싱 ~2분
conda activate pansori
python 06_merge.py --set full && python 07_validate.py --set full     # PASS 확인 필수
python 08_export.py --set full --out data/pansori                     # 오디오 재커팅 ~20분
```

**`01_segment.py` 를 다시 돌리면 안 된다.** 재현되지 않는다 — 다른 env 에서 돌리면
`librosa.effects.split` 이 경계를 3~11 ms 옮겨 이미 만든 `phones.jsonl` 의 음소가 전부
어긋난다. `wavs/` 가 필요하면 매니페스트에서 복원한다:

```sh
python pansori/preprocess/01b_recut_wavs.py --set full   # 31,529개, 새로 자르는 건 4,574개
```

**`07_validate.py` PASS 없이 binarize 하지 말 것.** binarizer 의 assert 실패는 스킵이
아니라 실행 중단이다. `ipa_to_coda.csv` 나 `convert_phone` 을 고쳤다면 그 전후로
`alignment/check_dict_agreement.py` 를 돌려 숫자로 판정한다 (1음절 현재 89.59%).

---

## 3.1 정렬기 — MFA ↔ SOFA

Stage 4 의 정렬기는 `preprocess/align.yaml` 의 **`aligner` 한 줄**이 정한다
(`--aligner mfa|sofa` 로 덮어쓸 수 있다). 어느 쪽이든 `aligned{suffix}/*.TextGrid` 를
남기므로 06 이후는 아무것도 달라지지 않는다.

| | MFA (기본) | SOFA |
|---|---|---|
| 설계 | 음성용 HMM-GMM | 노래 전용 |
| 음소 | IPA → `ipa_to_coda.csv` + `convert_phone` (일치율 89.59%) | `korean-coda.txt` 를 그대로 써서 **Coda 가 바로 나온다** |
| 파인튜닝 | 사실상 불가 | full/weak/no_label 로 직접 학습 (§3.2) |
| 들숨(AP) | 없음 | FoxBreatheLabeler 로 라벨 (`breath_ap.enabled`) |
| 환경 | `aligner` | `sofa` (`conda run` 으로 셸아웃) |

왜 바꾸는가: 정렬기 원본 TextGrid 기준 음소의 **34.03%** 가 0.03초 하한선에 박혀 있고
([`docs/LYRICBOOK_VERDICT.md`](docs/LYRICBOOK_VERDICT.md)), 06 이 흡수하는 "무음인데 노트가 있는"
구간이 코퍼스 시간의 **52.2%** 다. 둘 다 정렬기가 판소리의 지속음을 감당하지 못한다는
같은 이야기다. 창본 보정으로 가사 축을 손대 본 결과가 그 판정문의 결론이다 —
*"가사를 바꾸는 것보다 정렬 자체를 고치는 쪽의 여지가 훨씬 크다."*

### 설치 (한 번만)

두 env 모두 `dsp` 와 같은 두 함정을 밟는다. 만들자마자 막아 둘 것:

- **`PYTHONNOUSERSITE=1`** — 없으면 `~/.local` 의 낡은 패키지 때문에 pip 가 "이미 설치됨"
  으로 건너뛰어 env 가 조용히 반쪽으로 만들어진다 (SOFA 는 `PIL` 없음으로 죽는다).
- **`setuptools<81`** — `librosa<0.10` 이 `pkg_resources` 를 쓰는데 setuptools 81 이 그걸
  뺐다. `dsp` 가 lightning 때문에 같은 핀을 쓰고 있다.

```sh
git clone https://github.com/qiuqiao/SOFA ~/src/SOFA
conda create -y -n sofa python=3.10
echo 'export PYTHONNOUSERSITE=1' \
  > ~/miniconda3/envs/sofa/etc/conda/activate.d/no_user_site.sh   # 없으면 mkdir -p 먼저
conda activate sofa
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r ~/src/SOFA/requirements.txt "setuptools<81"
cp dictionaries/korean-coda.txt ~/src/SOFA/dictionary/
python infer.py --help          # 여기까지 떠야 한다

# 들숨(AP) 라벨까지 할 때만.
# FoxBreatheLabeler 에는 requirements.txt 가 없다 — 아래가 textgrid_add_ap.py 가
# 실제로 import 하는 것들이다 (trainCLS 가 lightning 과 pytorch_lightning 을 둘 다 쓴다).
git clone https://github.com/autumn-DL/FoxBreatheLabeler ~/src/FoxBreatheLabeler
conda create -y -n fbl python=3.10
echo 'export PYTHONNOUSERSITE=1' \
  > ~/miniconda3/envs/fbl/etc/conda/activate.d/no_user_site.sh
conda activate fbl
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install click "einops==0.6.1" "librosa<0.10.0" "matplotlib~=3.7.3" "numpy~=1.24.1" \
            "PyYAML~=6.0.1" textgrid tqdm "lightning>=2.0.0" pytorch-lightning \
            numba chardet "setuptools<81"
cd ~/src/FoxBreatheLabeler && python textgrid_add_ap.py --help    # 여기까지 떠야 한다
# 체크포인트(config.yaml + model_ckpt_steps_7000.ckpt):
#   https://github.com/autumn-DL/FoxBreatheLabeler/releases/latest
```

`align.yaml` 의 `sofa.root` / `sofa.ckpt` / `breath_ap.*` 를 받은 위치로 맞춘다.

## 3.2 SOFA 파인튜닝

한국어 SOFA 공개 체크포인트는 없다 (공개된 것은 만다린 `opencpop-extension`). 학습
데이터는 기존 MFA 정렬에서 부트스트랩하되, **그대로 쓰지 않는다** — 통째로 주면 SOFA 가
MFA 의 0.03초 바닥을 그대로 배운다. 발화별 floor_rate 로 갈라 나쁜 타이밍만 버린다.

```sh
conda activate pansori
python pansori/preprocess/01b_recut_wavs.py --set full          # data/full/wavs 복원
python pansori/preprocess/04b_sofa_data.py --set full --out ~/src/SOFA/data
#   floor_rate <= 0.15 → full_label (타이밍까지 신뢰)
#   초과              → weak_label (음소열만, 타이밍 버림)
#   정렬 없음          → no_label   (오디오만)
#   --max_floor_rate 로 조절. 문턱값별 발화 수를 같이 찍는다.
```

실측 (26,955발화): floor_rate 중앙값 **0.35**. `0.15` 에서 full 4,501 / weak 22,454 /
no 4,574 로 갈린다.

```sh
conda activate sofa && cd ~/src/SOFA
# configs/binarize_config.yaml 의 data_folder 를 위 --out 으로 맞춘다
python binarize.py

# 사전학습: releases 의 pretrained_multilingual_singing.zip (v1.0.1, 359MB)
# configs/train_config.yaml 의 model_name 을 korean_pansori 로
python train.py -p ckpt/pretrained_multilingual_singing.ckpt
tensorboard --logdir=ckpt/
```

**음소셋이 달라도 `-p` 를 쓴다.** `modules/task/forced_alignment.py:190-197` 의
`load_pretrained` 는 backbone 을 그대로 가져오고, vocab 크기가 다르면 **head 만 새로
초기화한다**. 즉 만다린 체크포인트에서 음향 backbone 을 물려받고 한국어 음소 분류기만
새로 배운다 — 처음부터 학습할 이유가 없다.

**추론에 zero-shot 은 불가능하다.** vocab 은 체크포인트 안에 박혀 있고
(`forced_alignment.py:309` 이 `self.vocab[ph]` 로 직접 찾는다), 만다린 vocab 에 Coda
음소가 없으므로 학습 전에 `--aligner sofa` 를 돌리면 `KeyError` 로 죽는다. 순서는
반드시 **학습 먼저**다.

### SOFA 로 코퍼스 전체 정렬 → A/B

`align.yaml` 에 `aligner: sofa` 와 학습된 `sofa.ckpt` 를 넣고:

```sh
conda activate aligner && cd pansori/preprocess
python 04_align.py --set full --aligner sofa --suffix _sofa

conda activate pansori && cd ../data && mkdir -p full_sofa && cd full_sofa
ln -s ../full/{segments.jsonl,notes.jsonl,spk_map.json,wavs} .
ln -s ../full/phones_sofa.jsonl phones.jsonl          # 다른 것은 이것뿐이다
cd ../../preprocess
python 06_merge.py --set full_sofa && python 07_validate.py --set full_sofa
python 08_export.py --set full_sofa --out data/pansori_sofa --wavs_from data/pansori

conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_sofa --name sofa
python pansori/eval/label_report.py --corpus data/pansori data/pansori_sofa \
       --html pansori/eval/out/label_report.html      # ★ 판정은 이 표로
```

들숨까지 하려면 `align.yaml` 의 `breath_ap.enabled: true` 로 두고 같은 절차를 돌린다.
`ph_seq` 에 `AP` 가 실리면 `make_configs.py` 가 `merged_phoneme_groups` 를 자동으로 뺀다.

`alignment/check_dict_agreement.py` 는 IPA→Coda **변환** 채점기이므로 SOFA 팔에는
적용되지 않는다 (변환 단계가 없다). `eval/intelligibility.py` 는 MFA 를 *측정 도구* 로
쓰므로 설정과 무관하게 MFA 고정이다 — 과거 수치와의 비교선을 지키기 위해서다.

---

## 4. 학습

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0

python pansori/build_dictionary.py         # dictionaries/korean-coda.txt
python pansori/make_configs.py             # configs/pansori/ 생성층

python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
python pansori/compute_spec_stats.py       # → configs/pansori/spec_stats.yaml

python pansori/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python pansori/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset
```

**spec 통계는 부트스트랩이 필요하다** — 기본값으로 한 번 binarize → 실측. `spec_min`/`spec_max`
는 학습 시점에만 쓰이고 `configs/pansori/acoustic.yaml` 이 `spec_stats.yaml` 을 base 로
얹고 있으므로, **재-binarize 도 config 재생성도 필요 없다.**

### 실험을 하나 더 만들려면

config 는 세 축으로 나뉘어 있고, upstream 의 `base_config` 체인이 다시 합친다:

```
configs/pansori/acoustic.yaml      kind  — 모델·배치·음소셋. 코퍼스를 모른다
configs/pansori/data_pansori.yaml  코퍼스 — datasets(화자 23명) + num_spk
configs/exp/ac_v1.yaml             실험  — 손으로 쓴다
```

앞의 둘은 `make_configs.py` 가 생성하므로 직접 고치지 않는다. 바꾸고 싶은 것은 **실험
파일에 적는다** — 그러면 `--hparams` 로 셸에만 남던 조건이 git 에 남는다:

```yaml
# configs/exp/ac_big.yaml
base_config:
  - configs/pansori/acoustic.yaml
  - configs/pansori/data_snap.yaml     # 데이터를 바꾸려면 이 줄

binary_data_dir: data/pansori_snap/binary_acoustic

max_updates: 50000                     # 학습 조건
num_ckpt_keep: 12
hidden_size: 512                       # 모델 구조
```

**뒤에 오는 것이 이긴다** (`utils/hparams.py:64-82`) — base 를 순서대로 덮고, 마지막에
실험 파일 자신의 키가 이긴다. 코퍼스를 새로 만들었으면 그 층만 생성하면 된다:

```sh
python pansori/make_configs.py --data data/pansori_snap --name snap
```

`pansori/train.py` 는 `scripts/train.py` + wandb 다. `--no-wandb` 를 주면 완전히 동일하게
동작한다. 끝난 학습의 텐서보드 로그는 `tb_to_wandb.py` 로 올린다 (`wandb sync
--sync-tensorboard` 는 지표 이름에 디렉토리를 붙여 같은 차트에 안 겹치므로 쓰지 말 것):

```sh
pip install wandb        # dsp env 에 없다. CLI 는 python -m wandb 로 부를 것
python pansori/tb_to_wandb.py --exp pansori_ac_v1 --run-id pansori_ac_v1_tb
```

variance 런은 부류별 음소 길이 지표(`dur_acc_onset` · `floor_rate_pred` · `phone_dur_acc`
그림)를 같이 낸다 — 읽는 법은 [`docs/EVAL.md`](docs/EVAL.md) §2.

---

## 5. 추론

```sh
bash pansori/examples/run_examples.sh          # 사다리 + 악보 합성 → examples/out/
python pansori/examples/measure.py             # 원본 대비 점수
python pansori/examples/seed_spread.py         # 시드 편차
python pansori/examples/build_report.py        # 설명 + 오디오 한 페이지
```

`run_examples.sh` 는 홀드아웃 발화마다 **사다리 4단**을 만든다. 아래로 갈수록 모델이 스스로
만들 것이 늘어나므로, 나쁠 때 어느 단계 탓인지 가를 수 있다.

| | 준 것 | 모델이 만든 것 |
|---|---|---|
| `1_orig` | — | — (원본 녹음) |
| `2_resynth` | `ph_dur` + `f0` | mel |
| `3_pitch` | `ph_dur` | `f0` + mel |
| `4_full` | 노트 + 가사 | `ph_dur` + `f0` + mel |

새 노래는 `examples/scores/` 에 파일 하나를 넣으면 된다:

```
_   rest  0.30      # '_' 는 쉼표 → SP
춘  A4    0.45
향  G4    0.40
-   F4    0.55      # '-' 는 앞 음절이 이어짐 (시김새·melisma)
```

**두 모델 다 reflow 샘플러다.** 시드만 바꿔도 f0 RMSE 가 ±17%, mel L1 이 ±8% 움직인다.
두 트랙의 차이가 `seed_spread.py` 의 편차보다 작으면 차이가 없는 것이다.

---

## 6. 평가

발음이 알아들리는가는 `compare/` 의 피치·스펙트럼 지표로 잴 수 없다. 세 개의 자가 그
구멍을 메운다 — 자세한 것은 [`docs/EVAL.md`](docs/EVAL.md).

```sh
python pansori/eval/label_report.py                  # C. 학습 라벨 진단, 몇 초
python pansori/eval/intelligibility.py --stage all   # B. 발음 명료도, 표본 40개 ~15분
                                                     # A. 학습중 지표는 wandb 에 자동
```

**셋 중 C 부터 볼 것.** 모델을 의심하기 전에 무엇을 보고 배웠는지 아는 편이 빠르다 —
이 코퍼스는 전체 음소의 22.3% 가 MFA 하한선 0.03초에 박혀 있다.
