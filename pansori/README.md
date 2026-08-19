# 판소리 fine-tune — openvpi/DiffSinger

판소리(한국 전통 서사창)로 DiffSinger 를 fine-tune 하기 위한 전처리 파이프라인.
이식 경위와 설계 판단은 [`docs/PORTING.md`](docs/PORTING.md), 6초 제약의 근거는
[`alignment/VERDICT.md`](alignment/VERDICT.md) 에 있다.

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
│   ├── best_ckpt.py       검증 손실 최저 체크포인트 선택
│   └── build_report.py    설명 + 오디오를 한 페이지로 렌더링
├── experiments/           세그멘테이션 알고리즘 시뮬레이션 (01 의 근거)
├── eval/                  resynth_ladder.py / f0_metrics.py / midi_baseline.py
│                          ※ 24 kHz 기준 구 스크립트. examples/ 가 대체하므로 폐기 검토
├── docs/                  PORTING.md, VIBRATO_VERDICT.md, SIGIMSAE_PLAN.md
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
| 2 | `02_speaker_map.py` | segments → `spk_map.json` | base |
| 3 | `03_transcribe.py` | wavs → `lyrics.jsonl` (whisper-large-v3) | base |
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` (음소 + **words tier**) | aligner |
| 5 | `05_notes.py` | rosvot_midi → `notes.jsonl` | aligner |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | pansori |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | pansori |
| 8 | `08_export.py` | 44.1 kHz 재커팅 + 화자별 분할 | pansori |

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
