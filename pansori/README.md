# 판소리 fine-tune — openvpi/DiffSinger

판소리(한국 전통 서사창)로 DiffSinger 를 fine-tune 하는 전처리·학습·평가 파이프라인.
이 문서는 **실행법**만 담는다 — 왜 그렇게 하는지는 아래 문서들에 있다.

**upstream 파일은 한 줄도 고치지 않았다.** 우리 코드는 `pansori/`, `dictionaries/korean-coda.txt`,
`configs/pansori_*.yaml` 에만 있다. rebase 를 계속 감당하려면 그 상태를 유지할 것.

| 문서 | 내용 |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | 원본 오디오 → 합성음, 단계별 전체 기록 |
| [`docs/TRAINING.md`](docs/TRAINING.md) | 학습 한 스텝 안에서 벌어지는 일 |
| [`docs/EVAL.md`](docs/EVAL.md) | 음소·발음 계측 — 세 개의 자 |
| [`docs/GOTCHAS.md`](docs/GOTCHAS.md) | 설정·데이터의 함정 (막히면 여기부터) |
| [`docs/LYRICS_AB.md`](docs/LYRICS_AB.md) | 가사 A/B 실험 절차와 결과 |
| [`docs/PORTING.md`](docs/PORTING.md) | MoonInTheRiver → openvpi 이식 경위 |
| [`docs/NEXT.md`](docs/NEXT.md) | 남은 일 |
| [`alignment/VERDICT.md`](alignment/VERDICT.md) | 왜 6초인가 — 세그멘테이션을 바꾸기 전에 |
| [`lyricbook/VERDICT.md`](lyricbook/VERDICT.md) | 창본 보정 판정 |
| [`compare/README.md`](compare/README.md) | 구 모델 대비 정량 비교 |

---

## 1. 디렉토리 구조

```
pansori/
├── preprocess/          01~08 전처리. 단계 사이는 jsonl 로 이어져 개별 재실행 가능
│   └── pansori_lib.py   공유 상수 · IPA→Coda 규칙 · 음소 부류 · jsonl IO
├── alignment/           MFA 타당성 검토 + IPA→Coda 매핑표
│   ├── VERDICT.md               ★ 6초 제약의 근거
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
├── make_configs.py      화자 목록을 읽어 configs/pansori_*.yaml 렌더링
├── make_ds.py           라벨 또는 악보 → .ds (추론 입력)
└── compute_spec_stats.py  이 코퍼스의 mel 포락선 실측 → spec_min/spec_max
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

`bash pansori/server/run.sh setup` 이 셋을 만들고 체크포인트도 받는다.

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
| 4 | `04_align.py` | lyrics + wavs → `phones.jsonl` | aligner |
| 5 | `05_notes.py` | rosvot_midi → `notes.jsonl` | aligner |
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | pansori |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | pansori |
| 8 | `08_export.py` | 44.1 kHz 재커팅 + 화자별 분할 | pansori |

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_mfa    # 정렬 결과 재파싱 ~2분
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

## 4. 학습

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0

python pansori/build_dictionary.py         # dictionaries/korean-coda.txt
python pansori/make_configs.py             # configs/pansori_{acoustic,variance}.yaml

python scripts/binarize.py --config configs/pansori_acoustic.yaml
python scripts/binarize.py --config configs/pansori_variance.yaml
python pansori/compute_spec_stats.py       # → configs/pansori_spec_stats.yaml
python pansori/make_configs.py             # 위 파일을 얹어 재생성

python pansori/train.py --config configs/pansori_acoustic.yaml --exp_name pansori_ac_v1 --reset
python pansori/train.py --config configs/pansori_variance.yaml --exp_name pansori_var_v1 --reset
```

**spec 통계는 부트스트랩이 필요하다** — 기본값으로 한 번 binarize → 실측 → config 재생성.
`spec_min`/`spec_max` 는 학습 시점에만 쓰이므로 **재-binarize 는 필요 없다.**

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
