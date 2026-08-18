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
│   └── ipa_to_coda.csv    korean_mfa IPA (108) → Coda (34)
├── experiments/           세그멘테이션 알고리즘 시뮬레이션 (01 의 근거)
├── eval/                  resynth_ladder.py / f0_metrics.py / midi_baseline.py
│                          ※ 아직 24 kHz 기준. 44.1 kHz + NsfHifiGAN 으로 개작 필요
├── docs/                  PORTING.md, VIBRATO_VERDICT.md, SIGIMSAE_PLAN.md
├── build_dictionary.py    한글 11,172음절 → Coda 34음소 사전 생성
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
| `base` / `pansori` | 3.13 / 3.10 | 전처리 01–03, 06–08 |
| `aligner` | 3.14 | MFA 3.4.1 + `korean_mfa` + mecab-ko → 04–05 |
| (신규 필요) | ≥ 3.10 | openvpi 본체. `lightning~=2.3.0` 등 `requirements.txt` |

openvpi 는 Python ≥ 3.10 / PyTorch ≥ 2.4 를 요구하므로 옛 `diffsinger`(3.8) env 는 쓸 수 없다.

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
| 6 | `06_merge.py` | phones + notes → `transcriptions.csv` | base |
| 7 | `07_validate.py` | transcriptions → PASS/FAIL | base |
| 8 | `08_export.py` | 44.1 kHz 재커팅 + 화자별 분할 | base |

정렬 결과(`aligned/`)가 남아 있으면 **MFA 재실행 없이** 4단계를 다시 파싱만 할 수 있다:

```sh
cd pansori/preprocess
conda activate aligner && python 04_align.py --set full --skip_mfa
python 06_merge.py --set full && python 07_validate.py --set full   # PASS 확인 필수
python 08_export.py --set full --out data/pansori
```

`07_validate.py` 는 openvpi binarizer 가 워커 안에서 assert 로 죽는 지점을 전부 앞당겨 검사한다.
**PASS 없이 binarize 하지 말 것** — 실패는 스킵이 아니라 실행 중단이다.

---

## 4. 학습

```sh
python pansori/build_dictionary.py          # dictionaries/korean-coda.txt
python pansori/make_configs.py              # configs/pansori_{acoustic,variance}.yaml

python scripts/binarize.py --config configs/pansori_acoustic.yaml
python pansori/compute_spec_stats.py        # spec_min/spec_max 실측 후 config 반영

python scripts/train.py --config configs/pansori_acoustic.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/pansori_variance.yaml --exp_name pansori_var_v1 --reset
```

선행 다운로드 (현재 `checkpoints/` 는 비어 있음):
`checkpoints/rmvpe/model.pt`, `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`.

---

## 5. 알아둘 것

- **화자는 `datasets:` 항목 단위로 지정된다.** 23명(명창 22 + pooled `unknown`)이 각각 자기
  디렉토리를 갖는다. `make_configs.py` 가 발화 수 내림차순으로 `spk_id` 를 매긴다.
- **홀드아웃 곡이 한 화자의 전부를 차지하면 자동 해제된다.** 정권진(226발화)이 그 경우라
  홀드아웃에서 빠졌다 — 학습 데이터 0인 화자는 임베딩이 아예 학습되지 않는다.
- **`f0_min: 55` / `f0_max: 750` 을 빠뜨리지 말 것.** 기본값 65 Hz 는 남성 통성을 잘라낸다.
- **가장 최근 체크포인트가 최선이 아니다.** 구 파이프라인에서는 56k step 부근에서 과적합이
  시작됐다. 아키텍처가 달라졌으니 수치는 이월되지 않지만 습관은 유지할 것.
