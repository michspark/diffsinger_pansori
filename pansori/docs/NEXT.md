# 남은 작업 — 다음 세션에 이 파일을 붙여넣으면 이어서 진행

브랜치 `feat/pansori`. 경위는 `pansori/docs/PORTING.md`, 실행법은 `pansori/README.md`.

## 끝난 것

포팅 세션 (`8212d64`~`f6ba635`):

- 원본 저장소(`~/Desktop/DiffSinger_moonInRiver`) 미커밋 작업물 53개 파일 커밋 (`341e927`)
- `pansori/` 전체 이식 + `dictionaries/korean-coda.txt`(11,172음절 → Coda 34음소) 생성
- `04_align.py` 가 MFA words tier 도 기록 → MFA 재실행 없이 `ph_num` 유도
- `06_merge.py` / `07_validate.py` openvpi CSV 스키마로 재작성
- `08_export.py` 로 44.1 kHz 모노 재커팅 + 화자 23명 분할
- `configs/pansori_{acoustic,variance}.yaml` 생성 (`make_configs.py`)

기동 세션:

- **env `dsp`** (Python 3.10 / torch 2.6.0+cu124 / lightning 2.3.3). `pip check` 통과
- **체크포인트 3종** 다운로드: `rmvpe`, `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`, `vr`(HN-SEP)
- **`check_dict_agreement.py` 신규** — 매핑 변경을 사전 대조 점수로 판정하는 하네스
- **라벨 결함 3갈래 수정** → 1음절 일치율 **87.92% → 89.59%**, 2음절 75.01% → 76.43%
- 라벨 재생성 (04 `--skip_mfa` → 06 → 07 PASS → 08). 26,955항목 중 **3,109개(11.5%)** 변경
- **binarize 완료** — acoustic 4.5 GB / variance 381 MB, train 27.8 h · valid 1.08 h, 23화자 전원 학습 데이터 보유
- **spec 통계 실측 → config 반영.** 상한이 실제로 기본값을 벗어났다 (아래 참조)
- **acoustic + variance 학습** — `checkpoints/pansori_{ac,var}_v1`, 둘 다 scratch
- **추론 경로 개통** — `pansori/make_ds.py` (라벨/악보 → `.ds`) + `examples/run_examples.sh`.
  가사→음소→variance→acoustic→wav 가 끝까지 돈다
- **f0 평탄화 질문에 답이 나왔다** — 체계적 평탄화는 재현되지 않는다 (아래 §D)
- **리포트 발행** — 파이프라인 전 구간 설명 + 합성 예시 15개 (`build_report.py`)

## 지금 돌고 있는 것

```sh
tail -f logs/train_pansori_ac_v1.log
tensorboard --logdir checkpoints
bash pansori/examples/run_examples.sh    # 예시 재생성 (최적 체크포인트 자동 선택)
```

**variance 는 step 32,000 에서 중단했다.** `validation/total_loss` 가 20k(0.08761)를 최저로
세 번 연속 올랐고(0.09049 → 0.09877 → 0.10423), 상승분이 전부 `dur_loss` 였다.
`num_ckpt_keep: 8` · 4k 간격이라 80k 까지 갔으면 20k 체크포인트가 삭제됐을 것이다.
최적 체크포인트는 `checkpoints/best/` 에 복사해 뒀다.

**acoustic 은 계속 돈다.** 검증 손실이 아직 추세라기보다 진동이다.

### 체크포인트는 헤드마다 따로 고른다

| 지표 | 최저 step | 비고 |
|---|---|---|
| acoustic `validation/total_loss` | 학습 중 | `best_ckpt.py pansori_ac_v1` |
| variance `validation/dur_loss` | **16,000** | 이후 계속 상승 |
| variance `validation/pitch_loss` | **32,000** | 중단 시점까지 하강 중 |
| variance `validation/total_loss` | 20,000 | dur 이 지배하는 절충값 |

`total_loss` 로 고른 20k 로 피치를 뽑으면 32k 보다 **f0 오차가 25% 크다**. 그래서
`run_examples.sh` 는 헤드별로 고르고 variance 를 두 번 돌린다 —
1차 `--predict dur`, 2차는 그 산출물에 `--predict pitch` (`ph_dur` 는 manual 로 읽힌다).

## 해야 할 것

### A. 학습 지켜보기 (진행 중)
1. 과적합 시작점을 TensorBoard `validation/total_loss` 로 잡을 것. 학습 loss 는 계속
   내려가도 검증이 돌아서면 거기가 쓸 체크포인트다
2. VRAM 이 24 GB 중 9.0 GB 뿐이다. 급하면 `make_configs.py` 의 `max_batch_frames` 를
   올릴 수 있다 — 다만 lr 은 현재 배치 기준이므로 같이 손봐야 하고, upstream 기본값과의
   비교 가능성을 잃는다

### B. 추론 — 개통됨
3. `bash pansori/examples/run_examples.sh` 가 사다리 4단(원본/재합성/피치예측/전체추론)과
   악보 합성을 만든다. 악보는 `pansori/examples/scores/*.txt` 에 `음절 노트 길이초` 형식으로
   적으면 되고 `-` 가 시김새, `_` 가 쉼표다. 새 악보를 넣으면 자동으로 3화자 합성된다.
   - **`--spk` 는 한글 화자명을 못 받는다** (`utils/infer_utils.py:55` 가 `[0-9A-Za-z_-]+`).
     `make_ds.py` 가 `.ds` 의 `spk_mix` 에 직접 써서 우회한다 — upstream 은 안 건드린다
   - **`note_slur` 가 필요하다.** ph_dur 없이 추론할 때 노트를 단어에 묶는 정보인데 라벨에
     없어서 실측 타이밍에서 DP 로 복원한다. 악보에는 `-` 로 직접 표기된다

### C. 평가
4. `pansori/eval/resynth_ladder.py`, `f0_metrics.py` 가 아직 24 kHz/80 mel/구 추론 API 기준이다.
   다만 **`pansori/examples/` 가 그 역할을 상당 부분 대체**한다 — 사다리(`run_examples.sh`),
   점수(`measure.py`), 평탄화(`f0_report.py --seeds N`), 샘플링 편차(`seed_spread.py`).
   구 스크립트는 개작보다 폐기가 맞을 수 있다
5. **f0 평탄화는 재현되지 않았다 — 이 항목은 닫는다.** 시드 5회 중앙값으로 잰 진폭비:
   재합성 0.86–0.98, f0 예측 0.72–0.85, dur·f0 예측 0.82–1.07. 예측 트랙 6개 중 4개는
   범위가 1.0 을 가로지른다. **감쇠의 2–14% 는 acoustic·보코더에서 이미 생기므로 피치
   예측기 탓이 아니다.** contour tokenizer 를 붙일 근거는 적어도 이 측정에는 없다.
   ※ acoustic 학습이 끝나면 다시 재볼 것

### D. 남은 라벨 이슈 (측정만 해둠)
`check_dict_agreement.py --syllables N --focus <기호>` 로 언제든 다시 잴 수 있다.
고치면 04(`--skip_mfa`) → 06 → 07 이 2분, 08 재커팅이 20분이다. **MFA 재실행은 불필요.**

6. **ㅈ/ㅉ/ㅊ + 활음** (96건+): 죠는 [tɕo] 라 MFA 는 `j o`, 사전은 `j y o`.
   한국어에서 죠/조는 동음이므로 이건 **사전 쪽 문제**다 — `ipa_to_coda.csv` 에 `y` 를 붙이면
   훨씬 흔한 조까지 `j y o` 가 된다. 고치려면 `build_dictionary.py` 를 손대야 한다
7. **요 → `y eo` vs 사전 `y o`** (156건): 모음 쪽 이슈. 아직 안 봤다
8. **ㄴ 이 `d` 로** (내/나/니/너, 합계 700건+): MFA 오인식이지 매핑 문제가 아니다.
   매핑으로는 못 고친다
9. **무음 흡수 임계**: 50 ms 만 겹쳐도 무음 전체를 흡수. 커버율 50% 미만인 흡수 구간이
    6.2%(3,772개)
10. 잔여 불일치의 다수는 **정상적인 연음**이다 (은 → `eu n`, 좋 → `j o`). 문맥 없는 음절
    사전으로는 재현할 수 없는 것이 맞다 — 일치율 100% 는 목표가 아니다

### E. 정리 / 결정
11. `~/Desktop/DiffSinger_moonInRiver` 를 origin 에 push 할지 (로컬 커밋만 되어 있음)
12. 정리 대상: 구 24 kHz `wavs/`(5.5 GB), 구 `data/binary/pansori*`(16 GB) — 검증 후
13. **들숨(AP) 라벨링**: 지금은 `merged_phoneme_groups: [[SP, AP]]` 로 무음과 합쳐 뒀다.
    판소리는 들숨이 크게 들리므로 따로 라벨할 값어치가 있을 수 있다. 하려면 06 에
    에너지 기반 판정을 넣고 재-binarize

## 신경 쓸 것

- **`git merge` / `git remote add` 금지.** 두 저장소는 공통 조상이 없다
- **구 checkpoint 3개는 재사용 불가.** 24 kHz·80 mel·FastSpeech2 → 텐서가 하나도 안 맞는다
- **`f0_min: 55` / `f0_max: 750`** 을 빠뜨리면 남성 통성 저음이 통째로 무성 처리된다
- **`spec_max` 는 실측값을 써야 한다.** openvpi 기본값은 `[0]` 인데 이 코퍼스 실측 상한은
  **+2.838** 이다. 그대로 두면 diffusion 타깃이 정규화 후 [-1, 1] 을 벗어난다.
  하한은 반대로 안전하다 — `nvSTFT` 가 `log(clamp(x, 1e-5))` 로 압축해 -11.513 아래가 없다
- **`PYTHONNOUSERSITE=1` 없이는 env 가 깨진다.** `~/.local/lib/python3.10/site-packages` 의
  낡은 tensorflow 가 tensorboard import 를 protobuf 에서 죽이고, 거기 numpy 때문에 pip 이
  요구사항을 "이미 설치됨"으로 오판한다. `dsp` env 의 `activate.d` 에 넣어 뒀다
- **전처리는 `base` env 로 못 돌린다** (numba vs numpy 2.5 충돌). `pansori` env(3.10)를 쓸 것
- **한 번 뽑은 결과로 모델을 판단하지 말 것.** 두 모델 다 reflow 샘플러라 시드만 바꿔도
  f0 RMSE 가 ±17%, mel L1 이 ±8% 움직인다 (`seed_spread.py`). 트랙끼리 비교하려면
  시드를 고정하거나 여러 번 뽑아 중앙값을 쓸 것
- **최근 체크포인트가 최선이 아니다 — 이제 도구가 지킨다.** `best_ckpt.py` 가 검증 손실 최저
  step 을 고르고, `num_ckpt_keep` 으로 진짜 최저가 이미 삭제됐으면 그 사실을 알린다
- **정권진(226발화)은 홀드아웃에서 자동 해제됨** — 곡 하나가 화자 전부라 학습 데이터가 0이 됐음
