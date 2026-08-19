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
- **acoustic + variance 학습 기동** — `checkpoints/pansori_{ac,var}_v1`, 둘 다 scratch

## 지금 돌고 있는 것

```sh
tail -f logs/train_pansori_ac_v1.log
tail -f logs/train_pansori_var_v1.log
tensorboard --logdir checkpoints          # 둘 다 한 화면에 뜬다
```

| | acoustic | variance |
|---|---|---|
| 첫 validation (4k step) | `val_loss` 0.196 | `val_loss` 0.200 |
| 학습 loss 추이 | `mel_loss` 1.09 → 0.16 | `dur_loss` 88 → 0.08, `pitch_loss` 1.26 → 0.013 |
| VRAM | 6.1 GB | 2.9 GB |
| TensorBoard | 오디오 30 + mel 20 (보코더 확인됨) | dur/pitch 플롯 20 |

**GPU 한 장을 나눠 쓴다.** 합계 9.0 / 24.5 GB 라 메모리는 여유롭지만 SM 을 경합해
acoustic 이 7 → 5 it/s 로 느려졌다 (variance 는 13 it/s). 100k step 기준 variance 약 2시간,
acoustic 약 5.5시간이고, variance 가 끝나면 acoustic 속도는 되돌아온다.

**가장 최근 체크포인트가 최선이 아니다** — 구 파이프라인에서는 56k step 부근에서 과적합이
시작됐다. 아키텍처가 달라 수치는 이월되지 않지만 습관은 유지할 것.

## 해야 할 것

### A. 학습 지켜보기 (진행 중)
1. 과적합 시작점을 TensorBoard `validation/total_loss` 로 잡을 것. 학습 loss 는 계속
   내려가도 검증이 돌아서면 거기가 쓸 체크포인트다
2. VRAM 이 24 GB 중 9.0 GB 뿐이다. 급하면 `make_configs.py` 의 `max_batch_frames` 를
   올릴 수 있다 — 다만 lr 은 현재 배치 기준이므로 같이 손봐야 하고, upstream 기본값과의
   비교 가능성을 잃는다

### B. 추론
3. `scripts/infer.py acoustic <ds> --exp pansori_ac_v1` 로 새 가사+노트 합성.
   **구 파이프라인에 없던 경로** — 사전이 생겨서 가사→음소 추론이 가능해졌다.
   variance 를 앞에 붙이면 노트만으로 dur/f0 까지 생성된다

### C. 평가 (개작 필요)
4. `pansori/eval/resynth_ladder.py`, `f0_metrics.py` 가 아직 24 kHz/80 mel/구 추론 API 기준.
   44.1 kHz + NsfHifiGAN + `scripts/infer.py` 로 개작
5. `pansori/eval/midi_baseline.py` 를 새 variance 모델에 돌려 **f0 평탄화가 여전한지 재측정**.
   구 진단("결정론적 회귀 + L1 → 조건부 중앙값 → 평탄화")의 전제가 reflow 생성 모델에는
   그대로 적용되지 않는다. 여기 결과가 나오기 전엔 contour tokenizer 를 붙이지 말 것

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
- **정권진(226발화)은 홀드아웃에서 자동 해제됨** — 곡 하나가 화자 전부라 학습 데이터가 0이 됐음
