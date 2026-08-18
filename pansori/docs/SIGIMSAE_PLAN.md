# 시김새를 되살리는 계획 — pitch-contour-tokenizer 활용

`pansori_aux`(cascade stage 1)의 f0가 판소리처럼 들리지 않는 이유는 측정으로 확인됐다.
이 문서는 `~/Desktop/pitch-contour-tokenizer`(ISMIR 2026, VQ-VAE 판소리 contour 토크나이저)를
어떻게 붙일지에 대한 계획이다.

측정 근거는 `pansori/eval/midi_baseline.py`, 재현 명령은 §6.

---

## 1. 진단 — 무엇이 문제인지는 이미 안다

`midi_baseline.py`, test 24발화, CREPE 기준:

| | 노트편향 RMS | 노트내 변동 | (노트 안쪽만) | CREPE와 상관 |
|---|---|---|---|---|
| CREPE (정답) | 131.2 | **161.4** | **138.0** | 1.000 |
| ROSVOT 노트 계단 | 0.0 | 0.0 | 0.0 | — |
| `pansori_aux` | 70.7 | **31.8** | **17.4** | **0.145** |

모델은 노트 안에서 정답의 **12~20%**만 움직이고, 그 움직임조차 정답과 거의 무관하다.
음정 정확도(동일 프레임 194.3 vs 계단 203.2)는 계단보다 겨우 8.9 cents 낫다.
**모델은 노트를 평평하게 부른다.**

원인 가설: `add_f0_loss`는 log₂ 공간의 **L1 회귀**다. 시김새는 시점을 정확히 예측할 수 없으므로,
결정론적 출력 + L1의 최적해는 **조건부 중앙값 = 평평한 선**이다. 파라미터화를 바꿔도
결정론적 회귀 + L1이면 같은 평탄화가 재발한다.

---

## 2. 토크나이저가 정확히 무엇인가

`src/models/model_zoo.py`, `src/configs/model/foffset_conv1d_vqvae.yaml` 기준.

- 판소리·단가 **1,914곡**의 CREPE contour로 학습 (`splits/inhouse_crepe_split.csv`)
- **1 토큰 = 1.28초** (128프레임 @ 100 Hz)의 **median을 뺀** contour 모양
- 코드북 **256개** = 판소리 국소 음높이 패턴 사전, latent 32차원
- `OffsetMSELoss`: tempo(50/54/60) × zoom(0.85/1.0/1.15) × y-offset(±0.1) 후보 중
  **최소 오차만 역전파** → 토큰이 타이밍·음역 변화에 불변
- 논문은 이 코드북으로 시김새 6종(`southern_flick`, `southern_vibrato`, `western_vibrato`,
  `soft_vibrato`, `upward_accent`, `descending_slide`)을 분류한다
  → **코드북이 시김새를 구분한다는 근거가 이미 있다**

---

## 3. 붙이기 좋은 이유 두 가지 (확인함)

**단위가 이미 같다.** 토크나이저 입력은 `norm_midi = midi/12 = log₂(f0) − 3.0314`
(`dataset_zoo.py:45-64`, `norm_midi=False`라 tonic=0). DiffSinger의 f0 타깃은 이미 `log₂(f0)`이고
(`pitch_norm: log`), 게다가 토크나이저는 window마다 median을 뺀다 —
**상수항이 완전히 소거된다.** 변환은 리샘플링 하나뿐이다.

**프레임 비율이 정수다.** DiffSinger는 hop 128 @ 24 kHz = **187.5 Hz**, 토크나이저는 **100 Hz**.
비율 15:8이므로

> **240 DiffSinger 프레임 = 128 토크나이저 프레임 = 1.28초**

`F.interpolate(size=128, mode='linear')` 한 줄로 정확하고 미분 가능하게 변환된다.

---

## 4. 붙이기 어려운 이유 세 가지 (정직하게)

**(a) 토큰이 너무 거칠다.** 1토큰 = 1.28초 = 240프레임. 6초 발화가 4.7토큰이다.
따라서 **"f0 회귀를 토큰 분류로 교체"라는 순진한 안은 성립하지 않는다** — 음표 단위 제어를 잃는다.
토크나이저는 합성 표현이 아니라 분석 도구로 설계됐다.

**(b) 파이썬 환경이 다르다.** 토크나이저는 3.10–3.13, `diffsinger` 환경은 **3.8**이다.
다만 확인한 바:

| | diffsinger 환경 |
|---|---|
| torch | **2.0.1** (충분히 최신) |
| `nn.Dropout1d` | 있음 (게다가 `cnn_dropout: 0`이라 생성조차 안 됨) |
| `torch.nanmedian` | 있음 |
| torchaudio | **없음** — 단 `process_offset`(디코더)에서만 쓰인다 |

→ **인코더 경로는 `Conv1d / ReLU / MaxPool1d / median`뿐이라 3.8에서 그대로 돈다.**
`ConvEncoder` + `calc_conv_params` + median 로직만 포팅하면 된다 (~80줄).

**(c) 학습된 표현이 "노트 대비 잔차"가 아니다.** 1.28초 window의 median은 대략 그 구간의 평균
음높이다. 여러 음이 든 window의 토큰은 **음 전이까지** 담는다. 노트 계단 위에 그대로 더하면
이중 계산이 된다. Stage C가 이걸 정면으로 다룬다.

---

## 5. 단계별 계획

### Stage A — 심판으로 쓴다 (반나절, 위험 0)

목적 둘: ① 평탄화를 시김새 어휘로 정량화 ② **Stage B가 통할지 판단하는 게이트**.

준비:
```sh
cd ~/Desktop/pitch-contour-tokenizer
curl -LO https://github.com/SeongUkJu/pitch-contour-tokenizer/releases/download/v1.0.0/checkpoints.tar.gz
tar xzf checkpoints.tar.gz        # → weights/all_transformation_vqvae/ 등 4종, 0.5 MB
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
(2 GB 데이터 자산은 **필요 없다** — 우리 CREPE contour를 쓴다.)

작업: `pansori/eval/contour_tokens.py` 신규.
`midi_baseline.py`와 동일한 24발화에 대해 CREPE f0 / 모델 f0 / 노트 계단을
187.5→100 Hz 리샘플 후 128프레임 window로 토큰화하고 비교한다.

| 지표 | 의미 |
|---|---|
| 토큰 일치율 | GT 토큰 == 모델 토큰 |
| 토큰 분포 KLD | `D(모델 ‖ GT)` — 모델이 소수 토큰에 몰리면 큼 |
| 사용 토큰 수 / perplexity | GT는 몇 종, 모델은 몇 종 |
| latent 거리 | `‖z_pred − z_gt‖` |

**게이트 B0 — 반드시 먼저 확인할 것:**
> 같은 구간의 **CREPE contour**와 **계단(완전 평평) contour**를 토큰화했을 때 **다른 토큰이 나오는가?**

- 다르다 → 코드북이 "평평함 vs 시김새"에 민감하다 → Stage B의 손실이 실제 gradient를 준다
- 같다 → Stage B는 무의미하다 → **Stage C로 직행**

---

### ✅ Stage A 결과 (완료)

`pansori/eval/contour_tokens.py`, test 150발화 → 1.28초 window 318개
(결측으로 1,134개 버림 — 뒤의 한계 참조). 체크포인트 `all_transformation_vqvae`.

**게이트 B0 = 84.9% 통과.** 코드북은 시김새와 평평함을 확실히 구분한다. **Stage B 진행 가능.**

| 토큰 일치 | |
|---|---|
| `token(CREPE) ≠ token(계단)` — **게이트 B0** | **84.9 %** |
| `token(모델) == token(계단)` | **54.4 %** |
| `token(모델) == token(CREPE)` | **17.3 %** |

모델은 정답보다 계단의 토큰을 **3.1배** 자주 낸다. `within_std`가 보여준 평탄화가
시김새 어휘 수준에서 그대로 재확인됐다.

| 어휘 | 사용 토큰 | perplexity | CREPE와 KLD |
|---|---|---|---|
| CREPE (정답) | 112 | 78.1 | — |
| ROSVOT 노트 계단 | 73 | 21.3 | 1.991 |
| 모델 f0 | 71 | 29.7 | 1.743 |

| latent 거리 (Stage B 손실의 직접 대리지표) | |
|---|---|
| 무작위 CREPE 쌍 (기준선) | 0.454 |
| `‖z(계단) − z(CREPE)‖` | 0.248 (기준선의 55%) |
| `‖z(모델) − z(CREPE)‖` | **0.247 (기준선의 54%)** |

**이게 이번 측정의 핵심이다.** contour 모양 공간에서 모델의 f0는 **평평한 MIDI 계단보다
정답에 조금도 더 가깝지 않다** (0.247 vs 0.248). 음정에서 얻은 9.2 cents 우위는 모양에서
아무것도 사지 못했다. 그리고 이 값이 바로 Stage B가 낮출 대상이므로 **여지가 최대**다
(0.247 → 0, 무작위 기준선 0.454).

둘 다 무작위(0.454)보다는 가깝다 — 계단도 모델도 CREPE와 **음 골격**은 공유하고,
**시김새만** 공유하지 않는다는 뜻이다. 일관된 해석이다.

산출물: `pansori/review/contour_tokens.html` (코드북 상위 토큰 모양 포함),
`infer_out/contours_aux150.npz`

**한계 두 가지**
- window 1,452개 중 **1,134개(78%)를 버렸다.** repo 필터가 연속 3프레임(30 ms) 이상 무성인
  window를 버리는데, 1.28초 판소리 악구에는 자음·숨이 거의 항상 들어간다. 남은 22%는
  긴 지속모음 구간에 치우쳐 있다 — 시김새가 사는 곳이라 편향 방향은 나쁘지 않지만,
  **"발성이 끊기지 않는 구간"의 통계**라는 점은 명시해 둔다.
- window를 hop 32로 겹쳐 뽑았다(`--hop`). 표본이 독립이 아니므로 비율의 신뢰구간은
  실제보다 좁게 보인다. 결론의 크기(84.9% vs 17.3%)가 워낙 커서 방향은 뒤집히지 않는다.

### Stage B — contour perceptual loss (본 처방, 1~2일)

`pansori_aux` 학습에 손실 항 하나만 추가한다. 모델 구조는 그대로.

```
log₂f0 ──resample 240→128── window ──frozen encoder── z (32-dim)
loss_contour = ‖ z(pred) − z(gt) ‖²
total_loss += λ_contour · loss_contour
```

**왜 이게 평탄화를 깨는가** — 이게 이 계획의 핵심 논거다:

1. 인코더는 window마다 median을 빼므로 **절대 음정을 보지 않는다.** 순수하게 모양만 본다.
2. `OffsetMSELoss`로 학습돼서 **타이밍이 조금 어긋나도 같은 latent**가 나온다.
   L1이 평평해지는 이유가 "시점을 못 맞추니 평균내는 게 낫다"인데,
   **이 latent는 시점에 관대하므로 "움직여라"는 신호가 살아남는다.**
3. 평평한 contour와 시김새 contour의 latent가 멀다(게이트 B0에서 검증).

구현 요점:

- `usr/contour_tokenizer.py` — `modules.ConvEncoder` + `model_utils.calc_conv_params` +
  `Conv1DAE.subtract_median_from_x` 포팅. state_dict에서 `encoder.*`만 로드,
  `requires_grad=False`, `eval()`. 체크포인트는 torch 2.8 저장본이지만 zipfile 직렬화라
  torch 2.0.1에서 읽힌다 (첫 작업으로 **로드 테스트부터** 할 것).
- 리샘플: 240프레임 단위로 잘라 `F.interpolate(size=128, mode='linear', align_corners=False)`
- **unvoiced 처리**: 원 데이터셋은 `confidence < 0.5`를 보간으로 메우고
  (`dataset_zoo.py:183-204`), 연속 결측이 `num_interpolate`(2)를 넘는 window는 버린다
  (`filter_receptive_field`). 같은 규칙을 쓴다 — uv 프레임은 `norm_interp_f0`가 이미 보간해 두었고,
  결측이 긴 window는 손실에서 제외.
- λ 스케일: z는 32차원, 입력은 옥타브 단위. warmup 몇 스텝에서 `loss_f0`와 크기를 재서 맞춘다.
  시작 후보 **λ = 0.1 ~ 1.0**.
- 성공 지표는 **이미 있다**: `midi_baseline.py`의 `within_std`, `within_corr`.
  - 목표: `within_std` 17.4 → GT 138의 절반 이상, `within_corr` 0.145 → 0.4+
  - 감시: `val/f0_cents_rmse`가 크게 나빠지면 λ가 과하다

### 🔄 Stage B 진행 상황

**끝난 것**

- 체크포인트가 `diffsinger` env(py3.8 / torch 2.0.1)에서 **로드된다.** 51텐서, 인코더는
  Conv1d(k=3) 6층 + maxpool `[4,2,2,2,2,2]`(곱=128), `quantizer.embedding` 256×32.
- `usr/contour_tokenizer.py` — 인코더 + 코드북 포팅. 업스트림 키 이름 그대로 로드된다.
- `pansori/eval/tokenizer_parity.py` — **bit-exact 검증 통과.** 실제 contour 366 window에서
  latent 최대 오차 `0.000e+00`, 토큰 일치 100%, 다중 window 경로도 일치.
  포팅을 건드리면 이 테스트를 다시 돌릴 것.
- **gradient 확인**: pitch_predictor 23개 파라미터 전부에 grad가 흐르고, 인코더에도
  `predictor_grad=0.1` 경유로 흐른다. 토크나이저는 frozen.
  `pitch_predictor.linear.bias`의 grad만 ~0(9.3e-10)인데 이건 **설계대로**다 —
  bias는 contour를 통째로 평행이동시키고 median 차감이 그걸 지운다. 절대음정 불변성의 증거.
- **λ 결정**: `pansori_aux`에서 f0 L1 = 0.1464(log2, = 176 cents), contour MSE = 0.0023.
  → λ=20이면 contour 항이 f0 항의 약 0.3배. (검증셋 기준으로는 0.43배로 나왔다.)
  참고로 contour MSE 0.0023은 Stage A의 `‖z‖=0.247`에서 예측되는 0.0019와 맞는다 —
  독립 구현 둘이 교차검증된 셈이다.

**실험**: `usr/configs/midi/cascade/pansori/aux_contour.yaml`, exp `pansori_aux_c1`.
`pansori_aux`의 best(13k)에서 `load_ckpt`로 이어받아 λ=20, hop=32로 20k step.
처음부터 다시 학습하지 않는 이유는, 묻는 것이 "**이미 음을 맞게 부르는 predictor에
시김새를 넣을 수 있는가**"이고 그 답이 수천 step 안에 나오기 때문이다.

**주의 — window 유지율**: 배치 패딩까지 무성으로 잡히면서 probe에서 window의 10.3%만
살아남았다(hop=64). hop=32로 낮춰 보완했지만, 실제 학습은 `sort_by_len` 덕에 패딩이
적어 이보다 낫다. `contour` 손실이 0 근처로 붙으면 window가 전멸한 것이므로 먼저 의심할 것.

**아직 없는 것 — 대조군.** λ=0으로 같은 step 수를 이어 학습한 run이 없다. 다만 λ=0의
거동은 이미 안다(`pansori_aux`가 10k 이후 `val/f0` 평평, `uv`/`pdur` 악화, within 17.4 cents).
결과가 애매하면 대조군을 돌려야 한다.

### ❌ Stage B 결과 — 전제가 틀렸다. 중단.

`pansori_aux_c1`, λ=20, hop=32, `pansori_aux` best에서 이어받아 4,000 step.

**검증 손실이 움직이지 않는다.**

| step | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|
| contour | 0.0354 | 0.0365 | 0.0351 | 0.0350 |
| f0 | 0.1536 | 0.1627 | 0.1549 | 0.1568 |

시김새도 거의 그대로다 (150발화):

| | `pansori_aux` | `pansori_aux_c1` |
|---|---|---|
| 노트내 변동 | 33.6 | 36.7 |
| (노트 안쪽만) | 18.9 | 20.6 |
| CREPE와 상관 | 0.129 | 0.126 |

**λ 부족이 아니다.** pitch_predictor에서 contour 손실의 grad norm은 0.3025,
f0 L1은 0.2575 — **contour 쪽이 1.175배 세다.** window도 배치당 48.2개로 충분하다.
gradient가 f0보다 센데 손실이 안 내려간다 = **더 내려갈 수 없는 지점**이라는 뜻이다.

**왜인지 — latent의 위상 민감도를 재보면 나온다.** (아래는 gap 필터·window 내 보간을
적용하지 않은 느슨한 전처리라 학습 손실값과 절대 비교는 안 되고, 표 안에서만 비교된다.)

| contour | latent MSE |
|---|---|
| 계단 (완전 평평) | 0.01727 |
| 모델 c1 | 0.01428 |
| **CREPE를 30 ms 밀기** | **0.00123** |
| **CREPE를 100 ms 밀기** | **0.00686** |
| **CREPE를 250 ms 밀기** | **0.01499** |
| 완전 0 contour | 0.00991 |

**똑같은 시김새를 250 ms 어긋나게 내면 평평한 계단과 다를 바 없다** (0.01499 vs 0.01727).
100 ms만 어긋나도 계단 거리의 40%를 잃는다.

즉 §Stage B가 근거로 삼은 "`OffsetMSELoss` 덕에 타이밍에 관대한 latent"는
**문제가 되는 스케일에서 성립하지 않는다.** 그 후보들은 tempo ±10%와 디코더 window 내
소폭 이동이지 임의 위상이 아니다. 따라서 L1을 평평하게 만든 논리 — 시점을 못 맞히니
평균이 최적 — 가 latent MSE에도 그대로 적용된다. **MSE는 어느 공간에서 재든 평균을 낸다.**

부수적으로: 완전히 평평한 0 contour(0.00991)가 계단(0.01727)보다 정답에 가깝다.
이 latent가 가장 크게 벌하는 것은 평평함이 아니라 **계단의 음 도약**이다.

**결론: Stage B는 구조적으로 막혔다. 손실 공간을 바꾸는 것으로는 안 되고,
출력을 평균낼 수 없게 만들어야 한다.** → §5.5 또는 Stage C.

산출물은 남겨둔다: `usr/contour_tokenizer.py`(bit-exact 검증됨)와
`pansori/eval/contour_tokens.py`는 Stage C에서 그대로 쓰인다.
`checkpoints/pansori_aux_c1`은 삭제해도 무방하다.

리스크와 완화 (사후에 보면 세 번째가 실제로 일어났다 — 다만 "아무 시김새나 내도 통과"가
아니라 그 반대로, 위상이 틀리면 안 내느니만 못해서 안 내는 쪽이 최적이 됐다):

| 리스크 | 완화 |
|---|---|
| median의 gradient가 선택된 원소로만 흐름 | median을 `detach()`해 상수로 취급하고 빼기만 |
| maxpool 6층으로 gradient 희석 | λ 상향, 또는 중간층 feature도 매칭(perceptual loss 관행) |
| **latent가 tempo/zoom에 불변이라 "아무 시김새나" 내도 통과** | 예상된 결과다. `within_std`는 오르는데 `within_corr`가 안 오르면 **부분 성공**으로 읽는다 — 판소리처럼 들리는 게 목적이면 시점이 정확할 필요는 없다. 다만 미리 인정하고 지표를 그렇게 해석할 것 |

### Stage C — 잔차 토크나이저 재학습 (구조적 해법, 1주+)

Stage B가 부족할 때. §4(c)를 정면으로 푼다.

1. **잔차 corpus 생성** — `CREPE − 노트 계단`(cents)을 pseudo-frequency로:
   `f = 440 · 2^(resid_cents/1200)`. CREPE는 `pansori/data/full/f0/*.npy`에 있고,
   계단은 `midi_baseline.render_staircase`가 만든다. `frequency,confidence` 두 컬럼 CSV로 쓰면
   토크나이저 파이프라인이 그대로 받는다.
2. **더 짧은 window로 재학습** — `dataset.params.window` 128 → **32~48** (0.32~0.48초 =
   시김새 한 개 단위). 나머지는 `foffset_vqvae` 그대로.
   ```sh
   python src/train.py --config-name foffset_vqvae \
       data.data_pth=/path/to/residual_csvs dataset.params.window=48
   ```
3. **pitch predictor를 토큰 분류기로 교체** — window마다 256-way cross-entropy,
   디코딩된 잔차를 노트 계단에 더한다.
   **cross-entropy는 평균내지 않는다. 이것이 L1 평탄화의 구조적 해결이다.**

여기까지 가면 음표 제어는 계단이, 시김새는 토큰이 담당한다.

---

## 5.5. 병행 권고 — 토크나이저 없이도 되는 싼 실험 (반나절)

평탄화의 근본 원인은 **결정론적 회귀 + L1**이지 파라미터화가 아니다.
이건 토크나이저 없이도 검증할 수 있고, **Stage C의 전제(분류 기반 f0)를 미리 확인**해 준다.

`utils/pitch_utils.py`에 이미 **256-bin 이산 f0 표현**이 있다 —
`f0_to_coarse`가 `pitch_embed`를 위해 매 프레임 1..255를 만든다 (해상도 ≈ 21 cents/bin).

- `PitchPredictor`의 `odim`을 2 → 257 (256 bin + uv)로
- `add_f0_loss`의 L1을 `f0_to_coarse(f0)` 타깃에 대한 cross-entropy로
- 추론은 argmax 또는 샘플링

싸고, 토크나이저 작업과 직교하며, "L1이 범인인가"를 하루 안에 답한다.

### 🔄 §5.5 구현 (진행 중)

`usr/pansori_fs2.py` + `usr/configs/midi/cascade/pansori/aux_cls.yaml`, exp `pansori_aux_cls1`.
**upstream 파일은 하나도 건드리지 않았다** — 전부 `usr/` 안의 subclass/override다
(`build_tts_model`, `add_pitch_loss`, `plot_pitch`, `load_ckpt`, `add_pitch`).

**bin 격자는 새로 만들었다.** 원래 계획은 `utils.pitch_utils.f0_to_coarse`의 256칸을
재사용하는 것이었는데, 재보니 그건 mel 척도라 음역마다 해상도가 다르다:

| f0 | `f0_to_coarse` 해상도 |
|---|---|
| 61.7 Hz (남성 통성, MIDI 35) | **72.7 cents/bin** |
| 220 Hz | 24.2 |
| 740 Hz | 10.4 |

135 cents짜리 시김새가 저음에서 2칸에 뭉개지고, 55–750 Hz가 256칸 중 3–192번만 쓴다.
대신 **cents 등간격** 격자를 쓴다: 55–750 Hz = 3.77옥타브 = 4,527 cents → 256칸에
**17.7 cents/bin, 전 음역 균일**. 측정된 양자화 왕복 오차는 평균 4.4 / 최대 8.8 cents로
이론 상한과 일치하고, 되살리려는 135 cents에 비하면 무시할 수준이다.
`f0_to_coarse`는 `pitch_embed`용으로 그대로 둔다.

**스모크 테스트에서 드러난 것 하나** — 랜덤 초기화 상태에서 디코드 표준편차:

| 디코드 | std |
|---|---|
| `sample` | 1.09 |
| `argmax` | 0.60 |
| `expect` (확률 가중 평균) | **0.004** |

`expect`가 거의 상수로 붕괴한다. 평균화가 곧 평탄화라는 것이 눈앞에서 재현된 셈이고,
**샘플링이 이 처방의 핵심 부품**이라는 근거다. 세 모드를 모두 남겨 둔 이유는
`f0_decode=argmax`로 돌린 결과가 그대로 대조군이 되기 때문이다.

warm start: `pansori_aux` best에서 conv 스택을 물려받고, 출력층만
(2채널 → 257) shape 불일치로 버려져 새로 학습된다. f0 CE가 5.565 ≈ ln(256)에서
시작하는 것으로 확인했다.

**주의**: `losses['f0']`가 이제 CE라 회귀 run의 `val/f0`(L1, 0.15 근처)와 **비교할 수 없다.**
대신 `f0cent`(argmax 디코드의 평균 절대 cents 오차)를 같이 로깅하고, 이 값은 진단이지
손실이 아니므로 `val_loss`에서 빼둔다 — 안 그러면 200 규모의 값이 `save_best`를 장악한다.

### ❌ §5.5 결과 — 절반만 맞았다. 중단.

`pansori_aux_cls1`, 4,000 step. f0 CE 5.54 → 3.75, `f0cent` 928 → 186 cents로
**회귀 baseline의 음정 정확도(184 cents)에 도달**했다. 학습 자체는 정상이다.

**시김새 (150발화, 노트 안쪽 기준 정답 134.9)**

| | 노트내 변동 | 정답 대비 | 상관 |
|---|---|---|---|
| 회귀 baseline | 18.9 | 14% | 0.129 |
| Stage B | 20.6 | 15% | 0.126 |
| cls + argmax | 20.9 | 16% | 0.103 |
| **cls + sample** | **216.9** | **161%** | **0.015** |

**온도 스윕** (60발화, 정답 125.6)

| 디코드 | 노트내(안쪽) | 정답대비 | 상관 | cents RMSE | 옥타브점프 |
|---|---|---|---|---|---|
| argmax | 22.4 | 18% | 0.049 | 254.1 | 0.02% |
| sample T=0.2 | 47.5 | 38% | 0.048 | 260.3 | 0.49% |
| sample T=0.4 | 80.2 | 64% | 0.016 | 269.6 | 1.60% |
| sample T=0.6 | 117.0 | 93% | 0.027 | 286.0 | 3.40% |
| sample T=0.8 | 163.9 | 130% | 0.017 | 315.6 | 6.69% |
| sample T=1.0 | 217.7 | 173% | 0.018 | 350.7 | 10.35% |

**온도는 진폭 손잡이일 뿐이고 상관은 어느 지점에서도 0.02~0.05다.** T=0.6에서 진폭은
정답의 93%로 맞지만 내용이 무관하다. 스윗스팟이 없다.

**왜 — 예측 분포의 폭을 재면 나온다.**

| | |
|---|---|
| 예측 분포 표준편차 | **228 cents** |
| 유효 폭 exp(H) | 36칸 = **638 cents** |
| 시김새 폭 | **126 cents** |

**분포가 시김새보다 2배 넓고, 유효 폭이 5반음을 넘는다.** 이 불확실성의 정체는
"어떤 시김새인가"가 아니라 **"어떤 음인가"**다. 그래서 샘플링은 음 단위 오류를 뽑고,
그게 옥타브 점프로 나타난다.

**그래서 §5.5가 답한 것과 못 답한 것**

- ✅ "L1의 평균화가 범인인가" → **부분적으로 그렇다.** 평균화를 없애니 모델이 분산을
  표현할 수 있게 됐다(argmax→sample의 차이가 그 증거다).
- ❌ 하지만 **충분하지 않다.** 프레임별 주변분포(marginal)에서 독립으로 뽑으면
  아무리 잘 뽑아도 백색잡음이다. 필요한 건 시간축 결합분포다.

**빠진 조각 두 개가 이제 명확하다:**

1. **잔차 파라미터화** — 노트를 조건으로 주고 그 위의 편차만 모델링해야, 샘플링되는
   분포가 시김새 스케일(±126 cents)이 되지 음 스케일(±228 cents)이 되지 않는다.
2. **시간적 일관성** — contour는 독립 추출의 나열이 아니다. 한 덩어리로 뽑아야 한다.

**Stage C가 정확히 이 둘을 동시에 준다.** 토큰 하나 = 잔차 shape 하나이자 시간적으로
일관된 한 덩어리다. 우회로였던 §5.5가 결국 Stage C의 설계를 뒷받침하는 근거가 됐다.

참고: 코드베이스의 `pitch_ar` 경로(AR pitch predictor)는 **깨져 있다** —
`modules/fastspeech/fs2.py:216`이 `self.pitch_predictor(decoder_inp, f0)`로 부르는데
`PitchPredictor.forward`는 인자를 하나만 받는다. 값싼 AR 옵션은 없다.

---

## 6. 순서와 게이트

```
Stage A (반나절) ──게이트 B0──┬─ 통과 ─→ Stage B (1~2일) ──┬─ within_std ≥ GT의 50% ─→ 끝, ds3 재학습
                              │                            └─ 미달 ─→ Stage C
                              └─ 실패 ─→ Stage C (1주+)

        └─ 병행: §5.5 cross-entropy f0 (반나절)
```

재현:
```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python pansori/eval/midi_baseline.py \
    --config usr/configs/midi/cascade/pansori/aux.yaml \
    --exp_name pansori_aux --target aux --n 24 --html
```

## 7. 아직 확인하지 않은 것

- 체크포인트를 torch 2.0.1 / Python 3.8에서 로드할 수 있는지 (Stage B의 첫 작업)
- 게이트 B0 — 코드북이 평평함과 시김새를 실제로 구분하는지
- 토크나이저 학습 corpus와 우리 `Pansori_Data`의 중복 여부.
  겹치면 Stage A의 held-out 성격이 약해진다 (`splits/inhouse_crepe_split.csv`의 곡명과 대조할 것)
- 24발화는 표본이 작다. 결론이 갈리면 `--n 100` 이상으로 재확인
