# 학습은 이렇게 진행된다 — 한 스텝씩

전처리가 끝난 뒤(`PIPELINE.md` A·B 단계) `scripts/train.py` 를 치면 안에서 무슨 일이 벌어지는지,
**배치 하나가 만들어져 파라미터가 한 번 갱신되기까지**를 순서대로 적은 문서다.

- 파이프라인 전체 기록은 [`PIPELINE.md`](PIPELINE.md), 실행법 요약은 [`../README.md`](../README.md).
- 여기 나오는 수치는 전부 이번에 실제로 돌린 `pansori_ac_v1` / `pansori_var_v1` 의 로그와
  텐서보드에서 뽑은 것이다. 추정치가 아니다.

---

## 0. 한 장 요약

**모델 두 개를 따로 학습한다.** 서로 주고받는 것 없이 완전히 독립이고, 순서도 상관없다
(이번에는 acoustic 을 돌리는 중간에 variance 를 같이 띄웠다).

| | 파라미터 | 무엇을 받아 | 무엇을 배우나 |
|---|---|---|---|
| **variance** `pansori_var_v1` | 25.1 M | 음소 + 노트(MIDI·길이) | ① 음소별 길이 `ph_dur` ② f0 곡선 |
| **acoustic** `pansori_ac_v1` | 69.1 M | 음소 + `ph_dur` + f0 + 화자 | mel 스펙트로그램 (128빈) |
| NSF-HiFiGAN | — | mel + f0 | 파형 — **학습하지 않는다.** 받아서 그대로 쓴다 |

한 스텝의 뼈대는 둘 다 같다:

```
  binary 데이터에서 배치 하나 꺼내기
        ↓
  [인코더] 음소 → 벡터 → 프레임 축으로 늘리기 = condition
        ↓
  [헤드]  condition 을 조건으로 정답을 맞히기
        ↓
  손실 → 역전파 → Muon+AdamW 한 번 갱신
        ↓
  4,000 스텝마다: 검증 + 그림/오디오 + 체크포인트
```

용어는 이 다섯 개만 알면 된다.

| 이름 | 모양 | 뜻 |
|---|---|---|
| `tokens` | `[T_ph]` | 음소 번호 열. 이 코퍼스는 36종(Coda 34 + `SP`/`AP` 병합 + PAD) |
| `ph_dur` | `[T_ph]` | 음소별 길이 (**프레임 수**, 초가 아니다) |
| `mel2ph` | `[T_s]` | 프레임마다 "몇 번째 음소인가". `ph_dur` 을 펼친 것 |
| `condition` | `[T_s, 384]` | 프레임 단위로 늘려 놓은 조건 벡터. 모든 헤드의 입력 |
| `base_pitch` | `[T_s]` | 노트를 프레임으로 펼쳐 부드럽게 만든 기준 음고 (semitone) |

**프레임**은 44,100 / 512 = **초당 86.13개**다. 평균 발화 3.82초 ≈ **329 프레임**.
아래에서는 이 평균 발화(음소 14개 · 329 프레임)를 계속 예로 든다.

---

## 1. `train.py` 를 치면 — 시작 전 준비

```sh
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
```

### 1-1. config 를 겹쳐 쌓는다

`base_config:` 를 따라 재귀적으로 병합한다. 실제 체인은 로그 첫 줄에 찍힌다:

```
| Hparams chains: ['configs/base.yaml', 'configs/acoustic.yaml',
                   'configs/pansori/spec_stats.yaml', 'configs/pansori/acoustic.yaml',
                   'configs/pansori/data_pansori.yaml', 'configs/exp/ac_v1.yaml']
```

**뒤에 오는 것이 이긴다.** `f0_min` 이 base 에서 65 였다가 pansori 에서 55 로 덮이는 식이다.
병합 결과는 `checkpoints/{exp}/config.yaml` 에 통째로 저장되고, **다음 실행부터는 이 파일이
기준이 된다.** `--reset` 은 "저장본을 무시하고 yaml 에서 다시 만들라"는 뜻이다
(`utils/hparams.py:87-115`). config 를 고쳤는데 `--reset` 없이 재실행하면 고친 게 안 먹는다.

### 1-2. task 클래스를 문자열로 찾아 띄운다

`task_cls: training.acoustic_task.AcousticTask` 를 `importlib` 으로 import 해서
`AcousticTask.start()` 를 부른다 (`scripts/train.py:22-27`). config 만 바꾸면 다른 모델이
돌아가는 구조다.

### 1-3. Trainer 조립 (`basics/base_task.py:398-440`)

| 설정 | 값 | 뜻 |
|---|---|---|
| `precision` | `16-mixed` | forward/backward 는 fp16, 가중치·옵티마이저는 fp32 |
| `max_steps` | 100,000 (ac) / 80,000 (var) | 여기 닿으면 종료 |
| `val_check_interval` | 4,000 | **epoch 이 아니라 step 기준** |
| `check_val_every_n_epoch` | `None` | epoch 경계는 아예 무시한다 |
| `accumulate_grad_batches` | 1 | 배치 1개 = 스텝 1개 |
| `gradient_clip_val` | 1.0 | |
| `num_sanity_val_steps` | 1 | 학습 전에 검증 배치 **1개**만 돌려 본다 |
| callbacks | `DsModelCheckpoint`, `DsTQDMProgressBar` | |
| logger | `DsTensorBoardLogger` → `checkpoints/{exp}/lightning_logs/latest` | |

### 1-4. 추론이 쓸 물건을 work_dir 로 복사한다

`spk_map.json` · `lang_map.json` · `dictionary-ko.txt` 를 binary 디렉토리에서
`checkpoints/{exp}/` 로 복사한다 (`base_task.py:444-462`).
**나중에 사전을 고쳐도 이 체크포인트는 학습 당시의 사전으로 추론한다.** 화자 번호와 음소 번호가
어긋나는 사고를 막는 장치다.

### 1-5. 이어하기가 기본값이다

```python
trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))
```

`checkpoints/{exp}/model_ckpt_steps_*.ckpt` 중 **step 이 가장 큰 것**을 자동으로 찾아 재개한다
(가장 최신이 아니라 **step 이 가장 큰** 것 — 파일 시각이 아니라 이름으로 고른다).
옵티마이저 상태·스케줄러·epoch 까지 복원한다. 처음부터 다시 하려면 디렉토리를 비워야 한다.

> 재개 시 옵티마이저 하이퍼파라미터는 **config 값이 체크포인트를 덮는다**
> (`base_task.py:471-514`). lr 을 고치고 재개하면 그 lr 이 먹고, 스케줄러는
> `simulate_lr_scheduler` 로 현재 step 까지 다시 굴려 맞춘다.

---

## 2. 배치 하나가 만들어지기까지

### 2-1. 디스크에서

`data/pansori/binary_acoustic/` 에 `train.data` (한 덩어리 바이너리) + `train.idx` (오프셋 표)
+ `train.meta` (길이·이름·화자명 같은 메타)가 있다. `IndexedDataset` 이 오프셋으로 바로 읽는다 —
전체를 메모리에 올리지 않는다.

이번 코퍼스 기준:

| | 발화 | 길이 |
|---|---|---|
| train | 25,946 | 100,101.68 s = **27.8 h** |
| valid | 1,009 | 3,881.78 s = **1.08 h** |

valid 는 **홀드아웃 곡 5개**에서 나온 것이다 — 이일주·김소희·조상현·박동진·박초월 5명뿐이고,
**나머지 18명은 valid 에 한 발화도 없다** (binarize 로그의 `valid respective duration`).
검증 손실은 이 5명에 대한 값이라는 뜻이다. 원래 6명이었는데, 홀드아웃 곡이 화자의 전부를
차지하던 정권진은 `make_configs.py` 가 자동으로 해제했다 (학습 데이터 0이면 화자 임베딩에
gradient 가 한 번도 안 간다).

### 2-2. 배치로 묶기 — `DsBatchSampler` (`utils/training_utils.py:76-197`)

epoch 마다 이 순서로 돈다:

1. **셔플** — 전체 인덱스를 무작위로 섞는다.
2. **길이순 정렬** — 길이를 6프레임 격자(`sampler_frame_count_grid`)로 반올림해 내림차순 정렬.
   → **비슷한 길이끼리 한 배치에 모인다. 패딩을 줄이려는 것이다.**
   (1번 셔플은 길이가 같은 것들 사이의 순서를 매 epoch 바꾸는 역할을 한다.)
3. **잘라 담기** — `max_batch_frames` 50,000 / `max_batch_size` 32 중 **먼저 걸리는 쪽**에서 끊는다.
4. **배치 순서 셔플** — 이렇게 만든 배치들의 순서를 다시 섞는다. 안 그러면 긴 발화만 나오는
   구간이 생긴다.

**실측: 항상 32가 먼저 걸린다.** 평균 329프레임 × 32 ≈ 10,500 프레임, 최장 6초(517프레임)라도
32개면 16,500 프레임이라 5만 프레임 상한에 절대 안 닿는다. 그래서 로그의 `batch_size` 는
언제나 32이고, **epoch 당 배치 수는 811개**(= ⌈25,946 / 32⌉)로 고정이다.

```
100,000 step ÷ 811 batch/epoch = 123.3 epoch   ← 로그의 마지막 줄이 "Epoch 123"
```

### 2-3. 패딩 — `collater`

배치 안에서 가장 긴 것에 맞춰 채운다. **채우는 값이 항목마다 다르다**:

| 항목 | 패딩 값 | 이유 |
|---|---|---|
| `tokens`, `mel2ph`, `ph_dur` | `0` | 0은 PAD 음소. `mel2ph > 0` 이 곧 유효 프레임 마스크 |
| `mel` | `log(1e-5)` = −11.513 | 진짜 무음. 0으로 채우면 최대 진폭이 된다 |
| `note_midi` | `-1` | 음수를 패딩 표시로 쓴다 (`padding_mask = note_midi < 0`) |
| `note_rest` | `True` | 패딩은 쉼표로 취급 |

---

## 3. acoustic 한 스텝 — 69.1 M

`training/acoustic_task.py:run_model` 하나가 아래 3-1 ~ 3-4 전부다.
배치 `B=32`, 예시 발화 `T_ph=14`, `T_s=329`, mel 빈 `M=128`.

### 3-1. condition 만들기 (`FastSpeech2Acoustic`)

```
tokens [32,14]  ──txt_embed(36→384)──────┐
mel2ph [32,329] ──세어서 dur 복원─→ log(1+dur) ──dur_embed──┤
                                          ├─→ encoder (4층, self-attn 2 head, RoPE)
                                          │        ↓  [32,14,384]
                                          │   mel2ph 로 gather = 프레임 축으로 늘리기
                                          │        ↓  [32,329,384]
                                     + stretch_embed   ← 이 음소가 얼마나 늘어났나
                                     + spk_embed       ← 화자 24종
                                     + pitch_embed(f0) ← log(1 + f0/700)
                                          ↓
                                    condition [32,329,384]
```

핵심은 **"늘리기"(length regulate)** 다. 음소 14개짜리 벡터열을 `mel2ph` 를 색인으로 써서
329 프레임으로 복사·확장한다. 6프레임짜리 음소면 그 벡터가 6번 반복된다.

**학습 때는 정답 `ph_dur`(에서 만든 `mel2ph`)을 쓴다.** 즉 acoustic 은 길이를 스스로 정하지
않는다 — 그건 variance 담당이다. 추론 때도 누군가(variance 또는 사람)가 길이를 줘야 한다.

`stretch_embed` 는 "이 프레임이 속한 음소가 원래보다 몇 배 늘어났나"를 넣어 준다. 복사만
하면 6프레임짜리 `a` 와 60프레임짜리 `a` 가 프레임 단위로는 구별이 안 되기 때문이다.
판소리처럼 한 음을 길게 끄는 노래에서 특히 중요한 입력이다.

### 3-2. aux decoder — mel 을 한 번에 그려 본다

`use_shallow_diffusion: true` 라서 ConvNeXt 6층짜리 보조 디코더가 `condition` 에서 mel 을
**생성 과정 없이 곧장** 뽑는다. 손실은 정규화 mel 에 대한 **L1**.

```python
aux_cond = condition * 0.1 + condition.detach() * 0.9    # aux_decoder_grad
```

보조 디코더의 gradient 는 **1/10 로만** 인코더에 흘린다. 인코더가 "대충 그리는 쪽"에
끌려가지 않게 막는 장치다 (`modules/toplevel.py:109`).

### 3-3. reflow — 진짜 mel 은 여기서 배운다

`modules/core/reflow.py:36-56`. 네 줄로 끝난다.

```python
spec  = norm_spec(gt_mel)                       # spec_min/max 로 [-1, 1] 정규화
t     = 0.4 + 0.6 * rand(B)                     # 배치의 발화마다 스칼라 하나
x_t   = noise + t * (spec - noise)              # noise ~ N(0, I)
v_pred = backbone(x_t, t*1000, cond)            # LynxNet2 6층 × 1024ch, kernel 31
loss   = L2(v_pred, spec - noise)               # 정답 속도 = 목적지 − 출발지
```

**노이즈와 정답 mel 을 직선으로 잇고, 그 위 무작위 한 지점에서 "어느 쪽으로 가야 하나"를
맞히게 한다.** 정답 방향은 언제나 `spec - noise` 로 상수다. 추론 때는 이 속도장을 따라
20번 나눠 걸어간다(euler).

**`T_start = 0.4` 가 이 모델의 성격을 결정한다.** `t` 를 `[0, 1]` 이 아니라 `[0.4, 1]` 에서만
뽑기 때문에 **acoustic 은 순수 노이즈에서 출발하는 법을 배우지 않는다.** 0.4 지점의 출발선은
추론 때 aux decoder 가 그린 mel 이 채운다:

```
추론:  x = 0.4 * aux_decoder(condition) + 0.6 * noise    ← 여기서 시작
       euler 20 step, dt = (1 − 0.4)/20 = 0.03
```

이게 shallow diffusion 이다. 밑바닥부터 안 그리니 20 스텝으로도 충분하다.

`main_loss_log_norm: false` (acoustic) 이므로 t 에 따른 손실 가중은 없다. 모든 t 가 동등하다.

> **패딩이 손실 값에 섞인다.** 손실은 패딩 프레임을 0으로 만든 뒤 `.mean()` 을 취하는데,
> 분모에는 패딩도 들어간다 (`modules/losses/reflow_loss.py:42-50`). 패딩이 많은 배치는
> 손실이 낮게 나온다. 길이순 정렬(2-2)이 패딩을 줄이는 것이 학습 안정에도 도움이 되는 이유고,
> **검증은 배치당 1발화라 패딩이 아예 없다** — 그래서 검증 손실은 이 영향에서 자유롭다.

### 3-4. 총 손실

```
total_loss = mel_loss + 0.2 × aux_mel_loss
```

로그에서 실제로 본 마지막 값: `mel_loss=0.15771`, `aux_mel_loss=0.01899`
(이미 0.2가 곱해진 값이다 — `lambda` 를 곱한 뒤 dict 에 담는다).

---

## 4. variance 한 스텝 — 25.1 M, 헤드 두 개

`training/variance_task.py:run_model`. **한 스텝에서 duration 과 pitch 를 동시에 학습한다.**

### 4-1. duration 헤드

```
tokens ──txt_embed──┐
onset  ─────────────┤   onset = 단어의 첫 음소인가 (ph2word 의 차분)
word_dur ───────────┤   단어 길이 (정답 ph_dur 을 단어별로 합산)
                    ├─→ encoder 4층 → encoder_out [B, T_ph, 384]
                    │        + midi_embed (음소별 평균 MIDI, 128종 임베딩)
                    │        + spk_embed
                    │              ↓
                    │      DurationPredictor (resnet 5층 × 256ch, Conv1d k=3)
                    │              ↓  log 도메인 출력
                    └─────→ ph_dur_pred [B, T_ph]
```

**학습 때 단어 길이는 정답에서 만들어 준다** (`variance_encoder.py:76-79`). 즉 duration 헤드가
배우는 것은 "이 문장이 몇 초냐"가 아니라 **"주어진 단어 길이를 음소들에 어떻게 나눠 줄 것이냐"**
쪽에 가깝다. 추론 때는 `.ds` 의 노트 길이가 그 자리를 대신한다.

손실은 세 겹이고, 전부 **log 도메인 MSE** 다 (`modules/losses/dur_loss.py`):

| 항 | 가중치 | 무엇을 맞히나 |
|---|---|---|
| `pdur` | 0.3 | 음소 하나하나의 길이 |
| `wdur` | 1.0 | 단어 단위 합계 |
| `sdur` | **3.0** | 문장 전체 길이 |

**문장 길이가 음소 길이보다 10배 무겁다.** 개별 음소가 조금씩 틀리는 것보다 총 길이가 밀리는
쪽이 훨씬 나쁘기 때문이다 — 총 길이가 어긋나면 뒤따르는 f0 와 mel 이 통째로 노트에서 벗어난다.
로그를 log 로 재는 것도 같은 이유다: 0.05초짜리 음소의 0.02초 오차와 2초짜리 음소의 0.02초
오차를 같게 취급하지 않는다.

### 4-2. pitch 헤드

```
condition (음소를 프레임으로 늘린 것, acoustic 과 같은 방식)
   + MelodyEncoder(note_midi, note_dur)  ← 노트 축을 따로 인코딩해 프레임으로 gather
   + pitch_retake_embed                  ← 4-3 참조
   + delta_pitch_embed
        ↓
   PitchRectifiedFlow      타깃 = pitch − base_pitch  (semitone 편차)
```

**타깃이 f0 자체가 아니라 "노트에서 얼마나 벗어났는가"** 다. 절대 음고는 이미 노트가 알려
주므로, 모델은 시김새·요성·이음새만 배우면 된다. 클립 범위 ±12 semitone, 정규화 범위 ±8.

`repeat_bins: 64` — 프레임당 값 하나짜리 곡선을 64칸으로 복제해 mel 처럼 2D 로 만든다.
같은 백본(LynxNet2)을 쓰기 위한 트릭이고, 추론 때는 64칸의 평균을 낸다.

acoustic 과 다른 점 두 가지:

- **`t ~ U(0, 1)`.** variance 는 shallow diffusion 이 아니다 (config 에 `use_shallow_diffusion`
  이 없어 `RectifiedFlow` 가 `t_start` 를 0으로 강제한다). 순수 노이즈에서 시작한다.
- **`main_loss_log_norm: true`.** t 에 로그정규 모양 가중이 붙어 **t≈0.5 부근을 무겁게** 본다
  (`reflow_loss.py:27-35`). 양 끝(거의 노이즈 / 거의 정답)은 덜 중요하게 친다.

### 4-3. retake mask — 왜 학습 때 정답을 일부 보여 주나

```python
pitch_retake = random_retake_masks(B, T)   # 평균적으로 프레임의 절반이 True
```

`True` 인 구간은 "다시 그려라", `False` 인 구간은 **정답 피치를 그대로 입력으로 넣어 준다.**
매 스텝 무작위로 연속 구간을 잡아 준다 (`variance_task.py:74-81`).

편집기에서 **"이 부분만 다시 뽑기"** 를 쓰려면 모델이 "주어진 앞뒤 피치에 이어 붙이는 법"을
알아야 한다. 전 구간을 항상 처음부터 그리게만 학습하면 부분 재생성 시 이음매가 튄다.
추론 때 전체를 새로 뽑으면 마스크가 전부 `True` 가 된다.

### 4-4. 총 손실

```
total_loss = 1.0 × dur_loss + 1.0 × pitch_loss
```

**여기가 나중에 문제가 된다.** 두 헤드는 서로 다른 것을 배우는데 손실은 하나로 합쳐지고,
`dur_loss` 가 `pitch_loss` 보다 자릿수가 커서 `total_loss` 는 사실상 `dur_loss` 다
(7절 표를 볼 것). 체크포인트를 `total_loss` 로 고르면 피치를 버리게 된다.

---

## 5. 역전파와 갱신

Lightning 이 알아서 하는 부분이지만 설정은 알아 둘 것.

| | 값 | 메모 |
|---|---|---|
| 정밀도 | `16-mixed` | fp16 forward + `GradScaler`. VRAM 24 GB 중 **6.1 GB** 만 썼다 |
| gradient clip | 1.0 (L2 norm) | |
| 옵티마이저 | `Muon_AdamW` | **파라미터를 둘로 나눈다** ↓ |
| lr | 6e-4 | betas (0.9, 0.98) |
| 스케줄러 | `StepLR(5000, 0.8)` | **step 단위**. 5,000 스텝마다 0.8배 |

**Muon 과 AdamW 를 섞어 쓴다** (`modules/optimizer/muon.py:161-200`):

- **Muon** — 2차원 이상의 일반 weight (attention·conv 커널 등). Newton–Schulz 반복으로
  gradient 를 직교화해서 적용한다. weight decay 0.1.
- **AdamW** — `nn.Embedding`, `AdamWLinear`, `AdamWConv1d` 로 선언된 것과 모든 1차원 파라미터
  (bias·LayerNorm). weight decay 0.

`f0`·`ph_dur` 같은 **스칼라 하나를 384차원으로 올리는 투영은 전부 `AdamWLinear`** 로 선언돼
있다. 입력이 1차원인 층은 Muon 의 직교화가 의미가 없어서다.

lr 실측: 100,000 스텝 시점에 **8.647e-06** — 초기값의 1/69 이다. 학습 후반은 사실상
미세조정 구간이고, 이게 7절에서 손실이 안 움직이는 이유의 절반이다.

**속도**: 약 **7.0 it/s** (RTX 4090, 배치 32). 811 배치 = 1 epoch ≈ 115초.
100,000 스텝이면 검증 제외 **약 4시간**.

---

## 6. 4,000 스텝마다 벌어지는 일

`val_check_interval: 4000`. epoch 과 무관하게 step 으로 끊는다.

### 6-1. 검증 — 두 번 돈다

```python
losses = self.run_model(sample, infer=False)     # ① 손실: 학습과 똑같이 계산
if 이 배치에 index < 10 인 발화가 있으면:
    out = self.run_model(sample, infer=True)     # ② 실제 생성: euler 20 step
```

- **① 은 valid 전체 1,009 발화에 대해** 돈다. `max_val_batch_size: 1` 이라 **한 발화씩** —
  패딩이 전혀 없다.
- **② 는 앞 10개 발화만.** 실제로 20스텝 샘플링하고, acoustic 은 NSF-HiFiGAN 으로 파형까지
  만들어 텐서보드에 **오디오와 mel 그림**을 올린다 (`num_valid_plots: 10`).
  variance 는 duration 막대그래프와 피치 곡선을 그린다.
- 검증은 `autocast` 를 끄고 fp32 로 돈다 (`base_task.py:266`).

> **검증 손실에도 난수가 들어 있다.** ①은 학습과 같은 코드라 `t` 와 노이즈를 매번 새로 뽑는다.
> 그래서 곡선의 한 칸(±0.005 정도) 차이는 신호가 아니다. **acoustic 의 28,000(0.17800)과
> 64,000(0.17988)은 사실상 구분되지 않는다.** 추세만 볼 것.

### 6-2. 체크포인트 — **손실 기준이 아니다**

```python
DsModelCheckpoint(monitor='step', mode='max', save_top_k=num_ckpt_keep)
```

`monitor='step'`, `mode='max'` — 즉 **"step 이 가장 큰 N개"**, 다시 말해 **최근 N개**를 남긴다.
검증 손실은 보관 여부에 아무 영향도 주지 않는다. acoustic `num_ckpt_keep: 8`,
variance `num_ckpt_keep: 5`.

영구 보존 규칙이 따로 있다 (`training_utils.py:269-286`): `step ≥ permanent_ckpt_start` 이고
`(step − start) % permanent_ckpt_interval == 0` 이면 삭제 대상에서 빠진다.
acoustic 은 start 60,000 / interval 10,000.

**그런데 이 규칙은 절반만 발동한다.** 저장 자체가 4,000 배수에서만 일어나므로
70,000 과 90,000 은 **애초에 존재한 적이 없다.** 실제로 살아남은 것:

```
영구:   60,000 · 80,000 · 100,000          ← 4,000 과 10,000 의 공배수만
최근 8: 72,000 76,000 80,000 84,000 88,000 92,000 96,000 100,000
```

`checkpoints/pansori_ac_v1/` 의 파일 목록이 정확히 이것이다(+ `best/` 에서 되돌려 놓은
12,000 · 28,000). **최저였던 28,000 은 학습이 끝났을 때 이미 지워져 있었다.**
`checkpoints/best/` 에 따로 복사해 두지 않았다면 남은 것 중 최저인 76,000(0.18335)으로
조용히 내려앉았을 것이다.

> 4,000 과 10,000 이 안 맞는 것은 upstream 기본값의 문제다. `permanent_ckpt_interval` 을
> `val_check_interval` 의 배수(예: 12,000)로 잡으면 규칙이 의도대로 작동한다.

---

## 7. 실제로 이렇게 흘러갔다

### acoustic (`validation/*`)

| step | total | mel | aux | |
|---:|---:|---:|---:|---|
| 0 | 1.25401 | 1.08727 | 0.16675 | sanity check, 배치 1개 — 곡선에서 제외할 것 |
| 4,000 | 0.19593 | 0.16167 | 0.03426 | |
| 12,000 | 0.18394 | 0.15173 | 0.03220 | |
| 16,000 | 0.20279 | 0.17124 | 0.03155 | 4,000 이후 최고 |
| **28,000** | **0.17800** | 0.14499 | 0.03301 | ← **최저** |
| 52,000 | 0.17981 | 0.14820 | 0.03161 | |
| 64,000 | 0.17988 | 0.14747 | 0.03241 | |
| 76,000 | 0.18335 | 0.15095 | 0.03240 | 남은 파일 중 최저 |
| 100,000 | 0.18834 | 0.15625 | 0.03209 | 마지막 — 최저보다 **5.8% 나쁘다** |

**4,000 스텝 만에 0.196 에 도달하고, 그 뒤 96,000 스텝 동안 0.178–0.203 사이를 진동만 했다.**
`aux_mel_loss` 는 8,000 스텝 이후 0.032 에서 사실상 정지한다. 설정대로 100,000 을 완주했지만
**필요했던 것은 3만 스텝 이하였다.**

### variance (`validation/*` + `metrics/*`)

| step | total | dur | pitch | rhythm_corr | pitch_r2 |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 0.20016 | 0.18772 | 0.01244 | 0.325 | 0.433 |
| 12,000 | 0.10261 | 0.09188 | 0.01073 | 0.500 | 0.465 |
| **16,000** | 0.08764 | **0.07797** | 0.00967 | 0.675 | 0.372 |
| 20,000 | **0.08761** | 0.07807 | 0.00955 | **0.750** | 0.434 |
| 24,000 | 0.09049 | 0.08169 | 0.00881 | 0.650 | 0.454 |
| 28,000 | 0.09877 | 0.09008 | 0.00868 | 0.575 | **0.501** |
| **32,000** | 0.10423 | 0.09561 | **0.00861** | 0.650 | 0.416 |

**두 헤드가 반대로 간다.** `dur_loss` 는 16,000 에서 바닥을 치고 그 뒤 계속 오르고,
`pitch_loss` 는 32,000 까지 계속 내려간다. `total_loss` 는 `dur` 가 자릿수로 지배하므로
사실상 `dur` 곡선이다 — 20,000 에서 최저를 찍은 것도 `dur` 때문이다.

`dur_loss` 가 3회 연속 오른 32,796 스텝에서 중단했다.

### 그래서 체크포인트는 헤드마다 따로 고른다

| 쓰는 곳 | 고른 기준 | step |
|---|---|---|
| acoustic | `validation/total_loss` | **28,000** |
| variance — 길이 | `validation/dur_loss` | **16,000** |
| variance — 피치 | `validation/pitch_loss` | **32,000** |

`total_loss` 로 고른 20,000 으로 피치를 뽑으면 32,000 보다 f0 오차가 **25% 크다.**
그래서 `examples/run_examples.sh` 는 **variance 를 두 번 돌린다** — 1차 `--predict dur`,
2차는 그 산출물에 `--predict pitch`.

도구: `python pansori/examples/best_ckpt.py {exp} [--metric validation/pitch_loss]`
(step 0 은 sanity check 라 후보에서 뺀다. 파일이 남아 있는 step 만 후보로 친다.)

---

## 8. 다음에 돌릴 때의 규칙

1. **`max_updates` 를 믿지 말 것.** 설정된 100,000 은 근거가 없는 기본값이었다.
   **검증 손실이 3회 연속(=12,000 스텝) 최저를 못 갱신하면 멈춘다** — variance 에는 이 규칙을
   적용해 32,796 에서 끊었고, acoustic 에는 적용하지 않아 GPU 3시간을 태웠다.
2. **멈추기 전에 `checkpoints/best/` 로 복사할 것.** `num_ckpt_keep` 은 손실을 보지 않는다.
   최저 체크포인트는 학습이 계속되는 동안 조용히 지워진다.
3. **헤드가 여러 개면 `total_loss` 를 쓰지 말 것.** 헤드별로 최저 step 을 따로 고른다.
4. **곡선의 한 칸 차이는 무시할 것.** 검증 손실 자체에 난수가 들어 있다 (6-1).
5. **variance 를 더 돌릴 여지가 있다.** `pitch_loss` 는 중단 시점까지 계속 내려가고 있었다.
   dur 헤드만 얼려 두고(`frozen_params`) pitch 를 더 돌리는 것이 다음 실험 후보다.

### 재실행 시 자주 밟는 것

- **config 를 고치고 `--reset` 없이 재실행** → `checkpoints/{exp}/config.yaml` 저장본이 이겨서
  고친 게 안 먹는다.
- **`--reset` 을 붙였는데 디렉토리를 안 비움** → config 는 새로 만들지만 `trainer.fit` 은
  여전히 마지막 체크포인트에서 **이어서** 돈다. `--reset` 은 config 만 리셋한다.
- **`spec_min`/`spec_max` 를 고쳤을 때** → 학습 시점에만 쓰는 값이라 **재-binarize 불필요.**
  `make_configs.py` 로 config 만 다시 만들면 된다.
- **두 학습은 독립이다.** 각 6 GB 대라 4090 한 장에서 동시에 돌아간다.

---

## 9. 학습이 추론에서 어떻게 뒤집히나

| | 학습 | 추론 |
|---|---|---|
| `mel2ph` | 정답 `ph_dur` 에서 | variance 가 예측한 `ph_dur` 에서 |
| f0 (acoustic 입력) | RMVPE 로 뽑은 실측 | variance 가 생성한 곡선 |
| reflow | `t` 무작위 1개, 속도 1회 예측 | `t_start`→1 을 20등분해 20회 예측 |
| retake mask | 무작위 연속 구간 | 전부 `True` (또는 편집 구간만) |
| aux decoder | 손실 계산용 | **0.4 지점의 출발선 생성용** |

이 대응 관계 때문에 `examples/run_examples.sh` 의 **사다리 4단**이 진단 도구가 된다 —
`2_resynth`(`ph_dur`+f0 제공) → `3_pitch`(`ph_dur`만) → `4_full`(노트+가사만) 로 내려가면서
모델이 스스로 만들어야 하는 것이 하나씩 늘어난다. 어디서 무너지는지가 곧 어느 학습이 부족한지다.

추론·평가는 [`PIPELINE.md`](PIPELINE.md) D·E 절.
