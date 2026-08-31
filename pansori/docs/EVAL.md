# 음소·발음 계측 — 세 개의 자

합성음이 **알아들리는가**를 재는 도구들. 파이프라인 전체 기록은
[`PIPELINE.md`](PIPELINE.md), 실행법 요약은 [`../README.md`](../README.md).

---

합성음의 발음이 부정확할 때, **그것을 잴 자가 이 저장소에 없었다.** `compare/` 의 지표
22개는 전부 피치 아니면 스펙트럼이고 (`compare/stats.py:40-60`), 가장 가까운 `voiced_agree`
조차 "이 프레임에서 소리를 내야 하는가"만 재므로 자음을 통째로 빼먹어도 떨어지지 않는다
(홀드아웃에서 0.95~0.97). [가사 A/B](LYRICS_AB.md) 를 포함해 지금까지의 모든 판정이 발음을 보지
못한 채 내려졌다. 이 절의 셋이 그 구멍을 메운다.

| 자 | 답하는 질문 | 무엇을 보나 | 비용 |
|---|---|---|---|
| **A** `variance_task.py` | 모델이 음소 길이를 정확히 예측하는가 | wandb, 매 검증 | 공짜 (자동) |
| **B** `eval/intelligibility.py` | 렌더된 소리가 실제로 알아들리는가 | 콘솔·wandb | 표본 40개 ~15분 |
| **C** `eval/label_report.py` | 학습 라벨 자체가 얼마나 망가졌나 | 콘솔·HTML | 몇 초 |

**셋 중 C 부터 볼 것.** 모델을 의심하기 전에 무엇을 보고 배웠는지 아는 편이 빠르다.

## 1. C — 라벨 진단

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

## 2. A — 학습중 wandb 지표

`pansori/train.py` 로 variance 를 돌리면 자동으로 붙는다 (README §4). config 의
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
옳은지는 이 지표가 답하지 않는다 — 그건 §3 의 일이다.

**함정 둘.**

- **범위가 다르다.** upstream 의 `ph_dur_acc` / `rhythm_corr` 은 `num_valid_plots`(10)
  게이트 안에 있어 **앞 10발화**만 채점한다 (`variance_task.py:253,260`). 새 `dur_acc_*` 는
  `num_valid_metric_items: 0` 으로 **검증셋 전체**를 본다. 두 수치를 직접 빼지 말 것.
  기존 둘은 이전 런과의 비교선을 위해 일부러 그대로 뒀다.
- **단위가 프레임이다.** binarize 가 `ph_dur` 을 mel 프레임으로 양자화한 뒤에야 모델이
  보므로 여기서도 프레임으로 잰다. 하한선은 3프레임(34.8 ms)이라 §1 의 **`≤30ms` 열이
  아니라 `≤3프레임` 열**과 맞춰 봐야 한다. 실측 검산: 검증셋 라벨의 ≤3프레임 비율은
  30.6899%, 학습이 기록한 `floor_rate_gt` 는 30.6974% — 13,379개 중 1개 차이(반올림 타이)다.

## 3. B — 발음 명료도 eval

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
3. **MFA 쪽 실질 지표는 `floor_rate` 와 `boundary_mae_ms` 둘이다.** `floor_rate` 는 §1 이
   라벨에 대는 것과 **같은 자를 합성음에 대는 것**이라, 두 표를 나란히 읽을 수 있다.
4. **Whisper 는 없는 소리를 채워 넣는다.** 그래서 두 backend 를 같이 본다. Whisper 가
   그럴듯한 문장을 뱉는데 MFA `floor_rate` 가 높으면, 그 문장은 상상이다.

**판독표 — 어느 단에서 떨어지면 무엇이 범인인가.**

| 증상 | 범인 | 다음에 볼 것 |
|---|---|---|
| `1_orig` 의 CER 이 0 근처가 아니다 | 계측 자체가 불안정 | 여기서 멈춘다 |
| `0_vocoder` 가 이미 나쁘다 | 보코더 · 추출기 | 그 아래 단은 읽을 필요 없다 |
| `2_resynth` 에서 이미 무너진다 | **라벨(MFA 정렬) 또는 음향모델** | §1 의 초성 하한선율 |
| `2_resynth` 는 멀쩡한데 `4_full` 에서 무너진다 | **duration 예측** | §2 의 `dur_acc_onset` |
| 전 단이 고르게 나쁘다 | 음소셋 · 사전 | `dictionaries/korean-coda.txt` |

**첫 관측 (n = 3, 시드 1 — 방향만 보는 예비 수치다).** 널 대조 0.000 통과. CER 은
`0_vocoder` 0.33 → `2_resynth` 1.00 → `4_full` 1.69, MFA 하한선율은 `1_orig` 30% →
`2_resynth` 40% → `4_full` 65%. 즉 **`2_resynth` 단계에서 이미 무너진다** — GT `ph_dur` 과
GT f0 를 주었는데도 그렇다. 판독표대로면 duration 예측이 아니라 라벨·음향모델 쪽이다.
표본이 3개뿐이므로 `--n 40 --seeds 3` 으로 다시 재고 나서 판정할 것.
