# pansori/compare — 구·신 모델 정량 비교

`~/Desktop/DiffSinger_moonInRiver`(MoonInTheRiver cascade, 24 kHz)와 이 저장소의
`pansori_ac_v1` + `pansori_var_v1`(openvpi reflow, 44.1 kHz)을 **같은 홀드아웃 발화에서
같은 자로** 재는 평가 패키지. 리포트: `report.html`.

## 결론 한 줄

**신 모델은 판소리처럼 움직이지만, 원본과 함께 움직이지는 않는다.**
노트내 변동비 0.29 → 0.92 (Holm p = 0.0005, 시드 잡음과 보코더 천장 격차를 모두 넘음),
원본과의 상관 0.035 → 0.037 (변화 없음).

## 실행

```sh
conda activate dsp && cd ~/Desktop/DiffSinger && export PYTHONPATH=. PYTHONNOUSERSITE=1

python pansori/compare/smoke.py                              # 게이트 5개 + 널 대조
python pansori/compare/render_new.py --items fair --seeds 5  # 19항목 × 5시드, ~40초
python pansori/compare/collect.py                            # → out/results.jsonl
python pansori/compare/stats.py                              # → out/summary.json + 콘솔 표
python pansori/compare/report.py                             # → report.html
```

홀드아웃 확장(절대 수치 + 대표성 검정):

```sh
python pansori/compare/render_new.py --items sample:200 --seeds 3
python pansori/compare/collect.py --items sample:200 --seeds 3 --only-new --no-spectral \
       --out pansori/compare/out/results_ext.jsonl
python pansori/compare/stats.py     # results_ext.jsonl 이 있으면 자동으로 붙는다
```

**`smoke.py` 가 실패하면 그 뒤 숫자는 전부 무효다.** 특히 G1(샘플레이트 무관성)이
깨지면 24 kHz 시스템과 44.1 kHz 시스템을 한 표에 놓을 수 없다.

## 이 코드가 아는 함정 다섯

1. **정권진 `a580300d` 5항목은 신 모델의 학습 데이터다.** 구 모델은 홀드아웃했지만
   신 모델 설정에서는 곡 하나가 화자 전부라 자동 해제됐다. 24항목 중 이 5개를 빼고
   **공정 쌍 19개**가 헤드라인이다. `items.assert_held_out()` 이 실행할 때마다
   `configs/exp/ac_v1.yaml` 의 `test_prefixes` 를 읽어 강제한다 — 주석이 아니다.
2. **`|Δf0| per frame` 은 프레임률 아티팩트다.** 구 파이프라인 5.33 ms vs 신 11.61 ms.
   같은 곡선이 신 쪽에서 ~2.2배로 나온다. 여기서는 전부 **10 ms 공통 격자의 cents/초**로
   잰다. `docs/NEXT.md §C5` 의 프레임 단위 비율은 시스템 간 비교에 쓸 수 없다.
3. **RMVPE 는 `f0_min`/`f0_max` 를 받기만 하고 쓰지 않는다** (`modules/pe/rmvpe/inference.py:63-78`).
   남성 통성에서 옥타브를 떨어뜨린다 — 19항목 중앙값 4.4%, 최악 28%. 보정 없이는
   **원본**의 노트내 변동이 650 cents 로 부풀어 공표값 161 과 어긋난다. 그래서 노트
   잔차를 (−600, 600] 로 감고(`sigimsae.octave_snap`), **기준과 모든 시스템에 동일하게**
   적용한다. 이 보정은 Tier 2 에만 쓴다 — 진짜 옥타브 오류는 음정 지표에서 벌점을 받아야 한다.
4. **구 `2_gt_crepe` 는 GT mel 이다**(보코더 천장). 신 `2_resynth`(음향모델 통과)와 같은
   것이 아니어서, 신 쪽에 `0_vocoder` 를 새로 만들었다. 대응은
   구 `5_pred_own` ↔ 신 `3_pitch` 하나뿐이다.
5. **두 저장소의 노트 격자는 동일하다.** `cumsum(note_dur)` 가 구 `notes.jsonl` 온셋을
   19항목 전부 0.000 ms 오차로 재현한다(게이트 G2). 그래서 계단 하나가 양쪽을 다 잰다 —
   구 `note_seg.py` 를 이식할 필요가 없다.

## 설계 결정

- **기준 f0 는 RMVPE 하나로 통일.** 다만 구 모델은 CREPE 타깃으로 학습됐으므로
  기준 교체가 불리할 수 있어, 두 기준의 차이(162 cents, 원본 노트내 변동 CREPE 133 vs
  RMVPE 160)를 리포트에 싣는다. 진폭 효과는 이 차이보다 크고, 음정 쪽은 그렇지 않아
  애초에 개선으로 주장하지 않았다.
- **시드**: 신 모델은 reflow 샘플러라 5시드 중앙값. 구 `pansori_aux` 는 결정론적
  회귀라 렌더 하나가 곧 모델의 답이다 — 이 비대칭이 정당한 이유가 그것이다.
  `pansori_ds3` 는 샘플러라 스펙트럼 전용·단일 draw 표기.
- **주장 규칙(사전 등록)**: 시드 IQR 의 2배 초과 **∧** Holm p < 0.05 **∧**
  보코더 천장 격차 초과. 세 값을 모두 표에 인쇄한다.
- **사인 비브라토 적합은 넣지 않는다.** `docs/VIBRATO_VERDICT.md` 가 이 코퍼스에서
  반증했다 — 같은 사인이 무관한 노트의 잔차도 똑같이 잘 맞았고(대조 0.829 vs 실측 0.782),
  깨끗한 주기 성분을 가진 노트는 3.9% 뿐이다.
- **PESQ/STOI 도 넣지 않는다.** 미설치이고, 협대역 전화음성 명료도용이라 판소리에서는
  권위 있어 보이는 무의미한 숫자가 나온다.
- `pansori/eval/` 은 import 하지 않는다 (`note_seg.py` 가 없어 실행 불가, `docs/NEXT.md §C4`
  가 폐기 권고). 정의만 이식했다. `f0_metrics.gross_error_rate` 는 분모가 공통 유성
  프레임이라 **어려운 프레임을 무성 처리하면 점수가 좋아지는 결함**이 있어 그대로 쓰지 않고,
  RPA-50 은 기준의 유성 프레임을 분모로 쓴다(두 버전을 나란히 보고한다).

## 파일

| 파일 | 책임 |
|---|---|
| `config.py` | 분석 격자 10 ms, 공통 스펙트럼 기저, 시드, 체크포인트, **사전 등록 종점 4개**. CLI 오버라이드 없음 |
| `items.py` | 홀드아웃 강제, 공정 19쌍, 층화 표본 |
| `notes.py` | 노트 격자·계단·core 마스크·경계 |
| `f0.py` | 10 ms RMVPE (디스크 캐시) + 구 CREPE 로더 |
| `sigimsae.py` | 잔차 분해, 옥타브 스냅, 모션(cents/s), wobble, 전이 |
| `pitch_metrics.py` | RPA/RCA/유무성/옥타브 점프 |
| `spectral.py` | 24 kHz 공통 기저 MCD/LSD, 대역 에너지 |
| `render_new.py` | 인프로세스 배치 렌더러 (prep → variance → acoustic 3프로세스) |
| `collect.py` | → `out/results.jsonl` (long 포맷) |
| `stats.py` | Wilcoxon · 부트스트랩 CI · Holm · 시드 바닥 · 확장 대표성 |
| `plots.py` | 인라인 SVG (테마 대응) |
| `report.py` | 자체완결 HTML |
| `smoke.py` | 게이트 G1–G5 + 널 대조 |

`render_new.py` 는 모델을 프로세스당 한 번만 올린다. `scripts/infer.py` 를 렌더마다
띄우면 같은 일이 ~2.6시간이고 대부분이 모델 로딩이다 — 실측 304렌더 19초.

## 알려진 한계

- `4_full`(길이까지 예측)은 구 사다리가 만든 적이 없어 비교 대상이 없다. 프레임 정렬도
  깨지므로 프레임 기반 지표에서 제외하고 분포 지표만 준다.
- 일부 홀드아웃 항목은 `note_slur` 복원이 불가능해(단어 > 노트) `4_full` 입력을 만들 수
  없다. 해당 항목은 그 단만 빠지고 나머지 비교에는 그대로 참여한다.
- 구 모델의 잡음 바닥은 측정하지 못했다(렌더가 하나뿐).
- n = 19 는 작다. 구 렌더가 24개뿐이라 정해진 크기이고, 신 항목을 더해도 쌍체 검정력은
  늘지 않는다. 그래서 200발화 확장은 절대 수치와 **대표성 검정**에만 쓴다.
