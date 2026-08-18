# MoonInTheRiver → openvpi 이식 기록

`~/Desktop/DiffSinger_moonInRiver`(MoonInTheRiver/DiffSinger 포크)에서 만든 판소리 파이프라인을
openvpi/DiffSinger 로 옮긴 기록이다. **merge 가 아니라 port** 다 — 두 저장소는 공통 조상이 없고
레이아웃·샘플레이트·모델 구조가 전부 다르다. `git merge` 나 `git remote add` 는 성립하지 않는다.

원본 저장소의 판소리 작업물은 이식 전에 `341e927` 로 커밋해 두었다. 그 이전에는 파이프라인의
대부분이 untracked 라 `git stash` 하나로 사라질 수 있는 상태였다.

---

## 1. 대응 관계

| MoonInTheRiver | openvpi | 조치 |
|---|---|---|
| 24 kHz / 80 mel / hop 128 | 44.1 kHz / 128 mel / hop 512 | 원본 오디오가 44.1 kHz 라 무손실 재커팅 |
| `transcriptions.txt` 7필드 (`is_slur`, `midi_dur`) | `transcriptions.csv` (`ph_num`, `note_seq`, `note_dur`) | 06 재작성 |
| 하드코딩 `CODA_36` | `dictionaries/*.txt` + language tag | `dictionaries/korean-coda.txt` 신규 |
| `usr/pansori_binarizer.py` 화자 오버라이드 | `datasets:` 의 `speaker`/`spk_id` | 삭제, 화자별 디렉토리로 |
| `data_gen_utils.py` f0_min 패치 | `f0_min`/`f0_max` 가 이미 config 키 | 패치 불필요 |
| CREPE `.npy` 주입 (`08_f0.py`) | `pe: rmvpe` 로 binarize 시 추출 | 삭제 → `08_export.py` 로 대체 |
| `pansori/warm_start.py` | `finetune_ckpt_path` / `finetune_ignored_params` | 삭제 |
| `pansori/train_wandb.py` | `scripts/train.py` + TensorBoard | 삭제 |
| cascade aux/ds 2단 구성 | acoustic / variance 로 이미 분리 | config 2개로 |
| 추론 경로 없음 | `scripts/infer.py` + `.ds` 파일 | 사전이 생겼으므로 가사→음소 경로도 열림 |

---

## 2. 라벨 스키마

옛 포맷은 멜로디를 **음소 축에 얹었다**: `note_seq`/`midi_dur`/`is_slur` 가 모두 `ph_seq` 와 같은
길이여야 해서, 한 모음이 N개 노트에 걸치면 그 모음을 N번 복제하고 2번째부터 `is_slur=1` 로 표시했다.

openvpi 는 두 축을 분리한다 (`preprocessing/variance_binarizer.py`). 노트는 자기 duration 을 갖고
프레임에 독립적으로 정렬된다. 그래서 **복제도 `is_slur` 도 사라졌다.** 시김새는 "한 단어 위에 노트
여러 개"로 그대로 표현된다 — 정보 손실이 아니라 표현이 더 자연스러워진 것이다.

새로 필요해진 것은 `ph_num`(단어별 음소 개수)뿐이다. MFA `words` tier 에서 유도한다. `04_align.py`
가 이제 이 tier 를 `phones.jsonl` 에 함께 기록한다 — **MFA 재실행 없이 파싱만 다시 하면 된다.**

`words` tier 는 어절 단위(공백 토큰)라 2음절 어절이 한 덩어리로 온다. 사전이 예측한 음소열과
실제 음소열이 정확히 일치하면 음절 단위로 쪼개고(나간다 → 2/3/2), 연음으로 개수가 달라진 어절은
(굴러 → `g u L eo`, 사전은 `g u L r eo`) 추측하지 않고 통째로 둔다.

유지한 것 — 포맷이 아니라 **이 데이터**에 관한 사실이기 때문이다:

1. **노트에 덮인 무음은 held vowel 이지 쉼이 아니다.** korean_mfa 는 speech 로 학습돼 긴 지속음을
   무음으로 라벨한다. 실측: 코퍼스 시간의 **52.2%(15.11 h)** 가 "MFA 는 무음, ROSVOT 은 노래"인
   구간이고, 그 구간의 노트 커버율 중앙값은 **1.00** 이다. 흡수하지 않으면 노래하는 구간에서 모델에
   조용히 하라고 가르치게 된다. 노트가 없는 무음(0.74 h)만 SP 가 된다.
2. **20 ms 미만 음소는 이웃에 접는다.** hop 512 / 44.1 kHz 에서 1프레임은 11.6 ms 다.

결과: 26,955 항목 전부 `07_validate.py` PASS, 0프레임 음소 0개, 평균 6.6단어 / 13.0노트.

---

## 3. 이식 중에 고친 실제 결함

`pansori_lib.convert_phone` 에 세 갈래 버그가 있었다. 단음절 어절 89,303개(모호성 없음)를 사전과
대조해 확인했고, 수정 후 사전 일치율이 **77.3% → 87.9%** 로 올랐다.

1. **`|` 와 활음이 같이 있는 항목에서 활음이 사라졌다.** `mʲ → "m|M y"` 를 `split("|")` 로 먼저
   가르면 `coda = "M y"` 라는 존재하지 않는 단일 토큰이 되고, 초성 쪽은 `["m"]` 만 남아 `y` 가
   증발했다. 며/면/벼 계열이 전부 활음을 잃었다 (며 → `m eo`, 794건).
2. **다음 음소가 활음이면 초성/종성 판정이 종성으로 떨어졌다.** `vowel_base("w")` 가 None 이라
   뭐 = `M w eo` 처럼 종성 ㅁ 이 어두에 왔다 (168건).
3. 수정 후 `y` 토큰이 크게 늘었다 (활음이 제자리를 찾음). OOV 0, 음소 34종 유지.

---

## 4. 남은 라벨 품질 이슈 (미결)

측정만 해두고 손대지 않았다. `ipa_to_coda.csv` 를 고치고 `04_align.py --skip_mfa` 를 다시 돌리면
(2분) 반영된다. **MFA 재실행은 필요 없다.**

| 이슈 | 근거 | 규모 |
|---|---|---|
| 연구개·성문 자음의 구개음화 활음 미반영 | `ɲ` 행 주석에 "y 삽입 필요"라고 적혀 있으나 coda 열에는 없음 | 냐 → `n a` 222건 등 |
| `ɸ → p` 매핑 | 후/호가 `p w u`/`p w o` 로 나옴. korean_mfa 에서 ɸ 는 ㅎ 의 원순모음 앞 변이 | 256건+ |
| `ʎ → "L|r"` 순서 | 다른 항목은 `초성|종성` 순인데 이것만 반대. 종성 위치에서 초성 토큰 `r` 이 나옴 | — |
| 사전의 ㅈ/ㅉ/ㅊ + 활음 | 죠는 [tɕo] 로 발음되어 MFA 는 `j o`, 사전은 `j y o` | 96건+ |
| 무음 흡수 임계 | 50 ms 만 겹쳐도 무음 전체를 흡수. 커버율 50% 미만인 흡수 구간이 6.2%(3,772개) | 소 |

잔여 불일치의 다수는 **정상적인 연음**이다 (은 → `eu n`, 좋 → `j o`). 문맥 없는 음절 사전으로는
재현할 수 없는 것이 맞고, 추론 시 학습 라벨과 어긋나는 지점이기도 하다.

---

## 5. 변하지 않는 제약

`alignment/VERDICT.md`, `alignment/FIELD_CHECK.md` 의 결론은 판소리와 이 데이터에 관한 것이라
아키텍처를 바꿔도 그대로 유효하다.

- **발화는 6초 이하.** korean_mfa 가 speech 로 학습돼 6초를 넘으면 정렬이 붕괴한다.
- **재세그멘테이션은 필수.** 라벨된 창 구간 37.5 h 중 21초 미만은 3.3 h 뿐이다.
- **8자리 uuid 접두로 조인.** `labels_2.json` 의 경로 문자열은 NFC/NFD 가 섞여 있다.
- **f0 범위 55–750 Hz.** 남성 통성은 MIDI 35(61.7 Hz)까지 내려간다. 기본값 65 Hz 면 잘린다.
- **한국어 종성은 중국어 word_boundary 로직을 깬다.** openvpi 는 `ph_num` 을 라벨에서 직접 받으므로
  이 문제는 구조적으로 사라졌다.

---

## 6. 폐기한 것

- **checkpoints `pansori_ds2` / `pansori_aux` / `pansori_ds3`** — 24 kHz·80 mel·FastSpeech2 라
  44.1 kHz·128 mel·reflow 모델과 텐서가 하나도 맞지 않는다. "56k step 에서 과적합" 같은 수렴 지점
  지식도 이월되지 않는다.
- **`0102_xiaoma_pe`, `0109_hifigan_bigpopcs_hop128`** — 24 kHz 자산.
- **시김새/contour tokenizer 계열** (`usr/contour_tokenizer.py`, `pansori_pitch.py`, `note_seg.py`)
  — 이번 이식 범위에서 제외. 진단 문서(`SIGIMSAE_PLAN.md`, `VIBRATO_VERDICT.md`)만 옮겼다.
  **진단의 전제가 달라졌다**: 평탄화의 원인 가설은 "결정론적 회귀 + L1 은 조건부 중앙값으로
  수렴한다"였는데, openvpi 의 pitch predictor 는 reflow 기반 생성 모델이다. 토크나이저를 붙이기
  전에 `eval/midi_baseline.py` 를 새 variance 모델에 돌려 평탄화가 여전한지부터 재보는 것이 맞다.
