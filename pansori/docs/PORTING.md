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

## 4. 라벨 품질 — 고친 것과 남은 것

`alignment/check_dict_agreement.py` 가 채점한다. MFA TextGrid 의 원본 IPA 를 Stage 4 와 같은
문맥 규칙으로 변환해 `dictionaries/korean-coda.txt` 와 대조한다. 사전은 `build_dictionary.py`
의 자모 표에서 나오고 `ipa_to_coda.csv` 와 독립이라 정답지 역할을 한다.

일치율 100% 는 목표가 아니다. **연음은 정상적인 한국어**이고 (은 → `eu n`, 좋 → `j o`)
문맥 없는 음절 사전으로는 재현할 수 없다. 차분으로 쓸 것 — 고치기 전에 재고, 고친 뒤에 재고,
숫자가 오르면 채택한다.

### 3갈래 수정 (1음절 87.92% → **89.59%**, 2음절 75.01% → **76.43%**)

| 결함 | 조치 | 기호별 일치율 |
|---|---|---|
| 연구개·성문 자음의 구개음화 활음 미반영 (냐 → `n a`) | `c cʰ cʰː c͈ ç ɲ ɟ ʝ ɕʰ ɕ͈` 에 ` y` 추가 | ç 22.8→**74.1**, ɲ 61.0→**84.5**, c 79.0→**96.2** |
| `ɸ → p` (후 → `p w u`). ɸ 는 ㅍ 이 아니라 **ㅎ 의 원순모음 앞 변이** | `ɸ→h`, `ɸʷ→h w` + `ʷ` 가 o/u 앞이면 활음 생략 | ɸ 0→**96.9**, ʷ 전체 **93.9** |
| `ʎ` 가 어느 위치에서도 틀림 (`L\|r` 순서 반대, 게다가 ㄹㄹ 겹침은 한 토큰으로 표현 불가) | `L r\|L y` — 모음 사이면 앞 음절 종성 + 뒤 음절 초성 두 토큰, 자음 앞이면 종성 하나 | ʎ 0→**84.5** |

세 번째를 위해 `convert_phone` 이 위치 대안에 여러 토큰을 담을 수 있게 됐다 (빨리 = `pp a` +
`L r` + `i`). 활음 생략 규칙도 일반화됐다: `ʲ` 는 `i` 앞에서, `ʷ` 는 `o`/`u` 앞에서 떨어진다 —
후행 모음이 이미 그 조음을 갖고 있으면 활음이 아니라 동시조음이기 때문이다.

음소 인벤토리는 34종 그대로고 OOV 는 0이다. 반영 결과 26,955항목 중 **3,109개(11.5%)** 의
`ph_seq` 가 바뀌었고 `07_validate.py` 는 PASS 다.

### 남은 것 (측정만)

| 이슈 | 근거 | 규모 |
|---|---|---|
| 사전의 ㅈ/ㅉ/ㅊ + 활음 | 죠는 [tɕo] 라 MFA 는 `j o`, 사전은 `j y o`. 죠/조가 동음이라 **사전 쪽** 문제 — 매핑에 `y` 를 붙이면 훨씬 흔한 조가 깨진다 | 96건+ |
| 요 → `y eo` vs 사전 `y o` | 모음 쪽 이슈, 미조사 | 156건 |
| ㄴ 이 `d` 로 (내/나/니/너) | MFA 오인식. 매핑으로는 못 고친다 | 700건+ |
| 무음 흡수 임계 | 50 ms 만 겹쳐도 무음 전체를 흡수. 커버율 50% 미만인 흡수 구간이 6.2%(3,772개) | 소 |

---

## 4b. openvpi 를 처음 돌리며 드러난 것

포팅 시점에는 알 수 없었던, 실제 실행에서만 나온 사실들이다.

- **`AP` 는 음소셋에 하드코딩돼 있다** (`utils/phoneme_utils.py:18`) 그리고 binarize 는 셋의
  모든 항목이 코퍼스에 나타나야 통과시킨다. 우리 라벨에 들숨 구분이 없으므로
  `merged_phoneme_groups: [[SP, AP]]` 로 한 인덱스에 합쳤다. 없는 구분을 지어내는 것보다
  라벨이 말하는 대로 두는 쪽이 맞다.
- **`spec_max` 기본값 `[0]` 은 이 코퍼스에 맞지 않는다.** 실측 상한이 **+2.838** 이라
  정규화 후 diffusion 타깃이 [-1, 1] 을 벗어난다. `configs/pansori_spec_stats.yaml` 을
  `base_config` 에 얹었다. 반대로 하한은 안전하다 — `nvSTFT` 가 `log(clamp(x, 1e-5))` 로
  압축하므로 -11.513 아래가 존재할 수 없고, 실측 하한도 -11.563(여유 0.05 포함)이다.
  **spec 값은 학습 시점에만 쓰이므로 재-binarize 는 필요 없다.**
- **acoustic binarize 도 HN-SEP 체크포인트를 요구한다.** breathiness/voicing/tension 을 전부
  꺼도 `DecomposedWaveform` 은 무조건 생성되고 기본 `hnsep: vr` 이 `checkpoints/vr/` 를 찾는다.
- **`requirements.txt` 에 `torchaudio` 가 없다** (upstream 누락). RMVPE 가 `Resample` 을 쓴다.
- **`~/.local` 이 conda env 앞에 끼어든다.** 거기 낡은 tensorflow 가 있으면 tensorboard
  import 가 protobuf 에서 죽고, 거기 numpy 가 있으면 pip 이 요구사항을 "이미 설치됨"으로
  오판해 env 안에 안 넣는다. `PYTHONNOUSERSITE=1` 이 필요하다.
- **`pansori/server/run.sh` 는 커밋된 적이 없었다.** 루트 `.gitignore` 의 `*.sh` 에 걸려
  `3e8d541` 이 README 만 담았다. `pansori/.gitignore` 에서 되돌렸다.
- **binarize 결과는 예상보다 작다.** acoustic 4.5 GB / variance 381 MB. 파형을 담지 않고
  44.1 kHz·hop 512 는 24 kHz·hop 128 보다 프레임이 적어서다 (구 16 GB 의 1/3).

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
