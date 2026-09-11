# 판소리 DiffSinger — 전체 파이프라인 단계별 기록

원본 오디오 한 덩어리에서 "부르는 모델"까지, 실제로 돌아간 순서 그대로 적은 문서다.
각 단계는 **무엇을 받아 무엇을 내놓는지 · 왜 그렇게 하는지 · 실측 수치 · 재실행 비용**을 같이 적었다.

- 실행법 요약은 [`../README.md`](../README.md), 이식 경위는 [`PORTING.md`](PORTING.md),
  **학습 한 스텝 안에서 벌어지는 일**은 [`TRAINING.md`](TRAINING.md),
  6초 제약의 근거는 [`SEGMENTATION_VERDICT.md`](SEGMENTATION_VERDICT.md), 남은 일은 [`NEXT.md`](NEXT.md).
- 브랜치 `feat/pansori`. **upstream 파일은 한 줄도 고치지 않았다.**

```
원본 394곡(28 GB)
  └ 01 세그멘테이션 ─ 02 화자매핑 ─ 03 전사 ─ 04 정렬 ─ 05 노트 ─ 06 병합 ─ 07 검증 ─ 08 내보내기
      └ 사전 생성 ─ config 생성 ─ binarize ─ spec 통계 부트스트랩
          └ acoustic 학습 ┐
            variance 학습 ┴─ 체크포인트 선택(헤드별)
                └ .ds 작성 ─ variance(dur) ─ variance(pitch) ─ acoustic ─ NSF-HiFiGAN ─ wav
                    └ 사다리 4단 · 악보 합성 ─ 측정 ─ 리포트
```

---

## 0. 준비물

### 하드웨어 / 원본

| 항목 | 값 |
|---|---|
| GPU | RTX 4090 (24 GB) — 실제 사용 6.1 GB |
| 원본 오디오 | `~/Desktop/Pansori_Data/Audio/` 394곡, 44.1 kHz 스테레오, **28 GB** (demucs 로 분리한 보컬) |
| 창 구간 라벨 | `labels_2.json` — 곡 안의 악조 구간(계면조/우조/평조/설렁제/경드름)과 아니리·창조 |
| 노트 MIDI | `rosvot_midi/*_vocal.mid` — ROSVOT 이 곡 단위로 뽑아 둔 음표 타임라인 |
| 가사 텍스트 | **없음** → 3단계에서 전사부터 해야 한다 |

원본 경로는 `preprocess/pansori_lib.py` 의 `PANSORI_DATA` 한 곳에만 하드코딩돼 있다.

### conda 환경 3개

| env | Python | 쓰는 곳 | 이유 |
|---|---|---|---|
| `pansori` | 3.10 | 01–03, 06–08 | `base` 는 numba ↔ numpy 2.5 충돌로 06 이 죽는다 |
| `aligner` | 3.14 | 04–05 | MFA 3.4.1 + `korean_mfa` + mecab-ko |
| `dsp` | 3.10 | binarize / train / infer | openvpi 본체. **`PYTHONNOUSERSITE=1` 필수** |

`PYTHONNOUSERSITE=1` 이 없으면 `~/.local/lib/python3.10/site-packages` 의 낡은 tensorflow 가
tensorboard import 를 protobuf 에서 죽이고, 거기 numpy 때문에 pip 이 요구사항을 "이미 설치됨"으로
오판해 env 안에 안 넣는다. `dsp` env 의 `activate.d` 에 넣어 뒀다.
`requirements.txt` 에는 **`torchaudio` 가 빠져 있다** (upstream 누락) — RMVPE 가 `Resample` 을 쓴다.

`bash pansori/server/run.sh setup` 이 위를 전부 한다.

---

## A. 전처리 8단계

단계 사이를 **jsonl 파일로 끊어 뒀다.** 재실행 비용이 단계마다 100배씩 차이나기 때문이다 —
MFA 정렬은 수 시간이지만 그 결과를 다시 파싱하는 것은 2분이다. 실제로 음소 매핑을 고쳤을 때
04는 정렬을 다시 돌리지 않고 파싱만 재실행했다.

모든 단계는 `pansori/data/{set}/` 아래에서 돌고 `--set` 으로 고른다 (실제 사용: `--set full`).

### 데이터가 줄어드는 지점

| 지점 | 남은 수 | 줄어든 이유 |
|---|---|---|
| 원본 오디오 | 394곡 | — |
| 창 구간 라벨과 교집합 | **387곡** | 라벨이 없거나 오디오가 없는 곡 |
| 01 세그멘테이션 | **31,529 발화 / 33.48 h** | 아니리·창조 제외, 2초 미만 조각 폐기 |
| 04 MFA 정렬 성공 | **26,955 발화** | 가사 2자 미만이라 정렬 대상 제외 + 정렬 실패 (−4,574) |
| 06 병합 | 26,955 | 노트 없음/전부 rest 로 탈락한 항목 0 |
| 08 최종 코퍼스 | **26,955 발화 / 28.93 h / 화자 23명** | — |

최종 라벨 규모: 음소 370,831개 · 노트 349,774개 · 단어 179,776개
(발화당 평균 음소 13.8 · 노트 13.0 · 단어 6.7).

---

### 01 — 세그멘테이션 `preprocess/01_segment.py`

| | |
|---|---|
| 입력 | `labels_2.json` + `Audio/*.wav` + `rosvot_midi/*_vocal.mid` |
| 출력 | `data/full/segments.jsonl` + `data/full/wavs/{item}.wav` |
| env | `pansori` |

곡을 **2–6초 발화**로 자른다. `item_name` 은 `{uuid8}_{index:04d}` 형식이고, 이 8자리 uuid 접두가
나중에 홀드아웃 단위가 된다 (`labels_2.json` 의 경로 문자열은 NFC/NFD 가 섞여 있어 파일명 대신
uuid 로 조인한다).

**왜 6초인가.** korean_mfa 는 말소리로 학습돼 긴 지속음에서 self-loop 페널티가 누적된다. 6초를
넘으면 정렬이 붕괴한다 — 음소가 0.03초 바닥값에 몰리고 나머지가 통째로 무음으로 먹힌다. 표본
10개 실측: 6초 이하에서는 1초 이상 침묵이 **한 건도 없고**, 6초 초과에서는 평균 **4.06초**가
침묵으로 잡혔다. 표본이 작아 확정된 임계값이라기보다 유력한 가설이지만, 재세그멘테이션 자체는
선택지가 아니다 — 라벨된 창 구간 37.5시간 중 21초 미만이 3.3시간뿐이다.

**어디서 자르는가** (`segment_span`):

1. `librosa.effects.split(top_db=35)` 로 비-무음 구간을 뽑고, 구간 사이 간격이
   **0.4초 이상**이면 진짜 숨자리로 보고 자른다. 실측 간격 분포가 중앙값 0.13초 / p90 0.66초라
   이 값이 숨 쉬는 자리와 **시김새 안의 짧은 멈춤**을 가른다.
2. 2초 미만 조각은 앞 그룹에 흡수한다 — 단, 합쳐서 4.5초(soft target)를 넘지 않을 때만.
3. 아직 6초를 넘는 그룹은 쪼갠다. 내부에 무음이 있으면 **폭 × 균형 점수**가 최대인 곳에서
   자르고(균형 항이 없으면 가장 넓은 멈춤이 가장자리에 붙어 2초짜리 조각만 떨어져 나간다),
   무음이 전혀 없는 긴 멜리스마는 **ROSVOT 노트 간격 중 가장 넓은 곳**으로 자른다 —
   지속음 한가운데를 자르지 않기 위해서다.

결과: 31,529 발화 / 33.48 h / 평균 3.82초 (min 2.00 / max 6.00).
악조 분포는 계면조 22,321 · 우조 7,057 · 평조 1,746 · 설렁제 310 · 경드름 95.

> 이 단계의 알고리즘 시뮬레이션은 `experiments/sim_merge.py`, `sim_final.py`, `diag_split.py`.

---

### 02 — 화자 매핑 `preprocess/02_speaker_map.py`

| | |
|---|---|
| 입력 | `segments.jsonl` |
| 출력 | `data/full/spk_map.json` |
| env | `pansori` |

파일명 `{uuid8}-{track}-{명창}-{바탕}_{대목}` 의 3번째 필드가 한글 2–4자면 명창으로 본다.

**명창을 특정할 수 없는 곡은 버리지 않고 `unknown` 한 사람으로 모은다.** 곡마다 다른 id 를 주면
발화 몇 개짜리 싱글턴 화자가 ~122명 생기고 그 임베딩은 학습이 안 된다. 모아 두면 6.16시간짜리
**단일 최대 화자**가 된다 (코퍼스의 21.3%).

---

### 03 — 가사 전사 `preprocess/03_transcribe.py`

| | |
|---|---|
| 입력 | `segments.jsonl` + `wavs/*.wav` |
| 출력 | `data/full/lyrics.jsonl` (`text_raw`, `text`) |
| env | `pansori` (GPU) |

`openai/whisper-large-v3`, fp16, 16 kHz, `language="korean"`. transformers 의 ASR *pipeline* 은
쓰지 않는다 — 깨진 torchcodec 을 무조건 import 한다. 모든 세그먼트가 30초 미만이라 long-form
청킹이 필요 없어서 모델을 직접 호출한다.

**반복 접기가 필요하다.** 지속음 하나가 Whisper 에게는 같은 음절의 반복으로 들려서
`지이이이…이(×200)` 같은 산출물이 나온다. `pansori_lib.normalize_repeats` 가 3회 이상 반복을
하나로 접고, 정렬은 접힌 쪽(`text`)을 쓴다. 원본(`text_raw`)은 검토용으로 남긴다.

Whisper 는 판소리에서 완벽하지 않다. 목표는 완벽한 가사가 아니라 **음소가 시간축에서 맞는 것**이다.

---

### 04 — 강제 정렬 `preprocess/04_align.py`

| | |
|---|---|
| 입력 | `segments.jsonl` + `lyrics.jsonl` + `wavs/*.wav` |
| 출력 | `data/full/phones.jsonl` (`phones`, **`words`**) + `mfa_corpus/`, `aligned/` |
| env | `aligner` |

1. `mfa_corpus/` 에 wav 심볼릭 링크 + `.lab` (정규화 텍스트)를 깐다. 한글 2자 미만은 제외.
2. `mfa align {corpus} korean_mfa korean_mfa {aligned} --clean --output_format long_textgrid`
3. TextGrid 를 파싱해 **IPA 108종 → 자모 기반 Coda 34종**으로 변환한다.

변환은 `pansori_lib.convert_phone` 이 하고, 문맥 규칙이 두 종류 붙는다 — 자음이 초성인지
종성인지, 그리고 활음(`y`/`w`)을 살릴지 뺄지. IPA 한 구간이 Coda 두 토큰으로 갈라지면
(`mʲ → m y`) 구간 길이를 균등 분할한다.

**`words` tier 를 함께 기록하는 것이 이 단계의 핵심 변경이다.** openvpi 가 요구하는 `ph_num`
(단어별 음소 개수)은 여기서만 유도할 수 있다. 남겨 두면 매핑을 고칠 때마다
`--skip_align`(옛 이름 `--skip_mfa` 도 받는다) 로 **정렬 없이 파싱만** 다시 돌릴 수 있다
(수 시간 → 2분).

정렬기는 `preprocess/align.yaml` 의 `aligner` 가 정한다 — `mfa`(기본) 또는 `sofa`.
백엔드의 속은 `preprocess/aligners.py` 에 있고, 둘 다 `aligned{suffix}/*.TextGrid` 를
남기므로 여기서 아래로는 아무것도 달라지지 않는다. SOFA 는 사전이 곧 음소셋이라
`dictionaries/korean-coda.txt` 를 주면 IPA→Coda 변환 단계 자체가 없다. 절차는
README §3.1–3.2.

무음은 `sil` 플레이스홀더로 남긴다 — 그것이 쉼인지 지속음인지는 노트를 봐야 알 수 있고,
그 판단은 06이 한다.

결과: 26,955 / 31,529 정렬 성공, OOV 0.

---

### 05 — 노트 슬라이싱 `preprocess/05_notes.py`

| | |
|---|---|
| 입력 | `segments.jsonl` + `phones.jsonl` (정렬에 살아남은 항목) |
| 출력 | `data/full/notes.jsonl` |
| env | `aligner` 또는 `pansori` |

ROSVOT 을 다시 돌리지 않는다. 곡 단위 MIDI 를 각 발화의 절대 시간 창으로 잘라 로컬 시간으로
옮길 뿐이다. 노트 음역은 **MIDI 35 (B1, 61.7 Hz) ~ 78 (F♯5, 740 Hz)** — 이 값이 나중에
`f0_min: 55` / `f0_max: 750` 의 근거가 된다.

---

### 06 — 병합 `preprocess/06_merge.py`

| | |
|---|---|
| 입력 | `phones.jsonl` + `notes.jsonl` + `segments.jsonl` + `dictionaries/korean-coda.txt` |
| 출력 | `data/full/transcriptions.csv` (`name, ph_seq, ph_dur, ph_num, note_seq, note_dur`) |
| env | `pansori` |

파이프라인에서 판단이 가장 많이 들어가는 단계다. 순서대로:

**① 음소를 단어에 배정** (`assign_words`) — 음소 **중점**이 어느 `words` 구간에 들어가는지로
판정한다. 04에서 균등 분할된 두 토큰도 중점이 원래 구간 안에 있으므로 같은 단어로 간다.

**② 노트가 덮은 무음은 흡수** (`absorb_silence`) — 이 파이프라인에서 가장 중요한 규칙이다.
korean_mfa 는 말소리로 학습돼 **긴 지속음을 무음으로 라벨한다.** 실측: 코퍼스 시간의
**52.2%(15.11시간)** 가 "MFA 는 무음, ROSVOT 은 노래"인 구간이고 그 구간의 노트 커버율
중앙값은 **1.00** 이다. 그대로 두면 노래하는 구간에서 모델에게 조용히 하라고 가르치게 된다.
그래서 노트가 50 ms 이상 덮은 무음은 **앞 음소의 끝을 늘려 흡수**하고, 노트가 없는
무음(0.74시간)만 `SP` 로 남긴다.

**③ 20 ms 미만 음소는 이웃에 접는다** (`merge_tiny`) — 44.1 kHz · hop 512 에서 한 프레임이
11.6 ms 라 그 아래는 0프레임으로 반올림된다. 접히는 쪽의 구간은 살아남은 이웃이 물려받는다.

**④ 경계 재유도** — 음소 구간이 발화를 빈틈없이 덮도록 첫 음소는 0.0, 마지막은 발화 길이로
고정한 뒤 `ph_dur` 를 경계에서 다시 뽑는다. 반올림이 총합을 파형 길이에서 밀어내지 못하게.

**⑤ `ph_num` 생성** (`group_words`) — MFA `words` tier 는 **어절 단위**(공백 토큰)라 2음절
어절이 한 덩어리로 온다. 사전이 예측한 음소열과 실제 음소열이 **정확히 일치하면** 음절 단위로
쪼개고 (나간다 → 2/3/2), 연음으로 개수가 달라진 어절은 (굴러 → `g u L eo`, 사전은
`g u L r eo`) **추측하지 않고 통째로 둔다.**

**⑥ 노트 타임라인** (`note_timeline`) — 노트 사이 빈틈을 `rest` 로 채워 `[0, duration]` 을
빈틈없이 타일링한다. 음이름은 `librosa.midi_to_note`.

> **왜 `is_slur` 가 사라졌나.** 옛(MoonInTheRiver) 포맷은 멜로디를 음소 축에 얹었다 —
> `note_seq`/`midi_dur`/`is_slur` 가 모두 `ph_seq` 와 같은 길이여야 해서 한 모음이 노트 N개에
> 걸치면 그 모음을 N번 복제하고 2번째부터 `is_slur=1` 을 붙였다. 판소리처럼 한 음절을 길게
> 굴리는 음악에서는 이 복제가 라벨의 상당 부분을 차지한다. openvpi 는 두 축을 분리해서
> (`preprocessing/variance_binarizer.py`) 노트가 자기 duration 을 갖고 프레임에 독립적으로
> 정렬된다. **복제도 슬러 플래그도 필요 없고, 시김새는 "한 단어 위에 노트 여러 개"로 그대로
> 표현된다** — 정보 손실이 아니라 표현이 더 자연스러워진 것이다. 실제 발화 `01ed19bb_0003`
> 기준으로 음소 16개가 옛 포맷에서는 24칸이 되고 그중 8칸이 복제였다.

---

### 07 — 검증 `preprocess/07_validate.py`

| | |
|---|---|
| 입력 | `transcriptions.csv` (+ `segments.jsonl`) |
| 출력 | stdout PASS/FAIL, FAIL 이면 exit 1 |
| env | `pansori` |

openvpi 의 binarizer 는 **워커 프로세스 안에서 assert 로 죽는다** — 항목을 스킵하는 게 아니라
40분짜리 작업 전체가 중간에 멈춘다. 그래서 그 assert 들을 전부 앞당겨 검사한다.

binarizer 가 부딪히는 순서대로:

- `len(ph_seq) == len(ph_dur)` — acoustic + variance
- `sum(ph_num) == len(ph_seq)` — variance, `predict_dur`
- `len(note_seq) == len(note_dur)` — variance, `predict_pitch`
- 비-rest 노트가 최소 하나 — variance, `predict_pitch`
- `ph_dur ≥ 0`, `note_dur ≥ 0`
- 모든 음소가 사전에서 해결됨 — `PhonemeDictionary.encode` KeyError

binarizer 가 **하지 않는** 검사도 같이 한다:

- `sum(ph_dur) ≈ sum(note_dur) ≈ wav 길이` (허용 0.05초) — 조용한 프레임 드리프트
- 0프레임으로 반올림되는 음소 (토큰 낭비, 치명적이지는 않음)

**PASS 없이 binarize 하지 않는다.** 현재 상태: 26,955 항목 전부 PASS, 0프레임 음소 0개.

---

### 08 — 내보내기 `preprocess/08_export.py`

| | |
|---|---|
| 입력 | `segments.jsonl` + `spk_map.json` + `transcriptions.csv` + 원본 `Audio/` |
| 출력 | `data/pansori/{화자}/wavs/*.wav` + `{화자}/transcriptions.csv` + `datasets.yaml` |
| env | `pansori` · 약 20분 |

01이 만든 wav 를 쓰지 않고 **원본에서 다시 자른다.** 원본이 이미 44.1 kHz 스테레오라
리샘플링 없이 스테레오→모노 평균 + 샘플 그리드 슬라이스만 한다 (PCM_16). 곡 하나를 열어
그 곡의 모든 세그먼트를 뽑는 식이라 I/O 가 곡 수만큼만 발생한다.

**화자 지정 방식이 바뀌었다.** 옛 파이프라인은 binarizer 서브클래스에서 화자를 오버라이드했지만
openvpi 는 config 의 `datasets:` 항목 단위로 지정한다 (`basics/base_binarizer.py:75-94`).
그래서 화자별 디렉토리로 부채꼴 전개하고, 발화 수 내림차순으로 정렬한 `datasets.yaml` 을
함께 쓴다 — `spk_id 0` 이 가장 데이터가 많은 화자가 되도록.

> 옛 `08_f0.py`(CREPE `.npy` 주입)는 폐기했다. openvpi 는 binarize 때 `pe: rmvpe` 로 f0 를
> 직접 뽑는다. 주입은 불필요할 뿐 아니라 **틀리다** — 옛 그리드는 hop 128 @ 24 kHz (5.33 ms),
> 지금은 hop 512 @ 44.1 kHz (11.6 ms)다.

최종 코퍼스 (`data/pansori/`, 8.7 GB):

| 화자 | 발화 | 시간 | 비중 | 홀드아웃 |
|---|---:|---:|---:|---|
| unknown | 5,759 | 6.16 h | 21.3% | |
| 이일주 | 3,792 | 4.06 h | 14.0% | `c235c995` |
| 김소희 | 3,275 | 3.58 h | 12.4% | `d207244a` |
| 조상현 | 2,690 | 2.85 h | 9.9% | `8b4d0824` |
| 안숙선 | 1,738 | 1.86 h | 6.4% | |
| 김일구 | 1,624 | 1.77 h | 6.1% | |
| 박동진 | 1,512 | 1.61 h | 5.6% | `e719dced` |
| 박초월 | 1,137 | 1.25 h | 4.3% | `d7ec123c` |
| 박록주 | 1,005 | 1.08 h | 3.7% | |
| 성우향 | 717 | 0.77 h | 2.6% | |
| 오정숙 | 691 | 0.73 h | 2.5% | |
| 정정렬 | 598 | 0.62 h | 2.1% | |
| 김연수 | 405 | 0.43 h | 1.5% | |
| 신금홍 | 319 | 0.35 h | 1.2% | |
| 남해성 | 259 | 0.27 h | 0.9% | |
| 성창순 | 254 | 0.27 h | 0.9% | |
| 박양덕 | 245 | 0.27 h | 0.9% | |
| 김수연 | 240 | 0.26 h | 0.9% | |
| 안향련 | 228 | 0.24 h | 0.8% | |
| 정권진 | 226 | 0.24 h | 0.8% | **자동 해제** |
| 김정문 | 148 | 0.15 h | 0.5% | |
| 이동백 | 47 | 0.05 h | 0.2% | |
| 김창룡 | 46 | 0.05 h | 0.2% | |

화자 편중이 극단적이다 — 상위 3명(unknown 제외 시 이일주·김소희·조상현)이 36%,
이동백·김창룡은 각각 3분 남짓이다.

---

### 중간 검문소 — 음소 매핑 채점기

`alignment/check_dict_agreement.py`. IPA→Coda 매핑표는 **손으로 만든 것**이라 고칠 때마다
추측이었다. 그래서 채점기부터 만들었다.

원리: `dictionaries/korean-coda.txt` 는 `build_dictionary.py` 의 자모 표에서 기계적으로
생성돼 **`ipa_to_coda.csv` 와 완전히 독립**이므로 정답지 역할을 할 수 있다. MFA TextGrid 의
원본 IPA 를 04와 같은 문맥 규칙으로 변환해 사전과 대조하면 변경의 옳고 그름이 숫자로 나온다.
모호성이 없는 **1음절 어절 89,303개**로 잰다.

이 하네스로 잡아낸 결함 3갈래:

| 결함 | 증상 | 조치 | 기호별 일치율 |
|---|---|---|---|
| 구개음화 활음 미반영 | 냐 → `n a` | `c cʰ cʰː c͈ ç ɲ ɟ ʝ ɕʰ ɕ͈` 에 ` y` 추가 | ç 22.8→**74.1** · ɲ 61.0→**84.5** · c 79.0→**96.2** |
| `ɸ → p` 오매핑 (ɸ 는 ㅍ 이 아니라 **ㅎ 의 원순모음 앞 변이**) | 후 → `p w u` | `ɸ→h`, `ɸʷ→h w`, `ʷ` 가 o/u 앞이면 활음 생략 | ɸ 0→**96.9** · ʷ 전체 **93.9** |
| `ʎ` 가 모든 위치에서 오답 | 빨리 → `pp a L i` | `L r\|L y` — 모음 사이면 앞 음절 종성 + 뒤 음절 초성 **두 토큰**, 자음 앞이면 종성 하나 | ʎ 0→**84.5** |

전체 일치율 **1음절 87.92 → 89.59%**, 2음절 75.01 → 76.43%.
이 수정 때문에 `convert_phone` 이 위치 대안에 **여러 토큰**을 담을 수 있게 됐다
(빨리 = `pp a` + `L r` + `i`). 활음 생략 규칙도 일반화됐다 — `ʲ` 는 `i` 앞에서, `ʷ` 는 `o`/`u`
앞에서 떨어진다. 후행 모음이 이미 그 조음을 갖고 있으면 활음이 아니라 동시조음이기 때문이다.

수정 반영: 26,955 항목 중 **3,109개(11.5%)** 의 `ph_seq` 가 바뀌었고 07은 PASS.
음소 인벤토리 34종 유지, OOV 0.

> **일치율 100% 는 목표가 아니다.** 남은 불일치의 다수는 정상적인 **연음**이다
> (은 → `eu n`, 좋 → `j o`). 문맥 없는 음절 사전으로는 재현할 수 없는 것이 맞다.
> 이 숫자는 절대 평가가 아니라 **변경 전후의 차분**으로 쓴다.

```sh
python pansori/alignment/check_dict_agreement.py                       # 1음절
python pansori/alignment/check_dict_agreement.py --syllables 2 --focus ʎ
```

---

## B. 학습 준비

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
```

### B1. 사전 생성 `build_dictionary.py`

한글 **11,172음절 전체** → Coda 34음소. `dictionaries/korean-coda.txt`.

옛 파이프라인에는 이 파일이 없었다 — 음소 인벤토리가 `CODA_36` 으로 하드코딩돼 있었고
**가사→음소 경로 자체가 없었다.** openvpi 는 `dictionaries:` 에 적힌 파일에서 음소셋을 만들고
(`utils/phoneme_utils.py`) 추론 때 같은 파일로 가사를 음소로 바꾼다. 즉 이 파일이 생기면서
**녹음 없는 새 노래를 부르게 하는 길이 열렸다.**

음절을 `[초성] [활음] 모음 [종성]` 으로 적는다 — 갸 → `g y a`, 의 → `eu i`, 왕 → `w a NG`.
생성 후 실제 코퍼스가 쓰는 34종과 대조해 불일치하면 **에러를 내고 멈춘다.**

한계는 명시해 뒀다: 음절 단위 매핑이라 **연음이 없다.** 학습 라벨은 MFA 산출이라 연음이
적용돼 있으므로 (책이 → `ch e g i`, `ch e K i` 아님) 추론 음소와 학습 음소가 일부 음절
경계에서 다르다. 음절 사전의 본질적 한계이고 upstream 도 같은 절충을 한다.

### B2. config 생성 `make_configs.py`

`configs/pansori/{acoustic,variance}.yaml` 을 **생성**한다 (직접 편집 금지 — 생성기를 고칠 것).
`datasets:` 블록이 08이 디스크에 실제로 쓴 것과 일치해야 하기 때문이다.

핵심 키:

| 키 | 값 | 이유 |
|---|---|---|
| `f0_min` / `f0_max` | **55 / 750** | 이 코퍼스 최저 노트가 MIDI 35(61.7 Hz). 기본값 65 Hz 면 남성 통성 저음이 통째로 무성 처리된다 |
| `merged_phoneme_groups` | `[[SP, AP]]` | `AP`(들숨)는 음소셋에 하드코딩돼 있고(`utils/phoneme_utils.py:18`) binarize 는 셋의 모든 항목이 코퍼스에 나타나야 통과시킨다. 우리 라벨에 들숨 구분이 없다 |
| `use_spk_id` / `num_spk` | true / 24 | 화자 23명 + 여유 1 |
| `pe` / `pe_ckpt` | `rmvpe` | binarize 때 f0 추출 |
| `max_batch_frames` | 50,000 (ac) / 80,000 (var) | |
| `pl_trainer_precision` | `16-mixed` | |
| 증강 | 전부 off | 판소리는 이미 음역이 넓고, key_shift/speed 임베딩이 붙으면 추론 시에도 그 값을 넘겨야 해 파이프라인이 하나 더 는다 |

**홀드아웃**은 곡 단위다 — `test_prefixes` 에 8자리 uuid 하나를 넣으면
`split_train_valid_set` 이 `startswith` 로 매칭해 곡 하나를 통째로 뺀다.
명창 6인에 하나씩 지정했는데, **홀드아웃 곡이 한 화자의 전부를 차지하면 생성기가 자동으로
해제한다.** 정권진(226발화)이 그 경우다 — 학습 데이터가 0이면 화자 임베딩에 gradient 가
한 번도 안 가는데 검증은 그 화자를 통과한다. 배울 수 없는 홀드아웃을 지키느니 화자를
학습시키는 쪽이 낫다.

### B3. binarize

```sh
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
```

결과: **acoustic 4.5 GB / variance 381 MB**, train **27.8 h** · valid **1.08 h**,
23화자 전원 학습 데이터 보유. 예상보다 작다 — 파형을 담지 않고, 44.1 kHz·hop 512 는
24 kHz·hop 128 보다 프레임이 적어서다 (구 파이프라인 16 GB 의 약 1/3).

> **acoustic binarize 도 HN-SEP 체크포인트를 요구한다.** breathiness/voicing/tension 을 전부
> 꺼도 `DecomposedWaveform` 은 무조건 생성되고 기본 `hnsep: vr` 이 `checkpoints/vr/` 를 찾는다.
> 선행 다운로드 3종: `checkpoints/rmvpe/`,
> `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`, `checkpoints/vr/`.

### B4. spec 통계 부트스트랩 — 순서가 있다

```sh
python pansori/compute_spec_stats.py     # → configs/pansori/spec_stats.yaml
python pansori/make_configs.py           # 위 파일을 base_config 에 얹어 재생성
```

openvpi 기본값 `spec_max: [0]` 은 이 코퍼스에 맞지 않는다. **실측 상한이 +2.838** 이라
그대로 두면 정규화 후 diffusion 타깃이 `[-1, 1]` 을 벗어난다. 하한은 반대로 안전하다 —
`nvSTFT` 가 `log(clamp(x, 1e-5))` 로 압축하므로 −11.513 아래가 존재할 수 없고 실측 하한은
−11.563(여유 0.05 포함)이다.

그래서 순서가 **기본값으로 한 번 binarize → 실측 → config 재생성 → 학습** 이다.
`spec_min`/`spec_max` 는 **학습 시점에만 쓰이므로 재-binarize 는 필요 없다.**

---

## C. 학습

```sh
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset
```

> 한 스텝 안에서 무슨 일이 벌어지는지 — 배치 구성부터 손실·옵티마이저·체크포인트 규칙까지는
> [`TRAINING.md`](TRAINING.md) 에 따로 적었다. 아래는 결과 요약이다.

둘 다 **scratch 학습**이다 (사전학습 체크포인트 미사용 — 공개 모델은 중국어 음소셋이라
phoneme embedding 이 통째로 어긋난다. config 에 `finetune_*` 자리는 만들어 뒀지만 `false`).

### 모델 두 개가 하는 일

| 모델 | 입력 | 출력 |
|---|---|---|
| **variance** | 가사(음소) + 노트 | 음소별 길이 `ph_dur` + f0 곡선 |
| **acoustic** | 음소 + `ph_dur` + f0 + 화자 | mel 스펙트로그램 (128빈) |
| **NSF-HiFiGAN** (학습 안 함) | mel | 파형 44.1 kHz |

둘 다 **reflow 기반 생성 모델**이고 euler 20 step 으로 샘플링한다. 이 사실이 뒤의 평가 전부를
지배한다 — 같은 입력이 매번 다른 답을 낸다.

### 학습 결과

| | 돌린 step | 최저 검증 | 마지막 | 상태 |
|---|---|---|---|---|
| acoustic | **100,000** (설정대로 완주, 123 epoch) | **0.17800 @ 28,000** | 0.18834 | 28k 이후 72k step 동안 0.178–0.203 진동만 |
| variance | **32,796** (중단, 40 epoch) | **0.08761 @ 20,000** | 0.10423 | `dur_loss` 3회 연속 상승해 중단 |

VRAM 은 24 GB 중 **6.1 GB** 만 썼다.

**acoustic 은 필요보다 3.5배 오래 돌았다.** 28,000 step 이 최선이었고 나머지 72,000 step 은
GPU 3시간을 태웠을 뿐이다. 마지막 체크포인트는 최선보다 **6% 나쁘다** — 학습 손실만 보고
있었으면 그걸 골랐을 것이다.

**더 위험한 것은 `num_ckpt_keep: 8` 이다.** 4,000 step 마다 저장하므로 최근 32,000 step 치만
남는다. 100,000 step 에서 멈췄을 때 **28,000 체크포인트는 이미 삭제돼 있었다.**
`checkpoints/best/` 에 따로 복사해 두지 않았다면 남은 것 중 최저인 76,000(0.18335)으로
조용히 내려앉았을 것이다.

### 체크포인트는 헤드마다 따로 고른다

variance 는 duration 과 pitch 를 함께 학습하는데 두 헤드의 검증 손실이 **서로 반대로 간다.**

| 지표 | 최저 step | 값 |
|---|---|---|
| acoustic `validation/total_loss` | **28,000** | 0.17800 |
| variance `validation/dur_loss` | **16,000** | 0.07797 — 이후 계속 상승 |
| variance `validation/pitch_loss` | **32,000** | 0.00861 — 중단 시점까지 하강 중 |
| variance `validation/total_loss` | 20,000 | dur 이 지배하는 절충값 |

`total_loss` 로 고른 20k 로 피치를 뽑으면 32k 보다 **f0 오차가 25% 크다.**
그래서 `run_examples.sh` 는 헤드별로 고르고 **variance 를 두 번 돌린다** —
1차 `--predict dur`, 2차는 그 산출물에 `--predict pitch` (이때 `ph_dur` 는 manual 로 읽힌다).
체크포인트 하나로 두 헤드를 다 만족시킬 수 없다면, 하나만 고를 이유도 없다.

도구가 이걸 지킨다:

- `examples/best_ckpt.py {exp} [--metric ...]` — 검증 손실 최저 step 을 고른다
- `checkpoints/best/{exp}_step{N}.ckpt` — 최적 체크포인트를 따로 보관 (1.6 GB, gitignore)
- `run_examples.sh` 가 실행 전에 실험 디렉토리로 **복원**한다 (이번에 실제로 발동했다)

현재 쓰는 조합: **acoustic 28,000 · variance dur 16,000 · variance pitch 32,000**
(`examples/out/CKPT_STEPS`).

---

## D. 추론

### D1. `.ds` 만들기 `make_ds.py`

openvpi 추론은 `.ds` 파일을 읽는데, 보통 OpenUtau 같은 에디터에서 나온다. 그걸 만드는 CLI 가
없고 판소리는 export 할 에디터 프로젝트도 없어서 양쪽 끝을 이 스크립트가 메운다.

**`from-label`** — 라벨에 있는 발화로 `.ds` 를 만든다. 무엇을 넣고 무엇을 뺄지로
**단계를 하나씩 분리**할 수 있다.

- `--with-dur` : 실제 `ph_dur` 주입
- `--with-f0` : 원본 wav 에서 f0 추출해 주입 (5 ms 그리드). binarizer 와 **같은 추출기·같은
  범위**를 쓰려고 config 를 읽어 `initialize_pe()` 로 뽑는다 — 기본값 65 Hz 면 남성 통성이 잘린다

`--with-dur` 없이 추론하려면 **`note_slur` 가 필요하다.** 노트를 단어에 묶는 정보인데
(`inference/ds_variance.py:193` — 0이 새 단어 시작, 1이 이어짐, 단어마다 정확히 하나가 0)
우리 라벨은 노트와 단어를 독립 축으로 두므로 그 묶음을 기록한 적이 없다. 그래서 실측 타이밍에서
**DP 로 복원한다** (`derive_note_slur`). 탐욕법이 아니라 DP 인 이유는 배정이 **단조 증가**여야
하고 모든 단어가 노트를 하나씩 받아야 하기 때문이다 — 탐욕법은 짧은 단어를 굶겨 조용히
망가진 `.ds` 를 만든다.

**`from-score`** — 녹음 없이 가사 + 멜로디만으로. 사전이 가사→음소를 맡는다.

```
_   rest  0.30      # '_' 는 쉼표 → SP
춘  A4    0.45
향  G4    0.40
-   F4    0.55      # '-' 는 앞 음절이 이어짐 (시김새·melisma)
```

`ph_seq`·`ph_num` 은 사전이, `note_slur` 는 `-` 표기가 그대로 만든다.
`examples/scores/` 에 `.txt` 를 넣으면 `run_examples.sh` 가 자동으로 3화자 합성한다.

> **`--spk` 는 한글 화자명을 못 받는다.** `utils/infer_utils.py:55` 가 `[0-9A-Za-z_-]+` 로
> 파싱한다. upstream 을 건드리지 않으려고 `make_ds.py` 가 `.ds` 의 `spk_mix` / `ph_spk_mix` 에
> 직접 쓴다 — `--spk` 가 채웠을 필드와 같은 dict 다.

### D2. 사다리 4단 `examples/run_examples.sh`

합성이 나빴을 때 **어느 단계 탓인지** 알 수 없으면 고칠 수가 없다. 그래서 홀드아웃 발화마다
실측값을 하나씩 빼고 예측값으로 바꾼 4단 사다리를 만든다.

| | 준 것 | 모델이 만든 것 |
|---|---|---|
| `1_orig` | — | — (원본 녹음, 상한) |
| `2_resynth` | `ph_dur` + `f0` | mel — **acoustic 단독 성능** |
| `3_pitch` | `ph_dur` | `f0` + mel — 피치 예측만 추가 |
| `4_full` | 노트 + 가사 | `ph_dur` + `f0` + mel — 전체 추론 |
| `5_score` | 악보 + 가사 (녹음 없음) | 전부 |

예시 발화 3개 — 남창/여창을 섞었다 (남성 통성은 `f0_min: 55` 없으면 통째로 무성 처리되는 구간):

- `d207244a_0085` 김소희 · 춘향가 이도령 이별 · 여창
- `8b4d0824_0078` 조상현 · 춘향가 그때의 어사또는 · 남창
- `e719dced_0197` 박동진 · 수궁가 · 남창

스크립트가 하는 일: `checkpoints/best/` 복원 → 헤드별 최적 체크포인트 선택 →
**시드 고정**(`SEED=20260818`) → 사다리 4단 × 3발화 → 악보 2개 × 3화자 → 스펙트로그램.

시드를 고정하지 않으면 같은 입력이 실행마다 다른 결과를 내고 표의 숫자가 흔들린다
(실측: f0 RMSE 가 회당 40~90 센트 움직인다).

---

## E. 평가

### E1. 사다리 점수 `examples/measure.py`

f0 RMSE 는 양쪽 다 유성으로 판정한 프레임에서 잰 **센트** 오차(100센트 = 반음),
mel L1 은 모델이 실제로 최적화하는 128빈 로그-mel 공간의 거리다. 둘 다 원본 기준, 낮을수록 가깝다.

| 발화 | 트랙 | f0 RMSE (cent) | mel L1 | 유무성 일치 |
|---|---|---:|---:|---:|
| 김소희 `d207244a_0085` | 2_resynth | **52.2** | 1.011 | 86.4% |
| | 3_pitch | 297.9 | 1.130 | 82.3% |
| | 4_full | 268.4 | 1.428 | 79.4% |
| 조상현 `8b4d0824_0078` | 2_resynth | **86.8** | 0.959 | 90.2% |
| | 3_pitch | 405.1 | 1.037 | 89.6% |
| | 4_full | 347.5 | 1.224 | 79.2% |
| 박동진 `e719dced_0197` | 2_resynth | **84.5** | 1.599 | 93.4% |
| | 3_pitch | 384.7 | 1.721 | 85.6% |
| | 4_full | 394.1 | 1.643 | 85.4% |

읽는 법: **재합성(실측 f0 주입)은 52–87센트**로 반음 안쪽이다 — acoustic + 보코더 경로는
제자리에 있다. 피치를 모델이 만들기 시작하면 268–405센트로 뛴다 — **현재 병목은 variance 의
피치 예측**이다. `4_full` 이 `3_pitch` 보다 나아 보이는 칸이 있는데, 아래 편차 폭을 보면
그 차이는 의미가 없다.

### E2. 시드 편차 `examples/seed_spread.py` — 숫자를 믿기 전에

**위 숫자들은 한 번의 추첨 결과다.** 두 모델 다 reflow 샘플러라 시드만 바꿔도
**f0 RMSE 가 ±17%, mel L1 이 ±8%** 움직인다. 트랙끼리 비교하려면 시드를 고정하거나
여러 번 뽑아 중앙값을 써야 한다. 표의 값 하나로 두 트랙의 우열을 가리기에는 폭이 너무 크다.

### E3. 시김새 평탄화 `examples/f0_report.py --seeds 5`

구 파이프라인의 진단은 **피치 평탄화**였다 — 결정론적 회귀에 L1 손실을 걸면 조건부 중앙값으로
수렴하고, 그러면 시김새(이 음악에서 표현의 대부분을 지고 있는 장식음)가 평균으로 뭉개진다는 것.

**그 논증은 새 모델에 그대로 옮겨오지 않는다.** openvpi 의 피치 예측기는 결정론적 회귀가 아니라
reflow 생성 모델이다. 그러니 가정할 게 아니라 다시 재야 했다.

재는 값: 프레임 사이 피치 이동량 `|Δf0|` 의 **중앙값을 원본 대비 비율**로. 음정은 맞히면서
그 사이의 움직임만 잃는 것이 평탄화의 서명이므로, 이 비율이 1보다 **일관되게** 낮으면 평탄화다.
"일관되게"가 중요하다 — 같은 발화가 시드에 따라 0.46 과 1.01 을 오갔다. 그래서 트랙마다
시드를 바꿔 5회 재생성하고 **중앙값과 범위**로 본다.

| 발화 | 트랙 | 진폭비 중앙값 | 5회 범위 | 1.0 포함 |
|---|---|---:|---|:--:|
| 조상현 | 실제 f0 | 0.859 | 0.810 – 0.875 | |
| | f0 예측 | 0.723 | 0.644 – 0.943 | |
| | dur·f0 예측 | 0.825 | 0.652 – 0.993 | |
| 김소희 | 실제 f0 | 0.871 | 0.810 – 0.951 | |
| | f0 예측 | 0.847 | 0.637 – 1.006 | ✓ |
| | dur·f0 예측 | 1.071 | 0.635 – 1.105 | ✓ |
| 박동진 | 실제 f0 | 0.977 | 0.914 – 1.040 | ✓ |
| | f0 예측 | 0.852 | 0.709 – 1.306 | ✓ |
| | dur·f0 예측 | 0.882 | 0.642 – 1.207 | ✓ |

**결론: 체계적 평탄화는 재현되지 않는다.** 예측 트랙 6개 중 4개는 범위가 1.0 을 가로지른다.
그리고 실측 f0 를 그대로 주입한 트랙조차 0.86–0.98 이므로 — **감쇠의 2–14% 는 acoustic 과
보코더에서 이미 생긴다.** 피치 예측기 탓이 아니다. contour tokenizer 를 붙일 근거는 적어도
이 측정에는 없다.

### E4. 스펙트로그램 `examples/spectrograms.py`

사다리 4단의 mel 을 한 장으로 그린다. 점수가 "왜" 나쁜지는 그림에서만 보인다.

### E5. 리포트 `examples/build_report.py`

설명 + 오디오를 한 페이지로 렌더링한다. 페이지가 주장하는 모든 수치는 손으로 적는 게 아니라
산출물에서 **다시 읽어 온다** — 코퍼스 표는 `data/pansori` 에서, 점수는 `measure.py`/`f0_report.py`
결과에서, 오디오는 `examples/out` 에서. 오디오는 base64 mp3 로 **박아 넣는다** (Artifact 의
CSP 가 외부 호스트를 전부 막아서 링크된 오디오는 재생되지 않는다).

발행본: <https://claude.ai/code/artifact/fe492821-71fa-4566-8b8d-459e6d619fc6>

---

## F. 전체 재현

```sh
# --- 전처리 (라벨만 고칠 때는 MFA 재실행 불필요) ---------------------------
cd pansori/preprocess
conda activate pansori && python 01_segment.py --out_dir ../data/full
python 02_speaker_map.py --set full
python 03_transcribe.py --set full
conda activate aligner
python 04_align.py --set full            # 매핑만 고쳤으면 --skip_align (~2분)
                                         # SOFA 로 돌리려면 --aligner sofa (README §3.1)
python 05_notes.py  --set full
conda activate pansori
python 06_merge.py    --set full
python 07_validate.py --set full         # PASS 확인 필수
python 08_export.py   --set full --out data/pansori   # 재커팅 ~20분

# --- 학습 준비 -------------------------------------------------------------
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
python pansori/build_dictionary.py
python pansori/make_configs.py
python scripts/binarize.py --config configs/exp/ac_v1.yaml
python scripts/binarize.py --config configs/exp/var_v1.yaml
python pansori/compute_spec_stats.py && python pansori/make_configs.py

# --- 학습 ------------------------------------------------------------------
python scripts/train.py --config configs/exp/ac_v1.yaml --exp_name pansori_ac_v1 --reset
python scripts/train.py --config configs/exp/var_v1.yaml --exp_name pansori_var_v1 --reset

# --- 추론 · 평가 · 리포트 ---------------------------------------------------
bash pansori/examples/run_examples.sh
python pansori/examples/measure.py
python pansori/examples/f0_report.py --seeds 5
python pansori/examples/build_report.py
```

---

## G. 밟으면 아픈 것들

**환경**
- `PYTHONNOUSERSITE=1` 없이 `dsp` 를 쓰면 tensorboard import 가 protobuf 에서 죽는다
- 전처리를 `base` env 로 돌리면 numba ↔ numpy 2.5 충돌로 06이 죽는다
- `requirements.txt` 에 `torchaudio` 가 없다 (RMVPE 가 쓴다)

**데이터**
- 발화는 **6초 이하** — 넘으면 MFA 정렬이 붕괴한다
- `labels_2.json` 의 경로 문자열은 NFC/NFD 가 섞여 있다 → **8자리 uuid 접두로 조인**
- 노트가 덮은 무음은 쉼이 아니라 **지속음**이다 (코퍼스 시간의 52.2%)

**config**
- `f0_min: 55` / `f0_max: 750` 을 빠뜨리면 남성 통성 저음이 통째로 무성 처리된다
- `spec_max` 기본값 `[0]` 을 그대로 두면 diffusion 타깃이 `[-1, 1]` 을 벗어난다 (실측 상한 +2.838)
- acoustic binarize 도 HN-SEP(`checkpoints/vr/`) 을 요구한다

**학습·추론**
- **가장 최근 체크포인트가 최선이 아니다.** 그리고 `num_ckpt_keep: 8` 은 최적을 지워 버린다
- variance 는 헤드마다 최적 step 이 다르다 → 두 번 돌린다
- **한 번 뽑은 결과로 모델을 판단하지 말 것** (f0 RMSE ±17%, mel L1 ±8%)
- `--spk` 는 한글 화자명을 못 받는다 → `.ds` 의 `spk_mix` 로 우회
- `git merge` / `git remote add` 금지 — 두 저장소는 공통 조상이 없다
- 구 체크포인트 3개는 재사용 불가 (24 kHz·80 mel·FastSpeech2)

---

## H. 아직 열려 있는 것

측정만 해두고 손대지 않은 것들 — 자세한 것은 [`NEXT.md`](NEXT.md).

| 이슈 | 근거 | 규모 |
|---|---|---|
| 사전의 ㅈ/ㅉ/ㅊ + 활음 | 죠는 [tɕo] 라 MFA 는 `j o`, 사전은 `j y o`. 죠/조가 동음이라 **사전 쪽** 문제 — 매핑에 `y` 를 붙이면 훨씬 흔한 조가 깨진다 | 96건+ |
| 요 → `y eo` vs 사전 `y o` | 모음 쪽 이슈, 미조사 | 156건 |
| ㄴ 이 `d` 로 (내/나/니/너) | MFA 오인식. 매핑으로는 못 고친다 | 700건+ |
| 무음 흡수 임계 | 50 ms 만 겹쳐도 무음 전체를 흡수. 커버율 50% 미만인 흡수 구간 | 6.2% (3,772개) |
| 들숨(AP) 라벨링 | 판소리는 들숨이 크게 들린다. 하려면 06에 에너지 기반 판정 + 재-binarize | — |
| 다음 학습 | `max_updates` 축소 또는 조기 종료, `num_ckpt_keep` 상향, 증강 검토 | — |
