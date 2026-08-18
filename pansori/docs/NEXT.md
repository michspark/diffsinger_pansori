# 남은 작업 — 다음 세션에 이 파일을 붙여넣으면 이어서 진행

브랜치 `feat/pansori`, 커밋 `8212d64`(포팅) + `3e8d541`(서버 러너).
경위는 `pansori/docs/PORTING.md`, 실행법은 `pansori/README.md` / `pansori/server/README.md`.

## 끝난 것

- 원본 저장소(`~/Desktop/DiffSinger_moonInRiver`) 미커밋 작업물 53개 파일 커밋 (`341e927`)
- `pansori/` 전체 이식 + `dictionaries/korean-coda.txt`(11,172음절 → Coda 34음소) 생성
- `04_align.py` 가 MFA words tier 도 기록하도록 확장 → MFA 재실행 없이 `ph_num` 유도
- `06_merge.py` / `07_validate.py` openvpi CSV 스키마로 재작성 → 26,955항목 **PASS**
- `convert_phone` 버그 3건 수정 (사전 일치율 77.3% → 87.9%)
- `08_export.py` 로 44.1 kHz 모노 재커팅 + 화자 23명 분할 → `data/pansori/` **8.7 GB**
- `configs/pansori_acoustic.yaml` / `pansori_variance.yaml` 생성 (`make_configs.py`)

## 해야 할 것

### A. 환경·자산 (서버, 선행 조건)
1. Python ≥ 3.10 env 생성 + `requirements.txt` 설치 — `bash pansori/server/run.sh setup`
   (기존 `diffsinger` 3.8 env 는 사용 불가. 어느 env 에도 `lightning` 이 없음)
2. `checkpoints/rmvpe/model.pt` 다운로드 — **binarize 필수**(f0 추출)
3. `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/` 다운로드 — 학습 검증용
4. 사전학습 acoustic/variance 체크포인트 확보 → `finetune_ckpt_path` 채우고
   `finetune_enabled: true`. **`finetune_ignored_params` 의 키를 실제 state_dict 와 대조할 것**
   (중국어 음소셋이라 phoneme embedding 은 반드시 제외). 없으면 scratch 로도 돈다.

### B. binarize
5. `bash pansori/server/run.sh binarize` — 단, 먼저 **1개 화자로 줄여 dry-run** 권장.
   통과 조건: 예외 없이 완료, `phone_set.json` 이 36항목(PAD+34+SP), train/valid 개수 정상,
   `test_prefixes` 경고 없음
6. `pansori/compute_spec_stats.py` 를 **128 mel / mel_base: e 기준으로 수정** 후 실행 →
   `spec_min`/`spec_max` 를 config 에 반영. (Opencpop 기본값이면 정규화 후 mel 이 +1 초과)

### C. 학습
7. `run.sh train-ac` / `run.sh train-var` — 100 step 스모크로 GPU 점유·loss 하강 먼저 확인
8. `scripts/infer.py acoustic <ds> --exp ...` 로 새 가사+노트 합성 확인
   (구 파이프라인에 없던 경로. 사전이 생겼으므로 이제 가능)

### D. 평가 (개작 필요)
9. `pansori/eval/resynth_ladder.py`, `f0_metrics.py` 가 아직 24 kHz/80 mel/구 추론 API 기준.
   44.1 kHz + NsfHifiGAN + `scripts/infer.py` 로 개작
10. `pansori/eval/midi_baseline.py` 를 새 variance 모델에 돌려 **f0 평탄화가 여전한지 재측정**.
    구 진단("결정론적 회귀 + L1 → 조건부 중앙값 → 평탄화")의 전제가 reflow 생성 모델에는
    그대로 적용되지 않는다. 여기 결과가 나오기 전엔 contour tokenizer 를 붙이지 말 것

### E. 미결 판단 (내가 손대지 않고 측정만 해둠 — `PORTING.md` §4)
11. `ipa_to_coda.csv` 잔여 결함: 구개음화 활음 미반영(냐 → `n a`, 222건), `ɸ → p`(후 → `p w u`,
    256건+), `ʎ` 의 `초성|종성` 순서 반대. 고치면 `04_align.py --skip_mfa` → `06` → `07` 재실행
    **2분**이면 반영됨 (MFA 재실행 불필요)
12. 사전의 ㅈ/ㅉ/ㅊ + 활음: 죠는 [tɕo] 라 MFA 는 `j o`, 사전은 `j y o`
13. 무음 흡수 임계: 50 ms 만 겹쳐도 무음 전체 흡수. 커버율 50% 미만이 6.2%(3,772구간)
14. `~/Desktop/DiffSinger_moonInRiver` 를 origin 에 push 할지 (로컬 커밋만 되어 있음)
15. 정리 대상: 구 24 kHz `wavs/`(5.5 GB), `data/binary/pansori*`(16 GB) — 검증 후

## 신경 쓸 것

- **`git merge` / `git remote add` 금지.** 두 저장소는 공통 조상이 없다
- **구 checkpoint 3개는 재사용 불가.** 24 kHz·80 mel·FastSpeech2 → 텐서가 하나도 안 맞는다
- **`f0_min: 55` / `f0_max: 750`** 을 빠뜨리면 남성 통성 저음이 통째로 무성 처리된다
- **디스크 여유 111 GB.** binarize 결과가 44.1 kHz·128 mel 이라 구 16 GB 의 2배 이상 예상
- **정권진(226발화)은 홀드아웃에서 자동 해제됨** — 곡 하나가 화자 전부라 학습 데이터가 0이 됐음
