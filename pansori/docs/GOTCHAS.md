# 알아둘 것 — 설정과 데이터의 함정

`README.md` 의 실행 절차를 따르다 막히면 여기부터 본다. 전부 실제로 한 번씩 밟은 것들이다.

---

- **화자는 `datasets:` 항목 단위로 지정된다.** 23명(명창 22 + pooled `unknown`)이 각각 자기
  디렉토리를 갖는다. `make_configs.py` 가 발화 수 내림차순으로 `spk_id` 를 매긴다.
- **홀드아웃 곡이 한 화자의 전부를 차지하면 자동 해제된다.** 정권진(226발화)이 그 경우라
  홀드아웃에서 빠졌다 — 학습 데이터 0인 화자는 임베딩이 아예 학습되지 않는다.
- **`f0_min: 55` / `f0_max: 750` 을 빠뜨리지 말 것.** 기본값 65 Hz 는 남성 통성을 잘라낸다.
- **`spec_max` 기본값 `[0]` 을 그대로 두지 말 것.** 이 코퍼스 실측 상한은 **+2.838** 이라
  정규화 후 diffusion 타깃이 [-1, 1] 을 벗어난다. 하한은 반대로 안전하다 — `nvSTFT` 가
  `log(clamp(x, 1e-5))` 로 압축해 -11.513 아래가 존재할 수 없다.
- **`AP` 는 `SP` 와 합쳐 뒀다.** openvpi 는 음소셋에 `AP`(들숨) 를 항상 넣고 binarize 는 셋의
  모든 항목이 코퍼스에 나타나야 통과시키는데, 우리 라벨에는 들숨 구분이 없다. 들숨을 따로
  라벨하게 되면 `make_configs.py` 의 `merged_phoneme_groups` 를 지우고 재-binarize 할 것.
- **가장 최근 체크포인트가 최선이 아니다.** 구 파이프라인에서는 56k step 부근에서 과적합이
  시작됐다. 아키텍처가 달라졌으니 수치는 이월되지 않지만 습관은 유지할 것.

---

## 학습·추론

- **선행 다운로드가 셋이다** (`server/run.sh setup` 이 받는다): `checkpoints/rmvpe/`,
  `checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/`, `checkpoints/vr/`.
  마지막 것은 HN-SEP 인데, breathiness 등을 전부 꺼도 acoustic binarize 가
  `DecomposedWaveform` 을 무조건 만들기 때문에 없으면 거기서 멈춘다.
- **wandb 는 한 번 삭제한 run id 를 다시 쓰지 못한다.** 잘못 올려서 지웠다면 `--run-id` 로
  새 id 를 주고 표시 이름만 `--exp` 로 남길 것.
- **체크포인트는 헤드마다 다르다.** variance 의 `dur_loss` 와 `pitch_loss` 는 서로 다른
  step 에서 최저다. `run_examples.sh` 가 헤드별로 골라 variance 를 두 번 돌린다.
- **`--spk` 는 한글 화자명을 못 받는다** (`utils/infer_utils.py:55`). `make_ds.py` 가
  `.ds` 의 `spk_mix` 에 직접 써서 우회한다.
