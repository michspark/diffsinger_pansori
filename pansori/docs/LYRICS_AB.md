# 가사 A/B — 창본 보정을 하고 / 안 하고

두 라벨로 각각 학습시켜 합성음을 비교한 실험의 **절차와 결과**다. 판정의 근거와 한계는
[`../lyricbook/VERDICT.md`](../lyricbook/VERDICT.md), 실행법 요약은
[`../README.md`](../README.md).

---

`03b_snap_lyrics.py` 는 Whisper 전사를 창본(`lyricbook/corpus.jsonl`)에 맞춰 교정한다.
[`VERDICT.md`](../lyricbook/VERDICT.md) 는 **MFA 정렬 지표로는** 이 교정이 오히려
나쁘다고 판정했지만, 같은 문서 §4 가 그 판정의 한계를 명시한다 — MFA 는 오디오에서 파생된
텍스트를 구조적으로 편애하므로 "어느 가사가 **학습에** 좋은가"는 답할 수 없다. 그 질문은
두 라벨로 각각 학습시켜 합성음을 비교해야 답이 나온다. 이 절이 그 절차다.

**바꾸는 것은 라벨 하나뿐이다.** 오디오·노트·홀드아웃·spec 통계·config·학습 설정이 전부
동일해야 차이를 가사 탓으로 돌릴 수 있다.

## 1. snapped 코퍼스 만들기

```sh
conda activate pansori
python pansori/preprocess/01b_recut_wavs.py --set full        # 04 의 입력 복원 (README §3)

conda activate aligner && cd pansori/preprocess
python 04_align.py --set full --lyrics lyrics_snapped.jsonl --suffix _snap   # 수 시간

# 06·07 은 data/{set}/ 의 고정된 파일명을 읽는다. 새 --set 을 링크로 조립하면
# 스크립트를 고치지 않고 그대로 돌릴 수 있다.
cd ../data && mkdir -p full_snap && cd full_snap
ln -s ../full/{segments.jsonl,notes.jsonl,spk_map.json,wavs} .
ln -s ../full/phones_snap.jsonl phones.jsonl                  # 다른 것은 이것뿐이다

conda activate pansori && cd ../../preprocess
python 06_merge.py    --set full_snap
python 07_validate.py --set full_snap                         # PASS 없이 진행 금지
python 08_export.py   --set full_snap --out data/pansori_snap --wavs_from data/pansori
```

`--wavs_from` 은 화자별 `wavs/` 를 기존 코퍼스로 심볼릭 링크한다. 두 변종의 오디오는
바이트 단위로 같으므로 14 GB 를 다시 자를 이유가 없고, 같은 파일을 가리키는 편이
"라벨 말고는 전부 같다"를 보장한다.

## 2. config 와 binarize

```sh
conda activate dsp && export PYTHONPATH=. PYTHONNOUSERSITE=1
python pansori/make_configs.py --data data/pansori_snap --name pansori_snap
python scripts/binarize.py --config configs/pansori_snap_acoustic.yaml
python scripts/binarize.py --config configs/pansori_snap_variance.yaml
```

**`compute_spec_stats.py` 를 다시 돌리지 않는다.** mel 통계는 오디오에서만 나오고 오디오는
동일하다. `configs/pansori_spec_stats.yaml` 을 그대로 재사용하는 것이 오히려 조건을
묶어 준다.

## 3. 학습 — 두 런을 한 group 으로

```sh
python pansori/train.py --config configs/pansori_snap_acoustic.yaml \
  --exp_name pansori_ac_snap --reset \
  --hparams max_updates=50000,num_ckpt_keep=12 \
  --lyrics-source snapped --wandb-group lyrics-ab

python pansori/train.py --config configs/pansori_snap_variance.yaml \
  --exp_name pansori_var_snap --reset \
  --hparams max_updates=32000,num_ckpt_keep=8 \
  --lyrics-source snapped --wandb-group lyrics-ab
```

- **스텝 수는 baseline 에 맞춘 것이다.** `pansori_var_v1` 은 config 의 80k 가 아니라 실제로
  **32k 에서 멈춰 있다**(체크포인트·tfevents 둘 다). 더 오래 돌리면 snapped 쪽 피치 헤드만
  더 학습돼 비교가 기운다. acoustic 은 baseline 이 100k 까지 갔지만 최적이 28k 였으므로
  50k 로 충분하다.
- **`num_ckpt_keep` 을 올린다.** 기본값(acoustic 8 / variance 5)이면 초반 체크포인트가
  지워지는데, 실측 최적은 acoustic 28k · dur 16k · pitch 32k 로 전부 초반이다.

## 4. 평가 — 같은 자로 두 번

`run_examples.sh` 는 환경변수로 모델과 코퍼스를 받는다 (README §5).

```sh
bash pansori/examples/run_examples.sh                          # whisper
python pansori/examples/measure.py --json pansori/examples/out/metrics.json

AC_EXP=pansori_ac_snap VAR_EXP=pansori_var_snap DATA=data/pansori_snap \
DS_DIR=pansori/examples/ds_snap OUT_DIR=pansori/examples/out_snap \
  bash pansori/examples/run_examples.sh
python pansori/examples/measure.py --out pansori/examples/out_snap \
  --json pansori/examples/out_snap/metrics.json
```

`SEED` 를 두 실행에서 같게 둘 것. reflow 는 확률적 샘플러라 시드만 바꿔도 f0 RMSE 가
±17% 움직인다 — **두 모델의 차이가 `seed_spread.py` 의 편차보다 작으면 차이가 없는 것이다.**

## 5. 결과 (2026-08-29 실행분)

**보정은 도움이 되지 않았다.** 홀드아웃 60쌍 × 3시드를 `2_resynth` 단에서 재면, 분리되는
지표는 하나뿐이고 그것이 **가사가 실제로 바뀐 발화에서만** 나온다 — 유무성 일치가
whisper 21승 6패 3무(p = 0.006), 대조군(음소열이 동일한 30쌍)에서는 10승 12패 8무
(p = 0.83). 전체 수치와 해석은 [`VERDICT.md`](../lyricbook/VERDICT.md) §9.

```sh
python pansori/examples/ab_ladder.py --n 60 --seeds 3      # 두 팔을 한 번에, ~5분
```

`ab_ladder.py` 는 팔마다 모델을 **한 번만** 올려 60항목을 렌더한다 (`run_examples.sh` 는
렌더마다 프로세스를 새로 띄워 25초씩 쓴다). 절반은 두 코퍼스에서 음소열이 글자 하나
다르지 않은 **대조군**이다 — 같은 모델 두 개로 재므로 학습 런 사이의 우연한 차이는 두
군에 똑같이 나타나고, 처치군에서만 나오는 차이라야 가사 탓이다.

## 6. 읽을 때 주의

- **두 런의 검증 손실을 높낮이로 비교하면 안 된다.** 각자 자기 라벨에 대해 계산된 값이다.
  wandb 에 나란히 보이는 것은 수렴 양상과 과적합 시점을 보기 위한 것이고, 우열 판정은
  §4 의 합성음 지표로만 한다.
- 두 변종에서 07 을 통과한 아이템 집합이 완전히 같지는 않다. 차이를 세어 명시할 것.
- 청취 검증은 여전히 없다. 이 실험은 "어느 라벨이 더 나은 합성음을 만드는가"만 답한다.

