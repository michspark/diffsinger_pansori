#!/usr/bin/env python3
"""Render the generated layers of configs/pansori/.

The old generator wrote one 191-line file per (corpus, kind). Of those lines 122
were the `datasets:` block, and the whisper/snapped pair differed in nothing but
a path -- so a second corpus cost a second 191-line copy, and a run's actual
conditions (max_updates, num_ckpt_keep) lived only in a `--hparams` string in
somebody's shell history.

So the config is split along the three axes that actually vary, and upstream's
own `base_config` chain puts them back together (utils/hparams.py:64-82 loads the
list in order, then lets the file's own keys win):

    configs/pansori/{acoustic,variance}.yaml   kind. 코퍼스를 모른다.
    configs/pansori/data_{name}.yaml           코퍼스. datasets + num_spk.
    configs/exp/*.yaml                         실험. 손으로 쓴다, git 에 남는다.

`binary_data_dir` is the one key that depends on both axes, so the experiment
file states it. That is not a workaround -- which binarized corpus a run reads is
the single most important fact about it, and it belongs where a reader looks.

    python pansori/make_configs.py                                   # data/pansori
    python pansori/make_configs.py --data data/pansori_snap --name snap
"""

import argparse
import csv
from pathlib import Path

# One held-out song per 명창, carried over from the MoonInTheRiver run so the
# test set stays directly comparable with the numbers in docs/PORTING.md.
# split_train_valid_set matches with startswith and item names are
# {uuid8}_{index}, so a bare uuid holds out a whole song.
HOLDOUT = {
    '조상현': '8b4d0824',   # 춘향가 그때의 어사또는
    '정권진': 'a580300d',   # 심청가 뺑파
    '이일주': 'c235c995',   # 심청가 심청 하직
    '박동진': 'e719dced',   # 수궁가
    '박초월': 'd7ec123c',   # 수궁가 토끼 수궁들어가는 대목
    '김소희': 'd207244a',   # 춘향가 이도령 이별
}

# 이 코퍼스의 mel 포락선. openvpi 기본값 spec_max:[0] 은 판소리에 맞지 않는다 —
# 실측 상한이 +2.84 라 정규화 후 diffusion 타깃이 [-1, 1] 밖으로 나간다.
# pansori/compute_spec_stats.py 가 binarize 결과에서 생성한다. 부트스트랩 순서는
# binarize(기본값) → compute_spec_stats.py → make_configs.py → 학습 이고,
# spec_min/max 는 학습 시점에만 쓰이므로 재-binarize 는 필요 없다.
# variance 는 mel 을 만들지 않으므로 acoustic 에만 붙인다.
EXTRA_BASE = {'acoustic': '\n  - configs/pansori/spec_stats.yaml'}

KIND_HEADER = """# 판소리 fine-tune — {kind} 공통층 (생성물)
# python pansori/make_configs.py 로 생성한다. 직접 편집하지 말고 생성기를 고칠 것.
#
# 이 파일은 **코퍼스를 모른다.** 실험 파일이 data_*.yaml 과 함께 base_config 에 얹고
# binary_data_dir 을 정한다 — configs/exp/ 참조.

base_config:
  - configs/{kind}.yaml{extra_base}

dictionaries:
  ko: dictionaries/korean-coda.txt
extra_phonemes: []

num_lang: 1
use_lang_id: false

binarization_args:
  num_workers: 4
use_spk_id: true

# 판소리 음역. 남성 통성은 ROSVOT 노트 기준 MIDI 35 = 61.7 Hz 까지 내려가므로
# 기본값 65 Hz 면 저음이 통째로 무성 처리된다. 측정 상한은 740 Hz.
f0_min: 55
f0_max: 750
pe: rmvpe
pe_ckpt: checkpoints/rmvpe/model.pt

# ---- fine-tune ------------------------------------------------------------
# 공개 사전학습 체크포인트를 받은 뒤 아래 두 줄을 켠다. 사전학습 모델은 중국어
# 음소셋이라 phoneme embedding 이 통째로 어긋나므로 반드시 무시 목록에 넣는다.
# 실제 state_dict 키는 체크포인트마다 다르므로 받은 뒤 대조할 것.
finetune_enabled: false
finetune_ckpt_path: null
finetune_ignored_params:
{ignored}
finetune_strict_shapes: false
"""

ACOUSTIC = """
# 첫 학습은 증강 없이 간다. 판소리는 이미 음역이 넓고, key_shift/speed 임베딩이
# 붙으면 추론 시에도 그 값을 넘겨야 해서 파이프라인이 하나 더 늘어난다.
use_key_shift_embed: false
use_speed_embed: false
augmentation_args:
  random_pitch_shifting:
    enabled: false
  fixed_pitch_shifting:
    enabled: false
  random_time_stretching:
    enabled: false

# spec_min/spec_max 는 위 base_config 의 spec_stats.yaml 에서 온다.
# 코퍼스를 다시 만들면 compute_spec_stats.py 로 다시 재고 이 스크립트를 다시 돌릴 것.

max_batch_frames: 50000
max_batch_size: 32
pl_trainer_precision: '16-mixed'
"""

VARIANCE = """
# upstream VarianceTask + 부류별 음소 길이 지표. upstream 파일은 그대로 두고
# task_cls 만 바꿔 끼운다 — 근거와 무엇이 더 붙는지는 pansori/variance_task.py 머리말.
# 기존 rhythm_corr / ph_dur_acc 도 그대로 나온다 (이전 런과의 비교선 유지).
task_cls: pansori.variance_task.PansoriVarianceTask

# 새 지표를 몇 발화까지 재는가. 0 = 검증셋 전체.
# upstream 의 rhythm_corr / ph_dur_acc 는 num_valid_plots(10) 에 묶여 있어 앞 10발화만
# 채점한다. duration 추론은 conv 하나라 사실상 공짜이므로 전체로 돌린다.
num_valid_metric_items: 0

predict_dur: true
predict_pitch: true
predict_energy: false
predict_breathiness: false
predict_voicing: false
predict_tension: false
use_glide_embed: false

max_batch_frames: 80000
max_batch_size: 32
pl_trainer_precision: '16-mixed'
"""

DATA_HEADER = """# 판소리 코퍼스 — {data} (생성물)
# python pansori/make_configs.py --data {data} --name {name} 로 생성한다.
#
# 화자 목록은 08_export.py 가 실제로 디스크에 쓴 것과 일치해야 하므로 손으로
# 관리하지 않는다. 여기에는 **코퍼스에만 달린 것**을 둔다 — kind 마다 다른
# binary_data_dir 은 실험 파일이 정한다 (configs/exp/).
"""

IGNORED = ['model.fs2.encoder.embed_tokens', 'model.fs2.txt_embed', 'model.fs2.spk_embed']

# openvpi 는 'SP'(무음) 와 'AP'(들숨) 를 음소셋에 항상 넣고
# (utils/phoneme_utils.py:18), binarize 는 셋의 모든 항목이 코퍼스에 실제로
# 나타나야 통과시킨다. 그래서 이 블록은 코퍼스가 정한다, 사람이 정하지 않는다 —
# 라벨에 AP 가 없는데 음소셋에 남겨 두면 binarize 가 죽고, 있는데 합쳐 두면
# 실제로 라벨한 들숨을 도로 뭉갠다. 어느 쪽인지는 transcriptions.csv 안에만 있다.
MERGED_SP_AP = """
# 이 코퍼스의 라벨에는 들숨 구분이 없다 (MFA 정렬) — 06_merge.py 는 노트가 덮지 않은
# 무음을 전부 SP 로 만든다. 그래서 둘을 한 인덱스로 합친다. 없는 구분을 지어내는 것보다
# 라벨이 말하는 대로 두는 쪽이 맞다.
merged_phoneme_groups:
  - [SP, AP]
"""

SPLIT_SP_AP = """
# 이 코퍼스는 들숨(AP) 을 실제로 라벨했다 (SOFA 정렬). 그래서 SP 와 합치지 않는다 —
# 합치면 방금 라벨한 구분을 도로 버리는 셈이다. upstream 기본값이
# `merged_phoneme_groups: []` (configs/acoustic.yaml:8) 이므로 아무것도 쓰지 않으면 된다.
"""


def corpus_has_ap(data_root: Path) -> bool:
    """이 코퍼스의 ph_seq 에 AP 토큰이 실제로 있는가.

    binarize 가 요구하는 조건 자체를 그대로 묻는다. datasets_block 이 어차피 같은
    CSV 들을 읽지만, 그쪽은 발화 수를 세고 이쪽은 음소를 보므로 따로 돈다.
    """
    for path in sorted(p for p in data_root.iterdir() if p.is_dir()):
        csv_path = path / 'transcriptions.csv'
        if not csv_path.exists():
            continue
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if 'AP' in row['ph_seq'].split():
                    return True
    return False


def datasets_block(data_root: Path, rel: str) -> tuple:
    speakers = []
    for path in sorted(p for p in data_root.iterdir() if p.is_dir() and (p / 'transcriptions.csv').exists()):
        with open(path / 'transcriptions.csv', encoding='utf-8') as f:
            names = [row['name'] for row in csv.DictReader(f)]
        held = sum(1 for n in names if n.startswith(HOLDOUT.get(path.name, '\0')))
        speakers.append((path.name, len(names), held))
    speakers.sort(key=lambda x: -x[1])          # spk_id 0 = 발화 최다 화자

    lines, dropped = ['', 'datasets:'], []
    for spk_id, (name, count, held) in enumerate(speakers):
        lines += [f'  - raw_data_dir: {rel}/{name}',
                  f'    speaker: {name}',
                  f'    spk_id: {spk_id}',
                  '    language: ko']
        # A holdout that swallows a speaker's entire dataset leaves that speaker
        # with no training items at all: the embedding never gets a gradient, yet
        # validation still runs through it. Better to train the speaker and give
        # up its holdout than to keep a hold-out we cannot learn from.
        if name in HOLDOUT and held < count:
            lines += ['    test_prefixes:', f"      - '{HOLDOUT[name]}'"]
        else:
            if name in HOLDOUT:
                dropped.append((name, count))
            lines += ['    test_prefixes: []']
        lines[-1] += f'      # 발화 {count:,}개'
    return '\n'.join(lines) + '\n', len(speakers), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/pansori')
    parser.add_argument('--name', default='pansori',
                        help='코퍼스 층의 이름 → configs/pansori/data_{name}.yaml')
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / 'configs' / 'pansori'
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = repo / args.data
    has_ap = corpus_has_ap(data_root)
    merged = SPLIT_SP_AP if has_ap else MERGED_SP_AP

    ignored = '\n'.join(f'  - {p}' for p in IGNORED)
    for kind, tail in (('acoustic', ACOUSTIC), ('variance', VARIANCE)):
        out = out_dir / f'{kind}.yaml'
        out.write_text(
            KIND_HEADER.format(kind=kind, extra_base=EXTRA_BASE.get(kind, ''),
                               ignored=ignored) + tail,
            encoding='utf-8')
        print(f'{out.relative_to(repo)}')
    print('들숨(AP) 라벨 있음 → merged_phoneme_groups 없음 (SP/AP 분리)' if has_ap
          else '들숨(AP) 라벨 없음 → merged_phoneme_groups: [[SP, AP]]')

    block, n_spk, dropped = datasets_block(data_root, args.data)
    out = out_dir / f'data_{args.name}.yaml'
    out.write_text(
        DATA_HEADER.format(data=args.data, name=args.name) + block
        + f'\n# 화자 {n_spk}명 (명창 + pooled unknown). num_spk 는 최대 spk_id 보다 커야 한다.\n'
        + f'num_spk: {n_spk + 1}\n' + merged,
        encoding='utf-8')
    print(f'{out.relative_to(repo)}  (화자 {n_spk}명)')

    for name, count in dropped:
        print(f'⚠ {name}: 홀드아웃 곡이 전체 {count:,}발화를 모두 차지 → 홀드아웃 해제 '
              f'(학습 데이터 0이면 화자 임베딩이 학습되지 않음)')
    missing = [s for s in HOLDOUT if not (data_root / s).exists()]
    if missing:
        print(f'⚠ 홀드아웃 지정 화자가 데이터에 없음: {missing}')


if __name__ == '__main__':
    main()
