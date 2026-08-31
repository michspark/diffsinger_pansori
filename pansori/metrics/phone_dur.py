#!/usr/bin/env python3
"""음소 길이 예측을 **부류별로** 재는 검증 지표.

upstream 은 이미 `modules/metrics/duration.py` 에 `PhonemeDurationAccuracy` 를 갖고
있고 `training/variance_task.py:136` 에서 그것을 등록한다. 그런데 그 지표는 코퍼스
전체를 하나의 숫자로 접는다 — 그리고 이 코퍼스에서 볼 만한 것은 정확히 그 접힘 안에
있다:

    초성 ≤3프레임  40.9%          종성 ≤3프레임  15.1%
    (pp 48 / tt 46 / b 42 / g 42)  (N 8.5 / NG 12 / M 13)

초성과 종성을 '자음' 하나로 묶으면 서로를 상쇄해 아무것도 안 보이고, 모음까지 섞으면
더 안 보인다. 그래서 여기서는 `pansori_lib.phone_class` 의 다섯 부류를 각각 따로 잰다.
(`pansori/eval/label_report.py` 가 같은 분류로 라벨 쪽을 재므로 두 수치가 직접 대응한다.)

**단위는 프레임이다.** binarize 가 ph_dur 을 mel 프레임으로 양자화한 뒤에야 모델이
보므로(`preprocessing/variance_binarizer.py:293-295`), 여기서 초로 되돌리는 것은
없는 정밀도를 지어내는 일이다. 하한선도 프레임으로 잡는다 — hop 512 / 44.1 kHz 에서
한 프레임이 11.61 ms 이고 MFA 의 30 ms 하한선은 3프레임(34.8 ms)에 걸린다. 그래서
`floor_rate_gt` 는 label_report 의 '≤30ms' 열이 아니라 '≤3프레임' 열과 맞춰 봐야 한다.

예측 길이는 재기 전에 `RhythmRegulator` 로 GT 단어 길이에 맞춘다 — 추론이 실제로
그렇게 하고(`modules/toplevel.py:262`), 단어 길이는 악보가 주는 값이지 모델이 푸는
문제가 아니기 때문이다. 모델이 푸는 문제는 **단어 안에서의 배분**이고, 그것만 재야 한다.
upstream `PhonemeDurationAccuracy` 와 같은 처리라서 두 수치가 서로 비교된다.

이 파일은 upstream 을 한 줄도 건드리지 않는다. 등록은 `pansori/variance_task.py` 의
서브클래스가 한다.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from torch import Tensor

matplotlib.rcParams['axes.unicode_minus'] = False

# 부류 이름은 한글이라 CJK 글리프가 있는 폰트가 필요하다. 없으면 tofu 상자가 찍히고
# wandb 에서 그림이 못 읽게 되므로, 폰트를 못 찾으면 라벨을 로마자로 떨어뜨린다.
# Noto Sans CJK 는 JP/KR 어느 쪽 이름으로 등록돼 있든 한글 글리프를 갖고 있다.
_CJK_CANDIDATES = ('Noto Sans CJK KR', 'Noto Sans CJK JP', 'NanumGothic',
                   'Malgun Gothic', 'AppleGothic', 'UnDotum')
_ASCII_CLASS = {'onset': 'onset', 'coda': 'coda', 'vowel': 'vowel',
                'glide': 'glide', 'sp': 'silence'}


def _cjk_font():
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


def build_token_class(phoneme_dictionary, classes) -> Tensor:
    """토큰 id -> 부류 index 룩업. 사전 하나당 한 번만 만든다.

    `PAD_INDEX` (0) 와 분류 불가 토큰은 -1 로 두어 어떤 부류에도 안 들어가게 한다.
    병합된 음소(SP/AP 처럼 한 id 에 여러 이름)는 `decode_one(scalar=False)` 가 튜플을
    주므로 첫 이름으로 분류한다 — 병합된 것들은 정의상 같은 부류다.
    """
    from pansori_lib import phone_class

    index = {name: i for i, name in enumerate(classes)}
    lookup = torch.full((phoneme_dictionary.vocab_size,), -1, dtype=torch.long)
    for token in range(1, phoneme_dictionary.vocab_size):
        phone = phoneme_dictionary.decode_one(token, scalar=False)
        if not isinstance(phone, str):
            phone = phone[0]
        lookup[token] = index.get(phone_class(phone), -1)
    return lookup


class _TokenClassMetric(torchmetrics.Metric):
    """토큰 부류로 걸러 재는 지표들의 공통 뼈대.

    `token_class` 는 버퍼라 `.to(device)` 를 따라간다. `cls_id=None` 이면 마스크
    안의 모든 음소를 쓴다 (부류 구분 없는 전체 지표).
    """

    def __init__(self, token_class: Tensor, cls_id=None, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('token_class', token_class.clone(), persistent=False)
        self.cls_id = cls_id

    def _select(self, tokens: Tensor, mask: Tensor) -> Tensor:
        if self.cls_id is None:
            return mask
        return mask & (self.token_class[tokens] == self.cls_id)


class PhoneClassDurationAccuracy(_TokenClassMetric):
    """한 부류 안에서 |pred - target| <= tolerance * target 인 음소의 비율.

    upstream `PhonemeDurationAccuracy` 와 같은 정의이고 tolerance 도 같은 0.2 를 쓴다.
    다른 것은 부류로 거른다는 점 하나뿐이라, 부류별 값들이 그 전체 값을 분해한 것이 된다.

    target 이 0 인 음소는 상대 오차가 정의되지 않으므로 제외한다 (프레임 양자화가
    0프레임 음소를 만들 수 있다).
    """

    def __init__(self, *, tolerance, token_class, cls_id=None, **kwargs):
        super().__init__(token_class, cls_id, **kwargs)
        self.tolerance = tolerance
        self.add_state('accurate', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor, tokens: Tensor, mask: Tensor) -> None:
        sel = self._select(tokens, mask) & (target > 0)
        if not sel.any():
            return
        accurate = (torch.abs(pred - target) <= target * self.tolerance) & sel
        self.accurate += accurate.sum()
        self.total += sel.sum()

    def compute(self) -> Tensor:
        return self.accurate / self.total.clamp(min=1)


class PhoneClassDurationMAE(_TokenClassMetric):
    """한 부류의 평균 절대오차 (ms).

    정확도만으로는 부족하다 — 초성은 대부분 2~4프레임이라 tolerance 20% 가 한 프레임도
    허용하지 않는 구간이 있고, 그러면 정확도가 바닥에 붙어 움직이지 않는다. ms 오차는
    그 아래에서도 계속 움직인다.
    """

    def __init__(self, *, token_class, timestep_ms, cls_id=None, **kwargs):
        super().__init__(token_class, cls_id, **kwargs)
        self.timestep_ms = timestep_ms
        self.add_state('error', default=torch.tensor(0., dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor, tokens: Tensor, mask: Tensor) -> None:
        sel = self._select(tokens, mask)
        if not sel.any():
            return
        self.error += torch.abs(pred.float() - target.float())[sel].sum()
        self.total += sel.sum()

    def compute(self) -> Tensor:
        return self.error / self.total.clamp(min=1) * self.timestep_ms


class DurationFloorRate(_TokenClassMetric):
    """길이가 하한선 이하인 음소의 비율. `on='target'` 이면 라벨을, `'pred'` 면 예측을 잰다.

    두 곡선을 나란히 보는 것이 요점이다. 예측이 라벨을 그대로 따라가면 모델이 라벨의
    병을 배운 것이고, 크게 낮으면 모델이 라벨과 다르게 굴고 있다는 뜻이다. 어느 쪽이
    옳은지는 이 지표가 답하지 않는다 — 그건 `pansori/eval/intelligibility.py` 의 일이다.
    """

    def __init__(self, *, floor_frames, token_class, on='pred', cls_id=None, **kwargs):
        super().__init__(token_class, cls_id, **kwargs)
        assert on in ('pred', 'target')
        self.floor_frames = floor_frames
        self.on = on
        self.add_state('floored', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor, tokens: Tensor, mask: Tensor) -> None:
        sel = self._select(tokens, mask)
        if not sel.any():
            return
        value = pred if self.on == 'pred' else target
        self.floored += ((value <= self.floor_frames) & sel).sum()
        self.total += sel.sum()

    def compute(self) -> Tensor:
        return self.floored / self.total.clamp(min=1)


class PerPhonemeDurationStats(torchmetrics.Metric):
    """음소 하나하나의 정확도를 쌓아 둔다. 스칼라로는 매크로 평균을 낸다.

    매크로 평균인 이유: 마이크로 평균(=upstream 의 `ph_dur_acc`)은 `a`, `i`, `o` 세
    모음이 전체의 25% 라 그쪽에 끌려간다. `pp` 345개가 통째로 무너져도 마이크로에서는
    0.1%p 도 안 움직인다. 두 평균을 같이 보면 "흔한 음소만 맞히고 있는가"가 보인다.

    막대그래프의 재료는 `per_phone()` 이 준다.
    """

    def __init__(self, *, tolerance, vocab_size, min_count=20, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.add_state('accurate', default=torch.zeros(vocab_size, dtype=torch.long),
                       dist_reduce_fx='sum')
        self.add_state('floored', default=torch.zeros(vocab_size, dtype=torch.long),
                       dist_reduce_fx='sum')
        self.add_state('count', default=torch.zeros(vocab_size, dtype=torch.long),
                       dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor, tokens: Tensor, mask: Tensor,
               floor_frames: int) -> None:
        sel = mask & (target > 0)
        if not sel.any():
            return
        flat_tokens = tokens[sel]
        accurate = (torch.abs(pred - target) <= target * self.tolerance)[sel]
        floored = (pred <= floor_frames)[sel]
        ones = torch.ones_like(flat_tokens)
        self.count.scatter_add_(0, flat_tokens, ones)
        self.accurate.scatter_add_(0, flat_tokens, accurate.long())
        self.floored.scatter_add_(0, flat_tokens, floored.long())

    def per_phone(self):
        """(token_ids, accuracy, floor_rate, count) — 표본이 min_count 이상인 음소만."""
        count = self.count.cpu().numpy()
        keep = np.flatnonzero(count >= self.min_count)
        if keep.size == 0:
            return keep, np.array([]), np.array([]), np.array([])
        n = count[keep]
        return (keep,
                self.accurate.cpu().numpy()[keep] / n,
                self.floored.cpu().numpy()[keep] / n,
                n)

    def compute(self) -> Tensor:
        keep = self.count >= self.min_count
        if not keep.any():
            return torch.tensor(0., device=self.count.device)
        return (self.accurate[keep].float() / self.count[keep].float()).mean()


# ---------------------------------------------------------------- figure

def phone_dur_to_figure(stats: PerPhonemeDurationStats, id_to_name, id_to_class,
                        class_labels, floor_frames, title=None):
    """음소별 정확도와 하한선 비율을 한 그림에.

    부류로 묶어 세로로 쌓는다. 색을 부류마다 다르게 주지 않는 이유는, 여기서 읽어야 할
    것이 '어느 부류인가'(위치가 이미 말해 준다)가 아니라 '얼마나 맞았는가'(크기)이기
    때문이다. 두 계열 — 정확도와 하한선율 — 만 색으로 가른다.
    """
    ids, acc, floor, count = stats.per_phone()
    if ids.size == 0:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, 'no phones above min_count', ha='center')
        plt.axis('off')
        return fig

    font = _cjk_font()
    if font is None:
        class_labels = _ASCII_CLASS
        acc_label = f'accuracy (+/-{stats.tolerance:.0%})'
        floor_label = f'predicted <= {floor_frames} frames'
        x_label = 'rate'
    else:
        plt.rcParams['font.family'] = font
        acc_label = f'정확도 (±{stats.tolerance:.0%})'
        floor_label = f'예측이 ≤{floor_frames}프레임인 비율'
        x_label = '비율'

    # 부류 순서는 class_labels 의 키 순서를 따른다 (PHONE_CLASSES 순: 초성이 먼저).
    # 알파벳순으로 두면 문제가 몰려 있는 초성이 그림 한가운데로 밀려난다.
    rank = {cls: i for i, cls in enumerate(class_labels)}
    order = sorted(range(len(ids)),
                   key=lambda i: (rank.get(id_to_class[ids[i]], len(rank)), -acc[i]))
    ids, acc, floor, count = (np.asarray(x)[order] for x in (ids, acc, floor, count))

    labels, boundaries, prev = [], [], None
    for pos, token in enumerate(ids):
        cls = id_to_class[token]
        if cls != prev:
            boundaries.append((pos, class_labels.get(cls, cls)))
            prev = cls
        labels.append(f'{id_to_name[token]}  n={count[pos]:,}')

    height = max(4.0, 0.32 * len(ids) + 1.6)
    fig, ax = plt.subplots(figsize=(11, height))
    y = np.arange(len(ids))
    ax.barh(y - 0.2, acc, height=0.38, color='#2a78d6', label=acc_label)
    ax.barh(y + 0.2, floor, height=0.38, color='#eb6834', label=floor_label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label)
    ax.grid(axis='x', alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for pos, name in boundaries:
        if pos > 0:
            ax.axhline(pos - 0.5, color='#7a868c', linewidth=0.8, linestyle=':')
        ax.text(1.005, pos, name, transform=ax.get_yaxis_transform(),
                fontsize=9, color='#4c565d', va='center')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    head = '음소별 길이 정확도' if font else 'per-phoneme duration accuracy'
    ax.set_title(f'{head} · {title}' if title else head, fontsize=12)
    fig.tight_layout()
    return fig
