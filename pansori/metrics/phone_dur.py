#!/usr/bin/env python3
"""Validation metrics that measure duration prediction **per phone class**.

upstream already has `PhonemeDurationAccuracy` in `modules/metrics/duration.py`
and registers it at `training/variance_task.py:136`. But that metric folds the
whole corpus into one number -- and what is worth seeing in this corpus is
exactly what is inside that fold:

    onsets <=3 frames  40.9%          codas <=3 frames  15.1%
    (pp 48 / tt 46 / b 42 / g 42)  (N 8.5 / NG 12 / M 13)

Lump onsets and codas together as 'consonants' and they cancel each other out
and nothing is visible; mix vowels in and less is visible still. So this measures
each of `pansori_lib.phone_class`'s five classes separately.
(`pansori/eval/label_report.py` measures the label side with the same
classification, so the two numbers correspond directly.)

**The unit is frames.** The model only sees ph_dur after binarize quantises it to
mel frames (`preprocessing/variance_binarizer.py:293-295`), so converting back to
seconds here would invent precision that does not exist. The floor is in frames
too -- at hop 512 / 44.1 kHz one frame is 11.61 ms, and MFA's 30 ms floor lands
on 3 frames (34.8 ms). Which is why `floor_rate_gt` lines up with
label_report's '<=3 frames' column, not its '<=30ms' one.

Predicted durations are fitted to the GT word durations with `RhythmRegulator`
before measuring -- because inference really does that
(`modules/toplevel.py:262`), and because word duration is a value the score
supplies, not a problem the model solves. What the model solves is **the division
within a word**, and that is what has to be measured. This is the same treatment
upstream's `PhonemeDurationAccuracy` applies, so the two numbers are comparable.

This file does not touch one line of upstream. Registration is done by the
subclass in `pansori/variance_task.py`.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from torch import Tensor

matplotlib.rcParams['axes.unicode_minus'] = False

# The class names are Hangul, so a font with CJK glyphs is needed. Without one the
# figure prints tofu boxes and becomes unreadable in wandb, so when no font is found
# the labels fall back to romanised names. Noto Sans CJK carries Hangul glyphs whether
# it is registered under its JP or its KR name.
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
    """A token id -> class index lookup. Built once per dictionary.

    `PAD_INDEX` (0) and unclassifiable tokens are set to -1 so they fall into no
    class at all. Merged phonemes (several names on one id, as with SP/AP) are
    classified by their first name, since `decode_one(scalar=False)` returns a
    tuple -- and merged phonemes are by definition in the same class.
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
    """The shared skeleton for metrics that filter by token class.

    `token_class` is a buffer, so it follows `.to(device)`. With `cls_id=None`
    every phoneme inside the mask is used (the overall metric, no class split).
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
    """The fraction of phonemes in a class where |pred - target| <= tolerance * target.

    The same definition as upstream's `PhonemeDurationAccuracy`, with the same
    tolerance of 0.2. The only difference is filtering by class, so the per-class
    values are a decomposition of that overall value.

    Phonemes whose target is 0 are excluded, since relative error is undefined
    there (frame quantisation can produce a zero-frame phoneme).
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
    """Mean absolute error for one class (ms).

    Accuracy alone is not enough -- onsets are mostly 2-4 frames, so there are
    ranges where a 20% tolerance does not permit even one frame, and accuracy sits
    pinned at the floor and does not move. The ms error keeps moving underneath
    it.
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
    """The fraction of phonemes at or below the floor. `on='target'` measures the
    labels, `'pred'` the predictions.

    The point is reading the two curves together. If the prediction tracks the
    labels, the model has learned the labels' disease; if it is much lower, the
    model is behaving differently from the labels. Which of those is correct is
    not something this metric answers -- that is
    `pansori/eval/intelligibility.py`'s job.
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
    """Accumulates per-phoneme accuracy. As a scalar it emits the macro average.

    Why macro: the micro average (= upstream's `ph_dur_acc`) is pulled towards the
    vowels `a`, `i` and `o`, which are 25% of everything. All 345 instances of
    `pp` collapsing would not move the micro figure by 0.1 pp. Read together, the
    two averages show whether only the common phonemes are being hit.

    The material for the bar chart comes from `per_phone()`.
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
        """(token_ids, accuracy, floor_rate, count) — phonemes with at least min_count samples."""
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
    """Per-phoneme accuracy and floor rate, on one figure.

    Grouped by class and stacked vertically. The classes are not given different
    colours because what has to be read here is not 'which class' (position
    already says that) but 'how accurate' (size). Only the two series --
    accuracy and floor rate -- are separated by colour.
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

    # Class order follows the key order of class_labels (PHONE_CLASSES order: onsets
    # first). Sorted alphabetically, the onsets -- where the problems are -- get pushed
    # into the middle of the figure.
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
