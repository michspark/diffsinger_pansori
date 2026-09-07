#!/usr/bin/env python3
"""Stage 4 의 정렬기 백엔드 — MFA 와 SOFA 를 같은 자리에 끼운다.

두 정렬기는 하는 일이 같다: (wav + 가사) 를 받아 `aligned{suffix}/*.TextGrid` 를
남기고, 그 TextGrid 를 Coda-36 음소열로 읽는다. 다른 것은 그 세 단계의 속이다.

    MFA   speech 용 HMM-GMM. korean_mfa 사전이 IPA 를 내므로 ipa_to_coda.csv 와
          convert_phone 의 문맥 규칙을 거쳐야 Coda 가 된다 (일치율 89.59%).
    SOFA  노래 전용. DictionaryG2P 가 사전 값을 그대로 음소로 쓰므로
          dictionaries/korean-coda.txt 를 주면 **Coda 가 바로 나온다** — 변환
          단계가 없어진다. 게다가 학습 라벨과 추론 g2p 가 같은 표를 쓰게 된다.

선택은 align.yaml 의 `aligner` 한 줄. 모양은 modules/pe/__init__.py 의
initialize_pe() 를 따랐다 — config 문자열로 고르는 얇은 팩토리.

SOFA·FoxBreatheLabeler 는 이 저장소 밖에 살고 각자 conda env 가 필요하다
(aligner env 는 py3.14 + MFA 라 torch 를 같이 넣을 수 없다). 그래서 셸아웃한다.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pansori_lib import (ALLOWED_PH, SILENCE_LABELS, convert_phone,
                         load_ipa_to_coda)


def hangul_syllables(text: str) -> List[str]:
    return [ch for ch in text if "가" <= ch <= "힣"]


def _open_tiers(tg_path: Path):
    """TextGrid -> (phones, words) 구간 리스트. 두 백엔드가 같은 티어 이름을 쓴다."""
    from praatio import textgrid as tgio

    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    return ([(iv.label.strip(), iv.start, iv.end) for iv in grid.getTier("phones").entries],
            [(iv.label.strip(), iv.start, iv.end) for iv in grid.getTier("words").entries])


def _words_tier(words) -> List[list]:
    """비음성 라벨을 뺀 words 티어. ph_num 의 근거가 된다 (06_merge.group_words)."""
    return [[label, start, end] for label, start, end in words
            if label.lower() not in SILENCE_LABELS and label not in ("SP", "AP")]


class Aligner:
    """corpus 를 짓고, 정렬기를 돌리고, TextGrid 를 Coda 음소열로 읽는다."""

    name = ""
    corpus_dirname = "corpus"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.min_hangul = cfg.get("min_hangul", 2)

    # -- 1. corpus ---------------------------------------------------------

    def lab_text(self, text: str) -> str:
        """가사 한 줄 -> 이 정렬기가 .lab 에서 기대하는 형태."""
        return text.strip()

    def build_corpus(self, out_dir: Path, segments, lyrics, suffix: str = "") -> Path:
        """{corpus}/{item}.wav (심링크) + {item}.lab 를 쓴다."""
        corpus = out_dir / f"{self.corpus_dirname}{suffix}"
        corpus.mkdir(exist_ok=True)
        text_by_item = {r["item_name"]: r["text"] for r in lyrics}

        n = 0
        for record in segments:
            item = record["item_name"]
            text = text_by_item.get(item, "").strip()
            if len(hangul_syllables(text)) < self.min_hangul:
                continue                      # nothing alignable
            wav_src = (out_dir / "wavs" / f"{item}.wav").resolve()
            wav_link = corpus / f"{item}.wav"
            if wav_link.exists() or wav_link.is_symlink():
                wav_link.unlink()
            os.symlink(wav_src, wav_link)
            (corpus / f"{item}.lab").write_text(self.lab_text(text), encoding="utf-8")
            n += 1
        print(f"{self.name} 코퍼스: {n}개 "
              f"(텍스트 부족분 제외 {len(segments) - n}개) → {corpus}")
        return corpus

    # -- 2. run / 3. parse -------------------------------------------------

    def run(self, corpus: Path, aligned: Path) -> None:
        raise NotImplementedError

    def parse(self, tg_path: Path) -> dict:
        raise NotImplementedError


class MfaAligner(Aligner):
    """korean_mfa + IPA→Coda 변환. 이 파이프라인이 처음부터 쓰던 경로."""

    name = "MFA"
    corpus_dirname = "mfa_corpus"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        mfa = cfg.get("mfa", {})
        self.acoustic = mfa.get("acoustic", "korean_mfa")
        self.dictionary = mfa.get("dictionary", "korean_mfa")
        self._mapping = None

    def run(self, corpus: Path, aligned: Path) -> None:
        """Invoke the MFA CLI. Assumes the `aligner` env is active."""
        subprocess.run(
            ["mfa", "align", str(corpus), self.dictionary, self.acoustic, str(aligned),
             "--clean", "--output_format", "long_textgrid"],
            check=True,
        )

    def parse(self, tg_path: Path) -> dict:
        """One TextGrid -> {"phones": [[coda,start,end],...], "words": [...]}.

        A single IPA interval can split into two Coda tokens (e.g. mʲ -> m y);
        the interval duration is split evenly (harmless for length parity).
        sil/spn is kept as a single 'sil' placeholder for Stage 6 to resolve.
        """
        if self._mapping is None:
            self._mapping = load_ipa_to_coda()
        intervals, words = _open_tiers(tg_path)

        phones = []
        for idx, (label, start, end) in enumerate(intervals):
            if label.lower() in SILENCE_LABELS:
                phones.append(["sil", start, end])
                continue
            # next non-silence IPA label drives the context rules
            next_ipa = None
            for nlab, _, _ in intervals[idx + 1:]:
                if nlab.lower() not in SILENCE_LABELS:
                    next_ipa = nlab
                    break
            tokens = convert_phone(label, next_ipa, self._mapping)
            step = (end - start) / len(tokens)
            for k, tok in enumerate(tokens):
                phones.append([tok, start + k * step, start + (k + 1) * step])

        return {"phones": phones, "words": _words_tier(words)}


class SofaAligner(Aligner):
    """SOFA (+ 선택적으로 FoxBreatheLabeler 의 AP 덧씌우기).

    사전을 korean-coda.txt 로 주므로 phones 티어가 이미 Coda-36 이다. parse 는
    변환하지 않고 통과시키기만 한다 — MFA 팔의 89.59% 변환 일치율이 100% 가
    되는 게 아니라, 틀릴 단계 자체가 없어진다.
    """

    name = "SOFA"
    corpus_dirname = "sofa_corpus"
    EOJEOL_DICT = "korean-coda-eojeol.txt"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sofa = cfg.get("sofa", {})
        self.root = Path(sofa["root"])
        self.conda_env = sofa.get("conda_env", "sofa")
        self.ckpt = Path(sofa["ckpt"])
        self.dictionary = Path(sofa["dictionary"])
        self.lab_unit = sofa.get("lab_unit", "syllable")
        self.mode = sofa.get("mode", "force")
        self.breath = cfg.get("breath_ap", {}) or {}
        # FBL 은 AP 가 없는 tg 위에서만 동작한다 (FoxBreatheLabeler readme).
        self.ap_detector = ("NoneAPDetector" if self.breath.get("enabled")
                            else sofa.get("ap_detector",
                                          "LoudnessSpectralcentroidAPDetector"))
        if self.lab_unit not in ("syllable", "eojeol"):
            sys.exit(f"sofa.lab_unit 은 syllable 또는 eojeol 이어야 합니다: {self.lab_unit}")

    # -- corpus ------------------------------------------------------------

    def lab_text(self, text: str) -> str:
        """SOFA 의 DictionaryG2P 는 공백 토큰마다 사전을 찾고 뒤에 SP 를 넣는다.

        syllable 이면 음절 하나가 곧 사전 표제어이므로 words 티어가 음절 단위로
        나온다 — 06_merge.group_words 가 휴리스틱으로 되돌리던 분할이 없어진다.
        eojeol 이면 SP 가 어절 경계에만 생기는 대신 어절 사전이 필요하다.
        """
        syllables = hangul_syllables(text)
        if self.lab_unit == "syllable":
            return " ".join(syllables)
        return " ".join(w for w in ("".join(hangul_syllables(t))
                                    for t in text.split()) if w)

    def build_corpus(self, out_dir: Path, segments, lyrics, suffix: str = "") -> Path:
        corpus = super().build_corpus(out_dir, segments, lyrics, suffix)
        if self.lab_unit == "eojeol":
            self.dictionary = self._write_eojeol_dictionary(corpus)
        return corpus

    def _write_eojeol_dictionary(self, corpus: Path) -> Path:
        """코퍼스에 실제로 나온 어절만 음절 사전에서 이어 붙여 새 사전을 만든다.

        SOFA 사전은 표제어 하나당 한 줄이라 어절 단위로 정렬하려면 어절 표제어가
        있어야 한다. build_dictionary.py 의 산출물을 그대로 이어 붙이므로 음소셋은
        바뀌지 않는다 (sandhi 도 그대로 없다 — 추론 g2p 와 같은 상태).
        """
        table = {}
        with open(self.dictionary, encoding="utf-8") as f:
            for line in f:
                syllable, phones = line.rstrip("\n").split("\t")
                table[syllable] = phones
        words = sorted({w for lab in corpus.glob("*.lab")
                        for w in lab.read_text(encoding="utf-8").split()})
        out = self.dictionary.parent / self.EOJEOL_DICT
        missing = 0
        with open(out, "w", encoding="utf-8") as f:
            for word in words:
                if any(ch not in table for ch in word):
                    missing += 1
                    continue
                f.write(f"{word}\t{' '.join(table[ch] for ch in word)}\n")
        print(f"어절 사전 {len(words) - missing:,}개 → {out}"
              + (f"  (사전에 없는 음절 포함 {missing}개 제외)" if missing else ""))
        return out

    # -- run ---------------------------------------------------------------

    def _conda(self, env: str, root: Path, script: str, args: List[str]) -> None:
        """다른 env 에서 도구를 부른다. cwd 는 그 저장소여야 한다.

        SOFA 의 infer.py 는 modules.* 를 상대 임포트하므로 cwd=root 가 아니면
        임포트에서 죽는다. 그래서 넘기는 경로는 전부 절대경로여야 한다.
        """
        command = ["conda", "run", "--no-capture-output", "-n", env,
                   "python", script] + args
        print("$ " + " ".join(command) + f"    (cwd={root})")
        subprocess.run(command, cwd=root, check=True)

    def run(self, corpus: Path, aligned: Path) -> None:
        if not self.ckpt.exists():
            sys.exit(f"SOFA 체크포인트가 없습니다: {self.ckpt}\n"
                     f"  align.yaml 의 sofa.ckpt 를 확인하거나 "
                     f"04b_sofa_data.py 로 학습 데이터를 만들어 먼저 학습하십시오.")

        self._conda(self.conda_env, self.root, "infer.py", [
            "--ckpt", str(self.ckpt),
            "--folder", str(corpus.resolve()),
            "--mode", self.mode,
            "--g2p", "Dictionary",
            "--dictionary", str(self.dictionary),
            "--ap_detector", self.ap_detector,
            "--in_format", "lab",
            "--out_formats", "textgrid",
        ])

        # SOFA 는 wav 와 같은 디렉터리의 TextGrid/ 아래에 쓴다. 하류(06, eval,
        # alignment/*)는 전부 aligned{suffix}/ 를 보므로 그리로 옮긴다.
        produced = corpus / "TextGrid"
        if not produced.is_dir():
            sys.exit(f"SOFA 출력이 없습니다: {produced}")
        if self.breath.get("enabled"):
            self._add_breath(corpus, produced, aligned)
        else:
            self._collect(produced, aligned)

    def _collect(self, produced: Path, aligned: Path) -> None:
        if aligned.exists():
            shutil.rmtree(aligned)
        aligned.mkdir(parents=True)
        n = 0
        for path in produced.glob("*.TextGrid"):
            shutil.move(str(path), aligned / path.name)
            n += 1
        shutil.rmtree(produced, ignore_errors=True)
        print(f"TextGrid {n:,}개 → {aligned}")

    def _add_breath(self, corpus: Path, produced: Path, aligned: Path) -> None:
        """FoxBreatheLabeler 로 SP 구간 위에 AP(들숨) 를 덧씌운다.

        FBL 은 SP/빈 라벨 위에만 AP 를 쓰므로, 앞선 infer 는 NoneAPDetector 로
        돌려 AP 없는 tg 를 만들어 둔 상태다 (__init__ 에서 강제).
        """
        ckpt = Path(self.breath["ckpt"])
        if not ckpt.exists():
            sys.exit(f"FoxBreatheLabeler 체크포인트가 없습니다: {ckpt}\n"
                     f"  align.yaml 의 breath_ap.ckpt 를 확인하거나 "
                     f"breath_ap.enabled 를 false 로 두십시오.")
        if aligned.exists():
            shutil.rmtree(aligned)
        aligned.mkdir(parents=True)
        self._conda(self.breath.get("conda_env", "fbl"), Path(self.breath["root"]),
                    "textgrid_add_ap.py", [
                        "--ckpt_path", str(ckpt),
                        "--wav_dir", str(corpus.resolve()),
                        "--tg_dir", str(produced.resolve()),
                        "--tg_out_dir", str(aligned.resolve()),
                        "--ap_threshold", str(self.breath.get("ap_threshold", 0.4)),
                        "--ap_dur", str(self.breath.get("ap_dur", 0.08)),
                        "--sp_dur", str(self.breath.get("sp_dur", 0.1)),
                    ])
        shutil.rmtree(produced, ignore_errors=True)
        print(f"들숨 라벨 완료 → {aligned}  ({len(list(aligned.glob('*.TextGrid'))):,}개)")

    # -- parse -------------------------------------------------------------

    def parse(self, tg_path: Path) -> dict:
        """변환 없음. 사전이 이미 Coda 를 내므로 라벨을 그대로 통과시킨다.

        sil 은 MFA 팔과 같이 자리표시자로 남긴다 — 노트가 덮은 무음을 held vowel
        로 볼지 SP 로 볼지는 Stage 6 이 ROSVOT 노트를 보고 정한다. AP 는 실제
        들숨이므로 그대로 음소로 흘려보낸다.
        """
        intervals, words = _open_tiers(tg_path)
        phones = []
        for label, start, end in intervals:
            if label == "AP":
                phones.append(["AP", start, end])
            elif label.lower() in SILENCE_LABELS:
                phones.append(["sil", start, end])
            elif label in ALLOWED_PH:
                phones.append([label, start, end])
            else:
                phones.append([f"<OOV:{label}>", start, end])
        return {"phones": phones, "words": _words_tier(words)}


ALIGNERS = {"mfa": MfaAligner, "sofa": SofaAligner}


def get_aligner(name: str, cfg: dict) -> Aligner:
    if name not in ALIGNERS:
        raise ValueError(f" [x] Unknown aligner: {name} (있는 것: {sorted(ALIGNERS)})")
    return ALIGNERS[name](cfg)
