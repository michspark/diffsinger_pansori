#!/usr/bin/env python3
"""Stage 4's aligner backends — MFA and SOFA behind the same seam.

Both aligners do the same job: take (wav + lyrics), leave
`aligned{suffix}/*.TextGrid`, and read that TextGrid back as a Coda-36 phoneme
sequence. What differs is the inside of those three steps.

    MFA   An HMM-GMM for speech. The korean_mfa dictionary emits IPA, so it has
          to pass through ipa_to_coda.csv and convert_phone's contextual rules to
          become Coda (89.59% agreement).
    SOFA  Built for singing. DictionaryG2P uses the dictionary's values directly
          as phonemes, so handing it dictionaries/korean-coda.txt makes **Coda
          come out natively** -- the conversion step disappears. On top of that,
          the training labels and the inference g2p end up using the same table.

The choice is one `aligner` line in align.yaml. The shape follows
initialize_pe() in modules/pe/__init__.py — a thin factory selected by a config
string.

SOFA and FoxBreatheLabeler live outside this repository and each needs its own
conda env (the aligner env is py3.14 + MFA, so torch cannot go in alongside).
Hence the shell-out.
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
    """TextGrid -> lists of (phones, words) spans. Both backends use the same tier names."""
    from praatio import textgrid as tgio

    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    return ([(iv.label.strip(), iv.start, iv.end) for iv in grid.getTier("phones").entries],
            [(iv.label.strip(), iv.start, iv.end) for iv in grid.getTier("words").entries])


def _words_tier(words) -> List[list]:
    """The words tier minus non-speech labels. This is what ph_num rests on (06_merge.group_words)."""
    return [[label, start, end] for label, start, end in words
            if label.lower() not in SILENCE_LABELS and label not in ("SP", "AP")]


class Aligner:
    """Build the corpus, run the aligner, read the TextGrid back as a Coda phoneme sequence."""

    name = ""
    corpus_dirname = "corpus"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.min_hangul = cfg.get("min_hangul", 2)

    # -- 1. corpus ---------------------------------------------------------

    def lab_text(self, text: str) -> str:
        """One line of lyrics -> the form this aligner expects in a .lab."""
        return text.strip()

    def build_corpus(self, out_dir: Path, segments, lyrics, suffix: str = "") -> Path:
        """Writes {corpus}/{item}.wav (a symlink) and {item}.lab."""
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
        print(f"{self.name} corpus: {n} items "
              f"({len(segments) - n} excluded for too little text) -> {corpus}")
        return corpus

    # -- 2. run / 3. parse -------------------------------------------------

    def run(self, corpus: Path, aligned: Path) -> None:
        raise NotImplementedError

    def parse(self, tg_path: Path) -> dict:
        raise NotImplementedError


class MfaAligner(Aligner):
    """korean_mfa + IPA->Coda conversion. The path this pipeline used from the start."""

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
    """SOFA (+ optionally FoxBreatheLabeler's AP overlay).

    Since the dictionary handed over is korean-coda.txt, the phones tier is
    already Coda-36. parse does not convert, it only passes through -- it is not
    that the MFA arm's 89.59% conversion agreement becomes 100%, it is that the
    step which could be wrong no longer exists.
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
        # FBL only works over a tg that has no AP (FoxBreatheLabeler readme).
        self.ap_detector = ("NoneAPDetector" if self.breath.get("enabled")
                            else sofa.get("ap_detector",
                                          "LoudnessSpectralcentroidAPDetector"))
        if self.lab_unit not in ("syllable", "eojeol"):
            sys.exit(f"sofa.lab_unit must be syllable or eojeol: {self.lab_unit}")

    # -- corpus ------------------------------------------------------------

    def lab_text(self, text: str) -> str:
        """SOFA's DictionaryG2P looks up the dictionary per whitespace token and
        inserts an SP after each.

        With `syllable`, one syllable is itself a dictionary headword, so the
        words tier comes out per syllable -- the split that 06_merge.group_words
        had to undo heuristically disappears. With `eojeol`, SP only appears at
        eojeol boundaries, but an eojeol dictionary is required.
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
        """Build a new dictionary by concatenating, from the syllable dictionary,
        only the eojeol that actually occur in the corpus.

        A SOFA dictionary has one line per headword, so aligning per eojeol needs
        eojeol headwords. Since build_dictionary.py's output is concatenated
        as-is, the phoneme set does not change (and sandhi stays absent, exactly
        as in the inference g2p).
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
        print(f"eojeol dictionary: {len(words) - missing:,} entries -> {out}"
              + (f"  ({missing} excluded for containing a syllable not in the dictionary)" if missing else ""))
        return out

    # -- run ---------------------------------------------------------------

    def _conda(self, env: str, root: Path, script: str, args: List[str]) -> None:
        """Invoke a tool in a different env. cwd has to be that repository.

        SOFA's infer.py imports modules.* relatively, so it dies at import
        unless cwd=root. Which means every path passed in must be absolute.
        """
        command = ["conda", "run", "--no-capture-output", "-n", env,
                   "python", script] + args
        print("$ " + " ".join(command) + f"    (cwd={root})")
        subprocess.run(command, cwd=root, check=True)

    def run(self, corpus: Path, aligned: Path) -> None:
        if not self.ckpt.exists():
            sys.exit(f"No SOFA checkpoint at: {self.ckpt}\n"
                     f"  Check sofa.ckpt in align.yaml, or build training data with "
                     f"04b_sofa_data.py and train SOFA first.")

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

        # SOFA writes into TextGrid/ beside the wavs. Everything downstream (06,
        # eval, alignment/*) looks at aligned{suffix}/, so move it there.
        produced = corpus / "TextGrid"
        if not produced.is_dir():
            sys.exit(f"No SOFA output at: {produced}")
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
        print(f"{n:,} TextGrids -> {aligned}")

    def _add_breath(self, corpus: Path, produced: Path, aligned: Path) -> None:
        """Overlay AP (breath) onto the SP spans using FoxBreatheLabeler.

        FBL only writes AP over SP or empty labels, so the preceding infer was
        run with NoneAPDetector to leave a tg with no AP (forced in __init__).
        """
        ckpt = Path(self.breath["ckpt"])
        if not ckpt.exists():
            sys.exit(f"No FoxBreatheLabeler checkpoint at: {ckpt}\n"
                     f"  Check breath_ap.ckpt in align.yaml, or set "
                     f"breath_ap.enabled to false.")
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
        print(f"breath labelling done -> {aligned}  ({len(list(aligned.glob('*.TextGrid'))):,} files)")

    # -- parse -------------------------------------------------------------

    def parse(self, tg_path: Path) -> dict:
        """No conversion. The dictionary already emits Coda, so labels pass through.

        sil is left as a placeholder, as in the MFA arm -- whether a note-covered
        silence is a held vowel or an SP is decided by Stage 6, looking at the
        ROSVOT notes. AP is a real breath, so it flows straight through as a
        phoneme.
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
        raise ValueError(f" [x] Unknown aligner: {name} (available: {sorted(ALIGNERS)})")
    return ALIGNERS[name](cfg)
