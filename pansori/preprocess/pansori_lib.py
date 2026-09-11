#!/usr/bin/env python3
"""Shared, verified helpers for the pansori DiffSinger preprocessing pipeline.

Every numbered stage script (01_..07_) imports from here so the logic that was
validated on the pilot lives in exactly one place. Heavy third-party imports
(pretty_midi, praatio, librosa) are done lazily inside the functions that need
them, so this module imports cleanly in every conda env the stages run under
(pansori / aligner / base), each of which has a different subset installed.
"""

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- paths

# preprocess/ -> pansori/ -> repo root
PREPROCESS_DIR = Path(__file__).resolve().parent
PANSORI_DIR = PREPROCESS_DIR.parent
REPO_DIR = PANSORI_DIR.parent
DATA_DIR = PANSORI_DIR / "data"
ALIGN_DIR = PANSORI_DIR / "alignment"

# Source pansori corpus (not under this repo).
PANSORI_DATA = Path("/home/sangheon/Desktop/Pansori_Data")
MIDI_DIR = PANSORI_DATA / "rosvot_midi"

# ---------------------------------------------------------------- align config


def load_align_config(path: Optional[Path] = None) -> Dict:
    """Stage 4 의 정렬기 설정. 경로 값은 ~ 확장 + repo 기준 절대경로로 돌려준다.

    두 정렬기(MFA / SOFA)는 서로 다른 conda env 에 살고 각자 체크포인트·사전
    경로를 들고 다닌다. 그걸 스크립트 상수로 두면 정렬기를 바꿀 때마다 코드를
    고쳐야 하므로 align.yaml 한 곳으로 모았다. 여기서 경로를 미리 풀어 두는 이유는
    04_align.py 가 conda run 으로 다른 cwd 에서 도구를 부르기 때문이다 — 상대경로가
    거기까지 살아 남지 못한다.
    """
    import yaml  # aligner/pansori/dsp 세 env 모두에 있다 (MFA·lightning 의존성)

    cfg_path = Path(path) if path else PREPROCESS_DIR / "align.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    for section, keys in (("sofa", ("root", "ckpt", "dictionary")),
                          ("breath_ap", ("root", "ckpt"))):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                cfg[section][key] = resolve_path(value)
    return cfg


def resolve_path(value: str) -> Path:
    """'~/x' -> home, 'a/b' -> repo-relative, '/a/b' -> as is."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_DIR / path).resolve()


# ---------------------------------------------------------------- audio

# openvpi/DiffSinger runs at 44.1 kHz (configs/acoustic.yaml: audio_sample_rate).
# The source Audio/ is already 44.1 kHz stereo, so cutting is a slice plus a
# stereo-to-mono fold — no resampling, and no loss the old 24 kHz path incurred.
TARGET_SAMPLE_RATE: int = 44100

# ---------------------------------------------------------------- phone sets

# MFA marks non-speech intervals with these (or an empty label).
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}

# The Coda-36 target inventory (see pansori/docs/SEGMENTATION_VERDICT.md).
CODA_VOWELS = {"a", "eo", "o", "u", "eu", "i", "e"}
CODA_36 = {
    "g", "n", "d", "r", "m", "b", "s", "j", "ch", "k", "t", "p", "h",
    "kk", "tt", "pp", "ss", "jj", "K", "N", "T", "L", "M", "P", "NG",
    "a", "eo", "o", "u", "eu", "i", "e", "y", "w",
}
# Legal ph_seq tokens = Coda-36 plus the two non-speech markers.
ALLOWED_PH = CODA_36 | {"SP", "AP"}

# Glides open a nucleus like a vowel does, but they are neither vowel nor
# consonant for the purpose of measuring durations -- they ride on the vowel.
CODA_GLIDES = {"y", "w"}

# Coda consonants are spelled uppercase by convention (convert_phone's 'A|B'
# rule), so the split is mechanical rather than a second hand-kept list.
CODA_CODAS = {p for p in CODA_36 if p.isupper()}
CODA_ONSETS = CODA_36 - CODA_VOWELS - CODA_GLIDES - CODA_CODAS

# The order the report and the training metrics both iterate in. Onsets first
# because that is where the duration floor concentrates.
PHONE_CLASSES = ("onset", "coda", "vowel", "glide", "sp")


def phone_class(phone: str) -> str:
    """Which duration-behaviour group a ph_seq token belongs to.

    Onset and coda are split apart rather than lumped as 'consonant' because
    they behave completely differently under forced alignment: in this corpus
    onsets sit at MFA's 30 ms floor ~40% of the time while codas do so ~12%.
    A single 'consonant' number averages that distinction away, which is the
    whole thing worth seeing.

    Unknown tokens return 'other' rather than raising -- a metric must never be
    the reason a training run dies.
    """
    if phone in ("SP", "AP"):
        return "sp"
    if phone in CODA_VOWELS:
        return "vowel"
    if phone in CODA_GLIDES:
        return "glide"
    if phone in CODA_CODAS:
        return "coda"
    if phone in CODA_ONSETS:
        return "onset"
    return "other"

# IPA vowel bases (diacritics stripped), for the "next phone is a vowel" tests.
IPA_VOWEL_BASE = {"i", "e", "ɛ", "a", "ɐ", "ʌ", "o", "u", "ɨ"}

# A ʷ diacritic before one of these is the vowel's own rounding, not an on-glide.
IPA_ROUND_VOWELS = {"o", "u"}

# Glides open a syllable nucleus just as a vowel does, so a consonant before one
# is an onset, not a coda (뭐 = m ɥ/w ʌ -> m w eo, never M w eo). They are not
# vowels for the palatalization test, hence a separate set.
IPA_GLIDE = {"j", "w", "ɥ", "ɰ"}

# ---------------------------------------------------------------- text (Stage 3)


def normalize_repeats(text: str) -> str:
    """Collapse Whisper's sustained-note artifact (지이이이…이 → 지이).

    A held sung vowel makes Whisper emit the same syllable hundreds of times.
    The real phoneme content is one long vowel, so we fold runs of an identical
    Hangul syllable (3+) down to a single copy, and clean up '이... 이...' forms.
    """
    text = re.sub(r"([가-힣])\1{2,}", r"\1", text)
    text = re.sub(r"([가-힣])(\.\.\.\s*\1){2,}", r"\1", text)
    text = re.sub(r"\.{2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hangul_only(text: str) -> str:
    """Drop everything that is not a Hangul syllable. Used for text matching."""
    return "".join(ch for ch in text if "\uac00" <= ch <= "\ud7a3")


def to_jamo(text: str) -> str:
    """Hangul syllables -> their NFD jamo, non-Hangul dropped.

    Matching Whisper's pansori output against a 창본 syllable-by-syllable fails:
    '굴러' and '군로' share no syllable at all. In jamo they are ㄱㅜㄹㄹㅓ vs
    ㄱㅜㄴㄹㅗ, which overlaps most of the way -- Whisper's errors are usually one
    jamo off, not one syllable off, so the finer unit is what carries the signal.
    """
    return unicodedata.normalize("NFD", hangul_only(text))


# ---------------------------------------------------------------- IPA -> Coda (Stage 4)


def load_ipa_to_coda() -> Dict[str, str]:
    """mfa_ipa -> coda string. '|' = position choice, ' ' = on-glide split."""
    mapping = {}
    with open(ALIGN_DIR / "ipa_to_coda.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["mfa_ipa"]] = row["coda"]
    return mapping


def vowel_base(ipa_label: Optional[str]) -> Optional[str]:
    """Vowel base of an IPA label if it is a vowel, else None."""
    if not ipa_label:
        return None
    base = ipa_label.replace("ː", "").replace("ʲ", "").replace("ʷ", "")
    return base if base in IPA_VOWEL_BASE else None


def _opens_nucleus(ipa_label: Optional[str]) -> bool:
    """True if this phone starts a syllable nucleus (a vowel or an on-glide)."""
    if not ipa_label:
        return False
    base = ipa_label.replace("ː", "").replace("ʲ", "").replace("ʷ", "")
    return base in IPA_VOWEL_BASE or base in IPA_GLIDE


def convert_phone(ipa_label: str, next_ipa: Optional[str],
                  mapping: Dict[str, str]) -> List[str]:
    """Apply the two context rules verified in FIELD_CHECK / VERDICT.

    - 'A|B' (position): next phone opens a nucleus -> A (onset) else B (coda).
    - 'X y' / 'X w' (on-glide split): keep the glide only where the following
      vowel does not already account for the diacritic (see below).

    Scored against dictionaries/korean-coda.txt by
    alignment/check_dict_agreement.py -- change these rules only with that
    number in hand.
    """
    coda_str = mapping.get(ipa_label)
    if coda_str is None:
        return [f"<OOV:{ipa_label}>"]

    next_vowel = vowel_base(next_ipa)

    # Peel the on-glide off first. Some entries carry both rules at once
    # ("m|M y" for mʲ); handling '|' first would swallow the glide into the
    # coda alternative and emit a single bogus token "M y".
    tokens = coda_str.split()
    glide = []
    if len(tokens) >= 2 and tokens[-1] in ("y", "w"):
        glide = [tokens[-1]]
        tokens = tokens[:-1]
    base = " ".join(tokens)

    if "|" in base:
        onset, coda = base.split("|")
        if not _opens_nucleus(next_ipa):
            # Coda position: an on-glide cannot follow a syllable-final consonant.
            return coda.split()
        # Either side may hold more than one token: a geminate spans two
        # syllables and needs both (ʎː in 빨리 is 빨's coda plus 리's onset).
        base = onset

    # The glide diacritics mark coarticulation, which is only sometimes a real
    # on-glide. Drop it when the following vowel already accounts for it:
    #   ʲ before 'i'      -> plain palatalization  (시 = ɕʰ i  -> s i,  not s y i)
    #   ʷ before 'o'/'u'  -> plain labialization   (호 = ɸʷ o  -> h o,  not h w o)
    # Before any other vowel the glide is contrastive and must be kept
    # (샤 = ɕʰ ɐ -> s y a, 화 = ɸʷ ɐ -> h w a).
    if glide == ["y"] and next_vowel == "i":
        glide = []
    elif glide == ["w"] and next_vowel in IPA_ROUND_VOWELS:
        glide = []
    return base.split() + glide


# ---------------------------------------------------------------- ROSVOT notes (Stage 5)


from functools import lru_cache


@lru_cache(maxsize=None)
def _song_notes(stem: str) -> Tuple[Tuple[int, float, float], ...]:
    """Parse a whole song's ROSVOT MIDI once. Cached per stem.

    Without this, slicing every utterance re-parses the full-song MIDI (a song
    with ~80 utterances would be parsed ~80 times); caching makes it one parse
    per song, turning Stage 5 from an hour into seconds on the full set.
    """
    import pretty_midi  # lazy: only the note stages need it

    midi = pretty_midi.PrettyMIDI(str(MIDI_DIR / f"{stem}_vocal.mid"))
    return tuple(
        (note.pitch, note.start, note.end)
        for instrument in midi.instruments
        for note in instrument.notes
    )


def slice_notes(stem: str, start_sec: float, end_sec: float) -> List[Tuple[int, float, float]]:
    """ROSVOT notes overlapping [start,end], clipped and shifted to local time.

    Returns [(midi_pitch, local_start, local_end), ...] sorted by start.
    """
    notes = []
    for pitch, note_start, note_end in _song_notes(stem):
        if note_end <= start_sec or note_start >= end_sec:
            continue
        notes.append((
            pitch,
            max(note_start, start_sec) - start_sec,
            min(note_end, end_sec) - start_sec,
        ))
    notes.sort(key=lambda x: x[1])
    return notes


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap between intervals [a0,a1] and [b0,b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


# ---------------------------------------------------------------- CREPE f0 (Stage 8)

# Whole-song CREPE output, one csv per song: time,frequency,confidence at 10ms.
# Verified to share Audio/'s timeline exactly (204.17s wav -> 20418 frames).
CREPE_DIR = PANSORI_DATA / "crepe"
CREPE_HOP_SEC = 0.01

# Pansori vocal range. ROSVOT's lowest note is MIDI 35 (61.7 Hz) and the highest
# measured is 740 Hz; raw CREPE reaches 1986 Hz, which is octave error, not voice.
F0_MIN: float = 55.0
F0_MAX: float = 750.0

# Local-median window for octave correction: 101 CREPE frames = ~1s. Long enough
# that a single bad run cannot drag the median with it, short enough to follow a
# real melodic line.
_OCTAVE_MEDIAN_FRAMES = 101


def _octave_correct(f0, voiced):
    """Pull octave-doubled/halved frames back onto the local melodic line.

    CREPE reports harmonics as the fundamental on pressed, noisy pansori voice.
    For each voiced frame we try shifting by whole octaves and keep the shift
    that lands closest to the local median, which no single error can move.
    Returns (corrected f0, voiced) with out-of-range frames marked unvoiced.
    """
    import numpy as np
    from scipy.ndimage import median_filter

    log_f0 = np.full(len(f0), np.nan, dtype=np.float64)
    log_f0[voiced] = np.log2(f0[voiced])
    if not voiced.any():
        return f0, voiced

    # Fill the unvoiced gaps so the median filter sees a continuous line.
    idx = np.arange(len(log_f0))
    filled = np.interp(idx, idx[voiced], log_f0[voiced])
    local_median = median_filter(filled, size=_OCTAVE_MEDIAN_FRAMES, mode="nearest")

    # Work on `filled` rather than log_f0: the two agree wherever voiced, and
    # this keeps NaN out of the argmin entirely (an all-NaN column would raise).
    shifts = np.arange(-2, 3)[:, None]                     # [-2..2] octaves
    candidates = filled[None, :] + shifts                  # [5, T]
    best = np.argmin(np.abs(candidates - local_median[None, :]), axis=0)
    corrected = np.where(voiced, np.exp2(filled + shifts[best, 0]), 0.0)

    in_range = (corrected >= F0_MIN) & (corrected <= F0_MAX)
    return np.where(in_range, corrected, 0.0), voiced & in_range


@lru_cache(maxsize=8)
def _song_crepe(stem: str, conf_threshold: float):
    """One song's CREPE track, octave-corrected. Cached per (song, threshold).

    maxsize is small on purpose: a long song is ~135k frames and segments arrive
    grouped by song, so a handful of entries covers the working set instead of
    holding all 391MB of csv in memory at once.
    """
    import numpy as np

    raw = np.loadtxt(CREPE_DIR / f"{stem}.f0.csv", delimiter=",", skiprows=1)
    f0, conf = raw[:, 1].astype(np.float64), raw[:, 2]
    voiced = (conf >= conf_threshold) & (f0 > 0)
    f0, voiced = _octave_correct(f0, voiced)
    return f0, voiced


def slice_f0(stem: str, start_sec: float, n_frames: int, hop_sec: float,
             conf_threshold: float = 0.5):
    """CREPE f0 resampled onto n_frames of the DiffSinger grid. Unvoiced = 0.

    Frame i is centred at start_sec + i*hop_sec in song time, matching how mel
    frames are laid out. Interpolation happens in log-f0 so an octave is a fixed
    distance; the voiced mask is taken nearest-neighbour, since averaging a
    voiced and an unvoiced frame has no meaning.
    """
    import numpy as np

    song_f0, song_voiced = _song_crepe(stem, conf_threshold)
    want = start_sec + np.arange(n_frames) * hop_sec
    src = np.arange(len(song_f0)) * CREPE_HOP_SEC

    nearest = np.clip(np.round((want - src[0]) / CREPE_HOP_SEC).astype(int),
                      0, len(song_f0) - 1)
    voiced = song_voiced[nearest]

    out = np.zeros(n_frames, dtype=np.float32)
    if song_voiced.any():
        log_line = np.interp(want, src[song_voiced], np.log2(song_f0[song_voiced]))
        out[voiced] = np.exp2(log_line[voiced]).astype(np.float32)
    return out


# ---------------------------------------------------------------- item / speaker


def extract_uuid(name: str) -> Optional[str]:
    """Leading 8-hex-char id that joins audio / labels / midi / items."""
    match = re.match(r"^([0-9a-f]{8})", name)
    return match.group(1) if match else None


# ---------------------------------------------------------------- json lines IO


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_dir(set_name: str) -> Path:
    """Per-run working dir, e.g. data/pilot_6s. Created if missing."""
    d = DATA_DIR / set_name
    d.mkdir(parents=True, exist_ok=True)
    return d
