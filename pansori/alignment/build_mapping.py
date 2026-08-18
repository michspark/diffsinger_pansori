#!/usr/bin/env python3
"""Generate the candidate korean_mfa(IPA) -> Coda-36 conversion table.

Every phone in mfa_phone_inventory.txt gets a row classified as:
  clean       - unconditional 1:1
  needs_rule  - correct target depends on surrounding phones
  unmapped    - no Coda equivalent

Two diacritics drive most of the rules:
  ʲ  palatalized  -> the syllable had an on-glide (ㅑㅕㅛㅠ) OR simply precedes
                     ㅣ. Only the former needs a `y` emitted.
  ʷ  labialized   -> on-glide ㅘㅝ etc -> emit `w`.
  ː  length       -> dropped; duration lives in ph_durs, not in the symbol.
"""

import csv
import unicodedata
from pathlib import Path

CHECK_DIR = Path(__file__).resolve().parent

# Base consonant: MFA symbol (diacritics stripped) -> Coda symbol.
# NOTE the deliberate mismatches. MFA `k` is ㄱ but Coda `k` is ㅋ; MFA `p` is
# ㅂ but Coda `p` is ㅍ. Never match these by string equality.
BASE_CONSONANT = {
    # 양순 ㅂㅍㅃ
    "p": ("b", "clean", "ㅂ 초성 (무성). MFA p = ㅂ, Coda p = ㅍ — 심볼 충돌 주의"),
    "b": ("b", "clean", "ㅂ 유성 변이 (모음 사이)"),
    "β": ("b", "clean", "ㅂ 마찰 변이"),
    "pʰ": ("p", "clean", "ㅍ"),
    "ɸ": ("p", "clean", "ㅍ 마찰 변이"),
    "p͈": ("pp", "clean", "ㅃ"),
    "p̚": ("P", "clean", "종성 ㅂ (미파음)"),
    # 치조 ㄷㅌㄸ
    "t": ("d", "clean", "ㄷ 초성. MFA t = ㄷ, Coda t = ㅌ — 심볼 충돌 주의"),
    "d": ("d", "clean", "ㄷ 유성 변이"),
    "tʰ": ("t", "clean", "ㅌ"),
    "t͈": ("tt", "clean", "ㄸ"),
    "t̚": ("T", "clean", "종성 ㄷ/ㅅ/ㅈ류 (미파음)"),
    # 연구개 ㄱㅋㄲ
    "k": ("g", "clean", "ㄱ 초성 (어두 무성). MFA k = ㄱ, Coda k = ㅋ — 심볼 충돌 주의"),
    "ɡ": ("g", "clean", "ㄱ 유성 변이 (모음 사이). 국악 = k u ɡ ɐ k̚"),
    "ɣ": ("g", "clean", "ㄱ 마찰 유성 변이"),
    "kʰ": ("k", "clean", "ㅋ"),
    "k͈": ("kk", "clean", "ㄲ"),
    "k̚": ("K", "clean", "종성 ㄱ (미파음)"),
    "x": ("h", "clean", "ㅎ 연구개 변이"),
    # 경구개 파열 — ㄱ계열이 ㅣ/반모음 앞에서 구개음화된 것
    "c": ("g", "needs_rule", "ㄱ + 구개음화. 교육 = c oː j u → g y o. 뒤 모음이 i가 아니면 y 삽입"),
    "ɟ": ("g", "needs_rule", "ㄱ 유성 + 구개음화. 시김새 = ɕʰ iː ɟ i m sʰ ɛː → 뒤가 i라 y 없음"),
    "cʰ": ("k", "needs_rule", "ㅋ + 구개음화"),
    "c͈": ("kk", "needs_rule", "ㄲ + 구개음화"),
    # 마찰 ㅅㅆ
    "s": ("s", "clean", "ㅅ"),
    "sʰ": ("s", "clean", "ㅅ (기식). 사랑 = sʰ ɐ ɾ ɐ ŋ"),
    "s͈": ("ss", "clean", "ㅆ"),
    "ɕʰ": ("s", "needs_rule", "ㅅ + 구개음화 (ㅣ/반모음 앞). 시 = ɕʰ i → s i, y 없음"),
    "ɕ͈": ("ss", "needs_rule", "ㅆ + 구개음화"),
    # 파찰 ㅈㅊㅉ
    "tɕ": ("j", "clean", "ㅈ"),
    "dʑ": ("j", "clean", "ㅈ 유성 변이"),
    "tɕʰ": ("ch", "clean", "ㅊ"),
    "tɕ͈": ("jj", "clean", "ㅉ"),
    # ㅎ
    "h": ("h", "clean", "ㅎ"),
    "ɦ": ("h", "clean", "ㅎ 유성 변이"),
    "ç": ("h", "needs_rule", "ㅎ + 구개음화. 향 = ç ɐ ŋ → h y a NG. y 삽입 필요"),
    "ʝ": ("h", "needs_rule", "ㅎ 유성 + 구개음화. 춘향 = tɕʰ u n ʝ ɐ ŋ"),
    # 비음 — 초성/종성이 같은 심볼이라 위치로 판정해야 함
    "m": ("m|M", "needs_rule", "ㅁ. 뒤에 모음이 오면 초성 m, 아니면 종성 M"),
    "n": ("n|N", "needs_rule", "ㄴ. 뒤에 모음이 오면 초성 n, 아니면 종성 N"),
    "ɲ": ("n|N", "needs_rule", "ㄴ + 구개음화. 아니리 = ɐ ɲ i(초성) vs 계면조 = ... ɲ dʑ o(종성)"),
    "ŋ": ("NG", "clean", "종성 ㅇ. 한국어에서 초성으로 출현하지 않음"),
    # 유음
    "ɾ": ("r", "clean", "초성/모음간 ㄹ (탄설음)"),
    "ɭ": ("L", "clean", "종성 ㄹ (설측음). 강물 = k ɐ ŋ m u ɭ"),
    "ʎ": ("L|r", "needs_rule", "구개음화된 ㄹ. 걸렸 = k ʌː ʎː ʌ t̚ 처럼 ㄹㄹ 겹침에서 출현"),
    # 반모음 (초성이 없을 때만 독립 음소로 출현)
    "j": ("y", "clean", "반모음 ㅣ계. 야 = j ɐ"),
    "w": ("w", "clean", "반모음 ㅜ계. 와 = w ɐ"),
    "ɥ": ("w", "clean", "ㅟ의 반모음. 위 = ɥ i → w i"),
    "ɰ": ("eu", "needs_rule", "ㅢ의 활음. 의 = ɰ i → Coda는 eu i (모음 2개)로 표기"),
}

BASE_VOWEL = {
    "ɐ": ("a", "clean", "ㅏ"),
    "ʌ": ("eo", "clean", "ㅓ"),
    "o": ("o", "clean", "ㅗ"),
    "u": ("u", "clean", "ㅜ"),
    "ɨ": ("eu", "clean", "ㅡ"),
    "i": ("i", "clean", "ㅣ"),
    "e": ("e", "clean", "ㅔ"),
    "ɛ": ("e", "clean", "ㅐ → Coda가 의도한 ㅐ/ㅔ 통합과 일치"),
}

LENGTH = "ː"          # ː
PALATAL = "ʲ"         # ʲ
LABIAL = "ʷ"          # ʷ


def strip_diacritics(phone: str):
    """Split a phone into (base, has_palatal, has_labial, has_length)."""
    has_length = LENGTH in phone
    has_palatal = PALATAL in phone
    has_labial = LABIAL in phone
    base = phone.replace(LENGTH, "").replace(PALATAL, "").replace(LABIAL, "")
    return base, has_palatal, has_labial, has_length


def main() -> None:
    phones = [
        line.strip()
        for line in open(CHECK_DIR / "mfa_phone_inventory.txt", encoding="utf-8")
        if line.strip()
    ]

    rows = []
    for phone in phones:
        base, palatal, labial, length = strip_diacritics(phone)

        if base in BASE_VOWEL:
            coda, confidence, note = BASE_VOWEL[base]
            if length:
                note += " / 장모음 → 길이는 ph_durs가 담당하므로 심볼에서 버림"
        elif base in BASE_CONSONANT:
            coda, confidence, note = BASE_CONSONANT[base]
            if length:
                note += " / 장자음"
        else:
            rows.append({
                "mfa_ipa": phone, "coda": "", "confidence": "unmapped",
                "note": f"BASE_CONSONANT/VOWEL에 정의되지 않음 (base={base})",
            })
            continue

        # ʲ / ʷ 는 선행 자음에 붙은 온글라이드 → Coda에서는 별도 음소로 분리
        if palatal:
            coda = f"{coda} y"
            confidence = "needs_rule"
            note += " / ʲ = 온글라이드 y 분리. 단, 뒤 모음이 i면 단순 구개음화이므로 y 생략"
        if labial:
            coda = f"{coda} w"
            confidence = "needs_rule"
            note += " / ʷ = 온글라이드 w 분리. 과일 = kʷ ɐ i ɭ → g w a i L"

        rows.append({
            "mfa_ipa": phone, "coda": coda,
            "confidence": confidence, "note": note,
        })

    out_path = CHECK_DIR / "ipa_to_coda.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mfa_ipa", "coda", "confidence", "note"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["confidence"]] = counts.get(row["confidence"], 0) + 1

    print(f"총 {len(rows)}개 음소 → {out_path.name}")
    for key in ("clean", "needs_rule", "unmapped"):
        print(f"  {key:11s}: {counts.get(key, 0):3d}")

    unmapped = [r for r in rows if r["confidence"] == "unmapped"]
    if unmapped:
        print("\n미매핑:")
        for row in unmapped:
            print("  ", row["mfa_ipa"], "|", row["note"])

    # Coda 36 중 실제로 도달 가능한 심볼을 확인 (도달 불가한 게 있으면 설계 구멍)
    reachable = set()
    for row in rows:
        for token in row["coda"].replace("|", " ").split():
            reachable.add(token)
    coda36 = set(
        "g n d r m b s j ch k t p h kk tt pp ss jj K N T L M P NG "
        "a eo o u eu i e y w".split()
    )
    print("\nCoda 36 중 도달 불가:", sorted(coda36 - reachable) or "없음")


if __name__ == "__main__":
    main()
