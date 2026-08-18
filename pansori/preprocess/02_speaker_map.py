#!/usr/bin/env python3
"""Stage 2: map each utterance to a speaker (명창) for multi-speaker training.

Filename form: {uuid8}-{track}-{명창}-{바탕}_{대목}
About a third of the songs omit the 명창 field; those become unknown_{uuid} so
they are still a single consistent speaker per song (one song = one singer).

Input : data/{set}/segments.jsonl   (item_name, song_stem)
Output: data/{set}/spk_map.json      {"speakers":[...], "item2spk":{item:spk}}
"""

import argparse
import json
import re

from pansori_lib import read_jsonl, set_dir, extract_uuid


def singer_from_stem(stem: str) -> str:
    """Return the 명창 if the 3rd dash-field is a Hangul name, else 'unknown'.

    Songs without a singer in the filename are pooled into ONE 'unknown'
    speaker rather than one-per-song: per-song ids would create ~122 singleton
    speakers whose embeddings can never be learned from a handful of utterances.
    """
    parts = stem.split("-")
    if len(parts) >= 3 and re.fullmatch(r"[가-힣]{2,4}", parts[2]):
        return parts[2]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: build speaker map")
    parser.add_argument("--set", default="pilot_6s", help="data/{set} working dir")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    records = read_jsonl(out_dir / "segments.jsonl")

    item2spk = {r["item_name"]: singer_from_stem(r["song_stem"]) for r in records}
    speakers = sorted(set(item2spk.values()))

    with open(out_dir / "spk_map.json", "w", encoding="utf-8") as f:
        json.dump({"speakers": speakers, "item2spk": item2spk},
                  f, ensure_ascii=False, indent=1)

    named = sum(1 for s in speakers if not s.startswith("unknown_"))
    print(f"발화 {len(item2spk)}개 → 화자 {len(speakers)}명 "
          f"(명창 {named}명 + unknown {len(speakers) - named}명)")
    print(f"저장: {out_dir / 'spk_map.json'}")


if __name__ == "__main__":
    main()
