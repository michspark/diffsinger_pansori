#!/usr/bin/env python3
"""Transcribe every wav in a segment dir with Whisper large-v3 -> lyrics.jsonl.

Thin driver around preprocess/03_transcribe.py's model functions. Used to
re-transcribe the 6s re-split so we can re-run the MFA alignment check on
short utterances and compare against the 15s baseline.
"""

import json
import sys
from pathlib import Path

import librosa

# Reuse the tested model/transcribe helpers from the Stage 3 script.
PREP = Path(__file__).resolve().parent.parent / "preprocess"
sys.path.insert(0, str(PREP))
import importlib.util

spec = importlib.util.spec_from_file_location("transcribe03", PREP / "03_transcribe.py")
t03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t03)

WHISPER_SAMPLE_RATE = 16000


def main() -> None:
    seg_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    records = [json.loads(line) for line in open(seg_dir / "segments.jsonl")]
    print(f"transcribing {len(records)} segments")

    model, processor, device, torch_dtype = t03.load_model()

    with open(out_path, "w", encoding="utf-8") as out_file:
        for index, record in enumerate(records, start=1):
            wav_path = seg_dir / "wavs" / f"{record['item_name']}.wav"
            audio, _ = librosa.load(wav_path, sr=WHISPER_SAMPLE_RATE, mono=True)
            result = t03.transcribe(model, processor, device, torch_dtype, audio)
            row = {
                "item_name": record["item_name"],
                "duration": record["duration"],
                "text": result["text"].strip(),
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 40 == 0 or index == len(records):
                print(f"  [{index}/{len(records)}]")

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
