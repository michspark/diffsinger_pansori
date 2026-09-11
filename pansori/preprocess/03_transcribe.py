#!/usr/bin/env python3
"""Stage 3: transcribe every segment with Whisper large-v3 -> lyrics.jsonl.

The raw transcription is kept for review, and a repeat-collapsed version
(pansori_lib.normalize_repeats) is what downstream alignment uses -- a held
sung note makes Whisper emit '지이이이…이(x200)', which the collapse folds back
to '지이'. Whisper is imperfect on pansori; the goal is that the *phonemes*
line up in time, not perfect lyrics (see pansori/docs/SEGMENTATION_VERDICT.md).

Input : data/{set}/segments.jsonl + data/{set}/wavs/*.wav
Output: data/{set}/lyrics.jsonl   {item_name, duration, text_raw, text}

Run in the `base` env (torch + transformers).
"""

import argparse

import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from pansori_lib import normalize_repeats, read_jsonl, write_jsonl, set_dir

MODEL_ID = "openai/whisper-large-v3"
WHISPER_SAMPLE_RATE = 16000  # Whisper's encoder is trained on 16kHz


def load_model():
    """Load Whisper large-v3 onto the GPU in fp16.

    The transformers ASR *pipeline* is avoided: it imports the (broken)
    torchcodec package unconditionally. Direct model calls are fine because
    every segment is under Whisper's 30s window (no long-form chunking needed).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device, torch_dtype


def transcribe(model, processor, device, torch_dtype, audio: np.ndarray) -> str:
    """Return the Korean transcription of one segment."""
    inputs = processor(audio, sampling_rate=WHISPER_SAMPLE_RATE,
                       return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device, dtype=torch_dtype)
    attention_mask = inputs.attention_mask.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            input_features, attention_mask=attention_mask,
            language="korean", task="transcribe",
            return_timestamps=True, max_new_tokens=440)
    decoded = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Whisper transcription")
    parser.add_argument("--set", default="pilot_6s", help="data/{set} working dir")
    args = parser.parse_args()

    out_dir = set_dir(args.set)
    records = read_jsonl(out_dir / "segments.jsonl")
    print(f"{len(records)}개 세그먼트 전사 시작 ({MODEL_ID})")

    model, processor, device, torch_dtype = load_model()

    rows = []
    for index, record in enumerate(records, start=1):
        wav_path = out_dir / "wavs" / f"{record['item_name']}.wav"
        audio, _ = librosa.load(wav_path, sr=WHISPER_SAMPLE_RATE, mono=True)
        raw = transcribe(model, processor, device, torch_dtype, audio)
        rows.append({
            "item_name": record["item_name"],
            "duration": record["duration"],
            "text_raw": raw,
            "text": normalize_repeats(raw),   # collapse sustained-note repeats
        })
        if index % 40 == 0 or index == len(records):
            print(f"  [{index}/{len(records)}]")

    write_jsonl(out_dir / "lyrics.jsonl", rows)
    collapsed = sum(1 for r in rows if r["text"] != r["text_raw"])
    print(f"저장: {out_dir / 'lyrics.jsonl'}  (반복접기 적용 {collapsed}개)")


if __name__ == "__main__":
    main()
