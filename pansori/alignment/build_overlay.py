#!/usr/bin/env python3
"""Overlay ROSVOT notes and MFA phonemes on one shared time axis, per segment.

Top strip = piano roll of the sliced ROSVOT notes (Stage 5 output).
Bottom strip = MFA phoneme alignment (Stage 4).
Both are absolutely positioned by time, so a note at t=0.8s sits directly above
the phoneme sounding at t=0.8s. Click any note or phoneme to play just that
span; this is how we eyeball whether the melody and the syllables agree.

Run in the `aligner` env (needs praatio + pretty_midi).
"""

import base64
import html
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import pretty_midi
from praatio import textgrid as tgio

CHECK_DIR = Path(__file__).resolve().parent
PANSORI = CHECK_DIR.parent
DATA = PANSORI / "data" / "pilot_6s"
MIDI_DIR = Path("/home/sangheon/Desktop/Pansori_Data/rosvot_midi")

SILENCE = {"", "sil", "sp", "spn", "<eps>"}
FLOOR_SEC = 0.035

# Curated sample: wordy (well-aligned) + slur-heavy + pure melisma.
# For melisma segments MFA only aligned the normalized lyric, so we point at
# aligned_norm; the note track is the interesting part there.
SAMPLE = [
    ("039d6b6e_0000", "aligned_6s"),   # wordy, notes≈vowels
    ("01ed19bb_0007", "aligned_6s"),   # notes > vowels -> slur territory
    ("01ed19bb_0019", "aligned_6s"),
    ("039d6b6e_0001", "aligned_6s"),
    ("02b8fd02_0011", "aligned_norm"),  # pure melisma: MFA silence vs ROSVOT note
    ("03abdd70_0069", "aligned_norm"),
]


def mp3_data_uri(wav_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", "96k", str(mp3_path)],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(mp3_path.read_bytes()).decode()
    finally:
        mp3_path.unlink(missing_ok=True)


def slice_notes(stem: str, start_sec: float, end_sec: float):
    midi = pretty_midi.PrettyMIDI(str(MIDI_DIR / f"{stem}_vocal.mid"))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end <= start_sec or note.start >= end_sec:
                continue
            notes.append((
                note.pitch,
                max(note.start, start_sec) - start_sec,
                min(note.end, end_sec) - start_sec,
            ))
    notes.sort(key=lambda x: x[1])
    return notes


def phones(item_name: str, aligned_subdir: str):
    tg_path = CHECK_DIR / aligned_subdir / f"{item_name}.TextGrid"
    if not tg_path.exists():
        return None
    grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    out = []
    for entry in grid.getTier("phones").entries:
        label = entry.label.strip()
        is_sil = label.lower() in SILENCE
        span = entry.end - entry.start
        kind = "sil" if is_sil else ("floor" if span <= FLOOR_SEC else "ok")
        out.append({"label": label or "·", "start": entry.start,
                    "dur": span, "kind": kind})
    return out


def wav_for(item_name: str, aligned_subdir: str) -> Path:
    corpus = "corpus_norm" if aligned_subdir == "aligned_norm" else "corpus_6s"
    return CHECK_DIR / corpus / f"{item_name}.wav"


def lab_for(item_name: str, aligned_subdir: str) -> str:
    corpus = "corpus_norm" if aligned_subdir == "aligned_norm" else "corpus_6s"
    p = CHECK_DIR / corpus / f"{item_name}.lab"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def render_segment(item_name, seg, notes, phone_list, audio_uri, lab, note_only) -> str:
    dur = seg["duration"]

    # Piano-roll vertical mapping: min..max pitch -> 88%..8% top.
    pitches = [p for p, _, _ in notes] or [60]
    lo, hi = min(pitches), max(pitches)
    span = max(1, hi - lo)

    def top_pct(pitch):
        return 8 + (1 - (pitch - lo) / span) * 78

    note_divs = []
    for pitch, start, end in notes:
        name = librosa.midi_to_note(pitch)
        hue = int(220 - 200 * (pitch - lo) / span)  # low=red-ish, high=blue-ish
        note_divs.append(
            f'<div class="note" data-start="{start:.4f}" data-dur="{end-start:.4f}" '
            f'style="left:{100*start/dur:.3f}%;width:{100*(end-start)/dur:.3f}%;'
            f'top:{top_pct(pitch):.1f}%;background:hsl({hue},70%,55%)" '
            f'title="{name} ({pitch})  {start:.2f}-{end:.2f}s">{html.escape(name)}</div>'
        )

    phone_divs = ""
    if phone_list:
        phone_divs = "".join(
            f'<div class="ph {p["kind"]}" data-start="{p["start"]:.4f}" '
            f'data-dur="{p["dur"]:.4f}" '
            f'style="left:{100*p["start"]/dur:.3f}%;width:{100*p["dur"]/dur:.3f}%" '
            f'title="{html.escape(p["label"])}  {p["start"]:.2f}s +{p["dur"]:.3f}s">'
            f'{html.escape(p["label"])}</div>'
            for p in phone_list
        )
    phone_row = (
        f'<div class="phones">{phone_divs}</div>' if phone_list
        else '<div class="phones empty">MFA 정렬: 지속음이 침묵 처리됨 — 위 ROSVOT 음표가 실제 소리</div>'
    )

    # second-tick scale
    ticks = "".join(
        f'<span style="left:{100*s/dur:.2f}%">{s}s</span>'
        for s in range(int(dur) + 1)
    )

    note_note = "" if not note_only else \
        ' <span class="tag">MFA 없음 · ROSVOT만</span>'

    return f"""
<article class="seg">
  <header>
    <code>{html.escape(item_name)}</code>
    <span class="badge">{dur:.2f}s · {html.escape(seg['mode'])}</span>
    <span class="badge">음표 {len(notes)}개 · 음역 {librosa.midi_to_note(lo)}~{librosa.midi_to_note(hi)}</span>
    {note_note}
  </header>
  <div class="lyric">{html.escape(lab)}</div>
  <audio controls preload="none" src="{audio_uri}"></audio>
  <div class="timeline">
    <div class="notes">{''.join(note_divs)}<div class="cursor"></div></div>
    {phone_row}
    <div class="ticks">{ticks}</div>
  </div>
</article>"""


def main() -> None:
    segments = {
        json.loads(l)["item_name"]: json.loads(l)
        for l in open(DATA / "segments.jsonl")
    }

    cards = []
    for item_name, aligned_subdir in SAMPLE:
        seg = segments[item_name]
        notes = slice_notes(seg["song_stem"], seg["song_start"], seg["song_end"])
        phone_list = phones(item_name, aligned_subdir)
        cards.append(render_segment(
            item_name, seg, notes, phone_list,
            mp3_data_uri(wav_for(item_name, aligned_subdir)),
            lab_for(item_name, aligned_subdir),
            note_only=(aligned_subdir == "aligned_norm"),
        ))

    page = f"""<title>ROSVOT 음표 × MFA 음소 겹쳐보기</title>
<style>
  :root {{ --bg:#fff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7; --card:#fafafa;
           --ok:#2563eb; --floor:#dc2626; --sil:#e4e4e7; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1115; --fg:#e4e4e7; --muted:#a1a1aa; --line:#27272a; --card:#18181b;
             --ok:#60a5fa; --floor:#f87171; --sil:#2a2a30; }} }}
  :root[data-theme="dark"] {{ --bg:#0f1115; --fg:#e4e4e7; --muted:#a1a1aa; --line:#27272a;
             --card:#18181b; --ok:#60a5fa; --floor:#f87171; --sil:#2a2a30; }}
  :root[data-theme="light"] {{ --bg:#fff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7;
             --card:#fafafa; --ok:#2563eb; --floor:#dc2626; --sil:#e4e4e7; }}
  body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem;
          max-width:70rem; line-height:1.5;
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif; }}
  h1 {{ font-size:1.4rem; margin:0 0 .3rem; }}
  .legend {{ color:var(--muted); font-size:.85rem; margin-bottom:1.6rem; }}
  .seg {{ background:var(--card); border:1px solid var(--line); border-radius:.7rem;
          padding:1rem 1.1rem; margin-bottom:1.1rem; }}
  header {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }}
  code {{ font-size:.8rem; color:var(--muted); }}
  .badge {{ font-size:.72rem; padding:.1rem .5rem; border-radius:999px;
            border:1px solid var(--line); color:var(--muted); }}
  .tag {{ font-size:.72rem; padding:.1rem .5rem; border-radius:999px;
          background:var(--floor); color:#fff; }}
  .lyric {{ margin:.5rem 0 .6rem; overflow-wrap:anywhere; }}
  audio {{ width:100%; margin-bottom:.7rem; }}
  .timeline {{ position:relative; }}
  .notes {{ position:relative; height:120px; border:1px solid var(--line);
            border-radius:5px 5px 0 0; background:
            repeating-linear-gradient(0deg,transparent,transparent 19px,var(--line) 19px,var(--line) 20px);
            overflow:hidden; }}
  .note {{ position:absolute; height:15px; border-radius:3px; cursor:pointer;
           font-size:.62rem; color:#000; line-height:15px; text-align:center;
           overflow:hidden; white-space:nowrap; box-shadow:0 0 0 1px rgba(0,0,0,.15); }}
  .note:hover {{ outline:2px solid var(--fg); z-index:5; }}
  .phones {{ position:relative; height:30px; border:1px solid var(--line); border-top:0;
             border-radius:0 0 5px 5px; }}
  .phones.empty {{ display:flex; align-items:center; justify-content:center;
                   font-size:.78rem; color:var(--floor); height:34px; }}
  .ph {{ position:absolute; top:0; height:100%; box-sizing:border-box;
         border-right:1px solid var(--bg); font-size:.72rem; line-height:30px;
         text-align:center; overflow:hidden; cursor:pointer; color:#fff; }}
  .ph.ok {{ background:var(--ok); }}
  .ph.floor {{ background:var(--floor); }}
  .ph.sil {{ background:var(--sil); color:var(--muted); }}
  .ph:hover {{ filter:brightness(1.15); }}
  .cursor {{ position:absolute; top:0; bottom:0; width:2px; background:var(--fg);
             left:0; opacity:0; pointer-events:none; z-index:8; }}
  .ticks {{ position:relative; height:1.1rem; }}
  .ticks span {{ position:absolute; font-size:.62rem; color:var(--muted);
                 transform:translateX(-50%); }}
</style>
<h1>ROSVOT 음표 × MFA 음소 겹쳐보기</h1>
<div class="legend">
  위=ROSVOT 음표(피아노롤, 높을수록 위·색은 음높이), 아래=MFA 음소 정렬.
  같은 시간축이라 세로로 겹칩니다. <b>음표나 음소를 클릭하면 그 구간만 재생</b>됩니다.<br>
  파란 음소=정상, 빨강=0.03초 하한선(정렬 포기), 회색=침묵.
</div>
{''.join(cards)}
<script>
document.querySelectorAll('.seg').forEach(function (seg) {{
  var audio = seg.querySelector('audio');
  var cursor = seg.querySelector('.cursor');
  var clipEnd = null;
  function play(el) {{
    var s = parseFloat(el.dataset.start), d = parseFloat(el.dataset.dur);
    if (isNaN(s)) return;
    clipEnd = s + d; audio.currentTime = s; audio.play();
  }}
  seg.querySelectorAll('.note,.ph').forEach(function (el) {{
    el.addEventListener('click', function (e) {{ e.stopPropagation(); play(el); }});
  }});
  audio.addEventListener('timeupdate', function () {{
    if (clipEnd !== null && audio.currentTime >= clipEnd) {{ audio.pause(); clipEnd = null; return; }}
    var t = audio.duration;
    if (isFinite(t) && t > 0) {{ cursor.style.opacity = 1; cursor.style.left = (100*audio.currentTime/t) + '%'; }}
  }});
  audio.addEventListener('seeking', function () {{ clipEnd = null; }});
}});
</script>
"""
    out = PANSORI / "review" / "note_overlay.html"
    out.write_text(page, encoding="utf-8")
    print(f"저장: {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
