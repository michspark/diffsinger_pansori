#!/usr/bin/env python3
"""Render the MFA alignment result as one self-contained HTML page.

Each segment gets its audio plus a proportional timeline of the aligned phones,
so the collapse pattern (phones pinned to MFA's 0.03s floor, followed by a long
"silence" swallowing the sustained note) is visible at a glance and checkable
by ear.

Run inside the `aligner` conda env (needs praatio).
"""

import base64
import csv
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from praatio import textgrid as tgio

CHECK_DIR = Path(__file__).resolve().parent

# Which alignment run to render. Defaults to the original 15s pilot; pass a tag
# (e.g. "6s") to render aligned_6s/ + corpus_6s/ against data/pilot_6s/.
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
_suffix = f"_{TAG}" if TAG else ""
_pilot = f"pilot_{TAG}" if TAG else "pilot"
ALIGNED_DIR = CHECK_DIR / f"aligned{_suffix}"
CORPUS_DIR = CHECK_DIR / f"corpus{_suffix}"
SEGMENTS = CHECK_DIR.parent / "data" / _pilot / "segments.jsonl"

SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}
FLOOR_SEC = 0.035      # MFA's minimum phone duration; at/below this it gave up
BIG_SILENCE_SEC = 1.0  # a silence this long mid-phrase means abandoned audio


def mp3_data_uri(wav_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", "96k", str(mp3_path)],
            check=True,
        )
        encoded = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
    finally:
        mp3_path.unlink(missing_ok=True)
    return f"data:audio/mpeg;base64,{encoded}"


def load_mapping() -> Dict[str, str]:
    """mfa_ipa -> coda symbol, for showing both notations side by side."""
    mapping = {}
    with open(CHECK_DIR / "ipa_to_coda.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["mfa_ipa"]] = row["coda"]
    return mapping


def build_rows() -> List[Dict]:
    durations = {
        json.loads(line)["item_name"]: json.loads(line)["duration"]
        for line in open(SEGMENTS, encoding="utf-8")
    }
    mapping = load_mapping()
    rows = []

    for tg_path in sorted(ALIGNED_DIR.glob("*.TextGrid")):
        name = tg_path.stem
        clip = durations[name]
        grid = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=True)

        intervals = []
        floor_count = speech_count = 0
        big_silence = 0.0
        for entry in grid.getTier("phones").entries:
            label = entry.label.strip()
            span = entry.end - entry.start
            is_sil = label.lower() in SILENCE_LABELS
            if is_sil:
                kind = "sil"
                if span >= BIG_SILENCE_SEC:
                    big_silence += span
                    kind = "bigsil"
            else:
                speech_count += 1
                if span <= FLOOR_SEC:
                    floor_count += 1
                    kind = "floor"
                else:
                    kind = "ok"
            intervals.append({
                "label": label or "·", "coda": mapping.get(label, ""),
                "start": entry.start, "dur": span, "kind": kind,
                "pct": 100.0 * span / clip,
            })

        # Word tier: humans judge a mis-timed word far more reliably than a
        # mis-timed phone, so this is the tier to click on first.
        words = []
        for entry in grid.getTier("words").entries:
            label = entry.label.strip()
            if not label or label.lower() in SILENCE_LABELS:
                continue
            words.append({
                "label": label,
                "start": entry.start,
                "dur": entry.end - entry.start,
            })

        rows.append({
            "name": name,
            "clip": clip,
            "words": words,
            "text": (CORPUS_DIR / f"{name}.lab").read_text(encoding="utf-8").strip(),
            "audio": mp3_data_uri(CORPUS_DIR / f"{name}.wav"),
            "intervals": intervals,
            "floor_pct": 100.0 * floor_count / max(1, speech_count),
            "big_silence": big_silence,
            "verdict": "collapsed" if big_silence >= BIG_SILENCE_SEC else "ok",
        })
    return rows


def render(rows: List[Dict]) -> str:
    cards = []
    for card_index, row in enumerate(sorted(rows, key=lambda r: r["clip"])):
        # Every clickable element carries the slice MFA assigned to it, so a
        # click plays exactly that span and nothing else.
        bars = "".join(
            f'<div class="iv {iv["kind"]}" data-start="{iv["start"]:.4f}" '
            f'data-dur="{iv["dur"]:.4f}" style="width:{iv["pct"]:.3f}%" '
            f'title="{html.escape(iv["label"])} → {html.escape(iv["coda"])}  '
            f'{iv["start"]:.2f}s +{iv["dur"]:.3f}s"></div>'
            for iv in row["intervals"]
        )
        word_chips = "".join(
            f'<span class="word" data-start="{w["start"]:.4f}" data-dur="{w["dur"]:.4f}">'
            f'{html.escape(w["label"])}<em>{w["dur"]:.2f}s</em></span>'
            for w in row["words"]
        ) or '<span class="none">(no words tier)</span>'
        phone_chips = "".join(
            f'<span class="chip {iv["kind"]}" data-start="{iv["start"]:.4f}" '
            f'data-dur="{iv["dur"]:.4f}">{html.escape(iv["label"])}'
            f'<em>{iv["dur"]:.2f}</em></span>'
            for iv in row["intervals"] if iv["kind"] != "sil"
        )
        badge = "bad" if row["verdict"] == "collapsed" else "good"
        cards.append(f"""
<article class="card" data-card="{card_index}">
  <header>
    <code>{html.escape(row['name'])}</code>
    <span class="badge">{row['clip']:.2f}s</span>
    <span class="badge">floor {row['floor_pct']:.0f}%</span>
    <span class="badge">1s+ silence {row['big_silence']:.2f}s</span>
    <span class="badge {badge}">{row['verdict']}</span>
  </header>
  <div class="lyric">{html.escape(row['text'])}</div>
  <audio controls preload="none" src="{row['audio']}"></audio>
  <div class="track">{bars}<div class="cursor"></div></div>
  <div class="hint">Click a word to play <b>only the span MFA assigned to it</b>. If the lyrics and what you hear agree, the alignment is right.</div>
  <div class="words">{word_chips}</div>
  <details><summary>per phoneme ({len([i for i in row['intervals'] if i['kind']!='sil'])})</summary>
    <div class="chips">{phone_chips}</div>
  </details>
</article>""")

    return f"""<title>MFA pansori alignment review</title>
<style>
  :root {{ --bg:#fff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7; --card:#fafafa;
           --ok:#2563eb; --floor:#dc2626; --sil:#d4d4d8; --bigsil:#f59e0b; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1115; --fg:#e4e4e7; --muted:#a1a1aa; --line:#27272a; --card:#18181b;
             --ok:#60a5fa; --floor:#f87171; --sil:#3f3f46; --bigsil:#fbbf24; }} }}
  :root[data-theme="dark"] {{ --bg:#0f1115; --fg:#e4e4e7; --muted:#a1a1aa; --line:#27272a;
             --card:#18181b; --ok:#60a5fa; --floor:#f87171; --sil:#3f3f46; --bigsil:#fbbf24; }}
  :root[data-theme="light"] {{ --bg:#fff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7;
             --card:#fafafa; --ok:#2563eb; --floor:#dc2626; --sil:#d4d4d8; --bigsil:#f59e0b; }}
  body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:2rem 1.25rem;
          max-width:64rem; line-height:1.6;
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif; }}
  h1 {{ font-size:1.4rem; margin:0 0 .3rem; }}
  .legend {{ color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; }}
  .key {{ display:inline-block; width:.8rem; height:.8rem; border-radius:2px;
          vertical-align:-1px; margin:0 .2rem 0 .8rem; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:.7rem;
           padding:1rem 1.1rem; margin-bottom:1rem; }}
  header {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; }}
  code {{ font-size:.8rem; color:var(--muted); }}
  .badge {{ font-size:.72rem; padding:.1rem .5rem; border-radius:999px;
            border:1px solid var(--line); color:var(--muted); }}
  .badge.bad {{ background:var(--floor); color:#fff; border-color:transparent; }}
  .badge.good {{ background:var(--ok); color:#fff; border-color:transparent; }}
  .lyric {{ margin:.5rem 0 .6rem; overflow-wrap:anywhere; }}
  audio {{ width:100%; margin-bottom:.6rem; }}
  .track {{ display:flex; height:1.6rem; border-radius:4px; overflow:hidden;
            border:1px solid var(--line); position:relative; cursor:pointer; }}
  .iv {{ height:100%; }}
  .cursor {{ position:absolute; top:0; bottom:0; width:2px; background:var(--fg);
             left:0; opacity:0; pointer-events:none; }}
  .hint {{ font-size:.78rem; color:var(--muted); margin:.7rem 0 .4rem; }}
  .words {{ display:flex; flex-wrap:wrap; gap:.3rem; }}
  .word {{ font-size:.95rem; padding:.15rem .55rem; border-radius:.4rem; cursor:pointer;
           border:1px solid var(--line); background:var(--bg); user-select:none; }}
  .word:hover {{ border-color:var(--ok); }}
  .word.playing, .chip.playing {{ background:var(--ok); color:#fff; border-color:transparent; }}
  .word em {{ font-style:normal; color:var(--muted); font-size:.7rem; margin-left:.3rem; }}
  .word.playing em {{ color:#fff; opacity:.8; }}
  .none {{ color:var(--muted); font-size:.8rem; }}
  details {{ margin-top:.7rem; }}
  summary {{ cursor:pointer; font-size:.8rem; color:var(--muted); }}
  .chip {{ cursor:pointer; }}
  .iv.ok {{ background:var(--ok); }}
  .iv.floor {{ background:var(--floor); }}
  .iv.sil {{ background:var(--sil); }}
  .iv.bigsil {{ background:var(--bigsil); }}
  .chips {{ margin-top:.6rem; display:flex; flex-wrap:wrap; gap:.25rem; }}
  .chip {{ font-size:.75rem; padding:.05rem .4rem; border-radius:4px;
           border:1px solid var(--line); font-variant-numeric:tabular-nums; }}
  .chip em {{ font-style:normal; color:var(--muted); margin-left:.25rem; font-size:.68rem; }}
  .chip.floor {{ border-color:var(--floor); color:var(--floor); }}
</style>
<h1>MFA pansori alignment review</h1>
<div class="legend">
  <span class="key" style="background:var(--ok)"></span>normal phoneme
  <span class="key" style="background:var(--floor)"></span>0.03 s floor (the aligner gave up)
  <span class="key" style="background:var(--sil)"></span>short silence
  <span class="key" style="background:var(--bigsil)"></span>silence of 1 s or more (a discarded span)
  <br>Ascending by length. Hover a bar to see the IPA -&gt; Coda correspondence.
</div>
{''.join(cards)}
<script>
document.querySelectorAll('.card').forEach(function (card) {{
  var audio = card.querySelector('audio');
  var cursor = card.querySelector('.cursor');
  var track = card.querySelector('.track');
  var clipEnd = null;          // stop point while playing a single slice
  var clipDuration = null;

  // Play only the span a clicked element was aligned to.
  function playSlice(element) {{
    var start = parseFloat(element.dataset.start);
    var dur = parseFloat(element.dataset.dur);
    if (isNaN(start)) return;
    clipEnd = start + dur;
    audio.currentTime = start;
    audio.play();
    card.querySelectorAll('.playing').forEach(function (e) {{
      e.classList.remove('playing');
    }});
    element.classList.add('playing');
  }}

  card.querySelectorAll('.word, .chip, .iv').forEach(function (element) {{
    element.addEventListener('click', function (event) {{
      event.stopPropagation();
      playSlice(element);
    }});
  }});

  // Pressing play on the audio control itself means "play the whole thing".
  audio.addEventListener('play', function () {{
    if (clipDuration === null && isFinite(audio.duration)) clipDuration = audio.duration;
  }});
  audio.addEventListener('seeked', function () {{
    // A manual scrub cancels the single-slice stop point.
    if (clipEnd !== null && Math.abs(audio.currentTime - (clipEnd)) > 30) clipEnd = null;
  }});

  audio.addEventListener('timeupdate', function () {{
    // Stop at the end of a clicked slice.
    if (clipEnd !== null && audio.currentTime >= clipEnd) {{
      audio.pause();
      clipEnd = null;
      return;
    }}
    // Karaoke cursor + highlight of whatever is currently sounding.
    var total = clipDuration || audio.duration;
    if (isFinite(total) && total > 0) {{
      cursor.style.opacity = 1;
      cursor.style.left = (100 * audio.currentTime / total) + '%';
    }}
    var now = audio.currentTime;
    card.querySelectorAll('.word').forEach(function (w) {{
      var s = parseFloat(w.dataset.start), d = parseFloat(w.dataset.dur);
      w.classList.toggle('playing', now >= s && now < s + d);
    }});
  }});

  audio.addEventListener('pause', function () {{
    card.querySelectorAll('.playing').forEach(function (e) {{
      e.classList.remove('playing');
    }});
  }});

  // Clicking empty track area seeks there.
  track.addEventListener('click', function (event) {{
    if (event.target !== track) return;
    var rect = track.getBoundingClientRect();
    var total = clipDuration || audio.duration;
    if (!isFinite(total)) return;
    clipEnd = null;
    audio.currentTime = total * (event.clientX - rect.left) / rect.width;
    audio.play();
  }});
}});
</script>
"""


def main() -> None:
    rows = build_rows()
    # Review HTML lives under pansori/review/ (CHECK_DIR = pansori/alignment).
    review_dir = CHECK_DIR.parent / "review"
    review_dir.mkdir(exist_ok=True)
    out = review_dir / f"alignment_review{_suffix}.html"
    out.write_text(render(rows), encoding="utf-8")
    print(f"saved: {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
