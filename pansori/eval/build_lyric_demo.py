#!/usr/bin/env python3
"""Render changbon-corrected lyrics as one self-contained HTML page, with audio.

Every verdict on the lyricbook so far has been decided by a metric -- MFA
alignment quality (LYRICBOOK_VERDICT.md) and synthesis distance after training
(LYRICS_AB.md) -- and both documents close by saying the same thing: nobody has
listened. This page is that missing step. It puts the recording next to both
candidate texts so the only question left is which one the singer actually sang.

Sampling is stratified over match_score, because the score bands fail in
different ways: near 1.0 the changbon merely respaces Whisper, while below 0.55
the span is often twice as long as what was sung (measured: p90 length ratio
2.00). A sample drawn flat would be mostly the easy end.

Blind mode hides which side is which and swaps them per item, so a judgement
can be recorded without knowing which arm it favours. Votes live in the
browser's localStorage and export as JSON.

Run inside the `pansori` env (stdlib only, plus ffmpeg on PATH).
"""

import argparse
import base64
import difflib
import html
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess"))

from pansori_lib import PANSORI_DIR, read_jsonl  # noqa: E402

HANGUL = re.compile(r"[가-힣]")

# The same bands the span-length analysis used, so the two can be read together.
BANDS = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.90), (0.90, 1.001)]


def mp3_data_uri(wav_path: Path, bitrate: str) -> str:
    """Same trick as alignment/build_review.py -- one file, no sidecar assets."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(
            mp3_path.read_bytes()).decode("ascii")
    finally:
        mp3_path.unlink(missing_ok=True)


def diff_html(left: str, right: str) -> tuple:
    """Mark the syllables that differ, on both sides."""
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    out_left, out_right = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        piece_left, piece_right = html.escape(left[i1:i2]), html.escape(right[j1:j2])
        if tag == "equal":
            out_left.append(piece_left)
            out_right.append(piece_right)
            continue
        if piece_left:
            out_left.append(f'<mark>{piece_left}</mark>')
        if piece_right:
            out_right.append(f'<mark>{piece_right}</mark>')
    return "".join(out_left), "".join(out_right)


def pick(rows, n, seed):
    """Stratify over match_score so every failure mode is represented."""
    rng = random.Random(seed)
    per_band, extra = divmod(n, len(BANDS))
    picked = []
    for index, (low, high) in enumerate(BANDS):
        pool = [r for r in rows if low <= r["match_score"] < high]
        want = per_band + (1 if index < extra else 0)
        picked += rng.sample(pool, min(want, len(pool)))
    picked.sort(key=lambda r: r["match_score"])
    return picked


def build_cards(picked, segments, wav_dir, bitrate, seed):
    rng = random.Random(seed + 1)
    cards = []
    for row in picked:
        item = row["item_name"]
        wav = wav_dir / f"{item}.wav"
        if not wav.exists():
            print(f"  missing audio, skipped: {item}", file=sys.stderr)
            continue
        whisper_html, book_html = diff_html(row["text_whisper"], row["text"])
        reference = row["book_ref"] or {}
        cards.append({
            "item": item,
            "song": segments.get(item, {}).get("song_stem", ""),
            "duration": row["duration"],
            "score": row["match_score"],
            "density": row["song_anchor_density"],
            "book": reference.get("book", ""),
            "block": reference.get("block", ""),
            "whisper": row["text_whisper"],
            "bookText": row["text"],
            "whisperHtml": whisper_html,
            "bookHtml": book_html,
            "syllables": [len(HANGUL.findall(row["text_whisper"])),
                          len(HANGUL.findall(row["text"]))],
            "swap": rng.random() < 0.5,
            "audio": mp3_data_uri(wav, bitrate),
        })
        print(f"  {len(cards):>3}/{len(picked)}  {item}  score={row['match_score']:.2f}")
    return cards


def render(cards, meta) -> str:
    payload = json.dumps(cards, ensure_ascii=False)
    head = json.dumps(meta, ensure_ascii=False)
    return TEMPLATE.replace("__CARDS__", payload).replace("__META__", head)


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>사설 스냅 청취 검증</title>
<style>
:root{--ink:#1b1b1f;--dim:#6a6a76;--line:#e2e2e8;--bg:#fafafc;--card:#fff;
      --whisper:#0b5fa5;--book:#a4501a;--mark:#ffe08a;}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 80px;background:var(--bg);color:var(--ink);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0;
     border-bottom:1px solid var(--line);margin-bottom:18px;
     display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:7px;
       background:var(--card);cursor:pointer}
button:hover{border-color:#b9b9c4}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tally{margin-left:auto;color:var(--dim);font-size:13px;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:16px;margin-bottom:14px}
.meta{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
      color:var(--dim);font-size:12px;margin-bottom:10px}
.score{font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;
       background:#f0f0f5;padding:2px 8px;border-radius:20px}
audio{width:100%;height:34px;margin-bottom:12px}
.pair{display:flex;flex-direction:column;gap:8px}
.side{display:flex;gap:12px;align-items:baseline;padding:9px 12px;border-radius:8px;
      background:#f7f7fa;border-left:3px solid var(--line)}
.side.w{border-left-color:var(--whisper)}
.side.b{border-left-color:var(--book)}
.tag{flex:0 0 62px;font-size:11px;font-weight:700;letter-spacing:.04em;color:var(--dim)}
.side.w .tag{color:var(--whisper)} .side.b .tag{color:var(--book)}
.txt{font-size:17px;line-height:1.7}
mark{background:var(--mark);padding:0 1px;border-radius:2px}
.vote{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.vote .q{font-size:12px;color:var(--dim);margin-right:2px}
.vote button.picked{background:#1f7a3f;color:#fff;border-color:#1f7a3f}
body.blind .hide-blind{visibility:hidden}
body.blind .side.b,body.blind .side.w{border-left-color:var(--line)}
.ab{flex:0 0 62px;font-size:11px;font-weight:700;color:var(--dim);display:none}
body.blind .ab{display:block}
body.blind .tag{display:none}
/* DOM order is always whisper-then-changbon; blind mode flips it visually. */
.side.w{order:1} .side.b{order:2}
body.blind .card[data-swap="1"] .side.b{order:1}
body.blind .card[data-swap="1"] .side.w{order:2}
.vote .q{order:0} .vote button[data-v="whisper"]{order:1}
.vote button[data-v="book"]{order:2} .vote button[data-v="unsure"]{order:3}
body.blind .card[data-swap="1"] .vote button[data-v="book"]{order:1}
body.blind .card[data-swap="1"] .vote button[data-v="whisper"]{order:2}
.lb{display:none} body.blind .ln{display:none} body.blind .lb{display:inline}
.note{color:var(--dim);font-size:12px;margin-top:26px;line-height:1.7}
</style></head><body>
<div class="wrap">
<h1>사설 스냅 청취 검증</h1>
<div class="sub" id="sub"></div>

<div class="bar">
  <button id="blind">블라인드 모드</button>
  <button id="export">판정 내보내기</button>
  <button id="clear">판정 지우기</button>
  <span class="tally" id="tally"></span>
</div>

<div id="list"></div>

<div class="note">
녹음을 듣고 <b>실제로 부른 것에 더 가까운 쪽</b>을 고릅니다. 표기가 옳은지가 아니라
소리와 맞는지가 기준입니다 — 학습 라벨은 소리를 따라가야 하기 때문입니다.<br>
블라인드 모드는 좌우를 항목마다 무작위로 바꾸고 라벨·점수·강조를 가립니다.
판정은 브라우저에 저장되고, 내보내기로 JSON을 받습니다.
</div>
</div>

<script>
const CARDS = __CARDS__, META = __META__;
const KEY = "pansori-lyric-demo/" + META.built;
let votes = {};
try { votes = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { votes = {}; }

document.getElementById("sub").textContent =
  META.n + "개 항목 · match_score " + META.range + " 구간별 층화추출 · seed " + META.seed
  + " · " + META.source;

const list = document.getElementById("list");
for (const c of CARDS) {
  const abW = c.swap ? "B" : "A", abB = c.swap ? "A" : "B";
  const side = (kind, ab, tag, text) => `
    <div class="side ${kind}">
      <span class="ab">${ab}</span>
      <span class="tag">${tag}</span>
      <span class="txt">${text}</span>
    </div>`;
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `
    <div class="meta">
      <span class="score hide-blind">${c.score.toFixed(2)}</span>
      <span>${c.item}</span>
      <span>${c.duration.toFixed(2)}s</span>
      <span class="hide-blind">${c.book} blk${c.block}</span>
      <span class="hide-blind">음절 ${c.syllables[0]}→${c.syllables[1]}</span>
    </div>
    <audio controls preload="metadata" src="${c.audio}"></audio>
    <div class="pair">
      ${side("w", abW, "WHISPER", c.whisperHtml)}
      ${side("b", abB, "창본", c.bookHtml)}
    </div>
    <div class="vote">
      <span class="q">더 가까운 쪽:</span>
      <button data-v="whisper"><span class="ln">whisper</span><span class="lb">${abW}</span></button>
      <button data-v="book"><span class="ln">창본</span><span class="lb">${abB}</span></button>
      <button data-v="unsure">판단 불가</button>
    </div>`;
  el.querySelectorAll(".vote button").forEach(b => {
    b.onclick = () => { votes[c.item] = b.dataset.v; save(); paint(); };
  });
  el.dataset.item = c.item;
  el.dataset.swap = c.swap ? "1" : "0";
  list.appendChild(el);
}

function save() { try { localStorage.setItem(KEY, JSON.stringify(votes)); } catch (e) {} }

function paint() {
  const count = { whisper: 0, book: 0, unsure: 0 };
  for (const el of list.children) {
    const v = votes[el.dataset.item];
    if (v) count[v]++;
    el.querySelectorAll(".vote button").forEach(b =>
      b.classList.toggle("picked", b.dataset.v === v));
  }
  document.getElementById("tally").textContent =
    `whisper ${count.whisper} · 창본 ${count.book} · 불가 ${count.unsure}` +
    ` · 남음 ${CARDS.length - count.whisper - count.book - count.unsure}`;
}

document.getElementById("blind").onclick = e => {
  document.body.classList.toggle("blind");
  e.target.classList.toggle("on", document.body.classList.contains("blind"));
};
document.getElementById("clear").onclick = () => {
  if (!confirm("판정을 모두 지웁니다.")) return;
  votes = {}; save(); paint();
};
document.getElementById("export").onclick = () => {
  const out = CARDS.map(c => ({
    item_name: c.item, song: c.song, match_score: c.score,
    text_whisper: c.whisper, text_book: c.bookText, vote: votes[c.item] || null,
  }));
  const blob = new Blob([JSON.stringify({ meta: META, votes: out }, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "lyric_listening_votes.json";
  a.click();
};

paint();
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="full")
    parser.add_argument("--lyrics", default="lyrics_snapped.jsonl")
    parser.add_argument("--data-root", default="",
                        help="override pansori/data (e.g. when run from a worktree)")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bitrate", default="96k")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else PANSORI_DIR / "data"
    set_dir = root / args.set
    rows = read_jsonl(set_dir / args.lyrics)
    segments = {r["item_name"]: r for r in read_jsonl(set_dir / "segments.jsonl")}

    # "Changed lyrics" means the Hangul changed -- 5.2% of adopted rows only
    # respace Whisper, and there is nothing to listen for in those.
    changed = [r for r in rows
               if r["source"] != "whisper"          # 03b says "book", 03c says "hybrid"
               and "".join(HANGUL.findall(r["text_whisper"]))
               != "".join(HANGUL.findall(r["text"]))]
    print(f"{len(rows):,} rows | rewritten {sum(1 for r in rows if r['source']!='whisper'):,}"
          f" | text actually changed {len(changed):,}")

    picked = pick(changed, args.n, args.seed)
    print(f"sampled {len(picked)} across {len(BANDS)} score bands; encoding audio")
    cards = build_cards(picked, segments, set_dir / "wavs", args.bitrate, args.seed)
    if not cards:
        raise SystemExit("no audio found; is data/{set}/wavs populated?")

    meta = {"n": len(cards), "seed": args.seed,
            "range": f"{BANDS[0][0]:.2f}–1.00",
            "source": args.lyrics, "built": f"{args.lyrics}:{args.seed}:{len(cards)}"}
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "out" / "lyric_demo.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cards, meta), encoding="utf-8")
    print(f"\n{out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
