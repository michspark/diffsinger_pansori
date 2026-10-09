#!/usr/bin/env python3
"""Does MFA's alignment degrade with utterance length? Measured on the corpus, paired.

The 6 s cap on utterances (01_segment.py MAX_UTTERANCE_SEC) was set from a pilot of ten
clips (pansori/docs/SEGMENTATION_VERDICT.md §2). Those clips and their alignments are
gone, and 01_segment.py must not be re-run (CLAUDE.md rule 3). But segments.jsonl keeps
each utterance's absolute span in its source recording, and consecutive utterances of a
song that are separated by less than the 0.4 s pause threshold were cut only for length.
Merging such a run of 2-4 utterances rebuilds a 6-20 s span from the same audio, and
aligning it once as a whole gives a paired comparison: the same audio, the same lyrics,
the same aligner, only the length differs.

Metrics, per clip, from MFA's own TextGrid:
  floor_pct     non-silence phones at MFA's 0.03 s minimum, % of phones
  voiced_sil    % of ROSVOT-note-covered time that MFA labels silence (the quantity
                06_merge has to absorb; 52.2% of corpus time at 2-6 s)
  silence_pct   % of the clip labelled silence (edges included)
  long_silence  summed silence spans of 1 s or more (the pilot's metric)
  max_silence   the longest single silence span

Stages (all in the `aligner` env: praatio, librosa, soundfile and the mfa CLI):

    python pansori/experiments/long_utterance_mfa.py --stage build
    python pansori/experiments/long_utterance_mfa.py --stage align
    python pansori/experiments/long_utterance_mfa.py --stage measure

Writes only under pansori/data/full/long_mfa/ and pansori/docs/img/.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "eval"))

from pansori_lib import load_align_config, read_jsonl, set_dir, write_jsonl  # noqa: E402

PAUSE_SEC = 0.4          # 01_segment.py SILENCE_GAP_THRESHOLD_SEC
FLOOR_SEC = 0.031        # compare_lyrics_alignment.py; MFA's minimum is 0.03 s
LONG_SILENCE_SEC = 1.0
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}
BUCKETS = [(6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]
REF_BUCKETS = [(2, 3), (3, 4), (4, 5), (5, 6)]
MAX_PARTS = 4


def bucket_of(duration, buckets):
    for lo, hi in buckets:
        if lo <= duration < hi:
            return f"{lo}-{hi}"
    return None


# ------------------------------------------------------------------ build

def candidate_chains(segments, aligned_dir):
    """Every run of 2..MAX_PARTS consecutive, same-mode, aligned utterances with
    0 <= gap < PAUSE_SEC between neighbours."""
    have = {p.stem for p in aligned_dir.glob("*.TextGrid")}
    out = []
    for i in range(len(segments)):
        first = segments[i]
        if first["item_name"] not in have:
            continue
        parts, gaps = [first], []
        for j in range(i + 1, min(i + MAX_PARTS, len(segments))):
            prev, nxt = segments[j - 1], segments[j]
            gap = nxt["song_start"] - prev["song_end"]
            if (nxt["uuid"] != first["uuid"] or nxt["mode"] != first["mode"]
                    or nxt["item_name"] not in have or not 0 <= gap < PAUSE_SEC):
                break
            parts.append(nxt)
            gaps.append(round(gap, 4))
            span = nxt["song_end"] - first["song_start"]
            b = bucket_of(span, BUCKETS)
            if b:
                out.append({"parts": [p["item_name"] for p in parts], "gaps": list(gaps),
                            "uuid": first["uuid"], "song_stem": first["song_stem"],
                            "mode": first["mode"], "song_start": first["song_start"],
                            "song_end": nxt["song_end"], "duration": round(span, 4),
                            "bucket": b})
    return out


def draw(cands, per_bucket, seed):
    """Per bucket, round-robin over songs so no recording dominates; no utterance is
    used by two chains, so the chains are independent samples."""
    rng = random.Random(seed)
    used, chosen = set(), []
    by_bucket = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_bucket[c["bucket"]][c["uuid"]].append(c)
    # long buckets are the scarce ones, so they pick first
    for lo, hi in reversed(BUCKETS):
        songs = by_bucket.get(f"{lo}-{hi}", {})
        queues = {u: rng.sample(cs, len(cs)) for u, cs in songs.items()}
        order = rng.sample(sorted(queues), len(queues))
        taken = 0
        while taken < per_bucket and any(queues.values()):
            for u in order:
                if taken >= per_bucket:
                    break
                q = queues[u]
                while q:
                    c = q.pop()
                    if not used.intersection(c["parts"]):
                        used.update(c["parts"])
                        chosen.append(c)
                        taken += 1
                        break
    return chosen


def stage_build(args, full, work):
    from compare_lyrics_alignment import build_corpus, cut_clips
    from aligners import get_aligner

    segments = read_jsonl(full / "segments.jsonl")
    lyrics = {r["item_name"]: r["text"] for r in read_jsonl(full / "lyrics.jsonl")}
    cands = candidate_chains(segments, full / "aligned")
    chains = draw(cands, args.per_bucket, args.seed)
    for n, c in enumerate(sorted(chains, key=lambda c: (c["uuid"], c["song_start"]))):
        c["item_name"] = f"long_{n:05d}"
        c["text"] = " ".join(lyrics[p].strip() for p in c["parts"])
    write_jsonl(work / "chains.jsonl", chains)
    # cut_clips reads song_stem/song_start/song_end; build_corpus reads item_name + field
    cut_clips(chains, chains, work / "wavs")
    aligner = get_aligner("mfa", load_align_config())
    build_corpus(chains, work / "wavs", work / "mfa_corpus_long", "text", aligner)

    counts = defaultdict(int)
    for c in chains:
        counts[c["bucket"]] += 1
    print(f"candidates {len(cands)}, drawn {len(chains)}")
    for lo, hi in BUCKETS:
        print(f"  {lo}-{hi} s: {counts[f'{lo}-{hi}']}")


# ------------------------------------------------------------------ align

def stage_align(args, full, work):
    from aligners import get_aligner
    aligner = get_aligner("mfa", load_align_config())
    (work / "aligned").mkdir(exist_ok=True)
    aligner.run(work / "mfa_corpus_long", work / "aligned")


# ------------------------------------------------------------------ measure

def read_grid(path):
    from praatio import textgrid as tgio
    grid = tgio.openTextgrid(str(path), includeEmptyIntervals=True)
    phones, sil = [], []
    for iv in grid.getTier("phones").entries:
        (sil if iv.label.strip().lower() in SILENCE_LABELS else phones).append(
            (iv.start, iv.end))
    return phones, sil


def overlap(spans_a, spans_b):
    """Total overlap of two lists of (start, end) spans."""
    total = 0.0
    for a0, a1 in spans_a:
        for b0, b1 in spans_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total


def tally(phones, sil, notes, duration):
    """The raw counts every metric is a ratio of, so parts can be pooled exactly."""
    lens = [b - a for a, b in phones]
    sils = [b - a for a, b in sil]
    return {
        "n_phones": len(lens),
        "n_floor": sum(x <= FLOOR_SEC for x in lens),
        "phone_time": sum(lens),
        "silence": sum(sils),
        "long_silence": sum(x for x in sils if x >= LONG_SILENCE_SEC),
        "max_silence": max(sils, default=0.0),
        "note_time": sum(b - a for a, b in notes),
        "voiced_silence": overlap(sil, notes),
        "duration": duration,
    }


def ratios(t):
    return {
        "floor_pct": 100 * t["n_floor"] / t["n_phones"] if t["n_phones"] else float("nan"),
        "voiced_sil": 100 * t["voiced_silence"] / t["note_time"] if t["note_time"] else float("nan"),
        "silence_pct": 100 * t["silence"] / t["duration"],
        "long_silence": t["long_silence"],
        "max_silence": t["max_silence"],
        "mean_phone_ms": 1000 * t["phone_time"] / t["n_phones"] if t["n_phones"] else float("nan"),
    }


def pool(tallies):
    out = {k: sum(t[k] for t in tallies) for k in tallies[0] if k != "max_silence"}
    out["max_silence"] = max(t["max_silence"] for t in tallies)
    return out


def bootstrap_ci(xs, rng, n=2000):
    means = []
    for _ in range(n):
        s = [xs[rng.randrange(len(xs))] for _ in xs]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def sign_test(diffs):
    """Two-sided exact sign test p, ties dropped."""
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def join_spans(spans, tol=1e-3):
    """Union of (start, end) spans; spans that touch become one."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def stage_measure(args, full, work):
    rng = random.Random(args.seed)
    segments = {r["item_name"]: r for r in read_jsonl(full / "segments.jsonl")}
    notes = {r["item_name"]: [(a, b) for _, a, b in r["notes"]]
             for r in read_jsonl(full / "notes.jsonl")}
    chains = read_jsonl(work / "chains.jsonl")

    part_tally, part_sil = {}, {}

    def part(item):
        if item not in part_tally:
            ph, sl = read_grid(full / "aligned" / f"{item}.TextGrid")
            part_tally[item] = tally(ph, sl, notes.get(item, []), segments[item]["duration"])
            part_sil[item] = sl
        return part_tally[item]

    def parts_timeline(c, offset):
        """The parts' silences on the chain's clock, with the inter-part gaps counted as
        silence and touching spans joined -- the longest silence the parts' alignments
        imply for the same audio, so a merged clip is not credited with a 'swallowed'
        span that is only part A's tail + the gap + part B's head."""
        spans = []
        for p in c["parts"]:
            part(p)
            spans += [(a + offset[p], b + offset[p]) for a, b in part_sil[p]]
        for p, q in zip(c["parts"], c["parts"][1:]):
            end_p = offset[p] + segments[p]["duration"]
            if offset[q] > end_p:
                spans.append((end_p, offset[q]))
        return [b - a for a, b in join_spans(spans)]

    # reference curve: every 2-6 s utterance MFA aligned
    ref = defaultdict(list)
    for path in sorted((full / "aligned").glob("*.TextGrid")):
        item = path.stem
        b = bucket_of(segments[item]["duration"], REF_BUCKETS)
        if b:
            ref[b].append(part(item))
    from aligners import hangul_syllables
    lyr = read_jsonl(full / "lyrics.jsonl")
    alignable = {r["item_name"] for r in lyr if len(hangul_syllables(r["text"])) >= 2}
    aligned_ref = {p.stem for p in (full / "aligned").glob("*.TextGrid")}
    ref_failed = {"alignable": len(alignable), "aligned": len(alignable & aligned_ref),
                  "failed_pct": 100 * len(alignable - aligned_ref) / len(alignable)}
    print(f"reference: {sum(len(v) for v in ref.values())} utterances; "
          f"2-6 s MFA failures {ref_failed}")

    rows, failed = [], defaultdict(int)
    drawn = defaultdict(int)
    for c in chains:
        drawn[c["bucket"]] += 1
        path = work / "aligned" / f"{c['item_name']}.TextGrid"
        if not path.exists():           # MFA's beam search gave up; every part aligned
            failed[c["bucket"]] += 1
            continue
        ph, sl = read_grid(path)
        offset = {p: segments[p]["song_start"] - c["song_start"] for p in c["parts"]}
        chain_notes = [(a + offset[p], b + offset[p]) for p in c["parts"]
                       for a, b in notes.get(p, [])]
        merged = tally(ph, sl, chain_notes, c["duration"])
        parts = pool([part(p) for p in c["parts"]])
        joined = parts_timeline(c, offset)
        parts["max_silence"] = max(joined, default=0.0)
        parts["long_silence"] = sum(x for x in joined if x >= LONG_SILENCE_SEC)
        rows.append({"item_name": c["item_name"], "bucket": c["bucket"],
                     "duration": c["duration"], "n_parts": len(c["parts"]),
                     "zero_gap": any(g == 0 for g in c["gaps"]),
                     "merged": ratios(merged), "parts": ratios(parts),
                     "swallowed": merged["max_silence"] - parts["max_silence"]
                     >= LONG_SILENCE_SEC})
    print(f"chains measured {len(rows)}, without a TextGrid {sum(failed.values())}")
    write_jsonl(work / "metrics.jsonl", rows)

    def ref_row(b, ts):
        r = [ratios(t) for t in ts]
        return {"bucket": b, "n": len(ts),
                "floor_pct": nanmean([x["floor_pct"] for x in r]),
                "voiced_sil": nanmean([x["voiced_sil"] for x in r]),
                "silence_pct": nanmean([x["silence_pct"] for x in r]),
                "share_long_silence": 100 * sum(x["long_silence"] > 0 for x in r) / len(r),
                "pooled": ratios(pool(ts))}

    def paired(rs, key):
        d = [r["merged"][key] - r["parts"][key] for r in rs
             if not (math.isnan(r["merged"][key]) or math.isnan(r["parts"][key]))]
        lo, hi = bootstrap_ci(d, rng)
        return {"n": len(d), "merged": nanmean([r["merged"][key] for r in rs]),
                "parts": nanmean([r["parts"][key] for r in rs]),
                "diff_mean": sum(d) / len(d), "diff_median": median(d),
                "ci95": [lo, hi], "sign_p": sign_test(d),
                "share_worse": 100 * sum(x > 0 for x in d) / len(d)}

    def block(rs):
        return {"n": len(rs),
                "duration_median": median([r["duration"] for r in rs]),
                "floor_pct": paired(rs, "floor_pct"),
                "voiced_sil": paired(rs, "voiced_sil"),
                "silence_pct": paired(rs, "silence_pct"),
                "mean_phone_ms": paired(rs, "mean_phone_ms"),
                "swallowed_pct": 100 * sum(r["swallowed"] for r in rs) / len(rs),
                "long_silence_merged": sum(r["merged"]["long_silence"] for r in rs) / len(rs),
                "long_silence_parts": sum(r["parts"]["long_silence"] for r in rs) / len(rs)}

    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)
    summary = {
        "written_by": "pansori/experiments/long_utterance_mfa.py",
        "aligner": "MFA korean_mfa (acoustic + dictionary)",
        "floor_sec": FLOOR_SEC, "pause_sec": PAUSE_SEC, "seed": args.seed,
        "reference_2_6s": [ref_row(f"{lo}-{hi}", ref[f"{lo}-{hi}"]) for lo, hi in REF_BUCKETS],
        "paired_by_bucket": {f"{lo}-{hi}": {**block(by_bucket[f"{lo}-{hi}"]),
                                            "drawn": drawn[f"{lo}-{hi}"],
                                            "failed": failed[f"{lo}-{hi}"],
                                            "failed_pct": 100 * failed[f"{lo}-{hi}"]
                                            / drawn[f"{lo}-{hi}"]}
                             for lo, hi in BUCKETS if by_bucket[f"{lo}-{hi}"]},
        "failed_all": {"drawn": len(chains), "failed": sum(failed.values())},
        "reference_failed_2_6s": ref_failed,
        "paired_all": block(rows),
        "paired_zero_gap": block([r for r in rows if r["zero_gap"]]),
        "paired_pause_gap": block([r for r in rows if not r["zero_gap"]]),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_summary(summary)
    plot(summary, ROOT / "docs" / "img" / "long_utterance_mfa.png")


def print_summary(s):
    print("\nreference, 2-6 s utterances as aligned in the corpus")
    print(f"{'bucket':>7} {'n':>6} {'floor%':>7} {'voicedSil%':>10} {'sil%':>6} {'has>=1s%':>8}")
    for r in s["reference_2_6s"]:
        print(f"{r['bucket']:>7} {r['n']:>6} {r['floor_pct']:7.1f} {r['voiced_sil']:10.1f} "
              f"{r['silence_pct']:6.1f} {r['share_long_silence']:8.1f}")
    print("\npaired: merged clip minus its own parts (mean diff [95% CI], sign p)")
    for b, r in list(s["paired_by_bucket"].items()) + [("all", s["paired_all"])]:
        f, v = r["floor_pct"], r["voiced_sil"]
        print(f"{b:>6} n={r['n']:>4}  floor {f['parts']:5.1f}->{f['merged']:5.1f} "
              f"{f['diff_mean']:+5.1f} [{f['ci95'][0]:+.1f},{f['ci95'][1]:+.1f}] p={f['sign_p']:.1e}"
              f"  voicedSil {v['parts']:5.1f}->{v['merged']:5.1f} {v['diff_mean']:+5.1f} "
              f"[{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] p={v['sign_p']:.1e}"
              f"  swallowed {r['swallowed_pct']:4.1f}%"
              + (f"  failed {r['failed']}/{r['drawn']}" if "drawn" in r else ""))


def plot(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ref = s["reference_2_6s"]
    pb = s["paired_by_bucket"]
    for ax, key, label in [(axes[0], "floor_pct", "phones at the 30 ms floor (%)"),
                           (axes[1], "voiced_sil", "note-covered time labelled silence (%)")]:
        x_ref = [int(r["bucket"].split("-")[0]) + 0.5 for r in ref]
        ax.plot(x_ref, [r[key] for r in ref], "o-", color="0.4", label="2–6 s utterances (corpus)")
        x = [int(b.split("-")[0]) + 1 for b in pb]
        ax.plot(x, [pb[b][key]["parts"] for b in pb], "s--", color="tab:blue",
                label="same audio as 2–6 s parts")
        ax.plot(x, [pb[b][key]["merged"] for b in pb], "o-", color="tab:red",
                label="same audio as one clip")
        ax.axvline(6, color="0.7", lw=0.8, ls=":")
        ax.set_xlabel("clip length (s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["build", "align", "measure"])
    ap.add_argument("--per-bucket", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    full = set_dir("full")
    work = full / "long_mfa"
    work.mkdir(exist_ok=True)
    {"build": stage_build, "align": stage_align, "measure": stage_measure}[args.stage](
        args, full, work)


if __name__ == "__main__":
    main()
