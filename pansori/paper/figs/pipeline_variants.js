// Candidate layouts for Fig. 1, one per slide file, so one can be chosen for the paper.
//
//   NODE_PATH=<dir with pptxgenjs> node pansori/paper/figs/pipeline_variants.js [v1 … v6]
//   for v in 1 2 3 4 5 6; do soffice --headless --convert-to pdf --outdir pansori/paper/figs/drafts \
//     pansori/paper/figs/drafts/pipeline_v$v.pptx; done
//
// v1–v3 draw the same content: one panel per paragraph of Section 3, each holding the steps
// that paragraph describes, coloured by paragraph. They differ only in layout:
//   v1  five equal columns, steps stacked as chips
//   v2  a chevron band naming the paragraphs, steps listed beneath as a timeline
//   v3  two tracks: the text path on top, notes and f0 below, merging into the singing model
//   v4–v6: see the second header below, before v4.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const OUT = path.join(__dirname, 'drafts');
const FONT = 'Arial', INK = '1F2D3A', LINK = '6B7682';

// One hue per paragraph: a saturated stroke for headers and borders, a pale tint for fills.
const HUE = {
  corpus:  { line: '3B5B8C', fill: 'EAF0F8' },
  seg:     { line: '23787A', fill: 'E3F2F1' },
  lyrics:  { line: '9A6A12', fill: 'FBF2E0' },
  align:   { line: 'A8443F', fill: 'F9E8E6' },
  model:   { line: '5B4B8A', fill: 'EDE9F6' },
};

// Section 3, paragraph by paragraph.
const S = {
  corpus: { title: 'Corpus', steps: ['Mode-labelled audio', 'HT-Demucs vocals', 'Drop aniri spans'], out: 'sung spans' },
  seg:    { title: 'Segmentation', steps: ['Cut at pauses', 'Split melismas at gaps', 'Cap at 6 s'], out: 'utterances' },
  pitch:  { title: 'Notes and f0', steps: ['ROSVOT notes', 'RMVPE f0'], out: 'notes, f0' },
  lyrics: { title: 'Lyrics', steps: ['Whisper large-v3', 'Collapse repeats', 'Anchor to changbon', 'Hybrid spelling'], out: 'phonemes' },
  align:  { title: 'Alignment', steps: ['MFA on Whisper text', 'Absorb held silence', 'SOFA round 1', 'SOFA round 2 (hybrid)'], out: 'boundaries' },
  model:  { title: 'Singing model', steps: ['Variance: durations, f0', 'Acoustic: mel', 'NSF-HiFiGAN vocoder'], out: 'waveform' },
};

function deck(W, H) {
  const pres = new pptxgen();
  pres.defineLayout({ name: 'FIG', width: W, height: H });
  pres.layout = 'FIG';
  const slide = pres.addSlide();
  slide.background = { color: 'FFFFFF' };

  const line = (x1, y1, x2, y2, color, head = true, width = 1.25) => slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1, line: { color, width, endArrowType: head ? 'triangle' : undefined },
  });

  // A panel: pale body, coloured border, a solid header band with square lower corners.
  const panel = (x, y, w, h, hue, title, band = 0.26) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07,
      fill: { color: hue.fill }, line: { color: hue.line, width: 1 } });
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: band, rectRadius: 0.07,
      fill: { color: hue.line }, line: { color: hue.line, width: 1 } });
    slide.addShape(pres.shapes.RECTANGLE, { x, y: y + band / 2, w, h: band / 2,
      fill: { color: hue.line }, line: { color: hue.line, width: 0 } });
    slide.addText(title, { x, y, w, h: band, fontFace: FONT, fontSize: 9.5, bold: true,
      color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0 });
  };

  const chip = (x, y, w, h, text, hue, size = 7.5) => slide.addText(text, {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x, y, w, h,
    fill: { color: 'FFFFFF' }, line: { color: hue.line, width: 0.75 },
    fontFace: FONT, fontSize: size, color: INK, align: 'center', valign: 'middle', margin: 2,
  });

  const label = (x, y, w, text, hue, align = 'center') => slide.addText(`→ ${text}`, {
    x, y, w, h: 0.18, fontFace: FONT, fontSize: 8, italic: true, bold: true, color: hue.line,
    align, valign: 'middle', margin: 0,
  });

  const text = (x, y, w, h, t, opts = {}) => slide.addText(t, {
    x, y, w, h, fontFace: FONT, fontSize: 7.5, color: INK, valign: 'middle', margin: 0, ...opts,
  });

  return { pres, slide, line, panel, chip, label, text };
}

// v1: five equal columns of stacked chips; notes and f0 fork from the end of Segmentation.
function v1() {
  const W = 8.0, M = 0.06, GAP = 0.2, N = 5, CW = (W - 2 * M - (N - 1) * GAP) / N;
  const BAND = 0.26, CH = 0.3, VG = 0.1, PAD = 0.07, ROWS = 4;
  const PH = BAND + PAD + ROWS * CH + (ROWS - 1) * VG + 0.06 + 0.2, Y = 0.04, H = PH + 0.08;
  const d = deck(W, H);
  const cx = i => M + i * (CW + GAP);
  const rowY = r => Y + BAND + PAD + r * (CH + VG);
  const order = ['corpus', 'seg', 'lyrics', 'align', 'model'];

  order.forEach((k, i) => {
    const hue = HUE[k], s = S[k], x = cx(i);
    d.panel(x, Y, CW, PH, hue, s.title, BAND);
    const steps = s.steps;
    steps.forEach((t, r) => {
      d.chip(x + PAD, rowY(r), CW - 2 * PAD, CH, t, hue);
      if (r < steps.length - 1) d.line(x + CW / 2, rowY(r) + CH, x + CW / 2, rowY(r + 1), hue.line, true, 1);
    });
    if (k === 'seg') {
      // The fork: notes and f0 are taken from the cut utterances, side by side.
      const half = (CW - 2 * PAD - 0.06) / 2, r = 3;
      S.pitch.steps.forEach((t, j) => {
        const xx = x + PAD + j * (half + 0.06);
        d.chip(xx, rowY(r), half, CH, t, hue, 7);
        d.line(x + CW / 2, rowY(r - 1) + CH, xx + half / 2, rowY(r), hue.line, true, 1);
      });
      d.label(x, Y + PH - 0.24, CW, 'utterances, notes, f0', hue);
    } else {
      d.label(x, Y + PH - 0.24, CW, s.out, hue);
    }
    if (i < N - 1) d.line(x + CW, Y + PH / 2, cx(i + 1), Y + PH / 2, LINK);
  });
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v1.pptx') });
}

// v2: a chevron band names the paragraphs left to right; each lists its steps on a timeline.
function v2() {
  const W = 8.0, M = 0.06, N = 5, TIP = 0.14, BAND = 0.32, Y = 0.04;
  const CW = (W - 2 * M + (N - 1) * TIP) / N;            // chevrons overlap by their tip
  const cx = i => M + i * (CW - TIP);
  const STEP = 0.21, LIST_Y = Y + BAND + 0.1, ROWS = 5;
  const BODY_H = 0.1 + ROWS * STEP + 0.26, H = Y + BAND + BODY_H + 0.04;
  const d = deck(W, H);
  const order = ['corpus', 'seg', 'lyrics', 'align', 'model'];

  order.forEach((k, i) => {
    const hue = HUE[k], s = S[k], x = cx(i);
    // Tinted column under its chevron.
    const colX = x + (i === 0 ? 0 : TIP), colW = CW - TIP - (i === 0 ? 0 : 0);
    d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x: colX + 0.02, y: Y + BAND - 0.04, w: colW - 0.04,
      h: BODY_H + 0.04, rectRadius: 0.06, fill: { color: hue.fill }, line: { color: hue.fill, width: 0 } });
    d.slide.addText(s.title, {
      shape: i === 0 ? d.pres.shapes.PENTAGON : d.pres.shapes.CHEVRON,
      x, y: Y, w: CW, h: BAND, fill: { color: hue.line }, line: { color: 'FFFFFF', width: 1.5 },
      fontFace: FONT, fontSize: 9.5, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0,
    });

    const steps = k === 'seg' ? [...s.steps, ...S.pitch.steps.map(t => `+ ${t}`)] : s.steps;
    const railX = colX + 0.14;
    const dotY = r => LIST_Y + r * STEP + STEP / 2;
    d.line(railX, dotY(0), railX, dotY(steps.length - 1), hue.line, false, 1.25);
    steps.forEach((t, r) => {
      const branch = t.startsWith('+ ');
      d.slide.addShape(d.pres.shapes.OVAL, { x: railX - 0.04, y: dotY(r) - 0.04, w: 0.08, h: 0.08,
        fill: { color: branch ? 'FFFFFF' : hue.line }, line: { color: hue.line, width: 1 } });
      d.text(railX + 0.09, dotY(r) - STEP / 2, colW - 0.3, STEP, branch ? t.slice(2) : t,
        { italic: branch, fontSize: 7.5 });
    });
    const out = k === 'seg' ? 'utterances, notes, f0' : s.out;
    d.label(railX - 0.06, Y + BAND + BODY_H - 0.24, colW - 0.1, out, hue, 'left');
  });
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v2.pptx') });
}

// v3: the text path runs along the top, notes and f0 along the bottom, and both enter the
// singing model, which spans the full height on the right.
function v3() {
  const W = 8.0, M = 0.06, GAP = 0.2, PW = 1.3, BAND = 0.26, PAD = 0.07, CH = 0.26, VG = 0.11;
  const Y = 0.04, ROWS = 4;
  const TOP_H = BAND + PAD + ROWS * CH + (ROWS - 1) * VG + 0.26;
  const BY = Y + TOP_H + 0.14, BOT_H = BAND + PAD + CH + PAD, H = BY + BOT_H + 0.04;
  const d = deck(W, H);
  const px = i => M + i * (PW + GAP);
  const rowY = (y0, r) => y0 + BAND + PAD + r * (CH + VG);
  const stack = (k, x, y0, w, steps = S[k].steps) => steps.forEach((t, r) => {
    d.chip(x + PAD, rowY(y0, r), w - 2 * PAD, CH, t, HUE[k]);
    if (r < steps.length - 1) d.line(x + w / 2, rowY(y0, r) + CH, x + w / 2, rowY(y0, r + 1), HUE[k].line, true, 1);
  });

  ['corpus', 'seg', 'lyrics', 'align'].forEach((k, i) => {
    d.panel(px(i), Y, PW, TOP_H, HUE[k], S[k].title, BAND);
    stack(k, px(i), Y, PW);
    const nxt = i < 3 ? px(i + 1) : px(4);
    d.line(px(i) + PW, Y + TOP_H / 2, nxt, Y + TOP_H / 2, LINK);
    d.label(px(i), Y + TOP_H - 0.23, PW, S[k].out, HUE[k]);
  });

  // Bottom track: notes and f0, taken from the segmented utterances.
  const bx = px(2), bw = 2 * PW + GAP, hueP = HUE.seg;
  d.panel(bx, BY, bw, BOT_H, hueP, S.pitch.title, BAND);
  const cw = (bw - 2 * PAD - 0.12) / 2;
  S.pitch.steps.forEach((t, j) => d.chip(bx + PAD + j * (cw + 0.12), rowY(BY, 0), cw, CH, t, hueP));
  const segMid = px(1) + PW / 2, bMid = BY + BOT_H / 2;
  d.line(segMid, Y + TOP_H, segMid, bMid, LINK, false);
  d.line(segMid, bMid, bx, bMid, LINK);
  d.line(bx + bw, bMid, px(4), bMid, LINK);

  // Singing model, full height.
  const mx = px(4), mw = W - M - mx, mh = BY + BOT_H - Y;
  d.panel(mx, Y, mw, mh, HUE.model, S.model.title, BAND);
  const n = S.model.steps.length, mch = 0.34, mvg = (mh - BAND - 2 * PAD - n * mch - 0.2) / (n - 1);
  S.model.steps.forEach((t, r) => {
    const yy = Y + BAND + PAD + r * (mch + mvg);
    d.chip(mx + PAD, yy, mw - 2 * PAD, mch, t, HUE.model, 8);
    if (r < n - 1) d.line(mx + mw / 2, yy + mch, mx + mw / 2, yy + mch + mvg, HUE.model.line, true, 1);
  });
  d.label(mx, Y + mh - 0.22, mw, S.model.out, HUE.model);
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v3.pptx') });
}

// ---------------------------------------------------------------------------------------------
// v4–v6: fewer words per step, and the arrows carry what is handed on. A step is one line of
// method; between steps a small glyph of the data in transit (a waveform, cut clips, phonemes,
// a segmented phone bar, notes under an f0 curve) shows how the material changes.
//   v4  title pills, method as free text beneath, payload glyph and label on each arrow
//   v5  a metro line: stations, data tags riding the line, the pitch branch splitting off
//   v6  data cards are the shapes, and each paragraph is written on the arrow that transforms them

const BRIEF = {
  corpus: 'HT-Demucs vocals,\naniri removed',
  seg:    'cut at pauses and\nnote gaps, ≤ 6 s',
  pitch:  'ROSVOT, RMVPE',
  lyrics: 'Whisper, respelled\nfrom the changbon',
  align:  'MFA, then SOFA\nfine-tuned twice',
  model:  'variance → acoustic\n→ vocoder',
};
const GREY = '5A6670';

// Glyphs, drawn as custom geometry in local inches: [x, y, moveTo?], or a pptxgenjs curve point.
function glyphs(d) {
  const { pres, slide } = d;
  const path_ = (x, y, w, h, pts, color, width = 1) => slide.addShape(pres.shapes.CUSTOM_GEOMETRY, {
    x, y, w, h, line: { color, width },
    points: pts.map(p => (Array.isArray(p) ? (p[2] ? { x: p[0], y: p[1], moveTo: true } : { x: p[0], y: p[1] }) : p)),
  });
  // Deterministic amplitude envelope, so every render draws the same waveform.
  const amp = (i, seed) => 0.25 + 0.75 * Math.abs(Math.sin(i * 0.37 + seed) * Math.cos(i * 0.11 + 2 * seed));
  const wave = (x, y, w, h, color, { seed = 1, n = 26, spans = [[0, 1]] } = {}) => {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      if (!spans.some(([a, b]) => t >= a && t <= b)) continue;
      const a = amp(i, seed) * h / 2;
      pts.push([t * w, h / 2 - a, true], [t * w, h / 2 + a]);
    }
    return path_(x, y, w, h, pts, color, 0.9);
  };
  const clips = (x, y, w, h, color) => wave(x, y, w, h, color,
    { seed: 2, n: 24, spans: [[0, 0.28], [0.38, 0.62], [0.72, 1]] });
  // A phone bar: an outlined strip cut at uneven boundaries.
  const bar = (x, y, w, h, color) => {
    const cuts = [0, 0.12, 0.3, 0.41, 0.63, 0.74, 0.88, 1];
    cuts.slice(0, -1).forEach((c, i) => slide.addShape(pres.shapes.RECTANGLE, {
      x: x + c * w, y: y + h * 0.15, w: (cuts[i + 1] - c) * w, h: h * 0.7,
      fill: { color: i % 2 ? 'FFFFFF' : color, transparency: i % 2 ? 0 : 45 }, line: { color, width: 0.75 },
    }));
  };
  const phones = (x, y, w, h, color) => slide.addText('p a n s o r i', {
    x: x - 0.1, y, w: w + 0.2, h, fontFace: FONT, fontSize: 7, bold: true, color,
    align: 'center', valign: 'middle', margin: 0, wrap: false, fit: 'none',
  });
  // Notes as level strokes, with an f0 line that slides between them and wavers on the held ones.
  const notes = (x, y, w, h, nc, fc) => {
    const lv = [0.7, 0.45, 0.55, 0.2], seg = w / lv.length, pts = [];
    lv.forEach((l, i) => pts.push([i * seg + 0.01, l * h, true], [(i + 1) * seg - 0.01, l * h]));
    path_(x, y, w, h, pts, nc, 2.2);
    const f = [];
    for (let i = 0; i <= 40; i++) {
      const t = i / 40, k = Math.min(lv.length - 1, Math.floor(t * lv.length)), u = t * lv.length - k;
      const nxt = lv[Math.min(lv.length - 1, k + 1)];
      const base = u > 0.8 ? lv[k] + (nxt - lv[k]) * (u - 0.8) / 0.2 : lv[k];
      f.push([t * w, (base - 0.13 + 0.04 * Math.sin(t * 26)) * h, i === 0]);
    }
    return path_(x, y, w, h, f, fc, 0.9);
  };
  return { wave, clips, bar, phones, notes, path_ };
}

// v4: solid title pills; the method sits beneath as plain text; each arrow carries a glyph of
// its payload with the payload's name under the shaft.
function v4() {
  const W = 8.0, M = 0.06, PW = 1.06, PH = 0.28, GAPW = 0.6, MW = W - 2 * M - 4 * PW - 4 * GAPW;
  const TY = 0.26, BY = 1.2, H = 1.78;
  const d = deck(W, H), g = glyphs(d);
  const px = i => M + i * (PW + GAPW);
  const pill = (x, y, w, k, title) => d.slide.addText(title, {
    shape: d.pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.5, x, y, w, h: PH,
    fill: { color: HUE[k].line }, line: { color: HUE[k].line, width: 0 },
    fontFace: FONT, fontSize: 9.5, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0,
  });
  const brief = (x, y, w, k) => d.text(x, y, w, 0.3, BRIEF[k], { fontSize: 7.5, color: GREY, align: 'center', valign: 'top' });
  const payload = (x0, x1, y, name, draw, hue) => {
    d.line(x0 + 0.03, y, x1 - 0.03, y, LINK);
    const gw = 0.42, gx = (x0 + x1) / 2 - gw / 2;
    draw(gx, y - 0.22, gw, 0.17, hue);
    d.text(x0 - 0.04, y + 0.03, x1 - x0 + 0.08, 0.15, name, { fontSize: 6.5, italic: true, color: hue, align: 'center' });
  };

  const order = ['corpus', 'seg', 'lyrics', 'align'];
  order.forEach((k, i) => { pill(px(i), TY, PW, k, S[k].title); brief(px(i), TY + PH + 0.04, PW, k); });
  const ay = TY + PH / 2;
  payload(px(0) + PW, px(1), ay, 'sung spans', (x, y, w, h, c) => g.wave(x, y, w, h, c, { seed: 0.4 }), HUE.corpus.line);
  payload(px(1) + PW, px(2), ay, 'utterances', g.clips, HUE.seg.line);
  payload(px(2) + PW, px(3), ay, 'phonemes', g.phones, HUE.lyrics.line);

  // Singing model: spans both rows; its three stages are free text joined by small arrows.
  const mx = px(4), mTop = TY, mBot = BY + PH;
  payload(px(3) + PW, mx, ay, 'boundaries', g.bar, HUE.align.line);
  d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x: mx, y: mTop, w: MW, h: mBot - mTop, rectRadius: 0.08,
    fill: { color: HUE.model.fill }, line: { color: HUE.model.line, width: 1 } });
  pill(mx, mTop, MW, 'model', S.model.title);
  const stages = ['variance', 'acoustic', 'vocoder'], sy = mTop + PH + 0.07, sh = 0.16;
  stages.forEach((t, r) => {
    const yy = sy + r * (sh + 0.08);
    d.text(mx, yy, MW, sh, t, { fontSize: 8, bold: true, color: HUE.model.line, align: 'center' });
    if (r < stages.length - 1) d.line(mx + MW / 2, yy + sh, mx + MW / 2, yy + sh + 0.08, HUE.model.line, true, 0.75);
  });
  g.wave(mx + 0.18, mBot - 0.25, MW - 0.36, 0.17, HUE.model.line, { seed: 3, n: 34 });

  // Pitch branch: leaves the utterances arrow, drops to its own row, and enters the model.
  const bx = px(2) + (PW + GAPW) / 2, bw = PW, by = BY;
  const jx = px(1) + PW + 0.08;
  d.slide.addShape(d.pres.shapes.OVAL, { x: jx - 0.03, y: ay - 0.03, w: 0.06, h: 0.06,
    fill: { color: LINK }, line: { color: LINK, width: 0 } });
  d.line(jx, ay, jx, by + PH / 2, LINK, false);
  d.line(jx, by + PH / 2, bx, by + PH / 2, LINK);
  pill(bx, by, bw, 'seg', S.pitch.title);
  d.text(bx, by + PH + 0.04, bw, 0.15, BRIEF.pitch, { fontSize: 7.5, color: GREY, align: 'center' });
  const nx0 = bx + bw, ny = by + PH / 2;
  d.line(nx0 + 0.03, ny, mx - 0.03, ny, LINK);
  g.notes(nx0 + (mx - nx0) / 2 - 0.22, ny - 0.24, 0.44, 0.19, HUE.seg.line, INK);
  d.text(nx0, ny + 0.03, mx - nx0, 0.15, 'notes, f0', { fontSize: 6.5, italic: true, color: HUE.seg.line, align: 'center' });
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v4.pptx') });
}

// v5: a metro map. Each paragraph is a station on one line; what travels between stations
// rides the line as a tag; the pitch branch splits after Segmentation and rejoins at the model.
function v5() {
  const W = 8.0, H = 1.74, LY = 0.6, BY = 1.18, R = 0.085, TW = 4.2;
  const d = deck(W, H), g = glyphs(d);
  const sx = [0.5, 1.85, 3.55, 5.0];                  // station centres, top line
  const mx = 6.3, mw = W - 0.06 - mx;                   // terminal: the singing model
  const order = ['corpus', 'seg', 'lyrics', 'align'];
  const track = (x1, y1, x2, y2, color) => d.line(x1, y1, x2, y2, color, false, TW);

  // Track segments first, coloured by the station they leave.
  order.forEach((k, i) => track(sx[i], LY, i < 3 ? sx[i + 1] : mx, LY, HUE[k].line));
  // Pitch branch: down from Segmentation, along the bottom, into the terminal.
  const jx = sx[1] + 0.5, kr = 0.16;
  g.path_(jx, LY, 2 * kr, BY - LY, [[0, 0, true], [0, BY - LY - kr],
    { x: kr, y: BY - LY, curve: { type: 'quadratic', x1: 0, y1: BY - LY } }, [2 * kr, BY - LY]], HUE.seg.line, TW);
  track(jx + 2 * kr - 0.01, BY, mx, BY, HUE.seg.line);

  const station = (x, y, k, title, brief, under = false) => {
    d.slide.addShape(d.pres.shapes.OVAL, { x: x - R, y: y - R, w: 2 * R, h: 2 * R,
      fill: { color: 'FFFFFF' }, line: { color: HUE[k].line, width: 2 } });
    const ty = under ? y + R + 0.02 : y - R - 0.24;
    d.text(x - 0.65, ty, 1.3, 0.2, title, { fontSize: 9.5, bold: true, color: HUE[k].line, align: 'center' });
    d.text(x - 0.65, under ? ty + 0.19 : y + R + 0.1, 1.3, 0.3, brief, { fontSize: 7.5, color: GREY, align: 'center', valign: 'top' });
  };
  order.forEach((k, i) => station(sx[i], LY, k, S[k].title, BRIEF[k]));
  const px = 4.0;
  station(px, BY, 'seg', S.pitch.title, BRIEF.pitch, true);

  // A tag riding the track: white card, glyph and name.
  const tag = (x, y, name, draw, k, gw = 0.4) => {
    const w = gw + 0.12, h = 0.34;
    d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x: x - w / 2, y: y - h / 2, w, h, rectRadius: 0.05,
      fill: { color: 'FFFFFF' }, line: { color: HUE[k].line, width: 0.75 } });
    draw(x - gw / 2, y - h / 2 + 0.04, gw, 0.15, HUE[k].line);
    d.text(x - w / 2, y + 0.02, w, 0.13, name, { fontSize: 6, italic: true, color: HUE[k].line, align: 'center' });
  };
  tag((sx[0] + sx[1]) / 2, LY, 'sung spans', (x, y, w, h, c) => g.wave(x, y, w, h, c, { seed: 0.4 }), 'corpus');
  tag((jx + sx[2]) / 2, LY, 'utterances', g.clips, 'seg');
  tag((sx[2] + sx[3]) / 2, LY, 'phonemes', g.phones, 'lyrics', 0.5);
  tag((sx[3] + mx) / 2, LY, 'boundaries', g.bar, 'align');
  tag((px + mx) / 2 + 0.45, BY, 'notes, f0', (x, y, w, h, c) => g.notes(x, y, w, h, c, INK), 'seg');

  // Terminal: the singing model, spanning both lines, ending in a waveform.
  const mTop = LY - 0.38, mBot = BY + 0.3;
  d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x: mx, y: mTop, w: mw, h: mBot - mTop, rectRadius: 0.1,
    fill: { color: HUE.model.line }, line: { color: HUE.model.line, width: 0 } });
  d.text(mx, mTop + 0.06, mw, 0.2, S.model.title, { fontSize: 9.5, bold: true, color: 'FFFFFF', align: 'center' });
  d.text(mx, mTop + 0.3, mw, 0.32, BRIEF.model, { fontSize: 7.5, color: 'FFFFFF', align: 'center', valign: 'top' });
  g.wave(mx + 0.2, mBot - 0.42, mw - 0.4, 0.22, 'FFFFFF', { seed: 3, n: 34 });
  d.text(mx, mBot - 0.19, mw, 0.14, 'waveform', { fontSize: 6.5, italic: true, color: 'FFFFFF', align: 'center' });
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v5.pptx') });
}

// v6: the data are the shapes. Each card shows what exists after a paragraph; the paragraph
// itself is written on the arrow that turns one card into the next.
function v6() {
  const W = 8.0, H = 1.72, CW = 0.58, CH = 0.42, M = 0.06, TY = 0.36, BY = 1.12, IN = 0.32;
  const d = deck(W, H), g = glyphs(d);
  const mw = 0.95, mx = W - M - mw;
  const AW = (mx - IN - M - 5 * CW) / 4;                // four arrows between cards; IN is the one into the model
  const cx = i => M + i * (CW + AW);
  const card = (x, y, name, draw, color = INK) => {
    d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x, y, w: CW, h: CH, rectRadius: 0.06,
      fill: { color: 'FFFFFF' }, line: { color: 'B8C0C8', width: 0.75 } });
    draw(x + 0.08, y + 0.08, CW - 0.16, CH - 0.16, color);
    d.text(x - 0.1, y + CH + 0.02, CW + 0.2, 0.14, name, { fontSize: 7, italic: true, color: INK, align: 'center' });
  };
  const step = (x0, x1, y, k, title, brief, above = true) => {
    d.line(x0 + 0.04, y, x1 - 0.04, y, HUE[k].line, true, 1.75);
    d.text(x0 - 0.04, y - 0.2, x1 - x0 + 0.08, 0.17, title, { fontSize: 8.5, bold: true, color: HUE[k].line, align: 'center' });
    if (brief) d.text(x0 - 0.02, y + 0.04, x1 - x0 + 0.04, 0.28, brief, { fontSize: 6.5, color: GREY, align: 'center', valign: 'top' });
  };

  const ay = TY + CH / 2;
  card(cx(0), TY, 'recordings', (x, y, w, h, c) => g.wave(x, y, w, h, c, { seed: 0.4, n: 30 }), GREY);
  card(cx(1), TY, 'sung spans', (x, y, w, h, c) => g.wave(x, y, w, h, c,
    { seed: 0.4, n: 30, spans: [[0, 0.45], [0.7, 1]] }), HUE.corpus.line);
  card(cx(2), TY, 'utterances', g.clips, HUE.seg.line);
  card(cx(3), TY, 'phonemes', g.phones, HUE.lyrics.line);
  card(cx(4), TY, 'boundaries', g.bar, HUE.align.line);
  step(cx(0) + CW, cx(1), ay, 'corpus', S.corpus.title, BRIEF.corpus);
  step(cx(1) + CW, cx(2), ay, 'seg', S.seg.title, BRIEF.seg);
  step(cx(2) + CW, cx(3), ay, 'lyrics', S.lyrics.title, BRIEF.lyrics);
  step(cx(3) + CW, cx(4), ay, 'align', S.align.title, BRIEF.align);

  // Pitch: from the utterances card down to a notes-and-f0 card, then right into the model.
  const ux = cx(2) + CW / 2, nx = cx(3), ny = BY + CH / 2;
  d.line(ux, TY + CH + 0.18, ux, ny, HUE.seg.line, false, 1.75);
  d.line(ux, ny, nx - 0.04, ny, HUE.seg.line, true, 1.75);
  d.text(ux + 0.06, ny - 0.2, nx - ux - 0.08, 0.17, S.pitch.title, { fontSize: 8.5, bold: true, color: HUE.seg.line, align: 'center' });
  d.text(ux + 0.06, ny + 0.04, nx - ux - 0.08, 0.14, BRIEF.pitch, { fontSize: 6.5, color: GREY, align: 'center' });
  card(nx, BY, 'notes, f0', (x, y, w, h, c) => g.notes(x, y, w, h, c, INK), HUE.seg.line);

  // The model: a solid block both paths enter, ending in the waveform it sings.
  const mTop = TY - 0.18, mBot = BY + CH + 0.04;
  d.line(cx(4) + CW + 0.04, ay, mx - 0.04, ay, LINK, true, 1.75);
  d.line(nx + CW + 0.04, ny, mx - 0.04, ny, LINK, true, 1.75);
  d.slide.addShape(d.pres.shapes.ROUNDED_RECTANGLE, { x: mx, y: mTop, w: mw, h: mBot - mTop, rectRadius: 0.1,
    fill: { color: HUE.model.fill }, line: { color: HUE.model.line, width: 1 } });
  d.text(mx, mTop + 0.06, mw, 0.18, S.model.title, { fontSize: 8.5, bold: true, color: HUE.model.line, align: 'center' });
  d.text(mx, mTop + 0.27, mw, 0.3, BRIEF.model, { fontSize: 7, color: GREY, align: 'center', valign: 'top' });
  g.wave(mx + 0.14, mBot - 0.42, mw - 0.28, 0.22, HUE.model.line, { seed: 3, n: 34 });
  d.text(mx, mBot - 0.19, mw, 0.14, 'waveform', { fontSize: 7, italic: true, color: HUE.model.line, align: 'center' });
  return d.pres.writeFile({ fileName: path.join(OUT, 'pipeline_v6.pptx') });
}

fs.mkdirSync(OUT, { recursive: true });
const ALL = { v1, v2, v3, v4, v5, v6 };
const pick = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(ALL);
Promise.all(pick.map(v => ALL[v]())).then(fs_ => fs_.forEach(f => console.log('wrote', f)));
