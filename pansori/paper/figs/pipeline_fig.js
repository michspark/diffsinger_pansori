// Pipeline figure for the AMIW paper (Section 3): one slide sized to the figure.
//
//   NODE_PATH=<dir with pptxgenjs> node pansori/paper/figs/pipeline_fig.js
//   python <pptx skill>/scripts/office/soffice.py --headless --convert-to pdf \
//     --outdir pansori/paper/figs pansori/paper/figs/pipeline.pptx
//
// Top row: the path the served (hyb2) corpus took. Bottom row: what was tried at that step
// and dropped (dashed), plus ROSVOT's notes as a side input (solid). Every number is the
// one the paper's text cites; change them there first.
const path = require('path');
const pptxgen = require('pptxgenjs');

const W = 8.0, H = 3.0;
const INK = '1F2D3A', KEEP = '2F5D62', KEEP_FILL = 'E6EFF0', DROP = 'A23B3B', MUTED = '5A6670';
const FONT = 'Arial';

const pres = new pptxgen();
pres.defineLayout({ name: 'FIG', width: W, height: H });
pres.layout = 'FIG';
const slide = pres.addSlide();
slide.background = { color: 'FFFFFF' };

const MARGIN = 0.15, GAP = 0.2, N = 6;
const BOX_W = (W - 2 * MARGIN - (N - 1) * GAP) / N;
const colX = i => MARGIN + i * (BOX_W + GAP);
const TOP_Y = 0.12, TOP_H = 1.5, LOW_Y = 1.92, LOW_H = 0.96;
const NB = '\u00A0';   // keeps a number on the same line as its unit

function box(i, y, h, title, body, kind) {
  const keep = kind !== 'drop';
  slide.addText([
    { text: title, options: { bold: true, fontSize: 10.5, color: keep ? INK : DROP, breakLine: true } },
    { text: body, options: { fontSize: 9, color: keep ? INK : MUTED } },
  ], {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
    x: colX(i), y, w: BOX_W, h,
    fill: { color: kind === 'keep' ? KEEP_FILL : 'FFFFFF' },
    line: { color: keep ? KEEP : DROP, width: 1, dashType: keep ? 'solid' : 'dash' },
    fontFace: FONT, valign: 'top', align: 'left', margin: 4, paraSpaceAfter: 2,
    objectName: `${kind}-${i}`,
  });
}

function arrow(x1, y1, x2, y2, opts) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1,
    line: { color: opts.color, width: 1.25, dashType: opts.dash ? 'dash' : 'solid',
            endArrowType: opts.head ? 'triangle' : undefined },
  });
}

const main = [
  ['394 recordings', 'HT-Demucs vocals; sung spans labelled by mode'],
  [`Cut to ≤${NB}6${NB}s`, `at pauses ≥${NB}0.4${NB}s, melismas at note gaps; 31,529${NB}utt.`],
  ['Hybrid lyrics', 'Whisper’s syllables, changbon spelling where exact matches pin it'],
  ['Fine-tuned SOFA', `labels bootstrapped from MFA, trusted timings only; 30${NB}ms floor 26.4%${NB}→${NB}2.7%`],
  ['Corpus', `silence under a note joins the phone; 26,955${NB}utt., 28.93${NB}h, 23${NB}singers`],
  ['DiffSinger', `variance 25.1${NB}M + acoustic 69.1${NB}M, rectified flow, 40k${NB}steps; checkpoint per head`],
];
main.forEach(([t, b], i) => box(i, TOP_Y, TOP_H, t, b, 'keep'));
for (let i = 0; i < N - 1; i++) {
  arrow(colX(i) + BOX_W, TOP_Y + TOP_H / 2, colX(i + 1), TOP_Y + TOP_H / 2, { color: KEEP, head: true });
}

// What each step replaced, and the input that joins at the merge.
const low = {
  1: ['Whole sung spans', `>${NB}6${NB}s: MFA aligns a mean 4.06${NB}s as silence`, 'drop'],
  2: ['Snap to changbon', `voiced/unvoiced lost 21–6 (p${NB}=${NB}0.006)`, 'drop'],
  3: ['MFA alone', '52.2% of time silent under a note', 'drop'],
  4: ['ROSVOT notes', 'sliced per utterance', 'input'],
  5: ['Select on total loss', '+25% f0 error', 'drop'],
};
for (const [k, [t, b, kind]] of Object.entries(low)) {
  const i = Number(k), cx = colX(i) + BOX_W / 2;
  box(i, LOW_Y, LOW_H, t, b, kind);
  if (kind === 'input') arrow(cx, LOW_Y, cx, TOP_Y + TOP_H, { color: KEEP, head: true });
  else arrow(cx, LOW_Y, cx, TOP_Y + TOP_H, { color: DROP, dash: true });
}

// Legend in the empty bottom-left cell.
const lx = colX(0), ly = LOW_Y + 0.12;
slide.addShape(pres.shapes.LINE, { x: lx, y: ly + 0.09, w: 0.24, h: 0, line: { color: KEEP, width: 1.25 } });
slide.addText('used', { x: lx + 0.3, y: ly, w: BOX_W, h: 0.18, fontFace: FONT, fontSize: 9,
  color: INK, margin: 0, isTextBox: true });
slide.addShape(pres.shapes.LINE, { x: lx, y: ly + 0.39, w: 0.24, h: 0, line: { color: DROP, width: 1.25, dashType: 'dash' } });
slide.addText('tried, dropped', { x: lx + 0.3, y: ly + 0.3, w: BOX_W, h: 0.18, fontFace: FONT, fontSize: 9,
  color: INK, margin: 0, isTextBox: true });

pres.writeFile({ fileName: path.join(__dirname, 'pipeline.pptx') }).then(f => console.log('wrote', f));
