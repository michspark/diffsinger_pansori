// Pipeline figure for the AMIW paper (Fig. 1): one slide sized to the figure.
//
//   NODE_PATH=<dir with pptxgenjs> node pansori/paper/figs/pipeline_fig.js
//   soffice --headless --convert-to pdf --outdir pansori/paper/figs pansori/paper/figs/pipeline.pptx
//
// One box per paragraph of Section 3, titled as the paragraph is: Corpus, Segmentation, Lyrics,
// Alignment, Singing model. The text path (top) and the pitch path (bottom) both start from the
// segmented utterances and meet at the singing model; each box ends with the label it hands
// on. Corpus sizes and step counts stay in the text: the figure shows the order of the steps.
const path = require('path');
const pptxgen = require('pptxgenjs');

const W = 8.0, H = 1.66;
const INK = '1F2D3A', KEEP = '2F5D62', KEEP_FILL = 'E6EFF0', MUTED = '5A6670';
const FONT = 'Arial';

const pres = new pptxgen();
pres.defineLayout({ name: 'FIG', width: W, height: H });
pres.layout = 'FIG';
const slide = pres.addSlide();
slide.background = { color: 'FFFFFF' };

const MARGIN = 0.05, GAP = 0.18, BOX_W = 1.1, ROW_H = 0.72;
const A_Y = 0.05, B_Y = A_Y + ROW_H + 0.12;
const colX = i => MARGIN + i * (BOX_W + GAP);
const midX = i => colX(i) + BOX_W / 2;
const NB = ' ';   // keeps a number on the same line as its unit

function box(x, y, w, title, body, out) {
  slide.addText([
    { text: title, options: { bold: true, fontSize: 10, color: INK, breakLine: true } },
    { text: body, options: { fontSize: 8.5, color: INK, breakLine: true } },
    { text: `→ ${out}`, options: { fontSize: 8.5, italic: true, color: KEEP } },
  ], {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
    x, y, w, h: ROW_H,
    fill: { color: KEEP_FILL }, line: { color: KEEP, width: 1 },
    fontFace: FONT, valign: 'top', align: 'left', margin: 4, paraSpaceAfter: 1,
  });
}

function line(x1, y1, x2, y2, color, head) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1,
    line: { color, width: 1.25, endArrowType: head ? 'triangle' : undefined },
  });
}

// Data preparation: the first four paragraphs, plus the notes and f0 the Segmentation
// paragraph takes from the cut utterances.
box(colX(0), A_Y, BOX_W, 'Corpus', 'HT-Demucs vocals; no aniri', 'sung spans');
box(colX(1), A_Y, BOX_W, 'Segmentation', `pauses, melisma gaps; ≤${NB}6${NB}s`, 'utterances');
box(colX(2), A_Y, BOX_W, 'Lyrics', 'Whisper + changbon', 'phonemes');
box(colX(3), A_Y, BOX_W, 'Alignment', 'SOFA, fine-tuned from MFA labels', 'boundaries');
box(colX(2), B_Y, 2 * BOX_W + GAP, 'Notes and f0', 'ROSVOT notes, RMVPE f0', 'notes, f0');

const midA = A_Y + ROW_H / 2, midB = B_Y + ROW_H / 2;
for (let i = 0; i < 3; i++) line(colX(i) + BOX_W, midA, colX(i + 1), midA, KEEP, true);
line(midX(1), A_Y + ROW_H, midX(1), midB, KEEP, false);
line(midX(1), midB, colX(2), midB, KEEP, true);

// Singing model: the four labels train the variance and acoustic models; the vocoder is
// pretrained.
const SX = colX(4), SW = W - MARGIN - SX, SY = A_Y, SH = B_Y + ROW_H - A_Y;
slide.addText([
  { text: 'Singing model', options: { bold: true, fontSize: 10, color: INK } },
  { text: '  DiffSinger (openvpi)', options: { fontSize: 8.5, italic: true, color: MUTED } },
], {
  shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
  x: SX, y: SY, w: SW, h: SH, fill: { color: 'FFFFFF' }, line: { color: INK, width: 1.5 },
  fontFace: FONT, valign: 'top', align: 'left', margin: 4,
});
line(colX(3) + BOX_W, midA, SX, midA, KEEP, true);
line(colX(3) + BOX_W, midB, SX, midB, KEEP, true);

const IN_PAD = 0.07, IN_GAP = 0.13, IN_H = 0.6;
const IN_W = (SW - 2 * IN_PAD - 2 * IN_GAP) / 3, IN_Y = SY + (SH - IN_H) / 2 + 0.08;
[['Variance', 'durations, f0'], ['Acoustic', 'mel spectrogram'], ['Vocoder', 'NSF-HiFiGAN']]
  .forEach(([title, body], k) => {
    const x = SX + IN_PAD + k * (IN_W + IN_GAP);
    slide.addText([
      { text: title, options: { bold: true, fontSize: 9, color: INK, breakLine: true } },
      { text: body, options: { fontSize: 7.5, color: INK } },
    ], {
      shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05,
      x, y: IN_Y, w: IN_W, h: IN_H, fill: { color: 'FFFFFF' }, line: { color: INK, width: 1 },
      fontFace: FONT, valign: 'top', align: 'left', margin: 2, paraSpaceAfter: 1,
    });
    if (k < 2) line(x + IN_W, IN_Y + IN_H / 2, x + IN_W + IN_GAP, IN_Y + IN_H / 2, INK, true);
  });

pres.writeFile({ fileName: path.join(__dirname, 'pipeline.pptx') }).then(f => console.log('wrote', f));
