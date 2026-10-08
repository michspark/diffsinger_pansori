// System figure for the AMIW paper (Fig. 1): one slide sized to the figure.
//
//   NODE_PATH=<dir with pptxgenjs> node pansori/paper/figs/pipeline_fig.js
//   soffice --headless --convert-to pdf --outdir pansori/paper/figs pansori/paper/figs/pipeline.pptx
//
// Top row (offline): how the recordings became the four labels DiffSinger trains on. Bottom
// row (online): the tool's plan, then the two models and the vocoder that render it. What
// was tried and dropped at each step is argued in Section 3, not drawn. Every number is the
// one the paper's text cites; change them there first.
const path = require('path');
const pptxgen = require('pptxgenjs');

const W = 8.0, H = 1.86;
const INK = '1F2D3A', KEEP = '2F5D62', KEEP_FILL = 'E6EFF0', TOOL = '6B4E16', TOOL_FILL = 'F6EEDD',
  MUTED = '5A6670';
const FONT = 'Arial';

const pres = new pptxgen();
pres.defineLayout({ name: 'FIG', width: W, height: H });
pres.layout = 'FIG';
const slide = pres.addSlide();
slide.background = { color: 'FFFFFF' };

const MARGIN = 0.1, GAP = 0.2, N = 6;
const BOX_W = (W - 2 * MARGIN - (N - 1) * GAP) / N;
const colX = i => MARGIN + i * (BOX_W + GAP);
const midX = i => colX(i) + BOX_W / 2;
const ROW_H = 0.62, A_Y = 0.05, B_Y = 0.97, NOTE_Y = B_Y + ROW_H + 0.04;
const BUS_Y = A_Y + ROW_H + 0.15;
const NB = ' ';   // keeps a number on the same line as its unit

const STYLE = {
  data:  { line: KEEP, fill: KEEP_FILL },
  model: { line: INK, fill: 'FFFFFF' },
  tool:  { line: TOOL, fill: TOOL_FILL },
};

function box(i, y, title, body, kind, w = BOX_W) {
  const s = STYLE[kind];
  slide.addText([
    { text: title, options: { bold: true, fontSize: 10, color: INK, breakLine: true } },
    { text: body, options: { fontSize: 8.5, color: INK } },
  ], {
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
    x: colX(i), y, w, h: ROW_H,
    fill: { color: s.fill }, line: { color: s.line, width: kind === 'model' ? 1.5 : 1 },
    fontFace: FONT, valign: 'top', align: 'left', margin: 4, paraSpaceAfter: 1,
    objectName: `${kind}-${i}`,
  });
}

function line(x1, y1, x2, y2, color, head) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: y2 < y1,
    line: { color, width: 1.25, endArrowType: head ? 'triangle' : undefined },
  });
}
const right = (i, y, color) => line(colX(i) + BOX_W, y + ROW_H / 2, colX(i + 1), y + ROW_H / 2, color, true);

function note(x, y, w, text, align) {
  slide.addText(text, { x, y, w, h: 0.2, fontFace: FONT, fontSize: 9, italic: true, color: MUTED,
    margin: 0, align, valign: 'middle', isTextBox: true });
}

// A. Offline: recordings to the four labels.
box(0, A_Y, 'Recordings', '394; HT-Demucs vocals', 'data');
box(1, A_Y, `Cut to ≤${NB}6${NB}s`, `31,529 utterances`, 'data');
box(2, A_Y, 'Phonemes', 'Whisper + changbon spelling', 'data');
box(3, A_Y, 'Phone timing', 'SOFA, fine-tuned from MFA labels', 'data');
box(4, A_Y, 'Notes', 'ROSVOT', 'data');
box(5, A_Y, 'f0', `RMVPE, 55${NB}Hz floor`, 'data');
right(0, A_Y, KEEP);
right(1, A_Y, KEEP);

// The four labels are one corpus, which trains the variance and acoustic models.
for (let i = 2; i < N; i++) line(midX(i), A_Y + ROW_H, midX(i), BUS_Y, KEEP, false);
line(midX(2), BUS_Y, midX(5), BUS_Y, KEEP, false);
line(midX(3), BUS_Y, midX(3), B_Y, KEEP, true);
line(midX(4), BUS_Y, midX(4), B_Y, KEEP, true);
note(MARGIN, BUS_Y - 0.1, midX(2) - MARGIN - 0.08,
  `corpus: 26,955 utt., 28.93${NB}h, 23 singers`, 'right');

// B. Online: the tool's plan, rendered by DiffSinger.
box(0, B_Y, 'Lyrics + mode', 'from the user', 'tool');
box(1, B_Y, 'Retrieve skeletons', `14,244 phrases`, 'tool');
box(2, B_Y, 'Plan', 'user binds syllables to notes', 'tool');
box(3, B_Y, 'Variance', `durations, f0; 25.1${NB}M`, 'model');
box(4, B_Y, 'Acoustic', `mel spectrogram; 69.1${NB}M`, 'model');
box(5, B_Y, 'Vocoder', `NSF-HiFiGAN; 44.1${NB}kHz audio`, 'model');
right(0, B_Y, TOOL);
right(1, B_Y, TOOL);
right(2, B_Y, TOOL);
right(3, B_Y, INK);
right(4, B_Y, INK);
note(colX(0), NOTE_Y, 3 * BOX_W + 2 * GAP, 'the tool: no melody model, editable before rendering', 'center');
note(colX(3), NOTE_Y, 3 * BOX_W + 2 * GAP, 'DiffSinger (openvpi), trained from scratch', 'center');

pres.writeFile({ fileName: path.join(__dirname, 'pipeline.pptx') }).then(f => console.log('wrote', f));
