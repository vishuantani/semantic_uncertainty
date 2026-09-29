"""Render the pipeline diagrams as print-ready figures.

Draws the three explainer figures (explainer/pipeline-overview.html) at paper
scale and writes each as both PNG (300 dpi, for slides and README) and PDF
(vector, for LaTeX \\includegraphics).

    python figures/pipeline/make_pipeline_figures.py

Geometry note: every axes is placed at [0, 0, 1, 1] with figsize chosen so that
100 data units == 1 inch on BOTH axes. That keeps circles round and rounded
corners uniform, and it means a font size of `pt(u)` points renders a glyph `u`
data units tall -- so the layout coordinates below read in the same units as the
lengths around them. The y-axis is inverted so y grows downward, matching the
SVG the layout was designed in.
"""

import pathlib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parent

UNITS_PER_INCH = 100.0
POINTS_PER_UNIT = 72.0 / UNITS_PER_INCH  # 0.72


def pt(units):
    """Data units -> font points, so text scales with the layout."""
    return units * POINTS_PER_UNIT


# --- palette -------------------------------------------------------------------
# Print figure, so this is a fixed light-ground palette rather than a themed one.
# The two accents are separated in lightness as well as hue, so the figure still
# reads when a reviewer prints it in greyscale.
INK       = '#14161C'   # primary text, box outlines that matter
INK_2     = '#4A4F5C'   # secondary text, arrows
INK_3     = '#7B8290'   # annotations, column guides
RULE      = '#C9CCD6'   # box outlines
RULE_SOFT = '#E4E6EC'   # phase bands, hairlines
SURFACE   = '#FFFFFF'
SURFACE_2 = '#F3F4F8'   # non-model compute chips
NEURAL    = '#3F3AA6'   # a neural network runs here
NEURAL_BG = '#EAE9F8'
DELTA     = '#A84A18'   # differs from the upstream implementation
DELTA_BG  = '#FAEFE7'

SANS = ['Helvetica', 'Arial', 'DejaVu Sans']
MONO = ['Menlo', 'DejaVu Sans Mono', 'Courier New']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': SANS,
    'pdf.fonttype': 42,   # embed TrueType so the PDF stays editable/searchable
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})


# --- primitives ----------------------------------------------------------------

def canvas(width_u, height_u):
    fig = plt.figure(figsize=(width_u / UNITS_PER_INCH, height_u / UNITS_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_u)
    ax.set_ylim(height_u, 0)          # inverted: y grows downward
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, face=SURFACE, edge=RULE, lw=1.0, r=5, z=2, ls='solid'):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))


def label(ax, x, y, text, size, color=INK, ha='left', va='baseline',
          weight='normal', mono=False, z=5):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=z,
            fontfamily='monospace' if mono else 'sans-serif',
            **({'fontname': MONO[0]} if mono else {}))


def caps(ax, x, y, text, size, color=INK_3, ha='left', z=5):
    """Small uppercase lane/role label."""
    label(ax, x, y, text.upper(), size, color=color, ha=ha, weight='bold', z=z)


def arrow(ax, x1, y1, x2, y2, color=INK_2, lw=1.1, z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=7,
        connectionstyle=f'arc3,rad={rad}',
        linewidth=lw, color=color, zorder=z,
        shrinkA=0, shrinkB=0))


def hline(ax, x1, x2, y, color=RULE_SOFT, lw=0.8, z=1, ls='solid'):
    ax.plot([x1, x2], [y, y], color=color, linewidth=lw, zorder=z, linestyle=ls)


def vline(ax, x, y1, y2, color=RULE_SOFT, lw=0.8, z=1, ls='solid'):
    ax.plot([x, x], [y1, y2], color=color, linewidth=lw, zorder=z, linestyle=ls)


def dot(ax, x, y, filled=True, r=4.2, z=6):
    ax.add_patch(Circle((x, y), r,
                        facecolor=INK if filled else SURFACE,
                        edgecolor=INK if filled else INK_2,
                        linewidth=0 if filled else 1.2, zorder=z))


def save(fig, stem):
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, **kw)
        print(f'  {path.relative_to(OUT.parents[1])}')
    plt.close(fig)


# =====================================================================================
# Figure 1 -- the pipeline end to end
# =====================================================================================

W1, H1 = 1330, 590
BOX_W, PITCH, X0 = 146, 168, 20
CX = [X0 + PITCH * i + BOX_W / 2 for i in range(7)]
LX = [X0 + PITCH * i for i in range(7)]

Y_BAND, H_BAND = 8, 22
Y_CHIP, H_CHIP = 44, 82
Y_STAGE, H_STAGE = 148, 114

PHASES = [('generate', 0, 0), ('condition the text', 1, 2),
          ('measure', 3, 5), ('evaluate', 6, 6)]

CHIPS = [
    ('generation model', 'opt-350m',          ['beam search ×5 beams', '+ sampling ×5'],        True),
    ('tokenizer only',   'opt-350m tok',      ['string truncation,', 're-tokenize on CPU'],     False),
    ('metrics',          'evaluate',          ['ROUGE-1/2/L and', 'exact match'],               False),
    ('entailment model', 'deberta-base-mnli', ['3-way MNLI head,', 'argmax per pair'],          True),
    ('evaluation model', 'opt-350m',          ['teacher-forced pass,', 'prompt masked out'],    True),
    ('no model',         'torch',             ['logsumexp over', 'clusters, entropy'],          False),
    ('no model',         'pandas / sklearn',  ['merge on question id,', 'roc_auc_score'],       False),
]

STAGES = [
    ('generate',     ['5 sampled answers and', '2 beam answers for', 'each of 40 questions'],   False),
    ('clean',        ['truncate answers at the', 'first turn marker, then', 'rebuild token ids'], True),
    ('score',        ['ROUGE-L and exact', 'match against gold,', 'raw and cleaned'],           True),
    ('similarities', ['group answers into', 'semantic sets by', 'two-way entailment'],          False),
    ('likelihoods',  ['NLL, unconditioned', 'NLL, PMI and hidden', 'states per sequence'],      True),
    ('confidence',   ['predictive entropy,', 'semantic entropy,', 'margin, set counts'],        False),
    ('analyze',      ['AUROC of every', 'measure against', 'correctness'],                      False),
]

# (filename, y, [(stage_index, is_write), ...])
LIFELINES = [
    ('opt-350m_generations.pkl', 318,
     [(0, True), (1, True), (2, True), (3, False), (4, False), (6, False)]),
    ('opt-350m_generations_similarities.pkl', 364,
     [(3, True), (4, False), (6, False)]),
    ('opt-350m_generations_likelihoods.pkl', 444,
     [(4, True), (5, False)]),
    ('aggregated_likelihoods_opt-350m_generations.pkl', 484,
     [(5, True), (6, False)]),
]


def figure_one():
    fig, ax = canvas(W1, H1)

    # phase bands
    for name, a, b in PHASES:
        x, w = LX[a], LX[b] + BOX_W - LX[a]
        box(ax, x, Y_BAND, w, H_BAND, face=RULE_SOFT, edge='none', r=3, z=1)
        caps(ax, x + w / 2, Y_BAND + 15, name, pt(9), ha='center')

    # compute lane -- the answer to "where does a model actually run"
    for i, (role, model, detail, neural) in enumerate(CHIPS):
        face, edge = (NEURAL_BG, NEURAL) if neural else (SURFACE_2, RULE)
        box(ax, LX[i], Y_CHIP, BOX_W, H_CHIP, face=face, edge=edge,
            lw=1.2 if neural else 1.0, r=4)
        caps(ax, CX[i], Y_CHIP + 22, role, pt(8.5),
             color=NEURAL if neural else INK_3, ha='center')
        label(ax, CX[i], Y_CHIP + 45, model,
              pt(11 if len(model) > 15 else 12.5),
              color=NEURAL if neural else INK_2, ha='center',
              weight='bold' if neural else 'normal', mono=True)
        for j, line in enumerate(detail):
            label(ax, CX[i], Y_CHIP + 62 + 13 * j, line, pt(9.5),
                  color=INK_2, ha='center')
        vline(ax, CX[i], Y_CHIP + H_CHIP, Y_STAGE, color=RULE, ls=(0, (2, 2)))

    # stage lane
    for i, (name, desc, changed) in enumerate(STAGES):
        box(ax, LX[i], Y_STAGE, BOX_W, H_STAGE, face=SURFACE, edge=RULE, lw=1.1, r=4)
        label(ax, LX[i] + 12, Y_STAGE + 23, f'{i + 1:02d}', pt(9.5), color=INK_3, mono=True)
        label(ax, LX[i] + 12, Y_STAGE + 46, name, pt(13.5), color=INK, weight='bold')
        for j, line in enumerate(desc):
            label(ax, LX[i] + 12, Y_STAGE + 66 + 14 * j, line, pt(9.5), color=INK_2)
        if changed:
            ax.add_patch(Circle((LX[i] + BOX_W - 13, Y_STAGE + 19), 3.6,
                                facecolor=DELTA, edgecolor='none', zorder=6))
        if i < 6:
            arrow(ax, LX[i] + BOX_W + 1, Y_STAGE + 57, LX[i + 1] - 1, Y_STAGE + 57)

    # artifact lane
    caps(ax, X0, 294, 'artifacts and who touches them', pt(9))
    for i in range(7):
        vline(ax, CX[i], Y_STAGE + H_STAGE, 548, color=RULE_SOFT)

    for fname, y, marks in LIFELINES:
        xs = [CX[i] for i, _ in marks]
        label(ax, min(xs), y - 9, fname, pt(9.5), color=INK_2, mono=True)
        hline(ax, min(xs), max(xs), y, color=INK_3, lw=1.2, z=3)
        for i, is_write in marks:
            dot(ax, CX[i], y, filled=is_write)

    label(ax, CX[1] + 16, 336, '+ cleaned_* fields', pt(8.8), color=INK_3)
    label(ax, CX[2] + 16, 336, '+ rouge / exact_match', pt(8.8), color=INK_3)

    # audit-only artifact: written, never read
    hline(ax, CX[3], CX[3] + 34, 404, color=INK_3, lw=1.2, z=3)
    dot(ax, CX[3], 404, filled=True)
    label(ax, CX[3] + 44, 407, 'deberta-base-mnli_preds.csv', pt(9.5), color=INK_2, mono=True)
    label(ax, CX[3] + 44, 419, 'every pair verdict, kept for audit — nothing downstream reads it',
          pt(8.8), color=INK_3)

    # terminal outputs
    hline(ax, CX[6], CX[6] + 34, 524, color=INK_3, lw=1.2, z=3)
    dot(ax, CX[6], 524, filled=True)
    for j, fname in enumerate(['overall_results.json', 'accuracy_verification.csv',
                               'sequence_embeddings.pkl']):
        label(ax, CX[6] + 44, 514 + 14 * j, fname, pt(9.5), color=INK_2, mono=True)

    # legend
    x = X0
    items = [('swatch', NEURAL_BG, NEURAL, 'neural forward pass'),
             ('swatch', SURFACE_2, RULE, 'CPU only, no model'),
             ('dot', DELTA, None, 'changed in this fork'),
             ('dot', INK, None, 'writes the file'),
             ('ring', SURFACE, INK_2, 'reads the file')]
    for kind, face, edge, text in items:
        if kind == 'swatch':
            box(ax, x, 566, 11, 11, face=face, edge=edge, lw=1.2, r=2, z=5)
        else:
            ax.add_patch(Circle((x + 5.5, 571.5), 4.6, facecolor=face,
                                edgecolor=edge or 'none',
                                linewidth=0 if edge is None else 1.2, zorder=5))
        label(ax, x + 19, 575, text, pt(9.5), color=INK_2)
        x += 19 + len(text) * 4.6 + 30

    save(fig, 'fig1_pipeline_overview')


# =====================================================================================
# Figure 2 -- inside `generate`: one model, two decoding rules
# =====================================================================================

def figure_two():
    fig, ax = canvas(1060, 400)

    # prompt
    box(ax, 20, 150, 190, 104, face=SURFACE_2, edge=RULE, r=4)
    caps(ax, 34, 174, 'prompt', pt(9))
    for j, line in enumerate(['10-shot preamble', 'Question: <question>', 'Answer:']):
        label(ax, 34, 196 + 16 * j, line, pt(9.5), color=INK_2, mono=True)
    label(ax, 34, 246, 'built once by parse_triviaqa.py', pt(8.8), color=INK_3)
    arrow(ax, 212, 202, 262, 202)

    # the single loaded model
    box(ax, 268, 132, 184, 140, face=NEURAL_BG, edge=NEURAL, lw=1.3, r=4)
    caps(ax, 360, 160, 'generation model', pt(8.5), color=NEURAL, ha='center')
    label(ax, 360, 184, 'facebook/opt-350m', pt(11), color=NEURAL,
          ha='center', weight='bold', mono=True)
    label(ax, 360, 203, 'AutoModelForCausalLM', pt(9.5), color=INK_2, ha='center')
    label(ax, 360, 230, 'one weight load,', pt(9.5), color=INK_2, ha='center')
    label(ax, 360, 244, 'two decoding rules', pt(9.5), color=INK_2, ha='center')

    # the two decoding branches
    arrow(ax, 452, 176, 552, 103, rad=0.28)
    arrow(ax, 452, 230, 552, 301, rad=-0.28)

    branches = [
        (44,  'beam search',
         ['num_beams=5  do_sample=False', 'num_return_sequences=2'],
         ['most_likely_generation', 'second_most_likely_generation'],
         'judged for correctness',
         ['stage 03 — ROUGE-L vs gold answers', 'stage 05 — feeds the margin measure']),
        (242, 'multinomial sampling',
         ['do_sample=True  T=1.0  top_p=1.0', 'drawn 5 times per question'],
         ['generated_texts[0...4]', 'the disagreement signal'],
         'used as the evidence',
         ['stage 04 — grouped into meaning sets', 'stage 06 — becomes every entropy']),
    ]
    for y, title, params, outputs, verdict, notes in branches:
        box(ax, 560, y, 252, 118, face=SURFACE, edge=RULE, lw=1.1, r=4)
        label(ax, 574, y + 26, title, pt(11), color=INK, weight='bold')
        for j, line in enumerate(params):
            label(ax, 574, y + 46 + 14 * j, line, pt(9.5), color=INK_2, mono=True)
        hline(ax, 574, 798, y + 72, color=RULE)
        for j, line in enumerate(outputs):
            label(ax, 574, y + 90 + 16 * j, line, pt(10), color=INK_2)

        arrow(ax, 814, y + 59, 854, y + 59)
        label(ax, 864, y + 48, verdict, pt(11), color=INK, weight='bold')
        for j, line in enumerate(notes):
            label(ax, 864, y + 65 + 14 * j, line, pt(9), color=INK_3)

    # both branches are stopped the same way
    vline(ax, 360, 272, 300, color=RULE, ls=(0, (2, 2)))
    box(ax, 20, 300, 420, 84, face=DELTA_BG, edge=DELTA, lw=1.2, r=4, ls=(0, (4, 3)))
    caps(ax, 34, 324, 'shared stopping control', pt(8.5), color=DELTA)
    label(ax, 34, 344, "eos_token_id  =  the '.' token", pt(9.5), color=INK_2, mono=True)
    label(ax, 34, 360, 'bad_words_ids =  15 banned turn markers', pt(9.5), color=INK_2, mono=True)
    label(ax, 34, 376, "the fork widened this list; opt-350m escapes the paper's six via lowercase",
          pt(8.8), color=INK_3)

    save(fig, 'fig2_generate_decoding')


# =====================================================================================
# Figure 3 -- clustering, and what it changes about the entropy
# =====================================================================================

def figure_three():
    fig, ax = canvas(1080, 470)

    # panel A: the sampled answers
    caps(ax, 20, 36, 'sampled answers', pt(9))
    label(ax, 20, 52, '5 samples, 4 unique after dedup', pt(9), color=INK_3)
    for j, text in enumerate(['William Golding', 'Golding',
                              'The author is William Golding', 'Peter Benchley']):
        box(ax, 20, 62 + 34 * j, 270, 26, face=SURFACE_2, edge=RULE, r=3)
        label(ax, 32, 79 + 34 * j, text, pt(9.5), color=INK_2, mono=True)
    label(ax, 20, 212, 'for the question “Who wrote Lord of the Flies?”', pt(9), color=INK_3)
    arrow(ax, 298, 126, 322, 126)

    # panel B: the entailment test
    caps(ax, 330, 36, 'bidirectional entailment', pt(9))
    # no arrows here: Helvetica has no U+2192 (the mono lines below use Menlo, which does)
    label(ax, 330, 52, '4 answers, 6 pairs, 12 DeBERTa forward inputs', pt(9), color=INK_3)
    box(ax, 330, 62, 390, 152, face=SURFACE, edge=RULE, lw=1.1, r=4)
    label(ax, 344, 88, 'one pair, run both ways', pt(10.5), color=INK, weight='bold')
    label(ax, 344, 112, 'Q + a1  [SEP]  Q + a2   →  entailment', pt(9.5), color=INK_2, mono=True)
    label(ax, 344, 128, 'Q + a2  [SEP]  Q + a1   →  neutral', pt(9.5), color=INK_2, mono=True)
    hline(ax, 344, 706, 140, color=RULE)
    label(ax, 344, 160, 'neither direction returned contradiction', pt(9.5), color=INK_2)
    label(ax, 344, 178, "so a2 is merged into a1's semantic set", pt(9.5), color=NEURAL, weight='bold')
    label(ax, 344, 200, 'a single contradiction, either way, keeps the two apart',
          pt(9), color=INK_3)
    label(ax, 330, 234, 'argmax over the 3-way MNLI head; only label 0 (contradiction) splits a pair',
          pt(9), color=INK_3)
    arrow(ax, 728, 126, 752, 126)

    # panel C: the resulting sets
    caps(ax, 760, 36, 'semantic sets', pt(9))
    label(ax, 760, 52, 'clusters, not strings, are what get counted', pt(9), color=INK_3)
    box(ax, 760, 62, 300, 106, face=NEURAL_BG, edge=NEURAL, lw=1.2, r=4)
    caps(ax, 774, 86, 'set 1', pt(8.5), color=NEURAL)
    for j, text in enumerate(['William Golding', 'Golding', 'The author is William Golding']):
        label(ax, 774, 110 + 18 * j, text, pt(9.5), color=INK_2, mono=True)
    box(ax, 760, 182, 300, 54, face=SURFACE_2, edge=RULE, r=4)
    caps(ax, 774, 204, 'set 2', pt(8.5))
    label(ax, 774, 226, 'Peter Benchley', pt(9.5), color=INK_2, mono=True)
    label(ax, 760, 262, 'number_of_semantic_sets = 2', pt(10), color=INK, mono=True)

    # the two entropies, differing only in the pooling step
    hline(ax, 20, 1060, 282, color=RULE_SOFT)
    panels = [
        (20, SURFACE, RULE, 1.1, INK_3, 'predictive entropy — over sequences',
         'PE = -(1/N) * sum_i log p(s_i | x)',
         ['Every sample is its own outcome. Three ways of',
          'saying Golding count as three, so a model that is',
          'confident but verbose reads as uncertain.'],
         'length-normalised variant: average_predictive_entropy'),
        (560, NEURAL_BG, NEURAL, 1.2, NEURAL, 'semantic entropy — over meaning sets',
         'SE = -(1/|C|) * sum_c logsumexp_{i in c} log p(s_i | x)',
         ['Likelihood is pooled inside each set before the',
          'average, so the three Golding samples contribute',
          'one term. Only real disagreement survives.'],
         'code subtracts --llh_shift (default 5.0) from each pooled term'),
    ]
    for x, face, edge, lw, headc, head, formula, body, foot in panels:
        box(ax, x, 296, 500, 158, face=face, edge=edge, lw=lw, r=4)
        caps(ax, x + 16, 320, head, pt(8.5), color=headc)
        label(ax, x + 16, 350, formula, pt(11.5), color=INK, mono=True)
        for j, line in enumerate(body):
            label(ax, x + 16, 382 + 16 * j, line, pt(10), color=INK_2)
        label(ax, x + 16, 436, foot, pt(9), color=INK_3)

    save(fig, 'fig3_semantic_clustering')


if __name__ == '__main__':
    print('rendering:')
    figure_one()
    figure_two()
    figure_three()
