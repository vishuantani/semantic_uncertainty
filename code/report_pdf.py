"""Render one run's headline metrics as a single-page PDF and upload it to W&B.

    python code/report_pdf.py --run_id opt1.3b-120q-eos --generation_model opt-1.3b

Page one is a dashboard, deliberately dense: it is meant to be put next to another run's
page and compared, and nothing on it is defined. Page two bins the questions by confidence
and plots observed accuracy per bin -- the one view that shows WHERE on the score range a
measure's separation lives, which AUROC compresses away and the cumulative risk-coverage
curve hides. Page three is the glossary, written against what this pipeline actually
computes rather than the textbook form of each measure. --no_bins and --no_glossary drop
their pages; both together give the bare one-pager.

EVERY NUMBER COMES FROM log_wandb_charts.ChartLogger. This stage computes no statistics of
its own -- it loads the frame, the bootstrap AUROCs and the coverage curves through that
class and only lays them out. That is the point: a report that recomputed its own AUROCs
could disagree with the W&B panels for the same run, and then neither could be trusted.
The two figures are the same figures too, via figure_risk_coverage/figure_auroc_bounds.

The page is A4 LANDSCAPE. Portrait would put the two charts in ~7.6cm slots, and since
matplotlib sizes text in points against the figure's inches, their 9pt labels would land
at about 2.5pt on the page. Landscape buys ~13cm per slot, which is enough to render each
figure at its final size and have the type come out at the size it claims.

Upload is deliberately two things at once:
  - wandb.save()     -> the Files tab, where W&B renders the PDF inline in the browser
  - wandb.Artifact   -> a versioned copy you can pull back later by run

--dry_run skips W&B entirely and just writes the file, which is also how this stage is
tested against a finished run without touching the tracked experiment.
"""

import argparse
import datetime
import os
import pathlib
import sys
import tempfile

import numpy as np
import wandb

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config  # noqa: E402
from log_wandb_charts import (BIN_COUNT, COVERAGE_GRID, INK, LABEL_OF,  # noqa: E402
                              MEASURES, MUTED, OPT_MODELS, PRIMARY_MEASURES,
                              REPORTED_COVERAGES, SURFACE, TARGET_ACCURACY, ChartLogger,
                              figure_accuracy_bins, figure_auroc_bounds,
                              figure_risk_coverage)

# Slot geometry. CHART_INCHES is the width each figure is RENDERED at as well as the width
# it is PLACED at, so 1 figure inch == 1 page inch and no text is scaled after the fact.
PAGE = landscape(A4)
MARGIN = 1.1 * cm
CONTENT_WIDTH = PAGE[0] - 2 * MARGIN
CHART_INCHES = 5.15
CHART_WIDTH = CHART_INCHES * 72              # reportlab points, 72 per inch
RISK_FIGSIZE = (CHART_INCHES, 3.02)
AUROC_FIGSIZE = (CHART_INCHES, 3.02)
# Every font in the figures is multiplied by this. At half the default canvas width the
# unscaled 9-12pt type covers twice the fraction of the figure it was laid out for, and
# the titles, legends and tick labels collide. 0.72 puts 9pt at ~6.5pt on the page, which
# is small print but legible, and restores the original proportions.
CHART_SCALE = 0.72

# Page two gets one chart across the full width, so it is rendered at its placed size with
# no font scaling at all.
BINS_FIGSIZE = (CONTENT_WIDTH / 72, 4.6)

# The bar chart gives each measure ~0.7in of x-axis. The full labels do not fit there even
# rotated, and the AUROC table directly above already prints them in full.
SHORT_LABEL = {
    'semantic entropy': 'semantic entropy',
    'ln predictive entropy': 'ln pred. entropy',
    'predictive entropy': 'pred. entropy',
    'unnormalised entropy/concepts': 'unnorm. ent./concepts',
    'number of semantic sets': '# semantic sets',
    'neg llh most likely gen': 'neg llh most likely',
    'avg neg llh most likely gen': 'avg neg llh most lkly',
}


# ---------------------------------------------------------------------------- glossary
# Page two. Definitions describe what THIS pipeline computes, not the general literature
# form of each measure -- where the two differ (length normalisation, which model scores
# the prior, the 0.3 correctness cut) the code is what is written down.
GLOSSARY = [
    ('Uncertainty measures', [
        ('predictive entropy',
         'Entropy over the sampled generations, using each sequence&rsquo;s summed '
         'log-probability. High when the model spreads its probability over many '
         'different continuations. Not length-normalised, so long answers depress it.'),
        ('ln predictive entropy',
         'Predictive entropy with each sequence&rsquo;s log-probability divided by its '
         'token count first, which removes the &ldquo;longer answers are always less '
         'likely&rdquo; bias. Logged as <font face="Courier">average_predictive_entropy</font>.'),
        ('semantic entropy',
         'The measure this repo exists for. Generations are first clustered into meaning '
         'classes by a bidirectional-entailment model, then entropy is taken over the '
         'CLUSTERS, so ten phrasings of one answer count once instead of ten times. '
         'Length-normalised. Logged as '
         '<font face="Courier">predictive_entropy_over_concepts</font>.'),
        ('unnormalised entropy/concepts',
         'Semantic entropy without the length normalisation &mdash; same clustering, raw '
         'summed log-probabilities.'),
        ('number of semantic sets',
         'How many distinct meaning clusters the generations fell into. The crudest '
         'signal here: it is a small integer, so many questions tie, which caps how well '
         'it can rank.'),
        ('neg llh most likely gen',
         'Negative log-likelihood of the single most-likely answer (beam/greedy), summed '
         'over its tokens. A one-sample confidence signal &mdash; it ignores the sampled '
         'generations entirely.'),
        ('avg neg llh most likely gen',
         'The same quantity divided by token count.'),
    ]),
    ('Selective prediction', [
        ('coverage',
         'The fraction of questions you choose to answer, taking the most confident '
         'first and abstaining on the rest. Every point on the risk-coverage curve is a '
         'deployable configuration.'),
        ('acc@N',
         'Accuracy among the most-confident N% of questions. Read the low-coverage end '
         'with care: at 10% coverage a single question moves it several points.'),
        ('area',
         'Area under the accuracy-vs-coverage curve, on a fixed 2% grid so runs with '
         'different question counts stay comparable. Higher is better and the base rate '
         'is the floor &mdash; this is the accuracy convention, not the risk convention '
         'where lower wins.'),
        ('cov@50',
         'The largest fraction of questions you can answer while still holding 50% '
         'accuracy. Usually the most quotable number on the page. 0 means the target is '
         'never reached at any coverage.'),
        ('oracle / base rate',
         'The two references on the risk-coverage chart. The oracle ranks by correctness '
         'itself &mdash; the ceiling no measure can beat. The base rate is the run&rsquo;s '
         'overall accuracy &mdash; the floor you get by answering everything.'),
    ]),
    ('Method and conventions', [
        ('AUROC',
         'The probability that a randomly chosen INCORRECT answer is scored more '
         'uncertain than a randomly chosen correct one. Equivalently: form every '
         '(wrong, right) pair of questions and count the fraction the measure ranks the '
         'right way round, ties counting half. 0.5 is chance. It depends only on the '
         'ORDER of the scores, never their scale, and is unaffected by the base rate.'),
        ('95% CI',
         'A non-parametric bootstrap over questions: resample the question set with '
         'replacement, recompute AUROC, take the 2.5th and 97.5th percentiles. A point '
         'estimate on this few positives is not a result &mdash; trust only the rows '
         'whose interval clears 0.5.'),
        ('beats chance',
         'Whether the lower bound of that interval sits above 0.5. This, not the AUROC '
         'itself, is what makes a measure a finding.'),
        ('correct',
         'ROUGE-L F-measure against the best-matching reference answer, thresholded at '
         '&gt; 0.3. ROUGE-L is longest-common-subsequence overlap turned into an F1, so '
         'precision punishes verbosity: a right answer buried in a wordy sentence scores '
         'as wrong. Every AUROC on this page ranks against this label.'),
        ('confidence bin',
         'Page two splits the questions into five equal groups by a measure&rsquo;s score, '
         'most confident first, and reports observed accuracy in each. Bins are '
         'DISJOINT, unlike coverage, which is cumulative &mdash; that is what lets them '
         'show whether a measure grades across its whole range or only isolates a good '
         'top group. Not a calibration chart: there is no claimed probability to check.'),
        ('Wilson interval',
         'The 95% binomial interval on a bin&rsquo;s accuracy. Wilson rather than the '
         'normal approximation because bins at the confident end sit near 0, where the '
         'normal interval runs below zero and collapses to zero width at exactly 0/n.'),
        ('semantic set',
         'A cluster of generations that mutually entail each other, per the DeBERTa '
         'entailment model in the similarities stage. The unit semantic entropy is '
         'computed over.'),
        ('not reported here',
         'Pointwise mutual information is computed by the likelihoods stage but no AUROC '
         'is taken on it. No calibration metric (ECE, Brier, reliability diagram) exists '
         'anywhere in this pipeline, and page two is not one: every number in this report '
         'measures ranking or selective accuracy, none of them measure whether a score is '
         'numerically meaningful. Getting a real ECE would mean fitting a mapping from '
         'entropy to P(correct) and validating it out-of-fold, which this many positives '
         'will not support.'),
    ]),
]

RULE = colors.HexColor('#DEDEE4')
INK_C = colors.HexColor(INK)
MUTED_C = colors.HexColor(MUTED)
SURFACE_C = colors.HexColor(SURFACE)
BAND = colors.HexColor('#F2F2F6')            # zebra fill, one step off SURFACE
GOOD = colors.HexColor('#00967A')            # reused from SERIES_COLOURS
WEAK = colors.HexColor('#C05F26')


def style(name, size, colour=INK_C, leading=None, bold=False, space_after=0):
    return ParagraphStyle(
        name, fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=size,
        leading=leading or size * 1.25, textColor=colour, alignment=TA_LEFT,
        spaceAfter=space_after)


TITLE = style('title', 16, bold=True)
SUBTITLE = style('subtitle', 8.5, MUTED_C)
SECTION = style('section', 9, MUTED_C, bold=True)
CELL = style('cell', 8)
CELL_HEAD = style('cellhead', 7.5, MUTED_C, bold=True)
FOOT = style('foot', 7.2, MUTED_C, leading=9.6)
KPI_VALUE = style('kpivalue', 15, INK_C, bold=True)
KPI_LABEL = style('kpilabel', 7, MUTED_C)
GLOSS_TERM = style('glossterm', 8, INK_C, bold=True)
GLOSS_BODY = style('glossbody', 7.4, MUTED_C, leading=9.4)
GLOSS_GROUP = style('glossgroup', 9, MUTED_C, bold=True)


def fmt(value, places=3):
    """None and NaN both mean 'could not be computed', and must not print as 0.000."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '--'
    return f'{value:.{places}f}'


class ReportBuilder:
    """Turns one run's ChartLogger output into a one-page PDF."""

    def __init__(self, args):
        self.args = args
        self.logger = ChartLogger(args)

    # ------------------------------------------------------------------ statistics

    def collect(self, run_name):
        """Every number the page shows, computed once, all of it via ChartLogger.

        Mirrors the assembly in ChartLogger._panel_* rather than calling those methods:
        they build wandb.Image objects and mutate a `logged` dict as a side effect, and
        this stage wants the statistics without either.
        """
        frame = self.logger._load_frame(run_name)
        correct = frame['correct']
        incorrect = 1 - correct

        auroc_rows = []
        for key, label in MEASURES:
            if key not in frame:
                continue
            result = self.logger._auroc_with_ci(incorrect, frame[key])
            if result is None:
                continue
            auroc_rows.append({'key': key, 'label': label, **result})
        auroc_rows.sort(key=lambda r: -r['auroc'])

        coverage_rows, curves = [], []
        for key in PRIMARY_MEASURES:
            if key not in frame:
                continue
            accuracies = self.logger._accuracy_by_coverage(correct, frame[key])
            curves.append((LABEL_OF[key], accuracies))
            at = {}
            for coverage in REPORTED_COVERAGES:
                index = int(np.argmin(np.abs(COVERAGE_GRID - coverage)))
                at[coverage] = accuracies[index]
            reached = [c for c, a in zip(COVERAGE_GRID, accuracies) if a >= TARGET_ACCURACY]
            coverage_rows.append({
                'key': key,
                'label': LABEL_OF[key],
                'at': at,
                'area': float(np.trapezoid(accuracies, COVERAGE_GRID)),
                'coverage_at_target': float(max(reached)) if reached else 0.0,
            })

        bin_panels = [
            {'key': key, 'label': LABEL_OF[key],
             'bins': self.logger._accuracy_by_bin(correct, frame[key])}
            for key in PRIMARY_MEASURES if key in frame
        ]

        return {
            'run_name': run_name,
            'bin_panels': bin_panels,
            'n_questions': int(len(correct)),
            'n_correct': int(correct.sum()),
            'accuracy': float(correct.mean()),
            'auroc_rows': auroc_rows,
            'coverage_rows': coverage_rows,
            'curves': curves,
            'oracle': self.logger._accuracy_by_coverage(correct, None),
        }

    # ------------------------------------------------------------------ page pieces

    def _header(self, data):
        stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        return [
            Paragraph('Semantic Uncertainty &mdash; Run Report', TITLE),
            Paragraph(
                f"<b>{data['run_name']}</b> &nbsp;&middot;&nbsp; {self.args.generation_model}"
                f" &nbsp;&middot;&nbsp; {data['n_questions']} questions"
                f" &nbsp;&middot;&nbsp; bootstrap {self.args.bootstrap_resamples} resamples,"
                f" seed {self.args.seed} &nbsp;&middot;&nbsp; generated {stamp}",
                SUBTITLE),
            Spacer(1, 0.32 * cm),
        ]

    def _kpis(self, data):
        best = data['auroc_rows'][0] if data['auroc_rows'] else None
        # Coverage at the target accuracy, taken from whichever measure reaches furthest:
        # it is the one number on the page that states a deployable configuration.
        best_cov = max(data['coverage_rows'], key=lambda r: r['coverage_at_target'],
                       default=None)

        cells = [
            (fmt(data['accuracy']), 'accuracy'),
            (f"{data['n_correct']} / {data['n_questions']}", 'correct / total'),
            (fmt(best['auroc']) if best else '--',
             f"best AUROC &mdash; {best['label']}" if best else 'best AUROC'),
            (f"{best_cov['coverage_at_target']:.0%}" if best_cov else '--',
             f"coverage at {TARGET_ACCURACY:.0%} acc &mdash; {best_cov['label']}"
             if best_cov else f'coverage at {TARGET_ACCURACY:.0%} acc'),
            (str(sum(1 for r in data['auroc_rows']
                     if r['ci_lo'] is not None and r['ci_lo'] > 0.5)),
             f"of {len(data['auroc_rows'])} measures beat chance"),
        ]
        table = Table([[Paragraph(v, KPI_VALUE) for v, _ in cells],
                       [Paragraph(l, KPI_LABEL) for _, l in cells]],
                      colWidths=[CONTENT_WIDTH / len(cells)] * len(cells))
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, 0), 2),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 0),
            ('TOPPADDING', (0, 1), (-1, 1), 1),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('LINEBELOW', (0, -1), (-1, -1), 0.6, RULE),
            ('BOTTOMPADDING', (0, -1), (-1, -1), 6),
        ]))
        return [table, Spacer(1, 0.3 * cm)]

    def _auroc_table(self, data, width):
        head = ['measure', 'AUROC', '95% CI', 'beats chance']
        rows = [[Paragraph(h, CELL_HEAD) for h in head]]
        flags = []
        for result in data['auroc_rows']:
            beats = result['ci_lo'] is not None and result['ci_lo'] > 0.5
            flags.append(beats)
            interval = ('--' if result['ci_lo'] is None
                        else f"{fmt(result['ci_lo'])} - {fmt(result['ci_hi'])}")
            rows.append([Paragraph(result['label'], CELL),
                         Paragraph(f"<b>{fmt(result['auroc'])}</b>", CELL),
                         Paragraph(interval, CELL),
                         Paragraph('yes' if beats else 'no', CELL)])

        table = Table(rows, colWidths=[w * width for w in (0.44, 0.16, 0.24, 0.16)],
                      repeatRows=1)
        commands = self._base_table_style(len(rows))
        # Colour only the verdict column: the AUROC itself is not good or bad, the
        # interval clearing chance is what makes it a finding.
        for index, beats in enumerate(flags, start=1):
            commands.append(('TEXTCOLOR', (3, index), (3, index), GOOD if beats else WEAK))
        table.setStyle(TableStyle(commands))
        return table

    def _coverage_table(self, data, width):
        head = (['measure'] + [f'acc@{int(c * 100)}' for c in REPORTED_COVERAGES]
                + ['area', f'cov@{int(TARGET_ACCURACY * 100)}'])
        rows = [[Paragraph(h, CELL_HEAD) for h in head]]
        for result in data['coverage_rows']:
            rows.append(
                [Paragraph(result['label'], CELL)]
                + [Paragraph(fmt(result['at'][c]), CELL) for c in REPORTED_COVERAGES]
                + [Paragraph(fmt(result['area']), CELL),
                   Paragraph(f"{result['coverage_at_target']:.0%}", CELL)])

        widths = [0.32, 0.13, 0.13, 0.13, 0.13, 0.16]
        table = Table(rows, colWidths=[w * width for w in widths], repeatRows=1)
        table.setStyle(TableStyle(self._base_table_style(len(rows))))
        return table

    @staticmethod
    def _base_table_style(n_rows):
        commands = [
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK_C),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]
        # Zebra banding rather than gridlines: the rows are short and a full grid would
        # add 30 rules to a page that already has two charts on it.
        for index in range(2, n_rows, 2):
            commands.append(('BACKGROUND', (0, index), (-1, index), BAND))
        return commands

    def _tables_row(self, data):
        gutter = 0.5 * cm
        left = (CONTENT_WIDTH - gutter) * 0.54
        right = (CONTENT_WIDTH - gutter) * 0.46
        inner = Table(
            [[Paragraph('DISCRIMINATION &mdash; AUROC, incorrectness as positive label',
                        SECTION),
              Paragraph('SELECTIVE PREDICTION &mdash; accuracy when answering only the '
                        'most confident', SECTION)],
             [self._auroc_table(data, left), self._coverage_table(data, right)]],
            colWidths=[left, right + gutter])
        inner.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (0, -1), 0),
            ('RIGHTPADDING', (0, 0), (0, -1), gutter),
            ('LEFTPADDING', (1, 0), (1, -1), 0),
            ('RIGHTPADDING', (1, 0), (1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ]))
        return [inner, Spacer(1, 0.26 * cm)]

    def _charts_row(self, data, tmpdir):
        """Both figures rendered at exactly the width they are placed at."""
        risk = figure_risk_coverage(
            data['curves'], data['oracle'], data['accuracy'], '',
            figsize=RISK_FIGSIZE, scale=CHART_SCALE, show_title=False)
        auroc = figure_auroc_bounds(
            data['auroc_rows'], '', figsize=AUROC_FIGSIZE, scale=CHART_SCALE,
            show_title=False,
            labels=[SHORT_LABEL.get(r['label'], r['label']) for r in data['auroc_rows']])

        images = []
        for name, figure, (_, height) in (('risk_coverage', risk, RISK_FIGSIZE),
                                          ('auroc_bounds', auroc, AUROC_FIGSIZE)):
            path = os.path.join(tmpdir, f'{name}.png')
            figure.tight_layout()
            figure.savefig(path, dpi=200, facecolor=SURFACE)
            figure.clf()
            images.append(Image(path, width=CHART_WIDTH, height=height * 72))

        row = Table(
            [[Paragraph('RISK-COVERAGE &mdash; accuracy vs. fraction answered', SECTION),
              Paragraph('AUROC &mdash; bars measured from chance, with 95% interval',
                        SECTION)],
             images],
            colWidths=[CONTENT_WIDTH / 2] * 2)
        row.setStyle(TableStyle([
            ('ALIGN', (0, 1), (0, 1), 'LEFT'), ('ALIGN', (1, 1), (1, 1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 3),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        return [row]

    def _footer(self, data):
        return [
            Spacer(1, 0.2 * cm),
            Paragraph(
                '<b>Conventions.</b> AUROC takes <b>incorrectness</b> as the positive '
                'label, so a good uncertainty measure scores above 0.5; it is the '
                'probability that a randomly chosen wrong answer is ranked more '
                'uncertain than a randomly chosen right one, and it depends only on the '
                'ranking of the scores, never their scale. Correctness is '
                '<b>rougeL_to_target &gt; 0.3</b> against the best-matching reference '
                'answer &mdash; a lexical proxy that penalises verbosity, so a correct '
                f"but wordy answer can score as wrong. With {data['n_correct']} correct "
                f"of {data['n_questions']}, one flipped label moves every AUROC by "
                f"roughly {1.0 / max(data['n_correct'], 1):.1%} of its pairs: read the "
                'intervals, not the point estimates. Risk-coverage uses the '
                '<b>accuracy</b> convention (higher is better, base rate is the floor), '
                'not the risk convention: <b>acc@N</b> is accuracy when answering the '
                'most-confident N% of questions, <b>area</b> is the area under that '
                f"curve, and <b>cov@{int(TARGET_ACCURACY * 100)}</b> is the largest "
                'fraction answerable while still holding '
                f"{TARGET_ACCURACY:.0%} accuracy. No calibration metric (ECE, Brier) is "
                'reported &mdash; this pipeline computes none.',
                FOOT),
        ]

    def _bins_page(self, data, tmpdir):
        """Page two: accuracy within each confidence bin, plus the counts behind it."""
        figure = figure_accuracy_bins(data['bin_panels'], data['accuracy'],
                                      figsize=BINS_FIGSIZE)
        path = os.path.join(tmpdir, 'accuracy_bins.png')
        figure.tight_layout()
        figure.savefig(path, dpi=200, facecolor=SURFACE)
        figure.clf()

        # The counts, because a bar at 0.08 on a 24-question bin is 2/24 and the reader
        # should be able to see that without measuring the whisker.
        head = ['measure'] + [f'bin {i + 1}' for i in range(BIN_COUNT)] + ['spread']
        rows = [[Paragraph(h, CELL_HEAD) for h in head]]
        for panel in data['bin_panels']:
            accuracies = [b['accuracy'] for b in panel['bins']]
            rows.append(
                [Paragraph(panel['label'], CELL)]
                + [Paragraph(f"{b['successes']}/{b['total']} &nbsp;<b>{b['accuracy']:.2f}</b>",
                             CELL) for b in panel['bins']]
                + [Paragraph(f'{max(accuracies) - min(accuracies):.2f}', CELL)])
        widths = [0.26] + [0.126] * BIN_COUNT + [0.11]
        table = Table(rows, colWidths=[w * CONTENT_WIDTH for w in widths], repeatRows=1)
        table.setStyle(TableStyle(self._base_table_style(len(rows))))

        return [
            PageBreak(),
            Paragraph('Accuracy by Confidence Bin', TITLE),
            Paragraph(
                f"Questions split into {BIN_COUNT} equal bins by each measure's score, "
                'most confident first. Bars are observed accuracy, whiskers are Wilson '
                '95% binomial intervals, the grey line is the base rate. '
                '<b>This is not a calibration chart</b> &mdash; see the note below.',
                SUBTITLE),
            Spacer(1, 0.4 * cm),
            Image(path, width=CONTENT_WIDTH, height=BINS_FIGSIZE[1] * 72),
            Spacer(1, 0.42 * cm),
            table,
            Spacer(1, 0.34 * cm),
            Paragraph(
                '<b>How to read it.</b> The bins are <b>disjoint</b>, unlike the '
                'risk-coverage curve on page one, which is cumulative. That is the point '
                'of having both: a cumulative curve cannot reveal a measure that '
                'separates its most-confident bin brilliantly and then ranks everything '
                'below it at chance, because the strong leading prefix stays inside every '
                'later average. Here that shows up as a <b>shape</b> &mdash; a steady '
                'decline means the measure grades across the whole range, a cliff '
                'followed by a flat tail means all of its signal is in the top bin. '
                'AUROC cannot distinguish the two, and can rank the flat-tailed measure '
                'higher. <b>What this is not:</b> a reliability diagram plots a claimed '
                'probability against the observed rate, and these measures are entropies '
                'in nats with no probability scale, so there is nothing here to be '
                'calibrated against. No ECE or Brier score is computed anywhere in this '
                f"pipeline. <b>Read the whiskers:</b> at {data['n_questions'] // BIN_COUNT} "
                'questions a bin the intervals are wide enough to overlap between '
                'adjacent bins, so the trend across a whole row is evidence but any '
                'single pair of bars is not.',
                FOOT),
        ]

    def _glossary(self, data):
        """Page two: what each name on page one means.

        Two columns, split where the RENDERED heights balance rather than where the entry
        counts do. Counting entries leaves one column 40% short here, because the AUROC
        and correctness definitions run to four or five lines while others run to one.
        Heights come from reportlab's own wrap(), so the balance is measured, not guessed.
        """
        gutter = 0.8 * cm
        width = (CONTENT_WIDTH - gutter) / 2

        blocks = []
        for group, items in GLOSSARY:
            blocks.append({'group': group, 'heading': True,
                           'flowables': [Paragraph(group.upper(), GLOSS_GROUP),
                                         Spacer(1, 0.1 * cm)]})
            for term, body in items:
                blocks.append({'group': group, 'heading': False,
                               'flowables': [Paragraph(term, GLOSS_TERM),
                                             Paragraph(body, GLOSS_BODY),
                                             Spacer(1, 0.16 * cm)]})
        for block in blocks:
            block['height'] = sum(f.wrap(width, 10 ** 6)[1] for f in block['flowables'])

        # The split that most evenly divides total height. A heading is never left as the
        # last thing in a column -- its entries would start in the next one.
        total = sum(b['height'] for b in blocks)
        running, best, best_gap = 0.0, 1, total
        for index in range(1, len(blocks)):
            running += blocks[index - 1]['height']
            if blocks[index - 1]['heading']:
                continue
            gap = abs(total - 2 * running)
            if gap < best_gap:
                best, best_gap = index, gap

        def render(items, carry=None):
            flowables = []
            if carry is not None:
                flowables += [Paragraph(f'{carry.upper()} (CONT.)', GLOSS_GROUP),
                              Spacer(1, 0.1 * cm)]
            for block in items:
                # A heading that opens a column needs no extra space above it.
                if block['heading'] and flowables:
                    flowables.append(Spacer(1, 0.25 * cm))
                flowables += block['flowables']
            return flowables

        left, right = blocks[:best], blocks[best:]
        # If the break lands inside a group, the second column opens on a bare term with
        # no idea which section it belongs to. Repeat the heading rather than move the
        # break, which would undo the height balancing.
        carry = right[0]['group'] if right and not right[0]['heading'] else None

        columns = Table([[render(left), render(right, carry)]],
                        colWidths=[width + gutter, width])
        columns.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (0, -1), gutter),
            ('RIGHTPADDING', (1, 0), (1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))

        return [
            PageBreak(),
            Paragraph('Glossary', TITLE),
            Paragraph(
                f"Definitions as implemented in this pipeline, for "
                f"<b>{data['run_name']}</b>. Where a measure here differs from its "
                'textbook form &mdash; length normalisation, which model scores the '
                'prior, the correctness threshold &mdash; the code is what is described.',
                SUBTITLE),
            Spacer(1, 0.42 * cm),
            columns,
        ]

    # ------------------------------------------------------------------ assembly

    def build(self, run_name, path):
        data = self.collect(run_name)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

        document = SimpleDocTemplate(
            path, pagesize=PAGE,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=0.9 * cm, bottomMargin=0.7 * cm,
            title=f'Semantic uncertainty report - {run_name}',
            author='code/report_pdf.py', subject=self.args.generation_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            story = []
            story += self._header(data)
            story += self._kpis(data)
            story += self._tables_row(data)
            story += self._charts_row(data, tmpdir)
            story += self._footer(data)
            if not self.args.no_bins:
                story += self._bins_page(data, tmpdir)
            if not self.args.no_glossary:
                story += self._glossary(data)
            document.build(story, onFirstPage=_paint_surface, onLaterPages=_paint_surface)

        return data


def _paint_surface(canvas, document):
    """The charts carry SURFACE as their own background; the page must match."""
    canvas.saveState()
    canvas.setFillColor(SURFACE_C)
    canvas.rect(0, 0, PAGE[0], PAGE[1], stroke=0, fill=1)
    canvas.restoreState()


def upload(path, run_name):
    """Files tab and a versioned artifact -- they are used at different times.

    Neither is allowed to fail the stage: the PDF is already on disk by this point, and a
    reporting stage that kills a pipeline over an upload hiccup is worse than one that
    prints a warning.
    """
    uploaded = []
    try:
        wandb.save(path, base_path=str(pathlib.Path(path).parent.parent))
        uploaded.append('files')
    except Exception as error:
        print(f'wandb.save failed: {type(error).__name__}: {error}')
    try:
        artifact = wandb.Artifact(f'{run_name}-report', type='report')
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        uploaded.append('artifact')
    except Exception as error:
        print(f'artifact upload failed: {type(error).__name__}: {error}')
    return uploaded


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--generation_model', type=str, default='opt-350m',
                        choices=OPT_MODELS)
    parser.add_argument('--evaluation_model', type=str, default=None,
                        help='Accepted for CLI symmetry with the other stages; the '
                             'confidence pickle is named by --generation_model alone.')
    parser.add_argument('--seed', type=int, default=10,
                        help='Seeds the bootstrap resampler. Match log_wandb_charts.py '
                             'and the PDF reproduces the W&B panels exactly.')
    parser.add_argument('--bootstrap_resamples', type=int, default=4000)
    parser.add_argument('--dry_run', action='store_true',
                        help='Write the PDF locally and skip W&B entirely. --run_id is '
                             'then used directly as the run name.')
    parser.add_argument('--no_bins', action='store_true',
                        help='Drop the accuracy-by-confidence-bin page.')
    parser.add_argument('--no_glossary', action='store_true',
                        help='Drop page two and emit the one-page dashboard alone.')
    parser.add_argument('--output', type=str, default=None,
                        help='Override the output path. Default is '
                             'output/reports/<run_name>/<model>_report.pdf')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        run_name = args.run_id
    else:
        wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
        run_name = wandb.run.name or args.run_id
    print(f'run_name={run_name}')

    path = args.output or (f'{config.output_dir}/reports/{run_name}/'
                           f'{args.generation_model}_report.pdf')
    data = ReportBuilder(args).build(run_name, path)

    size_kb = pathlib.Path(path).stat().st_size / 1024
    print(f'wrote {path}  ({size_kb:.0f} KB)')
    print(f"  accuracy={data['accuracy']:.4f}  "
          f"correct={data['n_correct']}/{data['n_questions']}")
    for result in data['auroc_rows']:
        print(f"  {result['label']:<32} {result['auroc']:.4f}  "
              f"[{fmt(result['ci_lo'])}, {fmt(result['ci_hi'])}]")

    if not args.dry_run:
        destinations = upload(path, run_name)
        print(f"uploaded to: {', '.join(destinations) if destinations else 'nothing'}")
        wandb.finish()


if __name__ == '__main__':
    main()
