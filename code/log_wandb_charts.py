"""Log this run's result charts to W&B.

NEW STAGE - NO UPSTREAM EQUIVALENT. Runs last, after analyze_results.py, and is
READ-ONLY with respect to the science: it opens the same pickles the analysis reads and
writes nothing back. It does not touch overall_results.json, it does not re-score
anything, and skipping it cannot change a result. Re-render any time with

    python code/run_pipeline.py --wandb_run_name <name> --model <model> --only charts

EACH REQUESTED CHART IS LOGGED TWICE - ONCE NATIVE, ONCE AS AN IMAGE
  charts/*   wandb custom-chart panels (Vega specs). Interactive: hover, zoom, and the
             backing table is sortable in the UI.
  images/*   the same chart drawn with matplotlib and logged as wandb.Image. An image
             panel has no Vega spec to resolve and no field mapping to get wrong, so what
             image_*() draws is exactly what appears in the run. This is the copy to trust
             when a charts/* panel renders as something other than what it claims to be -
             in this project they repeatedly resolved to bar charts.

             The pairing is not pure redundancy: images/auroc_bounds is a REAL GROUPED BAR
             CHART, which the native panel cannot be (see below).

TWO CHARTS, BY REQUEST
  charts/risk_coverage   'If the model may abstain, what does that buy?' Questions are
                         ranked by uncertainty, most confident first; each point asks how
                         accurate the model would be if it answered only that fraction and
                         abstained on the rest. Every point is a system you could ship.
                         Bracketed by an oracle ceiling (rank by correctness itself) and a
                         base-rate floor (rank at random), so the area between a curve and
                         the floor is exactly what the uncertainty estimate bought.

  charts/auroc_bounds    AUROC with its bootstrap 95% interval, as three coloured series
                         across the measures: lower bound, point estimate, upper bound.
                         The vertical gap between the bounds IS the interval; where two
                         measures' gaps overlap they are not separable, whatever their
                         point estimates say.

                         The NATIVE panel cannot be a grouped bar: wandb/bar/v0 exposes
                         only `label` and `value` - no series or colour channel - and this
                         version of wandb ships no grouped-bar spec, so it is three
                         coloured lines over a rank axis instead. images/auroc_bounds IS
                         the grouped bar, three bars per measure, with real measure names
                         on the x-axis.

                         Neither is STACKED, which was never on the table: a stacked bar's
                         height is the SUM of its segments, so lower+auroc+upper would
                         reach ~2.15. The image form bases its bars at 0.5 rather than 0,
                         so a bar's length is lift over chance - AUROC's real origin -
                         and a bound that fails to clear chance draws downwards.

SERIES ARE SEPARATED BY COLOUR, NEVER BY LINE STYLE
Native panels go through coloured_line() - see its docstring for why the choice of wandb
helper, not any styling argument, is what decides this. The images draw from
SERIES_COLOURS and never set a dash pattern; the oracle and base-rate references are grey
rather than dashed for the same reason.

X-AXIS OF charts/auroc_bounds IS A RANK
wandb/line/v0 plots a numeric x, so measures sit at ranks rather than named ticks.
tables/auroc_with_ci is sorted identically, so rank N is row N there - which is why that
one table is logged alongside the two panels rather than left out.

RE-RUNNING APPENDS, IT DOES NOT REPLACE
W&B history and media are append-only and there is no unlog operation, so every
invocation of this stage adds another copy of each panel to the run. Renaming or removing
a panel here does not remove the old one from a run that already has it - that needs the
panel deleted in the UI (view-level, cosmetic) or the run deleted (data-level). Deleting
a panel from a workspace view does NOT delete the logged data, and re-running this stage
does not restore a panel you deleted from the view - reset the workspace view for that.

AUROC CONVENTION
AUROC uses INCORRECTNESS as the positive label - an uncertainty measure should score high
when the model is wrong - matching analyze_results.py's `1 - correct`. Higher is better,
0.5 is chance. Questions whose measure is non-finite are dropped from that measure and
counted in n_masked_non_finite rather than imputed.
"""

import argparse
import pickle
import uuid

import textwrap

import matplotlib
matplotlib.use('Agg')   # a pipeline stage has no display; must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import wandb

import config

OPT_MODELS = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

# Every measure analyze_results.py ranks on, for charts/auroc_bounds. (key, label)
MEASURES = [
    ('predictive_entropy_over_concepts', 'semantic entropy'),
    ('average_predictive_entropy', 'ln predictive entropy'),
    ('predictive_entropy', 'predictive entropy'),
    ('unnormalised_entropy_over_concepts', 'unnormalised entropy/concepts'),
    ('number_of_semantic_sets', 'number of semantic sets'),
    ('neg_log_likelihood_of_most_likely_gen', 'neg llh most likely gen'),
    ('average_neg_log_likelihood_of_most_likely_gen', 'avg neg llh most likely gen'),
]

# charts/risk_coverage gets a SHORT list on purpose: six overlaid curves is the ceiling
# for legibility, and two of the six are the oracle and base-rate references.
PRIMARY_MEASURES = [
    'predictive_entropy_over_concepts',
    'average_predictive_entropy',
    'predictive_entropy',
    'neg_log_likelihood_of_most_likely_gen',
]

# A FIXED coverage grid rather than one point per question, so runs with different
# question counts overlay on the same axis. 2% steps also set the resolution of
# riskcov_coverage_at_*: a target reachable at 25% is reported as 24%.
COVERAGE_GRID = np.round(np.arange(0.02, 1.0001, 0.02), 4)
REPORTED_COVERAGES = [0.10, 0.20, 0.50]
TARGET_ACCURACY = 0.5
BOOTSTRAP_RESAMPLES = 4000

# Bins for the accuracy-by-confidence-bin panel. Five is the most this run's 120 questions
# support: at 24 questions a bin the 95% interval is already ~0.3 wide, and ten bins would
# put ~3 correct answers in each, where the ordering is noise.
BIN_COUNT = 5

LABEL_OF = dict(MEASURES)

# ------------------------------------------------------------------------- palette
# Validated with the dataviz palette checker against a light surface, ALL pairs rather
# than just adjacent ones: lightness band, chroma floor, CVD separation (worst pair
# dE 9.1 deutan), normal-vision floor and contrast-vs-surface all pass. Re-run it before
# substituting any colour here - 'these look distinct to me' is the judgement the checker
# exists to replace, and deuteranopia is where hand-picked palettes collapse.
SERIES_COLOURS = ['#5B55C9', '#00967A', '#C05F26', '#A8358F']

# The oracle and the base rate are REFERENCES, not results, so they get neutral greys.
# Still their own colours - never a dashed or dotted variant of a series colour, which is
# the line-style encoding this file exists to avoid - but recessive enough that the four
# measures read as the subject of the chart.
ORACLE_COLOUR = '#4A4A52'
FLOOR_COLOUR = '#8C8C96'

# An ORDINAL ramp, not three categories: lower bound < point estimate < upper bound, so a
# single hue light-to-dark carries the ordering rather than three unrelated hues implying
# three unrelated things. The lightest step sits under 3:1 against the surface, which is
# why every bar in that panel carries a visible value label.
BOUND_COLOURS = {'ci_lo': '#ADA9E4', 'auroc': '#5B55C9', 'ci_hi': '#332C8F'}

INK = '#26262B'
MUTED = '#6E6E78'
GRID_COLOUR = '#DEDEE4'
SURFACE = '#FCFCFB'


def to_numpy(value):
    """Tensors in these pickles may sit on mps/cuda, so never call .numpy() directly."""
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=float)


def wilson_interval(successes, total, z=1.96):
    """Binomial interval for a bin's accuracy.

    Wilson rather than the textbook normal approximation because the bins at the confident
    end of a 27%-accuracy run sit near 0 -- where the normal interval runs below zero and
    collapses to zero width at exactly 0/n, claiming certainty from no evidence.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def confidence_ranking(score):
    """Question indices ordered most-confident first.

    Non-finite scores sort last: a measure that could not be computed is the least
    trustworthy thing to answer on, so it belongs at the abstain end rather than wherever
    NaN happens to land. kind='stable' keeps ties in question order, so the curve is
    reproducible across runs.
    """
    return np.argsort(np.where(np.isfinite(score), score, np.inf), kind='stable')


def coloured_line(rows, x_name, y_name, series_name, title):
    """A multi-series line chart whose series are separated by COLOUR.

    This helper exists because the distinction is in the choice of wandb helper, not in
    any styling argument, and is easy to undo by accident:

      wandb.plot.line_series -> spec `wandb/lineseries/v0`, which carries series identity
                                in its own lineKey channel and renders them as varying
                                LINE STYLES.
      wandb.plot.line        -> spec `wandb/line/v0`, which exposes an explicit `stroke`
                                field - Vega-Lite's COLOUR channel - so one colour per
                                series.

    Long-format rows (one row per point, carrying its series name) are what that spec
    wants, and they double as the chart's own sortable table in the UI.
    """
    table = wandb.Table(columns=[series_name, x_name, y_name], data=rows)
    return wandb.plot.line(table, x_name, y_name, stroke=series_name, title=title)


def _style_axes(axes, scale=1.0):
    """Recessive grid and axes, so the data is the only assertive thing on the figure."""
    axes.set_facecolor(SURFACE)
    axes.set_axisbelow(True)
    axes.grid(True, axis='y', color=GRID_COLOUR, linewidth=0.8)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(GRID_COLOUR)
    axes.tick_params(colors=MUTED, labelsize=9 * scale, length=0)


def _wrapped_title(axes, title, width, scale=1.0):
    """Titles here double as the caption, so they are long enough to need wrapping.

    An unwrapped one is silently CLIPPED at the figure edge rather than shrunk, which is
    the kind of defect that only shows up once you look at the rendered PNG.
    """
    axes.set_title('\n'.join(textwrap.wrap(title, width)),
                   color=INK, fontsize=12 * scale, loc='left', pad=14 * scale)


def _finish(figure):
    """Hand the figure to W&B as a PNG and close it, so a long run cannot leak figures."""
    figure.tight_layout()
    image = wandb.Image(figure)
    plt.close(figure)
    return image


def figure_risk_coverage(curves, oracle, base_rate, title, figsize=(10.5, 5.8),
                         title_width=88, scale=1.0, show_title=True):
    """The risk-coverage curves as a matplotlib Figure.

    Split from image_risk_coverage so a second consumer -- report_pdf.py -- can embed the
    same figure in a PDF. _finish() closes the figure it is handed, so a function that
    ends in _finish has nothing left to give anyone else.

    figsize is a parameter because matplotlib font sizes are absolute POINTS relative to
    the figure's inches. Rendering the 10.5in default and then scaling it into a 5in PDF
    slot would shrink 9pt labels to ~4pt; the PDF instead asks for a figure the size of
    the slot it has, so the text lands at the size it says it is.

    `scale` exists because that alone is not enough: point sizes are absolute, so halving
    the canvas without touching them doubles how much of it the type covers, and the
    title, legend and axis labels start colliding. scale multiplies every font size (and
    the paddings measured in points), which is what keeps a half-width render laid out
    like the full-width one. `show_title` drops the in-figure title for callers whose
    surrounding document already has a heading for the chart.

    Why an image at all, when the wandb/line/v0 panel is logged too: an image panel is
    the one kind W&B cannot mis-render. There is no Vega spec for it to resolve and no
    field mapping to get wrong, so what is drawn here is exactly what appears in the run
    - which the native custom-chart panels, in this project, repeatedly were not.
    """
    figure, axes = plt.subplots(figsize=figsize, dpi=160)
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes, scale)

    # The band between floor and ceiling is the entire space an uncertainty estimate has
    # to play in. Filling it makes 'how much did this buy?' a distance you can see rather
    # than a sentence in the caption.
    axes.fill_between(COVERAGE_GRID, base_rate, oracle, color=ORACLE_COLOUR, alpha=0.05,
                      linewidth=0)
    axes.plot(COVERAGE_GRID, oracle, color=ORACLE_COLOUR, linewidth=2,
              label='oracle (ceiling)')
    axes.axhline(base_rate, color=FLOOR_COLOUR, linewidth=2,
                 label=f'base rate {base_rate:.3f} (floor)')

    for index, (label, accuracies) in enumerate(curves):
        axes.plot(COVERAGE_GRID, accuracies, linewidth=2, label=label,
                  color=SERIES_COLOURS[index % len(SERIES_COLOURS)])

    axes.set_xlim(0, 1)
    # Anchored just under the floor rather than at 0. Nothing can plot below the base rate
    # (answering the k most-confident questions cannot do worse than answering all of
    # them, in expectation), so a 0-anchored axis spends a quarter of its height on
    # territory no curve enters. Truncation is legitimate for lines - they encode position,
    # not length - and the floor line makes the datum explicit.
    axes.set_ylim(max(0.0, base_rate - 0.06), 1.02)
    axes.xaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    axes.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
    axes.set_xlabel('coverage - fraction of questions answered', color=MUTED,
                    fontsize=10 * scale)
    axes.set_ylabel('accuracy on the answered fraction', color=MUTED, fontsize=10 * scale)
    if show_title:
        _wrapped_title(axes, title, title_width, scale)

    # Six series is past the point where direct labels stay legible - the curves converge
    # on the right and cross on the left - so identity lives in the legend. Handles are
    # passed explicitly to put the measures before the two references: draw order has the
    # references first, so that they sit behind the curves, and the default legend follows
    # draw order.
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[2:] + handles[:2], labels[2:] + labels[:2],
                loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False,
                fontsize=9 * scale, labelcolor=MUTED,
                handlelength=1.6, columnspacing=1.1, handletextpad=0.5)
    return figure


def image_risk_coverage(curves, oracle, base_rate, title):
    """The risk-coverage figure as a wandb.Image."""
    return _finish(figure_risk_coverage(curves, oracle, base_rate, title))


def figure_auroc_bounds(results, title, figsize=(11, 6.2), title_width=92, scale=1.0,
                        show_title=True, labels=None):
    """AUROC with its bootstrap interval as a real grouped bar chart, as a Figure.

    Split from image_auroc_bounds for the same reason as figure_risk_coverage above.

    This is the chart originally asked for and the reason the image route is worth having:
    wandb/bar/v0 exposes only `label` and `value`, with no series or colour channel, and
    this version of wandb ships no grouped-bar spec, so the native panel had to fall back
    to three coloured lines over a rank axis.

    Bars are based at 0.5, so a bar's LENGTH is lift over chance. That is the honest form
    here: 0.5 is AUROC's real origin, a 0-anchored axis would spend 70% of its height on
    territory no measure can enter, and a truncated 0-anchored axis would misstate every
    ratio. A bound that fails to clear chance draws downwards, which is the thing you most
    need to see.
    """
    labels = labels or [result['label'] for result in results]
    positions = np.arange(len(results), dtype=float)
    width = 0.27

    figure, axes = plt.subplots(figsize=figsize, dpi=160)
    figure.patch.set_facecolor(SURFACE)
    _style_axes(axes, scale)

    for offset, key, name in ((-width, 'ci_lo', 'lower bound (2.5%)'),
                              (0.0, 'auroc', 'AUROC'),
                              (width, 'ci_hi', 'upper bound (97.5%)')):
        values = [result.get(key) for result in results]
        # width * 0.92 leaves a surface gap between neighbouring bars, so a group reads as
        # three marks rather than one striped block.
        bars = axes.bar(positions + offset,
                        [0.0 if v is None else v - 0.5 for v in values],
                        width * 0.92, bottom=0.5, linewidth=0,
                        color=BOUND_COLOURS[key], label=name)
        # SELECTIVE labels: only the point estimate. Numbering all three bars puts 21
        # figures on a 7-measure chart, which is the 'a number on every mark' failure -
        # and the bounds are already readable off the axis to the precision anyone reads
        # them at.
        if key != 'auroc':
            continue
        for bar, value in zip(bars, values):
            # Right-anchored, not centred: the label is wider than the bar, and the bar
            # to its right is the UPPER bound, which is always taller - so a centred
            # label collides with it. Extending leftwards instead puts the overhang above
            # the lower-bound bar, which is by construction shorter than this one.
            axes.annotate(f'{value:.3f}',
                          (bar.get_x() + bar.get_width(), value),
                          textcoords='offset points',
                          xytext=(1, 3 if value >= 0.5 else -11 * scale),
                          ha='right', fontsize=8.5 * scale, color=MUTED)

    axes.axhline(0.5, color=INK, linewidth=1)
    # Left-anchored: the bars are sorted descending, so the right-hand end of the chance
    # line is exactly where the shortest bars are and an annotation there overlaps them.
    axes.annotate('0.5 = chance', (-0.55, 0.5), textcoords='offset points',
                  xytext=(0, -14 * scale), ha='left', fontsize=9 * scale, color=MUTED)

    finite = [v for result in results for v in (result['auroc'], result.get('ci_lo'),
                                                result.get('ci_hi')) if v is not None]
    axes.set_ylim(min(0.47, min(finite) - 0.03), max(finite) + 0.09)
    axes.set_xlim(-0.6, len(results) - 0.4)
    axes.set_xticks(positions)
    # rotation_mode='anchor' rotates about the label's anchor point; without it a rotated
    # right-aligned label drifts left of the group it belongs to.
    axes.set_xticklabels(labels, rotation=20, ha='right', rotation_mode='anchor',
                         color=MUTED, fontsize=9 * scale)
    axes.set_ylabel('AUROC (bars measured from chance)', color=MUTED, fontsize=10 * scale)
    if show_title:
        _wrapped_title(axes, title, title_width, scale)
    axes.legend(loc='upper right', frameon=False, fontsize=9 * scale, labelcolor=MUTED,
                ncol=3, handlelength=1.4, columnspacing=1.0, handletextpad=0.4)
    return figure


def figure_accuracy_bins(panels, base_rate, figsize=(10.75, 3.6), scale=1.0,
                         title_width=30):
    """Observed accuracy within each confidence bin, as small multiples.

    NOT a calibration chart, and must never be labelled one. A reliability diagram plots a
    CLAIMED probability against the observed rate; these measures are entropies in nats
    with no probability scale, so there is nothing to be calibrated against. What this
    shows is where on the score range a measure's separation actually lives -- which AUROC
    compresses to a scalar, and which the risk-coverage curve hides because it is
    cumulative, so one strong leading bin props the running average up across the whole
    axis.

    Small multiples on a SHARED y-axis rather than four overlaid series: the comparison
    the panel exists for is between the four SHAPES, and twenty bars with twenty error
    bars on one axis is not a chart anyone reads shapes off.
    """
    # No gridspec_kw here: an explicit wspace is what tight_layout() calls incompatible,
    # and the caller tight_layouts this figure to fit its PDF slot exactly.
    figure, axes_row = plt.subplots(1, len(panels), figsize=figsize, dpi=160, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    if len(panels) == 1:
        axes_row = [axes_row]

    ceiling = max([b['ci_hi'] for panel in panels for b in panel['bins']] + [base_rate])

    for index, (axes, panel) in enumerate(zip(axes_row, panels)):
        _style_axes(axes, scale)
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        positions = np.arange(len(panel['bins']), dtype=float)
        accuracies = [b['accuracy'] for b in panel['bins']]

        axes.bar(positions, accuracies, 0.68, color=colour, linewidth=0)
        # Whiskers, not a shaded band: each bin is an independent binomial, and a band
        # would imply a continuous function through them.
        axes.errorbar(positions, accuracies,
                      yerr=[[a - b['ci_lo'] for a, b in zip(accuracies, panel['bins'])],
                            [b['ci_hi'] - a for a, b in zip(accuracies, panel['bins'])]],
                      fmt='none', ecolor=INK, elinewidth=1.1 * scale,
                      capsize=2.5 * scale, capthick=1.1 * scale, alpha=0.55)
        axes.axhline(base_rate, color=FLOOR_COLOUR, linewidth=1.4, zorder=0)

        for position, bin_ in zip(positions, panel['bins']):
            axes.annotate(f"{bin_['successes']}/{bin_['total']}",
                          (position, 0), textcoords='offset points',
                          xytext=(0, 3 * scale), ha='center', va='bottom',
                          fontsize=7 * scale, color=SURFACE, zorder=5)

        axes.set_xticks(positions)
        axes.set_xticklabels([str(i + 1) for i in range(len(positions))],
                             color=MUTED, fontsize=9 * scale)
        axes.set_xlabel('confidence bin', color=MUTED, fontsize=9 * scale)
        axes.set_xlim(-0.65, len(positions) - 0.35)
        axes.set_ylim(0, ceiling * 1.08)
        _wrapped_title(axes, panel['label'], title_width, scale)
        if index == 0:
            axes.set_ylabel('observed accuracy', color=MUTED, fontsize=10 * scale)
            axes.yaxis.set_major_formatter(lambda value, _: f'{value:.0%}')
            # Anchored right, over the LAST bin: these scores are sorted most-confident
            # first, so the tail bars are the short ones and the airspace above the base
            # rate is free there. At the left edge the label lands on the tallest bar.
            axes.annotate(f'base rate {base_rate:.0%}',
                          (len(positions) - 0.4, base_rate),
                          textcoords='offset points', xytext=(0, 4 * scale),
                          ha='right', fontsize=7.5 * scale, color=MUTED)

    return figure


def image_auroc_bounds(results, title):
    """The AUROC-bounds figure as a wandb.Image."""
    return _finish(figure_auroc_bounds(results, title))


class ChartLogger:
    def __init__(self, args):
        self.args = args

    # ---------------------------------------------------------------- loading

    def _load_frame(self, run_name):
        """One row per question: correctness plus every uncertainty measure."""
        model = self.args.generation_model
        with open(f'{config.output_dir}/sequences/{run_name}/'
                  f'{model}_generations.pkl', 'rb') as infile:
            generations = pickle.load(infile)
        with open(f'{config.output_dir}/confidence/{run_name}/'
                  f'aggregated_likelihoods_{model}_generations.pkl', 'rb') as infile:
            confidence = pickle.load(infile)

        # Row order in the confidence pickle is NOT the generations order:
        # get_overall_log_likelihoods builds 'ids' by appending as it walks its own input,
        # so the two files agree on content but not order. Getting this wrong silently
        # pairs each question with another question's uncertainty.
        ids = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
               for i in confidence['ids']]
        position = {question_id: index for index, question_id in enumerate(ids)}
        order = np.array([position[g['id'][0]] for g in generations])

        frame = {
            # analyze_results.py's definition, reproduced rather than re-invented so these
            # panels rank against exactly the labels the reported AUROCs used.
            'correct': (np.array([float(g['rougeL_to_target']) for g in generations])
                        > 0.3).astype(int),
        }
        for key, _ in MEASURES:
            if key in confidence:
                frame[key] = to_numpy(confidence[key]).reshape(-1)[order]

        return frame

    # ---------------------------------------------------------------- statistics

    def _auroc_with_ci(self, incorrect, score):
        """AUROC over the finite rows, with a non-parametric bootstrap over questions.

        The interval is the point of the second panel: a bare AUROC on a handful of
        positives is not a result. With 3 correct answers of 40, every achievable value is
        a multiple of 1/111 and one question flips it by 0.009 - which is how an 0.766 in
        this project became 0.565 once the data tripled.
        """
        finite = np.isfinite(score)
        if finite.sum() < 2 or len(set(incorrect[finite])) < 2:
            return None
        y, s = incorrect[finite], score[finite]

        generator = np.random.default_rng(self.args.seed)
        estimates = []
        for _ in range(self.args.bootstrap_resamples):
            draw = generator.integers(0, len(y), len(y))
            if len(set(y[draw])) < 2:
                continue
            estimates.append(sklearn.metrics.roc_auc_score(y[draw], s[draw]))

        low, high = np.percentile(estimates, [2.5, 97.5]) if estimates else (None, None)
        return {
            'auroc': float(sklearn.metrics.roc_auc_score(y, s)),
            'ci_lo': None if low is None else float(low),
            'ci_hi': None if high is None else float(high),
            'n_used': int(finite.sum()),
            'n_masked': int((~finite).sum()),
        }

    @staticmethod
    def _accuracy_by_coverage(correct, score=None):
        """Accuracy among the most-confident fraction, on COVERAGE_GRID.

        score=None gives the oracle: rank by correctness itself, the ceiling no
        uncertainty measure can beat. It holds 1.0 out to coverage = base rate, then
        declines as min(1, n_correct / k).
        """
        ranking = (np.argsort(-correct, kind='stable') if score is None
                   else confidence_ranking(score))
        cumulative = np.cumsum(correct[ranking])
        accuracies = []
        for coverage in COVERAGE_GRID:
            k = max(1, int(round(coverage * len(correct))))
            accuracies.append(float(cumulative[k - 1] / k))
        return accuracies

    @staticmethod
    def _accuracy_by_bin(correct, score, bin_count=BIN_COUNT):
        """Split the questions into equal-size confidence bins, most confident first.

        DISJOINT bins, unlike _accuracy_by_coverage's cumulative prefixes. That is the
        whole difference: a cumulative curve cannot show a measure that separates its top
        bin brilliantly and then ranks the remainder at chance, because the good prefix
        stays in every later average.
        """
        ranking = confidence_ranking(score)
        bins = []
        for index, group in enumerate(np.array_split(ranking, bin_count)):
            successes, total = int(correct[group].sum()), int(len(group))
            low, high = wilson_interval(successes, total)
            bins.append({'bin': index + 1, 'successes': successes, 'total': total,
                         'accuracy': successes / total if total else 0.0,
                         'ci_lo': low, 'ci_hi': high})
        return bins

    # ---------------------------------------------------------------- panels

    def _panel_risk_coverage(self, frame, logged):
        correct = frame['correct']
        base_rate = float(correct.mean())
        rows, curves = [], []

        for key in PRIMARY_MEASURES:
            if key not in frame:
                continue
            accuracies = self._accuracy_by_coverage(correct, frame[key])
            curves.append((LABEL_OF[key], accuracies))
            for coverage, accuracy in zip(COVERAGE_GRID, accuracies):
                rows.append([LABEL_OF[key], float(coverage), accuracy])

            # Area under accuracy-coverage. The literature usually plots RISK and reports
            # an area where LOWER is better; this is the accuracy convention, so higher is
            # better and base_rate is the floor to beat. Named to say which.
            logged[f'riskcov_area_accuracy/{key}'] = round(
                float(np.trapezoid(accuracies, COVERAGE_GRID)), 4)

            for coverage in REPORTED_COVERAGES:
                index = int(np.argmin(np.abs(COVERAGE_GRID - coverage)))
                logged[f'riskcov_acc_at_{int(coverage * 100)}pct/{key}'] = round(
                    accuracies[index], 4)

            # Usually the most quotable single number: how much you can answer and still
            # clear a target accuracy. 0 means the target is never reached at any coverage.
            reached = [c for c, a in zip(COVERAGE_GRID, accuracies) if a >= TARGET_ACCURACY]
            logged[f'riskcov_coverage_at_{int(TARGET_ACCURACY * 100)}pct_acc/{key}'] = round(
                float(max(reached)) if reached else 0.0, 4)

        oracle = self._accuracy_by_coverage(correct, None)
        for coverage, accuracy in zip(COVERAGE_GRID, oracle):
            rows.append(['oracle (ceiling)', float(coverage), accuracy])
        # Two points are enough for a flat reference, and keeping it to two stops the
        # floor competing visually with the real curves.
        for coverage in (float(COVERAGE_GRID[0]), 1.0):
            rows.append([f'base rate {base_rate:.3f} (floor)', coverage, base_rate])

        logged['riskcov_area_accuracy/oracle'] = round(
            float(np.trapezoid(oracle, COVERAGE_GRID)), 4)
        logged['riskcov_base_rate'] = round(base_rate, 4)

        title = 'Risk-coverage: accuracy when answering only the most confident fraction'
        return {
            'charts/risk_coverage': coloured_line(
                rows, 'coverage', 'accuracy', 'measure', title),
            'images/risk_coverage': image_risk_coverage(curves, oracle, base_rate, title),
        }

    def _panel_auroc_bounds(self, frame, logged):
        incorrect = 1 - frame['correct']
        results, masked = [], 0

        for key, label in MEASURES:
            if key not in frame:
                continue
            result = self._auroc_with_ci(incorrect, frame[key])
            if result is None:
                continue
            masked += result['n_masked']
            results.append({'label': label, 'key': key, **result})

            logged[f'auroc/{key}'] = round(result['auroc'], 4)
            if result['ci_lo'] is not None:
                logged[f'auroc_ci_lo/{key}'] = round(result['ci_lo'], 4)
                logged[f'auroc_ci_hi/{key}'] = round(result['ci_hi'], 4)

        logged['n_masked_non_finite'] = masked
        if not results:
            return {}

        results.sort(key=lambda r: -r['auroc'])
        rows = []
        for rank, result in enumerate(results, start=1):
            rows.append(['AUROC', rank, round(result['auroc'], 4)])
            if result['ci_lo'] is not None:
                rows.append(['lower bound (2.5%)', rank, round(result['ci_lo'], 4)])
                rows.append(['upper bound (97.5%)', rank, round(result['ci_hi'], 4)])
            rows.append(['chance', rank, 0.5])

        table = wandb.Table(
            columns=['measure_rank', 'measure', 'auroc', 'ci_lo', 'ci_hi', 'ci_width',
                     'ci_excludes_chance', 'n_questions_used'],
            data=[[rank, r['label'], round(r['auroc'], 4),
                   None if r['ci_lo'] is None else round(r['ci_lo'], 4),
                   None if r['ci_hi'] is None else round(r['ci_hi'], 4),
                   None if r['ci_lo'] is None else round(r['ci_hi'] - r['ci_lo'], 4),
                   bool(r['ci_lo'] is not None and r['ci_lo'] > 0.5), r['n_used']]
                  for rank, r in enumerate(results, start=1)])

        title = ('AUROC with 95% bootstrap interval - where two measures\' intervals '
                 'overlap they are not separable, whatever their point estimates say')
        return {
            'charts/auroc_bounds': coloured_line(
                rows, 'measure_rank', 'auroc', 'bound', title),
            'images/auroc_bounds': image_auroc_bounds(results, title),
            # Logged because the chart's x-axis is a rank: this table is sorted
            # identically, so it is how a reader maps rank N to a measure name.
            'tables/auroc_with_ci': table,
        }

    # ---------------------------------------------------------------- run

    def run(self, run_name):
        frame = self._load_frame(run_name)
        logged = {'n_questions': len(frame['correct']),
                  'n_correct': int(frame['correct'].sum())}
        panels = {}
        panels.update(self._panel_risk_coverage(frame, logged))
        panels.update(self._panel_auroc_bounds(frame, logged))

        wandb.log({**panels, **logged})
        return logged, sorted(panels)


def notes_markdown(run_name, logged):
    """Short prose on the run's overview tab, so the caveats travel with the run.

    Notes are not a panel, so this adds nothing to the chart area.
    """
    def show(key):
        return logged.get(key, 'n/a')

    return (
        f'## {run_name}\n\n'
        'Charts from `code/log_wandb_charts.py`. AUROC uses **incorrectness as the '
        'positive label** (higher = better, 0.5 = chance).\n\n'
        'Each chart is logged twice: an interactive `charts/*` custom-chart panel and an '
        '`images/*` matplotlib render of the same data. **Read the `images/*` pair** - an '
        'image panel has no chart spec to resolve, and `images/auroc_bounds` is a real '
        'grouped bar chart with named measures rather than the rank axis the native '
        'panel is forced onto.\n\n'
        '- **`charts/risk_coverage`** - accuracy when answering only the most-confident '
        'fraction and abstaining on the rest. Every point is a deployable configuration. '
        f"At 20% coverage: {show('riskcov_acc_at_20pct/average_predictive_entropy')} "
        f"against a {show('riskcov_base_rate')} base rate. Read the left end with care - "
        'at low coverage a single question moves it several points.\n'
        '- **`charts/auroc_bounds`** - AUROC with its bootstrap 95% interval. The x-axis '
        'is a rank; `tables/auroc_with_ci` is sorted identically and names each one. '
        'Trust only rows where `ci_excludes_chance` is true.\n'
        f"- **Positive class**: {show('n_correct')} correct of {show('n_questions')} "
        'questions. Correctness is `rougeL > 0.3`, which on short answers passes on '
        'surname overlap alone, so the labels every AUROC ranks against carry real error.'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_id', type=str, default='run_1')
    parser.add_argument('--generation_model', type=str, default='opt-350m', choices=OPT_MODELS)
    parser.add_argument('--evaluation_model', type=str, default=None,
                        help='Accepted for CLI symmetry with the other stages; the '
                             'confidence pickle is named by --generation_model alone.')
    parser.add_argument('--seed', type=int, default=10,
                        help='Seeds the bootstrap resampler, so intervals are reproducible.')
    parser.add_argument('--bootstrap_resamples', type=int, default=BOOTSTRAP_RESAMPLES,
                        help='0 logs point estimates with no interval, which is most of '
                             "this stage's runtime.")
    parser.add_argument('--no_notes', action='store_true',
                        help='Skip setting the run notes.')
    return parser.parse_args()


def main():
    args = parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:8]
    wandb.init(project='nlg_uncertainty', id=run_id, config=args, resume='allow')
    run_name = wandb.run.name or run_id
    print(f'run_id={run_id}  run_name={run_name}')

    logged, panel_names = ChartLogger(args).run(run_name)

    if not args.no_notes:
        try:
            wandb.run.notes = notes_markdown(run_name, logged)
        except Exception as error:  # notes are a nicety, never a reason to fail the stage
            print(f'could not set run notes: {type(error).__name__}: {error}')

    print('\npanels logged:')
    for name in panel_names:
        print(f'  {name}')
    print('\n' + '  '.join(f'{k}={v}' for k, v in logged.items() if '/' not in k))

    wandb.finish()


if __name__ == '__main__':
    main()
