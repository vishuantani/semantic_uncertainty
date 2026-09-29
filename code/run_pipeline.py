"""Run the whole improved semantic-uncertainty pipeline from a single command.

Replaces run_pipeline.sh for the `_improved` stack. Nine stages run in order, each
as its own subprocess so that (a) every stage keeps the CLI contract it already
has, (b) the model weights of one stage are released before the next loads its
own, and (c) analyze_results.py -- which parses its args at module scope -- can
be driven at all.

    python code/run_pipeline.py --wandb_run_name opt350m-baseline

Every stage calls wandb.init(project='nlg_uncertainty', id=run_id, resume='allow')
and then derive every file path from `wandb.run.name or run_id`. Exporting
WANDB_NAME into the child environment therefore does two jobs at once: it names
the single wandb run that the whole pipeline logs into, and it names the
output/{sequences,entailment,likelihoods,confidence}/<run_name>/ directories --
instead of leaving them at a wandb-generated name like `avid-darkness-3`.

Any flag you omit falls through to the stage script's own default, so the child
scripts stay the single source of truth for defaults. The exceptions are
--model and --seed, which this script owns because they must agree across stages.

The `score` stage is new_score_accuracy.py, which has no `_improved` suffix because
it has no upstream equivalent: correctness scoring was moved out of generate_improved.py
so that it runs AFTER cleaning. It must stay between `clean` and `similarities` -- it
writes exact_match and rouge*_to_target back into the generations pickle, and
analyze_results.py raises KeyError('rougeL_to_target') without it.

The p(True) baseline (get_prompting_based_uncertainty.py) is deliberately not
run: it has no `_improved` version, and neither compute_confidence_measure_improved.py
nor analyze_results.py reads its output.
"""

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time

# Every stage does a bare `import config`, whose paths ('./data', './output') are
# relative to the repo root. So we chdir to the root and put code/ on PYTHONPATH
# rather than trusting whatever cwd the caller happened to be in.
CODE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

sys.path.insert(0, str(CODE_DIR))
import config  # noqa: E402  (needs CODE_DIR on the path first)

STAGE_ORDER = ['generate', 'clean', 'score', 'similarities', 'likelihoods', 'confidence',
               'analyze', 'charts', 'report']

SCRIPTS = {
    'generate': 'generate_improved.py',
    'clean': 'clean_generated_strings_improved.py',
    'score': 'new_score_accuracy.py',
    'similarities': 'get_semantic_similarities_improved.py',
    'likelihoods': 'get_likelihoods_improved.py',
    'confidence': 'compute_confidence_measure_improved.py',
    'analyze': 'analyze_results.py',
    'charts': 'log_wandb_charts.py',
    'report': 'report_pdf.py',
}

OPT_MODELS = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']


# --- command construction ----------------------------------------------------------

def _flag(name, value):
    """Emit ['--name', 'value'] only when the value was actually supplied.

    A value starting with '-' is emitted as a single '--name=value' token instead.
    argparse treats a separate argv element beginning with a dash as another option and
    rejects it with "expected one argument"; the '=' form bypasses that heuristic. Without
    this, any flag whose value can start with a dash is unpassable to a child stage.
    """
    if value is None:
        return []
    text = str(value)
    return [f'{name}={text}'] if text.startswith('-') else [name, text]


def _switch(name, enabled):
    """store_true flags are forwarded only when set, so the child keeps its default."""
    return [name] if enabled else []


def _generate_args(a):
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--model', a.model),
        *_flag('--seed', a.seed),
        *_flag('--type_of_question', a.type_of_question),
        *_flag('--num_generations_per_prompt', a.num_generations_per_prompt),
        *_flag('--fraction_of_data_to_use', a.fraction_of_data_to_use),
        *_flag('--temperature', a.temperature),
        *_flag('--num_beams', a.num_beams),
        *_flag('--decoding_method', a.decoding_method),
        *_flag('--top_p', a.top_p),
        *_flag('--dataset', a.dataset),
        *_flag('--dataset_num_examples', a.dataset_num_examples),
        *_flag('--max_length_of_generated_sequence', a.max_length_of_generated_sequence),
        *_flag('--most_likely_num_beams', a.most_likely_num_beams),
        *_flag('--ban_list', a.ban_list),
        *(['--turn_marker_tokens', *a.turn_marker_tokens] if a.turn_marker_tokens else []),
        *_switch('--stop_on_turn_marker', a.stop_on_turn_marker),
        *_switch('--include_model_eos', a.include_model_eos),
        *_switch('--batch_samples', a.batch_samples),
        *_switch('--fix_question_parsing', a.fix_question_parsing),
        *_switch('--pickle_on_cpu', a.pickle_on_cpu),
        *_flag('--inspect_entries', a.inspect_entries),
    ]


def _clean_args(a):
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--seed', a.seed),
        *_flag('--strings_to_filter_on', a.strings_to_filter_on),
    ]


def _score_args(a):
    # No --seed: scoring is deterministic given the pickle.
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--dataset', a.dataset),
        *_flag('--score_on', a.score_on),
    ]


def _similarities_args(a):
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--seed', a.seed),
        *_flag('--entailment_model', a.entailment_model),
        *_flag('--similarity_style', a.similarity_style),
    ]


def _likelihoods_args(a):
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--evaluation_model', a.evaluation_model),
        *_flag('--seed', a.seed),
        *_switch('--drop_empty_generations', a.drop_empty_generations),
    ]


def _confidence_args(a):
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--evaluation_model', a.evaluation_model),
        *_flag('--seed', a.seed),
        *_flag('--llh_shift', a.llh_shift),
        *_switch('--drop_empty_generations', a.drop_empty_generations),
    ]


def _analyze_args(a):
    # analyze_results.py declares --verbose as `type=bool`, so `--verbose False`
    # would parse as True. The empty string is the only value that reaches it as
    # falsey, which is why --no_analyze_verbose passes '' rather than 'False'.
    verbose = [] if a.analyze_verbose else ['--verbose', '']
    return ['-n', a.run_id, *_flag('--model', a.model), *verbose]


def _charts_args(a):
    # Read-only stage: it opens the run's own pickles and logs panels to W&B. Safe to
    # re-run alone with --only charts, and safe to --skip charts entirely.
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--evaluation_model', a.evaluation_model),
        *_flag('--seed', a.seed),
        *_flag('--bootstrap_resamples', a.bootstrap_resamples),
        *_switch('--no_notes', not a.charts_notes),
    ]


def _report_args(a):
    # Read-only, like charts: it opens the run's pickles, renders one PDF page and uploads
    # it. Safe to re-run alone with --only report, and safe to --skip entirely. It shares
    # --seed and --bootstrap_resamples with the charts stage on purpose - matching those
    # is what makes the PDF's intervals identical to the ones in the W&B panels.
    return [
        *_flag('--run_id', a.run_id),
        *_flag('--generation_model', a.model),
        *_flag('--evaluation_model', a.evaluation_model),
        *_flag('--seed', a.seed),
        *_flag('--bootstrap_resamples', a.bootstrap_resamples),
    ]


ARG_BUILDERS = {
    'generate': _generate_args,
    'clean': _clean_args,
    'score': _score_args,
    'similarities': _similarities_args,
    'likelihoods': _likelihoods_args,
    'confidence': _confidence_args,
    'analyze': _analyze_args,
    'charts': _charts_args,
    'report': _report_args,
}


# --- CLI ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    orch = parser.add_argument_group('orchestration')
    orch.add_argument('--wandb_run_name', type=str, required=True,
                      help='Names the wandb run AND the output/*/<name>/ directories. '
                           'Use a fresh name for a fresh run.')
    orch.add_argument('--run_id', type=str, default=None,
                      help='wandb run id shared by every stage so they all log into one '
                           'run. Defaults to a slug of --wandb_run_name.')
    orch.add_argument('--start_from', type=str, default=None, choices=STAGE_ORDER,
                      help='Resume the pipeline at this stage instead of the beginning.')
    orch.add_argument('--only', nargs='+', default=None, choices=STAGE_ORDER, metavar='STAGE',
                      help='Run only these stages. Mutually exclusive with --start_from.')
    orch.add_argument('--skip', nargs='+', default=None, choices=STAGE_ORDER, metavar='STAGE',
                      help='Drop these stages from whatever selection is in effect.')
    orch.add_argument('--dry_run', action='store_true',
                      help='Print the exact command for each stage without running any.')
    orch.add_argument('--python', type=str, default=sys.executable,
                      help='Interpreter used for the stage subprocesses.')

    shared = parser.add_argument_group('shared across stages')
    shared.add_argument('--model', type=str, default='opt-350m', choices=OPT_MODELS,
                        help='Generation model. Passed as --model to generate and as '
                             '--generation_model to every later stage.')
    shared.add_argument('--evaluation_model', type=str, default=None,
                        help='Model that scores likelihoods. Defaults to --model.')
    shared.add_argument('--seed', type=int, default=10)

    gen = parser.add_argument_group('generate_improved.py')
    gen.add_argument('--type_of_question', type=str, default=None)
    gen.add_argument('--num_generations_per_prompt', type=int, default=None)
    gen.add_argument('--fraction_of_data_to_use', type=float, default=None)
    gen.add_argument('--temperature', type=float, default=None)
    gen.add_argument('--num_beams', type=int, default=None,
                     help='Beams for the SAMPLED generations. 1 = plain multinomial '
                          'sampling, which is what the uncertainty estimate wants.')
    gen.add_argument('--decoding_method', type=str, default=None,
                     choices=['beam_search', 'greedy'])
    gen.add_argument('--top_p', type=float, default=None)
    gen.add_argument('--dataset', type=str, default='trivia_qa', choices=['trivia_qa', 'coqa'])
    gen.add_argument('--dataset_num_examples', type=int, default=200,
                     help='Identifies the dataset filepath written by parse_triviaqa.py.')
    gen.add_argument('--max_length_of_generated_sequence', type=int, default=None)
    gen.add_argument('--most_likely_num_beams', type=int, default=None,
                     help='Beams for the deterministic answer. Must be >= 2.')
    gen.add_argument('--ban_list', type=str, default=None, choices=['upstream', 'extended'])
    gen.add_argument('--turn_marker_tokens', nargs='+', default=None, metavar='TOKEN')
    gen.add_argument('--stop_on_turn_marker', action='store_true')
    gen.add_argument('--include_model_eos', action='store_true',
                     help="Add the tokenizer's own eos id to the stop list. Without it "
                          'eos_token_id REPLACES the model eos, so the model can emit '
                          '</s> and keep generating past it.')
    gen.add_argument('--batch_samples', action='store_true',
                     help='Draw all N samples in one generate() call. Requires --num_beams 1.')
    gen.add_argument('--fix_question_parsing', action='store_true')
    gen.add_argument('--pickle_on_cpu', action='store_true')
    gen.add_argument('--inspect_entries', type=int, default=None)

    clean = parser.add_argument_group('clean_generated_strings_improved.py')
    clean.add_argument('--strings_to_filter_on', type=str, default=None,
                       choices=['upstream', 'updated'])

    score = parser.add_argument_group('new_score_accuracy.py')
    score.add_argument('--score_on', type=str, default=None, choices=['cleaned', 'raw'],
                       help="Which variant is aliased onto the bare metric keys "
                            "analyze_results.py reads. Both are always stored. 'raw' "
                            "reproduces upstream accuracy only.")

    sim = parser.add_argument_group('get_semantic_similarities_improved.py')
    sim.add_argument('--entailment_model', type=str, default=None)
    sim.add_argument('--similarity_style', type=str, default=None, choices=['ENSURE_ENTAILMENT', 'NOT_CONTRADICTION'],
                     help='Rule that decides when two answers land in the same semantic '
                          'set. Left unset so the stage script keeps ownership of the '
                          'default and of which values are valid.')

    conf = parser.add_argument_group('compute_confidence_measure_improved.py')
    conf.add_argument('--llh_shift', type=float, default=None)

    empty = parser.add_argument_group('empty-generation handling (likelihoods + confidence)')
    empty.add_argument('--drop_empty_generations', action='store_true',
                       help='Exclude samples that generated no tokens from each '
                            "question's entropy, renormalising over the samples that "
                            'survive. Passed to BOTH stages that read it. Without it a '
                            'single empty generation makes analyze_results.py raise '
                            '"Input contains NaN".')

    charts = parser.add_argument_group('log_wandb_charts.py')
    charts.add_argument('--bootstrap_resamples', type=int, default=None,
                        help='Resamples behind each AUROC confidence interval. 0 logs '
                             'point estimates only, which is much faster.')
    charts.add_argument('--no_charts_notes', dest='charts_notes', action='store_false',
                        help='Skip setting the run notes.')
    parser.set_defaults(charts_notes=True)

    ana = parser.add_argument_group('analyze_results.py')
    ana.add_argument('--no_analyze_verbose', dest='analyze_verbose', action='store_false',
                     help='Silence the per-metric printing in analyze_results.py.')
    parser.set_defaults(analyze_verbose=True)

    args = parser.parse_args()

    if args.only and args.start_from:
        parser.error('--only and --start_from are mutually exclusive.')

    return args


def select_stages(args):
    """Resolve --only / --start_from / --skip into an ordered stage list."""
    if args.only:
        stages = [s for s in STAGE_ORDER if s in set(args.only)]
    elif args.start_from:
        stages = STAGE_ORDER[STAGE_ORDER.index(args.start_from):]
    else:
        stages = list(STAGE_ORDER)

    if args.skip:
        stages = [s for s in stages if s not in set(args.skip)]

    return stages


def slugify(name):
    """wandb ids must be filesystem- and URL-safe; run names need not be."""
    slug = re.sub(r'[^0-9a-zA-Z._-]+', '-', name).strip('-')
    return slug or 'run'


# --- execution ---------------------------------------------------------------------

def run_stage(key, cmd, env, log_path):
    """Stream a stage's output to the console and a log file, returning (code, seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with open(log_path, 'wb') as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                # read1 rather than readline: tqdm redraws with \r and would
                # otherwise sit in the buffer until the bar finished.
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                log.write(chunk)
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise

    return code, time.time() - start


def format_duration(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}:{minutes:02d}:{secs:02d}'


def preflight(args, stages):
    """Fail before loading any model if the dataset generate needs is not there."""
    if 'generate' not in stages or args.dataset != 'trivia_qa':
        return
    dataset_path = REPO_ROOT / config.trivia_qa_path(args.dataset_num_examples).lstrip('./')
    if not dataset_path.exists():
        sys.exit(f'Dataset not found at {dataset_path}\n'
                 f'Run: python code/parse_triviaqa.py --num_examples {args.dataset_num_examples}')


def main():
    args = parse_args()
    stages = select_stages(args)

    if not stages:
        sys.exit('No stages selected.')

    args.run_id = args.run_id or slugify(args.wandb_run_name)
    args.evaluation_model = args.evaluation_model or args.model
    preflight(args, stages)

    env = os.environ.copy()
    # The one lever that names the wandb run and every output directory at once.
    env['WANDB_NAME'] = args.wandb_run_name
    env['PYTHONPATH'] = os.pathsep.join(p for p in [str(CODE_DIR), env.get('PYTHONPATH', '')] if p)
    env['PYTHONUNBUFFERED'] = '1'

    print(f'run_name  : {args.wandb_run_name}')
    print(f'run_id    : {args.run_id}')
    print(f'model     : {args.model} (eval: {args.evaluation_model})')
    print(f'stages    : {" -> ".join(stages)}')
    print(f'outputs   : {REPO_ROOT / "output"}/*/{args.wandb_run_name}/')
    print()

    log_dir = REPO_ROOT / 'output' / 'logs' / args.wandb_run_name
    timings = []

    for position, key in enumerate(stages, start=1):
        cmd = [args.python, str(CODE_DIR / SCRIPTS[key]), *ARG_BUILDERS[key](args)]
        printable = ' '.join(shlex.quote(part) for part in cmd)

        if args.dry_run:
            print(f'[{position}/{len(stages)}] {key}\n    {printable}\n')
            continue

        print(f'\n{"=" * 78}\n[{position}/{len(stages)}] {key}: {printable}\n{"=" * 78}',
              flush=True)

        code, elapsed = run_stage(key, cmd, env, log_dir / f'{position:02d}_{key}.log')
        timings.append((key, elapsed))

        if code != 0:
            print(f'\nStage {key!r} failed with exit code {code}.')
            print(f'Log: {log_dir / f"{position:02d}_{key}.log"}')
            print(f'Re-run the same command with --start_from {key} to resume from here.')
            sys.exit(code)

        print(f'\n[{position}/{len(stages)}] {key} finished in {format_duration(elapsed)}')

    if args.dry_run:
        print('Dry run: nothing executed.')
        return

    print(f'\n{"=" * 78}\nPipeline complete: {args.wandb_run_name}\n{"=" * 78}')
    for key, elapsed in timings:
        print(f'  {key:<14} {format_duration(elapsed)}')
    print(f'  {"TOTAL":<14} {format_duration(sum(e for _, e in timings))}')
    print(f'\nLogs    : {log_dir}')
    print(f'Results : {REPO_ROOT / "overall_results.json"}, '
          f'{REPO_ROOT / "accuracy_verification.csv"}')


if __name__ == '__main__':
    main()
