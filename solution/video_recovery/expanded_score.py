"""One optional public comparison after both video candidates are frozen.

This never fits a model or alters predictions. It is separate from recovery,
so rebuilding observations does not read the public possession answers.
"""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, subprocess, sys
from .config import ROOT, OUT

BEST_SHA = 'd2048553ec2ad4538369a5f786c24531d2d95439c0bf6f3a9102bf1a0aa240f5'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _fresh_out(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Outputs must stay under ignored data/')
    if path.exists() and any(path.iterdir()):
        raise ValueError('Score output is not empty; inspect the previous run instead of rescoring')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _verified_build(path):
    path = Path(path).resolve()
    report = json.loads((path/'benchmark_build.json').read_text())
    if report['status'] != 'experimental_predictions_ready':
        raise ValueError('No completed candidate in '+str(path))
    for key in ('baseline_reproduced', 'baseline_file_byte_identical', 'final_dead_mask_identical',
                'final_x_mask_identical', 'input_hashes_stable'):
        if report['checks'].get(key) is not True:
            raise ValueError('Candidate did not pass '+key)
    prediction = path/'predictions.csv'
    if _digest(prediction) != report['prediction_sha256']:
        raise ValueError('Prediction changed after its build')
    if not all(_digest(v['path']) == v['sha256'] for v in report['input_hashes'].values()):
        raise ValueError('A build input changed; do not score altered outputs')
    return report, prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-build', type=Path, default=OUT/'benchmark')
    parser.add_argument('--expanded-build', type=Path, default=OUT/'full_match/benchmark')
    parser.add_argument('--out', type=Path, default=OUT/'full_match/evaluation')
    args = parser.parse_args(argv)
    short, base_file = _verified_build(args.short_build)
    expanded, expanded_file = _verified_build(args.expanded_build)
    if _digest(base_file) != BEST_SHA:
        raise ValueError('Short-video baseline differs from the recorded comparator')
    added = int(expanded['recovered_frames'])-int(short['recovered_frames'])
    if added < 500:
        raise ValueError('Fewer than 500 additional observations; no meaningful expansion')
    out = _fresh_out(args.out)
    inputs = out/'inputs'; inputs.mkdir()
    shutil.copy2(base_file, inputs/'baseline_video.csv')
    shutil.copy2(expanded_file, inputs/'expanded_video.csv')
    lock = {'baseline_sha256': _digest(base_file), 'candidate_sha256': _digest(expanded_file),
            'short_build_report_sha256': _digest(Path(args.short_build)/'benchmark_build.json'),
            'expanded_build_report_sha256': _digest(Path(args.expanded_build)/'benchmark_build.json'),
            'candidate_frozen_before_public_answers': True,
            'public_spell_gain_minimum': .01, 'public_macro_decline_maximum': .01}
    (out/'score_lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--results', str(inputs),
                    '--gt', 'public', '--out', str(out/'official'), '--bootstrap', '0', '--no-detail'],
                   cwd=ROOT, check=True)
    with (out/'official/scoreboard.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    metrics = {Path(r['submission']).stem: r for r in rows if r['status'] == 'scored'}
    if set(metrics) != {'baseline_video', 'expanded_video'}:
        raise RuntimeError('Unexpected official evaluation output')
    baseline, new = metrics['baseline_video'], metrics['expanded_video']
    base_spell, new_spell = float(baseline['spell_f1_mean']), float(new['spell_f1_mean'])
    base_macro, new_macro = float(baseline['macro_f1']), float(new['macro_f1'])
    stable = _digest(base_file) == lock['baseline_sha256'] and _digest(expanded_file) == lock['candidate_sha256']
    if not stable: raise RuntimeError('Predictions changed during scoring')
    result = {'baseline_public_spell_f1': base_spell, 'expanded_public_spell_f1': new_spell,
              'gain': new_spell-base_spell, 'baseline_macro_f1': base_macro, 'expanded_macro_f1': new_macro,
              'additional_coordinates': added,
              'public_sample_lead': new_spell-base_spell >= .01 and new_macro-base_macro >= -.01,
              'hashes_stable': True, 'candidate_promoted': False, 'hidden_whole_match_score_known': False,
              'metrics': metrics}
    (out/'public_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__': main()
