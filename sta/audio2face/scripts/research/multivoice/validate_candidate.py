"""Select using development only, then evaluate the frozen v4 confirmation set."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
PILOT = Path(os.environ.get('A2F_MULTIVOICE_PILOT', APP / '.local/compact-multivoice')).resolve()
CONFIRM = Path(os.environ.get('A2F_CONFIRMATION_ROOT', APP / '.local/compact-v4-confirmation')).resolve()
OUT = Path(os.environ.get('A2F_VALIDATION_OUTPUT', APP / '.local/compact-v4-validation')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def select():
    protocol = read(CONFIRM / 'protocol.json')
    if sha(CONFIRM / 'planned.json') != protocol['planned_sha256']:
        raise ValueError('Frozen confirmation plan changed')
    reports = []
    reference = None
    for seed in protocol['seeds']:
        run = PILOT / 'runs' / f'control-replay-s{seed}'
        report = read(run / 'training.json')
        config = report['recipe']['config'].copy()
        if config.pop('seed') != seed:
            raise ValueError('Seed configuration mismatch')
        if reference is None:
            reference = {k: v for k, v in report['recipe'].items() if k != 'config'}
            reference['config_without_seed'] = config
        comparison = {k: v for k, v in report['recipe'].items() if k != 'config'}
        comparison['config_without_seed'] = config
        if comparison != reference or config != dict(arm='control', epochs=24, threads=4, lr=.00012, clean_replay=True):
            raise ValueError('Training recipes or inputs differ across seeds')
        if report['recipe']['code_sha256'] != protocol['training_code_sha256']:
            raise ValueError('Training source differs from frozen protocol')
        if sha(run / 'model.onnx') != report['model_sha256'] or report['parameters'] != 331047:
            raise ValueError('Incomplete or changed export')
        reports.append(dict(seed=seed, validation_selection=report['best_selection'],
                            best_epoch=report['best_epoch'], model=str(run / 'model.onnx'),
                            model_sha256=report['model_sha256'], bytes=report['bytes']))
    chosen = min(reports, key=lambda r: (r['validation_selection'], r['seed']))
    selection = dict(protocol_sha256=sha(CONFIRM / 'protocol.json'), planned_sha256=protocol['planned_sha256'],
                     runs=reports, representative=chosen,
                     selection='Development scores only, frozen before fresh confirmation evaluation')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'selection.json'
    if target.exists() and read(target) != selection:
        raise ValueError('Frozen representative selection changed')
    target.write_text(json.dumps(selection, indent=2))
    shutil.copy2(CONFIRM / 'protocol.json', OUT / 'protocol.json')
    shutil.copy2(chosen['model'], OUT / 'v4-candidate.onnx')
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def command(script, args, root):
    env = dict(os.environ, A2F_MULTIVOICE_OUTPUT=str(root), A2F_ABLATION_OUTPUT=str(PILOT))
    subprocess.run([sys.executable, str(script), *args], check=True, env=env)


def compare(seed_reports, baseline, selected_seed):
    voices = sorted({c['engine'] + ':' + c['voice'] for c in baseline['clips']})
    texts = sorted({c['text'] for c in baseline['clips']})
    def array(report):
        lookup = {(c['engine'] + ':' + c['voice'], c['text']): c for c in report['clips']}
        return np.asarray([[[lookup[v, t][key] for key in ['squared_gap_error', 'gap_values',
                                                         'teacher_contacts', 'student_contacts', 'contact_hits']]
                            for t in texts] for v in voices])
    reference = array(baseline)
    candidates = np.stack([array(r) for r in seed_reports])
    def score(a):
        total = a.sum(axis=-2)
        gap = np.sqrt(total[..., 0] / total[..., 1]).mean()
        f1 = (2 * total[..., 4] / np.maximum(1, total[..., 2] + total[..., 3])).mean()
        return np.array([gap, f1])
    base_score = score(reference)
    seed_scores = [score(a) for a in candidates]
    primary = np.mean(seed_scores, axis=0)
    groups = [[i for i, v in enumerate(voices) if v.startswith(engine + ':')] for engine in ['phonon', 'supertonic']]
    rng = np.random.default_rng(37)
    draws = []
    for _ in range(4000):
        si = rng.integers(len(candidates), size=len(candidates))
        vi = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        ti = rng.integers(len(texts), size=len(texts))
        draws.append(score(candidates[si][:, vi][:, :, ti]) - score(reference[vi][:, ti]))
    ci = np.quantile(draws, [.025, .975], axis=0)
    per_voice = {}
    for index, voice in enumerate(voices):
        b = score(reference[index:index + 1])
        a = np.mean([score(v[index:index + 1]) for v in candidates], axis=0)
        per_voice[voice] = dict(baseline_gap_rmse=float(b[0]), candidate_gap_rmse=float(a[0]),
                               gap_relative_change=float(a[0] / b[0] - 1),
                               baseline_contact_f1=float(b[1]), candidate_contact_f1=float(a[1]))
    per_engine = {}
    for engine, indices in zip(['phonon', 'supertonic'], groups):
        b = score(reference[indices])
        a = np.mean([score(v[indices]) for v in candidates], axis=0)
        per_engine[engine] = dict(baseline_gap_rmse=float(b[0]), candidate_gap_rmse=float(a[0]),
                                  gap_relative_change=float(a[0] / b[0] - 1),
                                  baseline_contact_f1=float(b[1]), candidate_contact_f1=float(a[1]))
    return dict(baseline_macro_gap_rmse=float(base_score[0]), candidate_macro_gap_rmse=float(primary[0]),
                gap_relative_change=float(primary[0] / base_score[0] - 1),
                baseline_macro_contact_f1=float(base_score[1]), candidate_macro_contact_f1=float(primary[1]),
                gap_delta_95ci=ci[:, 0].tolist(), contact_f1_delta_95ci=ci[:, 1].tolist(),
                seed_gap_rmse=[float(v[0]) for v in seed_scores],
                seed_contact_f1=[float(v[1]) for v in seed_scores],
                seed_gap_std=float(np.std([v[0] for v in seed_scores], ddof=1)),
                per_voice=per_voice, per_engine=per_engine,
                bootstrap='4000 paired crossed fine-tuning seed/voice/text draws; voices sampled within engine',
                limitation='Three seeds; two new voices per engine; not universal voice certification')


def evaluate(selection):
    baseline_path = WORK / 'arkit-student-v3/james-arkit-student-v3.onnx'
    command(HERE / 'evaluate.py', ['v3', str(baseline_path)], CONFIRM)
    command(APP / 'scripts/research/compact_ablation/evaluate_confirmation.py', ['baseline'], PILOT)
    for run in selection['runs']:
        name = 'v4-s' + str(run['seed'])
        if sha(run['model']) != run['model_sha256']:
            raise ValueError('Selected export changed')
        command(HERE / 'evaluate.py', [name, run['model']], CONFIRM)
        command(HERE / 'evaluate.py', [name, run['model']], PILOT)
        shutil.copy2(run['model'], PILOT / (name + '.onnx'))
        command(APP / 'scripts/research/compact_ablation/evaluate_confirmation.py', [name], PILOT)
    fresh = [read(CONFIRM / 'evaluation' / f'v4-s{r["seed"]}-test.json') for r in selection['runs']]
    baseline = read(CONFIRM / 'evaluation/v3-test.json')
    result = compare(fresh, baseline, selection['representative']['seed'])
    old = [read(PILOT / f'confirmation-v4-s{r["seed"]}.json')['all']['student'] for r in selection['runs']]
    old_base = read(PILOT / 'confirmation-baseline.json')['all']['student']
    old_gap = float(np.mean([v['lip_gap_rmse'] for v in old]))
    old_f1 = float(np.mean([v['contact_f1'] for v in old]))
    result['historical_easton'] = dict(baseline=old_base, seeds=old, mean_gap_rmse=old_gap,
                                      gap_relative_change=old_gap / old_base['lip_gap_rmse'] - 1,
                                      mean_contact_f1=old_f1, contact_f1_delta=old_f1 - old_base['contact_f1'])
    screen = read(CONFIRM / 'protocol.json')['candidate_screen']
    checks = dict(
        fresh_gap_reduction=-result['gap_relative_change'] >= screen['fresh_macro_gap_reduction_at_least'],
        each_engine_gap=all(-v['gap_relative_change'] >= screen['fresh_each_engine_gap_reduction_at_least'] for v in result['per_engine'].values()),
        fresh_contact_f1=result['candidate_macro_contact_f1'] - result['baseline_macro_contact_f1'] >= screen['fresh_macro_contact_f1_delta_at_least'],
        each_voice_gap=all(v['gap_relative_change'] <= screen['fresh_each_voice_rms_relative_regression_at_most'] for v in result['per_voice'].values()),
        historical_easton_gap=result['historical_easton']['gap_relative_change'] <= screen['historical_easton_mean_gap_relative_regression_at_most'],
        historical_easton_contact=result['historical_easton']['contact_f1_delta'] >= screen['historical_easton_mean_contact_f1_delta_at_least'])
    result.update(engineering_checks=checks, engineering_screen_passed=all(checks.values()),
                  representative=selection['representative'], confirmation_manifest_sha256=sha(CONFIRM / 'manifest.json'),
                  evaluation_source_sha256=sha(__file__), protocol_sha256=selection['protocol_sha256'])
    (OUT / 'validation-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['select', 'evaluate'], required=True)
    args = p.parse_args()
    selection = select()
    if args.stage == 'evaluate':
        evaluate(selection)
