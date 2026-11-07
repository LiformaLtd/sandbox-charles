"""Run matched width/seed fine-tunes and evaluate each selected checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seeds', type=int, nargs='+', default=[37, 73, 101])
    parser.add_argument('--widths', type=int, nargs='+', default=[96, 128, 192])
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--training-python', default=sys.executable)
    parser.add_argument('--evaluation-python', default=sys.executable)
    parser.add_argument('--onnx-site-packages', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.jobs) < 1 or any(w < 96 for w in args.widths):
        parser.error('Use positive epochs/jobs and widths of at least 96.')
    source = Path(__file__).resolve().parent
    output = Path(os.environ['A2F_ABLATION_OUTPUT']).resolve()
    work = Path(os.environ['A2F_RESEARCH_WORK']).resolve()
    frozen = output / 'frozen'
    frozen.mkdir(parents=True, exist_ok=True)
    for name in ['train.py', 'model.py', 'evaluate.py', 'evaluation_support.py', 'metrics.py']:
        original, snapshot = source / name, frozen / name
        if snapshot.exists() and snapshot.read_bytes() != original.read_bytes():
            raise RuntimeError('Frozen recipe differs. Choose a new output directory for a new recipe.')
        shutil.copy2(original, snapshot)
    env = {**os.environ, 'A2F_ABLATION_OUTPUT': str(output), 'A2F_RESEARCH_WORK': str(work)}
    compatibility = ''
    if args.onnx_site_packages:
        compatibility = f'sys.path.append({str(args.onnx_site_packages)!r});'
    boot = ('import numpy,torch,sys,runpy,os;' + compatibility +
            'script=sys.argv.pop(1);sys.path.insert(0,os.path.dirname(script));'
            'runpy.run_path(script,run_name="__main__")')
    # Warm the immutable feature cache before concurrent readers.
    prepare = (f'import sys,importlib.util;sys.path.insert(0,{str(frozen)!r});' +
               f'spec=importlib.util.spec_from_file_location("study",{str(frozen / "train.py")!r});'
               'm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);'
               'active=m.np.load(m.WORK/"arkit-student/rig-metrics.npz")["active"];'
               '[(m.dataset(root,split,active,16000)) for root in m.ROOTS '
               'for split in ("train","dev") if split=="train" or root!=m.ROOTS[1]]')
    with (output / 'prepare.log').open('w') as log:
        subprocess.run([args.training_python, '-u', '-c',
                        'import numpy,torch,sys;' + compatibility + prepare],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    queue = [(w, s) for s in args.seeds for w in args.widths]
    active, completed = {}, []
    try:
        while queue or active:
            while queue and len(active) < args.jobs:
                width, seed = queue.pop(0)
                name = f'w{width}-s{seed}'
                log = (output / f'{name}.log').open('w')
                command = [args.training_python, '-u', '-c', boot, str(frozen / 'train.py'),
                           '--name', name, '--width', str(width), '--seed', str(seed),
                           '--epochs', str(args.epochs)]
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[name] = process, log
                print('Started', name, flush=True)
            for name, (process, log) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                del active[name]
                if process.returncode:
                    raise RuntimeError(f'Training failed: {name}; inspect its log.')
                with (output / f'{name}-eval.log').open('w') as evaluation:
                    subprocess.run([args.evaluation_python, str(frozen / 'evaluate.py'), name,
                                    str(output / f'{name}.onnx')], env=env, stdout=evaluation,
                                   stderr=subprocess.STDOUT, check=True)
                completed.append(name)
                print('Completed and evaluated', name, flush=True)
            (output / 'progress.json').write_text(json.dumps({
                'completed': completed, 'active': list(active), 'queued': queue,
            }, indent=2))
            time.sleep(1)
    finally:
        for process, log in active.values():
            process.terminate()
            process.wait()
            log.close()
    recipe = hashlib.sha256((frozen / 'train.py').read_bytes() + (frozen / 'model.py').read_bytes()).hexdigest()
    (output / 'training-complete.json').write_text(json.dumps({
        'completed': completed, 'epochs': args.epochs, 'seeds': args.seeds,
        'widths': args.widths, 'recipe_sha256': recipe,
    }, indent=2))


if __name__ == '__main__':
    main()
