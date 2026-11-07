"""Create an isolated v3/v4 WebGPU comparison with a frozen selected export."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
OUT = Path(os.environ.get('A2F_VALIDATION_OUTPUT', APP / '.local/compact-v4-validation')).resolve()
REFERENCE = APP / 'scripts/research/compact_ablation'


def main():
    selection = json.loads((OUT / 'selection.json').read_text())['representative']
    path = Path(selection['model'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection['model_sha256']:
        raise ValueError('Validation-selected export changed')
    root = OUT / 'browser'
    env = dict(os.environ, A2F_ABLATION_OUTPUT=str(root))
    subprocess.run([sys.executable, str(REFERENCE / 'prepare_browser.py')], check=True, env=env)
    shutil.copy2(path, root / 'candidate.onnx')
    shutil.copytree(APP / '.local/supertonic', root / 'web/public/assets/supertonic', dirs_exist_ok=True)
    config = root / 'vite.config.mjs'
    source = config.read_text()
    needle = "'contextwide','data','folded'];"
    if source.count(needle) != 1:
        raise ValueError('Reference harness allowlist changed')
    config.write_text(source.replace(needle, "'contextwide','data','folded','candidate'];"))
    subprocess.run(['clang', '-O2', str(REFERENCE / 'sample.c'), '-o', str(root / 'sample')], check=True)
    (root / 'selected-model.json').write_text(json.dumps(selection, indent=2))
    print('Prepared validation-selected seed', selection['seed'], 'in', root)


if __name__ == '__main__':
    main()
