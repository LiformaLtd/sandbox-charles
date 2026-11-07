"""Reuse the frozen browser benchmark without altering the demo or old study."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()
REFERENCE = APP / 'scripts/research/compact_ablation'


def main():
    root = OUT / 'browser'
    root.mkdir(exist_ok=True)
    env = dict(os.environ, A2F_ABLATION_OUTPUT=str(root))
    subprocess.run([sys.executable, str(REFERENCE / 'prepare_browser.py')], check=True, env=env)
    # Vite does not execute Vercel's asset redirects. Serve the same pinned
    # Supertonic files locally for its full-page runtime/memory check.
    shutil.copytree(APP / '.local/supertonic', root / 'web/public/assets/supertonic', dirs_exist_ok=True)
    # The old harness already allows these three 96-channel names. Map them
    # explicitly rather than edit its shaders or production model selection.
    aliases = {'control': 'control', 'phoneme': 'data', 'features': 'folded', 'replay': 'replay'}
    for arm, alias in aliases.items():
        run = 'control-replay' if arm == 'replay' else arm
        shutil.copy2(OUT / 'runs' / (run + '-s37') / 'model.onnx', root / (alias + '.onnx'))
    config = root / 'vite.config.mjs'
    text = config.read_text()
    before = "'contextwide','data','folded'];"
    if text.count(before) != 1:
        raise ValueError('Reference browser model allowlist changed')
    config.write_text(text.replace(before, "'contextwide','data','folded','replay'];"))
    benchmark = (REFERENCE / 'browser_bench.mjs').read_text()
    old = "const base=process.env.BENCH_URL;"
    benchmark = benchmark.replace(old, old + "\nconst benchmarkEngine=process.env.BENCH_TTS??'phonon';\nif(!['phonon','supertonic'].includes(benchmarkEngine))throw Error('Unknown benchmark TTS');")
    benchmark = benchmark.replace('const variant=`${candidate}-${length}-${repeat}`;',
                                  'const variant=`${candidate}-${benchmarkEngine}-${length}-${repeat}`;')
    needle = "await mark('idle');"
    benchmark = benchmark.replace(needle, needle + "\n      await page.locator('input[name=engine][value='+benchmarkEngine+']').check();")
    benchmark = benchmark.replace("document.querySelector('#status').textContent==='Phonon is ready.'",
                                  "document.querySelector('#status').textContent===(document.querySelector('input[name=engine]:checked').value==='phonon'?'Phonon is ready.':'Supertonic 3 is ready.')")
    benchmark = benchmark.replace("assert.equal(await page.locator('#status').textContent(),'Phonon is ready.');",
                                  "assert.equal(await page.locator('#status').textContent(),benchmarkEngine==='phonon'?'Phonon is ready.':'Supertonic 3 is ready.');")
    benchmark = benchmark.replace('const summary={candidate,length,repeat,', 'const summary={candidate,engine:benchmarkEngine,length,repeat,')
    (root / 'browser_bench.mjs').write_text(benchmark)
    (root / 'aliases.json').write_text(json.dumps(aliases, indent=2))
    (root / 'benchmark-provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256((REFERENCE / 'browser_bench.mjs').read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(benchmark.encode()).hexdigest()), indent=2))
    subprocess.run(['clang', '-O2', str(REFERENCE / 'sample.c'), '-o', str(root / 'sample')], check=True)
    print('Prepared aliases:', aliases, '\nSet BENCH_TTS=phonon or supertonic; benchmark after training stops.')


if __name__ == '__main__':
    main()
