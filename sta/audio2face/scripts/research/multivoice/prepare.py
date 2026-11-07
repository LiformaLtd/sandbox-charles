"""Freeze a balanced English TTS pilot, pinned assets and optimized aligner sources."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import urllib.request

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2]
OUT = Path(os.environ.get('A2F_MULTIVOICE_OUTPUT', APP / '.local/compact-multivoice')).resolve()
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(re.findall(r"[a-z]+", text.lower().replace('’', "'")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    voice_path = Path(os.environ.get('A2F_MULTIVOICE_VOICES', HERE / 'voices.json')).resolve()
    prompt_path = Path(os.environ.get('A2F_MULTIVOICE_PROMPTS', HERE / 'prompts.json')).resolve()
    voices = json.loads(voice_path.read_text())
    prompts = json.loads(prompt_path.read_text())
    dictionary_root = Path(os.environ.get('A2F_DICT_ROOT', APP.parent / 'audio2face-dict'))
    aligner_root = Path(os.environ.get('A2F_ALIGNER_ROOT', APP.parents[1] / 'model-demos/ttp/tinyaligner-fast'))
    dictionary = json.loads((dictionary_root / 'static/models/lexicon.json').read_text())
    prior = set()
    for name in ['arkit-student', 'arkit-student-v2', 'arkit-student-v3', 'arkit-student-v3/extra']:
        for row in json.loads((WORK / name / 'manifest.json').read_text()):
            prior.add(normalize(row['text']))
    # Include newer research manifests, so evaluation prompts do not duplicate
    # either an old training example or an already inspected test sentence.
    for parent in [APP / '.local/compact-ablation', APP / '.local/compact-followup']:
        for path in parent.glob('*/manifest.json'):
            prior.update(normalize(r['text']) for r in json.loads(path.read_text()))
    pilot = APP / '.local/compact-multivoice/manifest.json'
    if pilot.exists() and pilot.resolve() != (OUT / 'manifest.json').resolve():
        prior.update(normalize(r['text']) for r in json.loads(pilot.read_text()))
    seen = set()
    for split, rows in prompts.items():
        for _, text in rows:
            normalized = normalize(text)
            if normalized in seen or normalized in prior:
                raise ValueError(f'Duplicate or historical prompt: {text}')
            seen.add(normalized)
            unknown = [w for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text)
                       if w.upper() not in dictionary['lexicon']]
            if unknown:
                raise ValueError(f'Unknown words: {unknown}')
    assets = {}
    def record(path):
        path = Path(path).resolve()
        assets[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return str(path)
    def fetch(url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        return record(path)
    config = {'voices': [], 'assets': assets, 'sample_rate': 16000,
              'phonon': {k: record(APP / '.local/phonon' / k)
                         for k in ['config.json', 'tokenizer.json', 'model.q8.gguf']},
              'supertonic_root': str((APP / '.local/supertonic').resolve())}
    for path in (APP / '.local/supertonic/onnx').iterdir():
        record(path)
    for voice in voices['voices']:
        v = dict(voice)
        if v['engine'] == 'phonon':
            url = f"https://huggingface.co/gradium/phonon-voices/resolve/{voices['phonon_voices_revision']}/voices/{v['id']}.safetensors"
            v['asset'] = fetch(url, OUT / 'assets/phonon' / (v['name'] + '.safetensors'))
        else:
            url = f"https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/{voices['supertonic_revision']}/voice_styles/{v['name']}.json"
            v['asset'] = fetch(url, OUT / 'assets/supertonic' / (v['name'] + '.json'))
        config['voices'].append(v)
    # Snapshot the current optimized frontend/CTC, not another task's mutable files.
    target = OUT / 'sources/aligner'
    target.mkdir(parents=True, exist_ok=True)
    for path in (aligner_root / 'src/lib/alignment').glob('*.ts'):
        text = path.read_text()
        text = re.sub(r"(from\s+['\"])(\./[^'\"]+)(['\"])",
                      lambda m: m[1] + m[2] + ('' if m[2].endswith('.ts') else '.ts') + m[3], text)
        destination = target / path.name
        if destination.exists() and destination.read_text() != text:
            raise ValueError(f'Aligner source changed; use a new study directory: {path}')
        destination.write_text(text)
        record(destination)
    for name in ['model.onnx', 'lexicon.json']:
        destination = OUT / 'assets/aligner' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(dictionary_root / 'static/models' / name, destination)
        record(destination)
    for source, name in [(dictionary_root / 'scripts/label.py', 'label.py'),
                         (APP / 'src/vendor/supertonic.js', 'supertonic.js'),
                         (APP / 'src/pcm.js', 'pcm.js'),
                         (APP / 'scripts/research/compact_ablation/model.py', 'model.py')]:
        destination = OUT / 'sources' / name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f'Source changed; use a new study directory: {source}')
        shutil.copy2(source, destination)
        record(destination)
    rows = []
    for v in config['voices']:
        for i, (kind, text) in enumerate(prompts[v['split']]):
            rows.append(dict(stem=f"{v['split']}-{v['engine']}-{v['name'].lower()}-{i:03}",
                             split=v['split'], engine=v['engine'], voice=v['name'], gender=v['gender'],
                             locale=v['locale'], text=text, kind=kind, temperature=.7,
                             supertonic_steps=8, supertonic_speed=1.05))
    config['definition_sha256'] = hashlib.sha256(voice_path.read_bytes() + prompt_path.read_bytes()).hexdigest()
    for name, value in [('assets.json', config), ('planned.json', rows)]:
        path = OUT / name
        text = json.dumps(value, indent=2) + '\n'
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Frozen definition changed: {path}')
        path.write_text(text)
    print(json.dumps({'clips': len(rows), 'voices': len(config['voices']), 'output': str(OUT)}))


if __name__ == '__main__':
    main()
