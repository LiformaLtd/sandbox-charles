"""Freeze new transcript coverage with pinned prior synthesis/teacher assets."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import numpy as np

APP = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('A2F_COVERAGE_OUTPUT', APP / '.local/compact-coverage-v2')).resolve()
PILOT = APP / '.local/compact-multivoice'
GAIN = APP / '.local/compact-gain-v2'
WORK = Path(os.environ['A2F_RESEARCH_WORK']).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, data):
    if isinstance(data, (dict, list)):
        data = json.dumps(data, indent=2).encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Frozen input changed: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def normalize(text):
    return ' '.join(re.findall('[a-z]+', text.lower()))


def text_coverage(texts, lexicon, frequencies):
    phones, pairs, triples = set(), set(), set()
    for text in texts:
        seq = [p for word in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*", text) for p in lexicon[word.upper()]]
        phones.update(seq)
        pairs.update(' '.join(p) for p in zip(seq, seq[1:]))
        triples.update(' '.join(p) for p in zip(seq, seq[1:], seq[2:]))
    return dict(distinct_texts=len(set(map(normalize, texts))), phones=len(phones), pairs=len(pairs), triples=len(triples),
                subtitle_occurrence_coverage=sum(r['occurrences'] for r in frequencies if r['pair'] in pairs)
                / sum(r['occurrences'] for r in frequencies))


def main():
    base = OUT / 'base'; base.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((HERE / 'prompts.json').read_text())
    assets = json.loads((PILOT / 'assets.json').read_text())
    lexicon = json.loads((PILOT / 'assets/aligner/lexicon.json').read_text())['lexicon']
    historical, replay_texts = set(), []
    for name in ['arkit-student', 'arkit-student-v2', 'arkit-student-v3', 'arkit-student-v3/extra']:
        for row in json.loads((WORK / name / 'manifest.json').read_text()):
            historical.add(normalize(row['text']))
            if name in ['arkit-student-v3', 'arkit-student-v3/extra'] and row['split'] == 'train':
                replay_texts.append(row['text'])
    for root in [PILOT, APP / '.local/compact-v4-confirmation', APP / '.local/compact-ablation', APP / '.local/compact-followup']:
        paths = [root / 'manifest.json', *root.glob('*/manifest.json')]
        for path in paths:
            if path.exists():
                historical.update(normalize(r['text']) for r in json.loads(path.read_text()))
    seen = set()
    for rows in prompts.values():
        for _, text in rows:
            norm = normalize(text)
            if norm in historical or norm in seen:
                raise ValueError('Duplicate/historical prompt: ' + text)
            seen.add(norm)
            unknown = [w for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)*", text) if w.upper() not in lexicon]
            if unknown:
                raise ValueError(f'Unknown words: {unknown}')
    planned = []
    training = {engine: [v for v in assets['voices'] if v['split'] == 'train' and v['engine'] == engine]
                for engine in ['phonon', 'supertonic']}
    # Catalogue order: Phonon male/male/female/female; pair opposite genders.
    training['supertonic'].sort(key=lambda v: (v['gender'] != 'female', v['name']))
    for split, texts in prompts.items():
        for i, (kind, text) in enumerate(texts):
            voices = [training[e][i % 4] for e in ['phonon', 'supertonic']] if split == 'train' else [v for v in assets['voices'] if v['split'] == split]
            for voice in voices:
                planned.append(dict(stem=f'coverage-{split}-{voice["engine"]}-{voice["name"].lower()}-{i:03}', split=split,
                                    engine=voice['engine'], voice=voice['name'], gender=voice['gender'], locale=voice['locale'],
                                    text=text, kind=kind, temperature=.7, supertonic_steps=8, supertonic_speed=1.05))
    for name in ['label.py', 'model.py', 'mlx_features.py', 'pcm.js', 'supertonic.js']:
        freeze(base / 'sources' / name, (PILOT / 'sources' / name).read_bytes())
    for path in (PILOT / 'sources/aligner').glob('*.ts'):
        freeze(base / 'sources/aligner' / path.name, path.read_bytes())
    target = base / 'assets'
    if not target.exists():
        target.symlink_to(PILOT / 'assets', target_is_directory=True)
    # Alignment rewrites this small file. Give it an owned copy, never a link
    # through which an older frozen study's file could be overwritten.
    freeze(base / 'phone-classes.json', (PILOT / 'phone-classes.json').read_bytes())
    assets['definition_sha256'] = sha(HERE / 'prompts.json')
    freeze(base / 'assets.json', assets); freeze(base / 'planned.json', planned)
    frequencies_path = APP.parent / 'audio2face-dict/.local/subtitle-analysis/pair-frequencies.json'
    frequencies = json.loads(frequencies_path.read_text())
    old_texts = [r[1] for r in json.loads((APP / 'scripts/research/multivoice/prompts.json').read_text())['train']]
    new_texts = [r[1] for r in prompts['train']]
    audit = {name: text_coverage(texts, lexicon, frequencies) for name, texts in [
        ('old_primary', old_texts), ('new_primary', new_texts),
        ('old_with_replay', old_texts + replay_texts), ('new_with_replay', new_texts + replay_texts)]}
    freeze(OUT / 'text-coverage.json', audit)
    baseline = GAIN / 'augmented/runs/control-replay-s37/model.onnx'
    if sha(baseline) != '84da9bfe22d221075d85e8a52622fc7d31b18d12b04905e0902a7ec64eb5ab2f':
        raise ValueError('Wrong published-v4 initialization')
    sources = [Path(__file__), HERE / 'prompts.json', APP / 'docs/plans/compact-coverage.md',
               *[APP / 'scripts/research/multivoice' / n for n in ['train.py', 'label.py', 'align.mjs', 'synthesize.mjs']]]
    protocol = dict(initialization=str(baseline), initialization_sha256=sha(baseline), seed=37, epochs=24, steps_per_epoch=31,
                    primary='new-original RMS vs matched control; >=5% improvement; no engine or contact regression',
                    primary_training_base_clips=256, new_training_texts=128, primary_conditions=['original', 'half', 'boost'],
                    extra_test_condition='quarter', test_base_clips=96, planned_sha256=sha(base / 'planned.json'),
                    subtitle_frequency_sha256=sha(frequencies_path), source_sha256={str(p.relative_to(APP)): sha(p) for p in sources})
    freeze(OUT / 'protocol.json', protocol)
    for source in sources:
        freeze(OUT / 'sources/frozen' / source.relative_to(APP), source.read_bytes())
    print(json.dumps(dict(planned_recordings=len(planned), text_coverage=audit), indent=2))


if __name__ == '__main__':
    main()
