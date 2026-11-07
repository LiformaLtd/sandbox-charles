import { copyFile, mkdir, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('../', import.meta.url));
const studio = resolve(process.env.JAMES_STUDIO_DIR ?? fileURLToPath(new URL('../../../audio2face/', import.meta.url)));
const aligner = fileURLToPath(new URL('../../../model-demos/ttp/tinyaligner/static/models/', import.meta.url));

async function present(path) {
  try { return (await stat(path)).size > 0; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

async function copy(source, target) {
  const destination = resolve(root, 'static', target);
  if (await present(destination)) return;
  if (!await present(source)) throw new Error(`Missing ${source}. Prepare the assets in the Audio2Face studio first.`);
  await copyFile(source, destination);
}

await mkdir(resolve(root, 'static/assets/phonon'), { recursive: true });
await mkdir(resolve(root, 'static/models'), { recursive: true });
for (const name of ['model.onnx', 'lexicon.json']) {
  if (await present(resolve(root, 'static/models', name))) continue;
  const source = resolve(aligner, name);
  if (await present(source)) await copyFile(source, resolve(root, 'static/models', name));
  else {
    const result = spawnSync(process.execPath, ['scripts/fetch-aligner.mjs'], { cwd: root, stdio: 'inherit' });
    if (result.status !== 0) throw new Error('Could not prepare TinyAligner assets.');
    break;
  }
}
for (const [source, target] of [
  ['rig.bin', 'rig.bin'], ['rig.json', 'rig.json'],
  ['arkit-student/james-blendshapes.bin', 'james-blendshapes.bin'],
  ...['model.q8.gguf', 'tokenizer.json', 'config.json', 'easton.safetensors'].map((name) =>
    ['phonon/' + name, 'phonon/' + name]),
]) await copy(resolve(studio, '.local', source), 'assets/' + target);
console.log('James, Phonon, and full TinyAligner assets are ready.');
