import { readFile } from 'node:fs/promises';
import { registerHooks } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.endsWith('.ts') && /^\.{1,2}\/[^.]+$/.test(specifier)) {
      return nextResolve(specifier + '.ts', context);
    }
    return nextResolve(specifier, context);
  },
});

const { benchmark } = await import('../src/lib/alignment/benchmark.ts');
const native = process.env.NATIVE_ORT;
const runtime = await import(native ? pathToFileURL(resolve(native)).href : 'onnxruntime-web');
const ort = native ? runtime.default : runtime;
const threads = Number(process.env.THREADS ?? 1);
if (!Number.isInteger(threads) || threads < 1) throw new Error('THREADS must be a positive integer');
if (!native) ort.env.wasm.numThreads = threads;
const modelDir = new URL('../static/models/', import.meta.url);
const variant = process.env.BENCHMARK_VARIANT ?? 'full';
if (!['full', 'english'].includes(variant)) throw new Error('Expected full or english benchmark variant');
if (variant === 'english' && process.env.BENCHMARK_MODEL) {
  throw new Error('Use BENCHMARK_VARIANT or BENCHMARK_MODEL for model selection');
}
const [model, dictionary, pcm] = await Promise.all([
  readFile(process.env.BENCHMARK_MODEL ?? new URL('model.onnx', modelDir)),
  readFile(new URL('lexicon.json', modelDir), 'utf8').then(JSON.parse),
  readFile(process.env.BENCHMARK_PCM ?? new URL('benchmark.f32', modelDir)),
]);
const [englishModel, englishPhones] = variant === 'english' ? await Promise.all([
  readFile(new URL('model-english.onnx', modelDir)),
  readFile(new URL('phones-english.json', modelDir), 'utf8').then(JSON.parse),
]) : [];
const fixture = new Float32Array(Uint8Array.from(pcm).buffer);
const referenceDir = process.env.REFERENCE_ALIGNMENT;
const reference = referenceDir ? {
  extractLogMel: (await import(pathToFileURL(resolve(referenceDir, 'audio.ts')).href)).extractLogMel,
  ctcForcedAlign: (await import(pathToFileURL(resolve(referenceDir, 'ctc.ts')).href)).ctcForcedAlign,
} : undefined;
if (reference) {
  const { extractLogMel } = await import('../src/lib/alignment/audio.ts');
  const signals = [
    Float32Array.of(0, 0),
    Float32Array.of(1, -1),
    new Float32Array(513),
    Float32Array.from({ length: 1601 }, (_, index) => index === 399 ? 1 : 0),
  ];
  for (const signal of signals) {
    const actual = extractLogMel(signal);
    const expected = reference.extractLogMel(signal);
    if (actual.frames !== expected.frames ||
        actual.mel.some((value, index) => !Object.is(value, expected.mel[index]))) {
      throw new Error(`Edge-case Mel mismatch at length ${signal.length}`);
    }
  }
}
const workers = [];
const loadWorkers = Number(process.env.LOAD_WORKERS ?? 0);
if (!Number.isInteger(loadWorkers) || loadWorkers < 0) throw new Error('Invalid LOAD_WORKERS count');
try {
  for (let index = 0; index < loadWorkers; index++) {
    const worker = new Worker(`
      const { parentPort, workerData } = require('node:worker_threads');
      async function load() {
        const ort = await import(workerData.runtime);
        ort.env.wasm.numThreads = 1;
        const session = await ort.InferenceSession.create(workerData.model, { executionProviders: ['wasm'] });
        const mel = new ort.Tensor('float32', new Float32Array(300 * 40), [1, 300, 40]);
        (await session.run({ mel })).log_probs.dispose();
        parentPort.postMessage('ready');
        for (;;) {
          (await session.run({ mel })).log_probs.dispose();
        }
      }
      void load();
    `, {
      eval: true,
      workerData: { runtime: import.meta.resolve('onnxruntime-web'), model: Uint8Array.from(model) },
    });
    workers.push(worker);
    await new Promise((ready, fail) => {
      worker.once('error', fail);
      worker.once('message', () => {
        worker.off('error', fail);
        ready();
      });
    });
  }
  const cpu = process.cpuUsage();
  const mainCpu = process.threadCpuUsage?.();
  const result = await benchmark({
    ort, provider: native ? 'cpu' : 'wasm', model: Uint8Array.from(englishModel ?? model), fixture, threads,
    dictionary: englishPhones ? { ...dictionary, phone2idx: englishPhones } : dictionary,
    comparison: englishModel ? { model: Uint8Array.from(model), phone2idx: dictionary.phone2idx } : undefined,
    text: process.env.BENCHMARK_TRANSCRIPT,
    runs: Number(process.env.RUNS ?? 7), reference, baseline: process.env.BASELINE === '1',
  });
  const elapsedCpu = process.cpuUsage(cpu);
  const elapsedMainCpu = mainCpu && process.threadCpuUsage(mainCpu);
  console.log(JSON.stringify({
    runtime: native ? 'native' : 'wasm',
    variant,
    node: process.version,
    loadWorkers: workers.length,
    processCpuMs: (elapsedCpu.user + elapsedCpu.system) / 1000,
    mainThreadCpuMs: elapsedMainCpu && (elapsedMainCpu.user + elapsedMainCpu.system) / 1000,
    maxRssKiB: process.resourceUsage().maxRSS,
    ...result,
  }, null, 2));
} finally {
  await Promise.all(workers.map((worker) => worker.terminate()));
}
