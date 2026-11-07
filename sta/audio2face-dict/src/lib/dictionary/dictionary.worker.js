import * as ort from 'onnxruntime-web/wasm';
import { align, prepareWords, transcriptWords } from '../alignment/engine.ts';
import { bakeAnimation, planAnimation } from './animate.mjs';
import { validClosureModel } from './closures.mjs';
import { validSelectionModel } from './selection-model.mjs';
import channels from '../studio/channels.json';
import { CHANNELS, FPS, SAMPLE_RATE } from './index.mjs';

ort.env.wasm.numThreads = 1;

async function fetchAsset(path, format) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}. See the setup instructions.`);
  return response[format]();
}

let lexicon;
let session;
let dictionary;

async function loadDictionary() {
  const index = await fetchAsset('/dictionary/index.json', 'json');
  if (index.version !== 1 || index.fps !== FPS ||
      JSON.stringify(index.channels) !== JSON.stringify(channels) ||
      !/^weights-[0-9a-f]{16}\.f32$/.test(index.weightsFile)) {
    throw new Error('Dictionary index is incompatible with the James rig. Rebuild the bundle.');
  }
  if (index.closureModel && (!index.lipRig || !validClosureModel(index.closureModel))) {
    throw new Error('Invalid closure model in the dictionary. Rebuild the bundle.');
  }
  if (index.selectionModel && (!index.lipRig || !validSelectionModel(index.selectionModel))) {
    throw new Error('Invalid selection model in the dictionary. Rebuild the bundle.');
  }
  const weights = await fetchAsset('/dictionary/' + index.weightsFile, 'arrayBuffer');
  if (weights.byteLength !== index.frameCount * CHANNELS * 4) {
    throw new Error('Dictionary animation data is incomplete. Rebuild the bundle.');
  }
  return { ...index, weights: new Float32Array(weights) };
}

self.onmessage = async ({ data: { id, type, text, pcm, recordedPauses = false } }) => {
  try {
    if (type === 'prepare') {
      lexicon ??= await fetchAsset('/models/lexicon.json', 'json');
      const words = transcriptWords(text);
      const needsPrediction = words.some((word) => !Object.hasOwn(lexicon.lexicon, word.toUpperCase()));
      const predict = needsPrediction ? (await import('../alignment/pronunciation.ts')).predictPhones : undefined;
      self.postMessage({ id, value: prepareWords(words, lexicon, predict) });
      return;
    }
    const alignmentStarted = performance.now();
    session ??= await ort.InferenceSession.create('/models/model.onnx', { executionProviders: ['wasm'] });
    const words = await align(session, pcm, SAMPLE_RATE, text);
    const phones = words.flatMap((word) => word.phones.map(({ word: phone, start, end }) => ({ phone, start, end })));
    const alignmentMs = performance.now() - alignmentStarted;
    self.postMessage({ id, status: 'Matching phonemes and blending animation…' });
    const samplingStarted = performance.now();
    dictionary ??= await loadDictionary();
    const started = performance.now();
    const plan = planAnimation(dictionary, phones, pcm, { recordedPauses });
    const frames = bakeAnimation(dictionary, plan);
    self.postMessage({ id, value: {
      frames, phonemeCount: phones.length, coverage: plan.coverage, usage: plan.usage, joins: plan.joins.length,
      alignmentMs, samplingMs: performance.now() - samplingStarted,
      lookupMs: performance.now() - started, corpus: dictionary.corpus,
      pauseUsage: plan.pauseDiagnostics ? { used: plan.pauseDiagnostics.usedPauses, detected: plan.pauseDiagnostics.detectedPauses } : undefined,
    } }, [frames.buffer]);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
