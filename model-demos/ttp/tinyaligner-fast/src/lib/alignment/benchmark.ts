import type * as Ort from 'onnxruntime-web/wasm';
import { extractLogMel } from './audio';
import { OUTPUT_FRAME_S, SAMPLE_RATE } from './config';
import { ctcForcedAlign, extractSegments } from './ctc';

export const transcript = 'The quick brown fox jumps over the lazy dog';

interface Options {
  ort: Pick<typeof Ort, 'InferenceSession' | 'Tensor'>;
  model: Uint8Array;
  fixture: Float32Array;
  dictionary: { lexicon: Record<string, string[]>; phone2idx: Record<string, number> };
  threads: number;
  provider?: 'cpu' | 'wasm';
  runs: number;
  reference?: { extractLogMel: typeof extractLogMel; ctcForcedAlign: typeof ctcForcedAlign };
  comparison?: { model: Uint8Array; phone2idx: Record<string, number> };
  baseline?: boolean;
  text?: string;
}

function median(values: number[]) {
  return values.sort((left, right) => left - right)[Math.floor(values.length / 2)];
}

function assertEqual(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string) {
  if (actual.length !== expected.length) throw new Error(`${label} shape differs`);
  for (let index = 0; index < actual.length; index++) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(`${label} differs at ${index}: ${actual[index]} vs ${expected[index]}`);
    }
  }
}

export async function benchmark({
  ort, model, fixture, dictionary, threads, runs, reference, comparison, baseline,
  provider = 'wasm', text = transcript,
}: Options) {
  if (!fixture.length || !Number.isInteger(runs) || runs < 1) throw new Error('Invalid fixture or run count');
  const words = text.toUpperCase().replaceAll('’', "'").match(/[A-Z]+(?:'[A-Z]+)*/g) ?? [];
  const utterance = words.flatMap((word) => {
    const pronunciation = dictionary.lexicon[word];
    if (!pronunciation) throw new Error(`Unknown fixture word: ${word}`);
    return pronunciation;
  });
  if (!utterance.length) throw new Error('Fixture transcript is empty');
  const createSession = (weights: Uint8Array) => ort.InferenceSession.create(weights, {
    executionProviders: [provider],
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
  });
  const started = performance.now();
  const session = await createSession(model);
  const loadMs = performance.now() - started;
  const cases = [];
  try {
    for (const copies of [1, 4, 12]) {
      const audio = new Float32Array(fixture.length * copies);
      for (let copy = 0; copy < copies; copy++) audio.set(fixture, copy * fixture.length);
      const phones = ['sil', ...Array.from({ length: copies }, () => utterance).flat(), 'sil'];
      const samples = [];

      async function measure(
        extract: typeof extractLogMel,
        decode: typeof ctcForcedAlign,
        activeSession = session,
        phone2idx = dictionary.phone2idx,
      ) {
        const ids = phones.map((phone) => phone2idx[phone]);
        const blank = phone2idx['<blank>'];
        if (blank === undefined || ids.some((id) => id === undefined)) {
          throw new Error('Fixture transcript has unknown phones');
        }
        const start = performance.now();
        const features = extract(audio);
        const input = new ort.Tensor('float32', features.mel, [1, features.frames, features.nMels]);
        const preprocessed = performance.now();
        const { log_probs: output } = await activeSession.run({ mel: input });
        const inferred = performance.now();
        const [frames, vocabulary] = output.dims;
        const logits = output.data as Float32Array;
        const path = decode(logits, frames, vocabulary, ids, blank);
        const segments = extractSegments(path, phones, OUTPUT_FRAME_S).filter((segment) => segment.word);
        const aligned = performance.now();
        input.dispose();
        output.dispose();
        if (segments.length !== phones.length) throw new Error('Fixture failed to align all phones');
        return {
          features, logits, path,
          preprocessMs: preprocessed - start,
          inferenceMs: inferred - preprocessed,
          ctcMs: aligned - inferred,
          totalMs: aligned - start,
        };
      }

      const warm = await measure(extractLogMel, ctcForcedAlign);
      if (baseline && !reference) throw new Error('Baseline mode requires reference modules');
      if (reference) {
        const original = await measure(reference.extractLogMel, reference.ctcForcedAlign);
        assertEqual(warm.features.mel, original.features.mel, 'Mel');
        assertEqual(warm.logits, original.logits, 'ONNX output');
        assertEqual(warm.path, original.path, 'CTC path');
      }
      const extract = baseline && reference ? reference.extractLogMel : extractLogMel;
      const decode = baseline && reference ? reference.ctcForcedAlign : ctcForcedAlign;
      for (let run = 0; run < runs; run++) samples.push(await measure(extract, decode).then(
        ({ preprocessMs, inferenceMs, ctcMs, totalMs }) => ({ preprocessMs, inferenceMs, ctcMs, totalMs })
      ));
      const totalMs = median(samples.map((sample) => sample.totalMs));
      cases.push({
        seconds: audio.length / SAMPLE_RATE,
        preprocessMs: median(samples.map((sample) => sample.preprocessMs)),
        inferenceMs: median(samples.map((sample) => sample.inferenceMs)),
        ctcMs: median(samples.map((sample) => sample.ctcMs)),
        totalMs,
        realtime: audio.length / SAMPLE_RATE * 1000 / totalMs,
      });
      if (comparison) {
        const original = await createSession(comparison.model);
        try {
          const compared = await measure(extractLogMel, ctcForcedAlign, original, comparison.phone2idx);
          assertEqual(warm.path, compared.path, 'CTC path against original model');
        } finally {
          await original.release();
        }
      }
    }
  } finally {
    await session.release();
  }
  return {
    loadMs, runs, threads, baseline: Boolean(baseline), exactParity: Boolean(reference),
    originalModelPathParity: Boolean(comparison), cases,
  };
}
