import * as ort from 'onnxruntime-web/wasm';
import { extractLogMel, resampleTo16k } from './audio';
import { OUTPUT_FRAME_S } from './config';
import { ctcForcedAlign, extractSegments } from './ctc';
import type { AlignedWord } from './types';

interface Dictionary {
  lexicon: Record<string, string[]>;
  phone2idx: Record<string, number>;
}

let dictionary: Dictionary | undefined;
let session: ort.InferenceSession | undefined;

export async function prepareText(text: string) {
  if (!/^[a-z\s.,!?;:'’“”"()\-–—]+$/i.test(text)) {
    throw new Error('Use English words and punctuation. Spell out numbers.');
  }

  if (!dictionary) {
    const response = await fetch('/models/lexicon.json');
    if (!response.ok) throw new Error('Dictionary unavailable. Run npm run assets to install it.');
    dictionary = await response.json() as Dictionary;
  }

  const words = text.replaceAll('’', "'").match(/[a-z]+(?:'[a-z]+)*/gi) ?? [];
  if (!words.length) throw new Error('Enter at least one English word.');
  const entries = words.map((word) => {
    const phones = dictionary!.lexicon[word.toUpperCase()];
    if (!phones) throw new Error(`“${word}” is missing from the pronunciation dictionary. Try another word.`);
    return { word, phones };
  });

  return { entries, phone2idx: dictionary.phone2idx };
}

export async function align(
  samples: Float32Array,
  sampleRate: number,
  { entries, phone2idx }: Awaited<ReturnType<typeof prepareText>>
): Promise<AlignedWord[]> {
  const duration = samples.length / sampleRate;
  const features = extractLogMel(await resampleTo16k(samples, sampleRate));
  const phones = ['sil', ...entries.flatMap((entry) => entry.phones), 'sil'];
  const ids = phones.map((phone) => phone2idx[phone]);

  if (!session) {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create('/models/model.onnx', { executionProviders: ['wasm'] });
  }
  const mel = new ort.Tensor('float32', features.mel, [1, features.frames, features.nMels]);
  const { log_probs } = await session.run({ mel });
  const [frames, vocabulary] = log_probs.dims;
  const requiredFrames = ids.length + ids.slice(1).filter((id, index) => id === ids[index]).length;
  if (frames < requiredFrames) throw new Error('The recording is too short to align this text.');
  const path = ctcForcedAlign(log_probs.data as Float32Array, frames, vocabulary, ids, phone2idx['<blank>']);
  const segments = extractSegments(path, phones, OUTPUT_FRAME_S).filter((segment) => segment.word);
  if (segments.length !== phones.length) throw new Error('Could not align every phoneme.');

  let offset = 1;
  return entries.map(({ word, phones }) => {
    const intervals = segments.slice(offset, offset + phones.length);
    offset += phones.length;
    for (const interval of intervals) interval.end = Math.min(duration, interval.end);
    return { word, start: intervals[0].start, end: intervals[intervals.length - 1].end, phones: intervals };
  });
}
