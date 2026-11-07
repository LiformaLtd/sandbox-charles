import * as ort from 'onnxruntime-web/wasm';
import { extractLogMel, resampleTo16k } from './audio.ts';
import { OUTPUT_FRAME_S } from './config.ts';
import { ctcForcedAlign, extractSegments } from './ctc.ts';
import type { AlignedWord } from './types.ts';

export interface Dictionary {
  lexicon: Record<string, string[]>;
  phone2idx: Record<string, number>;
}

export function transcriptWords(text: string) {
  if (!/^[a-z\s.,!?;:'’“”"()\-–—]+$/i.test(text)) {
    throw new Error('Use English words and punctuation. Spell out numbers.');
  }
  const words = text.replaceAll('’', "'").match(/[a-z]+(?:'[a-z]+)*/gi) ?? [];
  if (!words.length) throw new Error('Enter at least one English word.');
  return words;
}

export function prepareTranscript(text: string, dictionary: Dictionary) {
  return prepareWords(transcriptWords(text), dictionary);
}

export function prepareWords(
  words: string[],
  dictionary: Dictionary,
  predict?: (word: string) => string[],
) {
  const entries = words.map((word) => {
    const phones = dictionary.lexicon[word.toUpperCase()] ?? predict?.(word);
    if (!phones?.length || phones.some((phone) =>
      !/^[A-Z]+$/.test(phone) || !Object.hasOwn(dictionary.phone2idx, phone)
    )) {
      throw new Error(`TinyAligner could not determine a pronunciation for “${word}”. Try a phonetic spelling.`);
    }
    return { word, phones };
  });

  return { entries, phone2idx: dictionary.phone2idx };
}

export async function align(
  session: ort.InferenceSession,
  samples: Float32Array,
  sampleRate: number,
  { entries, phone2idx }: ReturnType<typeof prepareTranscript>
): Promise<AlignedWord[]> {
  const duration = samples.length / sampleRate;
  const features = extractLogMel(await resampleTo16k(samples, sampleRate));
  const phones = ['sil', ...entries.flatMap((entry) => entry.phones), 'sil'];
  const ids = phones.map((phone) => phone2idx[phone]);

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
