import { prepareTranscript } from '../src/lib/alignment/engine.ts';
import { candidatePrompts, development, pilot, test } from './prompts.mjs';

function combinations(phones, count) {
  return Array.from({ length: Math.max(0, phones.length - count + 1) }, (_, i) =>
    phones.slice(i, i + count).join(' '));
}

export function corpusSelector(dictionary) {
  const counts = new Map();
  const demand = new Map();
  const keyset = (text) => {
    const phones = prepareTranscript(text, dictionary).entries.flatMap((word) => word.phones);
    return [...new Set([...combinations(phones, 2), ...combinations(phones, 3)])];
  };
  const include = (keys) => {
    for (const key of keys) counts.set(key, 1 + (counts.get(key) ?? 0));
  };
  const normal = (text) => text.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
  const used = new Set([...pilot, ...development, ...test].map(normal));
  for (const text of pilot) include(keyset(text));
  for (const text of development) {
    for (const key of keyset(text)) demand.set(key, 1 + (demand.get(key) ?? 0));
  }

  const candidates = [];
  for (const text of candidatePrompts()) {
    if (used.has(normal(text))) continue;
    used.add(normal(text));
    candidates.push({ text, keys: keyset(text) });
  }

  return () => {
    if (!candidates.length) throw new Error('Ran out of corpus prompts before reaching the audio budget');
    let best = 0;
    let bestScore = -Infinity;
    for (const [i, candidate] of candidates.entries()) {
      let score = 0;
      for (const key of candidate.keys) {
        const count = counts.get(key) ?? 0;
        const pair = key.split(' ').length === 2;
        score += ((pair ? 1.5 : 1) + 2 * (demand.get(key) ?? 0)) / (1 + count) ** 2;
      }
      if (score > bestScore) {
        best = i;
        bestScore = score;
      }
    }
    const [candidate] = candidates.splice(best, 1);
    include(candidate.keys);
    return candidate.text;
  };
}
