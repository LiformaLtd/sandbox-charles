import type { Voice } from './api';

export function recordingFor(voice: Voice, voices: Voice[]): Voice {
  const seen = new Set<string>();
  while (voice.parentId && !seen.has(voice.id)) {
    seen.add(voice.id);
    const parent = voices.find(item => item.id === voice.parentId);
    if (!parent) break;
    voice = parent;
  }
  return voice;
}

export function methodName(voice: Voice): string {
  if (voice.fit?.algorithm === 'linear') return 'Fast linear fit';
  if (voice.fit?.focus === 'pronunciation') return 'Pronunciation fit';
  if (voice.fit) return 'Iterative fit';
  return 'Initial clone';
}

export function versionName(voice: Voice, family: Voice[]): string {
  const method = methodName(voice);
  const peers = family.filter(item => methodName(item) === method);
  const number = peers.findIndex(item => item.id === voice.id) + 1;
  return `${method}${peers.length > 1 ? ` ${number}` : ''}${voice.fit?.iterations ? ` · ${voice.fit.iterations} steps` : ''}`;
}

export function finalError(voice: Voice): number | undefined {
  return voice.fit?.finalError ?? (voice.fit?.algorithm === 'linear' ? undefined : voice.fit?.after.loss);
}

export function formatError(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return 'Unavailable';
  return value !== 0 && value < 0.0001 ? value.toExponential(3) : value.toFixed(4);
}
