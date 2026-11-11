import { makeWav, playChunk } from './audio';

export async function runLocalPhonon(
 input: { backend: 'native' | 'mlx'; text: string; voice: string },
 audio: AudioContext, signal: AbortSignal,
 onSetup: (ms: number) => void, onFirst: () => void
) {
 const response = await fetch('/api/phonon', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input), signal });
 if (!response.ok || !response.body) {
  const body = await response.json().catch(() => ({}));
  throw new Error(body.detail || `Phonon request failed (${response.status}).`);
 }
 const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
 const chunks: ArrayBuffer[] = [];
 const playback = { next: 0 };
 let pending = '', samples = 0, sampleRate = 0;
 let setup: number | undefined, generation: number | undefined;
 let complete = false;
 try {
  while (true) {
   const { value, done } = await reader.read(); if (done) break;
   const lines = (pending + value).split('\n'); pending = lines.pop() || '';
   for (const line of lines) {
    if (!line) continue;
    signal.throwIfAborted();
    const packet = JSON.parse(line) as { error?: string; setupMs?: number; sampleRate?: number; pcm?: string; generationMs?: number; done?: boolean };
    if (packet.error) throw new Error(packet.error);
    if (packet.sampleRate) {
     if (sampleRate && sampleRate !== packet.sampleRate) throw new Error('The audio sample rate changed during generation.');
     sampleRate = packet.sampleRate;
    }
    if (packet.setupMs !== undefined) { setup = packet.setupMs; onSetup(setup); }
    if (packet.pcm) {
     if (!sampleRate) throw new Error('Missing Phonon audio sample rate.');
     const bytes = Uint8Array.from(atob(packet.pcm), c => c.charCodeAt(0));
     const pcm = new Float32Array(bytes.buffer);
     playChunk(audio, playback, pcm, sampleRate);
     if (!samples) onFirst();
     chunks.push(bytes.buffer); samples += pcm.length;
    }
    if (packet.generationMs !== undefined) generation = packet.generationMs;
    if (packet.done) complete = true;
   }
  }
 } finally { await reader.cancel().catch(() => {}); }
 signal.throwIfAborted();
 if (pending || !complete || !samples || setup === undefined || generation === undefined) throw new Error('Phonon audio generation ended before completion.');
 return { audio: makeWav(chunks, samples, sampleRate), setup, generation, duration: samples / sampleRate };
}
