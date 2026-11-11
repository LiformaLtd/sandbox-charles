import { runLocalPhonon } from './phonon-server';
import type { PhononTTS } from 'phonon-tts';
import PhononWorkerUrl from './phonon/phonon.worker.ts?worker&url';
import phononWasm from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
import phononSimd from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';
import SupertonicWorker from './supertonic/browser.worker?worker';
import PocketWorker from './pocket/onnx/pocket.worker?worker';
import { runOnnx } from './pocket/onnx/client';
import { receiveAudio } from './pocket/stream-audio';
import { makeWav, playChunk } from './audio';
import { modelById, phononVoiceFiles, type Family, type Mode } from './catalogue';
import type { Packet } from './supertonic/model';
export interface Input { family: Family; model: string; mode: Mode; text: string; voice: string; language: string; speed: number; steps: number; reference?: File }
export interface Result { audio: Blob; setup: number; generation: number; duration: number }
export class Runtimes {
 private phonon?: PhononTTS;
 private phononModel = '';
 private voices = new Set<string>();
 private supertonic?: Worker;
 private pocket?: Worker;
 private destroyed = false;
 private rejectWorker?: (e: Error) => void;
 cancel() {
  this.supertonic?.terminate(); this.supertonic = undefined;
  this.pocket?.terminate(); this.pocket = undefined;
  this.rejectWorker?.(new DOMException('Stopped', 'AbortError')); this.rejectWorker = undefined;
  // A partially loaded Phonon instance is disposed when its load completes.
 }
 dispose() { this.destroyed = true; this.cancel(); this.phonon?.dispose(); this.phonon = undefined; }
 async run(input: Input, audio: AudioContext, signal: AbortSignal, status: (text: string) => void, onSetup: (ms: number) => void, onFirst: () => void): Promise<Result> {
  const started = performance.now();
  if (input.family === 'phonon') {
   if (input.mode === 'native' || input.mode === 'mlx') {
    status(`Loading local ${input.mode === 'mlx' ? 'MLX' : 'native'} Phonon…`);
    return runLocalPhonon({ backend: input.mode, text: input.text, voice: input.voice }, audio, signal, ms => { onSetup(ms); status('Generating speech…'); }, () => { status('Streaming speech…'); onFirst(); });
   }
   const { PhononTTS, concatPcm, encodeWav } = await import('phonon-tts');
   signal.throwIfAborted();
   if (!this.phonon || this.phononModel !== input.model) {
    this.phonon?.dispose(); this.phonon = undefined; this.voices.clear();
    status('Loading Phonon model…');
    try {
     const loaded = await PhononTTS.load({ model: modelById('phonon', input.model).checkpoint, lang: 'en', quant: 'q8', device: 'cpu', threads: 1, workerUrl: PhononWorkerUrl, wasmUrl: phononWasm, simdWasmUrl: phononSimd,
      onProgress: p => { if (!signal.aborted && !p.cached) status(`Loading ${p.file.split('/').pop()}${p.total ? ` · ${Math.round(p.loaded / p.total * 100)}%` : ''}`); } });
     if (this.destroyed || signal.aborted) { loaded.dispose(); signal.throwIfAborted(); throw new Error('The app was closed.'); }
     this.phonon = loaded; this.phononModel = input.model;
    } catch (e) { throw e; }
   }
   signal.throwIfAborted();
   if (!this.voices.has(input.voice)) {
    const url = phononVoiceFiles.get(input.voice); if (!url) throw new Error('Choose an available voice.');
    status(`Loading ${input.voice}…`); await this.phonon.addVoice(input.voice, url); this.voices.add(input.voice);
   }
   signal.throwIfAborted();
   const setup = performance.now() - started; onSetup(setup); status('Streaming speech…');
   const generated = performance.now();
   const stream = this.phonon.stream(input.text, { voice: input.voice, signal });
   const chunks: Float32Array[] = []; const playback = { next: 0 };
   try {
    for await (const pcm of stream) {
     signal.throwIfAborted();
     playChunk(audio, playback, pcm, this.phonon.sampleRate);
     if (!chunks.length) onFirst(); chunks.push(pcm);
    }
   } finally { if (signal.aborted) await stream.done.catch(() => {}); }
   const stats = await stream.done; signal.throwIfAborted();
   if (!chunks.length) throw new Error('Phonon returned no audio.');
   const pcm = concatPcm(chunks);
   return { audio: encodeWav(pcm, this.phonon.sampleRate), setup, generation: stats.totalMs ?? performance.now() - generated, duration: pcm.length / this.phonon.sampleRate };
  }
  if (input.family === 'pocket') {
   if (input.mode === 'browser') {
    status('Loading or generating Pocket ONNX…'); this.pocket ??= new PocketWorker();
    const result = await runOnnx(this.pocket, { text: input.text, voice: input.voice }, input.reference, audio, signal, onFirst);
    onSetup(result.setup); return result;
   }
   status('Loading or generating Pocket TTS…');
   let body: FormData | string;
   if (input.reference) {
    body = new FormData(); body.set('reference', input.reference); body.set('model', input.model); body.set('text', input.text);
   } else body = JSON.stringify({ model: input.model, voice: input.voice, text: input.text });
   const response = await fetch('/api/pocket?stream=true', { method: 'POST', headers: input.reference ? undefined : { 'Content-Type': 'application/json' }, body, signal });
   if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(error.detail || `Pocket inference failed (${response.status}).`); }
   const setup = Number(response.headers.get('x-setup-ms')); onSetup(setup); status('Streaming speech…');
   return { ...await receiveAudio(response, audio, started, { next: 0 }, onFirst), setup };
  }
  const chunks: ArrayBuffer[] = []; let samples = 0; let rate = 0; let setup = 0; let generation: number | undefined; let completed = false;
  const playback = { next: 0 };
  const accept = (p: Packet) => {
   signal.throwIfAborted(); if (p.error) throw new Error(p.error);
   if (p.status) status(p.status);
   if (p.setup !== undefined) { setup = p.setup; onSetup(setup); }
   if (p.generation !== undefined) generation = p.generation;
   if (p.done) completed = true;
   if (p.pcm) {
    if (!p.sampleRate) throw new Error('Missing sample rate.'); rate = p.sampleRate;
    playChunk(audio, playback, p.pcm, rate); if (!samples) onFirst();
    chunks.push(p.pcm.buffer); samples += p.pcm.length;
   }
  };
  const settings = { text: input.text, voice: input.voice, language: input.language, speed: input.speed, steps: input.steps };
  if (input.mode === 'browser') {
   this.supertonic ??= new SupertonicWorker();
   await new Promise<void>((resolve, reject) => {
    this.rejectWorker = reject;
    this.supertonic!.onerror = e => { this.supertonic?.terminate(); this.supertonic = undefined; reject(new Error(e.message || 'Browser inference failed.')); };
    this.supertonic!.onmessage = ({ data }: MessageEvent<Packet>) => { try { accept(data); if (data.done) resolve(); } catch (e) { reject(e); } };
    this.supertonic!.postMessage(settings);
   }).finally(() => { this.rejectWorker = undefined; });
  } else {
   const response = await fetch('/api/supertonic', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings), signal });
   if (!response.ok || !response.body) throw new Error(`Supertonic request failed (${response.status}).`);
   const reader = response.body.pipeThrough(new TextDecoderStream()).getReader(); let pending = '';
   try {
    while (true) {
     const { value, done } = await reader.read(); if (done) break;
     const lines = (pending + value).split('\n'); pending = lines.pop() || '';
     for (const line of lines) { if (!line) continue; const p = JSON.parse(line); if (p.pcm) p.pcm = new Float32Array(Uint8Array.from(atob(p.pcm), c => c.charCodeAt(0)).buffer); accept(p); }
    }
    if (pending) throw new Error('Incomplete audio stream.');
   } finally { await reader.cancel(); }
  }
  signal.throwIfAborted(); if (!completed || !samples || generation === undefined) throw new Error('Audio generation ended early.');
  return { audio: makeWav(chunks, samples, rate), setup, generation, duration: samples / rate };
 }
}
