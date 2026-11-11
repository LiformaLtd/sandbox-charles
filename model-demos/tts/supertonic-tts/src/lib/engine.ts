import { createRuntime } from './vendor/helper.js';
import { graphs, segments, type Settings, type Packet } from './model';

// The upstream JS classes accept either ONNX Runtime implementation.
export async function loadEngine(ort: any, asset: (path: string) => Promise<Uint8Array<ArrayBuffer>>, status: (text: string) => void, provider: 'cpu' | 'wasm') {
 const json = async (path: string) => JSON.parse(new TextDecoder().decode(await asset(path)));
 const { UnicodeProcessor, TextToSpeech, Style } = createRuntime(ort);
 const config = await json('onnx/tts.json');
 const processor = new UnicodeProcessor(await json('onnx/unicode_indexer.json'));
 const sessions = [];
 try {
  for (const graph of graphs) {
   status(`Loading ${graph.replaceAll('_', ' ')}…`);
   sessions.push(await ort.InferenceSession.create(await asset(`onnx/${graph}.onnx`), { executionProviders: [provider], intraOpNumThreads: 1, interOpNumThreads: 1 }));
  }
 } catch (error) { for (const session of sessions) await session.release(); throw error; }
 const engine = new TextToSpeech(config, processor, ...sessions);
 const styles = new Map<string, any>();
 return {
  async prepare(voice: string) {
   if (!styles.has(voice)) {
    const data = await json(`voice_styles/${voice}.json`);
    const tensor = (key: string) => new ort.Tensor('float32', Float32Array.from(data[key].data.flat(Infinity)), data[key].dims);
    styles.set(voice, new Style(tensor('style_ttl'), tensor('style_dp')));
   }
  },
  async generate(settings: Settings, emit: (packet: Packet) => void, cancelled: () => boolean) {
   const started = performance.now();
   const parts = segments(settings.text, settings.language);
   for (const [index, text] of parts.entries()) {
    if (cancelled()) return;
    emit({ status: `Generating segment ${index + 1} of ${parts.length}…` });
    const result = await engine._infer([text], [settings.language], styles.get(settings.voice), settings.steps, settings.speed);
    if (cancelled()) return;
    const sampleRate = engine.sampleRate;
    const length = Math.min(result.wav.length, Math.floor(result.duration[0] * sampleRate));
    if (length <= 0 || !Number.isFinite(length)) throw new Error('The model returned an invalid audio duration.');
    // Match the upstream 0.3 second gap between segments, with no leading gap.
    const gap = index ? Math.round(sampleRate * .3) : 0;
    const pcm = new Float32Array(length + gap);
    pcm.set(result.wav.slice(0, length), gap);
    emit({ pcm, sampleRate });
   }
   emit({ generation: performance.now() - started, done: true });
  }
 };
}
export type Engine = Awaited<ReturnType<typeof loadEngine>>;
