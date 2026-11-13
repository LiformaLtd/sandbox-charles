import * as ort from 'onnxruntime-web/webgpu';
import { AutoTokenizer, env } from '@huggingface/transformers';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import wasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';
import gpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm?url';
import gpuModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jspi.mjs?url';
import asyncWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import asyncModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import { onnxBase, onnxRepository, onnxRevision, type OnnxVersion, type OnnxDevice } from './onnx-options';

env.allowLocalModels = false;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl };

type Progress = (message: string) => void;
type Feeds = Record<string, ort.Tensor>;

async function asset(path: string, progress: Progress): Promise<Uint8Array> {
	const url = onnxBase + path;
	const cache = typeof caches !== 'undefined' ? await caches.open('phonon-onnx-v1').catch(() => undefined) : undefined;
	const cached = await cache?.match(url);
	if (cached) return new Uint8Array(await cached.arrayBuffer());
	progress(`Downloading ${path.split('/').pop()}…`);
	const response = await fetch(url);
	if (!response.ok) throw new Error(`Could not download ${path}: HTTP ${response.status}.`);
	const reader = response.body?.getReader();
	if (!reader) throw new Error(`Empty download for ${path}.`);
	const parts: Uint8Array[] = [];
	const total = Number(response.headers.get('content-length'));
	let loaded = 0;
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		parts.push(value); loaded += value.length;
		progress(`Downloading ${path.split('/').pop()} · ${total ? `${Math.min(100, Math.round(loaded / total * 100))}%` : `${(loaded / 1e6).toFixed(1)} MB`}`);
	}
	const bytes = new Uint8Array(loaded);
	let offset = 0;
	for (const part of parts) { bytes.set(part, offset); offset += part.length; }
	// Cache quota/private browsing failures must not prevent generation.
	await cache?.put(url, new Response(bytes)).catch(() => {});
	return bytes;
}

const ints = (data: number[], dims: number[]) => new ort.Tensor('int64', BigInt64Array.from(data, BigInt), dims);
const floats = (dims: number[], data?: Float32Array) => new ort.Tensor('float32', data ?? new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
function dispose(feeds: Feeds) { for (const tensor of new Set(Object.values(feeds))) tensor.dispose(); }

// This pinned export generates eight latent frames per call. Input text and voice
// are separate: the graph prepends the voice and BOS itself. Do not add chat tokens.
export class OnnxPhonon {
	private model!: ort.InferenceSession;
	private decoder!: ort.InferenceSession;
	private tokenizer!: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
	private voices = new Map<string, Float32Array>();
	private halfCache = false;

	static async load(version: OnnxVersion, device: OnnxDevice, progress: Progress) {
		if (device === 'webgpu') {
			const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu;
			const adapter = await gpu?.requestAdapter();
			if (!adapter) throw new Error('WebGPU is unavailable. Choose Browser CPU (WASM).');
			if ((version === 'fp16' || version === 'q4f16') && !adapter.features.has('shader-f16')) {
				throw new Error('This GPU does not support FP16. Choose Browser CPU (WASM) or another ONNX version.');
			}
			// ORT 1.31's native WebGPU provider needs its JSPI/Asyncify binary,
			// rather than the older JSEP module used by the CPU path.
			const jspi = 'Suspending' in WebAssembly;
			ort.env.wasm.wasmPaths = { wasm: jspi ? gpuWasmUrl : asyncWasmUrl, mjs: jspi ? gpuModuleUrl : asyncModuleUrl };
		}
		const engine = new OnnxPhonon();
		engine.halfCache = version === 'fp16' || version === 'q4f16';
		try {
			engine.tokenizer = await AutoTokenizer.from_pretrained(onnxRepository, { revision: onnxRevision });
			const suffix = version === 'fp32' ? '' : `_${version}`;
			for (const name of ['model', 'mimi_decoder'] as const) {
				const file = `${name}${suffix}.onnx`;
				const graph = await asset(`onnx/${file}`, progress);
				const weights = await asset(`onnx/${file}_data`, progress);
				progress(`Preparing ${file} on ${device === 'wasm' ? 'CPU' : 'GPU'}…`);
				const gpuFeedback: Record<string, 'gpu-buffer'> = {};
				if (device === 'webgpu' && name === 'model') {
					gpuFeedback.audio_latents = gpuFeedback['present.audio_latents'] = 'gpu-buffer';
					for (let layer = 0; layer < 6; layer++) for (const kind of ['key', 'value']) {
						gpuFeedback[`present.${layer}.${kind}`] = 'gpu-buffer';
					}
				}
				engine[name === 'model' ? 'model' : 'decoder'] = await ort.InferenceSession.create(graph, {
					executionProviders: [device], externalData: [{ path: `${file}_data`, data: weights }],
					...(device === 'webgpu' && name === 'model' ? { preferredOutputLocation: gpuFeedback } : {})
				});
			}
			return engine;
		} catch (error) { await engine.release(); throw error; }
	}

	async voice(file: string, progress: Progress) {
		let voice = this.voices.get(file);
		if (!voice) {
			const raw = new Float32Array((await asset(file, progress)).buffer);
			if (!raw.length || raw.length % 512 || !raw.every(Number.isFinite)) throw new Error('Invalid ONNX voice data.');
			const frames = raw.length / 512;
			voice = new Float32Array(raw.length);
			// Voice files are [1, 512, frames]; the generator expects [1, frames, 512].
			for (let f = 0; f < frames; f++) for (let c = 0; c < 512; c++) voice[f * 512 + c] = raw[c * frames + f];
			this.voices.set(file, voice);
		}
		return voice;
	}

	async *speak(text: string, voice: Float32Array) {
		// Sentence/clause chunks limit context and permit playback before the whole
		// request finishes. Each chunk gets a fresh cache, as in the reference runtime.
		const sentences = Array.from(new Intl.Segmenter('en', { granularity: 'sentence' }).segment(text), item => item.segment);
		const chunks: string[] = [];
		for (const sentence of sentences) {
			const words = sentence.trim().split(/\s+/u);
			for (let i = 0; i < words.length; i += 40) chunks.push(words.slice(i, i + 40).join(' '));
		}
		for (const chunk of chunks.filter(Boolean)) yield* this.sentence(chunk, voice);
	}

	private async *sentence(text: string, voice: Float32Array): AsyncGenerator<Float32Array> {
		const encoded = this.tokenizer(text, { add_special_tokens: false });
		const ids = Array.from(encoded.input_ids.data, Number);
		const voiceFrames = voice.length / 512;
		let cache: Feeds = { 'past_key_values.audio_latents': floats([1, 0, 32]) };
		for (let layer = 0; layer < 6; layer++) for (const kind of ['key', 'value']) {
			cache[`past_key_values.${layer}.${kind}`] = this.halfCache
				? new ort.Tensor('float16', new Uint16Array(0), [1, 8, 0, 64]) : floats([1, 8, 0, 64]);
		}
		const maxFrames = Math.ceil((ids.length / 3 + 2) * 12.5);
		const tail = text.trim().split(/\s+/u).length <= 4 ? 3 : 1;
		let frames = 0, emittedSamples = 0, stopFrame: number | undefined;
		let latents: ort.Tensor | undefined;
		try {
			while (frames < Math.min(stopFrame ?? maxFrames, maxFrames)) {
				const first = frames === 0;
				const past = cache['past_key_values.0.key'].dims[2];
				const newLength = (first ? ids.length + voiceFrames : 0) + 8;
				const noise = new Float32Array(8 * 32);
				for (let i = 0; i < noise.length; i += 2) {
					const r = Math.sqrt(-2 * Math.log(Math.max(Number.EPSILON, Math.random())) * 0.3);
					const angle = 2 * Math.PI * Math.random();
					noise[i] = r * Math.cos(angle); noise[i + 1] = r * Math.sin(angle);
				}
				const inputs: Feeds = {
					input_ids: ints(first ? ids : [], [1, first ? ids.length : 0]),
					audio_prompt_latents: floats([1, first ? voiceFrames : 0, 512], first ? voice : undefined),
					attention_mask: ints(Array(past + newLength).fill(1), [1, past + newLength]),
					position_ids: ints(Array.from({ length: newLength }, (_, i) => past + i), [1, newLength]),
					noise: floats([1, 8, 32], noise)
				};
				let outputs: Feeds;
				try { outputs = await this.model.run({ ...inputs, ...cache }); }
				finally { dispose(inputs); }
				dispose(cache);
				cache = {};
				latents?.dispose(); latents = outputs.audio_latents;
				if (stopFrame === undefined) {
					const eos = Array.from(outputs.eos_logits.data as Float32Array).findIndex(value => value > 4);
					if (eos !== -1) stopFrame = frames + eos + tail + 1;
				}
				for (const [name, tensor] of Object.entries(outputs)) {
					if (name.startsWith('present.')) cache[name.replace('present.', 'past_key_values.')] = tensor;
					else if (name !== 'audio_latents') tensor.dispose();
				}
				frames += 8;
				// The decoder is causal but exposes no streaming cache. Decode the
				// accumulated prefix and send only its new samples, instead of waiting
				// for sentence EOS. Each generator block represents 640 ms of audio.
				const numFrames = ints([Math.min(frames, stopFrame ?? maxFrames, maxFrames)], [1]);
				let decoded: Feeds;
				try { decoded = await this.decoder.run({ audio_latents: latents!, num_frames: numFrames }); }
				finally { numFrames.dispose(); }
				try {
					const all = decoded.audio_values.data as Float32Array;
					const pcm = Float32Array.from(all.subarray(emittedSamples));
					emittedSamples = all.length;
					if (!pcm.length || !pcm.every(Number.isFinite)) throw new Error('ONNX returned invalid audio.');
					yield pcm;
				} finally { dispose(decoded); }
			}
			if (stopFrame === undefined) throw new Error('ONNX generation reached its speech limit. Try a shorter sentence.');
		} finally { dispose(cache); latents?.dispose(); }
	}

	async release() {
		await this.model?.release(); await this.decoder?.release(); this.voices.clear();
	}
}
