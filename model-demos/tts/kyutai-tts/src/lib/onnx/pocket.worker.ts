import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm';
import wasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import wasmModule from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import { SentencePieceProcessor } from '@sctg/sentencepiece-js';

const revision = '9b874ef8e0be4a95a352a0fe0455e064e75404e0';
const base = `https://huggingface.co/maxime/pocket-tts-18m-onnx/resolve/${revision}/english_18m_refl_2026-10/`;
env.wasm.numThreads = 1;
env.wasm.wasmPaths = {
	wasm: new URL(wasm, self.location.origin).href,
	mjs: new URL(wasmModule, self.location.origin).href
};

type StateSpec = { name: string; output: string; shape: number[]; dtype: 'int64' | 'float32' };
type Bundle = {
	sample_rate: number;
	frame_rate: number;
	frame_size: number;
	dims: { latent: number; text_embedding: number };
	temperature: number;
	max_gen: { tokens_per_second: number; padding_seconds: number };
	eos: {
		threshold: number;
		min_frames: number;
		frames_after_eos: { short: number; long: number; max_short_words: number };
	};
	graphs: Record<'flow_lm_main' | 'mimi_decoder', { states: StateSpec[] }>;
	voices: Record<string, { start: number; end: number }>;
};

async function asset(name: string) {
	const url = base + name;
	const cache = await caches.open('pocket-tts-onnx-' + revision).catch(() => undefined);
	const cached = await cache?.match(url);
	if (cached) return cached.arrayBuffer();
	const response = await fetch(url, { credentials: 'omit', signal: AbortSignal.timeout(120_000) });
	if (!response.ok) throw new Error(`Could not download ${name} (HTTP ${response.status}).`);
	const copy = cache ? response.clone() : undefined;
	const bytes = await response.arrayBuffer();
	if (copy) await cache?.put(url, copy).catch(() => {});
	return bytes;
}

const sessions = new Map<string, Promise<InferenceSession>>();
function graph(name: string) {
	let session = sessions.get(name);
	if (!session) {
		session = asset(name + '.onnx')
			.then((bytes) => InferenceSession.create(bytes, { executionProviders: ['wasm'] }))
			.catch((error) => { sessions.delete(name); throw error; });
		sessions.set(name, session);
	}
	return session;
}

async function load() {
	const [manifest, tokenizerBytes, main, flow, decoder, conditioner] = await Promise.all([
		asset('bundle.json'),
		asset('tokenizer.model'),
		graph('flow_lm_main_int8'),
		graph('flow_lm_flow'),
		graph('mimi_decoder'),
		graph('text_conditioner')
	]);
	const bundle = JSON.parse(new TextDecoder().decode(manifest)) as Bundle;
	const tokenizer = new SentencePieceProcessor();
	const bytes = new Uint8Array(tokenizerBytes);
	let binary = '';
	for (const byte of bytes) binary += String.fromCharCode(byte);
	await tokenizer.loadFromB64StringModel(btoa(binary));
	return { bundle, tokenizer, main, flow, decoder, conditioner };
}

let runtime: ReturnType<typeof load> | undefined;
let voices: Promise<ArrayBuffer> | undefined;

function presetState(data: ArrayBuffer, capacity: number, specs: StateSpec[]) {
	const view = new DataView(data);
	let offset = 0;
	const u16 = () => { const value = view.getUint16(offset, true); offset += 2; return value; };
	const u32 = () => { const value = view.getUint32(offset, true); offset += 4; return value; };
	const readString = () => {
		const size = u16();
		const value = new TextDecoder().decode(new Uint8Array(data, offset, size));
		offset += size;
		return value;
	};
	readString(); // Preset name
	const count = u16();
	const state: Record<string, Tensor> = {};
	for (let index = 0; index < count; index++) {
		const name = readString();
		const dtype = view.getUint8(offset++);
		const rank = view.getUint8(offset++);
		const shape = Array.from({ length: rank }, u32);
		const size = u32();
		const bytes = data.slice(offset, offset + size);
		offset += size;
		if (dtype === 1) {
			state[name] = new Tensor('int64', new BigInt64Array(bytes), shape);
			continue;
		}
		const values = new Float32Array(bytes);
		// Each key/value plane has its own frame axis.
		const plane = values.length / 2;
		const paddedPlane = capacity * shape[3] * shape[4];
		const padded = new Float32Array(2 * paddedPlane);
		padded.set(values.subarray(0, plane));
		padded.set(values.subarray(plane), paddedPlane);
		state[name] = new Tensor('float32', padded, [shape[0], shape[1], capacity, shape[3], shape[4]]);
	}
	if (specs.some((spec) => !state[spec.name])) throw new Error('Incomplete ONNX voice state.');
	return state;
}

function zeroStates(specs: StateSpec[], capacity = 0) {
	return Object.fromEntries(specs.map((spec) => {
		const shape = spec.shape.map((dim) => dim === -1 ? capacity : dim);
		const size = shape.reduce((product, dim) => product * dim, 1);
		const values = spec.dtype === 'int64' ? new BigInt64Array(size) : new Float32Array(size);
		return [spec.name, new Tensor(spec.dtype, values, shape)];
	}));
}

function advance(specs: StateSpec[], state: Record<string, Tensor>, output: Record<string, Tensor>) {
	for (const spec of specs) {
		state[spec.name].dispose();
		state[spec.name] = output[spec.output];
	}
}

type Input = { text: string; voice: string; reference?: Float32Array };

async function generate(input: Input) {
	const started = performance.now();
	const cold = !runtime;
	runtime ??= load().catch((error) => { runtime = undefined; throw error; });
	const { bundle, tokenizer, main, flow, decoder, conditioner } = await runtime;
	const { sample_rate: sampleRate, dims, eos } = bundle;
	const mainSpecs = bundle.graphs.flow_lm_main.states;
	const decoderSpecs = bundle.graphs.mimi_decoder.states;
	const text = input.text.replace(/\s+/g, ' ').replace(/;/g, ',');
	const normalized = text[0].toUpperCase() + text.slice(1) + (/[.!?…]$/.test(text) ? '' : '.');
	const ids = tokenizer.encodeIds(normalized);
	if (!ids.length) throw new Error('Enter text to speak.');
	const maxFrames = Math.ceil((ids.length / bundle.max_gen.tokens_per_second
		+ bundle.max_gen.padding_seconds) * bundle.frame_rate);
	const emptySequence = new Tensor('float32', [], [1, 0, dims.latent]);
	const emptyText = new Tensor('float32', [], [1, 0, dims.text_embedding]);
	let state: Record<string, Tensor>;

	if (input.reference) {
		const [encoder, bos] = await Promise.all([graph('mimi_encoder'), asset('bos_before_voice.npy')]);
		const padded = new Float32Array(Math.ceil(input.reference.length / bundle.frame_size) * bundle.frame_size);
		padded.set(input.reference);
		const { conditioning } = await encoder.run({ audio: new Tensor('float32', padded, [1, 1, padded.length]) });
		const voiceFrames = conditioning.dims[1] + 1;
		const voice = new Float32Array(voiceFrames * dims.text_embedding);
		// The pinned NPY contains one little-endian float32 embedding after its header.
		voice.set(new Float32Array(bos, bos.byteLength - dims.text_embedding * 4));
		voice.set(conditioning.data as Float32Array, dims.text_embedding);
		conditioning.dispose();
		state = zeroStates(mainSpecs, voiceFrames + ids.length + maxFrames);
		const output = await main.run({
			sequence: emptySequence,
			text_embeddings: new Tensor('float32', voice, [1, voiceFrames, dims.text_embedding]),
			...state
		});
		advance(mainSpecs, state, output);
		output.conditioning.dispose();
		output.eos_logit.dispose();
	} else {
		const range = bundle.voices[input.voice];
		if (!range) throw new Error('Choose a supported ONNX voice.');
		voices ??= asset('voices.bin').catch((error) => { voices = undefined; throw error; });
		const data = (await voices).slice(range.start, range.end);
		// Each preset packs exactly its active frames, with 8 key/value caches.
		const view = new DataView(data);
		const nameEnd = 2 + view.getUint16(0, true);
		const firstTensor = nameEnd + 2;
		const firstShape = firstTensor + 2 + view.getUint16(firstTensor, true);
		const voiceFrames = Number(view.getBigInt64(firstShape + 2 + 4 + 4, true));
		state = presetState(data, voiceFrames + ids.length + maxFrames, mainSpecs);
	}

	const decoderState = zeroStates(decoderSpecs);
	const setup = performance.now() - started;
	const generationStarted = performance.now();
	const fadeFrames = Math.round(sampleRate * 0.005);
	try {
		const { embeddings } = await conditioner.run({
			token_ids: new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length])
		});
		const prompt = await main.run({ sequence: emptySequence, text_embeddings: embeddings, ...state });
		advance(mainSpecs, state, prompt);
		prompt.conditioning.dispose();
		prompt.eos_logit.dispose();
		embeddings.dispose();

		let latent = new Float32Array(dims.latent).fill(NaN);
		let eosFrame = Infinity;
		const tail = eos.frames_after_eos;
		const extra = text.split(' ').length <= tail.max_short_words ? tail.short : tail.long;
		for (let frame = 0; frame < maxFrames; frame++) {
			const output = await main.run({
				sequence: new Tensor('float32', latent, [1, 1, dims.latent]), text_embeddings: emptyText, ...state
			});
			advance(mainSpecs, state, output);
			if (frame >= eos.min_frames && Number(output.eos_logit.data[0]) > eos.threshold) {
				eosFrame = Math.min(eosFrame, frame);
			}
			output.eos_logit.dispose();
			if (frame >= eosFrame + extra) {
				output.conditioning.dispose();
				break;
			}
			const noise = Float32Array.from({ length: dims.latent }, () =>
				Math.sqrt(-2 * Math.log(1 - Math.random()) * bundle.temperature) * Math.cos(2 * Math.PI * Math.random())
			);
			const flowOutput = await flow.run({ c: output.conditioning, x: new Tensor('float32', noise, [1, dims.latent]) });
			output.conditioning.dispose();
			latent = new Float32Array(flowOutput.latent.data as Float32Array);
			flowOutput.latent.dispose();
			const decoded = await decoder.run({ latent: new Tensor('float32', latent, [1, 1, dims.latent]), ...decoderState });
			advance(decoderSpecs, decoderState, decoded);
			const pcm = new Float32Array(decoded.audio.data as Float32Array);
			decoded.audio.dispose();
			if (!frame) for (let i = 0; i < fadeFrames; i++) pcm[i] *= i / fadeFrames;
			self.postMessage({ type: 'audio', pcm, sampleRate }, { transfer: [pcm.buffer] });
		}
	} finally {
		for (const tensor of [...Object.values(state), ...Object.values(decoderState), emptySequence, emptyText]) tensor.dispose();
	}
	self.postMessage({ type: 'done', setup, generation: performance.now() - generationStarted, cold });
}

self.onmessage = ({ data }: MessageEvent<Input>) => {
	void generate(data).catch((error) => self.postMessage({
		type: 'error', detail: error instanceof Error ? error.message : String(error)
	}));
};
