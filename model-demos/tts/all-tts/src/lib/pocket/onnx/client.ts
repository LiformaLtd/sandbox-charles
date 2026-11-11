import { makeWav, playChunk } from '../stream-audio.js';

type WorkerReply =
	| { type: 'audio'; pcm: Float32Array<ArrayBuffer>; sampleRate: number }
	| { type: 'done'; setup: number; generation: number; cold: boolean }
	| { type: 'error'; detail: string };

export async function runOnnx(
	worker: Worker,
	input: { text: string; voice: string },
	reference: File | undefined,
	context: AudioContext,
	signal: AbortSignal,
	onFirstAudio: (ttfa: number) => void
) {
	const started = performance.now();
	let pcm: Float32Array<ArrayBuffer> | undefined;
	if (reference) {
		if (reference.size > 10 * 1024 * 1024) throw new Error('Choose an audio file up to 10 MB.');
		const decoder = new OfflineAudioContext(1, 1, 24000);
		const audio = await decoder.decodeAudioData(await reference.arrayBuffer()).catch(() => {
			throw new Error('This browser could not decode the reference audio.');
		});
		pcm = new Float32Array(Math.min(audio.length, 30 * audio.sampleRate));
		for (let channel = 0; channel < audio.numberOfChannels; channel++) {
			const samples = audio.getChannelData(channel);
			for (let i = 0; i < pcm.length; i++) pcm[i] += samples[i] / audio.numberOfChannels;
		}
	}
	signal.throwIfAborted();
	const decoding = performance.now() - started;
	const chunks: ArrayBuffer[] = [];
	const playback = { next: 0 };
	let samples = 0;
	let sampleRate = 24000;

	return new Promise<{ audio: Blob; setup: number; generation: number; duration: number; load: string }>((resolve, reject) => {
		function cleanup() {
			worker.removeEventListener('message', onMessage);
			worker.removeEventListener('error', onError);
			signal.removeEventListener('abort', onAbort);
		}
		function onAbort() {
			cleanup();
			reject(signal.reason);
		}
		function onError(event: ErrorEvent) {
			cleanup();
			reject(new Error(event.message || 'Could not start the ONNX worker.'));
		}
		function onMessage({ data }: MessageEvent<WorkerReply>) {
			try {
				if (data.type === 'audio') {
					if (!samples) onFirstAudio(performance.now() - started);
					sampleRate = data.sampleRate;
					playChunk(context, playback, data.pcm, sampleRate);
					chunks.push(data.pcm.buffer);
					samples += data.pcm.length;
					return;
				}
				cleanup();
				if (data.type === 'error') throw new Error(data.detail);
				if (!samples) throw new Error('The ONNX model returned no audio.');
				resolve({
					audio: makeWav(chunks, samples, sampleRate),
					setup: data.setup + decoding,
					generation: data.generation,
					duration: samples / sampleRate,
					load: data.cold ? 'cold' : 'warm'
				});
			} catch (error) {
				cleanup();
				reject(error);
			}
		}
		signal.addEventListener('abort', onAbort, { once: true });
		worker.addEventListener('error', onError);
		worker.addEventListener('message', onMessage);
		worker.postMessage({ ...input, reference: pcm }, pcm ? [pcm.buffer] : []);
	});
}
