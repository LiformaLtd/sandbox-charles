import { makeWav, playChunk } from './audio';
import type { OnnxVersion, OnnxDevice } from './onnx-options';

export class OnnxClient {
	private worker: Worker | undefined;
	private key = '';
	dispose() { this.worker?.terminate(); this.worker = undefined; this.key = ''; }

	run(input: { version: OnnxVersion; device: OnnxDevice; text: string; voiceFile: string },
		context: AudioContext, signal: AbortSignal,
		onProgress: (message: string) => void, onSetup: (ms: number) => void, onFirst: () => void) {
		signal.throwIfAborted();
		const key = `${input.version}/${input.device}`;
		if (!this.worker || this.key !== key) {
			this.dispose();
			this.worker = new Worker(new URL('./onnx.worker.ts', import.meta.url), { type: 'module' });
			this.key = key;
		}
		const worker = this.worker;
		return new Promise<{ audio: Blob; setup: number; generation: number; duration: number }>((resolve, reject) => {
			const chunks: ArrayBuffer[] = [];
			const playback = { next: 0 };
			let samples = 0, setup = 0;
			const cleanup = () => {
				signal.removeEventListener('abort', abort);
				worker.onmessage = null; worker.onerror = null;
			};
			const fail = (error: Error) => { cleanup(); this.dispose(); reject(error); };
			const abort = () => fail(new DOMException('Speech generation stopped.', 'AbortError'));
			signal.addEventListener('abort', abort, { once: true });
			worker.onerror = event => fail(new Error(event.message || 'The ONNX worker failed.'));
			worker.onmessage = ({ data }) => {
				try {
					if (data.type === 'progress') onProgress(data.message);
					else if (data.type === 'ready') { setup = data.setup; onSetup(setup); }
					else if (data.type === 'audio') {
						const pcm = data.pcm as Float32Array;
						playChunk(context, playback, pcm, 24000);
						if (!samples) onFirst();
						chunks.push(pcm.buffer as ArrayBuffer); samples += pcm.length;
					} else if (data.type === 'done') {
						if (!samples) throw new Error('ONNX returned no audio.');
						cleanup();
						resolve({ audio: makeWav(chunks, samples, 24000), setup, generation: data.generation, duration: samples / 24000 });
					} else if (data.type === 'error') fail(new Error(data.message));
				} catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
			};
			worker.postMessage(input);
		});
	}
}
