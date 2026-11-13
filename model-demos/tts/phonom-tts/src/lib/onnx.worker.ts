import { OnnxPhonon } from './onnx-engine';
import type { OnnxVersion, OnnxDevice } from './onnx-options';

let engine: OnnxPhonon | undefined;
self.onmessage = async ({ data }: MessageEvent<{ version: OnnxVersion; device: OnnxDevice; text: string; voiceFile: string }>) => {
	try {
		const started = performance.now();
		const progress = (message: string) => self.postMessage({ type: 'progress', message });
		engine ??= await OnnxPhonon.load(data.version, data.device, progress);
		const voice = await engine.voice(data.voiceFile, progress);
		self.postMessage({ type: 'ready', setup: performance.now() - started });
		const synthesisStarted = performance.now();
		for await (const pcm of engine.speak(data.text, voice)) {
			self.postMessage({ type: 'audio', pcm }, { transfer: [pcm.buffer] });
		}
		self.postMessage({ type: 'done', generation: performance.now() - synthesisStarted });
	} catch (error) {
		self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
	}
};
