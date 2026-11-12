export async function receiveAudio(
	response: Response,
	context: AudioContext,
	started: number,
	playback: { next: number },
	onFirstAudio: (ttfa: number) => void
) {
	const sampleRate = Number(response.headers.get('x-sample-rate'));
	if (!response.body || !sampleRate) throw new Error('The inference service did not return an audio stream.');

	const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
	const chunks: ArrayBuffer[] = [];
	let pending = '';
	let samples = 0;
	let generation: number | undefined;

	try {
		while (true) {
			const { value, done } = await reader.read();
			if (done) break;
			const lines = (pending + value).split('\n');
			pending = lines.pop() ?? '';
			for (const line of lines) {
				const packet = JSON.parse(line) as { pcm?: string; generationMs?: number; error?: string };
				if (packet.error) throw new Error(packet.error);
				if (packet.generationMs !== undefined) generation = packet.generationMs;
				if (!packet.pcm) continue;

				if (!samples) onFirstAudio(performance.now() - started);
				const bytes = Uint8Array.from(atob(packet.pcm), (byte) => byte.charCodeAt(0));
				const pcm = new Float32Array(bytes.buffer);
				playChunk(context, playback, pcm, sampleRate);
				chunks.push(bytes.buffer);
				samples += pcm.length;
			}
		}
	} finally {
		await reader.cancel().catch(() => {});
	}

	if (pending || generation === undefined || !samples) throw new Error('The audio stream ended early.');

	return { audio: makeWav(chunks, samples, sampleRate), duration: samples / sampleRate, generation };
}

export function playChunk(context: AudioContext, playback: { next: number }, pcm: Float32Array<ArrayBuffer>, sampleRate: number) {
	const buffer = context.createBuffer(1, pcm.length, sampleRate);
	buffer.copyToChannel(pcm, 0);
	const source = context.createBufferSource();
	source.buffer = buffer;
	source.connect(context.destination);
	const playAt = Math.max(context.currentTime, playback.next);
	source.start(playAt);
	playback.next = playAt + buffer.duration;
}

export function makeWav(chunks: ArrayBuffer[], samples: number, sampleRate: number) {
	const header = new ArrayBuffer(44);
	const view = new DataView(header);
	view.setUint32(0, 0x52494646); // RIFF
	view.setUint32(4, 36 + samples * 4, true);
	view.setUint32(8, 0x57415645); // WAVE
	view.setUint32(12, 0x666d7420); // fmt
	view.setUint32(16, 16, true);
	view.setUint16(20, 3, true); // IEEE float
	view.setUint16(22, 1, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 4, true);
	view.setUint16(32, 4, true);
	view.setUint16(34, 32, true);
	view.setUint32(36, 0x64617461); // data
	view.setUint32(40, samples * 4, true);
	return new Blob([header, ...chunks], { type: 'audio/wav' });
}
