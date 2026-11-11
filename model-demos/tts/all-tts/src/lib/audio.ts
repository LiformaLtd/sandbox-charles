export function playChunk(context: AudioContext, playback: { next: number }, pcm: Float32Array, sampleRate: number) {
	const buffer = context.createBuffer(1, pcm.length, sampleRate);
	buffer.getChannelData(0).set(pcm);
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
