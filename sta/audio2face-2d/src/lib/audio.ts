export async function decodeMono16k(bytes: ArrayBuffer): Promise<Float32Array> {
  const decoder = new AudioContext();
  let decoded: AudioBuffer;
  try { decoded = await decoder.decodeAudioData(bytes); }
  finally { await decoder.close(); }
  if (!(decoded.duration > 0) || decoded.duration > 30) throw new Error('Choose audio between 0 and 30 seconds long.');
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
  return (await offline.startRendering()).getChannelData(0).slice();
}

export function playbackWav(samples: Float32Array): { pcm: Float32Array; blob: Blob } {
  const buffer = new ArrayBuffer(44 + samples.length * 2); const view = new DataView(buffer);
  const text = (at: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
  text(0,'RIFF'); view.setUint32(4,36 + samples.length * 2,true); text(8,'WAVE');text(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
  view.setUint32(24,16000,true);view.setUint32(28,32000,true);view.setUint16(32,2,true);view.setUint16(34,16,true);
  text(36,'data');view.setUint32(40,samples.length*2,true);
  const pcm = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    if (!Number.isFinite(samples[i])) throw new Error('Audio contains invalid samples.');
    const v = Math.round(Math.max(-1,Math.min(1,samples[i])) * 32767);
    view.setInt16(44+i*2,v,true);pcm[i]=v/32768;
  }
  return { pcm, blob: new Blob([buffer], { type: 'audio/wav' }) };
}
