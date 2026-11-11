// Preserve the exact 3D comparison PCM instead of resampling 16 kHz WAV twice.
export function readPcm16Wav(bytes:ArrayBuffer):Float32Array|undefined {
  const view=new DataView(bytes);if(view.byteLength<12)return;
  const tag=(at:number)=>String.fromCharCode(view.getUint8(at),view.getUint8(at+1),view.getUint8(at+2),view.getUint8(at+3));
  if(tag(0)!=='RIFF'||tag(8)!=='WAVE')return;
  let supported=false,dataOffset=-1,dataSize=0;
  for(let offset=12;offset+8<=view.byteLength;){
    const size=view.getUint32(offset+4,true),start=offset+8;
    if(start+size>view.byteLength)throw new Error('The WAV file is truncated.');
    if(tag(offset)==='fmt '&&size>=16)supported=view.getUint16(start,true)===1&&view.getUint16(start+2,true)===1&&view.getUint32(start+4,true)===16000&&view.getUint16(start+12,true)===2&&view.getUint16(start+14,true)===16;
    if(tag(offset)==='data'){dataOffset=start;dataSize=size;}
    offset=start+size+(size%2);
  }
  if(!supported||dataOffset<0)return;
  if(dataSize%2)throw new Error('Invalid PCM16 WAV data.');
  const count=dataSize/2;if(!count||count>480000)throw new Error('Choose audio between 0 and 30 seconds long.');
  const pcm=new Float32Array(count);for(let i=0;i<count;i++)pcm[i]=view.getInt16(dataOffset+i*2,true)/32768;
  return pcm;
}

export async function decodeMono16k(bytes: ArrayBuffer): Promise<Float32Array> {
  const exact=readPcm16Wav(bytes);if(exact)return exact;
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
    const v = Math.max(-32768,Math.min(32767,Math.round(samples[i] * 32768)));
    view.setInt16(44+i*2,v,true);pcm[i]=v/32768;
  }
  return { pcm, blob: new Blob([buffer], { type: 'audio/wav' }) };
}
