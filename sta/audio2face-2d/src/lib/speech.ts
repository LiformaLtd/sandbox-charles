import { PhononTTS } from 'phonon-tts';
import workerUrl from './local-speech.worker.js?worker&url';
import wasmUrl from 'phonon-tts/wasm/phonon_tts_bg.wasm?url';
import simdWasmUrl from 'phonon-tts/wasm/phonon_tts_simd_bg.wasm?url';

let loading: Promise<PhononTTS> | undefined;
export async function releaseSpeech() {
  const pending=loading;loading=undefined;
  try { (await pending)?.dispose(); } catch { /* The generation handler owns initialization errors. */ }
}
export async function generateSpeech(text: string, signal: AbortSignal, report: (message: string) => void) {
  if (!text.trim() || text.length>500) throw new Error('Enter between 1 and 500 characters.');
  loading ??= PhononTTS.load({
    model: { weights: { q8:'/assets/lf-tts/model.q8.gguf' },tokenizer:'/assets/lf-tts/tokenizer.json',
      config:'/assets/lf-tts/config.json',voices:{ default:'/assets/lf-tts/easton.safetensors' },defaultVoice:'default' },
    lang:'en',quant:'q8',device:'cpu',threads:1,cache:false,
    workerUrl:new URL(workerUrl,location.href).href,wasmUrl,simdWasmUrl,
    onProgress:({ total,loaded })=>{ if(total!==null&&total>1000000)report(`Loading local speech… ${Math.min(100,Math.round(loaded/total*100))}%`); }
  }).catch(error=>{ loading=undefined;throw error; });
  const engine=await loading;signal.throwIfAborted();report('Generating local speech…');
  const chunks: Float32Array[]=[];let length=0;
  const stream=engine.stream(text,{ voice:'default',signal });
  stream.done.catch(()=>{});
  for await (const chunk of stream) {
    signal.throwIfAborted();length+=chunk.length;
    if(length>engine.sampleRate*30)throw new Error('Generated speech exceeded 30 seconds. Shorten the text.');
    chunks.push(chunk);
  }
  await stream.done;
  const samples=new Float32Array(length);let offset=0;
  for(const chunk of chunks){samples.set(chunk,offset);offset+=chunk.length;}
  const offline=new OfflineAudioContext(1,Math.ceil(length/engine.sampleRate*16000),16000);
  const buffer=offline.createBuffer(1,length,engine.sampleRate);buffer.copyToChannel(samples,0);
  const source=offline.createBufferSource();source.buffer=buffer;source.connect(offline.destination);source.start();
  return (await offline.startRendering()).getChannelData(0).slice();
}
