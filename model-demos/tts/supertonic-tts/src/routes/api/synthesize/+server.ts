import { error } from '@sveltejs/kit';
import { voices, languages, type Settings } from '../../../lib/model';
import { loadEngine, type Engine } from '../../../lib/engine';
import { assetBytes } from '../../../lib/server/assets';
let engine: Engine | undefined;
let queue = Promise.resolve();
export async function POST({ request, url }: { request: Request; url: URL }) {
 if (request.headers.get('origin') !== url.origin) error(403, 'Invalid request origin.');
 const raw = await request.text();
 if (raw.length > 10000) error(413, 'Request too large.');
 let settings: Settings;
 try { settings = JSON.parse(raw); } catch { error(400, 'Invalid request.'); }
 if (!settings || typeof settings.text !== 'string' || !settings.text.trim() || settings.text.length > 1000 || !voices.includes(settings.voice) || !languages.includes(settings.language) || !Number.isInteger(settings.steps) || settings.steps < 1 || settings.steps > 20 || !Number.isFinite(settings.speed) || settings.speed < .5 || settings.speed > 2) error(400, 'Invalid synthesis settings.');
 let cancelled = false;
 const encoder = new TextEncoder();
 const stream = new ReadableStream({
  start(controller) {
   const emit = (packet: object) => { if (!cancelled) controller.enqueue(encoder.encode(JSON.stringify(packet) + '\n')); };
   const started = performance.now();
   emit({ status: 'Preparing local model…' });
   queue = queue.then(async () => {
    if (cancelled) return;
    try {
     if (!engine) {
      const ort = await import('onnxruntime-node');
      engine = await loadEngine(ort, assetBytes, status => emit({ status }), 'cpu');
     }
     await engine.prepare(settings.voice);
     emit({ setup: performance.now() - started });
     await engine.generate(settings, packet => {
      if (packet.pcm) emit({ pcm: Buffer.from(packet.pcm.buffer).toString('base64'), sampleRate: packet.sampleRate });
      else emit(packet);
     }, () => cancelled);
    } catch (e) { console.error('Supertonic native synthesis failed', e); emit({ error: e instanceof Error ? e.message : String(e) }); }
    finally { if (!cancelled) controller.close(); }
   });
  },
  cancel() { cancelled = true; }
 });
 return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no' } });
}
