import { error } from '@sveltejs/kit';
import { stat } from 'node:fs/promises';
import { files } from '../../../../lib/model';
import { assetPath, assetStream } from '../../../../lib/server/assets';
export async function GET({ params }: { params: { file?: string } }) {
 const file = params.file || '';
 if (!files.includes(file)) error(404, 'Unknown model asset.');
 try {
  const path = await assetPath(file);
  return new Response(assetStream(path), { headers: { 'Content-Type': file.endsWith('.json') ? 'application/json' : 'application/octet-stream', 'Content-Length': String((await stat(path)).size), 'Cache-Control': 'public, max-age=31536000, immutable' } });
 } catch (e) { console.error('Supertonic asset loading failed', file, e); error(502, 'Could not download the model asset. Check the server connection and try again.'); }
}
