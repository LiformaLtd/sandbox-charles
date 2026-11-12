import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, stat, rename, rm, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { randomUUID } from 'node:crypto';
import { env } from 'node:process';
import { files, modelBase, revision } from '../model';
export const cache = () => env.SUPERTONIC_MODEL_DIR || join(homedir(), 'Source/liforma/ai-workflows/apple-silicon/.supertonic-3-cache', revision);
const pending = new Map<string, Promise<string>>();
export async function assetPath(file: string): Promise<string> {
 if (!files.includes(file)) throw new Error('Unknown model asset.');
 const path = join(cache(), file);
 try { if ((await stat(path)).size > 0) return path; } catch (e) { if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e; }
 const existing = pending.get(file); if (existing) return existing;
 const task = (async () => {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.partial`;
  try {
   const response = await fetch(modelBase + file, { signal: AbortSignal.timeout(300_000) });
   if (!response.ok || !response.body) throw new Error(`Model download failed: ${file} (${response.status})`);
   await pipeline(Readable.fromWeb(response.body as any), createWriteStream(temporary, { flags: 'wx' }));
   const size = (await stat(temporary)).size;
   const expected = Number(response.headers.get('content-length'));
   if (!size || (expected && size !== expected)) throw new Error(`Incomplete model download: ${file}`);
   await rename(temporary, path); return path;
  } finally { await rm(temporary, { force: true }); }
 })();
 pending.set(file, task);
 try { return await task; } finally { pending.delete(file); }
}
export async function assetBytes(file: string) { return new Uint8Array(await readFile(await assetPath(file))); }
export function assetStream(path: string) { return Readable.toWeb(createReadStream(path)) as ReadableStream<Uint8Array>; }
