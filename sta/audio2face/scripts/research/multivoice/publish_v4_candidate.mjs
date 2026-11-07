// Publish the exact validation-selected candidate under an immutable CDN prefix.
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, mkdir, copyFile, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';

const app = fileURLToPath(new URL('../../../', import.meta.url));
if (!process.env.A2F_CDN_ROOT || !process.env.A2F_LICENSE_ROOT) throw Error('Set A2F_CDN_ROOT and A2F_LICENSE_ROOT.');
const cdn = resolve(process.env.A2F_CDN_ROOT);
const require = createRequire(join(cdn, 'package.json'));
const env = require('dotenv').parse(await readFile(resolve(cdn, '../../.env')));
const setting = name => process.env[name]?.trim() || env[name]?.trim();
const origin = setting('PUBLIC_CDN_BASE_URL')?.replace(/\/$/, '');
const bucket = setting('S3_BUCKET');
if (origin !== 'https://staging-cdn.liforma.ai' || bucket !== 'liforma-staging') throw Error('Expected the existing staging CDN bucket.');
for (const key of ['S3_API_URL', 'S3_ACCESS_KEY_ID', 'S3_SECRET_ACCESS_KEY']) {
  if (!setting(key)) throw Error('Missing CDN configuration: ' + key);
}
const { S3Client, HeadObjectCommand } = require('@aws-sdk/client-s3');
const { putOrResumeFile, installUploadStopHandlers } = await import(join(cdn, 'scripts/lib/resumable-r2-upload.mjs'));
const client = new S3Client({ region: 'auto', endpoint: setting('S3_API_URL'), forcePathStyle: true,
  credentials: { accessKeyId: setting('S3_ACCESS_KEY_ID'), secretAccessKey: setting('S3_SECRET_ACCESS_KEY') } });
installUploadStopHandlers();
const validation = resolve(process.env.A2F_VALIDATION_OUTPUT || join(app, '.local/compact-v4-validation'));
const stage = join(validation, 'cdn-stage');
await mkdir(stage, { recursive: true });
const selected = JSON.parse(await readFile(join(validation, 'selection.json'))).representative;
const expected = 'b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d';
if (selected.seed !== 73 || selected.model_sha256 !== expected) throw Error('The validated candidate selection changed.');
async function digest(stream) {
  const h = createHash('sha256'); for await (const c of stream) h.update(c); return h.digest('hex');
}
const model = join(validation, 'v4-candidate.onnx');
if (await digest(createReadStream(model)) !== expected) throw Error('Candidate checksum mismatch.');
const prefix = 'models/speech-to-animation/lf-sta/arkit-student-v4-candidate';
for (const name of ['LICENSE.pdf', 'MAYA-ACE-LICENSE.txt', 'NOTICE.txt']) {
  await copyFile(join(process.env.A2F_LICENSE_ROOT, name), join(stage, name));
}
const files = [
  ['LICENSE.pdf', join(stage, 'LICENSE.pdf'), 'application/pdf'],
  ['MAYA-ACE-LICENSE.txt', join(stage, 'MAYA-ACE-LICENSE.txt'), 'text/plain'],
  ['NOTICE.txt', join(stage, 'NOTICE.txt'), 'text/plain'],
  ['james-arkit-student-v4.onnx', model, 'application/octet-stream'],
];
async function publish([name, filePath, contentType]) {
  const key = prefix + '/' + name;
  const sha256 = await digest(createReadStream(filePath));
  const { size: bytes } = await stat(filePath);
  const existing = await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key })).catch(error => {
    if (error.$metadata?.httpStatusCode === 404) return null; throw error;
  });
  if (existing && existing.ContentLength !== bytes) throw Error('Immutable CDN object differs: ' + key);
  if (!existing) await putOrResumeFile({ client, bucket, key, filePath, contentType, stateDir: join(stage, 'resume') });
  const response = await fetch(origin + '/' + key);
  if (!response.ok || await digest(response.body) !== sha256) throw Error('CDN byte verification failed: ' + key);
  console.log('Verified CDN', key, bytes, sha256);
  return { bytes, sha256 };
}
const manifest = { id: 'lf-sta', version: 'arkit-student-v4-candidate', status: 'candidate; physical iPhone validation pending',
  selected_seed: 73, selected_epoch: 19, output_channels: 52, active_channels: 39, files: {} };
for (const file of files) manifest.files[file[0]] = await publish(file);
const path = join(stage, 'manifest.json');
await writeFile(path, JSON.stringify(manifest, null, 2) + '\n');
await publish(['manifest.json', path, 'application/json']);
