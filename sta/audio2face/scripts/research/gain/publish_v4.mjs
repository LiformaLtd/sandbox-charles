// Publish the exact gain-study checkpoint requested by the user, without overwriting prior v4 weights.
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
const validation = resolve(process.env.A2F_VALIDATION_OUTPUT || join(app, '.local/compact-gain-v2'));
const stage = join(validation, 'cdn-stage-gain');
await mkdir(stage, { recursive: true });
const selected = JSON.parse(await readFile(join(validation, 'selection.json'))).augmented;
const expected = '84da9bfe22d221075d85e8a52622fc7d31b18d12b04905e0902a7ec64eb5ab2f';
if (selected.best_epoch !== 15 || selected.model_sha256 !== expected) throw Error('The validated candidate selection changed.');
async function digest(stream) {
  const h = createHash('sha256'); for await (const c of stream) h.update(c); return h.digest('hex');
}
const model = join(validation, 'augmented/runs/control-replay-s37/model.onnx');
if (await digest(createReadStream(model)) !== expected) throw Error('Candidate checksum mismatch.');
const prefix = 'models/speech-to-animation/lf-sta/arkit-student-v4-gain-r1';
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
const manifest = { id: 'lf-sta', version: 'arkit-student-v4-gain-r1', status: 'published at user request; single-seed gain-robustness update; physical iPhone validation pending',
  selected_seed: 37, selected_epoch: 15, output_channels: 52, active_channels: 39,
  predecessor_sha256: 'b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d',
  evaluation: { gain_improvement_vs_matched_control: 0.03412598611825446,
    quarter_level_improvement_vs_matched_control: 0.07292440482717377, original_level_improvement_vs_matched_control: 0.0032166770910080356,
    initial_5_percent_gate_passed: false, confirmation: 'one seed; previously inspected pilot voices/texts' },
  study_protocol_sha256: await digest(createReadStream(join(validation, 'protocol.json'))), files: {} };
for (const file of files) manifest.files[file[0]] = await publish(file);
const path = join(stage, 'manifest.json');
await writeFile(path, JSON.stringify(manifest, null, 2) + '\n');
await publish(['manifest.json', path, 'application/json']);
