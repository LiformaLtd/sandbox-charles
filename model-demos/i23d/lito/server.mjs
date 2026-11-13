import { spawn } from 'node:child_process';
import {
  accessSync, closeSync, constants, existsSync, mkdirSync, openSync, readFileSync,
  readdirSync, renameSync, rmSync, statSync, symlinkSync, writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import express from 'express';
import { createServer } from 'node:http';

let port = 5173;
const dataDir = resolve(process.env.LITO_DATA_DIR || '.lito-data');
const jobsDir = join(dataDir, 'jobs');
const runtimeDir = join(dataDir, 'runtime');
const cli = process.env.LITO_CLI;
const weights = process.env.LITO_WEIGHTS && resolve(process.env.LITO_WEIGHTS);
const idPattern = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;

if (!cli || !weights) throw new Error('Set LITO_CLI and LITO_WEIGHTS in .env before starting.');
const python = resolve(process.env.LITO_PYTHON || join(dirname(resolve(cli)), 'python'));
const refineScript = resolve('refine.py');
accessSync(cli, constants.X_OK);
accessSync(join(weights, 'image_to_3d', 'lito_dit_rgba.safetensors'), constants.R_OK);
accessSync(join(weights, 'tokenizer', 'lito_new.safetensors'), constants.R_OK);
accessSync(join(weights, 'dependencies', 'trellis', 'ckpts', 'ss_dec_conv3d_16l8_fp16.safetensors'), constants.R_OK);
accessSync('dist/index.html', constants.R_OK);

mkdirSync(jobsDir, { recursive: true });
const decoderDir = join(runtimeDir, 'weights', 'trellis2', 'microsoft');
mkdirSync(decoderDir, { recursive: true });
const decoderLink = join(decoderDir, 'TRELLIS-image-large');
// This path is owned by the demo and bridges the pinned CLI's legacy decoder lookup.
rmSync(decoderLink, { force: true });
symlinkSync(join(weights, 'dependencies', 'trellis'), decoderLink, 'dir');

function jobPath(id, name) {
  return join(jobsDir, id, name);
}

function readJob(id) {
  try {
    return JSON.parse(readFileSync(jobPath(id, 'job.json'), 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function writeJob(job) {
  const temporary = jobPath(job.id, 'job.tmp');
  writeFileSync(temporary, JSON.stringify(job));
  renameSync(temporary, jobPath(job.id, 'job.json'));
}

function imageExtension(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) return 'png';
  if (bytes.subarray(0, 3).equals(Buffer.from([255, 216, 255]))) return 'jpg';
  if (bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') return 'webp';
  return null;
}

const queue = [];
let active = false;
for (const id of readdirSync(jobsDir)) {
  if (!idPattern.test(id)) continue;
  const job = readJob(id);
  if (job?.state === 'queued') queue.push(id);
  if (job?.state === 'running' || job?.state === 'refining') {
    writeJob({ ...job, state: 'failed', error: 'The local server restarted during generation. Click Generate to try again.' });
  }
}

function runProcess(command, args, log) {
  const logFd = openSync(log, 'a');
  try {
    const child = spawn(command, args, { cwd: runtimeDir, stdio: ['ignore', logFd, logFd] });
    return new Promise((resolveProcess, rejectProcess) => {
      child.once('error', rejectProcess);
      child.once('close', (code, signal) => {
        if (signal) rejectProcess(new Error(`Process stopped with ${signal}`));
        else resolveProcess(code);
      });
    });
  } finally {
    closeSync(logFd);
  }
}

async function runNext() {
  if (active || queue.length === 0) return;
  active = true;
  const id = queue.shift();
  const job = readJob(id);
  let input = jobPath(id, job.input);
  const output = jobPath(id, 'splat.ply');
  const base = job.sourceDetail ? jobPath(id, 'base.ply') : output;
  const log = jobPath(id, 'generation.log');
  let stage = 'generation';
  try {
    writeJob({ ...job, state: 'running' });
    let sourceBackgroundRemoved;
    let sourceDetailReason;
    if (job.sourceDetail) {
      stage = 'source detail';
      const prepared = jobPath(id, 'keyed.png');
      const code = await runProcess(python, [refineScript, 'prepare', input, prepared], log);
      if (code === 0) {
        input = prepared;
        sourceBackgroundRemoved = true;
      } else if (code === 3) {
        sourceDetailReason = 'background';
      } else if (code !== 4) {
        throw new Error(`Source preparation exited with code ${code}.`);
      }
    }

    stage = 'generation';
    const code = await runProcess(resolve(cli), [
      'generate', input, '--weights-root', weights, '--output', base,
      '--max-init-coords-per-batch', 'none', '--seed', '42', '--print-metrics',
    ], log);
    if (code !== 0 || !existsSync(base) || !statSync(base).size) {
      throw new Error(`Generation exited with code ${code} or produced no splat.`);
    }

    let sourceDetailApplied = sourceDetailReason ? false : undefined;
    if (job.sourceDetail && !sourceDetailReason) {
      stage = 'source detail';
      writeJob({ ...job, state: 'refining' });
      const refined = jobPath(id, 'refined.ply');
      const code = await runProcess(python, [refineScript, 'project', base, input, refined], log);
      if (code === 3) {
        sourceDetailReason = 'alignment';
        sourceDetailApplied = false;
      } else {
        if (code !== 0 || !existsSync(refined) || !statSync(refined).size) {
          throw new Error(`Refinement exited with code ${code} or produced no splat.`);
        }
        renameSync(refined, output);
        rmSync(base, { force: true });
        sourceDetailApplied = true;
      }
    }
    if (sourceDetailApplied === false) renameSync(base, output);
    writeJob({ ...job, state: 'complete', sourceDetailApplied, sourceDetailReason, sourceBackgroundRemoved });
    console.log(`LiTo job ${id} completed`);
  } catch (error) {
    console.error(`LiTo job ${id} failed: ${error}; log: ${log}`);
    const hint = stage === 'source detail' ? ' Check that the source detail dependencies are installed.' : '';
    writeJob({ ...job, state: 'failed', error: `LiTo ${stage} failed. See the local job log for ${id}.${hint}` });
  } finally {
    active = false;
    void runNext();
  }
}

const app = express();
app.disable('x-powered-by');
app.use('/api/jobs', (_request, response, next) => {
  response.set('Cache-Control', 'no-store');
  next();
});
app.param('id', (request, response, next, id) => {
  if (!idPattern.test(id)) return response.status(400).json({ error: 'Invalid job ID.' });
  next();
});

app.put('/api/jobs/:id', express.raw({ type: 'application/octet-stream', limit: '20mb' }), (request, response) => {
  const { id } = request.params;
  const existing = readJob(id);
  if (existing) return response.json(existing);
  const extension = Buffer.isBuffer(request.body) && imageExtension(request.body);
  if (!extension) return response.status(415).json({ error: 'Choose a PNG, JPEG, or WebP image.' });
  const input = `input.${extension}`;
  mkdirSync(jobPath(id, '.'), { recursive: true });
  writeFileSync(jobPath(id, input), request.body);
  const job = { id, input, state: 'queued', sourceDetail: request.query.sourceDetail === '1' };
  writeJob(job);
  queue.push(id);
  response.status(202).json(job);
  runNext();
});
app.get('/api/jobs/:id', (request, response) => {
  const job = readJob(request.params.id);
  if (!job) return response.status(404).json({ error: 'Job not found.' });
  response.json(job);
});
app.get('/api/jobs/:id/splat.ply', (request, response) => {
  const job = readJob(request.params.id);
  if (job?.state !== 'complete') return response.status(404).json({ error: 'Splat is not ready.' });
  response.type('application/octet-stream').sendFile('splat.ply', { root: jobPath(job.id, '.') });
});

app.use('/vendor/spark', express.static('node_modules/@sparkjsdev/spark/dist'));
app.use('/vendor/three', express.static('node_modules/three/build'));
app.use('/vendor/three-addons', express.static('node_modules/three/examples/jsm'));
app.use(express.static('dist'));
app.use((error, _request, response, _next) => {
  if (response.headersSent) return response.end();
  if (error.status === 413) return response.status(413).json({ error: 'Choose an image smaller than 20 MB.' });
  console.error(error);
  response.status(500).json({ error: 'The local server could not complete the request. Check its log.' });
});
const server = createServer(app);
server.on('error', error => {
  if (error.code === 'EADDRINUSE' && port < 65535) {
    server.listen(++port, '127.0.0.1');
  } else {
    console.error('LiTo could not start its local server:', error);
    process.exitCode = 1;
  }
});
server.on('listening', () => {
  console.log(`LiTo demo listening at http://127.0.0.1:${port}`);
  runNext();
});
server.listen(port, '127.0.0.1');
