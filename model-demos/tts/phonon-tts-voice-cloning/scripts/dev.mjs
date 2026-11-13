import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:net';
import { fileURLToPath } from 'node:url';

process.chdir(fileURLToPath(new URL('..', import.meta.url)));
const backendPort = await new Promise((resolve, reject) => {
  const server = createServer();
  server.once('error', reject);
  server.listen(0, '127.0.0.1', () => {
    const port = server.address().port;
    server.close(() => resolve(port));
  });
});
const env = { ...process.env, PHONON_CLONE_TOKEN: randomBytes(32).toString('hex'),
  PHONON_CLONE_URL: `http://127.0.0.1:${backendPort}`, HF_HUB_OFFLINE: '1' };
const children = [];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) child.kill('SIGTERM');
  process.exitCode = code;
}
function start(command, args) {
  const child = spawn(command, args, { env, stdio: 'inherit' });
  children.push(child);
  child.on('error', error => { console.error(error.message); stop(1); });
  child.on('exit', code => { if (!stopping) stop(code || 1); });
}
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
start('.venv/bin/python', ['-m', 'uvicorn', 'server:app', '--app-dir', 'python', '--host', '127.0.0.1', '--port', String(backendPort)]);
start(process.execPath, ['node_modules/vite/bin/vite.js', process.argv.includes('--preview') ? 'preview' : 'dev']);
