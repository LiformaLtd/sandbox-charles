import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

export function createHeadPoseServer(root) {
  const assets = join(root, '.local/head-motion');
  const pending = new Map();
  let process;
  let nextId = 0;
  let idleTrack;

  function start() {
    const worker = spawn(join(assets, '.venv/bin/python'), [
      join(root, 'scripts/head_pose.py'), '--assets', assets,
    ], { stdio: ['pipe', 'pipe', 'inherit'] });
    process = worker;
    const lines = createInterface({ input: worker.stdout });
    function fail(error) {
      if (process !== worker) return;
      process = undefined;
      for (const request of pending.values()) request.reject(error);
    }
    worker.on('error', () => fail(new Error('Install the head motion models with sh scripts/setup_head_motion.sh.')));
    worker.stdin.on('error', () => fail(new Error('The head motion worker stopped accepting audio.')));
    worker.on('exit', () => fail(new Error('The head motion worker stopped. See the server log.')));
    lines.on('line', (line) => {
      let reply;
      try {
        reply = JSON.parse(line);
      } catch {
        fail(new Error('The head motion worker returned an invalid response.'));
        worker.kill();
        return;
      }
      const request = pending.get(reply.id);
      if (!request) return;
      if (reply.error) request.reject(new Error(reply.error));
      else request.resolve(reply);
    });
    return worker;
  }

  async function predict(request) {
    const worker = process ?? start();
    const id = nextId++;
    try {
      return await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Head motion inference timed out.')), 30_000);
        pending.set(id, {
          resolve: (reply) => { clearTimeout(timer); resolve(reply); },
          reject: (error) => { clearTimeout(timer); reject(error); },
        });
        worker.stdin.write(JSON.stringify({ id, ...request }) + '\n');
      });
    } finally {
      pending.delete(id);
    }
  }

  return {
    predict(pcm, startSample) {
      return predict({ pcm: pcm.toString('base64'), startSample });
    },
    idle() {
      idleTrack ??= predict({ idle: true }).catch((error) => {
        idleTrack = undefined;
        throw error;
      });
      return idleTrack;
    },
    close() {
      process?.kill();
    },
  };
}
