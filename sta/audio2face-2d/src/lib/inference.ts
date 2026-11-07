import type { Catalog, Runtime, Track } from './types';

export class InferenceClient {
  private worker: Worker | undefined;
  private reject: ((error: Error) => void) | undefined;
  cancel() {
    this.worker?.terminate();this.worker=undefined;
    this.reject?.(new DOMException('Preparation cancelled.','AbortError'));this.reject=undefined;
  }
  async run(pcm: Float32Array, model: Catalog['model'], runtime: Runtime, signal: AbortSignal,
    progress: (message: string, percent: number) => void): Promise<Track> {
    signal.throwIfAborted();
    if (this.reject) throw new Error('A speech preparation is already running.');
    const worker = this.worker ??= new Worker(new URL('./inference.worker.ts',import.meta.url),{ type: 'module' });
    return new Promise((resolve,reject) => {
      this.reject=reject;const abort=()=>this.cancel();
      const clean = () => { signal.removeEventListener('abort',abort);this.reject=undefined;worker.onmessage=null;worker.onerror=null; };
      signal.addEventListener('abort',abort,{ once: true });
      worker.onmessage=({ data })=> {
        if (data.type==='progress') progress(data.message,data.percent);
        if (data.type==='complete') { clean();resolve(data as Track); }
        if (data.type==='error') { clean();reject(new Error(data.message)); }
      };
      worker.onerror=event=> { clean();worker.terminate();this.worker=undefined;reject(new Error(event.message || 'Shape worker failed.')); };
      const samples=pcm.slice();
      worker.postMessage({ pcm: samples, url: model.url, checksum: model.sha256, runtime },[samples.buffer]);
    });
  }
}
