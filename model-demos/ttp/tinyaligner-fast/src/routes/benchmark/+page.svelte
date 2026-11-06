<script lang="ts">
  let result = $state('');
  let running = $state(false);

  async function run() {
    running = true;
    result = '';
    try {
      const params = new URLSearchParams(location.search);
      const variant = params.get('variant') ?? 'full';
      if (variant !== 'full' && variant !== 'english') throw new Error('Expected full or english benchmark variant');
      const threads = Number(params.get('threads') ?? 1);
      if (!Number.isInteger(threads) || threads < 1) throw new Error('Invalid thread count');
      if (threads > 1 && !crossOriginIsolated) {
        throw new Error('Threaded WASM requires a cross-origin-isolated benchmark page');
      }
      const [ort, { benchmark }] = await Promise.all([
        import('onnxruntime-web/wasm'), import('../../lib/alignment/benchmark'),
      ]);
      ort.env.wasm.numThreads = threads;
      ort.env.wasm.initTimeout = 15_000;
      async function asset(name: string) {
        const response = await fetch(`/models/${name}`);
        if (!response.ok) throw new Error(`Missing benchmark asset ${name}`);
        return response;
      }
      const [model, dictionary, fixture] = await Promise.all([
        asset('model.onnx').then(async (response) => new Uint8Array(await response.arrayBuffer())),
        asset('lexicon.json').then((response) => response.json()),
        asset('benchmark.f32').then(async (response) => new Float32Array(await response.arrayBuffer())),
      ]);
      const [englishModel, englishPhones] = variant === 'english' ? await Promise.all([
        asset('model-english.onnx').then(async (response) => new Uint8Array(await response.arrayBuffer())),
        asset('phones-english.json').then((response) => response.json()),
      ]) : [];
      result = JSON.stringify({
        userAgent: navigator.userAgent,
        crossOriginIsolated,
        variant,
        ...await benchmark({
          ort, model: englishModel ?? model, fixture, threads, runs: Number(params.get('runs') ?? 7),
          dictionary: englishPhones ? { ...dictionary, phone2idx: englishPhones } : dictionary,
          comparison: englishModel ? { model, phone2idx: dictionary.phone2idx } : undefined,
        }),
        effectiveThreads: ort.env.wasm.numThreads,
      }, null, 2);
    } catch (error) {
      result = String(error);
    } finally {
      running = false;
    }
  }
</script>

<h1>TinyAligner benchmark</h1>
<button onclick={run} disabled={running}>{running ? 'Running…' : 'Run benchmark'}</button>
<pre>{result}</pre>
