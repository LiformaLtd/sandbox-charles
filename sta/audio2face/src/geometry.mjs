const sampleRate = 16000;
const hop = 8000;
const fps = 60;
const outputSize = 88831;

export async function* audioWindows(source) {
  const chunks = (source[Symbol.asyncIterator] ?? source[Symbol.iterator]).call(source);
  let audio = new Float32Array(sampleRate);
  let length = 0;
  let ended = false;
  try {
    for (let start = -sampleRate; ; start += hop) {
      // Wait for the whole window during streaming. Only pad after the final chunk.
      while (!ended && length < start + sampleRate) {
        const { value, done } = await chunks.next();
        ended = done;
        if (done) break;
        if (!(value instanceof Float32Array) || !value.every(Number.isFinite)) {
          throw new TypeError('Expected finite 16 kHz mono PCM chunks');
        }
        if (length + value.length > audio.length) {
          const larger = new Float32Array(Math.max(2 * audio.length, length + value.length));
          larger.set(audio.subarray(0, length));
          audio = larger;
        }
        audio.set(value, length);
        length += value.length;
      }
      if (ended && start >= length - 4000) return;

      const samples = new Float32Array(sampleRate);
      const left = Math.max(start, 0);
      samples.set(audio.subarray(left, Math.min(start + sampleRate, length)), left - start);
      const sampleIndices = [];
      let first = 0;
      for (let i = 15; i < 45; i++) {
        const sample = start + Math.floor(i * sampleRate / fps);
        if (sample < 0 || sample >= length) continue;
        if (!sampleIndices.length) first = i;
        sampleIndices.push(sample);
      }
      const pcm = sampleIndices.length
        ? audio.slice(Math.max(0, start + 4000), Math.min(length, start + 12000))
        : undefined;
      const headStart = start < 0 ? start : Math.max(0, start - sampleRate);
      const headWindow = sampleIndices.length ? new Float32Array(start + sampleRate - headStart) : undefined;
      if (headWindow) {
        const headLeft = Math.max(0, headStart);
        headWindow.set(audio.subarray(headLeft, Math.min(start + sampleRate, length)), headLeft - headStart);
      }
      yield { samples, sampleIndices, first, pcm, headStart, headWindow };
    }
  } finally {
    if (!ended) await chunks.return?.();
  }
}

// Drive the batch-one graph with 16 kHz mono PCM and caller-supplied noise.
// Pass the same ONNX Runtime Web module used to create the session.
export async function* geometryBlocks(ort, session, audio, noise, identity = 1, inferHead) {
  const noiseShape = [1, 3, 60, outputSize];
  if (!(noise instanceof Float32Array) || noise.length !== 3 * 60 * outputSize) {
    throw new TypeError('Expected float32 diffusion noise with shape [1, 3, 60, 88831]');
  }
  if (!Number.isInteger(identity) || identity < 0 || identity > 2) {
    throw new RangeError('Identity must be 0 (Claire), 1 (James), or 2 (Mark)');
  }

  const tensor = (data, shape) => new ort.Tensor('float32', data, shape);
  const oneHot = new Float32Array(3);
  oneHot[identity] = 1;
  const shared = {
    identity: tensor(oneHot, [1, 3]),
    emotion: tensor(new Float32Array(30 * 10), [1, 30, 10]),
    noise: tensor(noise, noiseShape),
  };
  let state = tensor(new Float32Array(2 * 2 * 256), [2, 2, 1, 256]);

  try {
    const source = audio instanceof Float32Array ? [audio] : audio;
    for await (const { samples, sampleIndices, first, pcm, headStart, headWindow } of audioWindows(source)) {
      const head = headWindow && inferHead
        ? inferHead(headWindow, headStart).then((track) => ({ track }), (error) => ({ error }))
        : undefined;
      const window = tensor(samples, [1, sampleRate]);
      let output;
      try {
        output = await session.run({...shared, window, input_latents: state});
      } finally {
        window.dispose();
      }
      state.dispose();
      state = output.output_latents;
      try {
        if (sampleIndices.length) {
          const geometry = output.prediction.data.subarray(
            first * outputSize, (first + sampleIndices.length) * outputSize,
          );
          const result = await head;
          if (result?.error) throw result.error;
          yield { sampleIndices, geometry, pcm, head: result?.track };
        }
      } finally {
        output.prediction.dispose();
      }
    }
  } finally {
    state.dispose();
    for (const input of Object.values(shared)) input.dispose();
  }
}
