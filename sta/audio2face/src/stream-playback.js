const sampleRate = 16000;

export function createStreamPlayback(context) {
  const timeline = [];
  let cursor = 0;
  let samples = 0;
  let ended = Promise.resolve();

  return {
    enqueue(pcm) {
      const buffer = context.createBuffer(1, pcm.length, sampleRate);
      buffer.copyToChannel(pcm, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      ended = new Promise((resolve) => {
        source.onended = () => {
          source.disconnect();
          resolve();
        };
      });
      const time = Math.max(timeline.at(-1)?.end ?? 0, context.currentTime + 0.02);
      timeline.push({ time, end: time + buffer.duration, offset: samples, length: pcm.length });
      samples += pcm.length;
      source.start(time);
    },
    playhead() {
      const now = context.currentTime;
      if (!timeline.length || now < timeline[0].time) return { time: -1, playing: false };
      while (cursor + 1 < timeline.length && now >= timeline[cursor + 1].time) cursor++;
      const segment = timeline[cursor];
      // During a buffer gap, hold the last frame whose audio already played.
      const sample = segment.offset + Math.min(segment.length - 1, Math.max(0, (now - segment.time) * sampleRate));
      return { time: sample / sampleRate, playing: now < segment.end };
    },
    async finish(signal) {
      signal.throwIfAborted();
      let cancel;
      try {
        await Promise.race([ended, new Promise((resolve) => {
          cancel = resolve;
          signal.addEventListener('abort', cancel, { once: true });
        })]);
      } finally {
        signal.removeEventListener('abort', cancel);
      }
      signal.throwIfAborted();
    },
  };
}
