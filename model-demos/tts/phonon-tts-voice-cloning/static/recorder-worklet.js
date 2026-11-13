class VoiceRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(4096);
    this.count = 0;
    this.active = true;
    this.port.onmessage = event => {
      if (event.data === 'stop') {
        this.active = false;
        this.flush();
        this.port.postMessage({ done: true });
      }
    };
  }
  flush() {
    if (!this.count) return;
    const samples = this.samples.slice(0, this.count);
    this.port.postMessage({ samples }, [samples.buffer]);
    this.count = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (this.active && channel) {
      for (const sample of channel) {
        this.samples[this.count++] = sample;
        if (this.count === this.samples.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('voice-recorder', VoiceRecorder);
