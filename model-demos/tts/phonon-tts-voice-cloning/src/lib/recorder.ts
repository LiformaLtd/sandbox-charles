export function encodeWav(samples: Float32Array): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const string = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  string(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  string(8, 'WAVE'); string(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true); view.setUint32(28, 48000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  string(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function resample(samples: Float32Array, sampleRate: number): Promise<Blob> {
  const context = new OfflineAudioContext(1, Math.round(samples.length * 24000 / sampleRate), 24000);
  const input = context.createBuffer(1, samples.length, sampleRate);
  input.copyToChannel(new Float32Array(samples), 0);
  const source = context.createBufferSource(); source.buffer = input;
  source.connect(context.destination); source.start();
  return encodeWav((await context.startRendering()).getChannelData(0));
}

export async function normalizeFile(file: File): Promise<Blob> {
  if (file.size > 20_000_000) throw new Error('Choose a recording smaller than 20 MB.');
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (decoded.duration < 5 || decoded.duration > 20) throw new Error('Choose a recording between 5 and 20 seconds.');
    const mono = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const values = decoded.getChannelData(channel);
      for (let i = 0; i < mono.length; i++) mono[i] += values[i] / decoded.numberOfChannels;
    }
    return await resample(mono, decoded.sampleRate);
  } finally { await context.close(); }
}

export class Recorder {
  private context?: AudioContext;
  private stream?: MediaStream;
  private source?: MediaStreamAudioSourceNode;
  private node?: AudioWorkletNode;
  private chunks: Float32Array[] = [];
  private finish?: () => void;
  private stopping = false;

  async start(onLevel: (level: number) => void): Promise<void> {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone recording requires localhost or HTTPS.');
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      this.context = new AudioContext();
      await this.context.resume();
      await this.context.audioWorklet.addModule('/recorder-worklet.js');
      this.source = this.context.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.context, 'voice-recorder');
      this.node.port.onmessage = event => {
        if (event.data.done) { this.finish?.(); return; }
        const samples = event.data.samples as Float32Array;
        this.chunks.push(samples);
        let energy = 0;
        for (const sample of samples) energy += sample * sample;
        onLevel(Math.min(1, Math.sqrt(energy / samples.length) * 5));
      };
      this.source.connect(this.node);
      // Worklet output is silence; connecting keeps processing active without
      // monitoring microphone audio back through the speakers.
      this.node.connect(this.context.destination);
    } catch (error) { await this.dispose(); throw error; }
  }

  async stop(): Promise<Blob> {
    if (!this.node || !this.context || this.stopping) throw new Error('There is no active recording.');
    this.stopping = true;
    const sampleRate = this.context.sampleRate;
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error('The microphone stopped responding. Please record again.')), 2000);
      this.finish = () => { window.clearTimeout(timeout); resolve(); };
      this.node!.port.postMessage('stop');
    }).finally(() => this.dispose());
    const samples = new Float32Array(this.chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of this.chunks) { samples.set(chunk, offset); offset += chunk.length; }
    if (samples.length / sampleRate < 5) throw new Error('That recording was too short. Speak for at least 5 seconds.');
    return resample(samples.subarray(0, 20 * sampleRate), sampleRate);
  }

  async dispose(): Promise<void> {
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect(); this.node?.disconnect();
    if (this.context && this.context.state !== 'closed') await this.context.close();
  }
}
