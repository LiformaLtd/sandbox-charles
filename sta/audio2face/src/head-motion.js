const sampleRate = 16000;

export async function predictHead(window, startSample, signal) {
  const response = await fetch('/api/head-motion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', 'X-Start-Sample': String(startSample) },
    body: window,
    signal,
  });
  const track = await response.json();
  if (!response.ok) throw new Error(track.error || 'Head motion inference failed.');
  return { ...track, startSample };
}

function at(track, sample) {
  const index = Math.max(0, Math.min(track.angles.length - 1, (sample - track.startSample) * track.fps / sampleRate));
  const lower = Math.floor(index);
  const left = track.angles[lower];
  const right = track.angles[Math.min(lower + 1, track.angles.length - 1)];
  return left.map((angle, axis) => angle + (index - lower) * (right[axis] - angle));
}

function smooth(track, sample) {
  const before = at(track, sample - 640);
  const current = at(track, sample);
  const after = at(track, sample + 640);
  return current.map((angle, axis) => (before[axis] + 2 * angle + after[axis]) / 4);
}

export function joinHeadWindow(track, previous, sampleIndices) {
  const scale = [0.7, 0.7, 0.5];
  const limit = [12, 14, 8];
  return sampleIndices.map((sample) => {
    const angle = smooth(track, sample);
    // Consecutive windows overlap by at least 250 ms past the join. Blend
    // their predictions so the model's window boundaries are not visible.
    const blend = Math.min(1, (sample - sampleIndices[0]) / 3200);
    const amount = blend * blend * (3 - 2 * blend);
    const prior = previous && blend < 1 ? smooth(previous, sample) : angle;
    const leadIn = Math.min(1, sample / 5600);
    const fade = leadIn * leadIn * (3 - 2 * leadIn);
    return angle.map((value, axis) => {
      const combined = (prior[axis] + (value - prior[axis]) * amount) * scale[axis] * fade;
      return limit[axis] * Math.tanh(combined / limit[axis]);
    });
  });
}
