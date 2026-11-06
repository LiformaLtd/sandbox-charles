// The rig supplies the closed eyelid shape. Use a short, repeatable
// blink envelope on the viewer clock to drive it.
export function blinkAt(time) {
  const peaks = [0.9, 4.25, 7.05];
  const cycle = 10.2;
  const closeSeconds = 0.09;
  const holdSeconds = 0.017;
  const reopenSeconds = 0.19;
  const end = holdSeconds + reopenSeconds;
  const phase = time % cycle;
  for (const peak of peaks) {
    const delta = phase - peak;
    if (delta < -closeSeconds || delta > end) continue;
    if (delta >= 0 && delta <= holdSeconds) return 1;
    const t = delta < 0 ? 1 + delta / closeSeconds : (end - delta) / reopenSeconds;
    return t * t * (3 - 2 * t);
  }
  return 0;
}

const neutral = [0, 0, 0];

export async function loadIdleHead() {
  const response = await fetch('/api/idle-head-motion');
  const track = await response.json();
  if (!response.ok) throw new Error(track.error || 'DAWN idle inference failed.');
  return track;
}

export function idleHeadAt(track, time) {
  if (!track) return neutral;
  const { fps, angles } = track;
  const position = (time * fps) % angles.length;
  const index = Math.floor(position);
  const next = angles[(index + 1) % angles.length];
  const fade = Math.min(1, time / 0.5);
  return angles[index].map((angle, axis) => fade * (angle + (next[axis] - angle) * (position - index)));
}
