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

// Slow shifts between held poses. Pitch, yaw, and roll are in degrees.
const center = [0, 0, 0];
const lookLeft = [-0.7, 2.4, 0.8];
const lookRight = [1.1, -1.9, -0.6];
const lift = [-1.4, 0.5, -0.3];
const idlePoses = [
  [0, center],
  [1.1, center],
  [2.7, lookLeft],
  [5.4, lookLeft],
  [7.2, lookRight],
  [10.8, lookRight],
  [12.4, lift],
  [14.2, lift],
  [16, center],
];

export function idleHeadAt(time) {
  const phase = time % idlePoses.at(-1)[0];
  const next = idlePoses.findIndex(([start]) => start > phase);
  const [start, from] = idlePoses[next - 1];
  const [end, to] = idlePoses[next];
  const t = (phase - start) / (end - start);
  const ease = t * t * (3 - 2 * t);
  return from.map((angle, axis) => angle + (to[axis] - angle) * ease);
}
