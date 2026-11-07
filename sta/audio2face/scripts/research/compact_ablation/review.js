import { createJames } from './src/rig.js';
import { blinkAt, idleHeadAt, loadIdleHead } from './src/idle-motion.js';
const clip = document.querySelector('#clip');
const audio = document.querySelector('#audio');
const scrub = document.querySelector('#scrub');
const steady = document.querySelector('#steady');
const closeup = document.querySelector('#closeup');
const status = document.querySelector('#status');
const play = document.querySelector('#play');
play.addEventListener('click', () => {
  if (audio.paused) audio.play().catch(fail); else audio.pause();
});
audio.addEventListener('play', () => { play.textContent = 'Pause matched speech'; });
audio.addEventListener('pause', () => { play.textContent = 'Play matched speech'; });
let rigs = [], tracks = [], frames = 0, loaded = false, idle;
let selection = 0;
function fail(error) { console.error(error); status.classList.add('error'); status.textContent = error.message; }
async function fetchAsset(path, format = 'json') {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Missing comparison asset: ${path} (${response.status}).`);
  return response[format]();
}
function seek(frame) {
  frame = Math.max(0, Math.min(frames - 1, frame));
  audio.pause(); audio.currentTime = frame / 60; scrub.value = frame;
}
try {
  const manifest = await fetchAsset('/review/manifest.json');
  if (manifest.provisional) document.querySelector('h1').textContent += ' · provisional first seed';
  for (const [index, sample] of manifest.cases.entries()) {
    const option = document.createElement('option'); option.value = index; option.textContent = sample.title; clip.append(option);
  }
  rigs = await Promise.all(manifest.models.map(async ({ title }) => {
    const panel = document.createElement('article');
    const heading = document.createElement('h2'); heading.textContent = title;
    const viewport = document.createElement('div'); viewport.className = 'viewport';
    panel.append(heading, viewport); document.querySelector('#panels').append(panel);
    const rig = await createJames(viewport, { compact: true }); rig.view(true); return rig;
  }));
  idle = await loadIdleHead();
  async function choose() {
    const request = ++selection; loaded = false; audio.pause(); clip.disabled = true; scrub.disabled = true; play.disabled = true;
    const sample = manifest.cases[Number(clip.value)];
    status.textContent = 'Loading matched tracks…';
    const arrays = await Promise.all(manifest.models.map(({ id }) => fetchAsset(`/review/${sample.stem}-${id}.f32`, 'arrayBuffer')));
    if (request !== selection) return;
    tracks = arrays.map(bytes => {
      const data = new Float32Array(bytes);
      if (data.length !== sample.frames * 52 || !data.every(Number.isFinite)) throw new Error('Invalid comparison track.');
      return Array.from({ length: sample.frames }, (_, i) => data.subarray(i * 52, (i + 1) * 52));
    });
    document.querySelector('#gap-plot').src = `/review/${sample.stem}-gaps.png`;
    frames = sample.frames; audio.src = `/review/${sample.stem}.wav`; audio.load();
    scrub.max = frames - 1; scrub.value = 0; document.querySelector('#dialogue').textContent = sample.text;
    document.querySelector('#snapshots').replaceChildren();
    for (const { frame, title } of sample.snapshots) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = title; button.dataset.frame = frame;
      button.addEventListener('click', () => seek(frame)); document.querySelector('#snapshots').append(button);
    }
    loaded = true; clip.disabled = false; scrub.disabled = false; play.disabled = false; status.textContent = 'Ready. Play or scrub the same speech across all models.';
  }
  clip.addEventListener('change', () => choose().catch(fail));
  scrub.addEventListener('input', () => seek(Number(scrub.value)));
  closeup.addEventListener('change', () => rigs.forEach(rig => rig.view(closeup.checked)));
  await choose();
  const began = performance.now();
  function draw(now) {
    const time = (now - began) / 1000;
    const frame = Math.min(frames - 1, Math.max(0, Math.floor(audio.currentTime * 60 + 1e-5)));
    const rotation = steady.checked ? undefined : idleHeadAt(idle, time).map(angle => angle * 0.6);
    const blink = steady.checked ? 0 : blinkAt(time);
    for (const [i, rig] of rigs.entries()) {
      rig.pose(loaded ? tracks[i][frame] : undefined, blink, rotation); rig.draw();
    }
    if (loaded) {
      scrub.value = frame; document.querySelector('#position').textContent = `${frame} / ${frames - 1} · ${(frame / 60).toFixed(3)} s`;
      document.body.dataset.frame = frame;
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
} catch (error) { fail(error); }
