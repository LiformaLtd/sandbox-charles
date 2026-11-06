import './style.css';
import { createJames } from './rig.js';
import { createStreamPlayback } from './stream-playback.js';
import { concatPcm, encodeWav } from 'phonon-tts';
import { animationBlocks, loadModel } from './inference.js';
import { blinkAt, idleHeadAt } from './idle-motion.js';

const viewport = document.querySelector('#viewport');
const form = document.querySelector('#speech-form');
const input = document.querySelector('#speech');
const generate = document.querySelector('#generate');
const status = document.querySelector('#status');
const progress = document.querySelector('#progress');
const audio = document.querySelector('#audio');
const playback = document.querySelector('#playback');
const face = document.querySelector('#face');
const mouth = document.querySelector('#mouth');
const count = document.querySelector('#count');
const voiceSwitch = document.querySelector('#voice-switch');
const voiceDescription = document.querySelector('#voice-description');
const playbackEngine = document.querySelector('#playback-engine');
const stopStream = document.querySelector('#stop-stream');
const duration = document.querySelector('#duration');
const headMotion = document.querySelector('#head-motion');
const headMotionValue = document.querySelector('#head-motion-value');

let rig;
let frames = [];
let headRotations = [];
let activeFrame = -1;
let viewerStarted;
let lastPoseTime;
let speechWeight = 0;
let lastSpeechRotation = [0, 0, 0];
let sourceUrl;
let livePlayback;
let liveRun;
stopStream.addEventListener('click', () => liveRun?.abort());

function report(message, error = false) {
  status.textContent = message;
  status.classList.toggle('error', error);
}

function updateCount() {
  count.textContent = `${input.value.length} / ${input.maxLength}`;
}
input.addEventListener('input', updateCount);
updateCount();

const engines = {
  phonon: { label: 'Phonon', voice: 'Easton' },
  supertonic: { label: 'Supertonic 3', voice: 'M1' },
  pocket: { label: 'Pocket TTS', voice: 'Alba' },
};

function updateVoiceDescription() {
  const { voice } = engines[form.elements.engine.value];
  voiceDescription.textContent = `${voice} · generated in this browser`;
}
voiceSwitch.addEventListener('change', updateVoiceDescription);
updateVoiceDescription();

function updateHeadMotion() {
  const percent = `${headMotion.value}%`;
  headMotionValue.textContent = percent;
  headMotion.setAttribute('aria-valuetext', percent);
}
headMotion.addEventListener('input', updateHeadMotion);
updateHeadMotion();

function setView(closeup) {
  rig?.view(closeup);
  face.setAttribute('aria-pressed', String(!closeup));
  mouth.setAttribute('aria-pressed', String(closeup));
}
face.addEventListener('click', () => setView(false));
mouth.addEventListener('click', () => setView(true));

function draw(now) {
  requestAnimationFrame(draw);
  viewerStarted ??= now;
  if (lastPoseTime === undefined || Math.floor(now * 0.06) > Math.floor(lastPoseTime * 0.06)) {
    const elapsed = (now - viewerStarted) / 1000;
    const dt = (now - (lastPoseTime ?? now)) / 1000;
    lastPoseTime = now;
    const { time, playing } = livePlayback?.playhead() ?? {
      time: audio.ended ? -1 : audio.currentTime,
      playing: !audio.paused && !audio.ended && !audio.seeking && audio.readyState >= audio.HAVE_FUTURE_DATA,
    };
    // Keep the selected facial frame for paused inspection after a seek.
    const showFrame = frames.length && (time > 0 || (time === 0 && playing));
    const next = showFrame ? Math.min(frames.length - 1, Math.floor(time * 60)) : -1;
    const speechRotation = playing ? headRotations[next] : undefined;
    if (speechRotation) lastSpeechRotation = speechRotation;
    const target = speechRotation ? 1 : 0;
    speechWeight += (target - speechWeight) * (1 - Math.exp(-dt / 0.15));
    if (Math.abs(target - speechWeight) < 0.001) speechWeight = target;
    const strength = headMotion.valueAsNumber / 100;
    const rotation = strength ? idleHeadAt(elapsed).map((angle, axis) =>
      strength * (angle + (lastSpeechRotation[axis] - angle) * speechWeight)) : undefined;
    rig.pose(frames[next], blinkAt(elapsed), rotation);
    if (next !== activeFrame) {
      activeFrame = next;
      viewport.dataset.frame = next < 0 ? 'neutral' : String(next);
    }
  }
  rig.draw();
}

createJames(viewport).then((viewer) => {
  rig = viewer;
  generate.disabled = false;
  report('Ready. The model loads on the first generation.');
  requestAnimationFrame(draw);
}).catch((error) => {
  console.error(error);
  report(error.message, true);
});

function setReplay(wav, samples, label) {
  const nextUrl = URL.createObjectURL(wav);
  audio.src = nextUrl;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = nextUrl;
  playback.hidden = false;
  audio.hidden = false;
  playbackEngine.textContent = `Playback · ${label}`;
  duration.textContent = `${(samples / 16000).toFixed(1)} s · ${frames.length} frames`;
}

async function streamSpeech(text, engine) {
  const { label } = engines[engine];
  const previousFrames = frames;
  const previousHeadRotations = headRotations;
  const previousPlayback = !playback.hidden;
  const previousLabel = playbackEngine.textContent;
  const previousDuration = duration.textContent;
  const run = new AbortController();
  const { signal } = run;
  liveRun = run;
  const context = new AudioContext({ latencyHint: 'interactive' });
  let complete = false;
  let speech;
  try {
    // Resume during the click event so playback can start after async setup.
    const resumed = context.resume();
    livePlayback = createStreamPlayback(context);
    frames = [];
    headRotations = [];
    playback.hidden = false;
    audio.hidden = true;
    stopStream.hidden = false;
    playbackEngine.textContent = `Live playback · ${label}`;
    duration.textContent = '';
    const loadingSpeech = engine === 'phonon'
      ? import('./phonon.js').then(({ openPhononStream }) => openPhononStream(text, report, signal))
      : import('./onnx-tts.js').then(({ openOnnxStream }) => openOnnxStream(engine, text, report, signal));
    const [model, stream] = await Promise.all([loadModel(), loadingSpeech, resumed]);
    speech = stream;
    const blocks = [];
    let samples = 0;
    for await (const block of animationBlocks(model, speech.chunks, signal)) {
      signal.throwIfAborted();
      headRotations.push(...block.rotations);
      frames.push(...block.frames);
      blocks.push(block.pcm);
      samples += block.pcm.length;
      livePlayback.enqueue(block.pcm);
      if (blocks.length === 1) {
        progress.hidden = true;
        report('Streaming speech and animation…');
      }
      duration.textContent = `${(samples / 16000).toFixed(1)} s ready`;
    }
    await speech.done;
    signal.throwIfAborted();
    if (!samples || frames.length !== Math.ceil(samples * 60 / 16000)) {
      throw new Error(`${label} did not produce a complete animation track.`);
    }
    report('Finishing playback…');
    await livePlayback.finish(signal);
    setReplay(encodeWav(concatPcm(blocks), 16000), samples, label);
    complete = true;
    report('Animation ready to replay.');
  } catch (error) {
    if (signal.aborted) report('Streaming stopped.');
    else throw error;
  } finally {
    run.abort();
    liveRun = undefined;
    livePlayback = undefined;
    stopStream.hidden = true;
    await context.close();
    if (!complete) {
      frames = previousFrames;
      headRotations = previousHeadRotations;
      audio.hidden = false;
      playback.hidden = !previousPlayback;
      playbackEngine.textContent = previousLabel;
      duration.textContent = previousDuration;
    }
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || generate.disabled) return;
  const engine = form.elements.engine.value;
  audio.pause();
  generate.disabled = true;
  voiceSwitch.disabled = true;
  progress.removeAttribute('value');
  progress.hidden = false;
  report(`Preparing ${engines[engine].label} in your browser…`);
  try {
    await streamSpeech(text, engine);
  } catch (error) {
    console.error(error);
    report(error.message, true);
  } finally {
    progress.hidden = true;
    voiceSwitch.disabled = false;
    generate.disabled = false;
  }
});

window.addEventListener('pagehide', () => {
  liveRun?.abort();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
});
