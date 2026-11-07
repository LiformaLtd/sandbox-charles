import './style.css';
import { dialogueSamples } from './dialogue-samples.js';
import { createJames } from './rig.js';
import { createStreamPlayback } from './stream-playback.js';
import { concatPcm, encodeWav } from 'phonon-tts';
import { animationBlocks, loadModel } from './inference.js';
import { blinkAt, idleHeadAt, loadIdleHead } from './idle-motion.js';

const viewport = document.querySelector('#viewport');
const form = document.querySelector('#speech-form');
const input = document.querySelector('#speech');
const generate = document.querySelector('#generate');
const dialoguePrev = document.querySelector('#dialogue-prev');
const dialogueNext = document.querySelector('#dialogue-next');
const dialogueCounter = document.querySelector('#dialogue-counter');
const dialogueTitle = document.querySelector('#dialogue-sample');
let sampleIndex = 0;
input.value = dialogueSamples[sampleIndex].text;
const prepareTts = document.querySelector('#prepare-tts');
const prepareSta = document.querySelector('#prepare-sta');
const status = document.querySelector('#status');
const progress = document.querySelector('#progress');
const audio = document.querySelector('#audio');
const playback = document.querySelector('#playback');
const face = document.querySelector('#face');
const mouth = document.querySelector('#mouth');
const count = document.querySelector('#count');
const voiceSwitch = document.querySelector('#voice-switch');
const animationSwitch = document.querySelector('#animation-switch');
const runtimeSwitch = document.querySelector('#runtime-switch');
const voiceDescription = document.querySelector('#voice-description');
const playbackEngine = document.querySelector('#playback-engine');
const stopStream = document.querySelector('#stop-stream');
const duration = document.querySelector('#duration');
const headMotion = document.querySelector('#head-motion');
const headMotionValue = document.querySelector('#head-motion-value');

// iPad Safari can send a desktop user agent.
const appleMobile = /iPhone|iPad|iPod/.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const params = new URLSearchParams(location.search);
const staMode = params.get('sta');
const requestedModel = params.get('model');
const defaultModel = 'compact-v4';
const model = ['geometry', 'compact', 'compact-v2', 'compact-v3', 'compact-v4'].includes(requestedModel) ? requestedModel : defaultModel;
const compact = model !== 'geometry';
const cpuOnly = staMode === 'cpu';
const draftKey = 'james-model-switch-draft';
form.elements.animation.value = model;
form.elements.runtime.value = cpuOnly ? 'cpu' : 'webgpu';
animationSwitch.disabled = runtimeSwitch.disabled = true;
if (cpuOnly) document.querySelector('#sta-runtime').textContent = '60 fps · CPU';
let startInCloseup = false;
try {
  const draft = sessionStorage.getItem(draftKey);
  sessionStorage.removeItem(draftKey);
  if (draft) {
    const { dialogue, engine, motion, closeup, sample } = JSON.parse(draft);
    if (Number.isInteger(sample) && sample >= 0 && sample < dialogueSamples.length) sampleIndex = sample;
    input.value = dialogue;
    form.elements.engine.value = engine;
    headMotion.value = motion;
    startInCloseup = closeup === true;
  }
} catch {
  // Switching models also works when browser storage is unavailable.
}
async function switchAnimationSettings(event) {
  try {
    sessionStorage.setItem(draftKey, JSON.stringify({
      dialogue: input.value,
      sample: sampleIndex,
      engine: form.elements.engine.value,
      motion: headMotion.value,
      closeup: mouth.getAttribute('aria-pressed') === 'true',
    }));
  } catch {
    // Keep the selector usable when browser storage is unavailable.
  }
  const url = new URL(location.href);
  if (event.currentTarget === runtimeSwitch) {
    url.searchParams.set('sta', form.elements.runtime.value);
  } else {
    const choice = form.elements.animation.value;
    if (choice === defaultModel) url.searchParams.delete('model');
    else url.searchParams.set('model', choice);
  }
  setBusy(true);
  try {
    await selectedTts?.release();
  } finally {
    // Reload so the previous model's WASM heap and GPU device can be reclaimed.
    location.replace(url);
  }
}
animationSwitch.addEventListener('change', switchAnimationSettings);
runtimeSwitch.addEventListener('change', switchAnimationSettings);

let rig;
let frames = [];
let headRotations = [];
let activeFrame = -1;
let viewerStarted;
let lastPoseTime;
let idleTrack;
let idleStarted;
let speechWeight = 0;
let lastSpeechRotation = [0, 0, 0];
let sourceUrl;
let lastClip;
let livePlayback;
let liveRun;
let selectedTts;

const interruptionKey = 'james-mobile-stage';
let previousStage;
let savedStage;
let trackStage = false;
if (appleMobile) {
  try {
    previousStage = sessionStorage.getItem(interruptionKey);
    sessionStorage.removeItem(interruptionKey);
  } catch {
    // The demo can run when browser storage is unavailable.
  }
}
stopStream.addEventListener('click', () => liveRun?.abort());

function report(message, error = false) {
  if (trackStage && !error) {
    const stage = message.replace(/… \d+%$/, '…');
    if (stage !== savedStage) {
      savedStage = stage;
      try {
        sessionStorage.setItem(interruptionKey, stage);
      } catch {
        // Progress is still displayed when browser storage is unavailable.
      }
    }
  }
  status.textContent = message;
  status.classList.toggle('error', error);
}

function updateCount() {
  count.textContent = `${input.value.length} / ${input.maxLength}`;
  dialogueCounter.textContent = `${sampleIndex + 1} / ${dialogueSamples.length}`;
  dialogueTitle.textContent = input.value === dialogueSamples[sampleIndex].text
    ? dialogueSamples[sampleIndex].title : 'Custom dialogue';
}
input.addEventListener('input', updateCount);
updateCount();

function stepDialogue(direction) {
  if (generate.disabled) return;
  sampleIndex = (sampleIndex + direction + dialogueSamples.length) % dialogueSamples.length;
  input.value = dialogueSamples[sampleIndex].text;
  updateCount();
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = undefined;
  lastClip = undefined;
  frames = [];
  headRotations = [];
  activeFrame = -1;
  playback.hidden = true;
  report('Ready to generate speech.');
}
dialoguePrev.addEventListener('click', () => stepDialogue(-1));
dialogueNext.addEventListener('click', () => stepDialogue(1));

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
    const transitionSeconds = speechRotation ? 0.15 : 0.38;
    speechWeight += (target - speechWeight) * (1 - Math.exp(-dt / transitionSeconds));
    if (Math.abs(target - speechWeight) < 0.001) speechWeight = target;
    const strength = headMotion.valueAsNumber / 100;
    if (idleTrack) idleStarted ??= now;
    const idle = idleHeadAt(idleTrack, (now - (idleStarted ?? now)) / 1000);
    const rotation = strength ? idle.map((angle, axis) =>
      strength * (angle + (lastSpeechRotation[axis] - angle) * speechWeight)) : undefined;
    rig.pose(frames[next], blinkAt(elapsed), rotation);
    if (next !== activeFrame) {
      activeFrame = next;
      viewport.dataset.frame = next < 0 ? 'neutral' : String(next);
    }
  }
  rig.draw();
}

createJames(viewport, { compact }).then((viewer) => {
  rig = viewer;
  if (startInCloseup) setView(true);
  setBusy(false);
  report(previousStage
    ? `The previous run was interrupted at “${previousStage}”`
    : 'Ready. The model loads on the first generation.');
  requestAnimationFrame(draw);
  loadIdleHead().then((track) => {
    idleTrack = track;
  }).catch((error) => {
    console.error('Could not load idle head motion.', error);
    if (!liveRun && !sourceUrl) report(`Could not load idle head motion. ${error.message}`, true);
  });
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
    const prepareModel = () => loadModel({
      model,
      cpuOnly,
      mobile: appleMobile,
      report: appleMobile ? (message) => {
        if (!signal.aborted) report(message);
      } : undefined,
    });
    const startSpeech = async () => {
      const tts = await selectSpeechEngine(engine);
      signal.throwIfAborted();
      return engine === 'phonon'
        ? tts.openPhononStream(text, report, signal, appleMobile)
        : tts.openOnnxStream(engine, text, report, signal);
    };
    const loadingModel = prepareModel();
    // On mobile, avoid overlapping cold startup allocations. Both sessions
    // stay loaded, so preparing them separately skips startup here.
    const loadingSpeech = appleMobile ? loadingModel.then(startSpeech) : startSpeech();
    const [loadedModel, speech] = await Promise.all([loadingModel, loadingSpeech, resumed]);
    signal.throwIfAborted();
    const blocks = [];
    let samples = 0;
    for await (const block of animationBlocks(loadedModel, speech.chunks, signal)) {
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
    const pcm = concatPcm(blocks);
    report('Finishing playback…');
    await livePlayback.finish(signal);
    setReplay(encodeWav(pcm, 16000), samples, label);
    lastClip = { text, engine };
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

function setBusy(busy) {
  generate.disabled = busy;
  dialoguePrev.disabled = dialogueNext.disabled = busy;
  prepareTts.disabled = busy;
  prepareSta.disabled = busy;
  voiceSwitch.disabled = busy;
  animationSwitch.disabled = busy;
  runtimeSwitch.disabled = busy;
}

async function runWithProgress(message, action) {
  if (generate.disabled) return;
  setBusy(true);
  progress.removeAttribute('value');
  progress.hidden = false;
  trackStage = appleMobile;
  savedStage = undefined;
  report(message);
  try {
    await action();
  } catch (error) {
    console.error(error);
    report(error.message, true);
  } finally {
    if (trackStage) {
      trackStage = false;
      try {
        sessionStorage.removeItem(interruptionKey);
      } catch {
        // There is no saved stage to clear when browser storage is unavailable.
      }
    }
    progress.hidden = true;
    setBusy(false);
  }
}

async function selectSpeechEngine(engine) {
  const tts = engine === 'phonon' ? await import('./phonon.js') : await import('./onnx-tts.js');
  if (appleMobile && selectedTts?.engine !== engine) {
    await selectedTts?.release();
    selectedTts = {
      engine,
      release: engine === 'phonon' ? tts.releasePhonon : () => tts.releaseOnnxTts(engine),
    };
  }
  return tts;
}

prepareTts.addEventListener('click', () => {
  const engine = form.elements.engine.value;
  const { label } = engines[engine];
  runWithProgress(`Preparing ${label} in your browser…`, async () => {
    const tts = await selectSpeechEngine(engine);
    if (engine === 'phonon') {
      await tts.preparePhonon(report, appleMobile);
    } else {
      await tts.prepareOnnxTts(engine, report);
    }
    report(`${label} is ready.`);
  });
});

prepareSta.addEventListener('click', () => {
  runWithProgress('Preparing the STA model…', async () => {
    await loadModel({ model, cpuOnly, mobile: appleMobile, report });
    report('STA is ready.');
  });
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  const engine = form.elements.engine.value;
  if (generate.disabled) return;
  if (appleMobile && compact && engine === 'phonon'
      && lastClip?.text === text && lastClip.engine === engine) {
    audio.currentTime = 0;
    report('Replaying animation…');
    audio.play().catch((error) => report(`Could not replay the clip. ${error.message}`, true));
    return;
  }
  runWithProgress(`Preparing ${engines[engine].label} in your browser…`, () => {
    audio.pause();
    return streamSpeech(text, engine);
  });
});

audio.addEventListener('ended', () => {
  if (status.textContent === 'Replaying animation…') report('Animation ready to replay.');
});

window.addEventListener('pagehide', () => {
  liveRun?.abort();
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
});
