import './expressive-review.css';
import * as ort from 'onnxruntime-web/wasm';
import { createJames } from './rig.js';

const $ = (id) => document.getElementById(id);
const settings = $('settings');
const status = $('status');
const audio = $('audio');
const views = [$('baseline-view'), $('candidate-view'), $('teacher-view')];
const rate = 16000;
let manifest, rigs, baseline, candidate, pcm, audioUrl, ownedUrl, lipBank;
let frames, teacherFrames, lastFrame = -1, manual = false;

function report(message) { status.textContent = message; status.dataset.error = 'false'; }
function fail(error) {
  console.error('[James expressive review]', error);
  status.textContent = error.message;
  status.dataset.error = 'true';
}
async function fetchOk(url, format = 'arrayBuffer') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}.`);
  return response[format]();
}
async function checkedBytes(url, expected) {
  const bytes = await fetchOk(url);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const actual = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
  if (actual !== expected) throw new Error(`Asset checksum mismatch: ${url}. Rerun prepare_review.py.`);
  return bytes;
}
function invalidate() {
  audio.pause();
  audio.hidden = true;
  audio.removeAttribute('src');
  audio.load();
  frames = undefined;
  teacherFrames = undefined;
  delete views[2].dataset.frame;
  delete document.body.dataset.teacherSha;
  lastFrame = -1;
  manual = false;
}
function busy(value) { settings.disabled = value; $('show-target').disabled = value; }
function teacherReference() {
  const id = $('sample').value;
  const emotion = $('emotion').value === 'neutral' || Number($('strength').value) === 0
    ? 'neutral' : Number($('strength').value) === 100 ? $('emotion').value : undefined;
  if (!emotion) return undefined;
  const reference = manifest?.teacherReviews?.[`${id}--${emotion}`] || manifest?.teacherReviews?.[id]
    || (manifest?.teacherReview?.sample === id ? manifest.teacherReview : undefined);
  return reference?.emotion === emotion ? reference : undefined;
}
function teacherAvailable() {
  return Boolean(teacherReference());
}
function updateComparison() {
  const available = teacherAvailable();
  $('comparison-source').querySelector('[value="teacher"]').disabled = !available;
  if (!available) $('comparison-source').value = 'v4';
  const teacher = $('comparison-source').value === 'teacher';
  $('baseline-card').hidden = teacher;
  $('teacher-card').hidden = !teacher;
  $('teacher-summary').textContent = teacher
    ? `Left: prerecorded official NVIDIA teacher, ${teacherReference().emotion} James, seed 0; raw 3D output without ARKit fitting. Right: our browser model. Same audio and 60 fps clock; no motion amplification.`
    : 'Teacher replay is available for matching staged recordings and controls. Released v4 is shown on the left.';
  lastFrame = -1;
}
function gazeAngles(frame) {
  if (!frame) return [0, 0, 0, 0];
  return [25 * (frame[8] - frame[11]), 25 * (frame[9] - frame[10]),
    25 * (frame[1] - frame[4]), 25 * (frame[3] - frame[2])];
}
async function selectSample() {
  invalidate();
  busy(true);
  try {
    if (ownedUrl) { URL.revokeObjectURL(ownedUrl); ownedUrl = undefined; }
    const sample = manifest.samples.find((s) => s.id === $('sample').value);
    if (!sample) throw new Error('Select a recorded speech sample or upload audio.');
    const bytes = await checkedBytes(sample.pcm, sample.sha256);
    if (bytes.byteLength !== sample.samples * 4) throw new Error('The recording has the wrong sample count.');
    pcm = new Float32Array(bytes);
    audioUrl = sample.wav;
    $('upload').value = '';
    $('transcript').textContent = `“${sample.text}”`;
    updateComparison();
    report('Ready. Choose expression controls, then generate the comparison.');
  } finally { busy(false); }
}
function changed() {
  invalidate();
  updateComparison();
  $('strength-value').textContent = `${$('strength').value}%`;
  report('Settings changed. Generate the comparison to apply them.');
}

async function generate() {
  if (!pcm?.length) throw new Error('Load speech first.');
  invalidate(); busy(true);
  try {
    const count = Math.ceil(pcm.length * 60 / rate);
    const current = new Float32Array(count * 52);
    const expressive = new Float32Array(count * 52);
    const emotion = new Float32Array(60 * 10);
    const emotionIndex = manifest.emotions.indexOf($('emotion').value);
    if (emotionIndex >= 0) for (let f = 0; f < 60; f++) emotion[f * 10 + emotionIndex] = Number($('strength').value) / 100;
    const style = new Float32Array(60 * 3);
    const styleIndex = manifest.gazeStyles.indexOf($('gaze-style').value);
    for (let f = 0; f < 60; f++) style[f * 3 + styleIndex] = 1;
    const emotionTensor = new ort.Tensor('float32', emotion, [1, 60, 10]);
    const styleTensor = new ort.Tensor('float32', style, [1, 60, 3]);
    let cursor = 0, drift = 0;
    const begun = performance.now();
    try {
      for (let start = -8000; start < pcm.length - 4000; start += 8000) {
        const window = new Float32Array(rate);
        const lo = Math.max(0, start), hi = Math.min(pcm.length, start + rate);
        window.set(pcm.subarray(lo, hi), lo - start);
        const input = new ort.Tensor('float32', window, [1, rate]);
        let oldResult, fullResult;
        try {
          oldResult = await baseline.run({ pcm: input });
          fullResult = await candidate.run({ pcm: input, emotion: emotionTensor, gaze_style: styleTensor });
          const old = oldResult.blendshapes.data;
          const full = fullResult.blendshapes.data;
          const articulation = fullResult.articulation.data;
          if (old.length !== 3120 || full.length !== 3120 || articulation.length !== 3120) throw new Error('Unexpected model output dimensions.');
          for (let i = 0; i < old.length; i++) {
            if (!Number.isFinite(full[i]) || !Number.isFinite(old[i]) || !Number.isFinite(articulation[i])
              || full[i] < 0 || full[i] > 1) throw new Error('The model produced an invalid coefficient.');
            drift = Math.max(drift, Math.abs(articulation[i] - old[i]));
          }
          if (drift > 0.00003) throw new Error('The protected articulation differs from current v4. Comparison stopped.');
          for (let f = 15; f < 45; f++) {
            const sample = start + Math.floor(f * rate / 60);
            if (sample < 0 || sample >= pcm.length) continue;
            current.set(old.subarray(f * 52, (f + 1) * 52), cursor * 52);
            expressive.set(full.subarray(f * 52, (f + 1) * 52), cursor * 52);
            cursor++;
          }
        } finally {
          input.dispose();
          for (const result of [oldResult, fullResult]) if (result) for (const tensor of Object.values(result)) tensor.dispose();
        }
        report(`Generating synchronized animation… ${Math.round(100 * cursor / count)}%`);
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } finally { emotionTensor.dispose(); styleTensor.dispose(); }
    if (cursor !== count) throw new Error('The animation track is incomplete.');
    if ($('comparison-source').value === 'teacher') {
      const reference = teacherReference();
      const sample = manifest.samples.find((s) => s.id === $('sample').value);
      if (!teacherAvailable() || reference.audioSha256 !== sample?.sha256
        || reference.frames !== count || reference.channels !== 88831 || reference.fps !== 60) throw new Error('Teacher replay does not match this audio/control timeline.');
      const bytes = await checkedBytes(reference.url, reference.sha256);
      if (bytes.byteLength !== count * 88831 * 4) throw new Error('Incomplete teacher replay.');
      teacherFrames = new Float32Array(bytes);
      if (teacherFrames.some((v) => !Number.isFinite(v))) throw new Error('Invalid teacher replay geometry.');
      document.body.dataset.teacherSha = reference.teacherModelSha256;
    }
    let accuracy = 'Lip-gap RMS unmeasured for these controls/audio.';
    const condition = emotionIndex < 0 || Number($('strength').value) === 0 ? 'neutral'
      : Number($('strength').value) === 100 ? $('emotion').value : undefined;
    const key = `${$('sample').value}--${condition}`;
    const reference = lipBank?.references[key];
    if (reference) {
      if (reference.length !== count) throw new Error('Lip-gap reference has a different audio frame count.');
      let squared = 0;
      for (let f = 0; f < count; f++) for (let g = 0; g < 9; g++) {
        let gap = lipBank.rest[g];
        for (let c = 0; c < 52; c++) gap += expressive[f * 52 + c] * lipBank.basis[c][g];
        squared += (gap - reference[f][g]) ** 2;
      }
      const rms = 10 * Math.sqrt(squared / (count * 9));
      const limit = lipBank.limits[key];
      document.body.dataset.lipRmsMm = String(rms);
      document.body.dataset.lipRmsLimitMm = String(limit);
      if (rms > limit && !manifest.candidate.reviewOnly) throw new Error(`Active lip-gap RMS ${rms.toFixed(3)} mm exceeds ${limit.toFixed(3)} mm regression limit.`);
      accuracy = `Active lip-gap RMS ${rms.toFixed(3)} mm · limit ${limit.toFixed(3)} mm (matched prior +5%).`;
      if (rms > limit) accuracy += ' Above tolerance · visual review allowed.';
    } else {
      delete document.body.dataset.lipRmsMm;
      delete document.body.dataset.lipRmsLimitMm;
    }
    frames = [current, expressive];
    document.body.dataset.frameCount = String(count);
    document.body.dataset.articulationDrift = String(drift);
    document.body.dataset.inferenceMs = String(performance.now() - begun);
    audio.src = audioUrl; audio.hidden = false; audio.load();
    report(`Ready · ${count} synchronized frames · ${accuracy} Frozen speech-core reference verified. CPU research runtime.`);
    try { await audio.play(); }
    catch (error) {
      if (error.name !== 'NotAllowedError') throw error;
      report('Animation ready. Press play in the audio controls.');
    }
  } finally { busy(false); }
}

function showTarget() {
  invalidate(); manual = true;
  $('comparison-source').value = 'v4'; updateComparison();
  const weights = new Float32Array(52);
  const channel = Number($('target').value);
  weights[channel] = Number($('amount').value) / 100;
  if (channel === 51) weights[17] = 0.4;
  rigs.slice(0, 2).forEach((rig) => rig.pose(weights, 0));
  views.slice(0, 2).forEach((view) => { view.dataset.frame = 'manual'; view.dataset.target = String(channel); });
  report(`Manual rig inspection: ${manifest.channels[channel]}. Both rigs receive the same coefficients; this is not model output.`);
}

function render() {
  if (!manual && frames) {
    const count = frames[0].length / 52;
    const index = Math.max(0, Math.min(count - 1, Math.floor(audio.currentTime * 60)));
    if (index !== lastFrame) {
      rigs.slice(0, 2).forEach((rig, i) => rig.pose(frames[i].subarray(index * 52, (index + 1) * 52), 0));
      views.slice(0, 2).forEach((view) => { view.dataset.frame = String(index); });
      if (teacherFrames) {
        rigs[2].pose(teacherFrames.subarray(index * 88831, (index + 1) * 88831), 0);
        views[2].dataset.frame = String(index);
      }
      lastFrame = index;
    }
  }
  rigs.forEach((rig, i) => { if (!views[i].closest('article').hidden) rig.draw(); });
  requestAnimationFrame(render);
}

async function setup() {
  ort.env.wasm.numThreads = 1;
  manifest = await fetchOk('/assets/expressive/review.json', 'json');
  document.body.dataset.candidateSha = manifest.candidate.sha256;
  $('candidate-summary').textContent = `Candidate: ${manifest.candidate.label || 'Full-face research'} · ${manifest.candidate.sha256.slice(0, 12)}.`
    + (manifest.candidate.reviewOnly ? ' Visual review only; not promoted. '
      + (manifest.candidate.reviewResults || []).map((r) => `${r.sample} at ${r.emotion} 100%: +${r.changePercent.toFixed(1)}% lip-gap RMS error.`).join(' ') : '');
  if (manifest.candidate.reviewOnly) $('candidate-summary').textContent += ' ' + (manifest.candidate.reviewWarnings || []).join(' ');
  $('candidate-summary').textContent += ' ' + (manifest.candidate.reviewNotes || []).join(' ');
  if (manifest.lipReference) lipBank = JSON.parse(new TextDecoder().decode(await checkedBytes(manifest.lipReference.url, manifest.lipReference.sha256)));
  if (manifest.channels.length !== 52) throw new Error('The research rig must have 52 channels.');
  for (const s of manifest.samples) $('sample').add(new Option(s.label, s.id));
  for (const emotion of ['neutral', ...manifest.emotions]) $('emotion').add(new Option(emotion, emotion));
  if (manifest.teacherReview) {
    $('sample').value = manifest.teacherReview.sample;
    $('emotion').value = manifest.teacherReview.emotion;
  }
  updateComparison();
  manifest.channels.forEach((name, i) => $('target').add(new Option(name, String(i))));
  $('target').value = '43';
  rigs = await Promise.all([
    createJames(views[0], { compact: true, assetsBase: '/assets/expressive', blendshapeUrl: '/assets/expressive/baseline-blendshapes.bin' }),
    createJames(views[1], { compact: true, assetsBase: '/assets/expressive', blendshapeUrl: '/assets/expressive/full-blendshapes.bin', compactGaze: gazeAngles }),
    createJames(views[2], { assetsBase: '/assets/expressive' }),
  ]);
  rigs.forEach((rig) => rig.view(false, { target: [0, 5, 4], distance: 48 }));
  requestAnimationFrame(render);
  report('Loading current v4…');
  baseline = await ort.InferenceSession.create(await checkedBytes(manifest.baseline.url, manifest.baseline.sha256), { executionProviders: ['wasm'] });
  report('Loading the full-face candidate…');
  candidate = await ort.InferenceSession.create(await checkedBytes(manifest.candidate.url, manifest.candidate.sha256), { executionProviders: ['wasm'] });
  await selectSample();
  document.body.dataset.ready = 'true';
}

$('generate').addEventListener('click', () => generate().catch(fail));
$('sample').addEventListener('change', () => selectSample().catch(fail));
$('comparison-source').addEventListener('change', changed);
for (const id of ['emotion', 'gaze-style']) $(id).addEventListener('change', changed);
$('strength').addEventListener('input', changed);
$('detail').addEventListener('click', () => rigs?.forEach((rig) => rig.view(false, { target: [0, 5, 4], distance: 48 })));
$('mouth-detail').addEventListener('click', () => rigs?.forEach((rig) => rig.view(true)));
$('whole').addEventListener('click', () => rigs?.forEach((rig) => rig.view(false)));
$('show-target').addEventListener('click', showTarget);
for (const id of ['target', 'amount']) $(id).addEventListener('input', () => { if (manual) showTarget(); });
$('upload').addEventListener('change', async () => {
  const file = $('upload').files[0];
  if (!file) return;
  invalidate(); busy(true);
  let context;
  try {
    if (file.size > 20_000_000) throw new Error('Upload audio up to 20 MB and 30 seconds.');
    context = new AudioContext({ sampleRate: rate });
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    if (decoded.sampleRate !== rate) throw new Error('The browser could not decode audio at 16 kHz.');
    if (decoded.duration > 30 || !decoded.length) throw new Error('Upload audio up to 30 seconds.');
    pcm = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const values = decoded.getChannelData(c);
      for (let i = 0; i < pcm.length; i++) pcm[i] += values[i] / decoded.numberOfChannels;
    }
    if (pcm.some((v) => !Number.isFinite(v) || Math.abs(v) > 1)) throw new Error('Audio contains invalid or out-of-range samples.');
    if (ownedUrl) URL.revokeObjectURL(ownedUrl);
    ownedUrl = URL.createObjectURL(file); audioUrl = ownedUrl;
    $('sample').selectedIndex = -1;
    $('transcript').textContent = file.name;
    report('Uploaded audio is ready. Generate the comparison.');
  } catch (error) { pcm = undefined; fail(error); }
  finally { await context?.close(); busy(false); }
});
setup().catch((error) => { busy(true); fail(error); });
