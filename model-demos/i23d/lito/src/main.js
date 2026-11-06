import './style.css';
import { openSessionStore, readSession, saveSession } from './storage.js';

const picker = document.querySelector('#image');
const preview = document.querySelector('#preview');
const previewEmpty = document.querySelector('#preview-empty');
const sourceDetail = document.querySelector('#source-detail');
const generate = document.querySelector('#generate');
const status = document.querySelector('#status');
const viewport = document.querySelector('#viewport');
const viewerStatus = document.querySelector('#viewer-status');
const reset = document.querySelector('#reset');
const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
let db;
let session = {};
let busy = false;
let imageUrl;
let viewer;

function syncControls() {
  picker.disabled = !db || busy;
  sourceDetail.disabled = !db || busy || !session.image;
  generate.disabled = !db || busy || !session.image;
}

function showInput() {
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = session.image ? URL.createObjectURL(session.image) : undefined;
  if (imageUrl) preview.src = imageUrl;
  else preview.removeAttribute('src');
  preview.hidden = !imageUrl;
  previewEmpty.hidden = !!imageUrl;
}

function clearSplat() {
  viewer?.clear();
  reset.disabled = true;
  viewerStatus.hidden = false;
  viewerStatus.textContent = 'Your splat will appear here.';
}

async function showSplat(blob) {
  viewerStatus.hidden = false;
  viewerStatus.textContent = 'Loading the splat into the viewer…';
  try {
    if (!viewer) {
      const { createViewer } = await import('./viewer.js');
      viewer = createViewer(viewport);
    }
    await viewer.show(blob);
    viewerStatus.hidden = true;
    reset.disabled = false;
    status.textContent = 'Splat ready and saved in this browser.';
    if (session.sourceBackgroundRemoved) status.textContent += ' Screen background removed.';
    if (session.sourceDetailApplied === true) status.textContent += ' Source detail applied.';
    if (session.sourceDetailApplied === false) {
      if (session.sourceDetailReason === 'alignment') {
        status.textContent += ' Source detail skipped because the image and splat could not be aligned reliably.';
      } else if (session.sourceDetailReason === 'background') {
        status.textContent += ' Source detail skipped. Use a transparent image or a solid green or blue screen.';
      } else {
        status.textContent += ' This splat was saved without source detail. Generate again to try automatic screen removal.';
      }
    }
  } catch (error) {
    console.error(error);
    viewerStatus.textContent = 'The splat is saved, but the viewer could not display it.';
    status.textContent = error.message;
  }
}

class RequestError extends Error {
  constructor(response, message) {
    super(message);
    this.status = response.status;
  }
}

async function requireOk(response) {
  if (response.ok) return response;
  const body = await response.json().catch(() => ({}));
  throw new RequestError(response, body.error || `Request failed (${response.status}).`);
}

async function followJob() {
  busy = true;
  syncControls();
  const url = `/api/jobs/${session.jobId}`;
  try {
    for (;;) {
      let job;
      let splat;
      try {
        let response = await fetch(url, { cache: 'no-store' });
        if (response.status === 404) {
          // Persist the client ID before submission, then use the same ID if the tab closed mid-upload.
          status.textContent = 'Submitting image…';
          const submission = session.sourceDetail ? `${url}?sourceDetail=1` : url;
          response = await fetch(submission, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: session.image,
          });
        }
        job = await (await requireOk(response)).json();
        if (job.state === 'complete') {
          status.textContent = 'Saving the generated splat…';
          splat = await (await requireOk(await fetch(`${url}/splat.ply`))).blob();
        }
      } catch (error) {
        if (error instanceof RequestError && error.status < 500) throw error;
        status.textContent = 'Waiting for the local LiTo server to reconnect…';
        await wait(2000);
        continue;
      }
      if (job.state === 'failed') throw new Error(job.error);
      if (splat) {
        session = {
          ...session, jobId: null, splat,
          sourceDetailApplied: job.sourceDetailApplied,
          sourceDetailReason: job.sourceDetailReason,
          sourceBackgroundRemoved: job.sourceBackgroundRemoved,
        };
        await saveSession(db, session);
        await showSplat(splat);
        break;
      }
      if (job.state === 'refining') {
        status.textContent = 'Adding source detail. You can close this page and return later.';
      } else if (job.state === 'running') {
        status.textContent = 'Generating on this Mac. You can close this page and return later.';
      } else {
        status.textContent = 'Waiting for LiTo to start…';
      }
      await wait(1500);
    }
  } catch (error) {
    console.error(error);
    status.textContent = error.message;
  } finally {
    busy = false;
    syncControls();
  }
}

picker.addEventListener('change', async () => {
  const image = picker.files[0];
  if (!image) return;
  if (image.size > 20 * 1024 * 1024) {
    picker.value = '';
    status.textContent = 'Choose an image smaller than 20 MB.';
    return;
  }
  busy = true;
  syncControls();
  try {
    const selected = { image, sourceDetail: sourceDetail.checked };
    await saveSession(db, selected);
    session = selected;
    clearSplat();
    showInput();
    status.textContent = 'Image saved. Click Generate to begin.';
  } catch (error) {
    status.textContent = `Could not save the image: ${error.message}`;
  } finally {
    busy = false;
    syncControls();
  }
});

generate.addEventListener('click', async () => {
  busy = true;
  syncControls();
  try {
    const pending = { image: session.image, jobId: crypto.randomUUID(), sourceDetail: sourceDetail.checked };
    await saveSession(db, pending);
    session = pending;
    clearSplat();
    await followJob();
  } catch (error) {
    status.textContent = `Could not save the job: ${error.message}`;
  } finally {
    busy = false;
    syncControls();
  }
});
reset.addEventListener('click', () => viewer?.reset());

async function restore() {
  try {
    db = await openSessionStore();
    session = await readSession(db);
    sourceDetail.checked = !!session.sourceDetail;
    showInput();
    if (session.jobId) await followJob();
    else if (session.splat) {
      busy = true;
      try { await showSplat(session.splat); } finally { busy = false; }
    } else status.textContent = session.image ? 'Image restored. Click Generate to begin.' : 'Choose a source image to get started.';
    syncControls();
  } catch (error) {
    status.textContent = `Could not restore browser storage: ${error.message}`;
  }
}

void restore();
