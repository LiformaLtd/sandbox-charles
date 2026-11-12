# LTX 2.5 local video studio

Svelte + TypeScript app for local image-to-video on Apple Silicon. Upload a first
image, optionally an ending image, and describe the motion. Preview and download
the resulting MP4. No external fonts, analytics or media services are used.

Duration choices are 2, 5 and 10 seconds. Seed `-1` requests a fresh random seed
for each new generation; a fixed seed repeats the same variation for the same
inputs/settings. The server resolves and saves random seeds once per job, and
the actual seed appears beside the job status for reuse. New workspaces default
to `-1`; previously saved seed choices are retained. Inputs, duration and seed
continue to restore after refresh.

## Run

Requires Node 22+ and the separate LTX-2.5 API. From this app directory:

```sh
npm ci
npm run dev
```

Open **http://127.0.0.1:3025**. Port 3025 is fixed; startup fails if occupied.
`/api` proxies to **http://127.0.0.1:3026**. To change the API destination, copy
`.env.example` to `.env` and configure `LTX25_API_URL` before starting Vite.

Cross-repository dependency: model runtime/server is installed in the parent
Liforma workspace at `ai-workflows/apple-silicon/ltx25/`. From that directory's
parent (`apple-silicon`), run `./ltx25/run-server.sh`. See its README for the pinned
runtime install, 43.49GB model provisioning, local model override and 64GB memory
settings. This app has no implicit filesystem dependency on a particular checkout
path: it only calls the configured HTTP server.

The UI works for drafting and restoring saved outputs while the model server is
offline, and reports missing model files explicitly. It cannot generate without
the actual weights. The server never substitutes a mock output.

## Persistence

- IndexedDB `ltx25-demo/records`: draft image Files, optional ending image,
  prompt/settings; immutable submission snapshots keyed by UUID; latest output
  video as a Blob.
- localStorage `ltx25.activeJob.v1`: job identity/status for reconnecting.
- localStorage `ltx25.output.v1`: completed output reference/size. The MP4 binary
  lives in IndexedDB because localStorage's small text quota cannot reliably
  hold video. Saving a blob URL alone would not survive a page refresh.
- Output blob URLs are recreated after each refresh; the download button saves
  the stored MP4 with a `.mp4` filename.
- Submission UUIDs are persisted **before** POST. If a response is lost or the
  page closes mid-request, the app reconciles with the server and only resends
  that same snapshot/UUID when needed. It never resubmits the current edited
  draft under an old job ID.

Jobs belong to the disk-backed model worker. Closing/reloading this page or
restarting Vite does not cancel them. They reconnect on return. Even the HTTP API
can restart while its detached model worker continues. See the backend README
for worker/Mac restart recovery limits.

Use the same browser/profile and origin (127.0.0.1 vs localhost matters). Private
browsing, clearing site data, storage quota/eviction or deleting the job directory
can remove persistence. The app requests persistent browser storage and reports
storage errors; it does not silently claim a failed save succeeded. Server copies
remain on disk even if browser cache is cleared.

## Validation

```sh
npm run check
npm run build
npm test
```

Playwright uses installed Google Chrome. Tests cover actual file upload, draft
restoration, reconnecting without duplicate jobs, MP4 Blob caching, restored
playback/download and the missing-weights UI. API replies and a tiny synthetic
video fixture are controlled in those tests; they do not benchmark or verify
LTX inference. Runtime uploads, prompts and model weights must not be committed
to this public sandbox repository.

Real end-to-end validation on 2026-11-08 also passed against the installed Q8
model on an M5 Max / 64GB: 768×512, 49 frames at 24fps, with both a starting-image
job (71.9s CLI generation time) and a first/last-image job (58.1s). The first job
survived actual frontend/API process and browser restarts, then restored and
downloaded correctly. These measurements include model loading; they are two
observed runs, not a controlled benchmark. Accessible names were added to the
format/duration controls during that live verification.
