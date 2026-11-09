# Loop Lab — SvelteKit 3 / LTX-2.5

One image, five generated ten-second variations, an infinite shuffled video.
The same image is deliberately supplied as both the first and last frame of
all five model jobs. A prompt describes the action inside each loop.

## Start

```sh
npm ci
npm run dev
```

Vite starts at **127.0.0.1:5173** and automatically tries the next free port.
Use the URL printed in the terminal; there is no fixed 30xx frontend port.
The existing LTX-2.5 API must be running at **127.0.0.1:3026**. Copy `.env.example`
to `.env` to configure another model API via `LTX25_API_URL`.

Cross-repository dependency: the model server is in the same Liforma workspace
at `ai-workflows/apple-silicon/ltx25`. From `ai-workflows/apple-silicon`, start it
with `./ltx25/run-server.sh`. This sandbox only accesses it through HTTP; it does
not import Python or hard-code a filesystem checkout path. The original video
demo remains separate.

This app uses SvelteKit 3.1, Svelte 5, TypeScript 6 and Vite 8. Configuration lives
in `vite.config.ts` (Kit 3 no longer uses `svelte.config.js`). `src/env.ts` declares
the private API URL. `+server.ts` proxies only the known local model endpoints;
Vite development and adapter-node builds use the same proxy. For an adapter-node
build, set `BODY_SIZE_LIMIT=25M` to accommodate 20MB image uploads.

## Use

1. Choose a PNG/JPEG/WebP under 20MB, enter the loop prompt and choose a format.
2. **Build & start looping** loads the matching saved set or creates five jobs.
3. Playback is armed immediately. It begins when the first completed MP4 has
   downloaded and been cached; it does not wait for all five.
4. Newly finished clips enter the pool while the player continues. With only one
   available clip it repeats that clip. With two or more, every selection excludes
   the current clip. Each segment plays for ten seconds.
5. **Stop / Start** pauses and resumes playback. It does not cancel server jobs.
   Audio is off initially; toggle it on and adjust volume. Each clip can also be
   downloaded independently.

**Saved loop sets** lists previous scenes with their prompt, format and ready-clip count.
Choose one and press **Open saved set** to restore its image, prompt and clips
without another upload or generation. Opening replaces the current editor draft;
playback continues with the selected set if it was already running. **Refresh**
fetches sets created in another tab. The selected set survives page refresh,
and previously cached sets can be opened with the model API offline.

Prompt or image changes take effect when you press Build again. The same image,
prompt, size, pipeline and pinned model/runtime revisions identify a saved set.
An unchanged set reuses existing completed jobs rather than generating five more.
The five seeds are different, resolved once and saved with the set.

**Latest build wins.** An accepted Build atomically cancels every other queued or
running generation on this local model server. The worker stops and reaps the
old inference process (including encoder children), then immediately claims the
new set's first eligible clip. This can take a short process shutdown interval;
it does not wait for the old clip to finish rendering. Invalid submissions leave
the existing queue intact. Completed videos and source images are retained.
Opening a saved set only changes playback and does not cancel anything.
Rebuilding or resuming a cancelled set reuses completed clips and existing seeds,
queues its unfinished clips, and takes over the model in the same way.
Cancelled sets show **Resume unfinished clips** rather than a failure.

## Negative prompt

The editor starts with `speaking, talking, lip movements, open mouth, singing`.
The text is saved with the browser draft and every generated set. Nonblank text
enables the installed distilled runtime's `--nag --negative-prompt` guidance for
all five clips. Clearing the field disables that guidance. Changing the negative
prompt creates a distinct saved set; a blank field still reuses legacy sets that
were generated without negative guidance. Opening a saved set restores its actual
negative prompt, including an empty one. NAG is experimental guidance, so a
negative prompt discourages the described motion rather than guaranteeing its
absence.

## Scheduling and recovery

The server publishes all five job records in one SQLite transaction. Clip 2
explicitly depends on successful completion of clip 1, clip 3 on clip 2, and so on.
A single detached model worker claims only eligible jobs. This continues after
the browser, frontend, or HTTP API closes/restarts. An interrupted worker restarts
a generation from its saved seed; this is not a mid-diffusion checkpoint.

A failed generation blocks the later clips. Ready clips still play. **Retry failed
clip** retries only failed work with its existing seed; completed work is retained.
Logs are available on failed cards. No mocked output is used in production.

## Persistence and changing ports

IndexedDB is scoped to the browser **origin: scheme + hostname + port**.
The browser database `ltx25-loop-lab/records` holds the input File, draft prompt,
format, saved-set library, active-set snapshot, anchor images and completed MP4 Blobs.
Blob URLs are recreated after refresh. These are actual video bytes, not saved
transient object URLs. Cached clips can play while the API is offline.

The local server is authoritative for generated sets, normalized images, prompts,
seeds, job states and MP4s. A fresh origin (including a different automatically
selected port) recovers the most recent server set and repopulates its browser
cache. It does not regenerate the clips. Unsubmitted draft edits remain specific
to their original browser origin. On return to the same origin, its own draft and
active set take precedence over the server's most recent set.

A page refresh restores the set and clips but waits for Start before playing
again. Browsers may reject unmuted autoplay without user interaction; Start and
the audio toggle provide that interaction. Storage and decode failures are
reported visibly rather than replaced with silent mock clips.

## How the joins work

Two HTML video decoders alternate, with the next clip preloaded ahead of the
boundary. A single persistent canvas paints the output, so switching sources
never clears the displayed frame. Around the join the canvas converges briefly
to the common endpoint image (center-cropped to match model framing), then blends
into the next decoded clip. When audio is enabled, Web Audio gains crossfade for
50ms. Resources and old object URLs are released when changing sets/unmounting.

This avoids blank-frame joins and softens reconstruction differences. The model
still determines motion and endpoint accuracy: an abrupt pose, lighting change,
voice or environmental sound in a generated clip cannot be guaranteed seamless
by the player. Keep prompts compatible with returning to the shared image.
Playback timing is designed for a visible tab; browsers may throttle background
rendering/timers. All media/model computation stays on the local machine.

## Verify

```sh
npm run check
npm run build
npm test
```

The test runner starts its own Vite server at the next free port from 5173 and
passes the actual URL to Playwright. Tests use installed Google Chrome and small
synthetic video/audio fixtures. They verify incremental availability, auto-start
from one clip, two ten-second joins without blank canvas frames, no adjacent
repeats, audio controls, pause/resume, downloads, cached refresh and fresh-origin
server recovery. They do not run a model themselves.

Backend tests live in `ai-workflows/apple-silicon/ltx25/test_loops.py` and cover
content-based reuse, identical endpoint images, five distinct seeds, strict
success dependencies, blocked failure recovery and existing-job compatibility.
Large model files, generated media, private uploads, environments and dependency
folders must remain out of this public repository.

### Real model validation, 2026-11-09

Five real 512×512 / 241-frame clips using identical first/last images were
generated with different seeds. CLI times were 97.5, 104.6, 101.0, 101.8 and
107.6 seconds. Every next job started after its predecessor completed. The
actual app recovered them on a fresh origin and passed two ten-second playback
joins with 1,201 sampled canvas frames, no blank frames and no immediate
repeats, plus Stop, audio toggles, download and cached restoration. This is one
illustrated test scene, not a quality guarantee or general performance benchmark.

### Proxy regression coverage

Browser submissions use a native server-to-server fetch with explicit headers.
SvelteKit event.fetch inherits the browser Origin, which the model service
correctly rejects for dynamic frontend ports. The proxy still rejects foreign
browser origins. A real proxy regression test posts multipart data through
SvelteKit to an isolated upstream that rejects inherited browser origins; it
also verifies foreign-origin requests never reach that upstream. The live Build
button was tested against the model API and returned 202 while reusing the five
existing clips without new generation.
