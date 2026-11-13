# LiTo image-to-3D demo

Upload a PNG, JPEG, or WebP (up to 20 MB) and click **Generate**. The local server runs LiTo on Apple Silicon, and Three.js with Spark displays the resulting Gaussian PLY. A transparent PNG works well for an isolated subject.

The browser stores the selected image, pending job ID, and completed PLY in IndexedDB. Reopen the same URL in the same browser profile to resume polling or restore the saved splat. Keep the local server running while the page is closed. Completed jobs remain available after a server restart, and a job interrupted by a server restart can be submitted again.

Enable **Add source detail** before clicking **Generate** to project the original image onto the visible surface of the LiTo splat. The pass adds thin surface Gaussians from a 1024 × 1024 projection. It uses a transparent background or keys a consistent green or blue screen before running LiTo. If the source background is unsupported or alignment is unreliable, the demo saves the LiTo splat and explains why source detail was skipped. Surfaces hidden in the source view still come from LiTo.

## Setup

Requires Apple Silicon, Node.js 22 or newer, and a local LiTo MLX installation. The demo uses the [mlx-spatial LiTo CLI](https://github.com/appautomaton/mlx-spatial/blob/main/docs/lito.md) version 0.0.3 and its local converted model bundle. LiTo is covered by [Apple's research model license](https://github.com/apple-aiml-research/ml-lito/blob/main/LICENSE_MODEL).

In this directory, run:

```sh
npm ci
cp .env.example .env
```

Set `LITO_CLI` to the absolute path of the executable `mlx-spatial-lito` and `LITO_WEIGHTS` to the absolute path of the bundle root. The server bridges the CLI's legacy TRELLIS decoder lookup to the decoder inside the bundle.

To enable source detail, install the additional packages into the LiTo Python environment:

```sh
uv pip install --python /absolute/path/to/.venv/bin/python -r requirements-detail.txt
```

The server defaults to the `python` executable next to `LITO_CLI`. Set `LITO_PYTHON` if the interpreter is elsewhere.

```sh
npm run dev
```

Open the URL printed by the server. It starts at port 5173 and advances to the first free port. IndexedDB is scoped to the origin, so saved browser sessions are associated with the port used for that session.

The server queues jobs one at a time. It stores inputs, PLYs, and generation logs under `.lito-data/jobs/`, or under the directory configured with `LITO_DATA_DIR`. The browser also stores a copy of its current result. To reclaim space, stop the server and remove unneeded local job folders, then clear the site's browser data if desired.
