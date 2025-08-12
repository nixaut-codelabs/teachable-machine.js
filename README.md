<div align="center">

# teachable-machine.js

Ultra‑fast, production‑ready Teachable Machine inference for Node.js with smart RAM/Disk I/O, unified Image & Video classification, and optional native backend acceleration.

Repo: https://github.com/nixaut-codelabs/teachable-machine.js • NPM: teachable-machine.js

</div>

## Highlights

* **Ultra fast classification** — batched inference, turbo mode, and optional native backend (`@tensorflow/tfjs-node`).
* **Smart resource management** — in‑memory (RAM) pipeline by default, automatic disk fallback, strict cleanup guarantees.
* **Video classification (FFmpeg)** — sample frames evenly via FFmpeg; supports MP4, GIF, and more.
* **Backend control** — choose between pure `tfjs` and `tfjs-node` (native TensorFlow) for best performance.
* **Detailed parameters & outputs** — timings, backend, I/O mode (RAM/Disk), size limits, fallback, and cleanup status.
* **Unified API** — classify images and videos through a single, ergonomic interface.
* **Flexible inputs** — URLs, local paths, Buffers/Uint8Arrays, data URIs, and base64 strings.

Author: Nixaut • License: Apache‑2.0

---

## Installation

Install from NPM:

```bash
npm i teachable-machine.js
# or
bun add teachable-machine.js
```

This library relies on:

* `@tensorflow/tfjs` (installed transitively)
* `sharp` for image decoding/resizing
* `ffmpeg-static` for bundled FFmpeg (or use system FFmpeg via PATH)

Optional (highly recommended for speed):

```bash
npm i @tensorflow/tfjs-node
# or
bun add @tensorflow/tfjs-node
```

> Note: When you pass `backend: 'tfjs-node'` the library will try to use the native backend (`tensorflow`) for a large CPU speedup.

---

## Quick Start

```js
import TeachableMachine from 'teachable-machine.js';

const tm = await TeachableMachine.create({
  modelUrl: 'https://teachablemachine.withgoogle.com/models/your-model-id/',
  saveToDir: 'model',      // optional: cache locally
  loadFrom: 'auto',        // 'auto'|'dir'
  warmup: true,            // run one warmup forward pass
  ioMode: 'ram',           // 'ram' (default) | 'disk'
  backend: 'tfjs-node',    // 'tfjs' (default) | 'tfjs-node'
  preprocessUseWorkers: true // offload sharp preprocess to Worker Threads
});

// Single image
const img = await tm.classify({ input: 'local_or_url_or_buffer.jpg', mediaType: 'image' });
console.log(img.predictions);

// Batch images
const imgs = await tm.classifyImages({ images: ['1.jpg', '2.jpg'], batchSize: 2 });
console.log(imgs.results[0].predictions);

// Single video (mp4/gif)
const vid = await tm.classify({ input: 'video.mp4', mediaType: 'video', frames: 8, turboMode: true });
console.log(vid.aggregate.predictions);

// Batch videos
const vids = await tm.classifyVideos({ videos: ['a.mp4', 'b.gif'], frames: 8 });
console.log(vids[0].aggregate.predictions);

// Mixed: images + videos together
const mixed = await tm.classify({
  input: { images: ['1.jpg', '2.jpg'], videos: ['a.mp4', 'b.gif'] },
  frames: 8,
  turboMode: true
});
console.log(mixed.images?.results?.length, mixed.videos?.length);

tm.dispose();
```

---

## CLI (tmjs)

After installation, you can quickly try it from the terminal:

```bash
# Single image
tmjs --model https://teachablemachine.withgoogle.com/models/XXX/ image.jpg

# Video + GIF with turbo and frame count
tmjs --model ./model --media video --frames 8 --turbo video.mp4 gif.gif
```

Arguments:

- `--model <url|dir>`: Model source (URL or local directory)
- `--backend tfjs|tfjs-node`: Backend selection
- `--io ram|disk`: I/O mode
- `--media image|video|auto`: Input type routing
- `--frames N`, `--topK K`, `--maxBytes BYTES`, `--turbo`

Output is printed as JSON to stdout; suitable for CI and automation.

---

## Why this library?

* **Ultra fast classification**
  * Image turbo batching and video turbo mode (preprocess tensors + single batched forward).
  * Optional `@tensorflow/tfjs-node` backend for native CPU acceleration.
* **Smart resource management**
  * Default in‑RAM pipeline avoids disk I/O.
  * Automatic fallback to disk when RAM extractor yields no frames.
  * Guaranteed temp cleanup on success and failure.
  * Optional Worker Threads (`preprocessUseWorkers: true`) separate preprocessing from the main thread.
* **Video classification (FFmpeg)**
  * Frame sampling via FFmpeg with robust single‑pass PNG pipeline.
  * GIFs supported the same way.
* **Backend control**
  * `backend: 'tfjs' | 'tfjs-node'` on model creation.
* **Detailed diagnostics**
  * Outputs include backend, timings for each stage, I/O mode, fallback flag, sizeBytes/maxBytes.

---

## API

### `TeachableMachine.create(options)` → `Promise<TeachableMachine>`

Options:

* `modelUrl?: string` — Teachable Machine base URL (ends with `/`).
* `modelDir?: string` — directory containing a cached model.
* `loadFrom?: 'auto'|'dir'` — auto prefer local dir when available.
* `saveToDir?: string` — if provided, downloads and caches the model locally.
* `warmup?: boolean` — run one forward pass on zeros (default true).
* `ioMode?: 'ram'|'disk'` — RAM mode uses in‑memory pipeline; disk uses temp files.
* `backend?: 'tfjs'|'tfjs-node'` — select JS vs native backend at init.
* `preprocessUseWorkers?: boolean` — run sharp-based preprocessing in Worker Threads (keeps CPU-heavy work off the main thread).

Returns an instance with methods below.

### Image classification

* `classifyImages({ images, topK?, centerCrop=true, resizeOnCPU=true, batchSize? })`
  * `images`: single input or array (string | Buffer | Uint8Array | data URI | base64).
  * Returns either a single detailed result or a batch summary `{ count, timings, results }`.

### Video classification

* `classifyVideos({ videos, frames=10, topK?, centerCrop=true, resizeOnCPU=true, turboMode=false, extractionConcurrency?, preprocessConcurrency?, maxConcurrent=2, maxBytes=10*MB })`
  * `videos`: single input or array.
  * Returns per‑video detailed outputs with frame predictions and `aggregate.predictions`.

### Unified classification

* `classify({ input, mediaType='auto', frames=10, topK?, centerCrop=true, resizeOnCPU=true, turboMode=false, extractionConcurrency?, preprocessConcurrency?, maxConcurrent=2, maxBytes=10*MB, batchSize? })`
  * If `input` is an array or scalar, the route is chosen by `mediaType` and `frames`.
  * If `input` is `{ images, videos }`, both branches run and return `{ images, videos }`.

### Compatibility helper

* `classifyBatch({ imageUrls, ... })` — original batch images API.
* `batchImageClassify(opts)` — alias to `classifyBatch()`.

### Lifecycle

* `dispose()` — free model resources.

---

## TypeScript

The package ships with a `.d.ts` type file. Key types:

- `CreateOptions`
- `ImageResult`, `BatchImageResult`
- `VideoResult`, `FramePrediction`

Example:

```ts
import TeachableMachine, { CreateOptions, VideoResult } from 'teachable-machine.js';

const opts: CreateOptions = { backend: 'tfjs-node', preprocessUseWorkers: true };
const tm = await TeachableMachine.create(opts);
```

---

## I/O Modes, Size Limits, and Cleanup

* `ioMode: 'ram' | 'disk'` on `.create()`
  * RAM: No temp files; streams media into memory buffers.
  * Disk: Uses temp files; always cleaned up in `finally`.
* Size limits
  * `maxBytes` on video classification to prevent oversized inputs.
* Output `io` object
  * `{ mode, fallbackToDisk, tempCleaned, sizeBytes, maxBytes }` for audits & debugging.

---

## Performance tips

* Use `backend: 'tfjs-node'` for large CPU speedups (server/desktop).
* Enable `turboMode` for videos to preprocess frames and run a single batched forward pass.
* Use `batchSize` on images for better throughput.
* Tune `extractionConcurrency` and `preprocessConcurrency` to match CPU cores.

---

## FFmpeg requirements

Video/GIF classification requires FFmpeg.

* Bundled: `ffmpeg-static` is used if present.
* System: Otherwise a system `ffmpeg` binary on PATH is used.

We include a small diagnostic helper internally that prefers `ffmpeg-static` and falls back to system PATH.

---

## Examples

See `advanced-example.js` for a comprehensive demo with environment toggles:

* `BACKEND=tfjs-node` or `tfjs`
* `IO_MODE=ram` or `disk`
* `FRAMES=8`, `TURBO=true`, `MAX_BYTES=10485760`

Run with Bun:

```bash
bun run advanced-example.js
```

---

## Contributing

PRs are welcome! Please open an issue to discuss substantial changes first.

---

## License

This project is licensed under the Apache-2.0 License. See the `LICENSE` file for details.
