import * as tf from '@tensorflow/tfjs';
import os from 'os';
import { http } from './utils/net.js';
import { dirExists, ioFromDir, readMetadata, writeMetadata } from './utils/io.js';
import { getImageBuffer, toSizedRGBTensor, setPreprocessOptions } from './preprocess.js';
import { ensureFFmpeg, ensureLocalPathWithCleanup, probeDurationSec, sampleTimestamps, safeTimestamps, extractFrames, getMediaBuffer, probeDurationSecFromBuffer, extractFramesFromBuffer, extractFramesAutoFromBuffer } from './utils/ffmpeg.js';

tf.env().set('DEBUG', false);

const getTopKClasses = async (logits, classes, topK) => {
  const k = Math.min(topK ?? classes.length, classes.length);
  const { values, indices } = tf.topk(logits, k);
  const [v, i] = await Promise.all([values.data(), indices.data()]);
  values.dispose(); indices.dispose(); logits.dispose();
  return Array.from(i).map((idx, j) => ({ class: classes[idx], score: v[j] }));
};

export default class TeachableMachine {
  constructor(model) {
    this.model = model;
  }

  static async create({ modelUrl, modelDir, loadFrom = 'auto', saveToDir, warmup = true, ioMode = 'ram', backend = 'tfjs', preprocessUseWorkers = false } = {}) {
    try {
      if (backend === 'tfjs-node') {
        try {
          // Dynamically register tfjs-node backend if available
          await import("@tensorflow/tfjs-node");
          // Prefer native backend when present
          if (tf.getBackend() !== 'tensorflow') {
            await tf.setBackend('tensorflow');
            await tf.ready();
          }
        } catch (e) {
          throw new Error("Requested backend 'tfjs-node' but '@tensorflow/tfjs-node' is not installed. Install it with: npm i @tensorflow/tfjs-node");
        }
      }
      let model; let classes;

      if (loadFrom === 'dir' || (loadFrom === 'auto' && modelDir && await dirExists(modelDir))) {
        model = await tf.loadLayersModel(ioFromDir(modelDir));
        const metadata = await readMetadata(modelDir);
        if (!metadata.labels || !Array.isArray(metadata.labels)) throw new Error('Invalid metadata in local dir.');
        classes = metadata.labels;
      } else {
        if (!modelUrl) throw new Error('Model URL is missing!');
        const modelURL = `${modelUrl}model.json`;
        const metadataResponse = await http(`${modelUrl}metadata.json`).buffer();
        const metadata = JSON.parse(metadataResponse.toString());
        if (!metadata.labels || !Array.isArray(metadata.labels)) throw new Error("Invalid metadata: 'labels' field not found or is not an array.");
        classes = metadata.labels;
        model = await tf.loadLayersModel(modelURL);
        const targetDir = saveToDir ?? modelDir;
        if (targetDir) {
          await model.save(ioFromDir(targetDir));
          await writeMetadata(targetDir, { labels: classes });
        }
      }

      model.classes = classes;
      if (warmup && model.inputs?.[0]?.shape) {
        const h = model.inputs[0].shape[1];
        const w = model.inputs[0].shape[2];
        if (typeof h === 'number' && typeof w === 'number') {
          const dummy = tf.zeros([1, h, w, 3]);
          const out = model.predict(dummy);
          if (Array.isArray(out)) out.forEach(t => t.dispose()); else out.dispose();
          dummy.dispose();
        }
      }

      const tm = new TeachableMachine(model);
      tm.ioMode = ioMode === 'disk' ? 'disk' : 'ram';
      tm.backend = backend === 'tfjs-node' ? 'tfjs-node' : 'tfjs';
      // optional: enable worker-threaded preprocessing
      try { setPreprocessOptions({ useWorkers: !!preprocessUseWorkers }); } catch {}
      return tm;
    } catch (e) {
      throw new Error(`Model loading failed: ${e.message}`);
    }
  }

  async _decodeAndPredict(imageBuffer, { topK, centerCrop = true, resizeOnCPU = true } = {}) {
    const inShape = this.model?.inputs?.[0]?.shape;
    const targetH = inShape?.[1];
    const targetW = inShape?.[2];
    if (typeof targetH !== 'number' || typeof targetW !== 'number') throw new Error('Model input shape is not fully defined.');

    const t0 = Date.now();
    const sized = await toSizedRGBTensor(imageBuffer, targetW, targetH, { centerCrop });
    const imageTensor = tf.tensor3d(sized.data, [targetH, targetW, 3], 'int32');
    const t1 = Date.now();

    const logits = tf.tidy(() => {
      const offset = tf.scalar(127.5);
      const normalized = imageTensor.toFloat().sub(offset).div(offset);
      const batched = normalized.expandDims(0);
      return this.model.predict(batched);
    });
    const t2 = Date.now();
    imageTensor.dispose();

    const top = await getTopKClasses(logits, this.model.classes, topK);
    const t3 = Date.now();

    return {
      backend: tf.getBackend(),
      modelInfo: { classesCount: this.model.classes.length },
      preprocess: { target: { width: targetW, height: targetH }, centerCrop, resizeOnCPU },
      timings: { decodeResizeMs: t1 - t0, inferenceMs: t2 - t1, postprocessMs: t3 - t2, totalMs: t3 - t0 },
      predictions: top.map((p, idx) => ({ ...p, rank: idx + 1 }))
    };
  }

  async _classifyImage({ imageUrl, topK, centerCrop = true, resizeOnCPU = true }) {
    const tStart = Date.now();
    const imageBuffer = await getImageBuffer(imageUrl);
    const downloadEnd = Date.now();
    const inner = await this._decodeAndPredict(imageBuffer, { topK, centerCrop, resizeOnCPU });
    return {
      input: { imageUrl },
      ...inner,
      timings: { downloadMs: downloadEnd - tStart, ...inner.timings, endToEndMs: (downloadEnd - tStart) + inner.timings.totalMs }
    };
  }

  async classifyImages({ images, topK, centerCrop = true, resizeOnCPU = true, batchSize } = {}) {
    if (!images) throw new Error('images is required');
    if (Array.isArray(images)) {
      return this.classifyBatch({ imageUrls: images, topK, centerCrop, resizeOnCPU, batchSize });
    }
    return this._classifyImage({ imageUrl: images, topK, centerCrop, resizeOnCPU });
  }

  async classifyBatch({ imageUrls, topK, centerCrop = true, resizeOnCPU = true, batchSize }) {
    if (!Array.isArray(imageUrls) || imageUrls.length === 0) throw new Error('imageUrls must be a non-empty array');

    const inShape = this.model?.inputs?.[0]?.shape;
    const targetH = inShape?.[1];
    const targetW = inShape?.[2];
    if (typeof targetH !== 'number' || typeof targetW !== 'number') throw new Error('Model input shape is not fully defined.');

    const results = [];
    const tBatchStart = Date.now();

    const processChunk = async (urls) => {
      const t0 = Date.now();
      const dlResults = await Promise.all(urls.map(async (u) => {
        try {
          const buf = await getImageBuffer(u);
          return { ok: true, buf };
        } catch (e) {
          return { ok: false, err: e, url: u };
        }
      }));
      const okPairs = dlResults.map((r, i) => ({ idx: i, r })).filter(x => x.r.ok);
      const failPairs = dlResults.map((r, i) => ({ idx: i, r })).filter(x => !x.r.ok);
      // Immediately record failures so batch remains responsive
      for (const { idx, r } of failPairs) {
        results.push({
          input: { imageUrl: urls[idx] },
          error: r.err?.message || String(r.err),
          backend: tf.getBackend(),
          modelInfo: { classesCount: this.model.classes.length },
          timings: { downloadMs: 0, decodeResizeMs: 0, inferenceMs: 0, postprocessMs: 0, totalMs: 0 }
        });
      }
      const buffers = okPairs.map(p => p.r.buf);
      const tDownloadEnd = Date.now();
      const tensors = [];
      for (const buf of buffers) {
        const sized = await toSizedRGBTensor(buf, targetW, targetH, { centerCrop });
        tensors.push(tf.tensor3d(sized.data, [targetH, targetW, 3], 'int32'));
      }
      const tPrepEnd = Date.now();
      const logits = tf.tidy(() => {
        const offset = tf.scalar(127.5);
        const batch = tf.stack(tensors.map(t => t.toFloat().sub(offset).div(offset)));
        const out = this.model.predict(batch);
        return Array.isArray(out) ? out[0] : out;
      });
      tensors.forEach(t => t.dispose());
      const tInferEnd = Date.now();

      const k = Math.min(topK ?? this.model.classes.length, this.model.classes.length);
      const { values, indices } = tf.topk(logits, k);
      const [vals, inds] = await Promise.all([values.array(), indices.array()]);
      values.dispose(); indices.dispose(); logits.dispose();
      const tPostEnd = Date.now();

      // Map predictions back onto successful indices only
      okPairs.forEach(({ idx: okIdx }, row) => {
        const u = urls[okIdx];
        const preds = inds[row].map((clsIdx, j) => ({ class: this.model.classes[clsIdx], score: vals[row][j], rank: j + 1 }));
        results.push({
          input: { imageUrl: u },
          backend: tf.getBackend(),
          modelInfo: { classesCount: this.model.classes.length },
          preprocess: { target: { width: targetW, height: targetH }, centerCrop, resizeOnCPU: true },
          timings: { downloadMs: tDownloadEnd - t0, decodeResizeMs: tPrepEnd - tDownloadEnd, inferenceMs: tInferEnd - tPrepEnd, postprocessMs: tPostEnd - tInferEnd, totalMs: tPostEnd - t0 },
          predictions: preds
        });
      });
    };

    if (batchSize && batchSize > 0 && batchSize < imageUrls.length) {
      for (let i = 0; i < imageUrls.length; i += batchSize) {
        const chunk = imageUrls.slice(i, i + batchSize);
        await processChunk(chunk);
      }
    } else {
      await processChunk(imageUrls);
    }

    const tBatchEnd = Date.now();
    return { backend: tf.getBackend(), count: imageUrls.length, modelInfo: { classesCount: this.model.classes.length }, timings: { endToEndMs: tBatchEnd - tBatchStart }, results };
  }

  /**
   * Backward-compat alias for older code.
   * Delegates to classifyBatch().
   */
  async batchImageClassify(options) {
    return this.classifyBatch(options);
  }

  /**
   * Unified classify entry. Routes to image or video classification.
   * @param {object} options
   * @param {any|any[]} options.input - Image(s) or video(s)
   * @param {'auto'|'image'|'video'} [options.mediaType='auto']
   * @param {number} [options.frames=10]
   */
  async classify({ input, mediaType = 'auto', frames = 10, topK, centerCrop = true, resizeOnCPU = true, turboMode = false, extractionConcurrency, preprocessConcurrency, maxConcurrent = 2, maxBytes = 10 * 1024 * 1024, batchSize } = {}) {
    // Mixed object form: { images: [...], videos: [...] }
    if (input && typeof input === 'object' && !Array.isArray(input) && (input.images || input.videos)) {
      const tasks = [];
      if (input.images && input.images.length) {
        tasks.push(this.classifyImages({ images: input.images, topK, centerCrop, resizeOnCPU, batchSize }));
      } else {
        tasks.push(Promise.resolve(null));
      }
      if (input.videos && input.videos.length) {
        tasks.push(this.classifyVideos({ videos: input.videos, frames, topK, centerCrop, resizeOnCPU, turboMode, extractionConcurrency, preprocessConcurrency, maxConcurrent, maxBytes }));
      } else {
        tasks.push(Promise.resolve(null));
      }
      const [imagesRes, videosRes] = await Promise.all(tasks);
      return { images: imagesRes, videos: videosRes };
    }

    // Legacy/array form: decide route
    const isVideo = mediaType === 'video' || (Number.isFinite(frames) && frames > 0 && mediaType !== 'image');
    if (isVideo) {
      return this.classifyVideos({ videos: input, frames, topK, centerCrop, resizeOnCPU, turboMode, extractionConcurrency, preprocessConcurrency, maxConcurrent, maxBytes });
    }
    return this.classifyImages({ images: input, topK, centerCrop, resizeOnCPU, batchSize });
  }

  /**
   * Public wrapper for single or multiple videos.
   */
  async classifyVideos({ videos, ...rest } = {}) {
    return this.classifyVideo({ videoUrl: videos, ...rest });
  }

  dispose() { if (this.model) this.model.dispose(); }

  /**
   * Classifies one or more videos/GIFs by sampling evenly spaced frames with FFmpeg and running the image pipeline.
   * Accepts URL/path, Buffer/Uint8Array, data URI or base64 for single input, or an array of such inputs.
   * @param {object} options
   * @param {string|Buffer|Uint8Array|(string|Buffer|Uint8Array)[]} options.videoUrl
   * @param {number} [options.frames=10]
   * @param {number} [options.topK]
   * @param {boolean} [options.centerCrop=true]
   * @param {boolean} [options.resizeOnCPU=true]
   * @param {boolean} [options.turboMode=false]
   */
  async classifyVideo({ videoUrl, frames = 10, topK, centerCrop = true, resizeOnCPU = true, turboMode = false, extractionConcurrency, preprocessConcurrency, maxConcurrent = 2, maxBytes = 10 * 1024 * 1024 } = {}) {
    // Support single or multiple inputs
    if (Array.isArray(videoUrl)) {
      const tasks = videoUrl.map((u) => async () => this.classifyVideo({ videoUrl: u, frames, topK, centerCrop, resizeOnCPU, turboMode, extractionConcurrency, preprocessConcurrency, maxBytes }));
      const q = tasks.slice();
      const out = [];
      const runners = new Array(Math.max(1, Math.min(maxConcurrent, q.length))).fill(0).map(async function run() {
        while (q.length) {
          const job = q.shift();
          if (!job) break;
          out.push(await job());
        }
      });
      await Promise.all(runners);
      return out;
    }
    if (!videoUrl) throw new Error('videoUrl is required');
    if (!Number.isFinite(frames) || frames <= 0) throw new Error('frames must be a positive number');
    const tStart = Date.now();
    // Determine target model input size
    const inShape = this.model?.inputs?.[0]?.shape;
    const targetH = inShape?.[1];
    const targetW = inShape?.[2];
    if (typeof targetH !== 'number' || typeof targetW !== 'number') throw new Error('Model input shape is not fully defined.');
    const aggregateScores = new Array(this.model.classes.length).fill(0);
    let frameCount = 0;
    const ffmpegPath = await ensureFFmpeg();
    let cleanup = async () => {};
    let durationSec; let framesSource; let usedMode = this.ioMode; let fallbackToDisk = false; let sizeBytes = 0; let tempCleaned = false;
    try {
      if (this.ioMode === 'ram') {
        const mediaBuf = await getMediaBuffer(videoUrl);
        sizeBytes = mediaBuf.length;
        if (maxBytes && sizeBytes > maxBytes) throw new Error(`Media exceeds maxBytes (${sizeBytes} > ${maxBytes})`);
        durationSec = await probeDurationSecFromBuffer(ffmpegPath, mediaBuf);
        framesSource = mediaBuf;
      } else {
        const loc = await ensureLocalPathWithCleanup(videoUrl);
        cleanup = loc.cleanup;
        try {
          const { default: fs } = await import('fs/promises');
          const st = await fs.stat(loc.path).catch(() => null);
          sizeBytes = st?.size ?? 0;
          if (maxBytes && sizeBytes > maxBytes) throw new Error(`Media exceeds maxBytes (${sizeBytes} > ${maxBytes})`);
        } catch {}
        durationSec = await probeDurationSec(ffmpegPath, loc.path);
        framesSource = loc.path;
      }
      if (!durationSec || durationSec <= 0) throw new Error('Unable to determine video duration');

      const stampsRaw = sampleTimestamps(durationSec, Math.floor(frames));
      const stamps = safeTimestamps(durationSec, stampsRaw);
      const tPrepEnd = Date.now();
      const cpuCount = (typeof os?.cpus === 'function' && Array.isArray(os.cpus())) ? os.cpus().length : 4;
      const extractConc = Math.max(1, Math.min(16, extractionConcurrency ?? (turboMode ? Math.min(8, cpuCount) : 1)));
      let frameBuffers = usedMode === 'ram'
        ? await extractFramesAutoFromBuffer(ffmpegPath, framesSource, Math.floor(frames), durationSec)
        : await extractFrames(ffmpegPath, framesSource, stamps, { concurrency: extractConc });
      if (usedMode === 'ram' && (!frameBuffers || frameBuffers.length === 0)) {
        // Fallback to disk
        const loc = await ensureLocalPathWithCleanup(videoUrl);
        fallbackToDisk = true; usedMode = 'disk';
        try {
          frameBuffers = await extractFrames(ffmpegPath, loc.path, stamps, { concurrency: extractConc });
          const { default: fs } = await import('fs/promises');
          const st = await fs.stat(loc.path).catch(() => null);
          sizeBytes = st?.size ?? sizeBytes;
        } finally {
          await loc.cleanup();
        }
      }

      if (turboMode) {
        const t0 = Date.now();
        const prepConc = Math.max(1, Math.min(32, preprocessConcurrency ?? Math.min(8, cpuCount)));
        const jobs = frameBuffers.map((buf, i) => async () => {
          const sized = await toSizedRGBTensor(buf, targetW, targetH, { centerCrop });
          return tf.tensor3d(sized.data, [targetH, targetW, 3], 'int32');
        });
        const queue = jobs.slice();
        const tensors = new Array(jobs.length);
        const runners = new Array(Math.max(1, Math.min(prepConc, queue.length))).fill(0).map(async function run() {
          while (queue.length) {
            const idx = jobs.length - queue.length;
            const job = queue.shift();
            if (!job) break;
            tensors[idx] = await job();
          }
        });
        await Promise.all(runners);
        const t1 = Date.now();
        const logits = tf.tidy(() => {
          const offset = tf.scalar(127.5);
          const batch = tf.stack(tensors.map(t => t.toFloat().sub(offset).div(offset)));
          const out = this.model.predict(batch);
          return Array.isArray(out) ? out[0] : out;
        });
        tensors.forEach(t => t.dispose());
        const t2 = Date.now();
        const [probs] = await Promise.all([logits.array()]);
        logits.dispose();
        frameCount = probs.length;
        for (let f = 0; f < probs.length; f++) {
          for (let c = 0; c < probs[f].length; c++) aggregateScores[c] += probs[f][c];
        }
        const k = Math.min(topK ?? this.model.classes.length, this.model.classes.length);
        const results = [];
        for (let i = 0; i < probs.length; i++) {
          const scores = probs[i];
          const idxs = scores.map((s, ci) => ci).sort((a, b) => scores[b] - scores[a]).slice(0, k);
          const preds = idxs.map((ci, j) => ({ class: this.model.classes[ci], score: scores[ci], rank: j + 1 }));
          results.push({ frameIndex: i, timestampSec: stamps[i] ?? null, predictions: preds });
        }
        const t3 = Date.now();
        const tEnd = Date.now();
        const avg = aggregateScores.map(s => s / Math.max(1, frameCount));
        const order = avg.map((s, i) => i).sort((a, b) => avg[b] - avg[a]);
        const overall = order.map((ci, j) => ({ class: this.model.classes[ci], score: avg[ci], rank: j + 1 })).slice(0, k);
        const out = {
          input: { videoUrl, frames: Math.floor(frames), turboMode: true },
          backend: tf.getBackend(),
          modelInfo: { classesCount: this.model.classes.length },
          timings: {
            downloadPrepareMs: tPrepEnd - tStart,
            decodeResizeMs: t1 - t0,
            inferenceMs: t2 - t1,
            postprocessMs: t3 - t2,
            totalMs: tEnd - tStart
          },
          io: { mode: usedMode, fallbackToDisk, tempCleaned, sizeBytes, maxBytes },
          results,
          aggregate: { predictions: overall }
        };
        await cleanup(); tempCleaned = true;
        return out;
      } else {
        const results = [];
        let decodeResizeMs = 0; let inferenceMs = 0; let postprocessMs = 0;
        for (let i = 0; i < frameBuffers.length; i++) {
          const tA = Date.now();
          const sized = await toSizedRGBTensor(frameBuffers[i], targetW, targetH, { centerCrop });
          const imageTensor = tf.tensor3d(sized.data, [targetH, targetW, 3], 'int32');
          const tB = Date.now();
          const logits = tf.tidy(() => {
            const offset = tf.scalar(127.5);
            const norm = imageTensor.toFloat().sub(offset).div(offset).expandDims(0);
            const out = this.model.predict(norm);
            return Array.isArray(out) ? out[0] : out;
          });
          const tC = Date.now();
          imageTensor.dispose();
          const probs = await logits.array();
          logits.dispose();
          frameCount += 1;
          for (let c = 0; c < probs[0].length; c++) aggregateScores[c] += probs[0][c];
          const k = Math.min(topK ?? this.model.classes.length, this.model.classes.length);
          const scores = probs[0];
          const idxs = scores.map((s, ci) => ci).sort((a, b) => scores[b] - scores[a]).slice(0, k);
          const preds = idxs.map((ci, j) => ({ class: this.model.classes[ci], score: scores[ci], rank: j + 1 }));
          const tD = Date.now();
          results.push({ frameIndex: i, timestampSec: stamps[i] ?? null, predictions: preds });
          decodeResizeMs += tB - tA; inferenceMs += tC - tB; postprocessMs += tD - tC;
        }
        const tEnd = Date.now();
        const avg = aggregateScores.map(s => s / Math.max(1, frameCount));
        const order = avg.map((s, i) => i).sort((a, b) => avg[b] - avg[a]);
        const overall = order.map((ci, j) => ({ class: this.model.classes[ci], score: avg[ci], rank: j + 1 })).slice(0, Math.min(topK ?? this.model.classes.length, this.model.classes.length));
        const out = {
          input: { videoUrl, frames: Math.floor(frames), turboMode: false },
          backend: tf.getBackend(),
          modelInfo: { classesCount: this.model.classes.length },
          timings: {
            downloadPrepareMs: tPrepEnd - tStart,
            decodeResizeMs,
            inferenceMs,
            postprocessMs,
            totalMs: tEnd - tStart
          },
          io: { mode: usedMode, fallbackToDisk, tempCleaned, sizeBytes, maxBytes },
          results,
          aggregate: { predictions: overall }
        };
        await cleanup(); tempCleaned = true;
        return out;
      }
    } finally {
      try { await cleanup(); tempCleaned = true; } catch {}
    }
  }

  /**
   * Classifies multiple videos or GIFs. Processes videos sequentially by default to limit memory.
   * @param {object} options
   * @param {string[]} options.videoUrls
   * @param {number} [options.frames=10]
   * @param {number} [options.topK]
   * @param {boolean} [options.centerCrop=true]
   * @param {boolean} [options.resizeOnCPU=true]
   * @param {boolean} [options.turboMode=false]
   * @param {number} [options.extractionConcurrency]
   * @param {number} [options.preprocessConcurrency]
   * @param {number} [options.maxConcurrent=2]
   * @returns {Promise<object>} Batch result with per-video outputs.
   */
  async classifyVideoBatch({ videoUrls, ...rest }) {
    if (!Array.isArray(videoUrls) || videoUrls.length === 0) throw new Error('videoUrls must be a non-empty array');
    return this.classifyVideo({ videoUrl: videoUrls, ...rest });
  }
}
