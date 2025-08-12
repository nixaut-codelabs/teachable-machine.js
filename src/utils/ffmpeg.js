import ffmpegStatic from 'ffmpeg-static';
import { execFile, spawn } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import path from 'path';
import fs from 'fs/promises';
import { http } from './net.js';

const execFileAsync = promisify(execFile);

/**
 * Ensures FFmpeg is available and returns the executable path.
 * Prefers ffmpeg-static, falls back to system ffmpeg.
 * Throws if not available.
 */
export async function ensureFFmpeg() {
  if (ffmpegStatic) {
    return ffmpegStatic;
  }
  try {
    await execFileAsync('ffmpeg', ['-version'], { windowsHide: true });
    return 'ffmpeg';
  } catch {
    throw new Error('FFmpeg is required but not available. Install ffmpeg or add ffmpeg-static.');
  }
}

/**
 * Downloads a remote file to a temporary path and returns the local path.
 * For local paths, returns the original path.
 */
export async function ensureLocalPath(src) {
  if (!src || typeof src !== 'string') throw new Error('Invalid source');
  if (src.startsWith('http')) {
    const buf = await http(src).buffer();
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmvid-'));
    const file = path.join(dir, 'input');
    await fs.writeFile(file, buf);
    return file;
  }
  return src;
}

/**
 * Returns a local path for arbitrary media input and a cleanup() function if a temp dir was created.
 * Accepts: URL, local path, Buffer/Uint8Array, data URI, or plain base64 string.
 */
export async function ensureLocalPathWithCleanup(src) {
  let dir = null;
  const writeTemp = async (buffer) => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmvid-'));
    const file = path.join(dir, 'input');
    await fs.writeFile(file, buffer);
    return file;
  };

  if (Buffer.isBuffer(src) || src instanceof Uint8Array) {
    const buf = Buffer.isBuffer(src) ? src : Buffer.from(src);
    const file = await writeTemp(buf);
    return { path: file, cleanup: async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); } };
  }
  if (typeof src !== 'string' || src.length === 0) throw new Error('Invalid source');
  if (src.startsWith('http')) {
    const buf = await http(src).buffer();
    const file = await writeTemp(buf);
    return { path: file, cleanup: async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); } };
  }
  if (src.startsWith('data:')) {
    const base64 = src.split(',')[1];
    if (!base64) throw new Error('Invalid data URI');
    const buf = Buffer.from(base64, 'base64');
    const file = await writeTemp(buf);
    return { path: file, cleanup: async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); } };
  }
  const looksBase64 = /^[A-Za-z0-9+/=]+$/.test(src) && (src.length % 4 === 0);
  if (looksBase64) {
    try {
      const buf = Buffer.from(src, 'base64');
      const file = await writeTemp(buf);
      return { path: file, cleanup: async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); } };
    } catch {}
  }
  return { path: src, cleanup: async () => {} };
}

/**
 * Returns a Buffer for arbitrary media input kept fully in memory.
 * Accepts URL, local path, Buffer/Uint8Array, data URI, or plain base64 string.
 */
export async function getMediaBuffer(src) {
  if (Buffer.isBuffer(src) || src instanceof Uint8Array) {
    return Buffer.isBuffer(src) ? src : Buffer.from(src);
  }
  if (typeof src !== 'string' || src.length === 0) throw new Error('Invalid source');
  if (src.startsWith('http')) return await http(src).buffer();
  if (src.startsWith('data:')) {
    const base64 = src.split(',')[1];
    if (!base64) throw new Error('Invalid data URI');
    return Buffer.from(base64, 'base64');
  }
  const looksBase64 = /^[A-Za-z0-9+/=]+$/.test(src) && (src.length % 4 === 0);
  if (looksBase64) {
    try { return Buffer.from(src, 'base64'); } catch {}
  }
  return await fs.readFile(src);
}

/**
 * Parses HH:MM:SS.xx duration from ffmpeg stderr output.
 */
export function parseDurationSec(stderr) {
  const m = /Duration:\s*(\d+):(\d+):(\d+\.\d+)/.exec(stderr);
  if (!m) return null;
  const h = parseInt(m[1], 10);
  const min = parseInt(m[2], 10);
  const s = parseFloat(m[3]);
  return h * 3600 + min * 60 + s;
}

/**
 * Probes media duration in seconds using ffmpeg stderr parsing.
 */
export async function probeDurationSec(ffmpegPath, inputPath) {
  try {
    await execFileAsync(ffmpegPath, ['-i', inputPath], { windowsHide: true });
    return null;
  } catch (e) {
    const stderr = String(e?.stderr || e?.message || '');
    return parseDurationSec(stderr);
  }
}

/**
 * Probes media duration from a Buffer using ffmpeg reading stdin (pipe:0).
 */
export async function probeDurationSecFromBuffer(ffmpegPath, buffer) {
  return new Promise((resolve) => {
    const child = spawn(ffmpegPath, ['-i', 'pipe:0'], { windowsHide: true });
    let stderr = '';
    child.stderr.on('data', d => { stderr += d.toString(); });
    child.on('close', () => {
      resolve(parseDurationSec(stderr));
    });
    child.stdin.on('error', () => {});
    child.stdin.end(buffer);
  });
}

/**
 * Extracts N evenly spaced center timestamps across the duration.
 */
export function sampleTimestamps(durationSec, count) {
  if (!durationSec || durationSec <= 0 || !count || count <= 0) return [];
  const ts = [];
  for (let k = 0; k < count; k++) {
    const t = ((k + 0.5) / count) * durationSec;
    ts.push(t);
  }
  return ts;
}

/**
 * Clamps timestamps to a safe range within media duration to avoid end-of-file seeks.
 */
export function safeTimestamps(durationSec, timestamps, marginSec = 0.05) {
  const maxT = Math.max(0, durationSec - Math.max(0, marginSec));
  return timestamps.map(t => Math.min(Math.max(0, t), maxT));
}

/**
 * Extracts frames as PNG buffers at given timestamps using separate seeks.
 * @param {string} ffmpegPath
 * @param {string} inputPath
 * @param {number[]} timestampsSec
 * @param {{concurrency?: number}} [options]
 */
export async function extractFrames(ffmpegPath, inputPath, timestampsSec, options = {}) {
  const { concurrency = 1 } = options;
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'tmframes-'));
  const jobs = timestampsSec.map((t, idx) => async () => {
    const seek = Math.max(0, t);
    const out = path.join(dir, `frame_${String(idx).padStart(3, '0')}.png`);
    try {
      await execFileAsync(ffmpegPath, ['-y', '-ss', String(seek), '-i', inputPath, '-frames:v', '1', '-f', 'image2', out], { windowsHide: true });
    } catch {}
    try {
      return await fs.readFile(out);
    } catch {
      const retryT = Math.max(0, seek - 0.1);
      try {
        await execFileAsync(ffmpegPath, ['-y', '-ss', String(retryT), '-i', inputPath, '-frames:v', '1', '-f', 'image2', out], { windowsHide: true });
        return await fs.readFile(out);
      } catch {
        return undefined;
      }
    }
  });

  const queue = jobs.slice();
  const results = new Array(jobs.length);
  const runners = new Array(Math.max(1, Math.min(concurrency, queue.length))).fill(0).map(async function run() {
    while (queue.length) {
      const nextIdx = jobs.length - queue.length;
      const job = queue.shift();
      if (!job) break;
      const buf = await job();
      results[nextIdx] = buf;
    }
  });
  await Promise.all(runners);
  const buffers = results.filter(Boolean);
  await fs.rm(dir, { recursive: true, force: true });
  return buffers;
}

/**
 * Extract frames as PNG buffers directly from a media Buffer via stdin and image2pipe stdout.
 */
export async function extractFramesFromBuffer(ffmpegPath, buffer, timestampsSec, options = {}) {
  const { concurrency = 1 } = options;
  const jobs = timestampsSec.map((t) => async () => {
    const seek = Math.max(0, t);
    const args = ['-y', '-ss', String(seek), '-i', 'pipe:0', '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'];
    return new Promise((resolve) => {
      const child = spawn(ffmpegPath, args, { windowsHide: true });
      const chunks = [];
      child.stdout.on('data', d => chunks.push(d));
      child.stderr.on('data', () => {});
      child.on('close', () => {
        if (chunks.length === 0) {
          const retryArgs = ['-y', '-ss', String(Math.max(0, seek - 0.1)), '-i', 'pipe:0', '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'];
          const retry = spawn(ffmpegPath, retryArgs, { windowsHide: true });
          const rchunks = [];
          retry.stdout.on('data', d => rchunks.push(d));
          retry.on('close', () => resolve(rchunks.length ? Buffer.concat(rchunks) : undefined));
          retry.stdin.on('error', () => {});
          retry.stdin.end(buffer);
        } else {
          resolve(Buffer.concat(chunks));
        }
      });
      child.stdin.on('error', () => {});
      child.stdin.end(buffer);
    });
  });
  const queue = jobs.slice();
  const results = new Array(jobs.length);
  const runners = new Array(Math.max(1, Math.min(concurrency, queue.length))).fill(0).map(async function run() {
    while (queue.length) {
      const idx = jobs.length - queue.length;
      const job = queue.shift();
      if (!job) break;
      results[idx] = await job();
    }
  });
  await Promise.all(runners);
  return results.filter(Boolean);
}

/**
 * Extract approximately 'frames' evenly spaced frames from a Buffer in a single ffmpeg pass using fps filter.
 * Splits concatenated PNGs from stdout by signature.
 */
export async function extractFramesAutoFromBuffer(ffmpegPath, buffer, frames, durationSec) {
  const fps = Math.max(0.1, Math.min(60, frames / Math.max(0.1, durationSec)));
  const args = ['-hide_banner', '-loglevel', 'error', '-y', '-i', 'pipe:0', '-vf', `fps=${fps}`, '-vsync', 'vfr', '-frames:v', String(frames), '-f', 'image2pipe', '-vcodec', 'png', '-'];
  const chunks = [];
  await new Promise((resolve) => {
    const child = spawn(ffmpegPath, args, { windowsHide: true });
    child.stdout.on('data', d => chunks.push(d));
    child.on('close', () => resolve());
    child.stdin.on('error', () => {});
    child.stdin.end(buffer);
  });
  const all = chunks.length ? Buffer.concat(chunks) : Buffer.alloc(0);
  if (all.length === 0) return [];
  const sig = Buffer.from([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]);
  const idxs = [];
  for (let i = 0; i <= all.length - sig.length; i++) {
    let ok = true;
    for (let j = 0; j < sig.length; j++) { if (all[i + j] !== sig[j]) { ok = false; break; } }
    if (ok) idxs.push(i);
  }
  const outs = [];
  for (let k = 0; k < idxs.length; k++) {
    const start = idxs[k];
    const end = (k + 1 < idxs.length) ? idxs[k + 1] : all.length;
    outs.push(all.slice(start, end));
  }
  return outs.slice(0, frames);
}
