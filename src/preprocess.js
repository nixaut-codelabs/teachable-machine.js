import sharp from 'sharp';
import fs from 'fs/promises';
import { http } from './utils/net.js';
import { Worker } from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

export async function getImageBuffer(imageUrl) {
  if (Buffer.isBuffer(imageUrl) || imageUrl instanceof Uint8Array) {
    return Buffer.isBuffer(imageUrl) ? imageUrl : Buffer.from(imageUrl);
  }
  if (typeof imageUrl !== 'string' || imageUrl.length === 0) {
    throw new Error('imageUrl must be a non-empty string or Buffer/Uint8Array');
  }
  if (imageUrl.startsWith('data:image/')) {
    const base64Data = imageUrl.split(',')[1];
    if (!base64Data) throw new Error('Invalid Base64 data URI');
    return Buffer.from(base64Data, 'base64');
  }
  if (imageUrl.startsWith('http')) {
    try {
      return await http(imageUrl).buffer();
    } catch (error) {
      throw new Error(`Failed to download image. Status: ${error.response ? error.response.statusCode : error.message}`);
    }
  }
  try {
    return await fs.readFile(imageUrl);
  } catch (error) {
    const looksBase64 = /^[A-Za-z0-9+/=]+$/.test(imageUrl) && (imageUrl.length % 4 === 0);
    if (looksBase64) {
      try { return Buffer.from(imageUrl, 'base64'); } catch {}
    }
    if (error.code === 'ENOENT') throw new Error(`Local file not found: ${imageUrl}`);
    throw error;
  }
}

let PREPROCESS_OPTS = { useWorkers: false };

export function setPreprocessOptions(opts = {}) {
  PREPROCESS_OPTS = { ...PREPROCESS_OPTS, ...opts };
}

function runWorkerResize(imageBuffer, targetW, targetH, centerCrop) {
  return new Promise((resolve, reject) => {
    try {
      const __filename = fileURLToPath(import.meta.url);
      const __dirname = dirname(__filename);
      const workerPath = new URL('./workers/resize-worker.js', `file://${__dirname}/`).pathname;
      const worker = new Worker(workerPath, {
        workerData: null
      });
      worker.once('message', (msg) => {
        if (msg && msg.ok) {
          resolve({ data: Buffer.from(msg.data), width: targetW, height: targetH });
        } else {
          reject(new Error(msg?.error || 'Worker resize failed'));
        }
      });
      worker.once('error', reject);
      worker.postMessage({
        buffer: Buffer.isBuffer(imageBuffer) ? imageBuffer : Buffer.from(imageBuffer),
        targetW,
        targetH,
        centerCrop
      });
    } catch (err) {
      reject(err);
    }
  });
}

export async function toSizedRGBTensor(imageBuffer, targetW, targetH, { centerCrop = true } = {}) {
  if (PREPROCESS_OPTS.useWorkers) {
    try {
      return await runWorkerResize(imageBuffer, targetW, targetH, centerCrop);
    } catch (e) {
      // Fallback to local sharp on failure
    }
  }
  const { data } = await sharp(imageBuffer)
    .resize(targetW, targetH, { fit: centerCrop ? 'cover' : 'fill', fastShrinkOnLoad: true })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: targetW, height: targetH };
}
