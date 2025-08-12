import { parentPort } from 'worker_threads';
import sharp from 'sharp';

parentPort.on('message', async (msg) => {
  try {
    const { buffer, targetW, targetH, centerCrop } = msg || {};
    if (!buffer || !Number.isFinite(targetW) || !Number.isFinite(targetH)) {
      return parentPort.postMessage({ ok: false, error: 'Invalid worker message' });
    }
    const { data } = await sharp(buffer)
      .resize(targetW, targetH, { fit: centerCrop ? 'cover' : 'fill', fastShrinkOnLoad: true })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    parentPort.postMessage({ ok: true, data: new Uint8Array(data) });
  } catch (err) {
    parentPort.postMessage({ ok: false, error: err?.message || String(err) });
  }
});
