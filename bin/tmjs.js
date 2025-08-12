#!/usr/bin/env node
import { argv, exit } from 'node:process';
import fs from 'node:fs/promises';
import TeachableMachine from '../src/index.js';

function printHelp() {
  console.log(`tmjs - Teachable Machine CLI\n\nUsage:\n  tmjs --model <url|dir> [--backend tfjs|tfjs-node] [--io ram|disk] [--frames N] [--topK K] [--turbo] [--maxBytes BYTES] [--media image|video|auto] <inputs...>\n\nExamples:\n  tmjs --model https://teachablemachine.withgoogle.com/models/XXX/ image.jpg\n  tmjs --model ./model --media video --frames 8 --turbo video.mp4 gif.gif\n`);
}

function parseArgs() {
  const args = argv.slice(2);
  const opts = { positional: [] };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--help' || a === '-h') return { help: true };
    if (a === '--model') { opts.model = args[++i]; continue; }
    if (a === '--backend') { opts.backend = args[++i]; continue; }
    if (a === '--io') { opts.io = args[++i]; continue; }
    if (a === '--frames') { opts.frames = Number(args[++i]); continue; }
    if (a === '--topK') { opts.topK = Number(args[++i]); continue; }
    if (a === '--maxBytes') { opts.maxBytes = Number(args[++i]); continue; }
    if (a === '--turbo') { opts.turbo = true; continue; }
    if (a === '--media') { opts.media = args[++i]; continue; }
    if (!a.startsWith('-')) { opts.positional.push(a); continue; }
  }
  return opts;
}

async function main() {
  const opts = parseArgs();
  if (opts.help || !opts.model || opts.positional.length === 0) {
    printHelp();
    return;
  }
  const tm = await TeachableMachine.create({
    modelUrl: opts.model?.startsWith('http') ? opts.model : undefined,
    modelDir: !opts.model?.startsWith('http') ? opts.model : undefined,
    loadFrom: 'auto', saveToDir: !opts.model?.startsWith('http') ? opts.model : undefined,
    backend: opts.backend || 'tfjs',
    ioMode: (opts.io === 'disk') ? 'disk' : 'ram'
  });

  const mediaType = opts.media || 'auto';
  const frames = Number.isFinite(opts.frames) ? opts.frames : 10;
  const topK = Number.isFinite(opts.topK) ? opts.topK : undefined;
  const maxBytes = Number.isFinite(opts.maxBytes) ? opts.maxBytes : 10 * 1024 * 1024;
  const inputs = opts.positional;

  let res;
  if (mediaType === 'image') {
    res = await tm.classifyImages({ images: inputs, topK });
  } else if (mediaType === 'video') {
    res = await tm.classifyVideos({ videos: inputs, frames, topK, turboMode: !!opts.turbo, maxBytes });
  } else {
    // auto: treat inputs as images by default; if one item and --frames set, treat as video
    const input = (inputs.length === 1 && frames > 0) ? { videos: inputs } : { images: inputs };
    res = await tm.classify({ input, frames, topK, turboMode: !!opts.turbo, mediaType: 'auto', maxBytes });
  }
  console.log(JSON.stringify(res, null, 2));
}

main().catch((err) => { console.error(err?.stack || err?.message || String(err)); exit(1); });
