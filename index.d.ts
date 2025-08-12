// TypeScript type definitions for teachable-machine.js

export interface CreateOptions {
  modelUrl?: string;
  modelDir?: string;
  loadFrom?: 'auto' | 'dir';
  saveToDir?: string;
  warmup?: boolean;
  ioMode?: 'ram' | 'disk';
  backend?: 'tfjs' | 'tfjs-node';
  preprocessUseWorkers?: boolean;
}

export interface TimingInfo {
  downloadMs?: number;
  decodeResizeMs?: number;
  inferenceMs?: number;
  postprocessMs?: number;
  totalMs?: number;
  endToEndMs?: number;
}

export interface IOInfo {
  mode?: 'ram' | 'disk';
  fallbackToDisk?: boolean;
  tempCleaned?: boolean;
  sizeBytes?: number;
  maxBytes?: number;
}

export interface ImageResult {
  input: { imageUrl: any };
  backend: string;
  modelInfo?: { classesCount: number };
  preprocess?: { target: { width: number; height: number }; centerCrop?: boolean; resizeOnCPU?: boolean };
  timings: TimingInfo;
  predictions?: Array<{ class: string; score: number; rank: number }>;
  error?: string;
}

export interface BatchImageResult {
  backend: string;
  count: number;
  modelInfo?: { classesCount: number };
  timings: { endToEndMs: number };
  results: ImageResult[];
}

export interface FramePrediction {
  frameIndex: number;
  timestampSec: number | null;
  predictions: Array<{ class: string; score: number; rank: number }>;
}

export interface VideoResult {
  input: { videoUrl: any; frames: number; turboMode?: boolean };
  backend: string;
  modelInfo?: { classesCount: number };
  timings: Record<string, number>;
  frames?: FramePrediction[];
  aggregate: { predictions: Array<{ class: string; score: number; rank: number }> };
  io?: IOInfo;
  error?: string;
}

export default class TeachableMachine {
  static create(options?: CreateOptions): Promise<TeachableMachine>;
  constructor(model: any);
  backend: 'tfjs' | 'tfjs-node';
  ioMode: 'ram' | 'disk';

  classify(options: {
    input: any | any[] | { images?: any[]; videos?: any[] };
    mediaType?: 'auto' | 'image' | 'video';
    frames?: number; topK?: number; centerCrop?: boolean; resizeOnCPU?: boolean;
    turboMode?: boolean; extractionConcurrency?: number; preprocessConcurrency?: number; maxConcurrent?: number; maxBytes?: number; batchSize?: number;
  }): Promise<BatchImageResult | VideoResult | { images: BatchImageResult | ImageResult; videos: VideoResult[] | VideoResult | null } | ImageResult>;

  classifyImages(options: { images: any | any[]; topK?: number; centerCrop?: boolean; resizeOnCPU?: boolean; batchSize?: number }): Promise<ImageResult | BatchImageResult>;
  classifyBatch(options: { imageUrls: any[]; topK?: number; centerCrop?: boolean; resizeOnCPU?: boolean; batchSize?: number }): Promise<BatchImageResult>;
  batchImageClassify(options: { imageUrls: any[]; topK?: number; centerCrop?: boolean; resizeOnCPU?: boolean; batchSize?: number }): Promise<BatchImageResult>;

  classifyVideos(options: { videos: any | any[]; frames?: number; topK?: number; centerCrop?: boolean; resizeOnCPU?: boolean; turboMode?: boolean; extractionConcurrency?: number; preprocessConcurrency?: number; maxConcurrent?: number; maxBytes?: number }): Promise<VideoResult | VideoResult[]>;

  dispose(): void;
}
