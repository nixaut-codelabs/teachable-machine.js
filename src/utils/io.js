import fs from 'fs/promises';
import path from 'path';

export const dirExists = async (p) => {
  try { const s = await fs.stat(p); return s.isDirectory(); } catch { return false; }
};

export const ioFromDir = (dirPath) => ({
  load: async () => {
    const jsonPath = path.join(dirPath, 'model.json');
    const weightsPath = path.join(dirPath, 'weights.bin');
    const jsonRaw = await fs.readFile(jsonPath, 'utf-8');
    const { modelTopology, weightSpecs } = JSON.parse(jsonRaw);
    const weightData = new Uint8Array(await fs.readFile(weightsPath)).buffer;
    return { modelTopology, weightSpecs, weightData };
  },
  save: async (artifacts) => {
    await fs.mkdir(dirPath, { recursive: true });
    const json = JSON.stringify({ modelTopology: artifacts.modelTopology, weightSpecs: artifacts.weightSpecs });
    await fs.writeFile(path.join(dirPath, 'model.json'), json);
    await fs.writeFile(path.join(dirPath, 'weights.bin'), Buffer.from(artifacts.weightData));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON', weightDataBytes: artifacts.weightData.byteLength } };
  }
});

export const readMetadata = async (dirPath) => {
  const metaStr = await fs.readFile(path.join(dirPath, 'metadata.json'), 'utf-8');
  return JSON.parse(metaStr);
};

export const writeMetadata = async (dirPath, metadata) => {
  await fs.mkdir(dirPath, { recursive: true });
  await fs.writeFile(path.join(dirPath, 'metadata.json'), JSON.stringify(metadata, null, 2));
};
