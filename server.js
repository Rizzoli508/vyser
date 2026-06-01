import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import OpenAI, { toFile } from 'openai';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const upload = multer({ dest: 'uploads/' });

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(express.static('public'));

async function fileToOpenAI(filePath, mimetype, name) {
  const buffer = fs.readFileSync(filePath);
  return await toFile(buffer, name, { type: mimetype });
}

// Corrige o fundo para #F3F4F6 exato:
// - Pixels próximos ao fundo (dentro do threshold) → exatamente #F3F4F6
// - Pixels de sombra/óculos (mais escuros) → offset proporcional preservando contraste
async function correctBackground(imageBuffer) {
  const { width, height } = await sharp(imageBuffer).metadata();
  const sz  = 40;
  const TARGET = [243, 244, 246]; // #F3F4F6
  const THRESHOLD = 12; // pixels dentro de 12 unidades do fundo = background puro

  const raw = await sharp(imageBuffer).removeAlpha().raw().toBuffer();

  // Amostra os 4 cantos para obter a cor de referência do fundo
  let sum = [0, 0, 0], count = 0;
  for (const [cx, cy] of [[0,0],[width-sz,0],[0,height-sz],[width-sz,height-sz]]) {
    for (let dy = 0; dy < sz; dy++) {
      for (let dx = 0; dx < sz; dx++) {
        const i = ((cy + dy) * width + (cx + dx)) * 3;
        sum[0] += raw[i]; sum[1] += raw[i+1]; sum[2] += raw[i+2];
        count++;
      }
    }
  }
  const bgRef = sum.map(s => s / count);
  const off   = TARGET.map((t, i) => t - bgRef[i]);

  console.log(`[bg-correct] bgRef=(${bgRef.map(v=>Math.round(v)).join(',')}) off=(${off.map(v=>Math.round(v)).join(',')})`);

  // Flood fill a partir das bordas — só pixels conectados ao fundo real são substituídos
  const isBg = new Uint8Array(width * height);
  const queue = [];

  const seed = (x, y) => {
    const idx = y * width + x;
    if (isBg[idx]) return;
    const i = idx * 3;
    const dist = Math.max(Math.abs(raw[i]-bgRef[0]), Math.abs(raw[i+1]-bgRef[1]), Math.abs(raw[i+2]-bgRef[2]));
    if (dist <= THRESHOLD) { isBg[idx] = 1; queue.push(idx); }
  };

  // Semeia todas as bordas da imagem
  for (let x = 0; x < width;  x++) { seed(x, 0); seed(x, height - 1); }
  for (let y = 0; y < height; y++) { seed(0, y); seed(width - 1, y); }

  // BFS
  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % width, y = Math.floor(idx / width);
    for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const nidx = ny * width + nx;
      if (isBg[nidx]) continue;
      const ni = nidx * 3;
      const dist = Math.max(Math.abs(raw[ni]-bgRef[0]), Math.abs(raw[ni+1]-bgRef[1]), Math.abs(raw[ni+2]-bgRef[2]));
      if (dist <= THRESHOLD) { isBg[nidx] = 1; queue.push(nidx); }
    }
  }

  // Erosão da máscara: remove pixels de fundo vizinhos a pixels de não-fundo
  // Evita comer as bordas dos óculos (2 passes = 2px de margem)
  const erode = (mask) => {
    const out = new Uint8Array(mask);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (!mask[idx]) continue;
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]]) {
          const nx = x+dx, ny = y+dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          if (!mask[ny * width + nx]) { out[idx] = 0; break; }
        }
      }
    }
    return out;
  };

  const erodedBg = erode(erode(erode(isBg))); // 3 passes = 3px de margem segura

  // Aplica: fundo erodido → #F3F4F6 exato | resto → offset
  for (let idx = 0; idx < width * height; idx++) {
    const i = idx * 3;
    if (erodedBg[idx]) {
      raw[i]   = TARGET[0];
      raw[i+1] = TARGET[1];
      raw[i+2] = TARGET[2];
    } else {
      raw[i]   = Math.min(255, Math.max(0, Math.round(raw[i]   + off[0])));
      raw[i+1] = Math.min(255, Math.max(0, Math.round(raw[i+1] + off[1])));
      raw[i+2] = Math.min(255, Math.max(0, Math.round(raw[i+2] + off[2])));
    }
  }

  return sharp(raw, { raw: { width, height, channels: 3 } }).png().toBuffer();
}

const PROMPT_FRENTE = `Image 1 is a style reference — follow its background, lighting, shadow, and FRONT-FACING composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in a FRONT VIEW (straight at the camera), centered, with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_LADO = `Image 1 is a style reference — follow its background, lighting, shadow, and SIDE/PROFILE composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in a SIDE/PROFILE VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_INCLINADO = `Image 1 is a style reference — follow its background, lighting, shadow, and ANGLED/3-QUARTER composition exactly.
Images 2+ show the glasses model to use.

Generate a product photo of the glasses from Images 2+ in an ANGLED/3-QUARTER VIEW (like Image 1), with soft studio lighting and shadow style as Image 1. Reproduce the glasses shape, color, and details precisely.
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

const PROMPT_MODEL_SEM_EXPRESSAO = `Image 1 is the model reference photo. Image 2 shows the glasses. Image 3 shows the outfit.

Generate a professional fashion photo where the model from Image 1 is wearing the glasses from Image 2 and the clothing from Image 3.
- Preserve the model's face, skin, and hair exactly as in Image 1
- Allow only very subtle natural variation: slight micro-expression shift and minor hair strand movement — to create a natural feel
- Place the glasses naturally and precisely on the model's face, preserving their exact shape, color, lenses, and frame
- Dress the model in the exact outfit shown in Image 3
- Professional studio lighting, soft and clean
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture.`;

const PROMPT_MODEL_COM_EXPRESSAO = `Image 1 is the model reference photo. Image 2 shows the glasses. Image 3 shows the outfit. Image 4 is a facial expression reference.

Generate a professional fashion photo where the model from Image 1 is wearing the glasses from Image 2 and the clothing from Image 3.
- Preserve the model's face, skin, and hair exactly as in Image 1
- Replicate only the facial expression from Image 4 (mouth position, eye openness, brow shape) onto the model — do NOT copy the face, identity, skin tone or any other feature of the person in Image 4
- Place the glasses naturally and precisely on the model's face, preserving their exact shape, color, lenses, and frame
- Dress the model in the exact outfit shown in Image 3
- Professional studio lighting, soft and clean
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture.`;

const PROMPT_MODEL_FRENTE = PROMPT_MODEL_SEM_EXPRESSAO;
const PROMPT_MODEL_LADINHO = PROMPT_MODEL_SEM_EXPRESSAO;

const PROMPT_SOMBRA = `Generate a clean professional product photo of the glasses shown in the images.
- Glasses: front view, horizontally centered
- Soft drop shadow directly beneath the glasses
- Professional studio lighting
IMPORTANT: the background MUST be solid #F3F4F6. No gradients, no texture, no other color.`;

app.post('/api/generate', upload.array('images', 10), async (req, res) => {
  const uploadedPaths = (req.files || []).map(f => f.path);
  try {
    const { view } = req.body;
    if (!req.files || req.files.length === 0)
      return res.status(400).json({ error: 'Nenhuma imagem enviada.' });

    let prompt, refPath, refName;
    if (view === 'lado') {
      prompt = PROMPT_LADO;
      refPath = path.join(__dirname, 'public/references/lado.png');
      refName = 'ref-lado.png';
    } else if (view === 'inclinado') {
      prompt = PROMPT_INCLINADO;
      refPath = path.join(__dirname, 'public/references/inclinado.png');
      refName = 'ref-inclinado.png';
    } else {
      prompt = PROMPT_FRENTE;
      refPath = path.join(__dirname, 'public/references/frente.png');
      refName = 'ref-frente.png';
    }

    const refFile = await fileToOpenAI(refPath, 'image/png', refName);
    const productFiles = await Promise.all(
      req.files.map((f, i) => fileToOpenAI(f.path, f.mimetype, `oculos-${i + 1}.${f.mimetype.split('/')[1] || 'jpg'}`))
    );

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    console.log(`[generate] view=${view} ref + ${productFiles.length} produto(s)`);

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: [refFile, ...productFiles],
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    const corrected = await correctBackground(Buffer.from(b64, 'base64'));
    res.json({ image: corrected.toString('base64') });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/png-sombra', upload.array('images', 10), async (req, res) => {
  const uploadedPaths = (req.files || []).map(f => f.path);
  try {
    if (!req.files || req.files.length === 0)
      return res.status(400).json({ error: 'Nenhuma imagem enviada.' });

    const productFiles = await Promise.all(
      req.files.map((f, i) => fileToOpenAI(f.path, f.mimetype, `oculos-${i + 1}.${f.mimetype.split('/')[1] || 'jpg'}`))
    );

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    console.log(`[png-sombra] ${productFiles.length} imagem(ns)`);

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: productFiles,
      prompt: PROMPT_SOMBRA,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    const corrected = await correctBackground(Buffer.from(b64, 'base64'));
    res.json({ image: corrected.toString('base64') });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Lista modelos disponíveis em public/models/
// Convenção: nome-frente.jpg e nome-ladinho.jpg
app.get('/api/models', (req, res) => {
  const modelsDir = path.join(__dirname, 'public/models');
  const exts = ['.jpg', '.jpeg', '.png', '.webp'];
  const files = fs.readdirSync(modelsDir)
    .filter(f => exts.includes(path.extname(f).toLowerCase()));

  const map = {};
  files.forEach(f => {
    const base = path.basename(f, path.extname(f));
    const mFrente  = base.match(/^(.+)-frente$/i);
    const mLadinho = base.match(/^(.+)-ladinho$/i);
    if (mFrente) {
      const name = mFrente[1];
      if (!map[name]) map[name] = { name };
      map[name].frente = f;
    } else if (mLadinho) {
      const name = mLadinho[1];
      if (!map[name]) map[name] = { name };
      map[name].ladinho = f;
    } else {
      if (!map[base]) map[base] = { name: base };
      map[base].frente = f;
    }
  });

  res.json(Object.values(map).sort((a, b) => a.name.localeCompare(b.name)));
});

// Lista expressões disponíveis em public/expressions/
app.get('/api/expressions', (req, res) => {
  const dir = path.join(__dirname, 'public/expressions');
  const exts = ['.jpg', '.jpeg', '.png', '.webp'];
  if (!fs.existsSync(dir)) return res.json([]);
  const files = fs.readdirSync(dir)
    .filter(f => exts.includes(path.extname(f).toLowerCase()))
    .map(f => ({ file: f, name: path.basename(f, path.extname(f)) }));
  res.json(files);
});

const modelUpload = multer({ dest: 'uploads/' });

app.post('/api/generate-model', modelUpload.fields([
  { name: 'glasses', maxCount: 1 },
  { name: 'clothing', maxCount: 1 },
]), async (req, res) => {
  const glassesFile  = req.files?.['glasses']?.[0];
  const clothingFile = req.files?.['clothing']?.[0];
  const uploadedPaths = [glassesFile?.path, clothingFile?.path].filter(Boolean);
  try {
    const { modelFile, pose } = req.body;
    if (!glassesFile)  return res.status(400).json({ error: 'Envie a foto dos óculos.' });
    if (!clothingFile) return res.status(400).json({ error: 'Envie a foto da roupa.' });
    if (!modelFile)    return res.status(400).json({ error: 'Selecione um modelo.' });

    const modelPath = path.join(__dirname, 'public/models', modelFile);
    if (!fs.existsSync(modelPath))
      return res.status(400).json({ error: 'Modelo não encontrado.' });

    const { expressionFile } = req.body;
    const ext = path.extname(modelFile).toLowerCase();
    const mime = ext === '.png' ? 'image/png' : 'image/jpeg';

    const modelRef    = await fileToOpenAI(modelPath,            mime,                    'model.jpg');
    const glassesRef  = await fileToOpenAI(glassesFile.path,  glassesFile.mimetype,  'glasses.jpg');
    const clothingRef = await fileToOpenAI(clothingFile.path, clothingFile.mimetype, 'clothing.jpg');

    const images = [modelRef, glassesRef, clothingRef];
    let prompt = pose === 'ladinho' ? PROMPT_MODEL_LADINHO : PROMPT_MODEL_FRENTE;

    if (expressionFile) {
      const exprPath = path.join(__dirname, 'public/expressions', expressionFile);
      if (fs.existsSync(exprPath)) {
        const exprExt = path.extname(expressionFile).toLowerCase();
        const exprMime = exprExt === '.png' ? 'image/png' : 'image/jpeg';
        const exprRef = await fileToOpenAI(exprPath, exprMime, 'expression.jpg');
        images.push(exprRef);
        prompt = PROMPT_MODEL_COM_EXPRESSAO;
        console.log(`[generate-model] model=${modelFile} pose=${pose} expression=${expressionFile}`);
      }
    } else {
      console.log(`[generate-model] model=${modelFile} pose=${pose}`);
    }

    uploadedPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });

    const response = await client.images.edit({
      model: 'gpt-image-2',
      image: images,
      prompt,
      quality: 'medium',
    });

    const b64 = response.data[0].b64_json;
    if (!b64) throw new Error('OpenAI não retornou imagem.');

    // Não aplica correção de fundo — foto com modelo tem muitos detalhes e o offset distorce as cores
    res.json({ image: b64 });
  } catch (err) {
    uploadedPaths.forEach(p => { try { if (fs.existsSync(p)) fs.unlinkSync(p); } catch {} });
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3333, () => console.log('Vyser rodando em http://localhost:3333'));
