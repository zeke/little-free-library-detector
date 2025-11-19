#!/usr/bin/env node
/**
 * Little Free Library detector using ONNX Runtime
 * Usage: node detect.js <image_path>
 */

import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import { readFileSync } from 'fs';
import { resolve } from 'path';

// ImageNet normalization values
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

async function preprocessImage(imagePath) {
  // Resize and center crop to 224x224
  const imageBuffer = await sharp(imagePath)
    .resize(256, 256, { fit: 'cover' })
    .extract({ left: 16, top: 16, width: 224, height: 224 })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { data, info } = imageBuffer;
  const { width, height, channels } = info;

  // Convert to float32 and normalize
  const float32Data = new Float32Array(width * height * channels);

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      for (let c = 0; c < channels; c++) {
        const idx = (i * width + j) * channels + c;
        // Normalize: (pixel/255 - mean) / std
        const normalized = (data[idx] / 255.0 - MEAN[c]) / STD[c];
        // NCHW format: [batch, channel, height, width]
        const outputIdx = c * (height * width) + i * width + j;
        float32Data[outputIdx] = normalized;
      }
    }
  }

  return {
    data: float32Data,
    dims: [1, channels, height, width]
  };
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const scores = logits.map(l => Math.exp(l - maxLogit));
  const sumScores = scores.reduce((a, b) => a + b);
  return scores.map(s => s / sumScores);
}

async function detectLibrary(imagePath, modelPath, labelsPath) {
  // Load model
  const session = await ort.InferenceSession.create(modelPath);

  // Load labels
  const labels = readFileSync(labelsPath, 'utf-8')
    .trim()
    .split('\n');

  // Preprocess image
  const { data, dims } = await preprocessImage(imagePath);
  const tensor = new ort.Tensor('float32', data, dims);

  // Run inference
  const feeds = { input: tensor };
  const results = await session.run(feeds);
  const output = results.output.data;

  // Convert logits to probabilities
  const probs = softmax(Array.from(output));

  // Get prediction
  const maxIdx = probs.indexOf(Math.max(...probs));
  const prediction = labels[maxIdx];
  const confidence = probs[maxIdx];

  return {
    prediction,
    confidence,
    probabilities: Object.fromEntries(
      labels.map((label, i) => [label, probs[i]])
    )
  };
}

async function main() {
  const imagePath = process.argv[2];

  if (!imagePath) {
    console.error('Usage: node detect.js <image_path>');
    process.exit(1);
  }

  const modelPath = resolve('../models/library_detector.onnx');
  const labelsPath = resolve('../models/library_detector_labels.txt');

  console.log('Loading model...');
  const result = await detectLibrary(imagePath, modelPath, labelsPath);

  console.log('\n=== Prediction ===');
  console.log(`Class: ${result.prediction}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  console.log('\nAll probabilities:');
  for (const [label, prob] of Object.entries(result.probabilities)) {
    console.log(`  ${label}: ${(prob * 100).toFixed(2)}%`);
  }
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  main().catch(console.error);
}

export { detectLibrary };
