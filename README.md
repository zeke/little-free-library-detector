# Little Free Library Detector

Machine learning model that detects Little Free Libraries in images. Built with PyTorch, deployed as ONNX for browser and Node.js.

Current accuracy: 98%

## Usage

Add training images to these folders:
- `training-data/positives/` - images with little free libraries
- `training-data/negatives/` - images without libraries (mailboxes, birdhouses, etc.)

Run the build pipeline:

```bash
script/build
```

This downsizes images, splits data, trains the model, and exports to ONNX.

Test the model:

```bash
script/test path/to/image.jpg
```

Or test in browser:

```bash
script/serve
# Visit http://localhost:8080/inference/browser.html
```

## Scripts

- `script/build` - Complete pipeline: downsize → split → train → export
- `script/collect-negatives [count]` - Auto-download negative examples via SerpAPI
- `script/downsize` - Resize training images to max 1024x1024
- `script/split-data` - Split raw images into train/val/test
- `script/train` - Train the model (10 epochs default)
- `script/export` - Export to ONNX
- `script/test <image>` - Test on a single image
- `script/serve` - Start web server for browser demo

## Training Data

Recommended: 100+ images per class (200+ total)

Good negative examples:
- Mailboxes (most similar to libraries)
- Birdhouses
- Newspaper boxes
- Residential exteriors
- Street scenes

Hard negatives (things easily confused with libraries) work better than easy negatives (random scenes).

## Auto-collecting Negatives

You can automatically download negative examples from Google Images using SerpAPI:

```bash
# Get a free API key at https://serpapi.com/
echo 'SERPAPI_API_KEY=your-api-key-here' > .env

# Collect 100 negative examples
script/collect-negatives 100
```

This searches for mailboxes, birdhouses, yard scenes, and similar images. Review the downloaded images in `training-data/negatives/` and remove any that don't look good before training.

## Deployment

The ONNX model works in:
- Browser (client-side, no server needed)
- Node.js/Deno
- Python
- Edge devices (Raspberry Pi, mobile)

## License

MIT
