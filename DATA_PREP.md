# Data Preparation Guide

## Overview

You need two types of images:
1. **Positives (library images):** Photos containing little free libraries
2. **Negatives (not-library images):** Similar outdoor scenes without libraries

Put them in:
- `training-data/positives/` - Your little free library photos
- `training-data/negatives/` - Similar scenes without libraries

## Minimum Requirements

- **100+ images per class** (200+ total) for good results
- **Equal balance** between positives and negatives
- **Good diversity** in lighting, angles, weather, library styles

## Collecting Library Images

### From Your Own Photos
If you have many little free library photos, that's perfect! Use them as-is.

### Finding More Images
1. **Google Images:** Search "little free library"
   - Download manually or use tools like `googleimagesdownload`
   - Check licenses if publishing the model

2. **Flickr:** Creative Commons licensed images
   - Search: little free library
   - Filter by license type

3. **Take your own:** Walk around and photograph local libraries
   - Best option for unique training data
   - Ensures you have rights to use images

## Collecting Negative Images

Negatives are **critical** - the model needs to learn what libraries *aren't*.

Use **similar contexts** but without libraries:

### Best Negatives (Hard Examples - Most Valuable):
1. **Mailboxes** - Most similar in shape/size/location
2. **Birdhouses** - Small yard structures
3. **Newspaper boxes** - Street furniture
4. **Utility boxes** - Similar rectangular shapes
5. **Decorative yard signs** - Similar outdoor placement

### Good Negatives:
6. **Residential exteriors** - Houses, porches, yards
7. **Street scenes** - Sidewalks, poles, curbs
8. **Park structures** - Benches, signs
9. **Fences and gates**

### Where to Find:
1. **Your own photos:** Walk around and photograph yards, streets
2. **Google Images:** Search:
   - "mailbox"
   - "birdhouse"
   - "newspaper box"
   - "residential exterior"
   - "front yard"
3. **While collecting positives:** Take photos of nearby non-libraries

**Key insight:** Hard negatives (objects that could be confused with libraries like mailboxes) are more valuable than easy negatives (random indoor scenes).

## Organizing Images

### Automatic (Recommended):

Just put your images in the right folders and run:

```bash
script/organize-data
```

This will automatically split them 70/15/15 into train/val/test.

### Manual Split:

If you prefer to organize manually:
- **70%** training → `data/train/{library,not_library}`
- **15%** validation → `data/val/{library,not_library}`
- **15%** test → `data/test/{library,not_library}`

## Image Requirements

### Format
- **JPG or PNG**
- Any resolution (will be resized to 224x224)
- RGB color (no grayscale)

### Quality
- **Clear focus:** Not blurry
- **Good lighting:** Not too dark
- **Visible subject:** Library clearly visible (for positive examples)

### Diversity
- Different times of day
- Various weather conditions
- Multiple angles (front, side, close, far)
- Different library styles/colors

## Data Augmentation

The training script automatically applies:
- Random horizontal flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random crops

So you don't need to manually create variations.

## Quick Check

Before training, verify your data:

```bash
# Count images per class
find data/train/library -type f | wc -l
find data/train/not_library -type f | wc -l

# Should be roughly equal numbers
```

## Common Issues

**Imbalanced classes:**
- Model will bias toward majority class
- Solution: Balance the dataset or use class weighting

**Too similar images:**
- Model may memorize instead of learning features
- Solution: More diversity in angles, lighting, subjects

**Wrong labels:**
- One mislabeled image in 100 is OK
- Systematic errors hurt performance
- Solution: Spot-check random samples

## Quick Start Workflow

1. **Collect images:**
   - Put library photos in `training-data/positives/`
   - Auto-collect negatives: `script/collect-negatives 100`

2. **Organize:**
   ```bash
   script/organize-data
   ```

3. **Train:**
   ```bash
   script/train
   ```

4. **Export and test:**
   ```bash
   script/export
   script/test path/to/image.jpg
   ```

## Minimum Viable Dataset

To test the pipeline quickly:
- **50 positives** (library photos)
- **50 negatives** (mailboxes, birdhouses, etc.)

This will give you ~70-80% accuracy - good enough to see if it works.

For production use, aim for 100-200+ images per class.
