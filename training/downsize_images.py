#!/usr/bin/env python3
"""
Downsize training images to reduce file size while maintaining quality.
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def downsize_image(image_path, max_size=1024, quality=85):
    """
    Downsize an image if it's larger than max_size.

    Args:
        image_path: Path to the image
        max_size: Maximum dimension (width or height)
        quality: JPEG quality (1-100)
    """
    try:
        img = Image.open(image_path)

        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Check if resizing is needed
        width, height = img.size
        if width <= max_size and height <= max_size:
            return False  # No resize needed

        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize with high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save with specified quality
        if image_path.suffix.lower() in ['.jpg', '.jpeg']:
            img.save(image_path, 'JPEG', quality=quality, optimize=True)
        elif image_path.suffix.lower() == '.png':
            img.save(image_path, 'PNG', optimize=True)

        return True  # Image was resized

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Downsize training images')
    parser.add_argument('--source', type=str, default='../training-data',
                       help='Source directory containing positives/negatives')
    parser.add_argument('--max-size', type=int, default=1024,
                       help='Maximum width or height in pixels')
    parser.add_argument('--quality', type=int, default=85,
                       help='JPEG quality (1-100)')
    args = parser.parse_args()

    source_dir = Path(args.source)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(source_dir.glob(f'**/{ext}'))

    print(f"Found {len(all_images)} images")
    print(f"Downsizing images larger than {args.max_size}x{args.max_size}...")

    resized_count = 0
    for image_path in tqdm(all_images):
        if downsize_image(image_path, args.max_size, args.quality):
            resized_count += 1

    print(f"\nResized {resized_count} images")
    print(f"Skipped {len(all_images) - resized_count} images (already small enough)")

if __name__ == '__main__':
    main()
