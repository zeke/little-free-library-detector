#!/usr/bin/env python3
"""
Helper script to organize raw images into train/val/test splits.

Usage:
  python organize_data.py --source raw_library_photos/ --dest ../data --class library
  python organize_data.py --source raw_other_photos/ --dest ../data --class not_library
"""

import argparse
import shutil
from pathlib import Path
import random

def split_data(source_dir, dest_dir, class_name, train=0.7, val=0.15, test=0.15):
    """Split images into train/val/test sets."""

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return

    # Get all image files
    images = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        images.extend(source_path.glob(f'*.{ext}'))

    if not images:
        print(f"No images found in {source_dir}")
        return

    random.shuffle(images)

    # Calculate splits
    n = len(images)
    train_n = int(n * train)
    val_n = int(n * val)

    splits = {
        'train': images[:train_n],
        'val': images[train_n:train_n + val_n],
        'test': images[train_n + val_n:]
    }

    # Copy files
    dest_base = Path(dest_dir)
    for split, files in splits.items():
        dest = dest_base / split / class_name
        dest.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, dest / f.name)
            print(f"  Copied {f.name} → {split}/{class_name}/")

    print(f"\nSplit {n} images into class '{class_name}':")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")

def main():
    parser = argparse.ArgumentParser(description='Organize images into train/val/test splits')
    parser.add_argument('--source', required=True, help='Source directory with raw images')
    parser.add_argument('--dest', default='../data', help='Destination data directory')
    parser.add_argument('--class', dest='class_name', required=True,
                       help='Class name (e.g., "library" or "not_library")')
    parser.add_argument('--train', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test', type=float, default=0.15, help='Test split ratio')
    args = parser.parse_args()

    # Validate ratios
    if abs(args.train + args.val + args.test - 1.0) > 0.01:
        print("Error: train + val + test must equal 1.0")
        return

    split_data(args.source, args.dest, args.class_name, args.train, args.val, args.test)

if __name__ == '__main__':
    main()
