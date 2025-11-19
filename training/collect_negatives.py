#!/usr/bin/env python3
"""
Collect negative example images using SerpAPI (Google Images).

Usage:
  python collect_negatives.py --api-key YOUR_API_KEY --count 100

Get a free API key at: https://serpapi.com/
"""

import argparse
import requests
from pathlib import Path
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
import hashlib

# Search queries for good negative examples
# Ordered by relevance (most similar to libraries first)
NEGATIVE_QUERIES = [
    "mailbox front yard",
    "residential mailbox",
    "metal mailbox post",
    "garden birdhouse",
    "wooden birdhouse yard",
    "newspaper box street",
    "utility box residential",
    "decorative yard sign",
    "front porch residential",
    "residential fence gate",
]

def download_image(url, output_path, timeout=10):
    """Download image from URL and save to path."""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        response.raise_for_status()

        # Open and verify it's a valid image
        img = Image.open(BytesIO(response.content))

        # Convert RGBA to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background

        # Save as JPEG
        img.save(output_path, 'JPEG', quality=95)
        return True

    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False

def get_image_hash(image_path):
    """Get hash of image to detect duplicates."""
    img = Image.open(image_path)
    return hashlib.md5(img.tobytes()).hexdigest()

def search_images(query, api_key, num_images=10):
    """Search for images using SerpAPI."""
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "num": min(num_images, 100)  # SerpAPI max per request
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "error" in results:
        raise Exception(f"SerpAPI error: {results['error']}")

    images = results.get("images_results", [])
    return [img.get("original") for img in images if img.get("original")]

def collect_negatives(api_key, total_count, output_dir):
    """Collect negative example images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track seen image hashes to avoid duplicates
    seen_hashes = set()
    if output_path.exists():
        for img_file in output_path.glob("*.jpg"):
            try:
                seen_hashes.add(get_image_hash(img_file))
            except:
                pass

    images_per_query = total_count // len(NEGATIVE_QUERIES) + 1
    collected = 0

    print(f"Collecting {total_count} negative examples...")
    print(f"Output directory: {output_path}")
    print()

    for query_idx, query in enumerate(NEGATIVE_QUERIES):
        if collected >= total_count:
            break

        print(f"Searching: {query}")

        try:
            image_urls = search_images(query, api_key, images_per_query)
            print(f"  Found {len(image_urls)} results")

            for url in tqdm(image_urls, desc=f"  Downloading", leave=False):
                if collected >= total_count:
                    break

                # Create filename
                filename = f"neg_{query_idx:02d}_{collected:04d}.jpg"
                output_file = output_path / filename

                # Download image
                if download_image(url, output_file):
                    # Check for duplicates
                    try:
                        img_hash = get_image_hash(output_file)
                        if img_hash in seen_hashes:
                            output_file.unlink()
                            continue
                        seen_hashes.add(img_hash)
                    except:
                        output_file.unlink()
                        continue

                    collected += 1

                # Rate limiting
                time.sleep(0.5)

            print(f"  Collected: {collected}/{total_count}")
            print()

            # Small delay between queries
            time.sleep(1)

        except Exception as e:
            print(f"  Error searching '{query}': {e}")
            continue

    print(f"\n✓ Collected {collected} negative examples")
    print(f"  Saved to: {output_path}")

    return collected

def main():
    parser = argparse.ArgumentParser(
        description='Collect negative example images using SerpAPI',
        epilog='Get a free API key at https://serpapi.com/'
    )
    parser.add_argument('--api-key', required=True,
                       help='SerpAPI API key')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of images to collect (default: 100)')
    parser.add_argument('--output', default='../training-data/negatives',
                       help='Output directory (default: ../training-data/negatives)')
    args = parser.parse_args()

    try:
        collected = collect_negatives(args.api_key, args.count, args.output)

        if collected < args.count:
            print(f"\nNote: Only collected {collected}/{args.count} images")
            print("This might be due to:")
            print("  - API rate limits")
            print("  - Duplicate images filtered out")
            print("  - Download failures")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
