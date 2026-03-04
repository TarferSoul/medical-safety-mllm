"""
Scan a LlamaFactory-style dataset JSON for corrupted images.
Usage:
    python check_images.py --dataset /path/to/medical_xray_train.json --workers 16
    python check_images.py --dataset /path/to/medical_xray_train.json --workers 16 --fix  # auto-remove bad entries
"""

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageFile
from tqdm import tqdm

# Don't let truncated images silently pass
ImageFile.LOAD_TRUNCATED_IMAGES = False


def check_single_image(args):
    idx, img_path = args
    try:
        img = Image.open(img_path)
        img.load()       # force full pixel decode (this is where your training crashes)
        # Also test resize since that's the exact operation in mm_plugin.py
        img.resize((224, 224))
        img.close()
        return None
    except FileNotFoundError:
        return (idx, img_path, "FILE_NOT_FOUND")
    except Exception as e:
        return (idx, img_path, str(e))


def extract_image_paths(data, dataset_dir=None):
    """Extract (entry_index, image_path) pairs from various dataset formats."""
    tasks = []
    for i, item in enumerate(data):
        # Try common LlamaFactory formats
        images = item.get("images") or []

        # ShareGPT format: images might be a list of paths or a single path
        if isinstance(images, str):
            images = [images]

        # Also check inside messages for <image> tags with separate image field
        if not images:
            # Some formats use "image" (singular)
            img = item.get("image")
            if img:
                images = [img] if isinstance(img, str) else img

        for img_path in images:
            # Handle relative paths
            if dataset_dir and not os.path.isabs(img_path):
                img_path = os.path.join(dataset_dir, img_path)
            tasks.append((i, img_path))

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Check dataset images for corruption")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--fix", action="store_true", help="Output a cleaned JSON without bad entries")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries: {len(data)}")

    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))
    tasks = extract_image_paths(data, dataset_dir)
    print(f"Total images to check: {len(tasks)}")

    if not tasks:
        print("No images found! Check your dataset format (expected 'images' or 'image' field).")
        return

    # Parallel check
    bad_results = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(check_single_image, tasks, chunksize=64),
                           total=len(tasks), desc="Checking images"):
            if result is not None:
                bad_results.append(result)

    # Report
    print(f"\n{'='*60}")
    print(f"Total images checked: {len(tasks)}")
    print(f"Bad images found:     {len(bad_results)}")
    print(f"{'='*60}")

    if bad_results:
        bad_indices = set()
        for idx, path, err in sorted(bad_results):
            print(f"  [entry {idx}] {path}")
            print(f"    Error: {err}")
            bad_indices.add(idx)

        # Save bad image list
        bad_list_path = args.dataset.replace(".json", "_bad_images.json")
        with open(bad_list_path, "w", encoding="utf-8") as f:
            json.dump([{"entry_index": idx, "path": path, "error": err}
                       for idx, path, err in bad_results], f, indent=2, ensure_ascii=False)
        print(f"\nBad image list saved to: {bad_list_path}")

        # Optionally output cleaned dataset
        if args.fix:
            cleaned = [item for i, item in enumerate(data) if i not in bad_indices]
            clean_path = args.dataset.replace(".json", "_clean.json")
            with open(clean_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            print(f"Cleaned dataset saved to: {clean_path} ({len(cleaned)} entries, removed {len(bad_indices)})")
    else:
        print("All images are valid!")


if __name__ == "__main__":
    main()