"""
Filter a LlamaFactory-style dataset JSON for corrupted images and image count constraints.
Usage:
    python filter.py --dataset /path/to/medical_xray_train.json --workers 16
    python filter.py --dataset /path/to/medical_xray_train.json --workers 16 --max_images 5  # filter by image count
    python filter.py --dataset /path/to/medical_xray_train.json --workers 16 --fix  # auto-remove bad entries
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
    parser = argparse.ArgumentParser(description="Filter dataset images for corruption and image count")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum images per sample (default: no limit)")
    parser.add_argument("--fix", action="store_true", help="Output a cleaned JSON without bad entries")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries: {len(data)}")

    # Filter by image count first
    if args.max_images is not None:
        too_many_images_indices = set()
        for i, item in enumerate(data):
            images = item.get("images", [])
            if isinstance(images, str):
                images = [images]
            if len(images) > args.max_images:
                too_many_images_indices.add(i)

        if too_many_images_indices:
            print(f"\nEntries with >{args.max_images} images: {len(too_many_images_indices)}")
            # Save list of filtered entries
            filtered_list_path = args.dataset.replace(".json", f"_too_many_images.json")
            with open(filtered_list_path, "w", encoding="utf-8") as f:
                json.dump([{"entry_index": i, "image_count": len(data[i].get("images", []))}
                           for i in sorted(too_many_images_indices)], f, indent=2, ensure_ascii=False)
            print(f"Filtered list saved to: {filtered_list_path}")
    else:
        too_many_images_indices = set()

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
    if args.max_images is not None:
        print(f"Entries with >{args.max_images} images: {len(too_many_images_indices)}")
    print(f"{'='*60}")

    bad_indices = set()
    if bad_results:
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

    # Combine all filtered indices
    all_filtered_indices = bad_indices | too_many_images_indices

    if all_filtered_indices and args.fix:
        cleaned = [item for i, item in enumerate(data) if i not in all_filtered_indices]
        clean_path = args.dataset.replace(".json", "_clean.json")
        with open(clean_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        print(f"\nCleaned dataset saved to: {clean_path}")
        print(f"  Original entries: {len(data)}")
        print(f"  Removed (bad images): {len(bad_indices)}")
        if args.max_images is not None:
            print(f"  Removed (too many images): {len(too_many_images_indices)}")
        print(f"  Remaining entries: {len(cleaned)}")
    elif not bad_results and not too_many_images_indices:
        print("\nAll images are valid!")


if __name__ == "__main__":
    main()
