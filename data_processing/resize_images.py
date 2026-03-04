#!/usr/bin/env python3
"""
Resize all images in a directory tree proportionally while maintaining folder structure.

Usage:
    python resize_images.py --scale 0.5
    python resize_images.py --scale 0.25 --quality 95
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def get_all_image_files(root_dir):
    """Recursively find all image files in directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    image_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def resize_image(input_path, output_path, scale, quality=95):
    """
    Resize a single image by scale factor.

    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        scale: Scale factor (e.g., 0.5 for 50% size)
        quality: JPEG quality (1-100)
    """
    try:
        # Open image
        img = Image.open(input_path)

        # Calculate new size
        original_size = img.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        # Resize with high-quality resampling
        resized_img = img.resize(new_size, Image.LANCZOS)

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save with appropriate format
        if img.format == 'JPEG' or output_path.lower().endswith(('.jpg', '.jpeg')):
            resized_img.save(output_path, 'JPEG', quality=quality, optimize=True)
        else:
            resized_img.save(output_path, img.format)

        return True, original_size, new_size

    except Exception as e:
        return False, None, None, str(e)


def resize_directory(input_dir, output_dir, scale, quality=95):
    """
    Resize all images in directory tree.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        scale: Scale factor (e.g., 0.5 for 50% size)
        quality: JPEG quality (1-100)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Get all image files
    print(f"Scanning for images in {input_dir}...")
    image_files = get_all_image_files(input_dir)
    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        print("No images found!")
        return

    # Process images
    success_count = 0
    fail_count = 0
    failed_files = []

    print(f"\nResizing images with scale={scale}, quality={quality}...")

    for input_file in tqdm(image_files, desc="Processing"):
        # Calculate relative path
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = output_path / rel_path

        # Resize image
        result = resize_image(input_file, str(output_file), scale, quality)

        if result[0]:
            success_count += 1
        else:
            fail_count += 1
            failed_files.append((input_file, result[3] if len(result) > 3 else "Unknown error"))

    # Print summary
    print("\n" + "=" * 60)
    print("RESIZE SUMMARY")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor:     {scale} ({scale * 100:.0f}%)")
    print(f"JPEG quality:     {quality}")
    print(f"\nTotal images:     {len(image_files)}")
    print(f"Successfully resized: {success_count}")
    print(f"Failed:           {fail_count}")

    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files[:10]:  # Show first 10
            print(f"  {file_path}")
            print(f"    Error: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize all images in a directory tree proportionally"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/shared-storage-user/ai4good1-share/xieyuejin/gdown/gdrive/p10",
        help="Input directory containing images (default: /mnt/shared-storage-user/ai4good1-share/xieyuejin/gdown/gdrive/p10)"
    )

    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Scale factor (e.g., 0.5 for 50%% size, 0.25 for 25%% size)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100, default: 95)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: {input}_scale{scale})"
    )

    args = parser.parse_args()

    # Validate scale
    if args.scale <= 0 or args.scale > 1:
        parser.error("Scale must be between 0 and 1 (e.g., 0.5 for 50% size)")

    # Validate quality
    if args.quality < 1 or args.quality > 100:
        parser.error("Quality must be between 1 and 100")

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Generate output directory name with scale suffix
        input_path = Path(args.input)
        scale_str = f"{args.scale:.2f}".replace(".", "")
        output_dir = f"{args.input}_scale{scale_str}"

    print("Configuration:")
    print(f"  Input directory:  {args.input}")
    print(f"  Output directory: {output_dir}")
    print(f"  Scale factor:     {args.scale} ({args.scale * 100:.0f}%)")
    print(f"  JPEG quality:     {args.quality}")
    print()

    # Confirm if output directory exists
    if os.path.exists(output_dir):
        response = input(f"Output directory {output_dir} already exists. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)

    # Run resizing
    resize_directory(args.input, output_dir, args.scale, args.quality)
