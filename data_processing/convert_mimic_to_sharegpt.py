#!/usr/bin/env python3
"""
Convert MIMIC-CXR-JPG dataset to LlamaFactory ShareGPT format.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config


def read_report(report_path: str) -> str:
    """Read and return the content of a medical report."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_study_images(study_dir: Path) -> List[str]:
    """Get all JPG images in a study directory, sorted by name."""
    images = sorted(study_dir.glob('*.jpg'))
    return [str(img.absolute()) for img in images]


def process_dataset(
    data_dir: str,
    report_dir: str,
    output_file: str,
    instruction: str = "Generate a medical imaging report based on the X-ray image results.",
    max_samples: int = None,
    test_samples: int = 500,
    test_split: float = None,
    random_seed: int = 42
):
    """
    Process MIMIC-CXR-JPG dataset and convert to ShareGPT format.

    Args:
        data_dir: Path to image data directory (e.g., .../gdrive/p10)
        report_dir: Path to report directory (e.g., .../gdrive/p10_report/p10)
        output_file: Output JSON file path
        instruction: User instruction for the task
        max_samples: Maximum number of samples to process (None for all)
        test_samples: Number of samples for test set (default: 500). If None, use test_split.
        test_split: Fraction of data to use as test set (only used if test_samples is None)
        random_seed: Random seed for reproducible splits (default: 42)
    """

    data_path = Path(data_dir)
    report_path = Path(report_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not report_path.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    sharegpt_data = []

    # Get all patient directories
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"Found {len(patient_dirs)} patient directories")

    sample_count = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name

        # Get all study directories for this patient
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

        for study_dir in study_dirs:
            study_id = study_dir.name

            # Get corresponding report file
            report_file = report_path / patient_id / f"{study_id}.txt"

            if not report_file.exists():
                print(f"Warning: Report not found for {patient_id}/{study_id}")
                continue

            # Get all images for this study
            images = get_study_images(study_dir)

            if len(images) == 0:
                print(f"Warning: No images found for {patient_id}/{study_id}")
                continue

            # Read the medical report
            try:
                report_content = read_report(str(report_file))
            except Exception as e:
                print(f"Error reading report {report_file}: {e}")
                continue

            # Create <image> tags based on number of images
            image_tags = "".join(["<image>"] * len(images))

            # Create ShareGPT format entry
            entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{image_tags}{instruction}"
                    },
                    {
                        "from": "gpt",
                        "value": report_content
                    }
                ],
                "images": images
            }

            sharegpt_data.append(entry)
            sample_count += 1

            # Check if we've reached the max_samples limit
            if max_samples is not None and sample_count >= max_samples:
                break

        if max_samples is not None and sample_count >= max_samples:
            break

    # Split data into train and test sets using fixed random seed
    print(f"\nTotal samples collected: {len(sharegpt_data)}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle data
    random.shuffle(sharegpt_data)

    # Determine test set size
    if test_samples is not None:
        # Use fixed number of test samples
        test_size = min(test_samples, len(sharegpt_data))
        print(f"Using fixed test set size: {test_size} samples")
    elif test_split is not None:
        # Use percentage split
        test_size = int(len(sharegpt_data) * test_split)
        print(f"Using test split ratio {test_split}: {test_size} samples")
    else:
        # Default: use 500 samples or 10% if total is less than 5000
        test_size = min(500, int(len(sharegpt_data) * 0.1))
        print(f"Using default test set size: {test_size} samples")

    train_size = len(sharegpt_data) - test_size

    train_data = sharegpt_data[:train_size]
    test_data = sharegpt_data[train_size:]

    # Generate output filenames
    output_path = Path(output_file)
    base_name = output_path.stem
    extension = output_path.suffix

    train_file = f"{base_name}_train{extension}"
    test_file = f"{base_name}_test{extension}"

    # Save train set
    print(f"\nSaving {len(train_data)} training samples to {train_file}")
    with open("./dataset/" + train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Save test set
    print(f"Saving {len(test_data)} test samples to {test_file}")
    with open("./dataset/" + test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\nConversion complete!")
    print(f"  Train set: ./dataset/{train_file}")
    print(f"  Test set: ./dataset/{test_file}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(sharegpt_data)}")
    print(f"  Train samples: {len(train_data)} ({len(train_data)/len(sharegpt_data)*100:.1f}%)")
    print(f"  Test samples: {len(test_data)} ({len(test_data)/len(sharegpt_data)*100:.1f}%)")

    total_images = sum(len(entry['images']) for entry in sharegpt_data)
    print(f"  Total images: {total_images}")
    print(f"  Average images per sample: {total_images / len(sharegpt_data):.2f}")


if __name__ == "__main__":
    import argparse

    # Load configuration
    config = load_config()

    # Get paths from config
    paths = config.get_data_paths()
    data_config = config.get_data_processing_config()

    parser = argparse.ArgumentParser(description="Convert MIMIC-CXR-JPG to ShareGPT format")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=paths.get("mimic_images_dir", "/mnt/shared-storage-user/ai4good1-share/xieyuejin/gdown/gdrive/p10_scale033"),
        help="Path to image data directory (default: from config.yaml)"
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=paths.get("mimic_reports_dir", "/mnt/shared-storage-user/ai4good1-share/xieyuejin/gdown/gdrive/p10_report/p10"),
        help="Path to report directory (default: from config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mimic_cxr_scale033.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=data_config.get("default_instruction", "Generate a medical imaging report based on the X-ray image results."),
        help="User instruction for the task (default: from config.yaml)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=data_config.get("max_samples", None),
        help="Maximum number of samples to process (default: from config.yaml)"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=data_config.get("test_samples", 500),
        help="Number of samples for test set (default: from config.yaml)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=data_config.get("test_split", None),
        help="Fraction of data to use as test set (default: from config.yaml)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=data_config.get("random_seed", 42),
        help="Random seed for reproducible train/test splits (default: from config.yaml)"
    )

    args = parser.parse_args()

    # Handle test_samples argument
    test_samples = args.test_samples
    if args.test_samples == 0:  # Allow --test_samples 0 to trigger test_split usage
        test_samples = None

    print("Configuration loaded:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Report directory: {args.report_dir}")
    print(f"  Output file: {args.output}")
    print(f"  Test samples: {test_samples}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Instruction length: {len(args.instruction)} characters")
    print()

    process_dataset(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        output_file=args.output,
        instruction=args.instruction,
        max_samples=args.max_samples,
        test_samples=test_samples,
        test_split=args.test_split,
        random_seed=args.random_seed
    )
