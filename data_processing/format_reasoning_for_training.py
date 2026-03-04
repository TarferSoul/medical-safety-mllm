#!/usr/bin/env python3
"""
Format reasoning data into training format with <think> tags.

Input: reasoning_results_*.json (from generate_reasoning.py)
Output: Training data in ShareGPT format with thinking process

Format:
<think>
[reasoning process]
</think>

[final report]
"""

import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config

# Load configuration
config = load_config()

# Get reasoning formatting settings
format_config = config.get_reasoning_format_config()

# Default settings from config
DEFAULT_REASONING_INPUT = format_config.get('reasoning_input', None)
DEFAULT_OUTPUT = format_config.get('output', 'dataset/mimic_cxr_train_w_reasoning.json')
DEFAULT_NORMALIZED_INPUT = format_config.get('normalized_input', None)
DEFAULT_SPLIT_RATIO = format_config.get('split_ratio', 0.0)


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use in filenames.
    Removes paths, special characters, and limits length.
    """
    # Remove path components (e.g., "Qwen/Qwen3-VL-8B" -> "Qwen3-VL-8B")
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    # Replace special characters with underscores
    model_name = re.sub(r'[^\w\-.]', '_', model_name)

    # Remove consecutive underscores
    model_name = re.sub(r'_+', '_', model_name)

    # Limit length
    if len(model_name) > 50:
        model_name = model_name[:50]

    return model_name


def format_with_thinking(reasoning: str, report: str) -> str:
    """
    Format reasoning and report into training output with <think> tags.

    Args:
        reasoning: The thinking/reasoning process
        report: The final medical report

    Returns:
        Formatted string with <think> tags
    """
    # Clean up reasoning - remove any existing think tags
    # Handle case where reasoning already contains <think>...</think>
    if '<think>' in reasoning or '</think>' in reasoning:
        # Remove all think tags from reasoning
        reasoning = reasoning.replace('<think>', '').replace('</think>', '').strip()

    # Clean up report - remove any think tags (shouldn't happen, but just in case)
    if '<think>' in report or '</think>' in report:
        # Take only the part after the last </think> tag
        if '</think>' in report:
            report = report.split('</think>')[-1].strip()
        # Remove any remaining think tags
        report = report.replace('<think>', '').strip()

    # Format with think tags
    formatted = f"<think>\n{reasoning}\n</think>\n\n{report}"

    return formatted


def merge_with_normalized(reasoning_file: str, normalized_file: str = None) -> tuple:
    """
    Merge reasoning data with normalized reports if available.

    Args:
        reasoning_file: Path to reasoning_results JSON file
        normalized_file: Optional path to normalized dataset

    Returns:
        Tuple of (merged items list, reasoning model name)
    """
    print(f"Loading reasoning data from {reasoning_file}")
    with open(reasoning_file, 'r', encoding='utf-8') as f:
        reasoning_data = json.load(f)

    # Extract model name from first successful item
    reasoning_model = "unknown"
    for item in reasoning_data:
        if item.get('success', False) and 'model' in item:
            reasoning_model = item['model']
            break

    # Load normalized data if provided
    normalized_map = {}
    if normalized_file and Path(normalized_file).exists():
        print(f"Loading normalized data from {normalized_file}")
        with open(normalized_file, 'r', encoding='utf-8') as f:
            normalized_data = json.load(f)

        # Create a map of images to normalized reports
        for idx, item in enumerate(normalized_data):
            images_key = tuple(sorted(item['images']))
            normalized_map[images_key] = {
                'report': item['conversations'][1]['value'],
                'normalization': item.get('normalization', {}),
                'index': idx
            }

        print(f"Loaded {len(normalized_map)} normalized reports")

    # Merge data
    merged_items = []
    used_normalized = 0
    used_original = 0
    failed_reasoning = 0

    for item in reasoning_data:
        # Check if reasoning was successful
        if not item.get('success', False) or not item.get('reasoning'):
            failed_reasoning += 1
            continue

        images = item['images']
        reasoning = item['reasoning']

        # Try to get normalized report
        images_key = tuple(sorted(images))
        if images_key in normalized_map:
            norm_data = normalized_map[images_key]
            # Check if normalization was successful
            if norm_data['normalization'].get('success', False):
                final_report = norm_data['report']
                used_normalized += 1
                report_source = 'normalized'
            else:
                final_report = item['ground_truth']
                used_original += 1
                report_source = 'original'
        else:
            final_report = item['ground_truth']
            used_original += 1
            report_source = 'original'

        # Format with thinking tags
        formatted_output = format_with_thinking(reasoning, final_report)

        # Create training item in ShareGPT format
        training_item = {
            'conversations': [
                {
                    'from': 'human',
                    'value': '<image>' * len(images) + 'Generate a medical imaging report based on the X-ray image results.'
                },
                {
                    'from': 'gpt',
                    'value': formatted_output
                }
            ],
            'images': images,
            'metadata': {
                'reasoning_model': item.get('model', 'unknown'),
                'report_source': report_source,
                'sample_id': item.get('sample_id', -1)
            }
        }

        merged_items.append(training_item)

    print(f"\nMerging Statistics:")
    print(f"  Total reasoning samples: {len(reasoning_data)}")
    print(f"  Failed reasoning: {failed_reasoning}")
    print(f"  Successfully merged: {len(merged_items)}")
    print(f"  Using normalized reports: {used_normalized}")
    print(f"  Using original reports: {used_original}")
    print(f"  Reasoning model: {reasoning_model}")

    return merged_items, reasoning_model


def format_reasoning_dataset(reasoning_file: str, output_file: str,
                            normalized_file: str = None,
                            split_ratio: float = 0.1):
    """
    Format reasoning dataset for training.

    Args:
        reasoning_file: Input reasoning results JSON
        output_file: Output training data JSON
        normalized_file: Optional normalized dataset to use instead of original
        split_ratio: Ratio to split into test set (default 0.1)
    """

    # Merge reasoning with reports
    merged_items, reasoning_model = merge_with_normalized(reasoning_file, normalized_file)

    if not merged_items:
        print("Error: No valid items to format!")
        return

    # Add model name to output file
    model_name_clean = "reasoning_" + sanitize_model_name(reasoning_model)
    output_path = Path(output_file)
    output_stem = output_path.stem
    output_parent = output_path.parent
    output_suffix = output_path.suffix

    # Insert model name before file extension
    if model_name_clean and not model_name_clean.endswith("unknown"):
        new_output_file = output_parent / f"{output_stem}_{model_name_clean}{output_suffix}"
    else:
        new_output_file = output_file

    # Split into train and test if needed
    if split_ratio > 0:
        split_idx = int(len(merged_items) * (1 - split_ratio))
        train_items = merged_items[:split_idx]
        test_items = merged_items[split_idx:]

        # Save training set
        train_file = str(new_output_file).replace('.json', '_train.json')
        print(f"\nSaving {len(train_items)} training samples to {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_items, f, indent=2, ensure_ascii=False)

        # Save test set
        test_file = str(new_output_file).replace('.json', '_test.json')
        print(f"Saving {len(test_items)} test samples to {test_file}")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_items, f, indent=2, ensure_ascii=False)

        print(f"\nTrain/Test split: {len(train_items)}/{len(test_items)} ({split_ratio*100:.0f}% test)")
    else:
        # Save all as single file
        print(f"\nSaving {len(merged_items)} samples to {new_output_file}")
        with open(new_output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_items, f, indent=2, ensure_ascii=False)

    # Save metadata summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_reasoning_file": reasoning_file,
        "normalized_file": normalized_file,
        "output_file": str(new_output_file),
        "reasoning_model": reasoning_model,
        "total_samples": len(merged_items),
        "train_samples": len(train_items) if split_ratio > 0 else len(merged_items),
        "test_samples": len(test_items) if split_ratio > 0 else 0,
        "split_ratio": split_ratio
    }

    summary_file = str(new_output_file).replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")

    # Show example
    print(f"\n{'='*80}")
    print("Example formatted output (first sample):")
    print(f"{'='*80}")
    example = merged_items[0]['conversations'][1]['value']
    print(example[:1000] + "..." if len(example) > 1000 else example)
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format reasoning data for training with <think> tags"
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        # required=True,
        default='reasoning_results/reasoning_results_20251229_144040.json',
        help="Input reasoning results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        # required=True,
        default='dataset/mimic_cxr_train_normalized_v2_improved_w_qwen3vl32b_reason.json',
        help="Output training data JSON file"
    )
    parser.add_argument(
        "--normalized",
        type=str,
        default=None,
        help="Optional normalized dataset JSON (will use instead of original reports)"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0,
        help="Test split ratio (default: 0.1, set to 0 for no split)"
    )

    args = parser.parse_args()

    format_reasoning_dataset(
        reasoning_file=args.reasoning,
        output_file=args.output,
        normalized_file=args.normalized,
        split_ratio=args.split
    )
