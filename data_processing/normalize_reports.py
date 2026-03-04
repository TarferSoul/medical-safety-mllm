#!/usr/bin/env python3
"""
Normalize medical reports to a standard format using LLM.
Ensures all reports have consistent structure with all required sections.
"""

import sys
import json
import argparse
import threading
import time
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, create_openai_client

# Load configuration
config = load_config()

# Get API config - using normalize API
normalize_api_config = config.get_normalize_api_config()

# Get normalization settings
norm_config = config.get_normalization_config()

# Initialize client using unified interface
client = create_openai_client(normalize_api_config)

# Default model (can be overridden by command line arguments)
NORMALIZE_MODEL = normalize_api_config['model']

# Default settings from config
DEFAULT_CONCURRENCY = norm_config.get('concurrency_v1', 20)
DEFAULT_MAX_RETRIES = norm_config.get('max_retries', 5)
DEFAULT_MAX_SAMPLES = norm_config.get('max_samples', None)

# Normalization prompt template
NORMALIZE_PROMPT = """You are a medical report formatting assistant specializing in radiology reports.

Your task is to standardize a chest X-ray radiology report into a consistent format. You MUST output a report with the following structure:

FINAL REPORT
 EXAMINATION: [examination type and technique]

 INDICATION: [clinical indication for the study]

 TECHNIQUE: [imaging technique used]

 COMPARISON: [comparison to prior studies]

 FINDINGS:

 [Detailed description of radiographic findings]

 IMPRESSION:

 [Diagnostic conclusion and key findings summary]

**IMPORTANT RULES:**

1. **Preserve Medical Content**: Do NOT change, omit, or add any medical findings or diagnoses. Keep all clinical information exactly as described in the original report.

2. **Fill Missing Sections**:
   - If a section is missing in the original report, write "Not specified." or "None." for that section
   - For FINDINGS: If the original report only has IMPRESSION, extract detailed findings from the IMPRESSION and place them in FINDINGS, then keep the summary in IMPRESSION
   - For IMPRESSION: If only FINDINGS exists, create a concise summary for IMPRESSION based on FINDINGS

3. **Handle Placeholders**: Replace underscores (___) with appropriate generic placeholders:
   - Age/Gender: [AGE/GENDER] (e.g., "___F" -> "[AGE]F" or just "[AGE/GENDER]")
   - Dates: [DATE]
   - Study IDs: [PRIOR_STUDY]
   - Names: [PATIENT] or [PHYSICIAN]

4. **Standardize Section Headers**: Use exactly the format shown above with consistent capitalization and spacing.

5. **Remove WET READ**: If the report contains "WET READ", incorporate relevant information into the FINAL REPORT sections but remove the WET READ label.

6. **Format Consistency**:
   - Use consistent indentation (2 spaces after section headers)
   - Maintain paragraph structure
   - Keep proper spacing between sections

**Original Report:**
{original_report}

**Instructions:** Output ONLY the normalized report. Do not include any explanations, comments, or additional text outside the report structure.
"""


def normalize_single_report(item: dict, model: str, max_retries: int = 5) -> dict:
    """Normalize a single report using LLM."""
    sample_id = item.get('sample_id', 'unknown')
    original_report = item['conversations'][1]['value']

    # Format the prompt
    prompt = NORMALIZE_PROMPT.format(original_report=original_report)

    last_error = None

    # Retry loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a medical report formatting assistant. Output only the normalized report without any additional commentary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent formatting
                max_tokens=4096
            )

            # Get the normalized report
            normalized_report = response.choices[0].message.content.strip()

            # Handle thinking tags if present
            if "</think>" in normalized_report:
                normalized_report = normalized_report.split("</think>")[-1].strip()

            # Check if response is valid
            if not normalized_report or len(normalized_report) < 50:
                if attempt < max_retries - 1:
                    tqdm.write(f"⚠️  Sample {sample_id}: Got short response (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    tqdm.write(f"⚠️  Sample {sample_id}: Got short response after {max_retries} attempts, keeping original")
                    normalized_report = original_report

            # Verify key sections are present
            required_sections = ["FINAL REPORT", "EXAMINATION:", "INDICATION:", "FINDINGS:", "IMPRESSION:"]
            missing_sections = [s for s in required_sections if s not in normalized_report]

            if missing_sections and attempt < max_retries - 1:
                tqdm.write(f"⚠️  Sample {sample_id}: Missing sections {missing_sections} (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)
                continue

            # Create result item
            result_item = item.copy()
            result_item['conversations'][1]['value'] = normalized_report
            result_item['normalization'] = {
                'original_report': original_report,
                'normalized': True,
                'model': model,
                'attempts': attempt + 1,
                'timestamp': datetime.now().isoformat()
            }

            return result_item

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                tqdm.write(f"❌ Sample {sample_id}: {type(e).__name__} (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                tqdm.write(f"❌ Sample {sample_id}: Failed after {max_retries} attempts - {e}")
                # Keep original report on failure
                result_item = item.copy()
                result_item['normalization'] = {
                    'original_report': original_report,
                    'normalized': False,
                    'error': str(e),
                    'model': model,
                    'attempts': max_retries
                }
                return result_item

    # Fallback: keep original
    result_item = item.copy()
    result_item['normalization'] = {
        'original_report': original_report,
        'normalized': False,
        'error': f"Max retries reached: {str(last_error)}",
        'model': model,
        'attempts': max_retries
    }
    return result_item


def normalize_dataset(input_file: str, output_file: str, model: str,
                      concurrency: int = 20, max_retries: int = 5, max_samples: int = None):
    """Normalize all reports in the dataset with concurrent processing."""

    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Limit samples if specified
    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Processing first {max_samples} samples (testing mode)")

    # Add sample_id to each item for tracking
    for idx, item in enumerate(dataset):
        item['sample_id'] = idx

    print(f"\n{'='*80}")
    print(f"NORMALIZATION CONFIGURATION")
    print(f"{'='*80}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {model}")
    print(f"Total samples: {len(dataset)}")
    print(f"Concurrency: {concurrency}")
    print(f"Max retries: {max_retries}")
    print(f"{'='*80}\n")

    # Prepare results storage with thread safety
    results = [None] * len(dataset)
    results_lock = threading.Lock()
    completed_count = 0
    success_count = 0
    failed_count = 0

    # Process samples concurrently
    print("Normalizing reports...\n")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(normalize_single_report, item, model, max_retries): idx
            for idx, item in enumerate(dataset)
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(dataset), desc="Normalizing") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()

                    with results_lock:
                        results[idx] = result
                        completed_count += 1

                        # Track success/failure
                        if result.get('normalization', {}).get('normalized', False):
                            success_count += 1
                        else:
                            failed_count += 1

                        # Save intermediate results every 50 samples
                        if completed_count % 50 == 0:
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x.get('sample_id', 0))

                            # Remove sample_id before saving
                            for r in valid_results:
                                r.pop('sample_id', None)

                            intermediate_file = output_file.replace('.json', '_intermediate.json')
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)

                            tqdm.write(f"✓ Completed {completed_count}/{len(dataset)} samples (Success: {success_count}, Failed: {failed_count})")

                except Exception as e:
                    with results_lock:
                        # Keep original on exception
                        result_item = dataset[idx].copy()
                        result_item['normalization'] = {
                            'original_report': dataset[idx]['conversations'][1]['value'],
                            'normalized': False,
                            'error': f"Future exception: {str(e)}",
                            'model': model
                        }
                        results[idx] = result_item
                        completed_count += 1
                        failed_count += 1

                pbar.update(1)

    # Filter None and sort by original order
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x.get('sample_id', 0))

    # Remove sample_id from final results
    for r in results:
        r.pop('sample_id', None)

    # Save final results
    print(f"\nSaving normalized dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "output_file": output_file,
        "model": model,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "total_samples": len(dataset),
        "successfully_normalized": success_count,
        "failed_normalization": failed_count,
        "success_rate": f"{success_count/len(dataset)*100:.2f}%"
    }

    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("NORMALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {summary['model']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successfully normalized: {summary['successfully_normalized']} ({summary['success_rate']})")
    print(f"Failed normalization: {summary['failed_normalization']}")
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize medical reports to standard format using LLM")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file (mimic_cxr_sharegpt format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for normalized reports"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=NORMALIZE_MODEL,
        help=f"Model to use for normalization (default: from config.yaml = {NORMALIZE_MODEL})"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent API calls (default: from config.yaml = {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum number of retries for failed API calls (default: from config.yaml = {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of samples to process (default: from config.yaml = {DEFAULT_MAX_SAMPLES or 'all'})"
    )

    args = parser.parse_args()

    print("\nConfiguration loaded:")
    print(f"  API Base URL: {normalize_api_config['base_url']}")
    print(f"  Auth Mode: {normalize_api_config['auth_mode']}")
    print(f"  Default Model: {normalize_api_config['model']}")
    print(f"  Using Model: {args.model}")
    print()

    # Run normalization
    normalize_dataset(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_samples=args.max_samples
    )
