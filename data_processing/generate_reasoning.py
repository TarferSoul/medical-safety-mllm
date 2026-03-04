#!/usr/bin/env python3
"""
Generate reasoning script: Given X-ray images and ground truth report,
ask LLM to explain the reasoning process from images to report.
"""

import sys
import json
import argparse
import threading
import re
import time
import base64
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, create_openai_client

# Load configuration
config = load_config()

# Get API config - using reasoning API
reasoning_api_config = config.get_reasoning_api_config()

# Get reasoning settings
reasoning_config = config.get("reasoning_api", {})

# Initialize client using unified interface
client = create_openai_client(reasoning_api_config)

# Default model (can be overridden by command line arguments)
DEFAULT_MODEL = reasoning_api_config['model']

# Default settings from config
DEFAULT_INPUT = reasoning_config.get('input', None)
DEFAULT_OUTPUT_DIR = reasoning_config.get('output_dir', './reasoning_results')
DEFAULT_CONCURRENCY = reasoning_config.get('concurrency', 20)
DEFAULT_MAX_RETRIES = reasoning_config.get('max_retries', 3)
DEFAULT_MAX_SAMPLES = reasoning_config.get('max_samples', None)
DEFAULT_INTERMEDIATE_SAVE = reasoning_config.get('intermediate_save_interval', 20)

# Generation parameters from config
DEFAULT_MAX_TOKENS = reasoning_config.get('max_tokens', 32768)
DEFAULT_TEMPERATURE = reasoning_config.get('temperature', 0.7)
DEFAULT_TOP_K = reasoning_config.get('top_k', 20)
DEFAULT_ENABLE_THINKING = reasoning_config.get('enable_thinking', True)


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


# Reasoning prompt template
REASONING_PROMPT = """Role: Expert Radiologist.
Task: Reverse-engineer the radiologist's thought process. You are given a Chest X-ray and the Ground Truth Report. Your goal is to generate the "Chain of Thought" that connects the raw image pixels to the final diagnosis.

**Ground Truth Report:**
{ground_truth}

**Instructions:**
Analyze the case and output your reasoning in 4 strict steps.

**Step 1: Anatomy & Support Devices (The "Safety Check")**
- Observe the image quality (projection, rotation).
- Identify ALL medical devices (lines, tubes, clips, etc.).
- *Constraint:* Describe their precise anatomical termination points based ONLY on visual evidence.

**Step 2: Visual Feature Extraction (The "What")**
- Identify abnormal visual signals (opacities, lucencies, deformities).
- Use precise radiological terms (e.g., "patchy consolidation," "blunting," "silhouette sign").
- *CRITICAL RULE:* Do NOT mention the patient's clinical history (e.g., "AML," "sepsis," "transplant") in this step. Describe only what is physically visible on the image.

**Step 3: Diagnostic Synthesis (The "Why")**
- Now, combine the Visual Features from Step 2 with the Clinical History provided in the report.
- Explain the logic: "The visual finding of [Feature X] combined with the history of [Condition Y] supports the diagnosis of [Diagnosis Z]."
- Differentiate: Explain why the findings support the specific conclusion in the report over other possibilities (e.g., "Why is this likely pneumonia and not just edema?").

**Step 4: Conclusion Alignment**
- Summarize the primary finding that drives the immediate clinical management.

**Output Format:**
Provide the reasoning directly. Do not use conversational fillers."""


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def prepare_messages(images: list, ground_truth: str) -> list:
    """Prepare messages for API call with images and ground truth."""
    content = []

    # Add images
    for img_path in images:
        try:
            img_base64 = encode_image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        except Exception as e:
            tqdm.write(f"Warning: Failed to load image {img_path}: {e}")
            continue

    # Add reasoning prompt with ground truth
    prompt_text = REASONING_PROMPT.format(ground_truth=ground_truth)
    content.append({
        "type": "text",
        "text": prompt_text
    })

    return [{"role": "user", "content": content}]


def call_api(client, messages: list, model: str, max_retries: int = None,
             max_tokens: int = None, temperature: float = None,
             top_k: int = None, enable_thinking: bool = None) -> dict:
    """Call API with retry logic."""
    # Use defaults from config if not specified
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if enable_thinking is None:
        enable_thinking = DEFAULT_ENABLE_THINKING

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "top_k": top_k,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )

            if response.choices:
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model": model
                }
            else:
                return {
                    "success": False,
                    "error": "No response choices returned"
                }

        except Exception as e:
            # Handle API errors
            if hasattr(e, 'status_code'):
                error_msg = f"API error (status {e.status_code}): {getattr(e, 'response', e)}"
            else:
                error_msg = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {
                "success": False,
                "error": error_msg
            }

    return {
        "success": False,
        "error": "Max retries reached"
    }


def process_single_sample(client, idx: int, sample: dict, model: str) -> dict:
    """Process a single sample and generate reasoning."""
    try:
        # Extract data
        conversations = sample["conversations"]
        images = sample["images"]

        # Get ground truth
        ground_truth = next((conv["value"] for conv in conversations if conv["from"] == "gpt"), None)

        if not ground_truth:
            return {
                "sample_id": idx,
                "images": images,
                "ground_truth": None,
                "reasoning": None,
                "success": False,
                "error": "No ground truth found",
                "model": model
            }

        # Prepare messages
        messages = prepare_messages(images, ground_truth)

        # Call API
        result = call_api(client, messages, model)

        # Store result
        sample_result = {
            "sample_id": idx,
            "images": images,
            "ground_truth": ground_truth,
            "reasoning": result.get("content", "") if result["success"] else None,
            "success": result["success"],
            "error": result.get("error"),
            "model": model
        }
        return sample_result

    except Exception as e:
        return {
            "sample_id": idx,
            "images": sample.get("images", []),
            "ground_truth": None,
            "reasoning": None,
            "success": False,
            "error": f"Exception: {str(e)}",
            "model": model
        }


def generate_reasoning(client, test_data_path: str, output_dir: str, model: str,
                       max_samples: int = None, concurrency: int = 20,
                       intermediate_save_interval: int = None):
    """Generate reasoning for all samples with concurrent processing."""

    # Use default from config if not specified
    if intermediate_save_interval is None:
        intermediate_save_interval = DEFAULT_INTERMEDIATE_SAVE

    # Load test data
    print(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Loaded {len(test_data)} test samples")
    print(f"Using concurrency: {concurrency}")
    print(f"Model: {model}")

    # Create output directory with model-specific subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = "reasoning_" + sanitize_model_name(model)

    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this reasoning model
    output_path = base_output_path / model_name_clean
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results storage with thread safety
    results = [None] * len(test_data)
    results_lock = threading.Lock()
    completed_count = 0
    successful = 0
    failed = 0

    # Process samples concurrently
    print("\nGenerating reasoning...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_sample, client, idx, sample, model): idx
            for idx, sample in enumerate(test_data)
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(test_data), desc="Processing") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()

                    with results_lock:
                        results[idx] = result
                        completed_count += 1

                        if result["success"]:
                            successful += 1
                        else:
                            failed += 1
                            tqdm.write(f"Sample {idx} failed: {result.get('error')}")

                        # Save intermediate results
                        if completed_count % intermediate_save_interval == 0:
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x["sample_id"])

                            intermediate_file = output_path / f"intermediate_reasoning_{model_name_clean}_{timestamp}.json"
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    with results_lock:
                        results[idx] = {
                            "sample_id": idx,
                            "images": [],
                            "ground_truth": None,
                            "reasoning": None,
                            "success": False,
                            "error": f"Future exception: {str(e)}",
                            "model": model
                        }
                        completed_count += 1
                        failed += 1

                pbar.update(1)

    # Filter out None values and sort by sample_id
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x["sample_id"])

    # Save final results
    results_file = output_path / f"reasoning_results_{model_name_clean}_{timestamp}.json"
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "model": model,
        "concurrency": concurrency,
        "total_samples": len(test_data),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(test_data) * 100 if len(test_data) > 0 else 0
    }

    summary_file = output_path / f"reasoning_summary_{model_name_clean}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("REASONING GENERATION SUMMARY")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(f"Concurrency: {summary['concurrency']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.2f}%")
    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 60)

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning from X-ray images and ground truth reports")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input data JSON file (default: from config.yaml = {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: from config.yaml = {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of samples to process (default: from config.yaml = {DEFAULT_MAX_SAMPLES or 'all'})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: from config.yaml = {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent API calls (default: from config.yaml = {DEFAULT_CONCURRENCY})"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input:
        parser.error("--input is required. Please specify input file or set it in config.yaml")

    print("\nConfiguration loaded:")
    print(f"  API Base URL: {reasoning_api_config['base_url']}")
    print(f"  Auth Mode: {reasoning_api_config['auth_mode']}")
    print(f"  Model: {args.model}")
    print(f"  Input File: {args.input}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Max Samples: {args.max_samples or 'all'}")
    print()

    # Run reasoning generation
    generate_reasoning(
        client=client,
        test_data_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        max_samples=args.max_samples,
        concurrency=args.concurrency
    )
