#!/usr/bin/env python3
"""
Evaluation script for medical imaging model.
Uses OpenAI-style API to generate medical reports from X-ray images.
Supports concurrent API calls for faster evaluation.
"""

import openai
import base64
import json
import time
import sys
import threading
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# --- Configuration ---
BASE_URL = "https://h.pjlab.org.cn/kapi/workspace.kubebrain.io/ailab-ai4good1/rjob-a9c0266e1a8f513d-f61eb500bddbbf61-0-0.xieyuejin/8000/v1/"
API_AK = "fb4b11bcc25b0fd8ac2bdad43aff3692"
API_SK = "1be0a8ad0270381b02108d07ba05ce80"
MODEL = "Qwen3-8B-VL-Mimic-v251230-original-report"

# Paths
TEST_DATA_PATH = "/mnt/shared-storage-user/xieyuejin/MLLM-Safety/MedicalSafety/dataset/mimic_cxr_sharegpt_test.json"
OUTPUT_DIR = "/mnt/shared-storage-user/xieyuejin/MLLM-Safety/MedicalSafety/evaluation_results"


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


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def prepare_messages(images: list, prompt: str) -> list:
    """Prepare messages for API call with images."""
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
            print(f"Warning: Failed to load image {img_path}: {e}")
            continue

    # Add text prompt (remove <image> tags as we're using structured format)
    text_prompt = prompt.replace("<image>", "").strip()
    content.append({
        "type": "text",
        "text": text_prompt
    })

    return [{"role": "user", "content": content}]


def call_api(client, messages: list, max_retries: int = 3) -> dict:
    """Call API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )

            if response.choices:
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model": MODEL
                }
            else:
                return {
                    "success": False,
                    "error": "No response choices returned"
                }

        except openai.APIStatusError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return {
                "success": False,
                "error": f"API error (status {e.status_code}): {e.response.text}"
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "success": False,
                "error": str(e)
            }

    return {
        "success": False,
        "error": "Max retries reached"
    }


def process_single_sample(client, idx: int, sample: dict) -> dict:
    """Process a single sample and return the result."""
    try:
        # Extract data
        conversations = sample["conversations"]
        images = sample["images"]

        # Get human prompt
        human_msg = next((conv["value"] for conv in conversations if conv["from"] == "human"), None)
        if not human_msg:
            return {
                "sample_id": idx,
                "images": images,
                "prompt": None,
                "ground_truth": None,
                "prediction": None,
                "success": False,
                "error": "No human message found",
                "model": MODEL
            }

        # Get ground truth
        ground_truth = next((conv["value"] for conv in conversations if conv["from"] == "gpt"), None)

        # Prepare messages
        messages = prepare_messages(images, human_msg)

        # Call API
        result = call_api(client, messages)

        # Store result
        sample_result = {
            "sample_id": idx,
            "images": images,
            "prompt": human_msg,
            "ground_truth": ground_truth,
            "prediction": result.get("content", "") if result["success"] else None,
            "success": result["success"],
            "error": result.get("error"),
            "model": MODEL
        }
        return sample_result

    except Exception as e:
        return {
            "sample_id": idx,
            "images": sample.get("images", []),
            "prompt": None,
            "ground_truth": None,
            "prediction": None,
            "success": False,
            "error": f"Exception: {str(e)}",
            "model": MODEL
        }


def evaluate_dataset(client, test_data_path: str, output_dir: str, max_samples: int = None, concurrency: int = 20):
    """Evaluate model on test dataset with concurrent API calls."""

    # Load test data
    print(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Loaded {len(test_data)} test samples")
    print(f"Using concurrency: {concurrency}")

    # Create output directory with model-specific subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = "pred_" + sanitize_model_name(MODEL)

    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this model
    output_path = base_output_path / model_name_clean
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results storage with thread safety
    results = [None] * len(test_data)
    results_lock = threading.Lock()
    completed_count = 0
    successful = 0
    failed = 0

    # Process samples concurrently
    print("\nGenerating predictions...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_sample, client, idx, sample): idx
            for idx, sample in enumerate(test_data)
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(test_data), desc="Evaluating") as pbar:
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

                        # Save intermediate results every 20 samples
                        if completed_count % 20 == 0:
                            # Filter out None values and sort by sample_id
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x["sample_id"])

                            intermediate_file = output_path / f"intermediate_results_{model_name_clean}_{timestamp}.json"
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    with results_lock:
                        results[idx] = {
                            "sample_id": idx,
                            "images": [],
                            "prompt": None,
                            "ground_truth": None,
                            "prediction": None,
                            "success": False,
                            "error": f"Future exception: {str(e)}",
                            "model": MODEL
                        }
                        completed_count += 1
                        failed += 1

                pbar.update(1)

    # Filter out None values and sort by sample_id
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x["sample_id"])

    # Save final results
    results_file = output_path / f"evaluation_results_{model_name_clean}_{timestamp}.json"
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "model": MODEL,
        "concurrency": concurrency,
        "total_samples": len(test_data),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(test_data) * 100 if len(test_data) > 0 else 0
    }

    summary_file = output_path / f"evaluation_summary_{model_name_clean}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
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
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate medical imaging model")
    parser.add_argument(
        "--test_data",
        type=str,
        default=TEST_DATA_PATH,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to use"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=BASE_URL,
        help="API base URL"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Number of concurrent API calls (default: 20)"
    )

    args = parser.parse_args()

    # Update global variables if provided
    if args.model:
        MODEL = args.model
    if args.base_url:
        BASE_URL = args.base_url

    # Initialize client
    auth_string = f"{API_AK}:{API_SK}"
    b64_auth_string = base64.b64encode(auth_string.encode()).decode()

    try:
        client = openai.OpenAI(
            base_url=BASE_URL,
            api_key=b64_auth_string,
            default_headers={"Authorization": f"Basic {b64_auth_string}"}
        )
    except ImportError:
        print("Error: 'openai' library not installed. Please run 'pip install openai'.")
        sys.exit(1)

    # Run evaluation
    evaluate_dataset(
        client=client,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        concurrency=args.concurrency
    )
