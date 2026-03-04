#!/usr/bin/env python3
"""
Integrated evaluation script: Generate predictions and judge them in one workflow.

Workflow:
For each sample (in parallel):
  1. Generate prediction using prediction model
  2. Immediately evaluate prediction using judge model
  3. Return complete result with both prediction and judgment

This approach provides faster feedback and better memory efficiency.
"""

import sys
import json
import time
import threading
import argparse
import re
import base64
import openai
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, create_openai_client

# Load configuration
config = load_config()

# Get API configs
pred_api_config = config.get_prediction_api_config()
judge_api_config = config.get_judge_api_config()

# Get evaluation settings
eval_config = config.get_evaluation_config()

# Initialize clients using unified interface
pred_client = create_openai_client(pred_api_config)
judge_client = create_openai_client(judge_api_config)

# Default models (can be overridden by command line arguments)
DEFAULT_PRED_MODEL = pred_api_config['model']
DEFAULT_JUDGE_MODEL = judge_api_config['model']

# Default settings from config
DEFAULT_TEST_DATA = eval_config.get('test_data', None)
DEFAULT_OUTPUT_DIR = eval_config.get('output_dir', './evaluation_results')
DEFAULT_CONCURRENCY = eval_config.get('prediction_concurrency', 50)
DEFAULT_JUDGE_MAX_RETRIES = eval_config.get('judge_max_retries', 5)
DEFAULT_MAX_SAMPLES = eval_config.get('default_max_samples', None)
DEFAULT_INTERMEDIATE_SAVE = eval_config.get('intermediate_save_interval', 50)

# Generation parameters from config
DEFAULT_PRED_MAX_TOKENS = eval_config.get('prediction_max_tokens', 4096)
DEFAULT_PRED_TEMPERATURE = eval_config.get('prediction_temperature', 0.7)
DEFAULT_PRED_TOP_K = eval_config.get('prediction_top_k', 20)
DEFAULT_ENABLE_THINKING = eval_config.get('enable_thinking', True)

DEFAULT_JUDGE_MAX_TOKENS = eval_config.get('judge_max_tokens', 10000)
DEFAULT_JUDGE_TEMPERATURE = eval_config.get('judge_temperature', 0.0)

# --- Judge Prompt ---
JUDGE_PROMPT_TEMPLATE = """You are an expert Radiologist and Medical Evaluator. Your task is to evaluate the quality of a generated chest X-ray report (Candidate) by comparing it against the expert-written Ground Truth report (Reference).

### Evaluation Criteria:
1. **Clinical Accuracy (Most Important):** Does the Candidate identify the same pathologies (e.g., pneumonia, effusion, pneumothorax, edema) as the Reference?
2. **Negation correctness:** specific attention to "No", "Not", "Free of". Confusing "No pneumothorax" with "Small pneumothorax" is a critical error.
3. **Anatomical Correctness:** Are the findings localized correctly (e.g., Left Lower Lobe vs Right Upper Lobe)?
4. **Omissions vs Hallucinations:**
   - **Omission:** The Reference reports a disease, but the Candidate misses it.
   - **Hallucination:** The Reference says normal, but the Candidate invents a disease.

### Instructions:
- Ignore formatting differences (e.g., "Findings:" vs "FINDINGS").
- Ignore the "INDICATION" and "COMPARISON" sections unless they contain critical diagnostic findings not present elsewhere. Focus heavily on "FINDINGS" and "IMPRESSION".
- Synonyms are accepted (e.g., "Opacities" ≈ "Infiltrates" ≈ "Consolidation" in specific contexts; "Cardiomegaly" ≈ "Enlarged heart").

### Input Data:
**[Ground Truth Report]:**
{ground_truth}

**[Candidate Report]:**
{prediction}

### Output Format:
ONLY return a JSON object with the following structure, DO NOT RETURN ANYTHING ELSE:
{{
  "reasoning": "Step-by-step comparison of findings...",
  "error_types": ["None" | "Omission" | "Hallucination" | "Localization Error" | "Severity Error"],
  "clinical_accuracy_score": <float 0.0 to 10.0>
}}

**Scoring Rubric:**
- **10.0:** Perfect clinical alignment. Only stylistic differences.
- **8.0-9.0:** Minor discrepancies (e.g., missed a minor scar, or slight severity difference), but main diagnosis is correct.
- **5.0-7.0:** Correctly identifies normal vs abnormal, but misses specific details (e.g., wrong lobe) or hallucinates a minor finding.
- **1.0-4.0:** Major diagnostic error (e.g., Reference says "Pneumonia", Candidate says "Normal"; or Candidate hallucinates "Edema" when lungs are clear).
"""


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use in filenames.
    Removes paths, special characters, and limits length.
    """
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    model_name = re.sub(r'[^\w\-.]', '_', model_name)
    model_name = re.sub(r'_+', '_', model_name)
    if len(model_name) > 50:
        model_name = model_name[:50]
    return model_name


# ============================================================================
# PREDICTION AND JUDGMENT FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def prepare_prediction_messages(images: list, prompt: str) -> list:
    """Prepare messages for prediction API call with images."""
    content = []

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

    text_prompt = prompt.replace("<image>", "").strip()
    content.append({
        "type": "text",
        "text": text_prompt
    })

    return [{"role": "user", "content": content}]


def call_prediction_api(client, messages: list, model: str, max_retries: int = 3) -> dict:
    """Call prediction API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
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
                    "model": model
                }
            else:
                return {
                    "success": False,
                    "error": "No response choices returned"
                }

        except openai.APIStatusError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
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


def process_single_prediction(pred_client, idx: int, sample: dict, model: str) -> dict:
    """Process a single sample and generate prediction."""
    try:
        conversations = sample["conversations"]
        images = sample["images"]

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
                "model": model
            }

        ground_truth = next((conv["value"] for conv in conversations if conv["from"] == "gpt"), None)
        messages = prepare_prediction_messages(images, human_msg)
        result = call_prediction_api(pred_client, messages, model)

        return {
            "sample_id": idx,
            "images": images,
            "prompt": human_msg,
            "ground_truth": ground_truth,
            "prediction": result.get("content", "") if result["success"] else None,
            "success": result["success"],
            "error": result.get("error"),
            "model": model
        }

    except Exception as e:
        return {
            "sample_id": idx,
            "images": sample.get("images", []),
            "prompt": None,
            "ground_truth": None,
            "prediction": None,
            "success": False,
            "error": f"Exception: {str(e)}",
            "model": model
        }


def call_judge_api(judge_client, prompt: str, judge_model: str, max_retries: int = 5) -> dict:
    """Call judge API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10000
            )

            raw_content = response.choices[0].message.content

            # Remove <think> tags if present
            if "</think>" in raw_content:
                raw_content = raw_content.split("</think>")[-1].strip()

            evaluation_output = json.loads(raw_content)

            if not evaluation_output or len(evaluation_output) == 0:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {
                        "error": "Empty response from API after retries",
                        "reasoning": "API returned empty JSON",
                        "error_types": ["Empty Response"],
                        "clinical_accuracy_score": 0.0,
                        "attempts": max_retries
                    }

            return evaluation_output

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    "error": f"JSON decode error after {max_retries} attempts: {str(e)}",
                    "reasoning": "Failed to parse API response as JSON",
                    "error_types": ["JSON Parse Error"],
                    "clinical_accuracy_score": 0.0,
                    "raw_response": raw_content[:500] if 'raw_content' in locals() else "",
                    "attempts": max_retries
                }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    "error": f"Evaluation failed after {max_retries} attempts: {str(e)}",
                    "reasoning": "Evaluation failed",
                    "error_types": ["Evaluation Error"],
                    "clinical_accuracy_score": 0.0,
                    "attempts": max_retries
                }

    return {
        "error": "Max retries reached",
        "reasoning": "Evaluation failed",
        "error_types": ["Max Retries Exceeded"],
        "clinical_accuracy_score": 0.0,
        "attempts": max_retries
    }


def process_single_judgment(judge_client, item: dict, judge_model: str, max_retries: int = 5) -> dict:
    """Evaluate a single prediction using judge model."""
    gt_text = item['ground_truth']
    pred_text = item.get('prediction', '')

    # Remove <think> tags from prediction
    if pred_text and "</think>" in pred_text:
        pred_text = pred_text.split("</think>")[-1].strip()

    final_prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=gt_text,
        prediction=pred_text
    )

    evaluation_output = call_judge_api(judge_client, final_prompt, judge_model, max_retries)

    result_item = item.copy()
    result_item['eval_results'] = {
        "eval_input": final_prompt,
        "eval_output": evaluation_output
    }
    result_item['judge_model'] = judge_model

    return result_item


def process_sample_complete(pred_client, judge_client, idx: int, sample: dict,
                           pred_model: str, judge_model: str, judge_max_retries: int = 5) -> dict:
    """
    Process a single sample: predict then immediately judge.

    Args:
        pred_client: Prediction model client
        judge_client: Judge model client
        idx: Sample index
        sample: Sample data
        pred_model: Prediction model name
        judge_model: Judge model name
        judge_max_retries: Max retries for judge

    Returns:
        Complete result with both prediction and judgment
    """
    # Step 1: Generate prediction
    pred_result = process_single_prediction(pred_client, idx, sample, pred_model)

    # Step 2: If prediction succeeded, judge it immediately
    if pred_result["success"] and pred_result.get("prediction"):
        result = process_single_judgment(judge_client, pred_result, judge_model, judge_max_retries)
    else:
        # If prediction failed, add empty judgment
        result = pred_result.copy()
        result['eval_results'] = {
            "eval_input": "",
            "eval_output": {
                "error": "Prediction failed, no judgment performed",
                "reasoning": "Cannot judge without prediction",
                "error_types": ["Prediction Failed"],
                "clinical_accuracy_score": 0.0
            }
        }
        result['judge_model'] = judge_model

    return result


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def evaluate_and_judge(pred_client, judge_client, test_data_path: str, output_dir: str,
                       pred_model: str, judge_model: str, max_samples: int = None,
                       concurrency: int = 20, judge_max_retries: int = 5):
    """
    Complete evaluation workflow: predict then judge each sample immediately.

    Args:
        pred_client: OpenAI client for prediction model
        judge_client: OpenAI client for judge model
        test_data_path: Path to test data JSON
        output_dir: Output directory
        pred_model: Prediction model name
        judge_model: Judge model name
        max_samples: Max samples to process (None = all)
        concurrency: Number of concurrent samples to process
        judge_max_retries: Max retries for judge API
    """

    # Load test data
    print(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Loaded {len(test_data)} test samples")

    # Create output directory with model-specific subdirectory
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_model_clean = "pred_" + sanitize_model_name(pred_model)
    judge_model_clean = "judge_" + sanitize_model_name(judge_model)

    # Create subdirectory for this prediction model
    output_path = base_output_path / pred_model_clean
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PROCESS ALL SAMPLES (PREDICT + JUDGE IMMEDIATELY)
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING SAMPLES (PREDICT → JUDGE)")
    print("="*80)
    print(f"Prediction Model: {pred_model}")
    print(f"Judge Model: {judge_model}")
    print(f"Concurrency: {concurrency}")
    print(f"Judge Max Retries: {judge_max_retries}")

    results = [None] * len(test_data)
    results_lock = threading.Lock()
    pred_success = 0
    pred_failed = 0
    judge_success = 0
    judge_failed = 0
    scores = []
    scores_lock = threading.Lock()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(
                process_sample_complete,
                pred_client, judge_client, idx, sample,
                pred_model, judge_model, judge_max_retries
            ): idx
            for idx, sample in enumerate(test_data)
        }

        with tqdm(total=len(test_data), desc="Processing") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()

                    with results_lock:
                        results[idx] = result
                        completed_count += 1

                        # Track prediction success
                        if result.get("success"):
                            pred_success += 1
                        else:
                            pred_failed += 1
                            tqdm.write(f"❌ Sample {idx}: Prediction failed - {result.get('error', 'Unknown error')}")

                        # Check judge success and collect scores
                        if 'eval_results' in result and 'eval_output' in result['eval_results']:
                            eval_output = result['eval_results']['eval_output']

                            # Check if judge failed
                            if 'error' in eval_output:
                                judge_failed += 1
                                error_msg = eval_output.get('error', 'Unknown error')
                                # Only print if it's not a "prediction failed" error
                                if 'Prediction failed' not in error_msg:
                                    tqdm.write(f"⚠️  Sample {idx}: Judge failed - {error_msg[:100]}")
                            else:
                                judge_success += 1

                            # Collect scores
                            if 'clinical_accuracy_score' in eval_output:
                                score = eval_output['clinical_accuracy_score']
                                if isinstance(score, (int, float)) and score > 0:
                                    with scores_lock:
                                        scores.append(score)

                        # Save intermediate results every 50 samples
                        if completed_count % 50 == 0:
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x.get('sample_id', 0))

                            intermediate_file = output_path / f"evaluation_{pred_model_clean}_with_{judge_model_clean}_{timestamp}_intermediate.json"
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)
                            tqdm.write(f"✓ Saved intermediate results ({completed_count}/{len(test_data)} samples)")

                except Exception as e:
                    with results_lock:
                        results[idx] = {
                            "sample_id": idx,
                            "images": test_data[idx].get("images", []),
                            "prompt": None,
                            "ground_truth": None,
                            "prediction": None,
                            "success": False,
                            "error": f"Future exception: {str(e)}",
                            "model": pred_model,
                            "eval_results": {
                                "eval_input": "",
                                "eval_output": {
                                    "error": f"Processing exception: {str(e)}",
                                    "reasoning": "Processing failed",
                                    "error_types": ["Processing Error"],
                                    "clinical_accuracy_score": 0.0
                                }
                            },
                            "judge_model": judge_model
                        }
                        completed_count += 1
                        pred_failed += 1
                        judge_failed += 1
                        tqdm.write(f"💥 Sample {idx}: Processing exception - {str(e)[:100]}")

                pbar.update(1)

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x.get('sample_id', 0))

    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING FINAL RESULTS")
    print("="*80)

    final_file = output_path / f"evaluation_{pred_model_clean}_with_{judge_model_clean}_{timestamp}.json"
    print(f"Saving final results to: {final_file}")
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "prediction_model": pred_model,
        "judge_model": judge_model,
        "concurrency": concurrency,
        "total_samples": len(test_data),
        "prediction_success": pred_success,
        "prediction_failed": pred_failed,
        "prediction_success_rate": pred_success / len(test_data) * 100 if len(test_data) > 0 else 0,
        "judge_success": judge_success,
        "judge_failed": judge_failed,
        "judge_success_rate": judge_success / len(test_data) * 100 if len(test_data) > 0 else 0,
        "successfully_scored": len(scores),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0
    }

    summary_file = output_path / f"evaluation_{pred_model_clean}_with_{judge_model_clean}_{timestamp}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Prediction Model: {pred_model}")
    print(f"Judge Model: {judge_model}")
    print(f"Concurrency: {concurrency}")
    print(f"\nPrediction Results:")
    print(f"  Total samples: {len(test_data)}")
    print(f"  Success: {pred_success}")
    print(f"  Failed: {pred_failed}")
    print(f"  Success rate: {summary['prediction_success_rate']:.2f}%")
    print(f"\nJudge Results:")
    print(f"  Success: {judge_success}")
    print(f"  Failed: {judge_failed}")
    print(f"  Success rate: {summary['judge_success_rate']:.2f}%")
    print(f"  Successfully scored: {len(scores)}")
    if scores:
        print(f"  Average score: {summary['average_score']:.2f}/10.0")
        print(f"  Min score: {summary['min_score']:.2f}")
        print(f"  Max score: {summary['max_score']:.2f}")
    print(f"\nOutput files:")
    print(f"  Final results: {final_file}")
    print(f"  Summary: {summary_file}")
    print("="*80)

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated evaluation: Predict then judge each sample immediately"
    )

    # Data parameters
    parser.add_argument(
        "--test_data",
        type=str,
        default=DEFAULT_TEST_DATA,
        help=f"Path to test data JSON file (default: from config.yaml = {DEFAULT_TEST_DATA})"
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
        help=f"Maximum number of samples to evaluate (default: from config.yaml = {DEFAULT_MAX_SAMPLES or 'all'})"
    )

    # Prediction model parameters
    parser.add_argument(
        "--pred_model",
        type=str,
        default=DEFAULT_PRED_MODEL,
        help=f"Prediction model name (default: from config.yaml = {DEFAULT_PRED_MODEL})"
    )

    # Judge model parameters
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model name (default: from config.yaml = {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--judge_max_retries",
        type=int,
        default=DEFAULT_JUDGE_MAX_RETRIES,
        help=f"Maximum number of retries for judge API (default: from config.yaml = {DEFAULT_JUDGE_MAX_RETRIES})"
    )

    # Concurrency parameters
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent samples to process (default: from config.yaml = {DEFAULT_CONCURRENCY})"
    )

    args = parser.parse_args()

    # Validate test data path
    if not args.test_data:
        parser.error("--test_data is required. Please specify test data file or set it in config.yaml")

    print("\nConfiguration loaded:")
    print(f"  Prediction API Base URL: {pred_api_config['base_url']}")
    print(f"  Prediction Auth Mode: {pred_api_config['auth_mode']}")
    print(f"  Prediction Model: {args.pred_model}")
    print(f"  Judge API Base URL: {judge_api_config['base_url']}")
    print(f"  Judge Auth Mode: {judge_api_config['auth_mode']}")
    print(f"  Judge Model: {args.judge_model}")
    print(f"  Test Data: {args.test_data}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Judge Max Retries: {args.judge_max_retries}")
    print(f"  Max Samples: {args.max_samples or 'all'}")
    print()

    # Run integrated evaluation
    evaluate_and_judge(
        pred_client=pred_client,
        judge_client=judge_client,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        pred_model=args.pred_model,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
        judge_max_retries=args.judge_max_retries
    )
