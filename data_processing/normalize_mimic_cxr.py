#!/usr/bin/env python3
"""
Normalize MIMIC-CXR reports from local dataset.
Process by prefix directory (p10-p19) for better checkpoint recovery.
Output format: sample_id, image_paths, original_report, normalized_json
"""

import sys
import json
import argparse
import threading
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config, create_openai_client

config = load_config()
normalize_api_config = config.get_normalize_api_config()
norm_config = config.get_normalization_config()
client = create_openai_client(normalize_api_config)

EXTRACT_MODEL = normalize_api_config["model"]
JUDGE_MODEL = normalize_api_config["model"]

DEFAULT_CONCURRENCY = norm_config.get("concurrency", 100)
DEFAULT_MAX_RETRIES = norm_config.get("max_retries", 5)
DEFAULT_OUTPUT_DIR = (
    "/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/mimic-cxr"
)
DEFAULT_CHECKPOINT_DIR = (
    "/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/mimic-cxr/checkpoints"
)

EXTRACT_PROMPT = """You are a medical report parser. Your task is to extract information from a chest X-ray radiology report and output it as structured JSON.

**CRITICAL RULES:**
1. **Preserve All Medical Content**: Extract ALL medical findings, diagnoses, and observations EXACTLY as stated. Do NOT omit, summarize, or modify any clinical information.
2. **Handle Missing Sections**: If a section is not present in the original report, set its value to null.
3. **Handle Placeholders**: Replace underscores (___) with appropriate generic placeholders:
   - Age/Gender: [AGE/GENDER]
   - Dates: [DATE]
   - Study IDs: [PRIOR_STUDY]
4. **Extract WET READ**: If present, extract it separately.
5. **Findings vs Impression**:
   - If the report has both FINDINGS and IMPRESSION, extract them separately
   - If only IMPRESSION exists, extract detailed observations to "findings" and summary to "impression"
   - If only FINDINGS exists, extract main conclusion to "impression"

**Output Format:**
Output ONLY a valid JSON object with this exact structure:
{{
  "examination": "string or null",
  "indication": "string or null",
  "technique": "string or null",
  "comparison": "string or null",
  "findings": "string or null",
  "impression": "string or null",
  "wet_read": "string or null"
}}

**Original Report:**
{original_report}

**Instructions:** Output ONLY the JSON object. No additional text, no explanations, no markdown code blocks.
"""

JUDGE_PROMPT = """You are an expert medical report reviewer. Your task is to verify whether the extracted structured information is faithful to the original report.

**IMPORTANT: What to IGNORE:**
1. **Placeholder Replacement**: Replacing "___" with "[AGE/GENDER]", "[DATE]", "[PRIOR_STUDY]" etc. is ACCEPTABLE and should NOT be considered a modification
2. **Format Changes**: Section headers, capitalization, spacing differences are ACCEPTABLE
3. **Structural Reorganization**: Moving content between sections (e.g., extracting detailed findings from IMPRESSION to FINDINGS) is ACCEPTABLE as long as NO medical content is lost

**Verification Criteria (Focus ONLY on Medical Content):**

1. **Completeness**: Are ALL medical findings, diagnoses, and clinical observations from the original report present somewhere in the extracted JSON?
2. **No Hallucination**: Is there any MEDICAL information (diseases, findings, diagnoses) in the JSON that doesn't exist in the original?
3. **No Critical Omission**: Are there any IMPORTANT clinical findings in the original that are completely missing from the JSON?
4. **Accuracy**: Is the clinical meaning preserved (e.g., "No pneumothorax" must not become "Pneumothorax")?

**Scoring:**
- **PASS**: All medical findings are present and accurate, even if reorganized across sections. Placeholder replacements are fine.
- **FAIL**: Important medical findings are omitted, hallucinated, or clinical meaning is changed

**Original Report:**
{original_report}

**Extracted JSON:**
{extracted_json}

**Output Format:**
Output ONLY a JSON object:
{{
  "verdict": "PASS" or "FAIL",
  "reasoning": "Brief explanation focusing on medical content only",
  "issues": ["list of specific MEDICAL content issues if FAIL, empty list if PASS"]
}}

**Instructions:** Output ONLY the JSON object. No additional text.
"""


def call_llm_with_retry(
    prompt: str,
    model: str,
    max_retries: int = 5,
    temperature: float = 0.1,
    sample_id: str = "",
) -> dict:
    for attempt in range(max_retries):
        raw_content = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Output only valid JSON without any additional text or markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=16384,
                extra_body={"chat_template_kwargs": {"thinking": True}},
            )

            raw_content = response.choices[0].message.content
            if raw_content is None:
                raw_content = ""
            raw_content = raw_content.strip()

            if not raw_content:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return {"success": False, "error": "Empty response", "raw": ""}

            if "rasse" in raw_content:
                raw_content = raw_content.split("rasse>")[-1].strip()

            result = json.loads(raw_content)
            return {"success": True, "data": result, "raw": raw_content}

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {
                "success": False,
                "error": f"JSON decode error: {str(e)}",
                "raw": raw_content,
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return {
                "success": False,
                "error": f"API error: {str(e)}",
                "raw": raw_content,
            }

    return {"success": False, "error": "Max retries exceeded", "raw": ""}


def extract_structured_json(
    original_report: str, model: str, sample_id: str = ""
) -> dict:
    prompt = EXTRACT_PROMPT.format(original_report=original_report)
    result = call_llm_with_retry(prompt, model, sample_id=sample_id)

    if result["success"]:
        required_keys = [
            "examination",
            "indication",
            "technique",
            "comparison",
            "findings",
            "impression",
            "wet_read",
        ]
        if all(key in result["data"] for key in required_keys):
            return {"success": True, "json": result["data"], "model": model}
        else:
            return {
                "success": False,
                "error": f"Missing required keys. Got: {list(result['data'].keys())}",
                "raw": result["raw"],
            }
    else:
        return {
            "success": False,
            "error": result["error"],
            "raw": result.get("raw", ""),
        }


def judge_faithfulness(
    original_report: str, extracted_json: dict, model: str, sample_id: str = ""
) -> dict:
    prompt = JUDGE_PROMPT.format(
        original_report=original_report,
        extracted_json=json.dumps(extracted_json, indent=2),
    )

    result = call_llm_with_retry(prompt, model, temperature=0.0, sample_id=sample_id)

    if result["success"]:
        verdict_data = result["data"]
        if "verdict" in verdict_data:
            return {
                "success": True,
                "verdict": verdict_data["verdict"],
                "reasoning": verdict_data.get("reasoning", ""),
                "issues": verdict_data.get("issues", []),
                "model": model,
            }
        else:
            return {
                "success": False,
                "error": "Verdict not found in response",
                "raw": result["raw"],
            }
    else:
        return {
            "success": False,
            "error": result["error"],
            "raw": result.get("raw", ""),
        }


def collect_samples_from_prefix(
    prefix_dir: Path, skip_ids: set = None
) -> tuple[list, int]:
    """Collect samples from a single prefix directory (e.g., p10)."""
    samples = []
    skip_ids = skip_ids or set()
    skipped = 0

    patient_dirs = sorted([d for d in prefix_dir.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        study_files = sorted(
            [f for f in patient_dir.iterdir() if f.is_file() and f.suffix == ".txt"]
        )

        for report_file in study_files:
            study_id = report_file.stem
            sample_id = f"{patient_id}_{study_id}"

            if sample_id in skip_ids:
                skipped += 1
                continue

            study_dir = patient_dir / study_id
            images = sorted(list(study_dir.glob("*.jpg")))
            if not images:
                continue

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    original_report = f.read().strip()
            except Exception as e:
                print(f"Error reading {report_file}: {e}")
                continue

            sample = {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "study_id": study_id,
                "image_paths": [str(img.absolute()) for img in images],
                "original_report": original_report,
            }
            samples.append(sample)

    return samples, skipped


def process_single_sample(
    sample: dict, extract_model: str, judge_model: str, max_retries: int = 5
) -> dict:
    sample_id = sample["sample_id"]
    original_report = sample["original_report"]
    result = sample.copy()

    extracted_json = None
    for attempt in range(max_retries):
        extract_result = extract_structured_json(
            original_report, extract_model, sample_id=sample_id
        )

        if extract_result["success"]:
            extracted_json = extract_result["json"]
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                result["normalized_json"] = None
                result["normalization_status"] = "extraction_failed"
                result["error"] = extract_result["error"]
                return result

    for attempt in range(max_retries):
        judge_result = judge_faithfulness(
            original_report, extracted_json, judge_model, sample_id=sample_id
        )

        if not judge_result["success"]:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            else:
                result["normalized_json"] = None
                result["normalization_status"] = "judge_failed"
                result["error"] = judge_result["error"]
                return result

        if judge_result["verdict"] == "PASS":
            result["normalized_json"] = extracted_json
            result["normalization_status"] = "success"
            return result
        else:
            if attempt < max_retries - 1:
                for re_extract_attempt in range(max_retries):
                    extract_result = extract_structured_json(
                        original_report, extract_model, sample_id=sample_id
                    )
                    if extract_result["success"]:
                        extracted_json = extract_result["json"]
                        break
                    elif re_extract_attempt < max_retries - 1:
                        time.sleep(2**re_extract_attempt)
                    else:
                        result["normalized_json"] = None
                        result["normalization_status"] = "extraction_failed"
                        result["error"] = (
                            "Re-extraction failed after verification failure"
                        )
                        return result
                time.sleep(2**attempt)
            else:
                result["normalized_json"] = None
                result["normalization_status"] = "verification_failed"
                result["verification_issues"] = judge_result["issues"]
                return result

    result["normalized_json"] = None
    result["normalization_status"] = "unknown_error"
    return result


def process_prefix_directory(
    prefix_dir: Path,
    checkpoint_dir: Path,
    extract_model: str,
    judge_model: str,
    concurrency: int = 100,
    max_retries: int = 5,
) -> dict:
    """Process a single prefix directory (p10, p11, ..., p19)."""
    prefix_name = prefix_dir.name
    output_file = checkpoint_dir / f"mimic_cxr_{prefix_name}.json"
    intermediate_file = checkpoint_dir / f"mimic_cxr_{prefix_name}_intermediate.json"

    processed_ids = set()
    existing_results = []

    if intermediate_file.exists():
        try:
            with open(intermediate_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            processed_ids = {r["sample_id"] for r in existing_results}
            print(f"  Loaded {len(existing_results)} processed samples from checkpoint")
        except Exception as e:
            print(f"  Checkpoint file corrupted, starting fresh: {e}")
            processed_ids = set()
            existing_results = []

    print(f"  Collecting samples from {prefix_name}...")
    samples, skipped = collect_samples_from_prefix(prefix_dir, processed_ids)

    if not samples:
        print(f"  No new samples to process in {prefix_name}")
        return {"prefix": prefix_name, "total": len(existing_results), "new": 0}

    print(f"  Found {len(samples)} new samples (skipped {skipped} already processed)")

    results = existing_results.copy()
    results_lock = threading.Lock()
    success_count = sum(
        1 for r in existing_results if r.get("normalization_status") == "success"
    )
    failed_count = len(existing_results) - success_count
    completed_count = len(existing_results)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample, sample, extract_model, judge_model, max_retries
            ): sample
            for sample in samples
        }

        with tqdm(total=len(samples), desc=f"  Processing {prefix_name}") as pbar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                        completed_count += 1
                        if result.get("normalization_status") == "success":
                            success_count += 1
                        else:
                            failed_count += 1

                        if completed_count % 50 == 0:
                            with open(intermediate_file, "w", encoding="utf-8") as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    with results_lock:
                        result = sample.copy()
                        result["normalized_json"] = None
                        result["normalization_status"] = "exception"
                        result["error"] = str(e)
                        results.append(result)
                        failed_count += 1

                pbar.update(1)

    with open(intermediate_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output_samples = [
        {
            "sample_id": r["sample_id"],
            "image_paths": r["image_paths"],
            "original_report": r["original_report"],
            "normalized_json": r.get("normalized_json"),
        }
        for r in results
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(output_samples)} samples")

    summary = {
        "prefix": prefix_name,
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "success_rate": f"{success_count / len(results) * 100:.2f}%"
        if results
        else "N/A",
    }

    return summary


def normalize_all_prefixes(
    input_dir: str,
    output_dir: str,
    checkpoint_dir: str,
    extract_model: str,
    judge_model: str,
    concurrency: int = 100,
    max_retries: int = 5,
    start_from: str = None,
):
    """Process all prefix directories sequentially."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    checkpoint_path = Path(checkpoint_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    prefix_dirs = sorted(
        [
            d
            for d in input_path.iterdir()
            if d.is_dir() and d.name.startswith("p") and not d.name.startswith(".")
        ]
    )

    print(f"\n{'=' * 80}")
    print("NORMALIZING MIMIC-CXR REPORTS (BY PREFIX DIRECTORY)")
    print(f"{'=' * 80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Extract model: {extract_model}")
    print(f"Judge model: {judge_model}")
    print(f"Prefix directories found: {[d.name for d in prefix_dirs]}")
    print(f"Concurrency: {concurrency}")
    print(f"{'=' * 80}\n")

    all_summaries = []
    start_processing = start_from is None

    for prefix_dir in prefix_dirs:
        prefix_name = prefix_dir.name

        if not start_processing:
            if prefix_name == start_from:
                start_processing = True
            else:
                print(f"Skipping {prefix_name} (starting from {start_from})")
                output_file = checkpoint_path / f"mimic_cxr_{prefix_name}.json"
                if output_file.exists():
                    with open(output_file) as f:
                        existing = json.load(f)
                    success = sum(1 for r in existing if r.get("normalized_json"))
                    all_summaries.append(
                        {
                            "prefix": prefix_name,
                            "total": len(existing),
                            "success": success,
                            "skipped": True,
                        }
                    )
                continue

        print(f"\n{'=' * 80}")
        print(f"Processing {prefix_name}")
        print(f"{'=' * 80}")

        summary = process_prefix_directory(
            prefix_dir=prefix_dir,
            checkpoint_dir=checkpoint_path,
            extract_model=extract_model,
            judge_model=judge_model,
            concurrency=concurrency,
            max_retries=max_retries,
        )
        all_summaries.append(summary)

        print(
            f"\n  {prefix_name} complete: {summary['success']}/{summary['total']} success ({summary['success_rate']})"
        )

    print(f"\n{'=' * 80}")
    print("MERGING ALL RESULTS")
    print(f"{'=' * 80}")

    all_results = []
    for prefix_dir in prefix_dirs:
        output_file = checkpoint_path / f"mimic_cxr_{prefix_dir.name}.json"
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.extend(data)
                print(f"  {prefix_dir.name}: {len(data)} samples")

    merged_file = output_path / "mimic_cxr_normalized.json"
    with open(merged_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nMerged {len(all_results)} total samples to {merged_file}")

    total_success = sum(s.get("success", 0) for s in all_summaries)
    total_samples = sum(s.get("total", 0) for s in all_summaries)

    final_summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "extract_model": extract_model,
        "judge_model": judge_model,
        "concurrency": concurrency,
        "prefix_summaries": all_summaries,
        "total_samples": total_samples,
        "total_success": total_success,
        "total_failed": total_samples - total_success,
        "success_rate": f"{total_success / total_samples * 100:.2f}%"
        if total_samples > 0
        else "N/A",
    }

    summary_file = output_path / "mimic_cxr_normalized_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total samples: {total_samples}")
    print(f"Success: {total_success} ({final_summary['success_rate']})")
    print(f"Failed: {total_samples - total_success}")
    print(f"\nResults saved to:")
    print(f"  - Checkpoints: {checkpoint_path}/")
    print(f"  - Final output: {merged_file}")
    print(f"  - Summary: {summary_file}")
    print(f"{'=' * 80}")

    return all_results, final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize MIMIC-CXR reports by prefix directory"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/mimic-cxr/files",
        help="Input directory containing MIMIC-CXR data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for final merged file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--extract_model",
        type=str,
        default=EXTRACT_MODEL,
        help=f"Model for extraction (default: {EXTRACT_MODEL})",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL,
        help=f"Model for verification (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        help="Start from a specific prefix (e.g., p15)",
    )

    args = parser.parse_args()

    print("\nConfiguration:")
    print(f"  API Base URL: {normalize_api_config['base_url']}")
    print(f"  Extract Model: {args.extract_model}")
    print(f"  Judge Model: {args.judge_model}")
    print(f"  Input Dir: {args.input_dir}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print(f"  Start From: {args.start_from or 'p10 (beginning)'}")
    print()

    normalize_all_prefixes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        extract_model=args.extract_model,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        start_from=args.start_from,
    )
