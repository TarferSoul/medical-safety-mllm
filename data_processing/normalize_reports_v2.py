#!/usr/bin/env python3
"""
Two-stage normalization with LLM-as-judge verification:
Stage 1: Extract structured JSON from original report
Stage 2: Verify JSON faithfulness to original using LLM judge
Stage 3: Generate normalized report from verified JSON
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

# Default models (can be overridden by command line arguments)
EXTRACT_MODEL = normalize_api_config['model']
JUDGE_MODEL = normalize_api_config['model']

# Default settings from config
DEFAULT_CONCURRENCY = norm_config.get('concurrency', 100)
DEFAULT_MAX_RETRIES = norm_config.get('max_retries', 5)
DEFAULT_MAX_SAMPLES = norm_config.get('max_samples', None)
DEFAULT_INPUT = norm_config.get('input', None)

# ==================== Stage 1: Extract Structured JSON ====================

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

# ==================== Stage 2: Judge Verification ====================

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

# ==================== Helper Functions ====================

def call_llm_with_retry(prompt: str, model: str, max_retries: int = 5, temperature: float = 0.1) -> dict:
    """Call LLM with retry logic, return parsed response."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output only valid JSON without any additional text or markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=8192
            )

            raw_content = response.choices[0].message.content.strip()

            # Handle thinking tags
            if "</think>" in raw_content:
                raw_content = raw_content.split("</think>")[-1].strip()

            # Try to parse JSON
            result = json.loads(raw_content)
            return {"success": True, "data": result, "raw": raw_content}

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"success": False, "error": f"JSON decode error: {str(e)}", "raw": raw_content if 'raw_content' in locals() else ""}

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"success": False, "error": f"API error: {str(e)}", "raw": ""}

    return {"success": False, "error": "Max retries exceeded", "raw": ""}


def extract_structured_json(original_report: str, model: str) -> dict:
    """Stage 1: Extract structured JSON from original report."""

    prompt = EXTRACT_PROMPT.format(original_report=original_report)
    result = call_llm_with_retry(prompt, model)

    if result["success"]:
        # Validate JSON structure
        required_keys = ["examination", "indication", "technique", "comparison", "findings", "impression", "wet_read"]
        if all(key in result["data"] for key in required_keys):
            return {
                "success": True,
                "json": result["data"],
                "model": model
            }
        else:
            return {
                "success": False,
                "error": f"Missing required keys. Got: {list(result['data'].keys())}",
                "raw": result["raw"]
            }
    else:
        return {
            "success": False,
            "error": result["error"],
            "raw": result.get("raw", "")
        }


def judge_faithfulness(original_report: str, extracted_json: dict, model: str) -> dict:
    """Stage 2: Judge whether extracted JSON is faithful to original."""

    prompt = JUDGE_PROMPT.format(
        original_report=original_report,
        extracted_json=json.dumps(extracted_json, indent=2)
    )

    result = call_llm_with_retry(prompt, model, temperature=0.0)

    if result["success"]:
        verdict_data = result["data"]
        if "verdict" in verdict_data:
            return {
                "success": True,
                "verdict": verdict_data["verdict"],
                "reasoning": verdict_data.get("reasoning", ""),
                "issues": verdict_data.get("issues", []),
                "model": model
            }
        else:
            return {
                "success": False,
                "error": "Verdict not found in response",
                "raw": result["raw"]
            }
    else:
        return {
            "success": False,
            "error": result["error"],
            "raw": result.get("raw", "")
        }


def json_to_report(structured_json: dict) -> str:
    """Stage 3: Generate normalized report from verified JSON."""

    sections = []
    sections.append("FINAL REPORT")

    if structured_json.get("wet_read"):
        sections.append(f"WET READ: {structured_json['wet_read']}")
        sections.append("" + "_" * 78)

    # Format each section
    if structured_json.get("examination"):
        sections.append(f"EXAMINATION: {structured_json['examination']}")
    else:
        sections.append("EXAMINATION: Not specified.")

    sections.append("")

    if structured_json.get("indication"):
        sections.append(f"INDICATION: {structured_json['indication']}")
    else:
        sections.append("INDICATION: Not specified.")

    sections.append("")

    if structured_json.get("technique"):
        sections.append(f"TECHNIQUE: {structured_json['technique']}")
    else:
        sections.append("TECHNIQUE: Not specified.")

    sections.append("")

    if structured_json.get("comparison"):
        sections.append(f"COMPARISON: {structured_json['comparison']}")
    else:
        sections.append("COMPARISON: None.")

    sections.append("")
    sections.append("FINDINGS:")
    sections.append("")

    if structured_json.get("findings"):
        # Add indentation to findings
        findings_lines = structured_json['findings'].split('\n')
        for line in findings_lines:
            sections.append(f"{line}")
    else:
        sections.append("Not specified.")

    sections.append("")
    sections.append("IMPRESSION:")
    sections.append("")

    if structured_json.get("impression"):
        # Add indentation to impression
        impression_lines = structured_json['impression'].split('\n')
        for line in impression_lines:
            sections.append(f"{line}")
    else:
        sections.append("Not specified.")

    return '\n'.join(sections)


# ==================== Main Processing ====================

def process_single_sample(item: dict, extract_model: str, judge_model: str, max_retries: int = 5) -> dict:
    """Process a single sample through the complete pipeline."""

    sample_id = item.get('sample_id', 'unknown')
    original_report = item['conversations'][1]['value']

    result_item = item.copy()

    # Stage 1: Extract structured JSON
    extract_result = extract_structured_json(original_report, extract_model)

    if not extract_result["success"]:
        tqdm.write(f"❌ Sample {sample_id}: Extraction failed - {extract_result['error']}")
        result_item['normalization'] = {
            "stage": "extraction",
            "success": False,
            "error": extract_result["error"],
            "kept_original": True
        }
        return result_item

    extracted_json = extract_result["json"]

    # Stage 2: Judge faithfulness
    judge_result = judge_faithfulness(original_report, extracted_json, judge_model)

    if not judge_result["success"]:
        tqdm.write(f"⚠️  Sample {sample_id}: Judge failed - {judge_result['error']}, keeping original")
        result_item['normalization'] = {
            "stage": "judge",
            "success": False,
            "error": judge_result["error"],
            "extracted_json": extracted_json,
            "kept_original": True
        }
        return result_item

    # Check verdict
    if judge_result["verdict"] != "PASS":
        tqdm.write(f"⚠️  Sample {sample_id}: Failed verification - {judge_result['reasoning']}")
        result_item['normalization'] = {
            "stage": "judge",
            "success": False,
            "verdict": judge_result["verdict"],
            "reasoning": judge_result["reasoning"],
            "issues": judge_result["issues"],
            "extracted_json": extracted_json,
            "kept_original": True
        }
        return result_item

    # Stage 3: Generate normalized report
    try:
        normalized_report = json_to_report(extracted_json)

        # Replace the report content
        result_item['conversations'][1]['value'] = normalized_report
        result_item['normalization'] = {
            "stage": "complete",
            "success": True,
            "verdict": "PASS",
            "reasoning": judge_result["reasoning"],
            "extracted_json": extracted_json,
            "original_report": original_report,
            "kept_original": False,
            "extract_model": extract_model,
            "judge_model": judge_model
        }

        return result_item

    except Exception as e:
        tqdm.write(f"❌ Sample {sample_id}: Report generation failed - {str(e)}")
        result_item['normalization'] = {
            "stage": "generation",
            "success": False,
            "error": str(e),
            "extracted_json": extracted_json,
            "kept_original": True
        }
        return result_item


def normalize_dataset(input_file: str, output_file: str,
                      extract_model: str, judge_model: str,
                      concurrency: int = 100, max_retries: int = 5,
                      max_samples: int = None):
    """Normalize all reports in the dataset with verification."""

    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Processing first {max_samples} samples (testing mode)")

    # Add sample_id
    for idx, item in enumerate(dataset):
        item['sample_id'] = idx

    print(f"\n{'='*80}")
    print(f"TWO-STAGE NORMALIZATION WITH VERIFICATION")
    print(f"{'='*80}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Extract model: {extract_model}")
    print(f"Judge model: {judge_model}")
    print(f"Total samples: {len(dataset)}")
    print(f"Concurrency: {concurrency}")
    print(f"Max retries: {max_retries}")
    print(f"{'='*80}\n")

    # Results storage
    results = [None] * len(dataset)
    results_lock = threading.Lock()
    completed_count = 0
    verified_count = 0
    failed_count = 0

    # Process concurrently
    print("Processing reports...\n")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(process_single_sample, item, extract_model, judge_model, max_retries): idx
            for idx, item in enumerate(dataset)
        }

        with tqdm(total=len(dataset), desc="Processing") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()

                    with results_lock:
                        results[idx] = result
                        completed_count += 1

                        if result.get('normalization', {}).get('success', False):
                            verified_count += 1
                        else:
                            failed_count += 1

                        # Save intermediate results
                        if completed_count % 50 == 0:
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x.get('sample_id', 0))

                            for r in valid_results:
                                r.pop('sample_id', None)

                            intermediate_file = output_file.replace('.json', '_intermediate.json')
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)

                            tqdm.write(f"✓ Completed {completed_count}/{len(dataset)} (Verified: {verified_count}, Failed: {failed_count})")

                except Exception as e:
                    with results_lock:
                        result_item = dataset[idx].copy()
                        result_item['normalization'] = {
                            "stage": "unknown",
                            "success": False,
                            "error": f"Future exception: {str(e)}",
                            "kept_original": True
                        }
                        results[idx] = result_item
                        completed_count += 1
                        failed_count += 1

                pbar.update(1)

    # Final processing
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x.get('sample_id', 0))

    for r in results:
        r.pop('sample_id', None)

    # Save final results
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "output_file": output_file,
        "extract_model": extract_model,
        "judge_model": judge_model,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "total_samples": len(dataset),
        "verified_and_normalized": verified_count,
        "failed_verification": failed_count,
        "success_rate": f"{verified_count/len(dataset)*100:.2f}%"
    }

    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("NORMALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Extract Model: {summary['extract_model']}")
    print(f"Judge Model: {summary['judge_model']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Verified & Normalized: {summary['verified_and_normalized']} ({summary['success_rate']})")
    print(f"Failed/Kept Original: {summary['failed_verification']}")
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage report normalization with verification")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help=f"Input JSON file (default: from config.yaml = {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: auto-generated from input and model name)")
    parser.add_argument("--extract_model", type=str, default=EXTRACT_MODEL,
                       help=f"Model for extraction (default: from config.yaml = {EXTRACT_MODEL})")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL,
                       help=f"Model for verification (default: from config.yaml = {JUDGE_MODEL})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                       help=f"Concurrent requests (default: from config.yaml = {DEFAULT_CONCURRENCY})")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                       help=f"Max retries (default: from config.yaml = {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES,
                       help=f"Max samples for testing (default: from config.yaml = {DEFAULT_MAX_SAMPLES or 'all'})")

    args = parser.parse_args()

    # Validate input file
    if not args.input:
        parser.error("--input is required. Please specify input file or set it in config.yaml")

    # Auto-generate output filename if not specified
    if not args.output:
        input_path = Path(args.input)
        # Format: {input_name}_norm_by_{model_name}.json
        model_name = args.extract_model.replace('/', '-').replace(' ', '_')
        output_filename = f"{input_path.stem}_norm_by_{model_name}{input_path.suffix}"
        args.output = str(input_path.parent / output_filename)
        print(f"Output file auto-generated: {args.output}")

    print("\nConfiguration loaded:")
    print(f"  API Base URL: {normalize_api_config['base_url']}")
    print(f"  Auth Mode: {normalize_api_config['auth_mode']}")
    print(f"  Default Model: {normalize_api_config['model']}")
    print(f"  Extract Model: {args.extract_model}")
    print(f"  Judge Model: {args.judge_model}")
    print(f"  Input File: {args.input}")
    print(f"  Output File: {args.output}")
    print()

    normalize_dataset(
        input_file=args.input,
        output_file=args.output,
        extract_model=args.extract_model,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_samples=args.max_samples
    )
