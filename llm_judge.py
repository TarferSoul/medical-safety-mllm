#!/usr/bin/env python3
"""
LLM-as-Judge for Radiology Report Evaluation

Reads dict-keyed JSON results (from mllm_inference.py) and runs LLM-as-judge
evaluation, adding judge_score, judge_reasoning, judge_error_types fields.

Supports:
  - Multi-URL load balancing (URLPool)
  - Resume from existing output (skips already-judged samples)
  - Batch save every 100 samples
  - config.yaml defaults with CLI overrides
"""

import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path

import openai
import yaml
from tqdm import tqdm

save_lock = threading.Lock()


# ---------------------------------------------------------------------------
# URL Pool for load balancing (from mllm_inference.py)
# ---------------------------------------------------------------------------
class URLPool:
    """Thread-safe URL pool for round-robin load balancing."""

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("URL list cannot be empty")
        self.urls = urls
        self._cycle = cycle(urls)
        self._lock = threading.Lock()

    def get_next_url(self) -> str:
        with self._lock:
            return next(self._cycle)

    def __len__(self):
        return len(self.urls)


def create_client(api_url: str, api_key: str = None, api_ak: str = None, api_sk: str = None) -> openai.OpenAI:
    """Create an OpenAI client with either Bearer or AK/SK Basic auth."""
    if api_ak and api_sk:
        b64_auth = base64.b64encode(f"{api_ak}:{api_sk}".encode()).decode()
        return openai.OpenAI(
            base_url=api_url,
            api_key=b64_auth,
            default_headers={"Authorization": f"Basic {b64_auth}"},
        )
    elif api_key:
        return openai.OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
    else:
        raise ValueError("Must provide either --api_key or both --api_ak and --api_sk")


# ---------------------------------------------------------------------------
# Judge prompt (from evaluation_discard/llm_as_judge.py)
# ---------------------------------------------------------------------------
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
- **1.0-4.0:** Major diagnostic error (e.g., Reference says "Pneumonia", Candidate says "Normal"; or Candidate hallucinates "Edema" when lungs are clear)."""


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    if text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


# ---------------------------------------------------------------------------
# Judge a single sample
# ---------------------------------------------------------------------------
def judge_single_sample(
    url_pool: URLPool,
    api_key: str,
    api_ak: str,
    api_sk: str,
    judge_model: str,
    sample_id: str,
    data: dict,
    max_retries: int,
) -> tuple[str, dict]:
    """Judge a single sample. Returns (sample_id, updated_data)."""
    gt_text = data.get("report", "")
    pred_text = strip_think_tags(data.get("model_prediction", ""))

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth=gt_text,
        prediction=pred_text,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            api_url = url_pool.get_next_url()
            client = create_client(api_url, api_key, api_ak, api_sk)

            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10000,
            )

            raw_content = response.choices[0].message.content
            raw_content = strip_think_tags(raw_content)

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"```(?:json)?\s*(.*?)```", raw_content, re.DOTALL)
            if json_match:
                raw_content = json_match.group(1).strip()

            eval_output = json.loads(raw_content)

            if not eval_output:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return sample_id, {
                        **data,
                        "judge_score": 0.0,
                        "judge_reasoning": "Empty response from judge after retries",
                        "judge_error_types": ["Empty Response"],
                        "judge_model": judge_model,
                    }

            return sample_id, {
                **data,
                "judge_score": float(eval_output.get("clinical_accuracy_score", 0.0)),
                "judge_reasoning": eval_output.get("reasoning", ""),
                "judge_error_types": eval_output.get("error_types", []),
                "judge_model": judge_model,
            }

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries - 1:
                tqdm.write(f"  {sample_id}: JSON parse error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)
            else:
                tqdm.write(f"  {sample_id}: JSON parse error after {max_retries} attempts")
                return sample_id, {
                    **data,
                    "judge_score": 0.0,
                    "judge_reasoning": f"JSON parse error: {e}",
                    "judge_error_types": ["JSON Parse Error"],
                    "judge_model": judge_model,
                }

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                tqdm.write(f"  {sample_id}: {type(e).__name__} (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** (attempt + 1))
            else:
                tqdm.write(f"  {sample_id}: {type(e).__name__} after {max_retries} attempts - {e}")
                return sample_id, {
                    **data,
                    "judge_score": 0.0,
                    "judge_reasoning": f"Error: {e}",
                    "judge_error_types": ["Evaluation Error"],
                    "judge_model": judge_model,
                }

    return sample_id, {
        **data,
        "judge_score": 0.0,
        "judge_reasoning": f"Max retries reached: {last_error}",
        "judge_error_types": ["Max Retries Exceeded"],
        "judge_model": judge_model,
    }


# ---------------------------------------------------------------------------
# Run judge on all samples
# ---------------------------------------------------------------------------
def run_judge(
    url_pool: URLPool,
    api_key: str,
    api_ak: str,
    api_sk: str,
    judge_model: str,
    input_file: str,
    output_file: str,
    max_workers: int,
    max_retries: int,
    max_samples: int,
):
    """Load input, resume from existing output, run judge, batch save."""
    with open(input_file, "r") as f:
        input_data = json.load(f)

    # Resume: load existing output if present
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            save_data = json.load(f)
    else:
        save_data = {}

    # Filter to pending (no judge_score yet)
    pending = [
        (sid, data)
        for sid, data in input_data.items()
        if sid not in save_data or "judge_score" not in save_data.get(sid, {})
    ]

    if max_samples > 0:
        pending = pending[:max_samples]

    total = len(input_data)
    done = total - len(pending)
    print(f"Total: {total} | Already judged: {done} | Pending: {len(pending)} | Workers: {max_workers}")

    if not pending:
        print("Nothing to do.")
        # Still print stats from existing data
        print_stats(save_data)
        return

    # Ensure all input data is in save_data (for samples not being re-judged)
    for sid, data in input_data.items():
        if sid not in save_data:
            save_data[sid] = data

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                judge_single_sample,
                url_pool, api_key, api_ak, api_sk,
                judge_model, sid, data, max_retries,
            ): sid
            for sid, data in pending
        }

        save_counter = 0
        save_interval = 100

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            sid, result = future.result()
            with save_lock:
                save_data[sid] = result
                save_counter += 1

                if save_counter % save_interval == 0:
                    with open(output_file, "w") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Final save
        with save_lock:
            with open(output_file, "w") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_file}")
    print_stats(save_data)


def print_stats(data: dict):
    """Print summary statistics for judged samples."""
    scores = [
        v["judge_score"]
        for v in data.values()
        if isinstance(v.get("judge_score"), (int, float)) and v["judge_score"] > 0
    ]
    total = len(data)
    judged = sum(1 for v in data.values() if "judge_score" in v)
    errors = sum(
        1 for v in data.values()
        if v.get("judge_error_types") and v["judge_error_types"] != ["None"]
        and any(t in str(v.get("judge_error_types", []))
                for t in ["Parse Error", "Evaluation Error", "Max Retries", "Empty Response"])
    )

    print("\n" + "=" * 60)
    print("JUDGE STATISTICS")
    print("=" * 60)
    print(f"Total samples:     {total}")
    print(f"Judged:            {judged}")
    if scores:
        print(f"Successfully scored: {len(scores)}")
        print(f"Average score:     {sum(scores) / len(scores):.2f} / 10.0")
        print(f"Min score:         {min(scores):.2f}")
        print(f"Max score:         {max(scores):.2f}")
        print(f"Median score:      {sorted(scores)[len(scores) // 2]:.2f}")
    if errors:
        print(f"Errors:            {errors}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    if not Path(config_path).exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = load_config("config.yaml")
    api_cfg = config.get("api", {})
    judge_cfg = config.get("judge", {})

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge for radiology report evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=judge_cfg.get("input"), help="Input JSON file (dict-keyed results from mllm_inference.py)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: {input_stem}_judged.json)")
    parser.add_argument("--judge_model", type=str, default=judge_cfg.get("model_name", "QwQ"), help="Judge model name")

    # API config (judge.* overrides api.*)
    parser.add_argument("--api_url", type=str, default=judge_cfg.get("url", api_cfg.get("url")), help="Single API URL (legacy)")
    parser.add_argument("--api_urls", nargs="+", default=judge_cfg.get("urls", api_cfg.get("urls")), help="Multiple API URLs for load balancing")
    parser.add_argument("--api_key", type=str, default=None, help="Bearer token")
    parser.add_argument("--api_ak", type=str, default=judge_cfg.get("ak", api_cfg.get("ak")), help="AK for Basic auth")
    parser.add_argument("--api_sk", type=str, default=judge_cfg.get("sk", api_cfg.get("sk")), help="SK for Basic auth")

    # Execution (judge.* overrides execution.*)
    parser.add_argument("--max_workers", type=int, default=judge_cfg.get("max_workers", 100), help="Concurrent workers")
    parser.add_argument("--max_retries", type=int, default=judge_cfg.get("max_retries", 5), help="Max retries per sample")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples to judge, 0=all")

    args = parser.parse_args()

    if not args.input:
        parser.error("--input is required (or set judge.input in config.yaml)")

    # Resolve API URLs
    api_urls = args.api_urls
    if not api_urls and args.api_url:
        api_urls = [args.api_url]
    if not api_urls:
        raise ValueError("Must provide either --api_url or --api_urls (or set in config.yaml)")

    url_pool = URLPool(api_urls)

    # Resolve output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_judged.json")

    # Print config
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Judge model: {args.judge_model}")
    print(f"API URLs:    {len(url_pool)} endpoints")
    print(f"Workers:     {args.max_workers}")
    print(f"Max samples: {args.max_samples or 'all'}")

    run_judge(
        url_pool=url_pool,
        api_key=args.api_key,
        api_ak=args.api_ak,
        api_sk=args.api_sk,
        judge_model=args.judge_model,
        input_file=args.input,
        output_file=args.output,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
