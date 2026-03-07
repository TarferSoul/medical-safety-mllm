#!/usr/bin/env python3
"""
ReXrank MLLM Inference Script

Runs inference on all ReXrank sub-benchmarks (iu_xray, mimic-cxr, chexpert_plus)
using an OpenAI-compatible multimodal API.

Supports two authentication methods:
  1. Bearer token:  --api_key <token>
  2. AK/SK Basic:   --api_ak <ak> --api_sk <sk>

Output format matches ReXrank-metric submission format (JSON with model_prediction field).
"""

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sub-benchmark definitions
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "iu_xray_test": {
        "data_file": "ReXrank/data/iu_xray/ReXRank_IUXray_test.json",
        "img_root_arg": "img_root_iu_xray",
    },
    "iu_xray_valid": {
        "data_file": "ReXrank/data/iu_xray/ReXRank_IUXray_valid.json",
        "img_root_arg": "img_root_iu_xray",
    },
    "mimic-cxr_test": {
        "data_file": "ReXrank/data/mimic-cxr/ReXRank_MIMICCXR_test.json",
        "img_root_arg": "img_root_mimic_cxr",
    },
    "chexpert_plus": {
        "data_file": "ReXrank/data/chexpert_plus/ReXRank_CheXpertPlus.json",
        "img_root_arg": "img_root_chexpert_plus",
    },
    # Training set benchmarks (converted from ShareGPT format)
    "train_sample_100": {
        "data_file": "dataset/cxr_train_for_inference_100.json",
        "img_root_arg": None,  # Uses absolute paths
    },
    "train_sample_1000": {
        "data_file": "dataset/cxr_train_for_inference_1000.json",
        "img_root_arg": None,  # Uses absolute paths
    },
    "train_full": {
        "data_file": "dataset/cxr_train_for_inference_full.json",
        "img_root_arg": None,  # Uses absolute paths
    },
}

save_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
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
# Image / prompt helpers
# ---------------------------------------------------------------------------
def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect MIME type.

    Returns:
        (base64_data, mime_type) tuple
    """
    from PIL import Image

    with open(image_path, "rb") as f:
        img_data = f.read()

    # Detect image format using PIL
    try:
        with Image.open(image_path) as img:
            img_format = img.format.lower() if img.format else "jpeg"
            # Handle common variations
            if img_format == "jpg":
                img_format = "jpeg"
            mime_type = f"image/{img_format}"
    except Exception:
        # Fallback to jpeg if detection fails
        mime_type = "image/jpeg"

    b64_data = base64.b64encode(img_data).decode("utf-8")
    return b64_data, mime_type


def build_prompt(context: str, num_images: int) -> str:
    image_tags = "<image>" * num_images
    return (
        f"{image_tags}"
        "You are a radiology assistant. Given chest X-ray images and clinical context, "
        "generate a structured radiology report with Findings and Impression sections.\n\n"
        "Below is some context to assist your diagnosis:\n"
        f"{context}\n\n"
        "Please provide a detailed radiology report in the following format:\n"
        "Findings: [describe the chest X-ray findings]\n"
        "Impression: [provide the interpretation]\n\n"
        "Generate the report:"
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_api(client: openai.OpenAI, model_name: str, images: list[str], context: str) -> dict:
    """Call API and return full response with reasoning."""
    prompt = build_prompt(context, len(images))

    content = []
    for img_path in images:
        b64_data, mime_type = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
        })
    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=16384,
        temperature=1.0,
    )

    # Return both prediction and reasoning
    message = response.choices[0].message
    return {
        "prediction": message.content,
        "reasoning": getattr(message, "reasoning", None)
    }


# ---------------------------------------------------------------------------
# Per-study worker
# ---------------------------------------------------------------------------
def process_study(
    client: openai.OpenAI,
    model_name: str,
    study_id: str,
    data: dict,
    img_root_dir: str,
    max_retries: int,
    retry_delay: int,
) -> tuple[str, dict]:
    # Handle image paths
    if img_root_dir is None:
        # Use absolute paths directly (for training set)
        image_paths = data["image_path"]
    else:
        # Strip leading '/' to ensure os.path.join works correctly (for test sets)
        image_paths = [os.path.join(img_root_dir, p.lstrip('/')) for p in data["image_path"]]

    context = data.get("context", "")

    for attempt in range(max_retries):
        try:
            result = call_api(client, model_name, image_paths, context)
            return study_id, {
                **data,
                "model_prediction": result["prediction"],
                "model_reasoning": result["reasoning"]
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return study_id, {
                    **data,
                    "model_prediction": f"ERROR: {str(e)}",
                    "model_reasoning": None
                }

    return study_id, {
        **data,
        "model_prediction": "ERROR: Unknown",
        "model_reasoning": None
    }


# ---------------------------------------------------------------------------
# Run inference on a single benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    client: openai.OpenAI,
    model_name: str,
    bench_name: str,
    data_file: str,
    img_root_dir: str,
    output_dir: str,
    max_workers: int,
    max_retries: int,
    retry_delay: int,
    max_samples: int,
) -> str:
    with open(data_file, "r") as f:
        input_data = json.load(f)

    save_json_file = os.path.join(output_dir, f"{bench_name}.json")
    os.makedirs(output_dir, exist_ok=True)

    # Resume: load existing results
    if os.path.exists(save_json_file):
        with open(save_json_file, "r") as f:
            save_data = json.load(f)
    else:
        save_data = {}

    # Filter to pending tasks
    pending = [
        (sid, data)
        for sid, data in input_data.items()
        if sid not in save_data or not save_data.get(sid, {}).get("model_prediction")
        or str(save_data.get(sid, {}).get("model_prediction", "")).startswith("ERROR:")
    ]

    if max_samples > 0:
        pending = pending[:max_samples]

    total = len(input_data)
    done = total - len(pending)
    print(f"\n[{bench_name}] Total: {total} | Already done: {done} | Pending: {len(pending)} | Workers: {max_workers}")

    if not pending:
        print(f"[{bench_name}] Nothing to do, skipping.")
        return save_json_file

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_study,
                client,
                model_name,
                sid,
                data,
                img_root_dir,
                max_retries,
                retry_delay,
            ): sid
            for sid, data in pending
        }

        # Batch save every 100 samples
        save_counter = 0
        save_interval = 100

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{bench_name}]"):
            sid, result = future.result()
            with save_lock:
                save_data[sid] = result
                save_counter += 1

                # Save every 100 samples
                if save_counter % save_interval == 0:
                    with open(save_json_file, "w") as f:
                        json.dump(save_data, f, indent=4, ensure_ascii=False)

        # Final save for remaining samples
        with save_lock:
            with open(save_json_file, "w") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f"[{bench_name}] Saved to {save_json_file}")
    return save_json_file


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load config first
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser(
        description="ReXrank MLLM Inference — evaluate all sub-benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Auth examples:\n"
            "  Bearer token:  --api_key sk-xxx\n"
            "  AK/SK Basic:   --api_ak <ak> --api_sk <sk>\n"
            "\n"
            "Config file:\n"
            "  Copy config.yaml.example to config.yaml and fill in credentials.\n"
            "  Command-line arguments override config file values.\n"
        ),
    )
    # API
    parser.add_argument("--api_url", type=str, default=config.get("api", {}).get("url"), help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", type=str, default=None, help="Bearer token for auth")
    parser.add_argument("--api_ak", type=str, default=config.get("api", {}).get("ak"), help="Access key for AK/SK Basic auth")
    parser.add_argument("--api_sk", type=str, default=config.get("api", {}).get("sk"), help="Secret key for AK/SK Basic auth")
    parser.add_argument("--model_name", type=str, default=config.get("api", {}).get("model_name", "Qwen3-4B"), help="Model name to call")

    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARKS.keys()),
        choices=list(BENCHMARKS.keys()),
        help="Sub-benchmarks to evaluate (default: all)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Root directory containing ReXrank/data/ (default: script directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/<model_name>/)",
    )

    # Per-benchmark image root directories (absolute paths)
    parser.add_argument(
        "--img_root_iu_xray",
        type=str,
        required=False,
        default=config.get("image_paths", {}).get("iu_xray"),
        help="Absolute path to IU X-ray images directory (should point to images/ subdirectory)",
    )
    parser.add_argument(
        "--img_root_mimic_cxr",
        type=str,
        required=False,
        default=config.get("image_paths", {}).get("mimic_cxr"),
        help="Absolute path to MIMIC-CXR images directory (should point to files/ subdirectory)",
    )
    parser.add_argument(
        "--img_root_chexpert_plus",
        type=str,
        required=False,
        default=config.get("image_paths", {}).get("chexpert_plus"),
        help="Absolute path to CheXpert Plus images directory (should point to PNG/ subdirectory)",
    )

    # Execution
    parser.add_argument("--max_workers", type=int, default=config.get("execution", {}).get("max_workers", 100), help="Concurrent workers (default: 100)")
    parser.add_argument("--max_retries", type=int, default=config.get("execution", {}).get("max_retries", 3), help="Retries per sample (default: 3)")
    parser.add_argument("--retry_delay", type=int, default=config.get("execution", {}).get("retry_delay", 5), help="Base retry delay in seconds (default: 5)")
    parser.add_argument("--max_samples", type=int, default=config.get("execution", {}).get("max_samples", 0), help="Max samples per benchmark, 0=all (default: 0)")

    args = parser.parse_args()

    # Resolve output dir
    output_dir = args.output_dir or os.path.join("results", args.model_name)

    # Create client
    client = create_client(args.api_url, args.api_key, args.api_ak, args.api_sk)

    # Verify connectivity
    print(f"API URL:  {args.api_url}")
    print(f"Model:    {args.model_name}")
    print(f"Auth:     {'AK/SK Basic' if args.api_ak else 'Bearer token'}")
    print(f"Output:   {output_dir}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")

    results = {}
    for bench_name in args.benchmarks:
        bench = BENCHMARKS[bench_name]
        data_file = os.path.join(args.data_root, bench["data_file"])

        if not os.path.exists(data_file):
            print(f"\n[{bench_name}] Data file not found: {data_file}, skipping.")
            continue

        # Handle image root directory
        img_root_arg = bench["img_root_arg"]
        if img_root_arg is None:
            # Training set uses absolute paths
            img_root_dir = None
        else:
            # Test sets use relative paths
            img_root_dir = getattr(args, img_root_arg, None)
            if not img_root_dir:
                print(f"\n[{bench_name}] No --{img_root_arg} provided, skipping.")
                continue
            if not os.path.isdir(img_root_dir):
                print(f"\n[{bench_name}] Image root not found: {img_root_dir}, skipping.")
                continue

        out_file = run_benchmark(
            client=client,
            model_name=args.model_name,
            bench_name=bench_name,
            data_file=data_file,
            img_root_dir=img_root_dir,
            output_dir=output_dir,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_samples=args.max_samples,
        )
        results[bench_name] = out_file

    print("\n" + "=" * 60)
    print("Done. Output files:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
