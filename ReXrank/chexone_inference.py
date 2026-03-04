import argparse
import json
import os
import time
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def generate_report(
    model, processor, image_paths, context, device="cuda", max_new_tokens=1024
):
    """
    Generate radiology report using CheXOne model

    Args:
        model: Qwen2_5_VLForConditionalGeneration model
        processor: AutoProcessor
        image_paths: list of image file paths
        context: clinical context string
        device: device to run on
        max_new_tokens: max tokens to generate

    Returns:
        generated report text
    """
    # Build message content with images
    content = []
    for img_path in image_paths:
        content.append(
            {
                "type": "image",
                "image": img_path,
            }
        )

    # Add prompt with context
    prompt_text = f"Given the clinical context: {context}\n\nGenerate a detailed radiology report with Findings and Impression sections."
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""


def main():
    parser = argparse.ArgumentParser(
        description="Generate radiology reports using CheXOne"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file with image paths and context",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON file for predictions",
    )
    parser.add_argument(
        "--img_root_dir",
        type=str,
        default="/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/iu_xray/images",
        help="Root directory for images",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="StanfordAIMI/CheXOne",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent workers (default 50)",
    )

    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model {args.model_name}...")
    # Check if local path
    is_local = os.path.exists(args.model_name)
    local_files_only = is_local

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if args.device == "cuda" else None,
        local_files_only=local_files_only,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()
    model_lock = threading.Lock()

    processor = AutoProcessor.from_pretrained(
        args.model_name, local_files_only=local_files_only
    )
    print("Model loaded successfully")

    # Load input data
    with open(args.input_json, "r") as f:
        input_data = json.load(f)

    # Load existing predictions if any
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions)} existing predictions")
    else:
        predictions = {}

    # Filter pending studies
    pending_studies = [(k, v) for k, v in input_data.items() if k not in predictions]
    print(
        f"Processing {len(pending_studies)} remaining studies with concurrency={args.concurrency}..."
    )

    def process_single(args_tuple):
        study_id, data = args_tuple
        try:
            # Get image paths
            image_paths = data.get("image_path", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            image_paths = [os.path.join(args.img_root_dir, p) for p in image_paths]

            # Get context
            context = data.get("context", "")

            # Generate report with lock
            with model_lock:
                report = generate_report(
                    model,
                    processor,
                    image_paths,
                    context,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )

            return study_id, {"model_prediction": report}, None
        except Exception as e:
            return study_id, None, str(e)

    # Process with concurrency
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_single, item): item[0] for item in pending_studies
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating reports"
        ):
            study_id = futures[future]
            try:
                _, result, error = future.result()
                if error:
                    print(f"\nError processing {study_id}: {error}")
                else:
                    predictions[study_id] = {**input_data[study_id], **result}
                    # Save after each prediction
                    with open(args.output_json, "w") as f:
                        json.dump(predictions, f, indent=4)
            except Exception as e:
                print(f"\nException for {study_id}: {e}")

    print(f"\nDone! Predictions saved to {args.output_json}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
