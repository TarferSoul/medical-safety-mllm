import argparse
import json
import os
import base64
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(context, num_images):
    image_tokens = "".join([f"Image {i}: <image>\n" for i in range(num_images)])

    prompt = f"""You are a radiology assistant. Given chest X-ray images and clinical context, generate a structured radiology report with Findings and Impression sections.

{image_tokens}Clinical Context: {context}

Please provide a detailed radiology report in the following format:
Findings: [describe the chest X-ray findings]
Impression: [provide the interpretation]

Generate the report:"""

    return prompt


def call_multimodal_api(api_url, api_key, images, context, model_name="gpt-4v"):
    prompt = build_prompt(context, len(images))

    messages = [{"role": "user", "content": []}]

    for img_path in images:
        base64_img = encode_image(img_path)
        if "gpt" in model_name.lower() or "openai" in api_url.lower():
            image_url = f"data:image/jpeg;base64,{base64_img}"
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        elif "claude" in model_name.lower():
            messages[0]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_img,
                    },
                }
            )
        else:
            image_url = f"data:image/jpeg;base64,{base64_img}"
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

    messages[0]["content"].append({"type": "text", "text": prompt})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 1.0,
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    content = ""
    if "gpt" in model_name.lower() or "openai" in api_url.lower():
        content = result["choices"][0]["message"]["content"]
    elif "claude" in model_name.lower():
        content = result["content"][0]["text"]
    elif "qwen" in model_name.lower():
        content = result["choices"][0]["message"]["content"]
    else:
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    if isinstance(content, list):
        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        content = " ".join(text_parts) if text_parts else str(content)

    return content


save_lock = threading.Lock()


def process_study(
    study_id,
    data,
    api_url,
    api_key,
    img_root_dir,
    model_name,
    max_retries,
    retry_delay,
):
    image_paths = [os.path.join(img_root_dir, p) for p in data["image_path"]]
    context = data.get("context", "")

    for attempt in range(max_retries):
        try:
            prediction = call_multimodal_api(
                api_url, api_key, image_paths, context, model_name
            )
            return study_id, {**data, "model_prediction": prediction}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return study_id, {**data, "model_prediction": f"ERROR: {str(e)}"}

    return study_id, {**data, "model_prediction": "ERROR: Unknown"}


def run_inference(
    api_url,
    api_key,
    input_json_file,
    img_root_dir,
    save_json_file,
    model_name,
    max_workers=25,
    max_retries=3,
    retry_delay=5,
):
    with open(input_json_file, "r") as f:
        input_data = json.load(f)

    if os.path.exists(save_json_file):
        with open(save_json_file, "r") as f:
            save_data = json.load(f)
    else:
        save_data = {}

    os.makedirs(os.path.dirname(save_json_file), exist_ok=True)

    pending_tasks = [
        (study_id, data)
        for study_id, data in input_data.items()
        if study_id not in save_data
        or not save_data.get(study_id, {}).get("model_prediction")
    ]

    print(
        f"Total tasks: {len(input_data)}, Pending: {len(pending_tasks)}, Workers: {max_workers}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_study,
                study_id,
                data,
                api_url,
                api_key,
                img_root_dir,
                model_name,
                max_retries,
                retry_delay,
            ): study_id
            for study_id, data in pending_tasks
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            study_id, result = future.result()
            with save_lock:
                save_data[study_id] = result
                with open(save_json_file, "w") as f:
                    json.dump(save_data, f, indent=4)

    print(f"Results saved to {save_json_file}")
    return save_data


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal LLM for Radiology Report Generation"
    )
    parser.add_argument("--api_url", type=str, required=True, help="API endpoint URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--input_json_file",
        type=str,
        default="data/iu_xray/ReXRank_IUXray_valid.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--img_root_dir",
        type=str,
        default="/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/iu_xray/images",
        help="Image root directory",
    )
    parser.add_argument(
        "--save_json_file",
        type=str,
        default="results/iu_xray/mllm_report.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--max_workers", type=int, default=25, help="Number of concurrent workers"
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=int, default=5)

    args = parser.parse_args()

    run_inference(
        args.api_url,
        args.api_key,
        args.input_json_file,
        args.img_root_dir,
        args.save_json_file,
        args.model_name,
        args.max_workers,
        args.max_retries,
        args.retry_delay,
    )


if __name__ == "__main__":
    main()
