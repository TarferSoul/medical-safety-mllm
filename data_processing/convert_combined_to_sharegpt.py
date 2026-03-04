import argparse
import json
import os
import re
import multiprocessing as mp
from tqdm import tqdm


def capitalize_sentences(text):
    if not text:
        return text

    text = text.lower()
    sentences = re.split(r"([.!?]+\s*)", text)

    result = []
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:
            part = part[0].upper() + part[1:] if len(part) > 0 else part
        result.append(part)

    return "".join(result)


def clean_text(text):
    if not text:
        return text
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_context(sections, dataset_name):
    if sections is None:
        return ""

    context_parts = []

    if dataset_name == "chexpertplus":
        if sections.get("narrative"):
            context_parts.append(f"Examination: {sections['narrative']}")
        if sections.get("history"):
            context_parts.append(f"History: {sections['history']}")
        if sections.get("comparison"):
            context_parts.append(f"Comparison: {sections['comparison']}")

    elif dataset_name == "mimic_cxr":
        if sections.get("examination"):
            context_parts.append(f"Examination: {sections['examination']}")
        if sections.get("indication"):
            context_parts.append(f"Indication: {sections['indication']}")
        if sections.get("technique"):
            context_parts.append(f"Technique: {sections['technique']}")
        if sections.get("comparison"):
            context_parts.append(f"Comparison: {sections['comparison']}")

    elif dataset_name == "rexgradient":
        if sections.get("indication"):
            context_parts.append(f"Indication: {sections['indication']}")
        if sections.get("comparison"):
            context_parts.append(f"Comparison: {sections['comparison']}")

    if context_parts:
        return "Below is some context to assist your diagnosis:\n" + "\n".join(
            context_parts
        )
    return ""


def build_prompt_with_findings(num_images, context):
    image_tokens = "".join([f"<image>" for _ in range(num_images)])

    return f"""{image_tokens}You are a radiology assistant. Given chest X-ray images and clinical context, generate a structured radiology report with Findings and Impression sections.

{context}

Please provide a detailed radiology report in the following format:
Findings: [describe the chest X-ray findings]
Impression: [provide the interpretation]

Generate the report:"""


def build_prompt_without_findings(num_images, context):
    image_tokens = "".join([f"<image>" for _ in range(num_images)])

    return f"""{image_tokens}You are a radiology assistant. Given chest X-ray images and clinical context, generate a structured radiology report with Impression section.

{context}

Please provide a detailed radiology report in the following format:
Impression: [provide the interpretation]

Generate the report:"""


def process_item(item):
    image_paths = item.get("image_paths", [])
    if not image_paths:
        return None

    dataset_name = item.get("dataset", "unknown")
    report = item.get("report")
    sections = item.get("sections") or {}

    has_findings = False
    findings = ""
    impression = ""

    if dataset_name == "rexgradient":
        findings = sections.get("findings", "")
        impression = sections.get("impression", "")
        has_findings = bool(findings)
    elif dataset_name == "mimic_cxr":
        findings = sections.get("findings", "")
        impression = sections.get("impression", "")
        has_findings = bool(findings)
    elif dataset_name == "chexpertplus":
        findings = sections.get("findings", "")
        impression = sections.get("impression", "")
        has_findings = bool(findings)

    if dataset_name == "iu_xray":
        if report:
            has_findings = True
            findings = capitalize_sentences(report.strip())
            impression = ""
        else:
            return None

    if not findings and not impression:
        return None

    context = extract_context(sections, dataset_name)

    if has_findings:
        prompt = build_prompt_with_findings(len(image_paths), context)
        findings_clean = (
            clean_text(capitalize_sentences(findings)) if findings else "None"
        )
        impression_clean = (
            clean_text(capitalize_sentences(impression)) if impression else "None"
        )
        report_str = f"Findings: {findings_clean}\nImpression: {impression_clean}"
    else:
        prompt = build_prompt_without_findings(len(image_paths), context)
        impression_clean = (
            clean_text(capitalize_sentences(impression)) if impression else "None"
        )
        report_str = f"Impression: {impression_clean}"

    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": report_str},
        ],
        "images": image_paths,
        "_dataset": dataset_name,
    }


def convert_to_sharegpt(input_file, output_file, num_workers=8):
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items. Processing with {num_workers} workers...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    stats = {
        "chexpertplus": 0,
        "mimic_cxr": 0,
        "iu_xray": 0,
        "rexgradient": 0,
        "skipped": 0,
    }

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_item, data, chunksize=1000),
                total=len(data),
                desc="Converting",
            )
        )

    sharegpt_data = []
    for result in results:
        if result is None:
            stats["skipped"] += 1
        else:
            dataset_name = result.pop("_dataset", "unknown")
            if dataset_name in stats:
                stats[dataset_name] += 1
            sharegpt_data.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)

    print(f"\nConversion complete:")
    print(f"  Total samples: {len(sharegpt_data)}")
    for ds, count in stats.items():
        if count > 0:
            print(f"  {ds}: {count}")
    print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert combined medical imaging dataset to ShareGPT format"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/train.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../dataset/medical_xray_train.json",
        help="Output ShareGPT format JSON file path",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes",
    )

    args = parser.parse_args()
    convert_to_sharegpt(args.input_file, args.output_file, args.num_workers)


if __name__ == "__main__":
    main()
