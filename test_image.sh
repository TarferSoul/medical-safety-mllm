#!/bin/bash
# 使用 Python 生成 JSON，避免 base64 转义问题
python3 << 'PYTHON_SCRIPT'
import json
import base64
import requests

img_path = "/mnt/shared-storage-user/ai4good1-share/xieyuejin/datasets/mimic-cxr/files/p10/p10046166/s50051329/427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5.jpg"

with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# 使用 mllm_inference.py 中的 prompt 格式
context = "Age:50-60.Gender:M.Indication: Evaluate for interval change in patient with metastatic melanoma, presenting with confusion and somnolence. Evaluate for acute cardiopulmonary process."

prompt = f"""<image>You are a radiology assistant. Given chest X-ray images and clinical context, generate a structured radiology report with Findings and Impression sections.

Below is some context to assist your diagnosis:
{context}

Please provide a detailed radiology report in the following format:
Findings: [describe the chest X-ray findings]
Impression: [provide the interpretation]

Generate the report:"""

payload = {
    "model": "Qwen3.5-397B-A17B",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ],
    "max_tokens": 2048,
    "temperature": 1.0
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Basic ZmI0YjExYmNjMjViMGZkOGFjMmJkYWQ0M2FmZjM2OTI6MWJlMGE4YWQwMjcwMzgxYjAyMTA4ZDA3YmEwNWNlODA="
}

print("Sending request...")
response = requests.post(
    "http://s-20260226142108-tvkq9.ailab-ai4good1.pjh-service.org.cn/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=120
)

print(json.dumps(response.json(), indent=2))
PYTHON_SCRIPT
