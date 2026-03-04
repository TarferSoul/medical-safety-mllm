import json
import base64
import argparse
import threading
import time
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- API Configuration (same as test_api.py) ---
PREFIX = "rjob-37bd26a61a52bebe-f61eb500bddbbf61"
PORT = 19114
BASE_URL = "https://h.pjlab.org.cn/kapi/workspace.kubebrain.io/ailab-ai4good1/" + PREFIX + "-0.xieyuejin/" + str(PORT) + "/v1/"
API_AK = "fb4b11bcc25b0fd8ac2bdad43aff3692"
API_SK = "1be0a8ad0270381b02108d07ba05ce80"
JUDGE_MODEL = "QwQ"  # Judge model name


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


# Initialize client with authentication
auth_string = f"{API_AK}:{API_SK}"
b64_auth_string = base64.b64encode(auth_string.encode()).decode()

client = OpenAI(
    base_url=BASE_URL,
    api_key=b64_auth_string,
    default_headers={"Authorization": f"Basic {b64_auth_string}"}
)

# 定义 Prompt 模板
PROMPT_TEMPLATE = """You are an expert Radiologist and Medical Evaluator. Your task is to evaluate the quality of a generated chest X-ray report (Candidate) by comparing it against the expert-written Ground Truth report (Reference).

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


def evaluate_single_sample(item: dict, judge_model: str, max_retries: int = 5) -> dict:
    """Evaluate a single sample using LLM as judge with retry mechanism."""
    # 提取 GT 和 Pred
    gt_text = item['ground_truth']

    # 注意：你的 prediction 包含 <think> 标签，我们最好去掉它，只评估最终报告
    pred_text = item.get('prediction', '')
    if pred_text and "</think>" in pred_text:
        pred_text = pred_text.split("</think>")[-1].strip()

    # 格式化 Prompt
    final_prompt = PROMPT_TEMPLATE.format(
        ground_truth=gt_text,
        prediction=pred_text
    )

    last_error = None

    # 重试循环
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": final_prompt}
                ],
                # response_format={"type": "json_object"}, # 某些模型可能不支持此参数
                temperature=0.0,
                max_tokens=10000
            )

            # 获取原始响应内容
            raw_content = response.choices[0].message.content

            # 如果响应包含 <think> 标签，提取实际内容
            if "</think>" in raw_content:
                raw_content = raw_content.split("</think>")[-1].strip()

            # 尝试解析 JSON
            evaluation_output = json.loads(raw_content)

            # 检查是否为空响应
            if not evaluation_output or len(evaluation_output) == 0:
                if attempt < max_retries - 1:
                    tqdm.write(f"⚠️  Sample {item.get('sample_id', 'unknown')} got empty response (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                else:
                    tqdm.write(f"⚠️  Sample {item.get('sample_id', 'unknown')} got empty response after {max_retries} attempts")
                    tqdm.write(f"Raw content: {raw_content[:200]}...")
                    evaluation_output = {
                        "error": "Empty response from API after retries",
                        "reasoning": "API returned empty JSON",
                        "error_types": ["Empty Response"],
                        "clinical_accuracy_score": 0.0,
                        "raw_response": raw_content,
                        "attempts": max_retries
                    }

            # 将评估结果合并回原始数据，使用新格式
            result_item = item.copy()
            result_item['eval_results'] = {
                "eval_input": final_prompt,
                "eval_output": evaluation_output
            }
            result_item['judge_model'] = judge_model
            result_item['attempts'] = attempt + 1  # 记录实际尝试次数

            return result_item

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries - 1:
                tqdm.write(f"❌ Sample {item.get('sample_id', 'unknown')}: JSON decode error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                tqdm.write(f"❌ Sample {item.get('sample_id', 'unknown')}: JSON decode error after {max_retries} attempts - {e}")
                tqdm.write(f"Raw response: {response.choices[0].message.content[:200]}...")
                result_item = item.copy()
                result_item['eval_results'] = {
                    "eval_input": final_prompt,
                    "eval_output": {
                        "error": f"JSON decode error after {max_retries} attempts: {str(e)}",
                        "reasoning": "Failed to parse API response as JSON",
                        "error_types": ["JSON Parse Error"],
                        "clinical_accuracy_score": 0.0,
                        "raw_response": response.choices[0].message.content[:500],
                        "attempts": max_retries
                    }
                }
                result_item['judge_model'] = judge_model
                result_item['attempts'] = max_retries
                return result_item

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                tqdm.write(f"❌ Sample {item.get('sample_id', 'unknown')}: {type(e).__name__} (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                tqdm.write(f"❌ Sample {item.get('sample_id', 'unknown')}: {type(e).__name__} after {max_retries} attempts - {e}")
                result_item = item.copy()
                result_item['eval_results'] = {
                    "eval_input": final_prompt,
                    "eval_output": {
                        "error": f"Evaluation failed after {max_retries} attempts: {str(e)}",
                        "reasoning": "Evaluation failed",
                        "error_types": ["Evaluation Error"],
                        "clinical_accuracy_score": 0.0,
                        "attempts": max_retries
                    }
                }
                result_item['judge_model'] = judge_model
                result_item['attempts'] = max_retries
                return result_item

    # 不应该到达这里，但作为保险
    result_item = item.copy()
    result_item['eval_results'] = {
        "eval_input": final_prompt,
        "eval_output": {
            "error": f"Max retries reached: {str(last_error)}",
            "reasoning": "Evaluation failed",
            "error_types": ["Max Retries Exceeded"],
            "clinical_accuracy_score": 0.0,
            "attempts": max_retries
        }
    }
    result_item['judge_model'] = judge_model
    result_item['attempts'] = max_retries
    return result_item


def evaluate_results(input_file: str, output_file: str, judge_model: str, concurrency: int = 20, max_retries: int = 5):
    """Evaluate all samples in the input file with concurrent processing."""
    # 读取评估结果数据
    print(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"开始评估 {len(dataset)} 个样本，使用模型: {judge_model}")
    print(f"并发数: {concurrency}")
    print(f"最大重试次数: {max_retries}")

    # 准备结果存储（线程安全）
    results = [None] * len(dataset)
    results_lock = threading.Lock()
    scores = []
    scores_lock = threading.Lock()
    completed_count = 0

    # 并发处理
    print("\n评估中...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有任务，使用 enumerate 获取索引
        future_to_idx = {
            executor.submit(evaluate_single_sample, item, judge_model, max_retries): idx
            for idx, item in enumerate(dataset)
        }

        # 处理完成的任务
        with tqdm(total=len(dataset), desc="Evaluating") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()

                    with results_lock:
                        results[idx] = result
                        completed_count += 1

                        # 收集分数用于统计
                        if 'eval_results' in result and 'eval_output' in result['eval_results']:
                            eval_output = result['eval_results']['eval_output']
                            if 'clinical_accuracy_score' in eval_output:
                                score = eval_output['clinical_accuracy_score']
                                if isinstance(score, (int, float)) and score > 0:
                                    with scores_lock:
                                        scores.append(score)

                        # 每50个样本保存一次中间结果
                        if completed_count % 50 == 0:
                            valid_results = [r for r in results if r is not None]
                            valid_results.sort(key=lambda x: x.get('sample_id', 0))

                            intermediate_file = output_file.replace('.json', '_intermediate.json')
                            with open(intermediate_file, 'w', encoding='utf-8') as f:
                                json.dump(valid_results, f, indent=2, ensure_ascii=False)
                            tqdm.write(f"已完成 {completed_count}/{len(dataset)} 个样本，中间结果已保存")

                except Exception as e:
                    with results_lock:
                        results[idx] = {
                            "sample_id": idx,
                            "eval_results": {
                                "eval_input": "",
                                "eval_output": {
                                    "error": f"Future exception: {str(e)}",
                                    "reasoning": "Evaluation failed",
                                    "error_types": ["Evaluation Error"],
                                    "clinical_accuracy_score": 0.0
                                }
                            },
                            "judge_model": judge_model
                        }
                        completed_count += 1

                pbar.update(1)

    # 过滤 None 并排序
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x.get('sample_id', 0))

    # 保存结果
    print(f"\n保存结果到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 计算统计数据
    if scores:
        avg_score = sum(scores) / len(scores)
        print("\n" + "=" * 60)
        print("EVALUATION STATISTICS")
        print("=" * 60)
        print(f"Model: {judge_model}")
        print(f"Concurrency: {concurrency}")
        print(f"Total samples: {len(dataset)}")
        print(f"Successfully scored: {len(scores)}")
        print(f"Average clinical accuracy score: {avg_score:.2f}/10.0")
        print(f"Min score: {min(scores):.2f}")
        print(f"Max score: {max(scores):.2f}")
        print("=" * 60)
    else:
        print("\n⚠️  No valid scores collected")

    print(f"\n评估完成，结果已保存至 {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM as Judge for Medical Report Evaluation")
    parser.add_argument(
        "--input",
        type=str,
        # required=True,
        default="evaluation_results/evaluation_results_20251227_134405.json",
        help="Input JSON file with predictions and ground truth"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file with evaluations (auto-generated if not specified)"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL,
        help=f"Judge model name (default: {JUDGE_MODEL})"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Number of concurrent API calls (default: 100)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries for failed API calls (default: 5)"
    )

    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        input_path = Path(args.input)
        input_stem = input_path.stem
        judge_model_clean = "judge_" + sanitize_model_name(args.judge_model)
        output_filename = f"{input_stem}_with_{judge_model_clean}.json"
        args.output = str(input_path.parent / output_filename)

    evaluate_results(
        input_file=args.input,
        output_file=args.output,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        max_retries=args.max_retries
    )
