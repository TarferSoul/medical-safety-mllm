#!/usr/bin/env python3
import openai
import base64
import time
import sys

# --- 配置信息 ---
# 您的代理服务器地址
BASE_URL = "https://h.pjlab.org.cn/kapi/workspace.kubebrain.io/ailab-safevlagent/rjob-5bfad9016ad50e4f-f61eb500bddbbf61-0.xieyuejin/8000/v1/"
API_AK = "fb4b11bcc25b0fd8ac2bdad43aff3692"
API_SK = "1be0a8ad0270381b02108d07ba05ce80"
MODEL = "Qwen3-VL-235B-A22B-Thinking"  # Judge model name



# --- 客户端初始化 ---
auth_string = f"{API_AK}:{API_SK}"
b64_auth_string = base64.b64encode(auth_string.encode()).decode()

try:
    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key=b64_auth_string, # 占位符
        default_headers={"Authorization": f"Basic {b64_auth_string}"}
    )
except ImportError:
    print("错误：'openai' 库未安装。请运行 'pip install openai'。")
    sys.exit(1)


def check_available_models(client):
    """
    列出当前端口下的所有模型，并检查目标模型是否存在
    """
    print("=" * 60)
    print("🔍 正在获取可用模型列表...")
    
    try:
        # 调用 /v1/models 接口
        models_page = client.models.list()
        
        available_models = []
        print(f"\n当前端口 ({BASE_URL}) 挂载了以下模型:")
        print("-" * 40)
        
        for i, model_obj in enumerate(models_page):
            # 获取模型ID
            m_id = model_obj.id
            available_models.append(m_id)
            print(f"{i+1}. {m_id}")
            
        print("-" * 40)

        # 检查目标模型是否存在
        if MODEL in available_models:
            print(f"✅ 目标模型 '{MODEL}' 在列表中，可以进行测试。")
            return True
        else:
            print(f"⚠️  警告: 目标模型 '{MODEL}' 未在列表中找到！")
            print(f"    这可能会导致接下来的调用报错 'model not found'。")
            print(f"    请检查是否需要更换 MODEL 变量的值。")
            return False

    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return False


def test_api_with_openai_lib(client):
    """
    使用 openai 库测试对话功能
    """
    print("\n" + "=" * 60)
    print(f"🚀 开始测试对话生成 (模型: {MODEL})...")

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "You are an expert Radiologist and Medical Evaluator. Your task is to evaluate the quality of a generated chest X-ray report (Candidate) by comparing it against the expert-written Ground Truth report (Reference).\n\n### Evaluation Criteria:\n1. **Clinical Accuracy (Most Important):** Does the Candidate identify the same pathologies (e.g., pneumonia, effusion, pneumothorax, edema) as the Reference?\n2. **Negation correctness:** specific attention to \"No\", \"Not\", \"Free of\". Confusing \"No pneumothorax\" with \"Small pneumothorax\" is a critical error.\n3. **Anatomical Correctness:** Are the findings localized correctly (e.g., Left Lower Lobe vs Right Upper Lobe)?\n4. **Omissions vs Hallucinations:**\n   - **Omission:** The Reference reports a disease, but the Candidate misses it.\n   - **Hallucination:** The Reference says normal, but the Candidate invents a disease.\n\n### Instructions:\n- Ignore formatting differences (e.g., \"Findings:\" vs \"FINDINGS\").\n- Ignore the \"INDICATION\" and \"COMPARISON\" sections unless they contain critical diagnostic findings not present elsewhere. Focus heavily on \"FINDINGS\" and \"IMPRESSION\".\n- Synonyms are accepted (e.g., \"Opacities\" ≈ \"Infiltrates\" ≈ \"Consolidation\" in specific contexts; \"Cardiomegaly\" ≈ \"Enlarged heart\").\n\n### Input Data:\n**[Ground Truth Report]:**\nFINAL REPORT\n EXAMINATION:  CHEST (PORTABLE AP)\n \n INDICATION:  ___ year old man with IPH, now with fevers, also please check OGT\n placement  // interval change and OGT placement      interval change and OGT\n placement\n \n IMPRESSION: \n \n In comparison with the study of ___, there again are low lung volumes,\n chronic severe enlargement of the cardiac silhouette, pulmonary vascular\n congestion, and bilateral layering pleural effusions with compressive\n atelectasis at the bases.  Endotracheal tube and left subclavian catheter\n remain in standard position.  The nasogastric tube extends at least to the mid\n to lower body of the stomach, were crosses the lower margin of the image.\n\n**[Candidate Report]:**\nFINAL REPORT\n EXAMINATION:  CHEST (PORTABLE AP)\n \n INDICATION:  ___ year old man with cirrhosis, ascites, now with hypoxia  //\n interval change\n \n COMPARISON:  ___.\n \n IMPRESSION: \n \n As compared to the previous radiograph, the lung volumes have substantially\n decreased.  As a consequence, the size of the cardiac silhouette has\n increased, with mild pulmonary edema.  No larger pleural effusions.  No\n pneumonia.\n\n### Output Format:\nReturn a JSON object with the following structure:\n{\n  \"reasoning\": \"Step-by-step comparison of findings...\",\n  \"error_types\": [\"None\" | \"Omission\" | \"Hallucination\" | \"Localization Error\" | \"Severity Error\"],\n  \"clinical_accuracy_score\": <float 0.0 to 10.0>\n}\n\n**Scoring Rubric:**\n- **10.0:** Perfect clinical alignment. Only stylistic differences.\n- **8.0-9.0:** Minor discrepancies (e.g., missed a minor scar, or slight severity difference), but main diagnosis is correct.\n- **5.0-7.0:** Correctly identifies normal vs abnormal, but misses specific details (e.g., wrong lobe) or hallucinates a minor finding.\n- **1.0-4.0:** Major diagnostic error (e.g., Reference says \"Pneumonia\", Candidate says \"Normal\"; or Candidate hallucinates \"Edema\" when lungs are clear).\n"}
            ],
            max_tokens=16384,
            temperature=0.7,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        
        end_time = time.time()
        
        # 提取内容
        if response.choices:
            content = response.choices[0].message.content
            print(f"\n🤖 模型回复:\n{content}")
            
            # 简单的验证逻辑
            if "测试成功" in content or "成功" in content:
                print(f"\n✅ 测试通过! (耗时: {end_time - start_time:.3f}s)")
            else:
                print(f"\n⚠️  调用成功，但回复内容需人工确认 (耗时: {end_time - start_time:.3f}s)")
        else:
            print("⚠️  未返回 choices 数据")

    except openai.APIStatusError as e:
        print(f"\n❌ API 错误 (状态码 {e.status_code}):")
        print(e.response.text)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

    print("=" * 60)


if __name__ == "__main__":
    # 1. 先检查模型列表
    model_exists = check_available_models(client)
    
    # 2. 如果获取列表没有报错（即使模型名不匹配也尝试跑一下，或者是网络通畅），则继续测试对话
    # 如果想强制模型必须存在才跑，可以加上 if model_exists:
    test_api_with_openai_lib(client)