# 数据标准化与思考训练流程

本文档说明如何将MIMIC-CXR原始数据处理成带有思考过程（thinking）的训练数据。

## 完整流程概览

```
原始MIMIC-CXR数据
    ↓
[1] convert_mimic_to_sharegpt.py
    ↓
ShareGPT格式数据
    ↓
[2] normalize_reports_v2.py (可选但推荐)
    ↓
标准化+验证的数据
    ↓
[3] generate_reasoning.py
    ↓
带推理过程的数据
    ↓
[4] format_reasoning_for_training.py
    ↓
<think>...</think>格式训练数据
```

---

## 步骤1：原始数据转换

将MIMIC-CXR原始数据转换为ShareGPT格式。

```bash
python3 convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output mimic_cxr_sharegpt.json \
  --test_split 0.1
```

**输出：**
- `dataset/mimic_cxr_sharegpt_train.json` (19,972个样本)
- `dataset/mimic_cxr_sharegpt_test.json` (2,219个样本)

---

## 步骤2：报告标准化（推荐）

使用LLM标准化报告格式，并用LLM judge验证忠实度。

### 2.1 为什么需要标准化？

**原始数据质量问题：**
- 93.3%样本包含占位符 (`___`)
- 18.1%样本缺失IMPRESSION
- 27.9%样本缺失FINDINGS
- Section标题格式不统一

**标准化后的好处：**
- ✅ 所有报告都有完整的6个section
- ✅ 占位符替换为 `[AGE/GENDER]`, `[DATE]`, `[PRIOR_STUDY]`
- ✅ LLM judge验证，确保医学内容忠实于原始报告
- ✅ 提供结构化JSON，便于分析

### 2.2 运行标准化

```bash
# 测试100个样本
python3 normalize_reports_v2.py \
  --input dataset/mimic_cxr_sharegpt_test.json \
  --output dataset/mimic_cxr_sharegpt_test_normalized_v2.json \
  --max_samples 100 \
  --concurrency 100

# 标准化完整测试集（约30分钟）
python3 normalize_reports_v2.py \
  --input dataset/mimic_cxr_sharegpt_test.json \
  --output dataset/mimic_cxr_sharegpt_test_normalized_v2.json \
  --concurrency 100

# 标准化训练集（约2-3小时）
python3 normalize_reports_v2.py \
  --input dataset/mimic_cxr_sharegpt_train.json \
  --output dataset/mimic_cxr_sharegpt_train_normalized_v2.json \
  --concurrency 100
```

**输出：**
- 标准化的数据集JSON
- Summary JSON（包含验证通过率）
- 每个样本包含：
  - `normalization.extracted_json`: 结构化的报告信息
  - `normalization.verdict`: PASS/FAIL
  - `normalization.kept_original`: 是否保留原始报告

**验证机制：**
- 只有LLM judge验证**PASS**的样本才会被替换
- 验证失败的样本保留原始报告
- 混合数据集：验证通过用标准化版本，失败用原始版本

---

## 步骤3：生成推理过程

使用VLM模型为每个样本生成从影像到诊断的详细推理过程。

```bash
python3 generate_reasoning.py \
  --test_data dataset/mimic_cxr_sharegpt_train_normalized_v2.json \
  --output_dir reasoning_results \
  --model Qwen3-VL-235B-A22B-Thinking \
  --concurrency 50
```

**输出：**
- `reasoning_results/reasoning_results_{timestamp}.json`
- 每个样本包含：
  - `sample_id`: 样本ID
  - `images`: 图像路径
  - `ground_truth`: 原始或标准化的报告
  - `reasoning`: LLM生成的推理过程
  - `success`: 是否成功生成
  - `model`: 使用的模型

---

## 步骤4：格式化为训练数据

将推理过程和报告合并为带`<think>`标签的训练格式。

```bash
# 不使用标准化报告
python3 format_reasoning_for_training.py \
  --reasoning reasoning_results/reasoning_results_20251229_144040.json \
  --output dataset/thinking_training_data.json \
  --split 0.1

# 使用标准化报告（推荐）
python3 format_reasoning_for_training.py \
  --reasoning reasoning_results/reasoning_results_20251229_144040.json \
  --output dataset/thinking_training_data.json \
  --normalized dataset/mimic_cxr_sharegpt_train_normalized_v2.json \
  --split 0.1
```

**参数说明：**
- `--reasoning`: reasoning结果文件
- `--output`: 输出文件路径
- `--normalized`: （可选）标准化数据集，优先使用标准化报告
- `--split`: 训练/测试划分比例（0.1 = 10%测试集，0 = 不划分）

**输出格式：**
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image><image>Generate a medical imaging report based on the X-ray image results."
    },
    {
      "from": "gpt",
      "value": "<think>\n[详细的推理过程]\n</think>\n\n[最终报告]"
    }
  ],
  "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "metadata": {
    "reasoning_model": "Qwen3-VL-235B-A22B-Thinking",
    "report_source": "normalized",
    "sample_id": 0
  }
}
```

**输出文件：**
- `dataset/thinking_training_data_train.json`: 训练集
- `dataset/thinking_training_data_test.json`: 测试集
- `dataset/thinking_training_data_summary.json`: 统计信息

---

## 快速开始脚本

### 标准化数据
```bash
./run_normalization.sh
```

### 格式化思考数据
```bash
./run_format_thinking.sh
```

---

## 训练配置

将格式化后的数据添加到 `dataset/dataset_info.json`:

```json
{
  "thinking_training": {
    "file_name": "thinking_training_data_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

训练时使用支持`<think>`标签的模型（如Qwen3-VL-Thinking系列）。

---

## 数据质量统计

### 原始数据问题
- 占位符样本：93.3%
- 缺失IMPRESSION：18.1%
- 缺失FINDINGS：27.9%
- 格式完整：53.9%

### 标准化后
- 所有样本格式统一：100%
- 验证通过率：通常>80%
- 占位符规范化：100%
- 结构化JSON：100%

---

## 常见问题

**Q: 标准化会不会改变医学内容？**
A: 不会。LLM judge会验证标准化后的内容是否忠实于原始报告。只有验证PASS的样本才会使用标准化版本，失败的样本保留原始报告。

**Q: 为什么要生成推理过程？**
A: 推理过程帮助模型学习"如何思考"，而不仅仅是记忆报告模板。这提高了模型的可解释性和泛化能力。

**Q: 可以不标准化直接生成推理吗？**
A: 可以，但不推荐。标准化数据格式更统一，训练效果更好。

**Q: 并发数设置多少合适？**
A: 默认100。根据API限制和机器性能调整。QwQ模型较慢，可降低并发。

**Q: 处理全部数据需要多久？**
A:
- 标准化训练集（19,972）：2-3小时
- 生成推理（19,972）：10-15小时（取决于模型速度）
- 格式化：<5分钟

---

## 脚本清单

| 脚本 | 功能 | 耗时 |
|------|------|------|
| `convert_mimic_to_sharegpt.py` | 原始数据转换 | 5分钟 |
| `normalize_reports_v2.py` | 报告标准化+验证 | 2-3小时 |
| `generate_reasoning.py` | 生成推理过程 | 10-15小时 |
| `format_reasoning_for_training.py` | 格式化训练数据 | <5分钟 |
| `run_normalization.sh` | 标准化批处理 | - |
| `run_format_thinking.sh` | 格式化批处理 | - |

---

## 技术细节

### 标准化流程（normalize_reports_v2.py）

**Stage 1: Extract Structured JSON**
- 使用LLM从原始报告提取结构化信息
- 输出JSON包含7个字段：examination, indication, technique, comparison, findings, impression, wet_read

**Stage 2: Judge Verification**
- 使用LLM judge验证提取的JSON是否忠实于原始报告
- 判断标准：医学内容完整性、无幻觉、无关键遗漏
- 输出：PASS/FAIL + 理由

**Stage 3: Generate Normalized Report**
- 只有PASS的样本才从JSON生成标准化报告
- FAIL的样本保留原始报告
- 所有样本都记录`normalization`元数据

### 格式化流程（format_reasoning_for_training.py）

1. 加载reasoning数据
2. 如果提供了normalized数据，优先使用标准化报告
3. 清理reasoning中的`<think>`标签（如果已存在）
4. 格式化为：`<think>\n{reasoning}\n</think>\n\n{report}`
5. 转换为ShareGPT格式
6. 划分训练/测试集

---

## 许可与引用

本项目基于MIMIC-CXR数据集。使用前请确保遵守MIMIC-CXR的使用许可。
