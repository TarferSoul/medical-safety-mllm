# Evaluation Scripts

模型评估相关脚本，用于测试模型性能和生成评估报告。

## 脚本说明

### evaluate_and_judge.py (推荐使用)
**集成评估脚本** - 预测和评审一体化流程。

**功能**:
- 使用预测模型生成医学报告
- 立即使用Judge模型评估生成质量
- 单个命令完成完整评估流程
- 实时显示预测和评审错误

**使用**:
```bash
python evaluate_and_judge.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --output_dir ../evaluation_results \
  --pred_model "Qwen3-8B-VL-Mimic" \
  --judge_model "QwQ" \
  --concurrency 50 \
  --max_samples 100
```

**输出**:
- 结果保存在 `evaluation_results/pred_{模型名}/` 子目录
- 包含预测和评审的完整结果
- 实时错误提示（❌ 预测失败，⚠️ 评审失败，💥 处理异常）

---

### evaluate_model.py
**单独预测脚本** - 仅生成模型预测，不进行评审。

**功能**:
- 使用训练好的模型生成医学报告
- 并发处理大规模数据集
- 保存预测结果和统计信息

**使用**:
```bash
python evaluate_model.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --output_dir ../evaluation_results \
  --model "Qwen3-8B-VL-Mimic" \
  --concurrency 100 \
  --max_samples 100
```

**输出**:
- `evaluation_results/pred_{模型名}/evaluation_results_{模型名}_{时间戳}.json`
- 包含预测文本和ground truth对比

---

### llm_as_judge.py
**单独评审脚本** - 对已有的预测结果进行评审。

**功能**:
- 使用大语言模型评估报告质量
- 多维度评分（临床准确性、解剖位置、错误类型）
- 支持重试机制和并发处理

**使用**:
```bash
python llm_as_judge.py \
  --input evaluation_results/pred_ModelA/evaluation_results_xxx.json \
  --judge_model "QwQ" \
  --concurrency 100 \
  --max_retries 5
```

**输出**:
- 自动在输入文件目录生成 `{输入文件名}_with_judge_{评审模型}_xxx.json`
- 包含详细的评分和错误分析

**评审维度**:
- Clinical Accuracy (临床准确性)
- Negation Correctness (否定词准确性)
- Anatomical Correctness (解剖位置准确性)
- Error Types (错误类型：遗漏、幻觉、定位错误、严重程度错误)

---

### generate_reasoning.py
**推理生成脚本** - 生成从图像到诊断的推理过程。

**功能**:
- 给定X光图像和ground truth报告
- 让LLM解释诊断的推理过程
- 生成可解释的诊断思维链

**使用**:
```bash
python generate_reasoning.py \
  --test_data ../dataset/mimic_cxr_train.json \
  --output_dir ../reasoning_results \
  --model "Qwen3-VL-32B-Thinking" \
  --concurrency 20 \
  --max_samples 1000
```

**输出**:
- `reasoning_results/reasoning_{模型名}/reasoning_results_{模型名}_{时间戳}.json`
- 包含详细的4步推理过程

**推理步骤**:
1. Anatomy & Support Devices (解剖结构和医疗设备)
2. Visual Feature Extraction (视觉特征提取)
3. Diagnostic Synthesis (诊断综合)
4. Conclusion Alignment (结论对齐)

---

## 推荐工作流程

### 完整评估流程（推荐）
```bash
# 一步完成预测和评审
python evaluate_and_judge.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --concurrency 50
```

### 分步评估流程
```bash
# Step 1: 生成预测
python evaluate_model.py --max_samples 100

# Step 2: 评审预测结果
python llm_as_judge.py \
  --input evaluation_results/pred_ModelA/evaluation_results_xxx.json
```

### 推理数据生成流程
```bash
# Step 1: 生成推理过程
python generate_reasoning.py --max_samples 1000

# Step 2: 格式化为训练数据
cd ../data_processing
python format_reasoning_for_training.py \
  --reasoning ../reasoning_results/reasoning_{model}/xxx.json
```

## 输出目录结构

```
evaluation_results/
├── pred_Qwen3_8B_VL_Mimic/
│   ├── evaluation_xxx_with_judge_QwQ_xxx.json      # 完整结果
│   ├── evaluation_xxx_with_judge_QwQ_xxx_summary.json  # 统计摘要
│   └── evaluation_xxx_with_judge_QwQ_xxx_intermediate.json  # 中间结果
└── pred_Another_Model/
    └── ...

reasoning_results/
├── reasoning_Qwen3_VL_32B_Thinking/
│   ├── reasoning_results_xxx.json
│   └── reasoning_summary_xxx.json
└── ...
```
