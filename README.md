# Medical Imaging MLLM Safety Project

医学影像多模态大模型安全评估项目 - 基于MIMIC-CXR数据集的胸部X光报告生成与评估。

## 📁 项目结构

```
MedicalSafety/
├── data_processing/          # 数据处理脚本
│   ├── convert_mimic_to_sharegpt.py    # MIMIC数据转换
│   ├── normalize_reports.py            # 报告标准化
│   ├── format_reasoning_for_training.py # 推理数据格式化
│   └── README.md                       # 详细文档
│
├── evaluation/               # 模型评估脚本
│   ├── evaluate_and_judge.py          # 集成评估（推荐）
│   ├── evaluate_model.py              # 单独预测
│   ├── llm_as_judge.py                # 单独评审
│   ├── generate_reasoning.py          # 推理生成
│   └── README.md                      # 详细文档
│
├── utils/                    # 工具脚本
│   ├── test_api.py                    # API测试
│   └── README.md                      # 详细文档
│
├── dataset/                  # 训练和测试数据
│   ├── dataset_info.json              # LLaMA-Factory配置
│   ├── mimic_cxr_sharegpt_train.json  # 训练集
│   └── mimic_cxr_sharegpt_test.json   # 测试集
│
├── evaluation_results/       # 评估结果（按模型分类）
│   ├── pred_Model_A/
│   └── pred_Model_B/
│
├── reasoning_results/        # 推理结果（按模型分类）
│   ├── reasoning_Model_A/
│   └── reasoning_Model_B/
│
├── LLaMA-Factory/           # 训练框架（子模块）
│   ├── train/
│   └── examples/
│
├── CLAUDE.md                # Claude Code 工作指南
└── README.md                # 本文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 设置CUDA环境
export CUDA_HOME=/mnt/shared-storage-user/xieyuejin/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置参数设置

本项目使用统一的 `config.yaml` 配置文件管理所有超参数。

**编辑配置文件**:
```bash
# 编辑 config.yaml 修改API端点、模型名称、路径等
vim config.yaml
```

**主要配置项**:
- `prediction_api` - 预测模型API配置（支持双认证模式）
- `judge_api` - 评审模型API配置（支持双认证模式）
- `reasoning_api` - 推理模型API配置（支持双认证模式）
- `data_paths` - 数据文件路径
- `evaluation` - 评估超参数（并发数、重试次数等）

**双认证模式支持** ✨:
所有API都支持两种认证方式：
- **Basic Auth (AK/SK)**: 适用于内部API端点
- **Direct API Key**: 适用于公开API服务

在 `config.yaml` 中切换：
```yaml
prediction_api:
  auth_mode: true  # true=Basic Auth, false=Direct API Key
  # ... 配置两种模式的参数
```

详细说明：
- 配置系统文档：[CONFIG.md](CONFIG.md)
- 双认证模式指南：[DUAL_AUTH_GUIDE.md](DUAL_AUTH_GUIDE.md)
- 快速参考：[QUICKSTART_CONFIG.md](QUICKSTART_CONFIG.md)

**测试配置**:
```bash
# 查看配置示例（包含两种认证模式演示）
python utils/config_example.py
```

### 3. 数据准备

```bash
cd data_processing

# 转换MIMIC-CXR数据为ShareGPT格式（默认500个测试样本）
python convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output ../dataset/mimic_cxr_sharegpt.json

# 自定义测试集样本数
python convert_mimic_to_sharegpt.py \
  --data_dir /path/to/mimic/images \
  --report_dir /path/to/mimic/reports \
  --output ../dataset/mimic_cxr_sharegpt.json \
  --test_samples 1000
```

### 4. 模型训练

```bash
cd LLaMA-Factory

# 全参数微调
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_vl_8b_full_sft.yaml

# LoRA微调
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### 5. 模型评估

```bash
cd evaluation

# 推荐：一键完成预测和评审
python evaluate_and_judge.py \
  --test_data ../dataset/mimic_cxr_sharegpt_test.json \
  --pred_model "Qwen3-8B-VL-Mimic" \
  --judge_model "QwQ" \
  --concurrency 50

# 或者分步进行
python evaluate_model.py --max_samples 100
python llm_as_judge.py --input ../evaluation_results/pred_ModelName/xxx.json
```

## 📊 主要功能

### 数据处理
- ✅ MIMIC-CXR数据转换为ShareGPT格式
- ✅ 医学报告标准化和规范化
- ✅ 推理过程数据格式化（带`<think>`标签）

### 模型评估
- ✅ 自动化预测生成（支持多图输入）
- ✅ LLM-as-Judge评审机制
- ✅ 多维度评分（临床准确性、解剖位置、错误类型）
- ✅ 推理过程生成（可解释AI）
- ✅ 结果按模型分类存储

### 评估维度
- **Clinical Accuracy**: 病理特征识别准确性
- **Negation Correctness**: 否定词使用正确性
- **Anatomical Correctness**: 解剖位置定位准确性
- **Error Types**: 遗漏、幻觉、定位错误、严重程度错误

## 📖 详细文档

每个目录都包含详细的README文档：

- [data_processing/README.md](data_processing/README.md) - 数据处理脚本详细说明
- [evaluation/README.md](evaluation/README.md) - 评估脚本详细说明和工作流程
- [utils/README.md](utils/README.md) - 工具脚本说明
- [CLAUDE.md](CLAUDE.md) - Claude Code AI助手工作指南

## 💡 推荐工作流程

### 完整评估流程
```bash
# 1. 准备数据（首次运行）
cd data_processing
python convert_mimic_to_sharegpt.py --output ../dataset/mimic_cxr.json

# 2. 训练模型（在LLaMA-Factory中）
cd ../LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_vl_8b_full_sft.yaml

# 3. 评估模型（推荐使用集成脚本）
cd ../evaluation
python evaluate_and_judge.py --concurrency 50

# 4. 分析结果
cd ../evaluation_results/pred_YourModel/
# 查看生成的JSON文件和summary
```

### 带推理的训练流程
```bash
# 1. 生成推理过程
cd evaluation
python generate_reasoning.py --max_samples 1000

# 2. 格式化为训练数据
cd ../data_processing
python format_reasoning_for_training.py \
  --reasoning ../reasoning_results/reasoning_Model/xxx.json \
  --output ../dataset/mimic_cxr_with_reasoning.json

# 3. 使用带推理的数据训练模型
cd ../LLaMA-Factory
llamafactory-cli train your_reasoning_config.yaml
```

## 🔧 配置说明

### API配置
所有评估脚本都支持配置API端点和认证信息，在脚本顶部修改：
```python
BASE_URL = "https://your-api-endpoint/v1/"
API_AK = "your_access_key"
API_SK = "your_secret_key"
MODEL = "your_model_name"
```

### 并发配置
- 预测模型：建议并发数 50-100（取决于GPU资源）
- Judge模型：建议并发数 100+（主要是CPU任务）

### 输出组织
- 所有结果自动按模型名称分类存储
- 每次运行都有时间戳标识
- 每50个样本自动保存中间结果

## 📈 输出文件说明

### 评估结果文件
```
evaluation_results/pred_ModelName/
├── evaluation_xxx_with_judge_xxx.json           # 完整结果
├── evaluation_xxx_with_judge_xxx_summary.json   # 统计摘要
└── evaluation_xxx_with_judge_xxx_intermediate.json  # 中间结果
```

### 结果JSON格式
```json
{
  "sample_id": 0,
  "images": ["path/to/image.jpg"],
  "prompt": "Generate a medical report...",
  "ground_truth": "FINAL REPORT...",
  "prediction": "模型生成的报告",
  "success": true,
  "model": "Qwen3-8B-VL-Mimic",
  "eval_results": {
    "eval_input": "评审提示词",
    "eval_output": {
      "reasoning": "逐步对比分析...",
      "error_types": ["None"],
      "clinical_accuracy_score": 9.5
    }
  },
  "judge_model": "QwQ"
}
```

## 🤝 贡献指南

添加新脚本时：
1. 根据功能放入对应目录（data_processing / evaluation / utils）
2. 在对应目录的README.md中添加文档
3. 更新本README的项目结构部分
4. 确保脚本有清晰的文档字符串和使用说明

## 📝 License

本项目用于研究目的，使用MIMIC-CXR数据集需遵守其许可协议。

## 📧 联系方式

如有问题或建议，请提交Issue或联系项目维护者。
