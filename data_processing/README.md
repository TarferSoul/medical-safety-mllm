# Data Processing Scripts

数据处理相关脚本，用于准备和转换训练数据。

## 脚本说明

### convert_mimic_to_sharegpt.py
将MIMIC-CXR原始数据转换为ShareGPT格式，供LLaMA-Factory训练使用。

**功能**:
- 读取MIMIC-CXR图像和放射科报告
- 转换为对话格式（human/gpt）
- 自动划分训练集和测试集（默认500个测试样本）
- 自动从 `config.yaml` 加载默认配置

**使用**:
```bash
# 使用 config.yaml 中的默认配置
python convert_mimic_to_sharegpt.py

# 自定义输出文件名
python convert_mimic_to_sharegpt.py \
  --output mimic_cxr_custom.json

# 覆盖配置文件中的测试集样本数
python convert_mimic_to_sharegpt.py \
  --test_samples 1000

# 使用百分比划分（而不是固定数量）
python convert_mimic_to_sharegpt.py \
  --test_samples 0 \
  --test_split 0.1

# 使用自定义 prompt（覆盖 config.yaml 中的 default_instruction）
python convert_mimic_to_sharegpt.py \
  --instruction "Generate a detailed chest X-ray report."
```

**配置说明**:
脚本会自动从 `config.yaml` 加载以下默认值：
- `data_paths.mimic_images_dir`: 图像目录
- `data_paths.mimic_reports_dir`: 报告目录
- `data_processing.default_instruction`: 默认prompt（详细的医学报告生成指令）
- `data_processing.test_samples`: 测试集样本数（默认500）
- `data_processing.random_seed`: 随机种子（默认42）

**参数说明**:
- `--data_dir`: 图像目录（默认从config.yaml读取）
- `--report_dir`: 报告目录（默认从config.yaml读取）
- `--instruction`: 用户指令（默认从config.yaml读取详细prompt）
- `--test_samples`: 测试集样本数（默认从config.yaml读取）
- `--test_split`: 测试集比例（仅当test_samples=0时使用）
- `--max_samples`: 最多处理的样本数（用于快速测试）
- `--random_seed`: 随机种子（默认从config.yaml读取）

**提示**:
- 所有参数都可以通过命令行覆盖配置文件中的默认值
- 要查看当前使用的配置，可以运行 `python convert_mimic_to_sharegpt.py --help`
- 关于 prompt 模板的详细说明，请参考 [PROMPT_TEMPLATES.md](../PROMPT_TEMPLATES.md)

### normalize_reports.py / normalize_reports_v2.py
标准化医学报告格式，统一不同医生的书写风格。

**功能**:
- 使用LLM规范化报告格式
- 统一医学术语和缩写
- 保持临床信息的准确性
- 自动从 `config.yaml` 加载 API 配置

**v1 vs v2**:
- `normalize_reports.py`: 直接使用 LLM 标准化报告
- `normalize_reports_v2.py`: 三阶段流程（提取JSON → LLM验证 → 生成报告），更可靠

**使用**:
```bash
# normalize_reports.py
python normalize_reports.py \
  --input dataset/mimic_cxr_sharegpt_train.json \
  --output dataset/mimic_cxr_train_normalized.json \
  --concurrency 20

# normalize_reports_v2.py（推荐）
python normalize_reports_v2.py \
  --input dataset/mimic_cxr_sharegpt_train.json \
  --output dataset/mimic_cxr_train_normalized_v2.json \
  --concurrency 100
```

**配置说明**:
两个脚本都自动从 `config.yaml` 加载：
- **API配置**: `normalize_api` - API端点、认证模式、模型名称
- **超参数配置**: `normalization` - 并发数、重试次数、样本数等

配置项包括：
- `normalize_api.model_auth` / `model_direct` - 模型名称
- `normalization.concurrency` - 并发数（v2默认100）
- `normalization.concurrency_v1` - v1脚本并发数（默认20）
- `normalization.max_retries` - 最大重试次数（默认5）
- `normalization.max_samples` - 最大样本数（默认null=全部）
- `normalization.intermediate_save_interval` - 中间保存间隔（默认50）
- `normalization.extract_temperature` - 提取温度（默认0.1）
- `normalization.judge_temperature` - 验证温度（默认0.0）
- `normalization.max_tokens` - 最大token数（默认8192）

**参数说明**:
- `--input`: 输入JSON文件（默认从config.yaml读取）
- `--output`: 输出JSON文件（自动生成：`{input}_norm_by_{model}.json`）
- `--model` / `--extract_model` / `--judge_model`: 覆盖默认模型
- `--concurrency`: 并发数（v1默认20，v2默认100）
- `--max_samples`: 仅处理前N个样本（用于测试）

### generate_reasoning.py
生成诊断推理过程，用于训练模型的思维链（Chain of Thought）。

**功能**:
- 给定X光图像和Ground Truth报告，使用VLM生成推理过程
- 反向工程放射科医生的诊断思维过程
- 四步推理：解剖检查 → 视觉特征提取 → 诊断综合 → 结论对齐
- 自动从 `config.yaml` 加载 API 配置和处理参数

**使用**:
```bash
# 使用 config.yaml 中的默认配置
python generate_reasoning.py

# 自定义输入文件和并发数
python generate_reasoning.py \
  --input dataset/mimic_cxr_train_normalized.json \
  --concurrency 30

# 测试模式（仅处理前10个样本）
python generate_reasoning.py \
  --max_samples 10
```

**配置说明**:
脚本自动从 `config.yaml` 加载：
- **API配置**: `reasoning_api` - API端点、认证模式、模型名称
- **处理参数**: 输入文件、输出目录、并发数、重试次数等

配置项包括：
- `reasoning_api.model_auth` / `model_direct` - 模型名称
- `reasoning_api.input` - 输入文件路径
- `reasoning_api.output_dir` - 输出目录（默认 `reasoning_results/`）
- `reasoning_api.concurrency` - 并发数（默认20）
- `reasoning_api.max_retries` - 最大重试次数（默认3）
- `reasoning_api.max_samples` - 最大样本数（默认null=全部）
- `reasoning_api.intermediate_save_interval` - 中间保存间隔（默认20）
- `reasoning_api.max_tokens` - 最大token数（默认32768）
- `reasoning_api.temperature` - 温度参数（默认0.7）
- `reasoning_api.top_k` - Top-K采样（默认20）
- `reasoning_api.enable_thinking` - 启用思考模式（默认true）

**参数说明**:
- `--input`: 输入JSON文件（默认从config.yaml读取）
- `--output_dir`: 输出目录（默认从config.yaml读取）
- `--model`: 模型名称（默认从config.yaml读取）
- `--concurrency`: 并发数（默认从config.yaml读取）
- `--max_samples`: 仅处理前N个样本（用于测试）

**输出**:
- `reasoning_results/reasoning_{model}/reasoning_results_{model}_{timestamp}.json` - 推理结果
- `reasoning_results/reasoning_{model}/reasoning_summary_{model}_{timestamp}.json` - 统计摘要
- `reasoning_results/reasoning_{model}/intermediate_reasoning_{model}_{timestamp}.json` - 中间结果

### format_reasoning_for_training.py
将推理过程数据格式化为训练格式，添加`<think>`标签。

**功能**:
- 合并推理数据和标准化报告
- 添加思维过程标记
- 生成训练/测试集划分

**使用**:
```bash
python format_reasoning_for_training.py \
  --reasoning reasoning_results/reasoning_results_xxx.json \
  --normalized dataset/mimic_cxr_normalized.json \
  --output dataset/mimic_cxr_with_reasoning.json \
  --split 0.1
```

### generate_llamafactory_dataset.py
生成LLaMA-Factory训练数据集配置（待实现）。
