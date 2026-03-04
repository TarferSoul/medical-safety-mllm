# Configuration System Guide

本项目使用统一的 YAML 配置文件来管理所有脚本的超参数和设置。

## 📁 配置文件位置

- **主配置文件**: `config.yaml` (项目根目录)
- **配置加载器**: `utils/config_loader.py`
- **使用示例**: `utils/config_example.py`

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pyyaml
# 或安装所有依赖
pip install -r requirements.txt
```

### 2. 编辑配置文件

编辑 `config.yaml` 来修改默认设置：

```yaml
# 修改预测模型
prediction_api:
  model: "Your-Model-Name"

# 修改并发数
evaluation:
  prediction_concurrency: 100
  judge_concurrency: 200
```

### 3. 在脚本中使用配置

```python
from utils.config_loader import load_config

# 加载配置
config = load_config()

# 获取API配置
api_config = config.get_prediction_api_config()
model = api_config['model']
base_url = api_config['base_url']

# 获取评估设置
concurrency = config.get('evaluation.prediction_concurrency')
```

## 📋 配置文件结构

### API端点配置

```yaml
prediction_api:
  prefix: "..."
  port: 8000
  base_url: "https://..."
  api_ak: "..."
  api_sk: "..."
  model: "Model-Name"

judge_api:
  # 同上结构

reasoning_api:
  auth_mode: true  # true=Basic Auth, false=Direct API Key
  # ... 其他配置

normalize_api:
  # 报告标准化专用API配置
  # 同上结构
```

### 数据路径配置

```yaml
data_paths:
  mimic_images_dir: "/path/to/images"
  mimic_reports_dir: "/path/to/reports"
  test_data: "/path/to/test.json"
  output_dir: "/path/to/output"
```

### 评估配置

```yaml
evaluation:
  # 并发设置
  prediction_concurrency: 50
  judge_concurrency: 100
  reasoning_concurrency: 20

  # 重试设置
  prediction_max_retries: 3
  judge_max_retries: 5

  # 生成参数
  prediction_max_tokens: 4096
  prediction_temperature: 0.7
  prediction_top_k: 20
```

### 数据处理配置

```yaml
data_processing:
  default_instruction: "..."
  test_samples: 500
  test_split: null
  random_seed: 42
```

### 标准化配置

```yaml
normalization:
  # 并发设置
  concurrency: 100        # v2脚本并发数
  concurrency_v1: 20      # v1脚本并发数

  # 重试设置
  max_retries: 5

  # 处理设置
  max_samples: null       # 最大样本数（null=全部）
  intermediate_save_interval: 50  # 中间保存间隔

  # 温度设置
  extract_temperature: 0.1
  judge_temperature: 0.0
  normalize_temperature: 0.1

  # Token限制
  max_tokens: 8192
```

### 提示词配置

```yaml
judge_prompt:
  system_message: "..."
  template: |
    多行提示词模板
    使用 {ground_truth} 和 {prediction} 占位符

reasoning_prompt:
  template: |
    多行推理提示词模板
```

## 🔧 使用方法

### 方法1: 使用预定义的获取函数

```python
from utils.config_loader import load_config

config = load_config()

# 获取API配置（自动格式化URL）
pred_config = config.get_prediction_api_config()
judge_config = config.get_judge_api_config()
reasoning_config = config.get_reasoning_api_config(use_auth=True)
normalize_config = config.get_normalize_api_config()

# 获取路径配置
paths = config.get_data_paths()
test_data = paths['test_data']

# 获取评估配置
eval_config = config.get_evaluation_config()
concurrency = eval_config['prediction_concurrency']

# 获取标准化配置
norm_config = config.get_normalization_config()
norm_concurrency = norm_config['concurrency']
max_retries = norm_config['max_retries']

# 获取提示词
judge_prompt = config.get_judge_prompt()
reasoning_prompt = config.get_reasoning_prompt()
```

### 方法2: 使用点号表示法

```python
config = load_config()

# 直接访问嵌套配置
model = config.get('prediction_api.model')
concurrency = config.get('evaluation.prediction_concurrency')
test_samples = config.get('data_processing.test_samples')
norm_concurrency = config.get('normalization.concurrency')

# 带默认值
value = config.get('some.nonexistent.key', default_value)
```

### 方法3: 命令行参数覆盖配置

```python
import argparse
from utils.config_loader import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='Config file path')
parser.add_argument('--model', default=None, help='Override model')
parser.add_argument('--concurrency', type=int, default=None)

args = parser.parse_args()

# 加载配置
config = load_config(args.config)

# 命令行参数优先级更高
model = args.model or config.get('prediction_api.model')
concurrency = args.concurrency or config.get('evaluation.prediction_concurrency')
```

## 📚 完整示例

### 示例1: 预测脚本

```python
#!/usr/bin/env python3
import openai
import base64
from utils.config_loader import load_config

def main():
    # 加载配置
    config = load_config()

    # 获取API配置
    api_config = config.get_prediction_api_config()

    # 初始化客户端
    auth_string = f"{api_config['api_ak']}:{api_config['api_sk']}"
    b64_auth = base64.b64encode(auth_string.encode()).decode()

    client = openai.OpenAI(
        base_url=api_config['base_url'],
        api_key=b64_auth,
        default_headers={"Authorization": f"Basic {b64_auth}"}
    )

    # 获取评估配置
    eval_config = config.get_evaluation_config()

    # 使用配置中的参数
    response = client.chat.completions.create(
        model=api_config['model'],
        messages=[...],
        max_tokens=eval_config['prediction_max_tokens'],
        temperature=eval_config['prediction_temperature'],
        extra_body={
            "top_k": eval_config['prediction_top_k'],
            "chat_template_kwargs": {
                "enable_thinking": eval_config['enable_thinking']
            }
        }
    )

if __name__ == "__main__":
    main()
```

### 示例2: 数据处理脚本

```python
#!/usr/bin/env python3
from utils.config_loader import load_config

def main():
    # 加载配置
    config = load_config()

    # 获取路径
    paths = config.get_data_paths()
    data_config = config.get_data_processing_config()

    # 使用配置中的路径
    process_dataset(
        data_dir=paths['mimic_images_dir'],
        report_dir=paths['mimic_reports_dir'],
        output_file=paths['dataset_dir'] + "/mimic_cxr.json",
        instruction=data_config['default_instruction'],
        test_samples=data_config['test_samples'],
        random_seed=data_config['random_seed']
    )

if __name__ == "__main__":
    main()
```

### 示例3: 评审脚本

```python
#!/usr/bin/env python3
from utils.config_loader import load_config

def evaluate_single_sample(item, config):
    # 获取提示词模板
    prompt_template = config.get_judge_prompt()

    # 填充模板
    final_prompt = prompt_template.format(
        ground_truth=item['ground_truth'],
        prediction=item['prediction']
    )

    # 获取judge配置
    judge_config = config.get_judge_api_config()
    eval_config = config.get_evaluation_config()

    # 调用API
    response = client.chat.completions.create(
        model=judge_config['model'],
        messages=[
            {"role": "system", "content": config.get('judge_prompt.system_message')},
            {"role": "user", "content": final_prompt}
        ],
        temperature=eval_config['judge_temperature'],
        max_tokens=eval_config['judge_max_tokens']
    )

    return response

def main():
    config = load_config()
    # ... 使用配置进行评审
```

## 🎯 运行示例

查看所有配置使用示例：

```bash
cd utils
python config_example.py
```

这将展示：
1. 预测脚本如何使用配置
2. 评审脚本如何使用配置
3. 推理脚本如何使用配置
4. 数据处理如何使用配置
5. 点号表示法访问
6. 命令行参数覆盖

## ⚙️ 配置优先级

配置值的优先级顺序（从高到低）：

1. **命令行参数** - `--model "Model-A"`
2. **环境变量** - `export MODEL="Model-A"`（需要在代码中实现）
3. **配置文件** - `config.yaml` 中的值
4. **代码默认值** - 函数参数的默认值

示例：
```python
# 推荐的参数优先级处理
model = (
    args.model or                           # 命令行参数
    os.getenv('MODEL') or                   # 环境变量
    config.get('prediction_api.model') or   # 配置文件
    "default-model"                         # 默认值
)
```

## 📝 最佳实践

### 1. 敏感信息管理

**不要**将API密钥直接提交到版本控制：

```bash
# 创建 config.local.yaml 用于本地开发
cp config.yaml config.local.yaml

# 在 .gitignore 中添加
echo "config.local.yaml" >> .gitignore

# 在代码中优先加载本地配置
config = load_config('config.local.yaml' if os.path.exists('config.local.yaml') else 'config.yaml')
```

### 2. 多环境配置

为不同环境创建不同的配置文件：

```bash
config.yaml           # 默认配置
config.dev.yaml       # 开发环境
config.prod.yaml      # 生产环境
config.test.yaml      # 测试环境
```

使用时指定：
```bash
python evaluate_model.py --config config.prod.yaml
```

### 3. 配置验证

在脚本开始时验证必需的配置：

```python
config = load_config()

# 验证必需的配置项
required_keys = [
    'prediction_api.model',
    'prediction_api.base_url',
    'data_paths.test_data',
]

for key in required_keys:
    if config.get(key) is None:
        raise ValueError(f"Required config key missing: {key}")
```

### 4. 配置文档化

在配置文件中添加注释说明每个参数的作用：

```yaml
evaluation:
  # 预测并发数：同时处理的样本数
  # 建议值: 50-100，取决于GPU/API限制
  prediction_concurrency: 50

  # Judge并发数：同时评审的样本数
  # Judge通常是CPU任务，可以设置更高
  judge_concurrency: 100
```

## 🔍 故障排查

### 配置文件找不到

```python
FileNotFoundError: config.yaml not found
```

**解决方法**:
1. 确认 `config.yaml` 在项目根目录
2. 或指定完整路径: `load_config('/path/to/config.yaml')`
3. 或使用相对路径: `load_config('../config.yaml')`

### YAML解析错误

```python
yaml.scanner.ScannerError: ...
```

**解决方法**:
1. 检查YAML语法（缩进使用空格，不用Tab）
2. 检查特殊字符是否需要引号
3. 使用在线YAML验证器检查格式

### 配置值为None

```python
value = config.get('some.key')  # 返回 None
```

**解决方法**:
1. 检查key路径是否正确（区分大小写）
2. 使用默认值: `config.get('some.key', 'default')`
3. 打印配置调试: `print(config._config)`

## 📖 更多资源

- [YAML语法教程](https://yaml.org/spec/1.2/spec.html)
- [Python YAML库文档](https://pyyaml.org/wiki/PyYAMLDocumentation)
- 项目示例: `utils/config_example.py`

## 🤝 贡献

添加新的配置项时：
1. 在 `config.yaml` 中添加配置项和注释
2. 在 `config_loader.py` 中添加相应的获取方法（如需要）
3. 在 `config_example.py` 中添加使用示例
4. 更新本文档说明新配置的用途
