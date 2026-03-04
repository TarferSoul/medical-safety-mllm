# Configuration System Quick Reference

## 📁 文件位置

```
MedicalSafety/
├── config.yaml                  # 主配置文件（编辑这里修改参数）
├── CONFIG.md                    # 完整配置文档
├── requirements.txt             # 依赖（包含pyyaml）
└── utils/
    ├── config_loader.py         # 配置加载器
    └── config_example.py        # 使用示例
```

## ⚡ 快速开始

### 1. 安装依赖
```bash
pip install pyyaml
# 或
pip install -r requirements.txt
```

### 2. 编辑配置
```bash
vim config.yaml
```

### 3. 在脚本中使用
```python
from utils.config_loader import load_config

config = load_config()
model = config.get('prediction_api.model')
```

### 4. 测试配置
```bash
python utils/config_example.py
```

## 📝 常用配置修改

### 修改模型
```yaml
prediction_api:
  model: "Your-Model-Name"

judge_api:
  model: "Your-Judge-Model"
```

### 修改并发数
```yaml
evaluation:
  prediction_concurrency: 100  # 预测并发
  judge_concurrency: 200       # 评审并发
```

### 修改路径
```yaml
data_paths:
  test_data: "/path/to/test.json"
  output_dir: "/path/to/output"
```

### 修改生成参数
```yaml
evaluation:
  prediction_max_tokens: 8192
  prediction_temperature: 0.5
  prediction_top_k: 40
```

## 🔧 代码示例

### 基础使用
```python
from utils.config_loader import load_config

config = load_config()

# 方法1: 预定义函数
api_config = config.get_prediction_api_config()
model = api_config['model']
base_url = api_config['base_url']

# 方法2: 点号表示法
model = config.get('prediction_api.model')
concurrency = config.get('evaluation.prediction_concurrency')

# 方法3: 带默认值
value = config.get('some.key', 'default')
```

### CLI参数覆盖
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None)
parser.add_argument('--model', default=None)
parser.add_argument('--concurrency', type=int, default=None)
args = parser.parse_args()

config = load_config(args.config)

# CLI参数优先级更高
model = args.model or config.get('prediction_api.model')
concurrency = args.concurrency or config.get('evaluation.prediction_concurrency')
```

### 完整示例：预测脚本
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
    eval_config = config.get_evaluation_config()

    # 初始化客户端
    auth_string = f"{api_config['api_ak']}:{api_config['api_sk']}"
    b64_auth = base64.b64encode(auth_string.encode()).decode()

    client = openai.OpenAI(
        base_url=api_config['base_url'],
        api_key=b64_auth,
        default_headers={"Authorization": f"Basic {b64_auth}"}
    )

    # 使用配置参数
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

## 📋 配置优先级

从高到低：
1. **命令行参数** - `--model "Model-A"`
2. **环境变量** - `export MODEL="Model-A"`
3. **配置文件** - `config.yaml`
4. **代码默认值** - 函数参数默认值

## 🎯 预定义配置获取函数

```python
config = load_config()

# API配置
api_config = config.get_prediction_api_config()
judge_config = config.get_judge_api_config()
reasoning_config = config.get_reasoning_api_config(use_auth=True)

# 路径配置
paths = config.get_data_paths()

# 评估配置
eval_config = config.get_evaluation_config()

# 数据处理配置
data_config = config.get_data_processing_config()

# 提示词模板
judge_prompt = config.get_judge_prompt()
reasoning_prompt = config.get_reasoning_prompt()
```

## 🔍 配置文件主要结构

```yaml
# API端点
prediction_api:
  model: "..."
  base_url: "..."
  api_ak: "..."
  api_sk: "..."

judge_api: { ... }
reasoning_api: { ... }

# 数据路径
data_paths:
  mimic_images_dir: "..."
  test_data: "..."
  output_dir: "..."

# 评估设置
evaluation:
  prediction_concurrency: 50
  judge_concurrency: 100
  prediction_max_retries: 3
  prediction_max_tokens: 4096
  prediction_temperature: 0.7

# 数据处理
data_processing:
  test_samples: 500
  random_seed: 42

# 提示词模板
judge_prompt:
  template: |
    多行提示词...

reasoning_prompt:
  template: |
    多行提示词...
```

## 📚 更多资源

- **完整文档**: [CONFIG.md](CONFIG.md)
- **使用示例**: `python utils/config_example.py`
- **主配置**: `config.yaml`

## 💡 最佳实践

1. **敏感信息**: 创建 `config.local.yaml` 用于本地开发，不提交到git
2. **多环境**: 创建 `config.dev.yaml`, `config.prod.yaml` 等
3. **验证配置**: 在脚本开始时验证必需配置项
4. **注释说明**: 在配置文件中添加详细注释

## 🐛 故障排查

### 配置文件找不到
```python
FileNotFoundError: config.yaml not found
```
**解决**: 确认 `config.yaml` 在项目根目录，或使用 `load_config('/path/to/config.yaml')`

### YAML解析错误
```python
yaml.scanner.ScannerError: ...
```
**解决**: 检查YAML语法（缩进用空格，不用Tab）

### 配置值为None
```python
value = config.get('some.key')  # None
```
**解决**: 检查key路径是否正确，或使用默认值: `config.get('some.key', 'default')`
