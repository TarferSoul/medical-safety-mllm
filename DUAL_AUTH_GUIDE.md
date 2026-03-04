# 双认证模式配置指南

本项目支持两种API认证模式，所有脚本都兼容两种模式。

## 📋 两种认证模式

### 1. Basic Auth (AK/SK) 模式
- 使用 Access Key 和 Secret Key 进行认证
- 适用于内部API端点
- 需要 Base64 编码认证字符串

### 2. Direct API Key 模式
- 使用标准的 API Key 直接认证
- 适用于公开API服务
- 配置更简单

## ⚙️ 配置方法

### 在 config.yaml 中配置

每个API都支持两种模式，通过 `auth_mode` 切换：

```yaml
# Prediction API 示例
prediction_api:
  # 认证模式：true=Basic Auth, false=Direct API Key
  auth_mode: true  # 修改这里切换模式

  # Basic Auth 配置
  prefix: "rjob-..."
  port: 8000
  base_url_auth: "https://..."
  api_ak: "your_access_key"
  api_sk: "your_secret_key"
  model_auth: "Qwen3-8B-VL-Model"

  # Direct API Key 配置
  base_url_direct: "http://35.220.164.252:3888/v1"
  api_key_direct: "sk-XRPOyVYolEzdtwvcvUQtAyQIUzNPchDYK9mdXDUQPc0z7yNb"
  model_direct: "Qwen/Qwen3-VL-235B-A22B-Thinking"
```

### 切换认证模式

**方法1: 修改配置文件**
```yaml
# 使用 Basic Auth
auth_mode: true

# 使用 Direct API Key
auth_mode: false
```

**方法2: 在代码中覆盖**
```python
from utils.config_loader import load_config, create_openai_client

config = load_config()

# 强制使用 Basic Auth
api_config = config.get_prediction_api_config(use_auth=True)

# 强制使用 Direct API Key
api_config = config.get_prediction_api_config(use_auth=False)

# 创建客户端（自动识别认证模式）
client = create_openai_client(api_config)
```

## 💻 代码示例

### 完整示例：支持两种认证模式的脚本

```python
#!/usr/bin/env python3
"""
示例：支持两种认证模式的评估脚本
"""
import argparse
from utils.config_loader import load_config, create_openai_client

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auth_mode', choices=['basic', 'direct'],
                       default=None, help='Override auth mode')
    args = parser.parse_args()

    # 加载配置
    config = load_config()

    # 根据命令行参数选择认证模式
    if args.auth_mode == 'basic':
        use_auth = True
    elif args.auth_mode == 'direct':
        use_auth = False
    else:
        use_auth = None  # 使用配置文件中的默认值

    # 获取API配置
    api_config = config.get_prediction_api_config(use_auth=use_auth)

    print(f"Using auth mode: {api_config['auth_mode']}")
    print(f"Model: {api_config['model']}")
    print(f"Base URL: {api_config['base_url']}")

    # 创建客户端（自动处理两种认证方式）
    client = create_openai_client(api_config)

    # 使用客户端
    response = client.chat.completions.create(
        model=api_config['model'],
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=100
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
```

### 运行示例

```bash
# 使用配置文件中的默认模式
python your_script.py

# 强制使用 Basic Auth
python your_script.py --auth_mode basic

# 强制使用 Direct API Key
python your_script.py --auth_mode direct
```

## 🔄 统一客户端初始化

使用 `create_openai_client` 函数可以自动处理两种认证方式：

```python
from utils.config_loader import load_config, create_openai_client

# 加载配置
config = load_config()

# 获取API配置（自动根据 auth_mode 选择认证方式）
api_config = config.get_prediction_api_config()

# 创建客户端（自动处理 Basic Auth 或 Direct API Key）
client = create_openai_client(api_config)

# 使用统一的接口调用
response = client.chat.completions.create(
    model=api_config['model'],
    messages=[...],
    max_tokens=4096
)
```

## 📊 三个API的配置

所有三个API都支持两种认证模式：

### Prediction API (预测模型)
```yaml
prediction_api:
  auth_mode: true  # 或 false
  # ... Basic Auth 配置
  # ... Direct API Key 配置
```

### Judge API (评审模型)
```yaml
judge_api:
  auth_mode: true  # 或 false
  # ... Basic Auth 配置
  # ... Direct API Key 配置
```

### Reasoning API (推理模型)
```yaml
reasoning_api:
  auth_mode: true  # 或 false
  # ... Basic Auth 配置
  # ... Direct API Key 配置
```

## 🎯 最佳实践

### 1. 开发环境 vs 生产环境

```yaml
# config.dev.yaml (开发环境 - 使用 Direct API Key)
prediction_api:
  auth_mode: false
  base_url_direct: "http://localhost:8000/v1"
  api_key_direct: "dev-api-key"
  model_direct: "local-model"

# config.prod.yaml (生产环境 - 使用 Basic Auth)
prediction_api:
  auth_mode: true
  base_url_auth: "https://production-api.com/v1/"
  api_ak: "prod-access-key"
  api_sk: "prod-secret-key"
  model_auth: "production-model"
```

使用时指定配置文件：
```bash
# 开发环境
python script.py --config config.dev.yaml

# 生产环境
python script.py --config config.prod.yaml
```

### 2. 环境变量覆盖

```python
import os
from utils.config_loader import load_config

config = load_config()

# 允许环境变量覆盖认证模式
use_auth = os.getenv('USE_BASIC_AUTH', 'true').lower() == 'true'

api_config = config.get_prediction_api_config(use_auth=use_auth)
```

运行时：
```bash
# 使用 Basic Auth
USE_BASIC_AUTH=true python script.py

# 使用 Direct API Key
USE_BASIC_AUTH=false python script.py
```

### 3. 动态切换

```python
from utils.config_loader import load_config, create_openai_client

config = load_config()

# 尝试 Basic Auth，失败则切换到 Direct API Key
try:
    api_config = config.get_prediction_api_config(use_auth=True)
    client = create_openai_client(api_config)
    # 测试连接
    response = client.chat.completions.create(
        model=api_config['model'],
        messages=[{"role": "user", "content": "test"}],
        max_tokens=10
    )
    print("✓ Using Basic Auth")
except Exception as e:
    print(f"Basic Auth failed: {e}")
    print("Switching to Direct API Key...")
    api_config = config.get_prediction_api_config(use_auth=False)
    client = create_openai_client(api_config)
    print("✓ Using Direct API Key")
```

## 🔍 配置检查

创建一个配置检查脚本：

```python
#!/usr/bin/env python3
"""检查所有API的配置是否正确"""
from utils.config_loader import load_config, create_openai_client

def check_api(api_name, get_config_func):
    print(f"\n=== Checking {api_name} ===")

    # 检查 Basic Auth
    print("  Basic Auth mode:")
    try:
        config = get_config_func(use_auth=True)
        client = create_openai_client(config)
        print(f"    ✓ Model: {config['model']}")
        print(f"    ✓ URL: {config['base_url']}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # 检查 Direct API Key
    print("  Direct API Key mode:")
    try:
        config = get_config_func(use_auth=False)
        client = create_openai_client(config)
        print(f"    ✓ Model: {config['model']}")
        print(f"    ✓ URL: {config['base_url']}")
    except Exception as e:
        print(f"    ✗ Error: {e}")

if __name__ == "__main__":
    config = load_config()

    check_api("Prediction API", config.get_prediction_api_config)
    check_api("Judge API", config.get_judge_api_config)
    check_api("Reasoning API", config.get_reasoning_api_config)
```

## 📖 更多资源

- **配置示例**: `python utils/config_example.py`
- **完整文档**: [CONFIG.md](CONFIG.md)
- **快速参考**: [QUICKSTART_CONFIG.md](QUICKSTART_CONFIG.md)
