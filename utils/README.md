# Utility Scripts

工具脚本，用于测试、配置管理和调试。

## 脚本说明

### config_loader.py
统一配置管理工具，从 `config.yaml` 加载项目配置。

**功能**:
- 自动搜索配置文件位置
- 提供点号表示法访问嵌套配置
- 预定义的配置获取方法
- 单例模式避免重复加载

**使用**:
```python
from utils.config_loader import load_config

# 加载配置
config = load_config()

# 获取API配置
api_config = config.get_prediction_api_config()
model = api_config['model']

# 使用点号表示法
concurrency = config.get('evaluation.prediction_concurrency')
```

**详细文档**: 请参考 [CONFIG.md](../CONFIG.md)

---

### config_example.py
配置系统使用示例，展示各种配置访问方式。

**功能**:
- 演示预测、评审、推理脚本如何使用配置
- 展示点号表示法和字典访问
- 演示CLI参数覆盖配置

**运行**:
```bash
python config_example.py
```

**输出示例**:
```
============================================================
Example 1: Prediction Script with Config
============================================================
✓ Loaded configuration from: config.yaml

Prediction API Settings:
  Base URL: https://...
  Model: Qwen3-8B-VL-Mimic
  ...
```

---

### test_api.py
测试模型API接口是否正常工作。

**功能**:
- 测试OpenAI兼容API连接
- 发送示例图像和prompt
- 验证模型响应格式

**使用**:
```bash
python test_api.py
```

**配置**:
在脚本中修改以下参数：
- `BASE_URL`: API端点地址
- `API_AK` / `API_SK`: 认证密钥
- `MODEL`: 模型名称

**输出示例**:
```
Testing API connection...
Model: Qwen3-8B-VL-Mimic
Response received successfully:

FINAL REPORT
EXAMINATION: CHEST (PA AND LAT)
...
```

---

## 配置系统快速入门

### 1. 安装依赖
```bash
pip install pyyaml
```

### 2. 编辑配置文件
```bash
# 编辑项目根目录的 config.yaml
vim ../config.yaml
```

### 3. 在脚本中使用
```python
#!/usr/bin/env python3
from utils.config_loader import load_config

def main():
    # 加载配置
    config = load_config()

    # 获取配置值
    model = config.get('prediction_api.model')
    concurrency = config.get('evaluation.prediction_concurrency')

    print(f"Using model: {model}")
    print(f"Concurrency: {concurrency}")

if __name__ == "__main__":
    main()
```

### 4. 命令行覆盖配置
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None)
parser.add_argument('--model', default=None)
args = parser.parse_args()

config = load_config(args.config)
model = args.model or config.get('prediction_api.model')
```

## 配置文件结构

```yaml
# API端点配置
prediction_api:
  model: "Model-Name"
  base_url: "https://..."
  ...

# 数据路径配置
data_paths:
  test_data: "/path/to/test.json"
  output_dir: "/path/to/output"
  ...

# 评估超参数
evaluation:
  prediction_concurrency: 50
  judge_concurrency: 100
  prediction_max_tokens: 4096
  ...

# 提示词模板
judge_prompt:
  template: |
    提示词内容...
```

## 添加新工具

如果需要添加新的工具脚本，请：
1. 将脚本放在此目录
2. 在此README中添加说明
3. 确保脚本有清晰的文档字符串
4. 如果使用配置系统，在 config_example.py 中添加示例

## 常见配置任务

### 修改并发数
```yaml
evaluation:
  prediction_concurrency: 100  # 增加预测并发
  judge_concurrency: 200       # 增加评审并发
```

### 切换模型
```yaml
prediction_api:
  model: "New-Model-Name"

judge_api:
  model: "New-Judge-Model"
```

### 修改数据路径
```yaml
data_paths:
  test_data: "/new/path/to/test.json"
  output_dir: "/new/path/to/output"
```

### 调整生成参数
```yaml
evaluation:
  prediction_max_tokens: 8192    # 增加最大生成长度
  prediction_temperature: 0.5    # 降低温度更确定性
  prediction_top_k: 40           # 增加采样池
```

## 更多资源

- **完整配置文档**: [CONFIG.md](../CONFIG.md)
- **配置示例**: `config_example.py`
- **主配置文件**: `../config.yaml`
