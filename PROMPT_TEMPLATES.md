# Prompt Templates for Medical Report Generation

本项目提供多个 prompt 模板，适用于不同的闭源模型测试场景。

## 📋 Prompt 模板

### 1. 详细版（推荐用于闭源模型）

**位置**: `config.yaml` -> `data_processing.default_instruction`

**特点**:
- 明确指定报告结构（EXAMINATION、INDICATION、FINDINGS、IMPRESSION 等）
- 详细说明每个部分应该包含的内容
- 提供具体的格式要求和注意事项
- 适合 GPT-4V、Claude、Gemini 等闭源模型

**使用**:
```bash
cd data_processing
python convert_mimic_to_sharegpt.py  # 自动使用配置文件中的详细版
```

**完整内容**:
```
You are an expert radiologist. Analyze the provided chest X-ray image(s) and generate a comprehensive radiology report following the standard MIMIC-CXR format.

Your report must include the following sections in order:

1. **EXAMINATION**: State the type of imaging study (e.g., "CHEST (PA AND LAT)" or "CHEST (PORTABLE AP)")

2. **INDICATION**: Brief clinical context or reason for the examination (if relevant medical history or symptoms can be inferred)

3. **TECHNIQUE**: Describe the radiographic technique used

4. **COMPARISON**: Note any comparison with prior studies if applicable, or state "None" if not available

5. **FINDINGS**: Provide detailed observations including:
   - Lung fields (clarity, opacities, masses, nodules, infiltrates)
   - Pleural spaces (effusions, pneumothorax, thickening)
   - Cardiac silhouette (size, contours)
   - Mediastinum (widening, masses, lymphadenopathy)
   - Osseous structures (fractures, lesions)
   - Medical devices (tubes, lines, catheters, pacemakers) with precise positions
   - Soft tissues and other abnormalities

6. **IMPRESSION**: Concise summary of key findings and their clinical significance

Requirements:
- Use precise medical terminology and standard radiological language
- Be specific about anatomical locations (e.g., "right lower lobe," "left costophrenic angle")
- Describe medical devices with exact positioning
- Note both normal and abnormal findings
- Use professional, objective tone
- Avoid speculation; report only what is visible
- If findings are stable compared to prior imaging, note this explicitly

Generate the complete radiology report now:
```

---

### 2. 简化版（微调模型可用）

**特点**:
- 简洁明了
- 适合已经在医学数据上微调过的模型
- 信任模型已学会正确的格式

**使用**:
```bash
cd data_processing
python convert_mimic_to_sharegpt.py \
  --instruction "Generate a comprehensive chest X-ray radiology report in standard MIMIC-CXR format."
```

**完整内容**:
```
Generate a comprehensive chest X-ray radiology report in standard MIMIC-CXR format.
```

---

### 3. 中等版（平衡选项）

**特点**:
- 指定主要结构，但不过度详细
- 适合大部分模型

**使用**:
```bash
cd data_processing
python convert_mimic_to_sharegpt.py \
  --instruction "You are an expert radiologist. Generate a chest X-ray report with the following sections: EXAMINATION, INDICATION, TECHNIQUE, COMPARISON, FINDINGS, and IMPRESSION. Use standard medical terminology."
```

**完整内容**:
```
You are an expert radiologist. Generate a chest X-ray report with the following sections: EXAMINATION, INDICATION, TECHNIQUE, COMPARISON, FINDINGS, and IMPRESSION. Use standard medical terminology.
```

---

### 4. Few-shot 版（带示例）

**特点**:
- 提供示例报告格式
- 最适合没见过 MIMIC-CXR 的模型
- 占用更多 token

**使用**:
创建自定义 prompt 文件 `prompt_with_example.txt`:
```
You are an expert radiologist. Generate a chest X-ray report following this format:

EXAMPLE FORMAT:
---
EXAMINATION: CHEST (PA AND LAT)

INDICATION: Shortness of breath

TECHNIQUE: Frontal and lateral views of the chest

COMPARISON: None

FINDINGS: The lungs are clear bilaterally without focal consolidation, pleural effusion, or pneumothorax. The cardiac silhouette is normal in size. The mediastinal contours are unremarkable. No acute osseous abnormality is identified.

IMPRESSION: No acute cardiopulmonary process.
---

Now analyze the provided chest X-ray image(s) and generate a similar comprehensive report:
```

然后运行:
```bash
cd data_processing
python convert_mimic_to_sharegpt.py \
  --instruction "$(cat prompt_with_example.txt)"
```

---

## 🎯 不同模型推荐

### GPT-4V / GPT-4o
- **推荐**: 详细版或中等版
- **理由**: GPT-4 理解能力强，能很好地遵循复杂指令
```bash
python convert_mimic_to_sharegpt.py  # 使用默认详细版
```

### Claude 3.5 Sonnet / Opus
- **推荐**: 详细版
- **理由**: Claude 擅长遵循结构化指令
```bash
python convert_mimic_to_sharegpt.py  # 使用默认详细版
```

### Gemini Pro Vision
- **推荐**: 详细版 + Few-shot
- **理由**: Gemini 可能需要更多上下文
```bash
# 使用详细版 + 添加示例
python convert_mimic_to_sharegpt.py
```

### Qwen-VL / InternVL / LLaVA
- **推荐**: 中等版或简化版
- **理由**: 开源模型可能在医学领域微调过
```bash
python convert_mimic_to_sharegpt.py \
  --instruction "Generate a chest X-ray report with EXAMINATION, FINDINGS, and IMPRESSION sections."
```

### 自己微调的模型
- **推荐**: 简化版
- **理由**: 模型已经学会了格式
```bash
python convert_mimic_to_sharegpt.py \
  --instruction "Generate a medical imaging report based on the X-ray image results."
```

---

## 📊 A/B 测试建议

如果要对比不同模型，建议使用**相同的 prompt**：

### 步骤1: 准备统一的测试集
```bash
cd data_processing

# 使用详细版 prompt 生成测试集
python convert_mimic_to_sharegpt.py \
  --output ../dataset/test_unified_prompt.json \
  --test_samples 500
```

### 步骤2: 使用相同的 prompt 测试所有模型

测试集已经包含了详细的 prompt，确保所有模型收到相同的指令。

### 步骤3: 评估对比

使用相同的 judge 模型评估所有生成结果：
```bash
cd evaluation

# 评估模型 A
python evaluate_and_judge.py \
  --test_data ../dataset/test_unified_prompt.json \
  --pred_model "Model-A"

# 评估模型 B
python evaluate_and_judge.py \
  --test_data ../dataset/test_unified_prompt.json \
  --pred_model "Model-B"
```

---

## 🔧 自定义 Prompt

### 方法1: 修改配置文件
编辑 `config.yaml`:
```yaml
data_processing:
  default_instruction: |
    你的自定义 prompt...
```

### 方法2: 命令行参数
```bash
python convert_mimic_to_sharegpt.py \
  --instruction "Your custom prompt here"
```

### 方法3: 从文件读取
```bash
python convert_mimic_to_sharegpt.py \
  --instruction "$(cat your_prompt.txt)"
```

---

## 💡 Prompt 设计原则

### 1. 明确角色
✅ "You are an expert radiologist."
❌ "Analyze this image."

### 2. 指定格式
✅ "Generate a report with EXAMINATION, FINDINGS, and IMPRESSION sections."
❌ "Generate a report."

### 3. 提供具体要求
✅ "Use precise medical terminology and standard radiological language."
❌ "Write professionally."

### 4. 包含示例（可选）
✅ 提供一个完整的示例报告
❌ 只说"类似这样的格式"

### 5. 避免歧义
✅ "Describe medical devices with exact positioning."
❌ "Describe devices."

---

## 📖 相关文档

- **数据处理**: [data_processing/README.md](data_processing/README.md)
- **配置系统**: [CONFIG.md](CONFIG.md)
- **评估系统**: [evaluation/README.md](evaluation/README.md)

---

## 🐛 常见问题

### Q: 不同模型生成的格式不一致怎么办？
A: 使用**详细版 prompt** 并在 IMPRESSION 部分添加示例。

### Q: 某些模型忽略了某些 section 怎么办？
A: 在 prompt 中强调"Your report **must** include **all** of the following sections"。

### Q: 如何测试 prompt 有效性？
A: 使用少量样本（--max_samples 10）快速测试，检查生成格式是否符合预期。

---

## ✅ Prompt 验证清单

在使用新 prompt 之前，检查：

- [ ] 是否明确指定了角色（radiologist）？
- [ ] 是否列出了所有必需的 sections？
- [ ] 是否提供了每个 section 应该包含什么内容的说明？
- [ ] 是否指定了语气和专业水平？
- [ ] 是否提供了示例（如果模型不熟悉格式）？
- [ ] 是否测试过至少 3-5 个样本确认格式正确？
