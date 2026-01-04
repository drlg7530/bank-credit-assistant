# 系统提示词配置说明

## 📋 概述

`config/prompts.py` 文件统一管理项目中所有大模型相关的系统提示词模板。

## 📁 文件结构

### 1. QA对生成提示词

用于从文档内容生成问答对：

- **`QA_GENERATION_PROMPT_POLICY`** - 政策类QA对生成提示词
- **`QA_GENERATION_PROMPT_SYSTEM`** - 系统功能类QA对生成提示词

### 2. RAG系统提示词（预留）

用于基于检索到的文档片段生成答案：

- **`RAG_QUERY_PROMPT`** - 通用RAG查询提示词
- **`RAG_POLICY_QUERY_PROMPT`** - 政策查询专用提示词
- **`RAG_SYSTEM_QUERY_PROMPT`** - 系统功能查询专用提示词

### 3. 意图识别提示词（预留）

用于判断用户问题意图：

- **`INTENT_CLASSIFICATION_PROMPT`** - 意图分类提示词

### 4. 预测模型解释提示词（预留）

用于解释预测结果：

- **`PREDICTION_EXPLANATION_PROMPT`** - 客户贷款意向预测结果解释提示词

## 🔧 使用方法

### 方法1：使用辅助函数（推荐）

```python
from config.prompts import get_qa_generation_prompt, get_rag_query_prompt

# 获取QA生成提示词
prompt_template = get_qa_generation_prompt('policy')
prompt = prompt_template.format(answer="政策条款内容...")

# 获取RAG查询提示词（基于检索到的文档片段生成答案）
rag_prompt = get_rag_query_prompt('policy')
final_prompt = rag_prompt.format(context="检索到的文档内容...", question="用户问题...")
```

### 方法2：直接导入使用

```python
from config.prompts import QA_GENERATION_PROMPT_POLICY

prompt = QA_GENERATION_PROMPT_POLICY.format(answer="政策条款内容...")
```

## 📝 提示词模板格式

所有提示词模板使用 Python 的 `.format()` 方法进行变量填充：

- 使用 `{变量名}` 作为占位符
- 调用 `.format(变量名=值)` 填充变量

示例：
```python
template = "你好，{name}！今天是{date}。"
result = template.format(name="张三", date="2025-01-01")
```

## ✏️ 修改提示词

如需修改提示词，直接编辑 `config/prompts.py` 文件：

1. 找到对应的提示词常量
2. 修改提示词内容
3. 保持 `{变量名}` 占位符格式不变
4. 保存文件即可生效

## 📌 注意事项

1. **占位符格式**：必须使用 `{变量名}` 格式，不要使用其他格式（如 `%s`、`{}` 等）
2. **变量命名**：使用有意义的变量名，便于理解和使用
3. **注释说明**：新增提示词时，请添加注释说明其用途和参数
4. **向后兼容**：修改现有提示词时，注意保持变量名不变，避免影响已有代码

## 🔄 扩展提示词

如需添加新的提示词：

1. 在 `config/prompts.py` 中添加新的常量
2. 添加注释说明用途和参数
3. 如需辅助函数，在文件末尾添加相应的 `get_xxx_prompt()` 函数
4. 更新本文档说明

示例：
```python
# 新功能提示词
NEW_FEATURE_PROMPT = """你是一位专家。请根据以下内容回答问题。

内容：
{content}

问题：
{question}

请回答："""
```

## 📚 相关文件

- `scripts/qa_generation/generate_qa_pairs.py` - 使用QA生成提示词
- `src/rag/` - 未来将使用RAG查询提示词
- `src/intent/` - 未来将使用意图识别提示词

