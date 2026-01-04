# QA对生成脚本使用说明

## 脚本说明

本目录包含两个QA对生成脚本：

### 1. `generate_qa_from_docs.py`（推荐）

**功能**：从解析后的文档生成QA对，支持百炼API和本地模型

**特点**：
- 支持百炼（DashScope）API和本地模型两种模式
- 使用 `config/prompts.py` 中的提示词配置
- 自动按条款/章节/语义块切分文档
- 生成完整的metadata信息

**使用方法**：
```bash
# 使用百炼API（默认）
export DASHSCOPE_API_KEY=your-api-key
python scripts/qa_generation/generate_qa_from_docs.py

# 使用本地模型
export LLM_MODE=local
export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
python scripts/qa_generation/generate_qa_from_docs.py
```

**详细说明**：请参考脚本内的注释和配置说明。

### 2. `generate_qa_pairs.py`

**功能**：从解析后的文档中生成QA对（使用Ollama或其他API）

1. **提取条款级别功能点**：从Markdown文档中提取条款级别的段落作为答案（A）
2. **生成问题**：使用大模型API根据答案生成对应的问题（Q）
3. **分类保存**：按域（policy/system）分类保存QA对

## 使用方法

### 1. 安装依赖

```bash
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可选：如果需要CSV格式输出
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置大模型API

#### 方式1：使用Ollama（推荐，本地大模型）

1. 安装并启动Ollama：
   ```bash
   # 下载Ollama: https://ollama.ai
   # 启动服务后，下载模型
   ollama pull qwen2.5:7b
   ```

2. 设置环境变量（可选，默认值已配置）：
   ```bash
   export LLM_API_URL=http://localhost:11434/api/generate
   export LLM_MODEL=qwen2.5:7b
   ```

#### 方式2：使用OpenAI兼容API

```bash
export LLM_API_URL=https://api.openai.com/v1/chat/completions
export LLM_API_KEY=your-api-key
export LLM_MODEL=gpt-3.5-turbo
```

#### 方式3：使用其他本地大模型服务

修改脚本中的 `call_llm_api` 函数，适配你的API格式。

### 3. 运行脚本

```bash
# 运行 generate_qa_pairs.py
python scripts/qa_generation/generate_qa_pairs.py

# 或运行 generate_qa_from_docs.py（推荐）
python scripts/qa_generation/generate_qa_from_docs.py
```

## 输出结果

### 文件格式

1. **JSON格式**：`data/qa_pairs/{domain}/qa_pairs.json`
   ```json
   [
     {
       "id": "policy_0001",
       "domain": "policy",
       "question": "小额贷款公司的注册资本要求是什么？",
       "answer": "第五条设立小额贷款公司应符合...",
       "created_at": "2025-12-29T20:00:00"
     }
   ]
   ```

2. **CSV格式**（如果安装了pandas）：`data/qa_pairs/{domain}/qa_pairs.csv`

### QA对数量

根据需求配置：

- **政策类**：50~100条
- **系统功能类**：50~80条
- **客户分析类**：10~20条（暂未实现）

## 条款提取规则

脚本会识别以下条款标识：

1. `第一条`、`第二条` 等（第X条格式）
2. `第一章`、`第二章` 等（章节格式）
3. `（一）`、`（二）` 等（中文括号编号）
4. `(一)`、`(二)` 等（英文括号编号）
5. `1.`、`2.` 等（数字编号，需上下文判断）

## 问题生成提示词

脚本会根据域类型使用不同的提示词：

- **政策类**：生成业务人员可能问的政策问题
- **系统功能类**：生成操作相关的问题
- **客户分析类**：生成客户分析相关的问题

## 故障排除

### 问题1：API调用失败

**解决方案**：
- 检查大模型服务是否启动
- 检查API地址和端口是否正确
- 检查网络连接

### 问题2：生成的问题质量不高

**解决方案**：
- 尝试使用更大的模型（如 qwen2.5:14b）
- 调整提示词模板
- 手动筛选和编辑生成的QA对

### 问题3：提取的条款太少

**解决方案**：
- 检查文档格式是否正确
- 调整 `extract_clause_level_content` 函数中的正则表达式
- 手动添加更多条款标识模式

## 注意事项

1. **条款级别**：重点是"条款级别"，不是每句话都做QA
2. **质量检查**：生成后建议人工检查QA对的质量
3. **去重处理**：相同或相似的答案可能生成相似的问题，需要去重
4. **API费用**：如果使用云端API，注意控制调用次数和费用

## 后续优化建议

1. 添加QA对去重功能
2. 添加质量评分机制
3. 支持批量生成和增量更新
4. 添加QA对编辑和审核功能

