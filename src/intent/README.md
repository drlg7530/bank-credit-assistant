# 问题理解与能力路由模块使用说明

## 📋 功能说明

问题理解与能力路由模块实现了智能查询路由功能：

1. **问题理解**：判断用户输入是否包含多个语义上独立的问题，支持问题拆分
2. **意图识别**：使用大模型（CoT思维链推理）分析每个问题的意图类型
3. **能力路由**：根据意图类型选择对应的系统能力（RAG或预测模块）
4. **查询执行**：自动调用相应的处理模块并返回结果

## 🎯 意图类型

系统支持4种意图类型：

### 1. policy_query（政策查询）
- **特征**：询问政策规定、监管要求、合规标准等
- **示例**：
  - "小额贷款公司的注册资本要求是什么？"
  - "最新的LPR是多少？"
  - "普惠保险考核指标是什么？"
- **路由**：RAG系统（政策域）

### 2. system_query（系统操作）
- **特征**：询问系统功能、操作步骤、如何使用等
- **示例**：
  - "如何在系统中查询客户的授信额度？"
  - "如何发起一个新贷款申请？"
  - "如何查询贷款申请的风控评分？"
- **路由**：RAG系统（系统域）

### 3. customer_analysis（客户分析）
- **特征**：询问客户风险、贷款意向、数据分析等
- **示例**：
  - "这个客户的贷款意向如何？"
  - "分析客户张三的风险情况"
  - "预测客户李四的贷款需求"
- **路由**：预测模块（待实现）

### 4. general（一般性问题）
- **特征**：问候、闲聊、无法明确分类的问题
- **示例**：
  - "你好"
  - "谢谢"
  - "系统介绍"
- **路由**：通用回复

## 🔧 配置说明

### 大模型配置

支持两种模式：

#### 方式1：百炼API（推荐）

```bash
export LLM_MODE=bailian
export DASHSCOPE_API_KEY=your-api-key
export BAILIAN_MODEL=qwen-plus
```

#### 方式2：本地模型

```bash
export LLM_MODE=local
export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export FORCE_CPU=false  # 如果使用CPU，设为true
```

## 🚀 使用方法

### 方式1：单独使用意图识别

```python
from src.intent.classification import classify_intent, route_query

# 进行问题理解与意图识别
intent_result = classify_intent(
    question="小额贷款公司的注册资本要求是什么？",
    use_cot=True  # 使用CoT思维链推理
)

# 获取路由信息
route_info = route_query(intent_result, question)

print(f"意图类型: {intent_result.intent.value}")
print(f"路由目标: {intent_result.route_to}")
print(f"推理过程: {intent_result.reasoning}")
```

### 方式2：使用查询路由（推荐）

```python
from src.intent.router import route_and_query

# 完整的查询流程：问题理解 → 意图识别 → 路由 → 查询
result = route_and_query(
    question="小额贷款公司的注册资本要求是什么？",
    role='客户经理',
    use_cot=True,          # 使用CoT思维链推理
    enable_rewrite=True,   # 启用query改写
    enable_rerank=True     # 启用重排序
)

# 获取结果
print(result['answer'])
print(result['intent'])
print(result['route_to'])
```

### 方式3：命令行测试

```bash
# 基本测试
python -m src.intent.classification "小额贷款公司的注册资本要求是什么？"

# 禁用CoT
python -m src.intent.classification "查询政策" --no-cot

# 使用查询路由（完整流程）
python -m src.intent.router "如何在系统中查询客户授信额度？" --role 客户经理
```

## 📊 返回结果格式

### 意图识别结果

```python
IntentResult(
    intent=IntentType.POLICY_QUERY,  # 意图类型
    confidence=0.8,                  # 置信度（0-1）
    reasoning="...",                 # CoT推理过程
    route_to='rag_policy'            # 路由目标
)
```

### 查询路由结果

```python
{
    'intent': 'policy_query',           # 意图类型
    'intent_confidence': 0.8,            # 意图识别置信度
    'intent_reasoning': '...',          # CoT推理过程
    'route_to': 'rag_policy',           # 路由目标
    'question': '用户问题',              # 原始问题
    'answer': '生成的答案',              # 最终答案
    'module': 'rag',                    # 处理模块
    'domain': 'policy',                  # 域类型（如果是RAG）
    'rewritten_query': '...',           # 改写后的查询（如果启用）
    'search_results': [...]             # 检索结果（如果是RAG）
}
```

## 🧠 问题拆分与意图识别

### 问题拆分规则

模块会智能判断用户输入是否包含多个独立问题：
- **单一问题**：如果只有一个清晰问题，保持原问题不拆分
- **多问题拆分**：只有在用户明确提出多个不同目标时，才拆分为多个子问题
- **输出格式**：使用JSON数组，每个元素包含 `sub_question`、`intent`、`module`

### CoT思维链推理

问题理解与意图识别使用CoT（Chain of Thought）思维链推理，让模型逐步分析问题：

```
【步骤1：问题分析】
- 关键词提取
- 业务领域
- 查询对象

【步骤2：意图判断】
- 匹配意图类型定义
- 分析问题特征

【步骤3：推理过程】
- 说明判断理由
- 匹配的意图类型
- 判断依据

【步骤4：最终输出】
- 返回意图类型
```

### CoT示例

**输入问题**："小额贷款公司的注册资本要求是什么？"

**CoT推理过程**：
```
【步骤1：问题分析】
- 关键词提取：小额贷款公司、注册资本、要求
- 业务领域：政策法规
- 查询对象：监管政策

【步骤2：意图判断】
- 特征：询问政策规定、监管要求
- 匹配意图类型：policy_query（政策查询类）

【步骤3：推理过程】
- 问题特征：询问具体的政策规定和监管要求
- 匹配的意图类型：policy_query
- 判断依据：问题明确涉及监管政策的具体要求

【步骤4：最终输出】
policy_query
```

## ⚙️ 配置选项

### 启用/禁用CoT

```python
# 启用CoT（默认）
intent_result = classify_intent(question, use_cot=True)

# 禁用CoT（使用简单分类）
intent_result = classify_intent(question, use_cot=False)
```

### 降级处理

如果大模型调用失败，系统会自动使用基于关键词的降级处理：

```python
# 自动降级（无需手动调用）
# 系统会根据关键词匹配进行意图识别
```

## 🔄 工作流程

```
用户问题
    ↓
[步骤1] 问题理解与拆分
    ├─ 判断是否包含多个独立问题
    ├─ 单一问题：保持原问题
    └─ 多问题：拆分为子问题
    ↓
[步骤2] 意图识别（CoT推理）
    ├─ 为每个问题分析意图类型
    └─ 选择对应的系统能力
    ↓
[步骤3] 能力路由
    ├─ policy_query → policy_rag（政策域RAG）
    ├─ system_query → system_rag（系统域RAG）
    ├─ customer_analysis → prediction（预测模块）
    └─ general → general_response（通用回复）
    ↓
[步骤4] 执行查询
    ↓
返回结果
```

## 🐛 常见问题

### 1. 问题拆分不准确

**问题**：应该拆分的问题没有拆分，或不应该拆分的问题被拆分

**解决**：
- 检查用户问题是否明确包含多个独立目标
- 单一问题不应被拆分
- 查看模块输出的JSON数组结构

### 2. 意图识别不准确

**问题**：识别出的意图类型不正确

**解决**：
- 检查CoT推理过程，查看判断依据
- 如果置信度较低，可以手动指定意图类型
- 优化提示词，添加更多示例

### 3. CoT推理过程不完整

**问题**：模型返回的推理过程不完整

**解决**：
- 检查大模型配置是否正确
- 尝试使用更大的模型（如qwen-max）
- 增加max_tokens参数

### 4. 路由错误

**问题**：路由到了错误的模块

**解决**：
- 检查意图识别结果
- 查看路由映射配置
- 手动指定domain参数

## 📚 相关文件

- `src/intent/classification.py` - 问题理解与意图识别核心模块
- `src/intent/router.py` - 查询路由模块
- `config/prompts.py` - 问题理解与能力路由提示词（CoT）
- `src/rag/query.py` - RAG查询模块

## 🔄 下一步

完成问题理解与能力路由模块后，可以：

1. 集成到Web界面（Streamlit/Dash）
2. 实现预测模块（customer_analysis路由）
3. 优化问题拆分逻辑，提升多问题识别准确率
4. 优化CoT提示词，提升意图识别准确率
5. 添加问题理解和路由的历史记录和统计

