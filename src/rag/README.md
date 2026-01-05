# RAG查询模块使用说明

## 📋 功能说明

RAG查询模块实现了完整的检索增强生成流程：

1. **Query改写**：使用大模型将用户的自然语言问题改写为更适合向量检索的查询
2. **向量化**：将改写后的查询转换为embedding向量
3. **向量搜索**：在Elasticsearch中搜索相似文档
4. **重排序**：对搜索结果进行重排序（可配置）
5. **答案生成**：使用大模型基于检索结果生成最终答案

## 🔧 配置说明

### 1. RAG配置（`config/rag_config.py`）

```python
RAG_CONFIG = {
    'top_k': 10,                    # 初始检索返回的文档数量
    'rerank_top_k': 5,              # 重排序后返回的文档数量
    'min_score': 0.3,               # 最小相似度分数阈值
    'enable_rewrite': True,          # 是否启用query改写
    'enable_rerank': True,           # 是否启用重排序
    'rerank_method': 'similarity',  # 重排序方法：'similarity' 或 'bm25'
    'enable_streaming': True,        # 是否启用流式输出（默认True）
                                     # 仅对最后一次LLM总结结果调用使用流式输出，提升用户体验
}
```

### 2. 大模型配置

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

### 3. Embedding模型配置

```bash
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
export EMBEDDING_BATCH_SIZE=4
export FORCE_CPU=false  # 如果使用CPU，设为true
```

## 🚀 使用方法

### 方式1：作为模块导入使用

```python
from src.rag.query import rag_query

# 执行RAG查询
result = rag_query(
    query="小额贷款公司的注册资本要求是什么？",
    domain='policy',              # 'policy' 或 'system'
    role='客户经理',              # 用户角色（用于权限过滤）
    enable_rewrite=True,          # 是否启用query改写
    enable_rerank=True,           # 是否启用重排序
    filters=None                  # 额外的过滤条件
)

# 获取结果
answer = result['answer']                    # 最终答案
rewritten_query = result['rewritten_query']  # 改写后的查询
results = result['results']                  # 检索到的文档列表
```

### 方式2：命令行测试

```bash
# 基本查询
python -m src.rag.query "小额贷款公司的注册资本要求是什么？"

# 指定域类型
python -m src.rag.query "如何查询客户授信额度？" --domain system

# 指定用户角色
python -m src.rag.query "最新的LPR是多少？" --role 团队负责人

# 禁用query改写
python -m src.rag.query "查询政策" --no-rewrite

# 禁用重排序
python -m src.rag.query "查询政策" --no-rerank
```

## 📊 返回结果格式

```python
{
    'answer': '生成的最终答案',
    'query': '原始查询',
    'rewritten_query': '改写后的查询（如果启用了改写）',
    'results': [
        {
            'content': '文档内容',
            'score': 0.85,  # 相似度分数
            'metadata': {
                'domain': '政策',
                'doc_type': '监管政策',
                'source': '国家金融监督管理总局',
                'region': '全国',
                'publish_date': '2023-10-01',
                'doc_id': 'NFRA_2023_XX'
            },
            'chunk_id': 'policy_xxxx_chunk_0001'
        },
        # ... 更多结果
    ],
    'domain': 'policy'
}
```

## 🔍 工作流程

```
用户查询
    ↓
[步骤1] Query改写（可选，调用大模型）
    ↓
改写后的查询
    ↓
[步骤2] 向量化（使用Embedding模型）
    ↓
查询向量
    ↓
[步骤3] Elasticsearch向量搜索
    ↓
检索结果（top_k条）
    ↓
[步骤4] 重排序（可选）
    ↓
重排序后的结果（rerank_top_k条）
    ↓
[步骤5] 大模型生成最终答案
    ↓
最终答案
```

## ⚙️ 配置重排序

### 方法1：修改配置文件

编辑 `config/rag_config.py`：

```python
RAG_CONFIG = {
    'enable_rerank': True,           # 启用/禁用重排序
    'rerank_method': 'similarity',  # 重排序方法
    'rerank_top_k': 5,              # 重排序后保留的文档数
}
```

### 方法2：代码中动态配置

```python
from config.rag_config import update_rag_config

# 禁用重排序
update_rag_config(enable_rerank=False)

# 修改重排序方法
update_rag_config(rerank_method='bm25')

# 修改重排序后保留的文档数
update_rag_config(rerank_top_k=10)
```

## 🌊 流式输出配置

流式输出功能默认开启，仅对最后一次LLM总结结果调用使用流式输出，提升用户体验。

### 配置方式

编辑 `config/rag_config.py`：

```python
RAG_CONFIG = {
    'enable_streaming': True,        # 是否启用流式输出（默认True）
                                     # 仅对最后一次LLM总结结果调用使用流式输出
}
```

### 说明

- **流式输出位置**：仅在 `generate_answer()` 函数调用LLM生成最终答案时使用流式输出
- **其他LLM调用**：query改写、意图分类等其他LLM调用不使用流式输出
- **支持情况**：
  - ✅ **百炼API（dashscope）**：完全支持流式输出
  - ⚠️ **本地LLM（transformers）**：暂不支持真正的流式输出，会降级为普通调用

## 📝 权限控制

根据用户角色自动过滤文档：

- **客户经理**：只能查询自己权限范围内的文档
- **团队负责人**：可以查询所有域的文档
- **行长**：可以查询所有域的文档

权限过滤在Elasticsearch查询时自动应用。

## 🐛 常见问题

### 1. Query改写失败

**问题**：Query改写返回空或错误

**解决**：
- 检查大模型配置是否正确
- 如果改写失败，会自动使用原查询
- 可以通过 `enable_rewrite=False` 禁用改写

### 2. 向量搜索无结果

**问题**：搜索不到相关文档

**解决**：
- 检查Elasticsearch服务是否正常运行
- 检查索引是否存在：`curl https://localhost:9200/bank_credit_policy/_count`
- 降低 `min_score` 阈值
- 增加 `top_k` 数量

### 3. 答案生成失败

**问题**：大模型生成答案时出错

**解决**：
- 检查大模型配置
- 检查API密钥是否正确
- 查看错误日志

## 📚 相关文件

- `src/rag/query.py` - RAG查询主模块
- `scripts/rag/build_vector_db.py` - 向量库生成脚本
- `config/rag_config.py` - RAG配置
- `config/prompts.py` - 提示词配置
- `config/elasticsearch.py` - Elasticsearch配置

## 🔄 下一步

完成RAG查询模块后，可以：

1. 集成到Web界面（Streamlit/Dash）
2. 添加意图识别模块
3. 添加客户预测模块
4. 实现完整的业务系统

