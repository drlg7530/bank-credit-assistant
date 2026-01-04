# RAG向量库生成说明

## 📋 功能说明

本目录包含RAG系统的核心脚本：

### 1. QA对生成脚本（已移至 `scripts/qa_generation/generate_qa_from_docs.py`）
从解析后的文档生成QA对，包括：
1. 读取解析后的文档（`data/parsed/`）
2. 按条款/章节/语义块切分（不使用LLM）
3. 使用LLM生成QA对（Q为大模型生成、A为政策或操作手册的功能点）
4. 保存QA对到`data/qa_pairs/`目录

**注意**：QA对生成脚本已移至 `scripts/qa_generation/` 目录，请参考该目录下的说明文档。

### 2. `build_vector_db.py` - 向量库生成脚本
从QA对生成Elasticsearch索引库，包括：
1. 读取QA对JSON文件（`data/qa_pairs/`）
2. 将QA对转换为文档块格式
3. 生成Embedding向量
4. 保存到Elasticsearch（**单个混合索引，同时包含向量和文本字段**）

**数据存储方式**：
- **单个混合索引**：`{index_name}` 同时包含向量字段和文本字段
- **支持ES原生Hybrid Search**：在同一个索引中同时进行BM25文本搜索和KNN向量搜索
- **优势**：简化索引管理，提升搜索效果，ES自动合并两种搜索结果

**注意**：数据仅保存在Elasticsearch中，不再保存本地文件。

## 🔧 使用前准备

### 1. 安装Elasticsearch

#### 方式1：使用Docker（推荐）

```bash
# 拉取Elasticsearch镜像
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 运行Elasticsearch容器
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### 方式2：本地安装

下载并安装Elasticsearch：https://www.elastic.co/downloads/elasticsearch

### 2. 安装Python依赖库

```bash
# 基础依赖
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# Embedding模型（ModelScope）
pip install modelscope transformers torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# Elasticsearch客户端
pip install elasticsearch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可选：IK分词器（用于中文分词，提升搜索效果）
# 需要手动安装到Elasticsearch，参考：https://github.com/medcl/elasticsearch-analysis-ik
```

### 3. 配置Elasticsearch连接

编辑 `config/elasticsearch.py` 文件，修改连接配置：

```python
ES_CONFIG = {
    'hosts': ['localhost:9200'],  # ES服务器地址
    'timeout': 30,
    'max_retries': 3,
}
```

### 4. 准备数据

确保以下QA对JSON文件存在：
- `data/qa_pairs/policy/policy_qa_pairs.json` - 政策类QA对数据
- `data/qa_pairs/system/system_qa_pairs.json` - 系统功能类QA对数据

QA对JSON文件格式：
```json
[
  {
    "id": "policy_0001",
    "domain": "policy",
    "question": "问题内容",
    "answer": "答案内容",
    "created_at": "2025-12-30T01:21:13.609472"
  },
  ...
]
```

## 🚀 使用方法

### 步骤1：生成QA对（可选，如果已有QA对可跳过）

如果还没有QA对数据，需要先从解析后的文档生成QA对：

```bash
# 方式1：直接运行脚本
python scripts/qa_generation/generate_qa_from_docs.py

# 方式2：在项目根目录运行
cd D:\LLMproject\cursor\myProjects\bank_credit_agent
python scripts/qa_generation/generate_qa_from_docs.py
```

**配置说明**：
- 默认使用百炼API（需要设置 `DASHSCOPE_API_KEY` 环境变量）
- 也可以使用本地模型（设置 `LLM_MODE=local`）

**输出**：
- QA对保存到 `data/qa_pairs/policy/policy_qa_pairs.json`
- QA对保存到 `data/qa_pairs/system/system_qa_pairs.json`

### 步骤2：生成向量库

```bash
# 方式1：直接运行脚本
python scripts/rag/build_vector_db.py

# 方式2：在项目根目录运行
cd D:\LLMproject\cursor\myProjects\bank_credit_agent
python scripts/rag/build_vector_db.py
```

## ⚙️ 配置说明

### QA对生成配置（scripts/qa_generation/generate_qa_from_docs.py）

#### 大模型模式配置

**百炼API模式**（默认，推荐）：
```bash
export LLM_MODE=bailian
export DASHSCOPE_API_KEY=your-api-key
export BAILIAN_MODEL=qwen-plus
```

**本地模型模式**（适用于GPU环境）：
```bash
export LLM_MODE=local
export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
```

#### 提示词配置

脚本使用 `config/prompts.py` 中的提示词：
- `QA_GENERATION_PROMPT_POLICY` - 政策类QA对生成提示词
- `QA_GENERATION_PROMPT_SYSTEM` - 系统功能类QA对生成提示词

### Embedding模型配置（build_vector_db.py）

脚本默认使用ModelScope的 `Qwen/Qwen3-Embedding-0.6B` 模型

可以通过环境变量修改：
```bash
# Windows
set EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
python scripts/rag/build_vector_db.py

# Linux/Mac
export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
python scripts/rag/build_vector_db.py
```

可选模型：
- `Qwen/Qwen3-Embedding-0.6B` - 0.6B参数，速度快，适合开发测试（默认）
- `Qwen/Qwen3-Embedding-8B` - 8B参数，效果好，适合生产环境

注意：模型会自动从ModelScope下载到 `./models/embedding/` 目录

### 批处理配置

可以通过环境变量调整批处理大小（适合低配置机器）：
```bash
# Windows
set EMBEDDING_BATCH_SIZE=2
python scripts/rag/build_vector_db.py

# Linux/Mac
export EMBEDDING_BATCH_SIZE=2
python scripts/rag/build_vector_db.py
```

默认值：4（适合大多数机器）

## 📊 Metadata设计

每个文本块包含以下metadata（符合需求文档设计）：

```json
{
  "domain": "政策",                // 域类型（政策 | 系统）
  "doc_type": "监管政策",          // 文档类型（监管政策 | 内部制度 | 系统说明 | 操作手册）
  "source": "QA对数据",            // 来源（QA对文件路径）
  "region": "全国",                // 地区范围（全国 | 新疆 等）
  "publish_date": "2025-12-30",   // 发布时间（从QA对的created_at提取）
  "status": "生效",                // 状态（生效 | 已失效）
  "doc_id": "policy_policy_0001", // 文档编号
  "role": "客户经理",              // 权限角色（客户经理 | 团队负责人 | 行长，用于查询权限控制）
  "chunk_id": 0,                   // 文本块ID（QA对的索引）
  "qa_id": "policy_0001",          // QA对ID
  "question": "问题内容",          // 原始问题
  "answer": "答案内容",            // 原始答案
  "content": "问题：xxx\n答案：xxx" // 组合内容（用于搜索）
}
```

### Metadata字段说明

1. **domain（域）**：区分政策类和系统功能类
   - `政策` - 政策类QA对
   - `系统` - 系统功能类QA对

2. **doc_type（文档类型）**：文档分类
   - `监管政策` - 政策类QA对（默认）
   - `操作手册` - 系统功能类QA对（默认）

3. **publish_date（发布时间）**：从QA对的created_at字段提取
   - 格式：`YYYY-MM-DD`
   - 如果QA对中没有created_at，默认为 `2024-01-01`

4. **role（权限角色）**：查询权限控制
   - `客户经理` - 客户经理权限（默认）
   - `团队负责人` - 团队负责人权限
   - `行长` - 行长权限
   - 用于在查询时根据用户角色过滤文档

5. **region（地区）**：地区范围
   - `全国` - 全国性政策（默认）

6. **status（状态）**：文档状态
   - `生效` - 当前生效的文档（默认）
   - `已失效` - 已失效的文档

## 📁 输出说明

数据保存在Elasticsearch中，为每个域创建一个混合索引：

### 政策类（policy）
- **`bank_credit_policy`** - 混合索引（同时支持BM25文本搜索和KNN向量搜索）

### 系统功能类（system）
- **`bank_credit_system`** - 混合索引（同时支持BM25文本搜索和KNN向量搜索）

### 混合索引特性

每个混合索引同时包含：
- **向量字段**（`embedding`）：用于KNN向量相似度搜索
- **文本字段**（`content`, `title`, `question`, `answer`）：用于BM25全文搜索
- **支持ES原生Hybrid Search**：在同一个查询中同时使用BM25和向量搜索，ES自动合并结果

### 索引字段说明

每个混合索引包含以下字段：
- `embedding` - 向量字段（dense_vector类型，用于knn搜索）
- `content` - 文本内容（text类型，支持全文搜索，格式：问题：xxx\n答案：xxx）
- `title` - 标题（text类型，使用问题作为标题）
- `question` - 原始问题（text类型，支持全文搜索）
- `answer` - 原始答案（text类型，支持全文搜索）
- `qa_id` - QA对ID（keyword）
- `domain` - 域类型（keyword）
- `doc_type` - 文档类型（keyword）
- `source` - 来源文件路径（keyword）
- `region` - 地区范围（keyword）
- `publish_date` - 发布时间（date）
- `status` - 状态（keyword）
- `doc_id` - 文档ID（keyword）
- `role` - 权限角色（keyword）
- `chunk_id` - 文本块ID（keyword）
- `page_num` - 页码（integer，QA对默认为1）
- `token_count` - Token数量（integer）
- `file_type` - 文件类型（keyword，QA对为json）

## 🔍 查询示例

### 使用Elasticsearch API查询

#### 1. 混合搜索（Hybrid Search：BM25 + 向量，推荐）

```bash
# ES原生Hybrid Search：同时使用BM25和向量搜索
curl -X POST "localhost:9200/bank_credit_policy/_search" -H 'Content-Type: application/json' -d'
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],  # 查询向量
    "k": 10,
    "num_candidates": 100
  },
  "query": {
    "multi_match": {
      "query": "小额贷款公司设立",
      "fields": ["title^2", "content", "question", "answer"],
      "type": "best_fields"
    }
  },
  "size": 10
}'
```

**说明**：ES会自动合并BM25和向量搜索结果，综合评分排序。

#### 2. 纯向量搜索（KNN）

```bash
# 查看索引信息
curl -X GET "localhost:9200/bank_credit_policy/_count"

# 向量搜索（knn查询）
curl -X POST "localhost:9200/bank_credit_policy/_search" -H 'Content-Type: application/json' -d'
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],  # 查询向量
    "k": 10,
    "num_candidates": 100
  },
  "size": 10
}'
```

#### 3. 纯文本搜索（BM25）

```bash
# 文本搜索（BM25算法）
curl -X POST "localhost:9200/bank_credit_policy/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "multi_match": {
      "query": "小额贷款公司设立",
      "fields": ["title^2", "content", "question", "answer"],
      "type": "best_fields"
    }
  },
  "size": 10
}'
```

#### 4. 混合搜索 + metadata过滤

```bash
# Hybrid Search + metadata过滤（根据用户角色和域类型）
curl -X POST "localhost:9200/bank_credit_policy/_search" -H 'Content-Type: application/json' -d'
{
  "knn": {
    "field": "embedding",
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "num_candidates": 100,
    "filter": {
      "bool": {
        "must": [
          {"term": {"domain": "政策"}},
          {"term": {"role": "客户经理"}},
          {"term": {"status": "生效"}}
        ]
      }
    }
  },
  "query": {
    "multi_match": {
      "query": "小额贷款",
      "fields": ["title^2", "content"]
    }
  },
  "size": 10
}'
```

## 📝 注意事项

### QA对生成注意事项

1. **文档准备**：确保 `data/parsed/policy/` 和 `data/parsed/system/` 目录下有解析后的Markdown文档
2. **切分规则**：文档按条款/章节/语义块切分，不使用LLM，纯规则切分
3. **QA对质量**：生成的QA对质量取决于LLM模型和提示词，建议使用较好的模型（如qwen-plus）
4. **API密钥**：使用百炼API时需要设置 `DASHSCOPE_API_KEY` 环境变量

### 向量库生成注意事项

1. **首次运行**：模型会自动下载，可能需要一些时间
2. **内存占用**：处理大量QA对时，注意内存使用
3. **向量维度**：不同模型的向量维度不同，更换模型后需要重新生成向量库
4. **数据来源**：使用QA对JSON文件（`data/qa_pairs/`），不再使用parsed文档
5. **本地缓存**：向量库仅保存在Elasticsearch中，不再保存本地文件

## 🔄 更新向量库

当QA对数据更新后，重新运行脚本即可更新向量库。脚本会：
1. 删除旧混合索引
2. 重新创建混合索引（同时包含向量和文本字段）
3. 重新索引所有QA对数据（单个索引包含向量和文本字段）

## 🐛 常见问题

### 1. Elasticsearch连接失败

**问题**：无法连接到Elasticsearch服务器

**解决**：
- 检查Elasticsearch服务是否启动：`curl http://localhost:9200`
- 检查 `config/elasticsearch.py` 中的连接配置
- 确认防火墙和端口设置

### 2. IK分词器未安装

**问题**：创建索引时提示IK分词器未找到

**解决**：
- 脚本会自动降级到标准分词器
- 如需更好的中文分词效果，安装IK分词器：
  ```bash
  # 进入Elasticsearch安装目录
  ./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.11.0/elasticsearch-analysis-ik-8.11.0.zip
  # 重启Elasticsearch
  ```

### 3. 模型下载失败

**问题**：模型下载很慢或失败

**解决**：
- 使用国内镜像：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载模型到本地，修改模型路径

### 4. 内存不足

**问题**：处理大量QA对时内存不足

**解决**：
- 减少 `EMBEDDING_BATCH_SIZE` 环境变量（默认4，可以改为2）
- 使用更小的embedding模型（如0.6B版本）
- 设置环境变量 `FORCE_CPU=true` 强制使用CPU模式

### 5. 向量维度不匹配

**问题**：更换embedding模型后向量维度变化

**解决**：
- 更新 `config/elasticsearch.py` 中的 `vector_dimension` 配置
- 删除旧索引，重新运行脚本生成向量库（脚本会自动删除旧混合索引）

### 6. 混合搜索效果不佳

**问题**：使用Hybrid Search时结果不理想

**解决**：
- 可以单独使用文本搜索（BM25）或向量搜索（KNN）
- 调整混合搜索中文本搜索和向量搜索的权重（通过调整query和knn的boost参数）
- 检查分词器配置（IK分词器效果更好）
- 确保向量质量和文本字段质量都良好

### 7. QA对文件不存在

**问题**：提示QA对文件不存在

**解决**：
- 确保 `data/qa_pairs/policy/policy_qa_pairs.json` 和 `data/qa_pairs/system/system_qa_pairs.json` 文件存在
- 检查文件路径是否正确
- 确保JSON文件格式正确（数组格式）

## 📚 相关文档

- [Elasticsearch启动说明](./Elasticsearch启动说明.md)
- [测试脚本说明](./test_build_and_query.py) - 用于测试数据保存和查询功能
- [QA对生成脚本](../qa_generation/generate_qa_from_docs.py) - 从解析后的文档生成QA对（已移至 `scripts/qa_generation/` 目录）

## 🔄 完整工作流程

1. **文档解析**：使用MinerU等工具将原始文档解析为Markdown格式，保存到 `data/parsed/`
2. **QA对生成**：运行 `scripts/qa_generation/generate_qa_from_docs.py`，从解析后的文档生成QA对，保存到 `data/qa_pairs/`
3. **向量库生成**：运行 `build_vector_db.py`，从QA对生成Elasticsearch索引库
4. **查询测试**：运行 `test_build_and_query.py`，测试向量库的查询功能
