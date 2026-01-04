# 代码重构说明

## 📋 重构内容

已完成代码重构，将核心模块从 `scripts/` 目录移动到 `src/` 目录，符合项目结构设计规范。

**重构原则**：
- `src/` - 源代码和说明文档（可复用的核心代码）
- `scripts/` - 数据处理和模型训练的脚本（一次性运行）

## 🔄 文件移动

### 核心模块（已移动到 `src/`）

1. **RAG查询模块**
   - `scripts/rag/rag_query.py` → `src/rag/query.py`
   - 功能：Query改写、向量化、搜索、重排序、答案生成

2. **意图识别模块**
   - `scripts/intent/intent_classification.py` → `src/intent/classification.py`
   - 功能：意图识别（CoT推理）、路由决策

3. **查询路由模块**
   - `scripts/intent/query_router.py` → `src/intent/router.py`
   - 功能：整合意图识别和RAG查询

### 脚本文件（保留在 `scripts/`）

以下文件保留在 `scripts/` 目录，作为一次性运行的脚本：

- `scripts/rag/build_vector_db.py` - 向量库生成脚本
- `scripts/qa_generation/generate_qa_pairs.py` - QA对生成脚本
- `scripts/data_import/import_customer_data.py` - 数据导入脚本
- `scripts/data_processing/` - 数据处理脚本

### 文档移动

以下文档已移动到 `src/` 目录：

- `scripts/rag/README_RAG查询说明.md` → `src/rag/README.md`
- `scripts/intent/README_意图识别说明.md` → `src/intent/README.md`

## 📝 导入路径更新

### 新的导入方式

```python
# RAG查询
from src.rag.query import rag_query
# 或
from src.rag import rag_query

# 意图识别
from src.intent.classification import classify_intent, route_query
# 或
from src.intent import classify_intent, route_query

# 查询路由
from src.intent.router import route_and_query
# 或
from src.intent import route_and_query
```

### 旧导入方式（已更新）

所有 `from scripts.rag.rag_query` 和 `from scripts.intent` 的导入已更新为 `from src.rag.query` 和 `from src.intent`。

## 🗂️ 目录结构

```
项目根目录/
├── src/                          # 源代码和说明文档（可复用）
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── query.py              # RAG查询核心功能
│   │   └── README.md             # 使用说明
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── classification.py     # 意图识别核心功能
│   │   ├── router.py             # 查询路由功能
│   │   └── README.md             # 使用说明
│   └── utils/                    # 工具函数（待实现）
│
└── scripts/                       # 脚本（一次性运行）
    ├── rag/
    │   └── build_vector_db.py     # 向量库生成脚本
    ├── qa_generation/
    │   └── generate_qa_pairs.py  # QA对生成脚本
    └── data_processing/           # 数据处理脚本
```

## ✅ 已更新的文件

1. **核心模块**
   - `src/rag/query.py` - 新建
   - `src/intent/classification.py` - 新建
   - `src/intent/router.py` - 新建
   - `src/rag/__init__.py` - 新建
   - `src/intent/__init__.py` - 新建

2. **导入路径更新**
   - `src/intent/router.py` - 已更新导入路径

3. **文档移动和更新**
   - `scripts/rag/README_RAG查询说明.md` → `src/rag/README.md`（已移动并更新）
   - `scripts/intent/README_意图识别说明.md` → `src/intent/README.md`（已移动并更新）
   - `项目结构说明.md` - 已更新目录结构说明

4. **清理冗余文件**
   - 删除 `scripts/rag/rag_query.py`（冗余包装文件）
   - 删除 `scripts/intent/intent_classification.py`（冗余包装文件）
   - 删除 `scripts/intent/query_router.py`（已在src目录）
   - 删除 `scripts/intent/` 目录（已清空）

## 🚀 使用方法

### 方式1：作为模块导入（推荐）

```python
# 导入RAG查询
from src.rag.query import rag_query

# 导入意图识别
from src.intent.classification import classify_intent
from src.intent.router import route_and_query
```

### 方式2：使用Python模块方式

```bash
# 使用-m参数运行模块
python -m src.rag.query "查询问题"
python -m src.intent.classification "查询问题"
python -m src.intent.router "查询问题"
```

## ⚠️ 注意事项

1. **导入路径**：所有新代码应使用 `from src.xxx` 导入，而不是 `from scripts.xxx`
2. **项目结构**：
   - `src/` - 源代码和说明文档（可复用的核心代码）
   - `scripts/` - 数据处理和模型训练的脚本（一次性运行）
3. **文档位置**：所有模块的使用说明文档在对应的 `src/` 子目录下
4. **代码清晰**：已删除所有冗余的包装文件，保持代码结构清晰简单

## 🔍 验证重构

运行以下命令验证重构是否成功：

```bash
# 测试RAG查询
python -m src.rag.query "小额贷款公司的注册资本要求是什么？"

# 测试意图识别
python -m src.intent.classification "如何在系统中查询客户授信额度？"

# 测试查询路由
python -m src.intent.router "最新的LPR是多少？"
```

## 📚 相关文档

- `项目结构说明.md` - 项目目录结构说明
- `scripts/rag/README_RAG查询说明.md` - RAG查询使用说明
- `scripts/intent/README_意图识别说明.md` - 意图识别使用说明

