# Context模块说明文档

## 一、模块概述

Context模块是智能信贷业务辅助系统的上下文记忆管理模块，负责管理对话的会话级记忆（L1）和业务状态记忆（L2），以及Session的生命周期管理。

### 1.1 核心功能

- **Session管理**：基于时间失效的Session管理（30分钟不操作后失效）
- **L1记忆**：会话级记忆，存储每轮对话的query和answer
- **L2记忆**：业务状态记忆，存储结构化的业务操作状态
- **Session记录**：管理Session元信息（标题、创建时间等）
- **历史记录清理**：支持清理用户或Session的历史记录

### 1.2 模块架构

```
src/context/
├── __init__.py              # 模块导出
├── session_manager.py       # Session管理（基于用户ID生成session_id）
├── session_record.py        # Session记录管理（ES存储session元信息）
├── l1_memory.py            # L1会话级记忆（ES存储）
├── l2_memory.py            # L2业务状态记忆（Redis+ES存储）
├── memory_manager.py       # 记忆管理器（统一接口）
└── clear_history.py         # 历史记录清理模块
```

## 二、模块详细说明

### 2.1 Session管理模块 (`session_manager.py`)

#### 功能说明
- 管理Session的生命周期
- Session基于时间失效（默认30分钟不操作后失效）
- 支持Redis和ES两种存储方式

#### 核心类和方法

**SessionManager类**：
```python
class SessionManager:
    @staticmethod
    def create_session(user_id: int = 10000) -> str:
        """创建新session，返回session_id
        格式：sess_{user_id}_{timestamp}
        """
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """获取session信息（从Redis）"""
    
    @staticmethod
    def is_valid(session_id: str) -> bool:
        """检查session是否有效
        - Redis启用：检查Redis中是否存在且未过期
        - Redis未启用：检查ES中的session_record是否存在
        """
    
    @staticmethod
    def refresh_session(session_id: str) -> bool:
        """刷新session过期时间（重置30分钟倒计时）"""
```

**便捷函数**：
```python
def get_or_create_session(user_id: int = 10000, session_id: Optional[str] = None) -> str:
    """获取或创建session
    - 如果提供了session_id且有效，则继续使用该session
    - 如果session_id无效或未提供，则创建新session
    """
```

#### 使用示例
```python
from src.context import get_or_create_session

# 创建新session
session_id = get_or_create_session(user_id=10000)

# 继续使用现有session
session_id = get_or_create_session(user_id=10000, session_id="sess_10000_1705123456")
```

### 2.2 Session记录模块 (`session_record.py`)

#### 功能说明
- 管理Session的元信息（标题、第一个问题、创建时间等）
- 存储在ES中，用于前端显示历史对话列表

#### 核心类和方法

```python
class SessionRecord:
    def __init__(self, es_client: Optional[Elasticsearch] = None):
        """初始化Session记录管理器"""
    
    def create_session_record(self, session_id: str, user_id: int, first_question: str) -> bool:
        """创建session记录
        - 标题使用第一个问题
        """
    
    def get_session_record(self, session_id: str) -> Optional[Dict]:
        """获取session记录"""
    
    def list_session_records(self, user_id: int, limit: int = 50) -> List[Dict]:
        """获取用户的所有session记录列表（按创建时间倒序）"""
    
    def update_session_record(self, session_id: str) -> bool:
        """更新session记录的最后更新时间"""
    
    def delete_session_record(self, session_id: str) -> bool:
        """删除session记录"""
```

#### ES索引配置
- **索引名称**：`bank_credit_session_record`
- **字段设计**：
  ```json
  {
    "session_id": "keyword",
    "user_id": "integer",
    "title": "text",
    "first_question": "text",
    "created_at": "date",
    "updated_at": "date"
  }
  ```

#### 使用示例
```python
from src.context import SessionRecord
from elasticsearch import Elasticsearch

es_client = Elasticsearch(**ES_CONFIG)
session_record = SessionRecord(es_client=es_client)

# 创建session记录
session_record.create_session_record(
    session_id="sess_10000_1705123456",
    user_id=10000,
    first_question="查询押品怎么创建"
)

# 获取session列表
sessions = session_record.list_session_records(user_id=10000, limit=50)
```

### 2.3 L1记忆模块 (`l1_memory.py`)

#### 功能说明
- 存储每轮对话的query和answer
- 存储在ES中，支持审计和历史查询
- 同一session下的多轮对话，使用同一个session_id，turn_id递增

#### 核心类和方法

```python
class L1Memory:
    def __init__(self, es_client: Optional[Elasticsearch] = None):
        """初始化L1记忆管理器"""
    
    def save_user_query(self, session_id: str, turn_id: Optional[int] = None, 
                       content: str = "", user_id: int = 10000) -> Optional[int]:
        """保存用户query到L1
        - 如果turn_id为None，自动获取下一个（基于该session的最大turn_id+1）
        - 如果是第一个问题（turn_id=1），自动创建session记录
        - 返回turn_id
        """
    
    def save_assistant_answer(self, session_id: str, turn_id: Optional[int] = None, 
                             content: str = "") -> Optional[int]:
        """保存助手answer到L1
        - 如果turn_id为None，使用当前session的最大turn_id（同一轮对话的answer应该和query使用相同的turn_id）
        - 返回turn_id
        """
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取会话历史记录
        - 返回该session下的所有对话（按turn_id和时间戳排序）
        - 格式：每轮对话包含user和assistant两条记录
        """
```

#### ES索引配置
- **索引名称**：`bank_credit_l1_memory`
- **字段设计**：
  ```json
  {
    "session_id": "keyword",
    "turn_id": "integer",
    "role": "keyword",
    "content": "text",
    "timestamp": "date"
  }
  ```

#### 使用示例
```python
from src.context import L1Memory
from elasticsearch import Elasticsearch

es_client = Elasticsearch(**ES_CONFIG)
l1_memory = L1Memory(es_client=es_client)

# 保存用户query
turn_id = l1_memory.save_user_query(
    session_id="sess_10000_1705123456",
    content="查询押品怎么创建",
    user_id=10000
)

# 保存助手answer
l1_memory.save_assistant_answer(
    session_id="sess_10000_1705123456",
    turn_id=turn_id,
    content="押品创建流程为：..."
)

# 获取对话历史
history = l1_memory.get_conversation_history(
    session_id="sess_10000_1705123456",
    limit=50
)
```

### 2.4 L2记忆模块 (`l2_memory.py`)

#### 功能说明
- 存储结构化的业务状态记忆
- 支持继承和覆盖规则
- 存储在Redis（快速查询）和ES（审计）中

#### 核心类和方法

```python
class L2Memory:
    def __init__(self, redis_client=None, es_client=None):
        """初始化L2记忆管理器"""
    
    def get_l2_state(self, session_id: str) -> Optional[Dict]:
        """从Redis获取L2状态"""
    
    def update_l2_state(self, session_id: str, intent: str, entities: Dict, 
                       previous_l2: Optional[Dict] = None) -> Dict:
        """更新L2状态（实现继承和覆盖规则）
        - 规则1：intent不同 → 全量覆盖
        - 规则2：intent相同 + 对象未显式出现 → 继承
        - 规则3：对象显式变化 → 覆盖
        - 规则4：阶段只能单向推进
        """
    
    def save_l2_to_es(self, session_id: str, l2_state: Dict) -> bool:
        """保存L2状态到ES（审计）"""
    
    def save_l2_to_redis(self, session_id: str, l2_state: Dict) -> bool:
        """保存L2状态到Redis（快速查询）"""
```

#### 数据结构
```json
{
  "session_id": "sess_001",
  "current_customer_id": null,
  "operation_chain": [
    {
      "intent": "system_operation",
      "active_domain": ["system"],
      "business_object": "押品",
      "operation_stage": "创建",
      "last_action": "查询押品创建流程",
      "last_update": "2026-01-05T14:00:02Z"
    }
  ]
}
```

#### 使用示例
```python
from src.context import L2Memory
import redis
from elasticsearch import Elasticsearch

redis_client = redis.Redis(**REDIS_CONFIG)
es_client = Elasticsearch(**ES_CONFIG)
l2_memory = L2Memory(redis_client=redis_client, es_client=es_client)

# 获取L2状态
previous_l2 = l2_memory.get_l2_state(session_id="sess_10000_1705123456")

# 更新L2状态
entities = {
    'active_domain': ['system'],
    'business_object': '押品',
    'operation_stage': '创建',
    'last_action': '查询押品创建流程',
    'current_customer_id': None
}
l2_state = l2_memory.update_l2_state(
    session_id="sess_10000_1705123456",
    intent="system_operation",
    entities=entities,
    previous_l2=previous_l2
)
```

### 2.5 记忆管理器 (`memory_manager.py`)

#### 功能说明
- 提供统一的接口管理L1和L2记忆
- 单例模式，全局共享

#### 核心类和方法

```python
class MemoryManager:
    def __init__(self, redis_client=None, es_client=None):
        """初始化记忆管理器"""
    
    def get_or_create_session(self, user_id: int = 10000, session_id: Optional[str] = None) -> str:
        """获取或创建session"""
    
    def save_user_query(self, session_id: str, turn_id: Optional[int] = None, 
                       content: str = "", user_id: int = 10000) -> Optional[int]:
        """保存用户query到L1"""
    
    def save_assistant_answer(self, session_id: str, turn_id: Optional[int] = None, 
                             content: str = "") -> Optional[int]:
        """保存助手answer到L1"""
    
    def get_l2_state(self, session_id: str) -> Optional[Dict]:
        """获取L2状态"""
    
    def update_l2_state(self, session_id: str, intent: str, entities: Dict, 
                       previous_l2: Optional[Dict] = None) -> Dict:
        """更新L2状态"""
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> list:
        """获取会话历史记录"""
```

**全局函数**：
```python
def get_memory_manager(redis_client=None, es_client=None) -> MemoryManager:
    """获取全局记忆管理器实例（单例模式）"""
```

#### 使用示例
```python
from src.context import get_memory_manager

# 获取记忆管理器（单例）
memory_manager = get_memory_manager()

# 获取或创建session
session_id = memory_manager.get_or_create_session(user_id=10000)

# 保存用户query
turn_id = memory_manager.save_user_query(
    session_id=session_id,
    content="查询押品怎么创建",
    user_id=10000
)

# 保存助手answer
memory_manager.save_assistant_answer(
    session_id=session_id,
    turn_id=turn_id,
    content="押品创建流程为：..."
)

# 获取对话历史
history = memory_manager.get_conversation_history(session_id=session_id, limit=50)
```

### 2.6 历史记录清理模块 (`clear_history.py`)

#### 功能说明
- 清理指定用户或指定session的历史记录
- 支持清理L1记忆、Session记录、L2记忆

#### 核心函数

```python
def clear_user_history(user_id: int = 10000) -> dict:
    """清理指定用户的所有历史记录
    返回: {
        'success': bool,
        'l1_deleted': int,
        'session_deleted': int,
        'l2_deleted': int,
        'error': str (可选)
    }
    """

def clear_session_history(session_id: str) -> dict:
    """清理指定session的所有历史记录
    返回: {
        'success': bool,
        'l1_deleted': int,
        'session_deleted': int,
        'l2_deleted': int,
        'error': str (可选)
    }
    """
```

#### 使用示例

**Python代码调用**：
```python
from src.context import clear_user_history, clear_session_history

# 清理指定用户的所有历史记录
result = clear_user_history(user_id=10000)
print(f"删除了 {result['l1_deleted']} 条L1记忆记录")
print(f"删除了 {result['session_deleted']} 条Session记录")
print(f"删除了 {result['l2_deleted']} 条L2记忆记录")

# 清理指定session的所有历史记录
result = clear_session_history(session_id="sess_10000_1705123456")
```

**命令行调用**：
```bash
# 清理指定session
python -m src.context.clear_history --session-id sess_10000_1705123456

# 清理指定用户的所有历史记录
python -m src.context.clear_history --user-id 10000
```

## 三、完整使用流程

### 3.1 基本使用流程

```python
from src.context import get_memory_manager

# 1. 获取记忆管理器
memory_manager = get_memory_manager()

# 2. 获取或创建session
session_id = memory_manager.get_or_create_session(user_id=10000)

# 3. 保存用户query
turn_id = memory_manager.save_user_query(
    session_id=session_id,
    content="查询押品怎么创建",
    user_id=10000
)

# 4. 处理查询（意图识别、RAG等）
# ... 业务逻辑 ...

# 5. 保存助手answer
memory_manager.save_assistant_answer(
    session_id=session_id,
    turn_id=turn_id,
    content="押品创建流程为：..."
)

# 6. 获取对话历史
history = memory_manager.get_conversation_history(session_id=session_id)
```

### 3.2 多轮对话流程

```python
# 第一轮对话
session_id = memory_manager.get_or_create_session(user_id=10000)
turn_id_1 = memory_manager.save_user_query(session_id=session_id, content="查询押品怎么创建", user_id=10000)
# ... 处理查询 ...
memory_manager.save_assistant_answer(session_id=session_id, turn_id=turn_id_1, content="押品创建流程为：...")

# 第二轮对话（继续使用同一个session_id）
session_id = memory_manager.get_or_create_session(user_id=10000, session_id=session_id)
turn_id_2 = memory_manager.save_user_query(session_id=session_id, content="那怎么入库呢", user_id=10000)
# ... 处理查询 ...
memory_manager.save_assistant_answer(session_id=session_id, turn_id=turn_id_2, content="押品入库流程为：...")
```

## 四、配置说明

### 4.1 Redis配置 (`config/redis.py`)

```python
# Redis功能开关
REDIS_ENABLED = False  # 如果Redis未部署，设为False

# Session过期时间（30分钟）
SESSION_TTL = 1800

# L2记忆过期时间（30分钟，与session生命周期一致）
L2_MEMORY_TTL = 1800
```

### 4.2 Elasticsearch配置 (`config/elasticsearch.py`)

```python
# ES索引配置
INDEX_CONFIG = {
    'l1_memory': {
        'index_name': 'bank_credit_l1_memory',
    },
    'l2_memory': {
        'index_name': 'bank_credit_l2_memory',
    },
    'session_record': {
        'index_name': 'bank_credit_session_record',
    }
}
```

## 五、注意事项

### 5.1 Session生命周期

- **Session基于时间失效**：默认30分钟不操作后失效
- **Session有效期内**：所有对话使用同一个session_id，turn_id递增（1, 2, 3...）
- **Session失效后**：下次对话会创建新session

### 5.2 Redis未启用时的处理

- 如果Redis未启用，使用ES中的session_record判断session是否存在
- 无法精确判断30分钟过期时间，但可以保证功能可用
- 建议启用Redis以获得完整的Session生命周期管理

### 5.3 错误处理

- Redis或ES连接失败时，不应影响主流程
- 所有方法都有异常处理，失败时返回None或空列表
- 建议在生产环境中监控Redis和ES的连接状态

### 5.4 性能考虑

- L2记忆优先使用Redis（快速查询）
- ES用于审计和历史查询
- 批量操作时注意ES的性能限制

## 六、API接口

### 6.1 后端API（Flask）

- **GET /api/session-list**：获取用户的所有session列表
- **GET /api/conversation-history**：获取指定session的对话历史
- **POST /api/session-delete**：删除指定session的所有历史记录

### 6.2 前端功能

- Session列表显示（标题+创建时间）
- 点击session记录切换并加载对话
- 删除session功能（每条记录右侧有删除按钮）
- 新对话功能（创建新session）

## 七、常见问题

### 7.1 为什么每次对话都创建新session？

**原因**：
- Redis未启用，且前端没有正确保存和传递session_id
- Session已过期（30分钟未操作）

**解决方案**：
- 确保前端正确保存和传递currentSessionId
- 启用Redis以获得完整的Session生命周期管理

### 7.2 如何清理历史记录？

**方法1**：使用前端删除按钮
- 点击session记录右侧的删除按钮

**方法2**：使用Python代码
```python
from src.context import clear_session_history
result = clear_session_history(session_id="sess_10000_1705123456")
```

**方法3**：使用命令行
```bash
python -m src.context.clear_history --session-id sess_10000_1705123456
```

### 7.3 如何查看对话历史？

```python
from src.context import get_memory_manager

memory_manager = get_memory_manager()
history = memory_manager.get_conversation_history(
    session_id="sess_10000_1705123456",
    limit=50
)

# history格式：
# [
#     {"session_id": "...", "turn_id": 1, "role": "user", "content": "...", "timestamp": "..."},
#     {"session_id": "...", "turn_id": 1, "role": "assistant", "content": "...", "timestamp": "..."},
#     ...
# ]
```

## 八、更新日志

### v1.0.0 (2026-01-XX)
- ✅ 实现Session管理（基于时间失效）
- ✅ 实现L1记忆（会话级记忆）
- ✅ 实现L2记忆（业务状态记忆）
- ✅ 实现Session记录管理
- ✅ 实现历史记录清理功能
- ✅ 实现前端历史对话功能

