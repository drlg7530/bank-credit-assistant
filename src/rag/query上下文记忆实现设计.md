# Query上下文记忆实现设计文档

## 一、总体架构

### 1.1 分层设计（L0-L4）
- **L0**：单轮上下文（当前问题）- 不存储
- **L1**：会话级记忆（会话记录）- ES存储
- **L2**：业务状态记忆（结构化状态）- Redis+ES存储
- **L3**：长期偏好记忆（暂未开发）

### 1.2 模块划分
```
src/context/
├── __init__.py              # 模块导出
├── session_manager.py       # Session管理（基于用户ID生成session_id）
├── session_record.py        # Session记录管理（ES存储session元信息）
├── l1_memory.py            # L1会话级记忆（ES存储）
├── l2_memory.py            # L2业务状态记忆（Redis+ES存储）
└── memory_manager.py       # 记忆管理器（统一接口）
```

## 二、Session管理设计

### 2.1 Session ID生成规则
- **格式**：`sess_{user_id}_{timestamp}`
- **示例**：`sess_10000_1705123456`
- **默认用户ID**：10000
- **说明**：包含时间戳，便于排序和调试

### 2.2 Session生命周期
- **Session基于时间失效**：默认30分钟（1800秒）不操作后失效
- **Session存储位置**：
  - Redis：存储session状态，TTL：30分钟（1800秒）
  - ES：存储session记录（session_record索引），包含标题、第一个问题、创建时间等
- **Session失效机制**：
  - 如果Redis启用：使用Redis TTL判断session是否过期（30分钟）
  - 如果Redis未启用：使用ES中的session_record判断session是否存在（无法精确判断过期时间）
- **Session失效后**：下次对话会创建新session
- **Session有效期内**：所有对话都使用同一个session_id，只是turn_id递增（1, 2, 3...）

### 2.3 Session记录管理
- **Session记录索引**：`bank_credit_session_record`（ES）
- **Session记录字段**：
  ```json
  {
    "session_id": "sess_10000_1705123456",
    "user_id": 10000,
    "title": "查询押品怎么创建",          // 标题使用第一个问题
    "first_question": "查询押品怎么创建",  // 第一个问题
    "created_at": "2026-01-05T14:00:00Z",
    "updated_at": "2026-01-05T14:00:00Z"
  }
  ```
- **Session记录创建时机**：保存第一个问题（turn_id=1）时自动创建
- **Session记录用途**：用于前端显示历史对话列表（标题+时间）

### 2.4 接口设计
```python
class SessionManager:
    @staticmethod
    def create_session(user_id: int = 10000) -> str:
        """创建新session，返回session_id"""
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """获取session信息（从Redis）"""
    
    @staticmethod
    def is_valid(session_id: str) -> bool:
        """检查session是否有效
        - Redis启用：检查Redis中是否存在且未过期
        - Redis未启用：检查ES中的session_record是否存在
        """

class SessionRecord:
    def create_session_record(self, session_id: str, user_id: int, first_question: str) -> bool:
        """创建session记录（ES）"""
    
    def get_session_record(self, session_id: str) -> Optional[Dict]:
        """获取session记录（ES）"""
    
    def list_session_records(self, user_id: int, limit: int = 50) -> List[Dict]:
        """获取用户的所有session记录列表（按创建时间倒序）"""

def get_or_create_session(user_id: int = 10000, session_id: Optional[str] = None) -> str:
    """获取或创建session
    - 如果提供了session_id且有效，则继续使用该session
    - 如果session_id无效或未提供，则创建新session
    """
```

## 三、L1记忆设计（会话级记忆）

### 3.1 ES索引配置
- **索引名称**：`bank_credit_l1_memory`
- **字段设计**：
  ```json
  {
    "session_id": "keyword",      // 对话ID
    "turn_id": "integer",          // 对话轮次编号（从1开始自增）
    "role": "keyword",              // 角色：user/assistant
    "content": "text",              // 内容（用户原始输入或助手输出）
    "timestamp": "date"            // 时间戳（ISO格式）
  }
  ```

### 3.2 存储时机
- **用户query**：在query改写前写入L1
  - 如果是第一个问题（turn_id=1），同时创建session记录（标题使用第一个问题）
- **助手answer**：在LLM生成后写入L1

### 3.3 核心逻辑
- **同一session下的多轮对话**：
  - 使用同一个session_id
  - turn_id递增（1, 2, 3...）
  - 所有对话保存在同一个session记录中
- **Session失效后**：
  - 下次对话会创建新session（新的session_id，turn_id从1开始）
- **历史对话查询**：
  - 前端显示session列表（标题+创建时间）
  - 点击session记录时，加载该session下的所有对话（按时间顺序）

### 3.4 接口设计
```python
class L1Memory:
    def __init__(self, es_client):
        """初始化L1记忆管理器"""
    
    def save_user_query(self, session_id: str, turn_id: Optional[int], content: str, user_id: int = 10000) -> Optional[int]:
        """保存用户query到L1
        - 如果turn_id为None，自动获取下一个（基于该session的最大turn_id+1）
        - 如果是第一个问题（turn_id=1），自动创建session记录
        - 返回turn_id
        """
    
    def save_assistant_answer(self, session_id: str, turn_id: Optional[int], content: str) -> Optional[int]:
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

## 四、L2记忆设计（业务状态记忆）

### 4.1 数据结构
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

### 4.2 存储位置
- **Redis**：快速查询，Key格式：`l2_memory:{session_id}`，TTL：30分钟（与session生命周期一致）
- **ES**：审计存储，索引名称：`bank_credit_l2_memory`

### 4.3 ES索引配置
- **索引名称**：`bank_credit_l2_memory`
- **字段设计**：
  ```json
  {
    "session_id": "keyword",
    "current_customer_id": "keyword",
    "operation_chain": "object",      // 嵌套对象数组
    "timestamp": "date"                // 最后更新时间
  }
  ```

### 4.4 继承和覆盖规则
- **规则1**：intent不同 → 全量覆盖
- **规则2**：intent相同 + 对象未显式出现 → 继承
- **规则3**：对象显式变化 → 覆盖
- **规则4**：阶段只能单向推进（创建 → 入库 → 审批）

### 4.5 接口设计
```python
class L2Memory:
    def __init__(self, redis_client, es_client):
        """初始化L2记忆管理器"""
    
    def get_l2_state(self, session_id: str) -> Optional[Dict]:
        """从Redis获取L2状态"""
    
    def update_l2_state(self, session_id: str, intent: str, entities: Dict, 
                        previous_l2: Optional[Dict] = None) -> Dict:
        """更新L2状态（实现继承和覆盖规则）"""
    
    def save_l2_to_es(self, session_id: str, l2_state: Dict) -> bool:
        """保存L2状态到ES（审计）"""
    
    def save_l2_to_redis(self, session_id: str, l2_state: Dict) -> bool:
        """保存L2状态到Redis（快速查询）"""
```

## 五、流程集成

### 5.1 完整流程
```
用户 query
  ↓
写入 L1（原始 query）
  ↓
query 改写 LLM
  ↓
【读取 L2】
  ↓
意图识别 + 关键实体抽取 LLM（同一次调用）
  ↓
更新 / 写入 L2（规则+继承，不用额外 LLM）
  ↓
路由 & ES 查询（用 L2 过滤 domain/object）
  ↓
LLM 重组答案
  ↓
写入 L1（助手回答）
```

### 5.2 关键实体抽取
需要在意图识别时同时抽取：
- `intent`：意图类型
- `active_domain`：业务域（policy/system/risk等）
- `business_object`：业务对象（如：押品、客户等）
- `operation_stage`：操作阶段（如：创建、入库、审批等）

### 5.3 提示词扩展
需要扩展意图识别提示词，要求同时返回实体信息：
```json
{
  "intent": "system_operation",
  "active_domain": ["system"],
  "business_object": "押品",
  "operation_stage": "创建"
}
```

## 六、配置更新

### 6.1 ES配置（config/elasticsearch.py）
添加L1、L2和Session记录索引配置：
```python
INDEX_CONFIG = {
    'policy': {...},
    'system': {...},
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

### 6.2 Redis配置（config/redis.py）
- **SESSION_TTL**：30分钟（1800秒）
- **L2_MEMORY_TTL**：30分钟（1800秒，与session生命周期一致）

## 七、历史对话功能

### 7.1 前端功能
- **Session列表显示**：显示所有session记录（标题+创建时间），按创建时间倒序
- **切换Session**：点击session记录时，切换到该session，并加载该session下的所有对话
- **新对话功能**：点击"新对话"按钮，清空当前session，下次查询时创建新session
- **对话显示**：按时间顺序显示该session下的所有对话（user和assistant消息）

### 7.2 后端API
- **GET /api/session-list**：获取用户的所有session列表
- **GET /api/conversation-history**：获取指定session的对话历史

### 7.3 数据流程
1. 用户第一次提问：创建新session，turn_id=1，创建session记录
2. 用户在session有效期内继续提问：使用同一个session_id，turn_id递增
3. Session失效后：下次提问创建新session
4. 点击历史记录：加载该session下的所有对话（按turn_id和时间戳排序）

## 八、实现步骤

1. ✅ 更新ES配置，添加L1/L2/Session记录索引配置
2. ✅ 创建Session管理模块
3. ✅ 创建Session记录管理模块
4. ✅ 创建L1记忆模块（ES存储）
5. ✅ 创建L2记忆模块（Redis+ES存储）
6. ✅ 创建记忆管理器（统一接口）
7. ✅ 扩展意图识别提示词，支持实体抽取
8. ✅ 修改app.py，集成上下文记忆流程
9. ✅ 修改Flask API，支持session传递和历史对话查询
10. ✅ 实现前端历史对话功能

## 九、注意事项

1. **Session ID传递**：
   - 前端在每次请求时传递currentSessionId（如果存在）
   - 后端检查session是否有效，有效则继续使用，无效则创建新session
   - 后端返回session_id，前端更新currentSessionId

2. **Session生命周期**：
   - Session基于时间失效（30分钟不操作后失效）
   - 在session有效期内，所有对话使用同一个session_id，turn_id递增
   - Session失效后，下次对话创建新session

3. **Turn ID管理**：
   - 每个session的turn_id从1开始自增
   - 通过查询ES获取该session的最大turn_id，然后+1

4. **Session记录管理**：
   - 只有第一个问题（turn_id=1）时创建session记录
   - Session记录标题使用第一个问题
   - Session记录用于前端显示历史对话列表

5. **Redis未启用时的处理**：
   - 如果Redis未启用，使用ES中的session_record判断session是否存在
   - 无法精确判断30分钟过期时间，但可以保证功能可用

6. **L2继承逻辑**：需要仔细实现继承和覆盖规则

7. **错误处理**：Redis或ES连接失败时，不应影响主流程

8. **性能考虑**：L2读取和更新要快速，优先使用Redis

