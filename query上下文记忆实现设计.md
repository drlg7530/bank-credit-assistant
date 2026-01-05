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
- Session在Redis中存储，TTL：24小时（86400秒）
- Session过期时，L2记忆也会过期

### 2.3 接口设计
```python
class SessionManager:
    @staticmethod
    def create_session(user_id: int = 10000) -> str:
        """创建新session，返回session_id"""
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """获取session信息"""
    
    @staticmethod
    def is_valid(session_id: str) -> bool:
        """检查session是否有效"""

def get_or_create_session(user_id: int = 10000, session_id: Optional[str] = None) -> str:
    """获取或创建session"""
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
- **助手answer**：在LLM生成后写入L1

### 3.3 接口设计
```python
class L1Memory:
    def __init__(self, es_client):
        """初始化L1记忆管理器"""
    
    def save_user_query(self, session_id: str, turn_id: int, content: str) -> bool:
        """保存用户query到L1"""
    
    def save_assistant_answer(self, session_id: str, turn_id: int, content: str) -> bool:
        """保存助手answer到L1"""
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取会话历史（可选，暂不实现）"""
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
- **Redis**：快速查询，Key格式：`l2_memory:{session_id}`，TTL：24小时
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
添加L1和L2索引配置：
```python
INDEX_CONFIG = {
    'policy': {...},
    'system': {...},
    'l1_memory': {
        'index_name': 'bank_credit_l1_memory',
    },
    'l2_memory': {
        'index_name': 'bank_credit_l2_memory',
    }
}
```

### 6.2 Redis配置（config/redis.py）
已有配置，无需修改。

## 七、实现步骤

1. ✅ 更新ES配置，添加L1/L2索引配置
2. ✅ 创建Session管理模块
3. ✅ 创建L1记忆模块（ES存储）
4. ✅ 创建L2记忆模块（Redis+ES存储）
5. ✅ 创建记忆管理器（统一接口）
6. ✅ 扩展意图识别提示词，支持实体抽取
7. ✅ 修改app.py，集成上下文记忆流程
8. ✅ 修改Flask API，支持session传递

## 八、注意事项

1. **Session ID传递**：Flask API需要支持session_id参数（可选，不传则创建新session）
2. **Turn ID管理**：每个session的turn_id从1开始自增，需要从Redis或ES获取当前最大值
3. **L2继承逻辑**：需要仔细实现继承和覆盖规则
4. **错误处理**：Redis或ES连接失败时，不应影响主流程
5. **性能考虑**：L2读取和更新要快速，优先使用Redis

