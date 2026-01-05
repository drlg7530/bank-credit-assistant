"""
L2记忆模块（业务状态记忆）
功能：存储和管理业务状态，支持继承和覆盖规则
存储位置：Redis（快速查询）+ ES（审计）
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.redis import REDIS_ENABLED, REDIS_CONFIG, L2_MEMORY_KEY_PREFIX, L2_MEMORY_TTL
from config.elasticsearch import ES_CONFIG, INDEX_CONFIG

# ============================================================================
# 依赖检查
# ============================================================================

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠ 警告: redis未安装，L2记忆功能将不可用")

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠ 警告: elasticsearch未安装，L2记忆功能将不可用")


# ============================================================================
# L2记忆管理类
# ============================================================================

class L2Memory:
    """L2业务状态记忆管理器"""
    
    def __init__(self, redis_client=None, es_client=None):
        """
        初始化L2记忆管理器
        
        参数:
            redis_client: Redis客户端，如果为None则自动创建
            es_client: Elasticsearch客户端，如果为None则自动创建
        """
        self.index_name = INDEX_CONFIG.get('l2_memory', {}).get('index_name', 'bank_credit_l2_memory')
        
        # 初始化Redis客户端
        if redis_client:
            self.redis_client = redis_client
        else:
            # 检查Redis功能是否启用
            if REDIS_ENABLED and REDIS_AVAILABLE:
                try:
                    self.redis_client = redis.Redis(**REDIS_CONFIG)
                    self.redis_client.ping()
                    print("  ✓ L2记忆：Redis连接成功")
                except Exception as e:
                    print(f"  ⚠ L2记忆：Redis连接失败: {e}")
                    self.redis_client = None
            else:
                if not REDIS_ENABLED:
                    print("  ℹ L2记忆：Redis功能已关闭（config/redis.py中REDIS_ENABLED=False）")
                self.redis_client = None
        
        # 初始化ES客户端
        if es_client:
            self.es_client = es_client
        else:
            if ELASTICSEARCH_AVAILABLE:
                try:
                    self.es_client = Elasticsearch(**ES_CONFIG)
                    self.es_client.ping()
                    print("  ✓ L2记忆：Elasticsearch连接成功")
                except Exception as e:
                    print(f"  ⚠ L2记忆：Elasticsearch连接失败: {e}")
                    self.es_client = None
            else:
                self.es_client = None
        
        # 确保ES索引存在
        if self.es_client:
            self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """
        确保L2记忆索引存在，如果不存在则创建
        """
        if not self.es_client:
            return
        
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                # 创建索引映射
                mapping = {
                    "mappings": {
                        "properties": {
                            "session_id": {
                                "type": "keyword"
                            },
                            "current_customer_id": {
                                "type": "keyword"
                            },
                            "operation_chain": {
                                "type": "nested",  # 嵌套对象数组
                                "properties": {
                                    "intent": {"type": "keyword"},
                                    "active_domain": {"type": "keyword"},
                                    "business_object": {"type": "keyword"},
                                    "operation_stage": {"type": "keyword"},
                                    "last_action": {"type": "text"},
                                    "last_update": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
                                }
                            },
                            "timestamp": {
                                "type": "date",
                                "format": "strict_date_optional_time||epoch_millis"
                            }
                        }
                    }
                }
                
                self.es_client.indices.create(index=self.index_name, body=mapping)
                print(f"  ✓ L2记忆：创建索引 {self.index_name}")
            else:
                print(f"  ✓ L2记忆：索引 {self.index_name} 已存在")
        except Exception as e:
            print(f"  ⚠ L2记忆：创建索引失败: {e}")
    
    def get_l2_state(self, session_id: str) -> Optional[Dict]:
        """
        从Redis获取L2状态
        
        参数:
            session_id: session ID
        
        返回:
            Optional[Dict]: L2状态，如果不存在则返回None
        """
        if not self.redis_client:
            return None
        
        try:
            key = f"{L2_MEMORY_KEY_PREFIX}:{session_id}"
            data = self.redis_client.get(key)
            
            if data:
                # 解析JSON
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return json.loads(data)
            else:
                return None
        except Exception as e:
            print(f"  ⚠ L2记忆：从Redis获取状态失败: {e}")
            return None
    
    def _apply_inheritance_rules(self, new_intent: str, new_entities: Dict, previous_l2: Optional[Dict]) -> Dict:
        """
        应用继承和覆盖规则，生成新的L2状态
        
        规则：
        1. intent不同 → 全量覆盖
        2. intent相同 + 对象未显式出现 → 继承
        3. 对象显式变化 → 覆盖
        4. 阶段只能单向推进（创建 → 入库 → 审批）
        
        参数:
            new_intent: 新的意图类型
            new_entities: 新的实体信息（包含active_domain, business_object, operation_stage等）
            previous_l2: 上一轮的L2状态
        
        返回:
            Dict: 更新后的L2状态
        """
        # 如果没有上一轮状态，直接使用新状态
        if not previous_l2 or not previous_l2.get('operation_chain'):
            # 创建新的operation_chain
            operation_item = {
                "intent": new_intent,
                "active_domain": new_entities.get('active_domain', []),
                "business_object": new_entities.get('business_object', ''),
                "operation_stage": new_entities.get('operation_stage', ''),
                "last_action": new_entities.get('last_action', ''),
                "last_update": datetime.utcnow().isoformat() + "Z"
            }
            return {
                "session_id": new_entities.get('session_id', ''),
                "current_customer_id": new_entities.get('current_customer_id'),
                "operation_chain": [operation_item],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        # 获取上一轮的操作链（取最后一个）
        previous_chain = previous_l2.get('operation_chain', [])
        if not previous_chain:
            # 如果上一轮没有操作链，直接使用新状态
            operation_item = {
                "intent": new_intent,
                "active_domain": new_entities.get('active_domain', []),
                "business_object": new_entities.get('business_object', ''),
                "operation_stage": new_entities.get('operation_stage', ''),
                "last_action": new_entities.get('last_action', ''),
                "last_update": datetime.utcnow().isoformat() + "Z"
            }
            return {
                "session_id": previous_l2.get('session_id', new_entities.get('session_id', '')),
                "current_customer_id": previous_l2.get('current_customer_id') or new_entities.get('current_customer_id'),
                "operation_chain": [operation_item],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        previous_item = previous_chain[-1]
        previous_intent = previous_item.get('intent', '')
        previous_object = previous_item.get('business_object', '')
        previous_stage = previous_item.get('operation_stage', '')
        
        # 规则1：intent不同 → 全量覆盖
        if new_intent != previous_intent:
            operation_item = {
                "intent": new_intent,
                "active_domain": new_entities.get('active_domain', []),
                "business_object": new_entities.get('business_object', ''),
                "operation_stage": new_entities.get('operation_stage', ''),
                "last_action": new_entities.get('last_action', ''),
                "last_update": datetime.utcnow().isoformat() + "Z"
            }
            # 保留历史操作链，添加新操作
            new_chain = previous_chain + [operation_item]
            return {
                "session_id": previous_l2.get('session_id', new_entities.get('session_id', '')),
                "current_customer_id": previous_l2.get('current_customer_id') or new_entities.get('current_customer_id'),
                "operation_chain": new_chain,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        # 规则2和3：intent相同，处理对象和阶段
        new_object = new_entities.get('business_object', '')
        new_stage = new_entities.get('operation_stage', '')
        
        # 如果新对象为空，继承上一轮的对象
        if not new_object:
            new_object = previous_object
        
        # 如果新阶段为空，继承上一轮的阶段
        if not new_stage:
            new_stage = previous_stage
        
        # 规则3：对象显式变化 → 覆盖
        if new_object and new_object != previous_object:
            # 对象变化，创建新的操作项
            operation_item = {
                "intent": new_intent,
                "active_domain": new_entities.get('active_domain', previous_item.get('active_domain', [])),
                "business_object": new_object,
                "operation_stage": new_stage,
                "last_action": new_entities.get('last_action', ''),
                "last_update": datetime.utcnow().isoformat() + "Z"
            }
            new_chain = previous_chain + [operation_item]
        else:
            # 规则4：阶段只能单向推进
            # 定义阶段顺序
            stage_order = ['创建', '入库', '审批', '放款', '结清']
            previous_stage_index = stage_order.index(previous_stage) if previous_stage in stage_order else -1
            new_stage_index = stage_order.index(new_stage) if new_stage in stage_order else -1
            
            # 如果新阶段在顺序上更靠后，更新阶段；否则保持原阶段
            if new_stage_index > previous_stage_index:
                # 更新最后一个操作项
                updated_item = previous_item.copy()
                updated_item['operation_stage'] = new_stage
                updated_item['last_action'] = new_entities.get('last_action', updated_item.get('last_action', ''))
                updated_item['last_update'] = datetime.utcnow().isoformat() + "Z"
                new_chain = previous_chain[:-1] + [updated_item]
            else:
                # 保持原状态，只更新last_action
                updated_item = previous_item.copy()
                updated_item['last_action'] = new_entities.get('last_action', updated_item.get('last_action', ''))
                updated_item['last_update'] = datetime.utcnow().isoformat() + "Z"
                new_chain = previous_chain[:-1] + [updated_item]
        
        return {
            "session_id": previous_l2.get('session_id', new_entities.get('session_id', '')),
            "current_customer_id": previous_l2.get('current_customer_id') or new_entities.get('current_customer_id'),
            "operation_chain": new_chain,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def update_l2_state(self, session_id: str, intent: str, entities: Dict, 
                       previous_l2: Optional[Dict] = None) -> Dict:
        """
        更新L2状态（实现继承和覆盖规则）
        
        参数:
            session_id: session ID
            intent: 意图类型
            entities: 实体信息（包含active_domain, business_object, operation_stage, last_action等）
            previous_l2: 上一轮的L2状态，如果为None则从Redis获取
        
        返回:
            Dict: 更新后的L2状态
        """
        # 如果没有提供上一轮状态，从Redis获取
        if previous_l2 is None:
            previous_l2 = self.get_l2_state(session_id)
        
        # 准备实体信息
        entities_with_session = {
            'session_id': session_id,
            'current_customer_id': entities.get('current_customer_id'),
            'active_domain': entities.get('active_domain', []),
            'business_object': entities.get('business_object', ''),
            'operation_stage': entities.get('operation_stage', ''),
            'last_action': entities.get('last_action', '')
        }
        
        # 应用继承和覆盖规则
        new_l2_state = self._apply_inheritance_rules(intent, entities_with_session, previous_l2)
        
        # 保存到Redis和ES
        self.save_l2_to_redis(session_id, new_l2_state)
        self.save_l2_to_es(session_id, new_l2_state)
        
        return new_l2_state
    
    def save_l2_to_redis(self, session_id: str, l2_state: Dict) -> bool:
        """
        保存L2状态到Redis（快速查询）
        
        参数:
            session_id: session ID
            l2_state: L2状态数据
        
        返回:
            bool: 保存成功返回True
        """
        if not self.redis_client:
            print("  ⚠ L2记忆：Redis不可用，跳过保存")
            return False
        
        try:
            key = f"{L2_MEMORY_KEY_PREFIX}:{session_id}"
            value = json.dumps(l2_state, ensure_ascii=False)
            self.redis_client.setex(key, L2_MEMORY_TTL, value)
            print(f"  ✓ L2记忆：保存到Redis (session={session_id})")
            return True
        except Exception as e:
            print(f"  ⚠ L2记忆：保存到Redis失败: {e}")
            return False
    
    def save_l2_to_es(self, session_id: str, l2_state: Dict) -> bool:
        """
        保存L2状态到ES（审计）
        
        参数:
            session_id: session ID
            l2_state: L2状态数据
        
        返回:
            bool: 保存成功返回True
        """
        if not self.es_client:
            print("  ⚠ L2记忆：Elasticsearch不可用，跳过保存")
            return False
        
        try:
            # 使用session_id作为文档ID，实现覆盖更新
            self.es_client.index(index=self.index_name, id=session_id, body=l2_state)
            print(f"  ✓ L2记忆：保存到ES (session={session_id})")
            return True
        except Exception as e:
            print(f"  ⚠ L2记忆：保存到ES失败: {e}")
            return False

