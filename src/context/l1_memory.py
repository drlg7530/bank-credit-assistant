"""
L1记忆模块（会话级记忆）
功能：存储每轮对话的query和answer到Elasticsearch
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入ES配置
from config.elasticsearch import ES_CONFIG, INDEX_CONFIG

# ============================================================================
# 依赖检查
# ============================================================================

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠ 警告: elasticsearch未安装，L1记忆功能将不可用")


# ============================================================================
# L1记忆管理类
# ============================================================================

class L1Memory:
    """L1会话级记忆管理器"""
    
    def __init__(self, es_client: Optional[Elasticsearch] = None):
        """
        初始化L1记忆管理器
        
        参数:
            es_client: Elasticsearch客户端，如果为None则自动创建
        """
        self.index_name = INDEX_CONFIG.get('l1_memory', {}).get('index_name', 'bank_credit_l1_memory')
        
        if es_client:
            self.es_client = es_client
        else:
            if ELASTICSEARCH_AVAILABLE:
                try:
                    self.es_client = Elasticsearch(**ES_CONFIG)
                    # 测试连接
                    self.es_client.ping()
                    print("  ✓ L1记忆：Elasticsearch连接成功")
                except Exception as e:
                    print(f"  ⚠ L1记忆：Elasticsearch连接失败: {e}")
                    self.es_client = None
            else:
                self.es_client = None
        
        # 确保索引存在
        if self.es_client:
            self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """
        确保L1记忆索引存在，如果不存在则创建
        """
        if not self.es_client:
            return
        
        try:
            # 检查索引是否存在
            if not self.es_client.indices.exists(index=self.index_name):
                # 创建索引映射
                mapping = {
                    "mappings": {
                        "properties": {
                            "session_id": {
                                "type": "keyword"  # 用于精确匹配和聚合
                            },
                            "turn_id": {
                                "type": "integer"  # 对话轮次编号
                            },
                            "role": {
                                "type": "keyword"  # user或assistant
                            },
                            "content": {
                                "type": "text",     # 对话内容，支持全文搜索
                                "analyzer": "ik_max_word"  # 使用中文分词器（如果已安装）
                            },
                            "timestamp": {
                                "type": "date",     # ISO格式时间戳
                                "format": "strict_date_optional_time||epoch_millis"
                            }
                        }
                    }
                }
                
                self.es_client.indices.create(index=self.index_name, body=mapping)
                print(f"  ✓ L1记忆：创建索引 {self.index_name}")
            else:
                print(f"  ✓ L1记忆：索引 {self.index_name} 已存在")
        except Exception as e:
            print(f"  ⚠ L1记忆：创建索引失败: {e}")
    
    def _get_next_turn_id(self, session_id: str) -> int:
        """
        获取下一个turn_id（从1开始自增）
        
        参数:
            session_id: session ID
        
        返回:
            int: 下一个turn_id
        """
        if not self.es_client:
            return 1
        
        try:
            # 查询该session的最大turn_id
            query = {
                "query": {
                    "term": {"session_id": session_id}
                },
                "size": 0,
                "aggs": {
                    "max_turn_id": {
                        "max": {"field": "turn_id"}
                    }
                }
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            max_turn_id = response.get('aggregations', {}).get('max_turn_id', {}).get('value')
            
            if max_turn_id is None:
                return 1
            else:
                return int(max_turn_id) + 1
        except Exception as e:
            print(f"  ⚠ L1记忆：获取turn_id失败: {e}，使用默认值1")
            return 1
    
    def _get_current_turn_id(self, session_id: str) -> int:
        """
        获取当前session的最大turn_id（用于同一轮对话的answer）
        
        参数:
            session_id: session ID
        
        返回:
            int: 当前最大turn_id，如果没有则返回1
        """
        if not self.es_client:
            return 1
        
        try:
            # 查询该session的最大turn_id
            query = {
                "query": {
                    "term": {"session_id": session_id}
                },
                "size": 0,
                "aggs": {
                    "max_turn_id": {
                        "max": {"field": "turn_id"}
                    }
                }
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            max_turn_id = response.get('aggregations', {}).get('max_turn_id', {}).get('value')
            
            if max_turn_id is None:
                return 1
            else:
                return int(max_turn_id)
        except Exception as e:
            print(f"  ⚠ L1记忆：获取当前turn_id失败: {e}，使用默认值1")
            return 1
    
    def save_user_query(self, session_id: str, turn_id: Optional[int] = None, content: str = "") -> Optional[int]:
        """
        保存用户query到L1
        
        参数:
            session_id: session ID
            turn_id: 对话轮次编号，如果为None则自动获取下一个
            content: 用户原始输入
        
        返回:
            Optional[int]: 保存成功返回turn_id，失败返回None
        """
        if not self.es_client:
            print("  ⚠ L1记忆：Elasticsearch不可用，跳过保存")
            return None
        
        if not content:
            print("  ⚠ L1记忆：内容为空，跳过保存")
            return None
        
        try:
            # 如果没有提供turn_id，自动获取下一个
            if turn_id is None:
                turn_id = self._get_next_turn_id(session_id)
            
            # 构建文档
            doc = {
                "session_id": session_id,
                "turn_id": turn_id,
                "role": "user",
                "content": content,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # 保存到ES（使用session_id和turn_id组合作为文档ID，确保唯一性）
            doc_id = f"{session_id}_turn_{turn_id}_user"
            self.es_client.index(index=self.index_name, id=doc_id, body=doc)
            print(f"  ✓ L1记忆：保存用户query (session={session_id}, turn={turn_id})")
            return turn_id
        except Exception as e:
            print(f"  ⚠ L1记忆：保存用户query失败: {e}")
            return None
    
    def save_assistant_answer(self, session_id: str, turn_id: Optional[int] = None, content: str = "") -> Optional[int]:
        """
        保存助手answer到L1
        
        参数:
            session_id: session ID
            turn_id: 对话轮次编号，如果为None则使用当前session的最大turn_id（同一轮对话的answer应该和query使用相同的turn_id）
            content: 助手输出内容
        
        返回:
            Optional[int]: 保存成功返回turn_id，失败返回None
        """
        if not self.es_client:
            print("  ⚠ L1记忆：Elasticsearch不可用，跳过保存")
            return None
        
        if not content:
            print("  ⚠ L1记忆：内容为空，跳过保存")
            return None
        
        try:
            # 如果没有提供turn_id，使用当前session的最大turn_id（同一轮对话的answer应该和query使用相同的turn_id）
            if turn_id is None:
                turn_id = self._get_current_turn_id(session_id)
                print(f"  ℹ L1记忆：未提供turn_id，使用当前最大turn_id={turn_id}")
            
            # 构建文档
            doc = {
                "session_id": session_id,
                "turn_id": turn_id,
                "role": "assistant",
                "content": content,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # 保存到ES（使用session_id和turn_id组合作为文档ID，确保唯一性）
            doc_id = f"{session_id}_turn_{turn_id}_assistant"
            self.es_client.index(index=self.index_name, id=doc_id, body=doc)
            print(f"  ✓ L1记忆：保存助手answer (session={session_id}, turn={turn_id})")
            return turn_id
        except Exception as e:
            print(f"  ⚠ L1记忆：保存助手answer失败: {e}")
            return None
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        获取会话历史（可选功能，暂不实现详细逻辑）
        
        参数:
            session_id: session ID
            limit: 返回的最大记录数
        
        返回:
            List[Dict]: 会话历史记录列表，按turn_id和时间戳排序
        """
        if not self.es_client:
            return []
        
        try:
            query = {
                "query": {
                    "term": {"session_id": session_id}
                },
                "sort": [
                    {"turn_id": {"order": "asc"}},
                    {"timestamp": {"order": "asc"}}
                ],
                "size": limit * 2  # 每轮对话有user和assistant两条记录
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            hits = response.get('hits', {}).get('hits', [])
            
            # 提取文档内容
            history = []
            for hit in hits:
                history.append(hit.get('_source', {}))
            
            return history
        except Exception as e:
            print(f"  ⚠ L1记忆：获取会话历史失败: {e}")
            return []

