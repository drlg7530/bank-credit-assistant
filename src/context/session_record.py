"""
Session记录管理模块
功能：管理session级别的记录，每个session包含标题（第一个问题）、创建时间等信息
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
    print("⚠ 警告: elasticsearch未安装，Session记录功能将不可用")


# ============================================================================
# Session记录管理类
# ============================================================================

class SessionRecord:
    """Session记录管理器"""
    
    def __init__(self, es_client: Optional[Elasticsearch] = None):
        """
        初始化Session记录管理器
        
        参数:
            es_client: Elasticsearch客户端，如果为None则自动创建
        """
        # 使用独立的索引存储session记录
        self.index_name = INDEX_CONFIG.get('session_record', {}).get('index_name', 'bank_credit_session_record')
        
        if es_client:
            self.es_client = es_client
        else:
            if ELASTICSEARCH_AVAILABLE:
                try:
                    self.es_client = Elasticsearch(**ES_CONFIG)
                    # 测试连接
                    self.es_client.ping()
                    print("  ✓ Session记录：Elasticsearch连接成功")
                except Exception as e:
                    print(f"  ⚠ Session记录：Elasticsearch连接失败: {e}")
                    self.es_client = None
            else:
                self.es_client = None
        
        # 确保索引存在
        if self.es_client:
            self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """
        确保Session记录索引存在，如果不存在则创建
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
                                "type": "keyword"  # 用于精确匹配
                            },
                            "user_id": {
                                "type": "integer"  # 用户ID
                            },
                            "title": {
                                "type": "text",     # 标题（第一个问题），支持全文搜索
                                "analyzer": "ik_max_word"  # 使用中文分词器（如果已安装）
                            },
                            "first_question": {
                                "type": "text",     # 第一个问题，支持全文搜索
                                "analyzer": "ik_max_word"
                            },
                            "created_at": {
                                "type": "date",     # ISO格式时间戳
                                "format": "strict_date_optional_time||epoch_millis"
                            },
                            "updated_at": {
                                "type": "date",     # 最后更新时间
                                "format": "strict_date_optional_time||epoch_millis"
                            }
                        }
                    }
                }
                
                self.es_client.indices.create(index=self.index_name, body=mapping)
                print(f"  ✓ Session记录：创建索引 {self.index_name}")
            else:
                print(f"  ✓ Session记录：索引 {self.index_name} 已存在")
        except Exception as e:
            print(f"  ⚠ Session记录：创建索引失败: {e}")
    
    def create_session_record(self, session_id: str, user_id: int, first_question: str) -> bool:
        """
        创建session记录
        
        参数:
            session_id: session ID
            user_id: 用户ID
            first_question: 第一个问题（作为标题）
        
        返回:
            bool: 创建成功返回True，失败返回False
        """
        if not self.es_client:
            print("  ⚠ Session记录：Elasticsearch不可用，跳过创建")
            return False
        
        if not first_question:
            print("  ⚠ Session记录：第一个问题为空，跳过创建")
            return False
        
        try:
            # 构建文档
            now = datetime.utcnow().isoformat() + "Z"
            doc = {
                "session_id": session_id,
                "user_id": user_id,
                "title": first_question,  # 标题使用第一个问题
                "first_question": first_question,
                "created_at": now,
                "updated_at": now
            }
            
            # 保存到ES（使用session_id作为文档ID，确保唯一性）
            self.es_client.index(index=self.index_name, id=session_id, body=doc)
            print(f"  ✓ Session记录：创建session记录 (session={session_id}, title={first_question[:30]}...)")
            return True
        except Exception as e:
            print(f"  ⚠ Session记录：创建session记录失败: {e}")
            return False
    
    def update_session_record(self, session_id: str) -> bool:
        """
        更新session记录的最后更新时间
        
        参数:
            session_id: session ID
        
        返回:
            bool: 更新成功返回True，失败返回False
        """
        if not self.es_client:
            return False
        
        try:
            # 更新最后访问时间
            now = datetime.utcnow().isoformat() + "Z"
            self.es_client.update(
                index=self.index_name,
                id=session_id,
                body={
                    "doc": {
                        "updated_at": now
                    }
                }
            )
            return True
        except Exception as e:
            print(f"  ⚠ Session记录：更新session记录失败: {e}")
            return False
    
    def get_session_record(self, session_id: str) -> Optional[Dict]:
        """
        获取session记录
        
        参数:
            session_id: session ID
        
        返回:
            Optional[Dict]: session记录，如果不存在则返回None
        """
        if not self.es_client:
            return None
        
        try:
            response = self.es_client.get(index=self.index_name, id=session_id)
            if response.get('found'):
                source = response.get('_source', {})
                return {
                    'session_id': source.get('session_id'),
                    'user_id': source.get('user_id'),
                    'title': source.get('title'),
                    'first_question': source.get('first_question'),
                    'created_at': source.get('created_at'),
                    'updated_at': source.get('updated_at')
                }
            return None
        except Exception as e:
            print(f"  ⚠ Session记录：获取session记录失败: {e}")
            return None
    
    def list_session_records(self, user_id: int = 10000, limit: int = 50) -> List[Dict]:
        """
        获取用户的所有session记录列表（按创建时间倒序）
        
        参数:
            user_id: 用户ID，默认10000
            limit: 返回的最大记录数，默认50
        
        返回:
            List[Dict]: session记录列表，按创建时间倒序
            格式: [
                {
                    "session_id": "...",
                    "user_id": 10000,
                    "title": "...",
                    "first_question": "...",
                    "created_at": "...",
                    "updated_at": "..."
                },
                ...
            ]
        """
        if not self.es_client:
            return []
        
        try:
            query = {
                "query": {
                    "term": {"user_id": user_id}
                },
                "sort": [
                    {"created_at": {"order": "desc"}}  # 按创建时间倒序
                ],
                "size": limit
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            hits = response.get('hits', {}).get('hits', [])
            
            # 提取文档内容
            records = []
            for hit in hits:
                source = hit.get('_source', {})
                records.append({
                    'session_id': source.get('session_id'),
                    'user_id': source.get('user_id'),
                    'title': source.get('title'),
                    'first_question': source.get('first_question'),
                    'created_at': source.get('created_at'),
                    'updated_at': source.get('updated_at')
                })
            
            return records
        except Exception as e:
            print(f"  ⚠ Session记录：获取session列表失败: {e}")
            return []
    
    def delete_session_record(self, session_id: str) -> bool:
        """
        删除session记录
        
        参数:
            session_id: session ID
        
        返回:
            bool: 删除成功返回True，失败返回False
        """
        if not self.es_client:
            return False
        
        try:
            self.es_client.delete(index=self.index_name, id=session_id)
            print(f"  ✓ Session记录：删除session记录 (session={session_id})")
            return True
        except Exception as e:
            print(f"  ⚠ Session记录：删除session记录失败: {e}")
            return False

