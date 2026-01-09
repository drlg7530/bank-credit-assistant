"""
Session管理模块
功能：基于用户ID生成和管理session_id
"""

import time
import sys
from pathlib import Path
from typing import Optional, Dict
import redis

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入Redis配置
from config.redis import REDIS_ENABLED, REDIS_CONFIG, SESSION_KEY_PREFIX, SESSION_TTL

# ============================================================================
# Redis连接管理
# ============================================================================

# 全局Redis连接（延迟初始化）
_redis_client = None


def get_redis_client() -> redis.Redis:
    """
    获取Redis客户端（单例模式）
    
    返回:
        redis.Redis: Redis客户端对象，如果Redis未启用或连接失败则返回None
    """
    global _redis_client
    
    # 检查Redis功能是否启用
    if not REDIS_ENABLED:
        return None
    
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(**REDIS_CONFIG)
            # 测试连接
            _redis_client.ping()
            print("  ✓ Redis连接成功")
        except Exception as e:
            print(f"  ⚠ Redis连接失败: {e}")
            # 如果连接失败，返回None，但不抛出异常（允许降级运行）
            _redis_client = None
    
    return _redis_client


# ============================================================================
# Session管理类
# ============================================================================

class SessionManager:
    """Session管理器"""
    
    @staticmethod
    def create_session(user_id: int = 10000) -> str:
        """
        创建新session，返回session_id
        
        参数:
            user_id: 用户ID，默认10000
        
        返回:
            str: session_id，格式：sess_{user_id}_{timestamp}
        """
        # 生成session_id：sess_{user_id}_{timestamp}
        timestamp = int(time.time())
        session_id = f"sess_{user_id}_{timestamp}"
        
        # 保存session信息到Redis
        redis_client = get_redis_client()
        if redis_client:
            try:
                session_key = f"{SESSION_KEY_PREFIX}:{session_id}"
                session_data = {
                    'user_id': user_id,
                    'created_at': timestamp,
                    'last_access': timestamp
                }
                # 使用hash存储session信息
                redis_client.hset(session_key, mapping=session_data)
                # 设置过期时间
                redis_client.expire(session_key, SESSION_TTL)
                print(f"  ✓ 创建Session: {session_id}")
            except Exception as e:
                print(f"  ⚠ 保存Session到Redis失败: {e}，但继续使用session_id")
        
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict]:
        """
        获取session信息
        
        参数:
            session_id: session ID
        
        返回:
            Optional[Dict]: session信息，如果不存在则返回None
        """
        redis_client = get_redis_client()
        if not redis_client:
            return None
        
        try:
            session_key = f"{SESSION_KEY_PREFIX}:{session_id}"
            session_data = redis_client.hgetall(session_key)
            
            if session_data:
                # 更新最后访问时间
                redis_client.hset(session_key, 'last_access', int(time.time()))
                redis_client.expire(session_key, SESSION_TTL)
                
                # 转换字节为字符串
                result = {
                    'session_id': session_id,
                    'user_id': int(session_data.get(b'user_id', session_data.get('user_id', 0))),
                    'created_at': int(session_data.get(b'created_at', session_data.get('created_at', 0))),
                    'last_access': int(session_data.get(b'last_access', session_data.get('last_access', 0)))
                }
                return result
            else:
                return None
        except Exception as e:
            print(f"  ⚠ 获取Session失败: {e}")
            return None
    
    @staticmethod
    def is_valid(session_id: str) -> bool:
        """
        检查session是否有效
        
        参数:
            session_id: session ID
        
        返回:
            bool: 如果session存在且有效则返回True
        """
        # 如果Redis启用，使用Redis检查
        redis_client = get_redis_client()
        if redis_client:
            session = SessionManager.get_session(session_id)
            return session is not None
        
        # 如果Redis未启用，使用ES中的session_record检查
        # 如果ES中有该session的记录，说明session存在（即使Redis未启用，ES中的记录仍然存在）
        try:
            from src.context.session_record import SessionRecord
            from config.elasticsearch import ES_CONFIG
            
            # 创建ES客户端
            from elasticsearch import Elasticsearch
            es_client = Elasticsearch(**ES_CONFIG)
            
            # 检查ES中是否存在该session的记录
            session_record = SessionRecord(es_client=es_client)
            record = session_record.get_session_record(session_id)
            
            if record:
                # ES中有记录，说明session存在
                # 注意：Redis未启用时，无法判断session是否过期（30分钟）
                # 但为了功能可用，我们假设ES中的记录都是有效的
                return True
            else:
                return False
        except Exception as e:
            print(f"  ⚠ 检查Session有效性失败（ES）: {e}")
            # 如果ES检查也失败，为了功能可用，返回True（允许继续使用）
            # 这样可以避免Redis和ES都不可用时，每次都创建新session
            return True
    
    @staticmethod
    def refresh_session(session_id: str) -> bool:
        """
        刷新session过期时间
        
        参数:
            session_id: session ID
        
        返回:
            bool: 刷新成功返回True
        """
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        try:
            session_key = f"{SESSION_KEY_PREFIX}:{session_id}"
            if redis_client.exists(session_key):
                redis_client.expire(session_key, SESSION_TTL)
                return True
            return False
        except Exception as e:
            print(f"  ⚠ 刷新Session失败: {e}")
            return False


# ============================================================================
# 便捷函数
# ============================================================================

def get_or_create_session(user_id: int = 10000, session_id: Optional[str] = None) -> str:
    """
    获取或创建session
    
    核心逻辑：
    - session基于时间失效（默认30分钟不操作后失效，由Redis TTL控制）
    - 如果提供了session_id且有效（Redis中存在且未过期，或ES中有记录），则继续使用该session
    - 如果session_id无效（不存在或已过期），则创建新session
    - 在session有效期内，所有对话都使用同一个session_id，只是turn_id递增
    
    注意：
    - 如果Redis启用，使用Redis的TTL来判断session是否过期（30分钟）
    - 如果Redis未启用，使用ES中的session_record来判断session是否存在（无法判断是否过期）
    
    参数:
        user_id: 用户ID，默认10000
        session_id: 可选的session_id，如果提供且有效则使用，否则创建新的
    
    返回:
        str: session_id
    """
    # 如果提供了session_id，检查是否有效
    if session_id:
        if SessionManager.is_valid(session_id):
            # session有效，刷新过期时间（如果Redis启用）
            SessionManager.refresh_session(session_id)
            print(f"  ✓ 继续使用现有session: {session_id}")
            return session_id
        else:
            # session无效（不存在或已过期），创建新session
            print(f"  ⚠ 提供的session_id无效或已过期: {session_id}，创建新session")
    
    # 创建新session（首次对话或session失效后）
    new_session_id = SessionManager.create_session(user_id=user_id)
    print(f"  ✓ 创建新session: {new_session_id}")
    return new_session_id

