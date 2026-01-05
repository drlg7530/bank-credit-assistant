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
        session = SessionManager.get_session(session_id)
        return session is not None
    
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
    
    参数:
        user_id: 用户ID，默认10000
        session_id: 可选的session_id，如果提供且有效则使用，否则创建新的
    
    返回:
        str: session_id
    """
    # 如果提供了session_id，检查是否有效
    if session_id:
        if SessionManager.is_valid(session_id):
            # 刷新session过期时间
            SessionManager.refresh_session(session_id)
            return session_id
        else:
            print(f"  ⚠ 提供的session_id无效: {session_id}，创建新session")
    
    # 创建新session
    return SessionManager.create_session(user_id=user_id)

