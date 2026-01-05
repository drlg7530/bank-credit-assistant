"""
记忆管理器（统一接口）
功能：提供统一的接口管理L1和L2记忆
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入记忆模块
from src.context.session_manager import get_or_create_session
from src.context.l1_memory import L1Memory
from src.context.l2_memory import L2Memory

# ============================================================================
# 记忆管理器类
# ============================================================================

class MemoryManager:
    """记忆管理器（统一接口）"""
    
    def __init__(self, redis_client=None, es_client=None):
        """
        初始化记忆管理器
        
        参数:
            redis_client: Redis客户端（可选）
            es_client: Elasticsearch客户端（可选）
        """
        # 初始化L1和L2记忆管理器
        self.l1_memory = L1Memory(es_client=es_client)
        self.l2_memory = L2Memory(redis_client=redis_client, es_client=es_client)
    
    def get_or_create_session(self, user_id: int = 10000, session_id: Optional[str] = None) -> str:
        """
        获取或创建session
        
        参数:
            user_id: 用户ID，默认10000
            session_id: 可选的session_id
        
        返回:
            str: session_id
        """
        return get_or_create_session(user_id=user_id, session_id=session_id)
    
    def save_user_query(self, session_id: str, turn_id: Optional[int] = None, content: str = "") -> Optional[int]:
        """
        保存用户query到L1
        
        参数:
            session_id: session ID
            turn_id: 对话轮次编号（可选）
            content: 用户原始输入
        
        返回:
            Optional[int]: 保存成功返回turn_id，失败返回None
        """
        return self.l1_memory.save_user_query(session_id=session_id, turn_id=turn_id, content=content)
    
    def save_assistant_answer(self, session_id: str, turn_id: Optional[int] = None, content: str = "") -> Optional[int]:
        """
        保存助手answer到L1
        
        参数:
            session_id: session ID
            turn_id: 对话轮次编号（可选）
            content: 助手输出内容
        
        返回:
            Optional[int]: 保存成功返回turn_id，失败返回None
        """
        return self.l1_memory.save_assistant_answer(session_id=session_id, turn_id=turn_id, content=content)
    
    def get_l2_state(self, session_id: str) -> Optional[Dict]:
        """
        获取L2状态
        
        参数:
            session_id: session ID
        
        返回:
            Optional[Dict]: L2状态
        """
        return self.l2_memory.get_l2_state(session_id)
    
    def update_l2_state(self, session_id: str, intent: str, entities: Dict, 
                       previous_l2: Optional[Dict] = None) -> Dict:
        """
        更新L2状态
        
        参数:
            session_id: session ID
            intent: 意图类型
            entities: 实体信息
            previous_l2: 上一轮的L2状态（可选）
        
        返回:
            Dict: 更新后的L2状态
        """
        return self.l2_memory.update_l2_state(
            session_id=session_id,
            intent=intent,
            entities=entities,
            previous_l2=previous_l2
        )


# ============================================================================
# 全局记忆管理器实例（延迟初始化）
# ============================================================================

_global_memory_manager = None


def get_memory_manager(redis_client=None, es_client=None) -> MemoryManager:
    """
    获取全局记忆管理器实例（单例模式）
    
    参数:
        redis_client: Redis客户端（可选）
        es_client: Elasticsearch客户端（可选）
    
    返回:
        MemoryManager: 记忆管理器实例
    """
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(redis_client=redis_client, es_client=es_client)
    
    return _global_memory_manager

