"""
RAG系统配置
用于管理RAG查询相关的配置参数
"""

# ============================================================================
# RAG查询配置
# ============================================================================

# 检索配置
RAG_CONFIG = {
    # 检索参数
    'top_k': 5,                    # 初始检索返回的文档数量，es粗筛默认为top_k*10
    'rerank_top_k': 5,              # 重排序后返回的文档数量
    'min_score': 0.5,               # 最小相似度分数阈值（低于此分数的结果会被过滤）
    
    # 功能开关
    'enable_rewrite': True,          # 是否启用query改写（调用大模型改写查询）
    'enable_rerank': False,         # 是否启用重排序（默认不启用）
    
    # 重排序配置
    'rerank_method': 'similarity',  # 重排序方法：
                                    # - 'similarity': 基于相似度分数排序（默认）
                                    # - 'bm25': 基于BM25分数排序（需要文本匹配）
    
    # 答案生成配置
    'max_context_length': 3000,     # 上下文最大长度（字符数）
    'include_metadata': True,        # 是否在答案中包含元数据信息（来源、时间等）
    'enable_streaming': True,        # 是否启用流式输出（默认True，开启状态）
                                     # 仅对最后一次LLM总结结果调用使用流式输出，提升用户体验
}

# ============================================================================
# 权限配置
# ============================================================================

# 角色权限映射（用于查询过滤）
ROLE_PERMISSIONS = {
    '客户经理': {
        'can_query_policy': True,
        'can_query_system': True,
        'can_query_all_domains': False,  # 只能查询自己权限范围内的文档
    },
    '团队负责人': {
        'can_query_policy': True,
        'can_query_system': True,
        'can_query_all_domains': True,   # 可以查询所有域的文档
    },
    '行长': {
        'can_query_policy': True,
        'can_query_system': True,
        'can_query_all_domains': True,   # 可以查询所有域的文档
    }
}

# ============================================================================
# 辅助函数
# ============================================================================

def get_rag_config() -> dict:
    """
    获取RAG配置
    
    返回:
        dict: RAG配置字典
    """
    return RAG_CONFIG.copy()


def update_rag_config(**kwargs):
    """
    更新RAG配置
    
    参数:
        **kwargs: 要更新的配置项
    """
    RAG_CONFIG.update(kwargs)


def is_rerank_enabled() -> bool:
    """检查是否启用重排序"""
    return RAG_CONFIG.get('enable_rerank', False)


def is_rewrite_enabled() -> bool:
    """检查是否启用query改写"""
    return RAG_CONFIG.get('enable_rewrite', True)

