"""
Token监控和时间监控模块（已废弃）
此文件中的大部分代码已被新的 llm_monitor.py 替代
仅保留 extract_token_info_from_response 函数供其他模块使用
"""

from typing import Dict, Any

# ============================================================================
# 辅助函数：从API响应中提取token信息（保留供其他模块使用）
# ============================================================================

def extract_token_info_from_response(response: Any, model: str = None) -> Dict:
    """
    从API响应中提取token信息（增强错误处理）
    
    参数:
        response: API响应对象
        model: 模型名称
    
    返回:
        Dict: 包含 prompt_tokens, completion_tokens, total_tokens, model
    """
    result = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'model': model
    }
    
    try:
        # 处理dict格式（百炼API可能返回dict）
        if isinstance(response, dict):
            if 'usage' in response:
                usage = response['usage']
                if isinstance(usage, dict):
                    # 百炼API格式：优先使用 prompt_tokens 和 completion_tokens
                    # 如果没有，则尝试 input_tokens 和 output_tokens（兼容旧格式）
                    result['prompt_tokens'] = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    result['completion_tokens'] = usage.get('completion_tokens', usage.get('output_tokens', 0))
                    result['total_tokens'] = usage.get('total_tokens', 0)
                elif hasattr(usage, 'prompt_tokens'):
                    # 对象格式：优先使用 prompt_tokens 和 completion_tokens
                    result['prompt_tokens'] = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', 0))
                    result['completion_tokens'] = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', 0))
                    result['total_tokens'] = getattr(usage, 'total_tokens', 0)
                elif hasattr(usage, 'input_tokens'):
                    # 兼容旧格式
                    result['prompt_tokens'] = getattr(usage, 'input_tokens', 0)
                    result['completion_tokens'] = getattr(usage, 'output_tokens', 0)
                    result['total_tokens'] = getattr(usage, 'total_tokens', 0)
            if 'model' in response and result['model'] is None:
                result['model'] = response.get('model')
        
        # 处理对象格式（百炼API响应对象）
        elif hasattr(response, 'usage'):
            usage = response.usage
            try:
                # 尝试作为dict访问
                if isinstance(usage, dict):
                    # 优先使用 prompt_tokens 和 completion_tokens
                    result['prompt_tokens'] = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    result['completion_tokens'] = usage.get('completion_tokens', usage.get('output_tokens', 0))
                    result['total_tokens'] = usage.get('total_tokens', 0)
                else:
                    # 尝试作为对象属性访问：优先使用 prompt_tokens 和 completion_tokens
                    result['prompt_tokens'] = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', 0))
                    result['completion_tokens'] = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', 0))
                    result['total_tokens'] = getattr(usage, 'total_tokens', 0)
            except (AttributeError, TypeError, KeyError) as e:
                # 如果访问失败，保持默认值0（不抛出异常）
                pass
        
        # 处理OpenAI格式
        elif hasattr(response, 'usage') and result['total_tokens'] == 0:
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                result['prompt_tokens'] = usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                result['completion_tokens'] = usage.completion_tokens
            if hasattr(usage, 'total_tokens'):
                result['total_tokens'] = usage.total_tokens
        
    except Exception as e:
        # 如果提取失败，记录错误但不抛出异常（使用默认值0）
        # 这样不会影响主流程
        pass
    
    return result
