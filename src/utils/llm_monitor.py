"""
大模型调用监控模块
功能：通过装饰器监控LLM调用，记录token使用情况和耗时
"""

import time
import threading
from typing import Dict, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from threading import local

# 线程本地存储（用于在函数内部传递token信息）
_thread_local = local()


@dataclass
class LLMCallInfo:
    """单次LLM调用信息"""
    function_name: str          # 函数名
    module: str                 # 模块名
    start_time: float           # 开始时间（秒）
    end_time: float             # 结束时间（秒）
    latency_ms: float           # 耗时（毫秒）
    prompt_tokens: int          # 输入token数
    completion_tokens: int      # 输出token数


@dataclass
class RequestStats:
    """单次业务请求的统计信息"""
    module: str                 # 模块名
    total_latency_ms: float     # 总耗时（毫秒）
    total_prompt_tokens: int    # 总输入token数
    total_completion_tokens: int # 总输出token数
    call_count: int             # 调用次数
    calls: list[LLMCallInfo] = field(default_factory=list)  # 调用记录列表


class LLMMonitor:
    """LLM监控器（单例模式，线程安全）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """实现单例模式（线程安全）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMMonitor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化监控器（只执行一次）"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # 当前请求的统计信息（按模块分组）
            # 格式: {module: RequestStats}
            self._current_request_stats: Dict[str, RequestStats] = {}
            self._request_active = False
            self._initialized = True
    
    def start_request(self):
        """开始一个新的业务请求"""
        with self._lock:
            self._current_request_stats.clear()
            self._request_active = True
    
    def get_request_stats(self) -> dict:
        """
        获取当前请求的统计信息（不清空）
        
        返回:
            dict: 包含统计信息的字典，如果没有统计信息则返回None
                {
                    'total_tokens': int,           # 总token数
                    'total_prompt_tokens': int,     # 总输入token数
                    'total_completion_tokens': int, # 总输出token数
                    'total_latency_ms': float,      # 总耗时（毫秒）
                    'call_count': int              # 调用次数
                }
        """
        with self._lock:
            if not self._request_active:
                return None
            
            # 计算所有模块的总统计
            total_latency_ms = sum(stats.total_latency_ms for stats in self._current_request_stats.values())
            total_prompt_tokens = sum(stats.total_prompt_tokens for stats in self._current_request_stats.values())
            total_completion_tokens = sum(stats.total_completion_tokens for stats in self._current_request_stats.values())
            total_calls = sum(stats.call_count for stats in self._current_request_stats.values())
            
            # 如果没有统计信息，返回None
            if total_calls == 0:
                return None
            
            return {
                'total_tokens': total_prompt_tokens + total_completion_tokens,
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'total_latency_ms': total_latency_ms,
                'call_count': total_calls
            }
    
    def finish_request(self):
        """结束当前业务请求，打印总统计信息并清空"""
        with self._lock:
            if not self._request_active:
                return
            
            # 计算所有模块的总统计
            total_latency_ms = sum(stats.total_latency_ms for stats in self._current_request_stats.values())
            total_prompt_tokens = sum(stats.total_prompt_tokens for stats in self._current_request_stats.values())
            total_completion_tokens = sum(stats.total_completion_tokens for stats in self._current_request_stats.values())
            total_calls = sum(stats.call_count for stats in self._current_request_stats.values())
            
            # 如果有统计信息，输出总统计
            if total_calls > 0:
                print(
                    f"\n[LLM Monitor(完整请求)] 总Token数: {total_prompt_tokens + total_completion_tokens} (输入: {total_prompt_tokens}, 输出: {total_completion_tokens})  "
                    f"总耗时: {int(total_latency_ms)} ms  调用次数: {total_calls}"
                )
            
            # 清空统计信息
            self._current_request_stats.clear()
            self._request_active = False
    
    def record_call(
        self,
        function_name: str,
        module: str,
        start_time: float,
        end_time: float,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """记录一次LLM调用，并立即输出该模块的统计信息"""
        with self._lock:
            if not self._request_active:
                # 如果没有激活的请求，自动开始一个
                self._request_active = True
            
            # 计算耗时
            latency_ms = (end_time - start_time) * 1000
            
            # 创建调用信息
            call_info = LLMCallInfo(
                function_name=function_name,
                module=module,
                start_time=start_time,
                end_time=end_time,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # 更新或创建模块统计
            if module not in self._current_request_stats:
                self._current_request_stats[module] = RequestStats(
                    module=module,
                    total_latency_ms=0.0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    call_count=0
                )
            
            stats = self._current_request_stats[module]
            stats.total_latency_ms += latency_ms
            stats.total_prompt_tokens += prompt_tokens
            stats.total_completion_tokens += completion_tokens
            stats.call_count += 1
            stats.calls.append(call_info)
            
            # 立即输出该模块的统计信息（步骤执行完就输出）
            print(
                f"[监控统计({module})]总输入token: {stats.total_prompt_tokens} | 总输出token: {stats.total_completion_tokens} |  总耗时：{int(stats.total_latency_ms)} ms"
            )


# 全局单例实例
_monitor = LLMMonitor()


def get_request_stats() -> dict:
    """
    获取当前请求的统计信息
    
    返回:
        dict: 包含统计信息的字典，如果没有统计信息则返回None
    """
    return _monitor.get_request_stats()


def set_token_info(prompt_tokens: int, completion_tokens: int):
    """设置当前线程的token信息（供LLM调用函数使用）"""
    _thread_local.prompt_tokens = prompt_tokens
    _thread_local.completion_tokens = completion_tokens


def get_token_info() -> tuple[int, int]:
    """获取当前线程的token信息"""
    prompt_tokens = getattr(_thread_local, 'prompt_tokens', 0)
    completion_tokens = getattr(_thread_local, 'completion_tokens', 0)
    return prompt_tokens, completion_tokens


def extract_token_from_response(response: Any) -> tuple[int, int]:
    """
    从响应中提取token信息
    
    参数:
        response: LLM响应对象（可能是dict、对象或字符串）
    
    返回:
        tuple: (prompt_tokens, completion_tokens)
    """
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        # 处理dict格式
        if isinstance(response, dict):
            if 'usage' in response:
                usage = response['usage']
                if isinstance(usage, dict):
                    # 优先使用 prompt_tokens 和 completion_tokens（百炼API标准格式）
                    # 如果没有，则尝试 input_tokens 和 output_tokens（兼容旧格式）
                    prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                elif hasattr(usage, 'prompt_tokens'):
                    # 对象格式：优先使用 prompt_tokens 和 completion_tokens
                    prompt_tokens = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', 0))
                    completion_tokens = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', 0))
                elif hasattr(usage, 'input_tokens'):
                    # 兼容旧格式
                    prompt_tokens = getattr(usage, 'input_tokens', 0)
                    completion_tokens = getattr(usage, 'output_tokens', 0)
        
        # 处理对象格式
        elif hasattr(response, 'usage'):
            usage = response.usage
            try:
                if isinstance(usage, dict):
                    # 优先使用 prompt_tokens 和 completion_tokens
                    prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                    completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                else:
                    # 优先使用 prompt_tokens 和 completion_tokens
                    prompt_tokens = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', 0))
                    completion_tokens = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', 0))
            except (AttributeError, TypeError):
                pass
    
    except Exception:
        # 如果提取失败，返回0
        pass
    
    return prompt_tokens, completion_tokens


def llm_monitor(module: str):
    """
    LLM调用监控装饰器
    
    参数:
        module: 模块名称（用于标识和统计）
    
    使用示例:
        @llm_monitor(module="query_rewrite")
        def my_llm_function(prompt: str) -> str:
            # LLM调用代码
            return response
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            start_time = time.time()
            
            try:
                # 清空线程本地存储的token信息
                if hasattr(_thread_local, 'prompt_tokens'):
                    delattr(_thread_local, 'prompt_tokens')
                if hasattr(_thread_local, 'completion_tokens'):
                    delattr(_thread_local, 'completion_tokens')
                
                # 调用原函数
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # 优先从线程本地存储获取token信息（如果函数内部设置了）
                prompt_tokens, completion_tokens = get_token_info()
                
                # 如果没有从线程本地存储获取到，尝试从返回值提取
                if prompt_tokens == 0 and completion_tokens == 0:
                    prompt_tokens, completion_tokens = extract_token_from_response(result)
                
                # 记录调用信息
                _monitor.record_call(
                    function_name=function_name,
                    module=module,
                    start_time=start_time,
                    end_time=end_time,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                return result
                
            except Exception as e:
                # 即使出错也记录调用（token数可能为0）
                end_time = time.time()
                _monitor.record_call(
                    function_name=function_name,
                    module=module,
                    start_time=start_time,
                    end_time=end_time,
                    prompt_tokens=0,
                    completion_tokens=0
                )
                raise
        
        return wrapper
    return decorator


def start_request():
    """开始一个新的业务请求"""
    _monitor.start_request()


def finish_request():
    """结束当前业务请求，打印统计信息"""
    _monitor.finish_request()

