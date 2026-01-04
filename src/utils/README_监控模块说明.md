# Token监控和时间监控模块使用说明

## 📋 功能说明

监控模块提供了完整的Token监控和时间监控功能，符合需求文档中的设计规范。

### 主要功能

1. **单次LLM调用监控**：记录每次LLM调用的token使用情况和耗时
2. **用户请求级别聚合监控**：将一次用户请求的多次调用聚合起来
3. **时间监控**：统计不同步骤的耗时

## 🚀 使用方法

### 方式1：自动监控（已集成到LLM调用函数）

LLM调用函数已自动集成监控功能，无需额外配置：

```python
from src.rag.query import call_llm

# 调用时会自动记录监控信息
result = call_llm(
    prompt="你的提示词",
    module="policy_rag_answer"  # 指定模块名称
)
```

### 方式2：使用装饰器监控自定义函数

```python
from src.utils.monitor import monitor_llm_call

@monitor_llm_call(module="custom_module", model="qwen-plus")
def my_llm_function(prompt: str) -> str:
    # 你的LLM调用代码
    return response
```

### 方式3：使用上下文管理器监控步骤

```python
from src.utils.monitor import monitor_step

with monitor_step("intent_router") as step:
    # 执行意图识别
    result = classify_intent(question)
    step['result'] = result
```

### 方式4：用户请求级别的聚合监控

```python
from src.utils.monitor import RequestMonitor, get_monitor_manager

# 创建请求监控器
request_monitor = RequestMonitor(user_query="这个客户风险高不高？")

# 步骤1：意图识别
with monitor_step("intent_router", request_monitor.trace_id) as step1:
    intent_result = classify_intent(question)
    # 从监控管理器获取最近的调用记录
    recent_calls = get_monitor_manager().get_call_records(limit=1)
    if recent_calls:
        last_call = recent_calls[-1]
        request_monitor.add_step(
            module="intent_router",
            tokens=last_call['total_tokens'],
            latency_ms=last_call['latency_ms']
        )

# 步骤2：RAG查询
with monitor_step("policy_rag_answer", request_monitor.trace_id) as step2:
    answer = rag_query(query, domain='policy')
    recent_calls = get_monitor_manager().get_call_records(limit=1)
    if recent_calls:
        last_call = recent_calls[-1]
        request_monitor.add_step(
            module="policy_rag_answer",
            tokens=last_call['total_tokens'],
            latency_ms=last_call['latency_ms']
        )

# 完成请求监控
request_record = request_monitor.finish(success=True)
print(f"总Token数: {request_record.total_tokens}")
print(f"总耗时: {request_record.total_latency_ms:.2f} ms")
```

## 📊 数据结构

### 单次LLM调用记录

```python
{
    "trace_id": "uuid",
    "module": "intent_router",
    "model": "qwen-plus",
    "prompt_tokens": 420,
    "completion_tokens": 98,
    "total_tokens": 518,
    "latency_ms": 820.5,
    "timestamp": "2025-01-10 14:32:10",
    "success": True,
    "error": None
}
```

### 用户请求级别聚合记录

```python
{
    "trace_id": "uuid",
    "user_query": "这个客户风险高不高？现在政策还支不支持？",
    "steps": [
        {
            "module": "intent_router",
            "tokens": 518,
            "latency_ms": 820.5
        },
        {
            "module": "policy_rag_answer",
            "tokens": 1240,
            "latency_ms": 1520.3
        }
    ],
    "total_tokens": 1758,
    "total_latency_ms": 2340.8,
    "timestamp": "2025-01-10 14:32:15",
    "success": True
}
```

## 🔧 API参考

### MonitorManager

监控管理器（单例模式）

```python
from src.utils.monitor import get_monitor_manager

manager = get_monitor_manager()

# 获取单次调用记录
call_records = manager.get_call_records(limit=10)  # 获取最近10条

# 获取请求级别记录
request_records = manager.get_request_records(limit=10)

# 获取统计信息
stats = manager.get_statistics()
print(f"总调用次数: {stats['total_calls']}")
print(f"总Token数: {stats['total_tokens']}")
print(f"平均每次调用Token数: {stats['avg_tokens_per_call']:.2f}")

# 导出到JSON文件
manager.export_to_json("monitor_records.json")

# 清空记录
manager.clear_records()
```

### RequestMonitor

用户请求级别的监控器

```python
from src.utils.monitor import RequestMonitor

# 创建监控器
monitor = RequestMonitor(user_query="用户问题")

# 添加步骤
monitor.add_step(module="intent_router", tokens=518, latency_ms=820.5)

# 完成监控
record = monitor.finish(success=True)

# 获取当前统计
stats = monitor.get_current_stats()
```

## 📈 统计信息

```python
from src.utils.monitor import print_statistics

# 打印统计信息
print_statistics()
```

输出示例：
```
============================================================
监控统计信息
============================================================
总调用次数: 25
总Token数: 12580
平均每次调用Token数: 503.20
平均延迟: 1250.50 ms
总请求数: 5
============================================================
```

## 🔄 集成到现有代码

### 更新router.py以使用请求级别监控

```python
from src.utils.monitor import RequestMonitor, monitor_step, get_monitor_manager

def route_and_query(question: str, ...):
    # 创建请求监控器
    request_monitor = RequestMonitor(user_query=question)
    
    try:
        # 步骤1：意图识别
        with monitor_step("intent_router", request_monitor.trace_id):
            intent_result = classify_intent(question)
            # 获取最近的调用记录
            recent = get_monitor_manager().get_call_records(limit=1)
            if recent:
                request_monitor.add_step(
                    module="intent_router",
                    tokens=recent[0]['total_tokens'],
                    latency_ms=recent[0]['latency_ms']
                )
        
        # 步骤2：RAG查询
        with monitor_step("policy_rag_answer", request_monitor.trace_id):
            answer = rag_query(...)
            recent = get_monitor_manager().get_call_records(limit=1)
            if recent:
                request_monitor.add_step(
                    module="policy_rag_answer",
                    tokens=recent[0]['total_tokens'],
                    latency_ms=recent[0]['latency_ms']
                )
        
        # 完成监控
        request_monitor.finish(success=True)
        
        return result
    except Exception as e:
        request_monitor.finish(success=False)
        raise
```

## 📚 相关文件

- `src/utils/monitor.py` - 监控模块核心代码
- `src/rag/query.py` - 已集成监控的RAG查询模块
- `src/intent/classification.py` - 意图识别模块（可集成监控）

## ⚠️ 注意事项

1. **Token估算**：对于本地模型，如果无法获取精确token数，会使用字符数估算（1 token ≈ 4字符）
2. **性能影响**：监控功能对性能影响很小，但大量调用时建议定期清理记录
3. **内存管理**：监控记录存储在内存中，长时间运行建议定期导出并清理

