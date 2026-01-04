"""
查询路由模块
功能：整合问题理解、意图识别和RAG查询，实现完整的查询流程
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from src.intent.classification import classify_intent, route_query, IntentType
from src.rag.query import rag_query

# 导入监控模块
from src.utils.llm_monitor import llm_monitor

# ============================================================================
# 查询路由主函数
# ============================================================================

@llm_monitor(module="route_and_query")
def route_and_query(
    question: str,
    role: str = '客户经理',
    use_cot: bool = True,
    enable_rewrite: bool = True,
    enable_rerank: bool = False
) -> Dict:
    """
    完整的查询流程：query改写 → 问题理解 → 意图识别 → 能力路由 → 查询 → 返回结果
    
    参数:
        question: 用户问题
        role: 用户角色（用于权限过滤）
        use_cot: 是否使用CoT思维链推理
        enable_rewrite: 是否启用query改写（在路由之前改写，改写后的query用于意图识别和后续查询）
        enable_rerank: 是否启用重排序（RAG查询时）
    
    返回:
        Dict: 包含意图识别结果和查询结果的字典
    """
    print(f"\n{'='*60}")
    print(f"智能查询路由")
    print(f"{'='*60}")
    print(f"用户问题: {question}")
    print(f"用户角色: {role}")
    
    success = True
    rewritten_query = None
    
    try:
        # 注意：Query改写已在app.py中完成，这里直接使用传入的question
        # 如果enable_rewrite为True，说明需要在路由模块内部改写（兼容独立调用场景）
        if enable_rewrite:
            print(f"\n[步骤0] Query改写...")
            # 导入LLM调用和提示词
            from src.rag.query import call_llm
            from config.prompts import QUERY_REWRITE_PROMPT, TODAY
            
            @llm_monitor(module="query_rewrite")
            def _rewrite_query_with_monitor():
                prompt = QUERY_REWRITE_PROMPT.format(original_query=question, today=TODAY)
                rewritten = call_llm(prompt, module="query_rewrite")
                rewritten = rewritten.strip().strip('"').strip("'")
                if not rewritten or len(rewritten) < 3:
                    return question
                return rewritten
            
            rewritten_query = _rewrite_query_with_monitor()
            print(f"  改写后: {rewritten_query}")
            processed_query = rewritten_query
        else:
            # 如果禁用改写，直接使用传入的query（可能已经在app.py中改写过了）
            processed_query = question
            rewritten_query = None
        
        # 步骤1：问题理解与意图识别（使用改写后的query）
        print(f"\n[步骤1] 问题理解与意图识别（使用CoT）...")
        
        try:
            intent_result = classify_intent(processed_query, use_cot=use_cot)
            print(f"  ✓ 识别完成")
            print(f"    意图类型: {intent_result.intent.value}")
            print(f"    置信度: {intent_result.confidence:.2f}")
            print(f"    路由目标: {intent_result.route_to}")
        except Exception as e:
            # 改进错误处理：输出详细的错误信息
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"  2⚠ 问题理解与意图识别失败: {error_type}: {error_msg}")
            
            # 输出完整的异常信息
            print(f"  ⚠ 异常类型: {error_type}")
            print(f"  ⚠ 异常消息: {error_msg}")
            print(f"  ⚠ 异常参数: {getattr(e, 'args', 'N/A')}")
            
            # 如果是JSON解析相关的错误，输出更多调试信息
            if 'sub_question' in error_msg or 'JSON' in error_msg or 'json' in error_msg or 'JSONDecodeError' in error_type:
                print(f"  ⚠ 可能是JSON解析错误，错误详情: {error_msg}")
                print(f"  ⚠ 用户问题: {processed_query}")
                import traceback
                print(f"  ⚠ 完整错误堆栈:")
                traceback.print_exc()
            
            # 使用降级处理
            from src.intent.classification import fallback_intent_classification
            intent_result = fallback_intent_classification(processed_query)
            print(f"  ✓ 使用降级处理，意图类型: {intent_result.intent.value}")
        
        # 步骤2：路由决策（使用改写后的query）
        route_info = route_query(intent_result, processed_query)
        
        # 步骤3：根据路由结果执行查询
        result = {
            'intent': intent_result.intent.value,
            'intent_confidence': intent_result.confidence,
            'intent_reasoning': intent_result.reasoning,
            'route_to': intent_result.route_to,
            'question': question,  # 原始问题
            'rewritten_query': rewritten_query,  # 改写后的query（如果启用）
            'answer': None,
            'module': route_info['module']
        }
        
        if route_info['module'] == 'rag':
            # RAG查询（使用装饰器自动监控）
            print(f"\n[步骤2] 路由到RAG模块...")
            
            domain = route_info.get('domain', 'policy')
            
            try:
                # 如果已经在路由前改写过，传入改写后的query，并禁用RAG内部的改写
                rag_result = rag_query(
                    query=processed_query,  # 使用改写后的query
                    domain=domain,
                    role=role,
                    enable_rewrite=False,  # 已在路由前改写，不再重复改写
                    enable_rerank=enable_rerank
                )
                result['answer'] = rag_result['answer']
                result['rewritten_query'] = rewritten_query  # 使用路由前改写的结果
                result['search_results'] = rag_result.get('results', [])
                result['domain'] = domain
                print(f"  ✓ RAG查询完成")
            except Exception as e:
                result['error'] = f"RAG查询失败: {e}"
                result['answer'] = f"抱歉，查询时出现错误：{e}"
                print(f"  ⚠ RAG查询失败: {e}")
                success = False
        
        elif route_info['module'] == 'prediction':
            # 预测模块（待实现）
            print(f"\n[步骤2] 路由到预测模块...")
            result['answer'] = "预测模块功能正在开发中，敬请期待。"
            result['note'] = "此问题需要客户数据分析功能"
            print(f"  ℹ 预测模块待实现")
        
        else:
            # 一般性问题
            print(f"\n[步骤2] 一般性问题处理...")
            result['answer'] = "您好！我是银行信贷业务智能助手。我可以帮您：\n" \
                              "1. 查询政策规定和监管要求\n" \
                              "2. 提供系统操作指导\n" \
                              "3. 分析客户贷款意向（功能开发中）\n\n" \
                              "请告诉我您需要什么帮助？"
            print(f"  ✓ 返回通用回复")
        
        return result
        
    except Exception as e:
        success = False
        raise


# ============================================================================
# 主函数（用于测试）
# ============================================================================

def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='查询路由测试')
    parser.add_argument('question', type=str, help='用户问题')
    parser.add_argument('--role', type=str, default='客户经理', help='用户角色')
    parser.add_argument('--no-cot', action='store_true', help='禁用CoT思维链推理')
    parser.add_argument('--no-rewrite', action='store_true', help='禁用query改写')
    parser.add_argument('--no-rerank', action='store_true', help='禁用重排序')
    
    args = parser.parse_args()
    
    try:
        result = route_and_query(
            question=args.question,
            role=args.role,
            use_cot=not args.no_cot,
            enable_rewrite=not args.no_rewrite,
            enable_rerank=not args.no_rerank
        )
        
        print(f"\n{'='*60}")
        print(f"最终结果")
        print(f"{'='*60}")
        print(f"\n意图类型: {result['intent']}")
        print(f"置信度: {result['intent_confidence']:.2f}")
        print(f"路由目标: {result['route_to']}")
        print(f"处理模块: {result['module']}")
        
        print(f"\n答案：\n{result['answer']}")
        
        if result.get('rewritten_query'):
            print(f"\n改写后的查询：{result['rewritten_query']}")
        
        if result.get('search_results'):
            print(f"\n检索到的文档数量：{len(result['search_results'])}")
            for i, r in enumerate(result['search_results'][:3], 1):
                print(f"\n[文档{i}] 相似度: {r['score']:.4f}")
                print(f"  内容: {r['content'][:100]}...")
        
        if result.get('intent_reasoning'):
            print(f"\n问题理解与意图识别推理过程：")
            print(f"{'-'*60}")
            print(result['intent_reasoning'])
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

