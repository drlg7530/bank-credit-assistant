"""
智能信贷业务辅助系统 - 应用入口
接收用户query，执行完整的查询流程

流程：
用户输入 query
      │
      ▼
【Query 改写】 → 改写为**规范、清晰、便于后续处理**的形式
      │
      ▼
【意图识别 / 路由模块】 → 判断 query 属于哪个意图
      │
      ├─ policy_query → rag_policy 查询
      │
      ├─ system_query → rag_system 查询
      │
      ├─ customer_analysis → 预测模型
      │
      └─ general → 通用回复
      │
      ▼
【结果汇总 LLM】 → 将各模块结果整合、总结输出给用户
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入模块
from src.intent.router import route_and_query
from src.rag.query import call_llm
from src.utils.llm_monitor import llm_monitor, start_request, finish_request, get_request_stats
from src.context.memory_manager import get_memory_manager
from src.intent.classification import classify_intent

# 导入配置
from config.prompts import QUERY_REWRITE_PROMPT, TODAY


def query(
    question: str,
    role: str = '客户经理',
    enable_rewrite: bool = True,
    enable_rerank: bool = False,
    use_cot: bool = True,
    session_id: Optional[str] = None,
    user_id: int = 10000
) -> Dict:
    """
    智能查询主函数（应用入口）
    
    参数:
        question: 用户问题
        role: 用户角色（'客户经理'/'团队负责人'/'行长'，用于权限过滤）
        enable_rewrite: 是否启用query改写（默认True）
        enable_rerank: 是否启用重排序（默认False）
        use_cot: 是否使用CoT思维链推理（默认True）
        session_id: 可选的session_id，如果不提供则创建新的
        user_id: 用户ID，默认10000
    
    返回:
        Dict: 包含答案和监控信息的字典
            {
                'answer': str,              # 最终答案
                'question': str,            # 原始问题
                'rewritten_query': str,     # 改写后的查询（如果启用）
                'intent': str,              # 意图类型
                'route_to': str,            # 路由目标
                'module': str,              # 处理模块
                'monitor': dict,            # 监控信息
                'session_id': str,          # Session ID
                ...
            }
    
    使用示例:
        from app import query
        
        # 基本查询
        result = query("小额贷款公司的注册资本要求是什么？")
        print(result['answer'])
        
        # 指定角色
        result = query(
            "如何在系统中查询客户授信额度？",
            role='团队负责人'
        )
        
        # 禁用query改写
        result = query(
            "查询政策",
            enable_rewrite=False
        )
    """
    # 开始新的业务请求
    start_request()
    
    print(f"\n{'='*60}")
    print(f"智能信贷业务辅助系统")
    print(f"{'='*60}")
    print(f"用户问题: {question}")
    print(f"用户角色: {role}")
    
    # 初始化记忆管理器
    memory_manager = get_memory_manager()
    
    # 获取或创建session
    session_id = memory_manager.get_or_create_session(user_id=user_id, session_id=session_id)
    print(f"Session ID: {session_id}")
    
    success = True
    rewritten_query = None
    turn_id = None
    
    try:
        # 步骤0：写入L1（原始query）
        print(f"\n[步骤0] 写入L1（原始query）...")
        turn_id = memory_manager.save_user_query(session_id=session_id, content=question)
        # 如果保存失败，turn_id为None，使用默认值1
        if turn_id is None:
            turn_id = 1
            print(f"  ⚠ L1记忆保存失败，使用默认turn_id=1")
        
        # 步骤1：Query改写（在路由之前进行）
        if enable_rewrite:
            print(f"\n[步骤1] Query改写...")
            
            @llm_monitor(module="query_rewrite")
            def _rewrite_query_with_monitor():
                # 构建提示词
                prompt = QUERY_REWRITE_PROMPT.format(original_query=question, today=TODAY)
                # 调用大模型
                rewritten = call_llm(prompt, module="query_rewrite")
                # 清理结果
                rewritten = rewritten.strip().strip('"').strip("'")
                # 如果改写失败或结果为空，返回原查询
                if not rewritten or len(rewritten) < 3:
                    return question
                return rewritten
            
            rewritten_query = _rewrite_query_with_monitor()
            print(f"  改写后: {rewritten_query}")
            
            # 使用改写后的query进行后续处理
            processed_query = rewritten_query
        else:
            processed_query = question
        
        # 步骤2：读取L2
        print(f"\n[步骤2] 读取L2状态...")
        previous_l2 = memory_manager.get_l2_state(session_id=session_id)
        if previous_l2:
            print(f"  找到上一轮L2状态: {previous_l2.get('operation_chain', [])}")
        else:
            print(f"  未找到上一轮L2状态（首次对话）")
        
        # 步骤3：意图识别 + 关键实体抽取
        print(f"\n[步骤3] 意图识别 + 关键实体抽取...")
        intent_result = classify_intent(question=processed_query, use_cot=use_cot)
        
        # 步骤4：更新/写入L2
        print(f"\n[步骤4] 更新/写入L2状态...")
        entities = {
            'active_domain': intent_result.active_domain or [],
            'business_object': intent_result.business_object or '',
            'operation_stage': intent_result.operation_stage or '',
            'last_action': processed_query,
            'current_customer_id': None  # 暂不支持客户ID，后续可扩展
        }
        new_l2_state = memory_manager.update_l2_state(
            session_id=session_id,
            intent=intent_result.intent.value,
            entities=entities,
            previous_l2=previous_l2
        )
        print(f"  更新后的L2状态: {new_l2_state.get('operation_chain', [])}")
        
        # 步骤5：路由并查询（包含功能调用）
        print(f"\n[步骤5] 路由并查询...")
        
        # 调用路由模块（传入改写后的query，不再重复改写）
        result = route_and_query(
            question=processed_query,  # 使用改写后的query
            role=role,
            use_cot=use_cot,
            enable_rewrite=False,  # 已在步骤1改写，不再重复
            enable_rerank=enable_rerank
        )
        
        # 添加改写信息到结果
        result['rewritten_query'] = rewritten_query
        result['original_question'] = question
        result['session_id'] = session_id
        result['turn_id'] = turn_id
        
        # 步骤6：写入L1（助手回答）
        print(f"\n[步骤6] 写入L1（助手回答）...")
        memory_manager.save_assistant_answer(
            session_id=session_id,
            turn_id=turn_id,
            content=result.get('answer', '')
        )
        
        # 获取监控统计信息（在finish_request之前）
        monitor_stats = get_request_stats()
        if monitor_stats:
            result['monitor'] = monitor_stats
        
        # 完成请求监控，打印统计信息
        finish_request()
        
        return result
        
    except Exception as e:
        success = False
        
        # 即使出错，也尝试保存助手回答（错误信息）到L1
        try:
            if 'session_id' in locals() and 'turn_id' in locals() and turn_id is not None:
                memory_manager = get_memory_manager()
                memory_manager.save_assistant_answer(
                    session_id=session_id,
                    turn_id=turn_id,
                    content=f"抱歉，查询时出现错误：{str(e)}"
                )
        except Exception as save_error:
            print(f"  ⚠ 保存错误信息到L1失败: {save_error}")
        
        # 获取监控统计信息（在finish_request之前）
        monitor_stats = get_request_stats()
        
        # 即使出错也要完成请求监控
        finish_request()
        
        result = {
            'answer': f"抱歉，查询时出现错误：{str(e)}",
            'question': question,
            'rewritten_query': rewritten_query,
            'error': str(e),
            'success': False
        }
        
        # 如果session_id存在，添加到结果中
        if 'session_id' in locals():
            result['session_id'] = session_id
        if 'turn_id' in locals():
            result['turn_id'] = turn_id
        
        if monitor_stats:
            result['monitor'] = monitor_stats
        
        return result


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='智能信贷业务辅助系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本查询
  python app.py "小额贷款公司的注册资本要求是什么？"
  
  # 指定角色
  python app.py "如何在系统中查询客户授信额度？" --role 团队负责人
  
  # 禁用query改写
  python app.py "查询政策" --no-rewrite
  
  # 禁用重排序
  python app.py "查询政策" --no-rerank
        """
    )
    
    parser.add_argument('question', type=str, help='用户问题')
    parser.add_argument('--role', type=str, default='客户经理',
                       choices=['客户经理', '团队负责人', '行长'],
                       help='用户角色（用于权限过滤）')
    parser.add_argument('--no-rewrite', action='store_true',
                       help='禁用query改写')
    parser.add_argument('--no-rerank', action='store_true',
                       help='禁用重排序')
    parser.add_argument('--no-cot', action='store_true',
                       help='禁用CoT思维链推理')
    
    args = parser.parse_args()
    
    try:
        # 执行查询
        result = query(
            question=args.question,
            role=args.role,
            enable_rewrite=not args.no_rewrite,
            enable_rerank=not args.no_rerank,
            use_cot=not args.no_cot
        )
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"查询结果")
        print(f"{'='*60}")
        print(f"\n答案：\n{result['answer']}")
        
        if result.get('rewritten_query'):
            print(f"\n改写后的查询：{result['rewritten_query']}")
        
        if result.get('intent'):
            print(f"\n意图类型：{result['intent']}")
            print(f"路由目标：{result.get('route_to', 'N/A')}")
        
        if result.get('search_results'):
            print(f"\n检索到的文档数量：{len(result['search_results'])}")
            for i, r in enumerate(result['search_results'][:3], 1):
                print(f"\n[文档{i}] 相似度: {r.get('score', 0):.4f}")
                print(f"  内容: {r.get('content', '')[:100]}...")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

