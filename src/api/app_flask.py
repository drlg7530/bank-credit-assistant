"""
Flask Web应用
提供Web界面进行智能查询
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import traceback
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入查询函数（从项目根目录的app.py导入）
# 注意：这里需要确保app.py在Python路径中
try:
    from app import query
except ImportError:
    # 如果直接导入失败，尝试添加项目根目录到路径
    import sys
    sys.path.insert(0, str(project_root))
    from app import query

# 创建Flask应用
app = Flask(__name__, 
            template_folder=project_root / 'templates',
            static_folder=project_root / 'static')

# ============================================================================
# 快捷问题配置（从JSON文件中固定选择，所有角色显示相同问题）
# 注意：RAG查询不涉及权限，所有角色显示相同的问题
# ============================================================================

# 固定的快捷问题列表（从data/qa_pairs/下的JSON文件中随机选择并固定）
QUICK_QUESTIONS = {
    '客户经理': [
        # 政策类问题
        "最新的1年期LPR利率是多少？",
        "5年期以上LPR现在是多少？",
        "小微企业金融服务监管评价依据什么文件？",
        "普惠金融高质量发展实施意见是哪个文号？",
        "小微企业金融服务监管评价办法谁制定的？",
        "新疆融资担保公司评级主要依据哪些法规？",
        "商业银行托管业务监督管理办法适用于哪些机构？",
        "银行业保险业数字金融的相关政策有哪些？",
        # 系统操作类问题
        "如何录入新增押品信息",
        "在哪里查询授信审批进度",
        "用信申请时如何关联已有授信额度",
        "押品价值重估的操作步骤是什么",
        "系统里怎么提交授信方案至审批环节",
        "如何录入客户基本信息",
        "在哪里提交用信申请",
        "怎么查询押品评估状态",
        "授信审批流程如何操作",
        "系统里怎么上传贷款合同文件",
        "如何在系统中录入新客户基本信息",
    ],
    '团队负责人': [
        # 政策类问题
        "最新的1年期LPR利率是多少？",
        "5年期以上LPR现在是多少？",
        "小微企业金融服务监管评价依据什么文件？",
        "普惠金融高质量发展实施意见是哪个文号？",
        "小微企业金融服务监管评价办法谁制定的？",
        "新疆融资担保公司评级主要依据哪些法规？",
        "商业银行托管业务监督管理办法适用于哪些机构？",
        "银行业保险业数字金融的相关政策有哪些？",
        # 系统操作类问题
        "如何录入新增押品信息",
        "在哪里查询授信审批进度",
        "用信申请时如何关联已有授信额度",
        "押品价值重估的操作步骤是什么",
        "系统里怎么提交授信方案至审批环节",
        "如何录入客户基本信息",
        "在哪里提交用信申请",
        "怎么查询押品评估状态",
        "授信审批流程如何操作",
        "系统里怎么上传贷款合同文件",
        "如何在系统中录入新客户基本信息",
    ],
    '行长': [
        # 政策类问题
        "最新的1年期LPR利率是多少？",
        "5年期以上LPR现在是多少？",
        "小微企业金融服务监管评价依据什么文件？",
        "普惠金融高质量发展实施意见是哪个文号？",
        "小微企业金融服务监管评价办法谁制定的？",
        "新疆融资担保公司评级主要依据哪些法规？",
        "商业银行托管业务监督管理办法适用于哪些机构？",
        "银行业保险业数字金融的相关政策有哪些？",
        # 系统操作类问题
        "如何录入新增押品信息",
        "在哪里查询授信审批进度",
        "用信申请时如何关联已有授信额度",
        "押品价值重估的操作步骤是什么",
        "系统里怎么提交授信方案至审批环节",
        "如何录入客户基本信息",
        "在哪里提交用信申请",
        "怎么查询押品评估状态",
        "授信审批流程如何操作",
        "系统里怎么上传贷款合同文件",
        "如何在系统中录入新客户基本信息",
    ]
}


# ============================================================================
# 路由定义
# ============================================================================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html', 
                         roles=['客户经理', '团队负责人', '行长'],
                         quick_questions=QUICK_QUESTIONS)


@app.route('/api/query', methods=['POST'])
def api_query():
    """
    查询API接口
    
    请求参数:
        question: 用户问题
        role: 用户角色（'客户经理'/'团队负责人'/'行长'）
        enable_rewrite: 是否启用query改写（可选，默认True）
        enable_rerank: 是否启用重排序（可选，默认True）
    
    返回:
        JSON格式的查询结果
    """
    try:
        print(f"\n{'='*60}")
        print(f"收到API查询请求")
        print(f"{'='*60}")
        
        # 获取请求参数
        data = request.get_json()
        if not data:
            print("  ⚠ 请求数据为空")
            return jsonify({'success': False, 'error': '请求数据为空'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            print("  ⚠ 问题为空")
            return jsonify({'success': False, 'error': '问题不能为空'}), 400
        
        role = data.get('role', '客户经理')
        enable_rewrite = data.get('enable_rewrite', True)
        enable_rerank = data.get('enable_rerank', True)
        session_id = data.get('session_id', None)  # 可选的session_id
        user_id = data.get('user_id', 10000)  # 用户ID，默认10000
        
        print(f"  问题: {question}")
        print(f"  角色: {role}")
        print(f"  启用改写: {enable_rewrite}")
        print(f"  启用重排序: {enable_rerank}")
        print(f"  Session ID: {session_id}")
        print(f"  用户ID: {user_id}")
        
        # 检查是否启用流式输出（优先使用请求参数，其次使用配置）
        from config.rag_config import RAG_CONFIG
        # 如果请求中没有传enable_streaming参数，则使用配置文件中的值
        if 'enable_streaming' in data:
            enable_streaming = data.get('enable_streaming')
        else:
            enable_streaming = RAG_CONFIG.get('enable_streaming', False)
        
        print(f"  流式输出: {enable_streaming} (配置值: {RAG_CONFIG.get('enable_streaming', False)})")
        
        # 如果启用流式输出，直接返回流式响应（不先执行查询）
        if enable_streaming:
            return Response(
                stream_with_context(_generate_streaming_response(
                    question, role, enable_rewrite, enable_rerank, 
                    session_id, user_id
                )),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        # 非流式输出：调用查询函数
        try:
            result = query(
                question=question,
                role=role,
                enable_rewrite=enable_rewrite,
                enable_rerank=enable_rerank,
                use_cot=True,
                session_id=session_id,
                user_id=user_id
            )
            print(f"  ✓ 查询完成")
        except Exception as query_error:
            print(f"  ❌ 查询函数执行失败: {query_error}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'查询执行失败: {str(query_error)}'
            }), 500
        
        # 返回结果（非流式）
        return jsonify({
            'success': True,
            'answer': result.get('answer', ''),
            'question': result.get('question', question),
            'rewritten_query': result.get('rewritten_query'),
            'intent': result.get('intent'),
            'route_to': result.get('route_to'),
            'module': result.get('module'),
            'monitor': result.get('monitor', {}),
            'search_results': result.get('search_results', []),
            'session_id': result.get('session_id'),  # 返回session_id，前端可以保存用于后续请求
            'turn_id': result.get('turn_id')  # 返回turn_id
        })
        
    except Exception as e:
        # 错误处理
        error_msg = str(e)
        print(f"\n❌ API查询错误: {error_msg}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


def _generate_streaming_response(question, role, enable_rewrite, enable_rerank, session_id, user_id):
    """
    生成流式响应（Server-Sent Events格式）
    
    参数:
        question: 用户问题
        role: 用户角色
        enable_rewrite: 是否启用query改写
        enable_rerank: 是否启用重排序
        session_id: session ID
        user_id: 用户ID
    
    返回:
        Generator: 生成器，逐步返回SSE格式的数据
    """
    try:
        # 导入必要的模块
        from src.rag.query import generate_answer_stream
        from src.intent.router import route_and_query
        from src.context.memory_manager import get_memory_manager
        from src.intent.classification import classify_intent
        from config.rag_config import RAG_CONFIG
        from config.prompts import QUERY_REWRITE_PROMPT, TODAY
        from src.rag.query import call_llm
        
        # 读取流式输出配置（不强制启用）
        # RAG_CONFIG['enable_streaming'] 的值由配置文件决定
        
        # 发送初始信息（立即发送）
        start_msg = f"data: {json.dumps({'type': 'start', 'message': '开始处理查询...'}, ensure_ascii=False)}\n\n"
        # print(f"[流式响应] 发送start消息: {start_msg[:100]}...")
        yield start_msg
        
        # 步骤0：写入L1（原始query）
        memory_manager = get_memory_manager()
        session_id = memory_manager.get_or_create_session(user_id=user_id, session_id=session_id)
        turn_id = memory_manager.save_user_query(session_id=session_id, content=question)
        if turn_id is None:
            turn_id = 1
        
        # 步骤1：Query改写
        rewritten_query = question
        if enable_rewrite:
            prompt = QUERY_REWRITE_PROMPT.format(original_query=question, today=TODAY)
            rewritten_query = call_llm(prompt, module="query_rewrite")
            rewritten_query = rewritten_query.strip().strip('"').strip("'")
            if not rewritten_query or len(rewritten_query) < 3:
                rewritten_query = question
        
        # 步骤2-4：意图识别和L2更新（非流式，快速完成）
        previous_l2 = memory_manager.get_l2_state(session_id=session_id)
        intent_result = classify_intent(question=rewritten_query, use_cot=True)
        
        entities = {
            'active_domain': intent_result.active_domain or [],
            'business_object': intent_result.business_object or '',
            'operation_stage': intent_result.operation_stage or '',
            'last_action': rewritten_query,
            'current_customer_id': None
        }
        memory_manager.update_l2_state(
            session_id=session_id,
            intent=intent_result.intent.value,
            entities=entities,
            previous_l2=previous_l2
        )
        
        # 步骤5：路由并查询
        route_result = route_and_query(
            question=rewritten_query,
            role=role,
            use_cot=True,
            enable_rewrite=False,
            enable_rerank=enable_rerank
        )
        
        # 发送元数据
        metadata_msg = f"data: {json.dumps({'type': 'metadata', 'intent': route_result.get('intent'), 'route_to': route_result.get('route_to'), 'module': route_result.get('module')}, ensure_ascii=False)}\n\n"
        # print(f"[流式响应] 发送metadata消息")
        yield metadata_msg
        
        # 步骤6：流式生成答案
        answer_chunks = []
        if route_result.get('module') == 'rag' and route_result.get('search_results'):
            # 使用流式生成答案
            from src.rag.query import SearchResult
            # 将搜索结果字典转换为SearchResult对象
            search_results = []
            for r in route_result.get('search_results', []):
                search_result = SearchResult(
                    content=r.get('content', ''),
                    score=r.get('score', 0.0),
                    metadata=r.get('metadata', {}),
                    chunk_id=r.get('chunk_id', '')
                )
                search_results.append(search_result)
            
            domain = route_result.get('domain', 'policy')
            # print(f"[流式响应] 准备生成答案，搜索结果数量: {len(search_results)}, domain: {domain}")
            try:
                chunk_count = 0
                for chunk in generate_answer_stream(
                    query=rewritten_query,
                    search_results=search_results,
                    domain=domain
                ):
                    answer_chunks.append(chunk)
                    chunk_count += 1
                    chunk_msg = f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                    # if chunk_count <= 3:  # 只打印前3个chunk的日志
                    #     print(f"[流式响应] 发送chunk #{chunk_count}: {chunk[:50]}...")
                    yield chunk_msg
                # print(f"[流式响应] 总共发送了 {chunk_count} 个chunk")
                # if chunk_count == 0:
                #     print(f"[流式响应] ⚠️ 警告：没有收到任何chunk，可能流式生成失败")
            except Exception as e:
                error_msg = f"流式生成答案失败: {str(e)}"
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg}, ensure_ascii=False)}\n\n"
        else:
            # 非RAG模块，直接返回答案
            answer = route_result.get('answer', '')
            # 逐字符流式返回（模拟流式效果）
            for char in answer:
                answer_chunks.append(char)
                yield f"data: {json.dumps({'type': 'chunk', 'content': char}, ensure_ascii=False)}\n\n"
        
        # 完整答案
        full_answer = ''.join(answer_chunks)
        
        # 打印最终完整的LLM重组内容（方便排查问题）
        # print(f"\n{'='*80}")
        # print(f"[流式响应] 最终完整的LLM重组内容:")
        # print(f"{'='*80}")
        # print(full_answer)
        # print(f"{'='*80}\n")
        
        # 保存到L1
        memory_manager.save_assistant_answer(
            session_id=session_id,
            turn_id=turn_id,
            content=full_answer
        )
        
        # 获取监控统计信息
        from src.utils.llm_monitor import get_request_stats, finish_request
        monitor_stats = get_request_stats()
        finish_request()
        
        # 发送完成信息（包含监控统计）
        done_msg = f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'turn_id': turn_id, 'monitor': monitor_stats}, ensure_ascii=False)}\n\n"
        # print(f"[流式响应] 发送done消息")
        yield done_msg
        
    except Exception as e:
        error_msg = str(e)
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"


@app.route('/api/query/stream', methods=['POST'])
def api_query_stream():
    """
    流式查询API接口（Server-Sent Events）
    注意：此端点强制使用流式输出，如需根据配置控制，请使用 /api/query 端点
    
    请求参数:
        question: 用户问题
        role: 用户角色
        enable_rewrite: 是否启用query改写
        enable_rerank: 是否启用重排序
        session_id: 可选的session_id
        user_id: 用户ID
    
    返回:
        Server-Sent Events格式的流式响应
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请求数据为空'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'success': False, 'error': '问题不能为空'}), 400
        
        role = data.get('role', '客户经理')
        enable_rewrite = data.get('enable_rewrite', True)
        enable_rerank = data.get('enable_rerank', True)
        session_id = data.get('session_id', None)
        user_id = data.get('user_id', 10000)
        
        # 此端点强制使用流式输出
        return Response(
            stream_with_context(_generate_streaming_response(
                question, role, enable_rewrite, enable_rerank,
                session_id, user_id
            )),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/conversation-history', methods=['GET'])
def api_conversation_history():
    """
    获取对话历史记录
    
    查询参数:
        session_id: session ID（必需）
        limit: 返回的最大轮次数（可选，默认50）
    
    返回:
        JSON格式的对话历史记录列表
    """
    try:
        session_id = request.args.get('session_id', None)
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'session_id参数不能为空'
            }), 400
        
        limit = int(request.args.get('limit', 50))
        
        # 获取记忆管理器
        from src.context.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()
        
        # 获取历史记录
        history = memory_manager.get_conversation_history(session_id=session_id, limit=limit)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        print(f"  ❌ 获取对话历史失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'获取对话历史失败: {str(e)}'
        }), 500


@app.route('/api/quick-questions', methods=['GET'])
def api_quick_questions():
    """
    获取快捷问题列表
    
    查询参数:
        role: 用户角色（可选，如果不提供则返回所有角色的快捷问题）
    
    返回:
        JSON格式的快捷问题列表
    """
    role = request.args.get('role', None)
    
    if role and role in QUICK_QUESTIONS:
        return jsonify({
            'success': True,
            'role': role,
            'questions': QUICK_QUESTIONS[role]
        })
    else:
        # 返回所有角色的快捷问题
        return jsonify({
            'success': True,
            'questions': QUICK_QUESTIONS
        })


# ============================================================================
# 主函数
# ============================================================================

def main():
    """启动Flask应用"""
    print("\n" + "="*60)
    print("智能信贷业务辅助系统 - Web界面")
    print("="*60)
    print("访问地址: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # 禁用自动重载（use_reloader=False），避免模型加载时触发不必要的重启
    # 注意：修改代码后需要手动重启应用
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()

