"""
Flask Web应用
提供Web界面进行智能查询
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import traceback

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
        
        print(f"  问题: {question}")
        print(f"  角色: {role}")
        print(f"  启用改写: {enable_rewrite}")
        print(f"  启用重排序: {enable_rerank}")
        
        # 调用查询函数（添加内部异常处理）
        try:
            result = query(
                question=question,
                role=role,
                enable_rewrite=enable_rewrite,
                enable_rerank=enable_rerank,
                use_cot=True
            )
            print(f"  ✓ 查询完成")
        except Exception as query_error:
            print(f"  ❌ 查询函数执行失败: {query_error}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'查询执行失败: {str(query_error)}'
            }), 500
        
        # 返回结果
        return jsonify({
            'success': True,
            'answer': result.get('answer', ''),
            'question': result.get('question', question),
            'rewritten_query': result.get('rewritten_query'),
            'intent': result.get('intent'),
            'route_to': result.get('route_to'),
            'module': result.get('module'),
            'monitor': result.get('monitor', {}),
            'search_results': result.get('search_results', [])
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

