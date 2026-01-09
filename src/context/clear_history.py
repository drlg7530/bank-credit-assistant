"""
清理历史记录模块
功能：删除ES中指定用户或指定session的历史记录
"""

import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.elasticsearch import ES_CONFIG, INDEX_CONFIG

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("❌ 错误: elasticsearch未安装")


def clear_user_history(user_id: int = 10000):
    """
    清理指定用户的所有历史记录
    
    参数:
        user_id: 用户ID，默认10000
    
    返回:
        dict: 清理结果统计
    """
    if not ELASTICSEARCH_AVAILABLE:
        print("❌ Elasticsearch不可用")
        return {'success': False, 'error': 'Elasticsearch不可用'}
    
    try:
        # 创建ES客户端
        es_client = Elasticsearch(**ES_CONFIG)
        es_client.ping()
        print(f"✓ Elasticsearch连接成功")
    except Exception as e:
        print(f"❌ Elasticsearch连接失败: {e}")
        return {'success': False, 'error': f'Elasticsearch连接失败: {e}'}
    
    result = {
        'success': True,
        'l1_deleted': 0,
        'session_deleted': 0,
        'l2_deleted': 0
    }
    
    print(f"\n{'='*60}")
    print(f"开始清理用户 {user_id} 的历史记录")
    print(f"{'='*60}\n")
    
    # 1. 清理L1记忆（对话记录）
    l1_index = INDEX_CONFIG.get('l1_memory', {}).get('index_name', 'bank_credit_l1_memory')
    print(f"[1] 清理L1记忆索引: {l1_index}")
    
    try:
        if es_client.indices.exists(index=l1_index):
            # 先查询该用户的所有session_id
            session_index = INDEX_CONFIG.get('session_record', {}).get('index_name', 'bank_credit_session_record')
            
            if es_client.indices.exists(index=session_index):
                # 查询该用户的所有session_id
                query = {
                    "query": {
                        "term": {"user_id": user_id}
                    },
                    "size": 10000  # 获取所有session
                }
                response = es_client.search(index=session_index, body=query)
                session_ids = [hit['_source']['session_id'] for hit in response.get('hits', {}).get('hits', [])]
                
                if session_ids:
                    print(f"  找到 {len(session_ids)} 个session")
                    # 删除这些session的所有L1记忆
                    delete_query = {
                        "query": {
                            "terms": {"session_id": session_ids}
                        }
                    }
                    delete_result = es_client.delete_by_query(index=l1_index, body=delete_query)
                    deleted_count = delete_result.get('deleted', 0)
                    result['l1_deleted'] = deleted_count
                    print(f"  ✓ 删除了 {deleted_count} 条L1记忆记录")
                else:
                    print(f"  ℹ 未找到该用户的session记录")
            else:
                print(f"  ⚠ Session记录索引不存在，跳过L1记忆清理")
        else:
            print(f"  ℹ L1记忆索引不存在")
    except Exception as e:
        print(f"  ❌ 清理L1记忆失败: {e}")
        result['success'] = False
        result['error'] = f'清理L1记忆失败: {e}'
    
    # 2. 清理Session记录
    session_index = INDEX_CONFIG.get('session_record', {}).get('index_name', 'bank_credit_session_record')
    print(f"\n[2] 清理Session记录索引: {session_index}")
    
    try:
        if es_client.indices.exists(index=session_index):
            # 删除该用户的所有session记录
            delete_query = {
                "query": {
                    "term": {"user_id": user_id}
                }
            }
            delete_result = es_client.delete_by_query(index=session_index, body=delete_query)
            deleted_count = delete_result.get('deleted', 0)
            result['session_deleted'] = deleted_count
            print(f"  ✓ 删除了 {deleted_count} 条Session记录")
        else:
            print(f"  ℹ Session记录索引不存在")
    except Exception as e:
        print(f"  ❌ 清理Session记录失败: {e}")
        result['success'] = False
        if 'error' not in result:
            result['error'] = f'清理Session记录失败: {e}'
    
    # 3. 清理L2记忆（如果有user_id字段）
    l2_index = INDEX_CONFIG.get('l2_memory', {}).get('index_name', 'bank_credit_l2_memory')
    print(f"\n[3] 清理L2记忆索引: {l2_index}")
    
    try:
        if es_client.indices.exists(index=l2_index):
            # L2记忆通过session_id关联，先获取该用户的所有session_id
            session_index = INDEX_CONFIG.get('session_record', {}).get('index_name', 'bank_credit_session_record')
            if es_client.indices.exists(index=session_index):
                query = {
                    "query": {
                        "term": {"user_id": user_id}
                    },
                    "size": 10000
                }
                response = es_client.search(index=session_index, body=query)
                session_ids = [hit['_source']['session_id'] for hit in response.get('hits', {}).get('hits', [])]
                
                if session_ids:
                    delete_query = {
                        "query": {
                            "terms": {"session_id": session_ids}
                        }
                    }
                    delete_result = es_client.delete_by_query(index=l2_index, body=delete_query)
                    deleted_count = delete_result.get('deleted', 0)
                    result['l2_deleted'] = deleted_count
                    print(f"  ✓ 删除了 {deleted_count} 条L2记忆记录")
                else:
                    print(f"  ℹ 未找到该用户的session记录，跳过L2记忆清理")
            else:
                print(f"  ℹ Session记录索引不存在，跳过L2记忆清理")
        else:
            print(f"  ℹ L2记忆索引不存在")
    except Exception as e:
        print(f"  ❌ 清理L2记忆失败: {e}")
        result['success'] = False
        if 'error' not in result:
            result['error'] = f'清理L2记忆失败: {e}'
    
    print(f"\n{'='*60}")
    print(f"清理完成")
    print(f"{'='*60}\n")
    
    return result


def clear_session_history(session_id: str):
    """
    清理指定session的所有历史记录
    
    参数:
        session_id: session ID
    
    返回:
        dict: 清理结果统计
    """
    if not ELASTICSEARCH_AVAILABLE:
        return {'success': False, 'error': 'Elasticsearch不可用'}
    
    try:
        # 创建ES客户端
        es_client = Elasticsearch(**ES_CONFIG)
        es_client.ping()
    except Exception as e:
        return {'success': False, 'error': f'Elasticsearch连接失败: {e}'}
    
    result = {
        'success': True,
        'l1_deleted': 0,
        'session_deleted': 0,
        'l2_deleted': 0
    }
    
    print(f"\n{'='*60}")
    print(f"开始清理session {session_id} 的历史记录")
    print(f"{'='*60}\n")
    
    # 1. 清理L1记忆（对话记录）
    l1_index = INDEX_CONFIG.get('l1_memory', {}).get('index_name', 'bank_credit_l1_memory')
    print(f"[1] 清理L1记忆索引: {l1_index}")
    
    try:
        if es_client.indices.exists(index=l1_index):
            delete_query = {
                "query": {
                    "term": {"session_id": session_id}
                }
            }
            delete_result = es_client.delete_by_query(index=l1_index, body=delete_query)
            deleted_count = delete_result.get('deleted', 0)
            result['l1_deleted'] = deleted_count
            print(f"  ✓ 删除了 {deleted_count} 条L1记忆记录")
        else:
            print(f"  ℹ L1记忆索引不存在")
    except Exception as e:
        print(f"  ❌ 清理L1记忆失败: {e}")
        result['success'] = False
        result['error'] = f'清理L1记忆失败: {e}'
    
    # 2. 清理Session记录
    session_index = INDEX_CONFIG.get('session_record', {}).get('index_name', 'bank_credit_session_record')
    print(f"\n[2] 清理Session记录索引: {session_index}")
    
    try:
        if es_client.indices.exists(index=session_index):
            # 删除该session的记录
            try:
                es_client.delete(index=session_index, id=session_id)
                result['session_deleted'] = 1
                print(f"  ✓ 删除了 1 条Session记录")
            except Exception as e:
                # 如果删除失败（记录不存在），不影响整体流程
                print(f"  ℹ Session记录不存在或已删除: {e}")
        else:
            print(f"  ℹ Session记录索引不存在")
    except Exception as e:
        print(f"  ❌ 清理Session记录失败: {e}")
        result['success'] = False
        if 'error' not in result:
            result['error'] = f'清理Session记录失败: {e}'
    
    # 3. 清理L2记忆
    l2_index = INDEX_CONFIG.get('l2_memory', {}).get('index_name', 'bank_credit_l2_memory')
    print(f"\n[3] 清理L2记忆索引: {l2_index}")
    
    try:
        if es_client.indices.exists(index=l2_index):
            delete_query = {
                "query": {
                    "term": {"session_id": session_id}
                }
            }
            delete_result = es_client.delete_by_query(index=l2_index, body=delete_query)
            deleted_count = delete_result.get('deleted', 0)
            result['l2_deleted'] = deleted_count
            print(f"  ✓ 删除了 {deleted_count} 条L2记忆记录")
        else:
            print(f"  ℹ L2记忆索引不存在")
    except Exception as e:
        print(f"  ❌ 清理L2记忆失败: {e}")
        result['success'] = False
        if 'error' not in result:
            result['error'] = f'清理L2记忆失败: {e}'
    
    print(f"\n{'='*60}")
    print(f"清理完成")
    print(f"{'='*60}\n")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='清理历史记录')
    parser.add_argument('--user-id', type=int, default=None, help='用户ID（清理该用户的所有历史记录）')
    parser.add_argument('--session-id', type=str, default=None, help='Session ID（清理该session的所有历史记录）')
    
    args = parser.parse_args()
    
    if args.session_id:
        # 清理指定session
        print(f"\n⚠️  警告：此操作将删除session {args.session_id} 的所有历史记录！")
        confirm = input("确认继续？(yes/no): ")
        if confirm.lower() == 'yes':
            clear_session_history(session_id=args.session_id)
        else:
            print("操作已取消")
    elif args.user_id:
        # 清理指定用户
        print(f"\n⚠️  警告：此操作将删除用户 {args.user_id} 的所有历史记录！")
        confirm = input("确认继续？(yes/no): ")
        if confirm.lower() == 'yes':
            clear_user_history(user_id=args.user_id)
        else:
            print("操作已取消")
    else:
        print("请指定 --user-id 或 --session-id 参数")

