"""
Elasticsearch数据保存和查询测试脚本

功能：
1. 保存混合数据到Elasticsearch（同时包含向量和文本字段）
2. 执行查询测试（支持文本搜索、向量搜索和混合搜索）

使用说明：
1. 混合索引：单个索引同时包含向量字段和文本字段
2. 文本搜索：使用BM25算法进行全文搜索
3. 向量搜索：使用KNN算法进行相似度搜索
4. 混合搜索：ES原生Hybrid Search（BM25 + 向量搜索）

参考代码：
- cankao/elasticsearch_index_search.py 的 index_documents() 函数
- cankao/es_doc_search_embedding.py 的 index_documents_with_vectors() 函数
"""

import sys
import os
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.elasticsearch import ES_CONFIG, INDEX_CONFIG, BATCH_SIZE
from config.rag_config import get_rag_config

# 导入必要的函数
from scripts.rag.build_vector_db import (
    test_elasticsearch_connection,
    load_embedding_model,
    generate_embeddings,
    create_vector_index,
    index_documents_with_vectors
)

# Embedding模型配置（与build_vector_db.py保持一致）
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')

from src.rag.query import (
    search_vectors,
    generate_query_embedding,
    SearchResult
)

# ============================================================================
# 依赖检查
# ============================================================================

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠ 警告: elasticsearch未安装")
    print("   请安装: pip install elasticsearch -i https://pypi.tuna.tsinghua.edu.cn/simple")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠ 警告: numpy未安装")

# ============================================================================
# 步骤1：连接Elasticsearch
# ============================================================================

def connect_elasticsearch() -> Elasticsearch:
    """
    连接到Elasticsearch服务器
    
    功能：
    - 使用配置文件中的连接信息
    - 测试连接是否成功
    
    返回：
    - Elasticsearch客户端对象
    """
    print("\n" + "="*60)
    print("步骤1：连接Elasticsearch")
    print("="*60)
    
    if not ELASTICSEARCH_AVAILABLE:
        raise Exception("elasticsearch库未安装")
    
    try:
        # 创建Elasticsearch客户端（使用配置文件中的配置）
        es_client = Elasticsearch(**ES_CONFIG)
        
        # 测试连接
        cluster_info = es_client.info()
        print(f"✓ 成功连接到Elasticsearch")
        print(f"  集群名称: {cluster_info.get('cluster_name', '未知')}")
        print(f"  版本: {cluster_info.get('version', {}).get('number', '未知')}")
        return es_client
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        raise


# 注意：create_vector_index 和 index_documents_with_vectors 函数已从 build_vector_db.py 导入


# ============================================================================
# 步骤4：准备文档块数据（从QA对JSON文件读取）
# ============================================================================

def prepare_chunks_from_qa_pairs(domain: str = 'system') -> List[Dict]:
    """
    从QA对JSON文件中读取数据并转换为文档块格式
    
    参数：
    - domain: 域类型（policy/system）
    
    返回：
    - 文档块列表，每个块包含title, source, content等信息
    
    功能：
    - 读取 data/qa_pairs/{domain}/{domain}_qa_pairs.json 文件
    - 将每个QA对转换为文档块格式
    - 使用question和answer组合作为content
    """
    print("\n" + "="*60)
    print("步骤4：从QA对JSON文件读取文档块数据")
    print("="*60)
    
    # 构建QA对JSON文件路径
    qa_file = project_root / "data" / "qa_pairs" / domain / f"{domain}_qa_pairs.json"
    
    if not qa_file.exists():
        raise FileNotFoundError(f"QA对文件不存在: {qa_file}")
    
    print(f"📁 正在读取QA对文件: {qa_file}")
    
    try:
        import json
        # 读取JSON文件
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        print(f"✓ 成功读取 {len(qa_pairs)} 个QA对")
        
        # 将QA对转换为文档块格式
        chunks = []
        for idx, qa in enumerate(qa_pairs):
            # 提取QA对信息
            qa_id = qa.get('id', f'{domain}_{idx:04d}')
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            created_at = qa.get('created_at', '')
            
            # 组合question和answer作为content（便于搜索）
            # 格式：问题：xxx\n答案：xxx
            content = f"问题：{question}\n答案：{answer}"
            
            # 计算token数量（简单估算：1 token ≈ 4字符）
            token_count = len(content) // 4
            
            # 构建文档块数据
            chunk = {
                'title': question[:100] if len(question) > 100 else question,  # 使用问题作为标题
                'source': str(qa_file),  # 源文件路径
                'content': content,      # 问题和答案的组合内容
                'chunk_id': idx,         # 块ID（使用索引）
                'page_num': 1,           # 页码（QA对通常不分页）
                'token_count': token_count,  # Token数量
                'file_type': 'json',     # 文件类型
                'domain': '系统' if domain == 'system' else '政策',  # 域类型（中文）
                'doc_type': '操作手册' if domain == 'system' else '监管政策',  # 文档类型
                'region': '全国',        # 地区范围
                'publish_date': created_at[:10] if created_at else '2024-01-01',  # 发布时间（提取日期部分）
                'status': '生效',        # 状态
                'role': '客户经理',      # 默认角色
                # 额外字段：保存原始QA对信息
                'qa_id': qa_id,         # QA对ID
                'question': question,    # 原始问题
                'answer': answer         # 原始答案
            }
            chunks.append(chunk)
        
        print(f"✓ 成功转换 {len(chunks)} 个文档块")
        print(f"  示例：")
        if chunks:
            print(f"    - 标题: {chunks[0]['title']}")
            print(f"    - 内容长度: {len(chunks[0]['content'])} 字符")
            print(f"    - Token数: {chunks[0]['token_count']}")
        
        return chunks
        
    except json.JSONDecodeError as e:
        raise Exception(f"JSON文件解析失败: {e}")
    except Exception as e:
        raise Exception(f"读取QA对文件失败: {e}")


# 注意：index_documents_with_vectors 函数已从 build_vector_db.py 导入


# ============================================================================
# 步骤7：文本搜索测试
# ============================================================================

def test_text_search(es: Elasticsearch, index_name: str, query_text: str, top_k: int = 10):
    """
    测试文本搜索功能（非向量模式）
    
    参数：
    - es: Elasticsearch客户端对象
    - index_name: 索引名称
    - query_text: 搜索查询文本
    - top_k: 返回前K个结果
    
    功能：
    - 使用multi_match查询在title和content字段中搜索
    - 返回相关性评分最高的文档
    """
    print("\n" + "="*60)
    print("步骤7：文本搜索测试")
    print("="*60)
    
    print(f"🔍 搜索查询: {query_text}")
    print(f"   返回前 {top_k} 个结果")
    
    # 构建搜索查询
    # 使用multi_match查询，在title和content字段中搜索
    search_body = {
        "query": {
            "multi_match": {
                "query": query_text,              # 搜索查询
                "fields": ["title^2", "content"], # 搜索字段，title权重为2（更重要）
                "type": "best_fields"             # 最佳字段匹配
            }
        },
        "highlight": {  # 高亮显示匹配的文本
            "fields": {
                "title": {},
                "content": {
                    "fragment_size": 200,      # 片段大小
                    "number_of_fragments": 3   # 返回的片段数
                }
            },
            "pre_tags": ["<mark>"],   # 高亮开始标签
            "post_tags": ["</mark>"]  # 高亮结束标签
        },
        "size": top_k  # 返回结果数量
    }
    
    try:
        # 执行搜索
        response = es.search(
            index=index_name,
            query=search_body["query"],
            highlight=search_body["highlight"],
            size=search_body["size"]
        )
        
        # 解析搜索结果
        hits = response['hits']['hits']
        total = response['hits']['total']['value']
        
        print(f"\n✓ 搜索完成！")
        print(f"  找到 {total} 个相关文档")
        print(f"  返回前 {len(hits)} 个结果\n")
        
        # 显示结果
        if hits:
            for i, hit in enumerate(hits, 1):
                score = hit['_score']
                source = hit['_source']
                highlight = hit.get('highlight', {})
                
                print(f"[结果 {i}] 相似度: {score:.4f}")
                print(f"  标题: {source.get('title', '无标题')}")
                print(f"  来源: {source.get('source', '未知')}")
                print(f"  内容: {source.get('content', '')[:200]}...")
                if highlight:
                    print(f"  高亮: {highlight}")
                print()
        else:
            print("  ⚠ 未找到匹配的结果")
        
        return hits
        
    except Exception as e:
        print(f"✗ 搜索失败: {str(e)}")
        raise


# ============================================================================
# 步骤8：向量搜索测试
# ============================================================================

def test_vector_search(query_text: str, domain: str, role: str = '客户经理', top_k: int = 10):
    """
    测试向量搜索功能（向量模式）
    
    参数：
    - query_text: 查询文本
    - domain: 域类型
    - role: 用户角色
    - top_k: 返回前K个结果
    
    功能：
    - 将查询文本转换为向量
    - 使用knn查询进行向量搜索
    """
    print("\n" + "="*60)
    print("步骤8：向量搜索测试")
    print("="*60)
    
    print(f"查询文本: {query_text}")
    print(f"域类型: {domain}")
    print(f"用户角色: {role}")
    
    # 获取索引配置
    index_config = INDEX_CONFIG.get(domain)
    if not index_config:
        print(f"  ⚠ 未找到域 {domain} 的索引配置")
        return
    
    index_name = index_config['index_name']
    
    # 向量化查询
    print(f"\n[查询步骤1] 向量化查询...")
    try:
        query_vector = generate_query_embedding(query_text)
        print(f"  ✓ 向量维度: {query_vector.shape[0]}")
    except Exception as e:
        print(f"  ⚠ 向量化失败: {e}")
        return
    
    # 执行搜索
    print(f"\n[查询步骤2] Elasticsearch向量搜索...")
    try:
        config = get_rag_config()
        results = search_vectors(
            query_vector=query_vector,
            index_name=index_name,
            domain=domain,
            role=role,
            top_k=top_k or config['top_k']
        )
        
        print(f"\n✓ 检索到 {len(results)} 条结果\n")
        
        # 显示结果
        if results:
            print(f"{'='*60}")
            print(f"查询结果")
            print(f"{'='*60}")
            for i, result in enumerate(results, 1):
                print(f"\n[结果 {i}] 相似度: {result.score:.4f}")
                print(f"  内容: {result.content[:200]}...")
                print(f"  元数据: {result.metadata}")
        else:
            print("  ⚠ 未找到匹配的结果")
            print(f"\n  💡 提示：索引 {index_name} 中没有文档，需要先构建向量库")
            
    except Exception as e:
        print(f"  ⚠ 搜索失败: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 主函数
# ============================================================================

def test_hybrid_search(es: Elasticsearch, index_name: str, query_text: str, query_vector: np.ndarray, top_k: int = 10):
    """
    测试混合搜索功能（Hybrid Search：BM25 + 向量搜索）
    
    参数：
    - es: Elasticsearch客户端对象
    - index_name: 索引名称
    - query_text: 搜索查询文本
    - query_vector: 查询向量
    - top_k: 返回前K个结果
    
    功能：
    - 同时使用BM25文本搜索和KNN向量搜索
    - ES原生Hybrid Search会自动合并两种搜索结果
    """
    print("\n" + "="*60)
    print("步骤9：混合搜索测试（Hybrid Search：BM25 + 向量）")
    print("="*60)
    
    print(f"🔍 搜索查询: {query_text}")
    print(f"   返回前 {top_k} 个结果")
    print(f"   搜索模式: 混合搜索（BM25 + KNN向量搜索）")
    
    # 构建混合搜索查询
    # ES原生Hybrid Search：同时包含query（BM25）和knn（向量搜索）
    search_body = {
        # 向量搜索（KNN）
        "knn": {
            "field": "embedding",
            "query_vector": query_vector.tolist(),
            "k": top_k,
            "num_candidates": top_k * 10
        },
        # 文本搜索（BM25）
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^2", "content", "question", "answer"],
                "type": "best_fields"
            }
        },
        "size": top_k,
        "_source": {
            "excludes": ["embedding"]  # 不返回向量字段（减少传输量）
        }
    }
    
    try:
        # 执行混合搜索（ES 9.x版本：直接传递参数）
        response = es.search(
            index=index_name,
            knn=search_body["knn"],
            query=search_body["query"],
            size=search_body["size"],
            _source=search_body["_source"]
        )
        
        # 解析搜索结果
        hits = response['hits']['hits']
        total = response['hits']['total']['value']
        
        print(f"\n✓ 混合搜索完成！")
        print(f"  找到 {total} 个相关文档")
        print(f"  返回前 {len(hits)} 个结果\n")
        
        # 显示结果
        if hits:
            for i, hit in enumerate(hits, 1):
                score = hit['_score']
                source = hit['_source']
                
                print(f"[结果 {i}] 综合评分: {score:.4f}")
                print(f"  标题: {source.get('title', '无标题')}")
                print(f"  来源: {source.get('source', '未知')}")
                print(f"  内容: {source.get('content', '')[:200]}...")
                print()
        else:
            print("  ⚠ 未找到匹配的结果")
        
        return hits
        
    except Exception as e:
        print(f"✗ 混合搜索失败: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Elasticsearch数据保存和查询测试脚本')
    parser.add_argument('--mode', type=str, choices=['hybrid'], default='hybrid',
                       help='保存模式：hybrid（混合索引，同时包含向量和文本字段）')
    parser.add_argument('--domain', type=str, choices=['policy', 'system'], default='system',
                       help='域类型（默认: system）')
    parser.add_argument('--query', type=str, default=None,
                       help='测试查询文本（如果不提供，将提示手动输入）')
    parser.add_argument('--role', type=str, default='客户经理',
                       help='用户角色（默认: 客户经理）')
    parser.add_argument('--test-only', action='store_true',
                       help='仅执行查询测试，不保存数据')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Elasticsearch数据保存和查询测试脚本")
    print("=" * 60)
    print(f"保存模式: {args.mode}")
    print(f"域类型: {args.domain}")
    print(f"用户角色: {args.role}")
    if args.test_only:
        print("模式: 仅查询测试")
    print("=" * 60)
    
    # 检查依赖
    if not ELASTICSEARCH_AVAILABLE:
        print("⚠ 错误: elasticsearch未安装")
        return
    
    # 检查Elasticsearch连接
    if not test_elasticsearch_connection():
        print("\n⚠ 错误: Elasticsearch连接失败")
        return
    
    # 连接Elasticsearch
    es = connect_elasticsearch()
    
    # 获取索引配置
    index_config = INDEX_CONFIG.get(args.domain)
    if not index_config:
        print(f"⚠ 未找到域 {args.domain} 的索引配置")
        return
    
    index_name = index_config['index_name']
    
    # 如果不是仅测试模式，执行数据保存
    if not args.test_only:
        # 从QA对JSON文件读取数据
        chunks = prepare_chunks_from_qa_pairs(args.domain)
        
        # 保存混合数据（单个索引，同时包含向量和文本字段）
        if args.mode == 'hybrid':
            # 加载embedding模型
            print("\n正在加载embedding模型...")
            try:
                model, tokenizer = load_embedding_model(EMBEDDING_MODEL_NAME)
                
                # 生成向量
                print("正在生成向量...")
                texts = [chunk['content'] for chunk in chunks]
                embeddings = generate_embeddings(texts, model, tokenizer, batch_size=4)
                
                # 创建混合索引（同时支持向量搜索和文本搜索）
                create_vector_index(es, index_name, embeddings.shape[1])
                
                # 索引文档和向量（单个索引包含向量和文本字段）
                index_documents_with_vectors(es, index_name, chunks, embeddings)
            except Exception as e:
                print(f"⚠ 向量生成或保存失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 执行查询测试
    print("\n" + "="*60)
    print("查询测试")
    print("="*60)
    
    # 如果没有提供查询文本，提示用户输入
    if args.query is None or args.query.strip() == '':
        print(f"\n未提供查询问题，请手动输入：")
        query_text = input("请输入查询问题: ").strip()
        
        if not query_text:
            print("⚠ 错误: 查询问题不能为空")
            return
    else:
        query_text = args.query
    
    # 测试混合搜索（Hybrid Search：BM25 + 向量）
    if args.mode == 'hybrid':
        print("\n" + "="*60)
        print("测试混合搜索（Hybrid Search：BM25 + 向量）")
        print("="*60)
        try:
            # 生成查询向量
            if not NUMPY_AVAILABLE:
                print("⚠ 错误: numpy未安装，无法生成查询向量")
                return
            
            query_vector = generate_query_embedding(query_text)
            # 执行混合搜索
            test_hybrid_search(es, index_name, query_text, query_vector, top_k=10)
        except Exception as e:
            print(f"⚠ 混合搜索测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 可选：单独测试文本搜索和向量搜索
        print("\n" + "="*60)
        print("额外测试：文本搜索（BM25）")
        print("="*60)
        try:
            test_text_search(es, index_name, query_text, top_k=5)
        except Exception as e:
            print(f"⚠ 文本搜索测试失败: {e}")
        
        print("\n" + "="*60)
        print("额外测试：向量搜索（KNN）")
        print("="*60)
        try:
            test_vector_search(query_text, args.domain, args.role, top_k=5)
        except Exception as e:
            print(f"⚠ 向量搜索测试失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
