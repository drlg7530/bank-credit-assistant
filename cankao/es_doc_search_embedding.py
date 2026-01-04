"""
基于 Embedding 向量的 Elasticsearch 文档索引和搜索脚本

功能：
1. 连接到 Elasticsearch
2. 创建支持向量搜索的索引（如果不存在）
3. 解析 docs 文件夹中的文档并分块
4. 使用 text-embedding-v4 生成文档向量
5. 将文档块和向量索引到 Elasticsearch
6. 使用向量搜索执行查询
7. 显示搜索结果

使用前准备：
1. 安装依赖：
   pip install elasticsearch openai -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install "qwen-agent[rag]" -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 设置环境变量：
   export DASHSCOPE_API_KEY=your_api_key

3. 如果 ES 不在本地，请修改 ES_HOST 配置

作者：AI Assistant
日期：2025-12-27
"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from qwen_agent.tools.doc_parser import DocParser
from openai import OpenAI

# ====== 步骤 1：配置 Elasticsearch 连接 ======
# Elasticsearch 连接配置
ES_HOST = "localhost"  # ES 服务器地址
ES_PORT = 9200  # ES 端口
ES_USERNAME = "elastic"  # ES 用户名
ES_PASSWORD = "elastic"  # ES 密码（已更新为 elastic）
# 注意：如果 ES 使用 HTTPS，请确保密码正确

# 索引配置
INDEX_NAME = "pingan_employer_insurance_embedding"  # 索引名称（使用不同的索引名避免冲突）

# 文档文件夹路径
DOCS_FOLDER = "./docs"  # 文档文件夹路径

# Embedding 配置
EMBEDDING_MODEL = "text-embedding-v4"  # 使用 text-embedding-v4 模型
EMBEDDING_DIMENSIONS = 1024  # 向量维度（text-embedding-v4 支持 1024 维）
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key


# ====== 步骤 2：初始化 Embedding 客户端 ======
def init_embedding_client():
    """
    初始化 OpenAI 客户端（用于调用 text-embedding-v4）
    
    返回：
    - OpenAI 客户端对象
    """
    if not DASHSCOPE_API_KEY:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的 base_url
    )
    return client


# ====== 步骤 3：生成文本向量 ======
def generate_embedding(text: str, client: OpenAI) -> List[float]:
    """
    使用 text-embedding-v4 生成文本的向量嵌入
    
    参数：
    - text: 要生成向量的文本
    - client: OpenAI 客户端对象
    
    返回：
    - 向量列表（浮点数列表）
    
    功能：
    - 调用 text-embedding-v4 模型
    - 返回 1024 维的向量
    """
    try:
        # 调用 embedding API
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS,  # 指定向量维度
            encoding_format="float"  # 返回浮点数格式
        )
        
        # 提取向量
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"❌ 生成向量失败: {str(e)}")
        raise


# ====== 步骤 4：连接到 Elasticsearch ======
def connect_elasticsearch():
    """
    连接到 Elasticsearch 服务器
    
    功能：
    - 使用用户名和密码进行身份验证
    - 测试连接是否成功
    - 提供详细的错误诊断
    
    返回：
    - Elasticsearch 客户端对象
    """
    print("=" * 60)
    print("步骤 1：连接到 Elasticsearch")
    print("=" * 60)
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 先检查 ES 服务是否可达（简单测试）
    import socket
    print("🔍 检查 ES 服务状态...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((ES_HOST, ES_PORT))
        sock.close()
        if result != 0:
            print(f"❌ 无法连接到 {ES_HOST}:{ES_PORT}，端口可能未开放")
            print("   请确认 Elasticsearch 服务已启动")
            print("   启动命令: cd D:\\Software\\elasticsearch-9.2.3\\bin && elasticsearch.bat")
            raise Exception(f"ES 服务未运行或端口 {ES_PORT} 未开放")
        else:
            print(f"✅ 端口 {ES_PORT} 可达，ES 服务可能正在运行")
    except Exception as e:
        if "ES 服务未运行" in str(e):
            raise
        print(f"⚠️  网络检查失败: {str(e)}")
    
    # 尝试使用 requests 直接测试连接（更简单的方式）
    print("\n🔍 使用 requests 测试连接...")
    try:
        import requests
        from requests.auth import HTTPBasicAuth
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 测试 HTTPS 连接
        test_urls = [
            f"https://{ES_HOST}:{ES_PORT}",
            f"http://{ES_HOST}:{ES_PORT}"
        ]
        
        for test_url in test_urls:
            try:
                print(f"   测试 {test_url}...", end='', flush=True)
                response = requests.get(
                    test_url,
                    auth=HTTPBasicAuth(ES_USERNAME, ES_PASSWORD),
                    verify=False,  # 禁用 SSL 验证
                    timeout=10
                )
                if response.status_code == 200:
                    print(f" ✅ 成功！")
                    print(f"   响应: {response.json().get('cluster_name', '未知')}")
                    break
                else:
                    print(f" ❌ HTTP {response.status_code}")
            except requests.exceptions.SSLError as e:
                print(f" ⚠️  SSL 错误（继续尝试其他方式）")
            except Exception as e:
                print(f" ❌ 失败: {str(e)[:50]}")
    except ImportError:
        print("   ⚠️  requests 库未安装，跳过直接测试")
    except Exception as e:
        print(f"   ⚠️  测试失败: {str(e)}")
    
    print()
    
    # 尝试 HTTPS 和 HTTP 连接
    es_urls = [
        f"https://{ES_HOST}:{ES_PORT}",  # 优先尝试 HTTPS（因为用户说可以访问 https://localhost:9200）
        f"http://{ES_HOST}:{ES_PORT}"     # 备用 HTTP
    ]
    
    last_error = None
    for es_url in es_urls:
        try:
            print(f"正在尝试连接: {es_url}...")
            
            # 配置 SSL 上下文，完全禁用证书验证（适用于自签名证书）
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 构建 ES 客户端配置
            es_config = {
                'basic_auth': (ES_USERNAME, ES_PASSWORD) if ES_USERNAME and ES_PASSWORD else None,
                'request_timeout': 60,
                'max_retries': 1,
                'retry_on_timeout': False,
                'verify_certs': False,
                'ssl_show_warn': False,
                'connections_per_node': 1,
                'http_compress': True
            }
            
            # 如果是 HTTPS，添加 SSL 上下文
            if es_url.startswith('https'):
                es_config['ssl_context'] = ssl_context
            
            es = Elasticsearch([es_url], **es_config)
            
            # 测试连接（使用更短的超时）
            cluster_info = es.info(request_timeout=10)
            print(f"✅ 成功连接到 Elasticsearch: {es_url}")
            print(f"   集群名称: {cluster_info['cluster_name']}")
            print(f"   ES 版本: {cluster_info['version']['number']}")
            return es
        except Exception as e:
            last_error = e
            error_msg = str(e)
            error_type = type(e).__name__
            
            # 根据错误类型提供更详细的诊断
            if "RemoteDisconnected" in error_type or "Connection aborted" in error_msg:
                print(f"   ⚠️  {es_url} 连接被远程端关闭")
                print(f"      可能原因：")
                print(f"      1. ES 服务未完全启动")
                print(f"      2. 认证失败（用户名或密码错误）")
                print(f"      3. ES 配置了安全策略，拒绝连接")
            elif "timeout" in error_msg.lower():
                print(f"   ⚠️  {es_url} 连接超时")
                print(f"      可能原因：ES 服务未启动或网络问题")
            elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
                print(f"   ⚠️  {es_url} 认证失败")
                print(f"      请检查用户名和密码是否正确")
            else:
                print(f"   ⚠️  {es_url} 连接失败: {error_msg[:150]}")
            continue
    
    # 如果所有连接都失败，提供详细的诊断信息
    if last_error:
        print(f"\n❌ 连接 Elasticsearch 失败")
        print(f"   错误类型: {type(last_error).__name__}")
        print(f"   错误信息: {str(last_error)[:200]}")
        print(f"\n尝试连接的地址:")
        for es_url in es_urls:
            print(f"   - {es_url}")
        print(f"\n用户名: {ES_USERNAME}")
        print(f"密码: {'*' * len(ES_PASSWORD) if ES_PASSWORD else '(未设置)'}")
        
        print("\n" + "=" * 60)
        print("诊断建议：")
        print("=" * 60)
        print("1. 确认 Elasticsearch 服务已启动")
        print("   检查方法：在浏览器访问 https://localhost:9200")
        print("2. 检查用户名和密码是否正确")
        print("   可以在浏览器中测试认证")
        print("3. 检查 ES 日志文件，查看是否有错误信息")
        print("4. 尝试使用 curl 命令测试连接：")
        print(f'   curl -u {ES_USERNAME}:{ES_PASSWORD} -k https://localhost:9200')
        print("5. 如果 ES 使用 HTTP，确保脚本尝试了 HTTP 连接")
        print("=" * 60)
    
    raise last_error if last_error else Exception("无法连接到 Elasticsearch")


# ====== 步骤 5：创建支持向量的索引 ======
def create_vector_index(es: Elasticsearch, index_name: str):
    """
    创建支持向量搜索的 Elasticsearch 索引
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    
    功能：
    - 定义索引映射，包含 dense_vector 字段用于向量搜索
    - 同时保留文本字段用于混合搜索
    - 创建索引（如果已存在则跳过）
    """
    print("\n" + "=" * 60)
    print("步骤 2：创建支持向量搜索的索引")
    print("=" * 60)
    
    # 检查索引是否已存在
    if es.indices.exists(index=index_name):
        print(f"⚠️  索引 '{index_name}' 已存在，将使用现有索引")
        print("   如需重新创建，请先删除现有索引")
        return
    
    # 定义索引映射，包含向量字段
    index_mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                # 文档标题字段（文本搜索）
                "title": {
                    "type": "text",
                    "analyzer": "ik_max_word",  # 使用 IK 分词器（如果可用）
                    "search_analyzer": "ik_max_word",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                # 文档源文件路径
                "source": {
                    "type": "keyword"
                },
                # 文档块内容（文本搜索）
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                # 向量字段（用于向量搜索）
                "content_vector": {
                    "type": "dense_vector",  # 密集向量类型
                    "dims": EMBEDDING_DIMENSIONS,  # 向量维度：1024
                    "index": True,  # 启用向量索引
                    "similarity": "cosine"  # 使用余弦相似度
                },
                # 块 ID
                "chunk_id": {
                    "type": "integer"
                },
                # 页码
                "page_num": {
                    "type": "integer"
                },
                # Token 数量
                "token_count": {
                    "type": "integer"
                },
                # 文档类型
                "file_type": {
                    "type": "keyword"
                }
            }
        }
    }
    
    try:
        # 创建索引
        es.indices.create(
            index=index_name,
            settings=index_mapping["settings"],
            mappings=index_mapping["mappings"]
        )
        print(f"✅ 成功创建向量索引: {index_name}")
        print(f"   向量维度: {EMBEDDING_DIMENSIONS}")
        print(f"   相似度算法: cosine（余弦相似度）")
    except Exception as e:
        error_msg = str(e)
        # 如果 IK 分词器不可用，尝试使用标准分词器
        if "ik_max_word" in error_msg.lower():
            print(f"   ⚠️  IK 分词器不可用，使用标准分词器")
            # 修改映射，使用标准分词器
            index_mapping["mappings"]["properties"]["title"]["analyzer"] = "standard"
            index_mapping["mappings"]["properties"]["title"]["search_analyzer"] = "standard"
            index_mapping["mappings"]["properties"]["content"]["analyzer"] = "standard"
            index_mapping["mappings"]["properties"]["content"]["search_analyzer"] = "standard"
            
            es.indices.create(
                index=index_name,
                settings=index_mapping["settings"],
                mappings=index_mapping["mappings"]
            )
            print(f"✅ 成功创建向量索引: {index_name}（使用标准分词器）")
        else:
            print(f"❌ 创建索引失败: {str(e)}")
            raise


# ====== 步骤 6：解析文档并生成向量 ======
def parse_and_embed_documents(docs_folder: str, embedding_client: OpenAI) -> List[Dict]:
    """
    解析文档并生成向量嵌入
    
    参数：
    - docs_folder: 文档文件夹路径
    - embedding_client: Embedding 客户端对象
    
    返回：
    - 文档块列表，每个块包含文本内容和向量嵌入
    
    功能：
    - 解析文档并分块
    - 为每个文档块生成向量嵌入
    - 返回包含向量的文档块
    """
    print("\n" + "=" * 60)
    print("步骤 3：解析文档并生成向量嵌入")
    print("=" * 60)
    
    # 创建 DocParser 实例
    doc_parser = DocParser({
        'max_ref_token': 20000,
        'parser_page_size': 500
    })
    
    # 获取所有文档文件
    if not os.path.exists(docs_folder):
        raise FileNotFoundError(f"文档文件夹不存在: {docs_folder}")
    
    files = []
    for file in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file)
        if os.path.isfile(file_path):
            files.append(file_path)
    
    print(f"📁 找到 {len(files)} 个文档文件")
    print(f"📊 使用模型: {EMBEDDING_MODEL}")
    print(f"📏 向量维度: {EMBEDDING_DIMENSIONS}\n")
    
    # 解析所有文档并生成向量
    all_chunks = []
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 正在处理: {os.path.basename(file_path)}")
        
        try:
            # 解析文档
            record = doc_parser.call(params={'url': file_path})
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # 处理每个文档块
            for j, chunk in enumerate(record['raw'], 1):
                chunk_text = chunk['content']
                
                # 生成向量嵌入
                print(f"   生成向量 [{j}/{len(record['raw'])}]...", end='', flush=True)
                try:
                    embedding = generate_embedding(chunk_text, embedding_client)
                    print(" ✅")
                except Exception as e:
                    print(f" ❌ 失败: {str(e)}")
                    continue
                
                # 构建文档块数据
                chunk_data = {
                    'title': record['title'],
                    'source': file_path,
                    'content': chunk_text,
                    'content_vector': embedding,  # 添加向量字段
                    'chunk_id': chunk['metadata'].get('chunk_id', 0),
                    'page_num': chunk['metadata'].get('page_num', 1),
                    'token_count': chunk['token'],
                    'file_type': file_ext
                }
                all_chunks.append(chunk_data)
            
            print(f"   ✅ 完成，生成 {len(record['raw'])} 个向量块\n")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {str(e)}\n")
            continue
    
    print(f"✅ 文档处理完成！")
    print(f"   总共生成 {len(all_chunks)} 个带向量的文档块")
    
    return all_chunks


# ====== 步骤 7：索引文档到 Elasticsearch ======
def index_documents_with_vectors(es: Elasticsearch, index_name: str, chunks: List[Dict]):
    """
    将文档块和向量索引到 Elasticsearch
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    - chunks: 包含向量的文档块列表
    """
    print("\n" + "=" * 60)
    print("步骤 4：索引文档和向量到 Elasticsearch")
    print("=" * 60)
    
    if not chunks:
        print("⚠️  没有文档块需要索引")
        return
    
    # 准备批量索引的数据
    actions = []
    for chunk in chunks:
        # 生成文档 ID
        doc_id = f"{hashlib.sha256(chunk['source'].encode()).hexdigest()}_{chunk['chunk_id']}"
        
        # 构建要索引的文档（包含向量）
        action = {
            "_index": index_name,
            "_id": doc_id,
            "_source": chunk
        }
        actions.append(action)
    
    print(f"📤 准备索引 {len(actions)} 个文档块（包含向量）...")
    
    try:
        # 批量索引
        success_count, failed_items = bulk(es, actions, chunk_size=50, request_timeout=120)
        
        print(f"✅ 索引完成！")
        print(f"   成功索引: {success_count} 个文档")
        if failed_items:
            print(f"   失败: {len(failed_items)} 个文档")
        
        # 刷新索引
        es.indices.refresh(index=index_name)
        print(f"   索引已刷新，文档可立即搜索")
        
    except Exception as e:
        print(f"❌ 索引失败: {str(e)}")
        raise


# ====== 步骤 8：向量搜索 ======
def vector_search(es: Elasticsearch, index_name: str, search_query: str, top_k: int = 10, 
                  use_hybrid: bool = True):
    """
    使用向量搜索在 Elasticsearch 中检索文档
    
    参数：
    - es: Elasticsearch 客户端对象
    - index_name: 索引名称
    - search_query: 搜索查询字符串
    - top_k: 返回前 K 个结果
    - use_hybrid: 是否使用混合搜索（向量 + 关键词）
    
    返回：
    - 搜索结果列表
    
    功能：
    - 将查询文本转换为向量
    - 使用 knn 查询进行向量搜索
    - 可选：结合关键词搜索（混合搜索）
    """
    print("\n" + "=" * 60)
    print("步骤 5：执行向量搜索")
    print("=" * 60)
    
    print(f"🔍 搜索查询: {search_query}")
    print(f"   返回前 {top_k} 个结果")
    print(f"   搜索模式: {'混合搜索（向量 + 关键词）' if use_hybrid else '纯向量搜索'}")
    
    # 初始化 Embedding 客户端
    embedding_client = init_embedding_client()
    
    # 生成查询向量
    print(f"\n📊 正在生成查询向量...")
    try:
        query_vector = generate_embedding(search_query, embedding_client)
        print(f"   ✅ 查询向量生成完成（维度: {len(query_vector)}）")
    except Exception as e:
        print(f"   ❌ 生成查询向量失败: {str(e)}")
        raise
    
    # 构建搜索查询
    if use_hybrid:
        # 混合搜索：结合向量搜索和关键词搜索
        search_body = {
            "knn": {
                "field": "content_vector",  # 向量字段
                "query_vector": query_vector,  # 查询向量
                "k": top_k,  # 返回的最近邻数量
                "num_candidates": top_k * 10  # 候选数量（越大越准确，但越慢）
            },
            "query": {
                "multi_match": {
                    "query": search_query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields"
                }
            },
            "size": top_k,
            "_source": {
                "excludes": ["content_vector"]  # 不返回向量字段（减少传输量）
            }
        }
    else:
        # 纯向量搜索
        search_body = {
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10
            },
            "size": top_k,
            "_source": {
                "excludes": ["content_vector"]
            }
        }
    
    try:
        # 执行搜索
        response = es.search(index=index_name, **search_body)
        
        # 解析搜索结果
        hits = response['hits']['hits']
        total = response['hits']['total']['value']
        
        print(f"\n✅ 搜索完成！")
        print(f"   找到 {total} 个相关文档")
        print(f"   返回前 {len(hits)} 个结果\n")
        
        return hits
        
    except Exception as e:
        print(f"❌ 搜索失败: {str(e)}")
        raise


# ====== 步骤 9：显示搜索结果 ======
def display_search_results(hits: List[Dict]):
    """
    格式化并显示搜索结果
    
    参数：
    - hits: 搜索结果列表
    """
    print("=" * 60)
    print("步骤 6：显示搜索结果")
    print("=" * 60)
    
    if not hits:
        print("⚠️  没有找到相关文档")
        return
    
    # 遍历显示每个搜索结果
    for i, hit in enumerate(hits, 1):
        score = hit['_score']  # 相关性评分（向量相似度或混合评分）
        source = hit['_source']
        
        print(f"\n{'=' * 60}")
        print(f"结果 {i} (相关性评分: {score:.4f})")
        print(f"{'=' * 60}")
        
        # 显示标题
        print(f"📄 标题: {source.get('title', '无标题')}")
        
        # 显示来源文件
        source_file = os.path.basename(source.get('source', '未知'))
        print(f"📁 来源: {source_file}")
        
        # 显示文件类型和块信息
        print(f"📊 信息: 文件类型={source.get('file_type', '未知')}, "
              f"块ID={source.get('chunk_id', 0)}, "
              f"Token数={source.get('token_count', 0)}")
        
        # 显示内容预览
        content = source.get('content', '')
        preview = content[:300] + ('...' if len(content) > 300 else '')
        print(f"\n💡 内容预览:")
        print(f"   {preview}")
        
        print()
    
    print("=" * 60)


# ====== 主程序 ======
def main():
    """
    主程序：执行完整的向量索引和搜索流程
    
    流程：
    1. 连接到 Elasticsearch
    2. 创建支持向量的索引
    3. 解析文档并生成向量
    4. 索引文档和向量到 Elasticsearch
    5. 执行向量搜索
    6. 显示搜索结果
    """
    print("\n" + "🚀" * 30)
    print("基于 Embedding 向量的 Elasticsearch 文档搜索系统")
    print("🚀" * 30 + "\n")
    
    try:
        # 步骤 1：初始化 Embedding 客户端
        print("=" * 60)
        print("初始化 Embedding 客户端")
        print("=" * 60)
        embedding_client = init_embedding_client()
        print(f"✅ Embedding 客户端初始化成功")
        print(f"   模型: {EMBEDDING_MODEL}")
        print(f"   维度: {EMBEDDING_DIMENSIONS}\n")
        
        # 步骤 2：连接到 Elasticsearch
        es = connect_elasticsearch()
        
        # 步骤 3：创建向量索引
        create_vector_index(es, INDEX_NAME)
        
        # 步骤 4：解析文档并生成向量
        chunks = parse_and_embed_documents(DOCS_FOLDER, embedding_client)
        
        # 步骤 5：索引文档和向量
        if chunks:
            index_documents_with_vectors(es, INDEX_NAME, chunks)
        else:
            print("⚠️  没有文档块需要索引，跳过索引步骤")
        
        # 步骤 6：执行向量搜索
        search_query = "工伤保险和雇主险有什么区别？"
        hits = vector_search(es, INDEX_NAME, search_query, top_k=10, use_hybrid=True)
        
        # 步骤 7：显示搜索结果
        display_search_results(hits)
        
        print("\n" + "✅" * 30)
        print("所有步骤执行完成！")
        print("✅" * 30 + "\n")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# ====== 程序入口 ======
if __name__ == '__main__':
    exit(main())

